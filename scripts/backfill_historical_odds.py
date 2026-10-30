"""Backfill historical PRE-GAME odds from the-odds-api historical endpoint.

The live fetch_live_odds.py only pulls UPCOMING events, so games loaded by the
load_*_historical scripts have no odds rows and every market feature falls back
to NEUTRAL_DEFAULTS (the moneyline is the strongest single predictor we have).

v2 (2026-10-25) — why the old design was replaced
--------------------------------------------------
v1 took ONE snapshot per game-day at 03:00 UTC of the NEXT day, "after most
games started", on the theory that it returns the line "live right at puck
drop". It does not: the historical /odds endpoint returns what the live one
returns at that instant — upcoming events AND events in progress, the latter
with IN-PLAY prices. The insert path's in-play guard compares against NOW(), so
for a historical game it never fired. Combined with a ±2-day team-name match
that returned the first hit without date ordering, and an insert path where the
first snapshot to reach a key wins, the stored "closing" lines for 2022-25
NBA/NFL/NHL games were a mix of (a) day-early lines, (b) in-play lines for any
game in progress at 03:00 UTC, and (c) occasionally another game's lines when
the same teams met within two days. Measured on prod (audit §7, 2026-10-25):
historical NBA markets are far more confident than the live closing market
(mean |p-0.5| 0.20-0.215 vs 0.148; 141 games above 90% implied in 2022-23 vs 0
live), and the cross-book line dispersion they produce is 4-5x the live value.

v2 design:
  * PLAN: cluster each game's commence time into groups no wider than
    --window-hours; one snapshot per group at (group's first commence -
    --lead-minutes). Every game therefore gets a synchronized, strictly
    pre-game snapshot taken at most window+lead before its own tip.
  * MATCH: an event is accepted only if its commence_time is AFTER the
    snapshot's own timestamp (never in-play), and only against the games of
    the group the snapshot was planned for (exact normalized names, then a
    fuzzy match restricted to those few games, one-to-one, commence within
    --max-offset-hours of our match_date). Nothing two days away can attach.
  * CACHE: every raw response is written to --cache-dir and read back on
    re-runs, so credits are spent once; `--mode cache-only` stages a season
    for offline validation without touching the database.
  * MODES: insert (default) writes only games that have NO pre-match odds
    rows yet — purely additive. replace archives a game's existing rows to
    --archive-dir (CSV) and swaps in the v2 snapshot; it is a destructive
    data change and refuses to run without --confirm-replace.
  * QUOTA: stops before x-requests-remaining would fall below --quota-floor,
    so the live odds fetch (≈1.3k credits/day) is never starved.

Cost per call = regions × markets × 10 (the historical multiplier), i.e. 30
credits for us × h2h,spreads,totals. A season is ~1-1.5 calls per game-day.

Usage (inside the api container):
    python /app/scripts/backfill_historical_odds.py --sport basketball_nba \\
        --start 2025-10-01 --end 2026-06-30 --dry-run
    python /app/scripts/backfill_historical_odds.py --sport basketball_nba \\
        --start 2025-10-01 --end 2026-06-30 --cache-dir /app/backups/odds_hist_cache
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_historical_odds")

BASE_URL = "https://api.the-odds-api.com/v4"
# the-odds-api documents a 10x quota multiplier for the historical API.
HISTORICAL_COST_MULTIPLIER = 10

DEFAULT_WINDOW_HOURS = 3.0
DEFAULT_LEAD_MINUTES = 10
DEFAULT_MAX_OFFSET_HOURS = 3.0
DEFAULT_QUOTA_FLOOR = 15000
# Rate limits (429), 5xx and dropped connections are retried; anything else
# (e.g. 404 UNKNOWN_SPORT for a tournament key that wasn't open yet) is final.
FETCH_ATTEMPTS = 4
FETCH_BACKOFF_SECONDS = (2.0, 6.0, 20.0)
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
FUZZY_THRESHOLD = 0.8

MODES = ("insert", "cache-only", "replace")

# Reuse fetch_live_odds.py helpers without a hard package import.
# scripts/ isn't an installable package; we load by path.
_LIVE_PATH = Path(__file__).resolve().parent / "fetch_live_odds.py"
_spec = importlib.util.spec_from_file_location("_fetch_live_odds", _LIVE_PATH)
assert _spec and _spec.loader
fetch_live_odds = importlib.util.module_from_spec(_spec)
sys.modules["_fetch_live_odds"] = fetch_live_odds
_spec.loader.exec_module(fetch_live_odds)


# ── Pure planning / matching (unit-tested) ──────────────────────────────


@dataclass
class Game:
    match_id: str
    match_date: datetime
    home: str
    away: str
    venue: str = ""


@dataclass
class Snapshot:
    requested: datetime
    games: list[Game] = field(default_factory=list)

    @property
    def iso(self) -> str:
        return self.requested.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def plan_snapshots(
    games: list[Game],
    window_hours: float = DEFAULT_WINDOW_HOURS,
    lead_minutes: int = DEFAULT_LEAD_MINUTES,
) -> list[Snapshot]:
    """Group games by commence time (greedy: a group spans at most window_hours
    from its first game) and plan one snapshot per group at first - lead."""
    out: list[Snapshot] = []
    window = timedelta(hours=window_hours)
    lead = timedelta(minutes=lead_minutes)
    current: Snapshot | None = None
    first: datetime | None = None
    for g in sorted(games, key=lambda x: x.match_date):
        if current is None or first is None or g.match_date - first > window:
            first = g.match_date
            current = Snapshot(requested=first - lead)
            out.append(current)
        current.games.append(g)
    return out


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def match_events_to_games(
    events: list[dict],
    snapshot_time: datetime,
    games: list[Game],
    max_offset_hours: float = DEFAULT_MAX_OFFSET_HOURS,
    normalize=None,
) -> tuple[list[tuple[dict, Game]], dict[str, int]]:
    """One-to-one (event, game) pairs for ONE snapshot.

    An event qualifies only when its commence_time is strictly AFTER
    snapshot_time (a game in progress carries in-play prices) and within
    max_offset_hours of a candidate game's match_date. Exact normalized
    home+away names win; otherwise the best fuzzy pair at or above
    FUZZY_THRESHOLD among the snapshot's own games. Returns (pairs, counts)."""
    norm = normalize or fetch_live_odds.normalize_name
    offset = timedelta(hours=max_offset_hours)
    counts = {"events": len(events), "in_play_or_started": 0, "unmatched": 0, "exact": 0, "fuzzy": 0}
    free = list(games)
    pairs: list[tuple[dict, Game]] = []
    pending: list[tuple[dict, datetime]] = []
    for ev in events:
        commence = _parse_ts(ev.get("commence_time"))
        if commence is None or not ev.get("home_team") or not ev.get("away_team"):
            counts["unmatched"] += 1
            continue
        if commence <= snapshot_time:
            counts["in_play_or_started"] += 1
            continue
        pending.append((ev, commence))

    # Pass 1: exact names.
    rest: list[tuple[dict, datetime]] = []
    for ev, commence in pending:
        h, a = norm(ev["home_team"]), norm(ev["away_team"])
        hit = next(
            (g for g in free if norm(g.home) == h and norm(g.away) == a and abs(g.match_date - commence) <= offset),
            None,
        )
        if hit is None:
            rest.append((ev, commence))
            continue
        free.remove(hit)
        pairs.append((ev, hit))
        counts["exact"] += 1

    # Pass 2: fuzzy, restricted to this snapshot's remaining games.
    for ev, commence in rest:
        h, a = norm(ev["home_team"]), norm(ev["away_team"])
        best, best_score = None, 0.0
        for g in free:
            if abs(g.match_date - commence) > offset:
                continue
            score = (
                SequenceMatcher(None, norm(g.home), h).ratio() + SequenceMatcher(None, norm(g.away), a).ratio()
            ) / 2
            if score > best_score:
                best, best_score = g, score
        if best is not None and best_score >= FUZZY_THRESHOLD:
            free.remove(best)
            pairs.append((ev, best))
            counts["fuzzy"] += 1
        else:
            counts["unmatched"] += 1
    return pairs, counts


def credits_per_call(markets: str, regions: str) -> int:
    n_markets = len([m for m in markets.split(",") if m.strip()])
    n_regions = len([r for r in regions.split(",") if r.strip()])
    return n_markets * n_regions * HISTORICAL_COST_MULTIPLIER


# ── Tennis: one the-odds-api key per TOURNAMENT ─────────────────────────
#
# the-odds-api lists tennis per tournament (tennis_atp_wimbledon, ...), and only
# the Slams, the 1000s and the tour finals — exactly the set the live fetch
# prices (fetch_live_odds.SPORT_KEYS), so training and serving see the same
# provider. Our ESPN tennis rows carry no tournament name, but every one has a
# venue; (venue, date window) resolves the tournament. ESPN's "ATP Tour" feed
# also carries the women's draw at the Slams, so combined events query BOTH
# tour keys and let the player names decide. Windows are generous (qualifying
# rounds in them simply never match) and never overlap for one venue.
TENNIS_EVENTS: tuple[tuple[tuple[str, ...], tuple[str, str], tuple[str, ...]], ...] = (
    (("melbourne",), ("01-08", "02-05"), ("tennis_atp_australian_open", "tennis_wta_australian_open")),
    (("indian wells",), ("03-01", "03-24"), ("tennis_atp_indian_wells", "tennis_wta_indian_wells")),
    (("miami",), ("03-15", "04-08"), ("tennis_atp_miami", "tennis_wta_miami")),
    (("monte carlo", "monte-carlo"), ("04-01", "04-20"), ("tennis_atp_monte_carlo",)),
    (("madrid",), ("04-18", "05-12"), ("tennis_atp_madrid_open", "tennis_wta_madrid_open")),
    (("rome", "roma"), ("05-01", "05-25"), ("tennis_atp_italian_open", "tennis_wta_italian_open")),
    (("paris",), ("05-18", "06-12"), ("tennis_atp_french_open", "tennis_wta_french_open")),
    (("london", "wimbledon"), ("06-26", "07-18"), ("tennis_atp_wimbledon", "tennis_wta_wimbledon")),
    (("toronto", "montreal"), ("07-25", "08-20"), ("tennis_atp_canadian_open", "tennis_wta_canadian_open")),
    (("cincinnati", "mason"), ("08-05", "08-28"), ("tennis_atp_cincinnati_open", "tennis_wta_cincinnati_open")),
    (("new york", "flushing"), ("08-18", "09-14"), ("tennis_atp_us_open", "tennis_wta_us_open")),
    (("beijing",), ("09-20", "10-10"), ("tennis_wta_china_open",)),
    (("shanghai",), ("09-28", "10-22"), ("tennis_atp_shanghai_masters",)),
    (("wuhan",), ("10-01", "10-22"), ("tennis_wta_wuhan_open",)),
    (("paris",), ("10-25", "11-08"), ("tennis_atp_paris_masters",)),
    (("riyadh", "cancun", "fort worth"), ("10-25", "11-12"), ("tennis_wta_finals",)),
    (("turin", "torino"), ("11-08", "11-25"), ("tennis_atp_finals",)),
)


def _fold(text: str) -> str:
    """Casefold, strip accents and punctuation-ish noise for name/venue matching."""
    import unicodedata

    t = "".join(ch for ch in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(ch))
    return " ".join(t.lower().replace(".", " ").replace("-", " ").split())


def tennis_keys_for(venue: str, when: datetime) -> tuple[str, ...]:
    """the-odds-api tournament keys a tennis match may be listed under, or ()."""
    v = _fold(venue)
    md = when.strftime("%m-%d")
    for venues, (lo, hi), keys in TENNIS_EVENTS:
        if lo <= md <= hi and any(_fold(name) in v for name in venues):
            return keys
    return ()


def plan_tennis_snapshots(
    games: list[Game], window_hours: float = DEFAULT_WINDOW_HOURS, lead_minutes: int = DEFAULT_LEAD_MINUTES
) -> list[tuple[str, Snapshot]]:
    """(sport_key, Snapshot) tasks: games grouped by their tournament key set,
    clustered by start time exactly as plan_snapshots does, and each cluster
    requested once per key in the set."""
    by_keys: dict[tuple[str, ...], list[Game]] = {}
    for g in games:
        keys = tennis_keys_for(g.venue, g.match_date)
        if keys:
            by_keys.setdefault(keys, []).append(g)
    tasks: list[tuple[str, Snapshot]] = []
    for keys, gs in by_keys.items():
        for snap in plan_snapshots(gs, window_hours, lead_minutes):
            tasks.extend((k, snap) for k in keys)
    tasks.sort(key=lambda t: (t[1].requested, t[0]))
    return tasks


def match_tennis_events(
    events: list[dict],
    snapshot_time: datetime,
    games: list[Game],
    max_offset_hours: float = 12.0,
) -> tuple[list[tuple[dict, Game, bool]], dict[str, int]]:
    """One-to-one (event, game, swapped) for ONE tennis snapshot.

    Player order is arbitrary on both sides (ESPN's and the-odds-api's "home"
    player are just list positions), so a pair matches in either orientation;
    `swapped` says the event's home_team is OUR away player. Strictly pre-game
    (commence_time after the snapshot) and within max_offset_hours (order of
    play moves start times by hours). Exact folded names first, then the best
    fuzzy pair >= FUZZY_THRESHOLD among this snapshot's games."""
    offset = timedelta(hours=max_offset_hours)
    counts = {"events": len(events), "in_play_or_started": 0, "unmatched": 0, "exact": 0, "fuzzy": 0}
    free = list(games)
    pairs: list[tuple[dict, Game, bool]] = []
    pending: list[tuple[dict, datetime]] = []
    for ev in events:
        commence = _parse_ts(ev.get("commence_time"))
        if commence is None or not ev.get("home_team") or not ev.get("away_team"):
            counts["unmatched"] += 1
            continue
        if commence <= snapshot_time:
            counts["in_play_or_started"] += 1
            continue
        pending.append((ev, commence))
    rest: list[tuple[dict, datetime]] = []
    for ev, commence in pending:
        h, a = _fold(ev["home_team"]), _fold(ev["away_team"])
        hit = None
        for g in free:
            if abs(g.match_date - commence) > offset:
                continue
            gh, ga = _fold(g.home), _fold(g.away)
            if (gh, ga) == (h, a):
                hit = (g, False)
                break
            if (gh, ga) == (a, h):
                hit = (g, True)
                break
        if hit is None:
            rest.append((ev, commence))
            continue
        free.remove(hit[0])
        pairs.append((ev, hit[0], hit[1]))
        counts["exact"] += 1
    for ev, commence in rest:
        h, a = _fold(ev["home_team"]), _fold(ev["away_team"])
        best, best_score, best_swapped = None, 0.0, False
        for g in free:
            if abs(g.match_date - commence) > offset:
                continue
            gh, ga = _fold(g.home), _fold(g.away)
            straight = (SequenceMatcher(None, gh, h).ratio() + SequenceMatcher(None, ga, a).ratio()) / 2
            crossed = (SequenceMatcher(None, gh, a).ratio() + SequenceMatcher(None, ga, h).ratio()) / 2
            score, swapped = (straight, False) if straight >= crossed else (crossed, True)
            if score > best_score:
                best, best_score, best_swapped = g, score, swapped
        if best is not None and best_score >= FUZZY_THRESHOLD:
            free.remove(best)
            pairs.append((ev, best, best_swapped))
            counts["fuzzy"] += 1
        else:
            counts["unmatched"] += 1
    return pairs, counts


# ── I/O ─────────────────────────────────────────────────────────────────


def cache_path(cache_dir: str | None, sport_key: str, snap: Snapshot, markets: str, regions: str) -> Path | None:
    if not cache_dir:
        return None
    tag = f"{markets.replace(',', '+')}_{regions.replace(',', '+')}"
    return Path(cache_dir) / sport_key / f"{snap.iso}_{tag}.json"


def fetch_historical_snapshot(
    sport_key: str, snapshot_iso: str, markets: str, api_key: str, regions: str
) -> tuple[dict | None, int | None]:
    """One historical snapshot call, retrying rate limits, 5xx and connection
    errors with backoff. Returns (response or None on a graceful skip,
    x-requests-remaining or None). Raises on auth errors."""
    url = f"{BASE_URL}/historical/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "date": snapshot_iso,
    }
    r = None
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        if attempt > 1:
            time.sleep(FETCH_BACKOFF_SECONDS[min(attempt - 2, len(FETCH_BACKOFF_SECONDS) - 1)])
        try:
            r = requests.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            logger.warning("snapshot %s %s: %s (attempt %d/%d)", sport_key, snapshot_iso, exc, attempt, FETCH_ATTEMPTS)
            r = None
            continue
        if r.status_code not in RETRYABLE_STATUS:
            break
        logger.warning(
            "snapshot %s %s: HTTP %d (attempt %d/%d)", sport_key, snapshot_iso, r.status_code, attempt, FETCH_ATTEMPTS
        )
    if r is None:
        logger.error(
            "snapshot %s %s: no response after %d attempts — skipping", sport_key, snapshot_iso, FETCH_ATTEMPTS
        )
        return None, None
    if r.status_code == 401:
        raise RuntimeError("the-odds-api: invalid API key (401)")
    remaining_hdr = r.headers.get("x-requests-remaining")
    remaining = int(float(remaining_hdr)) if remaining_hdr not in (None, "") else None
    if r.status_code != 200:
        try:
            error_code = (r.json() or {}).get("error_code", "")
        except ValueError:
            error_code = ""
        logger.warning("snapshot %s %s: HTTP %d %s — skipping", sport_key, snapshot_iso, r.status_code, error_code)
        return None, remaining
    logger.info("snapshot %s: cost=%s remaining=%s", snapshot_iso, r.headers.get("x-requests-last", "?"), remaining_hdr)
    return r.json(), remaining


def load_games(cur, sport: str, start: date, end: date, mode: str) -> list[Game]:
    """Finished non-preseason games of `sport` in [start, end]. In insert mode,
    only games with NO pre-match odds rows at all (purely additive)."""
    only_missing = (
        "AND NOT EXISTS (SELECT 1 FROM odds o WHERE o.match_id = m.id AND COALESCE(o.is_live, false) = false)"
        if mode == "insert"
        else ""
    )
    cur.execute(
        f"""
        SELECT m.id::text AS match_id, m.match_date, ht.name AS home, at.name AS away, m.venue
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE l.sport = %s
          AND m.status = 'finished'
          AND m.match_date >= %s AND m.match_date < %s
          AND COALESCE(m.metadata->>'season_type', '') <> 'preseason'
          {only_missing}
        ORDER BY m.match_date
        """,
        (sport, start, end + timedelta(days=1)),
    )
    return [Game(r["match_id"], r["match_date"], r["home"], r["away"], r.get("venue") or "") for r in cur.fetchall()]


_ARCHIVE_COLUMNS = (
    "id",
    "match_id",
    "bookmaker",
    "market_type",
    "selection",
    "odds_decimal",
    "odds_american",
    "line",
    "implied_probability",
    "timestamp",
    "is_opening",
    "is_live",
    "created_at",
    "last_seen_at",
)


def archive_and_delete(cur, match_id: str, writer) -> int:
    """replace mode: copy the game's pre-match odds rows to the archive CSV,
    then delete them. Returns rows archived (== rows deleted)."""
    cur.execute(
        f"SELECT {', '.join(_ARCHIVE_COLUMNS)} FROM odds WHERE match_id = %s AND COALESCE(is_live, false) = false",
        (match_id,),
    )
    rows = cur.fetchall()
    for r in rows:
        writer.writerow([r[c] for c in _ARCHIVE_COLUMNS])
    cur.execute("DELETE FROM odds WHERE match_id = %s AND COALESCE(is_live, false) = false", (match_id,))
    return len(rows)


# ── Orchestration ───────────────────────────────────────────────────────


def run(
    database_url: str,
    sport_key: str,
    start: date,
    end: date,
    markets: str,
    api_key: str,
    regions: str,
    dry_run: bool,
    mode: str = "insert",
    cache_dir: str | None = None,
    archive_dir: str | None = None,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    lead_minutes: int = DEFAULT_LEAD_MINUTES,
    max_offset_hours: float = DEFAULT_MAX_OFFSET_HOURS,
    quota_floor: int = DEFAULT_QUOTA_FLOOR,
    confirm_replace: bool = False,
) -> dict[str, int]:
    sport = fetch_live_odds.sport_for_key(sport_key)
    if sport is None:
        raise ValueError(f"Unknown sport key prefix: {sport_key!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "replace" and not (confirm_replace or dry_run):
        raise RuntimeError(
            "replace mode deletes existing odds rows (archived to CSV first) — a destructive data change. "
            "Re-run with --confirm-replace once an operator has approved it."
        )

    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            games = load_games(cur, sport, start, end, mode)
    snaps = plan_snapshots(games, window_hours, lead_minutes)
    per_call = credits_per_call(markets, regions)
    cached = sum(1 for s in snaps if (p := cache_path(cache_dir, sport_key, s, markets, regions)) and p.exists())
    estimate = per_call * (len(snaps) - cached)
    logger.info(
        "Plan: sport=%s mode=%s games=%d snapshots=%d (cached %d) cost/call=%d estimate=%d credits",
        sport_key,
        mode,
        len(games),
        len(snaps),
        cached,
        per_call,
        estimate,
    )
    totals = {
        "games": len(games),
        "snapshots_planned": len(snaps),
        "snapshots_cached": cached,
        "estimated_credits": estimate,
        "snapshots_fetched": 0,
        "snapshots_skipped": 0,
        "games_matched": 0,
        "games_unmatched": 0,
        "events_in_play_rejected": 0,
        "fuzzy_matches": 0,
        "odds_rows_written": 0,
        "rows_archived": 0,
        "stopped_on_quota": 0,
    }
    if dry_run or not snaps:
        return totals

    archive_file = None
    writer = None
    if mode == "replace":
        Path(archive_dir or ".").mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_path = Path(archive_dir or ".") / f"odds_archive_{sport_key}_{start}_{end}_{stamp}.csv"
        archive_file = open(archive_path, "w", newline="")
        writer = csv.writer(archive_file)
        writer.writerow(_ARCHIVE_COLUMNS)
        logger.info("replace mode: archiving replaced rows to %s", archive_path)

    remaining: int | None = None
    try:
        with psycopg2.connect(database_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for snap in snaps:
                    path = cache_path(cache_dir, sport_key, snap, markets, regions)
                    if path is not None and path.exists():
                        response = json.loads(path.read_text())
                    else:
                        if remaining is not None and remaining - per_call < quota_floor:
                            logger.error(
                                "Quota floor reached (remaining=%d, floor=%d) — stopping; re-run resumes from cache",
                                remaining,
                                quota_floor,
                            )
                            totals["stopped_on_quota"] = 1
                            break
                        response, remaining = fetch_historical_snapshot(sport_key, snap.iso, markets, api_key, regions)
                        if response is None:
                            totals["snapshots_skipped"] += 1
                            continue
                        totals["snapshots_fetched"] += 1
                        if path is not None:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(json.dumps(response))
                    if mode == "cache-only":
                        continue

                    snap_time = _parse_ts(response.get("timestamp")) or snap.requested
                    pairs, counts = match_events_to_games(
                        response.get("data") or [], snap_time, snap.games, max_offset_hours
                    )
                    totals["events_in_play_rejected"] += counts["in_play_or_started"]
                    totals["fuzzy_matches"] += counts["fuzzy"]
                    totals["games_matched"] += len(pairs)
                    totals["games_unmatched"] += len(snap.games) - len(pairs)
                    if writer is not None:
                        # Clear EVERY planned game of this group, matched or not: an
                        # unmatched game is better off on NEUTRAL_DEFAULTS than keeping
                        # a v1 row that may be in-play or another game's price.
                        for game in snap.games:
                            totals["rows_archived"] += archive_and_delete(cur, game.match_id, writer)
                    for event, game in pairs:
                        totals["odds_rows_written"] += fetch_live_odds.insert_event_odds(
                            cur,
                            game.match_id,
                            event,
                            sport,
                            event["home_team"],
                            event["away_team"],
                            record_snapshots=False,
                            allow_post_commence_insert=True,
                        )
                    conn.commit()
    finally:
        if archive_file is not None:
            archive_file.close()
    logger.info("Totals: %s", totals)
    return totals


def run_tennis(
    database_url: str,
    start: date,
    end: date,
    markets: str,
    api_key: str,
    regions: str,
    dry_run: bool,
    mode: str = "insert",
    cache_dir: str | None = None,
    archive_dir: str | None = None,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    lead_minutes: int = DEFAULT_LEAD_MINUTES,
    max_offset_hours: float = 12.0,
    quota_floor: int = DEFAULT_QUOTA_FLOOR,
    confirm_replace: bool = False,
) -> dict[str, int]:
    """run() for tennis: snapshots are per TOURNAMENT key (TENNIS_EVENTS), the
    matcher ignores player order, and prices are written in OUR orientation."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "replace" and not (confirm_replace or dry_run):
        raise RuntimeError(
            "replace mode deletes existing odds rows (archived to CSV first) — a destructive data change. "
            "Re-run with --confirm-replace once an operator has approved it."
        )
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            games = load_games(cur, "tennis", start, end, mode)
    tasks = plan_tennis_snapshots(games, window_hours, lead_minutes)
    mapped = {g.match_id for _, snap in tasks for g in snap.games}
    per_call = credits_per_call(markets, regions)
    cached = sum(1 for key, snap in tasks if (p := cache_path(cache_dir, key, snap, markets, regions)) and p.exists())
    per_key: dict[str, int] = {}
    for key, _ in tasks:
        per_key[key] = per_key.get(key, 0) + 1
    totals = {
        "games": len(games),
        "games_in_listed_tournaments": len(mapped),
        "snapshot_calls_planned": len(tasks),
        "snapshots_cached": cached,
        "estimated_credits": per_call * (len(tasks) - cached),
        "snapshots_fetched": 0,
        "snapshots_skipped": 0,
        "games_matched": 0,
        "events_in_play_rejected": 0,
        "fuzzy_matches": 0,
        "swapped_orientation": 0,
        "odds_rows_written": 0,
        "rows_archived": 0,
        "stopped_on_quota": 0,
    }
    logger.info("Tennis plan: %s; calls per key: %s", totals, dict(sorted(per_key.items())))
    if dry_run or not tasks:
        return totals

    archive_file = None
    writer = None
    if mode == "replace":
        Path(archive_dir or ".").mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_path = Path(archive_dir or ".") / f"odds_archive_tennis_{start}_{end}_{stamp}.csv"
        archive_file = open(archive_path, "w", newline="")
        writer = csv.writer(archive_file)
        writer.writerow(_ARCHIVE_COLUMNS)
    matched: set[str] = set()
    archived: set[str] = set()
    remaining: int | None = None
    try:
        with psycopg2.connect(database_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for key, snap in tasks:
                    todo = [g for g in snap.games if g.match_id not in matched]
                    if not todo:
                        continue  # every game of this cluster was priced under the other tour's key
                    path = cache_path(cache_dir, key, snap, markets, regions)
                    if path is not None and path.exists():
                        response = json.loads(path.read_text())
                    else:
                        if remaining is not None and remaining - per_call < quota_floor:
                            logger.error("Quota floor reached (remaining=%d) — stopping; re-run resumes", remaining)
                            totals["stopped_on_quota"] = 1
                            break
                        response, remaining = fetch_historical_snapshot(key, snap.iso, markets, api_key, regions)
                        if response is None:
                            totals["snapshots_skipped"] += 1
                            continue
                        totals["snapshots_fetched"] += 1
                        if path is not None:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(json.dumps(response))
                    if mode == "cache-only":
                        continue
                    snap_time = _parse_ts(response.get("timestamp")) or snap.requested
                    pairs, counts = match_tennis_events(response.get("data") or [], snap_time, todo, max_offset_hours)
                    totals["events_in_play_rejected"] += counts["in_play_or_started"]
                    totals["fuzzy_matches"] += counts["fuzzy"]
                    for event, game, swapped in pairs:
                        if writer is not None and game.match_id not in archived:
                            totals["rows_archived"] += archive_and_delete(cur, game.match_id, writer)
                            archived.add(game.match_id)
                        ours_home, ours_away = (
                            (event["away_team"], event["home_team"])
                            if swapped
                            else (event["home_team"], event["away_team"])
                        )
                        totals["odds_rows_written"] += fetch_live_odds.insert_event_odds(
                            cur,
                            game.match_id,
                            event,
                            "tennis",
                            ours_home,
                            ours_away,
                            record_snapshots=False,
                            allow_post_commence_insert=True,
                        )
                        matched.add(game.match_id)
                        totals["swapped_orientation"] += int(swapped)
                    conn.commit()
    finally:
        if archive_file is not None:
            archive_file.close()
    totals["games_matched"] = len(matched)
    logger.info("Totals: %s", totals)
    return totals


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sport",
        default="icehockey_nhl",
        help="the-odds-api sport key (e.g. icehockey_nhl), or 'tennis' for every listed tournament (TENNIS_EVENTS).",
    )
    p.add_argument("--start", required=True, help="Inclusive start date YYYY-MM-DD (UTC).")
    p.add_argument("--end", required=True, help="Inclusive end date YYYY-MM-DD (UTC).")
    p.add_argument(
        "--markets",
        default=None,
        help="Comma-separated the-odds-api market keys. Defaults to the sport's full basket "
        "from fetch_live_odds.SPORT_MARKETS (h2h+spreads+totals for NHL).",
    )
    p.add_argument("--regions", default="us", help="the-odds-api region (default 'us').")
    p.add_argument("--api-key", default=os.environ.get("THE_ODDS_API_KEY"))
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--mode", choices=MODES, default="insert")
    p.add_argument("--cache-dir", default=None, help="Raw-response cache (read-through). Strongly recommended.")
    p.add_argument("--archive-dir", default="/app/backups", help="replace mode: where replaced rows are archived.")
    p.add_argument("--window-hours", type=float, default=DEFAULT_WINDOW_HOURS)
    p.add_argument("--lead-minutes", type=int, default=DEFAULT_LEAD_MINUTES)
    p.add_argument("--max-offset-hours", type=float, default=DEFAULT_MAX_OFFSET_HOURS)
    p.add_argument("--quota-floor", type=int, default=DEFAULT_QUOTA_FLOOR)
    p.add_argument("--confirm-replace", action="store_true", help="Required for --mode replace (destructive).")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and estimate quota cost without API calls or writes. Always run this first.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.api_key and not args.dry_run:
        logger.error("THE_ODDS_API_KEY not set (required unless --dry-run)")
        return 2
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2

    try:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError as e:
        logger.error("Bad --start/--end (expected YYYY-MM-DD): %s", e)
        return 2

    if args.sport == "tennis":
        # Per-tournament keys are resolved from each match's venue; the model
        # uses the match-winner price only, so h2h is the default (10 credits/call).
        totals = run_tennis(
            database_url=args.database_url,
            start=start,
            end=end,
            markets=args.markets or "h2h",
            api_key=args.api_key or "",
            regions=args.regions,
            dry_run=args.dry_run,
            mode=args.mode,
            cache_dir=args.cache_dir,
            archive_dir=args.archive_dir,
            window_hours=args.window_hours,
            lead_minutes=args.lead_minutes,
            quota_floor=args.quota_floor,
            confirm_replace=args.confirm_replace,
        )
        print(json.dumps(totals, indent=2))
        return 0

    sport = fetch_live_odds.sport_for_key(args.sport)
    if sport is None:
        logger.error("Unknown sport prefix: %s", args.sport)
        return 2

    markets = args.markets or fetch_live_odds.SPORT_MARKETS.get(sport)
    if not markets:
        logger.error("No default markets known for sport=%s; pass --markets explicitly", sport)
        return 2

    totals = run(
        database_url=args.database_url,
        sport_key=args.sport,
        start=start,
        end=end,
        markets=markets,
        api_key=args.api_key or "",
        regions=args.regions,
        dry_run=args.dry_run,
        mode=args.mode,
        cache_dir=args.cache_dir,
        archive_dir=args.archive_dir,
        window_hours=args.window_hours,
        lead_minutes=args.lead_minutes,
        max_offset_hours=args.max_offset_hours,
        quota_floor=args.quota_floor,
        confirm_replace=args.confirm_replace,
    )
    print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
