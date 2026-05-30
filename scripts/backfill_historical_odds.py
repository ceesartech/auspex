"""Backfill historical pre-match odds from the-odds-api historical endpoint.

The live fetch_live_odds.py only pulls UPCOMING events from
/v4/sports/{sport}/odds — meaning historical NHL games loaded via
load_nhl_historical.py have no odds rows. Without odds, the
moneyline / puck-line / total features in features_cache fall back
to NEUTRAL_DEFAULTS for every historical game, and the trained model
gets zero signal from the market line (the strongest single
predictor in sports modeling).

This script walks /v4/historical/sports/{sport}/odds for a date range,
takes one snapshot per game-day, and inserts whatever bookmaker lines
were live just before that day's first NHL game. Reuses the team
matching and market mapping logic from fetch_live_odds.py — same
sport-dispatch, same insertion guard, same `is_opening=false,
is_live=false` so feature compute treats them identically to live
odds.

Quota cost:
    The historical endpoint has a 10x quota multiplier compared to
    the live endpoint. Cost per call = regions × markets × 10.
    Default config (1 region, 3 markets) = 30 credits/call.
    NHL plays ~170 game days per season, so 5 seasons ≈ 25k credits.

    --dry-run prints the call count + credit estimate without spending
    any quota. Always run --dry-run first.

Usage:
    # Estimate cost before spending
    python scripts/backfill_historical_odds.py \\
        --sport icehockey_nhl --start 2022-10-01 --end 2025-06-30 --dry-run

    # Actually backfill the most recent 3 clean seasons (no COVID)
    python scripts/backfill_historical_odds.py \\
        --sport icehockey_nhl --start 2022-10-01 --end 2025-06-30

    # Trim markets to halve cost (just moneyline + totals, no puck line)
    python scripts/backfill_historical_odds.py \\
        --sport icehockey_nhl --start 2022-10-01 --end 2025-06-30 \\
        --markets h2h,totals
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_historical_odds")

BASE_URL = "https://api.the-odds-api.com/v4"
# the-odds-api documents a 10x quota multiplier for the historical API.
HISTORICAL_COST_MULTIPLIER = 10
# Snapshot is taken at this UTC hour on (game-date + day_offset). For NHL
# the modal evening slate runs 23:00 UTC - 02:00 UTC (7pm-10pm ET, plus
# West Coast late games). The default 03:00 UTC of the NEXT day = 11pm
# ET puts the snapshot AFTER most games have started, so the-odds-api
# returns the line that was live right at puck drop for Eastern games
# (true closing) and the line ~30min before tip for the few Western
# games still pending. Override per-sport with --snapshot-hour /
# --snapshot-day-offset if a sport has a different slate cadence.
#
# The "ideal" alternative is per-game snapshots (one call per match at
# commence_time - 5min). That's ~30 credits × games-in-window = ~114k
# credits for 3 seasons of NHL — over the 100k tier — so we defer it
# to a future enhancement and use the best single-snapshot timing today.
DEFAULT_SNAPSHOT_HOUR_UTC = 3
DEFAULT_SNAPSHOT_DAY_OFFSET = 1


# Reuse fetch_live_odds.py helpers without a hard package import.
# scripts/ isn't an installable package; we load by path.
_LIVE_PATH = Path(__file__).resolve().parent / "fetch_live_odds.py"
_spec = importlib.util.spec_from_file_location("_fetch_live_odds", _LIVE_PATH)
assert _spec and _spec.loader
fetch_live_odds = importlib.util.module_from_spec(_spec)
sys.modules["_fetch_live_odds"] = fetch_live_odds
_spec.loader.exec_module(fetch_live_odds)


def fetch_historical_snapshot(
    sport_key: str, snapshot_iso: str, markets: str, api_key: str, regions: str
) -> dict | None:
    """One historical snapshot call. Returns the parsed response dict
    (with `data` list of events) or None on a graceful skip (404 / 422
    / non-200). Raises on auth errors so a bad key fails fast."""
    url = f"{BASE_URL}/historical/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "date": snapshot_iso,
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 401:
        raise RuntimeError("the-odds-api: invalid API key (401)")
    if r.status_code in (404, 422):
        logger.warning("snapshot %s: HTTP %d — skipping", snapshot_iso, r.status_code)
        return None
    if r.status_code != 200:
        logger.warning("snapshot %s: HTTP %d — skipping", snapshot_iso, r.status_code)
        return None

    used = r.headers.get("x-requests-used", "?")
    remaining = r.headers.get("x-requests-remaining", "?")
    cost = r.headers.get("x-requests-last", "?")
    logger.info(
        "snapshot %s: events=? cost=%s quota used=%s remaining=%s",
        snapshot_iso,
        cost,
        used,
        remaining,
    )
    return r.json()


def game_dates_in_range(database_url: str, sport_key: str, start: date, end: date) -> list[date]:
    """Distinct dates (UTC) of finished NHL matches in the DB within
    [start, end]. Skipping dark dates is the cheapest way to cut quota
    cost in half — NHL's regular season is dense but Oct-Dec and
    Mar-Jun have plenty of dark Mondays."""
    sport = fetch_live_odds.sport_for_key(sport_key)
    if sport is None:
        raise ValueError(f"Unknown sport key prefix: {sport_key!r}")
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT DISTINCT DATE(m.match_date) AS d
                FROM matches m
                JOIN leagues l ON l.id = m.league_id
                WHERE l.sport = %s
                  AND m.status = 'finished'
                  AND m.match_date >= %s
                  AND m.match_date < %s
                ORDER BY d ASC
                """,
                (sport, start, end + timedelta(days=1)),
            )
            return [row["d"] for row in cur.fetchall()]


def snapshot_timestamp(
    d: date,
    hour_utc: int = DEFAULT_SNAPSHOT_HOUR_UTC,
    day_offset: int = DEFAULT_SNAPSHOT_DAY_OFFSET,
) -> str:
    """ISO-8601 timestamp for the snapshot to fetch for game-day `d`.
    The-odds-api returns the most recent snapshot at-or-before this
    timestamp. With NHL defaults (hour=3, offset=+1) we ask for
    11pm-ET-of-game-day, which sits AFTER most evening games started
    and returns their closing lines."""
    snapshot_date = d + timedelta(days=day_offset)
    return (
        datetime(snapshot_date.year, snapshot_date.month, snapshot_date.day, hour_utc, 0, 0, tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def process_snapshot_response(cur, sport: str, response: dict) -> tuple[int, int]:
    """Process one historical snapshot's data array. Returns (events_matched, odds_rows_inserted).
    Reuses fetch_live_odds.process_event end-to-end — the response shape inside `data` is
    identical to the live endpoint's top-level array."""
    data = response.get("data") or []
    unmatched: list[str] = []
    events_matched = 0
    odds_rows_inserted = 0
    for event in data:
        inserted = fetch_live_odds.process_event(cur, sport, event, unmatched_log=unmatched)
        if inserted > 0:
            events_matched += 1
            odds_rows_inserted += inserted
    if unmatched:
        logger.info(
            "  %d/%d events had no matching DB fixture. First 3: %s",
            len(unmatched),
            len(data),
            unmatched[:3],
        )
    return events_matched, odds_rows_inserted


def run(
    database_url: str,
    sport_key: str,
    start: date,
    end: date,
    markets: str,
    api_key: str,
    regions: str,
    dry_run: bool,
    snapshot_hour: int = DEFAULT_SNAPSHOT_HOUR_UTC,
    snapshot_day_offset: int = DEFAULT_SNAPSHOT_DAY_OFFSET,
) -> dict[str, int]:
    sport = fetch_live_odds.sport_for_key(sport_key)
    if sport is None:
        raise ValueError(f"Unknown sport key prefix: {sport_key!r}")

    dates = game_dates_in_range(database_url, sport_key, start, end)
    n_markets = len([m for m in markets.split(",") if m.strip()])
    n_regions = len([r for r in regions.split(",") if r.strip()])
    cost_per_call = n_markets * n_regions * HISTORICAL_COST_MULTIPLIER
    estimated_cost = cost_per_call * len(dates)

    logger.info(
        "Plan: sport=%s dates=%d markets=%s regions=%s cost/call=%d total_estimate=%d credits",
        sport_key,
        len(dates),
        markets,
        regions,
        cost_per_call,
        estimated_cost,
    )

    if dry_run:
        logger.info("Dry run — no API calls made.")
        return {
            "dates_planned": len(dates),
            "estimated_credits": estimated_cost,
            "events_matched": 0,
            "odds_rows_inserted": 0,
        }

    if not dates:
        logger.info("No game dates in range — nothing to do.")
        return {"dates_planned": 0, "events_matched": 0, "odds_rows_inserted": 0}

    totals = {"events_matched": 0, "odds_rows_inserted": 0, "dates_processed": 0, "dates_skipped": 0}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for d in dates:
                snap = snapshot_timestamp(d, hour_utc=snapshot_hour, day_offset=snapshot_day_offset)
                response = fetch_historical_snapshot(sport_key, snap, markets, api_key, regions)
                if response is None:
                    totals["dates_skipped"] += 1
                    continue
                events_matched, odds_inserted = process_snapshot_response(cur, sport, response)
                totals["events_matched"] += events_matched
                totals["odds_rows_inserted"] += odds_inserted
                totals["dates_processed"] += 1
                conn.commit()
    return totals


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sport", default="icehockey_nhl", help="the-odds-api sport key (e.g. icehockey_nhl).")
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
    p.add_argument(
        "--snapshot-hour",
        type=int,
        default=DEFAULT_SNAPSHOT_HOUR_UTC,
        help=f"UTC hour of the snapshot timestamp (default {DEFAULT_SNAPSHOT_HOUR_UTC}). "
        "For NHL, 3 sits 11pm ET = after most evening games start = closing lines.",
    )
    p.add_argument(
        "--snapshot-day-offset",
        type=int,
        default=DEFAULT_SNAPSHOT_DAY_OFFSET,
        help=f"Days added to game-date when picking the snapshot timestamp "
        f"(default {DEFAULT_SNAPSHOT_DAY_OFFSET}). Combined with --snapshot-hour, "
        "this lets each sport target a slate-appropriate close-of-play time.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate quota cost without making API calls. Always run this first.",
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
        snapshot_hour=args.snapshot_hour,
        snapshot_day_offset=args.snapshot_day_offset,
    )
    logger.info("Done. %s", totals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
