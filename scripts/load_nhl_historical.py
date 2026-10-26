"""Backfill NHL historical games and team stats from the NHL Stats API.

The NHL Stats API (https://api-web.nhle.com/v1, no key required) is the
canonical source for finished NHL games. We use two endpoints:

  * /club-schedule-season/{tri-code}/{season-id}
        Enumerate every game in a team-season. Each game appears under
        both teams' schedules, so we dedupe by game ID.
  * /gamecenter/{game-id}/right-rail
        Team-level stats (SOG, faceoffs, PP, hits, blocks, giveaways,
        takeaways, PIM) and the period-by-period linescore (used to
        derive regulation scores and OT/SO tiebreak flags).

  * /score/{YYYY-MM-DD}
        Live mode (--recent-days): the games that went final on a date.

Fixture identity (2026-10): every game is first ATTACHED to an existing
fixture — same club under any spelling (accent-insensitive, so ESPN's
'Montreal Canadiens' row matches the API's 'Montréal'), kickoff within
FIXTURE_TOLERANCE_HOURS, or the same NHL game id — and its stats are
keyed to that fixture's own team ids. Before this, the 2025-26 load
minted a second row for every ESPN-created fixture whose kickoff or team
row differed, and the live season got no stats at all. Only historical
games with no fixture are inserted (upsert key home/away/match_date);
--attach-only (the live pipeline task) never inserts. (match_id, team_id)
is unique on nhl_match_stats, so re-runs overwrite stat rows in place.

Usage:
    # Live pipeline tick: attach stats for games final in the last 3 days
    python scripts/load_nhl_historical.py --recent-days 3 --attach-only

    # Current season to date, attach-only, dry run first
    python scripts/load_nhl_historical.py --seasons current --attach-only --dry-run

    # Single season (good for sanity checks)
    python scripts/load_nhl_historical.py --seasons 2024-2025

    # Full roadmap backfill (10 seasons, ~13k games, ~1h with throttle)
    python scripts/load_nhl_historical.py \\
        --seasons 2015-2016,2016-2017,2017-2018,2018-2019,2019-2020,\\
2020-2021,2021-2022,2022-2023,2023-2024,2024-2025

    # Skip games already in nhl_match_stats (default), or refetch all
    python scripts/load_nhl_historical.py --seasons 2024-2025 --refetch
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import psycopg2
import requests
from psycopg2.extras import Json, RealDictCursor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import team_identity  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("load_nhl_historical")

NHL_API = "https://api-web.nhle.com/v1"

# NHL team tri-codes used by the API. Includes every franchise active
# at any point in the 10-season window (2015-16 → 2024-25):
#   * The 30 teams pre-Vegas
#   * VGK (2017-18 onwards)
#   * SEA (2021-22 onwards)
#   * UTA, the Utah franchise that replaced Arizona in 2024-25
# Inactive (or not-yet-active) teams in a given season's schedule call
# return an empty `games` array, which we treat as a no-op.
NHL_TEAM_CODES = [
    "ANA",
    "ARI",
    "BOS",
    "BUF",
    "CAR",
    "CBJ",
    "CGY",
    "CHI",
    "COL",
    "DAL",
    "DET",
    "EDM",
    "FLA",
    "LAK",
    "MIN",
    "MTL",
    "NJD",
    "NSH",
    "NYI",
    "NYR",
    "OTT",
    "PHI",
    "PIT",
    "SEA",
    "SJS",
    "STL",
    "TBL",
    "TOR",
    "UTA",
    "VAN",
    "VGK",
    "WPG",
    "WSH",
]

# gameType values from the schedule feed.
GAME_TYPE_PRESEASON = 1
GAME_TYPE_REGULAR = 2
GAME_TYPE_PLAYOFF = 3

# gameState values that mean the game is final and stats are settled.
FINAL_STATES = {"FINAL", "OFF", "OFFICIAL"}

# Be polite — NHL doesn't publish rate limits but we still don't want
# to hammer them. ~3 req/s is the working number used by the nhl-api
# Python clients in the wild.
REQUEST_DELAY_SEC = 0.3

# How far an NHL-API startTimeUTC may sit from an existing fixture's
# match_date and still be "the same game". ESPN and the NHL API normally
# agree to the minute; the slack covers a realigned kickoff, while staying
# well under the ~24 h gap between two genuine meetings of the same pair
# (the 2020-21 schedule had back-to-back same-opponent games).
FIXTURE_TOLERANCE_HOURS = 12.0

# main() exits non-zero when at least this many games needed stats and
# none could be loaded — the systemic-breakage signal (see main()).
SYSTEMIC_FAILURE_MIN_GAMES = 3


def strip_accents(text: str) -> str:
    """'Montréal' -> 'Montreal'. NFKD-decompose and drop combining marks."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def team_match_key(name: str | None) -> str:
    """Accent- and case-insensitive team key. ESPN ingests 'Montreal
    Canadiens' while the NHL API's placeName is 'Montréal', so the two
    feeds created two `teams` rows that exact normalized_name matching can
    never join (UNIQUE(normalized_name, sport) keeps them apart)."""
    if not name:
        return ""
    return " ".join(strip_accents(name).lower().replace(".", " ").split())


def current_nhl_season(today: date) -> str:
    """Season string the NHL is in on `today`. A season opens in October and
    its playoffs end in June; July-September belongs to the season about to
    start (preseason games are filtered out by gameType anyway)."""
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start}-{start + 1}"


def resolve_season_arg(value: str | None, today: date | None = None) -> str | None:
    """'current' -> the live season string; anything else passes through."""
    if value is None:
        return None
    if value.strip().lower() == "current":
        return current_nhl_season(today or datetime.now(timezone.utc).date())
    return value


def season_from_api(season_int: int | str | None) -> str | None:
    """20262027 -> '2026-2027' (the NHL API's season id, back to ours)."""
    s = str(season_int or "")
    if len(s) != 8 or not s.isdigit():
        return None
    return f"{s[:4]}-{s[4:]}"


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def choose_fixture(
    candidates: list[dict],
    match_dt: datetime,
    nhl_game_id: int | None,
    tolerance_hours: float = FIXTURE_TOLERANCE_HOURS,
) -> dict | None:
    """Pick the existing `matches` row an NHL-API game belongs to.

    `candidates` are rows already restricted to the right (home, away)
    team-id sets; each carries `match_date`, `status` and `external_ids`.
    A row that already holds this NHL game id wins outright. Otherwise the
    row closest in time within `tolerance_hours` wins; ties prefer a row
    that is already finished, then one carrying an ESPN id (the live
    fixture), so a stale 'scheduled' placeholder never beats the real game.
    A row already claimed by a DIFFERENT NHL game id is never chosen.
    Returns None when nothing is close enough — the caller decides whether
    that means "insert" (historical backfill) or "report and skip" (live)."""
    if nhl_game_id is not None:
        for c in candidates:
            ext = c.get("external_ids") or {}
            if str(ext.get("nhl", "")) == str(nhl_game_id):
                return c
    target = _as_utc(match_dt)
    best: tuple | None = None
    best_row: dict | None = None
    for c in candidates:
        md = c.get("match_date")
        if md is None:
            continue
        gap_h = abs((_as_utc(md) - target).total_seconds()) / 3600.0
        if gap_h > tolerance_hours:
            continue
        ext = c.get("external_ids") or {}
        if ext.get("nhl") and nhl_game_id is not None and str(ext["nhl"]) != str(nhl_game_id):
            continue
        rank = (gap_h, 0 if c.get("status") == "finished" else 1, 0 if "espn" in ext else 1)
        if best is None or rank < best:
            best, best_row = rank, c
    return best_row


def load_team_alias_ids(cur) -> dict[str, set[str]]:
    """{team_match_key: {team ids}} over every NHL team row, so one NHL-API
    team resolves to all rows that are the same club under another
    spelling (the ESPN 'Montreal Canadiens' vs NHL 'Montréal Canadiens')."""
    cur.execute("SELECT id::text AS id, name FROM teams WHERE sport = 'nhl'")
    out: dict[str, set[str]] = {}
    for r in cur.fetchall():
        out.setdefault(team_match_key(r["name"]), set()).add(r["id"])
    return out


def lookup_team_by_nhl_api_id(cur, nhl_team_id: int | None) -> dict | None:
    """The NHL-API-owned teams row for an API team id (stamped by
    ensure_nhl_team). Used when a payload lacks placeName/commonName,
    which is the case for the /score/{date} feed."""
    if nhl_team_id is None:
        return None
    cur.execute(
        """
        SELECT id::text AS id, name FROM teams
        WHERE sport = 'nhl' AND (external_ids->>'nhl_api_id')::bigint = %s
        ORDER BY updated_at DESC NULLS LAST
        LIMIT 1
        """,
        (int(nhl_team_id),),
    )
    return cur.fetchone()


def find_existing_fixture(
    cur,
    home_ids: set[str],
    away_ids: set[str],
    match_dt: datetime,
    nhl_game_id: int | None,
    tolerance_hours: float = FIXTURE_TOLERANCE_HOURS,
) -> dict | None:
    """Existing NHL fixture for this game, if any — see choose_fixture."""
    window = timedelta(hours=max(tolerance_hours, 1.0))
    gid = None if nhl_game_id is None else str(nhl_game_id)
    cur.execute(
        """
        SELECT m.id::text AS id, m.home_team_id::text AS home_team_id,
               m.away_team_id::text AS away_team_id, m.match_date, m.status,
               m.home_score, m.away_score, m.external_ids
        FROM matches m
        WHERE (m.home_team_id::text = ANY(%s) AND m.away_team_id::text = ANY(%s)
               AND m.match_date BETWEEN %s AND %s)
           OR (%s::text IS NOT NULL AND m.external_ids->>'nhl' = %s::text)
        """,
        (list(home_ids), list(away_ids), _as_utc(match_dt) - window, _as_utc(match_dt) + window, gid, gid),
    )
    return choose_fixture(list(cur.fetchall()), match_dt, nhl_game_id, tolerance_hours)


def attach_to_fixture(cur, fixture: dict, game: dict, metadata: dict) -> None:
    """Stamp NHL identity + loader metadata on an EXISTING fixture.

    Additive only: external_ids gains the 'nhl' key, metadata gains the
    loader's keys without overwriting what the ESPN path already wrote
    (existing keys win), and scores/status are filled only when the row
    has no result yet — fetch_upcoming --results owns those otherwise."""
    home_score = (game.get("homeTeam") or {}).get("score")
    away_score = (game.get("awayTeam") or {}).get("score")
    cur.execute(
        """
        UPDATE matches
           SET external_ids = COALESCE(external_ids, '{}'::jsonb) || %s,
               metadata = %s::jsonb || COALESCE(metadata, '{}'::jsonb),
               status = CASE WHEN home_score IS NULL AND %s::int IS NOT NULL THEN 'finished' ELSE status END,
               home_score = COALESCE(home_score, %s),
               away_score = COALESCE(away_score, %s),
               updated_at = NOW()
         WHERE id = %s
        """,
        (Json({"nhl": str(game.get("id"))}), Json(metadata), home_score, home_score, away_score, fixture["id"]),
    )


@dataclass(frozen=True)
class TeamGameStats:
    """Subset of /right-rail teamGameStats + linescore we persist.

    `power_play_goals` / `power_play_opps` come from the API's
    "powerPlay" formatted string "G/O". `penalty_kill_*` are derived
    from the opponent's power-play numbers (opp PP opps = our PK opps,
    opp PP goals = goals against us short-handed).
    """

    shots_on_goal: int
    power_play_goals: int
    power_play_opps: int
    power_play_pct: float | None
    pim: int
    hits: int
    blocked_shots: int
    giveaways: int
    takeaways: int
    faceoffs_won: int
    faceoffs_taken: int
    faceoff_win_pct: float | None
    period_scores: dict[str, int]


def season_id(season_str: str) -> str:
    """'2024-2025' -> '20242025' (the NHL API's season-id format)."""
    parts = season_str.split("-")
    if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 4 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"Bad --seasons entry: {season_str!r} (expected YYYY-YYYY)")
    return parts[0] + parts[1]


def get_json(url: str) -> dict | None:
    """Polite GET with throttle. Returns None on 404 (skip silently)."""
    time.sleep(REQUEST_DELAY_SEC)
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException as e:
        logger.warning("GET %s failed: %s", url, e)
        return None
    if r.status_code == 404:
        return None
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        logger.warning("GET %s -> HTTP %d: %s", url, r.status_code, e)
        return None
    try:
        return r.json()
    except ValueError:
        logger.warning("GET %s -> non-JSON body", url)
        return None


def parse_wins_taken(value: str | None) -> tuple[int, int]:
    """Parse the API's "wins/taken" formatted strings like "34/59" or
    "2/4". Returns (0, 0) on malformed input — we still record the
    game even if one stat string is missing."""
    if not value or "/" not in value:
        return 0, 0
    try:
        w, t = value.split("/", 1)
        return int(w), int(t)
    except ValueError:
        return 0, 0


def parse_team_game_stats(rr_payload: dict, side: str) -> TeamGameStats:
    """Extract one team's TeamGameStats from a /right-rail response.

    `side` is 'awayValue' or 'homeValue' — the field name the right-rail
    feed uses for each team's value in the teamGameStats list.
    """
    stats = {row["category"]: row.get(side) for row in rr_payload.get("teamGameStats") or []}

    pp_goals, pp_opps = parse_wins_taken(stats.get("powerPlay"))
    fo_wins, fo_taken = parse_wins_taken(stats.get("faceoffWins"))

    # Period scores: keyed by period number ("1", "2", "3", "OT") so the
    # NHL feature pipeline can derive regulation totals without re-reading
    # the right-rail payload. SO goals aren't included in the linescore —
    # they only affect the final score and we capture that via
    # matches.{home,away}_score + matches.metadata.period_type_final.
    period_scores: dict[str, int] = {}
    linescore = rr_payload.get("linescore") or {}
    score_key = "home" if side == "homeValue" else "away"
    for period in linescore.get("byPeriod") or []:
        desc = period.get("periodDescriptor") or {}
        ptype = desc.get("periodType")
        pnum = desc.get("number")
        # Regular periods: "1", "2", "3". Overtime: "OT".
        if ptype == "OT":
            key = "OT"
        elif pnum is not None:
            key = str(pnum)
        else:
            continue
        period_scores[key] = int(period.get(score_key) or 0)

    return TeamGameStats(
        shots_on_goal=int(stats.get("sog") or 0),
        power_play_goals=pp_goals,
        power_play_opps=pp_opps,
        power_play_pct=float(stats["powerPlayPctg"]) if stats.get("powerPlayPctg") is not None else None,
        pim=int(stats.get("pim") or 0),
        hits=int(stats.get("hits") or 0),
        blocked_shots=int(stats.get("blockedShots") or 0),
        giveaways=int(stats.get("giveaways") or 0),
        takeaways=int(stats.get("takeaways") or 0),
        faceoffs_won=fo_wins,
        faceoffs_taken=fo_taken,
        faceoff_win_pct=float(stats["faceoffWinningPctg"]) if stats.get("faceoffWinningPctg") is not None else None,
        period_scores=period_scores,
    )


def _parse_starting_goalie(team_block: dict) -> dict | None:
    """From boxscore.playerByGameStats.{home,away}Team, find the goalie
    marked starter=True and return a canonical {playerId, name} dict.
    Returns None if no starter is flagged or the team has no goalies
    (very rare — e.g., truly bizarre data error).
    """
    goalies = team_block.get("goalies") or []
    for g in goalies:
        if not g.get("starter"):
            continue
        pid = g.get("playerId")
        if pid is None:
            continue
        name_field = g.get("name") or {}
        name = name_field.get("default") if isinstance(name_field, dict) else str(name_field)
        return {"nhl_player_id": int(pid), "name": name or f"NHL Player {pid}"}
    return None


def extract_starting_goalies(boxscore_payload: dict) -> tuple[dict | None, dict | None]:
    """Pull (home_starter, away_starter) from a /boxscore response.
    Both elements may be None if the API omits the flag for that side."""
    pbg = boxscore_payload.get("playerByGameStats") or {}
    home = _parse_starting_goalie(pbg.get("homeTeam") or {})
    away = _parse_starting_goalie(pbg.get("awayTeam") or {})
    return home, away


def upsert_goalie_player(cur, goalie: dict, team_id: str | None) -> str | None:
    """Insert-or-update a players row for an NHL goalie keyed on the
    stable NHL player id (external_ids->>'nhl_player_id'). Returns the
    players.id. The unique partial index added in migration 008 makes
    the ON CONFLICT clause work.
    """
    name = goalie["name"]
    norm = " ".join(name.strip().lower().split())
    cur.execute(
        """
        INSERT INTO players (name, normalized_name, team_id, position, external_ids)
        VALUES (%s, %s, %s, 'G', %s)
        ON CONFLICT ((external_ids ->> 'nhl_player_id'))
            WHERE external_ids ? 'nhl_player_id'
        DO UPDATE
            SET team_id = COALESCE(EXCLUDED.team_id, players.team_id),
                name = EXCLUDED.name,
                normalized_name = EXCLUDED.normalized_name,
                updated_at = NOW()
        RETURNING id
        """,
        (name, norm, team_id, Json({"nhl_player_id": goalie["nhl_player_id"]})),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def fetch_boxscore_goalies(game_id: int) -> tuple[dict | None, dict | None] | None:
    """One /boxscore call to extract (home, away) starting goalies.
    Returns None on graceful skip (404/timeout); caller treats this as
    'no goalie data' and inserts NULL for that game.
    """
    payload = get_json(f"{NHL_API}/gamecenter/{game_id}/boxscore")
    if payload is None:
        return None
    return extract_starting_goalies(payload)


def regulation_outcome(home_periods: dict[str, int], away_periods: dict[str, int]) -> tuple[int, int, str]:
    """Sum periods 1-3 only (exclude OT/SO) to get the regulation score,
    then label the regulation winner. Used downstream for both the
    moneyline target (incl. OT/SO winner via matches.{home,away}_score)
    and the 60-minute regulation 3-way target."""
    reg_home = sum(home_periods.get(str(p), 0) for p in (1, 2, 3))
    reg_away = sum(away_periods.get(str(p), 0) for p in (1, 2, 3))
    if reg_home > reg_away:
        winner = "home"
    elif reg_away > reg_home:
        winner = "away"
    else:
        winner = "tie"
    return reg_home, reg_away, winner


def ensure_nhl_league(cur) -> str:
    cur.execute(
        """
        INSERT INTO leagues (name, country, sport, external_ids)
        VALUES ('NHL', 'USA', 'nhl', %s)
        ON CONFLICT (name, country, sport) DO UPDATE
            SET external_ids = leagues.external_ids || EXCLUDED.external_ids
        RETURNING id
        """,
        (Json({"nhl_api": "NHL"}),),
    )
    return cur.fetchone()["id"]


def ensure_nhl_team(cur, team_payload: dict, league_id: str) -> str | None:
    """Resolve an NHL team payload (from schedule's awayTeam/homeTeam) to
    a teams.id. NHL names are canonical, so we skip the soccer fuzzy
    path and rely on (normalized_name, sport='nhl') uniqueness."""
    place = (team_payload.get("placeName") or {}).get("default")
    common = (team_payload.get("commonName") or {}).get("default")
    abbrev = team_payload.get("abbrev")
    nhl_id = team_payload.get("id")
    if not (place and common):
        return None
    full = f"{place} {common}"
    norm = " ".join(full.strip().lower().split())
    # Reuse the club's row under any accent spelling (ESPN's "Montreal
    # Canadiens" vs our "Montréal Canadiens") before inserting a twin.
    cur.execute("SELECT id FROM teams WHERE normalized_name = %s AND sport = 'nhl'", (norm,))
    if cur.fetchone() is None:
        folded_id = team_identity.find_team_accent_insensitive(cur, "nhl", norm)
        if folded_id:
            cur.execute(
                "UPDATE teams SET external_ids = COALESCE(external_ids, '{}'::jsonb) || %s, "
                "league_id = COALESCE(league_id, %s) WHERE id = %s",
                (Json({"nhl_api_id": nhl_id, "nhl_api_abbrev": abbrev}), league_id, folded_id),
            )
            return folded_id
    cur.execute(
        """
        INSERT INTO teams (name, normalized_name, league_id, sport, external_ids)
        VALUES (%s, %s, %s, 'nhl', %s)
        ON CONFLICT (normalized_name, sport) DO UPDATE
            SET league_id = COALESCE(teams.league_id, EXCLUDED.league_id),
                external_ids = teams.external_ids || EXCLUDED.external_ids
        RETURNING id
        """,
        (full, norm, league_id, Json({"nhl_api_id": nhl_id, "nhl_api_abbrev": abbrev})),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def build_match_metadata(game: dict, reg_home: int, reg_away: int, reg_winner: str) -> dict:
    """matches.metadata keys this loader owns (regulation score + final
    period type for the regulation/OT markets, game type, NHL game id)."""
    return {
        "regulation_home_score": reg_home,
        "regulation_away_score": reg_away,
        "regulation_winner": reg_winner,
        "period_type_final": (game.get("gameOutcome") or {}).get("lastPeriodType") or "REG",
        "game_type": "playoff" if game.get("gameType") == GAME_TYPE_PLAYOFF else "regular",
        "nhl_game_id": game.get("id"),
    }


def upsert_match(
    cur,
    game: dict,
    league_id: str,
    home_id: str,
    away_id: str,
    reg_home: int,
    reg_away: int,
    reg_winner: str,
    season: str,
) -> str | None:
    """Upsert one game's row in `matches`. Returns the match id.

    Final score = NHL `awayTeam.score` / `homeTeam.score` (includes OT
    and SO goals). Regulation scores + final-period type land in
    matches.metadata for the feature pipeline.
    """
    try:
        match_dt = datetime.fromisoformat((game["startTimeUTC"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return None

    home_score = (game.get("homeTeam") or {}).get("score")
    away_score = (game.get("awayTeam") or {}).get("score")
    venue = (game.get("venue") or {}).get("default")
    metadata = build_match_metadata(game, reg_home, reg_away, reg_winner)

    cur.execute(
        """
        INSERT INTO matches (
            league_id, home_team_id, away_team_id, match_date, status,
            home_score, away_score, season, venue, external_ids, metadata
        )
        VALUES (%s, %s, %s, %s, 'finished', %s, %s, %s, %s, %s, %s)
        ON CONFLICT (home_team_id, away_team_id, match_date) DO UPDATE
            SET status = 'finished',
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score,
                venue = COALESCE(matches.venue, EXCLUDED.venue),
                external_ids = matches.external_ids || EXCLUDED.external_ids,
                metadata = matches.metadata || EXCLUDED.metadata,
                updated_at = NOW()
        RETURNING id
        """,
        (
            league_id,
            home_id,
            away_id,
            match_dt,
            home_score,
            away_score,
            season,
            venue,
            Json({"nhl": str(game.get("id"))}),
            Json(metadata),
        ),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def upsert_team_stats(cur, match_id: str, team_id: str, stats: TeamGameStats, opp_stats: TeamGameStats) -> None:
    """Persist one team's NHL stats. Goalie-side numbers (saves, save_pct,
    goals_against) are derived from the opponent's shots and goals — the
    right-rail feed doesn't expose them as team-level stats directly."""
    goals = sum(stats.period_scores.values())
    goals_against = sum(opp_stats.period_scores.values())
    shots_faced = opp_stats.shots_on_goal
    saves = max(shots_faced - goals_against, 0)
    save_pct = (saves / shots_faced) if shots_faced > 0 else None
    pk_pct = 1.0 - (opp_stats.power_play_goals / opp_stats.power_play_opps) if opp_stats.power_play_opps > 0 else None
    cur.execute(
        """
        INSERT INTO nhl_match_stats (
            match_id, team_id,
            shots_on_goal, goals,
            saves, save_pct, goals_against,
            power_play_goals, power_play_opportunities, power_play_pct,
            penalty_kill_goals_against, penalty_kill_opportunities, penalty_kill_pct,
            hits, blocked_shots,
            faceoffs_won, faceoffs_taken, faceoff_win_pct,
            giveaways, takeaways, pim,
            period_scores
        )
        VALUES (%s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s)
        ON CONFLICT (match_id, team_id) DO UPDATE SET
            shots_on_goal = EXCLUDED.shots_on_goal,
            goals = EXCLUDED.goals,
            saves = EXCLUDED.saves,
            save_pct = EXCLUDED.save_pct,
            goals_against = EXCLUDED.goals_against,
            power_play_goals = EXCLUDED.power_play_goals,
            power_play_opportunities = EXCLUDED.power_play_opportunities,
            power_play_pct = EXCLUDED.power_play_pct,
            penalty_kill_goals_against = EXCLUDED.penalty_kill_goals_against,
            penalty_kill_opportunities = EXCLUDED.penalty_kill_opportunities,
            penalty_kill_pct = EXCLUDED.penalty_kill_pct,
            hits = EXCLUDED.hits,
            blocked_shots = EXCLUDED.blocked_shots,
            faceoffs_won = EXCLUDED.faceoffs_won,
            faceoffs_taken = EXCLUDED.faceoffs_taken,
            faceoff_win_pct = EXCLUDED.faceoff_win_pct,
            giveaways = EXCLUDED.giveaways,
            takeaways = EXCLUDED.takeaways,
            pim = EXCLUDED.pim,
            period_scores = EXCLUDED.period_scores
        """,
        (
            match_id,
            team_id,
            stats.shots_on_goal,
            goals,
            saves,
            save_pct,
            goals_against,
            stats.power_play_goals,
            stats.power_play_opps,
            stats.power_play_pct,
            opp_stats.power_play_goals,
            opp_stats.power_play_opps,
            pk_pct,
            stats.hits,
            stats.blocked_shots,
            stats.faceoffs_won,
            stats.faceoffs_taken,
            stats.faceoff_win_pct,
            stats.giveaways,
            stats.takeaways,
            stats.pim,
            Json(stats.period_scores),
        ),
    )


def already_loaded_game_ids(cur, game_ids: list[int]) -> set[int]:
    """NHL game ids (of `game_ids`) whose fixture already carries stats.
    Keyed on the game id, not the season string, because an attached ESPN
    fixture's season label is ESPN's, not the loader's. Used to skip
    re-fetching on resumed runs and on every live tick."""
    if not game_ids:
        return set()
    cur.execute(
        """
        SELECT DISTINCT (m.external_ids->>'nhl')::bigint AS nhl_id
        FROM matches m
        JOIN nhl_match_stats s ON s.match_id = m.id
        WHERE m.external_ids->>'nhl' = ANY(%s)
        """,
        ([str(g) for g in game_ids],),
    )
    return {row["nhl_id"] for row in cur.fetchall() if row.get("nhl_id") is not None}


def discover_recent_games(days: int, today: date | None = None) -> dict[int, dict]:
    """{game_id: game} for regular-season + playoff games that went FINAL
    on any of the last `days` UTC dates (today included), via the per-date
    /score feed: one request per day instead of 32 club schedules, cheap
    enough for the 15-minute pipeline tick."""
    today = today or datetime.now(timezone.utc).date()
    games: dict[int, dict] = {}
    for back in range(days + 1):
        d = today - timedelta(days=back)
        payload = get_json(f"{NHL_API}/score/{d.isoformat()}")
        for g in (payload or {}).get("games") or []:
            if g.get("gameType") not in (GAME_TYPE_REGULAR, GAME_TYPE_PLAYOFF):
                continue
            if (g.get("gameState") or "").upper() not in FINAL_STATES:
                continue
            if g.get("id") is not None:
                games.setdefault(int(g["id"]), g)
    return games


def discover_games(season_str: str) -> dict[int, dict]:
    """Walk every team's season schedule and return {game_id: game} for
    regular-season + playoff games in FINAL state. Inactive teams in
    that season silently no-op."""
    games: dict[int, dict] = {}
    sid = season_id(season_str)
    for tri in NHL_TEAM_CODES:
        payload = get_json(f"{NHL_API}/club-schedule-season/{tri}/{sid}")
        if not payload:
            continue
        for g in payload.get("games") or []:
            if g.get("gameType") not in (GAME_TYPE_REGULAR, GAME_TYPE_PLAYOFF):
                continue
            if (g.get("gameState") or "").upper() not in FINAL_STATES:
                continue
            gid = g.get("id")
            if gid is None:
                continue
            games.setdefault(int(gid), g)
        logger.info("season %s team %s: %d cumulative finished games", season_str, tri, len(games))
    return games


# How often to log progress inside the per-game loop. The discovery
# step is chatty enough on its own; here we just need a heartbeat every
# ~30s of work (0.3s/req × 100 games ≈ 30s) so a multi-season run
# doesn't look hung between season-boundary banners.
GAME_LOG_INTERVAL = 100


def _resolve_team(cur, team_payload: dict, league_id: str) -> dict | None:
    """NHL-API team payload -> {id, name} of the loader-owned teams row.
    Club-schedule payloads carry placeName/commonName (ensure_nhl_team can
    create/refresh the row); /score payloads only carry the API id."""
    if (team_payload.get("placeName") or {}).get("default") and (team_payload.get("commonName") or {}).get("default"):
        tid = ensure_nhl_team(cur, team_payload, league_id)
        if tid:
            place = team_payload["placeName"]["default"]
            common = team_payload["commonName"]["default"]
            return {"id": str(tid), "name": f"{place} {common}"}
    return lookup_team_by_nhl_api_id(cur, team_payload.get("id"))


def _alias_ids(aliases: dict[str, set[str]], team: dict) -> set[str]:
    return set(aliases.get(team_match_key(team["name"]), set())) | {team["id"]}


def load_games(
    database_url: str,
    games: dict[int, dict],
    label: str,
    refetch: bool,
    attach_only: bool,
    dry_run: bool = False,
) -> dict[str, int]:
    """Load right-rail stats for `games` and attach them to fixtures.

    Every game is first matched against EXISTING fixtures (same club under
    any spelling, kickoff within FIXTURE_TOLERANCE_HOURS, or the same NHL
    game id). A match is attached to and its stats are keyed to THAT
    fixture's own team ids, so the ESPN-created live-season rows get the
    stats the feature builder reads. Only when nothing matches does the
    historical path insert a new fixture; with attach_only (the live job)
    an unmatched game is logged as an error and skipped — the live job
    must never mint a second row for a game ESPN already owns."""
    counts = {
        "games_seen": len(games),
        "games_loaded": 0,
        "games_attached": 0,
        "games_inserted": 0,
        "games_skipped": 0,
        "games_unmatched": 0,
        "games_failed": 0,
    }
    if not games:
        logger.info("=== %s: no finished games discovered ===", label)
        return counts

    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            league_id = ensure_nhl_league(cur)
            already = set() if refetch else already_loaded_game_ids(cur, list(games))
            aliases = load_team_alias_ids(cur)
            total = len(games)
            logger.info("=== %s: %d games to consider (%d already loaded) ===", label, total, len(already))

            for processed, (gid, game) in enumerate(sorted(games.items()), start=1):
                if gid in already:
                    counts["games_skipped"] += 1
                else:
                    try:
                        _load_one_game(cur, conn, gid, game, league_id, aliases, attach_only, dry_run, counts)
                    except Exception as e:  # noqa: BLE001 — one bad game shouldn't kill the run
                        conn.rollback()
                        logger.error("game %s failed: %s", gid, e)
                        counts["games_failed"] += 1

                # Heartbeat so a long silent loop looks like progress
                # instead of a hang.
                if processed % GAME_LOG_INTERVAL == 0 or processed == total:
                    logger.info("%s: processed %d/%d %s", label, processed, total, counts)
            if dry_run:
                conn.rollback()
    return counts


def _load_one_game(cur, conn, gid, game, league_id, aliases, attach_only, dry_run, counts) -> None:
    try:
        match_dt = datetime.fromisoformat(game["startTimeUTC"].replace("Z", "+00:00"))
    except (KeyError, ValueError):
        counts["games_failed"] += 1
        return
    home = _resolve_team(cur, game.get("homeTeam") or {}, league_id)
    away = _resolve_team(cur, game.get("awayTeam") or {}, league_id)
    if not (home and away):
        logger.error("game %s: NHL team ids not resolvable (%s / %s)", gid, game.get("homeTeam"), game.get("awayTeam"))
        counts["games_failed"] += 1
        return
    fixture = find_existing_fixture(cur, _alias_ids(aliases, home), _alias_ids(aliases, away), match_dt, gid)
    if fixture is None and attach_only:
        logger.error(
            "game %s (%s @ %s, %s): no existing fixture to attach to — skipped (attach-only)",
            gid,
            away["name"],
            home["name"],
            match_dt.isoformat(),
        )
        counts["games_unmatched"] += 1
        return

    rr = get_json(f"{NHL_API}/gamecenter/{gid}/right-rail")
    if rr is None:
        counts["games_failed"] += 1
        return
    home_stats = parse_team_game_stats(rr, "homeValue")
    away_stats = parse_team_game_stats(rr, "awayValue")
    reg_home, reg_away, reg_winner = regulation_outcome(home_stats.period_scores, away_stats.period_scores)

    if fixture is not None:
        attach_to_fixture(cur, fixture, game, build_match_metadata(game, reg_home, reg_away, reg_winner))
        match_id, home_tid, away_tid = fixture["id"], fixture["home_team_id"], fixture["away_team_id"]
        counts["games_attached"] += 1
    else:
        season_str = season_from_api(game.get("season")) or current_nhl_season(match_dt.date())
        match_id = upsert_match(
            cur, game, league_id, home["id"], away["id"], reg_home, reg_away, reg_winner, season_str
        )
        if not match_id:
            counts["games_failed"] += 1
            return
        home_tid, away_tid = home["id"], away["id"]
        counts["games_inserted"] += 1

    upsert_team_stats(cur, match_id, home_tid, home_stats, away_stats)
    upsert_team_stats(cur, match_id, away_tid, away_stats, home_stats)
    counts["games_loaded"] += 1
    # Commit per game so a crash mid-season leaves a usable partial
    # backfill instead of an aborted transaction (dry runs roll back).
    if dry_run:
        conn.rollback()
    else:
        conn.commit()


def load_season(
    database_url: str, season_str: str, refetch: bool, attach_only: bool = False, dry_run: bool = False
) -> dict[str, int]:
    logger.info("=== season %s: discovering games ===", season_str)
    return load_games(database_url, discover_games(season_str), f"season {season_str}", refetch, attach_only, dry_run)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--seasons",
        help="Comma-separated season strings (e.g. 2024-2025,2023-2024), or 'current'.",
    )
    src.add_argument(
        "--recent-days",
        type=int,
        help="Live mode: games that went final on the last N UTC days (per-date /score feed).",
    )
    p.add_argument(
        "--attach-only",
        action="store_true",
        help="Never insert a new fixture: attach stats to existing rows only (the live job).",
    )
    p.add_argument("--dry-run", action="store_true", help="Match and fetch, report counts, write nothing.")
    p.add_argument(
        "--refetch",
        action="store_true",
        help="Re-fetch right-rail stats for games already in nhl_match_stats.",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set and --database-url not provided")
        return 2
    runs: list[dict[str, int]] = []
    if args.recent_days is not None:
        counts = load_games(
            args.database_url,
            discover_recent_games(args.recent_days),
            f"last {args.recent_days}d",
            args.refetch,
            args.attach_only,
            args.dry_run,
        )
        logger.info("recent %dd: %s", args.recent_days, counts)
        runs.append(counts)
    else:
        seasons = [resolve_season_arg(s.strip()) for s in args.seasons.split(",") if s.strip()]
        for season in seasons:
            counts = load_season(args.database_url, season, args.refetch, args.attach_only, args.dry_run)
            logger.info("season %s: %s", season, counts)
            runs.append(counts)
    # Loud failure for the systemic case: several games needed loading and
    # NONE could be loaded (team map, fixture identity or the API broken).
    # One stubborn game is logged at ERROR above every tick but does not
    # fail the pipeline — a transient right-rail miss would otherwise page.
    needed = sum(c["games_seen"] - c["games_skipped"] for c in runs)
    loaded = sum(c["games_loaded"] for c in runs)
    if needed >= SYSTEMIC_FAILURE_MIN_GAMES and loaded == 0 and not args.dry_run:
        logger.error("%d finished NHL games needed stats and none were loaded", needed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
