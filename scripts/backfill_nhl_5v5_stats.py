"""Backfill 5v5 even-strength advanced stats from NHL play-by-play.

Phase 3d found puck-line and total models hit a feature-set ceiling
that goalies + pace alone couldn't break. The remaining headroom is in
5v5 shot-attempt data — Corsi differential at even strength is the
canonical NHL underlying-quality metric. Teams that out-Corsi opponents
at 5v5 tend to outscore them going forward, regardless of recent
goal-scoring variance.

For each finished NHL game lacking an nhl_match_advanced_stats row,
this script:
  1. Calls /gamecenter/{id}/play-by-play (free, ~140KB per game)
  2. Walks the plays, filters to 5v5 even-strength (situationCode='1551')
  3. Aggregates per team: shot attempts (Corsi), unblocked shots
     (Fenwick), shots on goal, goals
  4. Scores each shot with a distance-based xG approximation and sums
     per team
  5. Writes one row per (match_id, team_id) to nhl_match_advanced_stats

xG approximation (v1): xG = base_rate(shot_type) * exp(-distance / 30),
where base_rate is calibrated from NHL-wide shot-type conversion rates.
A properly trained shot-location logistic regression is Phase 3e.1
work; this approximation is enough to test whether 5v5 features unlock
puck-line/total improvements.

Idempotent via the UNIQUE(match_id, team_id) constraint on the target
table. Re-runs skip games that already have rows by default; --refetch
overrides for refreshing stale data.

Quota: free, no API key. ~6,500 games × 0.5-1s each = 60-120 min.

Usage:
    # Backfill all finished NHL games missing 5v5 stats
    python scripts/backfill_nhl_5v5_stats.py

    # Limit to one season
    python scripts/backfill_nhl_5v5_stats.py --season 2024-2025

    # Refetch even for games that already have rows
    python scripts/backfill_nhl_5v5_stats.py --season 2024-2025 --refetch
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_nhl_5v5_stats")

# Reuse the load_nhl_historical NHL HTTP helper without making scripts/
# a package — same pattern as the other backfill scripts.
_LOADER_PATH = Path(__file__).resolve().parent / "load_nhl_historical.py"
_spec = importlib.util.spec_from_file_location("_load_nhl_historical", _LOADER_PATH)
assert _spec and _spec.loader
load_nhl = importlib.util.module_from_spec(_spec)
sys.modules["_load_nhl_historical"] = load_nhl
_spec.loader.exec_module(load_nhl)


NHL_API = load_nhl.NHL_API
GAME_LOG_INTERVAL = 100

# 5v5 situation code in the NHL play-by-play feed. Format is a 4-digit
# string [away_goalie, away_skaters, home_skaters, home_goalie] where
# 1 = present, 0 = pulled. 1551 = both goalies in net + 5 skaters each
# = canonical even-strength play (the analytics-community filter for
# isolating team-quality signal from special teams + empty-net noise).
EVEN_STRENGTH_5V5 = "1551"

# Shot-event types that count toward Corsi (shot attempts). The full
# Corsi numerator is SOG + missed + blocked; Fenwick excludes blocked.
SHOT_EVENT_TYPES = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}
UNBLOCKED_TYPES = {"shot-on-goal", "goal", "missed-shot"}

# Base goal-conversion rates per NHL shot type (rough league-wide
# averages from public sources). The xG approximation multiplies the
# base by exp(-distance / 30) so close-range wrist shots get the
# highest xG and long slap shots from the point get the lowest. Real
# NHL xG uses ~15 features; this is a stopgap until a proper shot-
# location logistic regression lands (Phase 3e.1).
XG_BASE_RATE_BY_TYPE: dict[str, float] = {
    "wrist": 0.10,
    "snap": 0.09,
    "slap": 0.07,
    "backhand": 0.10,
    "tip-in": 0.18,
    "deflected": 0.15,
    "wrap-around": 0.06,
    # Sane fallback for any shot type not in the table.
    "_default": 0.09,
}

# NHL net is at (89, 0) on the offensive half of the rink. Coordinates
# in the play-by-play are in the team's offensive direction, so for both
# teams we measure distance to the same target.
NET_X = 89.0
NET_Y = 0.0


def shot_xg(shot_type: str | None, x_coord: float | None, y_coord: float | None) -> float:
    """Approximate expected goals for a single shot. Returns 0 when the
    coordinates are missing (rare — usually penalty-shot adjudicated
    events that don't have spatial data)."""
    if x_coord is None or y_coord is None:
        return 0.0
    base = XG_BASE_RATE_BY_TYPE.get((shot_type or "").lower(), XG_BASE_RATE_BY_TYPE["_default"])
    distance = math.sqrt((NET_X - float(x_coord)) ** 2 + (NET_Y - float(y_coord)) ** 2)
    # The exp(-d/30) decay gives ~0.85 at 5 feet, ~0.51 at 20 feet,
    # ~0.19 at 50 feet — broadly matching shooter conversion rates.
    return base * math.exp(-distance / 30.0)


def aggregate_5v5(plays: list[dict]) -> dict[int, dict[str, float]]:
    """Walk a /play-by-play response's plays array and aggregate 5v5
    even-strength shot stats per NHL team id. Returns
    {nhl_team_id: {corsi, fenwick, sog, goals, xg}}."""
    aggregates: dict[int, dict[str, float]] = {}
    for play in plays:
        if play.get("typeDescKey") not in SHOT_EVENT_TYPES:
            continue
        if play.get("situationCode") != EVEN_STRENGTH_5V5:
            continue
        details = play.get("details") or {}
        team_id = details.get("eventOwnerTeamId")
        if team_id is None:
            continue
        bucket = aggregates.setdefault(
            int(team_id),
            {"corsi": 0, "fenwick": 0, "sog": 0, "goals": 0, "xg": 0.0},
        )
        bucket["corsi"] += 1
        type_key = play.get("typeDescKey")
        if type_key in UNBLOCKED_TYPES:
            bucket["fenwick"] += 1
            # xG is only meaningful for unblocked shots — blocked shots
            # don't reach the net so their conversion probability is 0.
            bucket["xg"] += shot_xg(
                details.get("shotType"),
                details.get("xCoord"),
                details.get("yCoord"),
            )
        if type_key in ("shot-on-goal", "goal"):
            bucket["sog"] += 1
        if type_key == "goal":
            bucket["goals"] += 1
    return aggregates


def list_games_needing_5v5(conn, season: str | None, refetch: bool) -> list[tuple[str, int, str, str]]:
    """Eligible games are finished NHL games with an external NHL game
    id and no nhl_match_advanced_stats row (unless --refetch). Returns
    rows of (match_id, nhl_game_id, home_team_id, away_team_id)."""
    where_extras = "AND m.season = %s" if season else ""
    join_filter = (
        "LEFT JOIN nhl_match_advanced_stats a ON a.match_id = m.id"
        if not refetch
        else "LEFT JOIN nhl_match_advanced_stats a ON FALSE"
    )
    having = "AND a.id IS NULL" if not refetch else ""
    params: list = []
    if season:
        params.append(season)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT DISTINCT
                m.id::text AS match_id,
                (m.external_ids->>'nhl')::bigint AS nhl_game_id,
                m.home_team_id::text AS home_team_id,
                m.away_team_id::text AS away_team_id,
                m.match_date
            FROM matches m
            JOIN leagues l ON l.id = m.league_id
            {join_filter}
            WHERE m.status = 'finished'
              AND l.sport = 'nhl'
              AND m.external_ids ? 'nhl'
              {having}
              {where_extras}
            ORDER BY m.match_date ASC
            """,
            params,
        )
        return [(r["match_id"], r["nhl_game_id"], r["home_team_id"], r["away_team_id"]) for r in cur.fetchall()]


def lookup_team_id_by_nhl_id(cur, nhl_team_id: int) -> str | None:
    """Map an NHL API team id back to our internal teams.id by joining
    on external_ids->>'nhl_api_id'. load_nhl_historical.ensure_nhl_team
    stamps this on every NHL teams row."""
    cur.execute(
        """
        SELECT id::text AS id
        FROM teams
        WHERE sport = 'nhl'
          AND (external_ids->>'nhl_api_id')::bigint = %s
        LIMIT 1
        """,
        (nhl_team_id,),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def upsert_5v5_row(
    cur,
    match_id: str,
    team_id: str,
    corsi: int,
    fenwick: int,
    sog: int,
    goals: int,
    xg: float,
) -> None:
    cur.execute(
        """
        INSERT INTO nhl_match_advanced_stats
            (match_id, team_id, shot_attempts_5v5, unblocked_shots_5v5,
             shots_on_goal_5v5, goals_5v5, xg_5v5)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (match_id, team_id) DO UPDATE
            SET shot_attempts_5v5 = EXCLUDED.shot_attempts_5v5,
                unblocked_shots_5v5 = EXCLUDED.unblocked_shots_5v5,
                shots_on_goal_5v5 = EXCLUDED.shots_on_goal_5v5,
                goals_5v5 = EXCLUDED.goals_5v5,
                xg_5v5 = EXCLUDED.xg_5v5
        """,
        (match_id, team_id, corsi, fenwick, sog, goals, round(xg, 3)),
    )


def backfill(database_url: str, season: str | None, refetch: bool) -> dict:
    counts = {
        "games_seen": 0,
        "games_skipped_no_pbp": 0,
        "games_with_no_5v5_data": 0,
        "rows_written": 0,
    }
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            games = list_games_needing_5v5(conn, season, refetch)
            counts["games_seen"] = len(games)
            if not games:
                logger.info("No games need 5v5 backfill (season=%s refetch=%s)", season, refetch)
                return counts

            logger.info(
                "=== Backfilling 5v5 advanced stats for %d games (season=%s refetch=%s) ===",
                len(games),
                season,
                refetch,
            )

            for i, (match_id, nhl_game_id, home_team_id, away_team_id) in enumerate(games, start=1):
                payload = load_nhl.get_json(f"{NHL_API}/gamecenter/{nhl_game_id}/play-by-play")
                if payload is None:
                    counts["games_skipped_no_pbp"] += 1
                else:
                    plays = payload.get("plays") or []
                    by_nhl_team = aggregate_5v5(plays)
                    if not by_nhl_team:
                        counts["games_with_no_5v5_data"] += 1
                    else:
                        # Map the play-by-play's NHL API team ids to our
                        # internal teams.id. We already know home and away
                        # for this match — fall back to a per-team lookup
                        # for safety in case the NHL API uses a different
                        # id (rare but happens for relocated franchises).
                        for nhl_tid, stats in by_nhl_team.items():
                            internal_tid = lookup_team_id_by_nhl_id(cur, nhl_tid)
                            if internal_tid is None:
                                logger.warning(
                                    "game %s: NHL team id %s not in teams table — skipping team",
                                    nhl_game_id,
                                    nhl_tid,
                                )
                                continue
                            # Defensive: only write if it matches one of
                            # the match's two teams (filters out data
                            # errors where a stray team id leaks in).
                            if internal_tid not in (home_team_id, away_team_id):
                                continue
                            upsert_5v5_row(
                                cur,
                                match_id,
                                internal_tid,
                                stats["corsi"],
                                stats["fenwick"],
                                stats["sog"],
                                stats["goals"],
                                stats["xg"],
                            )
                            counts["rows_written"] += 1
                        conn.commit()

                if i % GAME_LOG_INTERVAL == 0 or i == len(games):
                    logger.info(
                        "progress: %d/%d (rows_written=%d no_pbp=%d no_5v5=%d)",
                        i,
                        len(games),
                        counts["rows_written"],
                        counts["games_skipped_no_pbp"],
                        counts["games_with_no_5v5_data"],
                    )

    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--season",
        default=None,
        help="Limit backfill to one season (e.g. 2024-2025). Default: all seasons.",
    )
    p.add_argument(
        "--refetch",
        action="store_true",
        help="Re-fetch even for games that already have 5v5 rows.",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    counts = backfill(args.database_url, args.season, args.refetch)
    logger.info("Done. %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
