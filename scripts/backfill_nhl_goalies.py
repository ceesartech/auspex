"""Retroactively populate starting_goalie_id on nhl_match_stats for
games already loaded via load_nhl_historical.py.

Phase 3c training revealed that without goalie-specific signal the
puck-line and total models can't beat naive baselines — the moneyline
feature in odds correlates with WINNER but tells you almost nothing
about MARGIN or TOTAL GOALS. The starting goalie is the single biggest
factor in scoring environment (a Vezina starter changes expected goals
against by ~1.0/game vs a backup). Phase 2's nhl_match_stats schema
already has the starting_goalie_id column; this script populates it.

For each finished NHL game without a starting_goalie_id, we:
  1. Call /gamecenter/{id}/boxscore to extract the starter per team
  2. Upsert a players row (position='G') keyed on the NHL player id
  3. Update nhl_match_stats.starting_goalie_id for that (match, team)

Idempotent via the ON CONFLICT clauses and the WHERE filter that skips
already-populated rows. Re-runs are safe and pick up newly-added games.

Quota: ~0.3s/game × ~6,500 NHL games in the corpus = ~32-35 min for
the first full backfill. Public NHL API (free, no key).

Usage:
    # Backfill goalies for every finished NHL game lacking one
    python scripts/backfill_nhl_goalies.py

    # Limit to a season for testing
    python scripts/backfill_nhl_goalies.py --season 2024-2025

    # Override the existing values (refetch even if goalie already set)
    python scripts/backfill_nhl_goalies.py --season 2024-2025 --refetch
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_nhl_goalies")

# Reuse the goalie extraction + upsert + HTTP helpers from
# load_nhl_historical.py without making scripts/ a package.
_LOADER_PATH = Path(__file__).resolve().parent / "load_nhl_historical.py"
_spec = importlib.util.spec_from_file_location("_load_nhl_historical", _LOADER_PATH)
assert _spec and _spec.loader
load_nhl = importlib.util.module_from_spec(_spec)
sys.modules["_load_nhl_historical"] = load_nhl
_spec.loader.exec_module(load_nhl)

# How often to log progress so a long silent backfill doesn't look hung.
GAME_LOG_INTERVAL = 100


def list_games_needing_goalies(conn, season: str | None, refetch: bool) -> list[tuple[str, int, str, str]]:
    """Return rows of (match_id, nhl_game_id, home_team_id, away_team_id)
    for finished NHL games whose nhl_match_stats rows still lack a
    starting_goalie_id. With refetch=True, drops the NULL filter so
    every game in scope gets re-fetched (used to refresh stale data)."""
    where_status = "m.status = 'finished'"
    where_sport = "l.sport = 'nhl'"
    where_has_nhl_id = "m.external_ids ? 'nhl'"
    where_goalie_filter = "" if refetch else "AND s.starting_goalie_id IS NULL"
    where_season = "AND m.season = %s" if season else ""

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        params: list = []
        if season:
            params.append(season)
        cur.execute(
            f"""
            SELECT DISTINCT
                m.id::text AS match_id,
                (m.external_ids->>'nhl')::bigint AS nhl_game_id,
                m.home_team_id::text AS home_team_id,
                m.away_team_id::text AS away_team_id
            FROM matches m
            JOIN leagues l ON l.id = m.league_id
            JOIN nhl_match_stats s ON s.match_id = m.id
            WHERE {where_status}
              AND {where_sport}
              AND {where_has_nhl_id}
              {where_goalie_filter}
              {where_season}
            ORDER BY m.match_date ASC
            """,
            params,
        )
        return [(r["match_id"], r["nhl_game_id"], r["home_team_id"], r["away_team_id"]) for r in cur.fetchall()]


def update_starting_goalie(cur, match_id: str, team_id: str, player_id: str) -> bool:
    """Set nhl_match_stats.starting_goalie_id for (match, team). Returns
    True if a row was updated. Doesn't insert — assumes load_nhl_historical
    already created the row (we ran it during Phase 1)."""
    cur.execute(
        """
        UPDATE nhl_match_stats
            SET starting_goalie_id = %s
        WHERE match_id = %s AND team_id = %s
        """,
        (player_id, match_id, team_id),
    )
    return cur.rowcount > 0


def backfill(database_url: str, season: str | None, refetch: bool) -> dict:
    """Walk eligible games, fetch /boxscore, upsert players + update
    nhl_match_stats. Per-game commit so a crash mid-run leaves a usable
    partial backfill."""
    counts = {
        "games_seen": 0,
        "games_skipped_no_boxscore": 0,
        "games_skipped_no_starter": 0,
        "rows_updated": 0,
        "players_upserted": 0,
    }

    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            games = list_games_needing_goalies(conn, season, refetch)
            counts["games_seen"] = len(games)
            if not games:
                logger.info("No games need goalie backfill (season=%s refetch=%s)", season, refetch)
                return counts

            logger.info(
                "=== Backfilling goalies for %d games (season=%s refetch=%s) ===",
                len(games),
                season,
                refetch,
            )

            for i, (match_id, nhl_game_id, home_team_id, away_team_id) in enumerate(games, start=1):
                result = load_nhl.fetch_boxscore_goalies(nhl_game_id)
                if result is None:
                    counts["games_skipped_no_boxscore"] += 1
                else:
                    home_g, away_g = result
                    any_updated = False

                    if home_g is not None:
                        pid = load_nhl.upsert_goalie_player(cur, home_g, home_team_id)
                        if pid and update_starting_goalie(cur, match_id, home_team_id, pid):
                            counts["players_upserted"] += 1
                            counts["rows_updated"] += 1
                            any_updated = True

                    if away_g is not None:
                        pid = load_nhl.upsert_goalie_player(cur, away_g, away_team_id)
                        if pid and update_starting_goalie(cur, match_id, away_team_id, pid):
                            counts["players_upserted"] += 1
                            counts["rows_updated"] += 1
                            any_updated = True

                    if not any_updated and home_g is None and away_g is None:
                        counts["games_skipped_no_starter"] += 1

                    conn.commit()

                if i % GAME_LOG_INTERVAL == 0 or i == len(games):
                    logger.info(
                        "progress: %d/%d (updated=%d players=%d no_box=%d no_starter=%d)",
                        i,
                        len(games),
                        counts["rows_updated"],
                        counts["players_upserted"],
                        counts["games_skipped_no_boxscore"],
                        counts["games_skipped_no_starter"],
                    )

    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--season",
        default=None,
        help="Limit backfill to one season (e.g. 2024-2025). Default: all seasons in matches table.",
    )
    p.add_argument(
        "--refetch",
        action="store_true",
        help="Re-fetch even for games that already have starting_goalie_id set.",
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
