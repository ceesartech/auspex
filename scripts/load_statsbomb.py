"""Load StatsBomb open-data match summaries and team-level xG aggregates.

StatsBomb publishes event-level data for select competitions for free on
GitHub (https://github.com/statsbomb/open-data). The full event-level dump is
~50GB; this loader pulls only what we need for outcome modelling:

  1. Competitions + matches metadata (small, ~MB).
  2. Per-match team aggregates derived from events: xG, xGA, shot count,
     pass count, possession proxy.

Use cases:
    # List available competitions
    python scripts/load_statsbomb.py --list

    # Load all matches from a competition into the database
    python scripts/load_statsbomb.py --competition 11 --season 90 \\
        --database-url postgres://...

    # Load match summaries only (no event aggregates - much faster)
    python scripts/load_statsbomb.py --competition 11 --season 90 --no-events
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("load_statsbomb")

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


@dataclass
class MatchAggregate:
    match_id: int
    competition_id: int
    season_id: int
    match_date: str
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    home_score: int
    away_score: int
    home_xg: float
    away_xg: float
    home_shots: int
    away_shots: int
    home_passes: int
    away_passes: int


def fetch_json(path: str) -> list | dict:
    url = f"{BASE_URL}/{path}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def list_competitions() -> list[dict]:
    return fetch_json("competitions.json")  # type: ignore[return-value]


def fetch_matches(competition_id: int, season_id: int) -> list[dict]:
    return fetch_json(f"matches/{competition_id}/{season_id}.json")  # type: ignore[return-value]


def aggregate_events(events: list[dict], home_team_id: int, away_team_id: int) -> tuple[dict, dict]:
    """Return (home_aggregate, away_aggregate) from a match's events."""
    agg = {
        home_team_id: {"xg": 0.0, "shots": 0, "passes": 0},
        away_team_id: {"xg": 0.0, "shots": 0, "passes": 0},
    }
    for ev in events:
        team_id = ev.get("team", {}).get("id")
        if team_id not in agg:
            continue
        type_name = ev.get("type", {}).get("name")
        if type_name == "Shot":
            agg[team_id]["shots"] += 1
            shot = ev.get("shot", {}) or {}
            xg = shot.get("statsbomb_xg")
            if xg is not None:
                agg[team_id]["xg"] += float(xg)
        elif type_name == "Pass":
            agg[team_id]["passes"] += 1
    return agg[home_team_id], agg[away_team_id]


def build_match_aggregate(match: dict, competition_id: int, season_id: int, with_events: bool) -> MatchAggregate | None:
    home = match["home_team"]
    away = match["away_team"]
    home_xg = away_xg = 0.0
    home_shots = away_shots = 0
    home_passes = away_passes = 0
    if with_events:
        try:
            events = fetch_json(f"events/{match['match_id']}.json")
        except requests.HTTPError as e:
            logger.warning("No events for match %s: %s", match["match_id"], e)
            return None
        home_a, away_a = aggregate_events(events, home["home_team_id"], away["away_team_id"])  # type: ignore[arg-type]
        home_xg, away_xg = home_a["xg"], away_a["xg"]
        home_shots, away_shots = home_a["shots"], away_a["shots"]
        home_passes, away_passes = home_a["passes"], away_a["passes"]
    return MatchAggregate(
        match_id=match["match_id"],
        competition_id=competition_id,
        season_id=season_id,
        match_date=match.get("match_date", ""),
        home_team_id=home["home_team_id"],
        home_team_name=home["home_team_name"],
        away_team_id=away["away_team_id"],
        away_team_name=away["away_team_name"],
        home_score=match.get("home_score", 0),
        away_score=match.get("away_score", 0),
        home_xg=home_xg,
        away_xg=away_xg,
        home_shots=home_shots,
        away_shots=away_shots,
        home_passes=home_passes,
        away_passes=away_passes,
    )


def load_competition_season(
    competition_id: int,
    season_id: int,
    with_events: bool,
    max_workers: int = 4,
) -> Iterable[MatchAggregate]:
    matches = fetch_matches(competition_id, season_id)
    logger.info("Found %d matches for competition=%s season=%s", len(matches), competition_id, season_id)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(build_match_aggregate, m, competition_id, season_id, with_events)
            for m in matches
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            agg = fut.result()
            if agg:
                yield agg
            if i % 20 == 0:
                logger.info("Processed %d/%d matches", i, len(matches))


def write_database(rows: list[MatchAggregate], database_url: str) -> int:
    import psycopg2
    from psycopg2.extras import execute_values

    if not rows:
        return 0
    create_sql = """
        CREATE TABLE IF NOT EXISTS raw_statsbomb_matches (
            match_id BIGINT PRIMARY KEY,
            competition_id INTEGER NOT NULL,
            season_id INTEGER NOT NULL,
            match_date TEXT,
            home_team_id BIGINT, home_team_name TEXT,
            away_team_id BIGINT, away_team_name TEXT,
            home_score INTEGER, away_score INTEGER,
            home_xg DOUBLE PRECISION, away_xg DOUBLE PRECISION,
            home_shots INTEGER, away_shots INTEGER,
            home_passes INTEGER, away_passes INTEGER,
            loaded_at TIMESTAMPTZ DEFAULT NOW()
        );
    """
    insert_sql = """
        INSERT INTO raw_statsbomb_matches (
            match_id, competition_id, season_id, match_date,
            home_team_id, home_team_name, away_team_id, away_team_name,
            home_score, away_score, home_xg, away_xg,
            home_shots, away_shots, home_passes, away_passes
        ) VALUES %s
        ON CONFLICT (match_id) DO UPDATE SET
            home_xg = EXCLUDED.home_xg,
            away_xg = EXCLUDED.away_xg,
            home_shots = EXCLUDED.home_shots,
            away_shots = EXCLUDED.away_shots,
            home_passes = EXCLUDED.home_passes,
            away_passes = EXCLUDED.away_passes,
            loaded_at = NOW();
    """
    payload = [
        (r.match_id, r.competition_id, r.season_id, r.match_date,
         r.home_team_id, r.home_team_name, r.away_team_id, r.away_team_name,
         r.home_score, r.away_score, r.home_xg, r.away_xg,
         r.home_shots, r.away_shots, r.home_passes, r.away_passes)
        for r in rows
    ]
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(create_sql)
            execute_values(cur, insert_sql, payload)
            conn.commit()
            return cur.rowcount


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="List available competitions and exit.")
    p.add_argument("--competition", type=int, help="StatsBomb competition_id.")
    p.add_argument("--season", type=int, help="StatsBomb season_id.")
    p.add_argument("--all-open", action="store_true",
                   help="Load every (competition, season) listed in competitions.json.")
    p.add_argument("--no-events", action="store_true",
                   help="Skip per-match event aggregation (much faster, no xG).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        comps = list_competitions()
        for c in comps:
            print(f"competition_id={c['competition_id']:<4} season_id={c['season_id']:<4} "
                  f"{c['competition_name']} / {c['season_name']}")
        return 0

    pairs: list[tuple[int, int]]
    if args.all_open:
        pairs = [(c["competition_id"], c["season_id"]) for c in list_competitions()]
    elif args.competition is not None and args.season is not None:
        pairs = [(args.competition, args.season)]
    else:
        logger.error("Pass --list, --all-open, or both --competition and --season.")
        return 1

    all_rows: list[MatchAggregate] = []
    for comp, season in pairs:
        try:
            for agg in load_competition_season(comp, season, not args.no_events, args.workers):
                all_rows.append(agg)
        except requests.HTTPError as e:
            logger.warning("Failed competition=%s season=%s: %s", comp, season, e)

    logger.info("Total matches loaded: %d", len(all_rows))
    if args.database_url:
        inserted = write_database(all_rows, args.database_url)
        logger.info("Upserted %d rows into raw_statsbomb_matches", inserted)
    else:
        logger.info("--database-url not set; printed-only mode")
        for r in all_rows[:5]:
            print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
