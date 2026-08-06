"""Bulk Open-Meteo weather backfill by TEAM STADIUM (weather revisit at 7x
corpus). fetch_weather.py works per-match with a per-run LIMIT 500 — right
for the daily tick, hopeless for 140k backfilled matches. This fetches the
archive per (stadium, 5-year chunk) — one API call covers every home match
of that team in the span (~5k calls total, well under Open-Meteo's ~10k/day
free budget) — then slices each match's kickoff window locally with the
SAME match_window_summary used by fetch_weather, writing identical
data_kind='actual' match_weather rows.

Coordinates come from geocode_team_stadiums.py's 'team-stadium:{team_id}'
venue_coords rows.

    python /app/scripts/bulk_weather_backfill.py
    python /app/scripts/bulk_weather_backfill.py --limit-teams 20
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Optional

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("bulk_weather_backfill")


def _load_fetch_weather():
    if "fetch_weather" in sys.modules:
        return sys.modules["fetch_weather"]
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("fetch_weather", os.path.join(here, "fetch_weather.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_weather"] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load_fetch_weather()
CHUNK_YEARS = 5
DELAY_S = 0.3


def stadium_teams(cur, limit: Optional[int]) -> list[dict]:
    cur.execute(
        """
        SELECT vc.id::text AS venue_id, vc.latitude, vc.longitude,
               replace(vc.normalized_venue_name, 'team-stadium:', '') AS team_id
        FROM venue_coords vc
        WHERE vc.normalized_venue_name LIKE 'team-stadium:%'
        ORDER BY vc.normalized_venue_name
        """
        + (" LIMIT %d" % int(limit) if limit else "")
    )
    return list(cur.fetchall())


def missing_home_matches(cur, team_id: str) -> list[dict]:
    cur.execute(
        """
        SELECT m.id::text AS match_id, m.match_date
        FROM matches m
        JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
        WHERE m.home_team_id = %s AND m.status = 'finished'
          AND NOT EXISTS (
              SELECT 1 FROM match_weather mw
              WHERE mw.match_id = m.id AND mw.data_kind = 'actual'
          )
        ORDER BY m.match_date
        """,
        (team_id,),
    )
    return list(cur.fetchall())


def fetch_chunk(lat: float, lon: float, start: str, end: str) -> Optional[dict]:
    try:
        r = requests.get(
            fw.ARCHIVE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": fw.HOURLY_VARS,
                "timezone": "UTC",  # align hourly slots with UTC match_date, matching fetch_weather
                "start_date": start,
                "end_date": end,
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("hourly")
    except (requests.RequestException, ValueError) as e:
        logger.warning("archive fetch failed (%s..%s): %s", start, end, e)
        return None


def run(database_url: str, limit_teams: Optional[int]) -> dict:
    counts = {"teams": 0, "calls": 0, "written": 0, "no_window": 0, "fetch_failed": 0}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            teams = stadium_teams(cur, limit_teams)
            logger.info("Backfilling weather for %d team stadiums", len(teams))
            for t in teams:
                matches = missing_home_matches(cur, t["team_id"])
                if not matches:
                    continue
                counts["teams"] += 1
                by_chunk: dict[int, list[dict]] = defaultdict(list)
                for m in matches:
                    by_chunk[m["match_date"].year // CHUNK_YEARS].append(m)
                for chunk in sorted(by_chunk):
                    ms = by_chunk[chunk]
                    start = min(m["match_date"] for m in ms).date().isoformat()
                    end = max(m["match_date"] for m in ms).date().isoformat()
                    hourly = fetch_chunk(float(t["latitude"]), float(t["longitude"]), start, end)
                    counts["calls"] += 1
                    time.sleep(DELAY_S)
                    if hourly is None:
                        counts["fetch_failed"] += 1
                        continue
                    for m in ms:
                        summary = fw.match_window_summary(hourly, m["match_date"])
                        if not summary:
                            counts["no_window"] += 1
                            continue
                        fw.write_weather(
                            cur,
                            match_id=m["match_id"],
                            venue_id=t["venue_id"],
                            data_kind="actual",
                            summary=summary,
                        )
                        counts["written"] += 1
                conn.commit()
                if counts["teams"] % 25 == 0:
                    logger.info("progress: %s", counts)
    logger.info("Done: %s", counts)
    return counts


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit-teams", type=int, help="Max stadiums this run (default: all).")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = p.parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    counts = run(args.database_url, args.limit_teams)
    return 0 if counts["written"] > 0 or counts["teams"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
