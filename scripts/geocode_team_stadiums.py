"""Geocode soccer teams' HOME STADIUMS via Wikidata (weather revisit, §4
weather chapter — corpus is now 7x, meeting the documented revisit bar).

The football-data.co.uk corpus carries NO venue strings, so the
venue-name-keyed weather path (matches.venue -> venue_coords) can never
resolve the 140k backfilled matches. This maps team -> home stadium ->
coordinates instead, writing venue_coords rows keyed
'team-stadium:{team_id}' that fetch_weather + bulk_weather_backfill use
as the fallback when a match has no venue string.

Wikidata flow (free, ~1 req/s politeness):
  1. wbsearchentities on the team name (plus expansions for our
     football-data short forms: 'Nott'm Forest' -> 'Nottingham Forest').
  2. Entity claims: P115 (home venue) -> venue entity.
  3. Venue claims: P625 (coordinates).
Misses are logged and counted — extend EXPANSIONS deliberately.

    python /app/scripts/geocode_team_stadiums.py
    python /app/scripts/geocode_team_stadiums.py --limit 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("geocode_team_stadiums")

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "auspex-personal-project/1.0 (stadium geocoding; contact: repo operator)"}
DELAY_S = 1.0

# football-data short form -> searchable club name. Extend from the miss log.
EXPANSIONS = {
    "Nott'm Forest": "Nottingham Forest",
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Sheffield Weds": "Sheffield Wednesday",
    "QPR": "Queens Park Rangers",
    "Wolves": "Wolverhampton Wanderers",
    "West Brom": "West Bromwich Albion",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "M'gladbach": "Borussia Monchengladbach",
    "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Sp Gijon": "Sporting Gijon",
    "Espanol": "Espanyol",
    "Sociedad": "Real Sociedad",
    "Betis": "Real Betis",
    "Vallecano": "Rayo Vallecano",
    "La Coruna": "Deportivo La Coruna",
    "Paris SG": "Paris Saint-Germain",
    "St Etienne": "Saint-Etienne",
    "Leverkusen": "Bayer Leverkusen",
    "Dortmund": "Borussia Dortmund",
}


def _get(params: dict) -> Optional[dict]:
    try:
        r = requests.get(WIKIDATA_API, params={**params, "format": "json"}, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning("wikidata request failed: %s", e)
        return None


def _search_entity(name: str) -> Optional[str]:
    data = _get({"action": "wbsearchentities", "search": name, "language": "en", "type": "item", "limit": 5})
    for hit in (data or {}).get("search", []):
        desc = (hit.get("description") or "").lower()
        if any(w in desc for w in ("football club", "soccer club", "sports club", "football team")):
            return hit["id"]
    hits = (data or {}).get("search", [])
    return hits[0]["id"] if hits else None


def _claim_target(entity_id: str, prop: str) -> Optional[dict]:
    data = _get({"action": "wbgetentities", "ids": entity_id, "props": "claims|labels"})
    ent = ((data or {}).get("entities") or {}).get(entity_id) or {}
    claims = (ent.get("claims") or {}).get(prop) or []
    if not claims:
        return None
    return claims[0].get("mainsnak", {}).get("datavalue", {}).get("value")


def stadium_coords_for(team_name: str) -> Optional[dict]:
    """(lat, lon, label) for a team's home stadium, or None."""
    search_name = EXPANSIONS.get(team_name, team_name)
    club_id = _search_entity(search_name)
    if not club_id:
        return None
    time.sleep(DELAY_S)
    venue_ref = _claim_target(club_id, "P115")  # home venue
    if not isinstance(venue_ref, dict) or "id" not in venue_ref:
        return None
    time.sleep(DELAY_S)
    coords = _claim_target(venue_ref["id"], "P625")  # coordinate location
    if not isinstance(coords, dict) or "latitude" not in coords:
        return None
    return {"lat": float(coords["latitude"]), "lon": float(coords["longitude"]), "venue_qid": venue_ref["id"]}


def teams_needing_coords(cur, limit: Optional[int]) -> list[dict]:
    cur.execute(
        """
        SELECT t.id::text AS team_id, t.name
        FROM teams t
        WHERE t.sport = 'soccer'
          AND EXISTS (SELECT 1 FROM matches m WHERE m.home_team_id = t.id AND m.status = 'finished')
          AND NOT EXISTS (
              SELECT 1 FROM venue_coords vc
              WHERE vc.normalized_venue_name = 'team-stadium:' || t.id::text
          )
        ORDER BY t.name
        """
        + (" LIMIT %s" % int(limit) if limit else "")
    )
    return list(cur.fetchall())


def store(cur, team_id: str, team_name: str, hit: dict) -> None:
    cur.execute(
        """
        INSERT INTO venue_coords (normalized_venue_name, display_name, latitude, longitude, is_indoor)
        VALUES (%s, %s, %s, %s, false)
        ON CONFLICT (normalized_venue_name) DO UPDATE
        SET latitude = EXCLUDED.latitude, longitude = EXCLUDED.longitude
        """,
        (f"team-stadium:{team_id}", f"{team_name} home stadium", hit["lat"], hit["lon"]),
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, help="Max teams this run (default: all).")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = p.parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2

    hits = misses = 0
    with psycopg2.connect(args.database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            teams = teams_needing_coords(cur, args.limit)
            logger.info("Geocoding stadiums for %d teams", len(teams))
            for t in teams:
                found = stadium_coords_for(t["name"])
                if found:
                    store(cur, t["team_id"], t["name"], found)
                    hits += 1
                else:
                    logger.info("  miss: %r", t["name"])
                    misses += 1
                if (hits + misses) % 50 == 0:
                    conn.commit()
                    logger.info("progress: %d hits / %d misses", hits, misses)
                time.sleep(DELAY_S)
            conn.commit()
    logger.info("Done: %d hits, %d misses", hits, misses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
