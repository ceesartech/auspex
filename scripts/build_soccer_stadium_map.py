"""Build the team -> home-stadium mapping for soccer weather lookup.

Soccer's `matches.venue` is NULL for 99.6% of historical rows because
the football-data.co.uk CSVs (our main source) ship no venue field,
and PR #17's FBRef scraper hit IP-blocking from hosting-provider
networks. This script generates a per-team fallback so the Visual
Crossing weather fetcher has coords for every soccer match, with
or without `matches.venue`.

Data source: Wikidata SPARQL endpoint. Wikidata indexes football
clubs (Q476028) with their home venue (P115) and stadium coords
(P625) — completely free, no API key, no rate-limit concerns at
our query scale. The SPARQL query below fans across every football
club Wikidata knows about (~3,000 clubs globally) with a home
venue tagged. Results are cross-referenced against our DB's
`teams.name` via case-insensitive matching plus a curated alias
list for known mismatches (e.g. "Man United" vs "Manchester
United F.C.").

Output: `data/soccer_team_stadiums.json` keyed by team UUID with
{stadium, latitude, longitude, timezone, is_indoor}. The VC
weather fetcher (scripts/fetch_weather_visual_crossing.py) reads
this file as a fallback when matches.venue is NULL.

Run from a residential connection (Wikidata is fine from any IP,
but our prod hosts shouldn't proxy this kind of one-off generator):

    python scripts/build_soccer_stadium_map.py \\
        --database-url postgresql://USER:PASS@HOST:PORT/DB \\
        --output data/soccer_team_stadiums.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("build_soccer_stadium_map")


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# Pull every football club Wikidata knows about that has BOTH a home
# venue and coordinates for that venue. ?clubLabel is the English
# label; ?coord is "Point(lon lat)" WKT we parse below. The query
# completes in ~5s and returns ~3000 rows.
SPARQL_QUERY = """
SELECT DISTINCT ?club ?clubLabel ?stadium ?stadiumLabel ?coord ?countryLabel WHERE {
  ?club wdt:P31/wdt:P279* wd:Q476028 .
  ?club wdt:P115 ?stadium .
  ?stadium wdt:P625 ?coord .
  OPTIONAL { ?club wdt:P17 ?country . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


def fetch_wikidata() -> list[dict]:
    """Run the SPARQL query, parse the result. Wikidata's response is
    JSON-LD-ish with each row's columns nested under ``value``."""
    logger.info("Querying Wikidata SPARQL (this takes ~5-10s)...")
    r = requests.get(
        WIKIDATA_ENDPOINT,
        params={"query": SPARQL_QUERY, "format": "json"},
        headers={"User-Agent": "auspex-soccer-stadium-builder/1.0 (chijiokekechi@gmail.com)"},
        timeout=60,
    )
    r.raise_for_status()
    rows = r.json()["results"]["bindings"]
    logger.info("Wikidata returned %d club rows.", len(rows))

    out = []
    for r_ in rows:
        club_label = r_.get("clubLabel", {}).get("value")
        stadium_label = r_.get("stadiumLabel", {}).get("value")
        coord = r_.get("coord", {}).get("value")  # "Point(lon lat)"
        country = r_.get("countryLabel", {}).get("value")
        if not (club_label and stadium_label and coord):
            continue
        m = re.match(r"Point\(([-0-9.]+) ([-0-9.]+)\)", coord)
        if not m:
            continue
        lon, lat = float(m.group(1)), float(m.group(2))
        out.append(
            {
                "club_name": club_label,
                "stadium": stadium_label,
                "latitude": lat,
                "longitude": lon,
                "country": country,
            }
        )
    return out


def _normalize(name: str) -> str:
    """Lowercase, drop punctuation + common boilerplate suffixes so
    "Manchester United F.C." and "Man United" can match."""
    s = name.lower()
    # Strip suffixes Wikidata appends.
    for suffix in (
        " f.c.",
        " fc",
        " a.f.c.",
        " afc",
        " s.c.",
        " sc",
        " c.f.",
        " cf",
        " sociedad anónima deportiva",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    # Drop punctuation, collapse whitespace.
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Curated overrides for teams whose DB name doesn't match Wikidata's
# canonical name even after normalisation. Add new entries when the
# match script logs "no match for ...". The KEY is the normalised
# DB team name; the VALUE is the normalised Wikidata club label.
TEAM_ALIASES: dict[str, str] = {
    # English shorthand.
    "man united": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "brighton": "brighton hove albion",
    "leicester": "leicester city",
    "leeds": "leeds united",
    "newcastle": "newcastle united",
    "west ham": "west ham united",
    # Continental.
    "bayern": "bayern munich",
    "psg": "paris saint germain",
    "atletico": "atlético madrid",
    "atletico madrid": "atlético madrid",
    "ac milan": "milan",
    "inter milan": "inter milan",
    "internazionale": "inter milan",
}


def match_team_to_stadium(
    db_team_name: str,
    wiki_index: dict[str, dict],
) -> Optional[dict]:
    """Look up a DB team's stadium info via normalised name + alias
    fallback. Returns None if no match (caller logs the miss)."""
    norm = _normalize(db_team_name)
    if norm in wiki_index:
        return wiki_index[norm]
    aliased = TEAM_ALIASES.get(norm)
    if aliased and aliased in wiki_index:
        return wiki_index[aliased]
    return None


def lookup_timezone(lat: float, lon: float) -> str:
    """Look up an IANA timezone for (lat, lon). Tries
    ``timezonefinder`` if installed (offline + fast); falls back to
    UTC with a warning otherwise (script still ships JSON; user can
    fill timezones later if needed)."""
    try:
        from timezonefinder import TimezoneFinder
    except ImportError:
        return "UTC"
    tf = TimezoneFinder()
    tz = tf.timezone_at(lat=lat, lng=lon)
    return tz or "UTC"


def list_soccer_teams(cur) -> list[dict]:
    cur.execute(
        """
        SELECT t.id::text AS id, t.name AS name, l.name AS league
        FROM teams t
        JOIN leagues l ON l.id = t.league_id
        WHERE l.sport = 'soccer'
        ORDER BY l.name, t.name
        """,
    )
    return cur.fetchall()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--output",
        default="data/soccer_team_stadiums.json",
        help="Where to write the mapping (relative paths from repo root).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    if not args.database_url:
        logger.error("DATABASE_URL not set.")
        return 1

    wiki_rows = fetch_wikidata()
    # Index by normalised name; if a name appears multiple times,
    # keep the first (Wikidata returns multiple language variants for
    # popular clubs, but they all point at the same stadium).
    wiki_index: dict[str, dict] = {}
    for row in wiki_rows:
        norm = _normalize(row["club_name"])
        if norm and norm not in wiki_index:
            wiki_index[norm] = row

    with psycopg2.connect(args.database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            db_teams = list_soccer_teams(cur)
    logger.info("Loaded %d soccer teams from DB.", len(db_teams))

    mapping: dict[str, dict] = {}
    matched = 0
    misses: list[str] = []
    for team in db_teams:
        hit = match_team_to_stadium(team["name"], wiki_index)
        if not hit:
            misses.append(f"{team['name']} ({team['league']})")
            continue
        tz = lookup_timezone(hit["latitude"], hit["longitude"])
        mapping[team["id"]] = {
            "team_name": team["name"],
            "league": team["league"],
            "stadium": hit["stadium"],
            "latitude": hit["latitude"],
            "longitude": hit["longitude"],
            "timezone": tz,
            # Wikidata has a "stadium is covered" property but coverage
            # is sparse. Default to False — caller can override per
            # known-indoor venue (Tottenham, Atlanta MLS).
            "is_indoor": False,
            "country": hit.get("country"),
        }
        matched += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False, sort_keys=True)
    logger.info(
        "Wrote %d team mappings to %s (%d misses).",
        matched,
        out_path,
        len(misses),
    )
    if misses:
        logger.info("Sample misses (first 30):")
        for m in misses[:30]:
            logger.info("  - %s", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
