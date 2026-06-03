"""Seed the venue_coords table with known stadium coordinates.

Static dataset — coords were sourced from Wikipedia / official
stadium pages and verified against the venue's own lat/lon
metadata where available. Update this file when a team moves
stadiums (e.g., Las Vegas Raiders → Allegiant Stadium 2020).

Three categories:
  * NFL: all 32 stadiums (8 are domes / climate-controlled and
    flagged is_indoor=True so weather features get skipped).
  * Grand Slam tennis: 4 main-court venues
    (Aus Open, Roland Garros, Wimbledon, US Open).
  * Top-5 European soccer leagues: ~100 grounds. Less reliable —
    smaller clubs occasionally share venues or play at neutral
    sites for European competition.

Usage:
    python /app/scripts/seed_venue_coords.py
    python /app/scripts/seed_venue_coords.py --update  # overwrite existing
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("seed_venue_coords")


@dataclass(frozen=True)
class Venue:
    display_name: str
    latitude: float
    longitude: float
    timezone: str
    is_indoor: bool = False
    # Aliases the venue is known by — lets the upserter cover vendor
    # name variations (e.g., "MetLife Stadium" + "Metlife"). Each
    # alias gets its own row in venue_coords with the same coords.
    aliases: tuple[str, ...] = ()


# ── NFL stadiums (32 teams) ─────────────────────────────────────────


NFL_VENUES = [
    # AFC East
    Venue("Highmark Stadium", 42.7738, -78.7869, "America/New_York"),  # Bills
    Venue("Hard Rock Stadium", 25.9580, -80.2389, "America/New_York"),  # Dolphins
    Venue("Gillette Stadium", 42.0909, -71.2643, "America/New_York"),  # Patriots
    Venue("MetLife Stadium", 40.8135, -74.0745, "America/New_York", aliases=("Metlife Stadium",)),  # Jets/Giants
    # AFC North
    Venue("M&T Bank Stadium", 39.2779, -76.6228, "America/New_York"),  # Ravens
    Venue("Paycor Stadium", 39.0954, -84.5161, "America/New_York"),  # Bengals
    Venue("Cleveland Browns Stadium", 41.5061, -81.6996, "America/New_York", aliases=("Huntington Bank Field",)),
    Venue("Acrisure Stadium", 40.4468, -80.0158, "America/New_York"),  # Steelers
    # AFC South
    Venue("NRG Stadium", 29.6847, -95.4107, "America/Chicago", is_indoor=True),  # Texans
    Venue("Lucas Oil Stadium", 39.7601, -86.1639, "America/Indiana/Indianapolis", is_indoor=True),  # Colts
    Venue("EverBank Stadium", 30.3239, -81.6373, "America/New_York", aliases=("EverBank Field", "TIAA Bank Field")),
    Venue("Nissan Stadium", 36.1665, -86.7713, "America/Chicago"),  # Titans
    # AFC West (Chiefs play at Arrowhead in Kansas City)
    Venue("GEHA Field at Arrowhead Stadium", 39.0489, -94.4839, "America/Chicago", aliases=("Arrowhead Stadium",)),
    # AFC West
    Venue("Empower Field at Mile High", 39.7439, -105.0201, "America/Denver"),  # Broncos
    Venue("Allegiant Stadium", 36.0908, -115.1830, "America/Los_Angeles", is_indoor=True),  # Raiders
    Venue("SoFi Stadium", 33.9535, -118.3392, "America/Los_Angeles", is_indoor=True),  # Chargers/Rams
    # NFC East (Cowboys, Eagles, Giants/Jets shared MetLife above, Commanders)
    Venue("AT&T Stadium", 32.7473, -97.0945, "America/Chicago", is_indoor=True),  # Cowboys
    Venue("Lincoln Financial Field", 39.9008, -75.1675, "America/New_York"),  # Eagles
    Venue("Northwest Stadium", 38.9078, -76.8645, "America/New_York", aliases=("FedExField",)),  # Commanders
    # NFC North
    Venue("Soldier Field", 41.8623, -87.6167, "America/Chicago"),  # Bears
    Venue("Ford Field", 42.3400, -83.0456, "America/Detroit", is_indoor=True),  # Lions
    Venue("Lambeau Field", 44.5013, -88.0622, "America/Chicago"),  # Packers
    Venue("U.S. Bank Stadium", 44.9737, -93.2581, "America/Chicago", is_indoor=True),  # Vikings
    # NFC South
    Venue("Mercedes-Benz Stadium", 33.7553, -84.4006, "America/New_York", is_indoor=True),  # Falcons
    Venue("Bank of America Stadium", 35.2258, -80.8530, "America/New_York"),  # Panthers
    Venue("Caesars Superdome", 29.9511, -90.0812, "America/Chicago", is_indoor=True),  # Saints
    Venue("Raymond James Stadium", 27.9759, -82.5033, "America/New_York"),  # Buccaneers
    # NFC West
    Venue("State Farm Stadium", 33.5276, -112.2626, "America/Phoenix", is_indoor=True),  # Cardinals
    Venue("Levi's Stadium", 37.4030, -121.9695, "America/Los_Angeles"),  # 49ers
    Venue("Lumen Field", 47.5952, -122.3316, "America/Los_Angeles"),  # Seahawks
]


# ── Grand Slam tennis venues ────────────────────────────────────────


TENNIS_VENUES = [
    Venue(
        "Melbourne Park",
        -37.8224,
        144.9789,
        "Australia/Melbourne",
        aliases=("Rod Laver Arena", "Margaret Court Arena", "John Cain Arena"),
    ),
    Venue(
        "Stade Roland Garros",
        48.8470,
        2.2480,
        "Europe/Paris",
        aliases=("Roland Garros", "Court Philippe-Chatrier", "Court Suzanne-Lenglen"),
    ),
    Venue(
        "All England Lawn Tennis Club",
        51.4341,
        -0.2140,
        "Europe/London",
        aliases=("Wimbledon", "Centre Court", "No. 1 Court"),
    ),
    Venue(
        "USTA Billie Jean King National Tennis Center",
        40.7497,
        -73.8458,
        "America/New_York",
        aliases=("Arthur Ashe Stadium", "Louis Armstrong Stadium"),
    ),
]


# ── Top-5 European soccer league grounds ────────────────────────────
# Coverage subset — the highest-coverage 20 grounds across the EPL,
# La Liga, Serie A, Bundesliga, and Ligue 1. Not exhaustive but
# enough that ~60-70% of fixtures match a venue. Smaller clubs and
# Cup ties at neutral sites won't match and the fetcher skips them.

SOCCER_VENUES = [
    # English Premier League
    Venue("Emirates Stadium", 51.5549, -0.1084, "Europe/London"),  # Arsenal
    Venue("Stamford Bridge", 51.4817, -0.1910, "Europe/London"),  # Chelsea
    Venue("Anfield", 53.4308, -2.9608, "Europe/London"),  # Liverpool
    Venue("Old Trafford", 53.4631, -2.2914, "Europe/London"),  # Man Utd
    Venue("Etihad Stadium", 53.4831, -2.2004, "Europe/London"),  # Man City
    Venue("Tottenham Hotspur Stadium", 51.6043, -0.0664, "Europe/London"),
    Venue("St James' Park", 54.9756, -1.6217, "Europe/London"),  # Newcastle
    # La Liga
    Venue("Santiago Bernabéu", 40.4530, -3.6883, "Europe/Madrid", aliases=("Santiago Bernabeu",)),
    Venue("Spotify Camp Nou", 41.3809, 2.1228, "Europe/Madrid", aliases=("Camp Nou",)),
    Venue("Metropolitano Stadium", 40.4362, -3.5994, "Europe/Madrid", aliases=("Wanda Metropolitano",)),
    # Serie A
    Venue("San Siro", 45.4781, 9.1239, "Europe/Rome", aliases=("Stadio Giuseppe Meazza",)),
    Venue("Allianz Stadium", 45.1097, 7.6411, "Europe/Rome", aliases=("Juventus Stadium",)),
    Venue("Stadio Olimpico", 41.9342, 12.4548, "Europe/Rome"),  # Roma + Lazio
    Venue("Stadio Diego Armando Maradona", 40.8279, 14.1932, "Europe/Rome", aliases=("Stadio San Paolo",)),
    # Bundesliga
    Venue("Allianz Arena", 48.2188, 11.6248, "Europe/Berlin"),  # Bayern Munich
    Venue("Signal Iduna Park", 51.4925, 7.4519, "Europe/Berlin", aliases=("Westfalenstadion",)),
    Venue("BayArena", 51.0381, 7.0026, "Europe/Berlin"),  # Bayer Leverkusen
    # Ligue 1
    Venue("Parc des Princes", 48.8414, 2.2530, "Europe/Paris"),  # PSG
    Venue("Groupama Stadium", 45.7651, 4.9819, "Europe/Paris", aliases=("Parc Olympique Lyonnais",)),
    Venue("Orange Velodrome", 43.2697, 5.3958, "Europe/Paris", aliases=("Stade Velodrome",)),
]


def normalize_venue(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def upsert_venue(cur, venue: Venue, alias: str, source: str, update: bool) -> str:
    """Insert one venue (or one alias). Returns 'inserted' / 'updated'
    / 'skipped' for the run summary."""
    norm = normalize_venue(alias)
    if not update:
        cur.execute("SELECT 1 FROM venue_coords WHERE normalized_venue_name = %s LIMIT 1", (norm,))
        if cur.fetchone():
            return "skipped"
    cur.execute(
        """
        INSERT INTO venue_coords
            (normalized_venue_name, display_name, latitude, longitude,
             timezone, is_indoor, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (normalized_venue_name) DO UPDATE
            SET display_name = EXCLUDED.display_name,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                timezone = EXCLUDED.timezone,
                is_indoor = EXCLUDED.is_indoor,
                source = EXCLUDED.source,
                updated_at = NOW()
        """,
        (
            norm,
            venue.display_name,
            venue.latitude,
            venue.longitude,
            venue.timezone,
            venue.is_indoor,
            source,
        ),
    )
    return "inserted" if cur.rowcount > 0 else "updated"


def seed_all(venues: Iterable[Venue], cur, source: str, update: bool) -> dict:
    counts = {"inserted": 0, "updated": 0, "skipped": 0}
    for v in venues:
        # Primary name + every alias each get a row pointing at the
        # same coords (insert separately rather than store a JSON
        # alias array — simpler lookup, no join needed).
        for name in (v.display_name, *v.aliases):
            result = upsert_venue(cur, v, name, source, update)
            counts[result] = counts.get(result, 0) + 1
    return counts


def run(database_url: str, update: bool) -> dict:
    totals = {"inserted": 0, "updated": 0, "skipped": 0}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for label, venues, source in (
                ("NFL", NFL_VENUES, "manual_nfl_seed_2026"),
                ("Tennis", TENNIS_VENUES, "manual_tennis_seed_2026"),
                ("Soccer", SOCCER_VENUES, "manual_soccer_seed_2026"),
            ):
                counts = seed_all(venues, cur, source, update)
                logger.info("%s: %s", label, counts)
                for k, v in counts.items():
                    totals[k] += v
            conn.commit()
    logger.info("Done. %s", totals)
    return totals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--update",
        action="store_true",
        help="Update existing rows (default: skip rows that already exist).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    run(args.database_url, args.update)
    return 0


if __name__ == "__main__":
    sys.exit(main())
