"""Populate matches.venue for soccer matches by scraping FBRef league
schedule pages.

99.6% of soccer matches in our DB have NULL `matches.venue` because
the football-data.co.uk CSV ingest doesn't include venue. FBRef's
per-league schedule pages list every match with its venue column,
so scraping ONE schedule page per (league, season) and joining the
results back to our matches is far cheaper than scraping per-match
pages (~38 HTTP requests vs ~38 × N_matches).

URL pattern (FBRef):
    https://fbref.com/en/comps/{COMP_ID}/{SEASON}/schedule/...

Each row of the schedule table is one match with columns Wk, Day,
Date, Time, Home, xG, Score, xG.1, Away, Attendance, Venue,
Referee, Match Report, Notes.

The script:
  1. For each (league, season) the user requests, fetches the
     schedule page once (with rate-limit + User-Agent set).
  2. Parses each match row → (home_team, away_team, date, venue).
  3. For each row, fuzzy-matches to our `matches` table
     (sport='soccer', NULL venue) by team name similarity + date
     proximity.
  4. UPDATEs matches.venue.

Designed to be RE-RUNNABLE: matches with non-NULL venue are
skipped, so a partial-run failure can pick up next time without
duplicate work.

Rate limit: 12s between requests by default. FBRef's robots.txt
allows scraping with reasonable cadence; this is conservative.

Run via:
    docker compose exec api python /app/scripts/scrape_fbref_venues.py \\
        --league "Premier League" --season 2024-2025

    docker compose exec api python /app/scripts/scrape_fbref_venues.py \\
        --all  # iterate over every (league, season) we have a comp_id for
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date as date_cls
from difflib import SequenceMatcher
from typing import Iterable, Optional

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("scrape_fbref_venues")


# Map auspex `leagues.name` -> FBRef competition ID. Only top divisions
# with stable IDs are included; expand as needed. FBRef comp IDs are
# stable (we don't expect them to change), so this can be a constant.
FBREF_COMP_IDS: dict[str, int] = {
    "Premier League": 9,
    "Serie A": 11,
    "La Liga": 12,
    "Ligue 1": 13,
    "Bundesliga": 20,
    "Eredivisie": 23,
    "Primeira Liga": 32,
    "Major League Soccer": 22,
}

BASE_URL = "https://fbref.com"
DEFAULT_TIMEOUT_SEC = 30
DEFAULT_REQUEST_DELAY_SEC = 12.0

# FBRef expects a real-ish User-Agent. Identifying as auspex (with
# contact-style ID) is more polite than spoofing a browser and gives
# them someone to block specifically if we're misbehaving.
USER_AGENT = "auspex-scraper/1.0 (https://github.com/ceesartech/auspex)"


def _normalize_team_name(name: str) -> str:
    """Lowercase + strip + collapse multiple spaces. FBRef occasionally
    writes 'Manchester United' where football-data.co.uk has 'Man
    United'; downstream uses fuzzy matching, but normalising removes
    casing noise first."""
    return " ".join(name.lower().split()) if name else ""


def fetch_schedule_page(
    comp_id: int,
    season: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> str:
    """Fetch the raw HTML for one league-season schedule page."""
    url = f"{BASE_URL}/en/comps/{comp_id}/{season}/schedule/"
    logger.info("Fetching %s", url)
    r = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.text


def parse_schedule_html(html: str) -> list[dict]:
    """Parse a FBRef schedule page into a list of
    {home, away, date, venue} dicts. Skips rows with empty Venue
    (FBRef sometimes lists future/postponed matches with no venue
    yet)."""
    from bs4 import BeautifulSoup  # imported here so --help works without bs4

    soup = BeautifulSoup(html, "html.parser")
    rows = []
    # Schedule table id depends on the comp but the data-stat attribute
    # mapping is stable. We index by data-stat to be robust to column
    # reordering.
    table = soup.find("table", class_=lambda c: c and "stats_table" in c)
    if table is None:
        logger.warning("No stats_table found in schedule HTML")
        return rows
    for tr in table.find("tbody").find_all("tr"):
        # Header rows have class 'spacer' or 'thead'; skip them.
        if "thead" in (tr.get("class") or []) or "spacer" in (tr.get("class") or []):
            continue

        cells = {c["data-stat"]: c.get_text(strip=True) for c in tr.find_all(["th", "td"]) if c.has_attr("data-stat")}
        # Required columns. FBRef stat keys: "home_team", "away_team",
        # "date", "venue". Skip when any are missing.
        home = cells.get("home_team")
        away = cells.get("away_team")
        date_str = cells.get("date")
        venue = cells.get("venue")
        if not (home and away and date_str and venue):
            continue
        rows.append(
            {
                "home": home,
                "away": away,
                "date": date_str,
                "venue": venue,
            }
        )
    return rows


def best_match(
    cur,
    home: str,
    away: str,
    date_str: str,
    *,
    day_window: int = 1,
) -> Optional[str]:
    """Find the best-matching auspex `matches.id` for this FBRef row,
    or None if no good match exists.

    Strategy: same-day exact match first; if multiple, pick the row
    whose team names have the highest fuzzy similarity to the FBRef
    names. Falls back to ±day_window-day search before giving up to
    handle timezone-induced day shifts."""
    # Try parsing the date — FBRef uses ISO-like "2024-08-17" or
    # "2024-08-17" depending on locale, and historical pages can use
    # textual months. SequenceMatcher is robust to small typos so we
    # don't need to perfectly normalise.
    try:
        target_date = date_cls.fromisoformat(date_str)
    except ValueError:
        # Fall back to letting the DB parse it; if it can't, no match.
        target_date = None

    if target_date is None:
        return None

    cur.execute(
        """
        SELECT m.id::text AS id, ht.name AS home_name, at.name AS away_name,
               m.match_date::date AS d
        FROM matches m
        JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE m.venue IS NULL
          AND m.match_date::date BETWEEN %s AND %s
        """,
        (
            (target_date.toordinal() - day_window),
            (target_date.toordinal() + day_window),
        ),
    )
    # The above param list passes ordinals as integers — PostgreSQL
    # won't accept those directly. Re-issue with date objects.
    from datetime import timedelta

    cur.execute(
        """
        SELECT m.id::text AS id, ht.name AS home_name, at.name AS away_name
        FROM matches m
        JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE m.venue IS NULL
          AND m.match_date::date BETWEEN %s AND %s
        """,
        (
            target_date - timedelta(days=day_window),
            target_date + timedelta(days=day_window),
        ),
    )
    candidates = cur.fetchall()
    if not candidates:
        return None

    home_n = _normalize_team_name(home)
    away_n = _normalize_team_name(away)

    def score(row: dict) -> float:
        return (
            SequenceMatcher(None, _normalize_team_name(row["home_name"]), home_n).ratio()
            + SequenceMatcher(None, _normalize_team_name(row["away_name"]), away_n).ratio()
        )

    # Highest combined similarity wins; tie-break by id to keep
    # the choice deterministic.
    candidates.sort(key=lambda r: (-score(r), r["id"]))
    best = candidates[0]
    if score(best) < 1.0:
        # Below ~0.5 per side means at least one team name is barely
        # similar — skip rather than write a wrong venue.
        return None
    return best["id"]


def update_venue(cur, match_id: str, venue: str) -> None:
    cur.execute(
        "UPDATE matches SET venue = %s, updated_at = NOW() WHERE id = %s",
        (venue, match_id),
    )


def league_seasons_to_scrape(
    cur,
    league_name: Optional[str],
    season: Optional[str],
) -> list[tuple[int, str, str]]:
    """Return (comp_id, season, league_name) tuples to scrape.

    If --league and --season are both provided, return that single
    combo (validated against FBREF_COMP_IDS). With --all, iterate
    over every league/season tuple we have a comp_id for AND at
    least one NULL-venue match.
    """
    if league_name and season:
        comp_id = FBREF_COMP_IDS.get(league_name)
        if comp_id is None:
            raise SystemExit(
                f"No FBRef comp_id known for league {league_name!r}. " f"Available: {sorted(FBREF_COMP_IDS.keys())}"
            )
        return [(comp_id, season, league_name)]

    # --all path: every (league, season) with at least one NULL-venue
    # match and a known comp_id. The DB's season column for soccer
    # uses the FBRef "YYYY-YYYY" format so it slots in directly.
    cur.execute(
        """
        SELECT DISTINCT l.name AS league_name, m.season AS season
        FROM matches m
        JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
        WHERE m.venue IS NULL AND m.season IS NOT NULL
        ORDER BY l.name, m.season
        """
    )
    out = []
    for row in cur.fetchall():
        comp_id = FBREF_COMP_IDS.get(row["league_name"])
        if comp_id is None:
            continue
        out.append((comp_id, row["season"], row["league_name"]))
    return out


def run(
    database_url: str,
    leagues_to_scrape: Iterable[tuple[int, str, str]],
    request_delay: float,
) -> dict:
    counts = {"matched": 0, "skipped": 0, "pages_fetched": 0, "rows_seen": 0}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for i, (comp_id, season, league_name) in enumerate(leagues_to_scrape):
                if i > 0:
                    time.sleep(request_delay)
                try:
                    html = fetch_schedule_page(comp_id, season)
                except requests.RequestException as e:
                    logger.warning(
                        "Skipping comp=%s season=%s: %s",
                        comp_id,
                        season,
                        e,
                    )
                    continue
                counts["pages_fetched"] += 1
                rows = parse_schedule_html(html)
                counts["rows_seen"] += len(rows)
                logger.info(
                    "%s %s: parsed %d schedule rows",
                    league_name,
                    season,
                    len(rows),
                )
                for row in rows:
                    match_id = best_match(cur, row["home"], row["away"], row["date"])
                    if match_id is None:
                        counts["skipped"] += 1
                        continue
                    update_venue(cur, match_id, row["venue"])
                    counts["matched"] += 1
                conn.commit()
                logger.info("%s %s: matched %d / skipped %d", league_name, season, counts["matched"], counts["skipped"])
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--league",
        help="auspex `leagues.name` — e.g. 'Premier League'.",
    )
    parser.add_argument(
        "--season",
        help="e.g. '2024-2025'. Must match matches.season.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Iterate over every (league, season) we have a comp_id for.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SEC,
        help="Seconds between consecutive FBRef requests.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logger.setLevel(args.log_level)
    if not args.database_url:
        logger.error("DATABASE_URL not set.")
        return 1
    if not (args.all or (args.league and args.season)):
        logger.error("Pass --all OR both --league and --season.")
        return 1

    with psycopg2.connect(args.database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            league_seasons = league_seasons_to_scrape(
                cur,
                args.league if not args.all else None,
                args.season if not args.all else None,
            )
    logger.info("Will scrape %d (league, season) pages.", len(league_seasons))
    counts = run(args.database_url, league_seasons, args.request_delay)
    logger.info("Done. %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
