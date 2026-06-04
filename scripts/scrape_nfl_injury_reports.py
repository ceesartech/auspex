"""SCAFFOLD: NFL injury-report scraper from pro-football-reference.

Memory `nfl-spread-total-efficient` flags QB injury data as the
biggest single signal for both NFL spread and total markets. This
script is the SCAFFOLD — schema (migration 018) + scraper architecture
+ unit-tested HTML parser — but is NOT production-ready out of the
box. Three pieces explicitly deferred:

  1. PFR URL pattern resolution. PFR week-by-week injury reports
     live at /years/{season_year}/week_{week}.htm — this script
     accepts that pattern but doesn't auto-discover the week
     numbers; the caller passes them.

  2. Robust week → (season_year, week_number) mapping. The user
     should backfill 3 seasons × 18 regular-season weeks × 22
     teams' injury rows. Each season's URL differs and the script
     accepts --season-year + --week as args.

  3. Match-row resolution. The scraper produces (player, status,
     team, week, snapshot_at) tuples — wiring those to specific
     `matches.id` rows requires a separate join step (the user
     can use scripts/load_football_data.py's team-name fuzzy
     matcher as a template).

Designed so the parser layer (tests in
tests/unit/test_scrape_nfl_injury_reports.py) is the load-bearing
piece. Once the user is satisfied that the parser handles their
real PFR HTML, the orchestration layer is a thin wrapper.

Run via:
    docker compose exec api python /app/scripts/scrape_nfl_injury_reports.py \\
        --season-year 2024 --week 1 --dry-run
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
logger = logging.getLogger("scrape_nfl_injury_reports")


BASE_URL = "https://www.pro-football-reference.com"
DEFAULT_TIMEOUT_SEC = 30
DEFAULT_REQUEST_DELAY_SEC = 6.0

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

# PFR injury status codes. Captured verbatim from the report;
# downstream feature computation maps these to numeric severity
# (e.g. starter-out → 1.0, starter-questionable → 0.5).
STATUS_CODES = {"Q", "D", "O", "IR", "PUP", "NSI"}


def fetch_injury_report_page(
    season_year: int,
    week: int,
    *,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> str:
    """Fetch the raw HTML for one week's injury report."""
    url = f"{BASE_URL}/years/{season_year}/week_{week}.htm"
    logger.info("Fetching %s", url)
    r = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.text


def parse_injury_table(html: str) -> list[dict]:
    """Parse a PFR injury report page into per-player rows.

    Returns a list of {team, player, position, status, injury_type}
    dicts. The PFR injury table uses `data-stat` attributes; we
    index by those for robustness to column reorders.

    Skips header rows and rows without a status code we recognise.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    rows = []
    table = soup.find("table", id="injuries")
    if table is None:
        # Fall back to any stats_table with player_name column.
        table = soup.find(
            "table",
            class_=lambda c: c and "stats_table" in c,
        )
    if table is None:
        logger.warning("No injury table found in PFR HTML")
        return rows
    tbody = table.find("tbody")
    if tbody is None:
        return rows
    for tr in tbody.find_all("tr"):
        cls = tr.get("class") or []
        if "thead" in cls or "spacer" in cls:
            continue
        cells = {c["data-stat"]: c.get_text(strip=True) for c in tr.find_all(["th", "td"]) if c.has_attr("data-stat")}
        player = cells.get("player_name") or cells.get("player")
        team = cells.get("team")
        status = (cells.get("status") or "").upper().strip()
        injury_type = cells.get("injury") or cells.get("body_part") or ""
        position = cells.get("position") or cells.get("pos") or ""
        if not (player and team and status):
            continue
        if status not in STATUS_CODES:
            # Unknown status — skip rather than guess.
            continue
        rows.append(
            {
                "team": team,
                "player": player,
                "position": position,
                "status": status,
                "injury_type": injury_type,
            }
        )
    return rows


def upsert_injury_row(
    cur,
    *,
    match_id: str,
    player_id: str,
    status: str,
    injury_type: Optional[str],
    is_starter: bool,
    snapshot_at: str,
    source: str,
) -> None:
    """Idempotent upsert on (match_id, player_id, snapshot_at). See
    migration 018 for the unique constraint shape."""
    cur.execute(
        """
        INSERT INTO nfl_injury_reports
            (match_id, player_id, status, injury_type, is_starter,
             snapshot_at, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (match_id, player_id, snapshot_at) DO UPDATE
            SET status = EXCLUDED.status,
                injury_type = EXCLUDED.injury_type,
                is_starter = EXCLUDED.is_starter,
                updated_at = NOW()
        """,
        (
            match_id,
            player_id,
            status,
            injury_type,
            is_starter,
            snapshot_at,
            source,
        ),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--season-year", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SEC,
        help="Seconds between PFR requests (default 6).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + log but don't write to the DB.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logger.setLevel(args.log_level)

    if not args.database_url and not args.dry_run:
        logger.error("DATABASE_URL not set; use --dry-run to parse without writes.")
        return 1

    try:
        html = fetch_injury_report_page(args.season_year, args.week)
    except requests.RequestException as e:
        logger.error("Fetch failed: %s", e)
        return 2
    time.sleep(args.request_delay)
    rows = parse_injury_table(html)
    logger.info("Parsed %d injury-report rows for %s week %s", len(rows), args.season_year, args.week)
    if args.dry_run:
        for r in rows[:25]:
            logger.info("[dry-run] %s", r)
        return 0

    # Real ingest goes here — left as a TODO for the user-driven
    # match/player resolution. The plumbing below is intentionally
    # minimal so the test suite covers the parser alone.
    with psycopg2.connect(args.database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            logger.warning("Insert path is a STUB. Implement match/player lookup before " "running outside --dry-run.")
            _ = cur, rows
    return 0


if __name__ == "__main__":
    sys.exit(main())
