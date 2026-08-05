"""Ingest Powerball / Mega Millions draw history into lottery_draws.

Source: NY Open Data (Socrata) — free, no API key, updated the morning after
each draw, full history (Powerball to 2010, Mega Millions to 2002):

  powerball:     https://data.ny.gov/resource/d6yy-54nr.json
                 winning_numbers = "08 30 41 48 54 04"  (5 mains + PB last)
  mega_millions: https://data.ny.gov/resource/5xaw-6ayf.json
                 winning_numbers = "14 21 51 55 65", mega_ball separate

Every row is validated against the game rules OF ITS ERA (lottery_rules.py):
main numbers distinct and within 1..main_max, bonus within 1..bonus_max for
the matrix in force on that draw date. Violations are logged loudly and
skipped — a burst of them means either a feed change or a game-rule change
that needs a new era entry, both of which should be seen, not swallowed.

Upsert is ON CONFLICT (game, draw_date) DO NOTHING — draw results are
immutable, so re-runs are cheap no-ops.

Usage (inside the api container):
    python /app/scripts/fetch_lottery_draws.py                 # incremental
    python /app/scripts/fetch_lottery_draws.py --backfill      # full history
    python /app/scripts/fetch_lottery_draws.py --game powerball
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime
from typing import Optional

import psycopg2
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_lottery_draws")

# Load lottery_rules directly by file path — the repo-root `services/`
# namespace package shadows services.api.src.services under pytest, so a
# package import is unreliable outside the container.
import importlib.util  # noqa: E402


def _load_rules():
    if "lottery_rules" in sys.modules:
        return sys.modules["lottery_rules"]
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        "/app/services/api/src/services/lottery_rules.py",
        os.path.join(here, "..", "services", "api", "src", "services", "lottery_rules.py"),
    ):
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("lottery_rules", candidate)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            sys.modules["lottery_rules"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("lottery_rules.py not found (container or repo checkout expected)")


rules_for = _load_rules().rules_for

SOCRATA_URLS = {
    "powerball": "https://data.ny.gov/resource/d6yy-54nr.json",
    "mega_millions": "https://data.ny.gov/resource/5xaw-6ayf.json",
}
GAMES = tuple(SOCRATA_URLS)
FETCH_LIMIT = 50000  # full history for both games is < 5k rows
TIMEOUT_S = 30


def parse_row(game: str, row: dict) -> Optional[dict]:
    """One Socrata record -> {draw_date, numbers, bonus_number, multiplier},
    or None (with a loud log) if malformed or invalid for its era."""
    raw_date = row.get("draw_date", "")
    try:
        draw_date = datetime.fromisoformat(raw_date.split(".")[0]).date()
    except ValueError:
        logger.error("%s: unparseable draw_date %r — skipping row", game, raw_date)
        return None

    try:
        parts = [int(x) for x in str(row.get("winning_numbers", "")).split()]
    except ValueError:
        logger.error("%s %s: unparseable winning_numbers %r", game, draw_date, row.get("winning_numbers"))
        return None

    multiplier: Optional[int] = None
    raw_mult = row.get("multiplier")
    if raw_mult not in (None, ""):
        try:
            multiplier = int(raw_mult)
        except ValueError:
            logger.warning("%s %s: unparseable multiplier %r — storing NULL", game, draw_date, raw_mult)

    if game == "powerball":
        # 5 mains + the Powerball packed into one field.
        if len(parts) != 6:
            logger.error("%s %s: expected 6 numbers, got %r", game, draw_date, parts)
            return None
        numbers, bonus = parts[:5], parts[5]
    else:
        if len(parts) != 5:
            logger.error("%s %s: expected 5 numbers, got %r", game, draw_date, parts)
            return None
        try:
            bonus = int(row.get("mega_ball", ""))
        except ValueError:
            logger.error("%s %s: unparseable mega_ball %r", game, draw_date, row.get("mega_ball"))
            return None
        numbers = parts

    era = rules_for(game, draw_date)
    if era is None:
        logger.error("%s %s: predates every known rules era — skipping", game, draw_date)
        return None
    if len(set(numbers)) != era.main_count or not all(1 <= n <= era.main_max for n in numbers):
        logger.error(
            "%s %s: mains %r invalid for era %s (5 distinct in 1..%d)",
            game,
            draw_date,
            numbers,
            era.start,
            era.main_max,
        )
        return None
    if not 1 <= bonus <= era.bonus_max:
        logger.error("%s %s: bonus %d invalid for era %s (1..%d)", game, draw_date, bonus, era.start, era.bonus_max)
        return None

    return {"draw_date": draw_date, "numbers": sorted(numbers), "bonus_number": bonus, "multiplier": multiplier}


def fetch_game(game: str, since: Optional[date]) -> list[dict]:
    """Fetch + parse all draws for a game, optionally only after `since`."""
    params: dict = {"$limit": FETCH_LIMIT, "$order": "draw_date ASC"}
    if since is not None:
        params["$where"] = f"draw_date > '{since.isoformat()}'"
    resp = requests.get(SOCRATA_URLS[game], params=params, timeout=TIMEOUT_S)
    resp.raise_for_status()
    rows = resp.json()
    parsed = [parse_row(game, r) for r in rows]
    good = [p for p in parsed if p is not None]
    bad = len(parsed) - len(good)
    if bad:
        logger.error("%s: %d/%d rows failed validation (see errors above)", game, bad, len(parsed))
    return good


def store_draws(cur, game: str, draws: list[dict]) -> int:
    written = 0
    for d in draws:
        cur.execute(
            """
            INSERT INTO lottery_draws (game, draw_date, numbers, bonus_number, multiplier)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (game, draw_date) DO NOTHING
            """,
            (game, d["draw_date"], d["numbers"], d["bonus_number"], d["multiplier"]),
        )
        written += cur.rowcount
    return written


def run(database_url: str, games: tuple, backfill: bool) -> dict:
    counts: dict = {}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            for game in games:
                since: Optional[date] = None
                if not backfill:
                    cur.execute("SELECT max(draw_date) FROM lottery_draws WHERE game = %s", (game,))
                    since = cur.fetchone()[0]
                draws = fetch_game(game, since)
                written = store_draws(cur, game, draws)
                counts[game] = written
                logger.info("%s: %d fetched, %d new rows written", game, len(draws), written)
            conn.commit()

    total_fetched = sum(counts.values())
    if backfill and total_fetched == 0:
        # A backfill that lands nothing means the feed or parser is broken —
        # fail loudly so the operator (or DAG) sees it.
        logger.error("Backfill wrote 0 rows across %s — feed or parser broken?", list(games))
        return {"counts": counts, "ok": False}
    return {"counts": counts, "ok": True}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", choices=GAMES, help="Only this game (default: both).")
    p.add_argument("--backfill", action="store_true", help="Fetch full history, not just new draws.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    games = (args.game,) if args.game else GAMES
    result = run(args.database_url, games, args.backfill)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
