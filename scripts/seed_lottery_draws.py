#!/usr/bin/env python3
"""Seed synthetic historical lottery draws for local dev / demo.

The general seed_dev_data.py is out of sync with the current schema, so this is
a standalone, schema-correct seeder for the `lottery_draws` table. It generates
uniformly-random draws for Powerball and Mega Millions across the most recent
scheduled draw dates, so the analysis / recommendation / backtest features have
data to work against.

The draws are uniformly random — which is exactly how real lottery draws behave.
This seed deliberately encodes NO pattern to "discover"; the engine's value is in
statistical-profile fitting and jackpot-share (EV) avoidance, not prediction.

Usage:
    python scripts/seed_lottery_draws.py --draws 200
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import date, timedelta

import psycopg2
from psycopg2.extras import execute_values

GAMES = {
    "powerball": {"main_count": 5, "main_max": 69, "bonus_max": 26, "weekdays": {0, 2, 5}},
    "mega_millions": {"main_count": 5, "main_max": 70, "bonus_max": 25, "weekdays": {1, 4}},
}


def recent_draw_dates(weekdays: set, n: int, today: date) -> list:
    """The `n` most recent scheduled draw dates on/before `today`."""
    dates = []
    d = today
    while len(dates) < n:
        if d.weekday() in weekdays:
            dates.append(d)
        d -= timedelta(days=1)
    return dates


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draws", type=int, default=200, help="Number of draws to seed per game (default 200).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible synthetic draws.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.database_url:
        print("DATABASE_URL not set")
        return 2

    rng = random.Random(args.seed)
    today = date.today()
    total = 0
    with psycopg2.connect(args.database_url) as conn:
        with conn.cursor() as cur:
            for game, cfg in GAMES.items():
                rows = []
                for dt in recent_draw_dates(cfg["weekdays"], args.draws, today):
                    numbers = sorted(rng.sample(range(1, cfg["main_max"] + 1), cfg["main_count"]))
                    bonus = rng.randint(1, cfg["bonus_max"])
                    jackpot = rng.choice([20, 40, 75, 120, 300, 500]) * 1_000_000
                    rows.append((game, dt, numbers, bonus, jackpot))
                execute_values(
                    cur,
                    """
                    INSERT INTO lottery_draws (game, draw_date, numbers, bonus_number, jackpot_amount)
                    VALUES %s
                    ON CONFLICT (game, draw_date) DO NOTHING
                    """,
                    rows,
                )
                total += len(rows)
                print(f"  {game}: prepared {len(rows)} draws")
        conn.commit()
    print(f"Done. Inserted up to {total} draws (existing dates skipped).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
