"""Backtest lottery combination strategies — honestly.

Two modes:

  --generate   Store one line per strategy for each game's NEXT scheduled draw,
               so there's a systematic record to settle later. Run this on a
               schedule (e.g. before each draw).

  (default)    Settle any persisted lines whose target draw has now happened
               (fill matched_main / matched_bonus / prize_tier), then print a
               per-strategy report.

IMPORTANT: lottery draws are uniform and independent. This measures strategy
behavior; observed prize-rates are expected to track pure chance (any-prize odds
are ~1 in 24.9 for Powerball, ~1 in 24 for Mega Millions, regardless of how the
numbers were chosen). The report prints observed vs. theoretical so the result
is never mistaken for an edge on winning.

Usage (inside the api container):
    python /app/scripts/lottery_backtest.py --generate
    python /app/scripts/lottery_backtest.py            # settle + report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("lottery_backtest")

# Make the pure engine importable both in the api container (/app) and from a
# repo checkout (running the script directly).
for _p in (
    "/app/services/api/src",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "services", "api", "src")),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from services.lottery_analysis import (  # noqa: E402
    GAME_CONFIG,
    generate_combinations,
    get_game_config,
    settle,
)

# Scheduled draw weekdays (Mon=0 … Sun=6): Powerball Mon/Wed/Sat, MM Tue/Fri.
DRAW_WEEKDAYS = {"powerball": {0, 2, 5}, "mega_millions": {1, 4}}
# Approx. odds of winning ANY prize, for the honesty anchor in the report.
ANY_PRIZE_ODDS = {"powerball": 24.9, "mega_millions": 24.0}
STRATEGIES = ("blend", "statistical", "ev", "hot", "due", "random")


def next_draw_date(game: str, today: date) -> date | None:
    days = DRAW_WEEKDAYS.get(game, set())
    if not days:
        return None
    for i in range(1, 8):
        d = today + timedelta(days=i)
        if d.weekday() in days:
            return d
    return None


def load_draws(cur, game: str, limit: int = 200):
    cur.execute(
        "SELECT numbers, bonus_number FROM lottery_draws WHERE game = %s ORDER BY draw_date DESC LIMIT %s",
        (game, limit),
    )
    rows = cur.fetchall()
    return [list(r["numbers"]) for r in rows], [r["bonus_number"] for r in rows]


def run_generate(conn, today: date, seed: int | None) -> int:
    written = 0
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        for game in GAME_CONFIG:
            cfg = get_game_config(game)
            target = next_draw_date(game, today)
            main, bonus = load_draws(cur, game)
            if not main:
                logger.warning("%s: no draws to analyze — skipping generation", game)
                continue
            for strat in STRATEGIES:
                combos = generate_combinations(main, bonus, cfg, strategy=strat, n=1, seed=seed)
                for c in combos:
                    cur.execute(
                        """
                        INSERT INTO lottery_predictions
                        (game, strategy, numbers, bonus_number, score, features,
                         rationale, target_draw_date)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                        """,
                        (
                            game,
                            strat,
                            c.numbers,
                            c.bonus_number,
                            c.score,
                            json.dumps(c.features),
                            c.rationale,
                            target,
                        ),
                    )
                    written += 1
        conn.commit()
    logger.info("Generated %d lines for the next draw(s)", written)
    return written


def run_settle(conn) -> int:
    settled = 0
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT lp.id, lp.game, lp.numbers, lp.bonus_number,
                   ld.numbers AS draw_numbers, ld.bonus_number AS draw_bonus
            FROM lottery_predictions lp
            JOIN lottery_draws ld
              ON ld.game = lp.game AND ld.draw_date = lp.target_draw_date
            WHERE lp.settled_at IS NULL AND lp.target_draw_date IS NOT NULL
            """)
        rows = cur.fetchall()
        for r in rows:
            res = settle(r["numbers"], r["bonus_number"], r["draw_numbers"], r["draw_bonus"], r["game"])
            cur.execute(
                """
                UPDATE lottery_predictions
                SET matched_main = %s, matched_bonus = %s, prize_tier = %s, settled_at = NOW()
                WHERE id = %s
                """,
                (res["matched_main"], res["matched_bonus"], res["prize_tier"], r["id"]),
            )
            settled += 1
        conn.commit()
    logger.info("Settled %d prediction(s)", settled)
    return settled


def run_report(conn) -> None:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT game, strategy,
                   COUNT(*) AS n,
                   COUNT(*) FILTER (WHERE prize_tier IS NOT NULL) AS prizes,
                   COUNT(*) FILTER (WHERE prize_tier = 'jackpot') AS jackpots,
                   AVG(matched_main)::float AS avg_main
            FROM lottery_predictions
            WHERE settled_at IS NOT NULL
            GROUP BY game, strategy
            ORDER BY game, strategy
            """)
        rows = cur.fetchall()
    if not rows:
        logger.info("No settled predictions yet — run with --generate before draws, then settle after.")
        return

    print("\nLottery strategy backtest (settled lines)")
    print("=" * 78)
    print(f"{'game':<14}{'strategy':<12}{'n':>6}{'prize%':>9}{'avg_main':>10}{'jackpots':>10}")
    print("-" * 78)
    current_game = None
    for r in rows:
        if r["game"] != current_game:
            current_game = r["game"]
            theo = ANY_PRIZE_ODDS.get(current_game)
            if theo:
                print(f"  [{current_game}] theoretical any-prize rate ≈ {100 / theo:.2f}% (1 in {theo})")
        prize_pct = (r["prizes"] / r["n"] * 100) if r["n"] else 0.0
        print(
            f"{r['game']:<14}{r['strategy']:<12}{r['n']:>6}{prize_pct:>8.2f}%"
            f"{(r['avg_main'] or 0):>10.3f}{r['jackpots']:>10}"
        )
    print("=" * 78)
    print(
        "Note: prize-rate differences between strategies are noise — number choice "
        "does not change win odds. The EV edge is jackpot-share avoidance (payout "
        "size when you win), which this hit-rate report does not capture."
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generate", action="store_true", help="Generate + store lines for the next draw (per strategy).")
    p.add_argument("--seed", type=int, default=None, help="Seed for reproducible --generate output.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    with psycopg2.connect(args.database_url) as conn:
        if args.generate:
            run_generate(conn, date.today(), args.seed)
        else:
            run_settle(conn)
            run_report(conn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
