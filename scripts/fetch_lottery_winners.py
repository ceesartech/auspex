"""Backfill + capture per-tier winner counts and jackpot amounts for Mega
Millions draws (lottery v1.1).

Source: megamillions.com ASMX endpoint GetDrawDataByTick — for any historical
draw date (verified back to at least 2019) it returns:
  Drawing    — the numbers (cross-checked against our Socrata-ingested row)
  Jackpot    — CurrentPrizePool (advertised $), CurrentCashValue, Winners
  PrizeTiers — winner counts per tier, split by multiplier band

This unlocks the two honest v1.1 analyses (scripts/fit_lottery_sales_popularity.py):
  1. tickets sold per draw ~= winners(0+MB tier) x its 1-in-N odds
  2. empirical popularity: excess winners at mains-dependent tiers vs the
     winning line's human-bias features (birthday numbers etc.)

Powerball is NOT covered: powerball.com sits behind a CDN that blocks
non-browser clients (verified from both dev machine and the VM), and no free
per-draw winners feed exists for it. MM data alone carries the popularity
fit — human pick-bias is not game-specific.

Tier index mapping (verified against published results):
  0=jackpot 1=match_5 2=match_4_bonus 3=match_4 4=match_3_bonus
  5=match_3 6=match_2_bonus 7=match_1_bonus 8=match_bonus
Winner counts are summed across a tier's multiplier rows (current era splits
by 2x..10x; Megaplier-era rows split base vs IsMegaplier — summing both is
the consistent per-era total, and the fit's within-draw ratios cancel the
convention).

Usage (inside the api container):
    python /app/scripts/fetch_lottery_winners.py              # fill all missing
    python /app/scripts/fetch_lottery_winners.py --limit 20   # daily DAG mode
    python /app/scripts/fetch_lottery_winners.py --since 2025-04-08
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime
from typing import Optional

import psycopg2
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_lottery_winners")

MM_BYTICK_URL = "https://www.megamillions.com/cmspages/utilservice.asmx/GetDrawDataByTick"
HEADERS = {
    "Content-Type": "application/json",
    # The ASMX endpoint 500s on default python-requests fingerprints.
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0",
}
TIMEOUT_S = 20
SLEEP_BETWEEN_CALLS_S = 0.25

TIER_LABELS = [
    "jackpot",
    "match_5",
    "match_4_bonus",
    "match_4",
    "match_3_bonus",
    "match_3",
    "match_2_bonus",
    "match_1_bonus",
    "match_bonus",
]


def dotnet_ticks(d: date) -> int:
    """UTC-midnight .NET ticks (100ns units since 0001-01-01) for a date.
    Verified fixtures: 2026-07-31 -> 639210528000000000,
    2019-06-14 -> 636960672000000000."""
    return int((datetime(d.year, d.month, d.day) - datetime(1, 1, 1)).total_seconds() * 10**7)


def parse_bytick_payload(raw: dict) -> Optional[dict]:
    """Decode the double-encoded ASMX response into
    {numbers, bonus, jackpot_amount, cash_value, jackpot_winners,
     winners_by_tier, tier_rows}. Returns None (logged) on any shape
    surprise — a burst of Nones means the feed changed."""
    try:
        payload = json.loads(raw["d"])
        drawing = payload["Drawing"]
        jackpot = payload.get("Jackpot") or {}
        tier_rows = payload.get("PrizeTiers") or []
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        logger.error("Unparseable ByTick payload: %s", e)
        return None

    numbers = sorted(int(drawing[f"N{i}"]) for i in range(1, 6))
    bonus = int(drawing["MBall"])

    totals: dict[str, int] = {label: 0 for label in TIER_LABELS}
    detail: list[dict] = []
    for row in tier_rows:
        tier_idx = row.get("Tier")
        if not isinstance(tier_idx, int) or not 0 <= tier_idx < len(TIER_LABELS):
            logger.error("Unknown tier index %r in PrizeTiers — skipping row", tier_idx)
            continue
        winners = int(row.get("Winners") or 0)
        totals[TIER_LABELS[tier_idx]] += winners
        detail.append(
            {
                "tier": TIER_LABELS[tier_idx],
                "winners": winners,
                "multiplier": row.get("Multiplier"),
                "is_megaplier": bool(row.get("IsMegaplier")),
            }
        )

    jackpot_winners = int(jackpot.get("Winners") or 0)
    if totals["jackpot"] != jackpot_winners:
        # Not fatal — but the two sources inside one payload should agree.
        logger.warning(
            "Jackpot winner mismatch: PrizeTiers says %d, Jackpot block says %d",
            totals["jackpot"],
            jackpot_winners,
        )

    return {
        "numbers": numbers,
        "bonus": bonus,
        "jackpot_amount": float(jackpot.get("CurrentPrizePool") or 0) or None,
        "cash_value": float(jackpot.get("CurrentCashValue") or 0) or None,
        "jackpot_winners": jackpot_winners,
        "winners_by_tier": totals,
        "tier_rows": detail,
    }


def fetch_draw_winners(session: requests.Session, draw_date: date) -> Optional[dict]:
    resp = session.post(
        MM_BYTICK_URL,
        json={"PlayDateTicks": dotnet_ticks(draw_date)},
        headers=HEADERS,
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    return parse_bytick_payload(resp.json())


def list_missing(cur, since: Optional[date], limit: Optional[int]) -> list[tuple]:
    # winners_by_tier has DEFAULT '{}' (migration 001), so a never-filled row
    # is {} rather than NULL — an IS NULL test alone silently no-ops the
    # whole backfill ("No draws missing" while every row is empty).
    q = """
        SELECT draw_date, numbers, bonus_number
        FROM lottery_draws
        WHERE game = 'mega_millions'
          AND (winners_by_tier IS NULL OR winners_by_tier = '{}'::jsonb)
    """
    params: list = []
    if since:
        q += " AND draw_date >= %s"
        params.append(since)
    q += " ORDER BY draw_date DESC"
    if limit:
        q += " LIMIT %s"
        params.append(limit)
    cur.execute(q, params)
    return cur.fetchall()


def store_winners(cur, draw_date: date, parsed: dict) -> None:
    cur.execute(
        """
        UPDATE lottery_draws
        SET winners_by_tier = %s::jsonb,
            jackpot_amount = %s,
            metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
        WHERE game = 'mega_millions' AND draw_date = %s
        """,
        (
            json.dumps(parsed["winners_by_tier"]),
            parsed["jackpot_amount"],
            json.dumps(
                {
                    "cash_value": parsed["cash_value"],
                    "jackpot_winners": parsed["jackpot_winners"],
                    "tier_rows": parsed["tier_rows"],
                    "winners_source": "megamillions.com GetDrawDataByTick",
                }
            ),
            draw_date,
        ),
    )


def run(database_url: str, since: Optional[date], limit: Optional[int]) -> dict:
    counts = {"attempted": 0, "written": 0, "mismatched_numbers": 0, "failed": 0}
    session = requests.Session()
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            missing = list_missing(cur, since, limit)
            if not missing:
                logger.info("No mega_millions draws missing winners data")
                return counts
            logger.info("Fetching winners for %d draws", len(missing))
            for draw_date, our_numbers, our_bonus in missing:
                counts["attempted"] += 1
                try:
                    parsed = fetch_draw_winners(session, draw_date)
                except requests.RequestException as e:
                    logger.error("%s: fetch failed: %s", draw_date, e)
                    counts["failed"] += 1
                    time.sleep(SLEEP_BETWEEN_CALLS_S)
                    continue
                if parsed is None:
                    counts["failed"] += 1
                    time.sleep(SLEEP_BETWEEN_CALLS_S)
                    continue
                # Cross-check the feed's numbers against our Socrata row —
                # a mismatch means a tick/date bug or feed corruption; do NOT
                # store winners against the wrong draw.
                if parsed["numbers"] != sorted(our_numbers) or parsed["bonus"] != our_bonus:
                    logger.error(
                        "%s: numbers mismatch (feed %s+%d vs ours %s+%d) — skipping",
                        draw_date,
                        parsed["numbers"],
                        parsed["bonus"],
                        sorted(our_numbers),
                        our_bonus,
                    )
                    counts["mismatched_numbers"] += 1
                    time.sleep(SLEEP_BETWEEN_CALLS_S)
                    continue
                store_winners(cur, draw_date, parsed)
                counts["written"] += 1
                if counts["written"] % 100 == 0:
                    conn.commit()
                    logger.info("progress: %d/%d written", counts["written"], len(missing))
                time.sleep(SLEEP_BETWEEN_CALLS_S)
            conn.commit()
    logger.info(
        "Done: %(written)d written, %(failed)d failed, %(mismatched_numbers)d mismatched of %(attempted)d",
        counts,
    )
    return counts


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", type=date.fromisoformat, help="Only draws on/after this date (YYYY-MM-DD).")
    p.add_argument("--limit", type=int, help="Max draws to fetch this run (daily DAG uses a small limit).")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    counts = run(args.database_url, args.since, args.limit)
    # Total failure with work to do = feed broken -> loud non-zero for the DAG.
    if counts["attempted"] > 0 and counts["written"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
