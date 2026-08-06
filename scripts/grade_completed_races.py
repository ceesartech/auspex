"""Grade race_predictions + settle race_recommendations for finished races.

Mirrors the matches-based grader (scripts/grade_completed_matches.py)
but operates on the horse-racing schema (migration 013): races,
race_entrants, race_predictions, race_recommendations.

For each finished race with finish_position=1 on some entrant:

  1. UPDATE race_predictions:
       actual_outcome = 1.0 if this entrant won, else 0.0
                        (entrants without a finish_position — typically
                        scratched / non-runners — get actual_outcome NULL)
       is_correct     = TRUE if the consensus favourite for THIS race
                        (the entrant with the highest confidence) won,
                        else FALSE. Same value applied to every row in
                        the race because horse-racing "the model picked
                        the winner" is a race-level fact, not per-entrant.

  2. UPDATE race_recommendations:
       status = 'won'  if selection's entrant won
              | 'void' if selection scratched / non-runner / race abandoned
              | 'lost' otherwise
       actual_result = winning horse name (or 'void' on cancellation)
       profit_loss   = stake × (odds - 1)  on won
                     = -stake               on lost
                     =  0                   on void
       settled_at    = NOW()

Idempotent: race_predictions rows with is_correct IS NOT NULL are
skipped, and race_recommendations rows with status NOT IN ('pending',
'placed') are skipped. Re-running on the same window is a cheap no-op
unless new predictions / recs landed AFTER the previous run.

Usage (inside the api container):
    python /app/scripts/grade_completed_races.py
    python /app/scripts/grade_completed_races.py --days 14
    python /app/scripts/grade_completed_races.py --race-id <uuid>
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from decimal import Decimal
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("grade_completed_races")


# ── DB I/O ────────────────────────────────────────────────────────────


# UK/IRE each-way place terms — MUST mirror ew_terms in
# generate_recommendations_horse_racing.py (places paid by field size).
def ew_places(field_size: int) -> int:
    if field_size <= 4:
        return 0
    if field_size <= 7:
        return 2
    return 3


def list_finished_races(cur, days: int, race_id: Optional[str]) -> list[dict]:
    """Finished races needing grading. --race-id short-circuits to one
    row (operator escape hatch for ad-hoc re-grading). Otherwise: races
    finished in the last `days` AND with at least one ungraded
    prediction OR unsettled rec."""
    if race_id:
        cur.execute(
            "SELECT id::text AS race_id FROM races WHERE id::text = %s",
            (race_id,),
        )
        return list(cur.fetchall())

    cur.execute(
        """
        SELECT r.id::text AS race_id
        FROM races r
        WHERE r.status = 'finished'
          AND r.race_date >= NOW() - (%s || ' days')::interval
          AND (
                EXISTS (
                  SELECT 1 FROM race_predictions rp
                  WHERE rp.race_id = r.id AND rp.is_correct IS NULL
                )
             OR EXISTS (
                  SELECT 1 FROM race_recommendations rec
                  WHERE rec.race_id = r.id AND rec.status IN ('pending', 'placed')
                )
          )
        ORDER BY r.race_date ASC
        """,
        (str(days),),
    )
    return list(cur.fetchall())


def fetch_race_outcome(cur, race_id: str) -> Optional[dict]:
    """Winner + horse-name lookup for one race. Returns
    {winner_entrant_id, winner_horse_name} or None if no entrant has
    finish_position=1 (typical for cancelled / abandoned races)."""
    cur.execute(
        """
        SELECT e.id::text AS winner_entrant_id, h.name AS winner_horse_name
        FROM race_entrants e
        JOIN horses h ON h.id = e.horse_id
        WHERE e.race_id = %s AND e.finish_position = 1
        ORDER BY e.disqualified ASC  -- prefer non-DQ if multiple rows
        LIMIT 1
        """,
        (race_id,),
    )
    row = cur.fetchone()
    return row if row else None


def fetch_consensus_favourite(cur, race_id: str) -> Optional[str]:
    """Highest-confidence entrant_id for the market_consensus_v1 model
    in this race. Returns None if no predictions exist (graded races
    that never had predictions written — rare, but possible for
    backfilled finished races)."""
    cur.execute(
        """
        SELECT entrant_id::text AS entrant_id
        FROM race_predictions
        WHERE race_id = %s
          AND model_name = 'market_consensus_v1'
          AND prediction_type = 'win'
        ORDER BY confidence DESC NULLS LAST
        LIMIT 1
        """,
        (race_id,),
    )
    row = cur.fetchone()
    return row["entrant_id"] if row else None


def grade_race_predictions(
    cur, race_id: str, winner_entrant_id: Optional[str], favourite_entrant_id: Optional[str]
) -> int:
    """UPDATE every ungraded race_predictions row in this race.
    actual_outcome is per-row (did THIS entrant win); is_correct is
    per-race (did the consensus pick win) replicated across rows so
    downstream metrics queries can ROLLUP cleanly. Returns the row
    count touched."""
    if winner_entrant_id is None:
        # Race finished without a recorded winner (abandoned, void).
        # Leave actual_outcome NULL but mark is_correct=FALSE so the
        # row doesn't sit in the ungraded queue forever — downstream
        # accuracy queries can filter on actual_outcome IS NOT NULL.
        cur.execute(
            """
            UPDATE race_predictions
            SET is_correct = FALSE,
                updated_at = NOW()
            WHERE race_id = %s
              AND is_correct IS NULL
            """,
            (race_id,),
        )
        return cur.rowcount

    favourite_won = favourite_entrant_id is not None and favourite_entrant_id == winner_entrant_id
    cur.execute(
        """
        UPDATE race_predictions
        SET actual_outcome = CASE WHEN entrant_id::text = %s THEN 1.0 ELSE 0.0 END,
            is_correct     = %s,
            updated_at     = NOW()
        WHERE race_id = %s
          AND is_correct IS NULL
        """,
        (winner_entrant_id, favourite_won, race_id),
    )
    return cur.rowcount


def settle_recommendations(
    cur,
    race_id: str,
    winner_entrant_id: Optional[str],
    winner_horse_name: Optional[str],
) -> dict:
    """Walk pending/placed recs for the race and settle them. Returns
    counts of {won, lost, void} for the run summary."""
    cur.execute(
        """
        SELECT
            rec.id::text         AS rec_id,
            rec.entrant_id::text AS entrant_id,
            rec.selection,
            rec.recommended_stake,
            rec.odds_at_recommendation,
            rec.bet_type,
            e.scratched          AS entrant_scratched,
            e.finish_position
        FROM race_recommendations rec
        JOIN race_entrants e ON e.id = rec.entrant_id
        WHERE rec.race_id = %s
          AND rec.status IN ('pending', 'placed')
          AND rec.bet_type IN ('win', 'place')
        """,
        (race_id,),
    )
    rows = list(cur.fetchall())
    # Field size for each-way place terms: non-scratched runners.
    cur.execute(
        "SELECT count(*) AS n FROM race_entrants WHERE race_id = %s AND NOT scratched",
        (race_id,),
    )
    places_paid = ew_places(int(cur.fetchone()["n"]))

    counts = {"won": 0, "lost": 0, "void": 0}
    for r in rows:
        stake = r["recommended_stake"] or Decimal("0")
        odds = r["odds_at_recommendation"] or Decimal("0")

        if winner_entrant_id is None:
            # Race void / abandoned — refund the implied stake.
            status = "void"
            profit_loss = Decimal("0")
            actual_result = "void"
        elif r["entrant_scratched"]:
            status = "void"
            profit_loss = Decimal("0")
            actual_result = winner_horse_name or "void"
        elif r["bet_type"] == "place":
            fp = r["finish_position"]
            if places_paid >= 2 and fp is not None and 1 <= fp <= places_paid:
                status = "won"
                profit_loss = stake * (odds - Decimal("1"))
            else:
                status = "lost"
                profit_loss = -stake
            actual_result = f"finished {fp}" if fp is not None else "unplaced"
        elif r["entrant_id"] == winner_entrant_id:
            status = "won"
            profit_loss = stake * (odds - Decimal("1"))
            actual_result = winner_horse_name or r["selection"]
        else:
            status = "lost"
            profit_loss = -stake
            actual_result = winner_horse_name or ""

        counts[status] += 1
        cur.execute(
            """
            UPDATE race_recommendations
            SET status        = %s,
                actual_result = %s,
                profit_loss   = %s,
                settled_at    = NOW(),
                updated_at    = NOW()
            WHERE id = %s
            """,
            (status, actual_result, profit_loss, r["rec_id"]),
        )
    return counts


def grade_race(cur, race_id: str) -> dict:
    """End-to-end grade one race. Returns per-race counters for the
    run summary."""
    outcome = fetch_race_outcome(cur, race_id)
    winner_entrant_id = outcome["winner_entrant_id"] if outcome else None
    winner_horse_name = outcome["winner_horse_name"] if outcome else None
    favourite_entrant_id = fetch_consensus_favourite(cur, race_id)

    predictions_graded = grade_race_predictions(cur, race_id, winner_entrant_id, favourite_entrant_id)
    rec_counts = settle_recommendations(cur, race_id, winner_entrant_id, winner_horse_name)
    return {
        "predictions_graded": predictions_graded,
        "recs_won": rec_counts["won"],
        "recs_lost": rec_counts["lost"],
        "recs_void": rec_counts["void"],
        "winner_known": winner_entrant_id is not None,
    }


# ── Orchestration ─────────────────────────────────────────────────────


def run(database_url: str, days: int, race_id: Optional[str]) -> dict:
    totals = {
        "races_processed": 0,
        "races_with_winner": 0,
        "predictions_graded": 0,
        "recs_won": 0,
        "recs_lost": 0,
        "recs_void": 0,
    }
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            races = list_finished_races(cur, days, race_id)
            if not races:
                logger.info("No finished races needing grading in last %d days", days)
                return totals
            logger.info("Grading %d finished races", len(races))
            for r in races:
                counts = grade_race(cur, r["race_id"])
                totals["races_processed"] += 1
                totals["races_with_winner"] += 1 if counts["winner_known"] else 0
                for k in ("predictions_graded", "recs_won", "recs_lost", "recs_void"):
                    totals[k] += counts[k]
            conn.commit()

    logger.info(
        "Graded %d races (%d with known winner): %d predictions; " "recs won=%d lost=%d void=%d",
        totals["races_processed"],
        totals["races_with_winner"],
        totals["predictions_graded"],
        totals["recs_won"],
        totals["recs_lost"],
        totals["recs_void"],
    )
    return totals


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=14, help="Lookback window in days (default 14).")
    p.add_argument("--race-id", help="Grade only one race by UUID (operator escape hatch).")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    run(args.database_url, args.days, args.race_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
