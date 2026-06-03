"""Market-consensus + online-learning predictions for horse races.

No training data available on Racing API Basic plan (no /results
endpoint), so v1 ships with a market-consensus baseline that works
from morning-line odds alone:

    raw_prob[i]     = 1 / morning_line_decimal[i]
    devigged[i]     = raw_prob[i] / sum(raw_prob)
    P(horse i wins) = devigged[i]

The devigging step removes the bookmaker overround so the
probabilities sum to 1.0 across the field. This is the standard
"closing-line wisdom-of-crowds" estimator — empirically the
hardest baseline to beat in horse racing markets.

Beyond v1, an online-learning layer slots in cleanly: as race
results land (via Racing API Standard upgrade or result scraping),
we adjust a small per-entity multiplier (jockey, trainer, course)
to Bayesian-update on observed outcomes. Until then the script
emits pure market-consensus predictions, which is genuinely useful
(it lets the rec engine + frontend operate on day-one data).

Output: one race_predictions row per (race, entrant) with
prediction_type='win', model_name='market_consensus_v1', and
confidence = devigged probability.

Usage:
    python /app/scripts/precompute_predictions_horse_racing.py
    python /app/scripts/precompute_predictions_horse_racing.py --days 3
    python /app/scripts/precompute_predictions_horse_racing.py --race-ids id1,id2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("precompute_predictions_horse_racing")

MODEL_NAME = "market_consensus_v1"
MODEL_VERSION = "1.0.0"

# Minimum number of entrants with morning-line odds we need to
# devigging confidently. Below 3, the field is so small that
# devigging just propagates whatever odds noise exists.
MIN_PRICED_ENTRANTS = 3

# Default per-entrant probability when no morning line is set —
# uniform over field size. Used as a fallback so even partially-
# priced races get a prediction.
def _uniform_prob(field_size: int) -> float:
    return 1.0 / max(field_size, 1)


def list_target_races(
    database_url: str,
    days: int,
    race_ids: Optional[list[str]],
    all_finished: bool = False,
) -> list[str]:
    """Race ids to score. By default: scheduled races in the next N days.
    With --race-ids: restricts to that list. With --all-finished:
    scores every finished race that doesn't already have a
    market_consensus_v1 prediction — used for retrospective backfill
    so the grader has something to evaluate."""
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if race_ids:
                cur.execute(
                    "SELECT id::text AS id FROM races WHERE id::text = ANY(%s)",
                    (race_ids,),
                )
                return [r["id"] for r in cur.fetchall()]
            if all_finished:
                # Backfill mode: finished races without any
                # market_consensus_v1 prediction yet. Skips races
                # already scored so re-running is cheap.
                cur.execute(
                    """
                    SELECT r.id::text AS id
                    FROM races r
                    WHERE r.status = 'finished'
                      AND NOT EXISTS (
                        SELECT 1 FROM race_predictions rp
                        WHERE rp.race_id = r.id
                          AND rp.model_name = %s
                          AND rp.model_version = %s
                      )
                    ORDER BY r.race_date ASC
                    """,
                    (MODEL_NAME, MODEL_VERSION),
                )
                return [r["id"] for r in cur.fetchall()]
            cur.execute(
                """
                SELECT id::text AS id FROM races
                WHERE status = 'scheduled'
                  AND race_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
                ORDER BY race_date ASC
                """,
                (str(days),),
            )
            return [r["id"] for r in cur.fetchall()]


def fetch_entrants(cur, race_id: str) -> list[dict]:
    """All non-scratched entrants for a race with their odds. Caller
    devigs across the returned set. Both morning_line_odds (pre-race
    bookmaker decimal) and starting_price (closing pari-mutuel) come
    back — devig() prefers ML when set and falls back to SP for
    retrospective scoring of historical races, where ML is NULL by
    design (the /results endpoint doesn't deliver pre-race lines)."""
    cur.execute(
        """
        SELECT
            e.id::text AS entrant_id,
            e.program_number,
            e.morning_line_odds,
            e.starting_price,
            e.scratched,
            h.name AS horse_name
        FROM race_entrants e
        JOIN horses h ON h.id = e.horse_id
        WHERE e.race_id = %s AND NOT e.scratched
        ORDER BY e.program_number ASC
        """,
        (race_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def _consensus_decimal(entrant: dict) -> Optional[float]:
    """Pick the odds value the consensus devig should treat as the
    market signal. Prefers morning_line_odds (live / upcoming races
    via the racecard endpoint); falls back to starting_price (post-
    race closing decimal from /results) for historical races where
    ML is NULL.

    Methodological note: using SP turns the baseline into a
    closing-line-consensus retrospect — strictly stronger than the
    morning line because the market has absorbed all pre-race
    information by the off. For evaluating the consensus baseline's
    floor accuracy on historical data this is the right signal; a
    real ML model that beats this is genuinely sharper than the
    market at the off."""
    ml = entrant.get("morning_line_odds")
    if ml is not None and float(ml) > 1.0:
        return float(ml)
    sp = entrant.get("starting_price")
    if sp is not None and float(sp) > 1.0:
        return float(sp)
    return None


def devig(entrants: list[dict]) -> dict[str, float]:
    """Compute devigged win probabilities per entrant. Returns
    {entrant_id: probability}. Probabilities sum to 1.0 across
    priced entrants; unpriced entrants (no usable decimal) get a
    uniform 1/field_size fallback then everything renormalises.

    Per-entrant signal precedence (see _consensus_decimal):
      morning_line_odds → starting_price → uniform fallback

    Examples:
      - 8-horse field, all with morning lines → exact devigging.
      - 8-horse field, 6 priced + 2 unpriced → priced devigged,
        unpriced get uniform, then full field renormalised so
        the row sums to 1.0.
      - 0 priced → all uniform.
    """
    field_size = len(entrants)
    if field_size == 0:
        return {}

    raw: dict[str, float] = {}
    priced_total = 0.0
    priced_count = 0
    for ent in entrants:
        decimal = _consensus_decimal(ent)
        if decimal is not None:
            inv = 1.0 / decimal
            raw[ent["entrant_id"]] = inv
            priced_total += inv
            priced_count += 1

    if priced_count < MIN_PRICED_ENTRANTS:
        # Not enough morning lines to devig — uniform across full field.
        uniform = _uniform_prob(field_size)
        return {ent["entrant_id"]: uniform for ent in entrants}

    # Devig the priced entries (they sum to 1.0 if priced_count ==
    # field_size; less if there are unpriced fillers).
    devigged: dict[str, float] = {eid: inv / priced_total for eid, inv in raw.items()}

    # Backfill unpriced entrants with uniform prior, then renormalise
    # to keep the whole-field sum at 1.0.
    uniform = _uniform_prob(field_size)
    total = sum(devigged.values())
    unpriced_eids = [ent["entrant_id"] for ent in entrants if ent["entrant_id"] not in devigged]
    if unpriced_eids:
        for eid in unpriced_eids:
            devigged[eid] = uniform
        # Renormalise everything so row sum = 1.0.
        s = sum(devigged.values())
        if s > 0:
            for eid in devigged:
                devigged[eid] = devigged[eid] / s
    return devigged


def store_prediction(
    cur,
    *,
    race_id: str,
    entrant_id: str,
    confidence: float,
    field_probs: dict[str, float],
) -> None:
    """Idempotent upsert on (race_id, entrant_id, model_name,
    model_version, prediction_type)."""
    cur.execute(
        """
        INSERT INTO race_predictions
            (race_id, entrant_id, model_name, model_version,
             prediction_type, confidence, probabilities, metadata)
        VALUES (%s, %s, %s, %s, 'win', %s, %s::jsonb, %s::jsonb)
        ON CONFLICT (race_id, entrant_id, model_name, model_version, prediction_type)
        DO UPDATE SET
            confidence = EXCLUDED.confidence,
            probabilities = EXCLUDED.probabilities,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        """,
        (
            race_id,
            entrant_id,
            MODEL_NAME,
            MODEL_VERSION,
            confidence,
            json.dumps(field_probs),
            json.dumps({"method": "morning_line_devig"}),
        ),
    )


def predict_for_race(cur, race_id: str) -> int:
    """Compute + store market-consensus predictions for one race.
    Returns the number of entrant rows written."""
    entrants = fetch_entrants(cur, race_id)
    if not entrants:
        return 0
    probs = devig(entrants)
    if not probs:
        return 0
    written = 0
    for ent in entrants:
        eid = ent["entrant_id"]
        if eid not in probs:
            continue
        store_prediction(
            cur,
            race_id=race_id,
            entrant_id=eid,
            confidence=probs[eid],
            field_probs=probs,
        )
        written += 1
    return written


def run(database_url: str, days: int, race_ids: Optional[list[str]], all_finished: bool = False) -> dict:
    targets = list_target_races(database_url, days, race_ids, all_finished=all_finished)
    if not targets:
        if all_finished:
            logger.info("No finished races without market_consensus_v1 predictions")
        else:
            logger.info("No upcoming races in the next %d days", days)
        return {"races": 0, "predictions": 0}

    logger.info("Scoring %d races with market-consensus baseline", len(targets))
    counts = {"races": 0, "predictions": 0}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for rid in targets:
                n = predict_for_race(cur, rid)
                if n > 0:
                    counts["races"] += 1
                    counts["predictions"] += n
            conn.commit()
    logger.info("Wrote %d predictions across %d races", counts["predictions"], counts["races"])
    return counts


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=2, help="Lookahead window in days (default 2).")
    p.add_argument("--race-ids", help="Comma-separated UUID list to score specific races.")
    p.add_argument(
        "--all-finished",
        action="store_true",
        help=(
            "Backfill: score every finished race that doesn't already "
            "have a market_consensus_v1 prediction. Uses starting_price "
            "as the consensus signal (morning_line_odds is NULL on "
            "historical /results rows). One-shot retrospective for "
            "grader / monitor evaluation."
        ),
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    race_ids = [s.strip() for s in args.race_ids.split(",") if s.strip()] if args.race_ids else None
    run(args.database_url, args.days, race_ids, all_finished=args.all_finished)
    return 0


if __name__ == "__main__":
    sys.exit(main())
