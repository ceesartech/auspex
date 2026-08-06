"""Grade predictions + settle recommendations for finished matches.

The closing-the-loop step that Phase 5 introduces. For every match
with status='finished' in the lookback window we:

  1. Compute actual_outcome per (prediction_type, model_name) via
     scripts/grading_outcomes.py — handles soccer + NHL markets.
  2. UPDATE predictions: actual_outcome, is_correct, profit_loss for
     every row whose is_correct is still NULL.
  3. UPDATE betting_recommendations: status (won/lost/void),
     actual_result, profit_loss, settled_at for every row whose status
     is still 'pending' or 'placed'.

Idempotent: rows already graded (is_correct IS NOT NULL) are left
alone, so re-running on the same window is a cheap no-op except for
matches that were freshly graded or whose predictions/recs landed
after the previous run.

The existing DB trigger calculate_match_outcome (migration 003)
covers soccer match_result / over_under / btts at the moment
status flips to 'finished'. This script is the comprehensive
follow-up: it grades NHL, covers markets the trigger misses, and
re-grades any predictions that were written AFTER the status flip
(the trigger won't fire again on those).

Usage (inside the api container):
    python /app/scripts/grade_completed_matches.py
    python /app/scripts/grade_completed_matches.py --days 30
    python /app/scripts/grade_completed_matches.py --match-id <uuid>  # ad-hoc
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

# Make the shared grading-outcomes helper importable.
sys.path.insert(0, os.path.dirname(__file__))

from grading_outcomes import actual_outcome, grade_prediction, grade_rec_selection, rec_status_and_pl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("grade_completed_matches")


# ── DB I/O ────────────────────────────────────────────────────────────


def list_finished_matches(cur, days: int, match_id: Optional[str]) -> list[dict]:
    """Finished matches needing grading. If --match-id is given we
    short-circuit to that one row (operator escape hatch for ad-hoc
    re-grading). Otherwise: matches that finished in the last `days`
    days AND have at least one ungraded prediction OR unsettled rec.
    The EXISTS subqueries keep this from churning over already-graded
    matches on every run."""
    if match_id is not None:
        cur.execute(
            """
            SELECT id::text AS match_id,
                   home_score, away_score, status, metadata
            FROM matches
            WHERE id = %s AND status = 'finished'
            """,
            (match_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    cur.execute(
        """
        SELECT m.id::text AS match_id,
               m.home_score, m.away_score, m.status, m.metadata
        FROM matches m
        WHERE m.status = 'finished'
          AND m.match_date >= NOW() - (%s || ' days')::interval
          AND (
              EXISTS (
                  SELECT 1 FROM predictions p
                  WHERE p.match_id = m.id AND p.is_correct IS NULL
              )
              OR EXISTS (
                  SELECT 1 FROM betting_recommendations br
                  WHERE br.match_id = m.id
                    AND (
                          br.status IN ('pending', 'placed')
                       -- trigger-settled but P&L never computed (the
                       -- migration-011 trigger sets status only) —
                       -- must reach grade_match for the money fill
                       OR (br.status IN ('won', 'lost', 'void')
                           AND br.profit_loss IS NULL)
                    )
              )
          )
        ORDER BY m.match_date ASC
        """,
        (str(days),),
    )
    return [dict(r) for r in cur.fetchall()]


def list_ungraded_predictions(cur, match_id: str) -> list[dict]:
    """Every prediction row for the match whose is_correct is still
    NULL. We grade and update each in place."""
    cur.execute(
        """
        SELECT id::text AS prediction_id,
               prediction_type, model_name, predicted_outcome
        FROM predictions
        WHERE match_id = %s AND is_correct IS NULL
        """,
        (match_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def fetch_match_features(cur, match_id: str) -> Optional[dict]:
    """Most-recent features_cache JSON for the match, regardless of
    feature_set. Used by NBA + NFL spread/total grading to read the
    closing line the model conditioned on (closing_spread_home /
    closing_total_line). Returns None if no features were ever cached
    — caller falls back to grading without line context, which means
    NBA/NFL spread/total predictions for that match stay ungraded
    until a feature row appears.

    Returns the freshest row by computed_at; for matches with multiple
    feature_sets, the latest pipeline write wins (nba_baseline for
    NBA, nfl_baseline for NFL, nhl_baseline for NHL, baseline for
    soccer). The dispatch only reads the shared closing-line keys,
    so soccer/NHL features pass through harmlessly."""
    cur.execute(
        """
        SELECT features
        FROM features_cache
        WHERE match_id = %s
        ORDER BY computed_at DESC
        LIMIT 1
        """,
        (match_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return row.get("features") or None


def update_prediction(cur, prediction_id: str, actual: Optional[str], correct: Optional[bool]) -> None:
    """Persist one prediction's grade. actual_outcome is set even on
    pushes so the recommendation settler can detect the push from the
    prediction row alone (without re-running the grading math)."""
    cur.execute(
        """
        UPDATE predictions
        SET actual_outcome = %s,
            is_correct = %s,
            updated_at = NOW()
        WHERE id = %s
        """,
        (actual, correct, prediction_id),
    )


def list_open_recs_for_match(cur, match_id: str) -> list[dict]:
    """Recommendations needing settlement math. Two shapes:
    (1) still in flight (pending/placed) — need status + P&L + settled_at;
    (2) already flipped won/lost/void by the migration-011 match-outcome
        DB TRIGGER, which sets status/actual_result but computes NO
        profit_loss — without this branch those rows stay P&L-less
        forever and every ROI rollup silently understates. The settle
        loop recomputes the same status and fills the money columns."""
    cur.execute(
        """
        SELECT br.id::text AS rec_id,
               br.prediction_id::text AS prediction_id,
               br.bet_type, br.selection,
               br.recommended_stake, br.odds_at_recommendation, br.status,
               p.actual_outcome AS p_actual_outcome,
               p.is_correct AS p_is_correct
        FROM betting_recommendations br
        LEFT JOIN predictions p ON p.id = br.prediction_id
        WHERE br.match_id = %s
          AND (
                br.status IN ('pending', 'placed')
             OR (br.status IN ('won', 'lost', 'void') AND br.profit_loss IS NULL)
          )
        """,
        (match_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def settle_recommendation(
    cur,
    rec_id: str,
    new_status: str,
    actual_result: Optional[str],
    profit_loss: Decimal,
) -> None:
    """Apply the won/lost/void settlement. settled_at goes to NOW()."""
    cur.execute(
        """
        UPDATE betting_recommendations
        SET status = %s,
            actual_result = %s,
            profit_loss = %s,
            settled_at = NOW(),
            updated_at = NOW()
        WHERE id = %s
        """,
        (new_status, actual_result, profit_loss, rec_id),
    )


# ── Math: profit_loss given outcome + stake + odds ───────────────────


def profit_loss_for(
    is_correct: Optional[bool],
    push: bool,
    stake: Decimal,
    odds: Decimal,
) -> tuple[str, Decimal]:
    """Compute (new_status, profit_loss) from a settled outcome.

    Win:  status='won',  P/L = stake × (odds - 1)
    Loss: status='lost', P/L = -stake
    Push: status='void', P/L = 0
    """
    if push:
        return "void", Decimal("0")
    if is_correct is True:
        return "won", (stake * (odds - Decimal("1"))).quantize(Decimal("0.01"))
    return "lost", (-stake).quantize(Decimal("0.01"))


# ── Orchestration ─────────────────────────────────────────────────────


def grade_match(cur, match: dict) -> dict:
    """Grade every ungraded prediction + settle every open
    recommendation for one match. Returns counts for the run summary."""
    counts = {"predictions_graded": 0, "predictions_skipped": 0, "recs_settled": 0}

    # Features carry the closing line for NBA spread/total grading.
    # Fetched once per match instead of per-prediction.
    features = fetch_match_features(cur, match["match_id"])

    for pred in list_ungraded_predictions(cur, match["match_id"]):
        actual = actual_outcome(
            prediction_type=pred["prediction_type"],
            model_name=pred["model_name"],
            predicted_outcome=pred["predicted_outcome"],
            home_score=match["home_score"],
            away_score=match["away_score"],
            metadata=match["metadata"],
            features=features,
        )
        if actual is None:
            counts["predictions_skipped"] += 1
            continue
        correct = grade_prediction(pred["predicted_outcome"], actual, prediction_type=pred["prediction_type"])
        update_prediction(cur, pred["prediction_id"], actual, correct)
        counts["predictions_graded"] += 1

    # Recs join to predictions for the actual_outcome — the prediction
    # update we just did is visible in the same txn via the join.
    # Settlement grades each rec's OWN (bet_type, selection) against the
    # final score — NOT the prediction's argmax verdict, which is wrong for
    # any non-argmax selection and for every lined market (audit §7
    # 2026-08-06). Ungradable selections stay open rather than guess.
    for rec in list_open_recs_for_match(cur, match["match_id"]):
        if match["home_score"] is None or match["away_score"] is None:
            continue
        outcome = grade_rec_selection(rec["bet_type"], rec["selection"], match["home_score"], match["away_score"])
        if outcome is None:
            continue
        new_status, pl = rec_status_and_pl(outcome, rec["recommended_stake"], rec["odds_at_recommendation"])
        settle_recommendation(cur, rec["rec_id"], new_status, outcome, pl)
        counts["recs_settled"] += 1

    return counts


def run(database_url: str, days: int, match_id: Optional[str]) -> dict:
    summary = {
        "matches_processed": 0,
        "predictions_graded": 0,
        "predictions_skipped": 0,
        "recs_settled": 0,
    }
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            matches = list_finished_matches(cur, days, match_id)
            if not matches:
                logger.info("No finished matches needing grading in the last %d days", days)
                return summary
            logger.info("Grading %d finished matches", len(matches))
            for m in matches:
                per = grade_match(cur, m)
                summary["matches_processed"] += 1
                for k in ("predictions_graded", "predictions_skipped", "recs_settled"):
                    summary[k] += per[k]
            conn.commit()
    logger.info(
        "Done. matches=%d, predictions_graded=%d (skipped=%d), recs_settled=%d",
        summary["matches_processed"],
        summary["predictions_graded"],
        summary["predictions_skipped"],
        summary["recs_settled"],
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--days",
        type=int,
        default=14,
        help="Lookback window for finished matches (default 14).",
    )
    p.add_argument(
        "--match-id",
        default=None,
        help="Grade ONE specific match by UUID (operator escape hatch).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    run(args.database_url, args.days, args.match_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
