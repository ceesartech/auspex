"""Generate mma value-bet recommendations from stored predictions + live odds.

MMA is the first 1v1 sport in the system — single market in v1
(moneyline). No spread, no total in the recommendation engine (the
fetch path captures totals odds but the model isn't trained on them
yet — that's a v2 follow-up).

Same EV / Kelly math as the team-sport engines:
  - Quarter Kelly stake sizing
  - 0.80 probability cap before EV/Kelly to defend against
    miscalibrated tails (mma ML models often emit 0.90+ on
    heavy favorites, which would size huge stakes)
  - 3% minimum EV threshold
  - 0.40 minimum raw model probability

Idempotent: drops pending mma recs for the match before re-inserting.

Telegram digest: this script does NOT queue alerts itself. The
precompute_predictions_mma script already queues high-confidence
picks; recommendation rows surface via the API + frontend.

Usage (inside the api container):
    python /app/scripts/generate_recommendations_mma.py
    python /app/scripts/generate_recommendations_mma.py --days 14 --ev-threshold 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

# Reuse the math helpers from the soccer engine — same Kelly + EV formulas.
sys.path.insert(0, os.path.dirname(__file__))

from generate_recommendations import confidence_rating, expected_value, get_bankroll, kelly_fraction  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("generate_recommendations_mma")

KELLY_FRACTION = 0.25

# Same defensive cap as NBA/NFL — mma moneyline models trained
# on ATP+WTA can emit 0.90+ on heavy favorites (Djokovic v lower-
# ranked qualifier). Cap before EV/Kelly to bound stake size.
PROB_CAP_FOR_EV = 0.80


def cap_prob(p: float, cap: float = PROB_CAP_FOR_EV) -> float:
    """Defensive cap on model probability for EV / Kelly math."""
    if p > cap:
        return cap
    return p


# Single market in v1 — moneyline only. Future commits can add
# total games / set-betting once linescore parsing lands in
# compute_features_mma.
MMA_MARKETS: dict[str, tuple[str, str]] = {
    "ensemble_mma_ml": ("moneyline", "Fight Winner"),
}

# Label index in probabilities dict (must match TaskSpec.labels in
# services/api/src/services/prediction_service.py). MMA 1v1 maps
# index 0 → "home" (player1), index 1 → "away" (player2).
MMA_LABELS: dict[str, list[str]] = {
    "moneyline": ["home", "away"],
}


# ── Per-market price + prob alignment ──────────────────────────────


def best_odds_for_market(cur, match_id: str, market_type: str) -> list[dict]:
    """Best (highest) pre-match decimal odds per selection. MMA
    moneyline has no line, so this is a simpler version of the
    NBA/NFL helpers — no target_line filter."""
    cur.execute(
        """
        SELECT DISTINCT ON (selection)
               selection, line, bookmaker, odds_decimal
        FROM odds
        WHERE match_id = %s AND market_type = %s AND NOT is_live
        ORDER BY selection, odds_decimal DESC
        """,
        (match_id, market_type),
    )
    return list(cur.fetchall())


def load_mma_predictions(cur, match_id: str) -> dict[str, dict]:
    """Latest mma prediction row per ensemble for one match.
    Returns {ensemble_name: {prediction_id, probabilities}}."""
    cur.execute(
        """
        SELECT DISTINCT ON (model_name)
               model_name, id::text AS prediction_id, probabilities
        FROM predictions
        WHERE match_id = %s AND model_name IN %s
        ORDER BY model_name, updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        """,
        (match_id, tuple(MMA_MARKETS.keys())),
    )
    out: dict[str, dict] = {}
    for r in cur.fetchall():
        out[r["model_name"]] = {
            "prediction_id": r["prediction_id"],
            "probabilities": r["probabilities"] or {},
        }
    return out


# ── Recommendation orchestration ────────────────────────────────────


def list_upcoming_mma(cur, days: int) -> list[str]:
    cur.execute(
        """
        SELECT m.id::text AS match_id
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'mma'
          AND m.status = 'scheduled'
          AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
        ORDER BY m.match_date ASC
        """,
        (str(days),),
    )
    return [r["match_id"] for r in cur.fetchall()]


def delete_pending(cur, match_id: str) -> None:
    """Idempotent: drop still-actionable mma picks before re-inserting.
    Never touches picks the user has placed or that have settled."""
    cur.execute(
        """
        DELETE FROM betting_recommendations
        WHERE match_id = %s
          AND status = 'pending'
          AND bet_type = 'moneyline'
        """,
        (match_id,),
    )


def insert_recommendation(cur, rec: dict) -> None:
    cur.execute(
        """
        INSERT INTO betting_recommendations
        (prediction_id, match_id, bet_type, selection, odds_at_recommendation,
         bookmaker, confidence_rating, expected_value, kelly_stake,
         recommended_stake, reasoning, risk_factors)
        VALUES (%(prediction_id)s, %(match_id)s, %(bet_type)s, %(selection)s,
                %(odds)s, %(bookmaker)s, %(conf)s, %(ev)s, %(kelly_stake)s,
                %(rec_stake)s, %(reasoning)s, %(risk)s::jsonb)
        """,
        rec,
    )


def _risk_factors(prob: float, odds_decimal: float) -> list[str]:
    risks: list[str] = []
    if odds_decimal >= 4.0:
        risks.append("longshot")
    if prob < 0.45:
        risks.append("low_model_probability")
    return risks


def recommend_for_match(
    cur,
    match_id: str,
    bankroll: float,
    ev_threshold: float,
    prob_floor: float,
) -> int:
    preds = load_mma_predictions(cur, match_id)
    if not preds:
        return 0

    delete_pending(cur, match_id)
    inserted = 0

    for ensemble_name, (market, market_label) in MMA_MARKETS.items():
        pred = preds.get(ensemble_name)
        if pred is None:
            continue
        labels = MMA_LABELS[market]
        probs = pred["probabilities"]

        for offer in best_odds_for_market(cur, match_id, market):
            selection = offer["selection"]
            if selection not in labels:
                continue
            raw_prob = float(probs.get(selection, 0.0))
            if raw_prob < prob_floor:
                continue
            prob = cap_prob(raw_prob)
            odds_decimal = float(offer["odds_decimal"])
            ev = expected_value(prob, odds_decimal)
            if ev < ev_threshold:
                continue
            k = kelly_fraction(prob, odds_decimal)
            stake = bankroll * k * KELLY_FRACTION

            cap_note = f" (capped from raw {raw_prob:.0%})" if prob < raw_prob else ""
            reasoning = (
                f"{market_label} {selection}: model {prob:.0%}{cap_note}, "
                f"book {1/odds_decimal:.0%} (@ {odds_decimal:.2f}) → "
                f"EV {ev:+.1%}, quarter-Kelly stake ${stake:.2f}."
            )
            rec = {
                "prediction_id": pred["prediction_id"],
                "match_id": match_id,
                "bet_type": market,
                "selection": selection,
                "odds": odds_decimal,
                "bookmaker": offer["bookmaker"],
                "conf": confidence_rating(ev, prob),
                "ev": ev,
                "kelly_stake": k,
                "rec_stake": round(stake, 2),
                "reasoning": reasoning,
                "risk": json.dumps(_risk_factors(prob, odds_decimal)),
            }
            insert_recommendation(cur, rec)
            inserted += 1

    return inserted


def run(database_url: str, days: int, ev_threshold: float, prob_floor: float) -> dict:
    counts = {"matches_processed": 0, "recommendations": 0}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            bankroll = get_bankroll(cur)
            logger.info("MMA bankroll for sizing: $%.2f", bankroll)
            matches = list_upcoming_mma(cur, days)
            if not matches:
                logger.info("No upcoming mma matches in the next %d days", days)
                return counts
            logger.info("Generating mma recommendations for %d matches", len(matches))
            for match_id in matches:
                inserted = recommend_for_match(cur, match_id, bankroll, ev_threshold, prob_floor)
                counts["matches_processed"] += 1
                counts["recommendations"] += inserted
            conn.commit()
    logger.info(
        "Wrote %d mma value-bet recommendations across %d matches",
        counts["recommendations"],
        counts["matches_processed"],
    )
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=14)
    p.add_argument(
        "--ev-threshold",
        type=float,
        default=0.03,
        help="Minimum positive EV for a pick to be recommended (default 0.03 = 3%).",
    )
    p.add_argument(
        "--prob-floor",
        type=float,
        default=0.40,
        help="Minimum model probability for a pick to be considered (default 0.40).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    counts = run(args.database_url, args.days, args.ev_threshold, args.prob_floor)
    logger.info("Done. %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
