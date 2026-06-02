"""Generate NFL value-bet recommendations from stored predictions + live odds.

NFL-specific value-bet engine — parallel to scripts/generate_recommendations_nba.py
with the same line-as-feature design constraints.

NFL spread + total are LINE-AS-FEATURE: one trained ensemble per market
emits P(home covers <closing_line>) where the closing line was baked
into the input features. The model never independently knows about
lines other than the closing one. So we can only recommend bets at
lines NEAR the closing line — at the snapshot's line the probability
is correct; at a meaningfully different line it's not.

v1 keeps this simple: recommend ONLY at the same line we trained
against. ±0.5 tolerance lets us pick up books that differ by a half
point. NFL has key numbers (3, 7, 10, 14) — a future commit could
add a key-number gate to avoid recommending bets on lines that
straddle one.

Telegram digest: this script does NOT queue alerts itself. The
precompute_predictions_nfl script already queues high-confidence
picks; recommendation rows surface in the API + frontend.

Usage (inside the api container):
    python /app/scripts/generate_recommendations_nfl.py
    python /app/scripts/generate_recommendations_nfl.py --days 14 --ev-threshold 0.05
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
logger = logging.getLogger("generate_recommendations_nfl")

# Quarter Kelly — same fraction as soccer + NBA engines.
KELLY_FRACTION = 0.25

# Cap model probability before EV / Kelly math. Same defensive cap as
# NBA — NFL ensembles will be similarly miscalibrated on the tail
# (small training corpus, ~285 games/season). 0.80 leaves a safety
# margin against worst-bucket overconfidence. A future commit can
# replace the hard cap with a learned calibrator once we have
# settled bets to fit it on.
PROB_CAP_FOR_EV = 0.80


def cap_prob(p: float, cap: float = PROB_CAP_FOR_EV) -> float:
    """Defensive cap on model probability for EV / Kelly math."""
    if p > cap:
        return cap
    return p


# Ensemble names → (prediction_type, market_label).
NFL_MARKETS: dict[str, tuple[str, str]] = {
    "ensemble_nfl_ml": ("moneyline", "Moneyline"),
    "ensemble_nfl_sp": ("spread", "Spread"),
    "ensemble_nfl_tot": ("total", "Total"),
}

# Label index in probabilities dict (must match TaskSpec.labels in
# services/api/src/services/prediction_service.py).
NFL_LABELS: dict[str, list[str]] = {
    "moneyline": ["home", "away"],
    "spread": ["home", "away"],
    "total": ["over", "under"],
}


# ── Per-market price + prob alignment ──────────────────────────────


def closing_line_for_match(cur, match_id: str, market_type: str) -> float | None:
    """Most-recent pre-match line offered (averaged across books) for
    the spread or total market on this match. Same home-perspective
    pinning as NBA: odds rows for home carry the signed home line
    (-7 for a 7-point favorite), and away rows the inverse (+7);
    averaging both perspectives gives ~0 for any matchup with a
    favorite and fails the ±0.5 filter. Pin to home so the target_line
    is the signed home line."""
    selection_filter = "selection = 'home'" if market_type == "spread" else ""
    where_clauses = [
        "match_id = %s",
        "market_type = %s",
        "NOT is_live",
        "line IS NOT NULL",
    ]
    if selection_filter:
        where_clauses.append(selection_filter)
    cur.execute(
        f"""
        SELECT AVG(line) AS avg_line
        FROM odds
        WHERE {' AND '.join(where_clauses)}
        """,
        (match_id, market_type),
    )
    row = cur.fetchone()
    if row is None or row.get("avg_line") is None:
        return None
    return float(row["avg_line"])


def best_odds_for_market(cur, match_id: str, market_type: str, target_line: float | None) -> list[dict]:
    """Best (highest) pre-match decimal odds per selection at (or very
    near) the target_line. For moneyline, target_line is None → any
    line passes. For spread/total, ±0.5 tolerance lets a book that
    differs by a half point still match."""
    if target_line is None:
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
    else:
        cur.execute(
            """
            SELECT DISTINCT ON (selection)
                   selection, line, bookmaker, odds_decimal
            FROM odds
            WHERE match_id = %s AND market_type = %s AND NOT is_live
              AND line IS NOT NULL
              AND ABS(line - %s) <= 0.5
            ORDER BY selection, odds_decimal DESC
            """,
            (match_id, market_type, target_line),
        )
    return list(cur.fetchall())


def load_nfl_predictions(cur, match_id: str) -> dict[str, dict]:
    """Latest NFL prediction row per ensemble for one match. Returns
    {ensemble_name: {prediction_id, probabilities}}."""
    cur.execute(
        """
        SELECT DISTINCT ON (model_name)
               model_name, id::text AS prediction_id, probabilities
        FROM predictions
        WHERE match_id = %s AND model_name IN %s
        ORDER BY model_name, updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        """,
        (match_id, tuple(NFL_MARKETS.keys())),
    )
    out: dict[str, dict] = {}
    for r in cur.fetchall():
        out[r["model_name"]] = {
            "prediction_id": r["prediction_id"],
            "probabilities": r["probabilities"] or {},
        }
    return out


# ── Recommendation orchestration ────────────────────────────────────


def list_upcoming_nfl(cur, days: int) -> list[str]:
    cur.execute(
        """
        SELECT m.id::text AS match_id
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'nfl'
          AND m.status = 'scheduled'
          AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
        ORDER BY m.match_date ASC
        """,
        (str(days),),
    )
    return [r["match_id"] for r in cur.fetchall()]


def delete_pending(cur, match_id: str) -> None:
    """Idempotent: drop still-actionable NFL picks before re-inserting.
    Never touches picks the user has placed or that have settled."""
    cur.execute(
        """
        DELETE FROM betting_recommendations
        WHERE match_id = %s
          AND status = 'pending'
          AND bet_type IN ('moneyline', 'spread', 'total')
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


def _selection_with_line(market: str, selection: str, line: float | None) -> str:
    """Display string for the bet. moneyline → just 'home' / 'away'.
    spread / total → carry the line so the recommendation is
    unambiguous when surfaced."""
    if market == "moneyline" or line is None:
        return selection
    line_str = f"{line:+g}" if market == "spread" else f"{line:g}"
    return f"{selection}_{line_str}"


def recommend_for_match(
    cur,
    match_id: str,
    bankroll: float,
    ev_threshold: float,
    prob_floor: float,
) -> int:
    preds = load_nfl_predictions(cur, match_id)
    if not preds:
        return 0

    delete_pending(cur, match_id)
    inserted = 0

    for ensemble_name, (market, market_label) in NFL_MARKETS.items():
        pred = preds.get(ensemble_name)
        if pred is None:
            continue
        labels = NFL_LABELS[market]
        probs = pred["probabilities"]
        target_line = None
        if market in ("spread", "total"):
            target_line = closing_line_for_match(cur, match_id, market)
            if target_line is None:
                # No closing line on file — model was trained
                # conditional on the line, so skip the market.
                continue

        for offer in best_odds_for_market(cur, match_id, market, target_line):
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

            offered_line = offer.get("line")
            display_sel = _selection_with_line(market, selection, offered_line)

            cap_note = f" (capped from raw {raw_prob:.0%})" if prob < raw_prob else ""
            reasoning = (
                f"{market_label} {display_sel}: model {prob:.0%}{cap_note}, "
                f"book {1/odds_decimal:.0%} (@ {odds_decimal:.2f}) → "
                f"EV {ev:+.1%}, quarter-Kelly stake ${stake:.2f}."
            )
            rec = {
                "prediction_id": pred["prediction_id"],
                "match_id": match_id,
                "bet_type": market,
                "selection": display_sel,
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
            logger.info("NFL bankroll for sizing: $%.2f", bankroll)
            matches = list_upcoming_nfl(cur, days)
            if not matches:
                logger.info("No upcoming NFL matches in the next %d days", days)
                return counts
            logger.info("Generating NFL recommendations for %d matches", len(matches))
            for match_id in matches:
                inserted = recommend_for_match(cur, match_id, bankroll, ev_threshold, prob_floor)
                counts["matches_processed"] += 1
                counts["recommendations"] += inserted
            conn.commit()
    logger.info(
        "Wrote %d NFL value-bet recommendations across %d matches",
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
