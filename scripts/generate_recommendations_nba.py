"""Generate NBA value-bet recommendations from stored predictions + live odds.

NBA-specific value-bet engine — parallel to scripts/generate_recommendations.py
(soccer) but adapted for the line-as-feature design.

Key difference from soccer/NHL:
  Soccer derives ~15 markets from the Dixon-Coles scoreline; the predictions
  table holds one row per (market, line) keyed by 'over_under_2.5'-style
  selections. The rec engine reads the model prob directly from
  probabilities[selection_key].

  NBA spread + total are LINE-AS-FEATURE. One trained ensemble per market
  emits P(home covers <closing_line>) where the closing line was baked into
  the input features. The model never independently knows about lines other
  than the closing one. So we can only recommend bets at lines NEAR the
  closing line — at the snapshot's line the probability is correct; at a
  meaningfully different line it's not.

  v1 keeps this simple: recommend ONLY at the same line we trained against
  (the closing line in features_cache). Other-line bets are skipped. A v2
  could re-predict at the offered line if we keep the model line-conditional.

For each upcoming NBA match we:
  1. Read the latest prediction rows for moneyline / spread / total.
  2. Look up the BEST available pre-match decimal price at the closing line.
  3. Compute EV = model_prob × decimal_odds − 1 per offered selection.
  4. Quarter-Kelly stake on positive EV picks that clear ev_threshold +
     prob_floor.
  5. Idempotent insert into betting_recommendations (pending rows are
     dropped + re-inserted so stale picks evaporate).

Telegram digest: this script does NOT queue alerts itself. The
precompute_predictions_nba script already queues high-confidence picks;
recommendation rows surface in the API + frontend. (A future commit could
add a value-bet-flavored Alert variant.)

Usage (inside the api container):
    python /app/scripts/generate_recommendations_nba.py
    python /app/scripts/generate_recommendations_nba.py --days 14 --ev-threshold 0.05
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

import rec_gating  # noqa: E402
from generate_recommendations import confidence_rating, expected_value, get_bankroll, kelly_fraction  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("generate_recommendations_nba")

# Quarter Kelly — same fraction as the soccer engine. Conservative
# enough that a 5% miscalibrated edge doesn't blow up the bankroll.
KELLY_FRACTION = 0.25

# Cap the model probability before EV / Kelly math.
#
# Why: the NBA spread + total ensembles hit val accuracy of ~60% with
# MCE (worst-bucket calibration error) ~0.20. That means the model's
# most-confident predictions can be ~20 points overconfident vs the
# true rate. A raw 0.87 emit might really be 0.67-0.87 in expectation.
#
# Without a cap, miscalibration on the tail translates directly into
# huge implied EV (0.87 × 1.95 - 1 = +0.70) and proportionally huge
# Kelly stakes. Capping at 0.80 gives a 7-point safety margin against
# the worst observed bucket and prevents a single overconfident
# matchup from sizing the bankroll.
#
# This affects the rec sizing ONLY — predictions.confidence stays
# uncapped (the model's raw output is preserved for monitoring +
# calibration analysis). A future commit can replace the hard cap
# with a learned calibrator (isotonic / Platt) once we have settled
# bets to fit it on.
PROB_CAP_FOR_EV = 0.80


def cap_prob(p: float, cap: float = PROB_CAP_FOR_EV) -> float:
    """Defensive cap on model probability for EV / Kelly math."""
    if p > cap:
        return cap
    return p


# Ensemble names → (prediction_type, market_label). Used to look up
# the right model rows for each NBA market.
NBA_MARKETS: dict[str, tuple[str, str]] = {
    "ensemble_nba_ml": ("moneyline", "Moneyline"),
    "ensemble_nba_sp": ("spread", "Spread"),
    "ensemble_nba_tot": ("total", "Total"),
}

# Label index in probabilities dict (must match TaskSpec.labels in
# services/api/src/services/prediction_service.py).
NBA_LABELS: dict[str, list[str]] = {
    "moneyline": ["home", "away"],
    "spread": ["home", "away"],
    "total": ["over", "under"],
}

# ── Gating (audit 2026-09) ─────────────────────────────────────────
#
# Which streams may emit recommendations lives in scripts/rec_gating.py so a
# sport is disabled in exactly one place. Predictions keep being produced and
# graded either way — that is how we measure — only rec emission is gated.
# The check below is per-bet_type, so a future partial re-enable (say
# moneyline back on while spread stays off) needs no change here.
SPORT = "nba"
PRIMARY_BET_TYPE = "moneyline"
EMITTED_BET_TYPES: tuple[str, ...] = ("moneyline", "spread", "total")


# ── Per-market price + prob alignment ──────────────────────────────


def closing_line_for_match(cur, match_id: str, market_type: str) -> float | None:
    """Most-recent pre-match line offered (averaged across books) for
    the spread or total market on this match. Returns None if no odds
    landed — used to decide whether we can recommend a spread/total
    bet (we only recommend at the line the model was trained on).

    Spread quirk: odds rows for the home side carry a SIGNED line
    (-5 for a 5-point favorite), and away rows the inverse (+5).
    Averaging both perspectives gives ~0 for any matchup with a
    favorite, which then fails the ±0.5 line-match filter in
    best_odds_for_match → 0 spread recs. Pin the average to the
    home perspective so the target_line is the signed home line
    (-5 in the example). Totals don't have this issue because the
    line is the same on both sides (218.5 for over AND under).
    """
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
    line passes. For spread/total, we filter to lines within ±0.5 of
    the closing line — books occasionally differ by a half point and
    we still trust the model's edge there."""
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


def load_nba_predictions(cur, match_id: str) -> dict[str, dict]:
    """Latest NBA prediction row per ensemble for one match. Returns
    {ensemble_name: {prediction_id, probabilities}}."""
    cur.execute(
        """
        SELECT DISTINCT ON (model_name)
               model_name, id::text AS prediction_id, probabilities
        FROM predictions
        WHERE match_id = %s AND model_name IN %s
        ORDER BY model_name, updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        """,
        (match_id, tuple(NBA_MARKETS.keys())),
    )
    out: dict[str, dict] = {}
    for r in cur.fetchall():
        out[r["model_name"]] = {
            "prediction_id": r["prediction_id"],
            "probabilities": r["probabilities"] or {},
        }
    return out


# ── Recommendation orchestration ────────────────────────────────────


def list_upcoming_nba(cur, days: int) -> list[str]:
    cur.execute(
        """
        SELECT m.id::text AS match_id
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'nba'
          AND m.status = 'scheduled'
          AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
        ORDER BY m.match_date ASC
        """,
        (str(days),),
    )
    return [r["match_id"] for r in cur.fetchall()]


def delete_pending(cur, match_id: str) -> None:
    """Idempotent: drop still-actionable NBA picks before re-inserting.
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
    suppressed: dict[str, int] | None = None,
) -> int:
    """Emit gated value bets for one match. `suppressed` accumulates the
    per-reason counts of candidates the gate rejected so run() can log the
    summary — silence is the failure mode we are guarding against."""
    if suppressed is None:
        suppressed = {}
    preds = load_nba_predictions(cur, match_id)
    if not preds:
        return 0

    delete_pending(cur, match_id)
    inserted = 0

    for ensemble_name, (market, market_label) in NBA_MARKETS.items():
        pred = preds.get(ensemble_name)
        if pred is None:
            continue
        # Per-bet_type gate. run() short-circuits when EVERY emitted bet_type
        # is off; this handles a partial disable so one market can be turned
        # off without touching the rest of the generator.
        gate = rec_gating.gate_for(SPORT, market)
        if not gate.enabled:
            suppressed["stream_disabled"] = suppressed.get("stream_disabled", 0) + 1
            continue
        labels = NBA_LABELS[market]
        probs = pred["probabilities"]
        # Only spread / total need the line gate. Moneyline matches
        # any offered price.
        target_line = None
        if market in ("spread", "total"):
            target_line = closing_line_for_match(cur, match_id, market)
            if target_line is None:
                # No closing line on file — the model was trained
                # conditional on the line, so we can't recommend
                # bets without one. Skip the market.
                continue

        for offer in best_odds_for_market(cur, match_id, market, target_line):
            selection = offer["selection"]
            if selection not in labels:
                continue
            raw_prob = float(probs.get(selection, 0.0))
            if raw_prob < prob_floor:
                continue
            # Apply the cap for EV / Kelly math. Predictions table
            # keeps the uncapped value; reasoning string surfaces both
            # the raw model output and the capped working number so
            # the user understands the rec sizing.
            prob = cap_prob(raw_prob)
            odds_decimal = float(offer["odds_decimal"])
            ev = expected_value(prob, odds_decimal)
            if ev < ev_threshold:
                continue
            # Odds / EV / gap caps. model_prob is the RAW (uncapped) model
            # probability on purpose: PROB_CAP_FOR_EV is an EV/stake defence,
            # not the model's belief, so capping first would hide exactly the
            # model-vs-market disagreement the gap cap exists to bound.
            market_prob = None
            if gate.max_gap is not None:
                market_prob = rec_gating.market_consensus_prob(cur, match_id, market, selection, offer.get("line"))
            ok, reason = rec_gating.passes_gate(
                SPORT, market, odds=odds_decimal, ev=ev, model_prob=raw_prob, market_prob=market_prob
            )
            if not ok:
                key = reason or "gated"
                suppressed[key] = suppressed.get(key, 0) + 1
                continue
            k = kelly_fraction(prob, odds_decimal)
            stake = rec_gating.cap_stake(bankroll * k * KELLY_FRACTION, bankroll)

            offered_line = offer.get("line")
            display_sel = _selection_with_line(market, selection, offered_line)

            # Surface the cap explicitly when it fires so the user can
            # see why the rec is more conservative than the raw model
            # would suggest.
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
                "rec_stake": stake,
                "reasoning": reasoning,
                "risk": json.dumps(_risk_factors(prob, odds_decimal)),
            }
            insert_recommendation(cur, rec)
            inserted += 1

    return inserted


def run(database_url: str, days: int, ev_threshold: float, prob_floor: float) -> dict:
    counts = {"matches_processed": 0, "recommendations": 0}
    gates = {bt: rec_gating.gate_for(SPORT, bt) for bt in EMITTED_BET_TYPES}
    if not any(g.enabled for g in gates.values()):
        # Loud, but NOT an exception: the DAG task must still succeed so the
        # pipeline keeps producing and grading predictions. Only rec emission
        # stops.
        logger.info(
            "recommendation generation for %s is gated OFF: %s",
            SPORT,
            gates[PRIMARY_BET_TYPE].note,
        )
        # Stopping emission is not enough. The recs the LAST pre-gate run wrote
        # are still status='pending' on upcoming fixtures, and
        # vw_active_recommendations (status IN ('pending','placed') AND
        # match_date > NOW()) keeps serving them to the API and the Telegram
        # digest as live picks. The per-match delete_pending never runs on this
        # path, so withdraw them here — a disabled stream must have an empty
        # book, not a frozen one.
        pruned = rec_gating.purge_pending_recs_for_sport(database_url, SPORT, EMITTED_BET_TYPES)
        counts["pruned_pending"] = pruned
        return counts
    suppressed: dict[str, int] = {}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            bankroll = get_bankroll(cur)
            logger.info("NBA bankroll for sizing: $%.2f", bankroll)
            matches = list_upcoming_nba(cur, days)
            if not matches:
                logger.info("No upcoming NBA matches in the next %d days", days)
                return counts
            logger.info("Generating NBA recommendations for %d matches", len(matches))
            for match_id in matches:
                inserted = recommend_for_match(cur, match_id, bankroll, ev_threshold, prob_floor, suppressed)
                counts["matches_processed"] += 1
                counts["recommendations"] += inserted
            conn.commit()
    logger.info(
        "Wrote %d NBA value-bet recommendations across %d matches",
        counts["recommendations"],
        counts["matches_processed"],
    )
    if suppressed:
        logger.info(
            "Gating suppressed %d candidate nba recs: %s",
            sum(suppressed.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(suppressed.items())),
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
