"""Generate tennis value-bet recommendations from stored predictions + live odds.

Tennis is the first 1v1 sport in the system — single market in v1
(moneyline). No spread, no total in the recommendation engine (the
fetch path captures totals odds but the model isn't trained on them
yet — that's a v2 follow-up).

Same EV / Kelly math as the team-sport engines:
  - Quarter Kelly stake sizing
  - 0.80 probability cap before EV/Kelly to defend against
    miscalibrated tails (tennis ML models often emit 0.90+ on
    heavy favorites, which would size huge stakes)
  - 3% minimum EV threshold
  - 0.40 minimum raw model probability

Idempotent: drops pending tennis recs for the match before re-inserting.

Telegram digest: this script does NOT queue alerts itself. The
precompute_predictions_tennis script already queues high-confidence
picks; recommendation rows surface via the API + frontend.

Usage (inside the api container):
    python /app/scripts/generate_recommendations_tennis.py
    python /app/scripts/generate_recommendations_tennis.py --days 14 --ev-threshold 0.05
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
from generate_recommendations import (  # noqa: E402
    FRESH_FIRST_ORDER_SQL,
    MAX_ODDS_AGE_HOURS,
    ODDS_AGE_SELECT_SQL,
    STALE_ODDS_REASON,
    confidence_rating,
    expected_value,
    get_bankroll,
    is_stale_odds,
    kelly_fraction,
    odds_age_hours,
    with_odds_age,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("generate_recommendations_tennis")

KELLY_FRACTION = 0.25

# Same defensive cap as NBA/NFL — tennis moneyline models trained
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
# compute_features_tennis.
TENNIS_MARKETS: dict[str, tuple[str, str]] = {
    "ensemble_tennis_ml": ("moneyline", "Match Winner"),
}

# Label index in probabilities dict (must match TaskSpec.labels in
# services/api/src/services/prediction_service.py). Tennis 1v1 maps
# index 0 → "home" (player1), index 1 → "away" (player2).
TENNIS_LABELS: dict[str, list[str]] = {
    "moneyline": ["home", "away"],
}

# ── Gating (audit 2026-09) ─────────────────────────────────────────
#
# Which streams may emit recommendations lives in scripts/rec_gating.py so a
# sport is disabled in exactly one place. Predictions keep being produced and
# graded either way — that is how we measure — only rec emission is gated.
SPORT = "tennis"
# Tennis has a single market in v1, so the primary bet_type is the only
# bet_type this generator can emit.
PRIMARY_BET_TYPE = "moneyline"
EMITTED_BET_TYPES: tuple[str, ...] = ("moneyline",)


# ── Per-market price + prob alignment ──────────────────────────────


def best_odds_for_market(cur, match_id: str, market_type: str, max_age_hours: float = MAX_ODDS_AGE_HOURS) -> list[dict]:
    """Best (highest) pre-match decimal odds per selection. Tennis
    moneyline has no line, so this is a simpler version of the
    NBA/NFL helpers — no target_line filter.

    Each row also carries the quote's age in hours, and the fresh-first
    tie-break runs BEFORE odds_decimal DESC so a book that stopped quoting
    can no longer win the selection with a price it no longer offers (audit
    2026-09)."""
    cur.execute(
        f"""
        SELECT DISTINCT ON (selection)
               selection, line, bookmaker, odds_decimal,
               {ODDS_AGE_SELECT_SQL}
        FROM odds
        WHERE match_id = %s AND market_type = %s AND NOT is_live
        ORDER BY selection, {FRESH_FIRST_ORDER_SQL}, odds_decimal DESC
        """,
        (match_id, market_type, max_age_hours),
    )
    return list(cur.fetchall())


def load_tennis_predictions(cur, match_id: str) -> dict[str, dict]:
    """Latest tennis prediction row per ensemble for one match.
    Returns {ensemble_name: {prediction_id, probabilities}}."""
    cur.execute(
        """
        SELECT DISTINCT ON (model_name)
               model_name, id::text AS prediction_id, probabilities
        FROM predictions
        WHERE match_id = %s AND model_name IN %s
        ORDER BY model_name, updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        """,
        (match_id, tuple(TENNIS_MARKETS.keys())),
    )
    out: dict[str, dict] = {}
    for r in cur.fetchall():
        out[r["model_name"]] = {
            "prediction_id": r["prediction_id"],
            "probabilities": r["probabilities"] or {},
        }
    return out


# ── Recommendation orchestration ────────────────────────────────────


def list_upcoming_tennis(cur, days: int, min_player_history: int = 10) -> list[str]:
    """Scheduled tennis matches in the window whose BOTH players have at
    least `min_player_history` finished matches in the corpus.

    Eligibility gate mirrored from soccer (audit doc §1.1.4): players
    absent from the training corpus (qualifiers, challengers stepping up)
    get prior-shaped probabilities that the EV gate converts into
    fictitious +EV recs — the mechanism behind this sport's stale
    pending-rec pileup. No history, no rec."""
    cur.execute(
        """
        WITH player_history AS (
            SELECT t.id AS player_id, COUNT(h.id) AS finished
            FROM teams t
            LEFT JOIN matches h
              ON (h.home_team_id = t.id OR h.away_team_id = t.id)
             AND h.status = 'finished'
            GROUP BY t.id
        )
        SELECT m.id::text AS match_id,
               (hh.finished >= %(n)s AND ah.finished >= %(n)s) AS eligible
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        JOIN player_history hh ON hh.player_id = m.home_team_id
        JOIN player_history ah ON ah.player_id = m.away_team_id
        WHERE l.sport = 'tennis'
          AND m.status = 'scheduled'
          AND m.match_date BETWEEN NOW() AND NOW() + (%(days)s || ' days')::interval
        ORDER BY m.match_date ASC
        """,
        {"days": str(days), "n": min_player_history},
    )
    rows = cur.fetchall()
    eligible = [r["match_id"] for r in rows if r["eligible"]]
    skipped = len(rows) - len(eligible)
    if skipped:
        # Loud, not silent: gated volume must be visible in the DAG log.
        logger.info(
            "Eligibility gate: skipped %d/%d upcoming tennis matches (a player has < %d finished matches in-corpus)",
            skipped,
            len(rows),
            min_player_history,
        )
    return eligible


def delete_pending(cur, match_id: str) -> None:
    """Idempotent: drop still-actionable tennis picks before re-inserting.
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
    suppressed: dict[str, int] | None = None,
    max_odds_age_hours: float = MAX_ODDS_AGE_HOURS,
) -> int:
    """Emit gated value bets for one match. `suppressed` accumulates the
    per-reason counts of candidates the gate rejected — and the ones the
    odds-freshness guard rejected, under STALE_ODDS_REASON — so run() can
    log the summary; silence is the failure mode we are guarding against."""
    if suppressed is None:
        suppressed = {}
    preds = load_tennis_predictions(cur, match_id)
    if not preds:
        return 0

    delete_pending(cur, match_id)
    inserted = 0

    for ensemble_name, (market, market_label) in TENNIS_MARKETS.items():
        pred = preds.get(ensemble_name)
        if pred is None:
            continue
        # Stream-level gate. run() short-circuits when EVERY emitted bet_type
        # is off; this handles a partial disable so one market can be turned
        # off without touching the rest of the generator.
        gate = rec_gating.gate_for(SPORT, market)
        if not gate.enabled:
            suppressed["stream_disabled"] = suppressed.get("stream_disabled", 0) + 1
            continue
        labels = TENNIS_LABELS[market]
        probs = pred["probabilities"]

        for offer in best_odds_for_market(cur, match_id, market, max_odds_age_hours):
            selection = offer["selection"]
            if selection not in labels:
                continue
            # FRESHNESS GUARD (audit 2026-09). best_odds_for_market already
            # preferred the warmest quote for this selection, so reaching the
            # bound here means every book's price for it has gone cold. A
            # price we cannot claim was available cannot price a rec, and the
            # refusal is counted rather than silently dropped.
            age_hours = odds_age_hours(offer)
            if is_stale_odds(age_hours, max_odds_age_hours):
                suppressed[STALE_ODDS_REASON] = suppressed.get(STALE_ODDS_REASON, 0) + 1
                continue
            raw_prob = float(probs.get(selection, 0.0))
            if raw_prob < prob_floor:
                continue
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
                "rec_stake": stake,
                "reasoning": reasoning,
                # odds_at_recommendation is `odds_decimal`, the price the
                # freshness guard accepted; its age rides along in
                # risk_factors so CLV can be audited after the fact.
                "risk": json.dumps(with_odds_age(_risk_factors(prob, odds_decimal), age_hours)),
            }
            insert_recommendation(cur, rec)
            inserted += 1

    return inserted


def run(database_url: str, days: int, ev_threshold: float, prob_floor: float, min_player_history: int = 10) -> dict:
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
            logger.info("Tennis bankroll for sizing: $%.2f", bankroll)
            matches = list_upcoming_tennis(cur, days, min_player_history)
            if not matches:
                logger.info("No upcoming tennis matches in the next %d days", days)
                return counts
            logger.info("Generating tennis recommendations for %d matches", len(matches))
            for match_id in matches:
                inserted = recommend_for_match(cur, match_id, bankroll, ev_threshold, prob_floor, suppressed)
                counts["matches_processed"] += 1
                counts["recommendations"] += inserted
            conn.commit()
    logger.info(
        "Wrote %d tennis value-bet recommendations across %d matches",
        counts["recommendations"],
        counts["matches_processed"],
    )
    if suppressed:
        logger.info(
            "Gating suppressed %d candidate tennis recs: %s",
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
    p.add_argument(
        "--min-player-history",
        type=int,
        default=10,
        help="Both players need this many finished matches in-corpus to be rec-eligible (default 10; 0 disables).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    counts = run(args.database_url, args.days, args.ev_threshold, args.prob_floor, args.min_player_history)
    logger.info("Done. %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
