"""Generate horse racing value-bet recommendations.

Horse racing is structurally different from the team / 1v1 sports
in the rec engine:
  * A race has N entrants, not a 2-way market. Each runner gets a
    devigged win probability (from race_predictions) and the
    recommendation candidate is per-(race, entrant).
  * Bookmaker odds DO NOT live in the `odds` table — that's keyed
    on match_id (sport=soccer/NFL/NBA/...). For horse racing they
    live in race_entrants.metadata.bookmaker_odds, captured by
    load_racing_api.upsert_entrant from the racecard's `odds[]`
    array.
  * The market-consensus baseline IS the devigged morning line.
    Comparing it against the SAME morning-line decimal would give
    zero EV by construction — so value bets only exist where a
    specific bookmaker offers LONGER odds than the consensus
    (best-of-N pricing). Without the bookmaker_odds capture, no
    recommendations can ever fire.

Same EV / Kelly math as the team-sport engines:
  - Quarter Kelly stake sizing
  - 5% minimum EV threshold (default; tunable)
  - 0.10 minimum raw model probability (default; tunable). Horse
    racing fields are wide so even 10% win prob is a credible
    candidate (5-1 odds-on favorites land here).
  - No prob-cap: devigged probs in fields of 8+ rarely exceed 60%
    even for prohibitive favorites, so the soccer/tennis cap isn't
    load-bearing here.

Writes to the race_recommendations table (not betting_recommendations
— horse racing has its own schema for multi-runner shapes; see
migration 013 for the rationale).

Idempotent: drops pending win-market recs for the race before
re-inserting.

Usage (inside the api container):
    python /app/scripts/generate_recommendations_horse_racing.py
    python /app/scripts/generate_recommendations_horse_racing.py \
        --days 2 --ev-threshold 0.05 --prob-floor 0.10
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

# Reuse the EV/Kelly + bankroll helpers from the soccer engine. Same
# math; horse racing doesn't need a different formulation.
sys.path.insert(0, os.path.dirname(__file__))
from generate_recommendations import confidence_rating, expected_value, get_bankroll, kelly_fraction  # noqa: E402
from telegram_notify import Alert, enqueue_alerts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("generate_recommendations_horse_racing")

KELLY_FRACTION = 0.25

# Model precedence (prob source): consensus-only for EV math. The
# LambdaMART ranker has BETTER top-1 accuracy (+2.0pts on the 13k-
# race corpus) but WORSE test Brier (0.1050 vs 0.0831), and the
# recs engine consumes probabilities directly (EV = prob × odds).
# A worse-calibrated model inflates the value-bet count by 17x —
# verified empirically (memory: horse-racing-ml-ranker-v1).
#
# Listing the ranker even as a FALLBACK is wrong: every race the
# ranker scores but consensus hasn't yet (e.g., races added between
# the two precompute tasks' runs) silently picks up the ranker's
# probabilities and the ~30-picks/day budget blows out again. Keep
# this list consensus-only for PROBABILITIES.
MODEL_PRECEDENCE: list[tuple[str, str]] = [
    ("market_consensus_v1", "1.0.0"),
]

# HYBRID mode: while consensus owns the PROBABILITY side of the EV
# math, the ranker's RANK is genuinely better (+2.0pts top-1 on the
# 13k-race held-out test). The hybrid filter uses the ranker to
# narrow each race down to its top-RANKER_TOP_N entrants, then
# evaluates EV on those using consensus probs. This combines what
# each model is good at:
#   - ranker: WHICH horses are likely to win (rank order)
#   - consensus: HOW LIKELY each is (calibrated probability)
#
# When RANKER_TOP_N is None, the filter is disabled and the recs
# engine considers every entrant — the prior consensus-only
# behaviour. When the ranker hasn't scored a race (e.g., because
# its precompute task ran AFTER the race-card list), the filter
# falls back to "consider all entrants" for that race so consensus-
# only races still get full coverage.
RANKER_MODEL_NAME = "lightgbm_ranker_v1"
RANKER_MODEL_VERSION = "1.0.0"
RANKER_TOP_N: Optional[int] = 3


# ── Best-of-N pricing across bookmakers ────────────────────────────


def best_decimal(bookmaker_odds: list[dict]) -> Optional[dict]:
    """Pick the bookmaker offering the longest decimal odds for this
    horse — that's the bettor's most favourable price, and the only
    one that can carry positive EV vs the devigged consensus across
    the same array.

    Returns {bookmaker, decimal} for the winner, or None if the input
    array is empty / malformed.
    """
    if not bookmaker_odds:
        return None
    best = None
    for entry in bookmaker_odds:
        if not isinstance(entry, dict):
            continue
        decimal = entry.get("decimal")
        if decimal is None:
            continue
        try:
            d = float(decimal)
        except (TypeError, ValueError):
            continue
        if d <= 1.0:
            continue
        if best is None or d > best["decimal"]:
            best = {"bookmaker": entry.get("bookmaker"), "decimal": d}
    return best


# ── DB I/O ─────────────────────────────────────────────────────────


def list_upcoming_races(cur, days: int) -> list[dict]:
    """Scheduled races in the next N days that have at least one
    market_consensus_v1 prediction. Returns the race_id alongside
    the track + race_date the Telegram alert needs to identify the
    race — skips races without predictions so we don't waste cycles."""
    eligible_model_names = [m for m, _ in MODEL_PRECEDENCE]
    cur.execute(
        """
        SELECT DISTINCT
            r.id::text   AS race_id,
            r.track_name,
            r.race_date,
            r.race_number
        FROM races r
        JOIN race_predictions rp ON rp.race_id = r.id
        WHERE r.status = 'scheduled'
          AND r.race_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
          AND rp.model_name = ANY(%s)
        ORDER BY r.race_date ASC, r.id::text
        """,
        (str(days), eligible_model_names),
    )
    return list(cur.fetchall())


def load_race_candidates(cur, race_id: str) -> list[dict]:
    """Per-entrant pricing + prediction context for one race. Returns
    list of {entrant_id, prediction_id, confidence, bookmaker_odds,
    horse_name, model_name, ranker_confidence}.

    The DISTINCT ON + ORDER BY (entrant_id, precedence) picks the
    highest-precedence available model per entrant for the
    consensus-driven probability. Separately we join in the LATEST
    ranker prediction per entrant (when one exists) so the hybrid
    filter can rank candidates by the LambdaMART score while still
    using consensus_prob for EV — see the RANKER_TOP_N block in
    recommend_for_race for how the two are combined."""
    eligible: list[str] = [m for m, _ in MODEL_PRECEDENCE]
    # Build the CASE WHEN ladder dynamically so this works with any
    # number of models. The ELSE branch puts unknown models at the
    # bottom, but the ANY(%s) filter excludes them anyway — defensive.
    when_clauses = " ".join(f"WHEN {ghr_arg(i)} THEN {i + 1}" for i in range(len(eligible)))
    case_sql = f"CASE rp.model_name {when_clauses} ELSE {len(eligible) + 1} END"
    cur.execute(
        f"""
        SELECT DISTINCT ON (entrant_id)
            entrant_id,
            prediction_id,
            confidence,
            bookmaker_odds,
            horse_name,
            model_name,
            ranker_confidence
        FROM (
            SELECT
                e.id::text          AS entrant_id,
                rp.id::text         AS prediction_id,
                rp.confidence       AS confidence,
                e.metadata->'bookmaker_odds' AS bookmaker_odds,
                h.name              AS horse_name,
                rp.model_name       AS model_name,
                rrk.confidence      AS ranker_confidence,
                {case_sql}          AS precedence
            FROM race_entrants e
            JOIN race_predictions rp ON rp.entrant_id = e.id
            JOIN horses h ON h.id = e.horse_id
            LEFT JOIN race_predictions rrk
                ON rrk.entrant_id = e.id
               AND rrk.model_name = %s
               AND rrk.model_version = %s
               AND rrk.prediction_type = 'win'
            WHERE e.race_id = %s
              AND NOT e.scratched
              AND rp.model_name = ANY(%s)
              AND rp.prediction_type = 'win'
        ) ranked
        ORDER BY entrant_id, precedence ASC
        """,
        (*eligible, RANKER_MODEL_NAME, RANKER_MODEL_VERSION, race_id, eligible),
    )
    return list(cur.fetchall())


def ghr_arg(_i: int) -> str:
    """Return a literal positional placeholder for the dynamic CASE
    WHEN clause. Kept as a tiny helper so the call site reads as a
    single f-string template — substituting %s directly inside the
    f-string would be ambiguous with the outer execute() params."""
    return "%s"


def delete_pending(cur, race_id: str) -> None:
    """Drop still-actionable win-market recs for this race before
    re-inserting. Never touches recs the user placed or that settled."""
    cur.execute(
        """
        DELETE FROM race_recommendations
        WHERE race_id = %s
          AND status = 'pending'
          AND bet_type = 'win'
        """,
        (race_id,),
    )


def insert_recommendation(cur, rec: dict) -> None:
    cur.execute(
        """
        INSERT INTO race_recommendations
          (race_prediction_id, race_id, entrant_id, bet_type, selection,
           odds_at_recommendation, bookmaker, confidence_rating,
           expected_value, kelly_stake, recommended_stake, reasoning,
           risk_factors)
        VALUES
          (%(prediction_id)s, %(race_id)s, %(entrant_id)s, 'win',
           %(selection)s, %(odds)s, %(bookmaker)s, %(conf)s, %(ev)s,
           %(kelly_stake)s, %(rec_stake)s, %(reasoning)s, %(risk)s::jsonb)
        """,
        rec,
    )


def _risk_factors(prob: float, odds_decimal: float, field_size: int) -> list[str]:
    risks: list[str] = []
    if odds_decimal >= 10.0:
        # Decimal 10+ = 9/1 or longer. Longshots are statistically
        # noisier — variance dominates EV math at this end.
        risks.append("longshot")
    if prob < 0.15:
        # Below 15% even after devig — the consensus considers this
        # horse unlikely. The recommendation is essentially a bet
        # AGAINST the consensus's confidence ordering.
        risks.append("low_consensus_probability")
    if field_size >= 14:
        # Large fields amplify variance and break-up traffic patterns.
        # Even a sharp pick can lose to bad luck in a 16-runner sprint.
        risks.append("large_field")
    return risks


# ── Alert factory ───────────────────────────────────────────────────


def horse_racing_alert(
    *,
    track_name: str,
    race_date,
    race_number: Optional[int],
    horse_name: str,
    odds_decimal: float,
    bookmaker: Optional[str],
    confidence: float,
    expected_value: float,
    recommended_stake: float,
) -> Alert:
    """Translate a horse racing value bet into the shared Alert shape.
    The Alert dataclass was designed for 2-team / 1v1 sports, so the
    fit isn't 1:1:
      * `home_team` carries the horse name (the actual selection).
      * `away_team` is "Field" — the consensus implied prob is
        derived ACROSS the field, so the contrast is horse-vs-field
        not horse-vs-horse.
      * `league_name` carries the track name (Newton Abbot, Curragh,
        ...) plus race number when known.
      * Setting `expected_value` flips the digest formatter into
        value-bet mode (odds + EV + stake instead of probability
        breakdown). See telegram_notify._format_alert_line."""
    race_label = f"{track_name}"
    if race_number is not None:
        race_label += f" R{race_number}"
    return Alert(
        sport="horse_racing",
        league_name=race_label,
        home_team=horse_name,
        away_team="Field",
        match_date=race_date,
        market_label="Win",
        predicted_outcome=horse_name,
        confidence=float(confidence),
        probabilities={"win": float(confidence)},
        odds_decimal=float(odds_decimal),
        expected_value=float(expected_value),
        recommended_stake=float(recommended_stake),
        bookmaker=bookmaker,
    )


# ── Recommendation orchestration ────────────────────────────────────


def recommend_for_race(
    cur,
    race: dict,
    bankroll: float,
    ev_threshold: float,
    prob_floor: float,
) -> list[Alert]:
    """Generate (and DB-insert) value-bet recs for one race; return
    the matching Alert objects ready for Redis. Returns [] when the
    race has no qualifying picks.

    Hybrid filter (RANKER_TOP_N): when the LambdaMART ranker has
    scored this race, narrow the candidate set to the top-N entrants
    by ranker confidence before evaluating EV. Consensus probability
    still drives the math; the ranker just picks which horses to
    consider. For races without ranker scores, every entrant is
    evaluated as before."""
    race_id = race["race_id"]
    candidates = load_race_candidates(cur, race_id)
    if not candidates:
        return []
    delete_pending(cur, race_id)
    field_size = len(candidates)

    # Hybrid filter: if RANKER_TOP_N is set AND every candidate has a
    # ranker_confidence, narrow to the top-N by ranker rank. The
    # "every candidate has a ranker_confidence" guard handles the
    # transient state where the ranker precompute hasn't run yet for
    # this race — fall back to "consider all entrants" so consensus-
    # only races still get full coverage.
    if RANKER_TOP_N is not None and RANKER_TOP_N > 0:
        if all(c.get("ranker_confidence") is not None for c in candidates):
            candidates = sorted(
                candidates,
                key=lambda c: float(c["ranker_confidence"]),
                reverse=True,
            )[:RANKER_TOP_N]

    alerts: list[Alert] = []

    for cand in candidates:
        prob = float(cand["confidence"])
        if prob < prob_floor:
            continue
        bookmaker_odds = cand["bookmaker_odds"]
        if isinstance(bookmaker_odds, str):
            # psycopg2 returns JSONB as str unless cursor configured
            # otherwise — handle both shapes defensively.
            try:
                bookmaker_odds = json.loads(bookmaker_odds)
            except (TypeError, ValueError):
                bookmaker_odds = None
        best = best_decimal(bookmaker_odds or [])
        if not best:
            continue
        odds_decimal = best["decimal"]
        ev = expected_value(prob, odds_decimal)
        if ev < ev_threshold:
            continue
        k = kelly_fraction(prob, odds_decimal)
        stake = round(bankroll * k * KELLY_FRACTION, 2)

        # Display label: "consensus" when this came through the
        # consensus-only path; "hybrid" when the ranker also scored
        # the race and filtered this pick into the top-N. Useful for
        # post-hoc analysis (does ranker filtering improve win rate?).
        had_ranker_score = cand.get("ranker_confidence") is not None
        label = "hybrid" if (had_ranker_score and RANKER_TOP_N) else "consensus"
        reasoning = (
            f"Win: {cand['horse_name']} — {label} {prob:.0%}, "
            f"book {1/odds_decimal:.0%} (@ {odds_decimal:.2f} on "
            f"{best['bookmaker']}) → EV {ev:+.1%}, quarter-Kelly "
            f"stake ${stake:.2f}."
        )
        rec = {
            "prediction_id": cand["prediction_id"],
            "race_id": race_id,
            "entrant_id": cand["entrant_id"],
            "selection": cand["horse_name"],
            "odds": odds_decimal,
            "bookmaker": best["bookmaker"],
            "conf": confidence_rating(ev, prob),
            "ev": ev,
            "kelly_stake": k,
            "rec_stake": stake,
            "reasoning": reasoning,
            "risk": json.dumps(_risk_factors(prob, odds_decimal, field_size)),
        }
        insert_recommendation(cur, rec)
        alerts.append(
            horse_racing_alert(
                track_name=race["track_name"],
                race_date=race["race_date"],
                race_number=race.get("race_number"),
                horse_name=cand["horse_name"],
                odds_decimal=odds_decimal,
                bookmaker=best["bookmaker"],
                confidence=prob,
                expected_value=ev,
                recommended_stake=stake,
            )
        )

    return alerts


def run(
    database_url: str,
    days: int,
    ev_threshold: float,
    prob_floor: float,
    notify: bool,
) -> dict:
    counts = {"races_processed": 0, "recommendations": 0, "alerts_queued": 0, "queue_depth": 0}
    all_alerts: list[Alert] = []
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            bankroll = get_bankroll(cur)
            logger.info("Horse racing bankroll for sizing: $%.2f", bankroll)
            races = list_upcoming_races(cur, days)
            if not races:
                logger.info("No upcoming races with predictions in next %d days", days)
                return counts
            logger.info("Generating horse racing recommendations for %d races", len(races))
            for race in races:
                alerts = recommend_for_race(cur, race, bankroll, ev_threshold, prob_floor)
                counts["races_processed"] += 1
                counts["recommendations"] += len(alerts)
                all_alerts.extend(alerts)
            conn.commit()

    queue_depth = enqueue_alerts(all_alerts) if (notify and all_alerts) else 0
    counts["alerts_queued"] = len(all_alerts) if notify else 0
    counts["queue_depth"] = queue_depth
    logger.info(
        "Wrote %d horse racing value-bet recommendations across %d races; " "queued %d alerts (queue depth now %d)",
        counts["recommendations"],
        counts["races_processed"],
        counts["alerts_queued"],
        queue_depth,
    )
    return counts


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=2, help="Lookahead window in days (default 2).")
    p.add_argument(
        "--ev-threshold",
        type=float,
        default=0.05,
        help="Minimum positive EV (default 0.05 = 5%%).",
    )
    p.add_argument(
        "--prob-floor",
        type=float,
        default=0.10,
        help="Minimum consensus probability (default 0.10).",
    )
    p.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip the Telegram-queue enqueue (DB writes still happen).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    run(args.database_url, args.days, args.ev_threshold, args.prob_floor, not args.no_notify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
