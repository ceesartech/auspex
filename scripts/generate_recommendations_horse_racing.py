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

# Model precedence: the consensus baseline comes FIRST for EV math
# even though the ranker has better top-1 accuracy. The reason is
# strict — the recs engine consumes probabilities (EV = prob × odds)
# and the ranker is BETTER at RANKING but WORSE at CALIBRATION on the
# current corpus:
#
#   variant                            top-1 acc   Brier
#   market_consensus_v1                19.9%       0.0831
#   lightgbm_ranker_v1 (13k corpus)    22.7%       0.0898
#
# A model with worse Brier produces probabilities that don't match
# real win frequencies (verified empirically: ranker-prob-driven EV
# fired 520 picks across 672 races, a 17x false-positive rate vs the
# consensus-only run which sat around 30/day). Keeping the consensus
# at the top of the precedence list means the recs engine uses the
# more honest probabilities for EV; the ranker stays in the predictions
# table for analysis + future-work iteration on its own probability
# calibration (memory: horse-racing-ml-ranker-v1 has the full
# breakdown + the open paths to closing the gap).
MODEL_PRECEDENCE: list[tuple[str, str]] = [
    ("market_consensus_v1", "1.0.0"),
    ("lightgbm_ranker_v1", "1.0.0"),
]


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
    horse_name, model_name}.

    The DISTINCT ON + ORDER BY (entrant_id, precedence) picks the
    BEST available model per entrant — the ranker when present, the
    consensus baseline otherwise. Surfaces model_name so the
    recommendation reasoning can show which model fired."""
    eligible: list[str] = [m for m, _ in MODEL_PRECEDENCE]
    # CASE WHEN ladder ranks rows by model precedence so DISTINCT ON
    # picks the first match per entrant. Index in the array (1-based
    # for the CASE statement) maps to precedence rank.
    cur.execute(
        """
        SELECT DISTINCT ON (entrant_id)
            entrant_id,
            prediction_id,
            confidence,
            bookmaker_odds,
            horse_name,
            model_name
        FROM (
            SELECT
                e.id::text          AS entrant_id,
                rp.id::text         AS prediction_id,
                rp.confidence       AS confidence,
                e.metadata->'bookmaker_odds' AS bookmaker_odds,
                h.name              AS horse_name,
                rp.model_name       AS model_name,
                CASE rp.model_name
                    WHEN %s THEN 1
                    WHEN %s THEN 2
                    ELSE          3
                END                 AS precedence
            FROM race_entrants e
            JOIN race_predictions rp ON rp.entrant_id = e.id
            JOIN horses h ON h.id = e.horse_id
            WHERE e.race_id = %s
              AND NOT e.scratched
              AND rp.model_name = ANY(%s)
              AND rp.prediction_type = 'win'
        ) ranked
        ORDER BY entrant_id, precedence ASC
        """,
        (eligible[0], eligible[1] if len(eligible) > 1 else "", race_id, eligible),
    )
    return list(cur.fetchall())


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
    race has no qualifying picks."""
    race_id = race["race_id"]
    candidates = load_race_candidates(cur, race_id)
    if not candidates:
        return []
    delete_pending(cur, race_id)
    field_size = len(candidates)
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

        # Display label per model so the user sees WHICH model fired.
        # The ranker is sharper than consensus on the held-out test
        # (+2.2pts top-1) so showing the source matters for trust +
        # post-hoc analysis.
        model_label = "ranker" if cand.get("model_name") == "lightgbm_ranker_v1" else "consensus"
        reasoning = (
            f"Win: {cand['horse_name']} — {model_label} {prob:.0%}, "
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
