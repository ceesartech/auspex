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
from generate_recommendations import (  # noqa: E402
    confidence_rating,
    expected_value,
    get_bankroll,
    kelly_fraction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("generate_recommendations_horse_racing")

KELLY_FRACTION = 0.25
MODEL_NAME = "market_consensus_v1"
MODEL_VERSION = "1.0.0"


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


def list_upcoming_races(cur, days: int) -> list[str]:
    """Scheduled races in the next N days that have at least one
    market_consensus_v1 prediction. Skip races without predictions
    so we don't waste cycles."""
    cur.execute(
        """
        SELECT DISTINCT r.id::text AS race_id
        FROM races r
        JOIN race_predictions rp ON rp.race_id = r.id
        WHERE r.status = 'scheduled'
          AND r.race_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
          AND rp.model_name = %s
        ORDER BY r.id::text
        """,
        (str(days), MODEL_NAME),
    )
    return [r["race_id"] for r in cur.fetchall()]


def load_race_candidates(cur, race_id: str) -> list[dict]:
    """Per-entrant pricing + prediction context for one race. Returns
    list of {entrant_id, prediction_id, confidence, bookmaker_odds,
    horse_name}."""
    cur.execute(
        """
        SELECT
            e.id::text AS entrant_id,
            rp.id::text AS prediction_id,
            rp.confidence,
            e.metadata->'bookmaker_odds' AS bookmaker_odds,
            h.name AS horse_name
        FROM race_entrants e
        JOIN race_predictions rp ON rp.entrant_id = e.id
        JOIN horses h ON h.id = e.horse_id
        WHERE e.race_id = %s
          AND NOT e.scratched
          AND rp.model_name = %s
          AND rp.model_version = %s
          AND rp.prediction_type = 'win'
        """,
        (race_id, MODEL_NAME, MODEL_VERSION),
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


# ── Recommendation orchestration ────────────────────────────────────


def recommend_for_race(
    cur,
    race_id: str,
    bankroll: float,
    ev_threshold: float,
    prob_floor: float,
) -> int:
    candidates = load_race_candidates(cur, race_id)
    if not candidates:
        return 0
    delete_pending(cur, race_id)
    field_size = len(candidates)
    inserted = 0

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
        stake = bankroll * k * KELLY_FRACTION

        reasoning = (
            f"Win: {cand['horse_name']} — consensus {prob:.0%}, "
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
            "rec_stake": round(stake, 2),
            "reasoning": reasoning,
            "risk": json.dumps(_risk_factors(prob, odds_decimal, field_size)),
        }
        insert_recommendation(cur, rec)
        inserted += 1

    return inserted


def run(database_url: str, days: int, ev_threshold: float, prob_floor: float) -> dict:
    counts = {"races_processed": 0, "recommendations": 0}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            bankroll = get_bankroll(cur)
            logger.info("Horse racing bankroll for sizing: $%.2f", bankroll)
            races = list_upcoming_races(cur, days)
            if not races:
                logger.info("No upcoming races with predictions in next %d days", days)
                return counts
            logger.info("Generating horse racing recommendations for %d races", len(races))
            for race_id in races:
                inserted = recommend_for_race(cur, race_id, bankroll, ev_threshold, prob_floor)
                counts["races_processed"] += 1
                counts["recommendations"] += inserted
            conn.commit()
    logger.info(
        "Wrote %d horse racing value-bet recommendations across %d races",
        counts["recommendations"],
        counts["races_processed"],
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
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    run(args.database_url, args.days, args.ev_threshold, args.prob_floor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
