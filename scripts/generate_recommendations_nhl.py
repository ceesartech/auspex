"""Generate NHL value-bet recommendations from stored predictions + live odds.

Parallel to scripts/generate_recommendations.py (soccer). For every upcoming
NHL match we compare the model's per-market probabilities (written to the
`predictions` table by precompute_predictions_nhl.py) against the best
available bookmaker price (the `odds` table) and emit a row to
`betting_recommendations` for each positive-EV pick, plus a Telegram alert
into the shared digest queue.

Three markets are evaluated (regulation is skipped — sportsbooks don't
typically price the 3-class regulation-winner market, so no odds to
compare against):

  - moneyline:  probabilities {home, away} ↔ odds (moneyline, home|away)
  - puck_line:  probabilities {cover, no_cover} ↔ odds (spread, home -1.5
                / away +1.5)  — "cover" = home covers -1.5; "no_cover" =
                away covers +1.5 (equivalently home doesn't cover -1.5)
  - total:      probabilities {over, under} ↔ odds (total, over|under,
                line=5.5)  — non-5.5 lines are skipped

Pricing matches the soccer engine:

  EV per unit  = model_prob × decimal_odds − 1
  Kelly f*     = (model_prob × decimal_odds − 1) / (decimal_odds − 1)
  stake        = bankroll × max(0, f*) × 0.25   (quarter Kelly)

Idempotent per match: deletes that match's still-`pending` recommendations
before re-inserting the current set. Placed / settled rows are never
touched.

Usage (inside the api container):
    python /app/scripts/generate_recommendations_nhl.py
    python /app/scripts/generate_recommendations_nhl.py --days 14 --ev-threshold 0.05
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

# Add the scripts dir so the shared telegram_notify helper imports.
sys.path.insert(0, os.path.dirname(__file__))

import rec_gating  # noqa: E402
from telegram_notify import Alert, enqueue_alerts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("generate_recommendations_nhl")


# NHL-specific market constants. NHL_TOTAL_LINE is the canonical book
# total — other lines are ignored. The puck-line is always ±1.5.
NHL_TOTAL_LINE = 5.5
NHL_PUCK_LINE_ABS = 1.5

# Predictions emit these prediction_type values for NHL (see
# services/api/src/services/prediction_service.py TASKS registry):
#   moneyline:  "moneyline"
#   puck_line:  "spread"
#   total:      "total"
# Regulation ("match_result") is intentionally absent.
NHL_PREDICTION_TYPES = ("moneyline", "spread", "total")


# Bookmaker odds market_type → friendly market_label for Telegram + bet_type
# for the DB column. Keep keys aligned with NHL_PREDICTION_TYPES.
MARKET_LABEL_FOR_PREDICTION_TYPE = {
    "moneyline": "Moneyline",
    "spread": "Puck Line",
    "total": "Total Goals O/U 5.5",
}


# ── Gating (audit 2026-09) ───────────────────────────────────────────────
#
# Which streams may emit recommendations lives in scripts/rec_gating.py so a
# sport is disabled in exactly one place. Predictions keep being produced and
# graded either way — that is how we measure — only rec emission is gated.
# The check below is per-bet_type (the DB bet_type, so 'puck_line' not the
# odds market_type 'spread'), so a future partial re-enable needs no change
# here.
SPORT = "nhl"
PRIMARY_BET_TYPE = "moneyline"
EMITTED_BET_TYPES: tuple[str, ...] = ("moneyline", "puck_line", "total")


# ── Pure helpers (unit-tested in test_recommendations_nhl.py) ─────────────


def expected_value(prob: float, odds_decimal: float) -> float:
    """EV per unit staked for a back bet (no push, no commission)."""
    return prob * odds_decimal - 1.0


def kelly_fraction(prob: float, odds_decimal: float) -> float:
    """Kelly stake fraction, clamped to >= 0 (no bet on non-positive edge)."""
    b = odds_decimal - 1.0
    if b <= 0:
        return 0.0
    f = (prob * odds_decimal - 1.0) / b
    return f if f > 0 else 0.0


def confidence_rating(ev: float, prob: float) -> str:
    """Bucket (EV, prob) into the betting_recommendations.confidence_rating
    CHECK domain: low / medium / high / very_high. Same thresholds as the
    soccer engine so consumers don't need a sport switch."""
    if ev >= 0.15 and prob >= 0.55:
        return "very_high"
    if ev >= 0.10:
        return "high"
    if ev >= 0.05:
        return "medium"
    return "low"


def model_prob_for_odds(prediction_type: str, probs: dict, selection: str, line: Optional[float]) -> Optional[float]:
    """Map an (odds market_type, selection, line) row back to the matching
    entry in the predictions.probabilities JSONB. Returns None when the
    odds row doesn't correspond to a market the model priced (e.g. a
    non-5.5 total)."""
    sel = (selection or "").strip().lower()
    if prediction_type == "moneyline":
        return float(probs[sel]) if sel in ("home", "away") and sel in probs else None
    if prediction_type == "spread":
        # Home -1.5 maps to "cover"; away +1.5 maps to "no_cover". Any
        # other puck-line variant (rare in the API) is skipped.
        if line is None:
            return None
        if sel == "home" and abs(line + NHL_PUCK_LINE_ABS) < 1e-6:
            return float(probs["cover"]) if "cover" in probs else None
        if sel == "away" and abs(line - NHL_PUCK_LINE_ABS) < 1e-6:
            return float(probs["no_cover"]) if "no_cover" in probs else None
        return None
    if prediction_type == "total":
        # Only model the canonical 5.5 line; alternative totals are
        # ignored to keep recs focused on what the ensemble was trained on.
        if line is None or abs(line - NHL_TOTAL_LINE) > 1e-6:
            return None
        return float(probs[sel]) if sel in ("over", "under") and sel in probs else None
    return None


def display_selection(prediction_type: str, selection: str, line: Optional[float]) -> str:
    """User-facing selection label embedded in betting_recommendations.selection
    and the Telegram alert. Encodes the line for lined markets so the row
    is self-describing without joining back to odds."""
    sel = (selection or "").strip().lower()
    if prediction_type == "moneyline":
        return sel
    if prediction_type == "spread" and line is not None:
        return f"{sel} {line:+g}"
    if prediction_type == "total" and line is not None:
        return f"{sel} {line:g}"
    return sel


def risk_factors(prob: float, odds_decimal: float) -> list[str]:
    """Same heuristics as the soccer engine — kept identical so downstream
    consumers (frontend badges, monitoring) don't need a sport switch."""
    risks: list[str] = []
    if odds_decimal >= 6.0:
        risks.append("longshot")
    if prob < 0.15:
        risks.append("low_model_probability")
    return risks


# ── DB I/O ────────────────────────────────────────────────────────────────


def list_upcoming_nhl(database_url: str, days: int) -> list[dict]:
    """Scheduled NHL matches in the next N days that have at least one
    NHL ensemble prediction already written. Joins on predictions so we
    don't waste a round-trip on matches that haven't been scored yet."""
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT DISTINCT
                    m.id::text AS match_id,
                    m.match_date,
                    ht.name AS home_team,
                    at.name AS away_team,
                    l.name AS league_name
                FROM matches m
                JOIN leagues l ON l.id = m.league_id
                JOIN teams ht  ON ht.id = m.home_team_id
                JOIN teams at  ON at.id = m.away_team_id
                JOIN predictions p ON p.match_id = m.id
                WHERE m.status = 'scheduled'
                  AND l.sport = 'nhl'
                  AND m.match_date BETWEEN NOW() AND NOW() + (%s || ' days')::interval
                  AND p.model_name LIKE 'ensemble_nhl_%%'
                ORDER BY m.match_date ASC
                """,
                (str(days),),
            )
            return [dict(r) for r in cur.fetchall()]


def load_nhl_predictions(cur, match_id: str) -> dict[str, dict]:
    """Latest NHL prediction per prediction_type for a match. Returns
    {prediction_type: {"prediction_id": str, "probabilities": dict}}
    keyed by the values in NHL_PREDICTION_TYPES (moneyline/spread/total).

    Mirror of soccer's load_market_predictions but filtered to NHL
    ensembles via model_name LIKE 'ensemble_nhl_%'."""
    cur.execute(
        """
        SELECT DISTINCT ON (prediction_type)
               prediction_type, id::text AS prediction_id, probabilities
        FROM predictions
        WHERE match_id = %s
          AND model_name LIKE 'ensemble_nhl_%%'
          AND prediction_type = ANY(%s)
        ORDER BY prediction_type, updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        """,
        (match_id, list(NHL_PREDICTION_TYPES)),
    )
    out: dict[str, dict] = {}
    for r in cur.fetchall():
        out[r["prediction_type"]] = {
            "prediction_id": r["prediction_id"],
            "probabilities": r["probabilities"] or {},
        }
    return out


def best_odds(cur, match_id: str) -> list[dict]:
    """Best (highest) pre-match decimal price per (market_type, selection,
    line). Same shape as soccer — only the consumed market_type values
    differ."""
    cur.execute(
        """
        SELECT DISTINCT ON (market_type, selection, COALESCE(line, -1))
               market_type, selection, line, bookmaker, odds_decimal
        FROM odds
        WHERE match_id = %s AND is_live = false
          AND market_type IN ('moneyline', 'spread', 'total')
        ORDER BY market_type, selection, COALESCE(line, -1), odds_decimal DESC
        """,
        (match_id,),
    )
    return list(cur.fetchall())


def get_bankroll(cur) -> float:
    """Same source as the soccer engine — user_preferences.bankroll."""
    cur.execute("SELECT preference_value FROM user_preferences WHERE preference_key = 'bankroll' LIMIT 1")
    row = cur.fetchone()
    if row and row.get("preference_value") is not None:
        val = row["preference_value"]
        if isinstance(val, dict):
            return float(val.get("value", 1000.0))
        try:
            return float(val)
        except (TypeError, ValueError):
            return 1000.0
    return 1000.0


def delete_pending(cur, match_id: str) -> None:
    """Soccer engine and this one share the betting_recommendations table
    but write different bet_type values. Scoping the delete to NHL
    bet_types means a match that somehow had both NHL and soccer rows
    (e.g. a cross-pollination test) doesn't lose its soccer picks.
    bet_type IN (...) is the cleanest scope filter."""
    cur.execute(
        """
        DELETE FROM betting_recommendations
        WHERE match_id = %s
          AND status = 'pending'
          AND bet_type IN ('moneyline', 'puck_line', 'total')
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


# ── Per-match orchestration ───────────────────────────────────────────────


def recommend_for_match(
    cur,
    match: dict,
    bankroll: float,
    ev_threshold: float,
    prob_floor: float,
    suppressed: dict[str, int] | None = None,
) -> list[Alert]:
    """Generate + persist value bets for one NHL match. Returns the list
    of Telegram alerts to enqueue (one per recommendation).

    `suppressed` accumulates the per-reason counts of candidates the gate
    rejected so run() can log the summary — silence is the failure mode we
    are guarding against."""
    if suppressed is None:
        suppressed = {}
    preds = load_nhl_predictions(cur, match["match_id"])
    if not preds:
        return []
    odds_rows = best_odds(cur, match["match_id"])
    if not odds_rows:
        return []

    delete_pending(cur, match["match_id"])

    alerts: list[Alert] = []
    for row in odds_rows:
        market_type = row["market_type"]  # 'moneyline' / 'spread' / 'total'
        if market_type not in preds:
            continue
        # DB bet_type != odds market_type for the puck line.
        bet_type = "puck_line" if market_type == "spread" else market_type
        # Per-bet_type gate. run() short-circuits when EVERY emitted bet_type
        # is off; this handles a partial disable so one market can be turned
        # off without touching the rest of the generator.
        gate = rec_gating.gate_for(SPORT, bet_type)
        if not gate.enabled:
            suppressed["stream_disabled"] = suppressed.get("stream_disabled", 0) + 1
            continue
        probs = preds[market_type]["probabilities"]
        odds_decimal = float(row["odds_decimal"])
        if odds_decimal <= 1.0:
            continue
        line = float(row["line"]) if row["line"] is not None else None

        prob = model_prob_for_odds(market_type, probs, row["selection"], line)
        if prob is None:
            continue

        ev = expected_value(prob, odds_decimal)
        if ev < ev_threshold or prob < prob_floor:
            continue

        f = kelly_fraction(prob, odds_decimal)
        if f <= 0:
            continue
        # Odds / EV / gap caps. `prob` here IS the raw model probability —
        # unlike the NBA/NFL/1v1 engines this one applies no PROB_CAP_FOR_EV,
        # so there is nothing to un-cap before bounding model-vs-market
        # disagreement.
        market_prob = None
        if gate.max_gap is not None:
            market_prob = rec_gating.market_consensus_prob(cur, match["match_id"], market_type, row["selection"], line)
        ok, reason = rec_gating.passes_gate(
            SPORT, bet_type, odds=odds_decimal, ev=ev, model_prob=prob, market_prob=market_prob
        )
        if not ok:
            key = reason or "gated"
            suppressed[key] = suppressed.get(key, 0) + 1
            continue
        kelly_stake = round(bankroll * f, 2)
        rec_stake = rec_gating.cap_stake(bankroll * f * 0.25, bankroll)  # quarter Kelly, per-bet capped
        sel_display = display_selection(market_type, row["selection"], line)
        market_label = MARKET_LABEL_FOR_PREDICTION_TYPE.get(market_type, market_type)
        reasoning = (
            f"NHL {market_label}: model {prob:.1%} vs implied {1.0 / odds_decimal:.1%} "
            f"@ {odds_decimal:.2f} ({row['bookmaker']}); EV {ev:+.1%}, "
            f"quarter-Kelly stake {rec_stake:.2f}."
        )
        insert_recommendation(
            cur,
            {
                "prediction_id": preds[market_type]["prediction_id"],
                "match_id": match["match_id"],
                "bet_type": bet_type,
                "selection": sel_display,
                "odds": odds_decimal,
                "bookmaker": row["bookmaker"],
                "conf": confidence_rating(ev, prob),
                "ev": round(ev, 4),
                "kelly_stake": kelly_stake,
                "rec_stake": rec_stake,
                "reasoning": reasoning,
                "risk": json.dumps(risk_factors(prob, odds_decimal)),
            },
        )
        alerts.append(
            Alert(
                sport="nhl",
                league_name=match["league_name"],
                home_team=match["home_team"],
                away_team=match["away_team"],
                match_date=match["match_date"],
                market_label=market_label,
                predicted_outcome=sel_display,
                confidence=prob,
                probabilities={k: float(v) for k, v in probs.items()},
                odds_decimal=odds_decimal,
                expected_value=round(ev, 4),
                recommended_stake=rec_stake,
                bookmaker=row["bookmaker"],
            )
        )
    return alerts


def run(database_url: str, days: int, ev_threshold: float, prob_floor: float, notify: bool) -> dict:
    """Walk upcoming NHL matches and write recommendations + queue
    Telegram alerts. Returns count summary suitable for DAG return."""
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
        return {
            "matches_evaluated": 0,
            "recommendations": 0,
            "alerts_queued": 0,
            "pruned_pending": pruned,
        }

    upcoming = list_upcoming_nhl(database_url, days)
    if not upcoming:
        logger.info("No upcoming NHL matches with predictions in the next %d days", days)
        return {"matches_evaluated": 0, "recommendations": 0, "alerts_queued": 0}
    logger.info("Evaluating %d upcoming NHL matches", len(upcoming))

    recs_written = 0
    all_alerts: list[Alert] = []
    suppressed: dict[str, int] = {}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            bankroll = get_bankroll(cur)
            logger.info("Bankroll: %.2f", bankroll)
            for m in upcoming:
                alerts = recommend_for_match(cur, m, bankroll, ev_threshold, prob_floor, suppressed)
                if alerts:
                    recs_written += len(alerts)
                    all_alerts.extend(alerts)
            conn.commit()

    queue_depth = enqueue_alerts(all_alerts) if (notify and all_alerts) else 0
    logger.info(
        "Wrote %d NHL recommendations across %d matches; queued %d alerts (queue depth now %d)",
        recs_written,
        len(upcoming),
        len(all_alerts) if notify else 0,
        queue_depth,
    )
    if suppressed:
        logger.info(
            "Gating suppressed %d candidate nhl recs: %s",
            sum(suppressed.values()),
            ", ".join(f"{k}={v}" for k, v in sorted(suppressed.items())),
        )
    return {
        "matches_evaluated": len(upcoming),
        "recommendations": recs_written,
        "alerts_queued": len(all_alerts) if notify else 0,
        "queue_depth": queue_depth,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=7, help="How many days forward to evaluate (default 7).")
    p.add_argument(
        "--ev-threshold",
        type=float,
        default=0.05,
        help="Minimum positive expected value (5%% default).",
    )
    p.add_argument(
        "--prob-floor",
        type=float,
        default=0.10,
        help="Minimum model probability to consider a pick (10%% default).",
    )
    p.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip Telegram dispatch — recs still get written to the DB.",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    counts = run(
        args.database_url,
        args.days,
        args.ev_threshold,
        args.prob_floor,
        notify=not args.no_notify,
    )
    logger.info("Done. %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
