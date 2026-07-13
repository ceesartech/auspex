"""Generate value-bet recommendations from stored predictions + live odds.

For every upcoming match we compare the model's per-market probabilities
(written to the `predictions` table by precompute_predictions.py) against the
best available bookmaker price (the `odds` table). Where the model thinks an
outcome is more likely than the price implies — a positive expected value — we
write a row to `betting_recommendations` with a Kelly-sized stake.

This fills a gap: nothing in production wrote `betting_recommendations` before
(the API only ever *read* it). Markets we don't ingest odds for (e.g. correct
score, team totals, winning margin) simply produce no recommendations — they
remain prediction-only until a price feed exists.

Pricing:
  EV per unit staked = model_prob * decimal_odds - 1
  Kelly fraction f*  = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
  recommended stake  = bankroll * max(0, f*) * 0.25   (quarter Kelly)
Asian-handicap integer lines can push (stake refunded); EV uses the raw
win/push masses (EV = win*O + push - 1) while Kelly sizes on the no-push
conditional.

Idempotent: re-running deletes a match's still-`pending` recommendations and
re-inserts the current set (so stale picks whose edge evaporated are pruned);
`placed`/`won`/`lost` rows are never touched.

Usage (inside the api container):
    python /app/scripts/generate_recommendations.py
    python /app/scripts/generate_recommendations.py --days 14 --ev-threshold 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("generate_recommendations")

# Bookmaker market_type -> the predictions.prediction_type that carries the
# model probabilities for it. The 1x2 headline is special: its prediction
# lives under 'match_result', but odds + the API filter use '1x2'.
ODDS_TO_PREDICTION: dict[str, str] = {
    "1x2": "match_result",
    "over_under": "over_under",
    "btts": "btts",
    "double_chance": "double_chance",
    "draw_no_bet": "draw_no_bet",
    "asian_handicap": "asian_handicap",
    "correct_score": "correct_score",
    # Halftime markets (PR #11 predictions, migration 014 +
    # migration 016 odds). HT odds + HT predictions share the same
    # name so the existing match-by-prediction-type path needs no
    # special handling.
    "match_result_ht": "match_result_ht",
    "over_under_ht": "over_under_ht",
    "btts_ht": "btts_ht",
}

# Markets whose selection embeds a betting line (for the displayed selection).
LINED_MARKETS = {"over_under", "asian_handicap", "team_total", "over_under_ht"}


# ── Pure helpers (unit-tested in test_recommendation_math.py) ─────────────


def _fmt_line(line: float) -> str:
    """Format a line for a selection key: 2.5 -> "2.5", -1.0 -> "-1", 0.0 ->
    "0". MUST match market_derivation._fmt_line so model keys reconstruct."""
    f = float(line)
    return str(int(f)) if f.is_integer() else str(f)


def expected_value(prob: float, odds_decimal: float) -> float:
    """EV per unit staked for a simple (no-push) back bet."""
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
    CHECK domain: low / medium / high / very_high."""
    if ev >= 0.15 and prob >= 0.55:
        return "very_high"
    if ev >= 0.10:
        return "high"
    if ev >= 0.05:
        return "medium"
    return "low"


def model_key_for_odds(market_type: str, selection: str, line) -> str | None:
    """Reconstruct the model `probabilities` key for an odds (selection, line),
    or None if it can't be matched. For asian_handicap the key is the
    home-perspective '<line>_<side>' (the away point is sign-flipped); the
    caller also reads '<line>_push'."""
    sel = (selection or "").strip().lower()
    if market_type == "1x2":
        return sel if sel in ("home", "draw", "away") else None
    if market_type == "btts":
        return sel if sel in ("yes", "no") else None
    if market_type == "double_chance":
        s = (selection or "").strip().upper()
        return s if s in ("1X", "12", "X2") else None
    if market_type == "draw_no_bet":
        return sel if sel in ("home", "away") else None
    if market_type == "over_under":
        if sel not in ("over", "under") or line is None:
            return None
        return f"{sel}_{_fmt_line(line)}"
    if market_type == "asian_handicap":
        if sel not in ("home", "away") or line is None:
            return None
        # the-odds-api gives each side its own signed point; the model indexes
        # by home-perspective line. Home keeps its point, away flips sign.
        home_line = float(line) if sel == "home" else -float(line)
        return f"{_fmt_line(home_line)}_{sel}"
    if market_type == "correct_score":
        return selection.strip() if selection else None
    # Halftime markets — selection-key conventions mirror the FT
    # equivalents so derive_soccer_halftime_markets and odds-fetch
    # land the same keys without extra plumbing.
    if market_type == "match_result_ht":
        return sel if sel in ("home", "draw", "away") else None
    if market_type == "btts_ht":
        return sel if sel in ("yes", "no") else None
    if market_type == "over_under_ht":
        if sel not in ("over", "under") or line is None:
            return None
        return f"{sel}_{_fmt_line(line)}"
    return None


def selection_value(market_type: str, model_probs: dict, selection: str, line, odds_decimal: float):
    """Return (prob_for_kelly, expected_value, raw_prob) for backing this
    selection, or None if the model doesn't price it. Handles asian-handicap
    pushes (stake refunded): EV uses raw win+push, Kelly uses the no-push
    conditional win/(1-push)."""
    key = model_key_for_odds(market_type, selection, line)
    if key is None:
        return None
    if market_type == "asian_handicap":
        win = model_probs.get(key)
        if win is None:
            return None
        line_prefix = key.rsplit("_", 1)[0]
        push = model_probs.get(f"{line_prefix}_push", 0.0) or 0.0
        ev = win * odds_decimal + push - 1.0
        denom = 1.0 - push
        p_kelly = (win / denom) if denom > 0 else 0.0
        return p_kelly, ev, float(win)
    p = model_probs.get(key)
    if p is None:
        return None
    return p, expected_value(p, odds_decimal), float(p)


def display_selection(market_type: str, selection: str, line) -> str:
    """Selection string stored on the recommendation (line embedded for lined
    markets, e.g. 'over_2.5', 'home_-0.5')."""
    if line is not None and market_type in LINED_MARKETS:
        return f"{selection}_{_fmt_line(line)}"
    return selection


# ── DB access ─────────────────────────────────────────────────────────────


def list_upcoming(cur, days: int, min_team_history: int = 10) -> list[dict]:
    """Scheduled matches in the window whose BOTH teams have at least
    `min_team_history` finished matches in the corpus.

    Eligibility gate (audit doc §1.1.4): the Poisson/Dixon-Coles ensemble
    members emit their global prior for teams absent from the training
    corpus, so out-of-corpus fixtures (World Cup, newly-added summer
    leagues) produce base-rate probabilities that the EV gate converts
    into fictitious +EV long-shot recs. No history, no rec — until the
    corpus backfill (audit doc §3.6) gives those leagues real coverage."""
    cur.execute(
        """
        WITH team_history AS (
            SELECT t.id AS team_id, COUNT(h.id) AS finished
            FROM teams t
            LEFT JOIN matches h
              ON (h.home_team_id = t.id OR h.away_team_id = t.id)
             AND h.status = 'finished'
            GROUP BY t.id
        )
        SELECT m.id::text AS match_id,
               (hh.finished >= %(n)s AND ah.finished >= %(n)s) AS eligible
        FROM matches m
        JOIN team_history hh ON hh.team_id = m.home_team_id
        JOIN team_history ah ON ah.team_id = m.away_team_id
        WHERE m.status = 'scheduled'
          AND m.match_date BETWEEN NOW() AND NOW() + (%(days)s || ' days')::interval
        ORDER BY m.match_date ASC
        """,
        {"days": str(days), "n": min_team_history},
    )
    rows = cur.fetchall()
    eligible = [r["match_id"] for r in rows if r["eligible"]]
    skipped = len(rows) - len(eligible)
    if skipped:
        # Loud, not silent: gated volume must be visible in the DAG log.
        logger.info(
            "Eligibility gate: skipped %d/%d upcoming matches (a team has " "< %d finished matches in-corpus)",
            skipped,
            len(rows),
            min_team_history,
        )
    return eligible


def load_market_predictions(cur, match_id: str) -> dict[str, dict]:
    """Latest ensemble prediction per prediction_type for a match. Returns
    {prediction_type: {"prediction_id": str, "probabilities": dict}}."""
    cur.execute(
        """
        SELECT DISTINCT ON (prediction_type)
               prediction_type, id::text AS prediction_id, probabilities
        FROM predictions
        WHERE match_id = %s AND model_name = 'ensemble'
        ORDER BY prediction_type, updated_at DESC NULLS LAST, created_at DESC NULLS LAST
        """,
        (match_id,),
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
    line), with the bookmaker offering it."""
    cur.execute(
        """
        SELECT DISTINCT ON (market_type, selection, COALESCE(line, -1))
               market_type, selection, line, bookmaker, odds_decimal
        FROM odds
        WHERE match_id = %s AND is_live = false
        ORDER BY market_type, selection, COALESCE(line, -1), odds_decimal DESC
        """,
        (match_id,),
    )
    return list(cur.fetchall())


def get_bankroll(cur) -> float:
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
    """Drop still-actionable picks before re-inserting; never touch picks the
    user has already placed or that have settled."""
    cur.execute(
        "DELETE FROM betting_recommendations WHERE match_id = %s AND status = 'pending'",
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
    if odds_decimal >= 6.0:
        risks.append("longshot")
    if prob < 0.15:
        risks.append("low_model_probability")
    return risks


def recommend_for_match(
    cur,
    match_id: str,
    bankroll: float,
    ev_threshold: float,
    prob_floor: float,
) -> int:
    """Generate + persist value bets for one match. Returns rows inserted."""
    preds = load_market_predictions(cur, match_id)
    if not preds:
        return 0
    odds_rows = best_odds(cur, match_id)
    if not odds_rows:
        return 0

    delete_pending(cur, match_id)

    written = 0
    for row in odds_rows:
        market_type = row["market_type"]
        ptype = ODDS_TO_PREDICTION.get(market_type)
        if ptype is None or ptype not in preds:
            continue
        probs = preds[ptype]["probabilities"]
        odds_decimal = float(row["odds_decimal"])
        if odds_decimal <= 1.0:
            continue
        line = float(row["line"]) if row["line"] is not None else None

        valued = selection_value(market_type, probs, row["selection"], line, odds_decimal)
        if valued is None:
            continue
        p_kelly, ev, p_raw = valued
        if ev < ev_threshold or p_kelly < prob_floor:
            continue

        f = kelly_fraction(p_kelly, odds_decimal)
        if f <= 0:
            continue
        kelly_stake = round(bankroll * f, 2)
        rec_stake = round(bankroll * f * 0.25, 2)  # quarter Kelly
        sel = display_selection(market_type, row["selection"], line)
        reasoning = (
            f"Model {p_raw:.1%} vs implied {1.0 / odds_decimal:.1%} @ {odds_decimal:.2f} "
            f"({row['bookmaker']}); EV {ev:+.1%}, quarter-Kelly stake {rec_stake:.2f}."
        )
        insert_recommendation(
            cur,
            {
                "prediction_id": preds[ptype]["prediction_id"],
                "match_id": match_id,
                "bet_type": market_type,
                "selection": sel,
                "odds": odds_decimal,
                "bookmaker": row["bookmaker"],
                "conf": confidence_rating(ev, p_kelly),
                "ev": round(ev, 4),
                "kelly_stake": kelly_stake,
                "rec_stake": rec_stake,
                "reasoning": reasoning,
                "risk": json.dumps(_risk_factors(p_raw, odds_decimal)),
            },
        )
        written += 1
    return written


def run(
    database_url: str,
    days: int,
    ev_threshold: float,
    prob_floor: float,
    min_team_history: int = 10,
) -> dict:
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            bankroll = get_bankroll(cur)
            match_ids = list_upcoming(cur, days, min_team_history)
            logger.info("Evaluating %d upcoming match(es) (bankroll=%.2f)", len(match_ids), bankroll)
            total = 0
            matches_with_recs = 0
            for mid in match_ids:
                try:
                    n = recommend_for_match(cur, mid, bankroll, ev_threshold, prob_floor)
                except Exception as e:
                    logger.warning("Recommendation generation failed for %s: %s", mid, e)
                    conn.rollback()
                    continue
                if n:
                    matches_with_recs += 1
                    total += n
                conn.commit()
    logger.info("Wrote %d recommendation(s) across %d match(es)", total, matches_with_recs)
    return {"recommendations": total, "matches": matches_with_recs}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=7, help="Look-ahead window (default 7).")
    p.add_argument(
        "--ev-threshold",
        type=float,
        default=0.05,
        help="Minimum expected value to recommend a bet (default 0.05 = 5%%).",
    )
    p.add_argument(
        "--prob-floor",
        type=float,
        default=0.10,
        help="Minimum model probability for a pick — filters thin longshots (default 0.10).",
    )
    p.add_argument(
        "--min-team-history",
        type=int,
        default=10,
        help="Skip matches where either team has fewer than this many "
        "finished matches in-corpus (eligibility gate; default 10, 0 disables).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    run(args.database_url, args.days, args.ev_threshold, args.prob_floor, args.min_team_history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
