"""Offline backtest of the recommendation strategy on historical data.

Walks every finished match in a date window, simulates what the live
rec engine WOULD have produced at the supplied thresholds (EV gate,
prob floor, Kelly fraction, NBA prob cap), grades each simulated
pick against the real outcome, and reports realized P/L.

Why this exists:
  The live rec engine's knobs are hardcoded (ev_threshold=0.03,
  prob_floor=0.40, KELLY_FRACTION=0.25, NBA cap=0.80). To answer
  "what if we tightened EV to 5%?" or "what if we ran full Kelly
  instead of quarter?" we need to re-run rec logic OFFLINE — without
  writing to betting_recommendations or touching production state.

What this is NOT:
  * Not a walk-forward retrain. We use predictions ALREADY in the DB
    (written by the precompute scripts at the time the games were
    upcoming). If those predictions were trained on data that included
    the match they predict, the backtest is overfit — but in steady
    state, every prediction written during a 15-min pipeline run is
    out-of-sample (the trained model only sees finished matches).
  * Not a full cross-validation framework. Single window, single
    threshold set per run.
  * Doesn't touch the betting_recommendations table.

Output:
  Markdown table by (sport, market) + overall summary to stdout.
  Optional --output writes the structured result to JSON so multiple
  runs can be diffed.

Usage:
    # Baseline run with live thresholds
    python /app/scripts/backtest_recommendations.py \\
        --start 2024-10-01 --end 2025-06-30 \\
        --ev-threshold 0.03 --prob-floor 0.40 \\
        --kelly-fraction 0.25 --nba-prob-cap 0.80 \\
        --bankroll 1000

    # Tighter EV gate to see if profitability improves
    python /app/scripts/backtest_recommendations.py \\
        --start 2024-10-01 --end 2025-06-30 \\
        --ev-threshold 0.05 --output bt_ev5.json

The Kelly + EV math is imported from scripts/generate_recommendations.py
(the live soccer rec engine) so we backtest the EXACT same formulas
— no chance of "backtest looked profitable, prod didn't" due to math
drift.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Reuse the live math + grading helpers — single source of truth.
sys.path.insert(0, os.path.dirname(__file__))

from generate_recommendations import expected_value, kelly_fraction  # noqa: E402
from grading_outcomes import actual_outcome, is_push  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backtest_recommendations")


# Maps the ensemble name written to predictions.model_name → the
# canonical market name + odds market_type filter. Mirrors NBA_MARKETS
# in generate_recommendations_nba.py and the analogous mapping in
# generate_recommendations.py for soccer.
ENSEMBLE_TO_MARKET = {
    # Soccer derives ~15 markets but they all live under model_name='ensemble'.
    # We approach the soccer side by reading prediction_type instead — see
    # _market_for_prediction below.
    "ensemble_nhl_ml": ("nhl", "moneyline", "moneyline"),
    "ensemble_nhl_reg": ("nhl", "regulation", "match_result"),  # not bookable
    "ensemble_nhl_pl": ("nhl", "puck_line", "spread"),
    "ensemble_nhl_tot": ("nhl", "total", "total"),
    "ensemble_nba_ml": ("nba", "moneyline", "moneyline"),
    "ensemble_nba_sp": ("nba", "spread", "spread"),
    "ensemble_nba_tot": ("nba", "total", "total"),
}


@dataclass
class MarketAggregate:
    """Per-(sport, market) backtest tally."""

    sport: str
    market: str
    recs: int = 0
    won: int = 0
    lost: int = 0
    void: int = 0
    total_staked: Decimal = field(default_factory=lambda: Decimal("0"))
    total_pnl: Decimal = field(default_factory=lambda: Decimal("0"))

    @property
    def hit_rate(self) -> float:
        decided = self.won + self.lost
        return (self.won / decided) if decided > 0 else 0.0

    @property
    def roi_pct(self) -> float:
        if self.total_staked == 0:
            return 0.0
        return float((self.total_pnl / self.total_staked) * Decimal("100"))


@dataclass
class BacktestResult:
    """Full run's structured output."""

    start: str
    end: str
    params: dict
    overall: MarketAggregate
    by_market: list[MarketAggregate]

    def to_dict(self) -> dict:
        def _serialize(agg: MarketAggregate) -> dict:
            return {
                "sport": agg.sport,
                "market": agg.market,
                "recs": agg.recs,
                "won": agg.won,
                "lost": agg.lost,
                "void": agg.void,
                "total_staked": float(agg.total_staked),
                "total_pnl": float(agg.total_pnl),
                "hit_rate": round(agg.hit_rate, 4),
                "roi_pct": round(agg.roi_pct, 2),
            }

        return {
            "start": self.start,
            "end": self.end,
            "params": self.params,
            "overall": _serialize(self.overall),
            "by_market": [_serialize(m) for m in self.by_market],
        }


# ── Pure helpers (unit-tested) ───────────────────────────────────────


def simulate_pnl(
    *,
    actual: Optional[str],
    selection: str,
    stake: Decimal,
    odds: Decimal,
) -> tuple[str, Decimal]:
    """Per-rec simulation: returns (outcome, profit_loss).

    Win:  selection matched actual → stake × (odds - 1)
    Loss: selection didn't match    → -stake
    Push: outcome is a push string  → 0 (status='void' in prod)
    Ungradable: actual is None      → ('skip', 0) — caller doesn't count it
    """
    if actual is None:
        return "skip", Decimal("0")
    if is_push(actual):
        return "void", Decimal("0")
    if selection == actual:
        return "won", (stake * (odds - Decimal("1"))).quantize(Decimal("0.01"))
    return "lost", (-stake).quantize(Decimal("0.01"))


def apply_prob_cap(sport: str, raw_prob: float, nba_cap: float) -> float:
    """Apply the live confidence cap. NBA-only: clip to nba_cap.
    Soccer + NHL pass through untouched (no cap in live code)."""
    if sport == "nba":
        return min(raw_prob, nba_cap)
    return raw_prob


def _market_for_prediction(model_name: str, prediction_type: str) -> Optional[tuple[str, str, str]]:
    """Resolve (sport, market, odds_market_type) for one prediction.
    Returns None for prediction types we don't backtest (regulation,
    derived soccer-only markets without a clean odds match)."""
    if model_name in ENSEMBLE_TO_MARKET:
        sport, market, odds_market = ENSEMBLE_TO_MARKET[model_name]
        # NHL regulation isn't a bookable market — skip.
        if market == "regulation":
            return None
        return sport, market, odds_market
    # Soccer ensemble — only backtest the 1x2 / match_result headline
    # in v1. Derived markets (over_under_X.Y, asian_handicap, etc.)
    # are skipped because their key-encoded selection strings
    # complicate the odds-table join; covered by accuracy widget
    # via the production grading.
    if model_name == "ensemble" and prediction_type == "match_result":
        return "soccer", "match_result", "1x2"
    return None


# ── DB queries ───────────────────────────────────────────────────────


def list_predictions_in_window(cur, start: str, end: str, prediction_version: Optional[str] = None) -> list[dict]:
    """Every prediction row whose match finished within [start, end].
    Includes match scores + metadata for the grading step + the
    sport/league for routing.

    `prediction_version`: when provided, restricts to predictions with
    `model_version = <prediction_version>`. Used by the walk-forward
    workflow to backtest only the wf_<split_date>-versioned rows
    written by scripts/walk_forward_predictions.py — without this
    filter, the backtest would mix walk-forward + production
    predictions and the numbers wouldn't mean anything.
    """
    cur.execute(
        """
        SELECT
            p.id::text AS prediction_id,
            p.match_id::text AS match_id,
            p.model_name,
            p.model_version,
            p.prediction_type,
            p.predicted_outcome,
            p.probabilities,
            p.confidence,
            l.sport AS sport,
            m.home_score,
            m.away_score,
            m.metadata,
            m.match_date
        FROM predictions p
        JOIN matches m ON m.id = p.match_id
        JOIN leagues l ON l.id = m.league_id
        WHERE m.status = 'finished'
          AND m.home_score IS NOT NULL
          AND m.away_score IS NOT NULL
          AND m.match_date BETWEEN %s::date AND %s::date
          AND (%s IS NULL OR p.model_version = %s)
        ORDER BY m.match_date ASC, p.id ASC
        """,
        (start, end, prediction_version, prediction_version),
    )
    return [dict(r) for r in cur.fetchall()]


def fetch_best_odds(cur, match_id: str, odds_market_type: str, target_line: Optional[float] = None) -> dict[str, dict]:
    """Best (highest) pre-match decimal price per selection, optionally
    line-filtered for spread/total. Same filter logic as
    best_odds_for_market in the rec engines."""
    if target_line is None:
        cur.execute(
            """
            SELECT DISTINCT ON (selection)
                   selection, line, bookmaker, odds_decimal
            FROM odds
            WHERE match_id = %s AND market_type = %s AND NOT is_live
            ORDER BY selection, odds_decimal DESC
            """,
            (match_id, odds_market_type),
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
            (match_id, odds_market_type, target_line),
        )
    return {r["selection"]: dict(r) for r in cur.fetchall()}


def fetch_features(cur, match_id: str) -> Optional[dict]:
    """Latest features_cache row for the match. NBA spread/total need
    the closing line for both the odds join and the grading dispatch."""
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


# ── Orchestration ────────────────────────────────────────────────────


def run(
    *,
    database_url: str,
    start: str,
    end: str,
    ev_threshold: float,
    prob_floor: float,
    kelly_fraction_arg: float,
    nba_prob_cap: float,
    bankroll: float,
    prediction_version: Optional[str] = None,
) -> BacktestResult:
    bankroll_d = Decimal(str(bankroll))
    aggregates: dict[tuple[str, str], MarketAggregate] = {}

    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            preds = list_predictions_in_window(cur, start, end, prediction_version=prediction_version)
            logger.info("Loaded %d predictions in [%s .. %s]", len(preds), start, end)

            for pred in preds:
                routing = _market_for_prediction(pred["model_name"], pred["prediction_type"])
                if routing is None:
                    continue
                sport, market, odds_market = routing
                key = (sport, market)
                agg = aggregates.setdefault(key, MarketAggregate(sport=sport, market=market))

                # Features pulled once per match for NBA spread/total
                # (used by both the line filter and the grading
                # dispatch). Cheap if reused across same-match rows
                # but for v1 we just refetch per-prediction — the
                # query is indexed and fast.
                features = fetch_features(cur, pred["match_id"]) if sport == "nba" else None

                # NBA spread + total: line comes from features_cache
                # so the odds filter targets the same line the model
                # was conditional on.
                target_line: Optional[float] = None
                if sport == "nba" and market in ("spread", "total"):
                    key_in_features = "closing_spread_home" if market == "spread" else "closing_total_line"
                    line_val = (features or {}).get(key_in_features)
                    if line_val is None:
                        continue
                    target_line = float(line_val)

                offers = fetch_best_odds(cur, pred["match_id"], odds_market, target_line)
                probs: dict = pred["probabilities"] or {}

                # For each offered selection, evaluate the EV against
                # the model's prob and recommend if it clears the gate.
                for selection, offer in offers.items():
                    raw_prob = float(probs.get(selection, 0.0))
                    if raw_prob < prob_floor:
                        continue
                    prob = apply_prob_cap(sport, raw_prob, nba_prob_cap)
                    odds = float(offer["odds_decimal"])
                    ev = expected_value(prob, odds)
                    if ev < ev_threshold:
                        continue
                    k = kelly_fraction(prob, odds)
                    stake = (bankroll_d * Decimal(str(k)) * Decimal(str(kelly_fraction_arg))).quantize(Decimal("0.01"))
                    if stake <= 0:
                        continue

                    actual = actual_outcome(
                        prediction_type=pred["prediction_type"],
                        model_name=pred["model_name"],
                        predicted_outcome=pred["predicted_outcome"],
                        home_score=pred["home_score"],
                        away_score=pred["away_score"],
                        metadata=pred["metadata"],
                        features=features,
                    )
                    status, pnl = simulate_pnl(
                        actual=actual,
                        selection=selection,
                        stake=stake,
                        odds=Decimal(str(odds)),
                    )
                    if status == "skip":
                        continue

                    agg.recs += 1
                    if status == "won":
                        agg.won += 1
                    elif status == "lost":
                        agg.lost += 1
                    else:  # void
                        agg.void += 1
                    agg.total_staked += stake
                    agg.total_pnl += pnl

    # Build overall by summing aggregates. Hit rate / ROI on the
    # overall row are computed on the summed staked + pnl rather
    # than averaging per-market rates.
    overall = MarketAggregate(sport="ALL", market="ALL")
    for agg in aggregates.values():
        overall.recs += agg.recs
        overall.won += agg.won
        overall.lost += agg.lost
        overall.void += agg.void
        overall.total_staked += agg.total_staked
        overall.total_pnl += agg.total_pnl

    return BacktestResult(
        start=start,
        end=end,
        params={
            "ev_threshold": ev_threshold,
            "prob_floor": prob_floor,
            "kelly_fraction": kelly_fraction_arg,
            "nba_prob_cap": nba_prob_cap,
            "bankroll": float(bankroll_d),
            "prediction_version": prediction_version,
        },
        overall=overall,
        by_market=sorted(aggregates.values(), key=lambda a: (a.sport, a.market)),
    )


def render_markdown(result: BacktestResult) -> str:
    lines = []
    lines.append(f"# Backtest: {result.start} → {result.end}")
    lines.append("")
    p = result.params
    lines.append(
        f"**Params:** EV≥{p['ev_threshold']:.2%}, prob≥{p['prob_floor']:.0%}, "
        f"Kelly×{p['kelly_fraction']:.2f}, NBA cap={p['nba_prob_cap']:.2f}, "
        f"bankroll=${p['bankroll']:.0f}"
    )
    lines.append("")
    o = result.overall
    lines.append(
        f"**Overall:** {o.recs} recs · {o.won}W / {o.lost}L / {o.void}V "
        f"({o.hit_rate:.1%} hit) · staked ${float(o.total_staked):,.2f} · "
        f"P/L ${float(o.total_pnl):+,.2f} · ROI {o.roi_pct:+.2f}%"
    )
    lines.append("")
    lines.append("| Sport | Market | Recs | W | L | V | Hit | Staked | P/L | ROI |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a in result.by_market:
        lines.append(
            f"| {a.sport} | {a.market} | {a.recs} | {a.won} | {a.lost} | {a.void} | "
            f"{a.hit_rate:.1%} | ${float(a.total_staked):,.2f} | "
            f"${float(a.total_pnl):+,.2f} | {a.roi_pct:+.2f}% |"
        )
    return "\n".join(lines)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="Inclusive start date YYYY-MM-DD.")
    p.add_argument("--end", required=True, help="Inclusive end date YYYY-MM-DD.")
    p.add_argument("--ev-threshold", type=float, default=0.03, help="Minimum EV for rec (default 0.03).")
    p.add_argument("--prob-floor", type=float, default=0.40, help="Minimum model prob (default 0.40).")
    p.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Stake = bankroll × Kelly × this fraction (default 0.25 = quarter Kelly).",
    )
    p.add_argument(
        "--nba-prob-cap",
        type=float,
        default=0.80,
        help="Cap NBA model probability before EV/Kelly (default 0.80, matches live).",
    )
    p.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll for stake sizing.")
    p.add_argument("--output", help="Write structured result to this JSON path.")
    p.add_argument(
        "--prediction-version",
        help="Filter predictions to exact model_version (e.g. 'wf_2024-01-01' to backtest "
        "only walk-forward predictions written by scripts/walk_forward_predictions.py).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2

    result = run(
        database_url=args.database_url,
        start=args.start,
        end=args.end,
        ev_threshold=args.ev_threshold,
        prob_floor=args.prob_floor,
        kelly_fraction_arg=args.kelly_fraction,
        nba_prob_cap=args.nba_prob_cap,
        bankroll=args.bankroll,
        prediction_version=args.prediction_version,
    )

    print(render_markdown(result))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info("Wrote structured result to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
