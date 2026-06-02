"""Walk graded predictions per (sport, market), compute calibration
metrics, alert on drift.

Runs on a rolling window (default 30 days). For each
(sport, prediction_type, model_name) slice with at least
--min-samples graded predictions:

  1. Pull (predicted_prob_for_picked_outcome, is_correct) pairs.
     Pushes (is_correct IS NULL) are excluded — they're neither
     right nor wrong and would distort calibration.
  2. Compute ECE / MCE / Brier / log-loss + accuracy via
     calibration_metrics.calibration_report.
  3. Compare against DriftThresholds; collect findings.
  4. Render a markdown summary to stdout.
  5. If any 'alert'-severity findings exist, post a single Telegram
     message via telegram_notify.send_telegram_digest. Warnings
     stay in the markdown but don't page.

Idempotent: the script doesn't write anything to the DB. Re-running
on the same window produces the same numbers (modulo new
graded rows landing between runs).

Usage (inside the api container):

    # 30-day rolling check
    python /app/scripts/monitor_models.py

    # Last 7 days only
    python /app/scripts/monitor_models.py --days 7

    # Custom thresholds (e.g. tighter ECE for a market with more data)
    python /app/scripts/monitor_models.py --ece-alert 0.07

The script is wired into the auspex_pipeline DAG (every 15 min) so
drift surfaces on the same cadence as the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Reuse the pure calibration math + the shared telegram digest.
sys.path.insert(0, os.path.dirname(__file__))

from calibration_metrics import (  # noqa: E402
    CalibrationReport,
    DriftFinding,
    DriftThresholds,
    calibration_report,
    detect_drift,
)
from telegram_notify import Alert, send_telegram_digest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("monitor_models")


# Friendly market labels for the report + alert. Mirrors the maps in
# precompute_predictions_*.py so the operator sees the same labels
# they see in the prediction UI.
MARKET_DISPLAY = {
    ("soccer", "match_result"): "Soccer 1X2",
    ("nhl", "moneyline"): "NHL Moneyline",
    ("nhl", "match_result"): "NHL Regulation",
    ("nhl", "spread"): "NHL Puck Line",
    ("nhl", "total"): "NHL Total",
    ("nba", "moneyline"): "NBA Moneyline",
    ("nba", "spread"): "NBA Spread",
    ("nba", "total"): "NBA Total",
}


# ── DB I/O ───────────────────────────────────────────────────────────


def fetch_slices(cur, days: int, min_samples: int) -> list[dict]:
    """Return rows of {sport, prediction_type, model_name, count} that
    pass the min-samples gate, sorted so the report is stable across
    runs."""
    cur.execute(
        """
        SELECT l.sport,
               p.prediction_type,
               p.model_name,
               COUNT(*) AS n
        FROM predictions p
        JOIN matches m ON m.id = p.match_id
        JOIN leagues l ON l.id = m.league_id
        WHERE p.is_correct IS NOT NULL
          AND m.status = 'finished'
          AND m.match_date >= NOW() - (%s || ' days')::interval
        GROUP BY l.sport, p.prediction_type, p.model_name
        HAVING COUNT(*) >= %s
        ORDER BY l.sport, p.prediction_type, p.model_name
        """,
        (str(days), min_samples),
    )
    return [dict(r) for r in cur.fetchall()]


def fetch_slice_pairs(
    cur,
    sport: str,
    prediction_type: str,
    model_name: str,
    days: int,
) -> tuple[list[float], list[int]]:
    """Pull (predicted_prob_for_picked, is_correct) pairs for one
    (sport, market, model_name) slice. We use the picked outcome's
    probability — predictions.probabilities is JSONB keyed by label,
    so we look up probabilities[predicted_outcome]."""
    cur.execute(
        """
        SELECT (p.probabilities->>p.predicted_outcome)::float AS picked_prob,
               (p.is_correct)::int AS correct
        FROM predictions p
        JOIN matches m ON m.id = p.match_id
        JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = %s
          AND p.prediction_type = %s
          AND p.model_name = %s
          AND p.is_correct IS NOT NULL
          AND m.status = 'finished'
          AND m.match_date >= NOW() - (%s || ' days')::interval
        """,
        (sport, prediction_type, model_name, str(days)),
    )
    rows = cur.fetchall()
    predicted: list[float] = []
    actual: list[int] = []
    for r in rows:
        if r["picked_prob"] is None:
            continue
        predicted.append(float(r["picked_prob"]))
        actual.append(int(r["correct"]))
    return predicted, actual


# ── Reporting ────────────────────────────────────────────────────────


def display_market(sport: str, prediction_type: str) -> str:
    return MARKET_DISPLAY.get((sport, prediction_type), f"{sport}/{prediction_type}")


def render_report(slices: list[dict]) -> str:
    """Markdown summary of every slice's calibration. Findings are
    appended below the per-slice rows."""
    lines = []
    lines.append("# Model monitoring report")
    lines.append("")
    lines.append("| Market | Model | n | Acc | Brier | LogLoss | ECE | MCE |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in slices:
        report: CalibrationReport = s["report"]
        market = display_market(s["sport"], s["prediction_type"])
        lines.append(
            f"| {market} | {s['model_name']} | {report.n} | "
            f"{report.accuracy:.1%} | {report.brier_score:.3f} | "
            f"{report.log_loss:.3f} | {report.ece:.3f} | {report.mce:.3f} |"
        )
    return "\n".join(lines)


def render_findings(findings: list[DriftFinding]) -> str:
    """Per-finding bullet list, grouped by severity. Returned empty
    string if no findings — caller checks before appending."""
    if not findings:
        return ""
    alerts = [f for f in findings if f.severity == "alert"]
    warns = [f for f in findings if f.severity == "warn"]
    parts = ["", "## Findings"]
    if alerts:
        parts.append("")
        parts.append("**ALERTS** (paged):")
        for f in alerts:
            parts.append(f"- {display_market(f.sport, f.market)}: {f.message}")
    if warns:
        parts.append("")
        parts.append("**Warnings** (logged):")
        for f in warns:
            parts.append(f"- {display_market(f.sport, f.market)}: {f.message}")
    return "\n".join(parts)


# ── Alerting ─────────────────────────────────────────────────────────


def build_alert(findings: list[DriftFinding]) -> Optional[Alert]:
    """Bundle alert-severity findings into a single Alert that the
    shared telegram_notify dispatcher can send. Warn-severity
    findings stay out — they're in the markdown for the operator to
    read on their own cadence."""
    alerts = [f for f in findings if f.severity == "alert"]
    if not alerts:
        return None

    # Build a representative summary line. Multiple alerts collapse
    # into one Alert so we don't spam the channel — the body lists
    # each one.
    body_lines = [f"⚠️ {f.message}" for f in alerts]
    label = "Model drift" if len(alerts) > 1 else f"Drift: {display_market(alerts[0].sport, alerts[0].market)}"

    return Alert(
        sport="monitoring",
        league_name="System",
        home_team="model_monitor",
        away_team="",
        match_date=datetime.utcnow(),
        market_label=label,
        predicted_outcome="drift_detected",
        confidence=1.0,
        probabilities={"body": "\n".join(body_lines)},
    )


# ── Orchestration ────────────────────────────────────────────────────


def run(database_url: str, days: int, min_samples: int, thresholds: DriftThresholds) -> dict:
    summary: list[dict] = []
    all_findings: List[DriftFinding] = []

    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            slices = fetch_slices(cur, days, min_samples)
            logger.info("Found %d slices to monitor (window=%dd, min_samples=%d)", len(slices), days, min_samples)

            for s in slices:
                predicted, actual = fetch_slice_pairs(cur, s["sport"], s["prediction_type"], s["model_name"], days)
                if len(predicted) < min_samples:
                    continue
                report = calibration_report(predicted, actual)
                findings = detect_drift(
                    sport=s["sport"],
                    market=s["prediction_type"],
                    report=report,
                    thresholds=thresholds,
                )
                summary.append(
                    {
                        "sport": s["sport"],
                        "prediction_type": s["prediction_type"],
                        "model_name": s["model_name"],
                        "report": report,
                        "findings": findings,
                    }
                )
                all_findings.extend(findings)

    md = render_report(summary)
    if all_findings:
        md += "\n" + render_findings(all_findings)
    print(md)

    alert_message = build_alert(all_findings)
    sent = 0
    if alert_message is not None:
        sent = send_telegram_digest([alert_message], header="Model monitoring — drift alert")
        logger.info("Drift alert dispatched (sent=%d messages)", sent)

    return {
        "slices": len(summary),
        "alerts": sum(1 for f in all_findings if f.severity == "alert"),
        "warnings": sum(1 for f in all_findings if f.severity == "warn"),
        "telegram_messages": sent,
    }


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=30, help="Rolling window in days (default 30).")
    p.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Skip slices with fewer graded preds than this (default 30 — below it the "
        "calibration numbers are too noisy to act on).",
    )
    # Threshold overrides. The defaults match calibration_metrics.DriftThresholds; the
    # CLI knobs are here for ops to dial sensitivity per environment.
    p.add_argument("--ece-warn", type=float, default=0.05)
    p.add_argument("--ece-alert", type=float, default=0.10)
    p.add_argument("--mce-warn", type=float, default=0.15)
    p.add_argument("--mce-alert", type=float, default=0.25)
    p.add_argument("--brier-drift-warn", type=float, default=0.02)
    p.add_argument("--brier-drift-alert", type=float, default=0.05)
    p.add_argument("--accuracy-floor", type=float, default=0.524)
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2

    thresholds = DriftThresholds(
        ece_warn=args.ece_warn,
        ece_alert=args.ece_alert,
        mce_warn=args.mce_warn,
        mce_alert=args.mce_alert,
        brier_drift_warn=args.brier_drift_warn,
        brier_drift_alert=args.brier_drift_alert,
        accuracy_floor=args.accuracy_floor,
    )
    counts = run(args.database_url, args.days, args.min_samples, thresholds)
    logger.info("Done. %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
