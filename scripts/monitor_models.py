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

Coverage: the matches-based sports (soccer / NHL / NBA / NFL / tennis
/ MMA) come from `predictions`; horse racing comes from
`race_predictions` (per-entrant consensus win prob vs actual win) so
the consensus model's calibration is monitored too.

Persistence: each run writes every slice's calibration report to
`model_performance_logs` (JSONB `metrics`) so calibration is a
trackable TIME SERIES, not just a per-run alert — you can chart ECE /
Brier drift over weeks. The Telegram drift alerting is unchanged.

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
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

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


def fetch_horse_racing_slices(cur, days: int, min_samples: int) -> list[dict]:
    """Horse-racing calibration slices from race_predictions. Sport is
    always 'horse_racing'; the pairs are per-ENTRANT (consensus win
    prob vs whether the horse actually won), which is a sharper
    calibration signal than the race-level favourite-strike-rate the
    accuracy widget shows."""
    cur.execute(
        """
        SELECT 'horse_racing' AS sport,
               rp.prediction_type,
               rp.model_name,
               COUNT(*) AS n
        FROM race_predictions rp
        JOIN races r ON r.id = rp.race_id
        WHERE rp.actual_outcome IS NOT NULL
          AND r.status = 'finished'
          AND r.race_date >= NOW() - (%s || ' days')::interval
        GROUP BY rp.prediction_type, rp.model_name
        HAVING COUNT(*) >= %s
        ORDER BY rp.prediction_type, rp.model_name
        """,
        (str(days), min_samples),
    )
    return [dict(r) for r in cur.fetchall()]


def fetch_horse_racing_pairs(
    cur,
    prediction_type: str,
    model_name: str,
    days: int,
) -> tuple[list[float], list[int]]:
    """(consensus_win_prob, did_win) pairs for one horse-racing slice.
    rp.confidence is the devigged consensus win probability;
    rp.actual_outcome is 1.0 if the entrant won, 0.0 otherwise."""
    cur.execute(
        """
        SELECT rp.confidence AS picked_prob,
               (rp.actual_outcome)::int AS correct
        FROM race_predictions rp
        JOIN races r ON r.id = rp.race_id
        WHERE rp.model_name = %s
          AND rp.prediction_type = %s
          AND rp.actual_outcome IS NOT NULL
          AND rp.confidence IS NOT NULL
          AND r.status = 'finished'
          AND r.race_date >= NOW() - (%s || ' days')::interval
        """,
        (model_name, prediction_type, str(days)),
    )
    predicted: list[float] = []
    actual: list[int] = []
    for r in cur.fetchall():
        predicted.append(float(r["picked_prob"]))
        actual.append(int(r["correct"]))
    return predicted, actual


def persist_reports(cur, summary: list[dict], days: int) -> int:
    """Write each slice's calibration report to model_performance_logs
    so ECE / Brier / accuracy become a queryable DAILY time series.

    Idempotent per day: the monitor runs every 15 min, but we only want
    one row per (model, sport, market) per day, so the INSERT is guarded
    by a NOT EXISTS check on (model_name, sport, evaluation_date=today,
    metrics->>'prediction_type'). The first run of the day writes the
    row; later runs skip it. prediction_type is stored inside the JSONB
    metrics because model_performance_logs has no such column.

    Returns the number of rows actually written. Best-effort — callers
    wrap this in try/except so a persistence failure can't break the
    drift-alert path."""
    written = 0
    for s in summary:
        report = s["report"]
        metrics = asdict(report)
        metrics["prediction_type"] = s["prediction_type"]
        cur.execute(
            """
            INSERT INTO model_performance_logs
                (model_name, model_version, sport, evaluation_date,
                 metrics, sample_size, date_range_start, date_range_end, notes)
            SELECT %s, %s, %s, CURRENT_DATE, %s, %s,
                   (NOW() - (%s || ' days')::interval)::date, CURRENT_DATE, %s
            WHERE NOT EXISTS (
                SELECT 1 FROM model_performance_logs
                WHERE model_name = %s
                  AND sport = %s
                  AND evaluation_date = CURRENT_DATE
                  AND metrics->>'prediction_type' = %s
            )
            """,
            (
                s["model_name"],
                "monitor-rolling",
                s["sport"],
                Json(metrics),
                report.n,
                str(days),
                f"calibration monitor: {s['sport']}/{s['prediction_type']} rolling {days}d",
                s["model_name"],
                s["sport"],
                s["prediction_type"],
            ),
        )
        written += cur.rowcount
    return written


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


# ── Constant-prior canary (audit doc §1.1.5) ─────────────────────────

# The worst incident this system had (a month of constant-prior soccer
# predictions) produced rows that were individually plausible but
# collectively identical. Calibration drift metrics need weeks of graded
# outcomes to notice that; this canary catches it on UNGRADED rows within
# one monitoring tick by asking: across the most recent serve-path
# predictions, how many DISTINCT probability vectors are there?
CANARY_WINDOW = 100  # most recent predictions to inspect
CANARY_MIN_ROWS = 30  # below this, too few rows to judge (quiet season)
CANARY_MIN_DISTINCT = 5  # fewer distinct home-probs than this ⇒ alert


def constant_prior_canary(cur) -> List[DriftFinding]:
    """Alert if the last CANARY_WINDOW soccer ensemble match_result
    predictions collapse to < CANARY_MIN_DISTINCT distinct home
    probabilities — the signature of every member failing and the
    ensemble (or a lone Poisson/DC survivor) serving its global prior."""
    cur.execute(
        """
        SELECT COUNT(*) AS n_rows,
               COUNT(DISTINCT p.probabilities->>'home') AS n_distinct
        FROM (
            SELECT pr.probabilities
            FROM predictions pr
            JOIN matches m ON m.id = pr.match_id
            JOIN leagues l ON l.id = m.league_id
            WHERE l.sport = 'soccer'
              AND pr.prediction_type = 'match_result'
              AND pr.model_name = 'ensemble'
            ORDER BY pr.created_at DESC
            LIMIT %(window)s
        ) p
        """,
        {"window": CANARY_WINDOW},
    )
    row = cur.fetchone()
    n_rows, n_distinct = row["n_rows"], row["n_distinct"]
    if n_rows < CANARY_MIN_ROWS:
        logger.info("Constant-prior canary: only %d recent soccer predictions — skipping", n_rows)
        return []
    if n_distinct >= CANARY_MIN_DISTINCT:
        logger.info(
            "Constant-prior canary OK: %d distinct home-probs across last %d soccer predictions",
            n_distinct,
            n_rows,
        )
        return []
    return [
        DriftFinding(
            sport="soccer",
            market="match_result",
            metric="distinct_probs",
            severity="alert",
            current=float(n_distinct),
            threshold=float(CANARY_MIN_DISTINCT),
            message=(
                f"CONSTANT-PRIOR CANARY: only {n_distinct} distinct home probabilities "
                f"across the last {n_rows} soccer ensemble predictions (need ≥ "
                f"{CANARY_MIN_DISTINCT}). The serve path is likely emitting a global "
                f"prior — check the feature__ bridge and ensemble member failures "
                f"(see audit doc §1.1)."
            ),
        )
    ]


# ── Orchestration ────────────────────────────────────────────────────


def run(database_url: str, days: int, min_samples: int, thresholds: DriftThresholds) -> dict:
    summary: list[dict] = []
    all_findings: List[DriftFinding] = []

    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            slices = fetch_slices(cur, days, min_samples)
            slices += fetch_horse_racing_slices(cur, days, min_samples)
            logger.info("Found %d slices to monitor (window=%dd, min_samples=%d)", len(slices), days, min_samples)

            for s in slices:
                if s["sport"] == "horse_racing":
                    predicted, actual = fetch_horse_racing_pairs(
                        cur, s["prediction_type"], s["model_name"], days
                    )
                else:
                    predicted, actual = fetch_slice_pairs(
                        cur, s["sport"], s["prediction_type"], s["model_name"], days
                    )
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

            # Constant-prior canary (audit doc §1.1.5) — unlike the
            # calibration slices above, this inspects UNGRADED recent
            # predictions, so it fires within one tick of a serve-path
            # regression instead of weeks later.
            all_findings.extend(constant_prior_canary(cur))

            # Persist the calibration time series (best-effort — must not
            # break the drift-alert path below).
            try:
                n_persisted = persist_reports(cur, summary, days)
                conn.commit()
                logger.info("Persisted %d calibration reports to model_performance_logs", n_persisted)
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                logger.warning("Calibration persistence failed (alerting unaffected): %s", exc)

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
