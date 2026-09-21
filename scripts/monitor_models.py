"""Walk graded predictions per (sport, market), compute calibration
metrics, alert on drift.

Runs on a rolling window (default 30 days). For each
(sport, prediction_type, model_name) slice with at least
--min-samples graded predictions:

  1. Pull (predicted_prob_for_picked_outcome, is_correct) pairs,
     ONE PER MATCH. Pushes (is_correct IS NULL) are excluded — they're
     neither right nor wrong and would distort calibration — and so
     are preseason games and stale retrain duplicates (see
     _DEDUPED_GRADED_PREDICTIONS_SQL). Reported n is therefore much
     smaller than it was before 2026-09 (NFL 220 -> ~47, MMA
     444 -> ~121); the drift thresholds are n-sensitive, so read a
     changed n as a corrected denominator, not as lost data.
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
Brier drift over weeks. Since 2026-09-21 the stored `mce` ignores buckets
under MIN_BUCKET_N_FOR_MCE; `mce_raw` and `mce_min_bucket_n` sit beside it.

Usage (inside the api container):

    # 30-day rolling check
    python /app/scripts/monitor_models.py

    # Last 7 days only
    python /app/scripts/monitor_models.py --days 7

    # Custom thresholds (e.g. tighter ECE for a market with more data)
    python /app/scripts/monitor_models.py --ece-alert 0.07

The script runs from its own hourly DAG (dags/monitor_models_dag.py). An
identical set of alert-severity findings pages at most once per
_PAGE_DEDUP_TTL_SECONDS (Redis SETNX, fail-open), so a persisting condition
is a daily page rather than an hourly one.

Every paged line names its slice and n. Findings on slices below
DriftThresholds.min_n_to_page graded predictions, or on streams that
scripts/rec_gating.py has switched OFF, are downgraded to warn (still
computed, logged and persisted — just not paged) with the reason in the
message.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

# Reuse the pure calibration math + the shared telegram digest.
sys.path.insert(0, os.path.dirname(__file__))

# The preseason-exclusion predicate is defined ONCE, in
# services/ml-models/src/utils/training_data.py — see the long comment
# there for the three marker states. The api container (where this
# runs) already has that tree on PYTHONPATH; the repo-relative path
# keeps local dev + the unit tests working. Position 0 also guarantees
# `utils` resolves to ml-models' package rather than the same-named one
# under services/data-ingestion/src.
_ML_MODELS_SRC = str(Path(__file__).resolve().parent.parent / "services" / "ml-models" / "src")
for _p in ("/app/services/ml-models/src", _ML_MODELS_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from calibration_metrics import (  # noqa: E402
    MIN_BUCKET_N_FOR_MCE,
    CalibrationReport,
    DriftFinding,
    DriftThresholds,
    calibration_report,
    detect_drift,
    maximum_calibration_error,
)
from telegram_notify import Alert, send_telegram_digest  # noqa: E402
from utils.training_data import preseason_exclusion_sql  # noqa: E402

try:  # rec_gating is a sibling script; the monitor must keep working without it.
    import rec_gating  # noqa: E402
except Exception as _exc:  # noqa: BLE001
    rec_gating = None  # type: ignore[assignment]
    logging.getLogger("monitor_models").warning("rec_gating unavailable (%s) — no gated-stream downgrade", _exc)

# Only count graded rows for games that were actually played for real.
NOT_PRESEASON_SQL = preseason_exclusion_sql("m")

# The monitor keys slices by predictions.prediction_type; rec_gating keys
# streams by betting_recommendations.bet_type. They coincide except where a
# generator renames the market on the way out (see each generator's
# insert). Keyed by (sport, prediction_type) because the same
# prediction_type means different streams in different sports.
_PREDICTION_TYPE_TO_BET_TYPE = {
    ("soccer", "match_result"): "1x2",
    ("nhl", "spread"): "puck_line",  # generate_recommendations_nhl.py writes bet_type 'puck_line'
    # NHL regulation (stored as prediction_type 'match_result') has no rec
    # stream — the NHL generator skips it by design — so it deliberately has
    # no entry here and resolves to the sport default.
}

# A Telegram message is chunked above ~3,900 chars, and a partially
# delivered chunked page would look like success (sent > 0) while the
# alerts in the failed chunk went nowhere. Cap the page so it can never
# chunk; the full list is always in the run log and the markdown report.
_MAX_PAGE_LINES = 12

# How many outcomes a pick is chosen from — the input to the 2-way
# break-even check. NEVER inferred from the stored probability vector:
# multi-line markets keep every line in one JSONB (asian_handicap carries
# 51 keys, over_under 12) yet each pick is a 2-way bet, so a shape-based
# count would silently switch the break-even check OFF for a live money
# stream. Unknown types fall back to the vector count only when it says
# "more than 2", which errs toward NOT paging "unprofitable" on a market we
# have not classified — and logs so the map gets extended.
_N_CLASSES_BY_TYPE = {
    "match_result": 3,
    "match_result_ht": 3,
    "double_chance": 3,
    "correct_score": 13,
    "winning_margin": 7,
    "total_goals": 7,
    "ht_ft_double_result": 9,
    "result_btts": 6,
    "result_over_under": 6,
    "clean_sheet": 4,
    "win_to_nil": 4,
}
_TWO_WAY_MULTI_LINE = frozenset({"asian_handicap", "over_under", "team_total", "over_under_ht"})

# A persisting condition should page once a day, not once an hour. The
# monitor DAG runs hourly; without this the 2026-09-21 findings would have
# paged 24 times for the same ten numbers.
_PAGE_DEDUP_TTL_SECONDS = 23 * 3600


def n_classes_for(prediction_type: str, vector_count: Optional[int]) -> int:
    if prediction_type in _TWO_WAY_MULTI_LINE:
        return 2
    if prediction_type in _N_CLASSES_BY_TYPE:
        return _N_CLASSES_BY_TYPE[prediction_type]
    if vector_count and vector_count > 2:
        logger.warning(
            "prediction_type %r is not in _N_CLASSES_BY_TYPE; using its %d-key vector as the class count "
            "(break-even check skipped) — add it to the map",
            prediction_type,
            vector_count,
        )
        return int(vector_count)
    return 2


def _stream_is_gated(sport: str, prediction_type: str) -> bool:
    """True when scripts/rec_gating.py has the recommendation stream for this
    slice switched OFF. Findings on a gated stream are real — that is usually
    why it is gated — but they must not page every hour; detect_drift
    downgrades them to warn so the model is still measured back to life
    from the persisted time series. Fails OPEN (not gated) so a rec_gating
    problem can never silence a real page."""
    if rec_gating is None:
        return False
    try:
        bet_type = _PREDICTION_TYPE_TO_BET_TYPE.get((sport, prediction_type), prediction_type)
        return not rec_gating.gate_for(sport, bet_type).enabled
    except Exception as exc:  # noqa: BLE001
        logger.warning("gated-stream lookup failed for %s/%s (%s) — treating as not gated", sport, prediction_type, exc)
        return False


# ── Page dedup ───────────────────────────────────────────────────────
#
# One key PER FINDING (sport:market:model:metric), not per alert set: a
# set-level key re-pages whenever an unrelated finding appears or clears,
# and a single metric flapping around its threshold would page once per
# distinct subset. A new condition pages immediately; a persisting one
# pages at most once per TTL.
#
# Keys are marked ONLY AFTER a successful Telegram send. Marking before
# the send (the first draft did) turns any transient send failure into
# 23 hours of silence with a log line falsely claiming "already paged" —
# the hourly rerun used to be the retry, and it must stay one. The DAG is
# max_active_runs=1, so read-then-mark has no race.
#
# Every Redis problem fails OPEN (page): a duplicate page is a nuisance, a
# swallowed one is the failure mode this repo is built to avoid. That
# includes a Redis that accepts TCP but never answers — the client here
# carries socket timeouts precisely so it cannot hang the monitor.


def _dedup_client(redis_url: Optional[str] = None):
    url = redis_url or os.environ.get("REDIS_URL")
    if not url:
        return None
    try:
        from redis import Redis  # type: ignore

        return Redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("drift page dedup client unavailable (%s) — paging without dedup", exc)
        return None


def _page_key(f: DriftFinding) -> str:
    return f"auspex:drift_page:{f.sport}:{f.market}:{f.model_name or '-'}:{f.metric}"


def select_new_alerts(findings: list[DriftFinding], *, redis_url: Optional[str] = None) -> list[DriftFinding]:
    """The alert-severity findings that have NOT been paged within the TTL.
    Read-only: nothing is marked here. Fails open (every alert is 'new')."""
    alerts = [f for f in findings if f.severity == "alert"]
    if not alerts:
        return []
    client = _dedup_client(redis_url)
    if client is None:
        return alerts
    try:
        flags = client.mget([_page_key(f) for f in alerts])
    except Exception as exc:  # noqa: BLE001
        logger.warning("drift page dedup read failed (%s) — paging everything", exc)
        return alerts
    return [f for f, seen in zip(alerts, flags) if not seen]


def mark_paged(findings: list[DriftFinding], *, redis_url: Optional[str] = None) -> int:
    """Record a SUCCESSFUL page for each alert-severity finding. Call this
    only after send_telegram_digest reported at least one message sent.
    Returns the number of keys written (0 when Redis is unavailable)."""
    alerts = [f for f in findings if f.severity == "alert"]
    client = _dedup_client(redis_url)
    if client is None or not alerts:
        return 0
    stamp = datetime.now(timezone.utc).isoformat()
    written = 0
    for f in alerts:
        try:
            client.set(_page_key(f), stamp, ex=_PAGE_DEDUP_TTL_SECONDS)
            written += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("drift page dedup mark failed for %s (%s)", _page_key(f), exc)
            break
    return written


def page_fingerprint(findings: list[DriftFinding]) -> str:
    """Short stable id of the alert set, for log lines only."""
    keys = sorted({_page_key(f) for f in findings if f.severity == "alert"})
    return hashlib.sha1("|".join(keys).encode()).hexdigest()[:16]


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


# ── One row per MATCH, not per predictions row ───────────────────────
#
# Every weekly retrain writes a FRESH predictions row for the same
# (match, market) under a new model_version, and grading fills
# is_correct on all of them. A naive COUNT(*) therefore counted the
# same game once per retrain: the NFL 30-day slice reported
# n = 220 / 220 / 226 for only 47 / 49 / 49 DISTINCT matches (4.7x),
# and MMA's headline "n = 444" was 121 distinct fights. Those extra
# rows are near-duplicates of each other, so ECE/Brier confidence
# intervals computed from them were far too tight.
#
# DISTINCT ON (match_id, prediction_type) ... ORDER BY ... created_at
# DESC keeps the NEWEST graded row per match+market — the version
# actually serving now — and drops the historical copies. Preseason
# games are excluded too (the current NFL 30-day slice is 100
# percent preseason).
#
# Pushes stay excluded via is_correct IS NOT NULL: they are neither
# right nor wrong and would distort calibration.
_DEDUPED_GRADED_PREDICTIONS_SQL = f"""        SELECT DISTINCT ON (p.match_id, p.prediction_type)
               p.match_id,
               p.created_at,
               l.sport,
               p.prediction_type,
               p.model_name,
               (p.probabilities->>p.predicted_outcome)::float AS picked_prob,
               (p.is_correct)::int AS correct,
               (CASE WHEN jsonb_typeof(p.probabilities) = 'object'
                     THEN (SELECT COUNT(*) FROM jsonb_object_keys(p.probabilities))
                     ELSE 0 END)::int AS n_classes
        FROM predictions p
        JOIN matches m ON m.id = p.match_id
        JOIN leagues l ON l.id = m.league_id
        WHERE p.is_correct IS NOT NULL
          AND m.status = 'finished'
          AND m.match_date >= NOW() - (%(days)s || ' days')::interval
          AND {NOT_PRESEASON_SQL}
        ORDER BY p.match_id, p.prediction_type, p.created_at DESC
"""


def fetch_slices(cur, days: int, min_samples: int) -> list[dict]:
    """Return rows of {sport, prediction_type, model_name, count} that
    pass the min-samples gate, sorted so the report is stable across
    runs.

    n is per-MATCH, not per predictions row. Retrain duplicates are
    collapsed by _DEDUPED_GRADED_PREDICTIONS_SQL and preseason games
    are dropped, so reported n fell sharply when this landed
    (NFL 220 -> ~47, MMA 444 -> ~121). The DriftThresholds are
    n-sensitive — a slice that used to clear --min-samples on inflated
    duplicates may now sit below it and stop being reported at all.
    That is the honest number; re-tune the gate, don't re-inflate n.

    model_name comes from the surviving (newest) row, so a slice is
    attributed to the version actually serving.

    n_classes is the modal number of outcomes in the slice's probability
    vectors (2 for moneyline/spread/total/BTTS, 3 for 1X2, 13 for correct
    score). detect_drift needs it because the 52.4% accuracy floor is a
    2-way break-even that is meaningless for k-way picks.
    """
    cur.execute(
        f"""
        SELECT sport,
               prediction_type,
               model_name,
               COUNT(*) AS n,
               MODE() WITHIN GROUP (ORDER BY n_classes) AS n_classes
        FROM (
{_DEDUPED_GRADED_PREDICTIONS_SQL}
        ) d
        GROUP BY sport, prediction_type, model_name
        HAVING COUNT(*) >= %(min_samples)s
        ORDER BY sport, prediction_type, model_name
        """,
        {"days": str(days), "min_samples": min_samples},
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
    so we look up probabilities[predicted_outcome].

    One pair per MATCH: the same de-duplication fetch_slices uses runs
    here first, so the counts in the report and the arrays the
    calibration math sees can never disagree. Preseason is excluded and
    pushes stay out (is_correct IS NOT NULL).
    """
    cur.execute(
        f"""
        SELECT picked_prob, correct
        FROM (
{_DEDUPED_GRADED_PREDICTIONS_SQL}
        ) d
        WHERE d.sport = %(sport)s
          AND d.prediction_type = %(prediction_type)s
          AND d.model_name = %(model_name)s
        """,
        {
            "days": str(days),
            "sport": sport,
            "prediction_type": prediction_type,
            "model_name": model_name,
        },
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
        # `mce` changed meaning on 2026-09-21 (buckets under
        # MIN_BUCKET_N_FOR_MCE no longer count). Store the floor and the raw
        # all-buckets maximum alongside it so the time series is
        # self-describing across the boundary and both are queryable.
        metrics["mce_min_bucket_n"] = MIN_BUCKET_N_FOR_MCE
        metrics["mce_raw"] = maximum_calibration_error(report.buckets, min_bucket_n=0)
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


def _finding_label(f: DriftFinding) -> str:
    """'<Sport Market> [model] (n=N)' — the same prefix on every rendering
    of a finding, so a number is never shown without the slice it belongs
    to. The model is shown only when it disambiguates (horse racing's
    ranker and consensus are both 'win')."""
    model = f" [{f.model_name}]" if f.model_name else ""
    return f"{display_market(f.sport, f.market)}{model} (n={f.n if f.n else '?'})"


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
            parts.append(f"- {_finding_label(f)}: {f.message}")
    if warns:
        parts.append("")
        parts.append("**Warnings** (logged):")
        for f in warns:
            parts.append(f"- {_finding_label(f)}: {f.message}")
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
    # each one. EVERY line names its slice and n: the 2026-09-21 page
    # was ten bare "MCE 0.923 >= 0.250" lines with no sport, market or
    # sample size, and could not be acted on as written.
    ordered = sorted(alerts, key=lambda f: (f.sport, f.market, f.model_name, f.metric))
    body_lines = [f"⚠️ {_finding_label(f)}: {f.message}" for f in ordered[:_MAX_PAGE_LINES]]
    if len(ordered) > _MAX_PAGE_LINES:
        body_lines.append(f"… +{len(ordered) - _MAX_PAGE_LINES} more alert(s) — full list in the monitor run log")
    label = "Model drift" if len(alerts) > 1 else f"Drift: {display_market(alerts[0].sport, alerts[0].market)}"

    return Alert(
        sport="monitoring",
        league_name="System",
        home_team="model_monitor",
        away_team="",
        match_date=datetime.now(timezone.utc),
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

    # connect_timeout: a Postgres that accepts TCP but never answers must not
    # hang the monitor (see the DAG's `timeout 9m` for the outer bound).
    with psycopg2.connect(database_url, connect_timeout=10) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            slices = fetch_slices(cur, days, min_samples)
            slices += fetch_horse_racing_slices(cur, days, min_samples)
            logger.info("Found %d slices to monitor (window=%dd, min_samples=%d)", len(slices), days, min_samples)

            for s in slices:
                if s["sport"] == "horse_racing":
                    predicted, actual = fetch_horse_racing_pairs(cur, s["prediction_type"], s["model_name"], days)
                else:
                    predicted, actual = fetch_slice_pairs(cur, s["sport"], s["prediction_type"], s["model_name"], days)
                if len(predicted) < min_samples:
                    continue
                report = calibration_report(predicted, actual)
                findings = detect_drift(
                    sport=s["sport"],
                    market=s["prediction_type"],
                    report=report,
                    thresholds=thresholds,
                    # Horse-racing slices carry no n_classes (per-entrant
                    # binary pairs); the accuracy floor is skipped for
                    # racing inside detect_drift anyway.
                    n_classes=n_classes_for(s["prediction_type"], s.get("n_classes")),
                    stream_gated=_stream_is_gated(s["sport"], s["prediction_type"]),
                    model_name=s["model_name"],
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

    alerts = [f for f in all_findings if f.severity == "alert"]
    sent = 0
    suppressed = False
    send_failed = False
    if alerts:
        new_alerts = select_new_alerts(all_findings)
        if not new_alerts:
            suppressed = True
            logger.info(
                "Drift alert NOT re-paged: every alert in set %s was paged within the last %dh",
                page_fingerprint(all_findings),
                _PAGE_DEDUP_TTL_SECONDS // 3600,
            )
        else:
            # The page carries EVERY current alert (context), but only a
            # not-yet-paged one triggers it. Keys are marked only on success.
            alert_message = build_alert(all_findings)
            sent = send_telegram_digest([alert_message], header="Model monitoring — drift alert")
            if sent > 0:
                marked = mark_paged(all_findings)
                logger.info(
                    "Drift alert dispatched (sent=%d messages, %d new, %d dedup keys)", sent, len(new_alerts), marked
                )
            elif os.environ.get("ENABLE_TELEGRAM_NOTIFICATIONS", "false").lower() == "true":
                # Telegram is ON and nothing went out: that is a delivery
                # failure, not a quiet hour. Say so at ERROR and let main()
                # fail the task so Airflow's failure hook pages instead.
                send_failed = True
                logger.error(
                    "Drift alert SEND FAILED with %d alert(s) pending (set %s) — nothing marked, will retry next run",
                    len(alerts),
                    page_fingerprint(all_findings),
                )
            else:
                logger.info("Drift alert built but Telegram is disabled (sent=0); nothing marked")

    return {
        "slices": len(summary),
        "alerts": len(alerts),
        "warnings": sum(1 for f in all_findings if f.severity == "warn"),
        "telegram_messages": sent,
        "telegram_suppressed_duplicate": suppressed,
        "telegram_send_failed": send_failed,
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
    if counts.get("telegram_send_failed"):
        # A drift alert existed and could not be delivered. Exit non-zero so
        # the Airflow task fails and its failure hook pages — a green task
        # with a lost page is exactly the silent failure to avoid.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
