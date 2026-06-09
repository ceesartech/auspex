"""Pure calibration math for model monitoring.

Compute ECE / MCE / Brier / log-loss / reliability-diagram buckets
over a list of (predicted_prob, actual_outcome) pairs. Used by
scripts/monitor_models.py to detect when a model's confidence is
drifting from its actual hit rate — the textbook signal that a
deployed model is going stale.

No DB / network / sklearn dependency. Single source of truth for
how the system measures calibration; the API's accuracy widget
(services/api/src/routes/accuracy.py) reports accuracy but doesn't
compute calibration — this fills that gap.

Conventions:
  * `predicted_prob` is the model's probability for the SELECTED
    outcome (the one it picked as predicted_outcome). NOT a vector
    of class probabilities — we collapse to scalar before passing
    in. This matches the practical question "when the model is X%
    confident, how often is it right?"
  * `actual` is 1 if the model's pick was correct, 0 otherwise.
  * Pushes (is_correct=NULL in DB) MUST be filtered out before
    calling these functions — they're neither right nor wrong, so
    counting them either way inflates or deflates the bucket.

References:
  * ECE: Naeini et al. (2015) "Obtaining well-calibrated probabilities"
  * Brier: Brier (1950); decomposes into reliability + resolution
  * Log-loss: standard cross-entropy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

# Default bucket count. 10 = 10% wide bins, the standard. Higher
# counts (e.g. 20) give finer-grained reliability diagrams but
# require more data per bin to be statistically meaningful.
DEFAULT_BUCKETS = 10


@dataclass(frozen=True)
class Bucket:
    """One row of the reliability diagram. `mean_predicted` is the
    average of predicted_prob values that fell in this bucket;
    `mean_actual` is the average of `actual` values for the same
    rows. A well-calibrated model has `mean_predicted` ≈ `mean_actual`
    in every bucket."""

    lower: float  # bucket left edge
    upper: float  # bucket right edge
    n: int  # number of predictions in this bucket
    mean_predicted: float  # mean of predicted_prob values
    mean_actual: float  # mean of actual values (= hit rate in bucket)


@dataclass(frozen=True)
class CalibrationReport:
    """Aggregated calibration result for a (sport, market) slice."""

    n: int  # total predictions used (excludes pushes)
    accuracy: float  # mean(actual) — overall hit rate
    brier_score: float  # mean((pred - actual)^2). Lower = better.
    log_loss: float  # mean(-[y log p + (1-y) log(1-p)]). Lower = better.
    ece: float  # Expected Calibration Error. 0 = perfect calibration.
    mce: float  # Max bucket calibration error. Worst single bucket.
    buckets: List[Bucket]


# ── Single-metric helpers (independently testable) ───────────────────


def brier_score(predicted: Sequence[float], actual: Sequence[int]) -> float:
    """Brier score: mean squared error between predicted prob and
    actual outcome (0 or 1). Lower is better; 0 = perfect prediction,
    0.25 = always predicting 50%. For 2-class problems this
    decomposes neatly into reliability + resolution."""
    if not predicted:
        return 0.0
    return sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(predicted)


def log_loss(predicted: Sequence[float], actual: Sequence[int], eps: float = 1e-15) -> float:
    """Cross-entropy log-loss. Lower is better; 0 = perfect, ln(2) ≈
    0.693 = always predicting 50%. We clip predicted to [eps, 1-eps]
    so log(0) doesn't blow up if a model emits a degenerate 0.0 or
    1.0 (XGBoost can do this on rare classes)."""
    if not predicted:
        return 0.0
    total = 0.0
    for p, a in zip(predicted, actual):
        p = max(eps, min(1 - eps, p))
        total += -(a * math.log(p) + (1 - a) * math.log(1 - p))
    return total / len(predicted)


def reliability_buckets(
    predicted: Sequence[float],
    actual: Sequence[int],
    n_buckets: int = DEFAULT_BUCKETS,
) -> List[Bucket]:
    """Partition predictions into equal-width buckets by predicted
    probability and compute per-bucket statistics. Empty buckets are
    OMITTED from the returned list (you can't compute calibration
    error on a bucket with no data)."""
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")
    if not predicted:
        return []

    width = 1.0 / n_buckets
    # Initialize accumulators for each bucket.
    sums_pred = [0.0] * n_buckets
    sums_actual = [0.0] * n_buckets
    counts = [0] * n_buckets

    for p, a in zip(predicted, actual):
        # Clamp to [0, 1] defensively; cap the topmost edge so a
        # predicted prob of exactly 1.0 lands in the last bucket
        # rather than out-of-range.
        p_clamped = min(max(p, 0.0), 1.0)
        # Use p * n_buckets instead of p / width — exact for clean
        # decimals (0.6 * 10 = 6.0 exactly, but 0.6 / 0.1 = 5.999...
        # due to float representation of 0.1).
        idx = min(int(p_clamped * n_buckets), n_buckets - 1)
        sums_pred[idx] += p
        sums_actual[idx] += a
        counts[idx] += 1

    out: List[Bucket] = []
    for i in range(n_buckets):
        if counts[i] == 0:
            continue
        out.append(
            Bucket(
                lower=i * width,
                upper=(i + 1) * width,
                n=counts[i],
                mean_predicted=sums_pred[i] / counts[i],
                mean_actual=sums_actual[i] / counts[i],
            )
        )
    return out


def expected_calibration_error(buckets: Sequence[Bucket], total_n: int) -> float:
    """ECE: weighted average of per-bucket calibration gaps,
    weighted by bucket size. 0 = perfect calibration (every bucket's
    predicted prob matches its actual hit rate); 0.5 = a model that
    says 100% on coin flips. Production-quality models are usually
    < 0.05; > 0.10 is a red flag."""
    if total_n == 0:
        return 0.0
    return sum((b.n / total_n) * abs(b.mean_predicted - b.mean_actual) for b in buckets)


def maximum_calibration_error(buckets: Sequence[Bucket]) -> float:
    """MCE: worst single bucket's calibration gap. Catches local
    miscalibration (e.g., a model that's well-calibrated overall but
    catastrophically wrong in its highest-confidence bucket) that
    ECE's weighted average could smooth over. The 0.80 NBA cap was
    set because the spread model's MCE was ~0.21 — caller can
    monitor whether retraining narrows this."""
    if not buckets:
        return 0.0
    return max(abs(b.mean_predicted - b.mean_actual) for b in buckets)


# ── Aggregate ────────────────────────────────────────────────────────


def calibration_report(
    predicted: Sequence[float],
    actual: Sequence[int],
    n_buckets: int = DEFAULT_BUCKETS,
) -> CalibrationReport:
    """One-call summary that runs every metric over the same inputs.
    Returns a CalibrationReport dataclass; the monitor script then
    compares the fields against drift thresholds."""
    n = len(predicted)
    if n == 0:
        return CalibrationReport(
            n=0,
            accuracy=0.0,
            brier_score=0.0,
            log_loss=0.0,
            ece=0.0,
            mce=0.0,
            buckets=[],
        )
    if len(actual) != n:
        raise ValueError(f"predicted ({n}) and actual ({len(actual)}) must be same length")

    buckets = reliability_buckets(predicted, actual, n_buckets=n_buckets)
    return CalibrationReport(
        n=n,
        accuracy=sum(actual) / n,
        brier_score=brier_score(predicted, actual),
        log_loss=log_loss(predicted, actual),
        ece=expected_calibration_error(buckets, total_n=n),
        mce=maximum_calibration_error(buckets),
        buckets=buckets,
    )


# ── Drift detection ──────────────────────────────────────────────────


@dataclass(frozen=True)
class DriftThresholds:
    """Per-metric tolerance bands. Caller can construct different
    threshold sets per market (e.g., NBA totals get stricter ECE
    because that model's already-known to be wobbly).
    """

    # ECE >= warn → log a warning; ECE >= alert → Telegram fires.
    ece_warn: float = 0.05
    ece_alert: float = 0.10
    # MCE catches local miscalibration the weighted ECE smooths.
    mce_warn: float = 0.15
    mce_alert: float = 0.25
    # Brier doesn't have a fixed "good" threshold — drift = current
    # is materially worse than baseline. baseline_brier supplied per
    # call by the monitor script (training-time Brier or rolling
    # historical avg).
    brier_drift_warn: float = 0.02  # absolute increase from baseline
    brier_drift_alert: float = 0.05
    # Hit rate dropping below break-even on a 2-class market is an
    # automatic alert (the strategy can no longer profit at any vig).
    accuracy_floor: float = 0.524  # -110 break-even


@dataclass(frozen=True)
class DriftFinding:
    """One detected calibration / accuracy issue. The monitor script
    accumulates these into a Telegram digest."""

    sport: str
    market: str
    metric: str  # 'ece' | 'mce' | 'brier' | 'accuracy'
    severity: str  # 'warn' | 'alert'
    current: float
    threshold: float
    message: str


def detect_drift(
    *,
    sport: str,
    market: str,
    report: CalibrationReport,
    thresholds: DriftThresholds,
    baseline_brier: float = 0.25,
) -> List[DriftFinding]:
    """Compare a CalibrationReport against thresholds, return a list
    of DriftFindings (empty if everything passes). Each finding is
    independent — a single market can fail multiple metrics at once
    and we report all of them so the operator sees the full picture
    rather than just the first violation."""
    findings: List[DriftFinding] = []

    # ECE
    if report.ece >= thresholds.ece_alert:
        findings.append(
            DriftFinding(
                sport=sport,
                market=market,
                metric="ece",
                severity="alert",
                current=report.ece,
                threshold=thresholds.ece_alert,
                message=f"ECE {report.ece:.3f} >= {thresholds.ece_alert:.3f} — model significantly miscalibrated",
            )
        )
    elif report.ece >= thresholds.ece_warn:
        findings.append(
            DriftFinding(
                sport=sport,
                market=market,
                metric="ece",
                severity="warn",
                current=report.ece,
                threshold=thresholds.ece_warn,
                message=f"ECE {report.ece:.3f} >= {thresholds.ece_warn:.3f} — calibration drifting",
            )
        )

    # MCE
    if report.mce >= thresholds.mce_alert:
        findings.append(
            DriftFinding(
                sport=sport,
                market=market,
                metric="mce",
                severity="alert",
                current=report.mce,
                threshold=thresholds.mce_alert,
                message=f"MCE {report.mce:.3f} >= {thresholds.mce_alert:.3f} — worst-bucket gap too wide",
            )
        )
    elif report.mce >= thresholds.mce_warn:
        findings.append(
            DriftFinding(
                sport=sport,
                market=market,
                metric="mce",
                severity="warn",
                current=report.mce,
                threshold=thresholds.mce_warn,
                message=f"MCE {report.mce:.3f} >= {thresholds.mce_warn:.3f} — worst bucket drifting",
            )
        )

    # Brier vs baseline
    brier_drift = report.brier_score - baseline_brier
    if brier_drift >= thresholds.brier_drift_alert:
        findings.append(
            DriftFinding(
                sport=sport,
                market=market,
                metric="brier",
                severity="alert",
                current=report.brier_score,
                threshold=baseline_brier + thresholds.brier_drift_alert,
                message=(
                    f"Brier {report.brier_score:.3f} is {brier_drift:+.3f} above baseline "
                    f"{baseline_brier:.3f} — model predictions less accurate"
                ),
            )
        )
    elif brier_drift >= thresholds.brier_drift_warn:
        findings.append(
            DriftFinding(
                sport=sport,
                market=market,
                metric="brier",
                severity="warn",
                current=report.brier_score,
                threshold=baseline_brier + thresholds.brier_drift_warn,
                message=(
                    f"Brier {report.brier_score:.3f} is {brier_drift:+.3f} above baseline " f"{baseline_brier:.3f}"
                ),
            )
        )

    # Hard accuracy floor — below this, the strategy can't profit
    # at -110 vig. Only meaningful for 2-class markets; 3-class
    # (soccer 1X2) coin-flip is 33%, so the same floor doesn't apply.
    # We still report it but as a warning rather than alert because
    # the rec engine has its own EV gate that prevents low-confidence
    # bets from being placed; the accuracy floor is mostly a signal
    # the model itself is worse than random.
    # The 52.4% accuracy floor is a 2-way (-110 moneyline) break-even.
    # It's meaningless for horse racing, where the calibration pairs are
    # per-ENTRANT win probabilities — a ~10% hit rate in a ~10-runner
    # field is normal, not "unprofitable" — so skip the floor there
    # (ECE/MCE/Brier still apply). Profitability for racing is judged by
    # realized recs ROI (the accuracy widget), not per-entrant hit rate.
    if sport != "horse_racing" and report.accuracy < thresholds.accuracy_floor and report.n >= 30:
        # Require >= 30 samples to avoid tripping on a 0/5 fluke.
        findings.append(
            DriftFinding(
                sport=sport,
                market=market,
                metric="accuracy",
                severity="alert",
                current=report.accuracy,
                threshold=thresholds.accuracy_floor,
                message=(
                    f"Accuracy {report.accuracy:.1%} < {thresholds.accuracy_floor:.1%} "
                    f"break-even — strategy unprofitable at -110 vig"
                ),
            )
        )

    return findings
