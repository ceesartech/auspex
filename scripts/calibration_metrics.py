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
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence

# Default bucket count. 10 = 10% wide bins, the standard. Higher
# counts (e.g. 20) give finer-grained reliability diagrams but
# require more data per bin to be statistically meaningful.
DEFAULT_BUCKETS = 10

# MCE is the WORST bucket's |predicted - actual|, so a bucket holding one
# prediction has an MCE of either ~0 or ~1 depending on a single coin flip.
# On 2026-09-21 the drift page carried "MCE 0.923" for the horse-racing
# consensus (ECE 0.004 — beautifully calibrated): its 0.9-1.0 bucket held
# exactly ONE entrant, which lost. Soccer BTTS "MCE 0.705" was one match in
# the 0.7-0.8 bucket. A bucket below this many predictions says nothing
# about calibration and is excluded from the maximum.
MIN_BUCKET_N_FOR_MCE = 20

# A 10-bin calibration read (ECE / MCE) on fewer predictions than this is
# noise. The 2026-09-21 page carried "ECE 0.218" for NFL totals on n=30
# (two weeks of games — buckets of 23, 5 and 2). Review simulated a
# PERFECTLY calibrated 2-way model (outcomes drawn from the model's own p,
# 4,000 trials) and asked how often ECE >= 0.10 fires anyway:
#   picks ~U(0.50, 0.75):  n=100 11.6% | n=150 3.8% | n=200 1.1% | n=400 0.1%
#   picks ~U(0.35, 0.95):  n=100 34.6% |             n=200 4.2%
# so a floor of 100 only moved the false pages from n=30 to n=100. 200 is
# where a healthy model stops paging. This floor applies ONLY to the
# bin-based metrics: accuracy has its own binomial bound and Brier its own
# guard, and an NFL slice (~65 games per 30-day window) must still be able
# to page on a real accuracy or Brier failure. Findings under the floor are
# still computed, logged and persisted — downgraded to warn, not dropped.
MIN_N_TO_PAGE = 200
BIN_BASED_METRICS = frozenset({"ece", "mce"})

# One-sided 95% z for the accuracy break-even test below.
_Z_95_ONE_SIDED = 1.64


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


def worst_bucket(buckets: Sequence[Bucket], min_bucket_n: int = MIN_BUCKET_N_FOR_MCE) -> Optional[Bucket]:
    """The bucket with the largest |predicted - actual| gap among those
    holding at least `min_bucket_n` predictions, or None when no bucket
    qualifies. Exposed separately from the MCE number so an alert can say
    WHICH bucket is wrong ("0.6-0.7, n=54: predicted 0.548, actual 0.444")
    instead of a bare figure the reader cannot act on."""
    eligible = [b for b in buckets if b.n >= min_bucket_n]
    if not eligible:
        return None
    return max(eligible, key=lambda b: abs(b.mean_predicted - b.mean_actual))


def maximum_calibration_error(buckets: Sequence[Bucket], min_bucket_n: int = MIN_BUCKET_N_FOR_MCE) -> float:
    """MCE: worst single bucket's calibration gap. Catches local
    miscalibration (e.g., a model that's well-calibrated overall but
    catastrophically wrong in its highest-confidence bucket) that
    ECE's weighted average could smooth over. The 0.80 NBA cap was
    set because the spread model's MCE was ~0.21 — caller can
    monitor whether retraining narrows this.

    Buckets with fewer than `min_bucket_n` predictions are ignored (see
    MIN_BUCKET_N_FOR_MCE for why); pass min_bucket_n=0 to get the raw
    all-buckets maximum. Returns 0.0 when no bucket qualifies — there is
    then no evidence of local miscalibration, which is different from
    evidence of good calibration; callers that care should also look at
    the population-weighted ECE."""
    worst = worst_bucket(buckets, min_bucket_n)
    if worst is None:
        return 0.0
    return abs(worst.mean_predicted - worst.mean_actual)


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
    # ONLY applied when detect_drift is told n_classes == 2: a 3-way
    # 1X2 pick at 48% or a 13-way correct-score pick at 12% is doing
    # fine, and the 2026-09-21 page reported both as "unprofitable".
    accuracy_floor: float = 0.524  # -110 break-even
    # Below this many graded predictions a slice's findings are still
    # computed and logged but downgraded from alert to warn (not paged).
    min_n_to_page: int = MIN_N_TO_PAGE
    # Buckets smaller than this do not count toward MCE.
    min_bucket_n_for_mce: int = MIN_BUCKET_N_FOR_MCE


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
    # Graded predictions behind the finding. Every rendering of a finding
    # MUST show it: the 2026-09-21 page listed ten bare numbers with no
    # sport, market or n, and eight of them were single-bucket or
    # small-sample noise that the n alone would have exposed. Defaults to
    # 0 so constructors that predate the field (the constant-prior canary)
    # keep working; 0 renders as "n=?" downstream.
    n: int = 0
    # Which model produced the slice. Two slices can share (sport, market)
    # — horse racing's ranker and consensus are both 'win' — and must not
    # render as one line or collapse into one dedup key.
    model_name: str = ""


def _downgrade(
    findings: List[DriftFinding], reason: str, only_metrics: Optional[frozenset] = None
) -> List[DriftFinding]:
    """Turn alert-severity findings into warns, appending WHY, so the
    information survives in the markdown report and the persisted time
    series but nobody gets paged for it. The reason is part of the message
    on purpose — a downgrade that leaves no trace is a silent failure.
    `only_metrics` restricts the downgrade to those metrics (the n floor
    applies to the bin-based ones only); None means every finding."""
    out: List[DriftFinding] = []
    for f in findings:
        if f.severity == "alert" and (only_metrics is None or f.metric in only_metrics):
            out.append(replace(f, severity="warn", message=f"{f.message} ({reason})"))
        else:
            out.append(f)
    return out


def detect_drift(
    *,
    sport: str,
    market: str,
    report: CalibrationReport,
    thresholds: DriftThresholds,
    baseline_brier: float = 0.25,
    n_classes: int = 2,
    stream_gated: bool = False,
    model_name: str = "",
) -> List[DriftFinding]:
    """Compare a CalibrationReport against thresholds, return a list
    of DriftFindings (empty if everything passes). Each finding is
    independent — a single market can fail multiple metrics at once
    and we report all of them so the operator sees the full picture
    rather than just the first violation.

    `n_classes` is how many outcomes the market has (2 for moneyline /
    spread / total / BTTS, 3 for 1X2, 13 for correct score, ...). The
    accuracy floor is a 2-way break-even and is applied ONLY when it is 2.
    `stream_gated` says the recommendation stream for this market is
    switched off in scripts/rec_gating.py; findings on a gated stream are
    real (that is usually WHY it is gated) but must not page every hour —
    they are downgraded to warn so the model can still be measured back to
    life from the persisted time series.

    Two more downgrades happen at the end, both leaving their reason in
    the message: a slice with fewer than thresholds.min_n_to_page graded
    predictions, and a gated stream."""
    findings: List[DriftFinding] = []

    n = report.n

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
                n=n,
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
                n=n,
            )
        )

    # MCE — recomputed here over qualifying buckets only, so a report
    # built with the raw all-buckets maximum (or by an older caller) is
    # judged on the same footing. Naming the bucket is what makes the
    # number actionable.
    worst = worst_bucket(report.buckets, thresholds.min_bucket_n_for_mce) if report.buckets else None
    if worst is not None:
        mce = abs(worst.mean_predicted - worst.mean_actual)
    elif report.buckets:
        # Buckets exist but none reaches the size floor: that is NO evidence
        # of local miscalibration, not evidence of it. Falling back to the
        # stored raw maximum here would re-admit the single-entrant page.
        mce = 0.0
    else:
        # No bucket detail at all (a hand-built report) — judge the stored
        # number as-is; there is nothing to filter.
        mce = report.mce
    where = (
        f" [bucket {worst.lower:.1f}-{worst.upper:.1f}, n={worst.n}: "
        f"predicted {worst.mean_predicted:.3f}, actual {worst.mean_actual:.3f}]"
        if worst is not None
        else ""
    )
    if mce >= thresholds.mce_alert:
        findings.append(
            DriftFinding(
                sport=sport,
                market=market,
                metric="mce",
                severity="alert",
                current=mce,
                threshold=thresholds.mce_alert,
                message=f"MCE {mce:.3f} >= {thresholds.mce_alert:.3f} — worst-bucket gap too wide{where}",
                n=n,
            )
        )
    elif mce >= thresholds.mce_warn:
        findings.append(
            DriftFinding(
                sport=sport,
                market=market,
                metric="mce",
                severity="warn",
                current=mce,
                threshold=thresholds.mce_warn,
                message=f"MCE {mce:.3f} >= {thresholds.mce_warn:.3f} — worst bucket drifting{where}",
                n=n,
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
                n=n,
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
                n=n,
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
    # The comment above has said "3-class doesn't apply" since this was
    # written, but the code only ever excluded horse racing — so the 2026-09-21
    # page reported soccer 1X2 at 48% and correct score at 12% as
    # "unprofitable". n_classes now enforces it.
    # A raw point estimate would page a genuinely profitable model on a
    # large fraction of days (review: a true-55% model reads below 52.4% on
    # 31% of draws at n=100, 21% at n=200, 14% at n=400). So the ALERT needs
    # the one-sided 95% upper bound to sit below break-even — "even giving
    # the model the benefit of the doubt it is unprofitable" — and a point
    # estimate below the floor whose bound is not is a warn.
    if n_classes == 2 and sport != "horse_racing" and report.accuracy < thresholds.accuracy_floor and report.n >= 30:
        acc = report.accuracy
        se = math.sqrt(max(acc * (1.0 - acc), 1e-12) / report.n)
        upper = acc + _Z_95_ONE_SIDED * se
        confident = upper < thresholds.accuracy_floor
        findings.append(
            DriftFinding(
                sport=sport,
                market=market,
                metric="accuracy",
                severity="alert" if confident else "warn",
                current=acc,
                threshold=thresholds.accuracy_floor,
                message=(
                    f"Accuracy {acc:.1%} < {thresholds.accuracy_floor:.1%} break-even "
                    f"(95% upper bound {upper:.1%}) — "
                    + (
                        "strategy unprofitable at -110 vig even at the top of its confidence interval"
                        if confident
                        else "below break-even but within noise at this n"
                    )
                ),
                n=n,
            )
        )

    findings = [replace(f, model_name=model_name) for f in findings]

    # Downgrades, each leaving its reason in the message. A gated stream
    # is checked first so a small gated slice names the more useful reason.
    # The n floor is scoped to the bin-based metrics (see MIN_N_TO_PAGE).
    if stream_gated:
        findings = _downgrade(findings, "recommendation stream is gated off — logged, not paged")
    if report.n < thresholds.min_n_to_page:
        findings = _downgrade(
            findings,
            f"n={report.n} < {thresholds.min_n_to_page} for a binned read — logged, not paged",
            only_metrics=BIN_BASED_METRICS,
        )

    return findings
