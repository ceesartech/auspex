"""Unit tests for calibration_metrics — pure math.

Lock the formulas against textbook expectations so a future refactor
of monitoring can't quietly change what "ECE > 0.10" means.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibration_metrics", "calibration_metrics.py")


# ── Brier ────────────────────────────────────────────────────────────


class TestBrier:
    def test_perfect_prediction_is_zero(self):
        # Predicting 1.0 on every winner + 0.0 on every loser.
        assert cm.brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0

    def test_always_fifty_is_quarter(self):
        # Predicting 0.5 on everything: each error term is 0.25.
        assert cm.brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.25

    def test_worst_case(self):
        # Predicting 1.0 on losers + 0.0 on winners → squared error 1
        # per row → mean 1.0. This is the worst-possible Brier.
        assert cm.brier_score([1.0, 0.0], [0, 1]) == 1.0

    def test_empty_returns_zero(self):
        # Defensive — empty input shouldn't crash the monitor.
        assert cm.brier_score([], []) == 0.0


# ── Log loss ─────────────────────────────────────────────────────────


class TestLogLoss:
    def test_perfect_prediction_is_zero(self):
        # log_loss is bounded below by 0; perfect 1.0 / 0.0 with eps
        # clipping gives ~0 (not exactly 0 due to eps).
        ll = cm.log_loss([1.0, 0.0, 1.0], [1, 0, 1])
        assert ll < 1e-10

    def test_always_fifty_is_ln_two(self):
        # log_loss for p=0.5 on any binary label is -log(0.5) = ln(2).
        ll = cm.log_loss([0.5, 0.5, 0.5], [1, 0, 1])
        assert ll == pytest.approx(math.log(2), abs=1e-9)

    def test_clips_zero_one_via_eps(self):
        # Degenerate 0.0 predictions (XGBoost sometimes does this)
        # must NOT blow up to infinity. The eps clip ensures finite.
        ll = cm.log_loss([0.0, 1.0], [1, 0])  # wrong both times
        assert math.isfinite(ll)
        # And the magnitude is large (≈ -log(eps)) but bounded.
        assert ll > 10

    def test_empty_returns_zero(self):
        assert cm.log_loss([], []) == 0.0


# ── Reliability buckets ──────────────────────────────────────────────


class TestReliabilityBuckets:
    def test_assigns_to_right_bucket(self):
        # 0.05 → bucket [0.0, 0.1); 0.15 → [0.1, 0.2); etc.
        buckets = cm.reliability_buckets([0.05, 0.15, 0.55], [0, 1, 1], n_buckets=10)
        # 3 distinct buckets populated, sorted by lower edge.
        assert [b.lower for b in buckets] == [0.0, 0.1, 0.5]

    def test_empty_buckets_omitted(self):
        # Bucket [0.2, 0.3) and [0.3, 0.4) etc. should NOT appear
        # in the result — they're empty.
        buckets = cm.reliability_buckets([0.05, 0.55], [0, 1], n_buckets=10)
        assert len(buckets) == 2

    def test_top_edge_lands_in_last_bucket(self):
        # Predicted prob of exactly 1.0 must NOT land out-of-range
        # (would crash or be silently dropped).
        buckets = cm.reliability_buckets([1.0], [1], n_buckets=10)
        assert len(buckets) == 1
        assert buckets[0].lower == 0.9
        assert buckets[0].upper == 1.0

    def test_per_bucket_means(self):
        # Bucket [0.6, 0.7) with predictions [0.6, 0.65] and actuals
        # [1, 0]: mean_pred = 0.625, mean_actual = 0.5.
        buckets = cm.reliability_buckets([0.6, 0.65], [1, 0], n_buckets=10)
        assert len(buckets) == 1
        b = buckets[0]
        assert b.n == 2
        assert b.mean_predicted == pytest.approx(0.625)
        assert b.mean_actual == pytest.approx(0.5)

    def test_clamps_out_of_range_probabilities(self):
        # Defensive: a corrupted probability outside [0, 1] should
        # clamp, not crash.
        buckets = cm.reliability_buckets([1.5, -0.3], [1, 0], n_buckets=10)
        # 1.5 clamps to 1.0 → last bucket. -0.3 clamps to 0.0 → first.
        assert len(buckets) == 2

    def test_n_buckets_validation(self):
        with pytest.raises(ValueError, match="n_buckets"):
            cm.reliability_buckets([0.5], [1], n_buckets=0)


# ── ECE / MCE ────────────────────────────────────────────────────────


class TestEce:
    def test_perfect_calibration_is_zero(self):
        # Every bucket: mean_pred == mean_actual. ECE = 0.
        buckets = [
            cm.Bucket(lower=0.0, upper=0.1, n=10, mean_predicted=0.05, mean_actual=0.05),
            cm.Bucket(lower=0.9, upper=1.0, n=10, mean_predicted=0.95, mean_actual=0.95),
        ]
        assert cm.expected_calibration_error(buckets, total_n=20) == pytest.approx(0.0)

    def test_weighted_by_bucket_size(self):
        # Big bucket with small error + small bucket with big error
        # → weighted ECE leans toward the big bucket's small error.
        buckets = [
            cm.Bucket(lower=0.0, upper=0.1, n=900, mean_predicted=0.05, mean_actual=0.10),  # gap 0.05
            cm.Bucket(lower=0.9, upper=1.0, n=100, mean_predicted=0.95, mean_actual=0.50),  # gap 0.45
        ]
        ece = cm.expected_calibration_error(buckets, total_n=1000)
        # 900/1000 * 0.05 + 100/1000 * 0.45 = 0.045 + 0.045 = 0.09
        assert ece == pytest.approx(0.09)

    def test_total_zero_returns_zero(self):
        # Empty input shouldn't divide by zero.
        assert cm.expected_calibration_error([], total_n=0) == 0.0


class TestMce:
    def test_returns_worst_bucket(self):
        # MCE = max gap, ignoring bucket size. The small-but-very-wrong
        # bucket is what we want to surface.
        buckets = [
            cm.Bucket(lower=0.0, upper=0.1, n=900, mean_predicted=0.05, mean_actual=0.10),
            cm.Bucket(lower=0.9, upper=1.0, n=100, mean_predicted=0.95, mean_actual=0.50),
        ]
        assert cm.maximum_calibration_error(buckets) == pytest.approx(0.45)

    def test_empty_returns_zero(self):
        assert cm.maximum_calibration_error([]) == 0.0


# ── Top-level report ─────────────────────────────────────────────────


class TestCalibrationReport:
    def test_returns_all_metrics(self):
        predicted = [0.7] * 100
        actual = [1] * 70 + [0] * 30  # actual hit rate exactly 70%
        report = cm.calibration_report(predicted, actual)
        assert report.n == 100
        assert report.accuracy == 0.7
        # Perfect calibration → ECE ≈ 0.
        assert report.ece < 0.01
        # Brier: (0.7 - 1)^2 * 0.7 + (0.7 - 0)^2 * 0.3 = 0.063 + 0.147 = 0.21
        assert report.brier_score == pytest.approx(0.21, abs=1e-9)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cm.calibration_report([0.5], [1, 0])

    def test_empty_returns_zero_report(self):
        report = cm.calibration_report([], [])
        assert report.n == 0
        assert report.accuracy == 0.0
        assert report.buckets == []


# ── Drift detection ──────────────────────────────────────────────────


class TestDetectDrift:
    def _report(self, *, n=100, accuracy=0.65, brier=0.2, ece=0.03, mce=0.10):
        # Build a minimal report-like object for drift tests.
        return cm.CalibrationReport(
            n=n,
            accuracy=accuracy,
            brier_score=brier,
            log_loss=0.5,
            ece=ece,
            mce=mce,
            buckets=[],
        )

    def test_clean_report_no_findings(self):
        report = self._report(ece=0.03, mce=0.10, accuracy=0.65, brier=0.20)
        findings = cm.detect_drift(
            sport="nba",
            market="moneyline",
            report=report,
            thresholds=cm.DriftThresholds(),
        )
        assert findings == []

    def test_high_ece_alerts(self):
        report = self._report(ece=0.15)
        findings = cm.detect_drift(
            sport="nba",
            market="moneyline",
            report=report,
            thresholds=cm.DriftThresholds(),
        )
        # ECE 0.15 > alert threshold 0.10 → alert.
        ece_findings = [f for f in findings if f.metric == "ece"]
        assert len(ece_findings) == 1
        assert ece_findings[0].severity == "alert"

    def test_moderate_ece_warns(self):
        report = self._report(ece=0.07)
        findings = cm.detect_drift(
            sport="nba",
            market="moneyline",
            report=report,
            thresholds=cm.DriftThresholds(),
        )
        ece_findings = [f for f in findings if f.metric == "ece"]
        assert len(ece_findings) == 1
        assert ece_findings[0].severity == "warn"

    def test_mce_alerts_independently_of_ece(self):
        # A model could have low ECE (weighted-average ok) but high
        # MCE (worst bucket terrible). MCE catches that.
        report = self._report(ece=0.02, mce=0.30)
        findings = cm.detect_drift(
            sport="nba",
            market="moneyline",
            report=report,
            thresholds=cm.DriftThresholds(),
        )
        # No ECE finding, but MCE alert fires.
        assert any(f.metric == "mce" and f.severity == "alert" for f in findings)
        assert not any(f.metric == "ece" for f in findings)

    def test_accuracy_below_breakeven_alerts(self):
        # Hit rate 45% can't profit at -110 vig → alert.
        report = self._report(accuracy=0.45, n=100)
        findings = cm.detect_drift(
            sport="nba",
            market="moneyline",
            report=report,
            thresholds=cm.DriftThresholds(),
        )
        assert any(f.metric == "accuracy" and f.severity == "alert" for f in findings)

    def test_low_n_skips_accuracy_floor(self):
        # 5 graded predictions is too few to fire the floor — a 0/5
        # streak is statistical noise, not drift.
        report = self._report(accuracy=0.0, n=5)
        findings = cm.detect_drift(
            sport="nba",
            market="moneyline",
            report=report,
            thresholds=cm.DriftThresholds(),
        )
        assert not any(f.metric == "accuracy" for f in findings)

    def test_horse_racing_skips_accuracy_floor(self):
        # The 52.4% floor is a 2-way moneyline break-even; horse-racing
        # calibration pairs are per-entrant win probs (~10% hit rate in
        # a full field is normal), so the floor must NOT fire there even
        # at a very low accuracy. ECE/MCE/Brier still apply.
        report = self._report(accuracy=0.10, n=10000)
        findings = cm.detect_drift(
            sport="horse_racing",
            market="win",
            report=report,
            thresholds=cm.DriftThresholds(),
        )
        assert not any(f.metric == "accuracy" for f in findings)

    def test_brier_drift_compared_to_baseline(self):
        # Brier of 0.26 with baseline 0.20 → drift 0.06 → alert
        # (alert threshold is 0.05). Using 0.06 not 0.05 to step
        # cleanly past the threshold; float math (0.25 - 0.20 =
        # 0.04999...) makes the knife-edge unreliable.
        report = self._report(brier=0.26)
        findings = cm.detect_drift(
            sport="nba",
            market="moneyline",
            report=report,
            thresholds=cm.DriftThresholds(),
            baseline_brier=0.20,
        )
        brier_findings = [f for f in findings if f.metric == "brier"]
        assert len(brier_findings) == 1
        assert brier_findings[0].severity == "alert"

    def test_multiple_findings_independent(self):
        # One slice can violate multiple metrics — we report all of
        # them, not just the first.
        report = self._report(ece=0.15, mce=0.30, accuracy=0.40, n=100)
        findings = cm.detect_drift(
            sport="nba",
            market="moneyline",
            report=report,
            thresholds=cm.DriftThresholds(),
        )
        metrics = {f.metric for f in findings}
        assert metrics == {"ece", "mce", "accuracy"}


class TestDriftThresholds:
    def test_default_values_locked(self):
        # Lock the defaults: these end up in the Telegram alerts so
        # changing them is a behavior change the operator must opt
        # into. The CLI override gates ad-hoc tuning per environment.
        t = cm.DriftThresholds()
        assert t.ece_warn == 0.05
        assert t.ece_alert == 0.10
        assert t.mce_warn == 0.15
        assert t.mce_alert == 0.25
        assert t.brier_drift_warn == 0.02
        assert t.brier_drift_alert == 0.05
        assert t.accuracy_floor == 0.524  # -110 vig break-even

    def test_thresholds_orderable(self):
        # warn < alert for every metric — otherwise we'd alert
        # before warning, which means warn is dead code.
        t = cm.DriftThresholds()
        assert t.ece_warn < t.ece_alert
        assert t.mce_warn < t.mce_alert
        assert t.brier_drift_warn < t.brier_drift_alert
