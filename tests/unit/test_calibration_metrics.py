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
        # n=400: a binned read pages only at/above MIN_N_TO_PAGE (a healthy
        # model at n=100 trips ECE>=0.10 ~12% of the time — see the constant).
        report = self._report(ece=0.15, n=400)
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
        # MCE (worst bucket terrible). MCE catches that. n=400 so the
        # binned-read floor does not downgrade it.
        report = self._report(ece=0.02, mce=0.30, n=400)
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
        # Hit rate 45% can't profit at -110 vig → alert. At n=400 its
        # one-sided 95% upper bound (~49.1%) is still under break-even; at
        # n=100 it would be ~53.2% and only warn (within noise).
        report = self._report(accuracy=0.45, n=400)
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


# ---------------------------------------------------------------------------
# The 2026-09-21 drift page: ten unlabeled findings, eight of them artifacts
# of thresholds that ignored bucket size, sample size and class count.
# Each class below reproduces one of those shapes from the real numbers.
# ---------------------------------------------------------------------------


def _b(lo, hi, n, pred, actual):
    return cm.Bucket(lower=lo, upper=hi, n=n, mean_predicted=pred, mean_actual=actual)


# The horse-racing consensus buckets exactly as persisted that day: ECE 0.005,
# and an "MCE 0.880/0.923" that was one entrant in the top bucket.
HORSE_BUCKETS = [
    _b(0.0, 0.1, 11437, 0.053, 0.051),
    _b(0.1, 0.2, 4232, 0.134, 0.137),
    _b(0.2, 0.3, 882, 0.233, 0.252),
    _b(0.3, 0.4, 112, 0.345, 0.348),
    _b(0.4, 0.5, 119, 0.442, 0.437),
    _b(0.5, 0.6, 54, 0.548, 0.444),
    _b(0.6, 0.7, 70, 0.658, 0.614),
    _b(0.7, 0.8, 33, 0.726, 0.636),
    _b(0.8, 0.9, 8, 0.833, 1.000),
    _b(0.9, 1.0, 1, 0.923, 0.000),
]


class TestMceIgnoresTinyBuckets:
    def test_raw_mce_is_the_single_entrant_bucket(self):
        assert cm.maximum_calibration_error(HORSE_BUCKETS, min_bucket_n=0) == pytest.approx(0.923)

    def test_filtered_mce_is_the_worst_bucket_with_enough_data(self):
        # 0.5-0.6 (n=54): |0.548 - 0.444| = 0.104 — below the 0.15 warn line.
        assert cm.maximum_calibration_error(HORSE_BUCKETS) == pytest.approx(0.104, abs=1e-9)
        assert cm.maximum_calibration_error(HORSE_BUCKETS) < cm.DriftThresholds().mce_warn

    def test_worst_bucket_names_the_bucket(self):
        worst = cm.worst_bucket(HORSE_BUCKETS)
        assert worst is not None
        assert (worst.lower, worst.upper, worst.n) == (0.5, 0.6, 54)

    def test_no_qualifying_bucket_means_no_evidence_not_a_page(self):
        tiny = [_b(0.7, 0.8, 1, 0.704, 0.0), _b(0.8, 0.9, 3, 0.829, 0.666)]  # the soccer BTTS tail
        assert cm.worst_bucket(tiny) is None
        assert cm.maximum_calibration_error(tiny) == 0.0

    def test_default_report_uses_the_filtered_mce(self):
        # Same buckets built from raw pairs: 25 well-calibrated rows plus ONE
        # confident miss must not produce an MCE alert.
        predicted = [0.5] * 25 + [0.95]
        actual = [1, 0] * 12 + [1] + [0]
        report = cm.calibration_report(predicted, actual)
        assert report.mce < 0.15
        assert cm.maximum_calibration_error(report.buckets, min_bucket_n=0) > 0.9


class TestDetectDriftScopesTheAccuracyFloor:
    def _report(self, *, n=1102, accuracy=0.48, buckets=None):
        return cm.CalibrationReport(
            n=n, accuracy=accuracy, brier_score=0.2, log_loss=0.5, ece=0.02, mce=0.05, buckets=buckets or []
        )

    def _accuracy_findings(self, **kw):
        findings = cm.detect_drift(
            sport=kw.pop("sport", "soccer"),
            market=kw.pop("market", "match_result"),
            report=self._report(**{k: v for k, v in kw.items() if k in ("n", "accuracy", "buckets")}),
            thresholds=cm.DriftThresholds(),
            n_classes=kw.get("n_classes", 2),
        )
        return [f for f in findings if f.metric == "accuracy"]

    def test_three_way_pick_at_48pct_is_not_unprofitable(self):
        assert self._accuracy_findings(accuracy=0.480, n_classes=3) == []

    def test_thirteen_way_correct_score_at_12pct_is_not_unprofitable(self):
        assert self._accuracy_findings(market="correct_score", accuracy=0.123, n_classes=13) == []

    def test_two_way_pick_below_break_even_still_alerts(self):
        found = self._accuracy_findings(sport="mma", market="moneyline", accuracy=0.488, n_classes=2)
        assert len(found) == 1 and found[0].severity == "alert"

    def test_horse_racing_stays_exempt_regardless_of_class_count(self):
        assert self._accuracy_findings(sport="horse_racing", market="win", accuracy=0.093, n_classes=2) == []


class TestDetectDriftDowngrades:
    def _report(self, n, ece=0.218, buckets=None):
        return cm.CalibrationReport(
            n=n, accuracy=0.567, brier_score=0.25, log_loss=0.6, ece=ece, mce=0.653, buckets=buckets or []
        )

    def test_every_finding_carries_n(self):
        findings = cm.detect_drift(
            sport="nfl", market="total", report=self._report(n=300), thresholds=cm.DriftThresholds()
        )
        assert findings and all(f.n == 300 for f in findings)

    def test_small_slice_is_downgraded_with_the_reason_in_the_message(self):
        # NFL totals on n=30: ECE 0.218 is an alert-level number on a sample
        # that cannot support a 10-bin read.
        findings = cm.detect_drift(
            sport="nfl", market="total", report=self._report(n=30), thresholds=cm.DriftThresholds()
        )
        assert findings, "the finding must still exist — it is logged and persisted"
        assert all(f.severity == "warn" for f in findings)
        assert all("n=30 < 200" in f.message and "not paged" in f.message for f in findings)

    def test_slice_at_the_floor_pages(self):
        findings = cm.detect_drift(
            sport="nfl", market="total", report=self._report(n=200), thresholds=cm.DriftThresholds()
        )
        assert any(f.severity == "alert" for f in findings)

    def test_gated_stream_is_downgraded_with_the_reason_in_the_message(self):
        findings = cm.detect_drift(
            sport="mma",
            market="moneyline",
            report=self._report(n=500),
            thresholds=cm.DriftThresholds(),
            stream_gated=True,
        )
        assert findings and all(f.severity == "warn" for f in findings)
        assert all("gated off" in f.message for f in findings)

    def test_downgrade_floor_is_a_threshold_knob(self):
        thr = cm.DriftThresholds(min_n_to_page=10)
        findings = cm.detect_drift(sport="nfl", market="total", report=self._report(n=30), thresholds=thr)
        assert any(f.severity == "alert" for f in findings)

    def test_mce_finding_names_the_bucket_and_uses_the_filtered_value(self):
        # A big bad bucket alerts and says where; a single-row 0.92 gap does not.
        buckets = [_b(0.5, 0.6, 400, 0.55, 0.52), _b(0.6, 0.7, 300, 0.65, 0.35), _b(0.9, 1.0, 1, 0.923, 0.0)]
        report = cm.CalibrationReport(
            n=701, accuracy=0.6, brier_score=0.2, log_loss=0.5, ece=0.05, mce=0.923, buckets=buckets
        )
        findings = cm.detect_drift(sport="soccer", market="btts", report=report, thresholds=cm.DriftThresholds())
        mce = [f for f in findings if f.metric == "mce"]
        assert len(mce) == 1 and mce[0].severity == "alert"
        assert mce[0].current == pytest.approx(0.30, abs=1e-9)
        assert "bucket 0.6-0.7, n=300" in mce[0].message

    def test_legacy_constructor_without_n_still_works(self):
        f = cm.DriftFinding(
            sport="soccer", market="1x2", metric="canary", severity="alert", current=1.0, threshold=5.0, message="x"
        )
        assert f.n == 0


class TestReviewFindings:
    """Pinned from the adversarial review of the first draft."""

    def _report(self, *, n, accuracy=0.6, ece=0.02, mce=0.05, buckets=None):
        return cm.CalibrationReport(
            n=n, accuracy=accuracy, brier_score=0.24, log_loss=0.6, ece=ece, mce=mce, buckets=buckets or []
        )

    def test_accuracy_below_floor_but_within_noise_is_a_warn_not_a_page(self):
        # A true-55% model reads 51.5% on a quarter of days at n=100; its
        # one-sided 95% upper bound (~59.7%) is nowhere near "unprofitable".
        findings = cm.detect_drift(
            sport="nba", market="spread", report=self._report(n=100, accuracy=0.515), thresholds=cm.DriftThresholds()
        )
        acc = [f for f in findings if f.metric == "accuracy"]
        assert len(acc) == 1 and acc[0].severity == "warn"
        assert "within noise" in acc[0].message and "upper bound" in acc[0].message

    def test_accuracy_confidently_below_floor_pages(self):
        findings = cm.detect_drift(
            sport="mma", market="moneyline", report=self._report(n=400, accuracy=0.45), thresholds=cm.DriftThresholds()
        )
        acc = [f for f in findings if f.metric == "accuracy"]
        assert len(acc) == 1 and acc[0].severity == "alert"

    def test_n_floor_does_not_silence_accuracy_on_an_nfl_sized_slice(self):
        # ~65 NFL games per 30-day window can never reach 200; a real
        # accuracy failure there must still page. Only ECE/MCE are floored.
        findings = cm.detect_drift(
            sport="nfl",
            market="total",
            report=self._report(n=65, accuracy=0.40, ece=0.218),
            thresholds=cm.DriftThresholds(),
        )
        by_metric = {f.metric: f for f in findings}
        assert by_metric["accuracy"].severity == "alert"
        assert by_metric["ece"].severity == "warn" and "binned read" in by_metric["ece"].message

    def test_all_tiny_buckets_means_no_mce_finding_even_with_a_raw_high_mce(self):
        tiny = [_b(0.7, 0.8, 1, 0.704, 0.0), _b(0.8, 0.9, 3, 0.829, 0.666)]
        findings = cm.detect_drift(
            sport="soccer",
            market="btts",
            report=self._report(n=1102, mce=0.705, buckets=tiny),
            thresholds=cm.DriftThresholds(),
        )
        assert [f for f in findings if f.metric == "mce"] == []

    def test_model_name_is_stamped_on_every_finding(self):
        findings = cm.detect_drift(
            sport="horse_racing",
            market="win",
            report=self._report(n=16948, ece=0.12),
            thresholds=cm.DriftThresholds(),
            model_name="lightgbm_ranker_v1",
        )
        assert findings and all(f.model_name == "lightgbm_ranker_v1" for f in findings)
