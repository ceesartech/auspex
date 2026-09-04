"""Unit tests for the soccer Dixon-Coles refit A/B harness.

DB-free and ml-models-free: everything here exercises the PURE pieces
that decide whether the experiment is honest —

  * monthly folds never leak an eval month into its own fit,
  * the paired SE is built from per-match differences, not from two
    independent means,
  * the per-league base-rate comparator is itself walk-forward (fit only
    on matches before the eval month), not an in-sample cheat,
  * the ship verdict flips correctly around the -0.005 gate AND names the
    baseline it judged against, because the verdict differs by baseline.

The model-fitting and DB paths are deliberately not covered here — the
Validate phase runs the real harness against prod.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ab = _load("ab_soccer_dixon_coles_refit", "ab_soccer_dixon_coles_refit.py")


def _frame(rows):
    """Minimal match frame: (date, league, over_2_5, outcome_1x2)."""
    return pd.DataFrame(
        [
            {
                "match_date": pd.Timestamp(d, tz="UTC"),
                "league": lg,
                "over_2_5": over,
                "btts": over,
                "outcome_1x2": outcome,
            }
            for d, lg, over, outcome in rows
        ]
    )


# ── Folds ────────────────────────────────────────────────────────────


class TestFolds:
    def test_one_fold_per_month_inclusive_of_end_day(self):
        folds = ab.build_folds("2025-08-01", "2026-08-31")
        assert [f.label for f in folds] == [
            "2025-08",
            "2025-09",
            "2025-10",
            "2025-11",
            "2025-12",
            "2026-01",
            "2026-02",
            "2026-03",
            "2026-04",
            "2026-05",
            "2026-06",
            "2026-07",
            "2026-08",
        ]
        # The last day of --end must be inside the last fold's window.
        last = folds[-1]
        assert last.eval_start == pd.Timestamp("2026-08-01", tz="UTC")
        assert last.eval_end == pd.Timestamp("2026-09-01", tz="UTC")

    def test_year_rollover(self):
        folds = ab.build_folds("2025-12-15", "2026-01-10")
        assert [f.label for f in folds] == ["2025-12", "2026-01"]
        # First fold is clipped to the requested start...
        assert folds[0].eval_start == pd.Timestamp("2025-12-15", tz="UTC")
        # ...but the FIT cutoff stays at the month boundary, which is
        # earlier, so clipping can never open a leak.
        assert folds[0].fit_cutoff == pd.Timestamp("2025-12-01", tz="UTC")
        assert folds[-1].eval_end == pd.Timestamp("2026-01-11", tz="UTC")

    def test_bad_span_raises(self):
        with pytest.raises(ValueError):
            ab.build_folds("2026-01-10", "2026-01-09")

    def test_fit_never_contains_the_eval_month(self):
        rows = []
        for month in range(1, 13):
            for day in (2, 14, 27):
                rows.append((f"2025-{month:02d}-{day:02d}", "L1", month % 2, 0))
        frame = _frame(rows)

        for fold in ab.build_folds("2025-03-01", "2025-06-30"):
            fit, ev = ab.split_fold(frame, fold)
            assert len(ev) > 0
            # Nothing in the fit frame is on/after the eval month start.
            assert (fit["match_date"] < fold.fit_cutoff).all()
            # And in particular no eval row leaked into the fit frame.
            assert fit["match_date"].max() < ev["match_date"].min()
            assert set(fit.index).isdisjoint(set(ev.index))
            # Eval stays inside its own window.
            assert (ev["match_date"] >= fold.eval_start).all()
            assert (ev["match_date"] < fold.eval_end).all()

    def test_fit_grows_monotonically_across_folds(self):
        rows = [(f"2025-{m:02d}-10", "L1", 1, 0) for m in range(1, 13)]
        frame = _frame(rows)
        sizes = [len(ab.split_fold(frame, f)[0]) for f in ab.build_folds("2025-04-01", "2025-08-31")]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]


# ── Paired statistics ────────────────────────────────────────────────


class TestPairedDelta:
    def test_uses_per_match_differences_not_two_independent_means(self):
        # Two arms that differ by a CONSTANT -0.01 per match but each
        # have huge spread. Paired SE must be ~0; an unpaired SE built
        # from the two marginal distributions would be large.
        rng = np.random.default_rng(0)
        base = rng.uniform(0.0, 1.0, size=500)
        challenger = base - 0.01

        out = ab.paired_delta(challenger, base)
        assert out["n"] == 500
        assert out["delta"] == pytest.approx(-0.01, abs=1e-12)
        assert out["se"] == pytest.approx(0.0, abs=1e-12)

        # The naive unpaired alternative would be enormous by comparison.
        unpaired_se = float(np.sqrt(challenger.var(ddof=1) / len(challenger) + base.var(ddof=1) / len(base)))
        assert unpaired_se > 100 * 1e-6

    def test_se_matches_the_textbook_paired_formula(self):
        challenger = np.array([0.10, 0.20, 0.30, 0.40])
        baseline = np.array([0.15, 0.18, 0.40, 0.30])
        out = ab.paired_delta(challenger, baseline)
        diff = challenger - baseline
        assert out["delta"] == pytest.approx(diff.mean())
        assert out["se"] == pytest.approx(diff.std(ddof=1) / np.sqrt(len(diff)))
        assert out["t"] == pytest.approx(out["delta"] / out["se"])

    def test_drops_pairs_where_either_side_is_missing(self):
        challenger = np.array([0.1, 0.2, np.nan, 0.4])
        baseline = np.array([0.2, np.nan, 0.3, 0.5])
        out = ab.paired_delta(challenger, baseline)
        # Only indices 0 and 3 have both sides — the market baseline only
        # exists where odds exist, so n must shrink, not silently pad.
        assert out["n"] == 2
        assert out["delta"] == pytest.approx(-0.1)

    def test_empty_overlap_is_reported_not_crashed(self):
        out = ab.paired_delta([np.nan, np.nan], [0.1, 0.2])
        assert out["n"] == 0 and out["delta"] is None

    def test_shape_mismatch_is_loud(self):
        with pytest.raises(ValueError):
            ab.paired_delta([0.1, 0.2], [0.1])


class TestBrier:
    def test_binary_is_per_match(self):
        out = ab.brier_binary([0.5, 0.25], [1.0, 0.0])
        assert out.shape == (2,)
        assert out == pytest.approx([0.25, 0.0625])

    def test_multiclass_sums_over_classes(self):
        proba = np.array([[0.6, 0.3, 0.1]])
        out = ab.brier_multiclass(proba, np.array([0]))
        assert out[0] == pytest.approx(0.16 + 0.09 + 0.01)


# ── Walk-forward base-rate comparator ────────────────────────────────


class TestBaseRates:
    def test_per_league_baseline_is_walk_forward_not_in_sample(self):
        # League L1 is a 0%-overs league for two years, then flips to
        # 100% overs during the eval month. An in-sample per-league rate
        # would see the flip; the walk-forward one must not.
        rows = [(f"2024-{m:02d}-10", "L1", 0, 0) for m in range(1, 13)]
        rows += [(f"2025-{m:02d}-10", "L1", 0, 0) for m in range(1, 7)]
        rows += [("2025-07-05", "L1", 1, 0), ("2025-07-19", "L1", 1, 0)]
        frame = _frame(rows)

        fold = [f for f in ab.build_folds("2025-07-01", "2025-07-31")][0]
        fit, ev = ab.split_fold(frame, fold)
        rates = ab.fit_base_rates(fit, "over_2_5", k=200.0)

        # Truth in the eval month is 1.0; the walk-forward comparator
        # must still be pinned near 0.
        assert ev["over_2_5"].mean() == 1.0
        assert rates.global_rate == 0.0
        assert rates.by_league["L1"] == pytest.approx(0.0)
        assert rates.predict(ev["league"])[0] == pytest.approx(0.0)

        # Sanity: an in-sample fit on the FULL frame would have moved.
        in_sample = ab.fit_base_rates(frame, "over_2_5", k=200.0)
        assert in_sample.global_rate > 0.0

    def test_shrinkage_protects_thin_leagues(self):
        # A big league at 0.60 sets the global rate; a 4-match league at
        # 1.00 must be pulled almost all the way back to it.
        rows = [(f"2025-01-{d:02d}", "BIG", 1 if d % 10 < 6 else 0, 0) for d in range(1, 31)]
        rows += [(f"2025-02-{d:02d}", "THIN", 1, 0) for d in range(1, 5)]
        frame = _frame(rows)
        rates = ab.fit_base_rates(frame, "over_2_5", k=200.0)
        assert rates.by_league["THIN"] < rates.global_rate + 0.02
        assert rates.by_league["THIN"] > rates.global_rate

    def test_unknown_league_falls_back_to_the_global_rate(self):
        frame = _frame([(f"2025-01-{d:02d}", "L1", 1, 0) for d in range(1, 21)])
        rates = ab.fit_base_rates(frame, "over_2_5", k=200.0)
        assert rates.predict(["NEVER_SEEN"])[0] == pytest.approx(rates.global_rate)

    def test_empty_fit_frame_is_safe(self):
        rates = ab.fit_base_rates(_frame([]).assign(over_2_5=[]), "over_2_5")
        assert rates.n_fit == 0 and rates.global_rate == 0.5

    def test_class_base_rates_are_walk_forward_and_shrunk(self):
        rows = [(f"2025-0{m}-10", "L1", 0, 0) for m in (1, 2, 3, 4, 5, 6)]
        rows += [("2025-07-10", "L1", 0, 2), ("2025-07-20", "L1", 0, 2)]
        frame = _frame(rows)
        fold = ab.build_folds("2025-07-01", "2025-07-31")[0]
        fit, _ev = ab.split_fold(frame, fold)
        rates = ab.fit_class_base_rates(fit, "outcome_1x2", 3, k=200.0)
        probs = ab.predict_class_base_rates(rates, ["L1"])
        assert probs.shape == (1, 3)
        assert probs[0].sum() == pytest.approx(1.0)
        # Away wins never happened before July -> class 2 stays at zero.
        assert probs[0][2] == pytest.approx(0.0)
        assert probs[0][0] == pytest.approx(1.0)


# ── Ship verdict ─────────────────────────────────────────────────────


def _clustered(delta, se, baseline, se_cluster=None, n_folds=13):
    """ship_verdict with a clustered SE supplied. Every real call has one —
    the harness always scores >= 2 folds — so the isolated unit tests must
    supply one too, or they would only ever exercise the no-cluster branch."""
    return ab.ship_verdict(
        delta,
        se,
        baseline,
        delta_cluster=delta,
        se_cluster=(se if se_cluster is None else se_cluster),
        n_folds=n_folds,
    )


class TestShipVerdict:
    def test_flips_around_the_threshold(self):
        just_over = _clustered(-0.0051, 0.0005, "base_league")
        just_under = _clustered(-0.0049, 0.0005, "base_league")
        assert just_over["ship"] is True
        assert just_under["ship"] is False
        assert "-0.005" in just_under["reason"] or "-0.0050" in just_under["reason"]

    def test_exactly_at_the_threshold_ships(self):
        assert _clustered(-0.005, 0.0001, "served")["ship"] is True

    def test_big_delta_inside_the_noise_does_not_ship(self):
        # This is the trap that killed two past "wins": a delta that
        # clears -0.005 but whose paired 95% upper bound is positive.
        v = _clustered(-0.006, 0.004, "served")
        assert v["ship"] is False
        assert "upper bound" in v["reason"]

    def test_a_win_that_only_survives_the_per_match_se_does_not_ship(self):
        """The reason the clustered SE exists. Per-match SE 0.0006 makes
        `delta + 2*se < 0` automatic for anything clearing -0.005, so the
        noise condition would be a restatement of the threshold. A between-
        fold spread big enough to swallow the delta must still block it."""
        v = ab.ship_verdict(-0.0060, 0.0006, "base_global", delta_cluster=-0.0060, se_cluster=0.0040, n_folds=13)
        assert v["ship"] is False
        assert "clustered" in v["reason"]
        assert v["upper_95"] < 0  # the per-match bound passed on its own
        assert v["upper_95_cluster"] > 0

    def test_no_clustered_se_is_not_a_pass(self):
        v = ab.ship_verdict(-0.02, 0.0001, "served")
        assert v["ship"] is False
        assert "fold-clustered" in v["reason"]

    def test_positive_delta_never_ships(self):
        assert _clustered(0.02, 0.0001, "market")["ship"] is False

    def test_verdict_names_the_baseline_it_used(self):
        for baseline in ab.BASELINE_KEYS:
            v = _clustered(-0.02, 0.001, baseline)
            assert v["baseline"] == baseline
            assert f"vs {baseline}" in v["line"]

    def test_same_delta_can_ship_against_one_baseline_and_not_another(self):
        # The whole reason the verdict must name its baseline.
        vs_legacy = _clustered(-0.0086, 0.0007, "refit_legacy")
        vs_league = _clustered(-0.0025, 0.0006, "base_league")
        assert vs_legacy["ship"] is True
        assert vs_league["ship"] is False
        assert vs_legacy["line"] != vs_league["line"]

    def test_non_finite_inputs_do_not_ship(self):
        assert ab.ship_verdict(None, None, "served")["ship"] is False
        assert ab.ship_verdict(float("nan"), 0.001, "served")["ship"] is False


# ── De-vig + blend swap ──────────────────────────────────────────────


class TestDevig:
    def test_two_way_removes_the_overround(self):
        p = ab.devig([1.90, 1.90])
        assert p is not None
        assert p.sum() == pytest.approx(1.0)
        assert p[0] == pytest.approx(0.5)

    def test_three_way_normalises(self):
        p = ab.devig([2.50, 3.40, 2.90])
        assert p is not None and p.sum() == pytest.approx(1.0)

    def test_rejects_impossible_prices(self):
        assert ab.devig([1.90, 0.5]) is None
        assert ab.devig([]) is None
        assert ab.devig([np.nan, 1.9]) is None

    def test_consensus_devigs_each_book_before_taking_the_median(self):
        # A wide-margin book (both sides short) must not drag the
        # consensus, because it is de-vigged before the median.
        quotes = [[2.00, 2.00], [2.10, 1.90], [1.70, 1.70]]
        q = ab.consensus_devigged(quotes)
        assert q is not None and q.probs.sum() == pytest.approx(1.0)
        assert q.probs[0] == pytest.approx(0.5, abs=0.03)

    def test_consensus_with_no_usable_book_is_none(self):
        assert ab.consensus_devigged([[0.9, 0.9]]) is None
        assert ab.consensus_devigged([]) is None

    def test_consensus_carries_the_depth_it_was_built_from(self):
        """A 'ceiling' built from one book at a 17% margin is not a
        ceiling. The depth has to survive to the report, or a reader
        assumes it."""
        q = ab.consensus_devigged([[2.00, 2.00], [2.10, 1.90]])
        assert q is not None
        assert q.n_books == 2
        assert q.book_sum_median == pytest.approx(1.0, abs=0.01)
        thin = ab.consensus_devigged([[1.70, 1.70]])
        assert thin is not None and thin.n_books == 1
        assert thin.book_sum_median == pytest.approx(2 / 1.7, abs=1e-6)  # ~17.6% overround

    def test_min_books_drops_a_one_book_consensus(self):
        assert ab.consensus_devigged([[2.00, 2.00]], min_books=2) is None
        assert ab.consensus_devigged([[2.00, 2.00], [1.95, 2.05]], min_books=2) is not None

    def test_market_depth_summary_reports_coverage_and_overround(self):
        quotes = {
            "a": ab.consensus_devigged([[2.00, 2.00], [2.10, 1.90], [1.95, 2.05]]),
            "b": ab.consensus_devigged([[1.80, 2.20], [1.85, 2.15]]),
        }
        eval_ids = ["a", "b"] + [f"x{i}" for i in range(98)]
        d = ab.market_depth_summary(quotes, eval_ids)
        assert d["coverage"] == pytest.approx(0.02)
        assert d["n_matches"] == 2
        assert d["books_min"] == 2 and d["books_max"] == 3
        assert d["overround_median"] is not None

    def test_depth_coverage_counts_only_rows_that_were_scored(self):
        """A quote for a match that never entered the eval set is not
        coverage — measuring against the odds query's own output could
        print a coverage above 100%."""
        quotes = {"a": ab.consensus_devigged([[2.00, 2.00], [1.95, 2.05]])}
        d = ab.market_depth_summary(quotes, ["b", "c"])
        assert d["coverage"] == 0.0 and d["n_matches"] == 0
        assert ab.market_depth_summary(quotes, [])["coverage"] is None


class TestBlendSwap:
    def test_identity_swap_is_a_no_op(self):
        ens = np.array([[0.45, 0.28, 0.27]])
        member = np.array([[0.5, 0.2, 0.3]])
        out = ab.blend_swap_1x2(ens, member, member, 0.0455)
        assert out[0] == pytest.approx(ens[0])

    def test_effect_scales_with_the_member_weight(self):
        ens = np.array([[0.45, 0.28, 0.27]])
        old = np.array([[0.50, 0.20, 0.30]])
        new = np.array([[0.30, 0.30, 0.40]])
        small = ab.blend_swap_1x2(ens, old, new, 0.0455)
        big = ab.blend_swap_1x2(ens, old, new, 0.50)
        assert abs(small[0][0] - ens[0][0]) < abs(big[0][0] - ens[0][0])
        # A 4.55% member can only move the blend a few points.
        assert abs(small[0][0] - ens[0][0]) < 0.02

    def test_output_stays_a_probability_vector(self):
        ens = np.array([[0.05, 0.05, 0.90]])
        old = np.array([[0.9, 0.05, 0.05]])
        new = np.array([[0.02, 0.02, 0.96]])
        out = ab.blend_swap_1x2(ens, old, new, 1.0)
        assert out.sum(axis=1)[0] == pytest.approx(1.0)
        assert (out >= 0).all()


# ── Summaries + CLI ──────────────────────────────────────────────────


class TestSummarise:
    def test_reports_bias_with_its_sign(self):
        probs = np.array([0.60, 0.60, 0.60, 0.60])
        truth = np.array([1.0, 0.0, 0.0, 0.0])
        s = ab.summarise(ab.brier_binary(probs, truth), probs, truth)
        assert s["n"] == 4
        assert s["mean_predicted"] == pytest.approx(0.60)
        assert s["realised"] == pytest.approx(0.25)
        # Long overs -> positive bias. A sign flip must show up as a
        # negative number here, not as a smaller magnitude.
        assert s["bias"] == pytest.approx(0.35)

    def test_short_side_bias_is_negative(self):
        probs = np.full(4, 0.10)
        truth = np.array([1.0, 1.0, 0.0, 1.0])
        s = ab.summarise(ab.brier_binary(probs, truth), probs, truth)
        assert s["bias"] < 0

    def test_all_missing_is_reported_as_empty(self):
        s = ab.summarise(np.array([np.nan]), np.array([np.nan]), np.array([1.0]))
        assert s["n"] == 0 and s["brier"] is None


class TestFoldClustering:
    """The challenger is one fitted model per fold, not one per match."""

    def test_clustered_se_is_reported_alongside_the_per_match_se(self):
        rng = np.random.default_rng(0)
        folds = np.repeat(np.arange(13), 700)
        # A per-fold offset shared by every match in that fold: exactly the
        # correlation the per-match SE assumes away.
        offsets = rng.normal(0.0, 0.004, size=13)[folds]
        challenger = 0.25 + offsets + rng.normal(0, 0.01, size=folds.size)
        baseline = np.full(folds.size, 0.25)
        out = ab.paired_delta(challenger, baseline, folds=folds)
        assert out["n_folds"] == 13
        # The clustered SE must be materially LARGER than the per-match one
        # here — that is the whole point of computing it.
        assert out["se_cluster"] > 3 * out["se"]

    def test_without_folds_the_clustered_fields_are_absent_not_zero(self):
        out = ab.paired_delta([0.1, 0.2, 0.3], [0.2, 0.2, 0.2])
        assert out["se_cluster"] is None and out["n_folds"] is None

    def test_a_single_fold_cannot_estimate_between_fold_spread(self):
        out = ab.paired_delta([0.1, 0.2], [0.2, 0.2], folds=[0, 0])
        assert out["n_folds"] == 1
        assert out["se_cluster"] is None  # not 0.0, which would read as certainty


class TestStratifiedByCoverage:
    def test_verdict_can_be_read_on_the_both_known_stratum_alone(self):
        """refit-vs-served on the full population mixes a fitting change
        with a corpus/coverage change. The both-known rows are the ones
        where both arms are predicting the same kind of object."""
        known = np.array([2.0] * 50 + [0.0] * 50)
        folds = np.array([0.0] * 25 + [1.0] * 25 + [0.0] * 25 + [1.0] * 25)
        scores = {
            # Identical on both-known rows; the whole gap is on the rows
            # where the served arm knows neither team.
            "served": np.concatenate([np.full(50, 0.24), np.full(50, 0.30)]),
            "refit": np.concatenate([np.full(50, 0.24), np.full(50, 0.24)]),
            "base_global": np.full(100, 0.25),
        }
        block, lines = ab.stratified_deltas(scores, ["served", "refit"], known, folds)
        assert block["both_known"]["n"] == 50
        assert block["neither_known"]["n"] == 50
        # No difference at all where both teams are known...
        assert block["both_known"]["deltas"]["refit"]["served"]["delta"] == pytest.approx(0.0)
        # ...and the entire headline gap lives on the uncovered rows.
        assert block["neither_known"]["deltas"]["refit"]["served"]["delta"] == pytest.approx(-0.06)
        assert any("both_known" in line for line in lines)

    def test_empty_stratum_is_reported_not_crashed(self):
        known = np.array([2.0, 2.0])
        scores = {"served": np.array([0.2, 0.2]), "refit": np.array([0.2, 0.2])}
        block, lines = ab.stratified_deltas(scores, ["served", "refit"], known, np.array([0.0, 1.0]))
        assert block["neither_known"]["n"] == 0
        assert any("no rows" in line for line in lines)


class TestDecayMatchedBaseRates:
    def _frame_with_dates(self):
        """Old matches that never go over, recent ones that always do — so
        the weighting choice moves the constant by a lot."""
        rows = []
        for i in range(200):
            rows.append(("2020-01-01", "L1", 0))
        for i in range(200):
            rows.append(("2026-01-01", "L1", 1))
        return pd.DataFrame(
            [{"match_date": pd.Timestamp(d, tz="UTC"), "league": lg, "over_2_5": o} for d, lg, o in rows]
        )

    def test_recency_weighting_moves_the_constant_comparator(self):
        frame = self._frame_with_dates()
        ref = pd.Timestamp("2026-02-01", tz="UTC")
        flat = ab.fit_base_rates(frame, "over_2_5", k=0.0)
        w = ab.decay_weights(frame, ref, 0.00095)
        decayed = ab.fit_base_rates(frame, "over_2_5", k=0.0, weights=w, decay=0.00095)
        assert flat.global_rate == pytest.approx(0.5)
        # The recent (all-over) half dominates once it is weighted.
        assert decayed.global_rate > 0.85
        assert decayed.effective_n < flat.effective_n

    def test_zero_decay_reproduces_the_unweighted_comparator(self):
        frame = self._frame_with_dates()
        w = ab.decay_weights(frame, pd.Timestamp("2026-02-01", tz="UTC"), 0.0)
        assert np.all(w == 1.0)
        assert ab.fit_base_rates(frame, "over_2_5", k=0.0, weights=w).global_rate == pytest.approx(0.5)

    def test_class_base_rates_take_the_same_weighting(self):
        frame = self._frame_with_dates().assign(outcome_1x2=lambda d: d["over_2_5"].astype(int) * 2)
        ref = pd.Timestamp("2026-02-01", tz="UTC")
        w = ab.decay_weights(frame, ref, 0.00095)
        flat = ab.fit_class_base_rates(frame, "outcome_1x2", 3, k=0.0)
        decayed = ab.fit_class_base_rates(frame, "outcome_1x2", 3, k=0.0, weights=w, decay=0.00095)
        assert flat["global"][2] == pytest.approx(0.5)
        assert decayed["global"][2] > 0.85

    def test_misaligned_weights_are_loud(self):
        frame = self._frame_with_dates()
        with pytest.raises(ValueError):
            ab.fit_base_rates(frame, "over_2_5", weights=np.ones(3))


class TestCli:
    def test_defaults_match_the_documented_protocol(self):
        args = ab.parse_args([])
        assert args.start == "2025-08-01"
        assert args.end == "2026-08-31"
        # 730-day half-life, the sweep winner.
        assert args.decay == pytest.approx(0.00095)
        assert np.log(2) / args.decay == pytest.approx(730, abs=5)
        assert args.reg == 20.0
        assert args.max_goals == 10
        assert args.limit == 0

    def test_missing_database_url_fails_loudly(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert ab.main(["--database-url", ""]) == 2

    def test_unknown_config_is_rejected(self):
        args = ab.parse_args(["--database-url", "postgres://x", "--configs", "nope"])
        with pytest.raises(ValueError):
            ab.run(args)

    def test_known_configs_include_one_ablation_per_defect(self):
        assert set(ab.CONFIG_SPECS) >= {
            "served",
            "refit",
            "refit_legacy",
            "refit_double_norm",
            "refit_no_decay",
            "refit_no_reg",
            "refit_global_baseline",
            "refit_max_goals_6",
        }
        # 'served' means "the on-disk artifact", not "a refit with default
        # hyperparameters" — the distinction is the whole experiment.
        assert ab.CONFIG_SPECS["served"] is None
        # 'refit' carries no overrides of its own: it takes the CLI values
        # on top of DIXON_COLES_CONFIG, so the harness always measures
        # whatever is about to ship.
        assert ab.CONFIG_SPECS["refit"] == {}
        assert ab.CONFIG_SPECS["refit_no_decay"]["time_decay"] == 0.0
        assert ab.CONFIG_SPECS["refit_no_reg"]["regularization"] == 0.0
        assert ab.CONFIG_SPECS["refit_global_baseline"]["per_league_baselines"] is False
        assert ab.CONFIG_SPECS["refit_max_goals_6"]["max_goals"] == 6

    def test_there_is_an_arm_that_isolates_the_scale_gauge(self):
        """Defect #1 (attack AND defense both normalised) is the headline
        claim. Without an arm that turns ONLY that back on, the only
        contrast carrying it is refit-vs-served, which also differs by
        corpus size and team coverage — so it would be attributing a
        retrain to a fitting fix."""
        assert ab.CONFIG_SPECS["refit_double_norm"] == {"normalise_defense": True}
        # And a full legacy arm: same folds, same corpus, old fitter.
        legacy = ab.CONFIG_SPECS["refit_legacy"]
        assert legacy["normalise_defense"] is True
        assert legacy["per_league_baselines"] is False
        assert legacy["time_decay"] == 0.0
        assert legacy["regularization"] == 0.001
        assert legacy["max_goals"] == 6
        # It runs by default, or the attribution is unavailable in practice.
        assert "refit_legacy" in ab.DEFAULT_CONFIGS.split(",")
        assert "refit_legacy" in ab.BASELINE_KEYS

    def test_base_rate_decay_defaults_to_the_challengers_own_decay(self):
        """The constant comparator must not be denied the recency weighting
        the model is granted — that asymmetry is worth ~11% of the ship gate."""
        assert ab.parse_args([]).base_rate_decay is None  # None => follow --decay
        assert ab.parse_args(["--base-rate-decay", "0"]).base_rate_decay == 0.0
        # Both weightings are scored, so the verdict is never stated only
        # against the handicapped one.
        assert "base_global_dw" in ab.BASELINE_KEYS
        assert "base_league_dw" in ab.BASELINE_KEYS

    def test_market_reference_requires_more_than_one_book_by_default(self):
        assert ab.parse_args([]).min_books == 2

    def test_league_shrinkage_and_base_rate_k_are_separate_knobs(self):
        # One shrinks the MODEL's fitted per-league scoring baselines, the
        # other shrinks the per-league BASE-RATE comparator. Conflating
        # them would let a knob tune the model and its own judge together.
        args = ab.parse_args(["--league-shrinkage", "50", "--shrink-k", "400"])
        assert args.league_shrinkage == 50.0
        assert args.shrink_k == 400.0

    def test_gate_constants_match_the_audit(self):
        assert ab.SHIP_THRESHOLD == -0.005
        assert ab.REFERENCE_NOISE_FLOOR_SE == 0.009
        assert ab.BASELINE_KEYS == (
            "served",
            "refit_legacy",
            "base_global",
            "base_global_dw",
            "base_league",
            "base_league_dw",
            "market",
        )


class TestReportFormatting:
    def test_section_prints_every_arm_and_names_the_verdict_baseline(self):
        arms = {
            "served": {"n": 100, "brier": 0.2526, "mean_predicted": 0.5672, "realised": 0.5236, "bias": 0.0436},
            "refit": {"n": 100, "brier": 0.2440, "mean_predicted": 0.4964, "realised": 0.5236, "bias": -0.0272},
            "base_league": {"n": 100, "brier": 0.2465, "mean_predicted": 0.52, "realised": 0.5236, "bias": -0.0036},
            "market": {"n": 60, "brier": 0.2432, "mean_predicted": 0.53, "realised": 0.5236, "bias": 0.0064},
        }
        deltas = {
            "refit": {
                "served": dict(
                    ab.paired_delta([0.1] * 100, [0.2] * 100), verdict=_clustered(-0.0086, 0.0007, "served")
                ),
                "base_league": dict(
                    ab.paired_delta([0.1] * 100, [0.11] * 100),
                    verdict=_clustered(-0.0025, 0.0006, "base_league"),
                ),
            }
        }
        lines = ab.format_market_section("over_2.5", True, arms, deltas)
        text = "\n".join(lines)
        # Every arm, flattering or not, appears.
        for key in arms:
            assert key in text
        assert "PRIMARY" in text
        assert "SHIP" in text and "DO-NOT-SHIP" in text
        # The market row's smaller n must survive into the table.
        assert "60" in text

    def test_section_survives_a_missing_metric(self):
        arms = {"served": {"n": 0, "brier": None, "mean_predicted": None, "realised": None, "bias": None}}
        lines = ab.format_market_section("btts_yes", False, arms, {})
        assert any("served" in line for line in lines)
