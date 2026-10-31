"""Unit tests for the NHL Dixon-Coles A/B harness (scripts/ab_nhl_dixon_coles.py)
— the pure pieces that decide whether the experiment is honest: seasons and the
thin-fold pooling, grading through grading_outcomes itself, the serve-path
read-out of all 8 served markets, the constants, the disqualification, the
pre-registered regression rule and decision — plus run() end to end on a
synthetic frame. The DB path is exercised by running the harness on prod."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "services" / "ml-models" / "src"))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ab = _load("ab_nhl_dixon_coles", "ab_nhl_dixon_coles.py")


def _matrix(h=3.1, a=2.8):
    from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix

    return build_dc_matrix(h, a, 0.0, max_goals=MAX_GOALS_DERIVE)


def _ev(rows):
    """Eval rows from (home_score, away_score, regulation_winner) triples."""
    return pd.DataFrame(
        {
            "home_team": [f"H{i}" for i in range(len(rows))],
            "away_team": [f"A{i}" for i in range(len(rows))],
            "home_score": [r[0] for r in rows],
            "away_score": [r[1] for r in rows],
            "regulation_winner": [r[2] for r in rows],
        }
    )


# ── seasons and folds ────────────────────────────────────────────────


class TestSeasons:
    @pytest.mark.parametrize(
        "ts, season",
        [
            ("2024-10-05", 2024),
            ("2025-06-20", 2024),
            ("2021-07-07", 2020),  # the 2020-21 final ran into July
            ("2026-08-01", 2026),
            ("2026-01-15T03:00:00+00:00", 2025),
        ],
    )
    def test_season_is_the_year_it_starts(self, ts, season):
        assert ab.season_of(ts) == season

    def test_label(self):
        assert ab.season_label(2024) == "2024-25" and ab.season_label(2099) == "2099-00"


class TestPoolThinFolds:
    def _folds(self):
        return ab.ft.build_folds("2025-03-01", "2025-10-31")  # Mar..Oct 2025

    def test_playoff_tails_join_april_and_the_offseason_is_skipped(self):
        folds = self._folds()
        counts = [240, 170, 40, 6, 0, 0, 12, 200]  # Mar Apr May Jun Jul Aug Sep Oct
        kept, pooled, skipped = ab.pool_thin_folds(folds, counts)
        assert [f.label for f in kept] == ["2025-03", "2025-04", "2025-10"]
        april = kept[1]
        # Still fitted at April 1 — strictly before every pooled row.
        assert april.fit_cutoff == pd.Timestamp("2025-04-01", tz="UTC")
        assert april.eval_end == pd.Timestamp("2025-07-01", tz="UTC")
        assert pooled == [
            {"fold": "2025-05", "n_eval": 40, "into": "2025-04"},
            {"fold": "2025-06", "n_eval": 6, "into": "2025-04"},
        ]
        # A thin September opens a new season: never pooled across the summer.
        assert skipped == [
            {"fold": "2025-07", "n_eval": 0},
            {"fold": "2025-08", "n_eval": 0},
            {"fold": "2025-09", "n_eval": 12},
        ]

    def test_a_thin_first_fold_is_skipped(self):
        folds = self._folds()[:2]
        kept, pooled, skipped = ab.pool_thin_folds(folds, [10, 170])
        assert [f.label for f in kept] == ["2025-04"] and pooled == [] and skipped[0]["fold"] == "2025-03"

    def test_never_pools_across_a_gap(self):
        folds = ab.ft.build_folds("2025-03-01", "2025-05-31")
        kept, pooled, skipped = ab.pool_thin_folds(folds, [200, 0, 30])
        assert [f.label for f in kept] == ["2025-03"]
        assert [s["fold"] for s in skipped] == ["2025-04", "2025-05"] and pooled == []

    def test_pooled_rows_are_in_the_eval_split_not_the_fit(self):
        folds = self._folds()
        kept, _, _ = ab.pool_thin_folds(folds, [240, 170, 40, 6, 0, 0, 12, 200])
        frame = pd.DataFrame(
            {"match_date": pd.to_datetime(["2025-03-30", "2025-04-02", "2025-05-20", "2025-06-10"], utc=True)}
        )
        fit, ev = ab.ft.split_fold(frame, kept[1])
        assert len(fit) == 1 and len(ev) == 3


# ── grading through grading_outcomes ─────────────────────────────────


class TestGradedVector:
    @pytest.mark.parametrize(
        "market, key, score, expected",
        [
            ("over_under", "over_5.5", (4, 2), 1.0),
            ("over_under", "over_5.5", (3, 2), 0.0),
            ("over_under", "over_4.5", (3, 2), 1.0),
            ("over_under", "over_6.5", (4, 3), 1.0),
            ("puck_line", "-1.5_home", (4, 2), 1.0),
            ("puck_line", "-1.5_home", (3, 2), 0.0),
            ("puck_line", "1.5_home", (2, 3), 1.0),  # home loses by one: covers +1.5
            ("puck_line", "1.5_home", (1, 3), 0.0),
            ("clean_sheet", "home_yes", (2, 0), 1.0),
            ("clean_sheet", "away_yes", (2, 0), 0.0),
            ("clean_sheet", "away_yes", (0, 1), 1.0),
            ("win_to_nil", "home_yes", (3, 0), 1.0),
            ("win_to_nil", "away_yes", (0, 0), 0.0),
            ("win_to_nil", "away_yes", (1, 2), 0.0),
        ],
    )
    def test_final_score_markets(self, market, key, score, expected):
        y = ab.graded_vector(market, (key,), score[0], score[1], "home")
        assert y.tolist() == [expected]

    def test_regulation_markets_read_regulation_winner_not_the_final_score(self):
        # A 3-2 OT win is a regulation TIE: draw, and 1X / X2 cover it, 12 does not.
        assert ab.graded_vector("regulation_1x2", ab.REGULATION_CLASSES, 3, 2, "tie").tolist() == [0, 1, 0]
        assert ab.graded_vector("regulation_1x2", ab.REGULATION_CLASSES, 3, 2, "draw").tolist() == [0, 1, 0]
        for key, won in (("1X", 1.0), ("12", 0.0), ("X2", 1.0)):
            assert ab.graded_vector("double_chance", (key,), 3, 2, "tie").tolist() == [won]
        assert ab.graded_vector("regulation_1x2", ab.REGULATION_CLASSES, 2, 4, "away").tolist() == [0, 0, 1]

    @pytest.mark.parametrize("missing", [None, float("nan")])
    def test_regulation_markets_without_regulation_winner_are_ungradable(self, missing):
        assert ab.graded_vector("regulation_1x2", ab.REGULATION_CLASSES, 3, 2, missing) is None
        assert ab.graded_vector("double_chance", ("12",), 3, 2, missing) is None
        assert ab.graded_vector("over_under", ("over_5.5",), 3, 2, missing).tolist() == [0.0]

    def test_total_goals_top_band_is_eight_plus(self):
        keys = next(u.keys for u in ab.UNITS if u.name == "total_goals")
        assert ab.graded_vector("total_goals", keys, 6, 3, "home").tolist() == [0] * 8 + [1]
        assert ab.graded_vector("total_goals", keys, 4, 3, "home")[7] == 1.0

    def test_correct_score_other_wins_off_the_listed_keys(self):
        probs = {"3-2": 0.1, "2-1": 0.1, "other": 0.8}
        keys = list(probs)
        assert ab.graded_vector("correct_score", keys, 3, 2, "home", probabilities=probs).tolist() == [1, 0, 0]
        assert ab.graded_vector("correct_score", keys, 7, 1, "home", probabilities=probs).tolist() == [0, 0, 1]

    def test_a_partition_without_exactly_one_winner_raises(self):
        with pytest.raises(ValueError, match="graded correct"):
            ab.graded_vector("regulation_1x2", ("home", "away"), 3, 2, "tie")

    def test_fixed_truth_shapes_and_ungradable_rows(self):
        t = ab.fixed_truth(_ev([(4, 2, "home"), (3, 2, None)]))
        assert set(t) == {u.name for u in ab.FIXED_UNITS} | {"correct_score_grid"}
        assert t["correct_score_grid"].shape == (2, 121)
        assert t["over_5.5"].shape == (2, 1) and t["over_5.5"][:, 0].tolist() == [1.0, 0.0]
        assert t["regulation_1x2"].shape == (2, 3) and np.isnan(t["regulation_1x2"][1]).all()
        assert t["regulation_1x2"][0].tolist() == [1, 0, 0]
        assert np.isnan(t["double_chance_12"][1, 0])
        assert t["total_goals"].shape == (2, 9)


# ── the serve-path read-out ──────────────────────────────────────────


class TestServedMarkets:
    def test_units_cover_exactly_what_precompute_predictions_nhl_serves(self):
        pytest.importorskip("psycopg2")
        nhl = _load("precompute_predictions_nhl", "precompute_predictions_nhl.py")
        assert (
            set(ab.SERVED_MARKETS)
            == set(nhl.HOCKEY_DERIVE_TO_PREDICTION)
            == set(nhl.HOCKEY_DERIVE_TO_PREDICTION.values())
        )
        assert {u.market for u in ab.UNITS} == set(ab.SERVED_MARKETS)

    def test_served_markets_drop_the_moneyline_and_carry_every_unit_key(self):
        m = ab.served_markets(_matrix())
        assert "moneyline" not in m and set(m) == set(ab.SERVED_MARKETS)
        for unit in ab.FIXED_UNITS:
            for key in unit.keys:
                assert key in m[unit.market], (unit.name, key)
        assert len(m["correct_score"]) == 13

    def test_readout_matches_the_derivation_and_scores_by_hand(self):
        from predictors.market_derivation import derive_hockey_markets

        P = _matrix()
        ref = derive_hockey_markets(P)
        ev = _ev([(3, 2, "tie")])
        probs = ab.readout([ab.served_markets(P)], ev, [P])
        assert probs["over_5.5"][0, 0] == pytest.approx(ref["over_under"]["over_5.5"])
        assert probs["puck_line_+1.5"][0, 0] == pytest.approx(ref["puck_line"]["1.5_home"])
        assert probs["double_chance_X2"][0, 0] == pytest.approx(ref["double_chance"]["X2"])
        assert probs["regulation_1x2"][0].tolist() == pytest.approx(
            [ref["regulation_1x2"][c] for c in ab.REGULATION_CLASSES]
        )
        # correct_score_listed truth is on the arm's own keys: one winner, 3-2 if listed, else 'other'.
        cs_keys = list(ref["correct_score"])
        y = probs["_cs_listed_truth"][0]
        assert y.sum() == 1.0
        assert cs_keys[int(y.argmax())] == ("3-2" if "3-2" in cs_keys else "other")
        truth = ab.fixed_truth(ev)
        s = ab.score_arm(probs, truth)
        assert set(s) == set(ab.UNIT_NAMES)
        assert s["over_5.5"][0] == pytest.approx((ref["over_under"]["over_5.5"] - 0.0) ** 2)
        p_reg = np.array([ref["regulation_1x2"][c] for c in ab.REGULATION_CLASSES])
        assert s["regulation_1x2"][0] == pytest.approx(float(np.sum((p_reg - [0, 1, 0]) ** 2)))
        p_cs = np.array([ref["correct_score"][k] for k in cs_keys])
        assert s["correct_score_listed"][0] == pytest.approx(float(np.sum((p_cs - y) ** 2)))
        # correct_score_grid: the whole 11x11 matrix, scored against the realised 3-2 cell.
        assert probs["correct_score_grid"].shape == (1, 121)
        assert probs["correct_score_grid"][0] == pytest.approx(P.ravel())
        onehot = np.zeros((11, 11))
        onehot[3, 2] = 1.0
        assert truth["correct_score_grid"][0].tolist() == onehot.ravel().tolist()
        assert s["correct_score_grid"][0] == pytest.approx(float(np.sum((P - onehot) ** 2)))
        assert ab.grid_log_score(probs, truth)[0] == pytest.approx(-np.log(P[3, 2]))

    def test_ungradable_rows_score_nan_for_their_units_only(self):
        ev = _ev([(3, 2, None)])
        P = _matrix()
        s = ab.score_arm(ab.readout([ab.served_markets(P)], ev, [P]), ab.fixed_truth(ev))
        assert np.isnan(s["regulation_1x2"][0]) and np.isnan(s["double_chance_1X"][0])
        assert np.isfinite(s["over_5.5"][0]) and np.isfinite(s["correct_score_listed"][0])
        assert np.isfinite(s["correct_score_grid"][0])

    def test_readout_refuses_a_matrix_off_the_shared_grid(self):
        P = _matrix()
        with pytest.raises(RuntimeError, match="shared 11x11 grid"):
            ab.readout([ab.served_markets(P)], _ev([(3, 2, "home")]), [P[:7, :7] / P[:7, :7].sum()])

    def test_model_matrices_never_pass_a_league(self):
        seen = []

        class Fake:
            rho = None  # an artifact without rho serves rho 0

            def lambdas_for_match(self, home, away, league="NOT PASSED"):
                seen.append(league)
                return 3.0, 2.7

        mats = ab.model_matrices(Fake(), pd.DataFrame({"home_team": ["A"], "away_team": ["B"], "league_id": ["L"]}))
        assert seen == ["NOT PASSED"]
        assert mats[0] == pytest.approx(_matrix(3.0, 2.7))

    def test_expected_totals(self):
        P = np.zeros((3, 3))
        P[2, 1] = P[0, 0] = 0.5
        assert ab.expected_totals([P]).tolist() == [1.5]


# ── constants ────────────────────────────────────────────────────────


class TestConstants:
    def _fit(self):
        return pd.DataFrame(
            {
                "home_score": [4, 2, 3, 1, 5, 2],
                "away_score": [1, 3, 2, 4, 0, 1],
                "regulation_winner": ["home", "away", "tie", "away", "home", None],
                "season": [2023, 2023, 2023, 2024, 2024, 2024],
            }
        )

    def test_regulation_markets_come_from_regulation_winner_rates(self):
        fit = self._fit()
        ev = pd.DataFrame({"season": [2024]})
        (m,), (P,) = ab.constant_markets(fit, ev, None, by_season=False)
        # 5 labelled rows: home 2, tie 1, away 2 (the unlabelled one is ignored).
        assert m["regulation_1x2"] == pytest.approx({"home": 0.4, "draw": 0.2, "away": 0.4})
        assert m["double_chance"] == pytest.approx({"1X": 0.6, "12": 0.8, "X2": 0.6})
        assert P.sum() == pytest.approx(1.0)
        # Final-score markets are the empirical rates: totals 5,5,5,5,5,3 -> 5/6 over 4.5, none over 5.5.
        assert m["over_under"]["over_4.5"] == pytest.approx(5 / 6)
        assert m["over_under"]["over_5.5"] == pytest.approx(0.0)
        assert m["clean_sheet"]["away_yes"] == pytest.approx(0.0)  # home never blanked
        assert m["clean_sheet"]["home_yes"] == pytest.approx(1 / 6)

    def test_per_season_shrinks_toward_the_global_rate(self):
        fit = self._fit()
        ev = pd.DataFrame({"season": [2024, 2099]})
        glob, _ = ab.constant_markets(fit, ev, None, by_season=False)
        seas, _ = ab.constant_markets(fit, ev, None, by_season=True)
        # An unseen season is the global constant; a seen one moves (a little, k=200) toward its own rate.
        assert seas[1]["over_under"]["over_4.5"] == pytest.approx(glob[1]["over_under"]["over_4.5"])
        assert seas[0]["regulation_1x2"]["away"] != pytest.approx(glob[0]["regulation_1x2"]["away"], abs=1e-6)

    def test_scorelines_beyond_the_grid_are_capped(self):
        assert ab.scoreline_index([12, 0], [1, 11], cap=10).tolist() == [10 * 11 + 1, 10]


# ── disqualification, regressions, decision ──────────────────────────


def _valid_probs(n=2):
    P = _matrix()
    return ab.readout([ab.served_markets(P)] * n, _ev([(3, 2, "home")] * n), [P] * n)


class TestDisqualification:
    def test_trainer_refusal_disqualifies_a_challenger_even_with_valid_probabilities(self):
        reason = ab.disqualification("refit", probs=_valid_probs(), refusal="the fit did not converge")
        assert reason and "would refuse" in reason and "did not converge" in reason

    def test_valid_challenger_passes(self):
        assert ab.disqualification("pinned", probs=_valid_probs()) is None

    def test_invalid_probabilities_disqualify_a_challenger(self):
        probs = _valid_probs()
        probs["regulation_1x2"][0] = [0.5, 0.5, 0.5]
        assert "regulation_1x2" in ab.disqualification("refit_frozen", probs=probs)
        probs = _valid_probs()
        probs["over_5.5"][1, 0] = np.nan
        assert "over_5.5" in ab.invalid_probabilities(probs)

    def test_served_and_constants_raise_instead(self):
        for arm in ("served", "base_global", "base_season", "base_season_dw"):
            with pytest.raises(RuntimeError, match=arm):
                ab.disqualification(arm, refusal="anything")

    def test_invalid_matrices(self):
        good = _matrix()
        assert ab.invalid_matrices([good]) is None
        assert "1 of 2" in ab.invalid_matrices([good, np.full((3, 3), np.nan)])
        assert "1 of 2" in ab.invalid_matrices([good, good * 2])


def _pair(delta, se=None, delta_cluster=None, se_cluster=None, n_folds=None):
    """A per-unit pair dict as run()'s pair() builds it, the bounds included."""
    return {
        "delta": delta,
        "se": se,
        "delta_cluster": delta_cluster,
        "se_cluster": se_cluster,
        "n_folds": n_folds,
        "lower_95_cluster_t": ab.t_lower(delta_cluster, se_cluster, n_folds),
        "upper_95_cluster_t": ab.ht.t_upper(delta_cluster, se_cluster, n_folds),
    }


class TestRegressionRule:
    """Amendment A2: a unit regresses only when SIGNIFICANTLY worse — the 95%
    lower bound of its dBrier (fold-clustered t, else per-match) above 0."""

    def test_t_lower_mirrors_t_upper(self):
        from scipy.stats import t

        assert ab.t_lower(0.003, 0.0005, 8) == pytest.approx(0.003 - t.ppf(0.975, 7) * 0.0005)
        lo, hi = ab.t_lower(0.001, 0.0004, 5), ab.ht.t_upper(0.001, 0.0004, 5)
        assert (lo + hi) / 2 == pytest.approx(0.001)
        assert ab.t_lower(0.003, 0.0005, 1) is None and ab.t_lower(None, 0.1, 8) is None

    def test_a_lower_bound_above_zero_regresses(self):
        p = _pair(0.003, se=0.0004, delta_cluster=0.003, se_cluster=0.0005, n_folds=8)
        assert p["lower_95_cluster_t"] > 0
        assert "significantly worse" in ab.regression(p)

    def test_the_smokes_over_4_5_is_not_a_regression(self):
        # The 2026-10-31 smoke: refit_frozen over_4.5 +0.0024, cluster SE 0.00099 over 3
        # folds — t-interval about [-0.0018, +0.0067]. It blocked under the first rule.
        p = _pair(0.00239, se=0.0011, delta_cluster=0.00240, se_cluster=0.00099, n_folds=3)
        assert p["lower_95_cluster_t"] == pytest.approx(-0.00186, abs=5e-5)
        assert p["upper_95_cluster_t"] > 0 and p["delta"] > 0.002
        assert ab.regression(p) is None

    def test_a_significant_regression_below_the_old_margin_still_blocks(self):
        # No fixed margin any more: +0.001 with a tight interval is significantly worse.
        p = _pair(0.001, se=0.0001, delta_cluster=0.001, se_cluster=0.0001, n_folds=8)
        assert ab.regression(p) is not None

    def test_an_improvement_never_regresses(self):
        assert ab.regression(_pair(-0.01, se=0.001, delta_cluster=-0.01, se_cluster=0.001, n_folds=8)) is None

    def test_reads_the_clustered_bound_not_the_per_match_one(self):
        # tight per-match SE (bound > 0), wide fold spread (clustered bound < 0): no regression
        p = _pair(0.002, se=0.0002, delta_cluster=0.002, se_cluster=0.002, n_folds=8)
        assert p["delta"] - 2 * p["se"] > 0 and p["lower_95_cluster_t"] < 0
        assert ab.regression(p) is None

    def test_falls_back_to_the_per_match_bound(self):
        assert ab.lower_bound(_pair(0.003, se=0.001)) == pytest.approx(0.001)
        assert ab.regression(_pair(0.003, se=0.001)) is not None
        assert ab.regression(_pair(0.003, se=0.002)) is None

    def test_an_unassessable_unit_blocks(self):
        assert "unassessable" in ab.regression(_pair(None))
        assert "no 95% bound" in ab.regression(_pair(0.004, se=float("nan")))
        assert "unassessable" in ab.regression({})


class TestBlockingRegressions:
    """The decision reads every unit except the primary and the context-only
    correct_score_listed — and correct_score_grid does block."""

    @staticmethod
    def _units(**worse):
        fine = _pair(-0.001, se=0.0005, delta_cluster=-0.001, se_cluster=0.0005, n_folds=8)
        bad = _pair(0.02, se=0.001, delta_cluster=0.02, se_cluster=0.001, n_folds=8)
        return {u: (bad if worse.get(u) else fine) for u in ab.UNIT_NAMES}

    def test_the_unit_sets(self):
        assert ab.CONTEXT_UNITS == ("correct_score_listed",)
        assert set(ab.BLOCKING_UNITS) == set(ab.UNIT_NAMES) - {"over_5.5", "correct_score_listed"}
        assert "correct_score_grid" in ab.BLOCKING_UNITS and len(ab.BLOCKING_UNITS) == 14

    def test_listed_correct_score_never_blocks(self):
        assert ab.blocking_regressions(self._units(correct_score_listed=True)) == {}

    def test_the_primary_is_left_to_rule_a(self):
        assert ab.blocking_regressions(self._units(**{"over_5.5": True})) == {}

    def test_the_grid_and_every_other_blocking_unit_block(self):
        for unit in ab.BLOCKING_UNITS:
            assert set(ab.blocking_regressions(self._units(**{unit: True}))) == {unit}

    def test_a_missing_blocking_unit_blocks(self):
        units = self._units()
        del units["total_goals"]
        assert set(ab.blocking_regressions(units)) == {"total_goals"}


class TestCorrectScoreGrid:
    """Amendment A1. The replacement unit is a proper score on one partition:
    an arm with inflated (or deflated) lambdas cannot beat the true
    distribution in expectation. Exact expectation over every cell of the grid,
    scored through readout + score_arm themselves — and the product's listed
    score, kept as context, is shown to be improper (why it may not block)."""

    TRUE = (3.15, 2.95)

    def _expected(self, scale, unit):
        from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix

        cap = MAX_GOALS_DERIVE
        p_true = build_dc_matrix(*self.TRUE, 0.0, max_goals=cap)
        Q = build_dc_matrix(self.TRUE[0] * scale, self.TRUE[1] * scale, 0.0, max_goals=cap)
        cells = [(i, j) for i in range(cap + 1) for j in range(cap + 1)]
        ev = _ev([(i, j, "home" if i > j else ("away" if i < j else "tie")) for i, j in cells])
        n = len(cells)
        probs = ab.readout([ab.served_markets(Q)] * n, ev, [Q] * n)
        s = ab.score_arm(probs, ab.fixed_truth(ev))[unit]
        return float(sum(p_true[i, j] * s[k] for k, (i, j) in enumerate(cells)))

    @pytest.mark.parametrize("scale", [1.05, 1.10, 0.95, 0.90])
    def test_a_mis_levelled_arm_does_not_beat_the_truth_on_the_grid(self, scale):
        assert self._expected(1.0, "correct_score_grid") < self._expected(scale, "correct_score_grid")

    def test_the_listed_score_rewards_inflated_lambdas(self):
        # The defect the amendment removes from the decision: 0.7067 true vs 0.6925 at +10%.
        true, inflated = self._expected(1.0, "correct_score_listed"), self._expected(1.10, "correct_score_listed")
        assert true == pytest.approx(0.7067, abs=1e-4) and inflated == pytest.approx(0.6925, abs=1e-4)
        assert inflated < true

    def test_grid_truth_is_the_graded_final_scoreline_capped_at_the_grid(self):
        t = ab.fixed_truth(_ev([(3, 2, "home"), (12, 1, "home"), (0, 11, "away")]))["correct_score_grid"]
        assert t.shape == (3, 121) and t.sum(axis=1).tolist() == [1.0, 1.0, 1.0]
        assert int(t[0].argmax()) == 3 * 11 + 2 and ab.go.correct_score_outcome(3, 2) == "3-2"
        assert int(t[1].argmax()) == 10 * 11 + 1 and int(t[2].argmax()) == 0 * 11 + 10

    def test_grid_log_score_is_inf_on_a_zero_cell_never_nan(self):
        P = np.zeros((11, 11))
        P[3, 2] = 1.0
        ev = _ev([(3, 2, "home"), (2, 3, "away")])
        ls = ab.grid_log_score(ab.readout([ab.served_markets(P)] * 2, ev, [P] * 2), ab.fixed_truth(ev))
        assert ls[0] == pytest.approx(0.0) and np.isposinf(ls[1])


class TestTeamCoverage:
    def test_known_count_and_unknown_teams(self):
        teams = {"Utah Utah Hockey Club", "Boston Bruins"}
        home, away = ["Utah Mammoth", "Boston Bruins", "Utah Mammoth"], ["Boston Bruins", "Utah Utah Hockey Club", "X"]
        assert ab.known_count(home, away, teams).tolist() == [1.0, 2.0, 0.0]
        assert ab.unknown_teams(home, away, teams) == {"Utah Mammoth": 2, "X": 1}


class TestPreRegistrationRecord:
    def test_the_amendment_is_recorded_with_its_reasons(self):
        doc = ab.__doc__
        assert "AMENDMENT 2026-10-31, made after the 3-fold smoke run (n=665) and before the full run" in doc
        for tag, reason in (("[A1]", "IMPROPER score"), ("[A2]", "regression wording"), ("[A3]", "IN-SAMPLE")):
            assert doc.count(tag) >= 2 and reason in doc  # the superseded passage and the amendment itself
        assert "RULE AS AMENDED (what decide() implements)" in doc
        # The original rule stays in the record, marked as superseded.
        assert "> +0.002 with a 95% upper" in doc and "[A2: replaced by the lower-bound rule.]" in doc
        # How much of the decision window the smoke saw, and that its unseen folds are reported (never deciding).
        assert "41% of the amended decision window's 1,603 rows (3\n  of its 8 folds)" in doc
        assert "the five folds the smoke never saw (2025-10..2026-02)" in doc


class TestDecision:
    V = {"refit_frozen": {"ship_strict": True}, "refit": {"ship_strict": False}}

    def test_ships_only_on_primary_pass_and_no_regression(self):
        d = ab.decide(self.V, {"refit_frozen": {}}, {})
        assert d["ship"] and d["arm"] == "refit_frozen"

    def test_a_regression_blocks(self):
        d = ab.decide(self.V, {"refit_frozen": {"total_goals": "dBrier +0.003 ..."}}, {})
        assert not d["ship"] and "total_goals" in d["reason"] and d["primary_pass"]

    def test_a_primary_miss_blocks(self):
        d = ab.decide({"refit_frozen": {"ship_strict": False}}, {"refit_frozen": {}}, {})
        assert not d["ship"] and "fails ship_strict" in d["reason"]

    def test_refit_never_decides_and_disqualified_frozen_cannot_ship(self):
        d = ab.decide({"refit": {"ship_strict": True}}, {"refit": {}}, {"refit_frozen": {"reason": "no converge"}})
        assert not d["ship"] and "disqualified: no converge" in d["reason"]

    def test_no_decision_fold_cannot_ship(self):
        d = ab.decide(self.V, {"refit_frozen": {}}, {}, unavailable="no kept fold starts on or after 2025-05-01")
        assert not d["ship"] and "2025-05-01" in d["reason"]


class TestArms:
    def test_pinned_is_exactly_the_trainers_config_and_refit_the_soccer_one(self):
        from predictors.model_config import DIXON_COLES_CONFIG, LEGACY_DIXON_COLES_HYPERPARAMETERS

        c = ab.arm_configs()
        assert c["pinned"] == ab.trainer.nhl_config()
        assert c["pinned"].name == c["refit"].name == "dixon_coles_nhl"
        assert c["pinned"].hyperparameters == {
            **DIXON_COLES_CONFIG.hyperparameters,
            **LEGACY_DIXON_COLES_HYPERPARAMETERS,
        }
        assert c["refit"].hyperparameters == DIXON_COLES_CONFIG.hyperparameters

    def test_decayed_arms_meet_the_decayed_constant(self):
        assert ab.matched_constant("refit") == ab.matched_constant("refit_frozen") == "base_season_dw"
        assert ab.matched_constant("served") == ab.matched_constant("pinned") == "base_season"

    def test_fit_arm_forces_rho_to_zero_and_round_trips(self, tmp_path):
        frame = nhl_frame(pd.Timestamp("2024-10-01", tz="UTC"), pd.Timestamp("2025-01-31", tz="UTC"), seed=3)
        model, result, path, refusal = ab.fit_arm(frame, ab.arm_configs()["pinned"], str(tmp_path), "p")
        assert refusal is None and model.rho == 0.0 and result["rho"] == 0.0
        assert json.loads(Path(path).read_text())["rho"] == 0.0


class TestAssembleFrame:
    def test_join_and_audit(self):
        frame = pd.DataFrame({"match_id": ["a", "b"], "home_score": [3, 2], "away_score": [2, 1]})
        reg = pd.DataFrame({"match_id": ["a", "b", "c"], "regulation_winner": ["tie", None, "home"]})
        out, audit = ab.assemble_frame(frame, reg)
        assert out["regulation_winner"].tolist()[0] == "tie"
        assert audit == {"rows": 2, "without_regulation_winner": 1, "regulation_labels": {"tie": 1}}

    def test_a_frame_row_missing_from_the_regulation_query_raises(self):
        frame = pd.DataFrame({"match_id": ["a", "zz"]})
        reg = pd.DataFrame({"match_id": ["a"], "regulation_winner": ["home"]})
        with pytest.raises(RuntimeError, match="the two queries disagree"):
            ab.assemble_frame(frame, reg)


# ── run() end to end on a synthetic frame ────────────────────────────


def nhl_frame(start, end, seed=0, per_day=3, thin=None):
    """A synthetic NHL frame shaped like NHL_DC_TRAINING_QUERY's plus
    regulation_winner. No games July-September; ``thin`` maps month -> games."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(12)]
    attack = {t: rng.uniform(0.85, 1.15) for t in teams}
    rows = []
    mid = 0
    for d in pd.date_range(start, end, freq="D"):
        if d.month in (7, 8, 9):
            continue
        n = per_day
        if thin and d.month in thin:
            n = 1 if d.day <= thin[d.month] else 0
        for _ in range(n):
            h, a = rng.choice(teams, 2, replace=False)
            hs, as_ = int(rng.poisson(3.2 * attack[h])), int(rng.poisson(2.9 * attack[a]))
            if hs == as_:
                reg = "tie"
                if rng.random() < 0.5:
                    hs += 1
                else:
                    as_ += 1
            else:
                reg = "home" if hs > as_ else "away"
            rows.append(
                dict(
                    match_id=f"m{mid}",
                    match_date=d,
                    league_id="NHL",
                    home_team=h,
                    away_team=a,
                    home_score=hs,
                    away_score=as_,
                    match_outcome=0 if hs > as_ else 1,
                    regulation_winner=reg if rng.random() > 0.02 else None,
                )
            )
            mid += 1
    return pd.DataFrame(rows)


def rename_team(frame, old, new, since):
    """The franchise rename the gate must survive: ``old`` plays as ``new`` from ``since`` on."""
    out = frame.copy()
    later = pd.to_datetime(out["match_date"], utc=True) >= pd.Timestamp(since, tz="UTC")
    for col in ("home_team", "away_team"):
        out.loc[later & (out[col] == old), col] = new
    return out


class TestSyntheticRun:
    """run() on a synthetic frame, no DB: the bookkeeping that makes the paired
    comparisons honest (pooled playoff tails, disqualified arms dropped
    everywhere, the frozen arm equal to the first DECISION fold's refit and
    scored on decision folds only, the decision read on the decision window
    only), valid JSON, and the pre-registered decision line."""

    @staticmethod
    def _setup(monkeypatch, tmp_path, decision_start="2024-11-01"):
        frame = nhl_frame(
            pd.Timestamp("2023-10-01", tz="UTC"), pd.Timestamp("2025-02-28", tz="UTC"), thin={5: 20, 6: 4}
        )
        # T11 becomes 'Utah Utah Hockey Club' after served's fit and 'Utah Mammoth'
        # after the frozen fit's cutoff (2024-11-01), as the real franchise did.
        frame = rename_team(frame, "T11", "Utah Utah Hockey Club", "2024-04-01")
        frame = rename_team(frame, "Utah Utah Hockey Club", "Utah Mammoth", "2024-12-01")
        reg = frame[["match_id", "regulation_winner"]]
        monkeypatch.setattr(
            ab, "load_frame", lambda url: ab.assemble_frame(frame.drop(columns="regulation_winner"), reg)
        )
        if decision_start is not None:
            # The synthetic frame ends 2025-02; move the decision window into it.
            monkeypatch.setattr(ab, "DECISION_START", decision_start)
        from predictors.poisson_models import DixonColesPredictor

        served = DixonColesPredictor(ab.arm_configs()["pinned"])
        served.train(frame[frame["match_date"] < "2024-03-01"][ab.FIT_COLS])
        served.rho = 0.0
        path = tmp_path / "served.bin"
        served.save(str(path))
        args = ab.parse_args(["--start", "2024-04-01", "--served-artifact", str(path), "--database-url", "x"])
        return args, frame

    def test_bookkeeping_and_decision_line(self, monkeypatch, tmp_path):
        args, frame = self._setup(monkeypatch, tmp_path)
        r = ab.run(args)
        json.dumps(ab.ht.clean_json(r), allow_nan=False)
        assert r["folds"] == ["2024-04", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02"]
        assert [p["fold"] for p in r["pooled_folds"]] == ["2024-05", "2024-06"]
        assert [s["fold"] for s in r["skipped_folds"]] == ["2024-07", "2024-08", "2024-09"]
        assert r["disqualified"] == {}
        dw = r["decision_window"]
        assert dw["folds"] == ["2024-11", "2024-12", "2025-01", "2025-02"]
        assert dw["frozen_fit_cutoff"] == "2024-11-01T00:00:00+00:00"
        md = pd.to_datetime(frame["match_date"], utc=True)
        n_decision = int(((md >= "2024-11-01") & (md < "2025-03-01")).sum())
        assert dw["n_eval"] == n_decision
        frozen_fit = next(f for f in r["fits"] if f["arm"] == "refit_frozen")
        assert frozen_fit["fold"] == "frozen@2024-11-01" and frozen_fit["n_fit"] == int((md < "2024-11-01").sum())
        assert set(r["verdicts_vs_served"]) == {"pinned", "refit", "refit_frozen"}
        # The decision reads the decision window only — every arm on the same rows.
        for arm in ("pinned", "refit", "refit_frozen"):
            assert r["vs_served"][arm]["over_5.5"]["n"] == n_decision
            assert r["vs_served"][arm]["over_5.5"]["n_folds"] == 4
            assert r["verdicts_vs_served"][arm]["n_folds"] == 4
        # The full window is context: the walk-forward arms over every row, refit_frozen absent.
        assert r["vs_served_full_window"]["refit"]["over_5.5"]["n"] == r["n_eval"] > n_decision
        assert "refit_frozen" not in r["vs_served_full_window"] and "refit_frozen" not in r["brier_full_window"]
        per_fold = {a: {row["fold"]: row for row in rows} for a, rows in r["per_fold_vs_served"].items()}
        assert per_fold["refit_frozen"]["2024-04"]["delta"] is None and not per_fold["refit"]["2024-10"]["decides"]
        assert per_fold["refit_frozen"]["2024-11"]["decides"]
        assert per_fold["refit_frozen"]["2024-11"]["delta"] == pytest.approx(per_fold["refit"]["2024-11"]["delta"])
        assert per_fold["refit_frozen"]["2025-02"]["delta"] is not None
        assert set(r["brier"]["served"]) == set(ab.UNIT_NAMES)
        assert all(r["brier"][a]["correct_score_grid"] is not None for a in r["brier"])
        assert set(r["correct_score_grid_log"]["arms"]) == set(r["brier"])
        assert r["correct_score_grid_log"]["vs_served"]["refit_frozen"]["n"] == n_decision
        # regulation units skip the rows without regulation_winner; the rest grade every row.
        full = r["n_gradable_full_window"]
        assert full["regulation_1x2"] < full["over_5.5"] == full["correct_score_grid"] == r["n_eval"]
        assert r["n_gradable"]["over_5.5"] == n_decision
        assert r["decision"]["arm"] == "refit_frozen"
        assert set(r["regressions"]) == {"pinned", "refit", "refit_frozen"}
        for regs in r["regressions"].values():
            assert not set(regs) & {"over_5.5", "correct_score_listed"}
        assert {"2023-24", "2024-25"} == set(r["by_season"])
        assert set(r["frozen_known_strata"]) == {"0", "1", "2"}
        assert sum(blk["n"] for blk in r["frozen_known_strata"].values()) == n_decision
        # Team coverage by the FROZEN fit's teams, not only served's: Utah HC (Nov 2024)
        # is unknown to served but known to the frozen fit; Utah Mammoth to neither.
        dec = frame[(md >= "2024-11-01") & (md < "2025-03-01")]
        mammoth = int((dec["home_team"] == "Utah Mammoth").sum() + (dec["away_team"] == "Utah Mammoth").sum())
        assert set(r["unknown_teams"]["served"]) == {"Utah Utah Hockey Club", "Utah Mammoth"}
        assert r["unknown_teams"]["refit_frozen"] == {"Utah Mammoth": mammoth} and mammoth > 0
        assert r["unknown_teams"]["served"]["Utah Mammoth"] == mammoth
        assert r["frozen_known_strata"]["2"]["n"] > r["served_known_strata"]["2"]["n"]
        assert r["frozen_known_strata"]["1"]["n"] == mammoth
        text = ab.format_report(r)
        assert "DECISION (pre-registered, amended 2026-10-31: refit_frozen must pass ship_strict" in text
        assert "DECISION WINDOW (amended, A3): folds from 2024-11-01" in text

    def test_without_a_decision_fold_nothing_can_ship(self, monkeypatch, tmp_path):
        args, _ = self._setup(monkeypatch, tmp_path, decision_start=None)  # the real 2025-05-01
        assert ab.DECISION_START == "2025-05-01"
        r = ab.run(args)
        assert r["decision_window"]["folds"] == [] and r["frozen_artifact"] is None
        assert not r["decision"]["ship"] and "decision start 2025-05-01" in r["decision"]["reason"]
        assert "refit_frozen" not in r["brier"] and "refit_frozen" not in r["verdicts_vs_served"]
        json.dumps(ab.ht.clean_json(r), allow_nan=False)
        assert "DO-NOT-SHIP" in ab.format_report(r)

    def test_every_challenger_disqualified_still_reports(self, monkeypatch, tmp_path):
        args, _ = self._setup(monkeypatch, tmp_path)
        orig = ab.fit_arm

        def refused(fit_df, cfg, tmp, tag):
            model, res, path, _ = orig(fit_df, cfg, tmp, tag)
            return model, res, path, "the fit did not converge"

        monkeypatch.setattr(ab, "fit_arm", refused)
        r = ab.run(args)
        assert r["verdicts_vs_served"] == {} and set(r["disqualified"]) == {"pinned", "refit", "refit_frozen"}
        assert not r["decision"]["ship"] and "disqualified" in r["decision"]["reason"]
        text = ab.format_report(r)
        assert "DO-NOT-SHIP" in text and "no challenger survived" in text
        json.dumps(ab.ht.clean_json(r), allow_nan=False)

    def test_the_default_served_path_must_be_the_legacy_fit(self, monkeypatch, tmp_path):
        args, _ = self._setup(monkeypatch, tmp_path)
        monkeypatch.setattr(ab, "SERVED_ARTIFACT", args.served_artifact)  # 12 teams, not the legacy 33
        with pytest.raises(RuntimeError, match="not the 2026-06-04 legacy fit"):
            ab.run(args)
