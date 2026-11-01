"""Unit tests for the soccer ensemble scoreline-members A/B harness: the
pieces that decide whether the experiment is honest. That covers the blend
arithmetic and the arm assembly, the serve-path row builder and fidelity
refusals, the out-of-sample window and the fold guard, the in-memory merge
simulation, the trainer-frame parity and save/load round-trip of the member
fits, the in-sample-optimal weight bound, the report's signs and verdicts, and
run() end to end on a synthetic corpus with the DB and registry faked."""

from __future__ import annotations

import importlib.util
import itertools
import re
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


ab = _load("ab_soccer_ensemble_members", "ab_soccer_ensemble_members.py")

WEIGHTS = {
    "xgboost_soccer_match_result": 0.4484264305538307,
    "lightgbm_soccer_match_result": 0.1051055545620673,
    "neural_network_soccer_match_result": 0.35445971101950746,
    "poisson_soccer_match_result": 0.046004151932297274,
    "dixon_coles_soccer_match_result": 0.046004151932297274,
}


class _Const:
    """A fitted member that emits one fixed 1x2 for every row."""

    is_fitted = True

    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)

    def predict_proba(self, X):
        return np.tile(self.p, (len(X), 1))


def _probs(n=4, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for m in ab.ALL_MEMBERS:
        raw = rng.random((n, 3)) + 0.1
        out[m] = raw / raw.sum(axis=1, keepdims=True)
    return out


class TestBlend:
    def test_matches_the_ensemble_predictor_arithmetic(self):
        from predictors.ensemble import EnsemblePredictor
        from predictors.model_config import ENSEMBLE_CONFIG

        probs = _probs()
        ens = EnsemblePredictor(ENSEMBLE_CONFIG)
        for m in ab.ALL_MEMBERS:
            ens.add_model(m, _Const(probs[m][0]), weight=WEIGHTS[m])
        served = ab.blend({m: probs[m][:1] for m in ab.ALL_MEMBERS}, WEIGHTS, ab.ALL_MEMBERS)
        assert served == pytest.approx(ens.predict_proba(pd.DataFrame({"x": [0]})), abs=1e-12)

    def test_drop_renormalises_the_learned_weights(self):
        probs = _probs()
        drop = ab.blend(probs, WEIGHTS, ab.LEARNED_MEMBERS)
        w = {m: WEIGHTS[m] for m in ab.LEARNED_MEMBERS}
        total = sum(w.values())
        expect = sum(w[m] / total * probs[m] for m in ab.LEARNED_MEMBERS)
        assert drop == pytest.approx(expect, abs=1e-12)
        assert drop.sum(axis=1) == pytest.approx(np.ones(len(drop)))

    def test_scoreline_members_carry_9_2_percent(self):
        w = ab.normalised_weights(WEIGHTS, ab.ALL_MEMBERS)
        assert w["poisson_soccer_match_result"] + w["dixon_coles_soccer_match_result"] == pytest.approx(
            0.0920, abs=1e-4
        )

    def test_a_member_without_a_weight_is_refused(self):
        with pytest.raises(ValueError, match="no weight"):
            ab.normalised_weights({"a": 1.0}, ("a", "b"))

    def test_oracle_replaces_only_the_members_share(self):
        probs = _probs(n=3)
        served = ab.blend(probs, WEIGHTS, ab.ALL_MEMBERS)
        w = ab.normalised_weights(WEIGHTS, ab.ALL_MEMBERS)
        member_sum = sum(w[m] * probs[m] for m in ab.SCORELINE_MEMBERS)
        y = np.array([0, 1, 2])
        out = ab.oracle_blend(served, member_sum, y, w["poisson_soccer_match_result"] * 2)
        learned_only = sum(w[m] * probs[m] for m in ab.LEARNED_MEMBERS)
        assert out == pytest.approx(learned_only + np.eye(3)[y] * (1 - sum(w[m] for m in ab.LEARNED_MEMBERS)))
        assert out.sum(axis=1) == pytest.approx(np.ones(3))


class TestServeRow:
    def test_mirrors_plain_keys_and_sets_identity_last(self):
        row, n = ab.serve_row(
            {"odds_home": 2.1, "feature__x": 1.0, "home_team": 7.0},
            {"home_team": "Arsenal", "away_team": "Chelsea", "league_id": "L1"},
            {},
        )
        assert n == 0
        assert row["feature__odds_home"] == 2.1 and row["odds_home"] == 2.1
        assert row["feature__x"] == 1.0 and "feature__feature__x" not in row
        # identity columns win over a same-named numeric feature, as in production
        assert row["home_team"] == "Arsenal" and row["league_id"] == "L1"
        assert row["feature__home_team"] == 7.0

    def test_non_numeric_values_take_the_corpus_median(self):
        row, n = ab.serve_row({"a": None, "b": "x", "c": 3}, {}, {"a": 0.5})
        assert n == 2
        assert row["a"] == 0.5 and row["feature__a"] == 0.5
        assert row["b"] is None  # no median for the key: stays missing, and rows_missing_features catches it
        assert ab.needs_medians({"a": None}) and not ab.needs_medians({"a": 1.0})

    def test_rows_missing_a_required_feature_are_flagged(self):
        frame = pd.DataFrame({"f1": [1.0, np.nan, 2.0], "f2": [1.0, 1.0, None]})
        assert ab.rows_missing_features(frame, ["f1", "f2"]).tolist() == [False, True, True]
        assert ab.rows_missing_features(frame, ["f1", "absent"]).all()


class TestOutOfSample:
    def _frame(self):
        dates = pd.date_range("2024-03-28", periods=8, freq="D", tz="UTC")
        return pd.DataFrame({"match_id": [f"m{i}" for i in range(8)], "match_date": dates})

    def test_only_rows_after_the_newest_seen_row_and_not_seen(self):
        frame = self._frame()
        seen = {"m0", "m1", "m3"}  # m2 is unseen but inside the seen span
        mask, cutoff, n_late = ab.out_of_sample_mask(frame, seen)
        assert cutoff == frame["match_date"][3]
        assert mask.tolist() == [False, False, False, False, True, True, True, True]
        assert n_late == 1

    def test_refuses_when_no_seen_id_is_in_the_frame(self):
        with pytest.raises(RuntimeError, match="none of the incumbent"):
            ab.out_of_sample_mask(self._frame(), {"zzz"})


class TestMergeSimulation:
    def _frame(self):
        t = pd.Timestamp("2026-08-01 15:00", tz="UTC")
        return pd.DataFrame(
            {
                "match_id": ["fd1", "espn1", "espn2", "other"],
                "match_date": [t, t + pd.Timedelta(hours=1), t + pd.Timedelta(days=7), t],
                "home_team_id": ["old", "new", "x", "y"],
                "away_team_id": ["opp", "opp", "new", "z"],
                "home_team": ["Manchester City", "Man City", "X", "Y"],
                "away_team": ["Opp", "Opp", "Man City", "Z"],
                "home_score": [2, 2, 0, 1],
                "away_score": [1, 1, 3, 1],
                "has_features": [False, True, True, True],
            }
        )

    def _map(self):
        return pd.DataFrame(
            {"new_team_id": ["new"], "old_team_id": ["old"], "new_name": ["Man City"], "old_name": ["Manchester City"]}
        )

    def test_repoints_names_and_folds_the_twin_keeping_the_row_with_features(self):
        out, stats = ab.simulate_team_merge(self._frame(), self._map())
        assert sorted(out["match_id"]) == ["espn1", "espn2", "other"]
        kept = out.set_index("match_id")
        assert kept.loc["espn1", "home_team"] == "Manchester City" and kept.loc["espn1", "home_team_id"] == "old"
        assert kept.loc["espn2", "away_team"] == "Manchester City"
        assert stats["rows_repointed"] == 2 and stats["twin_rows_folded"] == 1 and stats["twin_clusters"] == 1

    def test_without_features_on_either_twin_the_earlier_row_is_kept(self):
        frame = self._frame()
        frame.loc[frame["match_id"] == "espn1", "has_features"] = False
        out, _ = ab.simulate_team_merge(frame, self._map())
        assert sorted(out["match_id"]) == ["espn2", "fd1", "other"]

    def test_twins_that_disagree_on_the_score_are_refused(self):
        frame = self._frame()
        frame.loc[0, "home_score"] = 3
        with pytest.raises(RuntimeError, match="disagree on the score"):
            ab.simulate_team_merge(frame, self._map())

    def test_untouched_same_pair_rows_are_left_alone(self):
        frame = self._frame()
        extra = frame.iloc[[3]].assign(match_id="other2", match_date=frame["match_date"][3] + pd.Timedelta(hours=2))
        out, stats = ab.simulate_team_merge(pd.concat([frame, extra], ignore_index=True), self._map())
        assert {"other", "other2"} <= set(out["match_id"])  # a fork no merge touched is not this tool's to fold

    def test_the_shipped_map_has_the_columns_the_simulation_reads(self):
        m = pd.read_csv(SCRIPTS_DIR / "data" / "soccer_team_merge_map.csv", comment="#", dtype=str)
        assert {"new_team_id", "old_team_id", "new_name", "old_name"} <= set(m.columns)
        assert m["new_team_id"].is_unique and not set(m["new_team_id"]) & set(m["old_team_id"])


class TestTrainerParity:
    def test_match_query_selects_the_training_querys_population(self):
        from utils.training_data import DEFAULT_TRAINING_QUERY

        def norm(s):
            return re.sub(r"\s+", " ", s).strip()

        train = norm(DEFAULT_TRAINING_QUERY)
        for predicate in (
            "WHERE l.sport = 'soccer' AND m.status = 'finished' AND m.home_score IS NOT NULL "
            "AND m.away_score IS NOT NULL",
            "JOIN teams ht ON m.home_team_id = ht.id JOIN teams at ON m.away_team_id = at.id",
            "ht.name AS home_team, at.name AS away_team",
            "m.league_id::text AS league_id",
        ):
            assert predicate in train
            assert predicate in norm(ab.MATCHES_SQL)

    def test_members_fit_on_the_trainers_columns_only(self, tmp_path):
        rng = np.random.default_rng(1)
        teams = [f"t{i}" for i in range(6)]
        n = 300
        home = rng.choice(teams, n)
        away = np.array([rng.choice([t for t in teams if t != h]) for h in home])
        fit_df = pd.DataFrame(
            {
                "home_team": home,
                "away_team": away,
                "home_score": rng.poisson(1.5, n),
                "away_score": rng.poisson(1.1, n),
                "match_date": pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
                "league_id": "L1",
                # a league NAME column must not be read: the trainer's frame carries league_id only
                "league_name": ["A", "B"] * (n // 2),
            }
        )
        dc, res = ab.fit_member(ab.DIXON_COLES, fit_df, str(tmp_path), "t")
        assert res["team_count"] == 6
        assert set(dc.league_baselines) == {"L1"}  # grouped on league_id; no name aliases from league_name
        p = ab.member_proba(dc, fit_df.head(5))
        assert p.shape == (5, 3) and p.sum(axis=1) == pytest.approx(np.ones(5))
        pois, _ = ab.fit_member(ab.POISSON, fit_df, str(tmp_path), "t")
        assert pois.league_baselines == {}  # POISSON_CONFIG fits one global baseline
        assert ab.known_teams(dc, ["t0", "zz"], ["t1", "t2"]).tolist() == [2.0, 1.0]

    def test_a_refit_is_served_through_save_and_the_serving_load(self, tmp_path, monkeypatch):
        from predictors.poisson_models import DixonColesPredictor

        loaded = []
        real_load = DixonColesPredictor.load

        def spy(self, p):
            loaded.append((self, p))
            return real_load(self, p)

        monkeypatch.setattr(DixonColesPredictor, "load", spy)
        dc, _ = ab.fit_member(ab.DIXON_COLES, _corpus().head(300), str(tmp_path), "rt")
        assert len(loaded) == 1 and dc is loaded[0][0]  # the object scored is the one load() rebuilt
        assert loaded[0][1].endswith("dixon_coles_soccer_match_result_rt.bin")


class TestAssembleArms:
    def _inputs(self, n=5):
        probs = _probs(n=n, seed=4)
        learned = {m: probs[m] for m in ab.LEARNED_MEMBERS}
        rng = np.random.default_rng(9)
        scoreline = {}
        for arm in ab.MEMBER_ARMS:
            scoreline[arm] = {}
            for m in ab.SCORELINE_MEMBERS:
                raw = rng.random((n, 3)) + 0.1
                scoreline[arm][m] = raw / raw.sum(axis=1, keepdims=True)
        market = np.full((n, 3), np.nan)
        market[::2] = [0.5, 0.3, 0.2]
        return learned, scoreline, market, np.arange(n) % 3

    def test_each_arm_is_the_blend_it_names(self):
        learned, scoreline, market, y = self._inputs()
        p = ab.assemble_arms(learned, scoreline, market, y, WEIGHTS)
        w = ab.normalised_weights(WEIGHTS, ab.ALL_MEMBERS)
        for arm in ab.MEMBER_ARMS:
            expect = sum(w[m] * learned[m] for m in ab.LEARNED_MEMBERS)
            expect = expect + sum(w[m] * scoreline[arm][m] for m in ab.SCORELINE_MEMBERS)
            assert p[arm] == pytest.approx(expect, abs=1e-12)
        assert set(p) == set(ab.ARMS) | {ab.BOUND_ARM, "oracle"}

    def test_drop_is_the_learned_members_alone_renormalised(self):
        learned, scoreline, market, y = self._inputs()
        p = ab.assemble_arms(learned, scoreline, market, y, WEIGHTS)
        total = sum(WEIGHTS[m] for m in ab.LEARNED_MEMBERS)
        expect = sum(WEIGHTS[m] / total * learned[m] for m in ab.LEARNED_MEMBERS)
        assert p["drop"] == pytest.approx(expect, abs=1e-12)
        # and no scoreline member can move it
        scoreline["served"] = {m: np.tile([1.0, 0.0, 0.0], (len(y), 1)) for m in ab.SCORELINE_MEMBERS}
        assert ab.assemble_arms(learned, scoreline, market, y, WEIGHTS)["drop"] == pytest.approx(expect, abs=1e-12)

    def test_the_bound_replaces_both_members_by_the_market_on_market_rows_only(self):
        learned, scoreline, market, y = self._inputs()
        p = ab.assemble_arms(learned, scoreline, market, y, WEIGHTS)
        assert np.isfinite(p[ab.BOUND_ARM][::2]).all() and np.isnan(p[ab.BOUND_ARM][1::2]).all()
        swapped = {m: np.tile([0.5, 0.3, 0.2], (len(y), 1)) for m in ab.SCORELINE_MEMBERS}
        expect = ab.blend({**learned, **swapped}, WEIGHTS, ab.ALL_MEMBERS)
        assert p[ab.BOUND_ARM][::2] == pytest.approx(expect[::2], abs=1e-12)
        rows = np.arange(len(y))
        assert (p["oracle"][rows, y] > p["served"][rows, y]).all()  # the members' share went to the outcome


class _FakeEnsemble:
    """predict_proba returns a fixed per-row table, optionally perturbed."""

    def __init__(self, table, batch_offset=0.0, single_offset=0.0, nan_row=None):
        self.table = np.asarray(table, dtype=float)
        self.batch_offset, self.single_offset, self.nan_row = batch_offset, single_offset, nan_row

    def predict_proba(self, X):
        out = self.table[X.index.to_numpy()].copy()
        if len(X) == 1:
            return out + self.single_offset
        if self.nan_row is not None:
            out[self.nan_row] = np.nan
        return out + self.batch_offset


class TestServePathFidelity:
    def _x(self, n=10):
        rng = np.random.default_rng(2)
        raw = rng.random((n, 3)) + 0.1
        return pd.DataFrame({"f": np.arange(n)}), raw / raw.sum(axis=1, keepdims=True)

    def test_a_faithful_ensemble_passes(self):
        x, p = self._x()
        batch, single, n = ab.serve_path_fidelity(_FakeEnsemble(p), x, p)
        assert batch == 0.0 and single == 0.0 and n == len(x)

    def test_a_nan_in_the_ensembles_batch_is_refused(self):
        x, p = self._x()
        with pytest.raises(RuntimeError, match="non-finite"):
            ab.serve_path_fidelity(_FakeEnsemble(p, nan_row=3), x, p)

    def test_a_nan_in_the_harness_blend_is_refused(self):
        x, p = self._x()
        mine = p.copy()
        mine[0] = np.nan
        with pytest.raises(RuntimeError, match="non-finite"):
            ab.serve_path_fidelity(_FakeEnsemble(p), x, mine)

    def test_a_blend_off_by_more_than_the_tolerance_is_refused(self):
        x, p = self._x()
        with pytest.raises(RuntimeError, match="differ by"):
            ab.serve_path_fidelity(_FakeEnsemble(p, batch_offset=1e-4), x, p)

    def test_single_row_predictions_that_differ_from_the_batch_are_refused(self):
        x, p = self._x()
        with pytest.raises(RuntimeError, match="one-row-at-a-time"):
            ab.serve_path_fidelity(_FakeEnsemble(p, single_offset=1e-4), x, p)

    def test_max_gap_refuses_a_shape_mismatch(self):
        with pytest.raises(RuntimeError, match="shapes"):
            ab.max_gap("x", np.zeros((2, 3)), np.zeros((3, 3)))


class TestFoldGuard:
    def test_a_fit_row_inside_the_eval_month_is_refused(self):
        fold = ab.ft.build_folds("2024-05-01", "2024-05-31")[0]
        before = pd.DataFrame({"match_date": pd.to_datetime(["2024-04-30 20:00"], utc=True)})
        inside = pd.DataFrame({"match_date": pd.to_datetime(["2024-05-02 15:00"], utc=True)})
        ab.check_fold_disjoint(before, inside, fold)
        with pytest.raises(RuntimeError, match="overlap"):
            ab.check_fold_disjoint(pd.concat([before, inside]), inside, fold)
        with pytest.raises(RuntimeError, match="overlap"):
            ab.check_fold_disjoint(before, pd.concat([before, inside]), fold)


class TestOptimalSimplexWeights:
    def _sources(self, n=400, seed=5):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 3, n)
        sources = []
        for sharp in (0.2, 0.5, 0.0):
            raw = rng.random((n, 3)) + 0.2 + sharp * np.eye(3)[y]
            sources.append(raw / raw.sum(axis=1, keepdims=True))
        return sources, y

    def test_matches_a_dense_grid_search(self):
        sources, y = self._sources()
        w, brier = ab.optimal_simplex_weights(sources, y)
        assert w.min() >= 0 and w.sum() == pytest.approx(1.0)
        grid = []
        for a, b in itertools.product(np.linspace(0, 1, 101), repeat=2):
            if a + b <= 1 + 1e-12:
                ww = np.array([a, b, 1 - a - b])
                grid.append(ab.ft.brier_multiclass(np.tensordot(ww, np.stack(sources), axes=1), y).mean())
        assert brier <= min(grid) + 1e-12
        assert brier == pytest.approx(min(grid), abs=1e-4)
        assert brier == pytest.approx(ab.ft.brier_multiclass(np.tensordot(w, np.stack(sources), axes=1), y).mean())

    def test_a_perfect_source_takes_all_the_weight(self):
        sources, y = self._sources()
        w, brier = ab.optimal_simplex_weights(sources + [np.eye(3)[y]], y)
        assert w[-1] == pytest.approx(1.0) and brier == pytest.approx(0.0, abs=1e-12)

    def test_non_finite_sources_are_refused(self):
        sources, y = self._sources()
        sources[1][0, 0] = np.nan
        with pytest.raises(RuntimeError, match="non-finite"):
            ab.optimal_simplex_weights(sources, y)


def _stacked(n_folds=6, per_fold=150, seed=11):
    """Synthetic per-row arrays shaped like run()'s. Built so that refit is
    WORSE than served and drop is far BETTER, by construction."""
    rng = np.random.default_rng(seed)
    n = n_folds * per_fold
    y = rng.integers(0, 3, n)

    def rand_p():
        raw = rng.random((n, 3)) + 0.2
        return raw / raw.sum(axis=1, keepdims=True)

    learned = {m: rand_p() for m in ab.LEARNED_MEMBERS}
    scoreline = {arm: {m: rand_p() for m in ab.SCORELINE_MEMBERS} for arm in ab.MEMBER_ARMS}
    market = rand_p()
    market[1::3] = np.nan
    p = ab.assemble_arms(learned, scoreline, market, y, WEIGHTS)
    p["refit"] = 0.7 * p["served"] + 0.3 * np.eye(3)[(y + 1) % 3]
    p["drop"] = 0.7 * p["served"] + 0.3 * np.eye(3)[y]
    s = {
        "y": y,
        "fold": np.repeat([f"2025-{k + 1:02d}" for k in range(n_folds)], per_fold).astype(object),
        "market": market,
        "espn_era": (np.arange(n) >= n // 2).astype(float),
        "identity_split": (np.arange(n) % 10 == 0).astype(float),
    }
    for arm in ab.MEMBER_ARMS:
        s[f"known::{arm}"] = np.full(n, 2.0)
        for m in ab.SCORELINE_MEMBERS:
            s[f"q::{arm}::{m}"] = scoreline[arm][m]
    for m in ab.LEARNED_MEMBERS:
        s[f"q::learned::{m}"] = learned[m]
    for arm, arr in p.items():
        s[f"p::{arm}"] = arr
    return s


class TestBuildReport:
    def test_a_worse_challenger_reads_positive_and_a_better_one_ships(self):
        s = _stacked()
        r = ab.build_report(s, meta={})
        y = s["y"]
        worse = ab.ft.brier_multiclass(s["p::refit"], y) - ab.ft.brier_multiclass(s["p::served"], y)
        better = ab.ft.brier_multiclass(s["p::drop"], y) - ab.ft.brier_multiclass(s["p::served"], y)
        assert r["vs_served"]["refit"]["delta"] == pytest.approx(worse.mean()) and worse.mean() > 0
        assert r["vs_served"]["drop"]["delta"] == pytest.approx(better.mean()) and better.mean() < -0.005
        assert r["verdicts_vs_served"]["refit"]["ship"] is False
        assert r["verdicts_vs_served"]["refit"]["ship_strict"] is False
        assert r["verdicts_vs_served"]["drop"]["ship_strict"] is True
        assert r["per_fold_vs_served"]["refit"][0]["delta"] > 0

    def test_the_bound_is_scored_on_the_market_rows_only(self):
        s = _stacked()
        r = ab.build_report(s, meta={})
        has = np.isfinite(s["market"]).all(axis=1)
        expect = ab.ft.brier_multiclass(s[f"p::{ab.BOUND_ARM}"][has], s["y"][has]).mean()
        assert r["market_rows"] == int(has.sum()) == len(has) - len(has[1::3])
        assert r["brier"][ab.BOUND_ARM] == pytest.approx(expect)
        assert r["bound"]["members_as_market_vs_served"]["n"] == int(has.sum())
        on, off = r["bound"]["refit_vs_served_on_market_rows"], r["bound"]["refit_vs_served_off_market_rows"]
        assert on["n"] == int(has.sum()) and off["n"] == int((~has).sum())

    def test_in_sample_optimal_weights_never_lose_to_the_production_weights(self):
        s = _stacked()
        opt = ab.build_report(s, meta={})["bound"]["in_sample_optimal_weights"]
        # the five served members at the production weights are one point of the simplex searched
        assert opt["served"]["delta_vs_served"] <= 1e-12
        assert sum(opt["refit"]["weights"].values()) == pytest.approx(1.0)
        assert set(opt["learned_only"]["weights"]) == set(ab.LEARNED_MEMBERS)

    def test_a_nan_in_any_scored_arm_is_refused_not_zeroed(self):
        for arm in ("served", "refit", "drop", "refit_pretest", "oracle"):
            s = _stacked()
            s[f"p::{arm}"][3] = np.nan
            with pytest.raises(RuntimeError, match=f"arm {arm}"):
                ab.build_report(s, meta={})

    def test_a_bound_row_that_disagrees_with_the_market_mask_is_refused(self):
        s = _stacked()
        s[f"p::{ab.BOUND_ARM}"][1] = s["p::served"][1]  # finite where there is no market
        with pytest.raises(RuntimeError, match="finite on"):
            ab.build_report(s, meta={})


class _Learned:
    """A fitted learned member: a fixed function of one serve-row feature."""

    is_fitted = True
    feature_names = ["feature__f1"]

    def __init__(self, scale):
        self.scale = scale

    def predict_proba(self, X):
        f = pd.to_numeric(X["feature__f1"]).to_numpy(dtype=float)
        raw = np.stack([np.exp(self.scale * f), np.ones_like(f), np.exp(-self.scale * f)], axis=1)
        return raw / raw.sum(axis=1, keepdims=True)


def _corpus(seed=7):
    """Two matches a day, 2023-01-01 .. 2024-06-30, eight teams, one league,
    with every column MATCHES_SQL selects plus the outcome."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01 15:00", "2024-06-30 15:00", freq="12h", tz="UTC")
    rows = []
    for i, d in enumerate(dates):
        h, a = rng.choice(8, 2, replace=False)
        rows.append(
            {
                "match_id": f"m{i:05d}",
                "match_date": d,
                "league_id": "L1",
                "league_name": "League One",
                "home_team_id": f"t{h}",
                "away_team_id": f"t{a}",
                "home_team": f"Team {h}",
                "away_team": f"Team {a}",
                "home_score": int(rng.poisson(1.5)),
                "away_score": int(rng.poisson(1.1)),
            }
        )
    frame = pd.DataFrame(rows)
    frame["outcome"] = np.where(
        frame["home_score"] > frame["away_score"], 0, np.where(frame["home_score"] == frame["away_score"], 1, 2)
    )
    return frame


class TestRunEndToEnd:
    """run() on a synthetic corpus, with only the DB reads and the registry
    faked: the serve rows, fidelity checks, member refits, fold guard, arm
    assembly, report and per-match CSV are the real code."""

    SEEN_BEFORE = pd.Timestamp("2024-03-16", tz="UTC")

    def _setup(self, tmp_path, monkeypatch):
        from predictors.ensemble import EnsemblePredictor
        from predictors.model_config import ENSEMBLE_CONFIG

        frame = _corpus()
        seen_rows = frame[frame["match_date"] < self.SEEN_BEFORE]
        seen = set(seen_rows["match_id"])
        ens = EnsemblePredictor(ENSEMBLE_CONFIG)
        for m, scale in zip(ab.LEARNED_MEMBERS, (0.3, 0.5, 0.8)):
            ens.add_model(m, _Learned(scale), weight=WEIGHTS[m])
        for m in ab.SCORELINE_MEMBERS:  # the incumbent: fitted on the oldest 70% of its train+val
            model, _ = ab.fit_member(m, seen_rows.head(int(0.7 * len(seen_rows))), str(tmp_path), "incumbent")
            ens.add_model(m, model, weight=WEIGHTS[m])
        rng = np.random.default_rng(1)
        late = frame[frame["match_date"] >= pd.Timestamp("2024-03-01", tz="UTC")]
        features = {mid: {"f1": float(rng.normal())} for mid in late["match_id"]}
        no_features = late["match_id"].iloc[-5]
        del features[no_features]
        features[late["match_id"].iloc[-9]] = {"f1": None}  # takes the corpus median, as production does
        quote = ab.ft.MarketQuote(probs=np.array([0.46, 0.27, 0.27]), n_books=3, book_sum_median=1.05)
        market = {mid: quote for mid in late["match_id"].iloc[::2]}
        calls = {"medians": 0}

        def medians(url):
            calls["medians"] += 1
            return {"f1": 0.25}

        monkeypatch.setattr(ab, "load_frame", lambda url: frame.copy())
        monkeypatch.setattr(ab, "load_ensemble", lambda path: (ens, "fake-ensemble"))
        monkeypatch.setattr(ab, "member_fingerprint", lambda path, m: {"member": m})
        monkeypatch.setattr(ab, "load_seen_ids", lambda path: set(seen))
        # The incumbent's fingerprint: its members were fitted on the oldest
        # int(0.7 x rows) of these rows, as the trainer does.
        monkeypatch.setattr(ab, "load_incumbent_rows", lambda path: len(seen_rows))
        monkeypatch.setattr(ab, "load_features", lambda url, start, end: dict(features))
        monkeypatch.setattr(ab, "load_medians", medians)
        monkeypatch.setattr(ab, "load_market_1x2", lambda url, start, end, min_books: dict(market))
        args = ab.parse_args(
            [
                "--database-url",
                "postgresql:///unused",
                "--model-path",
                str(tmp_path / "registry"),
                "--merge-map",
                "",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        return frame, ens, features, market, calls, args

    def test_runs_and_every_arm_is_rebuilt_from_its_members(self, tmp_path, monkeypatch):
        frame, _ens, features, market, calls, args = self._setup(tmp_path, monkeypatch)
        r = ab.run(args)
        m = r["meta"]
        assert r["folds"] == ["2024-04", "2024-05", "2024-06"]
        assert [f["fold"] for f in m["skipped_folds"]] == ["2024-03"]  # 32 rows < MIN_FOLD_EVAL
        in_eval = frame[(frame["match_date"] >= pd.Timestamp("2024-04-01", tz="UTC"))]
        assert r["n_eval"] == int(in_eval["match_id"].isin(features.keys()).sum()) == len(in_eval) - 1
        assert m["eval_rows_without_features"] == 1 and m["median_filled_values"] == 1 and calls["medians"] == 1
        assert m["fidelity_batch_max_abs"] <= ab.FIDELITY_ATOL and m["fidelity_single_row_max_abs"] <= ab.FIDELITY_ATOL
        for fit in m["fits"]:
            if fit["fold"] not in ("pretest", "converged"):
                assert pd.Timestamp(fit["fit_newest"]) < pd.Timestamp(f"{fit['fold']}-01", tz="UTC")
        df = pd.read_csv(r["per_match_csv"])
        assert len(df) == r["n_eval"] and set(df["fold"]) == set(r["folds"])

        def cols(prefix):
            return df[[f"{prefix}::{c}" for c in ab.CLASSES]].to_numpy(dtype=float)

        learned = {mm: cols(f"q::learned::{mm}") for mm in ab.LEARNED_MEMBERS}
        assert cols("p::drop") == pytest.approx(ab.blend(learned, WEIGHTS, ab.LEARNED_MEMBERS), abs=1e-12)
        for arm in ab.MEMBER_ARMS:
            members = {**learned, **{mm: cols(f"q::{arm}::{mm}") for mm in ab.SCORELINE_MEMBERS}}
            assert cols(f"p::{arm}") == pytest.approx(ab.blend(members, WEIGHTS, ab.ALL_MEMBERS), abs=1e-12)
        has = df["match_id"].isin(market.keys()).to_numpy()
        assert np.isfinite(cols(f"p::{ab.BOUND_ARM}")[has]).all()
        assert np.isnan(cols(f"p::{ab.BOUND_ARM}")[~has]).all()
        d = (df["brier::refit"] - df["brier::served"]).mean()
        assert r["vs_served"]["refit"]["delta"] == pytest.approx(d)
        # The pre-registered converged arm: the served Dixon-Coles untouched,
        # its split proven by reproducing the served Poisson exactly, and on
        # this one-league corpus (one component, so the served fit converged
        # too) the converged Poisson is the served one.
        conv = ab.CONVERGED
        split = m["converged_arm_split"]
        assert split["reproduction_max_abs"] <= ab.SPLIT_ATOL and split["assignments_tried"] == 1
        dc = ab.DIXON_COLES
        assert np.array_equal(cols(f"q::{conv}::{dc}"), cols(f"q::served::{dc}"))
        assert cols(f"q::{conv}::{ab.POISSON}") == pytest.approx(cols(f"q::served::{ab.POISSON}"), abs=1e-5)
        assert r["convergence_decision"]["arm"] == conv and r["convergence_decision"]["ship"] is False
        assert any(f["fold"] == "converged" and f["converged_at"] for f in m["fits"])

    def test_a_fit_that_reaches_its_eval_month_fails_the_run(self, tmp_path, monkeypatch):
        *_, args = self._setup(tmp_path, monkeypatch)
        real_split = ab.ft.split_fold

        def leaky(frame, fold):
            _fit, ev = real_split(frame, fold)
            return frame, ev  # the whole corpus, the eval month included

        monkeypatch.setattr(ab.ft, "split_fold", leaky)
        with pytest.raises(RuntimeError, match="overlap"):
            ab.run(args)

    def test_a_nan_from_the_ensemble_fails_the_run(self, tmp_path, monkeypatch):
        _frame, ens, *_rest, args = self._setup(tmp_path, monkeypatch)
        real = ens.predict_proba

        def poisoned(X, apply_calibration=True):
            out = np.array(real(X), dtype=float)
            if len(X) > 1:
                out[0] = np.nan
            return out

        monkeypatch.setattr(ens, "predict_proba", poisoned)
        with pytest.raises(RuntimeError, match="non-finite"):
            ab.run(args)


def _tied_corpus(seed=21):
    """Two leagues that never meet (the soccer split's shape), one match a
    day, with SIX matches sharing one kickoff right at the 70% boundary."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2020-01-01 15:00", tz="UTC")
    for i in range(400):
        league = "A" if i % 2 else "B"
        h, a = rng.choice(6, 2, replace=False)
        rows.append(
            {
                "match_id": f"m{i:04d}",
                "match_date": start + pd.Timedelta(days=i),
                "league_id": league,
                "league_name": f"League {league}",
                "home_team": f"{league}{h}",
                "away_team": f"{league}{a}",
                "home_score": int(rng.poisson(0.9 if league == "A" else 2.0)),
                "away_score": int(rng.poisson(0.7 if league == "A" else 1.7)),
            }
        )
    frame = pd.DataFrame(rows)
    n_fit = int(len(frame) * ab.TRAIN_RATIO)  # rows = len(frame): everything is "seen"
    frame.loc[n_fit - 3 : n_fit + 2, "match_date"] = frame.loc[n_fit - 3, "match_date"]  # 6 tied rows
    return frame.sort_values("match_date", kind="stable").reset_index(drop=True), n_fit


class TestConvergedArm:
    def test_the_served_split_is_found_among_the_boundary_ties_and_proven(self, tmp_path):
        frame, n_fit = _tied_corpus()
        ties = list(range(n_fit - 3, n_fit + 3))
        # The trainer's unstable sort put the LAST three tied rows in train.
        served_rows = pd.concat([frame.iloc[: n_fit - 3], frame.loc[ties[3:]]])
        served, res = ab.fit_member(ab.POISSON, served_rows, str(tmp_path), "served")
        assert res["fit_diagnostics"]["connected_components"] == 2 and not res["converged"]
        split, info = ab.served_member_split(frame, set(frame["match_id"]), len(frame), served, str(tmp_path))
        assert set(split["match_id"]) == set(served_rows["match_id"])
        assert info["boundary_tied_rows"] == 6 and info["boundary_rows_in_train"] == 3
        assert info["assignments_tried"] == 20 and info["reproduction_max_abs"] <= ab.SPLIT_ATOL

    def test_a_split_that_reproduces_nothing_fails_the_run(self, tmp_path):
        frame, n_fit = _tied_corpus()
        other, _ = ab.fit_member(ab.POISSON, frame.iloc[: n_fit - 10], str(tmp_path), "other")
        with pytest.raises(RuntimeError, match="reproduces the served Poisson"):
            ab.served_member_split(frame, set(frame["match_id"]), len(frame), other, str(tmp_path))

    def test_the_converged_fit_converges_where_the_served_one_cycles(self, tmp_path):
        frame, n_fit = _tied_corpus()
        _served, res_served = ab.fit_member(ab.POISSON, frame.iloc[:n_fit], str(tmp_path), "s")
        conv, res = ab.fit_member(ab.POISSON, frame.iloc[:n_fit], str(tmp_path), "c", ab.CONVERGED_HYPERPARAMETERS)
        assert res_served["fit_diagnostics"]["period_two_oscillation"] is True
        assert res["converged"] is True and res["fit_diagnostics"]["gauge"] == "per_component"
        assert conv.gauge == "global"  # served through the plain production config, as the API loads it

    def test_incumbent_rows_come_from_the_fingerprint(self, tmp_path):
        d = tmp_path / ab.ENSEMBLE_NAME
        d.mkdir()
        with pytest.raises(RuntimeError, match="missing"):
            ab.load_incumbent_rows(str(tmp_path))
        (d / "held_out_metrics.json").write_text('{"fingerprint": {"rows": 170855}}')
        assert ab.load_incumbent_rows(str(tmp_path)) == 170855
        (d / "held_out_metrics.json").write_text('{"fingerprint": {}}')
        with pytest.raises(RuntimeError, match="no fingerprint rows"):
            ab.load_incumbent_rows(str(tmp_path))


class TestConvergenceDecision:
    @staticmethod
    def _pair(delta, se, se_cluster, n_folds=31, n=1000):
        return {"delta": delta, "se": se, "delta_cluster": delta, "se_cluster": se_cluster, "n_folds": n_folds, "n": n}

    def _report(self, main, units):
        conv = ab.CONVERGED
        return {
            "verdicts_vs_served": {conv: ab.verdict(main, "served")},
            "strata": {name: {"n": 100, conv: pair} for name, pair in units.items()},
        }

    def test_unit_lower_bound_is_clustered_when_it_can_be(self):
        assert ab.unit_lower_95(self._pair(0.002, 0.0001, 0.001)) == pytest.approx(0.002 - 1.96 * 0.001)
        one_fold = self._pair(0.002, 0.0001, None, n_folds=1)
        assert ab.unit_lower_95(one_fold) == pytest.approx(0.002 - 1.96 * 0.0001)
        assert ab.unit_lower_95({"delta": None, "se": None}) is None

    def test_a_gate_pass_with_no_regressing_unit_ships(self):
        main = self._pair(-0.006, 0.0005, 0.001)
        units = {"espn_era_2026_06_on": self._pair(0.001, 0.001, 0.002)}  # worse, but inside the noise
        d = ab.convergence_decision(self._report(main, units))
        assert d["ship"] is True and d["units_regressed"] == [] and d["line"].startswith("SHIP")

    def test_a_regressing_unit_blocks_a_gate_pass(self):
        main = self._pair(-0.006, 0.0005, 0.001)
        units = {"identity_split_fixtures": self._pair(0.004, 0.0005, 0.001), "not_identity_split": main}
        d = ab.convergence_decision(self._report(main, units))
        assert d["ship"] is False and [u["unit"] for u in d["units_regressed"]] == ["identity_split_fixtures"]
        assert "blocking unit" in d["line"]

    def test_a_null_result_does_not_ship(self):
        main = self._pair(-0.0001, 0.00005, 0.0001)
        d = ab.convergence_decision(self._report(main, {"espn_era_2026_06_on": main}))
        assert d["ship"] is False and "gate is not met" in d["line"]


class TestVerdict:
    def _pair(self, delta, se, se_cluster, n_folds=31):
        return {
            "delta": delta,
            "se": se,
            "delta_cluster": delta,
            "se_cluster": se_cluster,
            "n_folds": n_folds,
        }

    def test_ship_needs_the_gate_and_both_bounds(self):
        assert ab.verdict(self._pair(-0.006, 0.0005, 0.0010), "served")["ship_strict"] is True
        v = ab.verdict(self._pair(-0.006, 0.0005, 0.00297), "served")  # z=2 bound clears, t(30) bound does not
        assert v["ship"] is True and v["ship_strict"] is False
        assert ab.verdict(self._pair(-0.004, 0.0001, 0.0001), "served")["ship_strict"] is False

    def test_standalone_metrics(self):
        proba = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
        y = np.array([0, 2])
        m = ab.standalone_metrics(proba, y)
        assert m["brier"] == pytest.approx(((0.4**2 + 0.3**2 + 0.1**2) + (0.2**2 + 0.5**2 + 0.7**2)) / 2)
        assert m["log_loss"] == pytest.approx(-(np.log(0.6) + np.log(0.3)) / 2)
        assert 0 <= m["ece"] <= 1 and m["real_away"] == pytest.approx(0.5)

    def test_standalone_metrics_refuses_non_finite_rows_instead_of_dropping_them(self):
        proba = np.array([[0.6, 0.3, 0.1], [np.nan, np.nan, np.nan]])
        with pytest.raises(RuntimeError, match="non-finite"):
            ab.standalone_metrics(proba, np.array([0, 2]))
