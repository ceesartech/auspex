"""Unit tests for the soccer ensemble scoreline-members A/B harness: the pure
pieces that decide whether the experiment is honest. That covers the blend
arithmetic each arm uses, the serve-path row builder, the out-of-sample
window, the in-memory merge simulation, the trainer-frame parity of the
member fits, and the verdict. The DB and model paths are exercised by running
the harness."""

from __future__ import annotations

import importlib.util
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
                "match_date": [t + pd.Timedelta(hours=1), t, t + pd.Timedelta(days=7), t],
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
