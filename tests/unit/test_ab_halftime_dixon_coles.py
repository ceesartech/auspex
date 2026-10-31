"""Unit tests for the half-time Dixon-Coles A/B harness — the pure pieces
that decide whether the experiment is honest: outcome labels, the
serve-path market read-out, calibration, and the trainer arm's fit cutoff.
The fitting and DB paths are exercised by running the harness on prod."""

from __future__ import annotations

import importlib.util
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


ab = _load("ab_halftime_dixon_coles", "ab_halftime_dixon_coles.py")


def _ev(scores, ft=None):
    ft = ft or scores
    return pd.DataFrame(
        {
            "home_score": [h for h, _ in scores],
            "away_score": [a for _, a in scores],
            "ft_home": [h for h, _ in ft],
            "ft_away": [a for _, a in ft],
        }
    )


class TestRealised:
    def test_labels_match_the_training_query(self):
        t = ab.realised(_ev([(1, 0), (0, 0), (0, 2), (1, 1)]))
        assert t["match_result_ht"].tolist() == [0, 1, 2, 1]  # home/draw/away, as HT_TRAINING_QUERY
        assert t["over_0.5"].tolist() == [1, 0, 1, 1]
        assert t["over_1.5"].tolist() == [0, 0, 1, 1]
        assert t["over_2.5"].tolist() == [0, 0, 0, 0]
        assert t["btts_ht"].tolist() == [0, 0, 0, 1]

    def test_ht_ft_index_is_ht_outcome_times_3_plus_ft_outcome(self):
        t = ab.realised(_ev([(1, 0), (0, 0), (0, 1)], ft=[(1, 1), (2, 0), (0, 3)]))
        assert t["ht_ft"].tolist() == [0 * 3 + 1, 1 * 3 + 0, 2 * 3 + 2]  # home_draw, draw_home, away_away
        assert [ab.HTFT_KEYS[i] for i in t["ht_ft"]] == ["home_draw", "draw_home", "away_away"]


class TestServePathReadout:
    def test_markets_from_matrix_reads_the_served_keys(self):
        from predictors.market_derivation import build_dc_matrix

        m = ab.markets_from_matrix(build_dc_matrix(0.7, 0.5, -0.03, max_goals=10), build_dc_matrix(0.8, 0.6, 0.0))
        assert m["match_result_ht"].sum() == pytest.approx(1.0)
        assert m["ht_ft"].sum() == pytest.approx(1.0)
        # HT marginal of the joint equals the HT 1x2
        assert m["ht_ft"].reshape(3, 3).sum(axis=1) == pytest.approx(m["match_result_ht"], abs=1e-9)
        assert m["match_result_ht"][0] > m["match_result_ht"][2]  # home lambda > away lambda
        assert 0 < m["over_1.5"] < m["over_0.5"] < 1
        assert 0 < m["btts_ht"] < 1

    def test_score_arm_is_per_match_brier(self):
        probs = {
            "match_result_ht": np.array([[1.0, 0.0, 0.0], [0.2, 0.5, 0.3]]),
            "over_0.5": np.array([1.0, 0.5]),
            "over_1.5": np.array([0.0, 0.5]),
            "over_2.5": np.array([0.0, 0.0]),
            "btts_ht": np.array([0.0, 1.0]),
            "ht_ft": np.full((2, 9), 1 / 9),
        }
        s = ab.score_arm(probs, ab.realised(_ev([(1, 0), (0, 0)])))
        assert s["ht_ft"].tolist() == pytest.approx([8 / 81 + (1 - 1 / 9) ** 2] * 2)
        assert s["match_result_ht"].tolist() == pytest.approx([0.0, 0.2**2 + 0.5**2 + 0.3**2])
        assert s["btts_ht"].tolist() == pytest.approx([0.0, 1.0])


class TestCalibration:
    def test_reports_predicted_vs_realised_and_argmax(self):
        proba = np.array([[0.5, 0.3, 0.2], [0.4, 0.35, 0.25], [0.2, 0.5, 0.3]])
        c = ab.draw_calibration(proba, np.array([1, 1, 0]))
        assert c["pred_draw"] == pytest.approx((0.3 + 0.35 + 0.5) / 3)
        assert c["real_draw"] == pytest.approx(2 / 3)
        assert c["argmax_home"] == pytest.approx(2 / 3) and c["argmax_draw"] == pytest.approx(1 / 3)
        assert c["argmax_hit"] == pytest.approx(0.0)


class TestTrainerCutoff:
    def _frame(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        home = [f"h{i}" for i in range(10)]
        away = [f"a{i}" for i in range(10)]
        return pd.DataFrame({"match_date": dates, "home_team": home, "away_team": away}), dates

    def test_cutoff_is_the_last_row_of_the_trainers_80_percent(self):
        frame, dates = self._frame()
        assert ab.trainer_fit_cutoff(frame) == dates[7]  # rows [0, 8) were fitted
        assert ab.trainer_fit_cutoff(frame, n_fit=5) == dates[4]

    def test_slice_must_hold_exactly_the_artifacts_teams(self):
        frame, _ = self._frame()
        head = frame.iloc[:5]
        teams = dict.fromkeys(set(head["home_team"]) | set(head["away_team"]), 1.0)
        ab.check_trainer_slice(frame, 5, teams)  # matches: no raise
        with pytest.raises(RuntimeError, match="cannot be reconstructed"):
            ab.check_trainer_slice(frame, 6, teams)


class TestLoudFailures:
    def test_non_normalised_or_non_finite_probabilities_are_reported(self):
        good = {
            "match_result_ht": np.array([[0.4, 0.4, 0.2]]),
            "over_0.5": np.array([0.6]),
            "over_1.5": np.array([0.3]),
            "over_2.5": np.array([0.1]),
            "btts_ht": np.array([0.2]),
            "ht_ft": np.full((1, 9), 1 / 9),
        }
        assert ab.invalid_probabilities(good) is None
        assert "ht_ft" in ab.invalid_probabilities({**good, "ht_ft": np.full((1, 9), 0.2)})
        assert "non-normalised" in ab.invalid_probabilities({**good, "match_result_ht": np.array([[0.5, 0.4, 0.2]])})
        assert "btts_ht" in ab.invalid_probabilities({**good, "btts_ht": np.array([np.nan])})

    def test_t_bound_is_stricter_than_z2_on_few_folds(self):
        assert ab.t_upper(-0.006, 0.002, 15) > -0.006 + 2 * 0.002
        assert ab.t_upper(-0.006, None, 15) is None and ab.t_upper(-0.006, 0.002, 1) is None

    def test_clean_json_drops_non_finite_floats(self):
        out = ab.clean_json({"a": float("nan"), "b": [np.float64(1.5), np.int64(2)], "c": np.bool_(True)})
        assert out == {"a": None, "b": [1.5, 2], "c": True}


class TestArms:
    def test_refit_arms_differ_only_in_the_intended_knobs(self):
        o = ab.arm_overrides()
        assert o["refit"] == {}
        assert o["pinned"]["per_league_baselines"] is False and o["pinned"]["max_goals"] == 6
        assert o["pinned_league"] == {**o["pinned"], "per_league_baselines": True}
