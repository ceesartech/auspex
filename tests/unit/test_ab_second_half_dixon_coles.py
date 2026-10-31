"""Unit tests for the second-half Dixon-Coles A/B harness — the pure pieces
that decide whether the experiment is honest: the 2H targets and the frame
join, outcome labels, the serve-path read-out, the HT-state-conditional
constant, and the disqualification rule. The fitting and DB paths are
exercised by running the harness on prod."""

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


ab = _load("ab_second_half_dixon_coles", "ab_second_half_dixon_coles.py")


def _ev(ht, ft):
    """Eval rows from (HT, FT) score pairs, 2H goals derived the trainer's way."""
    h2h, h2a = ab.second_half_goals([h for h, _ in ht], [a for _, a in ht], [h for h, _ in ft], [a for _, a in ft])
    return pd.DataFrame(
        {
            "ht_home": [h for h, _ in ht],
            "ht_away": [a for _, a in ht],
            "ft_home": [h for h, _ in ft],
            "ft_away": [a for _, a in ft],
            "h2_home": h2h,
            "h2_away": h2a,
        }
    )


def _valid(n=1):
    return {
        "ht_ft": np.full((n, 9), 1 / 9),
        "h2_1x2": np.tile([0.4, 0.3, 0.3], (n, 1)),
        "h2_over_0.5": np.full(n, 0.7),
        "h2_over_1.5": np.full(n, 0.4),
    }


class TestSecondHalfTargets:
    def test_targets_are_full_time_minus_half_time(self):
        h2h, h2a = ab.second_half_goals([1, 0, 2], [0, 2, 2], [3, 0, 2], [1, 2, 4])
        assert h2h.tolist() == [2, 0, 0] and h2a.tolist() == [1, 0, 2]

    def test_full_time_below_half_time_raises_never_clips(self):
        # SECOND_HALF_TRAINING_QUERY's GREATEST(.., 0) would clip this to 0;
        # its FT >= HT filter drops it — so seeing one here is a defect.
        with pytest.raises(ValueError, match="below the half-time"):
            ab.second_half_goals([2], [0], [1], [0])


class TestAssembleFrame:
    def _scores(self):
        return pd.DataFrame(
            {
                "match_id": ["a", "b", "c", "d", "e"],
                "ht_home": [1, 0, 2, 1, 0],
                "ht_away": [0, 0, 1, 1, 0],
                "ft_home": [2, 1, 1, np.nan, 0],
                "ft_away": [1, 0, 1, np.nan, 0],
            }
        )

    def _h2(self, ids=("a", "b"), home=(1, 1), away=(1, 0)):
        return pd.DataFrame({"match_id": list(ids), "home_score": list(home), "away_score": list(away)})

    def test_join_carries_ht_2h_ft_and_audits_the_dropped_ht_rows(self):
        frame, audit = ab.assemble_frame(self._h2(), self._scores())
        assert frame["h2_home"].tolist() == [1, 1] and frame["h2_away"].tolist() == [1, 0]
        assert frame["ht_home"].tolist() == [1, 0] and frame["ft_home"].tolist() == [2, 1]
        assert audit["ht_rows"] == 5 and audit["second_half_rows"] == 2
        assert audit["dropped_ft_below_ht"] == ["c"]  # HT 2-1, FT 1-1
        assert audit["dropped_no_full_time"] == ["d"]
        assert audit["dropped_other"] == ["e"]  # e.g. a team row missing from the trainer's joins

    def test_query_targets_that_disagree_with_ft_minus_ht_raise(self):
        with pytest.raises(RuntimeError, match="!= FT - HT"):
            ab.assemble_frame(self._h2(home=(1, 0)), self._scores())

    def test_a_second_half_row_without_scores_raises(self):
        with pytest.raises(RuntimeError, match="no HT/FT score"):
            ab.assemble_frame(self._h2(ids=("a", "zz")), self._scores())


class TestRealised:
    def test_ht_ft_index_is_ht_outcome_times_3_plus_ft_outcome(self):
        t = ab.realised(_ev([(1, 0), (0, 0), (0, 1)], [(1, 2), (2, 0), (0, 3)]))
        assert t["ht_ft"].tolist() == [0 * 3 + 2, 1 * 3 + 0, 2 * 3 + 2]
        assert [ab.HTFT_KEYS[i] for i in t["ht_ft"]] == ["home_away", "draw_home", "away_away"]
        assert t["ht_1x2"].tolist() == [0, 1, 2]

    def test_second_half_markets_come_from_the_2h_goals_not_the_full_time(self):
        # HT 1-0 FT 1-2: the 2H alone is 0-2 (away) though home led at HT.
        t = ab.realised(_ev([(1, 0), (2, 2), (0, 0), (1, 1)], [(1, 2), (2, 2), (1, 0), (3, 2)]))
        assert t["h2_1x2"].tolist() == [2, 1, 0, 0]
        assert t["h2_over_0.5"].tolist() == [1, 0, 1, 1]
        assert t["h2_over_1.5"].tolist() == [1, 0, 0, 1]


class TestServePathReadout:
    def test_h2_markets_agree_with_the_served_matrix_readers(self):
        from predictors.market_derivation import build_dc_matrix, derive_soccer_halftime_markets

        P = build_dc_matrix(0.95, 0.7, -0.05, max_goals=10)
        m = ab.h2_markets(P)
        ref = derive_soccer_halftime_markets(P)
        assert m["h2_1x2"].tolist() == pytest.approx([ref["match_result_ht"][c] for c in ab.CLASSES], abs=1e-12)
        assert m["h2_over_0.5"] == pytest.approx(ref["over_under_ht"]["over_0.5"], abs=1e-12)
        assert m["h2_over_1.5"] == pytest.approx(ref["over_under_ht"]["over_1.5"], abs=1e-12)

    def test_htft_rows_sum_to_the_ht_marginal(self):
        from predictors.market_derivation import build_dc_matrix

        P_ht, P_2h = build_dc_matrix(0.7, 0.5, -0.03), build_dc_matrix(0.9, 0.7, 0.0)
        v = ab.htft_vector(P_ht, P_2h)
        assert v.sum() == pytest.approx(1.0)
        assert v.reshape(3, 3).sum(axis=1) == pytest.approx(ab.one_x_two(P_ht), abs=1e-9)

    def test_score_arm_scores_only_the_markets_an_arm_emits(self):
        truth = ab.realised(_ev([(1, 0)], [(1, 0)]))  # home_home, 2H 0-0
        s = ab.score_arm({"ht_ft": np.full((1, 9), 1 / 9)}, truth)
        assert set(s) == {"ht_ft"}
        assert s["ht_ft"].tolist() == pytest.approx([8 / 81 + (1 - 1 / 9) ** 2])
        full = ab.score_arm(_valid(), truth)
        assert full["h2_1x2"].tolist() == pytest.approx([0.4**2 + 0.7**2 + 0.3**2])  # 2H draw
        assert full["h2_over_0.5"].tolist() == pytest.approx([0.49])


class TestConditionalConstant:
    def test_shrinks_each_leagues_transition_toward_the_global_one(self):
        # global: from HT home -> FT home 3, FT draw 1; from HT draw -> FT draw 2.
        ht_idx = [0, 0, 0, 0, 1, 1]
        ft_idx = [0, 0, 0, 1, 1, 1]
        leagues = ["L1", "L1", "L2", "L2", "L1", "L2"]
        cond = ab.fit_conditional_htft(ht_idx, ft_idx, leagues, k=2.0)
        assert cond["global"][0] == pytest.approx([0.75, 0.25, 0.0])
        assert cond["global"][1] == pytest.approx([0.0, 1.0, 0.0])
        assert cond["global"][2] == pytest.approx([1 / 3] * 3)  # no HT-away rows: uniform, not NaN
        # L1 from HT home: counts (2, 0, 0) + 2 * (0.75, 0.25, 0) over 2 + 2
        assert cond["by_league"]["L1"][0] == pytest.approx([3.5 / 4, 0.5 / 4, 0.0])
        # L2 from HT home: counts (1, 1, 0)
        assert cond["by_league"]["L2"][0] == pytest.approx([2.5 / 4, 1.5 / 4, 0.0])
        for T in [cond["global"], *cond["by_league"].values()]:
            assert T.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])

    def test_weights_count_and_zero_weight_rows_vanish(self):
        cond = ab.fit_conditional_htft([0, 0], [0, 2], [None, None], weights=[1.0, 0.0])
        assert cond["global"][0] == pytest.approx([1.0, 0.0, 0.0])
        assert cond["by_league"] == {}
        with pytest.raises(ValueError, match="align"):
            ab.fit_conditional_htft([0, 0], [0, 2], [None, None], weights=[1.0])

    def test_prediction_keeps_the_ht_marginal_and_falls_back_for_unseen_leagues(self):
        cond = ab.fit_conditional_htft([0, 1, 2, 0], [0, 1, 2, 1], ["L1", "L1", "L1", "L1"], k=1.0)
        p_ht = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
        out = ab.predict_conditional_htft(cond, p_ht, ["L1", "unseen"])
        assert out.sum(axis=1) == pytest.approx([1.0, 1.0])
        assert out.reshape(2, 3, 3).sum(axis=2) == pytest.approx(p_ht)  # HT marginal untouched
        assert out[0].reshape(3, 3) == pytest.approx(p_ht[0][:, None] * cond["by_league"]["L1"])
        assert out[1].reshape(3, 3) == pytest.approx(p_ht[1][:, None] * cond["global"])


class TestDisqualification:
    def test_non_convergence_disqualifies_a_challenger_even_with_finite_probabilities(self):
        reason = ab.disqualification("pinned", probs=_valid(), train_result={"iterations": None})
        assert reason is not None and "did not converge" in reason
        assert "did not converge" in ab.disqualification("refit", train_result={})  # no key = not converged

    def test_converged_and_valid_is_not_disqualified(self):
        assert ab.disqualification("refit", train_result={"iterations": 30}) is None
        assert ab.disqualification("refit_frozen", probs=_valid(3)) is None

    def test_invalid_probabilities_disqualify_a_challenger(self):
        bad = {**_valid(2), "ht_ft": np.array([[1 / 9] * 9, [0.2] * 9])}
        assert "ht_ft" in ab.disqualification("refit", probs=bad)
        assert "h2_over_1.5" in ab.disqualification("refit", probs={**_valid(), "h2_over_1.5": np.array([np.nan])})

    def test_served_constants_and_the_ht_side_raise_instead(self):
        bad = {**_valid(), "h2_1x2": np.array([[0.6, 0.3, 0.3]])}
        for arm in ("served", "base_league", "base_cond_dw"):
            with pytest.raises(RuntimeError, match=arm):
                ab.disqualification(arm, probs=bad)
        with pytest.raises(RuntimeError, match="ht_side: fit did not converge"):
            ab.disqualification("ht_side", train_result={"iterations": None})

    def test_an_ht_ft_only_arm_is_checked_on_what_it_emits(self):
        assert ab.invalid_probabilities({"ht_ft": np.full((2, 9), 1 / 9)}) is None
        neg = np.full((1, 9), 1 / 9)
        neg[0, 0], neg[0, 1] = -0.1, 2 / 9 + 0.1  # negative mass, the row still sums to 1
        assert neg.sum() == pytest.approx(1.0)
        assert "non-finite, negative" in ab.invalid_probabilities({"ht_ft": neg})

    def test_invalid_matrices(self):
        good = np.full((2, 2), 0.25)
        assert ab.invalid_matrices([good, good]) is None
        assert "1 of 2" in ab.invalid_matrices([good, np.full((2, 2), np.nan)])


class TestArms:
    def test_fitted_arms_use_exactly_the_trainers_configs(self):
        from predictors.model_config import DIXON_COLES_CONFIG, LEGACY_DIXON_COLES_HYPERPARAMETERS

        c = ab.arm_configs()
        assert c["ht_side"].name == "dixon_coles_ht_soccer"
        assert c["ht_side"].hyperparameters == DIXON_COLES_CONFIG.hyperparameters
        assert c["refit"].hyperparameters == DIXON_COLES_CONFIG.hyperparameters
        assert c["pinned"].name == c["refit"].name == "dixon_coles_h2_soccer"
        assert c["pinned"].hyperparameters == {
            **DIXON_COLES_CONFIG.hyperparameters,
            **LEGACY_DIXON_COLES_HYPERPARAMETERS,
        }

    def test_decayed_arms_meet_the_decayed_constants(self):
        assert ab.matched_constant("refit") == "base_league_dw" and ab.matched_conditional("refit") == "base_cond_dw"
        assert ab.matched_constant("refit_frozen") == "base_league_dw"
        assert ab.matched_constant("served") == "base_league" and ab.matched_conditional("pinned") == "base_cond"


class TestCalibration:
    def test_htft_calibration_per_cell(self):
        proba = np.zeros((2, 9))
        proba[0, 4] = proba[1, 0] = 1.0  # draw_draw, home_home
        c = ab.htft_calibration(proba, np.array([4, 4]))
        assert c["pred"]["draw_draw"] == pytest.approx(0.5) and c["real"]["draw_draw"] == pytest.approx(1.0)
        assert c["max_abs_gap"] == pytest.approx(0.5) and c["worst_cell"] in ("home_home", "draw_draw")
        assert c["argmax_hit"] == pytest.approx(0.5)
