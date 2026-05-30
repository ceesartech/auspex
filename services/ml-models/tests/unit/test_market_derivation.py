"""Unit tests for the analytic market-derivation engine.

Pure-math tests over the joint scoreline matrix: invariants (which markets
must sum to 1), over/under monotonicity, Asian-handicap push masses, golden
values against closed-form references, that build_dc_matrix reproduces the
trained Dixon-Coles model's own score matrix, and that reconciliation hits a
target 1x2 while preserving within-region shape.
"""

import math

import numpy as np
import pytest
from predictors.market_derivation import (
    AH_LINES,
    OU_LINES,
    _fmt_line,
    build_dc_matrix,
    derive_from_lambdas,
    derive_markets,
    derive_soccer_markets,
    reconcile_matrix_to_1x2,
)


@pytest.mark.unit
class TestBuildMatrix:
    def test_normalized(self):
        P = build_dc_matrix(1.5, 1.1, rho=-0.13, max_goals=10)
        assert P.shape == (11, 11)
        assert P.min() >= 0
        assert abs(P.sum() - 1.0) < 1e-12

    def test_rho_zero_is_independent(self):
        P = build_dc_matrix(1.4, 1.2, rho=0.0, max_goals=8)
        ph = P.sum(axis=1)
        pa = P.sum(axis=0)
        assert np.allclose(P, np.outer(ph, pa), atol=1e-12)

    def test_higher_grid_barely_changes_tail_market(self):
        # Justifies max_goals=10: extending the grid changes O/U 5.5 negligibly.
        m10 = derive_soccer_markets(build_dc_matrix(1.8, 1.5, -0.1, max_goals=10))
        m14 = derive_soccer_markets(build_dc_matrix(1.8, 1.5, -0.1, max_goals=14))
        assert abs(m10["over_under"]["over_5.5"] - m14["over_under"]["over_5.5"]) < 1e-3


@pytest.mark.unit
class TestInvariants:
    @pytest.fixture
    def markets(self):
        return derive_soccer_markets(build_dc_matrix(1.7, 1.2, rho=-0.12, max_goals=10))

    def test_sum_to_one_markets(self, markets):
        for mt in [
            "1x2",
            "draw_no_bet",
            "btts",
            "odd_even",
            "total_goals",
            "winning_margin",
            "result_btts",
            "result_over_under",
            "correct_score",
        ]:
            assert abs(sum(markets[mt].values()) - 1.0) < 1e-9, mt

    def test_double_chance_relations(self, markets):
        o = markets["1x2"]
        dc = markets["double_chance"]
        assert abs(dc["1X"] - (1 - o["away"])) < 1e-9
        assert abs(dc["12"] - (1 - o["draw"])) < 1e-9
        assert abs(dc["X2"] - (1 - o["home"])) < 1e-9

    def test_over_under_per_line_and_monotonic(self, markets):
        ou = markets["over_under"]
        overs = []
        for line in OU_LINES:
            lbl = _fmt_line(line)
            assert abs(ou[f"over_{lbl}"] + ou[f"under_{lbl}"] - 1.0) < 1e-9
            overs.append(ou[f"over_{lbl}"])
        assert all(overs[i] >= overs[i + 1] - 1e-12 for i in range(len(overs) - 1))

    def test_asian_handicap_masses(self, markets):
        ah = markets["asian_handicap"]
        for line in AH_LINES:
            lbl = _fmt_line(line)
            total = ah[f"{lbl}_home"] + ah[f"{lbl}_away"] + ah[f"{lbl}_push"]
            assert abs(total - 1.0) < 1e-9
            assert 0.0 <= ah[f"{lbl}_push"] <= 1.0
        # Half-integer lines never push; integer lines may.
        assert ah["-0.5_push"] == 0.0
        assert ah["0_push"] > 0.0

    def test_team_total_pairs(self, markets):
        tt = markets["team_total"]
        for side in ("home", "away"):
            for lbl in ("0.5", "1.5", "2.5"):
                assert abs(tt[f"{side}_over_{lbl}"] + tt[f"{side}_under_{lbl}"] - 1.0) < 1e-9

    def test_probabilities_in_range(self, markets):
        for mt, sels in markets.items():
            for k, v in sels.items():
                assert -1e-9 <= v <= 1.0 + 1e-9, (mt, k, v)


@pytest.mark.unit
class TestGoldenValues:
    def test_deterministic_matrix(self):
        # Mass only on (2,1), (1,1), (0,2) — every market is a closed form.
        N = 5
        P = np.zeros((N, N))
        P[2, 1] = 0.5  # home win, both score, total 3 (odd)
        P[1, 1] = 0.3  # draw, both score, total 2 (even)
        P[0, 2] = 0.2  # away win, home blanked, total 2 (even)
        m = derive_soccer_markets(P, top_n_scores=3)

        assert abs(m["1x2"]["home"] - 0.5) < 1e-12
        assert abs(m["1x2"]["draw"] - 0.3) < 1e-12
        assert abs(m["1x2"]["away"] - 0.2) < 1e-12
        assert abs(m["btts"]["yes"] - 0.8) < 1e-12
        assert abs(m["btts"]["no"] - 0.2) < 1e-12
        assert abs(m["total_goals"]["3"] - 0.5) < 1e-12
        assert abs(m["total_goals"]["2"] - 0.5) < 1e-12
        assert abs(m["odd_even"]["odd"] - 0.5) < 1e-12
        assert abs(m["correct_score"]["2-1"] - 0.5) < 1e-12
        assert abs(m["correct_score"]["other"]) < 1e-12
        # Clean sheet: home keeps it when away scores 0 — never here.
        assert abs(m["clean_sheet"]["home_yes"]) < 1e-12
        # Away keeps a clean sheet on (0,2) → away scores while home blanked.
        assert abs(m["clean_sheet"]["away_yes"] - 0.2) < 1e-12
        # Asian handicap line 0: push = draw mass.
        assert abs(m["asian_handicap"]["0_push"] - 0.3) < 1e-12

    def test_reference_independent_poisson(self):
        lam_h, lam_a = 1.6, 1.1
        P = build_dc_matrix(lam_h, lam_a, rho=0.0, max_goals=12)
        m = derive_soccer_markets(P)
        # P(total=0) = e^-(lh+la)
        assert abs(m["total_goals"]["0"] - math.exp(-(lam_h + lam_a))) < 1e-4
        # BTTS yes = (1-e^-lh)(1-e^-la) under independence.
        exp_yes = (1 - math.exp(-lam_h)) * (1 - math.exp(-lam_a))
        assert abs(m["btts"]["yes"] - exp_yes) < 1e-4


@pytest.mark.unit
class TestReconciliation:
    def test_hits_target(self):
        P = build_dc_matrix(1.5, 1.0, rho=-0.1, max_goals=10)
        m = derive_soccer_markets(reconcile_matrix_to_1x2(P, (0.55, 0.25, 0.20)))
        assert abs(m["1x2"]["home"] - 0.55) < 1e-9
        assert abs(m["1x2"]["draw"] - 0.25) < 1e-9
        assert abs(m["1x2"]["away"] - 0.20) < 1e-9

    def test_preserves_within_region_ratio(self):
        P = build_dc_matrix(1.5, 1.0, rho=-0.1, max_goals=10)
        r_before = P[2, 0] / P[3, 1]  # two home-win cells
        P2 = reconcile_matrix_to_1x2(P, (0.55, 0.25, 0.20))
        r_after = P2[2, 0] / P2[3, 1]
        assert abs(r_before - r_after) < 1e-9

    def test_target_normalized_internally(self):
        P = build_dc_matrix(1.4, 1.4, rho=0.0, max_goals=8)
        m = derive_soccer_markets(reconcile_matrix_to_1x2(P, (2.0, 1.0, 1.0)))
        assert abs(m["1x2"]["home"] - 0.5) < 1e-9

    def test_zero_region_guard(self):
        N = 4
        P = np.zeros((N, N))
        P[2, 0] = 0.6  # home win
        P[1, 1] = 0.4  # draw — no away-win mass at all
        P2 = reconcile_matrix_to_1x2(P, (0.5, 0.3, 0.2))
        assert np.isfinite(P2).all()


@pytest.mark.unit
class TestBuilderMatchesModel:
    def test_build_matches_score_matrix(self, dixon_coles_config, sample_match_data):
        from predictors.poisson_models import DixonColesPredictor

        dc = DixonColesPredictor(dixon_coles_config)
        dc.train(sample_match_data)
        teams = sorted(dc.team_attack.keys())[:2]
        h_lam, a_lam = dc.lambdas_for_match(teams[0], teams[1])

        ref = dc._score_matrix(h_lam, a_lam)  # the model's own matrix (max_goals=4)
        got = build_dc_matrix(h_lam, a_lam, dc.rho, max_goals=dc.max_goals)
        assert got.shape == ref.shape
        assert np.allclose(got, ref, atol=1e-10)


@pytest.mark.unit
class TestDispatch:
    def test_soccer(self):
        P = build_dc_matrix(1.3, 1.1, rho=-0.1)
        assert "1x2" in derive_markets("soccer", P)

    def test_unknown_sport_raises(self):
        P = build_dc_matrix(1.3, 1.1)
        with pytest.raises(ValueError):
            derive_markets("cricket", P)

    def test_derive_from_lambdas_reconciles(self):
        m = derive_from_lambdas(1.5, 1.0, -0.1, target_1x2=(0.5, 0.3, 0.2))
        assert abs(m["1x2"]["home"] - 0.5) < 1e-9
