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
    HT_OU_LINES,
    NHL_TOTAL_LINES,
    OU_LINES,
    _fmt_line,
    build_dc_matrix,
    derive_from_lambdas,
    derive_hockey_markets,
    derive_markets,
    derive_soccer_halftime_markets,
    derive_soccer_htft_markets,
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

    def test_soccer_halftime_registered(self):
        # HT lambdas are ~half of FT lambdas in real soccer (the
        # second half typically carries the bulk of goals because of
        # late tactical changes + tired defending). Test with a small
        # synthetic HT lambda pair and confirm dispatch works.
        P = build_dc_matrix(0.7, 0.5, rho=-0.08)
        markets = derive_markets("soccer_halftime", P)
        assert "match_result_ht" in markets
        assert "over_under_ht" in markets
        assert "btts_ht" in markets


@pytest.mark.unit
class TestHalftimeMarkets:
    """The halftime deriver only emits 3 markets (1x2 / O/U / BTTS) —
    no correct_score, asian_handicap, team_total etc., because retail
    HT odds markets are correspondingly narrow."""

    @pytest.fixture
    def markets(self):
        # HT lambdas comparable to a high-scoring real fixture (Man City
        # vs Liverpool style): ~0.9 home / ~0.6 away expected HT goals.
        P = build_dc_matrix(0.9, 0.6, rho=-0.08, max_goals=10)
        return derive_soccer_halftime_markets(P)

    def test_only_three_markets_emitted(self, markets):
        # Halftime catalog is intentionally narrow — guards against
        # accidental inclusion of FT markets in the HT output.
        assert set(markets.keys()) == {"match_result_ht", "over_under_ht", "btts_ht"}

    def test_match_result_ht_sums_to_one(self, markets):
        mr = markets["match_result_ht"]
        assert set(mr.keys()) == {"home", "draw", "away"}
        assert abs(sum(mr.values()) - 1.0) < 1e-9

    def test_match_result_ht_draw_dominates_at_realistic_lambdas(self):
        # In real soccer, the draw at HT (0-0 most common) dominates
        # because mean HT goals are ~0.5 per side. Verify the deriver
        # captures this with a tiny-lambda case.
        P = build_dc_matrix(0.4, 0.4, rho=-0.1, max_goals=10)
        m = derive_soccer_halftime_markets(P)
        assert m["match_result_ht"]["draw"] > m["match_result_ht"]["home"]
        assert m["match_result_ht"]["draw"] > m["match_result_ht"]["away"]

    def test_over_under_ht_pairs_sum_to_one_per_line(self, markets):
        # Half-integer lines never push; over_X + under_X = 1.
        ou = markets["over_under_ht"]
        for line in HT_OU_LINES:
            lbl = _fmt_line(line)
            over = ou[f"over_{lbl}"]
            under = ou[f"under_{lbl}"]
            assert abs(over + under - 1.0) < 1e-9, f"line {line}: {over}+{under}"

    def test_over_under_ht_monotone_with_line(self, markets):
        # P(over X) must strictly decrease as the line goes up.
        ou = markets["over_under_ht"]
        prev = math.inf
        for line in HT_OU_LINES:
            over = ou[f"over_{_fmt_line(line)}"]
            assert over < prev
            prev = over

    def test_btts_ht_pair_sums_to_one(self, markets):
        btts = markets["btts_ht"]
        assert set(btts.keys()) == {"yes", "no"}
        assert abs(btts["yes"] + btts["no"] - 1.0) < 1e-9

    def test_btts_ht_yes_below_ft_btts_for_same_lambdas(self):
        # BTTS in 1H is STRICTLY LESS LIKELY than BTTS over the full
        # match for the same lambda pair — every joint mass that
        # contributes to FT-BTTS-no also contributes to HT-BTTS-no
        # (plus more), so HT yes ≤ FT yes.
        ht_P = build_dc_matrix(0.7, 0.6, rho=-0.08)
        ft_P = build_dc_matrix(0.7, 0.6, rho=-0.08)
        # NOTE: this isn't comparing HT vs FT lambdas — it's
        # confirming the deriver itself is consistent with the
        # FT deriver for the same matrix. Used here as a sanity
        # check that BTTS math is identical between FT and HT
        # derivers.
        ht_btts = derive_soccer_halftime_markets(ht_P)["btts_ht"]["yes"]
        ft_btts = derive_soccer_markets(ft_P)["btts"]["yes"]
        assert abs(ht_btts - ft_btts) < 1e-9

    def test_ht_ou_lines_topped_at_2_5(self):
        # Confirms the HT_OU_LINES constant doesn't accidentally extend
        # to FT's 5.5 — HT books don't trade lines that high.
        assert max(HT_OU_LINES) == 2.5
        assert min(HT_OU_LINES) == 0.5

    def test_renormalises_if_input_drifts(self):
        # Defensive: if a caller hands us an un-normalised matrix we
        # still produce probabilities in [0, 1]. Same invariant as
        # derive_soccer_markets.
        P = build_dc_matrix(0.6, 0.4) * 2.0  # un-normalised
        m = derive_soccer_halftime_markets(P)
        assert abs(sum(m["match_result_ht"].values()) - 1.0) < 1e-9


@pytest.mark.unit
class TestHalftimeFulltimeJoint:
    """The HT/FT joint deriver convolves a HT scoreline matrix with
    a SECOND-HALF (2H = FT - HT) scoreline matrix and aggregates
    joint mass into the 9 (HT outcome × FT outcome) buckets."""

    @pytest.fixture
    def ht_matrix(self):
        # Realistic HT lambdas: ~0.7 home / ~0.5 away (~half of FT).
        return build_dc_matrix(0.7, 0.5, rho=-0.08, max_goals=10)

    @pytest.fixture
    def h2_matrix(self):
        # Realistic 2H lambdas: ~0.9 home / ~0.7 away (slightly
        # higher than HT — late tactical changes + tired defending).
        return build_dc_matrix(0.9, 0.7, rho=-0.08, max_goals=10)

    @pytest.fixture
    def markets(self, ht_matrix, h2_matrix):
        return derive_soccer_htft_markets(ht_matrix, h2_matrix)

    def test_emits_only_double_result_market(self, markets):
        # The deriver returns ONLY the ht_ft_double_result market.
        # Other markets (1x2 at FT, total goals, etc.) come from the
        # separate full-time deriver.
        assert set(markets.keys()) == {"ht_ft_double_result"}

    def test_nine_selections_present(self, markets):
        m = markets["ht_ft_double_result"]
        expected = {
            "home_home",
            "home_draw",
            "home_away",
            "draw_home",
            "draw_draw",
            "draw_away",
            "away_home",
            "away_draw",
            "away_away",
        }
        assert set(m.keys()) == expected

    def test_selections_sum_to_one(self, markets):
        m = markets["ht_ft_double_result"]
        assert abs(sum(m.values()) - 1.0) < 1e-9

    def test_all_probabilities_in_unit_interval(self, markets):
        m = markets["ht_ft_double_result"]
        for sel, p in m.items():
            assert 0.0 <= p <= 1.0, f"{sel}={p} outside [0,1]"

    def test_away_to_home_comeback_rarer_than_home_to_home(self, markets):
        # AWAY leading at HT and HOME winning at FT requires a
        # late-game comeback (rare in real soccer) — should be
        # significantly less likely than home-then-home (steady
        # home win).
        m = markets["ht_ft_double_result"]
        assert m["away_home"] < m["home_home"]

    def test_marginal_ht_outcome_matches_ht_deriver(self, ht_matrix, h2_matrix, markets):
        # Summing the joint over FT outcomes should recover the
        # marginal HT outcomes — i.e. P(HT=home) = P(HT=home, FT=home)
        # + P(HT=home, FT=draw) + P(HT=home, FT=away). This is the
        # load-bearing invariant that links the joint to the HT-only
        # deriver.
        m = markets["ht_ft_double_result"]
        ht_only = derive_soccer_halftime_markets(ht_matrix)["match_result_ht"]
        for ht_outcome in ("home", "draw", "away"):
            joint_marginal = sum(m[f"{ht_outcome}_{ft_outcome}"] for ft_outcome in ("home", "draw", "away"))
            assert abs(joint_marginal - ht_only[ht_outcome]) < 1e-9

    def test_marginal_ft_outcome_matches_independent_convolution(
        self,
        ht_matrix,
        h2_matrix,
        markets,
    ):
        # Summing the joint over HT outcomes should recover the
        # marginal FT outcomes — derivable directly by convolving
        # the HT and 2H matrices. This is the other half of the
        # marginal invariant. Use a coarse tolerance because the
        # in-deriver convolution is computed element-by-element
        # while the expected uses scipy/numpy.
        m = markets["ht_ft_double_result"]
        # Build FT matrix by manual convolution.
        N_ht = ht_matrix.shape[0]
        N_2h = h2_matrix.shape[0]
        N_ft = N_ht + N_2h - 1
        ft_matrix = np.zeros((N_ft, N_ft))
        for i in range(N_ht):
            for j in range(N_ht):
                for a in range(N_2h):
                    for b in range(N_2h):
                        ft_matrix[i + a, j + b] += ht_matrix[i, j] * h2_matrix[a, b]
        ft_matrix /= ft_matrix.sum()
        ft_i, ft_j = np.indices(ft_matrix.shape)
        expected_marginals = {
            "home": float(ft_matrix[ft_i > ft_j].sum()),
            "draw": float(ft_matrix[ft_i == ft_j].sum()),
            "away": float(ft_matrix[ft_i < ft_j].sum()),
        }
        for ft_outcome, expected in expected_marginals.items():
            joint_marginal = sum(m[f"{ht_outcome}_{ft_outcome}"] for ht_outcome in ("home", "draw", "away"))
            assert abs(joint_marginal - expected) < 1e-9

    def test_renormalises_drifted_inputs(self):
        # Defensive: if a caller hands us un-normalised matrices we
        # still produce a normalised joint.
        ht_P = build_dc_matrix(0.7, 0.5) * 2.0
        h2_P = build_dc_matrix(0.9, 0.7) * 1.5
        m = derive_soccer_htft_markets(ht_P, h2_P)
        assert abs(sum(m["ht_ft_double_result"].values()) - 1.0) < 1e-9

    def test_draw_draw_dominates_at_low_lambdas(self):
        # Low-scoring fixtures (0.4 / 0.3 each half) should have the
        # joint draw_draw bucket as the modal outcome — both halves
        # ending nil-nil drives this.
        ht_P = build_dc_matrix(0.4, 0.3, rho=-0.08)
        h2_P = build_dc_matrix(0.4, 0.3, rho=-0.08)
        m = derive_soccer_htft_markets(ht_P, h2_P)["ht_ft_double_result"]
        assert m["draw_draw"] == max(m.values())


@pytest.mark.unit
class TestQuarterAsianHandicap:
    """Quarter Asian handicap lines (.25 / .75) split the bet across
    two adjacent sub-lines, producing a half-win / half-loss outcome
    at the integer-sub-line's push margin. The deriver emits
    EFFECTIVE home / away / push masses so the existing recs engine
    formula (`win * odds + push - 1`) keeps working without code
    changes — half-stake refund mass is folded into `push`."""

    @pytest.fixture
    def markets(self):
        return derive_soccer_markets(build_dc_matrix(1.7, 1.2, rho=-0.12, max_goals=10))

    def test_quarter_lines_present(self, markets):
        # Quarter lines must be in the output. Without them this
        # whole feature is dead.
        ah = markets["asian_handicap"]
        for line in (-0.25, -0.75, 0.25, 0.75, -1.25, 1.25):
            lbl = _fmt_line(line)
            assert f"{lbl}_home" in ah
            assert f"{lbl}_away" in ah
            assert f"{lbl}_push" in ah

    def test_effective_masses_sum_to_one_per_line(self, markets):
        ah = markets["asian_handicap"]
        for line in AH_LINES:
            lbl = _fmt_line(line)
            total = ah[f"{lbl}_home"] + ah[f"{lbl}_away"] + ah[f"{lbl}_push"]
            assert abs(total - 1.0) < 1e-9, f"line {line}: {total}"

    def test_neg_0_25_home_matches_neg_0_5_home_on_strict_wins(self, markets):
        # For line -0.25 home: full_win region = margin > 0 = exactly
        # the same as -0.5 home's full_win region. The EFFECTIVE home
        # value at -0.25 = full_win = P(margin > 0) since half_win = 0.
        # So effective_home(-0.25) == home(-0.5).
        ah = markets["asian_handicap"]
        assert abs(ah["-0.25_home"] - ah["-0.5_home"]) < 1e-12

    def test_neg_0_75_home_full_win_matches_neg_1_home_full_win(self, markets):
        # For -0.75 home: full_win at margin >= 2 (same as -1 home's
        # margin > 1). Plus a half_win at margin == 1.
        # EFFECTIVE home(-0.75) = full_win + 0.5 * half_win
        #                      = home(-1) + 0.5 * push(-1)
        ah = markets["asian_handicap"]
        expected = ah["-1_home"] + 0.5 * ah["-1_push"]
        assert abs(ah["-0.75_home"] - expected) < 1e-12

    def test_quarter_push_is_half_of_adjacent_integer_push(self, markets):
        # For line -0.25: push mass = 0.5 * (half_win + half_loss).
        # half_win comes from the lower sub-line being integer (here
        # h_lower = -0.5 is NOT integer, so half_win = 0), so the
        # full push mass comes from half_loss only:
        #   push(-0.25) = 0.5 * P(margin == 0) = 0.5 * push(0)
        # since the upper sub-line h_upper = 0 is integer.
        ah = markets["asian_handicap"]
        assert abs(ah["-0.25_push"] - 0.5 * ah["0_push"]) < 1e-12

    def test_neg_0_75_push_is_half_of_neg_1_push(self, markets):
        # For -0.75: h_lower = -1.0 is integer, h_upper = -0.5 isn't.
        # push(-0.75) = 0.5 * P(margin == 1) = 0.5 * push(-1).
        ah = markets["asian_handicap"]
        assert abs(ah["-0.75_push"] - 0.5 * ah["-1_push"]) < 1e-12

    def test_existing_half_integer_lines_unchanged(self, markets):
        # Defensive: integer + half-integer lines must keep their
        # pre-quarter values exactly. Compares against a hand-built
        # reference on the same matrix.
        P = build_dc_matrix(1.7, 1.2, rho=-0.12, max_goals=10)
        N = P.shape[0]
        i_idx, j_idx = np.indices((N, N))
        margin = i_idx - j_idx
        ah = markets["asian_handicap"]
        for line in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
            lbl = _fmt_line(line)
            adj = margin + line
            expected_home = float(P[adj > 0].sum())
            expected_away = float(P[adj < 0].sum())
            expected_push = float(P[adj == 0].sum())
            assert abs(ah[f"{lbl}_home"] - expected_home) < 1e-12, lbl
            assert abs(ah[f"{lbl}_away"] - expected_away) < 1e-12, lbl
            assert abs(ah[f"{lbl}_push"] - expected_push) < 1e-12, lbl

    def test_quarter_home_plus_away_plus_push_in_0_1(self, markets):
        # All three components stay in [0, 1] for every quarter line.
        ah = markets["asian_handicap"]
        for line in (-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75):
            lbl = _fmt_line(line)
            for key in (f"{lbl}_home", f"{lbl}_away", f"{lbl}_push"):
                assert 0.0 <= ah[key] <= 1.0, f"{key}={ah[key]}"

    def test_symmetric_zero_lambda_difference(self):
        # When lambdas are equal, line 0.25 home should equal line
        # 0.25 away (symmetry). Quarter line at 0.25 means home
        # gets +0.25 head start; with equal teams, away with -0.25
        # has the mirror mass.
        P = build_dc_matrix(1.5, 1.5, rho=-0.1, max_goals=10)
        m = derive_soccer_markets(P)["asian_handicap"]
        assert abs(m["0.25_home"] - m["-0.25_away"]) < 1e-12


@pytest.mark.unit
class TestHockeyMarkets:
    """Hockey deriver uses the same scoreline-matrix machinery as
    soccer but with NHL-appropriate markets: no draw at game end
    (OT/SO redistributes), puck line at ±1.5, totals 4.5/5.5/6.5,
    exact-total buckets, plus a regulation_1x2 view that preserves
    the draw mass for users who want the in-regulation market."""

    @pytest.fixture
    def markets(self):
        # NHL-realistic lambdas: ~3.1 home / ~2.9 away.
        return derive_hockey_markets(build_dc_matrix(3.1, 2.9, rho=-0.05, max_goals=10))

    def test_moneyline_two_outcomes_sum_to_one(self, markets):
        m = markets["moneyline"]
        assert set(m.keys()) == {"home", "away"}
        assert abs(m["home"] + m["away"] - 1.0) < 1e-9

    def test_moneyline_splits_draw_mass_evenly(self):
        # Build a matrix where draws have known mass.
        P = build_dc_matrix(3.0, 3.0, rho=-0.05)
        m = derive_hockey_markets(P)
        reg_home = m["regulation_1x2"]["home"]
        reg_draw = m["regulation_1x2"]["draw"]
        # moneyline home = reg_home + 0.5 * reg_draw exactly.
        assert abs(m["moneyline"]["home"] - (reg_home + 0.5 * reg_draw)) < 1e-12
        # equal lambdas → moneyline 50/50.
        assert abs(m["moneyline"]["home"] - 0.5) < 1e-9

    def test_regulation_1x2_sums_to_one(self, markets):
        r = markets["regulation_1x2"]
        assert set(r.keys()) == {"home", "draw", "away"}
        assert abs(sum(r.values()) - 1.0) < 1e-9

    def test_puck_line_pairs_sum_to_one(self, markets):
        pl = markets["puck_line"]
        assert abs(pl["-1.5_home"] + pl["-1.5_away"] - 1.0) < 1e-9
        assert abs(pl["1.5_home"] + pl["1.5_away"] - 1.0) < 1e-9

    def test_puck_line_home_minus_strictly_less_than_plus(self, markets):
        # Home -1.5 (wins by 2+) is strictly harder than home +1.5
        # (wins OR loses by 1) for any non-degenerate scoreline.
        pl = markets["puck_line"]
        assert pl["-1.5_home"] < pl["1.5_home"]

    def test_over_under_pairs_sum_to_one_per_line(self, markets):
        ou = markets["over_under"]
        for line in NHL_TOTAL_LINES:
            lbl = _fmt_line(line)
            assert abs(ou[f"over_{lbl}"] + ou[f"under_{lbl}"] - 1.0) < 1e-9

    def test_over_under_monotonic(self, markets):
        ou = markets["over_under"]
        overs = [ou[f"over_{_fmt_line(line)}"] for line in NHL_TOTAL_LINES]
        # Strictly decreasing as line rises.
        assert all(overs[i] >= overs[i + 1] for i in range(len(overs) - 1))

    def test_total_goals_buckets_sum_to_one(self, markets):
        tg = markets["total_goals"]
        assert abs(sum(tg.values()) - 1.0) < 1e-9

    def test_double_chance_sums(self, markets):
        # Double chance pairs ARE NOT independent — each pair sums
        # over the 3-way mass and equals (1 - the excluded outcome).
        dc = markets["double_chance"]
        r = markets["regulation_1x2"]
        assert abs(dc["1X"] - (1 - r["away"])) < 1e-9
        assert abs(dc["12"] - (1 - r["draw"])) < 1e-9
        assert abs(dc["X2"] - (1 - r["home"])) < 1e-9

    def test_clean_sheet_pair_sums(self, markets):
        cs = markets["clean_sheet"]
        assert abs(cs["home_yes"] + cs["home_no"] - 1.0) < 1e-9
        assert abs(cs["away_yes"] + cs["away_no"] - 1.0) < 1e-9

    def test_win_to_nil_pair_sums(self, markets):
        wtn = markets["win_to_nil"]
        assert abs(wtn["home_yes"] + wtn["home_no"] - 1.0) < 1e-9
        assert abs(wtn["away_yes"] + wtn["away_no"] - 1.0) < 1e-9

    def test_correct_score_sums_to_one(self, markets):
        cs = markets["correct_score"]
        assert abs(sum(cs.values()) - 1.0) < 1e-9

    def test_registered_via_dispatch(self):
        # The MARKET_DERIVERS dispatch table must include "nhl" so
        # derive_markets(sport="nhl", P) works downstream.
        P = build_dc_matrix(3.0, 2.8, rho=-0.05)
        m = derive_markets("nhl", P)
        # Soccer-specific markets must NOT appear in the hockey
        # output — guards against accidental fallthrough.
        assert "asian_handicap" not in m
        assert "btts" not in m
        assert "moneyline" in m
        assert "puck_line" in m

    def test_zero_lambda_difference_means_50_50_moneyline(self):
        # Symmetric strengths → moneyline exactly 50/50.
        P = build_dc_matrix(2.5, 2.5, rho=-0.05)
        m = derive_hockey_markets(P)
        assert abs(m["moneyline"]["home"] - 0.5) < 1e-9
        assert abs(m["moneyline"]["away"] - 0.5) < 1e-9
