"""Unit tests for predictors.hockey_regulation — the NHL regulation-scoreline
structure: regulation scores from final scores, the margin tilt, the
overtime resolution, the payload's validation, the joint maximum-likelihood
fit (it recovers known parameters from simulated games and leaves a plain
Poisson alone), and derive_hockey_markets' two-matrix mode plus the
DixonColesPredictor round trip that keeps every legacy artifact unchanged."""

import json

import numpy as np
import pandas as pd
import pytest
from predictors import hockey_regulation as hr
from predictors.market_derivation import MAX_GOALS_DERIVE, build_dc_matrix, derive_hockey_markets, derive_markets
from predictors.model_config import DIXON_COLES_CONFIG, ModelConfig
from predictors.poisson_models import DixonColesPredictor

CAP = MAX_GOALS_DERIVE


def _cfg(**hp) -> ModelConfig:
    from dataclasses import replace

    return replace(
        DIXON_COLES_CONFIG, name="dixon_coles_nhl", hyperparameters={**DIXON_COLES_CONFIG.hyperparameters, **hp}
    )


@pytest.mark.unit
class TestRegulationScores:
    def test_overtime_games_drop_the_winners_goal(self):
        h, a = hr.regulation_scores([4, 2, 3, 1], [3, 3, 1, 4], [True, True, False, False])
        assert h.tolist() == [3, 2, 3, 1] and a.tolist() == [3, 2, 1, 4]

    @pytest.mark.parametrize(
        "home, away, past",
        [([5], [3], [True]), ([3], [3], [False]), ([2], [2], [True])],
        ids=["two-goal OT game", "tied regulation-decided game", "tied OT game"],
    )
    def test_contradictions_raise(self, home, away, past):
        with pytest.raises(ValueError, match="not a regulation score"):
            hr.regulation_scores(home, away, past)

    def test_misaligned_inputs_raise(self):
        with pytest.raises(ValueError, match="aligned"):
            hr.regulation_scores([1, 2], [0], [False])


@pytest.mark.unit
class TestTiltAndOvertime:
    def test_no_buckets_is_the_identity(self):
        P = build_dc_matrix(3.1, 2.8, 0.0, max_goals=CAP)
        assert np.allclose(hr.tilt(P, hr.margin_factor_grid((), (), CAP)), P)

    def test_a_tie_factor_moves_mass_onto_the_diagonal_only_by_ratio(self):
        P = build_dc_matrix(3.1, 2.8, 0.0, max_goals=CAP)
        R = hr.tilt(P, hr.margin_factor_grid((0,), (np.log(1.5),), CAP))
        assert abs(R.sum() - 1.0) < 1e-12
        d = np.trace(P)
        assert np.trace(R) == pytest.approx(1.5 * d / (1 + 0.5 * d))
        off = ~np.eye(CAP + 1, dtype=bool)
        assert np.allclose(R[off] / P[off], (R[off] / P[off])[0])  # off-diagonal shape untouched

    def test_factor_grid_by_absolute_margin(self):
        G = hr.margin_factor_grid((0, 2), (0.3, -0.4), 4)
        assert G[1, 1] == pytest.approx(np.exp(0.3)) and G[3, 1] == G[1, 3] == pytest.approx(np.exp(-0.4))
        assert G[2, 1] == 1.0 and G[4, 0] == 1.0
        with pytest.raises(ValueError, match="log-factor"):
            hr.margin_factor_grid((0, 1), (0.1,), 4)

    def test_resolve_overtime_moves_every_tie_one_goal_up(self):
        R = hr.tilt(build_dc_matrix(3.0, 2.7, 0.0, max_goals=CAP), hr.margin_factor_grid((0,), (0.4,), CAP))
        F = hr.resolve_overtime(R, 0.53)
        assert np.trace(F) == 0.0 and abs(F.sum() - 1.0) < 1e-12
        assert F[3, 2] == pytest.approx(R[3, 2] + 0.53 * R[2, 2])
        assert F[2, 3] == pytest.approx(R[2, 3] + 0.47 * R[2, 2])
        i, j = np.indices(F.shape)
        # E[final total] = E[regulation total] + P(regulation tie), but for the
        # corner cell, which moves one goal DOWN (2 * R[cap, cap], ~1e-6 here).
        corner = 2.0 * R[CAP, CAP]
        assert (F * (i + j)).sum() == pytest.approx((R * (i + j)).sum() + np.trace(R) - corner, abs=1e-12)
        # a stack resolves row by row
        assert np.allclose(hr.resolve_overtime(np.stack([R, R]), 0.53)[1], F)

    def test_the_corner_tie_stays_on_the_grid(self):
        R = np.zeros((3, 3))
        R[2, 2] = 1.0
        F = hr.resolve_overtime(R, 0.25)
        assert F[2, 1] == 0.25 and F[1, 2] == 0.75 and np.trace(F) == 0.0


@pytest.mark.unit
class TestStructurePayload:
    def test_round_trip(self):
        s = hr.RegulationStructure((0, 1, 2), (0.2, -0.6, -0.2), 0.52)
        payload = json.loads(json.dumps(s.to_payload()))
        assert hr.RegulationStructure.from_payload(payload) == s

    @pytest.mark.parametrize(
        "payload, match",
        [
            ("x", "not an object"),
            ({"version": 2, "margin_buckets": [], "margin_log_factors": [], "ot_home_share": 0.5}, "version"),
            ({"version": 1, "margin_buckets": [0], "margin_log_factors": [], "ot_home_share": 0.5}, "log-factor"),
            (
                {"version": 1, "margin_buckets": [1, 0], "margin_log_factors": [0, 0], "ot_home_share": 0.5},
                "increasing",
            ),
            (
                {"version": 1, "margin_buckets": [0], "margin_log_factors": [float("nan")], "ot_home_share": 0.5},
                "finite",
            ),
            ({"version": 1, "margin_buckets": [0], "margin_log_factors": [3.5], "ot_home_share": 0.5}, "exceeds"),
            ({"version": 1, "margin_buckets": [0], "margin_log_factors": [0.1], "ot_home_share": 1.0}, r"\(0, 1\)"),
            ({"version": 1, "margin_buckets": [0], "ot_home_share": 0.5}, "malformed"),
        ],
    )
    def test_malformed_payloads_are_refused(self, payload, match):
        with pytest.raises(ValueError, match=match):
            hr.RegulationStructure.from_payload(payload)

    def test_structure_of(self):
        m = DixonColesPredictor(_cfg())
        assert hr.structure_of(m) is None
        m.regulation_structure = {
            "version": 1,
            "margin_buckets": [0],
            "margin_log_factors": [9.0],
            "ot_home_share": 0.5,
        }
        with pytest.raises(ValueError):
            hr.structure_of(m)


def simulate(theta_by_margin, ot_share=0.55, rounds=60, seed=3):
    """Games between 12 teams whose regulation scoreline is the tilted Poisson
    with the given {margin: log-factor}; ties resolved with ``ot_share``."""
    rng = np.random.default_rng(seed)
    teams = [f"T{k}" for k in range(12)]
    attack = dict(zip(teams, rng.uniform(0.85, 1.15, len(teams))))
    defense = dict(zip(teams, rng.uniform(0.85, 1.15, len(teams))))
    i, j = np.indices((CAP + 1, CAP + 1))
    G = np.exp(sum(t * (np.abs(i - j) == m) for m, t in theta_by_margin.items())) if theta_by_margin else 1.0
    rows, day = [], pd.Timestamp("2023-10-01", tz="UTC")
    for _ in range(rounds):
        for h in teams:
            for a in rng.choice([t for t in teams if t != h], 2, replace=False):
                P = build_dc_matrix(3.15 * attack[h] * defense[a], 2.9 * attack[a] * defense[h], 0.0, max_goals=CAP)
                Q = (P * G).ravel()
                cell = rng.choice(Q.size, p=Q / Q.sum())
                hs, as_ = divmod(int(cell), CAP + 1)
                rows.append(
                    dict(home_team=h, away_team=a, home_score=hs, away_score=as_, match_date=day, league_id="N")
                )
        day += pd.Timedelta(days=2)
    df = pd.DataFrame(rows)
    tie = (df.home_score == df.away_score).to_numpy()
    ot = np.where(tie, (rng.random(len(df)) < ot_share).astype(float), np.nan)
    return df, ot


@pytest.mark.unit
class TestFitStructure:
    def test_recovers_known_margin_factors_and_the_level(self):
        truth = {0: 0.30, 1: -0.55, 2: -0.25}
        df, ot = simulate(truth, rounds=70)
        m = DixonColesPredictor(_cfg(time_decay=0.0, regularization=0.001))
        m.train(df)
        m.rho = 0.0
        s, res = hr.fit_structure(m, df, ot, (0, 1, 2), CAP)
        assert res["iterations"] is not None
        assert np.allclose(s.margin_log_factors, [truth[0], truth[1], truth[2]], atol=0.08)
        assert s.ot_home_share == pytest.approx(np.nanmean(ot))
        assert res["fitted_mean_total_goals"] == pytest.approx(res["observed_mean_total_goals"], rel=0.01)
        assert res["fitted_tie_rate"] == pytest.approx(res["observed_tie_rate"], abs=1e-4)  # ML matches the bucket
        assert res["loglik_gain"] > 50

    def test_a_plain_poisson_frame_fits_near_zero_factors(self):
        df, ot = simulate({}, rounds=150, seed=5)  # 3,600 games: a tie log-factor's SE is ~0.045
        m = DixonColesPredictor(_cfg(time_decay=0.0, regularization=0.001))
        m.train(df)
        m.rho = 0.0
        s, res = hr.fit_structure(m, df, ot, (0, 1), CAP)
        assert np.all(np.abs(s.margin_log_factors) < 0.15)
        assert res["loglik_gain"] < 6  # two free parameters on pure noise: ~chi2(2)/2

    def test_no_buckets_keeps_the_poisson_strengths(self):
        df, ot = simulate({0: 0.3}, rounds=30)
        m = DixonColesPredictor(_cfg(time_decay=0.0, regularization=0.001))
        m.train(df)
        m.rho = 0.0
        before = dict(m.team_attack)
        s, res = hr.fit_structure(m, df, ot, (), CAP)
        assert s.margin_buckets == () and res["iterations"] is not None
        # The same moment equations (but for the truncation at CAP): the fit only
        # walks on along the Jacobi iteration's slow direction, gaining nothing.
        assert abs(res["loglik_gain"]) < 0.5
        assert all(abs(m.team_attack[t] - before[t]) < 0.01 for t in before)

    def test_refuses_a_frame_without_overtime_games_or_with_an_unknown_team(self):
        df, ot = simulate({0: 0.3}, rounds=10)
        m = DixonColesPredictor(_cfg(time_decay=0.0))
        m.train(df)
        m.rho = 0.0
        with pytest.raises(ValueError, match="overtime share"):
            hr.fit_structure(m, df, np.full(len(df), np.nan), (0,), CAP)
        bad = df.copy()
        bad.loc[0, "home_team"] = "Nobody"
        with pytest.raises(ValueError, match="not trained on"):
            hr.fit_structure(m, bad, ot, (0,), CAP)


@pytest.mark.unit
class TestTwoMatrixMarkets:
    def _pair(self):
        P = build_dc_matrix(3.2, 2.9, 0.0, max_goals=CAP)
        s = hr.RegulationStructure((0, 1), (0.3, -0.6), 0.52)
        return P, s.matrices(P)

    def test_legacy_mode_is_unchanged(self):
        P = build_dc_matrix(3.2, 2.9, 0.0, max_goals=CAP)
        assert derive_hockey_markets(P) == derive_markets("nhl", P)

    def test_regulation_markets_read_the_regulation_matrix_the_rest_the_final_one(self):
        P, (R, F) = self._pair()
        m = derive_markets("nhl", R, final=F)
        assert m["regulation_1x2"]["draw"] == pytest.approx(np.trace(R))
        assert m["double_chance"]["1X"] == pytest.approx(m["regulation_1x2"]["home"] + m["regulation_1x2"]["draw"])
        legacy_final = derive_hockey_markets(F)  # every final-score market as if F were the one matrix
        for market in ("puck_line", "over_under", "total_goals", "clean_sheet", "win_to_nil", "correct_score"):
            assert m[market] == legacy_final[market]
        assert m["moneyline"]["home"] + m["moneyline"]["away"] == pytest.approx(1.0)
        assert m["moneyline"]["home"] == pytest.approx(np.tril(F, -1).sum())
        assert not any(k.split("-")[0] == k.split("-")[-1] for k in m["correct_score"] if k != "other")

    def test_mismatched_shapes_raise(self):
        P, (R, F) = self._pair()
        with pytest.raises(ValueError, match="does not match"):
            derive_hockey_markets(R, final=F[:-1, :-1])


@pytest.mark.unit
class TestArtifactRoundTrip:
    def _fitted(self):
        df, _ = simulate({0: 0.3}, rounds=8)
        m = DixonColesPredictor(_cfg(time_decay=0.0))
        m.train(df)
        m.rho = 0.0
        return m

    def test_without_a_structure_the_bytes_carry_no_key(self, tmp_path):
        m = self._fitted()
        m.save(str(tmp_path / "a.bin"))
        data = json.loads((tmp_path / "a.bin").read_text())
        assert hr.PAYLOAD_KEY not in data
        back = DixonColesPredictor(DIXON_COLES_CONFIG)
        back.load(str(tmp_path / "a.bin"))
        assert back.regulation_structure is None

    def test_a_structure_survives_save_and_load(self, tmp_path):
        m = self._fitted()
        s = hr.RegulationStructure((0, 1), (0.25, -0.5), 0.51)
        m.regulation_structure = s.to_payload()
        m.save(str(tmp_path / "b.bin"))
        back = DixonColesPredictor(DIXON_COLES_CONFIG)
        back.load(str(tmp_path / "b.bin"))
        assert hr.structure_of(back) == s
