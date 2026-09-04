"""Unit tests for the Poisson / Dixon-Coles fitting fixes.

Covers the four defects that made the served soccer Dixon-Coles level-biased:

1. the scale gauge — the multiplicative model has exactly ONE scale
   indeterminacy, so normalising BOTH attack and defense over-constrains it
   and pins mean(lambda) to the hard-coded baseline instead of the data;
2. per-league baselines, shrunk toward the global weighted means;
3. exponential recency weights actually applied to the MLE (and to the rho
   NLL) rather than serialised and ignored;
4. `regularization` read as a real shrinkage prior in effective matches.

Plus the backward-compatibility contract: an artifact written before any of
this must still load and serve.

The decay here is a weight inside a 4-parameter multiplicative Poisson MLE.
It is NOT the GBM training-frame horizon and NOT GBM recency weighting —
those are separate, closed levers (docs/SYSTEM_AUDIT_AND_ROADMAP.md §4).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from predictors.model_config import ModelConfig, ModelType, PredictionTask
from predictors.poisson_models import DixonColesPredictor, PoissonMatchPredictor

# ── helpers ──────────────────────────────────────────────────────────


def make_config(**hyper):
    """A Poisson/Dixon-Coles config with tight convergence so the tests
    measure the fixed point, not the iteration budget."""
    base = {
        "max_goals": 8,
        "home_advantage": 0.25,
        "regularization": 0.0,
        "max_iterations": 2000,
        "convergence_threshold": 1e-10,
        "rho_init": -0.05,
    }
    base.update(hyper)
    return ModelConfig(
        name="test_poisson_baselines",
        model_type=ModelType.DIXON_COLES,
        prediction_task=PredictionTask.MATCH_OUTCOME,
        version="test",
        hyperparameters=base,
        features=[],
        target_column="match_outcome",
        loss_function="poisson_nll",
        metrics=["accuracy"],
        training_config={},
    )


def make_league(
    league_id="L1",
    league_name=None,
    n=600,
    n_teams=10,
    home_rate=1.9,
    away_rate=1.3,
    seed=0,
    start="2020-01-01",
    team_prefix=None,
):
    """A synthetic league with a KNOWN mean total (home_rate + away_rate)
    and interchangeable teams, so the only thing a correct fit has to
    recover is the level."""
    rng = np.random.default_rng(seed)
    prefix = team_prefix if team_prefix is not None else league_id
    teams = [f"{prefix}_T{i}" for i in range(n_teams)]
    home = rng.integers(0, n_teams, n)
    away = (home + 1 + rng.integers(0, n_teams - 1, n)) % n_teams  # never self
    return pd.DataFrame(
        {
            "match_date": pd.date_range(start, periods=n, freq="D"),
            "league_id": league_id,
            "league": league_name if league_name is not None else f"{league_id} name",
            "home_team": [teams[i] for i in home],
            "away_team": [teams[i] for i in away],
            "home_score": rng.poisson(home_rate, n).astype(float),
            "away_score": rng.poisson(away_rate, n).astype(float),
        }
    )


def observed_mean_total(df):
    return float((df["home_score"] + df["away_score"]).mean())


def fitted_mean_total(model, df):
    """Mean of (home_lambda + away_lambda) the fitted model implies over
    the very rows it was fitted on."""
    totals = [
        sum(model.lambdas_for_match(row.home_team, row.away_team, league=row.league_id)) for row in df.itertuples()
    ]
    return float(np.mean(totals))


def legacy_double_normalised_fit(df, home_baseline, away_baseline, iterations=2000):
    """The PRE-FIX scheme, reimplemented here as the counterfactual: it
    normalises attack AND defense to mean 1 every iteration. Returns the
    mean total goals it implies over `df`."""
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    h = df["home_team"].map(idx).to_numpy()
    a = df["away_team"].map(idx).to_numpy()
    hs = df["home_score"].to_numpy(dtype=float)
    as_ = df["away_score"].to_numpy(dtype=float)
    n = len(teams)
    attack_num = np.bincount(h, weights=hs, minlength=n) + np.bincount(a, weights=as_, minlength=n)
    defense_num = np.bincount(h, weights=as_, minlength=n) + np.bincount(a, weights=hs, minlength=n)
    attack = np.ones(n)
    defense = np.ones(n)
    for _ in range(iterations):
        attack_den = np.bincount(h, weights=defense[a] * home_baseline, minlength=n) + np.bincount(
            a, weights=defense[h] * away_baseline, minlength=n
        )
        defense_den = np.bincount(h, weights=attack[a] * away_baseline, minlength=n) + np.bincount(
            a, weights=attack[h] * home_baseline, minlength=n
        )
        attack = np.where(attack_den > 0, attack_num / attack_den, attack)
        defense = np.where(defense_den > 0, defense_num / defense_den, defense)
        attack /= attack.mean()
        defense /= defense.mean()  # ← the extra constraint the model does not have
    return float(np.mean(attack[h] * defense[a] * home_baseline + attack[a] * defense[h] * away_baseline))


# ── 1. the scale gauge ───────────────────────────────────────────────


@pytest.mark.unit
class TestScaleIndeterminacy:
    def test_level_is_recovered_from_the_data(self):
        """A frame with a known mean total must be recovered by the fit
        even though the configured baseline (league_avg + 0.25) disagrees
        with it. This is the assertion that FAILS under the old
        double-normalised scheme."""
        df = make_league(n=1500, home_rate=1.9, away_rate=1.3, seed=1)
        model = PoissonMatchPredictor(make_config())
        result = model.train(df)

        observed = observed_mean_total(df)
        assert result["observed_mean_total_goals"] == pytest.approx(observed, rel=1e-9)
        assert result["fitted_mean_total_goals"] == pytest.approx(observed, rel=0.005)
        assert fitted_mean_total(model, df) == pytest.approx(observed, rel=0.01)

    def test_old_double_normalisation_is_pinned_to_the_baseline(self):
        """Documents the mechanism: normalising defense as well forces the
        fitted level onto the hard-coded baseline (here 3.45) instead of
        the observed one (~3.2)."""
        df = make_league(n=1500, home_rate=1.9, away_rate=1.3, seed=1)
        model = PoissonMatchPredictor(make_config())
        model.train(df)

        observed = observed_mean_total(df)
        baseline_sum = model.league_avg_goals + model.home_advantage + model.league_avg_goals
        legacy = legacy_double_normalised_fit(
            df,
            model.league_avg_goals + model.home_advantage,
            model.league_avg_goals,
        )
        # The legacy fit sits on the baseline, not on the data...
        assert legacy == pytest.approx(baseline_sum, rel=0.01)
        assert abs(legacy - observed) > 0.15
        # ...while the fixed fit sits on the data.
        assert abs(fitted_mean_total(model, df) - observed) < 0.05

    def test_normalise_defense_knob_reproduces_the_legacy_scheme(self):
        """The ablation knob the A/B harness needs. Without a switch that
        restores ONLY the double normalisation, no arm can attribute a
        result to the gauge fix as opposed to a bigger corpus — and the
        knob is worthless if it doesn't reproduce the old fit exactly."""
        df = make_league(n=1500, home_rate=1.9, away_rate=1.3, seed=1)
        fixed = PoissonMatchPredictor(make_config())
        fixed.train(df)
        legacy_model = PoissonMatchPredictor(make_config(normalise_defense=True))
        legacy_model.train(df)

        reference = legacy_double_normalised_fit(
            df,
            fixed.league_avg_goals + fixed.home_advantage,
            fixed.league_avg_goals,
        )
        assert fitted_mean_total(legacy_model, df) == pytest.approx(reference, rel=0.01)
        # It really is the biased fit, not a relabelled copy of the fixed one.
        observed = observed_mean_total(df)
        assert abs(fitted_mean_total(legacy_model, df) - observed) > 0.15
        assert float(np.mean(list(legacy_model.team_defense.values()))) == pytest.approx(1.0, abs=1e-6)

    def test_the_gauge_fix_is_the_default_and_nothing_opts_into_the_bug(self):
        model = PoissonMatchPredictor(make_config())
        assert model.normalise_defense is False
        from predictors.model_config import (
            DIXON_COLES_CONFIG,
            HOCKEY_POISSON_NHL_MONEYLINE,
            HOCKEY_POISSON_NHL_PUCK_LINE,
            HOCKEY_POISSON_NHL_REGULATION,
            HOCKEY_POISSON_NHL_TOTAL,
            POISSON_CONFIG,
            dixon_coles_config_pinned_to_legacy_fit,
        )

        for cfg in (
            DIXON_COLES_CONFIG,
            POISSON_CONFIG,
            HOCKEY_POISSON_NHL_MONEYLINE,
            HOCKEY_POISSON_NHL_REGULATION,
            HOCKEY_POISSON_NHL_PUCK_LINE,
            HOCKEY_POISSON_NHL_TOTAL,
            dixon_coles_config_pinned_to_legacy_fit(),
        ):
            assert cfg.hyperparameters.get("normalise_defense", False) is False

    def test_attack_gauge_is_still_fixed(self):
        """Removing the defense normalisation must not leave the fit
        unidentified: attack is still pinned to mean 1, and defense lands
        near 1 on its own because the baselines are the fitted means."""
        df = make_league(n=1200, seed=3)
        model = PoissonMatchPredictor(make_config(per_league_baselines=True))
        model.train(df)
        assert float(np.mean(list(model.team_attack.values()))) == pytest.approx(1.0, abs=1e-6)
        assert float(np.mean(list(model.team_defense.values()))) == pytest.approx(1.0, abs=0.05)


# ── 2. per-league baselines ──────────────────────────────────────────


@pytest.mark.unit
class TestPerLeagueBaselines:
    def test_thin_league_is_shrunk_toward_the_global_mean(self):
        big = make_league(league_id="BIG", n=2000, home_rate=1.9, away_rate=1.3, seed=4)
        thin = make_league(league_id="THIN", n=20, home_rate=0.6, away_rate=0.4, seed=5)
        df = pd.concat([big, thin], ignore_index=True)

        model = PoissonMatchPredictor(make_config(per_league_baselines=True, league_shrinkage=200.0))
        model.train(df)

        big_home, big_away = model.league_baselines["BIG"]
        thin_home, thin_away = model.league_baselines["THIN"]
        global_home = model.global_home_baseline

        thin_raw_home = float(thin["home_score"].mean())
        # 20 matches against 200 of prior → ~9% of the way to its own mean.
        expected = (20 * thin_raw_home + 200 * global_home) / 220.0
        assert thin_home == pytest.approx(expected, rel=1e-6)
        assert abs(thin_home - global_home) < 0.15 * abs(thin_raw_home - global_home)

        # The 2,000-match league keeps essentially its own level.
        assert big_home == pytest.approx(float(big["home_score"].mean()), rel=0.05)
        assert big_home > big_away
        assert thin_home + thin_away > 0

    def test_leagues_get_different_levels(self):
        low = make_league(league_id="LOW", n=1200, home_rate=1.15, away_rate=0.95, seed=6)
        high = make_league(league_id="HIGH", n=1200, home_rate=1.75, away_rate=1.45, seed=7)
        df = pd.concat([low, high], ignore_index=True)

        model = PoissonMatchPredictor(make_config(per_league_baselines=True))
        model.train(df)

        low_total = sum(model.league_baselines["LOW"])
        high_total = sum(model.league_baselines["HIGH"])
        assert high_total - low_total > 0.7
        assert low_total < sum((model.global_home_baseline, model.global_away_baseline)) < high_total

    def test_flag_off_keeps_the_single_global_baseline(self):
        df = pd.concat(
            [
                make_league(league_id="LOW", n=600, home_rate=1.1, away_rate=0.9, seed=8),
                make_league(league_id="HIGH", n=600, home_rate=1.8, away_rate=1.5, seed=9),
            ],
            ignore_index=True,
        )
        model = PoissonMatchPredictor(make_config(per_league_baselines=False))
        model.train(df)

        assert model.league_baselines == {}
        assert model.global_home_baseline is None
        low = model.lambdas_for_match("LOW_T0", "LOW_T1", league="LOW")
        high = model.lambdas_for_match("LOW_T0", "LOW_T1", league="HIGH")
        assert low == high  # no per-league state → league argument is inert

    def test_missing_league_column_falls_back_to_global(self, caplog):
        df = make_league(n=400, seed=10).drop(columns=["league_id", "league"])
        model = PoissonMatchPredictor(make_config(per_league_baselines=True))
        with caplog.at_level("WARNING"):
            model.train(df)
        assert model.league_baselines == {}
        assert model.global_home_baseline is not None
        assert any("per_league_baselines is on" in r.message for r in caplog.records)

    def test_league_name_alias_resolves_to_the_same_baselines(self):
        df = make_league(league_id="L7", league_name="Serie Test", n=800, seed=11)
        model = PoissonMatchPredictor(make_config(per_league_baselines=True))
        result = model.train(df)
        assert model.lambdas_for_match("L7_T0", "L7_T1", league="Serie Test") == model.lambdas_for_match(
            "L7_T0", "L7_T1", league="L7"
        )
        # One league fitted, two keys (id + name alias).
        assert result["league_count"] == 1
        assert set(model.league_baselines) == {"L7", "Serie Test"}


# ── unknown teams ────────────────────────────────────────────────────


@pytest.mark.unit
class TestUnknownTeamFallback:
    def test_unknown_team_in_a_known_league_gets_the_league_baseline(self):
        low = make_league(league_id="LOW", n=1200, home_rate=1.1, away_rate=0.9, seed=12)
        high = make_league(league_id="HIGH", n=1200, home_rate=1.8, away_rate=1.5, seed=13)
        model = PoissonMatchPredictor(make_config(per_league_baselines=True))
        model.train(pd.concat([low, high], ignore_index=True))

        for league in ("LOW", "HIGH"):
            h_lam, a_lam = model.lambdas_for_match("Promoted FC", "Newcomer FC", league=league)
            assert (h_lam, a_lam) == pytest.approx(model.league_baselines[league])

        # ...and an unknown/omitted league is the GLOBAL baseline, which
        # sits strictly between the two league levels.
        globals_ = model.lambdas_for_match("Promoted FC", "Newcomer FC")
        assert globals_ == pytest.approx((model.global_home_baseline, model.global_away_baseline))
        assert sum(model.lambdas_for_match("A", "B", league="LOW")) < sum(globals_)
        assert sum(globals_) < sum(model.lambdas_for_match("A", "B", league="HIGH"))
        assert model.lambdas_for_match("A", "B", league="NOT_A_LEAGUE") == pytest.approx(globals_)


# ── 3. time decay ────────────────────────────────────────────────────


@pytest.mark.unit
class TestTimeDecay:
    @staticmethod
    def _two_era_frame():
        """A high-scoring old era followed by a low-scoring recent one."""
        old = make_league(league_id="L1", n=1000, home_rate=2.2, away_rate=1.8, seed=14, start="2012-01-01")
        new = make_league(league_id="L1", n=1000, home_rate=1.2, away_rate=0.9, seed=15, start="2022-01-01")
        return pd.concat([old, new], ignore_index=True)

    def test_decay_moves_the_fitted_level_toward_the_recent_era(self):
        df = self._two_era_frame()
        flat = PoissonMatchPredictor(make_config(per_league_baselines=True, time_decay=0.0))
        flat.train(df)
        decayed = PoissonMatchPredictor(make_config(per_league_baselines=True, time_decay=0.00095))
        decayed.train(df)

        recent_total = observed_mean_total(df.tail(1000))
        assert sum(decayed.league_baselines["L1"]) < sum(flat.league_baselines["L1"])
        assert abs(sum(decayed.league_baselines["L1"]) - recent_total) < abs(
            sum(flat.league_baselines["L1"]) - recent_total
        )

    def test_zero_decay_reproduces_the_unweighted_fit(self):
        df = self._two_era_frame()
        implicit = PoissonMatchPredictor(make_config(per_league_baselines=True))  # no time_decay key
        implicit.train(df)
        explicit = PoissonMatchPredictor(make_config(per_league_baselines=True, time_decay=0.0))
        explicit.train(df)

        assert implicit.time_decay == 0.0
        assert np.all(implicit._time_weights(df) == 1.0)
        assert implicit.team_attack == explicit.team_attack
        assert implicit.team_defense == explicit.team_defense
        assert implicit.league_baselines == explicit.league_baselines

    def test_weights_are_exponential_in_age_days(self):
        df = make_league(n=100, seed=16, start="2024-01-01")
        model = PoissonMatchPredictor(make_config(time_decay=0.001))
        weights = model._time_weights(df)
        assert weights[-1] == pytest.approx(1.0)  # newest match is the reference
        assert weights[0] == pytest.approx(np.exp(-0.001 * 99))
        assert np.all(np.diff(weights) > 0)

    def test_decay_without_dates_is_loud_and_unweighted(self, caplog):
        df = make_league(n=300, seed=17).drop(columns=["match_date"])
        model = PoissonMatchPredictor(make_config(time_decay=0.001))
        with caplog.at_level("ERROR"):
            weights = model._time_weights(df)
        assert np.all(weights == 1.0)
        assert any("UNWEIGHTED" in r.message for r in caplog.records)

    def test_serialised_decay_is_the_decay_that_was_used(self, tmp_path):
        df = make_league(n=400, seed=18)
        model = DixonColesPredictor(make_config(time_decay=0.00095, per_league_baselines=True))
        model.train(df)
        path = tmp_path / "dc.json"
        model.save(str(path))
        payload = json.loads(Path(path).read_text())
        assert payload["time_decay"] == 0.00095

        reloaded = DixonColesPredictor(make_config(time_decay=0.0))
        reloaded.load(str(path))
        assert reloaded.time_decay == 0.00095

    def test_rho_fit_is_weighted(self):
        """rho is estimated on the same effective sample as the strengths:
        decaying away a low-scoring era must move it."""
        df = self._two_era_frame()
        flat = DixonColesPredictor(make_config(per_league_baselines=True, time_decay=0.0))
        flat.train(df)
        decayed = DixonColesPredictor(make_config(per_league_baselines=True, time_decay=0.002))
        decayed.train(df)
        assert flat.rho != pytest.approx(decayed.rho, abs=1e-4)


# ── 4. regularization as a shrinkage prior ───────────────────────────


@pytest.mark.unit
class TestShrinkagePrior:
    @staticmethod
    def _frame_with_a_thin_team():
        df = make_league(league_id="L1", n=1200, home_rate=1.5, away_rate=1.2, seed=19)
        thin = pd.DataFrame(
            {
                "match_date": pd.to_datetime(["2021-06-01", "2021-06-08", "2021-06-15"]),
                "league_id": "L1",
                "league": "L1 name",
                "home_team": ["Thin FC", "Thin FC", "Thin FC"],
                "away_team": ["L1_T0", "L1_T1", "L1_T2"],
                "home_score": [6.0, 7.0, 6.0],
                "away_score": [0.0, 0.0, 0.0],
            }
        )
        return pd.concat([df, thin], ignore_index=True)

    def test_thin_team_is_pulled_toward_its_league_baseline(self):
        df = self._frame_with_a_thin_team()
        unregularised = PoissonMatchPredictor(make_config(per_league_baselines=True, regularization=0.0))
        unregularised.train(df)
        shrunk = PoissonMatchPredictor(make_config(per_league_baselines=True, regularization=20.0))
        shrunk.train(df)

        assert unregularised.team_attack["Thin FC"] > 3.0  # 6.3 goals/game, unshrunk
        assert shrunk.team_attack["Thin FC"] < unregularised.team_attack["Thin FC"]
        assert abs(shrunk.team_attack["Thin FC"] - 1.0) < abs(unregularised.team_attack["Thin FC"] - 1.0)

        # The shrunk prediction lands near the LEAGUE baseline rather than
        # near a 6-goal fantasy.
        league_home = shrunk.league_baselines["L1"][0]
        h_lam, _ = shrunk.lambdas_for_match("Thin FC", "L1_T5", league="L1")
        assert h_lam < 3.0 * league_home

    def test_shrinkage_helps_convergence(self):
        df = self._frame_with_a_thin_team()
        config = make_config(per_league_baselines=True, regularization=20.0, max_iterations=500)
        result = PoissonMatchPredictor(config).train(df)
        assert result["iterations"] is not None


# ── backward compatibility ───────────────────────────────────────────


@pytest.mark.unit
class TestLegacyArtifacts:
    LEGACY = {
        "team_attack": {"Alpha": 1.2, "Beta": 0.85},
        "team_defense": {"Alpha": 0.9, "Beta": 1.1},
        "home_advantage": 0.25,
        "league_avg_goals": 1.3615,
        "max_goals": 6,
        "rho": -0.0339,
        "time_decay": 0.0018,
        "config": {},
    }

    def _write(self, tmp_path, payload):
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(payload))
        return str(path)

    def test_legacy_dixon_coles_artifact_loads_and_serves(self, tmp_path):
        model = DixonColesPredictor(make_config(per_league_baselines=True, time_decay=0.00095))
        model.load(self._write(tmp_path, self.LEGACY))

        assert model.is_fitted is True
        assert model.league_baselines == {}
        assert model.global_home_baseline is None
        assert model.max_goals == 6  # the artifact's value wins over the config
        assert model.rho == -0.0339

        # Serves exactly the pre-change lambdas: attack × defense ×
        # (league_avg_goals ± home_advantage), league argument or not.
        h_lam, a_lam = model.lambdas_for_match("Alpha", "Beta", league="anything")
        assert h_lam == pytest.approx(1.2 * 1.1 * (1.3615 + 0.25))
        assert a_lam == pytest.approx(0.85 * 0.9 * 1.3615)
        assert model.lambdas_for_match("Alpha", "Beta") == (h_lam, a_lam)

        proba = model.predict_proba(pd.DataFrame({"home_team": ["Alpha"], "away_team": ["Beta"]}))
        assert proba.shape == (1, 3)
        assert proba.sum() == pytest.approx(1.0)
        assert np.isfinite(model.predict_score("Alpha", "Beta")["over_2_5"])

    def test_legacy_poisson_artifact_loads_and_serves(self, tmp_path):
        payload = {k: v for k, v in self.LEGACY.items() if k not in ("rho", "time_decay")}
        model = PoissonMatchPredictor(make_config(per_league_baselines=True))
        model.load(self._write(tmp_path, payload))
        assert model.is_fitted is True
        assert model.lambdas_for_match("Alpha", "Beta", league="L1")[0] == pytest.approx(1.2 * 1.1 * (1.3615 + 0.25))

    def test_unknown_team_in_a_legacy_artifact_is_the_global_constant(self, tmp_path):
        model = DixonColesPredictor(make_config())
        model.load(self._write(tmp_path, self.LEGACY))
        assert model.lambdas_for_match("Ghost", "Phantom", league="L1") == pytest.approx((1.3615 + 0.25, 1.3615))

    def test_roundtrip_preserves_the_new_baselines(self, tmp_path):
        df = pd.concat(
            [
                make_league(league_id="LOW", n=800, home_rate=1.1, away_rate=0.9, seed=20),
                make_league(league_id="HIGH", n=800, home_rate=1.8, away_rate=1.5, seed=21),
            ],
            ignore_index=True,
        )
        model = DixonColesPredictor(make_config(per_league_baselines=True, time_decay=0.00095))
        model.train(df)

        path = str(tmp_path / "dc.json")
        model.save(path)
        reloaded = DixonColesPredictor(make_config())
        reloaded.load(path)

        assert reloaded.league_baselines == model.league_baselines
        assert reloaded.global_home_baseline == model.global_home_baseline
        assert reloaded.global_away_baseline == model.global_away_baseline
        for league in ("LOW", "HIGH", "NOT_A_LEAGUE"):
            assert reloaded.lambdas_for_match("LOW_T0", "HIGH_T1", league=league) == pytest.approx(
                model.lambdas_for_match("LOW_T0", "HIGH_T1", league=league)
            )


# ── 6. blast radius: DIXON_COLES_CONFIG is shared ────────────────────


@pytest.mark.unit
class TestSharedConfigScoping:
    """DIXON_COLES_CONFIG is not soccer-only by construction — three
    scripts build a Dixon-Coles straight from it and write the artifact
    themselves, bypassing train_all_models and therefore bypassing both the
    1x2 promote gate and the derived-market guard. The refit knobs were
    measured on the soccer FULL-TIME over-2.5 walk-forward and nowhere
    else, so those three take a pinned copy instead."""

    def test_the_pinned_copy_turns_off_every_unmeasured_knob(self):
        from predictors.model_config import DIXON_COLES_CONFIG, dixon_coles_config_pinned_to_legacy_fit

        pinned = dixon_coles_config_pinned_to_legacy_fit("dixon_coles_nhl")
        assert pinned.name == "dixon_coles_nhl"
        assert pinned.hyperparameters["max_goals"] == 6
        assert pinned.hyperparameters["time_decay"] == 0.0
        assert pinned.hyperparameters["regularization"] == 0.001
        assert pinned.hyperparameters["per_league_baselines"] is False
        # ...without mutating the soccer config it was derived from.
        assert DIXON_COLES_CONFIG.hyperparameters["max_goals"] == 10
        assert DIXON_COLES_CONFIG.hyperparameters["per_league_baselines"] is True
        assert DIXON_COLES_CONFIG.name == "dixon_coles"

    def test_the_pinned_fit_matches_the_pre_refit_behaviour(self):
        """Same frame, pinned config vs the explicit legacy hyperparameters:
        identical fitted level. If this drifts, an unmeasured knob has
        leaked into the NHL / halftime / second-half artifacts."""
        df = pd.concat(
            [
                make_league(league_id="A", n=700, home_rate=1.6, away_rate=1.2, seed=30),
                make_league(league_id="B", n=700, home_rate=2.1, away_rate=1.7, seed=31),
            ],
            ignore_index=True,
        )
        from predictors.model_config import dixon_coles_config_pinned_to_legacy_fit

        pinned = dixon_coles_config_pinned_to_legacy_fit()
        explicit = make_config(
            max_goals=6,
            time_decay=0.0,
            regularization=0.001,
            per_league_baselines=False,
            max_iterations=pinned.hyperparameters["max_iterations"],
            convergence_threshold=pinned.hyperparameters["convergence_threshold"],
            rho_init=pinned.hyperparameters["rho_init"],
        )
        a = PoissonMatchPredictor(pinned)
        a.train(df)
        b = PoissonMatchPredictor(explicit)
        b.train(df)
        assert a.max_goals == b.max_goals == 6
        assert a.per_league_baselines is False and a.league_baselines == {}
        assert fitted_mean_total(a, df) == pytest.approx(fitted_mean_total(b, df), rel=1e-9)

    def test_the_three_off_gate_trainers_do_not_use_the_soccer_config(self):
        """A grep-level guard. The whole defect is that these three read the
        SAME object the soccer refit mutates."""
        repo = Path(__file__).resolve().parents[4]
        for script in (
            "train_hockey_dixon_coles.py",
            "train_halftime_dixon_coles.py",
            "train_second_half_dixon_coles.py",
        ):
            source = (repo / "scripts" / script).read_text()
            assert "DixonColesPredictor(DIXON_COLES_CONFIG)" not in source, script
            assert "dixon_coles_config_pinned_to_legacy_fit" in source, script
