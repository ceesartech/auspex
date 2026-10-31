"""Unit tests for the half-time Dixon-Coles trainer's safety rails
(scripts/train_halftime_dixon_coles.py): the training-frame filter the A/B
gate measured, the post-fit refusals, and the manifest + atomic write that
scripts/promote_dc_model.py trusts. The DB path is exercised on prod."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import replace
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


trainer = _load("train_halftime_dixon_coles", "train_halftime_dixon_coles.py")


def ht_frame(rounds: int = 6, seed: int = 0) -> pd.DataFrame:
    """A small two-league half-time frame shaped like HT_TRAINING_QUERY's."""
    rng = np.random.default_rng(seed)
    leagues = {"L1": ["A", "B", "C", "D"], "L2": ["E", "F", "G", "H"]}
    rows = []
    day = pd.Timestamp("2025-01-04", tz="UTC")
    for r in range(rounds):
        for league, teams in leagues.items():
            for home in teams:
                for away in teams:
                    if home == away:
                        continue
                    hs, as_ = int(rng.poisson(0.7)), int(rng.poisson(0.55))
                    rows.append(
                        {
                            "match_id": f"{league}-{r}-{home}-{away}",
                            "match_date": day + pd.Timedelta(days=7 * r),
                            "league_id": league,
                            "home_team": home,
                            "away_team": away,
                            "home_score": hs,
                            "away_score": as_,
                            "match_outcome": 0 if hs > as_ else (1 if hs == as_ else 2),
                        }
                    )
    return pd.DataFrame(rows)


def fit(frame: pd.DataFrame):
    from predictors.model_config import DIXON_COLES_CONFIG
    from predictors.poisson_models import DixonColesPredictor

    model = DixonColesPredictor(replace(DIXON_COLES_CONFIG, name=trainer.MODEL_NAME))
    result = model.train(frame)
    return model, result


def fixtures(frame: pd.DataFrame):
    return frame[["home_team", "away_team", "league_id"]].drop_duplicates().itertuples(index=False)


@pytest.fixture(scope="module")
def fitted():
    frame = ht_frame()
    model, result = fit(frame)
    return frame, model, result


class FakeModel:
    def __init__(self, rho=-0.01, lambdas=(0.7, 0.5)):
        self.rho = rho
        self._lambdas = lambdas

    def lambdas_for_match(self, home, away, league=None):
        return self._lambdas


GOOD_RESULT = {"iterations": 40, "fitted_mean_total_goals": 1.214, "observed_mean_total_goals": 1.215}


# ── the training frame ───────────────────────────────────────────────


class TestTrainingQuery:
    def test_drops_rows_without_a_consistent_full_time_score(self):
        sql = " ".join(trainer.HT_TRAINING_QUERY.split())
        for clause in (
            "m.half_time_home IS NOT NULL",
            "m.half_time_away IS NOT NULL",
            "AND m.home_score IS NOT NULL",
            "AND m.away_score IS NOT NULL",
            "AND m.home_score >= m.half_time_home",
            "AND m.away_score >= m.half_time_away",
        ):
            assert clause in sql, clause

    def test_filters_are_on_the_table_columns_not_the_ht_aliases(self):
        # home_score/away_score are ALIASES of the half-time goals in the
        # SELECT list; the FT filter must name the table column (m.), and sit
        # in the WHERE clause, before the ORDER BY.
        sql = " ".join(trainer.HT_TRAINING_QUERY.split())
        where = sql[sql.index(" WHERE ") : sql.index(" ORDER BY ")]
        assert re.findall(r"(?<!m\.)\bhome_score\b", where) == []
        assert re.findall(r"(?<!m\.)\baway_score\b", where) == []
        assert sql.rstrip().endswith("ORDER BY m.match_date ASC")


# ── post-fit refusals ────────────────────────────────────────────────


class TestFitRefusal:
    def test_a_sound_fit_is_accepted(self, fitted):
        frame, model, result = fitted
        assert result["iterations"] is not None
        assert trainer.fit_refusal(result, model, fixtures(frame)) is None

    def test_unconverged(self):
        assert "did not converge" in trainer.fit_refusal(dict(GOOD_RESULT, iterations=None), FakeModel(), [])

    def test_level_bias_beyond_two_percent(self):
        r = dict(GOOD_RESULT, fitted_mean_total_goals=1.215 * 1.021)
        msg = trainer.fit_refusal(r, FakeModel(), [])
        assert msg and "outside +/-2%" in msg and "+2.10%" in msg
        r = dict(GOOD_RESULT, fitted_mean_total_goals=1.215 * 0.979)
        assert "outside" in trainer.fit_refusal(r, FakeModel(), [])

    def test_level_bias_inside_two_percent(self):
        r = dict(GOOD_RESULT, fitted_mean_total_goals=1.215 * 1.019)
        assert trainer.fit_refusal(r, FakeModel(), [("A", "B", "L1")]) is None

    @pytest.mark.parametrize(
        "fitted_total, observed_total",
        [(None, 1.2), (1.2, None), (float("nan"), 1.2), (1.2, 0.0), (True, 1.0)],
    )
    def test_missing_or_degenerate_level_check(self, fitted_total, observed_total):
        r = dict(GOOD_RESULT, fitted_mean_total_goals=fitted_total, observed_mean_total_goals=observed_total)
        assert "no usable level check" in trainer.fit_refusal(r, FakeModel(), [])

    def test_runaway_rho(self):
        assert "exceeds 0.1" in trainer.fit_refusal(GOOD_RESULT, FakeModel(rho=0.13), [])
        assert "exceeds 0.1" in trainer.fit_refusal(GOOD_RESULT, FakeModel(rho=-0.11), [])
        assert trainer.fit_refusal(GOOD_RESULT, FakeModel(rho=-0.1), [("A", "B", "L1")]) is None

    @pytest.mark.parametrize("rho", [float("nan"), float("inf"), None])
    def test_non_finite_rho(self, rho):
        assert "not finite" in trainer.fit_refusal(GOOD_RESULT, FakeModel(rho=rho), [])

    def test_invalid_served_matrix(self):
        msg = trainer.fit_refusal(GOOD_RESULT, FakeModel(lambdas=(float("nan"), 0.5)), [("Home FC", "Away FC", "L9")])
        assert msg and "invalid served scoreline matrix" in msg and "Home FC v Away FC" in msg


# ── manifest + atomic write ──────────────────────────────────────────


def manifest_fields(frame, result, **overrides):
    fields = dict(result=result, train_df=frame, run_id="run-1", git_sha="abc123", holdout_fraction=0.0)
    fields.update(overrides)
    return fields


class TestWriteArtifact:
    def test_manifest_describes_the_written_model(self, fitted, tmp_path):
        frame, model, result = fitted
        manifest = trainer.write_artifact(model, tmp_path, manifest_fields(frame, result))
        model_bin = tmp_path / "model.bin"
        on_disk = json.loads((tmp_path / "manifest.json").read_text())
        assert on_disk == manifest
        assert manifest["model_sha256"] == hashlib.sha256(model_bin.read_bytes()).hexdigest()
        assert manifest["model"] == "dixon_coles_ht_soccer"
        assert manifest["run_id"] == "run-1" and manifest["git_sha"] == "abc123"
        assert manifest["n_rows"] == len(frame)
        assert pd.Timestamp(manifest["newest_match_date"]) == frame["match_date"].max()
        assert manifest["holdout_fraction"] == 0.0
        assert manifest["converged_at"] == result["iterations"]
        assert manifest["fitted_mean_total_goals"] == result["fitted_mean_total_goals"]
        assert manifest["observed_mean_total_goals"] == result["observed_mean_total_goals"]
        assert manifest["rho"] == model.rho
        assert manifest["teams"] == 8 and manifest["leagues"] == 2
        assert pd.Timestamp(manifest["written_at"]).tzinfo is not None
        assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json", "model.bin"]  # no *.tmp left
        # model.bin is the predictor's own serialisation — it loads back.
        from predictors.model_config import DIXON_COLES_CONFIG
        from predictors.poisson_models import DixonColesPredictor

        loaded = DixonColesPredictor(DIXON_COLES_CONFIG)
        loaded.load(str(model_bin))
        assert loaded.rho == model.rho and loaded.team_attack == model.team_attack

    def test_failed_save_leaves_the_previous_artifact_untouched(self, fitted, tmp_path):
        frame, model, result = fitted
        (tmp_path / "model.bin").write_text("old model")
        (tmp_path / "manifest.json").write_text("old manifest")

        class Exploding:
            rho = -0.01
            team_attack = {"A": 1.0}

            def save(self, path):
                Path(path).write_text("half a mod")
                raise OSError("disk full")

        with pytest.raises(OSError):
            trainer.write_artifact(Exploding(), tmp_path, manifest_fields(frame, result))
        assert (tmp_path / "model.bin").read_text() == "old model"
        assert (tmp_path / "manifest.json").read_text() == "old manifest"

    def test_failed_manifest_leaves_the_previous_model_untouched(self, fitted, tmp_path):
        frame, model, result = fitted
        (tmp_path / "model.bin").write_text("old model")
        (tmp_path / "manifest.json").write_text("old manifest")
        # allow_nan=False: a NaN in the manifest raises instead of writing
        # invalid JSON — before either file is renamed into place.
        bad = dict(result, fitted_mean_total_goals=float("nan"))
        with pytest.raises(ValueError):
            trainer.write_artifact(model, tmp_path, manifest_fields(frame, bad))
        assert (tmp_path / "model.bin").read_text() == "old model"
        assert (tmp_path / "manifest.json").read_text() == "old manifest"

    def test_default_run_id_is_a_utc_stamp(self):
        assert re.fullmatch(r"\d{8}T\d{6}Z", trainer.default_run_id())


class TestOutputDirGuard:
    @pytest.mark.parametrize(
        "path, refused",
        [
            ("/app/models/staging", True),
            ("/app/models/staging/dixon_coles_ht_soccer/1.0.0", True),
            ("/app/models/staging/../staging/x", True),
            ("/app/models/staging-ht/dixon_coles_ht_soccer/1.0.0", False),
            ("/app/models/production/dixon_coles_ht_soccer/1.0.0", False),
        ],
    )
    def test_never_writes_into_the_auto_promoted_staging_tree(self, path, refused):
        assert trainer.writes_into_auto_promoted_staging(Path(path)) is refused

    def test_default_output_dir_is_outside_it(self):
        assert not trainer.writes_into_auto_promoted_staging(trainer.DEFAULT_OUTPUT_DIR)

    def test_main_refuses_before_touching_the_db(self):
        assert trainer.main(["--database-url", "postgresql://unused", "--output-dir", "/app/models/staging/x"]) == 1
