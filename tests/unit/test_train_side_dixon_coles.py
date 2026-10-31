"""Unit tests for the second-half and NHL Dixon-Coles trainers' safety rails
(scripts/train_second_half_dixon_coles.py, scripts/train_hockey_dixon_coles.py):
fit every row, refuse an unsound fit, write model.bin + manifest.json
atomically into their OWN staging dirs, never production — and stay out of
scripts/promote_dc_model.py, because neither artifact has passed its gate.
main() runs end to end on synthetic frames; the DB path is exercised on prod."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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


h2 = _load("train_second_half_dixon_coles", "train_second_half_dixon_coles.py")
nhl = _load("train_hockey_dixon_coles", "train_hockey_dixon_coles.py")
TRAINERS = pytest.mark.parametrize("trainer", [h2, nhl], ids=["h2", "nhl"])


def h2_frame(rounds: int = 6, seed: int = 0) -> pd.DataFrame:
    """A two-league second-half frame shaped like SECOND_HALF_TRAINING_QUERY's."""
    rng = np.random.default_rng(seed)
    leagues = {"L1": ["A", "B", "C", "D"], "L2": ["E", "F", "G", "H"]}
    rows = []
    day = pd.Timestamp("2025-01-04", tz="UTC")
    for r in range(rounds):
        for league, teams in leagues.items():
            for home in teams:
                for away in teams:
                    if home != away:
                        hs, as_ = int(rng.poisson(0.85)), int(rng.poisson(0.7))
                        rows.append(
                            dict(
                                match_id=f"{league}-{r}-{home}-{away}",
                                match_date=day + pd.Timedelta(days=7 * r),
                                league_id=league,
                                home_team=home,
                                away_team=away,
                                home_score=hs,
                                away_score=as_,
                                match_outcome=0 if hs > as_ else (1 if hs == as_ else 2),
                            )
                        )
    return pd.DataFrame(rows)


def nhl_frame(rounds: int = 8, seed: int = 0) -> pd.DataFrame:
    """A one-league NHL frame shaped like NHL_DC_TRAINING_QUERY's: FINAL scores, never tied."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    day = pd.Timestamp("2025-10-08", tz="UTC")
    for r in range(rounds):
        for home in teams:
            for away in teams:
                if home != away:
                    hs, as_ = int(rng.poisson(3.2)), int(rng.poisson(2.9))
                    if hs == as_:
                        hs += int(rng.random() < 0.5)
                        as_ += int(hs == as_)
                    rows.append(
                        dict(
                            match_id=f"{r}-{home}-{away}",
                            match_date=day + pd.Timedelta(days=4 * r),
                            league_id="NHL",
                            home_team=home,
                            away_team=away,
                            home_score=hs,
                            away_score=as_,
                            match_outcome=0 if hs > as_ else 1,
                        )
                    )
    return pd.DataFrame(rows)


FRAMES = {h2: h2_frame, nhl: nhl_frame}


@pytest.fixture
def fake_db(monkeypatch):
    """Serve a synthetic frame from load_training_frame, recording the query."""
    import utils.training_data as td

    seen = {}

    def install(frame):
        def load(*, database_url=None, input_csv=None, query=None):
            seen["query"] = query
            return frame.copy()

        monkeypatch.setattr(td, "load_training_frame", load)
        return seen

    return install


# ── where they write ─────────────────────────────────────────────────


class TestOutputDirs:
    def test_each_trainer_stages_into_its_own_dir(self):
        assert h2.DEFAULT_OUTPUT_DIR == Path("/app/models/staging-h2/dixon_coles_h2_soccer/1.0.0")
        assert nhl.DEFAULT_OUTPUT_DIR == Path("/app/models/staging-nhl/dixon_coles_nhl/1.0.0")
        assert h2.MODEL_NAME == "dixon_coles_h2_soccer" and nhl.MODEL_NAME == "dixon_coles_nhl"

    @TRAINERS
    def test_the_default_is_allowed(self, trainer):
        assert trainer.halftime.output_dir_refusal(trainer.DEFAULT_OUTPUT_DIR) is None

    @TRAINERS
    @pytest.mark.parametrize(
        "out",
        [
            "/app/models/staging/x",
            "/app/models/production/dixon_coles_h2_soccer/1.0.0",
            "/app/models/production/dixon_coles_nhl/1.0.0",
            "/app/models/staging-nhl/../production/dixon_coles_nhl/1.0.0",
        ],
    )
    def test_main_refuses_staging_and_production_before_touching_the_db(self, trainer, out, caplog, fake_db):
        seen = fake_db(pd.DataFrame())
        with caplog.at_level("ERROR", logger=trainer.LOGGER.name):
            assert trainer.main(["--database-url", "postgresql://unused", "--output-dir", out]) == 1
        assert "Refusing --output-dir" in caplog.text
        assert "query" not in seen

    @TRAINERS
    def test_main_needs_a_database_url_and_a_sane_holdout(self, trainer, tmp_path, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert trainer.main(["--output-dir", str(tmp_path)]) == 1
        assert trainer.main(["--database-url", "x", "--output-dir", str(tmp_path), "--holdout-fraction", "1.0"]) == 1


# ── configs and rho ──────────────────────────────────────────────────


class TestConfigs:
    def test_second_half_fits_the_soccer_config_under_its_own_name(self):
        from predictors.model_config import DIXON_COLES_CONFIG

        cfg = h2.h2_config()
        assert cfg.name == "dixon_coles_h2_soccer"
        assert cfg.hyperparameters == DIXON_COLES_CONFIG.hyperparameters
        assert DIXON_COLES_CONFIG.name == "dixon_coles"  # not mutated

    def test_nhl_keeps_the_pinned_legacy_config_until_its_gate_passes(self):
        from predictors.model_config import DIXON_COLES_CONFIG, LEGACY_DIXON_COLES_HYPERPARAMETERS

        cfg = nhl.nhl_config()
        assert cfg.name == "dixon_coles_nhl"
        assert cfg.hyperparameters == {**DIXON_COLES_CONFIG.hyperparameters, **LEGACY_DIXON_COLES_HYPERPARAMETERS}

    def test_force_rho_zero(self):
        class M:
            rho = 8.2e27

        m, result = M(), {"rho": 8.2e27}
        assert nhl.force_rho_zero(m, result) == 8.2e27
        assert m.rho == 0.0 and result == {"rho": 0.0, "rho_fitted": 8.2e27}
        m.rho = float("nan")
        assert nhl.force_rho_zero(m, {}) is None and m.rho == 0.0

    def test_served_fixtures_carry_no_league(self):
        frame = pd.DataFrame(
            {"home_team": ["A", "A", "B"], "away_team": ["B", "B", "A"], "league_id": ["NHL", "NHL", "NHL"]}
        )
        assert list(nhl.served_fixtures(frame)) == [("A", "B", None), ("B", "A", None)]

    def test_second_half_query_is_the_gates_frame(self):
        sql = " ".join(h2.SECOND_HALF_TRAINING_QUERY.split())
        for clause in (
            "GREATEST(m.home_score - m.half_time_home, 0) AS home_score",
            "AND m.home_score >= m.half_time_home",
            "AND m.away_score >= m.half_time_away",
            "ORDER BY m.match_date ASC",
        ):
            assert clause in sql, clause

    def test_nhl_query_excludes_preseason(self):
        from utils.training_data import preseason_exclusion_sql

        assert preseason_exclusion_sql("m") in nhl.NHL_DC_TRAINING_QUERY
        assert "l.sport = 'nhl'" in nhl.NHL_DC_TRAINING_QUERY


# ── main() end to end ────────────────────────────────────────────────


class TestMain:
    @TRAINERS
    def test_fits_every_row_and_stages_a_manifest(self, trainer, tmp_path, fake_db):
        frame = FRAMES[trainer]()
        seen = fake_db(frame)
        out = tmp_path / "stage"
        assert trainer.main(["--database-url", "x", "--output-dir", str(out), "--run-id", "run-7"]) == 0
        assert seen["query"] == (h2.SECOND_HALF_TRAINING_QUERY if trainer is h2 else nhl.NHL_DC_TRAINING_QUERY)
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["model"] == trainer.MODEL_NAME
        assert manifest["run_id"] == "run-7" and manifest["holdout_fraction"] == 0.0
        assert manifest["n_rows"] == len(frame)  # every row, not the oldest 80%
        assert manifest["converged_at"] is not None
        assert manifest["model_sha256"] == hashlib.sha256((out / "model.bin").read_bytes()).hexdigest()
        assert sorted(p.name for p in out.iterdir()) == ["manifest.json", "model.bin"]
        payload = json.loads((out / "model.bin").read_text())
        assert payload["config"]["name"] == trainer.MODEL_NAME
        if trainer is nhl:
            assert payload["rho"] == manifest["rho"] == 0.0
            assert payload["league_baselines"] == {}  # pinned config: one global baseline
        else:
            assert abs(payload["rho"]) <= 0.1 and payload["league_baselines"]
        assert manifest["leagues"] == (0 if trainer is nhl else 2)

    @TRAINERS
    def test_an_unconverged_fit_writes_nothing(self, trainer, tmp_path, fake_db, monkeypatch, caplog):
        fake_db(FRAMES[trainer]())
        name = "h2_config" if trainer is h2 else "nhl_config"
        base = getattr(trainer, name)()
        starved = replace(base, hyperparameters={**base.hyperparameters, "max_iterations": 1})
        monkeypatch.setattr(trainer, name, lambda: starved)
        out = tmp_path / "stage"
        out.mkdir()
        (out / "model.bin").write_text("previous model")
        with caplog.at_level("ERROR", logger=trainer.LOGGER.name):
            assert trainer.main(["--database-url", "x", "--output-dir", str(out)]) == 1
        assert "did not converge" in caplog.text and "refusing to write" in caplog.text
        assert sorted(p.name for p in out.iterdir()) == ["model.bin"]
        assert (out / "model.bin").read_text() == "previous model"

    @TRAINERS
    def test_an_empty_frame_is_refused(self, trainer, tmp_path, fake_db, caplog):
        fake_db(pd.DataFrame())
        with caplog.at_level("ERROR", logger=trainer.LOGGER.name):
            assert trainer.main(["--database-url", "x", "--output-dir", str(tmp_path / "s")]) == 1
        assert "empty" in caplog.text and not (tmp_path / "s").exists()

    def test_nhl_refusal_checks_the_matrices_production_builds_after_forcing_rho(self, monkeypatch, tmp_path):
        """The runaway rho is forced to 0 BEFORE fit_refusal, so the rho bound
        cannot refuse a sound NHL fit — and fit_refusal sees league None."""
        calls = {}
        orig = nhl.halftime.fit_refusal

        def spy(result, model, fixtures):
            fixtures = list(fixtures)
            calls["rho"], calls["leagues"] = model.rho, {f[2] for f in fixtures}
            return orig(result, model, fixtures)

        monkeypatch.setattr(nhl.halftime, "fit_refusal", spy)
        import utils.training_data as td

        monkeypatch.setattr(td, "load_training_frame", lambda **kw: nhl_frame())
        assert nhl.main(["--database-url", "x", "--output-dir", str(tmp_path / "s")]) == 0
        assert calls == {"rho": 0.0, "leagues": {None}}


# ── neither is promotable ────────────────────────────────────────────


class TestNotPromotable:
    def test_promote_dc_model_has_no_h2_or_nhl_entry(self):
        promote = _load("promote_dc_model", "promote_dc_model.py")
        assert set(promote.REGISTRY) == {"ht"}

    @TRAINERS
    def test_docstrings_say_so_and_name_the_gate(self, trainer):
        doc = trainer.__doc__
        assert "NOT PROMOTABLE" in doc and "promote_dc_model.py has no" in doc
        gate = "ab_second_half_dixon_coles.py" if trainer is h2 else "ab_nhl_dixon_coles.py"
        assert gate in doc

    def test_the_halftime_manifest_default_is_unchanged(self, tmp_path):
        """build_manifest / write_artifact take a model name now; the half-time
        trainer's own name stays the default."""
        frame = h2_frame()
        from predictors.poisson_models import DixonColesPredictor

        model = DixonColesPredictor(h2.h2_config())
        result = model.train(frame)
        fields = dict(result=result, train_df=frame, run_id="r", git_sha="g", holdout_fraction=0.0)
        assert h2.halftime.write_artifact(model, tmp_path / "a", fields)["model"] == "dixon_coles_ht_soccer"
        named = h2.halftime.write_artifact(model, tmp_path / "b", fields, model_name="dixon_coles_h2_soccer")
        assert named["model"] == "dixon_coles_h2_soccer"
        assert h2.halftime.build_manifest(model, model_sha256="x", **fields)["model"] == "dixon_coles_ht_soccer"
