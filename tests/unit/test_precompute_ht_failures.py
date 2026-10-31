"""Half-time failures in the soccer serve path are loud
(scripts/precompute_predictions.py).

A missing or unloadable HT / 2H Dixon-Coles artifact disables the HT and
HT/FT markets on every run, so it logs at ERROR, not INFO. Per-match HT and
HT/FT derivation failures stay per-match warnings, but the run ends with ONE
error summarising how many matches failed — a systematic failure must not
read like noise."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
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


precompute = _load("precompute_predictions", "precompute_predictions.py")


# ── the artifact loader ──────────────────────────────────────────────


def test_missing_artifact_logs_error(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="precompute_predictions"):
        assert precompute._load_dixon_coles_artifact(str(tmp_path / "nope" / "model.bin"), "HT") is None
    rec = [r for r in caplog.records if "artifact not found" in r.getMessage()]
    assert rec and rec[0].levelno == logging.ERROR and "HT markets disabled" in rec[0].getMessage()


def test_unloadable_artifact_logs_error(tmp_path, caplog):
    bad = tmp_path / "model.bin"
    bad.write_text("{not json")
    with caplog.at_level(logging.INFO, logger="precompute_predictions"):
        assert precompute._load_dixon_coles_artifact(str(bad), "2H") is None
    rec = [r for r in caplog.records if "load failed" in r.getMessage()]
    assert rec and rec[0].levelno == logging.ERROR


# ── end-of-run summary ───────────────────────────────────────────────


class FakeEnsemble:
    models: dict = {}  # no FT Dixon-Coles member: the FT derivation block is skipped

    def predict_proba(self, df):
        return np.array([[0.5, 0.3, 0.2]])


class FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return {"features": {"elo_diff": 12.0}}


class FakeConn(FakeCursor):
    def cursor(self, cursor_factory=None):
        return FakeCursor()

    def commit(self):
        pass


class DCModel:
    """Stands in for a loaded HT / 2H DixonColesPredictor."""

    def __init__(self, fail_for=()):
        self.rho = -0.01
        self.fail_for = set(fail_for)

    def lambdas_for_match(self, home, away, league=None):
        if home in self.fail_for:
            raise KeyError(home)
        return 0.7, 0.5


@pytest.fixture
def serve_path(monkeypatch):
    """run() with its DB, Redis and model-registry edges stubbed out."""

    class PredictionService:
        def __init__(self, db=None):
            self.models = {"soccer:match_result": FakeEnsemble()}

    services = types.ModuleType("services")
    cache_mod = types.ModuleType("services.cache_service")
    cache_mod.CacheService = type("CacheService", (), {"set_prediction": lambda self, *a, **k: None})
    pred_mod = types.ModuleType("services.prediction_service")
    pred_mod.PredictionService = PredictionService
    pred_mod.get_model_version = lambda: "v-test"
    pred_mod.load_models_into_process = lambda: None
    for name, mod in (
        ("services", services),
        ("services.cache_service", cache_mod),
        ("services.prediction_service", pred_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    matches = [
        {"match_id": f"m{i}", "home_team": home, "away_team": "Away", "league_id": "L1"}
        for i, home in enumerate(["Good", "Bad", "Good2"])
    ]
    monkeypatch.setattr(precompute, "list_upcoming", lambda url, days: matches)
    monkeypatch.setattr(precompute, "compute_feature_medians", lambda url: {})
    monkeypatch.setattr(precompute, "psycopg2", types.SimpleNamespace(connect=lambda url: FakeConn()))
    monkeypatch.setattr(precompute, "store_prediction", lambda *a, **k: None)
    monkeypatch.setattr(precompute, "store_market_predictions", lambda cur, mid, markets, *a, **k: len(markets))

    def configure(ht_fail_for=(), h2_fail_for=()):
        monkeypatch.setattr(precompute, "_load_halftime_dixon_coles", lambda: DCModel(ht_fail_for))
        monkeypatch.setattr(precompute, "_load_second_half_dixon_coles", lambda: DCModel(h2_fail_for))

    return configure


def summaries(caplog):
    return [r for r in caplog.records if "Half-time derivation failures this run" in r.getMessage()]


def test_failures_end_the_run_with_one_error(serve_path, caplog):
    serve_path(ht_fail_for={"Bad"}, h2_fail_for={"Good2"})
    with caplog.at_level(logging.INFO, logger="precompute_predictions"):
        out = precompute.run("postgresql://unused", days=7, notify_threshold=0.99, notify=False)
    assert out["predicted"] == 3
    rec = summaries(caplog)
    assert len(rec) == 1 and rec[0].levelno == logging.ERROR
    # HT failed for "Bad" (so its HT/FT was never attempted); HT/FT failed for "Good2".
    assert "HT markets failed on 1 of 3 matches, HT/FT on 1 of 2" in rec[0].getMessage()


def test_a_clean_run_logs_no_summary(serve_path, caplog):
    serve_path()
    with caplog.at_level(logging.INFO, logger="precompute_predictions"):
        out = precompute.run("postgresql://unused", days=7, notify_threshold=0.99, notify=False)
    assert out["predicted"] == 3 and out["market_rows"] > 0
    assert summaries(caplog) == []
