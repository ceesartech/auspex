"""monitor_models while horse racing is dormant (2026-11-01): the horse slices
keep being computed from a shrinking 30-day window of old races, so once no
race prediction has been CREATED for HORSE_DORMANT_AFTER_HOURS their findings
are logged as warnings, not paged — and only then."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO / "scripts"
ML_SRC = REPO / "services" / "ml-models" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mm = _load("monitor_models")
cm = _load("calibration_metrics")


class OneRowCursor:
    def __init__(self, row):
        self.row, self.sql = row, []

    def execute(self, sql, params=None):
        self.sql.append((sql, params))

    def fetchone(self):
        return self.row


def test_horse_racing_dormant_reads_the_newest_prediction():
    cur = OneRowCursor({"dormant": True})
    assert mm.horse_racing_dormant(cur) is True
    sql, params = cur.sql[0]
    assert "MAX(created_at)" in sql and "race_predictions" in sql and params == (mm.HORSE_DORMANT_AFTER_HOURS,)
    assert mm.horse_racing_dormant(OneRowCursor({"dormant": False})) is False
    assert mm.horse_racing_dormant(OneRowCursor(None)) is True
    assert mm.HORSE_DORMANT_AFTER_HOURS >= 72  # an enabled pipeline goes ~66 h without a race over Christmas


@pytest.mark.parametrize("dormant", [False, True])
def test_a_dormant_horse_stream_is_logged_not_paged(monkeypatch, dormant):
    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def cursor(self, cursor_factory=None):
            return self

        def commit(self):
            pass

        def rollback(self):
            pass

    sent = []
    horse_slice = {"sport": "horse_racing", "prediction_type": "win", "model_name": "market_consensus_v1", "n": 400}
    monkeypatch.setattr(mm.psycopg2, "connect", lambda *a, **k: Conn())
    monkeypatch.setattr(mm, "fetch_slices", lambda cur, days, n: [])
    monkeypatch.setattr(mm, "fetch_horse_racing_slices", lambda cur, days, n: [dict(horse_slice)])
    # Badly miscalibrated: predicted 0.9, never won.
    monkeypatch.setattr(mm, "fetch_horse_racing_pairs", lambda cur, t, m, d: ([0.9] * 400, [0] * 400))
    monkeypatch.setattr(mm, "horse_racing_dormant", lambda cur: dormant)
    monkeypatch.setattr(mm, "constant_prior_canary", lambda cur: [])
    monkeypatch.setattr(mm, "feature_default_canary", lambda cur: [])
    monkeypatch.setattr(mm, "persist_reports", lambda cur, summary, days: 0)
    monkeypatch.setattr(mm, "select_new_alerts", lambda findings: [f for f in findings if f.severity == "alert"])
    monkeypatch.setattr(mm, "mark_paged", lambda findings: 0)
    monkeypatch.setattr(mm, "send_telegram_digest", lambda alerts, header=None: sent.append(alerts) or 1)
    monkeypatch.setattr(mm, "_stream_is_gated", lambda sport, ptype: False)
    out = mm.run("postgresql://x", 30, 30, cm.DriftThresholds())
    if dormant:
        assert out["alerts"] == 0 and out["warnings"] > 0 and sent == []
    else:
        assert out["alerts"] > 0 and len(sent) == 1
