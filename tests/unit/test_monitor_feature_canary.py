"""The neutral-default feature canary in scripts/monitor_models.py.

Every compute_features_*.py swaps a missing input for a NEUTRAL_DEFAULT, so a
broken feed makes features go flat without any error. The canary pages when a
probe's default rate JUMPS against its own trailing baseline, and reports a
probe stuck at default in both windows as a standing (warn-only) defect.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
ML_SRC = REPO_ROOT / "services" / "ml-models" / "src"
for p in (str(SCRIPTS_DIR), str(ML_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mm = _load("monitor_models", "monitor_models.py")


@pytest.mark.parametrize(
    "counts, expected",
    [
        ((40, 30, 200, 20), "alert"),  # 75% recent vs 10% baseline: a feed broke
        ((40, 10, 200, 20), None),  # 25% recent: a jump, but below the 50% floor
        ((40, 39, 200, 196), "warn"),  # stuck at default in both windows
        ((40, 30, 200, 120), None),  # 75% vs 60%: no jump, not stuck
        ((10, 10, 200, 0), None),  # recent window too small to say anything
        ((40, 38, 0, 0), "warn"),  # no baseline at all but stuck now
        ((40, 30, 5, 0), None),  # thin baseline: a jump there is not trusted
    ],
)
def test_classify_default_rate(counts, expected):
    severity, recent, base = mm.classify_default_rate(*counts)
    assert severity == expected
    assert recent == pytest.approx(counts[1] / counts[0])


def test_small_window_still_reports_its_rates():
    severity, recent, base = mm.classify_default_rate(12, 0, 61, 0)
    assert severity is None
    assert recent == 0.0 and base == 0.0  # not None: the log must show the real baseline


def test_every_probe_resolves_a_numeric_neutral_default():
    for sport, feature_set, module_name, key, label in mm.FEATURE_CANARY_PROBES:
        assert mm._neutral_default(module_name, key) is not None, (sport, key)


class _Cursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append(params)
        if isinstance(self.rows[0], Exception):
            raise self.rows.pop(0)

    def fetchone(self):
        return self.rows.pop(0)


def test_canary_emits_an_alert_for_a_jump_and_a_warn_for_a_stuck_probe(monkeypatch):
    monkeypatch.setattr(
        mm,
        "FEATURE_CANARY_PROBES",
        [
            ("nba", "nba_baseline", "compute_features_nba", "odds_home_ml", "moneyline odds"),
            ("nhl", "nhl_baseline", "compute_features_nhl", "home_goalie_roll_save_pct", "starting goalie"),
            ("soccer", "baseline", "compute_features", "odds_home", "1x2 odds"),
        ],
    )
    cur = _Cursor(
        [
            {"n_recent": 30, "d_recent": 27, "n_base": 100, "d_base": 5},
            {"n_recent": 20, "d_recent": 20, "n_base": 180, "d_base": 180},
            {"n_recent": 200, "d_recent": 52, "n_base": 600, "d_base": 160},
        ]
    )
    findings = mm.feature_default_canary(cur)
    by_sport = {f.sport: f for f in findings}
    assert set(by_sport) == {"nba", "nhl"}
    assert by_sport["nba"].severity == "alert" and "JUMPED" in by_sport["nba"].message
    assert by_sport["nhl"].severity == "warn" and "STUCK" in by_sport["nhl"].message
    assert by_sport["nba"].n == 30
    # the SQL compares against the module's own default, not a copied literal
    assert cur.calls[0]["default"] == pytest.approx(mm._neutral_default("compute_features_nba", "odds_home_ml"))


def test_a_failing_probe_becomes_a_visible_warning_not_a_crash(monkeypatch):
    monkeypatch.setattr(
        mm, "FEATURE_CANARY_PROBES", [("nba", "nba_baseline", "compute_features_nba", "odds_home_ml", "odds")]
    )
    findings = mm.feature_default_canary(_Cursor([RuntimeError("relation does not exist")]))
    assert len(findings) == 1
    assert findings[0].severity == "warn"
    assert "could not evaluate" in findings[0].message
