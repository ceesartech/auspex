"""Unit tests for precompute_predictions_nfl — per-market thresholds +
Alert construction. Mirrors test_precompute_predictions_nba — same
surface area (threshold map sanity, Alert builder, market labels)
since the NFL precompute is a near-direct port of the NBA one.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("telegram_notify", "telegram_notify.py")
ppn = _load("precompute_predictions_nfl", "precompute_predictions_nfl.py")


class TestMarketThresholds:
    def test_thresholds_reflect_per_market_model_ceiling(self):
        # Same invariant as NBA — moneyline ≥ spread/total. If someone
        # bumps spread/total above moneyline they'll see near-zero
        # alerts because the model can't reach that confidence on the
        # smaller NFL training corpus.
        t = ppn.MARKET_NOTIFY_THRESHOLDS
        assert t["spread"] <= t["moneyline"]
        assert t["total"] <= t["moneyline"]

    def test_all_nfl_markets_have_a_threshold(self):
        for market in ("moneyline", "spread", "total"):
            assert market in ppn.MARKET_NOTIFY_THRESHOLDS

    def test_thresholds_inside_realistic_band(self):
        for v in ppn.MARKET_NOTIFY_THRESHOLDS.values():
            assert 0.55 <= v <= 0.75


class TestNflAlert:
    def test_translates_to_friendly_market_label(self):
        alert = ppn.nfl_alert(
            league_name="NFL",
            home_team="Kansas City Chiefs",
            away_team="Buffalo Bills",
            match_date=datetime(2025, 1, 12, 18, 0),
            market="spread",
            predicted_outcome="home",
            confidence=0.62,
            probabilities={"home": 0.62, "away": 0.38},
        )
        assert alert.sport == "nfl"
        assert alert.market_label == "Spread"
        assert alert.confidence == 0.62
        assert alert.probabilities == {"home": 0.62, "away": 0.38}

    def test_unknown_market_falls_through_to_raw(self):
        alert = ppn.nfl_alert(
            league_name="NFL",
            home_team="A",
            away_team="B",
            match_date=datetime(2025, 1, 12, 18, 0),
            market="first_quarter_winner",
            predicted_outcome="home",
            confidence=0.71,
            probabilities={"home": 0.71, "away": 0.29},
        )
        assert alert.market_label == "first_quarter_winner"


class TestMarketDisplayLabels:
    def test_all_thresholded_markets_have_display_label(self):
        assert set(ppn.MARKET_DISPLAY_LABELS) == set(ppn.MARKET_NOTIFY_THRESHOLDS)


class TestConstants:
    def test_feature_set_pinned_to_nfl_baseline(self):
        # Lock the feature_set/version pair — must match what
        # scripts/compute_features_nfl.py writes, or list_upcoming_nfl
        # returns 0 rows silently.
        assert ppn.FEATURE_SET == "nfl_baseline"
        assert ppn.FEATURE_VERSION == "v1"
