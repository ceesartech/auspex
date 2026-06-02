"""Unit tests for precompute_predictions_nba — per-market thresholds +
Alert construction.

The Telegram dispatch path itself is covered by test_telegram_notify
(shared with NHL). Here we only verify the NBA-specific surface:
threshold map sanity, Alert builder, market labels.
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


# Load telegram_notify first so the precompute import resolves.
_load("telegram_notify", "telegram_notify.py")
ppn = _load("precompute_predictions_nba", "precompute_predictions_nba.py")


class TestMarketThresholds:
    def test_thresholds_reflect_per_market_model_ceiling(self):
        # The threshold floor must match what each market's model can
        # actually produce. Moneyline (75% val accuracy) clears 0.60
        # often on heavy favorites; spread/total (~60% accuracy) sit
        # closer to their confidence ceiling and need a SLIGHTLY
        # lower floor — 0.58 — to surface picks at all.
        # If someone bumps spread/total above moneyline they'll get
        # near-zero alerts because the model can't reach that confidence.
        t = ppn.MARKET_NOTIFY_THRESHOLDS
        assert t["spread"] <= t["moneyline"]
        assert t["total"] <= t["moneyline"]

    def test_all_nba_markets_have_a_threshold(self):
        for market in ("moneyline", "spread", "total"):
            assert market in ppn.MARKET_NOTIFY_THRESHOLDS

    def test_thresholds_inside_realistic_band(self):
        # Above coin-flip (0.50) for 2-class markets; below the
        # "model is broken" ceiling (~0.85). Locks the band so a
        # well-meaning tweak can't push the floor under 0.55.
        for v in ppn.MARKET_NOTIFY_THRESHOLDS.values():
            assert 0.55 <= v <= 0.75


class TestNbaAlert:
    def test_translates_to_friendly_market_label(self):
        # The Alert that lands in the digest must NOT carry raw
        # snake_case 'spread' / 'total' — has to be the human label.
        alert = ppn.nba_alert(
            league_name="NBA",
            home_team="Boston Celtics",
            away_team="Dallas Mavericks",
            match_date=datetime(2025, 6, 13, 0, 30),
            market="spread",
            predicted_outcome="home",
            confidence=0.64,
            probabilities={"home": 0.64, "away": 0.36},
        )
        assert alert.sport == "nba"
        assert alert.market_label == "Spread"
        assert alert.confidence == 0.64
        assert alert.probabilities == {"home": 0.64, "away": 0.36}

    def test_unknown_market_falls_through_to_raw(self):
        # Defensive — a market we haven't added to MARKET_DISPLAY_LABELS
        # yet still produces a valid Alert, just with the raw value.
        alert = ppn.nba_alert(
            league_name="NBA",
            home_team="A",
            away_team="B",
            match_date=datetime(2025, 6, 13, 0, 30),
            market="first_quarter_winner",
            predicted_outcome="home",
            confidence=0.71,
            probabilities={"home": 0.71, "away": 0.29},
        )
        assert alert.market_label == "first_quarter_winner"


class TestMarketDisplayLabels:
    def test_all_thresholded_markets_have_display_label(self):
        # The display map + threshold map must cover the same keys —
        # otherwise a digest line could render with raw 'spread'.
        assert set(ppn.MARKET_DISPLAY_LABELS) == set(ppn.MARKET_NOTIFY_THRESHOLDS)


class TestConstants:
    def test_feature_set_pinned_to_nba_baseline(self):
        # Lock the feature_set/version pair — must match what
        # scripts/compute_features_nba.py writes, or list_upcoming_nba
        # returns 0 rows silently.
        assert ppn.FEATURE_SET == "nba_baseline"
        assert ppn.FEATURE_VERSION == "v1"
