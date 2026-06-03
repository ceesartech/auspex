"""Unit tests for precompute_predictions_mma — per-market thresholds +
Alert construction. Mirror of test_precompute_predictions_nfl —
mma ships a single market (moneyline) in v1, so the test set is
smaller.
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
ppt = _load("precompute_predictions_mma", "precompute_predictions_mma.py")


class TestMarketThresholds:
    def test_moneyline_threshold_at_tour_modal_favorite(self):
        # 0.62 matches the UFC modal favorite implied probability
        # (-200 = 67%). Below that the alert is just echoing the
        # book; above it the model needs to be saying something
        # the market isn't.
        assert ppt.MARKET_NOTIFY_THRESHOLDS["moneyline"] == 0.62

    def test_thresholds_inside_realistic_band(self):
        for v in ppt.MARKET_NOTIFY_THRESHOLDS.values():
            assert 0.55 <= v <= 0.75

    def test_only_moneyline_market_in_v1(self):
        # Locked here so adding a market without wiring the full
        # path (TaskSpec + recs engine + grading) fails this test
        # rather than silently breaking the digest.
        assert set(ppt.MARKET_NOTIFY_THRESHOLDS) == {"moneyline"}


class TestMMAAlert:
    def test_translates_to_friendly_market_label(self):
        alert = ppt.mma_alert(
            league_name="UFC",
            home_team="Israel Adesanya",
            away_team="Dricus Du Plessis",
            match_date=datetime(2024, 8, 17, 22, 0),
            market="moneyline",
            predicted_outcome="home",
            confidence=0.70,
            probabilities={"home": 0.70, "away": 0.28},
        )
        assert alert.sport == "mma"
        # MMA uses "Fight Winner" for the user-facing label —
        # makes more sense than "Moneyline" in a 1v1 context.
        assert alert.market_label == "Fight Winner"
        assert alert.confidence == 0.70
        assert alert.probabilities == {"home": 0.70, "away": 0.28}

    def test_unknown_market_falls_through_to_raw(self):
        # Defensive — set-betting / total-games markets aren't
        # wired in v1; if one slipped through with an alert, it
        # should still produce a valid Alert.
        alert = ppt.mma_alert(
            league_name="UFC",
            home_team="A",
            away_team="B",
            match_date=datetime(2024, 8, 17, 22, 0),
            market="method_of_victory",
            predicted_outcome="KO",
            confidence=0.55,
            probabilities={"KO": 0.55, "3-1": 0.30, "3-2": 0.15},
        )
        assert alert.market_label == "method_of_victory"


class TestMarketDisplayLabels:
    def test_all_thresholded_markets_have_display_label(self):
        # The display map + threshold map must cover the same keys —
        # otherwise a digest line could render with raw 'moneyline'.
        assert set(ppt.MARKET_DISPLAY_LABELS) == set(ppt.MARKET_NOTIFY_THRESHOLDS)

    def test_moneyline_renders_as_match_winner(self):
        # MMA 1v1 → "Fight Winner" reads more naturally than
        # "Moneyline" (which is a team-sport term).
        assert ppt.MARKET_DISPLAY_LABELS["moneyline"] == "Fight Winner"


class TestConstants:
    def test_feature_set_pinned_to_mma_baseline(self):
        # Lock the feature_set/version pair — must match what
        # scripts/compute_features_mma.py writes, or
        # list_upcoming_mma returns 0 rows silently.
        assert ppt.FEATURE_SET == "mma_baseline"
        assert ppt.FEATURE_VERSION == "v1"
