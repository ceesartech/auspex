"""Unit tests for precompute_predictions_nhl — per-market thresholds +
Alert construction. The actual Telegram dispatch path is covered by
test_telegram_notify.py since both scripts share the same helper."""

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


# Load telegram_notify first so precompute_predictions_nhl's import resolves.
_load("telegram_notify", "telegram_notify.py")
ppn = _load("precompute_predictions_nhl", "precompute_predictions_nhl.py")


class TestMarketThresholds:
    def test_thresholds_are_inside_realistic_band(self):
        # Each threshold should sit above the coin-flip floor (0.50 for
        # 2-class, 0.33 for 3-class) but below the "model is broken"
        # ceiling (0.85+). The previous puck_line/total of 0.70 was
        # high enough to produce zero NHL alerts during playoff
        # wind-down — locked here so we don't drift back up.
        t = ppn.MARKET_NOTIFY_THRESHOLDS
        assert 0.55 <= t["moneyline"] <= 0.75
        assert 0.45 <= t["regulation"] <= 0.65
        assert 0.55 <= t["puck_line"] <= 0.65
        assert 0.55 <= t["total"] <= 0.65

    def test_all_nhl_markets_have_a_threshold(self):
        for market in ("moneyline", "regulation", "puck_line", "total"):
            assert market in ppn.MARKET_NOTIFY_THRESHOLDS


class TestNhlAlert:
    def test_translates_to_friendly_market_label(self):
        # The Alert that lands in the digest must NOT carry the raw
        # snake_case 'puck_line' — it has to be translated to "Puck Line"
        # at the alert-construction boundary so the digest stays
        # presentational.
        alert = ppn.nhl_alert(
            league_name="NHL",
            home_team="Canadiens",
            away_team="Rangers",
            match_date=datetime(2025, 3, 15, 19, 0),
            market="puck_line",
            predicted_outcome="cover",
            confidence=0.74,
            probabilities={"cover": 0.74, "no_cover": 0.26},
        )
        assert alert.sport == "nhl"
        assert alert.market_label == "Puck Line"
        assert alert.confidence == 0.74
        assert alert.probabilities == {"cover": 0.74, "no_cover": 0.26}

    def test_unknown_market_falls_through_to_raw_string(self):
        # Defensive: a market we haven't added to MARKET_DISPLAY_LABELS
        # yet should still produce an Alert, just with the raw value.
        # The label map is a best-effort prettifier, not a validator.
        alert = ppn.nhl_alert(
            league_name="NHL",
            home_team="A",
            away_team="B",
            match_date=datetime(2025, 3, 15, 19, 0),
            market="first_period_winner",
            predicted_outcome="home",
            confidence=0.71,
            probabilities={"home": 0.71, "away": 0.29},
        )
        assert alert.market_label == "first_period_winner"


class TestMarketDisplayLabels:
    def test_all_thresholded_markets_have_display_label(self):
        # The display map and threshold map must cover the same keys —
        # otherwise a digest line could render with raw 'puck_line'.
        assert set(ppn.MARKET_DISPLAY_LABELS) == set(ppn.MARKET_NOTIFY_THRESHOLDS)
