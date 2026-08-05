"""Unit tests for fetch_lottery_winners — ticks conversion, ASMX payload
parsing, tier aggregation. No network or DB."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fw = _load("fetch_lottery_winners", REPO / "scripts" / "fetch_lottery_winners.py")


class TestDotnetTicks:
    def test_verified_fixtures(self):
        # Both values verified live against GetDrawDataByTick responses.
        assert fw.dotnet_ticks(date(2026, 7, 31)) == 639210528000000000
        assert fw.dotnet_ticks(date(2019, 6, 14)) == 636960672000000000

    def test_one_day_is_864e9_ticks(self):
        assert fw.dotnet_ticks(date(2026, 8, 1)) - fw.dotnet_ticks(date(2026, 7, 31)) == 864_000_000_000


def _payload(drawing: dict, jackpot: dict, tiers: list[dict]) -> dict:
    return {"d": json.dumps({"Drawing": drawing, "Jackpot": jackpot, "PrizeTiers": tiers})}


DRAWING = {"PlayDate": "2026-07-31T00:00:00", "N1": 4, "N2": 18, "N3": 26, "N4": 43, "N5": 51, "MBall": 4}
JACKPOT = {"CurrentPrizePool": 50_000_000.0, "CurrentCashValue": 21_500_000.0, "Winners": 0}


class TestParsePayload:
    def test_current_era_multiplier_rows_are_summed_per_tier(self):
        tiers = [
            {"Tier": 0, "IsMegaplier": False, "Winners": 0, "Multiplier": ""},
            {"Tier": 2, "IsMegaplier": False, "Winners": 3, "Multiplier": "2x"},
            {"Tier": 2, "IsMegaplier": False, "Winners": 1, "Multiplier": "10x"},
            {"Tier": 8, "IsMegaplier": False, "Winners": 40_000, "Multiplier": "2x"},
            {"Tier": 8, "IsMegaplier": False, "Winners": 12_000, "Multiplier": "3x"},
        ]
        parsed = fw.parse_bytick_payload(_payload(DRAWING, JACKPOT, tiers))
        assert parsed["numbers"] == [4, 18, 26, 43, 51]
        assert parsed["bonus"] == 4
        assert parsed["jackpot_amount"] == 50_000_000.0
        assert parsed["cash_value"] == 21_500_000.0
        assert parsed["winners_by_tier"]["match_4_bonus"] == 4
        assert parsed["winners_by_tier"]["match_bonus"] == 52_000
        assert parsed["winners_by_tier"]["jackpot"] == 0

    def test_megaplier_era_base_and_megaplier_rows_both_count(self):
        tiers = [
            {"Tier": 5, "IsMegaplier": False, "Winners": 20_000, "Multiplier": None},
            {"Tier": 5, "IsMegaplier": True, "Winners": 6_000, "Multiplier": None},
        ]
        parsed = fw.parse_bytick_payload(_payload(DRAWING, JACKPOT, tiers))
        assert parsed["winners_by_tier"]["match_3"] == 26_000

    def test_unknown_tier_index_skipped_loudly(self):
        tiers = [{"Tier": 42, "IsMegaplier": False, "Winners": 5, "Multiplier": ""}]
        parsed = fw.parse_bytick_payload(_payload(DRAWING, JACKPOT, tiers))
        assert all(v == 0 for v in parsed["winners_by_tier"].values())

    def test_malformed_payload_returns_none(self):
        assert fw.parse_bytick_payload({"d": "not json"}) is None
        assert fw.parse_bytick_payload({}) is None

    def test_tier_labels_match_rules_registry_order(self):
        # 9 tiers, jackpot first, match_bonus last — the mapping the whole
        # v1.1 analysis rests on.
        assert fw.TIER_LABELS[0] == "jackpot"
        assert fw.TIER_LABELS[1] == "match_5"
        assert fw.TIER_LABELS[8] == "match_bonus"
        assert len(fw.TIER_LABELS) == 9
