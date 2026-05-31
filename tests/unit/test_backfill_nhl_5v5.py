"""Unit tests for backfill_nhl_5v5_stats — pure parsing + xG math.
No DB or HTTP."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bf = _load("backfill_nhl_5v5_stats", "backfill_nhl_5v5_stats.py")


# ── xG approximation ────────────────────────────────────────────────


class TestShotXG:
    def test_zero_distance_returns_base_rate(self):
        # Shot at the net (89, 0) — exp(-0/30) = 1.0, so xG = base_rate.
        xg = bf.shot_xg("wrist", 89, 0)
        assert xg == pytest.approx(bf.XG_BASE_RATE_BY_TYPE["wrist"], rel=1e-9)

    def test_long_distance_decays(self):
        # Slap shot from the far blue line (x=25, y=0) — distance = 64.
        # exp(-64/30) ≈ 0.118. xG = 0.07 * 0.118 ≈ 0.0083.
        xg = bf.shot_xg("slap", 25, 0)
        expected = bf.XG_BASE_RATE_BY_TYPE["slap"] * math.exp(-64 / 30)
        assert xg == pytest.approx(expected, rel=1e-6)

    def test_unknown_shot_type_uses_default(self):
        xg = bf.shot_xg("space_laser", 89, 0)
        assert xg == pytest.approx(bf.XG_BASE_RATE_BY_TYPE["_default"], rel=1e-9)

    def test_missing_coordinates_returns_zero(self):
        # Penalty-shot adjudications don't carry coordinates.
        assert bf.shot_xg("wrist", None, 0) == 0.0
        assert bf.shot_xg("wrist", 89, None) == 0.0
        assert bf.shot_xg(None, None, None) == 0.0

    def test_close_wrist_higher_than_far_slap(self):
        # Sanity check: a close-range wrist is more dangerous than a
        # point shot. If this ever inverts, the xG model is broken.
        close_wrist = bf.shot_xg("wrist", 80, 0)  # ~9 ft from net
        far_slap = bf.shot_xg("slap", 30, 0)  # ~59 ft from net
        assert close_wrist > far_slap


# ── 5v5 strength filter + Corsi/Fenwick/SOG/Goals aggregation ──────


def _shot_event(
    type_desc: str,
    team_id: int,
    situation: str = "1551",
    x: float = 70,
    y: float = 0,
    shot_type: str = "wrist",
) -> dict:
    return {
        "typeDescKey": type_desc,
        "situationCode": situation,
        "details": {
            "eventOwnerTeamId": team_id,
            "xCoord": x,
            "yCoord": y,
            "shotType": shot_type,
        },
    }


class TestAggregate5v5:
    def test_5v5_only(self):
        # Two SOG events at 5v5 + one at 5v4 (power-play) — only the
        # 5v5 shots should count toward the team's Corsi.
        plays = [
            _shot_event("shot-on-goal", team_id=1, situation="1551"),
            _shot_event("shot-on-goal", team_id=1, situation="1541"),  # PP, excluded
            _shot_event("shot-on-goal", team_id=1, situation="1551"),
        ]
        agg = bf.aggregate_5v5(plays)
        assert agg[1]["corsi"] == 2
        assert agg[1]["sog"] == 2

    def test_corsi_includes_all_attempt_types(self):
        # Corsi = SOG + missed + blocked. Fenwick = SOG + missed (NOT blocked).
        plays = [
            _shot_event("shot-on-goal", team_id=1),
            _shot_event("missed-shot", team_id=1),
            _shot_event("blocked-shot", team_id=1),
            _shot_event("goal", team_id=1),
        ]
        agg = bf.aggregate_5v5(plays)
        assert agg[1]["corsi"] == 4
        assert agg[1]["fenwick"] == 3  # blocked excluded
        assert agg[1]["sog"] == 2  # SOG + goal
        assert agg[1]["goals"] == 1

    def test_per_team_aggregation(self):
        plays = [
            _shot_event("shot-on-goal", team_id=1),
            _shot_event("shot-on-goal", team_id=2),
            _shot_event("shot-on-goal", team_id=2),
        ]
        agg = bf.aggregate_5v5(plays)
        assert agg[1]["corsi"] == 1
        assert agg[2]["corsi"] == 2

    def test_non_shot_events_ignored(self):
        plays = [
            {"typeDescKey": "faceoff", "situationCode": "1551", "details": {"eventOwnerTeamId": 1}},
            {"typeDescKey": "hit", "situationCode": "1551", "details": {"eventOwnerTeamId": 1}},
            _shot_event("shot-on-goal", team_id=1),
        ]
        agg = bf.aggregate_5v5(plays)
        assert agg[1]["corsi"] == 1

    def test_missing_team_id_skipped(self):
        plays = [
            {"typeDescKey": "shot-on-goal", "situationCode": "1551", "details": {}},
            _shot_event("shot-on-goal", team_id=1),
        ]
        agg = bf.aggregate_5v5(plays)
        assert list(agg.keys()) == [1]
        assert agg[1]["corsi"] == 1

    def test_blocked_shots_dont_add_xg(self):
        # xG only summed for unblocked attempts — blocked shots never
        # reach the net so their conversion probability is 0.
        plays = [_shot_event("blocked-shot", team_id=1, x=80, y=0)]
        agg = bf.aggregate_5v5(plays)
        assert agg[1]["corsi"] == 1
        assert agg[1]["fenwick"] == 0
        assert agg[1]["xg"] == 0.0


# ── Constants lockdown ──────────────────────────────────────────────


class TestConstants:
    def test_5v5_situation_code(self):
        # If this changes, the analytics community broke their convention.
        # Locked here so any drift triggers a test failure rather than
        # silently mis-filtering shot events.
        assert bf.EVEN_STRENGTH_5V5 == "1551"

    def test_xg_base_rates_calibrated_to_shot_quality(self):
        # Tip-ins and deflections should have the highest base xG (close-range
        # redirections); wrap-arounds the lowest of the tracked types.
        rates = bf.XG_BASE_RATE_BY_TYPE
        assert rates["tip-in"] > rates["wrist"] > rates["slap"]
        assert rates["wrap-around"] < rates["slap"]
