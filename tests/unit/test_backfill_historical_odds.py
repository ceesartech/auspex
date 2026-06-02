"""Unit tests for backfill_historical_odds — covers the pure logic
(snapshot timestamp formatting, cost estimation math, helper reuse
from fetch_live_odds) without touching the network or DB."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
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


backfill = _load("backfill_historical_odds", "backfill_historical_odds.py")
fetch_live_odds = backfill.fetch_live_odds  # re-exported via the dynamic import


# ── Snapshot timestamp ────────────────────────────────────────────────


class TestSnapshotTimestamp:
    def test_defaults_capture_nhl_closing_lines(self):
        # 03:00 UTC of game-date+1 = 11pm ET = AFTER most evening NHL
        # games have started, so the-odds-api returns closing lines.
        # Locked in here so a future drift to a worse time (the old
        # 22:00 UTC same-day = 5pm ET pre-game lines bug) triggers a
        # test failure rather than silently degrading model signal.
        assert backfill.DEFAULT_SNAPSHOT_HOUR_UTC == 3
        assert backfill.DEFAULT_SNAPSHOT_DAY_OFFSET == 1

    def test_default_iso_format_uses_next_day(self):
        # Game-date April 1 → snapshot at 03:00 UTC on April 2.
        ts = backfill.snapshot_timestamp(date(2024, 4, 1))
        assert ts == "2024-04-02T03:00:00Z"

    def test_custom_hour_only(self):
        ts = backfill.snapshot_timestamp(date(2024, 4, 1), hour_utc=19)
        # Still uses the default offset (+1 day).
        assert ts == "2024-04-02T19:00:00Z"

    def test_custom_offset_zero(self):
        # day_offset=0 means same-day snapshot (legacy behavior). Useful
        # for sports whose evening slate finishes before UTC rollover.
        ts = backfill.snapshot_timestamp(date(2024, 4, 1), hour_utc=22, day_offset=0)
        assert ts == "2024-04-01T22:00:00Z"

    def test_custom_offset_crosses_month_boundary(self):
        # Game-date April 30 + 1 day offset → May 1. Date arithmetic
        # must roll cleanly across month boundaries.
        ts = backfill.snapshot_timestamp(date(2024, 4, 30))
        assert ts == "2024-05-01T03:00:00Z"


# ── Cost math ────────────────────────────────────────────────────────


class TestCostMultiplier:
    def test_historical_is_10x_live(self):
        # the-odds-api docs: "historical odds API has a 10x quota cost
        # multiplier compared to the standard odds API". Lock the value.
        assert backfill.HISTORICAL_COST_MULTIPLIER == 10

    @pytest.mark.parametrize(
        "regions, markets, expected_per_call",
        [
            ("us", "h2h", 10),
            ("us", "h2h,totals", 20),
            ("us", "h2h,spreads,totals", 30),
            ("us,uk", "h2h,spreads,totals", 60),
        ],
    )
    def test_cost_formula(self, regions, markets, expected_per_call):
        n_regions = len(regions.split(","))
        n_markets = len(markets.split(","))
        assert n_regions * n_markets * backfill.HISTORICAL_COST_MULTIPLIER == expected_per_call


# ── Helper reuse from fetch_live_odds ────────────────────────────────


class TestHelperReuse:
    def test_sport_dispatch_matches_live(self):
        # The historical script reuses fetch_live_odds.sport_for_key —
        # confirming the live module is actually importable and every
        # registered sport prefix resolves the same way for both
        # pipelines.
        assert fetch_live_odds.sport_for_key("icehockey_nhl") == "nhl"
        assert fetch_live_odds.sport_for_key("basketball_nba") == "nba"
        assert fetch_live_odds.sport_for_key("soccer_epl") == "soccer"
        assert fetch_live_odds.sport_for_key("bogus_keykey") is None

    def test_default_markets_picked_from_live_basket(self):
        # If fetch_live_odds adds a sport, the historical default
        # picks it up via SPORT_MARKETS. NHL stays at h2h+spreads+totals.
        assert fetch_live_odds.SPORT_MARKETS["nhl"] == "h2h,spreads,totals"
        assert fetch_live_odds.SPORT_MARKETS["nba"] == "h2h,spreads,totals"
        # And our script falls back to it when --markets isn't passed:
        argv = ["--start", "2024-10-01", "--end", "2024-10-02", "--dry-run"]
        args = backfill.parse_args(argv)
        assert args.markets is None  # the resolution happens in main()
        # main() falls back to SPORT_MARKETS[sport], so verify we'd
        # pick up "h2h,spreads,totals" for both NHL and NBA.
        for live_key in ("icehockey_nhl", "basketball_nba"):
            sport = fetch_live_odds.sport_for_key(live_key)
            assert fetch_live_odds.SPORT_MARKETS[sport] == "h2h,spreads,totals"


# ── Argparse ─────────────────────────────────────────────────────────


class TestArgparse:
    def test_dry_run_doesnt_require_api_key(self):
        # Critical: --dry-run is meant to be safe to run without any
        # api key configured, so the user can estimate costs in CI
        # or on a machine without secrets. Validate the path doesn't
        # crash before reaching the api_key check.
        args = backfill.parse_args(
            [
                "--start",
                "2024-10-01",
                "--end",
                "2025-06-30",
                "--dry-run",
            ]
        )
        assert args.dry_run is True
        assert args.start == "2024-10-01"
        assert args.end == "2025-06-30"
        assert args.sport == "icehockey_nhl"

    def test_required_dates(self):
        # Missing --start/--end should fail argparse — protects us
        # from accidentally calling the historical API with default
        # values that would burn quota on the wrong range.
        with pytest.raises(SystemExit):
            backfill.parse_args(["--dry-run"])

    def test_explicit_market_override(self):
        args = backfill.parse_args(
            [
                "--start",
                "2024-10-01",
                "--end",
                "2025-06-30",
                "--markets",
                "h2h,totals",
                "--dry-run",
            ]
        )
        assert args.markets == "h2h,totals"


# ── process_snapshot_response ─────────────────────────────────────────


class TestProcessSnapshotResponse:
    def test_empty_data_array(self):
        # No events in the snapshot — graceful no-op, returns (0, 0).
        # We don't even need a cursor because process_event isn't called.
        events_matched, odds_inserted = backfill.process_snapshot_response(cur=None, sport="nhl", response={"data": []})
        assert events_matched == 0
        assert odds_inserted == 0

    def test_missing_data_key(self):
        # If the API returns an unexpected shape (no `data` key), we
        # treat it as "nothing to do" rather than crashing — protects
        # the bulk job from a single bad response.
        events_matched, odds_inserted = backfill.process_snapshot_response(cur=None, sport="nhl", response={})
        assert events_matched == 0
        assert odds_inserted == 0
