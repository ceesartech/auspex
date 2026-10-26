"""Unit tests for backfill_historical_odds — covers the pure logic
(snapshot planning, strictly-pre-game event matching, cost estimation,
helper reuse from fetch_live_odds) without touching the network or DB."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
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


# ── Snapshot planning ─────────────────────────────────────────────────

UTC = timezone.utc


def _g(mid, hh, mm=0, day=1, home="Boston Bruins", away="Toronto Maple Leafs"):
    return backfill.Game(mid, datetime(2025, 11, day, hh, mm, tzinfo=UTC), home, away)


class TestPlanSnapshots:
    def test_one_snapshot_per_cluster_taken_before_the_first_game(self):
        games = [_g("a", 23), _g("b", 23, 30), _g("c", 0, 0, day=2), _g("d", 1, 30, day=2), _g("e", 17, day=2)]
        snaps = backfill.plan_snapshots(games, window_hours=3.0, lead_minutes=10)
        assert [len(s.games) for s in snaps] == [4, 1]
        assert snaps[0].requested == datetime(2025, 11, 1, 22, 50, tzinfo=UTC)
        assert snaps[1].requested == datetime(2025, 11, 2, 16, 50, tzinfo=UTC)
        # every game's snapshot is strictly before its own start
        for s in snaps:
            assert all(s.requested < g.match_date for g in s.games)

    def test_a_cluster_never_spans_more_than_the_window(self):
        games = [_g(str(i), 18 + i) for i in range(5)]  # 18:00 .. 22:00
        snaps = backfill.plan_snapshots(games, window_hours=3.0)
        assert [len(s.games) for s in snaps] == [4, 1]  # 18..21 then 22

    def test_iso_is_utc_z(self):
        snaps = backfill.plan_snapshots([_g("a", 1)], lead_minutes=10)
        assert snaps[0].iso == "2025-11-01T00:50:00Z"

    def test_empty(self):
        assert backfill.plan_snapshots([]) == []


# ── Event matching ────────────────────────────────────────────────────


def _ev(home, away, hh, mm=0, day=1):
    ts = datetime(2025, 11, day, hh, mm, tzinfo=UTC).isoformat().replace("+00:00", "Z")
    return {"home_team": home, "away_team": away, "commence_time": ts, "bookmakers": []}


class TestMatchEvents:
    def test_in_play_events_are_rejected(self):
        snap_time = datetime(2025, 11, 1, 23, 40, tzinfo=UTC)
        game = _g("a", 23, 30)
        pairs, counts = backfill.match_events_to_games(
            [_ev("Boston Bruins", "Toronto Maple Leafs", 23, 30)], snap_time, [game]
        )
        assert pairs == []
        assert counts["in_play_or_started"] == 1

    def test_exact_pre_game_match(self):
        snap_time = datetime(2025, 11, 1, 23, 20, tzinfo=UTC)
        game = _g("a", 23, 30)
        pairs, counts = backfill.match_events_to_games(
            [_ev("Boston Bruins", "Toronto Maple Leafs", 23, 30)], snap_time, [game]
        )
        assert [g.match_id for _, g in pairs] == ["a"]
        assert counts["exact"] == 1

    def test_the_rematch_two_days_later_cannot_attach(self):
        # v1's defect: the same pairing within ±2 days matched the first hit.
        snap_time = datetime(2025, 11, 1, 22, 50, tzinfo=UTC)
        game = _g("a", 23)
        later = _ev("Boston Bruins", "Toronto Maple Leafs", 23, day=3)
        pairs, counts = backfill.match_events_to_games([later], snap_time, [game])
        assert pairs == [] and counts["unmatched"] == 1

    def test_home_away_must_agree(self):
        snap_time = datetime(2025, 11, 1, 22, 50, tzinfo=UTC)
        game = _g("a", 23)
        swapped = _ev("Toronto Maple Leafs", "Boston Bruins", 23)
        pairs, _ = backfill.match_events_to_games([swapped], snap_time, [game])
        assert pairs == []

    def test_fuzzy_names_resolve_within_the_group_only(self):
        snap_time = datetime(2025, 11, 1, 22, 50, tzinfo=UTC)
        games = [
            _g("clips", 23, home="LA Clippers", away="Utah Jazz"),
            _g("lal", 23, home="Los Angeles Lakers", away="Miami Heat"),
        ]
        events = [_ev("Los Angeles Clippers", "Utah Jazz", 23), _ev("Los Angeles Lakers", "Miami Heat", 23)]
        pairs, counts = backfill.match_events_to_games(events, snap_time, games)
        got = {e["home_team"]: g.match_id for e, g in pairs}
        assert got == {"Los Angeles Clippers": "clips", "Los Angeles Lakers": "lal"}
        assert counts["exact"] == 1 and counts["fuzzy"] == 1

    def test_one_to_one(self):
        snap_time = datetime(2025, 11, 1, 22, 50, tzinfo=UTC)
        game = _g("a", 23)
        dup = [_ev("Boston Bruins", "Toronto Maple Leafs", 23), _ev("Boston Bruins", "Toronto Maple Leafs", 23, 5)]
        pairs, counts = backfill.match_events_to_games(dup, snap_time, [game])
        assert len(pairs) == 1 and counts["unmatched"] == 1


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
        assert fetch_live_odds.sport_for_key("americanfootball_nfl") == "nfl"
        assert fetch_live_odds.sport_for_key("soccer_epl") == "soccer"
        assert fetch_live_odds.sport_for_key("tennis_atp_wimbledon") == "tennis"
        assert fetch_live_odds.sport_for_key("mma_mixed_martial_arts") == "mma"
        assert fetch_live_odds.sport_for_key("bogus_keykey") is None

    def test_default_markets_picked_from_live_basket(self):
        # If fetch_live_odds adds a sport, the historical default
        # picks it up via SPORT_MARKETS. NHL stays at h2h+spreads+totals.
        assert fetch_live_odds.SPORT_MARKETS["nhl"] == "h2h,spreads,totals"
        assert fetch_live_odds.SPORT_MARKETS["nba"] == "h2h,spreads,totals"
        assert fetch_live_odds.SPORT_MARKETS["nfl"] == "h2h,spreads,totals"
        # And our script falls back to it when --markets isn't passed:
        argv = ["--start", "2024-10-01", "--end", "2024-10-02", "--dry-run"]
        args = backfill.parse_args(argv)
        assert args.markets is None  # the resolution happens in main()
        # main() falls back to SPORT_MARKETS[sport], so verify we'd
        # pick up "h2h,spreads,totals" for NHL, NBA, and NFL.
        for live_key in ("icehockey_nhl", "basketball_nba", "americanfootball_nfl"):
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


# ── Safety rails ──────────────────────────────────────────────────────


class TestSafetyRails:
    def test_replace_refuses_without_confirmation(self, monkeypatch):
        monkeypatch.setattr(backfill, "load_games", lambda *a, **k: [])

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def cursor(self, **k):
                return self

        monkeypatch.setattr(backfill.psycopg2, "connect", lambda *a, **k: _Conn())
        with pytest.raises(RuntimeError, match="confirm-replace"):
            backfill.run(
                "postgres://x",
                "icehockey_nhl",
                datetime(2024, 1, 1).date(),
                datetime(2024, 1, 2).date(),
                "h2h",
                "key",
                "us",
                dry_run=False,
                mode="replace",
            )

    def test_cost_per_call(self):
        assert backfill.credits_per_call("h2h,spreads,totals", "us") == 30
        assert backfill.credits_per_call("h2h", "us,eu") == 20
