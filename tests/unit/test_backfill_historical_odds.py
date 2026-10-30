"""Unit tests for backfill_historical_odds — covers the pure logic
(snapshot planning, strictly-pre-game event matching, cost estimation,
helper reuse from fetch_live_odds) without touching the network or DB."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
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


# ── Tennis: per-tournament keys, player order agnostic ────────────────


def _tg(mid, when, home, away, venue):
    return backfill.Game(mid, when, home, away, venue)


class TestTennisKeys:
    def test_paris_resolves_by_date(self):
        may = datetime(2025, 5, 28, 10, tzinfo=UTC)
        nov = datetime(2025, 11, 1, 10, tzinfo=UTC)
        assert backfill.tennis_keys_for("Paris, France", may) == ("tennis_atp_french_open", "tennis_wta_french_open")
        assert backfill.tennis_keys_for("paris, France", nov) == ("tennis_atp_paris_masters",)

    def test_accents_and_case_do_not_matter_and_unlisted_events_map_to_nothing(self):
        when = datetime(2023, 4, 10, tzinfo=UTC)
        assert backfill.tennis_keys_for("Monte-Carlo, Monaco", when) == ("tennis_atp_monte_carlo_masters",)
        assert backfill.tennis_keys_for("Santiago de Querétaro, Mexico", when) == ()
        assert backfill.tennis_keys_for("Paris, France", datetime(2025, 3, 1, tzinfo=UTC)) == ()

    def test_combined_events_plan_one_call_per_tour_key(self):
        g = [_tg("a", datetime(2025, 1, 20, 0, tzinfo=UTC), "Novak Djokovic", "Carlos Alcaraz", "Melbourne, Australia")]
        tasks = backfill.plan_tennis_snapshots(g)
        assert [k for k, _ in tasks] == ["tennis_atp_aus_open_singles", "tennis_wta_aus_open_singles"]
        assert tasks[0][1].requested == datetime(2025, 1, 19, 23, 50, tzinfo=UTC)

    def test_unlisted_tournaments_cost_nothing(self):
        g = [_tg("a", datetime(2025, 3, 1, tzinfo=UTC), "A B", "C D", "Santiago de Querétaro, Mexico")]
        assert backfill.plan_tennis_snapshots(g) == []


class TestTennisKeyTable:
    def test_every_key_is_a_real_provider_key(self):
        keys = {k for _, _, ks in backfill.TENNIS_EVENTS for k in ks}
        assert keys <= set(fetch_live_odds.TENNIS_PROVIDER_KEYS)

    def test_500_level_events_resolve(self):
        oct_ = datetime(2025, 10, 22, tzinfo=UTC)
        assert backfill.tennis_keys_for("Basel, Switzerland", oct_) == ("tennis_atp_swiss_indoors",)
        assert backfill.tennis_keys_for("Vienna, Austria", oct_) == ("tennis_atp_vienna_open",)
        assert backfill.tennis_keys_for("Tokyo, Japan", oct_) == (
            "tennis_atp_japan_open",
            "tennis_wta_pan_pacific_open",
        )

    def test_queens_and_wimbledon_share_london_without_overlapping(self):
        assert backfill.tennis_keys_for("London, Great Britain", datetime(2025, 6, 18, tzinfo=UTC)) == (
            "tennis_atp_queens_club_champ",
            "tennis_wta_queens_club_champ",
        )
        assert backfill.tennis_keys_for("London, Great Britain", datetime(2025, 7, 2, tzinfo=UTC)) == (
            "tennis_atp_wimbledon",
            "tennis_wta_wimbledon",
        )

    def test_a_venue_whose_event_moved_resolves_in_both_windows(self):
        for when in (datetime(2025, 5, 20, tzinfo=UTC), datetime(2023, 7, 20, tzinfo=UTC)):
            assert backfill.tennis_keys_for("Hamburg, Germany", when) == ("tennis_atp_hamburg_open",)


class TestProviderAnchoredPlanning:
    """ESPN's tennis times run hours after the provider's commence_time, so
    snapshots must be clustered on the provider's time (Paris 2025-10-29:
    ESPN 13:30 / 17:40 vs provider 11:10 / 14:30)."""

    def _ev(self, eid, home, away, when):
        return {
            "id": eid,
            "home_team": home,
            "away_team": away,
            "commence_time": when.isoformat().replace("+00:00", "Z"),
        }

    def test_listing_calls_cover_each_game_day_per_key(self):
        g = [_tg("a", datetime(2025, 10, 29, 13, 30, tzinfo=UTC), "A B", "C D", "Paris, France")]
        tasks = backfill.tennis_listing_tasks(g)
        assert tasks == [
            ("tennis_atp_paris_masters", "2025-10-28T16:00:00Z"),
            ("tennis_atp_paris_masters", "2025-10-29T00:00:00Z"),
            ("tennis_atp_paris_masters", "2025-10-29T08:00:00Z"),
        ]

    def test_snapshots_sit_before_the_providers_start_not_espns(self):
        day = datetime(2025, 10, 29, tzinfo=UTC)
        games = [
            _tg("m1", day.replace(hour=13, minute=30), "Camilo Ugo Carabelli", "Alexander Zverev", "Paris, France"),
            _tg("m2", day.replace(hour=17, minute=40), "Lorenzo Musetti", "Lorenzo Sonego", "Paris, France"),
        ]
        events = {
            "tennis_atp_paris_masters": [
                self._ev("e1", "Alexander Zverev", "Camilo Ugo Carabelli", day.replace(hour=11, minute=10)),
                self._ev("e2", "Lorenzo Musetti", "Lorenzo Sonego", day.replace(hour=14, minute=30)),
            ]
        }
        tasks = backfill.plan_tennis_anchored(games, events, window_hours=3.0, lead_minutes=10)
        # 11:10 and 14:30 are 3 h 20 apart: two clusters, each 10 min before the PROVIDER's start
        assert [(k, snap.requested) for k, snap in tasks] == [
            ("tennis_atp_paris_masters", day.replace(hour=11, minute=0)),
            ("tennis_atp_paris_masters", day.replace(hour=14, minute=20)),
        ]
        for (_k, snap), eid in zip(tasks, ("e1", "e2")):
            # the snapshot's game carries the provider's start, and its event is strictly pre-game there
            ev = [e for e in events["tennis_atp_paris_masters"] if e["id"] == eid]
            pairs, counts = backfill.match_tennis_events(ev, snap.requested, snap.games)
            assert len(pairs) == 1 and counts["in_play_or_started"] == 0
        # under the old ESPN anchor, a snapshot at 13:20 finds Zverev's match already started
        late = backfill.match_tennis_events(
            events["tennis_atp_paris_masters"][:1], day.replace(hour=13, minute=20), games
        )
        assert late[1]["in_play_or_started"] == 1

    def test_games_the_provider_never_listed_plan_nothing(self):
        g = [_tg("m1", datetime(2025, 10, 29, 13, tzinfo=UTC), "A B", "C D", "Paris, France")]
        assert backfill.plan_tennis_anchored(g, {"tennis_atp_paris_masters": []}) == []

    def test_anchor_is_a_cli_choice(self):
        args = backfill.parse_args(
            ["--sport", "tennis", "--start", "2025-01-01", "--end", "2025-12-31", "--anchor", "provider"]
        )
        assert args.anchor == "provider"


class TestTennisMatcher:
    def _ev(self, home, away, when):
        return {"home_team": home, "away_team": away, "commence_time": when.isoformat().replace("+00:00", "Z")}

    def test_either_player_order_matches_and_reports_the_swap(self):
        start = datetime(2025, 1, 20, 2, tzinfo=UTC)
        games = [_tg("m1", start, "Félix Auger-Aliassime", "Jannik Sinner", "Melbourne")]
        events = [self._ev("Jannik Sinner", "Felix Auger-Aliassime", start + timedelta(hours=3))]
        pairs, counts = backfill.match_tennis_events(events, start - timedelta(minutes=10), games)
        assert [(g.match_id, swapped) for _, g, swapped in pairs] == [("m1", True)]
        assert counts["exact"] == 1

    def test_in_play_events_are_rejected(self):
        start = datetime(2025, 1, 20, 2, tzinfo=UTC)
        games = [_tg("m1", start, "A Player", "B Player", "Melbourne")]
        pairs, counts = backfill.match_tennis_events([self._ev("A Player", "B Player", start)], start, games)
        assert pairs == [] and counts["in_play_or_started"] == 1

    def test_fuzzy_names_resolve_within_the_snapshot_only(self):
        start = datetime(2025, 1, 20, 2, tzinfo=UTC)
        games = [_tg("m1", start, "Alex de Minaur", "Ben Shelton", "Melbourne")]
        events = [self._ev("Alex De Minaur", "Benjamin Shelton", start + timedelta(hours=1))]
        pairs, counts = backfill.match_tennis_events(events, start - timedelta(minutes=10), games)
        assert len(pairs) == 1 and counts["fuzzy"] == 1


class TestTennisCli:
    def test_tennis_is_accepted_as_a_sport(self):
        args = backfill.parse_args(["--sport", "tennis", "--start", "2025-01-01", "--end", "2025-12-31", "--dry-run"])
        assert args.sport == "tennis"


class _Resp:
    def __init__(self, status, body=None, remaining="4000000"):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = {"x-requests-remaining": remaining, "x-requests-last": "10"}

    def json(self):
        return self._body


class TestFetchRetries:
    """A 429 burst lost ~270 tennis snapshots on 2026-10-30; one transient
    failure left two NHL games on contaminated v1 lines."""

    def _fetch(self, monkeypatch, responses):
        calls = []

        def fake_get(url, params, timeout):
            calls.append(params["date"])
            nxt = responses.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

        monkeypatch.setattr(backfill.requests, "get", fake_get)
        monkeypatch.setattr(backfill.time, "sleep", lambda s: None)
        out = backfill.fetch_historical_snapshot("icehockey_nhl", "2024-03-22T22:50:00Z", "h2h", "k", "us")
        return out, calls

    def test_rate_limit_is_retried(self, monkeypatch):
        (resp, remaining), calls = self._fetch(monkeypatch, [_Resp(429), _Resp(429), _Resp(200, {"data": []})])
        assert resp == {"data": []} and remaining == 4000000 and len(calls) == 3

    def test_connection_error_is_retried(self, monkeypatch):
        (resp, _), calls = self._fetch(
            monkeypatch, [backfill.requests.ConnectionError("reset"), _Resp(200, {"data": []})]
        )
        assert resp == {"data": []} and len(calls) == 2

    def test_unknown_sport_is_final(self, monkeypatch):
        (resp, remaining), calls = self._fetch(monkeypatch, [_Resp(404, {"error_code": "UNKNOWN_SPORT"})])
        assert resp is None and remaining == 4000000 and len(calls) == 1

    def test_gives_up_after_the_attempt_budget(self, monkeypatch):
        (resp, _), calls = self._fetch(monkeypatch, [_Resp(503)] * backfill.FETCH_ATTEMPTS)
        assert resp is None and len(calls) == backfill.FETCH_ATTEMPTS

    def test_auth_error_still_raises(self, monkeypatch):
        with pytest.raises(RuntimeError):
            self._fetch(monkeypatch, [_Resp(401)])
