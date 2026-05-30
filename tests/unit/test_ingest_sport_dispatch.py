"""Unit tests for sport dispatch in the ingest scripts.

Covers the pure functions added when we generalised fetch_upcoming.py
and fetch_live_odds.py for multi-sport ingestion (Phase 1 of the NHL
vertical slice). DB/network calls are intentionally not exercised here
— those belong in integration tests against the data-ingestion fixture
stack.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(module_name: str, filename: str):
    """Load a top-level script as an importable module without needing
    a package structure under scripts/. Registers the module in
    sys.modules before exec so @dataclass can resolve cls.__module__."""
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


fetch_upcoming = _load("fetch_upcoming", "fetch_upcoming.py")
fetch_live_odds = _load("fetch_live_odds", "fetch_live_odds.py")
load_nhl_historical = _load("load_nhl_historical", "load_nhl_historical.py")


# ── fetch_upcoming: sport config / season conventions ────────────────


class TestSportConfigs:
    def test_both_sports_registered(self):
        assert set(fetch_upcoming.SPORT_CONFIGS.keys()) == {"soccer", "nhl"}

    def test_soccer_config_keeps_espn_path(self):
        cfg = fetch_upcoming.SPORT_CONFIGS["soccer"]
        assert cfg.sport == "soccer"
        assert cfg.espn_path == "soccer"
        assert "eng.1" in cfg.leagues
        assert cfg.club_suffixes  # soccer relies on token stripping

    def test_nhl_config_uses_hockey_path(self):
        cfg = fetch_upcoming.SPORT_CONFIGS["nhl"]
        assert cfg.sport == "nhl"
        # ESPN URL is /sports/hockey/nhl/scoreboard — the "nhl" segment
        # comes from the league slug, NOT espn_path. Double-checking
        # this here because a bad value silently 404s every fetch.
        assert cfg.espn_path == "hockey"
        assert "nhl" in cfg.leagues
        # NHL team names are unambiguous — no token stripping.
        assert not cfg.club_suffixes
        assert not cfg.club_prefixes

    @pytest.mark.parametrize(
        "sport, slug, expected_suffix",
        [
            ("soccer", "eng.1", "/sports/soccer/eng.1/scoreboard"),
            ("nhl", "nhl", "/sports/hockey/nhl/scoreboard"),
        ],
    )
    def test_constructed_scoreboard_url(self, sport, slug, expected_suffix):
        """Lock in the ESPN URL shape per sport. Regression guard for
        the "hockey/nhl" + "nhl" double-segment bug."""
        cfg = fetch_upcoming.SPORT_CONFIGS[sport]
        url = f"{fetch_upcoming.ESPN_BASE}/{cfg.espn_path}/{slug}/scoreboard"
        assert url.endswith(expected_suffix)


class TestSeasonFunctions:
    @pytest.mark.parametrize(
        "month, year, expected",
        [
            (1, 2025, "2024-2025"),  # mid-season
            (5, 2025, "2024-2025"),  # late season
            (8, 2025, "2025-2026"),  # August: new season started
            (12, 2025, "2025-2026"),
        ],
    )
    def test_soccer_season_boundary(self, month, year, expected):
        dt = datetime(year, month, 15, tzinfo=timezone.utc)
        assert fetch_upcoming._season_soccer(dt) == expected

    @pytest.mark.parametrize(
        "month, year, expected",
        [
            (1, 2025, "2024-2025"),
            (6, 2025, "2024-2025"),  # late playoffs / Cup final
            (8, 2025, "2024-2025"),  # offseason — last completed season
            (9, 2025, "2025-2026"),  # preseason starts
            (10, 2025, "2025-2026"),  # regular season opener
            (12, 2025, "2025-2026"),
        ],
    )
    def test_nhl_season_boundary(self, month, year, expected):
        dt = datetime(year, month, 15, tzinfo=timezone.utc)
        assert fetch_upcoming._season_nhl(dt) == expected


class TestStripClubTokens:
    def test_soccer_strips_suffix(self):
        cfg = fetch_upcoming.SPORT_CONFIGS["soccer"]
        assert fetch_upcoming._strip_club_tokens("rosenborg bk", cfg) == "rosenborg"
        assert fetch_upcoming._strip_club_tokens("arsenal fc", cfg) == "arsenal"

    def test_nhl_is_noop(self):
        cfg = fetch_upcoming.SPORT_CONFIGS["nhl"]
        # NHL teams have city + nickname; no suffix tokens to strip.
        assert fetch_upcoming._strip_club_tokens("toronto maple leafs", cfg) == "toronto maple leafs"


# ── fetch_live_odds: key → sport dispatch + market mapping ───────────


class TestFindMatchIdStatusScope:
    """Regression guard: find_match_id was filtering to status='scheduled'
    only, which silently dropped every historical-backfill event because
    those games are status='finished'. The fix broadens the scope to
    IN ('scheduled', 'finished') so both pipelines (live odds for
    upcoming games, historical odds for finished games) hit. Test
    inspects the rendered SQL so a future refactor that drops 'finished'
    fails loudly here instead of silently inserting zero rows."""

    def test_includes_both_scheduled_and_finished(self):
        import inspect

        src = inspect.getsource(fetch_live_odds.find_match_id)
        # Must include both statuses — not just 'scheduled' (the
        # pre-bug state) and not just 'finished' (would break live
        # odds).
        assert "'scheduled'" in src and "'finished'" in src
        assert "IN ('scheduled', 'finished')" in src or "IN ( 'scheduled', 'finished' )" in src


class TestSportForKey:
    @pytest.mark.parametrize(
        "key, expected",
        [
            ("soccer_epl", "soccer"),
            ("soccer_usa_mls", "soccer"),
            ("icehockey_nhl", "nhl"),
            ("basketball_nba", None),  # not registered yet — should fall through
            ("", None),
        ],
    )
    def test_sport_for_key(self, key, expected):
        assert fetch_live_odds.sport_for_key(key) == expected


class TestSoccerMarketMapping:
    def test_h2h_home(self):
        mt, sel, keep = fetch_live_odds.map_outcome("soccer", "h2h", "Liverpool", "Liverpool", "Arsenal", None)
        assert (mt, sel, keep) == ("1x2", "home", False)

    def test_h2h_draw(self):
        mt, sel, keep = fetch_live_odds.map_outcome("soccer", "h2h", "Draw", "Liverpool", "Arsenal", None)
        assert (mt, sel, keep) == ("1x2", "draw", False)

    def test_totals_only_25_line(self):
        mt, sel, keep = fetch_live_odds.map_outcome("soccer", "totals", "Over", "A", "B", 2.5)
        assert (mt, sel, keep) == ("over_under", "over", True)

    def test_totals_skips_other_lines(self):
        # Soccer features are tied to the 2.5 line — drop everything else.
        mt, _, _ = fetch_live_odds.map_outcome("soccer", "totals", "Over", "A", "B", 3.5)
        assert mt is None

    def test_spreads_not_supported(self):
        mt, _, _ = fetch_live_odds.map_outcome("soccer", "spreads", "Liverpool", "Liverpool", "Arsenal", -1.0)
        assert mt is None


class TestNhlMarketMapping:
    def test_h2h_is_moneyline_no_draw(self):
        mt, sel, keep = fetch_live_odds.map_outcome(
            "nhl", "h2h", "Toronto Maple Leafs", "Toronto Maple Leafs", "Boston Bruins", None
        )
        assert (mt, sel, keep) == ("moneyline", "home", False)

    def test_h2h_away(self):
        mt, sel, keep = fetch_live_odds.map_outcome(
            "nhl", "h2h", "Boston Bruins", "Toronto Maple Leafs", "Boston Bruins", None
        )
        assert (mt, sel, keep) == ("moneyline", "away", False)

    def test_h2h_draw_outcome_rejected(self):
        # NHL moneyline never settles as a draw — if the-odds-api ever
        # emits one, we should drop it rather than store an illegal row.
        mt, _, _ = fetch_live_odds.map_outcome("nhl", "h2h", "Draw", "Toronto Maple Leafs", "Boston Bruins", None)
        assert mt is None

    def test_spreads_puck_line(self):
        mt, sel, keep = fetch_live_odds.map_outcome(
            "nhl", "spreads", "Toronto Maple Leafs", "Toronto Maple Leafs", "Boston Bruins", -1.5
        )
        assert (mt, sel, keep) == ("spread", "home", True)

    def test_spreads_requires_point(self):
        mt, _, _ = fetch_live_odds.map_outcome(
            "nhl", "spreads", "Toronto Maple Leafs", "Toronto Maple Leafs", "Boston Bruins", None
        )
        assert mt is None

    @pytest.mark.parametrize("point", [5.5, 6.0, 6.5])
    def test_totals_keeps_every_line(self, point):
        # No model exists yet; we want all available NHL totals lines.
        mt, sel, keep = fetch_live_odds.map_outcome("nhl", "totals", "Under", "A", "B", point)
        assert (mt, sel, keep) == ("total", "under", True)


# ── load_nhl_historical: pure parsing/derivations ────────────────────


class TestParseWinsTaken:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("34/59", (34, 59)),
            ("0/2", (0, 2)),
            (None, (0, 0)),
            ("", (0, 0)),
            ("notanumber", (0, 0)),
        ],
    )
    def test_parse_wins_taken(self, raw, expected):
        assert load_nhl_historical.parse_wins_taken(raw) == expected


class TestRegulationOutcome:
    def test_home_wins_in_regulation(self):
        home = {"1": 1, "2": 2, "3": 0}
        away = {"1": 0, "2": 1, "3": 1}
        assert load_nhl_historical.regulation_outcome(home, away) == (3, 2, "home")

    def test_tie_after_regulation_then_ot(self):
        # 3-3 after regulation, home wins in OT — moneyline says home,
        # but the regulation 3-way target says "tie".
        home = {"1": 1, "2": 1, "3": 1, "OT": 1}
        away = {"1": 1, "2": 1, "3": 1}
        reg_home, reg_away, winner = load_nhl_historical.regulation_outcome(home, away)
        assert (reg_home, reg_away, winner) == (3, 3, "tie")

    def test_away_blowout(self):
        home = {"1": 0, "2": 0, "3": 1}
        away = {"1": 2, "2": 3, "3": 1}
        assert load_nhl_historical.regulation_outcome(home, away) == (1, 6, "away")


class TestSeasonId:
    def test_valid(self):
        assert load_nhl_historical.season_id("2024-2025") == "20242025"

    @pytest.mark.parametrize("bad", ["2024", "24-25", "2024_2025", "abcd-2025"])
    def test_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            load_nhl_historical.season_id(bad)


class TestParseTeamGameStats:
    """Anchor test against a realistic /right-rail payload — ensures the
    parser still grabs the right fields if the schema drifts."""

    def _payload(self):
        return {
            "teamGameStats": [
                {"category": "sog", "awayValue": 23, "homeValue": 31},
                {"category": "faceoffWinningPctg", "awayValue": 0.423729, "homeValue": 0.576271},
                {"category": "faceoffWins", "awayValue": "25/59", "homeValue": "34/59"},
                {"category": "powerPlay", "awayValue": "0/2", "homeValue": "1/4"},
                {"category": "powerPlayPctg", "awayValue": 0.0, "homeValue": 0.25},
                {"category": "pim", "awayValue": 8, "homeValue": 4},
                {"category": "hits", "awayValue": 34, "homeValue": 29},
                {"category": "blockedShots", "awayValue": 18, "homeValue": 15},
                {"category": "giveaways", "awayValue": 22, "homeValue": 12},
                {"category": "takeaways", "awayValue": 2, "homeValue": 5},
            ],
            "linescore": {
                "byPeriod": [
                    {"periodDescriptor": {"number": 1, "periodType": "REG"}, "away": 2, "home": 0},
                    {"periodDescriptor": {"number": 2, "periodType": "REG"}, "away": 1, "home": 0},
                    {"periodDescriptor": {"number": 3, "periodType": "REG"}, "away": 1, "home": 1},
                ],
                "totals": {"away": 4, "home": 1},
            },
        }

    def test_home_stats(self):
        home = load_nhl_historical.parse_team_game_stats(self._payload(), "homeValue")
        assert home.shots_on_goal == 31
        assert home.power_play_goals == 1
        assert home.power_play_opps == 4
        assert home.power_play_pct == pytest.approx(0.25)
        assert home.faceoffs_won == 34
        assert home.faceoffs_taken == 59
        assert home.faceoff_win_pct == pytest.approx(0.576271, abs=1e-5)
        assert home.period_scores == {"1": 0, "2": 0, "3": 1}

    def test_away_stats_period_scores(self):
        away = load_nhl_historical.parse_team_game_stats(self._payload(), "awayValue")
        assert away.shots_on_goal == 23
        assert away.power_play_goals == 0
        assert away.period_scores == {"1": 2, "2": 1, "3": 1}
