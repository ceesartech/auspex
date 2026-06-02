"""Unit tests for the Phase 10 (NFL) scaffolding.

E1 lands the lookup tables + dispatch entries that the rest of the
NFL pipeline (features, training, predictions, recs) hangs off. If
any of these tables drift relative to the other sports' shape, the
later phases will fail in less obvious ways (e.g., precompute_nfl
silently returning 0 matches because the sport string doesn't match
the leagues row).
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


fu = _load("fetch_upcoming", "fetch_upcoming.py")
flo = _load("fetch_live_odds", "fetch_live_odds.py")


# ── fetch_upcoming wiring ─────────────────────────────────────────────


class TestFetchUpcomingNflConfig:
    def test_nfl_registered_in_sport_configs(self):
        # If the SportConfig is missing, --sport nfl crashes at CLI
        # validation before doing anything.
        assert "nfl" in fu.SPORT_CONFIGS

    def test_nfl_uses_football_espn_path(self):
        # ESPN path /sports/football/{league_slug}/scoreboard. Same
        # gotcha as NBA/NHL — espn_path is just "football", NOT
        # "football/nfl" (would 404 by doubling the slug).
        cfg = fu.SPORT_CONFIGS["nfl"]
        assert cfg.espn_path == "football"

    def test_nfl_leagues_registered(self):
        # Single league entry, mirroring NBA/NHL shape.
        cfg = fu.SPORT_CONFIGS["nfl"]
        assert "nfl" in cfg.leagues
        code, name, country = cfg.leagues["nfl"]
        assert code == "NFL"
        assert name == "NFL"
        assert country == "USA"

    def test_nfl_sport_string_lowercase(self):
        # Must match leagues.sport column convention (lowercase). If
        # this drifts, INSERTs into leagues mismatch existing rows
        # and create duplicates per-cased.
        assert fu.SPORT_CONFIGS["nfl"].sport == "nfl"

    def test_nfl_season_func_handles_season_boundary(self):
        # NFL season runs Sep→Feb. Aug-Dec maps to the new season
        # (preseason starts early Aug); Jan-Jul maps to the season
        # that finished that Feb.
        season_func = fu.SPORT_CONFIGS["nfl"].season_func

        # August → new season starting that year
        assert season_func(datetime(2024, 8, 5)) == "2024-2025"
        # September preseason → new season
        assert season_func(datetime(2024, 9, 8)) == "2024-2025"
        # December regular season → still the season that started Aug
        assert season_func(datetime(2024, 12, 25)) == "2024-2025"
        # January playoffs → season that started prior Aug
        assert season_func(datetime(2025, 1, 12)) == "2024-2025"
        # Early February — Super Bowl. Still 2024-2025 season.
        assert season_func(datetime(2025, 2, 9)) == "2024-2025"
        # Late July — between seasons. Maps to the LAST season (the
        # one that ended that Feb), not the upcoming one (which the
        # cutoff month 8 hasn't started yet).
        assert season_func(datetime(2025, 7, 20)) == "2024-2025"


# ── fetch_live_odds (the-odds-api) dispatch ─────────────────────────


class TestFetchLiveOddsNflDispatch:
    def test_americanfootball_prefix_routes_to_nfl(self):
        # the-odds-api uses "americanfootball_nfl" / "americanfootball_ncaaf"
        # etc. Our prefix table maps the whole family to 'nfl'. If we
        # later add NCAAF, the prefix would still route to 'nfl'; that's
        # fine for v1 (we don't process NCAAF) but flag it explicitly:
        assert flo.sport_for_key("americanfootball_nfl") == "nfl"

    def test_nfl_in_default_sport_set(self):
        # fetch_live_odds.py has a default sports list it pulls if
        # the CLI doesn't override. NFL needs to be in it or the
        # 15-min cron job won't pull NFL odds even after we've
        # added the dispatch.
        assert "americanfootball_nfl" in flo.DEFAULT_SPORTS

    def test_nfl_markets_match_nba_nhl_pattern(self):
        # h2h + spreads + totals — same line-as-feature design as
        # NBA. The training query for NFL will use closing line as
        # a feature, same as NBA's spread/total handling.
        assert flo.SPORT_MARKETS["nfl"] == "h2h,spreads,totals"

    def test_nfl_market_keys_registered(self):
        # KNOWN_MARKET_KEYS gates which markets process_event will
        # consume. Missing the entry = NFL odds get logged as
        # "unknown market" warnings and dropped.
        assert flo.KNOWN_MARKET_KEYS["nfl"] == {"h2h", "spreads", "totals"}


# ── map_outcome NFL branch ───────────────────────────────────────────


class TestMapOutcomeNfl:
    """The map_outcome branch translates the-odds-api's payload into
    (market_type, selection, keep_line) tuples that match our DB
    schema. Same shape as NBA — h2h/spreads/totals with variable
    lines — so the tests mirror the NBA cases."""

    def test_h2h_home_routes_to_moneyline_home(self):
        result = flo.map_outcome(
            sport="nfl",
            market_key="h2h",
            outcome_name="Kansas City Chiefs",
            home_team_name="Kansas City Chiefs",
            away_team_name="Buffalo Bills",
            point=None,
        )
        assert result == ("moneyline", "home", False)

    def test_h2h_away_routes_to_moneyline_away(self):
        result = flo.map_outcome(
            sport="nfl",
            market_key="h2h",
            outcome_name="Buffalo Bills",
            home_team_name="Kansas City Chiefs",
            away_team_name="Buffalo Bills",
            point=None,
        )
        assert result == ("moneyline", "away", False)

    def test_spreads_keeps_line(self):
        # NFL spread line varies per game (-14.5 to +14.5 typical).
        # keep_line=True is the line-as-feature signal — the line
        # column on the odds row gets populated.
        result = flo.map_outcome(
            sport="nfl",
            market_key="spreads",
            outcome_name="Kansas City Chiefs",
            home_team_name="Kansas City Chiefs",
            away_team_name="Buffalo Bills",
            point=-7.0,
        )
        assert result == ("spread", "home", True)

    def test_spreads_without_point_dropped(self):
        # Defensive — the-odds-api occasionally omits the point on
        # malformed payloads. Drop the row rather than insert NULL
        # line.
        result = flo.map_outcome(
            sport="nfl",
            market_key="spreads",
            outcome_name="Kansas City Chiefs",
            home_team_name="Kansas City Chiefs",
            away_team_name="Buffalo Bills",
            point=None,
        )
        assert result == (None, None, False)

    def test_totals_over_routes(self):
        # NFL totals typically 38-55. keep_line=True so the model
        # sees the actual closing line at predict time.
        result = flo.map_outcome(
            sport="nfl",
            market_key="totals",
            outcome_name="Over",
            home_team_name="Kansas City Chiefs",
            away_team_name="Buffalo Bills",
            point=47.5,
        )
        assert result == ("total", "over", True)

    def test_totals_under_routes(self):
        result = flo.map_outcome(
            sport="nfl",
            market_key="totals",
            outcome_name="Under",
            home_team_name="Kansas City Chiefs",
            away_team_name="Buffalo Bills",
            point=47.5,
        )
        assert result == ("total", "under", True)

    def test_unknown_market_returns_nones(self):
        # Player props / first-half / quarter markets aren't wired
        # yet — they should drop, not crash.
        result = flo.map_outcome(
            sport="nfl",
            market_key="player_props",
            outcome_name="Patrick Mahomes Over 250.5",
            home_team_name="Kansas City Chiefs",
            away_team_name="Buffalo Bills",
            point=250.5,
        )
        assert result == (None, None, False)
