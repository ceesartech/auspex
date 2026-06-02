"""Unit tests for the Phase 11 (Tennis) scaffolding.

Mirror of test_nfl_scaffolding. T1 lands the lookup tables + dispatch
entries the rest of the tennis pipeline hangs off. Tennis differs from
the team sports in two structural ways tested here:

  * 1v1 / individual sport: ESPN competitors use 'athlete' not 'team'
    and lack the homeAway field. process_event uses positional ordering
    (index 0 → home, index 1 → away) and reads athlete.displayName via
    SportConfig.is_individual=True.
  * No spreads market in the default US odds basket: tennis recovers
    h2h (match-winner) + totals (total games over/under). Set-handicap
    is a UK/EU specialty layered in later.
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


class TestFetchUpcomingTennisConfig:
    def test_tennis_registered_in_sport_configs(self):
        assert "tennis" in fu.SPORT_CONFIGS

    def test_tennis_uses_tennis_espn_path(self):
        # ESPN path /sports/tennis/{atp|wta}/scoreboard. Same gotcha
        # as NBA/NHL — espn_path is JUST "tennis" so the slug "atp"
        # doesn't double up to /tennis/atp/atp.
        cfg = fu.SPORT_CONFIGS["tennis"]
        assert cfg.espn_path == "tennis"

    def test_tennis_leagues_registered(self):
        # Two tours, both 1v1: ATP (men) + WTA (women).
        cfg = fu.SPORT_CONFIGS["tennis"]
        assert "atp" in cfg.leagues
        assert "wta" in cfg.leagues

    def test_atp_metadata(self):
        cfg = fu.SPORT_CONFIGS["tennis"]
        code, name, country = cfg.leagues["atp"]
        assert code == "ATP"
        assert name == "ATP Tour"
        assert country == "World"

    def test_wta_metadata(self):
        cfg = fu.SPORT_CONFIGS["tennis"]
        code, name, country = cfg.leagues["wta"]
        assert code == "WTA"
        assert name == "WTA Tour"
        assert country == "World"

    def test_tennis_sport_string_lowercase(self):
        assert fu.SPORT_CONFIGS["tennis"].sport == "tennis"

    def test_tennis_is_individual_flag(self):
        # is_individual=True triggers the positional+athlete handling
        # in process_event. If this flag drifts, tennis matches silently
        # drop because there's no 'homeAway' field on competitors.
        assert fu.SPORT_CONFIGS["tennis"].is_individual is True

    def test_team_sports_individual_flag_stays_false(self):
        # Regression guard — adding the is_individual flag MUST default
        # False so soccer/NHL/NBA/NFL keep their team-sport semantics.
        for sport in ("soccer", "nhl", "nba", "nfl"):
            assert fu.SPORT_CONFIGS[sport].is_individual is False

    def test_tennis_season_is_calendar_year(self):
        # ATP/WTA tour year is a single calendar year (Jan-Nov regular
        # season + Nov Finals). String shape is just the 4-digit year,
        # NOT "YYYY-YYYY" like soccer/NHL/NBA/NFL.
        season_func = fu.SPORT_CONFIGS["tennis"].season_func
        assert season_func(datetime(2024, 1, 20)) == "2024"  # Aus Open
        assert season_func(datetime(2024, 7, 5)) == "2024"  # Wimbledon
        assert season_func(datetime(2024, 12, 28)) == "2024"  # year-end
        assert season_func(datetime(2025, 1, 5)) == "2025"


# ── fetch_live_odds (the-odds-api) dispatch ─────────────────────────


class TestFetchLiveOddsTennisDispatch:
    def test_tennis_prefix_routes_to_tennis(self):
        # the-odds-api uses event-specific keys like
        # tennis_atp_australian_open, tennis_wta_wimbledon, etc. All
        # share the "tennis_" prefix → 'tennis'.
        assert flo.sport_for_key("tennis_atp_australian_open") == "tennis"
        assert flo.sport_for_key("tennis_atp_french_open") == "tennis"
        assert flo.sport_for_key("tennis_atp_wimbledon") == "tennis"
        assert flo.sport_for_key("tennis_atp_us_open") == "tennis"
        assert flo.sport_for_key("tennis_wta_australian_open") == "tennis"
        assert flo.sport_for_key("tennis_wta_french_open") == "tennis"
        assert flo.sport_for_key("tennis_wta_wimbledon") == "tennis"
        assert flo.sport_for_key("tennis_wta_us_open") == "tennis"

    def test_tennis_grand_slams_in_default_set(self):
        # The 15-min pipeline pulls default sports. Without these, the
        # cron won't fetch tennis odds even after dispatch is wired.
        # Off-tournament keys 422 silently so the cost is one
        # round-trip per dormant key.
        for key in (
            "tennis_atp_australian_open",
            "tennis_atp_french_open",
            "tennis_atp_wimbledon",
            "tennis_atp_us_open",
            "tennis_wta_australian_open",
            "tennis_wta_french_open",
            "tennis_wta_wimbledon",
            "tennis_wta_us_open",
        ):
            assert key in flo.DEFAULT_SPORTS

    def test_tennis_markets_h2h_and_totals(self):
        # No spreads — tennis spread (set-handicap) is a UK/EU specialty
        # not in the default US odds basket. h2h covers moneyline;
        # totals covers total-games over/under.
        assert flo.SPORT_MARKETS["tennis"] == "h2h,totals"

    def test_tennis_market_keys_registered(self):
        # KNOWN_MARKET_KEYS gates which markets process_event will
        # consume. Missing the entry = tennis odds get logged as
        # "unknown market" warnings and dropped.
        assert flo.KNOWN_MARKET_KEYS["tennis"] == {"h2h", "totals"}


# ── map_outcome tennis branch ───────────────────────────────────────


class TestMapOutcomeTennis:
    """The tennis branch of map_outcome — 1v1 with no draw, no
    spreads market. Player names appear in outcome_name directly
    (vs team sports where outcome_name is the team name)."""

    def test_h2h_player1_routes_to_moneyline_home(self):
        result = flo.map_outcome(
            sport="tennis",
            market_key="h2h",
            outcome_name="Novak Djokovic",
            home_team_name="Novak Djokovic",
            away_team_name="Carlos Alcaraz",
            point=None,
        )
        assert result == ("moneyline", "home", False)

    def test_h2h_player2_routes_to_moneyline_away(self):
        result = flo.map_outcome(
            sport="tennis",
            market_key="h2h",
            outcome_name="Carlos Alcaraz",
            home_team_name="Novak Djokovic",
            away_team_name="Carlos Alcaraz",
            point=None,
        )
        assert result == ("moneyline", "away", False)

    def test_totals_over_routes(self):
        # Tennis totals: typically ~22.5 games for best-of-3,
        # ~37.5 for best-of-5. keep_line=True so the model sees the
        # actual line at predict time (line-as-feature design).
        result = flo.map_outcome(
            sport="tennis",
            market_key="totals",
            outcome_name="Over",
            home_team_name="Novak Djokovic",
            away_team_name="Carlos Alcaraz",
            point=22.5,
        )
        assert result == ("total", "over", True)

    def test_totals_under_routes(self):
        result = flo.map_outcome(
            sport="tennis",
            market_key="totals",
            outcome_name="Under",
            home_team_name="Novak Djokovic",
            away_team_name="Carlos Alcaraz",
            point=22.5,
        )
        assert result == ("total", "under", True)

    def test_totals_without_point_dropped(self):
        # Defensive — same as NFL/NBA: drop the row rather than
        # insert NULL line.
        result = flo.map_outcome(
            sport="tennis",
            market_key="totals",
            outcome_name="Over",
            home_team_name="Novak Djokovic",
            away_team_name="Carlos Alcaraz",
            point=None,
        )
        assert result == (None, None, False)

    def test_spreads_returns_none(self):
        # No spreads market on tennis — set-handicap not in the default
        # basket. If a future commit adds it, this test needs updating.
        result = flo.map_outcome(
            sport="tennis",
            market_key="spreads",
            outcome_name="Novak Djokovic",
            home_team_name="Novak Djokovic",
            away_team_name="Carlos Alcaraz",
            point=-1.5,
        )
        assert result == (None, None, False)

    def test_unknown_market_returns_nones(self):
        # Set-betting / correct-score markets aren't wired yet —
        # should drop, not crash.
        result = flo.map_outcome(
            sport="tennis",
            market_key="set_betting",
            outcome_name="3-0 Djokovic",
            home_team_name="Novak Djokovic",
            away_team_name="Carlos Alcaraz",
            point=None,
        )
        assert result == (None, None, False)


# ── process_event individual-sport handling ─────────────────────────


class TestProcessEventIndividual:
    """The is_individual=True flag changes how process_event reads
    competitors. Tennis events from ESPN don't have homeAway and use
    'athlete' not 'team'. Test the helper that resolves the name."""

    def test_competitor_name_uses_athlete_for_individual(self):
        comp = {"athlete": {"displayName": "Novak Djokovic"}}
        assert fu._competitor_name(comp, is_individual=True) == "Novak Djokovic"

    def test_competitor_name_uses_team_for_team_sport(self):
        comp = {"team": {"displayName": "Kansas City Chiefs"}}
        assert fu._competitor_name(comp, is_individual=False) == "Kansas City Chiefs"

    def test_competitor_name_falls_back_when_displayname_missing(self):
        # ESPN occasionally omits displayName for less-tracked athletes
        # — fall back to shortName then name to recover something
        # rather than dropping the match.
        comp = {"athlete": {"shortName": "N. Djokovic"}}
        assert fu._competitor_name(comp, is_individual=True) == "N. Djokovic"
        comp = {"athlete": {"name": "Djokovic"}}
        assert fu._competitor_name(comp, is_individual=True) == "Djokovic"

    def test_competitor_name_returns_none_when_all_missing(self):
        # No name fields at all → None. process_event drops the event.
        assert fu._competitor_name({}, is_individual=True) is None
        assert fu._competitor_name({"athlete": {}}, is_individual=True) is None
