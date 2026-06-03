"""Unit tests for the Phase 12 (MMA) scaffolding.

Mirror of test_tennis_scaffolding. M1 lands the lookup tables +
dispatch entries the rest of the MMA pipeline hangs off. MMA shares
the is_individual=True path with tennis (1v1, no homeAway) but
differs structurally: ESPN MMA events are CARDS (one event = one
fight night) with competitions[] = individual fights. No groupings
layer like tennis tournaments.
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


class TestFetchUpcomingMmaConfig:
    def test_mma_registered_in_sport_configs(self):
        assert "mma" in fu.SPORT_CONFIGS

    def test_mma_uses_mma_espn_path(self):
        # ESPN path /sports/mma/ufc/scoreboard. espn_path is just
        # "mma" so the slug "ufc" doesn't double up.
        cfg = fu.SPORT_CONFIGS["mma"]
        assert cfg.espn_path == "mma"

    def test_ufc_league_registered(self):
        cfg = fu.SPORT_CONFIGS["mma"]
        assert "ufc" in cfg.leagues
        code, name, country = cfg.leagues["ufc"]
        assert code == "UFC"
        assert name == "UFC"
        assert country == "USA"

    def test_mma_sport_string_lowercase(self):
        assert fu.SPORT_CONFIGS["mma"].sport == "mma"

    def test_mma_is_individual_flag(self):
        # MMA shares the 1v1 / athlete / positional-fallback path
        # with tennis. is_individual=True triggers _competitor_name's
        # athlete lookup + process_event's positional handling when
        # homeAway is absent.
        assert fu.SPORT_CONFIGS["mma"].is_individual is True

    def test_mma_season_is_calendar_year(self):
        # UFC schedules ~40-45 cards/year spanning the full calendar.
        # No cross-year season — single 4-digit year string.
        season_func = fu.SPORT_CONFIGS["mma"].season_func
        assert season_func(datetime(2024, 1, 20)) == "2024"
        assert season_func(datetime(2024, 12, 15)) == "2024"
        assert season_func(datetime(2025, 1, 5)) == "2025"


# ── fetch_live_odds (the-odds-api) dispatch ─────────────────────────


class TestFetchLiveOddsMmaDispatch:
    def test_mma_prefix_routes_to_mma(self):
        # the-odds-api uses a single global "mma_mixed_martial_arts"
        # key (vs tennis which has per-tournament keys). Prefix match
        # on "mma_" routes to the 'mma' label.
        assert flo.sport_for_key("mma_mixed_martial_arts") == "mma"

    def test_mma_in_default_set(self):
        assert "mma_mixed_martial_arts" in flo.DEFAULT_SPORTS

    def test_mma_markets_h2h_only(self):
        # Single market in v1. Method-of-victory / round-group /
        # fight-goes-the-distance are MMA specialty markets we'll
        # add via per-event additional markets in v2.
        assert flo.SPORT_MARKETS["mma"] == "h2h"

    def test_mma_market_keys_registered(self):
        assert flo.KNOWN_MARKET_KEYS["mma"] == {"h2h"}


# ── map_outcome MMA branch ──────────────────────────────────────────


class TestMapOutcomeMma:
    """MMA shares the 1v1 moneyline-only shape with tennis but
    without the totals market. h2h is the only thing wired."""

    def test_h2h_fighter1_routes_to_moneyline_home(self):
        result = flo.map_outcome(
            sport="mma",
            market_key="h2h",
            outcome_name="Israel Adesanya",
            home_team_name="Israel Adesanya",
            away_team_name="Dricus Du Plessis",
            point=None,
        )
        assert result == ("moneyline", "home", False)

    def test_h2h_fighter2_routes_to_moneyline_away(self):
        result = flo.map_outcome(
            sport="mma",
            market_key="h2h",
            outcome_name="Dricus Du Plessis",
            home_team_name="Israel Adesanya",
            away_team_name="Dricus Du Plessis",
            point=None,
        )
        assert result == ("moneyline", "away", False)

    def test_totals_returns_none(self):
        # No totals market on MMA in v1. If a future commit adds
        # round-total over/under, this test needs updating.
        result = flo.map_outcome(
            sport="mma",
            market_key="totals",
            outcome_name="Over",
            home_team_name="A",
            away_team_name="B",
            point=2.5,
        )
        assert result == (None, None, False)

    def test_spreads_returns_none(self):
        # MMA doesn't have a spread market — fights are 1v1 outcomes,
        # not point-spread comparisons.
        result = flo.map_outcome(
            sport="mma",
            market_key="spreads",
            outcome_name="Israel Adesanya",
            home_team_name="Israel Adesanya",
            away_team_name="Dricus Du Plessis",
            point=-1.5,
        )
        assert result == (None, None, False)

    def test_unknown_market_returns_nones(self):
        # method_of_victory / round_group not wired yet — drop, don't
        # crash.
        result = flo.map_outcome(
            sport="mma",
            market_key="method_of_victory",
            outcome_name="Adesanya by KO",
            home_team_name="Israel Adesanya",
            away_team_name="Dricus Du Plessis",
            point=None,
        )
        assert result == (None, None, False)
