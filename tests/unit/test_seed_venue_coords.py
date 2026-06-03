"""Unit tests for seed_venue_coords — venue dataset shape checks.

These tests don't hit Postgres — they verify the static venue
dataset is well-formed (no NaN coords, all NFL stadiums covered,
indoor flags correct for known domes).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


svc = _load("seed_venue_coords", "seed_venue_coords.py")


class TestNflVenues:
    def test_all_thirty_two_stadiums_present(self):
        # NFL = 32 teams. MetLife is shared by Giants+Jets (1 stadium
        # for 2 teams) and SoFi shared by Chargers+Rams (1 for 2),
        # so 32 teams - 2 = 30 unique stadiums.
        assert len(svc.NFL_VENUES) == 30

    def test_every_venue_has_valid_coords(self):
        for v in svc.NFL_VENUES:
            assert -90 <= v.latitude <= 90, f"{v.display_name}: bad latitude"
            assert -180 <= v.longitude <= 180, f"{v.display_name}: bad longitude"
            # NFL stadiums are all in continental US — sanity check
            # the box (excludes outlier typos like a flipped sign).
            assert 24 <= v.latitude <= 49, f"{v.display_name}: outside US latitude range"
            assert -125 <= v.longitude <= -66, f"{v.display_name}: outside US longitude range"

    def test_every_venue_has_us_timezone(self):
        valid_tz_prefixes = ("America/",)
        for v in svc.NFL_VENUES:
            assert any(
                v.timezone.startswith(p) for p in valid_tz_prefixes
            ), f"{v.display_name}: timezone {v.timezone} not America/*"

    def test_known_indoor_stadiums_flagged(self):
        # The 8 active NFL domes / climate-controlled venues. Weather
        # features get skipped for these — lock the flag explicitly.
        indoor_names = {v.display_name for v in svc.NFL_VENUES if v.is_indoor}
        assert indoor_names == {
            "NRG Stadium",  # Texans
            "Lucas Oil Stadium",  # Colts
            "Allegiant Stadium",  # Raiders
            "SoFi Stadium",  # Chargers/Rams
            "AT&T Stadium",  # Cowboys
            "Ford Field",  # Lions
            "U.S. Bank Stadium",  # Vikings
            "Mercedes-Benz Stadium",  # Falcons
            "Caesars Superdome",  # Saints
            "State Farm Stadium",  # Cardinals
        }


class TestTennisVenues:
    def test_grand_slams_plus_major_tour_stops(self):
        # 4 Grand Slams + ~28 major ATP/WTA tour stops. ESPN's tennis
        # venue field stores "City, Country" (not stadium names), so
        # the display_name doubles as the primary lookup key with
        # stadium/court names as aliases.
        assert len(svc.TENNIS_VENUES) >= 30

    def test_all_four_grand_slams_present(self):
        # Locked: removing a Slam would silently drop weather features
        # for the entire tournament's worth of matches.
        names = {v.display_name for v in svc.TENNIS_VENUES}
        assert "Melbourne, Australia" in names  # Australian Open
        assert "Paris, France" in names  # Roland Garros
        assert "London, Great Britain" in names  # Wimbledon
        assert "New York, USA" in names  # US Open

    def test_grand_slam_venues_have_court_aliases(self):
        # Slams have multiple courts (Centre Court, Court 1, etc.) and
        # the stadium-name aliases let the lookup match other ingest
        # paths that report stadium-level venues.
        slam_cities = {"Melbourne, Australia", "Paris, France", "London, Great Britain", "New York, USA"}
        for v in svc.TENNIS_VENUES:
            if v.display_name in slam_cities:
                assert v.aliases, f"{v.display_name}: missing court aliases"

    def test_no_tennis_venue_is_indoor(self):
        # All v1 tennis venues are outdoor (some have retractable roofs
        # but the model treats them as outdoor since weather affects
        # play before the roof closes). Indoor venues like Wiener
        # Stadthalle or Pala Alpitour have the venue marker but are
        # left as outdoor here for now — they all have weather-aware
        # decisions about roof opening.
        for v in svc.TENNIS_VENUES:
            assert not v.is_indoor


class TestSoccerVenues:
    def test_coverage_of_top_5_leagues(self):
        # Subset, not exhaustive. ~20 grounds across EPL + La Liga +
        # Serie A + Bundesliga + Ligue 1.
        assert len(svc.SOCCER_VENUES) >= 18

    def test_every_venue_has_european_timezone(self):
        for v in svc.SOCCER_VENUES:
            assert v.timezone.startswith("Europe/"), f"{v.display_name}: timezone {v.timezone} not Europe/*"

    def test_no_soccer_venue_is_indoor_by_default(self):
        # All top-5 league grounds are open-air. If a club moves to
        # a covered ground (rare), the seed needs an explicit update.
        for v in svc.SOCCER_VENUES:
            assert not v.is_indoor


class TestNormalizeVenue:
    def test_matches_fetch_weather_normalization(self):
        # The seed normalization MUST match fetch_weather's lookup
        # normalization or the lookups will miss every venue.
        # Lockdown so a future refactor can't break the contract.
        fw = _load("fetch_weather", "fetch_weather.py")
        for v in svc.NFL_VENUES[:3]:
            # Same shape (lowercase + collapsed whitespace).
            assert svc.normalize_venue(v.display_name) == fw.normalize_venue(v.display_name)


class TestCli:
    def test_default_update_false(self):
        args = svc.parse_args(["--database-url", "x"])
        assert args.update is False

    def test_update_flag(self):
        args = svc.parse_args(["--update", "--database-url", "x"])
        assert args.update is True
