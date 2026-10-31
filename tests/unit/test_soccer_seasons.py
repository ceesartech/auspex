"""scripts/soccer_seasons.py: the one soccer season-label scheme.

The evidence for each league (ESPN scoreboard season blocks, football-data
season labels) is in the module docstring; these tests pin the decisions and
that promote_raw, fetch_upcoming and football-data's own labels agree."""

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


ss = _load("soccer_seasons", "soccer_seasons.py")
pr = _load("promote_raw_for_soccer_seasons", "promote_raw.py")


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


class TestTheScheme:
    @pytest.mark.parametrize(
        "league", ["MLS", "BR1", "AR1", "CN1", "NO1", "SE1", "CL1", "CO1", "FI1", "IE1", "KR1", "WC", "GOLD", "LIB"]
    )
    def test_calendar_year_leagues_label_by_the_year(self, league):
        assert ss.season_for_kickoff(league, utc(2026, 3, 20)) == "2026"
        assert ss.season_for_kickoff(league, utc(2026, 7, 20)) == "2026"
        assert ss.season_for_kickoff(league, utc(2026, 11, 8)) == "2026"

    @pytest.mark.parametrize("league", ["MX1", "E0", "SP1", "AU1", "UCL", "AT1", None, "XX9"])
    def test_cross_year_leagues_cut_in_july(self, league):
        assert ss.season_for_kickoff(league, utc(2026, 6, 30, 23)) == "2025-2026"
        assert ss.season_for_kickoff(league, utc(2026, 7, 1)) == "2026-2027"
        assert ss.season_for_kickoff(league, utc(2027, 5, 25)) == "2026-2027"

    def test_j1_is_calendar_year_through_its_2026_special_season_then_aug_may(self):
        # ESPN: '2026-j1-100-year-vision-league' from 2026-02-06 to the
        # placement play-offs of 2026-06-06; 'regular-season' from 2026-08-08.
        assert ss.season_for_kickoff("JP1", utc(2025, 11, 30)) == "2025"
        assert ss.season_for_kickoff("JP1", utc(2026, 2, 14, 5)) == "2026"
        assert ss.season_for_kickoff("JP1", utc(2026, 6, 6, 5)) == "2026"
        assert ss.season_for_kickoff("JP1", utc(2026, 8, 8, 10)) == "2026-2027"
        assert ss.season_for_kickoff("JP1", utc(2027, 6, 6)) == "2026-2027"

    @pytest.mark.parametrize(
        "league, when",
        [
            ("AR1", utc(2019, 10, 1)),  # football-data '2019/2020'
            ("AR1", utc(2021, 1, 18)),  # the Copa Diego Maradona, football-data '2020'
            ("CL1", utc(2016, 10, 1)),
            ("BR1", utc(2021, 2, 26)),  # Brasileirao 2020 ran to 2021-02-26
            ("CN1", utc(2022, 1, 12)),  # CSL 2021 ran to 2022-01-12
        ],
    )
    def test_irregular_eras_have_no_date_rule(self, league, when):
        assert ss.season_for_kickoff(league, when) is None
        assert ss.season_scheme(league, when) is None

    def test_the_eras_open_where_the_seasons_turned_regular(self):
        assert ss.season_for_kickoff("AR1", utc(2021, 2, 12)) == "2021"
        assert ss.season_for_kickoff("CL1", utc(2018, 2, 1)) == "2018"
        assert ss.season_for_kickoff("BR1", utc(2021, 5, 29)) == "2021"
        assert ss.season_for_kickoff("CN1", utc(2022, 6, 3)) == "2022"

    def test_a_naive_kickoff_is_utc_and_an_aware_one_is_converted(self):
        assert ss.cross_year_label(datetime(2026, 7, 1, 0, 30)) == "2026-2027"
        local = datetime.fromisoformat("2026-06-30T20:30:00-04:00")  # 00:30Z on July 1
        assert ss.cross_year_label(local) == "2026-2027"
        assert ss.season_for_kickoff("MLS", datetime.fromisoformat("2026-12-31T21:00:00-05:00")) == "2027"


class TestOneSchemeEverywhere:
    def test_the_international_codes_are_promote_raws(self):
        international = {code for code, (_n, country) in pr.LEAGUE_MAP.items() if country == "International"}
        assert ss.INTERNATIONAL_CODES == international

    def test_promote_raw_parses_by_the_shared_set(self):
        assert pr.YEAR_CODED_LEAGUES is ss.FOOTBALL_DATA_YEAR_CODED
        assert not hasattr(pr, "CALENDAR_SEASON_LEAGUES")  # the misleading name is gone

    def test_the_year_coded_set_is_exactly_the_new_format_files_and_the_internationals(self):
        extra = _load("load_football_data_extra_for_seasons", "load_football_data_extra.py")
        codes = {code for code, _name, _country in extra.COUNTRY_MAP.values()}
        assert ss.FOOTBALL_DATA_YEAR_CODED == codes | ss.INTERNATIONAL_CODES

    @pytest.mark.parametrize(
        "league, year, kickoff",
        [
            ("MLS", "2025", utc(2025, 6, 14)),
            ("BR1", "2025", utc(2025, 9, 14)),
            ("AR1", "2024", utc(2024, 8, 3)),
            ("CN1", "2023", utc(2023, 7, 1)),
            ("NO1", "2026", utc(2026, 7, 26)),
            ("SE1", "2024", utc(2024, 10, 20)),
            ("FI1", "2025", utc(2025, 8, 9)),
            ("IE1", "2025", utc(2025, 9, 1)),
            ("JP1", "2025", utc(2025, 8, 16)),
        ],
    )
    def test_the_date_rule_reproduces_football_datas_calendar_labels(self, league, year, kickoff):
        assert ss.season_for_kickoff(league, kickoff) == pr.season_label(year, league) == year

    @pytest.mark.parametrize(
        "league, code, kickoff",
        [
            ("MX1", "2025/2026", utc(2025, 8, 9)),
            ("MX1", "2025/2026", utc(2026, 4, 12)),
            ("MX1", "2022/2023", utc(2022, 7, 2)),
            ("AT1", "2024/2025", utc(2025, 3, 1)),
        ],
    )
    def test_the_date_rule_reproduces_football_datas_cross_year_labels(self, league, code, kickoff):
        assert ss.season_for_kickoff(league, kickoff) == pr.season_label(code, league)
