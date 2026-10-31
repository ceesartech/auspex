"""promote_raw: which football-data rows become finished matches, what the
conflict branch may overwrite, and how a season code becomes a label.

The upsert is exercised for real against SQLite (3.35+ speaks the same
INSERT ... ON CONFLICT DO UPDATE ... WHERE ... RETURNING); only the
placeholder and NOW() are translated."""

from __future__ import annotations

import importlib.util
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pr = _load("promote_raw", "promote_raw.py")

# The two rows football-data published this way (raw_football_data, prod 2026-10-31).
AWARDED = {
    "league": "F2",
    "season": "2526",
    "Date": "05/12/2025",
    "Time": "19:00",
    "HomeTeam": "Bastia",
    "AwayTeam": "Red Star",
    "FTHG": "",
    "FTAG": "",
    "FTR": "",
    "HTHG": "0.0",
    "HTAG": "3.0",
    "HTR": "A",
}
UNPLAYED = {**AWARDED, "HTHG": "", "HTAG": "", "HTR": ""}
NORMAL = {**AWARDED, "FTHG": "2", "FTAG": "1", "FTR": "H", "HTHG": "1", "HTAG": "0", "HTR": "H"}


class TestRawResult:
    def test_awarded_row_is_not_a_result(self):
        # FT blank, "HT" 0-3 with HTR 'A' is the awarded score, not a half.
        assert pr.raw_result(AWARDED) is None

    def test_unplayed_row_is_not_a_result(self):
        assert pr.raw_result(UNPLAYED) is None
        assert pr.raw_result({**UNPLAYED, "FTHG": None, "FTAG": None}) is None

    def test_one_blank_full_time_column_is_not_a_result(self):
        assert pr.raw_result({**NORMAL, "FTAG": ""}) is None
        assert pr.raw_result({**NORMAL, "FTHG": None}) is None

    def test_normal_row_is_unchanged(self):
        assert pr.raw_result(NORMAL) == pr.RawResult(2, 1, 1, 0)
        assert pr.raw_result({**NORMAL, "FTHG": "2.0", "HTHG": "1.0"}) == pr.RawResult(2, 1, 1, 0)

    def test_file_without_half_time_columns(self):
        # The /new/ country files carry no HTHG/HTAG at all.
        row = {k: v for k, v in NORMAL.items() if k not in ("HTHG", "HTAG", "HTR")}
        assert pr.raw_result(row) == pr.RawResult(2, 1)

    def test_half_time_above_full_time_is_dropped(self):
        result = pr.raw_result({**NORMAL, "FTHG": "0", "FTAG": "1", "HTHG": "0", "HTAG": "3"})
        assert (result.home_score, result.away_score, result.ht_home, result.ht_away) == (0, 1, None, None)
        assert "exceeds full-time" in result.ht_dropped

    def test_split_half_time_pair_is_dropped(self):
        result = pr.raw_result({**NORMAL, "HTAG": ""})
        assert (result.ht_home, result.ht_away) == (None, None)
        assert "incomplete" in result.ht_dropped


class TestSeasonLabel:
    @pytest.mark.parametrize(
        "code, expected",
        [("2425", "2024-2025"), ("9900", "1999-2000"), ("0708", "2007-2008"), ("9394", "1993-1994")],
    )
    def test_european_season_codes(self, code, expected):
        assert pr.season_label(code, "E0") == expected
        assert pr.season_label(code) == expected

    @pytest.mark.parametrize("league", ["BR1", "MLS", "NO1", "SE1", "JP1", "CN1", "AR1", "FI1", "IE1"])
    def test_calendar_year_stays_a_year(self, league):
        assert pr.season_label("2016", league) == "2016"
        assert pr.season_label("2021", league) == "2021"  # not the 2020-21 code
        assert pr.season_label("2020", league) == "2020"

    def test_international_year(self):
        assert pr.season_label("1998", "WC") == "1998"

    def test_cross_year_spelling_matches_the_other_loaders(self):
        assert pr.season_label("2012/2013", "AT1") == "2012-2013"
        assert pr.season_label("2026/2027", "MX1") == "2026-2027"

    def test_without_a_league_only_consecutive_halves_expand(self):
        assert pr.season_label("2016") == "2016"
        assert pr.season_label("2021") == "2020-2021"  # ambiguous: the old reading wins

    def test_the_old_bug_is_gone(self):
        # '2016' read as a 'YYZZ' code was stored as '2020-2116'.
        for league in pr.LEAGUE_MAP:
            for year in range(1993, 2031):
                label = pr.season_label(str(year), league)
                if "-" in label:
                    start, end = map(int, label.split("-"))
                    assert end - start == 1, (league, year, label)

    def test_european_leagues_keep_their_labels(self):
        legacy = _load("backfill_season_labels", "backfill_season_labels.py").legacy_season_label
        for league in set(pr.LEAGUE_MAP) - pr.CALENDAR_SEASON_LEAGUES:
            for start in range(93, 127):
                code = f"{start % 100:02d}{(start + 1) % 100:02d}"
                assert pr.season_label(code, league) == legacy(code), (league, code)

    @pytest.mark.parametrize("code", ["", None, "abc", "20166"])
    def test_other_values_pass_through(self, code):
        assert pr.season_label(code, "BR1") == (code or "")

    def test_every_new_format_league_is_calendar_aware(self):
        extra = _load("load_football_data_extra", "load_football_data_extra.py")
        codes = {code for code, _name, _country in extra.COUNTRY_MAP.values()}
        assert codes <= pr.CALENDAR_SEASON_LEAGUES, codes - pr.CALENDAR_SEASON_LEAGUES


class _SqliteCursor:
    """Just enough of a RealDictCursor over SQLite for insert_match."""

    def __init__(self, conn: sqlite3.Connection):
        self._cur = conn.cursor()

    def execute(self, sql, params=()):
        self._cur.execute(sql.replace("%s", "?").replace("NOW()", "CURRENT_TIMESTAMP"), params)

    def fetchone(self):
        row = self._cur.fetchone()
        return None if row is None else {"id": row[0]}


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE matches (
            id INTEGER PRIMARY KEY,
            league_id TEXT, home_team_id TEXT, away_team_id TEXT, match_date TEXT,
            status TEXT CHECK (status IN ('scheduled', 'live', 'finished', 'postponed', 'cancelled')),
            home_score INTEGER, away_score INTEGER, half_time_home INTEGER, half_time_away INTEGER,
            season TEXT, updated_at TEXT,
            UNIQUE (home_team_id, away_team_id, match_date)
        )
        """
    )
    yield conn
    conn.close()


KICKOFF = "2025-12-05 19:00:00"


def _insert(db, home_score=2, away_score=1, ht_home=1, ht_away=0, season="2025-2026"):
    return pr.insert_match(
        _SqliteCursor(db),
        league_id="lg",
        home_id="h",
        away_id="a",
        match_dt=KICKOFF,
        season=season,
        home_score=home_score,
        away_score=away_score,
        ht_home=ht_home,
        ht_away=ht_away,
    )


def _existing(db, status, home_score=None, away_score=None, ht_home=None, ht_away=None, season="espn-season"):
    db.execute(
        "INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, home_score, away_score, "
        "half_time_home, half_time_away, season) VALUES ('lg', 'h', 'a', ?, ?, ?, ?, ?, ?, ?)",
        (KICKOFF, status, home_score, away_score, ht_home, ht_away, season),
    )


def _row(db):
    return db.execute(
        "SELECT status, home_score, away_score, half_time_home, half_time_away, season FROM matches"
    ).fetchone()


class TestInsertMatch:
    def test_new_row_is_finished_with_both_pairs(self, db):
        assert _insert(db) is not None
        assert _row(db) == ("finished", 2, 1, 1, 0, "2025-2026")

    def test_refuses_a_row_without_full_time(self, db):
        with pytest.raises(ValueError, match="full-time"):
            _insert(db, home_score=None, away_score=None, ht_home=0, ht_away=3)
        with pytest.raises(ValueError, match="full-time"):
            _insert(db, away_score=None)
        assert _row(db) is None

    def test_refuses_a_split_half_time_pair(self, db):
        with pytest.raises(ValueError, match="split"):
            _insert(db, ht_away=None)

    def test_rerun_without_half_time_keeps_the_existing_half_time(self, db):
        _existing(db, "finished", 2, 1, 1, 0)
        assert _insert(db, ht_home=None, ht_away=None) is not None
        assert _row(db)[:5] == ("finished", 2, 1, 1, 0)

    def test_scheduled_fixture_becomes_finished(self, db):
        _existing(db, "scheduled")
        assert _insert(db) is not None
        assert _row(db) == ("finished", 2, 1, 1, 0, "espn-season")  # season untouched

    def test_cancelled_row_is_left_alone(self, db):
        _existing(db, "cancelled")
        assert _insert(db) is None
        assert _row(db) == ("cancelled", None, None, None, None, "espn-season")


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self, cursor_factory=None):
        return _FakeCursor(self._rows)

    def commit(self):
        pass


@pytest.fixture
def promote_with(monkeypatch):
    """Run promote() over raw rows with every DB helper faked; returns the
    insert_match calls."""
    calls: list[dict] = []

    def run(rows, insert_returns="m-1", legacy_check=lambda cur, rows, league_ids, team_ids: None):
        monkeypatch.setattr(pr.psycopg2, "connect", lambda url: _FakeConn(rows))
        monkeypatch.setattr(pr, "check_legacy_kickoffs", legacy_check)
        monkeypatch.setattr(pr, "upsert_leagues", lambda cur, codes: {c: f"lg-{c}" for c in codes})
        monkeypatch.setattr(
            pr, "upsert_teams", lambda cur, names, by_team: {pr.normalize_team_name(n): f"t-{n}" for n in names}
        )
        monkeypatch.setattr(pr, "insert_match", lambda cur, **kw: calls.append(kw) or insert_returns)
        monkeypatch.setattr(pr, "insert_match_stats", lambda *a, **kw: None)
        monkeypatch.setattr(pr, "insert_odds", lambda *a, **kw: 0)
        return pr.promote("postgres://fake"), calls

    return run


class TestPromote:
    def test_awarded_row_is_skipped_loudly_and_normal_row_promoted(self, promote_with, caplog):
        normal = {**NORMAL, "league": "G1", "HomeTeam": "Panathinaikos", "AwayTeam": "Aris"}
        with caplog.at_level(logging.WARNING, logger="promote_raw"):
            stats, calls = promote_with([AWARDED, UNPLAYED | {"HomeTeam": "Niki Volos"}, normal])
        assert stats.skipped_no_result == 2 and stats.matches == 1
        assert [c["home_id"] for c in calls] == ["t-Panathinaikos"]
        assert (calls[0]["home_score"], calls[0]["away_score"], calls[0]["ht_home"], calls[0]["ht_away"]) == (
            2,
            1,
            1,
            0,
        )
        assert "Not promoting F2 05/12/2025 Bastia v Red Star" in caplog.text

    def test_calendar_year_league_gets_a_year_label(self, promote_with):
        brazil = {**NORMAL, "league": "BR1", "season": "2016", "HomeTeam": "Chapecoense-SC", "AwayTeam": "Santos"}
        _stats, calls = promote_with([brazil])
        assert calls[0]["season"] == "2016"

    def test_cancelled_existing_row_is_counted_not_promoted(self, promote_with, caplog):
        with caplog.at_level(logging.WARNING, logger="promote_raw"):
            stats, _calls = promote_with([NORMAL], insert_returns=None)
        assert (stats.matches, stats.left_cancelled, stats.match_stats) == (0, 1, 0)
        assert "is 'cancelled'" in caplog.text

    def test_the_legacy_kickoff_check_runs_before_any_match_is_written(self, promote_with):
        def refuse(cur, rows, league_ids, team_ids):
            assert set(league_ids) == {"F2"} and team_ids  # after the league/team upserts
            raise pr.LegacyKickoffsPresent("1 raw row(s) would be promoted as a second matches row")

        with pytest.raises(pr.LegacyKickoffsPresent):
            promote_with([NORMAL], legacy_check=refuse)
        _stats, calls = promote_with([NORMAL])
        assert len(calls) == 1  # the same rows promote once nothing is stranded

    def test_main_reports_the_refusal_and_exits_1(self, monkeypatch, caplog):
        def refuse(url, batch_size):
            raise pr.LegacyKickoffsPresent("63768 raw row(s) would be promoted as a second matches row")

        monkeypatch.setattr(pr, "promote", refuse)
        assert pr.main(["--database-url", "postgres://fake"]) == 1
        assert "Refusing to promote — nothing written: 63768 raw row(s)" in caplog.text


UTC = timezone.utc


class TestParseMatchDate:
    """football-data's Time is UK local time. Until 2026-10-31 it was stored
    as if it were UTC, so every summer kickoff landed 1 h late (65 fixtures
    the ESPN results path also wrote sit exactly 1 h apart on prod)."""

    def test_a_bst_kickoff_is_an_hour_earlier_in_utc(self):
        assert pr.parse_match_date("05/07/2025", "15:00") == datetime(2025, 7, 5, 14, 0, tzinfo=UTC)

    def test_a_gmt_kickoff_is_the_same_in_utc(self):
        assert pr.parse_match_date("05/12/2025", "19:00") == datetime(2025, 12, 5, 19, 0, tzinfo=UTC)

    def test_the_prod_duplicate_now_lands_on_the_espn_kickoff(self):
        # Shandong Taishan v Yunnan Yukun (CN1): ESPN row e77e612a is
        # 2026-07-10 11:35Z; the football-data row (90455957) sits at 12:35.
        assert pr.parse_match_date("10/07/2026", "12:35") == datetime(2026, 7, 10, 11, 35, tzinfo=UTC)

    def test_a_uk_after_midnight_kickoff_moves_back_a_day(self):
        # Internacional v Cruzeiro (BR1), 00:30 UK on 23/07 = 22/07 23:30Z.
        assert pr.parse_match_date("23/07/2026", "00:30") == datetime(2026, 7, 22, 23, 30, tzinfo=UTC)

    def test_the_two_digit_year_format_converts_too(self):
        assert pr.parse_match_date("13/08/16", "15:00") == datetime(2016, 8, 13, 14, 0, tzinfo=UTC)

    def test_the_repeated_october_hour_reads_as_bst(self):
        assert pr.parse_match_date("26/10/2025", "01:30") == datetime(2025, 10, 26, 0, 30, tzinfo=UTC)

    @pytest.mark.parametrize("raw_time", [None, "", "TBC", "25:99"])
    def test_a_date_only_row_keeps_its_naive_midnight(self, raw_time):
        # Unchanged: Postgres reads it in the session TimeZone, as before.
        got = pr.parse_match_date("05/07/2025", raw_time)
        assert got == datetime(2025, 7, 5) and got.tzinfo is None

    @pytest.mark.parametrize("raw_date", ["", None, "2025-07-05", "32/07/2025"])
    def test_an_unreadable_date_is_none(self, raw_date):
        assert pr.parse_match_date(raw_date, "15:00") is None

    def test_parse_wall_clock_is_the_legacy_naive_reading(self):
        assert pr.parse_wall_clock("05/07/2025", "15:00") == datetime(2025, 7, 5, 15, 0)
        assert pr.parse_wall_clock("05/07/2025", None) == datetime(2025, 7, 5)


SUMMER = {**NORMAL, "Date": "05/07/2025", "Time": "15:00"}  # 14:00Z; legacy row at 15:00Z
WINTER = {**NORMAL, "Date": "05/12/2025", "Time": "19:00"}  # 19:00Z either way
LEAGUE_IDS = {"F2": "lg-F2"}
TEAM_IDS = {"bastia": "t-bastia", "red star": "t-red-star"}


def _key(when):
    return ("t-bastia", "t-red-star", when)


class TestStrandedLegacyKickoffs:
    """A football-data run re-promotes the whole raw table. Without this check
    the first run after the timezone fix would miss ON CONFLICT on every
    summer row promote_raw wrote at the legacy kickoff and insert a second
    finished match for each: about 63,800 rows on prod (2026-10-31)."""

    def _stranded(self, rows, existing, db_tz=UTC):
        return pr.stranded_legacy_kickoffs(rows, LEAGUE_IDS, TEAM_IDS, set(existing), db_tz)

    def test_a_summer_row_at_the_legacy_kickoff_is_stranded(self):
        assert self._stranded([SUMMER], [_key(datetime(2025, 7, 5, 15, 0, tzinfo=UTC))]) == [SUMMER]

    def test_a_winter_row_is_unchanged_in_a_utc_database(self):
        assert self._stranded([WINTER], [_key(datetime(2025, 12, 5, 19, 0, tzinfo=UTC))]) == []

    def test_a_row_already_moved_to_utc_is_fine(self):
        assert self._stranded([SUMMER], [_key(datetime(2025, 7, 5, 14, 0, tzinfo=UTC))]) == []

    def test_a_legacy_row_beside_an_espn_twin_does_not_block(self):
        # The upsert lands on the ESPN row; the old twin is the operator's job.
        legacy, espn = datetime(2025, 7, 5, 15, 0, tzinfo=UTC), datetime(2025, 7, 5, 14, 0, tzinfo=UTC)
        assert self._stranded([SUMMER], [_key(legacy), _key(espn)]) == []

    def test_date_only_unresolved_and_unfinished_rows_are_ignored(self):
        midnight = datetime(2025, 7, 5, 0, 0, tzinfo=UTC)
        rows = [{**SUMMER, "Time": ""}, {**SUMMER, "HomeTeam": "Nowhere"}, {**SUMMER, "FTHG": ""}]
        assert self._stranded(rows, [_key(midnight), _key(datetime(2025, 7, 5, 15, 0, tzinfo=UTC))]) == []

    def test_the_legacy_kickoff_is_read_in_the_database_timezone(self):
        # A naive 15:00 in a Europe/Paris session was stored as 13:00Z.
        assert self._stranded(
            [SUMMER], [_key(datetime(2025, 7, 5, 13, 0, tzinfo=UTC))], db_tz=ZoneInfo("Europe/Paris")
        ) == [SUMMER]


class _GuardCursor:
    def __init__(self, existing, tz="UTC"):
        self._existing, self._tz, self._last = existing, tz, ""

    def execute(self, sql, params=None):
        self._last = sql

    def fetchone(self):
        return {"TimeZone": self._tz}

    def fetchall(self):
        return [{"home_id": h, "away_id": a, "match_date": when} for h, a, when in self._existing]


class TestCheckLegacyKickoffs:
    def test_raises_with_the_count_and_a_sample(self):
        cur = _GuardCursor([_key(datetime(2025, 7, 5, 15, 0, tzinfo=UTC))])
        with pytest.raises(pr.LegacyKickoffsPresent, match=r"^1 raw row\(s\).*F2 05/07/2025 Bastia v Red Star 15:00"):
            pr.check_legacy_kickoffs(cur, [SUMMER, WINTER], LEAGUE_IDS, TEAM_IDS)

    def test_passes_once_the_rows_sit_at_utc(self):
        cur = _GuardCursor([_key(datetime(2025, 7, 5, 14, 0, tzinfo=UTC))])
        assert pr.check_legacy_kickoffs(cur, [SUMMER, WINTER], LEAGUE_IDS, TEAM_IDS) is None

    def test_postgres_timestamps_in_another_zone_compare_by_instant(self):
        paris = datetime(2025, 7, 5, 17, 0, tzinfo=ZoneInfo("Europe/Paris"))  # 15:00Z
        with pytest.raises(pr.LegacyKickoffsPresent):
            pr.check_legacy_kickoffs(_GuardCursor([_key(paris)]), [SUMMER], LEAGUE_IDS, TEAM_IDS)
