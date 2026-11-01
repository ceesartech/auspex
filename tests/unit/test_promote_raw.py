"""promote_raw: which football-data rows become finished matches, what the
conflict branch may overwrite, and how a season code becomes a label.

The upsert is exercised for real against SQLite (3.35+ speaks the same
INSERT ... ON CONFLICT DO UPDATE ... WHERE ... RETURNING); only the
placeholder and NOW() are translated."""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sqlite3
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
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
        for league in set(pr.LEAGUE_MAP) - pr.YEAR_CODED_LEAGUES:
            for start in range(93, 127):
                code = f"{start % 100:02d}{(start + 1) % 100:02d}"
                assert pr.season_label(code, league) == legacy(code), (league, code)

    @pytest.mark.parametrize("code", ["", None, "abc", "20166"])
    def test_other_values_pass_through(self, code):
        assert pr.season_label(code, "BR1") == (code or "")

    def test_every_new_format_league_is_calendar_aware(self):
        extra = _load("load_football_data_extra", "load_football_data_extra.py")
        codes = {code for code, _name, _country in extra.COUNTRY_MAP.values()}
        assert codes <= pr.YEAR_CODED_LEAGUES, codes - pr.YEAR_CODED_LEAGUES


def _sqlite_ts(value):
    """A timestamp as fixed-width UTC text, so SQLite compares it the way
    Postgres compares timestamptz; anything else unchanged."""
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.strftime("%Y-%m-%dT%H:%M:%S.%f")
    return value


class _SqliteCursor:
    """Just enough of a RealDictCursor over SQLite for insert_match,
    update_existing_match and find_existing_matches: named %(x)s parameters
    become :x, '::text' casts go, "col->>'key'" becomes json_extract, and
    timestamps travel as fixed-width UTC text and come back aware."""

    def __init__(self, conn: sqlite3.Connection):
        self._cur = conn.cursor()

    def execute(self, sql, params=()):
        sql = re.sub(r"%\((\w+)\)s", r":\1", sql).replace("%s", "?").replace("NOW()", "CURRENT_TIMESTAMP")
        sql = re.sub(r"(\w+)->>'(\w+)'", r"json_extract(\1, '$.\2')", sql.replace("::text", ""))
        if isinstance(params, dict):
            params = {k: _sqlite_ts(v) for k, v in params.items()}
        else:
            params = tuple(_sqlite_ts(v) for v in params)
        self._cur.execute(sql, params)

    def _dict(self, row):
        out = dict(zip([d[0] for d in self._cur.description], row))
        when = out.get("match_date")
        if isinstance(when, str) and "T" in when:
            out["match_date"] = datetime.fromisoformat(when).replace(tzinfo=timezone.utc)
        return out

    def fetchone(self):
        row = self._cur.fetchone()
        return None if row is None else self._dict(row)

    def fetchall(self):
        return [self._dict(r) for r in self._cur.fetchall()]


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
            season TEXT, updated_at TEXT, external_ids TEXT DEFAULT '{}', metadata TEXT DEFAULT '{}',
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

    def run(
        rows,
        insert_returns="m-1",
        legacy_check=lambda cur, rows, league_ids, team_ids: None,
        existing=lambda cur, home_id, away_id, match_dt, fixture_day: [],
        aliases=lambda cur, home_id, away_id, match_dt, fixture_day: [],
    ):
        monkeypatch.setattr(pr.psycopg2, "connect", lambda url: _FakeConn(rows))
        monkeypatch.setattr(pr, "check_legacy_kickoffs", legacy_check)
        monkeypatch.setattr(pr, "find_existing_matches", existing)
        monkeypatch.setattr(pr, "find_alias_slot_rows", aliases)
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


class TestScreenRawRows:
    """Rows promote_raw refuses before anything (2026-11-01): an unknown league code (was a warning), a Div that
    names another division (2026-08-06: SC1/SC2 served at the SP1/SP2 URLs), a mis-decoded team name (the
    latin-1 read of a UTF-8 file: 'PreuÃ\\x9fen MÃ¼nster'). They are not promoted and decide no team's league."""

    GARBLED = "Preußen Münster".encode("utf-8").decode("latin-1")

    def test_each_problem_is_named(self):
        assert pr.raw_row_problem(NORMAL) is None
        assert pr.raw_row_problem({**NORMAL, "Div": "F2"}) is None
        assert pr.raw_row_problem({**NORMAL, "league": "XX9"}) == pr.RAW_UNKNOWN_LEAGUE
        assert pr.raw_row_problem({**NORMAL, "league": "SP1", "Div": "SC1"}) == pr.RAW_DIV_MISMATCH
        assert pr.raw_row_problem({**NORMAL, "league": "D2", "HomeTeam": self.GARBLED}) == pr.RAW_MOJIBAKE

    def test_the_scottish_lower_divisions_have_leagues(self):
        assert pr.LEAGUE_MAP["SC1"] == ("Championship", "Scotland")
        assert pr.LEAGUE_MAP["SC2"] == ("League One", "Scotland")
        assert pr.LEAGUE_MAP["SC3"] == ("League Two", "Scotland")

    def test_refused_rows_are_not_promoted_and_decide_no_teams_league(self, monkeypatch, caplog):
        leagues_by_team: dict = {}
        league_codes: set = set()
        calls: list = []

        def teams(cur, names, by_team):
            leagues_by_team.update(by_team)
            return {pr.normalize_team_name(n): f"t-{n}" for n in names}

        def leagues(cur, codes):
            league_codes.update(codes)
            return {c: f"lg-{c}" for c in codes}

        rows = [
            {**NORMAL, "league": "SP1", "Div": "SC1", "HomeTeam": "Ayr", "AwayTeam": "Arbroath"},
            {**NORMAL, "league": "D2", "HomeTeam": self.GARBLED, "AwayTeam": "Holstein Kiel"},
            {**NORMAL, "league": "XX9", "HomeTeam": "Nowhere", "AwayTeam": "Somewhere"},
            {**NORMAL, "league": "SC1", "Div": "SC1", "HomeTeam": "Morton", "AwayTeam": "Partick"},
        ]
        monkeypatch.setattr(pr.psycopg2, "connect", lambda url: _FakeConn(rows))
        monkeypatch.setattr(pr, "check_legacy_kickoffs", lambda cur, rows, league_ids, team_ids: None)
        monkeypatch.setattr(pr, "find_existing_matches", lambda *a: [])
        monkeypatch.setattr(pr, "find_alias_slot_rows", lambda *a: [])
        monkeypatch.setattr(pr, "upsert_leagues", leagues)
        monkeypatch.setattr(pr, "upsert_teams", teams)
        monkeypatch.setattr(pr, "insert_match", lambda cur, **kw: calls.append(kw) or "m-1")
        monkeypatch.setattr(pr, "insert_match_stats", lambda *a, **kw: None)
        monkeypatch.setattr(pr, "insert_odds", lambda *a, **kw: 0)
        with caplog.at_level(logging.ERROR, logger="promote_raw"):
            stats = pr.promote("postgres://fake")
        assert stats.refused_raw == 3 and stats.matches == 1
        assert [c["home_id"] for c in calls] == ["t-Morton"]
        assert set(leagues_by_team) == {"Morton", "Partick"} and league_codes == {"SC1"}
        assert "REFUSED 1 raw row(s) of SP1 2526 — Div names another division" in caplog.text
        assert "REFUSED 1 raw row(s) of D2 2526 — a team name looks mis-decoded" in caplog.text
        assert "REFUSED 1 raw row(s) of XX9 2526 — league code not in promote_raw.LEAGUE_MAP" in caplog.text

    def test_main_exits_3_when_rows_were_screened_out(self, monkeypatch, caplog):
        monkeypatch.setattr(pr, "promote", lambda url, batch_size: pr.Promotion(matches=5, refused_raw=68))
        with caplog.at_level(logging.ERROR, logger="promote_raw"):
            assert pr.main(["--database-url", "postgres://fake"]) == pr.REFUSED_EXIT_CODE
        assert "screened out 68" in caplog.text


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

    def test_a_legacy_row_beside_an_espn_twin_blocks_until_it_is_merged(self):
        # promote() would write onto the ESPN row and leave the legacy copy beside it for good: a duplicate in the
        # training corpus that only --dedupe merges (review finding, 2026-10-31).
        legacy, espn = datetime(2025, 7, 5, 15, 0, tzinfo=UTC), datetime(2025, 7, 5, 14, 0, tzinfo=UTC)
        assert self._stranded([SUMMER], [_key(legacy), _key(espn)]) == [SUMMER]
        assert self._stranded([SUMMER], [_key(espn)]) == []  # merged: the copy is gone

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

    def test_the_refusal_names_the_repair(self):
        cur = _GuardCursor([_key(datetime(2025, 7, 5, 15, 0, tzinfo=UTC))])
        with pytest.raises(pr.LegacyKickoffsPresent, match="repair_football_data_kickoffs.sh: --dedupe, then --shift"):
            pr.check_legacy_kickoffs(cur, [SUMMER], LEAGUE_IDS, TEAM_IDS)


# --- the insert path: write onto the existing row for the pair, never beside it ---------------------------------


class TestExistingMatchWindow:
    """Within 3 h of the kickoff and on football-data's Date, read as the UK
    date it is."""

    def test_a_daytime_kickoff_is_plus_minus_three_hours(self):
        lo, hi = pr.existing_match_window(datetime(2026, 7, 10, 11, 35, tzinfo=UTC), date(2026, 7, 10))
        assert (lo, hi) == (datetime(2026, 7, 10, 8, 35, tzinfo=UTC), datetime(2026, 7, 10, 14, 35, tzinfo=UTC))

    def test_a_late_us_kickoff_keeps_the_espn_row_on_the_uk_date(self):
        # FC Cincinnati v San Jose (MLS), prod: football-data 02/08/2026 00:30 UK = 23:30Z on the 1st; ESPN's
        # row kicked off at 01:46Z on the 2nd, a stale fixture row sits at 23:30Z. Both are on the UK date.
        lo, hi = pr.existing_match_window(datetime(2026, 8, 1, 23, 30, tzinfo=UTC), date(2026, 8, 2))
        assert lo == datetime(2026, 8, 1, 23, 0, tzinfo=UTC)  # midnight BST, not 20:30Z
        assert hi == datetime(2026, 8, 2, 2, 30, tzinfo=UTC)
        assert lo <= datetime(2026, 8, 2, 1, 46, tzinfo=UTC) <= hi

    def test_the_window_stops_at_the_end_of_the_uk_date(self):
        lo, hi = pr.existing_match_window(datetime(2025, 12, 5, 22, 0, tzinfo=UTC), date(2025, 12, 5))
        assert lo == datetime(2025, 12, 5, 19, 0, tzinfo=UTC)
        assert hi == datetime(2025, 12, 5, 23, 59, 59, 999999, tzinfo=UTC)

    def test_a_date_only_row_looks_at_its_whole_uk_date(self):
        lo, hi = pr.existing_match_window(datetime(2016, 8, 13), date(2016, 8, 13))
        assert lo == datetime(2016, 8, 12, 23, 0, tzinfo=UTC)  # BST midnight
        assert hi == datetime(2016, 8, 13, 22, 59, 59, 999999, tzinfo=UTC)
        assert lo <= datetime(2016, 8, 13, tzinfo=UTC) <= hi  # its own naive midnight, read as UTC


def _cand(id_, status="finished", espn_id=None, result_source=None, when=datetime(2026, 7, 10, 11, 35, tzinfo=UTC)):
    return {"id": id_, "status": status, "espn_id": espn_id, "result_source": result_source, "match_date": when}


class TestChooseExisting:
    def test_none(self):
        assert pr.choose_existing([]) == (None, None)

    def test_the_espn_row_beats_football_datas_own(self):
        espn, fd = _cand("espn", espn_id="747597"), _cand("fd")
        assert pr.choose_existing([fd, espn]) == (espn, None)

    def test_a_results_path_row_counts_as_espn(self):
        espn, fd = _cand("espn", result_source="espn_scoreboard"), _cand("fd")
        assert pr.choose_existing([fd, espn]) == (espn, None)

    def test_a_finished_espn_row_beats_a_stale_scheduled_one(self):
        # Atlas v Monterrey (MX1), prod: a pre-023 fixture row at 01:00Z and the ESPN result at 01:05Z.
        stale, result = _cand("stale", status="scheduled"), _cand("result", result_source="espn_scoreboard")
        assert pr.choose_existing([stale, result]) == (result, None)

    def test_two_level_rows_are_ambiguous(self):
        row, why = pr.choose_existing([_cand("a"), _cand("b")])
        assert row is None and "2 existing rows" in why

    def test_a_single_row_is_it(self):
        assert pr.choose_existing([_cand("only", status="scheduled")])[0]["id"] == "only"


def _result(home=2, away=1, ht_home=None, ht_away=None):
    return pr.RawResult(home, away, ht_home, ht_away)


class TestUpdateExistingMatch:
    """update_existing_match against SQLite, same table as TestInsertMatch."""

    def _existing(self, db, status="finished", ft=(None, None), ht=(None, None), espn=True):
        _existing(db, status, ft[0], ft[1], ht[0], ht[1])
        row = db.execute("SELECT id, match_date FROM matches").fetchone()
        return {
            "id": row[0],
            "match_date": row[1],
            "status": status,
            "home_score": ft[0],
            "away_score": ft[1],
            "half_time_home": ht[0],
            "half_time_away": ht[1],
            "espn_id": "747597" if espn else None,
            "result_source": None,
        }

    def test_a_scheduled_espn_row_gets_the_result(self, db):
        existing = self._existing(db, status="scheduled")
        assert pr.update_existing_match(_SqliteCursor(db), existing, _result(ht_home=1, ht_away=0)) == (
            existing["id"],
            "updated",
        )
        assert _row(db) == ("finished", 2, 1, 1, 0, "espn-season")  # season stays the row's own

    def test_an_espn_rows_score_is_never_overwritten(self, db):
        existing = self._existing(db, ft=(1, 1), ht=(0, 0))
        assert pr.update_existing_match(_SqliteCursor(db), existing, _result(2, 1)) == (None, "score_conflict")
        assert _row(db)[:5] == ("finished", 1, 1, 0, 0)

    def test_an_espn_row_without_half_time_takes_football_datas(self, db):
        existing = self._existing(db, ft=(2, 1))
        assert pr.update_existing_match(_SqliteCursor(db), existing, _result(ht_home=1, ht_away=1))[1] == "updated"
        assert _row(db)[:5] == ("finished", 2, 1, 1, 1)

    def test_an_espn_rows_half_time_is_kept_and_the_conflict_reported(self, db):
        existing = self._existing(db, ft=(2, 1), ht=(0, 0))
        assert pr.update_existing_match(_SqliteCursor(db), existing, _result(ht_home=1, ht_away=1))[1] == (
            "ht_conflict"
        )
        assert _row(db)[:5] == ("finished", 2, 1, 0, 0)

    def test_football_datas_own_row_takes_its_new_score_as_before(self, db):
        existing = self._existing(db, ft=(1, 1), ht=(0, 0), espn=False)
        assert pr.update_existing_match(_SqliteCursor(db), existing, _result(2, 1, 1, 0))[1] == "score_changed"
        assert _row(db)[:5] == ("finished", 2, 1, 1, 0)

    def test_a_half_time_is_only_replaced_by_a_complete_pair(self, db):
        existing = self._existing(db, ft=(2, 1), ht=(1, 0), espn=False)
        assert pr.update_existing_match(_SqliteCursor(db), existing, _result())[1] == "unchanged"  # nothing to write
        assert _row(db)[:5] == ("finished", 2, 1, 1, 0)
        assert pr.update_existing_match(_SqliteCursor(db), existing, _result(3, 1))[1] == "score_changed"
        assert _row(db)[:5] == ("finished", 3, 1, 1, 0)  # written, and the pair is still kept

    def test_a_finished_row_that_already_holds_the_result_is_not_written(self, db):
        """Until 2026-11-01 every run rewrote all ~168k rows (and bumped updated_at) for nothing."""
        for espn, ht in ((True, (1, 0)), (False, (1, 0)), (False, (None, None))):
            db.execute("DELETE FROM matches")
            existing = self._existing(db, ft=(2, 1), ht=ht, espn=espn)
            db.execute("UPDATE matches SET updated_at = 'before'")
            raw_ht = {} if ht == (None, None) else {"ht_home": 1, "ht_away": 0}
            assert pr.update_existing_match(_SqliteCursor(db), existing, _result(**raw_ht)) == (
                existing["id"],
                "unchanged",
            )
            assert db.execute("SELECT updated_at FROM matches").fetchone() == ("before",)

    def test_a_kept_half_time_conflict_is_reported_without_a_write(self, db):
        existing = self._existing(db, ft=(2, 1), ht=(0, 0))
        db.execute("UPDATE matches SET updated_at = 'before'")
        assert pr.update_existing_match(_SqliteCursor(db), existing, _result(ht_home=1, ht_away=1)) == (
            existing["id"],
            "ht_conflict",
        )
        assert db.execute("SELECT updated_at FROM matches").fetchone() == ("before",)

    def test_a_cancelled_row_is_left_alone(self, db):
        existing = self._existing(db, status="cancelled")
        assert pr.update_existing_match(_SqliteCursor(db), existing, _result()) == (None, "cancelled")
        assert _row(db)[0] == "cancelled"

    def test_a_row_that_moved_after_the_lookup_is_not_written(self, db):
        existing = self._existing(db, status="scheduled")
        db.execute("UPDATE matches SET status = 'finished', home_score = 3, away_score = 3")
        assert pr.update_existing_match(_SqliteCursor(db), existing, _result()) == (None, "changed")
        assert _row(db)[:3] == ("finished", 3, 3)


ESPN_ROW = {
    "id": "e77e612a",
    "match_date": datetime(2026, 7, 10, 11, 35, tzinfo=UTC),
    "status": "finished",
    "home_score": 4,
    "away_score": 3,
    "half_time_home": 3,
    "half_time_away": 1,
    "espn_id": None,
    "result_source": "espn_scoreboard",
}
SHANDONG = {
    **NORMAL,
    "league": "CN1",
    "season": "2026",
    "Date": "10/07/2026",
    "Time": "12:35",
    "HomeTeam": "Shandong Taishan",
    "AwayTeam": "Yunnan Yukun",
    "FTHG": "4",
    "FTAG": "3",
    "HTHG": "",
    "HTAG": "",
}


class TestWriteMatch:
    """promote() writes a fixture onto the existing row for its pair."""

    def _promote(self, monkeypatch, promote_with, existing_row, update_returns):
        lookups, updates = [], []

        def existing(cur, home_id, away_id, match_dt, fixture_day):
            lookups.append((match_dt, fixture_day))
            return [existing_row]

        def update(cur, row, result):
            updates.append((row["id"], result))
            return update_returns

        monkeypatch.setattr(pr, "update_existing_match", update)
        stats, inserts = promote_with([SHANDONG], existing=existing)
        return stats, inserts, lookups, updates

    def test_the_prod_duplicate_goes_onto_the_espn_row(self, monkeypatch, promote_with):
        # Shandong Taishan v Yunnan Yukun (CN1): football-data 12:35 UK = 11:35Z, ESPN's row e77e612a.
        stats, inserts, lookups, updates = self._promote(monkeypatch, promote_with, ESPN_ROW, ("e77e612a", "updated"))
        assert lookups == [(datetime(2026, 7, 10, 11, 35, tzinfo=UTC), date(2026, 7, 10))]
        assert updates == [("e77e612a", pr.RawResult(4, 3))] and inserts == []
        assert (stats.matches, stats.merged, stats.refused) == (1, 0, 0)  # same kickoff: not a merge

    def test_a_row_at_another_kickoff_is_counted_as_merged(self, monkeypatch, promote_with):
        moved = {**ESPN_ROW, "match_date": ESPN_ROW["match_date"] + timedelta(minutes=5)}
        stats, inserts, _lookups, _updates = self._promote(monkeypatch, promote_with, moved, ("e77e612a", "updated"))
        assert (stats.matches, stats.merged, inserts) == (1, 1, [])

    def test_a_score_conflict_writes_nothing_and_exits_3(self, monkeypatch, promote_with, caplog):
        with caplog.at_level(logging.ERROR, logger="promote_raw"):
            stats, inserts, _lookups, _updates = self._promote(
                monkeypatch, promote_with, {**ESPN_ROW, "home_score": 1}, (None, "score_conflict")
            )
        assert (stats.matches, stats.refused, inserts) == (0, 1, [])
        assert "is an ESPN row with final score 1-3 (football-data 4-3)" in caplog.text
        monkeypatch.setattr(pr, "promote", lambda url, batch_size: stats)
        assert pr.main(["--database-url", "postgres://fake"]) == pr.REFUSED_EXIT_CODE == 3

    def test_a_half_time_conflict_is_written_and_logged(self, monkeypatch, promote_with, caplog):
        with caplog.at_level(logging.ERROR, logger="promote_raw"):
            stats, _inserts, _lookups, _updates = self._promote(
                monkeypatch, promote_with, ESPN_ROW, ("e77e612a", "ht_conflict")
            )
        assert (stats.matches, stats.ht_conflicts, stats.refused) == (1, 1, 0)
        assert "kept the half-time 3-1" in caplog.text

    def test_an_ambiguous_pair_is_refused(self, promote_with, caplog):
        with caplog.at_level(logging.ERROR, logger="promote_raw"):
            stats, inserts = promote_with(
                [SHANDONG], existing=lambda cur, h, a, dt, day: [dict(ESPN_ROW), {**ESPN_ROW, "id": "other"}]
            )
        assert (stats.matches, stats.refused, inserts) == (0, 1, [])
        assert "merge them first" in caplog.text

    def test_no_existing_row_inserts(self, promote_with):
        stats, inserts = promote_with([SHANDONG])
        assert stats.matches == 1 and inserts[0]["match_dt"] == datetime(2026, 7, 10, 11, 35, tzinfo=UTC)

    def test_a_clean_run_exits_0(self, monkeypatch):
        monkeypatch.setattr(pr, "promote", lambda url, batch_size: pr.Promotion(matches=3))
        assert pr.main(["--database-url", "postgres://fake"]) == 0

    def test_same_kickoff_reads_a_date_only_row_in_the_session_zone(self):
        assert pr.same_kickoff(datetime(2016, 8, 13, tzinfo=UTC), datetime(2016, 8, 13))
        assert not pr.same_kickoff(datetime(2016, 8, 13, 1, tzinfo=UTC), datetime(2016, 8, 13))
        assert pr.same_kickoff(datetime(2026, 7, 10, 13, 35, tzinfo=ZoneInfo("Europe/Paris")), ESPN_ROW["match_date"])


# --- the existing-row lookup against SQLite: the real EXISTING_MATCH_SQL, choose_existing and the writes -----------


def _put(db, when, home="h", away="a", status="finished", score=(None, None), ht=(None, None), espn_id=None, rs=None):
    """A matches row; returns its id. `rs` is metadata.result_source."""
    db.execute(
        "INSERT INTO matches (league_id, home_team_id, away_team_id, match_date, status, home_score, away_score, "
        "half_time_home, half_time_away, season, external_ids, metadata) "
        "VALUES ('lg', ?, ?, ?, ?, ?, ?, ?, ?, 's', ?, ?)",
        (
            home,
            away,
            _sqlite_ts(when),
            status,
            *score,
            *ht,
            json.dumps({"espn": espn_id} if espn_id else {}),
            json.dumps({"result_source": rs} if rs else {}),
        ),
    )
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]


def _rows(db):
    return db.execute(
        "SELECT id, home_team_id, away_team_id, match_date, status, home_score, away_score FROM matches ORDER BY id"
    ).fetchall()


def _fixture(date_, time_, ft=("2", "1"), home="Home", away="Away"):
    teams = {"HomeTeam": home, "AwayTeam": away}
    scores = {"FTHG": ft[0], "FTAG": ft[1], "HTHG": "", "HTAG": ""}
    return {**NORMAL, "league": "MLS", "season": "2026", "Date": date_, "Time": time_, **teams, **scores}


def _write(db, raw, home="h", away="a"):
    stats = pr.Promotion()
    match_id = pr.write_match(
        _SqliteCursor(db),
        stats,
        raw,
        league_id="lg",
        home_id=home,
        away_id=away,
        match_dt=pr.parse_match_date(raw["Date"], raw["Time"]),
        result=pr.raw_result(raw),
    )
    return match_id, stats


def _lookup(db, when, day, home="h", away="a"):
    return pr.find_existing_matches(_SqliteCursor(db), home, away, when, day)


class TestFindExistingMatches:
    """EXISTING_MATCH_SQL itself (the review found every promote() test faked it): team ids, both bounds, the
    UK-date clamp, and which row choose_existing then picks."""

    K = datetime(2026, 7, 10, 11, 35, tzinfo=UTC)

    def test_only_the_same_home_and_away_ids_are_candidates(self, db):
        mine = _put(db, self.K, score=(4, 3))
        _put(db, self.K, home="h", away="other", score=(4, 3))
        _put(db, self.K, home="a", away="h", score=(4, 3))  # the reverse fixture
        assert [r["id"] for r in _lookup(db, self.K, date(2026, 7, 10))] == [mine]

    def test_the_three_hour_bounds_are_inclusive_and_stop_there(self, db):
        inside = [_put(db, self.K - timedelta(hours=3)), _put(db, self.K + timedelta(hours=3), away="a")]
        _put(db, self.K - timedelta(hours=3, seconds=1), home="h", away="a", status="scheduled")
        _put(db, self.K + timedelta(hours=3, minutes=1), home="h", away="a", status="scheduled")
        assert sorted(r["id"] for r in _lookup(db, self.K, date(2026, 7, 10))) == sorted(inside)

    def test_the_espn_row_is_chosen_over_football_datas_own(self, db):
        _put(db, self.K + timedelta(hours=1), score=(4, 3))  # the 1 h late football-data copy
        espn = _put(db, self.K, score=(4, 3), rs="espn_scoreboard")
        row, why = pr.choose_existing(_lookup(db, self.K, date(2026, 7, 10)))
        assert (row["id"], why) == (espn, None)
        assert row["result_source"] == "espn_scoreboard" and row["match_date"] == self.K

    def test_an_espn_competition_id_counts_too(self, db):
        _put(db, self.K + timedelta(hours=1), score=(4, 3))
        espn = _put(db, self.K, score=(4, 3), espn_id="747597")
        row, _why = pr.choose_existing(_lookup(db, self.K, date(2026, 7, 10)))
        assert (row["id"], row["espn_id"]) == (espn, "747597")

    def test_a_legacy_row_on_the_next_uk_date_is_outside_the_window(self, db):
        # 20/07/2026 23:30 UK = 22:30Z; the legacy row sits at 23:30Z, which is 00:30 on the 21st in the UK.
        stale = _put(db, datetime(2026, 7, 20, 22, 30, tzinfo=UTC), status="scheduled")
        _put(db, datetime(2026, 7, 20, 23, 30, tzinfo=UTC), score=(2, 1))
        got = _lookup(db, pr.parse_match_date("20/07/2026", "23:30"), date(2026, 7, 20))
        assert [r["id"] for r in got] == [stale]


class TestStaleFixtureRows:
    """A stale fixture row (unfinished, no ESPN identity: pre-migration 023) is never finished by promote_raw:
    the outcome trigger would grade its predictions and settle its recs (review finding, 2026-10-31)."""

    def test_update_existing_match_refuses_and_writes_nothing(self, db):
        row_id = _put(db, datetime(2026, 7, 20, 22, 30, tzinfo=UTC), status="scheduled")
        (existing,) = _lookup(db, datetime(2026, 7, 20, 22, 30, tzinfo=UTC), date(2026, 7, 20))
        assert pr.update_existing_match(_SqliteCursor(db), existing, _result()) == (None, "stale_fixture")
        assert _rows(db)[0][0] == row_id and _rows(db)[0][4:] == ("scheduled", None, None)

    def test_the_late_evening_bst_case_is_refused_not_finished(self, db, monkeypatch, caplog):
        # The reviewer's reproduction: 20/07/2026 23:30 UK. The window [19:30Z, 22:59:59Z] holds only the stale
        # row at the corrected kickoff (22:30Z); the never-shifted legacy copy (23:30Z) is on the next UK date.
        stale = _put(db, datetime(2026, 7, 20, 22, 30, tzinfo=UTC), status="scheduled")
        copy = _put(db, datetime(2026, 7, 20, 23, 30, tzinfo=UTC), score=(2, 1))
        with caplog.at_level(logging.ERROR, logger="promote_raw"):
            match_id, stats = _write(db, _fixture("20/07/2026", "23:30"))
        assert (match_id, stats.refused, stats.matches) == (None, 1, 0)
        assert "fixture row without ESPN identity" in caplog.text
        assert [(r[0], r[4], r[5]) for r in _rows(db)] == [(stale, "scheduled", None), (copy, "finished", 2)]
        monkeypatch.setattr(pr, "promote", lambda url, batch_size: stats)
        assert pr.main(["--database-url", "postgres://fake"]) == pr.REFUSED_EXIT_CODE

    def test_a_scheduled_espn_row_still_takes_the_result(self, db):
        espn = _put(db, datetime(2026, 7, 20, 22, 30, tzinfo=UTC), status="scheduled", espn_id="747597")
        match_id, stats = _write(db, _fixture("20/07/2026", "23:30"))
        assert (match_id, stats.refused) == (espn, 0) and _rows(db)[0][4:] == ("finished", 2, 1)

    def test_a_finished_football_data_row_beats_a_stale_one(self, db):
        _put(db, datetime(2026, 7, 10, 11, 0, tzinfo=UTC), status="scheduled")
        fd = _put(db, datetime(2026, 7, 10, 11, 35, tzinfo=UTC), score=(2, 1))
        match_id, stats = _write(db, _fixture("10/07/2026", "12:35"))
        assert (match_id, stats.refused) == (fd, 0)


class TestDuplicatesBeside:
    """Writing onto the ESPN row while football-data's own finished row for the pair sits beside it is a duplicate
    in the corpus: logged at ERROR, counted, exit 3 (review finding, 2026-10-31)."""

    def test_written_onto_the_espn_row_and_reported(self, db, monkeypatch, caplog):
        espn = _put(db, datetime(2026, 7, 10, 11, 35, tzinfo=UTC), score=(2, 1), rs="espn_scoreboard")
        copy = _put(db, datetime(2026, 7, 10, 12, 35, tzinfo=UTC), score=(2, 1))
        with caplog.at_level(logging.ERROR, logger="promote_raw"):
            match_id, stats = _write(db, _fixture("10/07/2026", "12:35"))
        assert (match_id, stats.duplicates, stats.refused) == (espn, 1, 0)
        assert f"sit beside it ({copy} at" in caplog.text and "--dedupe" in caplog.text
        monkeypatch.setattr(pr, "promote", lambda url, batch_size: stats)
        assert pr.main(["--database-url", "postgres://fake"]) == pr.REFUSED_EXIT_CODE

    def test_a_stale_row_beside_the_espn_row_is_not_a_duplicate(self, db):
        _put(db, datetime(2026, 7, 10, 11, 0, tzinfo=UTC), status="scheduled")
        espn = _put(db, datetime(2026, 7, 10, 11, 35, tzinfo=UTC), score=(2, 1), rs="espn_scoreboard")
        match_id, stats = _write(db, _fixture("10/07/2026", "12:35"))
        assert (match_id, stats.duplicates) == (espn, 0)

    def test_only_an_espn_row_can_have_duplicates_beside_it(self):
        fd, other = _cand("fd"), _cand("other")
        assert pr.duplicates_beside(fd, [fd, other]) == []
        espn = _cand("espn", espn_id="1")
        assert pr.duplicates_beside(espn, [espn, fd, _cand("stale", status="scheduled")]) == [fd]


# The team merge (scripts/merge_soccer_teams.py, run before the kickoff repair) deletes about 171 football-data rows
# whose ESPN twin survives, re-pointing that twin onto the football-data team ids. The raw rows stay, and every
# promote_raw run re-promotes the whole table: these are the prod pairs whose ESPN kickoff is NOT the corrected
# football-data one (dup_pairs_alias_160, 2026-10-31), where an exact-key upsert would insert the copy again.
AFTER_MERGE = [
    # (raw Date, raw Time, ESPN kickoff, score): Orlando City v Nashville SC, Inter Miami v Columbus Crew, Club Leon v
    # Pachuca; and the usual case, ESPN exactly on the corrected kickoff.
    ("26/07/2026", "00:30", datetime(2026, 7, 26, 0, 46, tzinfo=UTC), ("1", "0")),
    ("02/08/2026", "00:30", datetime(2026, 8, 2, 0, 10, tzinfo=UTC), ("2", "2")),
    ("02/08/2026", "02:00", datetime(2026, 8, 2, 1, 6, tzinfo=UTC), ("1", "0")),
    ("10/07/2026", "12:35", datetime(2026, 7, 10, 11, 35, tzinfo=UTC), ("4", "3")),
]


class TestAfterTheTeamMerge:
    @pytest.mark.parametrize("raw_date, raw_time, espn_kickoff, ft", AFTER_MERGE)
    def test_the_dropped_copy_is_not_inserted_again(self, db, raw_date, raw_time, espn_kickoff, ft):
        score = (int(ft[0]), int(ft[1]))
        espn = _put(db, espn_kickoff, home="old-h", away="old-a", score=score, rs="espn_scoreboard")
        match_id, stats = _write(db, _fixture(raw_date, raw_time, ft), home="old-h", away="old-a")
        assert (match_id, stats.matches, stats.refused, stats.duplicates) == (espn, 0, 0, 0)
        corrected = pr.parse_match_date(raw_date, raw_time)
        assert stats.merged == (corrected != espn_kickoff)
        assert [(r[0], r[3], r[5], r[6]) for r in _rows(db)] == [(espn, _sqlite_ts(espn_kickoff), *score)]

    @pytest.mark.parametrize("raw_date, raw_time, espn_kickoff, ft", AFTER_MERGE)
    def test_promote_raw_is_not_blocked_by_them(self, raw_date, raw_time, espn_kickoff, ft):
        raw = {**_fixture(raw_date, raw_time, ft), "league": "F2", "HomeTeam": "Bastia", "AwayTeam": "Red Star"}
        assert pr.stranded_legacy_kickoffs([raw], LEAGUE_IDS, TEAM_IDS, {_key(espn_kickoff)}, UTC) == []

    def test_an_espn_score_that_disagrees_is_refused_not_duplicated(self, db):
        espn = _put(db, AFTER_MERGE[0][2], home="old-h", away="old-a", score=(0, 0), rs="espn_scoreboard")
        match_id, stats = _write(db, _fixture("26/07/2026", "00:30", ("1", "0")), home="old-h", away="old-a")
        assert (match_id, stats.refused) == (None, 1)
        assert [(r[0], r[5], r[6]) for r in _rows(db)] == [(espn, 0, 0)]


# After repair_football_data_kickoffs --dedupe alias-merged a copy into the ESPN row still filed under the wrong club
# (Monterrey v Santos, the Brazilian club's row; prod 2026-10-31), the raw row (Monterrey v Santos Laguna) has no row
# for its own pair until repoint_team_matches moves that one. Inserting would put the match back in the corpus twice.
MX_KICKOFF = datetime(2026, 7, 19, 1, 5, tzinfo=UTC)  # football-data 19/07/2026 02:05 UK


class TestAliasSlotGuard:
    """write_match never inserts a fixture beside a row for another team pair that shares a club with it in the
    window: no club plays twice in a few hours (review finding, 2026-10-31: the re-point / kickoff-repair deadlock)."""

    def _mx(self, db):
        return _write(db, _fixture("19/07/2026", "02:05", ("3", "2")), home="mon", away="laguna")

    def test_the_alias_merged_fixture_is_not_inserted_again(self, db, monkeypatch, caplog):
        espn = _put(db, MX_KICKOFF, home="mon", away="santos-br", score=(3, 2), rs="espn_scoreboard")
        with caplog.at_level(logging.ERROR, logger="promote_raw"):
            match_id, stats = self._mx(db)
        assert (match_id, stats.refused, stats.matches) == (None, 1, 0)
        assert f"{espn} at" in caplog.text and "under other team ids" in caplog.text
        assert [r[0] for r in _rows(db)] == [espn]
        monkeypatch.setattr(pr, "promote", lambda url, batch_size: stats)
        assert pr.main(["--database-url", "postgres://fake"]) == pr.REFUSED_EXIT_CODE

    def test_a_shared_away_club_or_the_reverse_fixture_counts_too(self, db):
        _put(db, MX_KICKOFF + timedelta(hours=2), home="other", away="laguna", score=(0, 0))
        assert self._mx(db)[1].refused == 1
        db.execute("DELETE FROM matches")
        _put(db, MX_KICKOFF, home="laguna", away="mon", score=(2, 3))
        assert self._mx(db)[1].refused == 1

    def test_another_fixture_at_the_same_kickoff_is_no_alias(self, db):
        _put(db, MX_KICKOFF, home="x", away="y", score=(3, 2), rs="espn_scoreboard")
        match_id, stats = self._mx(db)
        assert (stats.refused, stats.matches) == (0, 0) and match_id is not None and len(_rows(db)) == 2

    def test_a_club_s_match_outside_the_window_is_no_alias(self, db):
        _put(db, MX_KICKOFF - timedelta(hours=3, minutes=1), home="mon", away="z", score=(1, 1))
        match_id, stats = self._mx(db)
        assert (stats.refused, match_id is not None, len(_rows(db))) == (0, True, 2)

    def test_once_re_pointed_it_writes_onto_the_row(self, db):
        espn = _put(db, MX_KICKOFF, home="mon", away="laguna", score=(3, 2), rs="espn_scoreboard")
        _put(db, MX_KICKOFF - timedelta(minutes=5), home="mon", away="laguna", status="scheduled")  # moved with it
        match_id, stats = self._mx(db)
        assert (match_id, stats.refused, stats.duplicates, len(_rows(db))) == (espn, 0, 0, 2)

    def test_the_query_returns_other_pairs_sharing_a_club_and_never_the_pair_itself(self, db):
        own = _put(db, MX_KICKOFF, home="mon", away="laguna", score=(3, 2))
        alias = _put(db, MX_KICKOFF, home="mon", away="santos-br", score=(3, 2), rs="espn_scoreboard")
        _put(db, MX_KICKOFF, home="x", away="y")
        got = pr.find_alias_slot_rows(_SqliteCursor(db), "mon", "laguna", MX_KICKOFF, date(2026, 7, 19))
        assert [r["id"] for r in got] == [alias] and own not in [r["id"] for r in got]

    def test_the_guard_is_not_asked_when_the_pair_has_its_row(self, monkeypatch, promote_with):
        asked: list = []
        monkeypatch.setattr(pr, "update_existing_match", lambda cur, row, result: ("e77e612a", "updated"))
        stats, inserts = promote_with(
            [SHANDONG],
            existing=lambda cur, h, a, dt, day: [dict(ESPN_ROW)],
            aliases=lambda cur, h, a, dt, day: asked.append(1) or [{"id": "x"}],
        )
        assert (asked, inserts, stats.matches, stats.refused) == ([], [], 1, 0)

    def test_promote_refuses_through_the_guard(self, promote_with):
        alias = {"id": "e78e7e49", "match_date": MX_KICKOFF, "status": "finished", "home_score": 3, "away_score": 2}
        stats, inserts = promote_with([SHANDONG], aliases=lambda cur, h, a, dt, day: [alias])
        assert (stats.matches, stats.refused, inserts) == (0, 1, [])


class TestCheckedMode:
    """--dry-run / --apply --expect-plan: one transaction, what it changed read back (read_changes), rolled back
    unless the apply's own plan equals its pin. The real run against a replica of prod: 4 match_stats and 13 odds
    inserted, everything else 0 (scripts/ops/promote_raw_after_repair.sh header)."""

    PIN = {k: 0 for k in pr.PLAN_KEYS} | {"match_stats_inserted": 4, "odds_inserted": 13}

    def test_expect_plan_needs_every_key_once(self):
        text = ",".join(f"{k}={v}" for k, v in self.PIN.items())
        assert pr.parse_expect_plan(text) == self.PIN
        for bad in (text.replace(",duplicates=0", ""), text + ",bogus=1", text.replace("refused=0", "refused=x")):
            with pytest.raises(pr.PlanRefused):
                pr.parse_expect_plan(bad)

    class _Cur:
        def __init__(self, found):
            self.found, self.sql = found, None

        def execute(self, sql, params=None):
            self.sql = sql

        def fetchall(self):
            for key, _table, sql in pr.CHECKED_CHANGES_SQL:
                if sql == self.sql:
                    return self.found.get(key, [])
            return []

    def test_read_changes_counts_every_query_and_the_refusals(self):
        found = {
            "match_stats_inserted": [{"key": f"m{i}/t", "old": "", "new": ""} for i in range(4)],
            "odds_inserted": [{"key": f"o{i}", "old": "", "new": "x"} for i in range(13)],
        }
        stats = pr.Promotion(refused=2, duplicates=1)
        counts, rows = pr.read_changes(self._Cur(found), stats)
        assert pr.plan_line(counts) == (
            "PLAN matches_inserted=0 matches_changed=0 match_stats_inserted=4 match_stats_changed=0 odds_inserted=13 "
            "teams_inserted=0 teams_changed=0 leagues_inserted=0 leagues_changed=0 predictions_changed=0 "
            "recs_changed=0 refused=2 duplicates=1"
        )
        assert len(rows) == 17 and rows[0] == {
            "plan_key": "match_stats_inserted",
            "table": "match_stats",
            "key": "m0/t",
            "old": "",
            "new": "",
        }

    class _Conn:
        def __init__(self):
            self.isolation, self.commits, self.rollbacks, self.closed = None, 0, 0, False

        def set_session(self, isolation_level=None):
            self.isolation = isolation_level

        def cursor(self, cursor_factory=None):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=None):
            pass

        def commit(self):
            self.commits += 1

        def rollback(self):
            self.rollbacks += 1

        def close(self):
            self.closed = True

    @pytest.fixture
    def checked(self, monkeypatch):
        conn = self._Conn()
        calls = []
        monkeypatch.setattr(pr.psycopg2, "connect", lambda url: conn)
        monkeypatch.setattr(pr, "promote_rows", lambda c, cur, stats, batch_size: calls.append(batch_size))
        monkeypatch.setattr(pr, "read_changes", lambda cur, stats: (Counter(self.PIN), []))
        return conn, calls

    def test_a_dry_run_is_one_repeatable_read_transaction_rolled_back(self, checked, tmp_path):
        conn, calls = checked
        _stats, counts = pr.promote_checked("x", apply=False, expected=None, out_csv=str(tmp_path / "d.csv"))
        assert calls == [None]  # no batch commits
        assert conn.isolation == "REPEATABLE READ" and (conn.commits, conn.rollbacks, conn.closed) == (0, 1, True)
        assert (tmp_path / "d.csv").read_text().startswith("plan_key,table,key,old,new")

    def test_an_apply_commits_only_at_its_pin(self, checked, tmp_path):
        conn, _ = checked
        pr.promote_checked("x", apply=True, expected=self.PIN, out_csv=str(tmp_path / "a.csv"))
        assert (conn.commits, conn.rollbacks) == (1, 0)

    def test_an_apply_that_drifted_rolls_back(self, checked, tmp_path):
        conn, _ = checked
        with pytest.raises(pr.PlanRefused, match="drifted"):
            pr.promote_checked("x", apply=True, expected={**self.PIN, "odds_inserted": 12}, out_csv=str(tmp_path / "a"))
        assert (conn.commits, conn.rollbacks, conn.closed) == (0, 1, True)

    def test_the_audit_csv_is_write_once(self, checked, tmp_path):
        conn, _ = checked
        (tmp_path / "d.csv").write_text("earlier\n")
        with pytest.raises(pr.PlanRefused, match="write-once"):
            pr.promote_checked("x", apply=False, expected=None, out_csv=str(tmp_path / "d.csv"))
        assert conn.commits == 0 and conn.closed

    @pytest.mark.parametrize(
        "argv",
        [
            ["--dry-run"],  # no --out-csv
            ["--apply", "--out-csv", "a.csv"],  # no pin
            ["--expect-plan", "refused=0", "--out-csv", "a.csv"],  # a pin without --apply
        ],
    )
    def test_main_refuses_an_incomplete_checked_command(self, argv, monkeypatch):
        monkeypatch.setattr(pr, "promote_checked", lambda *a, **k: pytest.fail("must not run"))
        assert pr.main(["--database-url", "x", *argv]) == 2

    def test_main_checked_dry_run_exits_3_on_a_refused_row(self, monkeypatch):
        monkeypatch.setattr(pr, "promote_checked", lambda *a, **k: (pr.Promotion(refused=1), Counter()))
        assert pr.main(["--database-url", "x", "--dry-run", "--out-csv", "d.csv"]) == pr.REFUSED_EXIT_CODE

    def test_main_checked_reports_a_legacy_kickoff_refusal(self, monkeypatch, caplog):
        def refuse(*a, **k):
            raise pr.LegacyKickoffsPresent("3 raw row(s) still have their legacy matches row")

        monkeypatch.setattr(pr, "promote_checked", refuse)
        assert pr.main(["--database-url", "x", "--dry-run", "--out-csv", "d.csv"]) == 1
        assert "Refusing to promote — nothing written" in caplog.text


# ── a re-run sends only what changes: unchanged leagues, teams and match_stats are neither rewritten nor locked ──


class _LeagueCursor:
    """The leagues table as upsert_leagues reads it; execute_values is captured by the test."""

    def __init__(self, leagues):
        self.leagues = leagues  # [(name, country, code)]
        self._rows: list[dict] = []

    def execute(self, sql, params=None):
        q = " ".join(sql.split())
        if q == " ".join(pr.LEAGUE_CODES_SQL.split()):
            self._rows = [{"name": n, "country": c, "code": code} for n, c, code in self.leagues]
        elif q.startswith("SELECT id, external_ids->>'football_data' AS code FROM leagues"):
            self._rows = [{"id": f"lg-{code}", "code": code} for _n, _c, code in self.leagues if code]
        else:
            raise AssertionError(f"unexpected SQL: {q}")

    def fetchall(self):
        return list(self._rows)


class TestUpsertLeagues:
    @pytest.fixture
    def sent(self, monkeypatch):
        rows: list[tuple] = []

        def fake_execute_values(cur, sql, argslist):
            assert "WHERE leagues.external_ids IS DISTINCT FROM" in " ".join(sql.split())
            for name, country, sport, ext in argslist:
                rows.append((name, country, ext.adapted))
                cur.leagues.append((name, country, ext.adapted["football_data"]))

        monkeypatch.setattr(pr, "execute_values", fake_execute_values)
        return rows

    def test_a_league_that_already_carries_its_code_is_not_sent(self, sent):
        cur = _LeagueCursor([("Premier League", "England", "E0"), ("Premiership", "Scotland", "SC0")])
        assert pr.upsert_leagues(cur, {"E0", "SC0"}) == {"E0": "lg-E0", "SC0": "lg-SC0"}
        assert sent == []

    def test_a_missing_league_or_code_is_sent_and_mapped(self, sent):
        cur = _LeagueCursor([("Premier League", "England", None), ("Premiership", "Scotland", "SC0")])
        ids = pr.upsert_leagues(cur, {"E0", "E1", "SC0"})
        assert sent == [
            ("Premier League", "England", {"football_data": "E0"}),
            ("Championship", "England", {"football_data": "E1"}),
        ]
        assert ids["E0"] == "lg-E0" and ids["E1"] == "lg-E1" and ids["SC0"] == "lg-SC0"

    def test_unknown_codes_only_is_empty_without_a_query(self, sent):
        assert pr.upsert_leagues(_LeagueCursor([]), {"ZZ9"}) == {} and sent == []


class TestInsertMatchStatsSql:
    """The statement is checked against Postgres in the 2026-11-01 rehearsal; here its two guards are pinned."""

    SQL = " ".join(pr.INSERT_MATCH_STATS_SQL.split())
    COLS = "shots, {p}shots_on_target, {p}corners, {p}fouls, {p}yellow_cards, {p}red_cards"

    def test_an_equal_row_is_not_inserted_so_it_never_conflicts(self):
        cols = self.COLS.format(p="s.")
        vals = self.COLS.format(p="v.")
        assert "WHERE NOT EXISTS ( SELECT 1 FROM match_stats s WHERE s.match_id = v.match_id" in self.SQL
        assert f"AND (s.{cols}) IS NOT DISTINCT FROM (v.{vals}) )" in self.SQL

    def test_the_conflict_branch_updates_only_a_differing_row(self):
        old = self.COLS.format(p="match_stats.")
        new = self.COLS.format(p="EXCLUDED.")
        assert "ON CONFLICT (match_id, team_id) DO UPDATE SET shots = EXCLUDED.shots," in self.SQL
        assert self.SQL.endswith(f"WHERE (match_stats.{old}) IS DISTINCT FROM (EXCLUDED.{new})")

    def test_it_passes_the_eight_values_once(self):
        seen = []

        class _Cur:
            def execute(self, sql, params):
                seen.append((sql, params))

        pr.insert_match_stats(
            _Cur(), "m", "t", shots=1, shots_on_target=2, corners=3, fouls=4, yellow_cards=5, red_cards=None
        )
        ((sql, params),) = seen
        assert sql is pr.INSERT_MATCH_STATS_SQL and params == ("m", "t", 1, 2, 3, 4, 5, None)
        assert sql.count("%s") == 8
