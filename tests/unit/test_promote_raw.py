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

    def run(rows, insert_returns="m-1"):
        monkeypatch.setattr(pr.psycopg2, "connect", lambda url: _FakeConn(rows))
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
