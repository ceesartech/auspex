"""backfill_season_labels: which matches rows get which season label.
Pure planning over in-memory rows — no DB."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
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


bsl = _load("backfill_season_labels", "backfill_season_labels.py")

UTC = ZoneInfo("UTC")
LEAGUES = {"BR1": "lg-br1", "AT1": "lg-at1", "E0": "lg-e0"}
TEAMS = {"chapecoense-sc": "t-chape", "santos": "t-santos", "rapid vienna": "t-rapid", "salzburg": "t-salz"}
TEAMS.update({"arsenal": "t-ars", "chelsea": "t-che"})


def _raw(league, season, home, away, date="11/12/2016", time="19:00"):
    return {"league": league, "season": season, "Date": date, "Time": time, "HomeTeam": home, "AwayTeam": away}


def _match(match_id, league_id, home_id, away_id, kickoff, season):
    return {
        "id": match_id,
        "league_id": league_id,
        "home_id": home_id,
        "away_id": away_id,
        "match_date": kickoff,
        "season": season,
    }


KICKOFF = datetime(2016, 12, 11, 19, 0, tzinfo=timezone.utc)


def _plan(raw_rows, matches):
    counts = Counter()
    by_key = {(m["home_id"], m["away_id"], m["match_date"]): m for m in matches}
    return bsl.plan(raw_rows, LEAGUES, TEAMS, by_key, UTC, counts), counts


class TestLegacyLabel:
    def test_reproduces_the_old_bug(self):
        assert bsl.legacy_season_label("2016") == "2020-2116"
        assert bsl.legacy_season_label("2021") == "2020-2021"
        assert bsl.legacy_season_label("1617") == "2016-2017"
        assert bsl.legacy_season_label("2016/2017") == "2016/2017"


class TestPlan:
    def test_calendar_year_row_is_relabelled(self):
        entries, _ = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2020-2116")],
        )
        assert [(e["match_id"], e["old_season"], e["new_season"]) for e in entries] == [("m1", "2020-2116", "2016")]

    def test_2021_that_looked_right_is_relabelled(self):
        kickoff = datetime(2021, 6, 6, 19, 0, tzinfo=timezone.utc)
        entries, _ = _plan(
            [_raw("BR1", "2021", "Chapecoense-SC", "Santos", date="06/06/2021")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", kickoff, "2020-2021")],
        )
        assert [(e["old_season"], e["new_season"]) for e in entries] == [("2020-2021", "2021")]

    def test_cross_year_row_gets_the_dash_spelling(self):
        kickoff = datetime(2016, 12, 11, 19, 0, tzinfo=timezone.utc)
        entries, _ = _plan(
            [_raw("AT1", "2016/2017", "Rapid Vienna", "Salzburg")],
            [_match("m1", "lg-at1", "t-rapid", "t-salz", kickoff, "2016/2017")],
        )
        assert [(e["old_season"], e["new_season"]) for e in entries] == [("2016/2017", "2016-2017")]

    def test_european_rows_are_untouched(self):
        entries, counts = _plan(
            [_raw("E0", "1617", "Arsenal", "Chelsea")],
            [_match("m1", "lg-e0", "t-ars", "t-che", KICKOFF, "2016-2017")],
        )
        assert entries == [] and counts["label_unchanged"] == 1

    def test_row_the_espn_path_labelled_is_left_alone(self):
        entries, counts = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2016-2017")],
        )
        assert entries == [] and counts["season_not_from_this_raw_row"] == 1

    def test_already_correct_row_is_counted(self):
        entries, counts = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2016")],
        )
        assert entries == [] and counts["already_correct"] == 1

    @pytest.mark.parametrize(
        "raw, counter",
        [
            (_raw("BR1", "2016", "Chapecoense-SC", "Santos", time="20:00"), "no_matches_row"),
            (_raw("BR1", "2016", "Unknown FC", "Santos"), "raw_row_unresolved"),
            (_raw("XX9", "2016", "Chapecoense-SC", "Santos"), "raw_row_unresolved"),
        ],
    )
    def test_unmatched_raw_rows_change_nothing(self, raw, counter):
        entries, counts = _plan([raw], [_match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2020-2116")])
        assert entries == [] and counts[counter] == 1

    def test_row_in_another_league_is_left_alone(self):
        entries, counts = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos")],
            [_match("m1", "lg-at1", "t-chape", "t-santos", KICKOFF, "2020-2116")],
        )
        assert entries == [] and counts["league_mismatch"] == 1

    def test_raw_duplicates_plan_one_write(self):
        raw = _raw("BR1", "2016", "Chapecoense-SC", "Santos")
        entries, _ = _plan([raw, dict(raw)], [_match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2020-2116")])
        assert len(entries) == 1

    def test_kickoff_is_read_in_the_database_timezone(self):
        # A naive 19:00 in a UTC+1 session is 18:00 UTC.
        kickoff = datetime(2016, 12, 11, 18, 0, tzinfo=timezone.utc)
        counts = Counter()
        by_key = {("t-chape", "t-santos", kickoff): _match("m1", "lg-br1", "t-chape", "t-santos", kickoff, "2020-2116")}
        entries = bsl.plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos")],
            LEAGUES,
            TEAMS,
            by_key,
            ZoneInfo("Europe/Paris"),
            counts,
        )
        assert [e["match_id"] for e in entries] == ["m1"]

    def test_a_summer_row_is_found_where_promote_raw_wrote_it(self):
        # promote_raw stored the UK wall clock (19:00) as 19:00Z until
        # 2026-10-31; parse_match_date now reads it as 18:00Z. The row still
        # sits at the legacy kickoff, so that is where the backfill looks first.
        legacy = datetime(2016, 7, 10, 19, 0, tzinfo=timezone.utc)
        entries, _ = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos", date="10/07/2016")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", legacy, "2020-2116")],
        )
        assert [(e["match_id"], e["new_season"]) for e in entries] == [("m1", "2016")]

    def test_a_summer_row_already_moved_to_utc_is_found_too(self):
        # The operator's UK-time shift may run before this backfill.
        shifted = datetime(2016, 7, 10, 18, 0, tzinfo=timezone.utc)
        entries, _ = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos", date="10/07/2016")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", shifted, "2020-2116")],
        )
        assert [(e["match_id"], e["new_season"]) for e in entries] == [("m1", "2016")]


class _Cursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Conn:
    def __init__(self):
        self.committed = 0

    def cursor(self, cursor_factory=None):
        return _Cursor(self)

    def commit(self):
        self.committed += 1

    def rollback(self):
        pass


class TestRun:
    @pytest.fixture
    def inputs(self, monkeypatch):
        raw = [_raw("BR1", "2016", "Chapecoense-SC", "Santos")]
        matches = {
            ("t-chape", "t-santos", KICKOFF): _match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2020-2116")
        }
        monkeypatch.setattr(bsl, "load_inputs", lambda cur: (raw, LEAGUES, TEAMS, matches, UTC))

    def test_dry_run_writes_only_the_csv(self, inputs, tmp_path, monkeypatch):
        monkeypatch.setattr(bsl, "execute_values", lambda *a, **kw: pytest.fail("dry run wrote"))
        counts = bsl.run(_Conn(), apply_changes=False, out_csv=str(tmp_path / "s.csv"))
        assert counts["planned"] == 1
        assert "m1,BR1" in (tmp_path / "s.csv").read_text()

    def test_apply_counts_rows_that_moved(self, inputs, tmp_path, monkeypatch):
        seen = {}

        def fake_execute_values(cur, sql, rows, page_size, fetch):
            seen["rows"] = rows
            return []  # the guard matched nothing: the row changed since the plan

        monkeypatch.setattr(bsl, "execute_values", fake_execute_values)
        conn = _Conn()
        counts = bsl.run(conn, apply_changes=True, out_csv=str(tmp_path / "s.csv"))
        assert seen["rows"] == [("m1", "2020-2116", "2016")]
        assert (counts["relabelled"], counts["changed_since_plan"], conn.committed) == (0, 1, 1)

    def test_audit_csv_is_write_once(self, inputs, tmp_path):
        out = tmp_path / "s.csv"
        out.write_text("x")
        with pytest.raises(FileExistsError):
            bsl.run(_Conn(), apply_changes=True, out_csv=str(out))
