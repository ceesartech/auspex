"""backfill_season_labels: which matches rows get which season label.
Pure planning over in-memory rows, plus run()/verify() over a stub
connection — no DB (the ops wrapper was rehearsed on a local replica)."""

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
CODES = {
    "lg-br1": "BR1",
    "lg-at1": "AT1",
    "lg-e0": "E0",
    "lg-mls": "MLS",
    "lg-cl1": "CL1",
    "lg-jp1": "JP1",
    "lg-mx1": "MX1",
    "lg-ar1": "AR1",
}
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


def _plan(raw_rows, matches, db_tz=UTC):
    counts = Counter()
    return bsl.plan(raw_rows, CODES, TEAMS, matches, db_tz, counts), counts


def _moves(entries):
    return [(e["match_id"], e["kind"], e["old_season"], e["new_season"]) for e in entries]


class TestLabels:
    def test_legacy_label_reproduces_the_old_bug(self):
        assert bsl.legacy_season_label("2016") == "2020-2116"
        assert bsl.legacy_season_label("2021") == "2020-2021"
        assert bsl.legacy_season_label("1617") == "2016-2017"
        assert bsl.legacy_season_label("2016/2017") == "2016/2017"

    @pytest.mark.parametrize(
        "label, bad",
        [
            ("2020-2116", True),
            ("2016/2017", True),
            ("2020-2021", False),  # looks right: only a raw tie can tell
            ("2016", False),
            ("2016-2017", False),
            ("", False),
            (None, False),
        ],
    )
    def test_bad_shape(self, label, bad):
        assert bsl.is_bad_shape(label) is bad


class TestRawTiedRows:
    def test_calendar_year_row_is_relabelled(self):
        entries, counts = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2020-2116")],
        )
        assert _moves(entries) == [("m1", "promote_raw", "2020-2116", "2016")]
        assert counts["relabel_bad_shape"] == 1 and entries[0]["raw_season"] == "2016"

    def test_2021_that_looked_right_is_relabelled(self):
        kickoff = datetime(2021, 6, 6, 19, 0, tzinfo=timezone.utc)
        entries, counts = _plan(
            [_raw("BR1", "2021", "Chapecoense-SC", "Santos", date="06/06/2021")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", kickoff, "2020-2021")],
        )
        assert _moves(entries) == [("m1", "promote_raw", "2020-2021", "2021")]
        assert counts["relabel_bad_shape"] == 0

    def test_cross_year_row_gets_the_dash_spelling(self):
        entries, _ = _plan(
            [_raw("AT1", "2016/2017", "Rapid Vienna", "Salzburg")],
            [_match("m1", "lg-at1", "t-rapid", "t-salz", KICKOFF, "2016/2017")],
        )
        assert _moves(entries) == [("m1", "promote_raw", "2016/2017", "2016-2017")]

    def test_european_rows_are_untouched(self):
        entries, counts = _plan(
            [_raw("E0", "1617", "Arsenal", "Chelsea")],
            [_match("m1", "lg-e0", "t-ars", "t-che", KICKOFF, "2016-2017")],
        )
        assert entries == [] and counts["already_correct"] == 1

    def test_an_espn_label_on_a_tied_calendar_row_is_relabelled(self):
        # ESPN created the row ('2016-2017', Aug->May) before promote_raw met it.
        entries, _ = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2016-2017")],
        )
        assert _moves(entries) == [("m1", "espn", "2016-2017", "2016")]

    def test_football_data_names_the_season_even_where_no_date_rule_does(self):
        # CSL 2021 ran to 2022-01-12: the raw season wins over the date.
        kickoff = datetime(2022, 1, 12, 11, 30, tzinfo=timezone.utc)
        codes = {**CODES, "lg-cn1": "CN1"}
        counts = Counter()
        entries = bsl.plan(
            [_raw("CN1", "2021", "Chapecoense-SC", "Santos", date="12/01/2022", time="11:30")],
            codes,
            TEAMS,
            [_match("m1", "lg-cn1", "t-chape", "t-santos", kickoff, "2020-2021")],
            UTC,
            counts,
        )
        assert _moves(entries) == [("m1", "promote_raw", "2020-2021", "2021")]

    def test_a_label_no_known_writer_produced_is_left_alone(self):
        entries, counts = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2015-2016")],
        )
        assert entries == [] and counts["left_unknown_writer"] == 1

    def test_already_correct_row_is_counted(self):
        entries, counts = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2016")],
        )
        assert entries == [] and counts["already_correct"] == 1

    @pytest.mark.parametrize(
        "raw, counter",
        [
            (_raw("BR1", "2016", "Chapecoense-SC", "Santos", time="20:00"), "raw_no_matches_row"),
            (_raw("BR1", "2016", "Unknown FC", "Santos"), "raw_row_unresolved"),
            (_raw("XX9", "2016", "Chapecoense-SC", "Santos"), "raw_row_unresolved"),
        ],
    )
    def test_unmatched_raw_rows_tie_nothing(self, raw, counter):
        # The row is then judged by its date: BR1 2016 has a date rule, but
        # '2020-2116' is no Aug->May label, so it is left alone.
        entries, counts = _plan([raw], [_match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2020-2116")])
        assert entries == [] and counts[counter] == 1 and counts["left_unknown_writer"] == 1

    def test_a_tie_to_another_leagues_raw_row_is_left_alone(self):
        entries, counts = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos")],
            [_match("m1", "lg-at1", "t-chape", "t-santos", KICKOFF, "2016/2017")],
        )
        assert entries == [] and counts["left_league_mismatch"] == 1

    def test_raw_duplicates_that_agree_plan_one_write(self):
        raw = _raw("BR1", "2016", "Chapecoense-SC", "Santos")
        entries, _ = _plan([raw, dict(raw)], [_match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2020-2116")])
        assert len(entries) == 1

    def test_raw_duplicates_that_disagree_leave_the_row_alone(self):
        raws = [_raw("BR1", "2016", "Chapecoense-SC", "Santos"), _raw("BR1", "2017", "Chapecoense-SC", "Santos")]
        entries, counts = _plan(raws, [_match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2020-2116")])
        assert entries == [] and counts["left_raw_disagree"] == 1

    def test_kickoff_is_read_in_the_database_timezone(self):
        # A naive 19:00 in a UTC+1 session is 18:00 UTC.
        kickoff = datetime(2016, 12, 11, 18, 0, tzinfo=timezone.utc)
        entries, _ = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", kickoff, "2020-2116")],
            db_tz=ZoneInfo("Europe/Paris"),
        )
        assert [e["match_id"] for e in entries] == ["m1"]

    def test_a_summer_row_is_found_where_promote_raw_wrote_it(self):
        # promote_raw stored the UK wall clock (19:00) as 19:00Z until
        # 2026-10-31; parse_match_date now reads it as 18:00Z.
        legacy = datetime(2016, 7, 10, 19, 0, tzinfo=timezone.utc)
        entries, _ = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos", date="10/07/2016")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", legacy, "2020-2116")],
        )
        assert [(e["match_id"], e["new_season"]) for e in entries] == [("m1", "2016")]

    def test_a_summer_row_already_moved_to_utc_is_found_too(self):
        shifted = datetime(2016, 7, 10, 18, 0, tzinfo=timezone.utc)
        entries, _ = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos", date="10/07/2016")],
            [_match("m1", "lg-br1", "t-chape", "t-santos", shifted, "2020-2116")],
        )
        assert [(e["match_id"], e["new_season"]) for e in entries] == [("m1", "2016")]

    def test_both_rows_of_a_one_hour_duplicate_take_the_raw_season(self):
        # football-data's row 1 h late beside the ESPN row of the same fixture.
        legacy = datetime(2016, 7, 10, 19, 0, tzinfo=timezone.utc)
        espn = datetime(2016, 7, 10, 18, 0, tzinfo=timezone.utc)
        entries, _ = _plan(
            [_raw("BR1", "2016", "Chapecoense-SC", "Santos", date="10/07/2016")],
            [
                _match("fd", "lg-br1", "t-chape", "t-santos", legacy, "2020-2116"),
                _match("es", "lg-br1", "t-chape", "t-santos", espn, "2016-2017"),
            ],
        )
        assert sorted(_moves(entries)) == [
            ("es", "espn", "2016-2017", "2016"),
            ("fd", "promote_raw", "2020-2116", "2016"),
        ]


class TestRowsWithoutARawRow:
    """The ESPN-created rows: judged by soccer_seasons for their league and kickoff."""

    @pytest.mark.parametrize(
        "league, kickoff, old, new",
        [
            ("lg-mls", datetime(2026, 9, 13, 23, 30, tzinfo=timezone.utc), "2026-2027", "2026"),
            ("lg-mls", datetime(2026, 3, 1, 1, 0, tzinfo=timezone.utc), "2025-2026", "2026"),
            ("lg-cl1", datetime(2026, 10, 4, 21, 0, tzinfo=timezone.utc), "2026-2027", "2026"),
            ("lg-jp1", datetime(2026, 4, 11, 5, 0, tzinfo=timezone.utc), "2025-2026", "2026"),
        ],
    )
    def test_calendar_year_leagues_lose_the_aug_may_label(self, league, kickoff, old, new):
        entries, _ = _plan([], [_match("m1", league, "t-chape", "t-santos", kickoff, old)])
        assert _moves(entries) == [("m1", "espn", old, new)]

    @pytest.mark.parametrize(
        "league, kickoff, label",
        [
            ("lg-jp1", datetime(2026, 10, 24, 5, 0, tzinfo=timezone.utc), "2026-2027"),  # J1 is Aug-May now
            ("lg-mx1", datetime(2026, 10, 4, 2, 0, tzinfo=timezone.utc), "2026-2027"),
            ("lg-e0", datetime(2026, 10, 4, 14, 0, tzinfo=timezone.utc), "2026-2027"),
        ],
    )
    def test_cross_year_labels_stay(self, league, kickoff, label):
        entries, counts = _plan([], [_match("m1", league, "t-chape", "t-santos", kickoff, label)])
        assert entries == [] and counts["already_correct"] == 1

    def test_an_irregular_era_is_left_alone(self):
        kickoff = datetime(2019, 10, 1, tzinfo=timezone.utc)
        entries, counts = _plan([], [_match("m1", "lg-ar1", "t-chape", "t-santos", kickoff, "2019-2020")])
        assert entries == [] and counts["left_no_date_rule"] == 1

    def test_another_label_is_left_alone(self):
        kickoff = datetime(2026, 9, 13, tzinfo=timezone.utc)
        entries, counts = _plan([], [_match("m1", "lg-mls", "t-chape", "t-santos", kickoff, "2025")])
        assert entries == [] and counts["left_unknown_writer"] == 1

    def test_a_league_without_a_code_is_left_alone(self):
        entries, counts = _plan([], [_match("m1", "lg-unknown", "t-chape", "t-santos", KICKOFF, "2016-2017")])
        assert entries == [] and counts["left_no_league_code"] == 1


class TestPlanLines:
    def test_plan_line_lists_every_counter(self):
        line = bsl.plan_line(Counter(relabel=3, relabel_espn=1, relabel_promote_raw=2))
        assert line.startswith("PLAN relabel=3 relabel_promote_raw=2 relabel_espn=1 relabel_bad_shape=0 ")
        assert all(f" {k}=" in line for k in bsl.PLAN_COUNTERS)

    def test_plan_leagues_line(self):
        entries = [{"league": "MLS", "kind": "espn"}, {"league": "AR1", "kind": "espn"}]
        assert bsl.plan_leagues_line(entries) == "PLAN_LEAGUES promote_raw=- espn=AR1,MLS"

    def test_parse_expect_plan(self):
        assert bsl.parse_expect_plan("relabel_promote_raw=5,relabel_espn=0") == {
            "relabel_promote_raw": 5,
            "relabel_espn": 0,
        }
        for bad in ("relabel=x", "nope=1", "relabel_espn", ""):
            with pytest.raises(ValueError):
                bsl.parse_expect_plan(bad)


class _Cursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append((sql, params))

    def fetchall(self):
        return self.conn.rows


class _Conn:
    def __init__(self, rows=()):
        self.committed = 0
        self.rolled_back = 0
        self.executed: list = []
        self.rows = list(rows)

    def cursor(self, cursor_factory=None):
        return _Cursor(self)

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1


class TestRun:
    @pytest.fixture
    def inputs(self, monkeypatch):
        raw = [_raw("BR1", "2016", "Chapecoense-SC", "Santos")]
        matches = [_match("m1", "lg-br1", "t-chape", "t-santos", KICKOFF, "2020-2116")]
        monkeypatch.setattr(bsl, "load_inputs", lambda cur: (raw, CODES, TEAMS, matches, UTC))

    def test_dry_run_writes_only_the_csv_and_prints_the_plan(self, inputs, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(bsl, "execute_values", lambda *a, **kw: pytest.fail("dry run wrote"))
        counts = bsl.run(_Conn(), apply_changes=False, out_csv=str(tmp_path / "s.csv"))
        assert counts["relabel"] == 1
        assert "m1,BR1" in (tmp_path / "s.csv").read_text()
        out = capsys.readouterr().out
        assert "PLAN relabel=1 relabel_promote_raw=1 relabel_espn=0 relabel_bad_shape=1 " in out
        assert "PLAN_LEAGUES promote_raw=BR1 espn=-" in out

    def test_apply_commits_what_it_planned(self, inputs, tmp_path, monkeypatch):
        seen = {}

        def fake_execute_values(cur, sql, rows, page_size, fetch):
            seen["rows"] = rows
            return [("m1",)]

        monkeypatch.setattr(bsl, "execute_values", fake_execute_values)
        conn = _Conn()
        counts = bsl.run(conn, apply_changes=True, out_csv=str(tmp_path / "s.csv"))
        assert seen["rows"] == [("m1", "2020-2116", "2016")]
        assert (counts["relabelled"], conn.committed) == (1, 1)

    def test_a_row_changed_since_the_plan_rolls_everything_back(self, inputs, tmp_path, monkeypatch):
        monkeypatch.setattr(bsl, "execute_values", lambda *a, **kw: [])  # the guard matched nothing
        conn = _Conn()
        with pytest.raises(RuntimeError, match="rolled back, nothing written"):
            bsl.run(conn, apply_changes=True, out_csv=str(tmp_path / "s.csv"))
        assert conn.committed == 0 and conn.rolled_back >= 2  # the read snapshot, then the write

    def test_plan_drift_refuses_before_any_write(self, inputs, tmp_path, monkeypatch):
        monkeypatch.setattr(bsl, "execute_values", lambda *a, **kw: pytest.fail("drift wrote"))
        with pytest.raises(bsl.PlanDrift):
            bsl.run(_Conn(), True, str(tmp_path / "s.csv"), {"relabel_promote_raw": 2, "relabel_espn": 0})

    def test_the_pinned_counts_let_the_apply_through(self, inputs, tmp_path, monkeypatch):
        monkeypatch.setattr(bsl, "execute_values", lambda *a, **kw: [("m1",)])
        counts = bsl.run(_Conn(), True, str(tmp_path / "s.csv"), {"relabel_promote_raw": 1, "relabel_espn": 0})
        assert counts["relabelled"] == 1

    def test_audit_csv_is_write_once(self, inputs, tmp_path):
        out = tmp_path / "s.csv"
        out.write_text("x")
        with pytest.raises(FileExistsError):
            bsl.run(_Conn(), apply_changes=True, out_csv=str(out))


class TestVerify:
    def test_counts_each_rows_label_now(self, tmp_path, capsys):
        apply_csv = tmp_path / "apply.csv"
        bsl.write_csv(
            str(apply_csv),
            [{"match_id": m, "old_season": "2020-2116", "new_season": "2016"} for m in ("m1", "m2", "m3", "m4")],
        )
        conn = _Conn(
            rows=[{"id": "m1", "season": "2016"}, {"id": "m2", "season": "2020-2116"}, {"id": "m3", "season": "x"}]
        )
        counts = bsl.verify(conn, str(apply_csv))
        assert (counts["now_new"], counts["still_old"], counts["other"], counts["missing"]) == (1, 1, 1, 1)
        assert "VERIFY rows=4 now_new=1 still_old=1 other=1 missing=1" in capsys.readouterr().out
        assert conn.executed and "UPDATE" not in conn.executed[0][0]


class TestMain:
    def test_expect_plan_needs_apply(self):
        assert bsl.main(["--database-url", "x", "--expect-plan", "relabel_espn=1"]) == 2

    def test_bad_expect_plan(self):
        assert bsl.main(["--database-url", "x", "--apply", "--expect-plan", "relabel_espn=x"]) == 2

    def test_plan_drift_exits_1(self, monkeypatch):
        class Conn(_Conn):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def drift(*a, **k):
            raise bsl.PlanDrift("drift")

        monkeypatch.setattr(bsl.psycopg2, "connect", lambda url: Conn())
        monkeypatch.setattr(bsl, "run", drift)
        assert bsl.main(["--database-url", "x", "--apply", "--expect-plan", "relabel_espn=1"]) == 1
