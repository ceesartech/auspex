"""repair_football_data_kickoffs: which football-data rows are copies and
merge into which ESPN row, which rows shift and where, what is refused, and
that every write is guarded, archived first and rolled back on drift. No DB:
planning is pure; the writes run against a recording cursor. The fixtures
are prod rows read 2026-10-31."""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
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


rfk = _load("repair_football_data_kickoffs", "repair_football_data_kickoffs.py")
pr = rfk.pr
UTC = timezone.utc

LEAGUES = {"CN1": "lg-cn1", "MLS": "lg-mls", "E0": "lg-e0", "MX1": "lg-mx1"}
TEAMS = {
    "shandong taishan": "t-sha",
    "yunnan yukun": "t-yun",
    "fc cincinnati": "t-cin",
    "san jose earthquakes": "t-sje",
    "arsenal": "t-ars",
    "chelsea": "t-che",
    "wuhan three towns": "t-wuh",
    "monterrey": "t-mon",
    "santos": "t-san",  # Santos FC (Brazil): ESPN filed Santos Laguna's Liga MX rows on it
    "santos laguna": "t-sl",
}


def _raw(date="10/07/2026", time="12:35", home="Shandong Taishan", away="Yunnan Yukun", ft=("4", "3"), league="CN1"):
    return {
        "league": league,
        "season": "2026",
        "Date": date,
        "Time": time,
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": ft[0],
        "FTAG": ft[1],
        "HTHG": "",
        "HTAG": "",
    }


def _m(
    id_,
    when,
    home="t-sha",
    away="t-yun",
    status="finished",
    score=(4, 3),
    ht=(None, None),
    espn_id=None,
    result_source=None,
    league="lg-cn1",
):
    return {
        "id": id_,
        "league_id": league,
        "home_id": home,
        "away_id": away,
        "match_date": when,
        "status": status,
        "home_score": score[0],
        "away_score": score[1],
        "half_time_home": ht[0],
        "half_time_away": ht[1],
        "espn_id": espn_id,
        "result_source": result_source,
    }


def _snap(raw_rows, matches, db_tz=UTC):
    return rfk.Snapshot(
        db_tz=db_tz,
        raw_rows=raw_rows,
        league_ids=LEAGUES,
        team_ids=TEAMS,
        team_names={v: k.title() for k, v in TEAMS.items()},
        matches=[dict(m) for m in matches],  # Snapshot annotates its rows; keep the fixtures clean
    )


def _t(*args):
    return datetime(*args, tzinfo=UTC)


# Shandong Taishan v Yunnan Yukun (CN1): ESPN row e77e612a at 11:35Z, football-data copy 90455957 at 12:35Z.
COPY = _m("90455957", _t(2026, 7, 10, 12, 35))
ESPN = _m("e77e612a", _t(2026, 7, 10, 11, 35), ht=(3, 1), result_source="espn_scoreboard")
SHANDONG = _raw()


def _facts(open_recs=0, post_kickoff_predictions=0, predictions=30, ungraded=30, recs=None, unsettled=0):
    return {
        "open_recs": open_recs,
        "post_kickoff_predictions": post_kickoff_predictions,
        "predictions": predictions,
        "ungraded_predictions": ungraded,
        "recs_by_status": recs if recs is not None else {"void": 1},
        "unsettled_recs": unsettled,
    }


def _plan_dedupe(
    raw_rows, matches, refs=None, guards=("predictions", "betting_recommendations"), facts=None, repoint_map=None
):
    snap = _snap(raw_rows, matches)
    counts: Counter = Counter()
    snap.set_pending(rfk.pending_repoints(snap, repoint_map or {}, counts))
    links = rfk.link_rows(snap, counts)
    candidates = rfk.dedupe_candidates(snap, links, counts)
    return rfk.plan_dedupe(snap, candidates, refs or {}, guards, counts, facts), counts


def _plan_shift(raw_rows, matches, db_tz=UTC, planned_away=frozenset(), repoint_map=None):
    snap = _snap(raw_rows, matches, db_tz)
    counts: Counter = Counter()
    snap.set_pending(rfk.pending_repoints(snap, repoint_map or {}, counts))
    return rfk.plan_shift(snap, rfk.link_rows(snap, counts), counts, planned_away), counts


class TestLinkRows:
    def test_a_summer_row_links_at_the_legacy_kickoff(self):
        counts: Counter = Counter()
        links = rfk.link_rows(_snap([SHANDONG], [COPY, ESPN]), counts)
        link = links["90455957"]
        assert (link.legacy, link.corrected) == (_t(2026, 7, 10, 12, 35), _t(2026, 7, 10, 11, 35))
        assert link.at_legacy and link.bst and link.raw_scores == {(4, 3)}
        assert set(links) == {"90455957"}  # the ESPN row has another kickoff: not football-data's

    def test_a_winter_row_kickoff_is_unchanged(self):
        row = _m("w", _t(2025, 12, 5, 19, 0))
        link = rfk.link_rows(_snap([_raw(date="05/12/2025", time="19:00")], [row]), Counter())["w"]
        assert link.at_legacy and not link.bst

    def test_a_moved_row_links_at_the_corrected_kickoff(self):
        moved = _m("90455957", _t(2026, 7, 10, 11, 35))
        link = rfk.link_rows(_snap([SHANDONG], [moved]), Counter())["90455957"]
        assert not link.at_legacy

    def test_another_row_on_the_corrected_slot_is_not_football_datas(self):
        # Once the copy is merged away, a stale fixture row (no score) can sit at the corrected kickoff.
        stale = _m("stale", _t(2026, 7, 10, 11, 35), status="scheduled", score=(None, None))
        counts: Counter = Counter()
        assert rfk.link_rows(_snap([SHANDONG], [stale]), counts) == {}
        assert counts["raw_corrected_slot_other_row"] == 1

    def test_the_legacy_kickoff_is_read_in_the_database_timezone(self):
        paris_legacy = _m("p", _t(2026, 7, 10, 10, 35))  # a naive 12:35 in a Europe/Paris session
        assert "p" in rfk.link_rows(_snap([SHANDONG], [paris_legacy], ZoneInfo("Europe/Paris")), Counter())

    @pytest.mark.parametrize(
        "raw, counter",
        [
            (_raw(time=""), "raw_date_only"),
            (_raw(ft=("", "")), "raw_no_result"),
            (_raw(home="Nowhere"), "raw_unresolved"),
            (_raw(league="E0"), "raw_league_mismatch"),
            (_raw(date="11/07/2026"), "raw_no_matches_row"),
        ],
    )
    def test_rows_that_do_not_link(self, raw, counter):
        counts: Counter = Counter()
        assert rfk.link_rows(_snap([raw], [COPY]), counts) == {}
        assert counts[counter] == 1

    def test_raw_duplicates_share_one_link(self):
        counts: Counter = Counter()
        links = rfk.link_rows(_snap([SHANDONG, {**SHANDONG, "FTHG": "5"}], [COPY]), counts)
        assert links["90455957"].raw_scores == {(4, 3), (5, 3)} and counts["raw_duplicate_rows"] == 1


class TestPlanDedupe:
    def test_the_prod_pair_merges_into_the_espn_row(self):
        (entry,), counts = _plan_dedupe([SHANDONG], [COPY, ESPN])
        assert (entry["action"], entry["copy_id"], entry["espn_row_id"]) == ("merge", "90455957", "e77e612a")
        assert entry["offset_min"] == "60" and entry["bst"] is True
        assert entry["ht_action"] == "none_on_copy" and entry["ht_write"] is None
        assert counts["dedupe_merge"] == 1

    def test_a_stale_fixture_row_beside_the_espn_row_is_passed_over(self):
        # Atlas v Monterrey (MX1): pre-023 'scheduled' row at 01:00Z, ESPN result 01:05Z, copy 02:05Z.
        stale = _m("stale", _t(2026, 7, 10, 11, 30), status="scheduled", score=(None, None))
        espn = {**ESPN, "match_date": _t(2026, 7, 10, 11, 30) + timedelta(minutes=5)}
        copy = _m("90455957", _t(2026, 7, 10, 12, 35))
        (entry,), _ = _plan_dedupe([SHANDONG], [stale, espn, copy])
        assert entry["action"] == "merge" and entry["espn_row_id"] == "e77e612a"
        assert entry["other_row_ids"] == "stale" and entry["offset_min"] == "60"

    def test_an_espn_row_well_after_the_legacy_kickoff_still_merges(self):
        # FC Cincinnati v San Jose (MLS): copy 00:30Z (legacy of 00:30 UK on 02/08), ESPN row 01:46Z.
        raw = _raw(date="02/08/2026", time="00:30", home="FC Cincinnati", away="San Jose Earthquakes", ft=("4", "2"))
        raw["league"] = "MLS"
        kw = dict(home="t-cin", away="t-sje", score=(4, 2), league="lg-mls")
        copy = _m("d4607acc", _t(2026, 8, 2, 0, 30), **kw)
        stale = _m("881fea0e", _t(2026, 8, 1, 23, 30), status="scheduled", **{**kw, "score": (None, None)})
        espn = _m("a88486bc", _t(2026, 8, 2, 1, 46), result_source="espn_scoreboard", **kw)
        (entry,), _ = _plan_dedupe([raw], [copy, stale, espn])
        assert (entry["action"], entry["espn_row_id"], entry["offset_min"]) == ("merge", "a88486bc", "-76")

    def test_without_fold_facts_a_stale_row_is_refused(self):
        stale = _m("stale", _t(2026, 7, 10, 11, 35), status="scheduled", score=(None, None))
        (entry,), _ = _plan_dedupe([SHANDONG], [stale, COPY])
        assert entry["action"] == "refuse" and "no fold facts" in entry["reason"]

    def test_two_football_data_rows_are_refused(self):
        other = _m("other", _t(2026, 7, 10, 11, 35))
        (entry,), _ = _plan_dedupe([SHANDONG], [other, COPY])
        assert entry["action"] == "refuse" and "two football-data rows" in entry["reason"]

    def test_different_scores_are_refused(self):
        (entry,), _ = _plan_dedupe([SHANDONG], [COPY, {**ESPN, "home_score": 3}])
        assert entry["action"] == "refuse" and "scores differ" in entry["reason"]

    def test_an_unfinished_espn_row_is_refused(self):
        (entry,), _ = _plan_dedupe([SHANDONG], [COPY, {**ESPN, "status": "postponed"}])
        assert entry["action"] == "refuse" and "not both finished" in entry["reason"]

    def test_a_copy_with_predictions_or_recs_is_refused(self):
        refs = {"90455957": {"predictions": 3, "features_cache": 1}}
        (entry,), _ = _plan_dedupe([SHANDONG], [COPY, ESPN], refs=refs)
        assert entry["action"] == "refuse" and entry["reason"] == "copy referenced by predictions (3)"

    def test_derived_rows_do_not_block(self):
        refs = {"90455957": {"features_cache": 1, "match_stats": 2, "odds": 13}}
        (entry,), _ = _plan_dedupe([SHANDONG], [COPY, ESPN], refs=refs)
        assert entry["action"] == "merge"
        assert json.loads(entry["copy_refs"]) == {"features_cache": 1, "match_stats": 2, "odds": 13}

    def test_two_espn_rows_are_ambiguous(self):
        espn2 = {**ESPN, "id": "espn2", "match_date": _t(2026, 7, 10, 11, 40)}
        (entry,), _ = _plan_dedupe([SHANDONG], [COPY, ESPN, espn2])
        assert entry["action"] == "refuse" and entry["reason"].startswith("ambiguous")

    def test_an_espn_row_wanted_by_two_copies_is_refused_for_both(self):
        copy2 = {**COPY, "id": "copy2", "match_date": _t(2026, 7, 10, 12, 40)}
        raw2 = _raw(time="12:40")
        entries, _ = _plan_dedupe([SHANDONG, raw2], [COPY, copy2, ESPN])
        assert [e["action"] for e in entries] == ["refuse", "refuse"]
        assert all("nearby row of 2" in e["reason"] for e in entries)

    def test_an_espn_row_4_hours_away_is_not_a_twin(self):
        far = {**ESPN, "match_date": _t(2026, 7, 10, 7, 35)}
        assert _plan_dedupe([SHANDONG], [COPY, far])[0] == []

    def test_an_espn_row_on_another_uk_date_is_not_a_twin(self):
        # 00:30 UK on 11/07 is 23:30Z on the 10th; an ESPN row at 21:30Z on the 10th is 2 h before but on
        # the UK date of the 10th.
        raw = _raw(date="11/07/2026", time="00:30")
        copy = _m("late", _t(2026, 7, 11, 0, 30))
        espn = {**ESPN, "match_date": _t(2026, 7, 10, 21, 30)}
        assert _plan_dedupe([raw], [copy, espn])[0] == []

    def test_a_linked_espn_row_is_never_a_copy(self):
        winter = _raw(date="05/12/2025", time="19:00")
        espn = {**ESPN, "match_date": _t(2025, 12, 5, 19, 0)}
        entries, counts = _plan_dedupe([winter], [espn])
        assert entries == [] and counts["linked_row_is_espn"] == 1


class TestHtPlan:
    @pytest.mark.parametrize(
        "copy_ht, espn_ht, expected",
        [
            ((None, None), (None, None), ("none_on_copy", None)),
            ((1, 0), (None, None), ("copy", (1, 0))),
            ((1, 0), (1, 0), ("same", None)),
            ((1, 0), (0, 0), ("conflict_kept_espn", None)),
            ((1, None), (None, None), ("incomplete_on_copy", None)),
            ((5, 0), (None, None), ("exceeds_espn_final", None)),
        ],
    )
    def test_cases(self, copy_ht, espn_ht, expected):
        copy, twin = _m("c", None, ht=copy_ht), _m("e", None, ht=espn_ht)
        assert rfk.ht_plan(copy, twin) == expected


class TestPlanShift:
    def test_a_summer_row_moves_back_an_hour(self):
        (entry,), counts = _plan_shift([SHANDONG], [COPY])
        assert (entry["action"], entry["old"], entry["new"]) == (
            "shift",
            _t(2026, 7, 10, 12, 35),
            _t(2026, 7, 10, 11, 35),
        )
        assert counts["shift_shift"] == 1

    def test_a_winter_row_stays(self):
        entries, counts = _plan_shift([_raw(date="05/12/2025", time="19:00")], [_m("w", _t(2025, 12, 5, 19, 0))])
        assert entries == [] and counts["gmt_kickoff_right"] == 1

    def test_a_moved_row_is_done(self):
        entries, counts = _plan_shift([SHANDONG], [_m("90455957", _t(2026, 7, 10, 11, 35))])
        assert entries == [] and counts["already_shifted"] == 1

    def test_the_corrected_slot_is_a_unique_collision(self):
        (entry,), _ = _plan_shift([SHANDONG], [COPY, ESPN])
        assert entry["action"] == "refuse" and "UNIQUE collision" in entry["reason"]
        assert entry["near_row_id"] == "e77e612a"

    def test_a_same_pair_row_within_3_hours_of_the_new_kickoff_is_refused(self):
        near = _m("near", _t(2026, 7, 10, 9, 0), status="scheduled", score=(None, None))
        (entry,), _ = _plan_shift([SHANDONG], [COPY, near])
        assert entry["action"] == "refuse" and "UNIQUE" not in entry["reason"] and "scheduled" in entry["reason"]

    def test_a_row_with_espn_identity_is_refused(self):
        (entry,), _ = _plan_shift([SHANDONG], [{**COPY, "espn_id": "123"}])
        assert entry["action"] == "refuse" and "ESPN identity" in entry["reason"]

    def test_a_row_edited_since_is_refused(self):
        (entry,), _ = _plan_shift([SHANDONG], [{**COPY, "home_score": 0}])
        assert entry["action"] == "refuse" and "edited since" in entry["reason"]

    def test_a_probable_alias_twin_minutes_away_is_shifted_and_listed(self):
        # ESPN filed the fixture under its own team rows, same league and score, kicking off 5 min later.
        alias = _m("alias", _t(2026, 7, 10, 11, 40), home="t-wuh", result_source="espn_scoreboard")
        (entry,), counts = _plan_shift([SHANDONG], [COPY, alias])
        assert (entry["action"], entry["alias_twin_id"], counts["shift_alias_twins"]) == ("shift", "alias", 1)

    def test_another_score_is_not_an_alias_twin(self):
        other = _m("other", _t(2026, 7, 10, 11, 40), home="t-wuh", score=(0, 0), result_source="espn_scoreboard")
        (entry,), _ = _plan_shift([SHANDONG], [COPY, other])
        assert entry["alias_twin_id"] == ""


class TestCheckCounts:
    def test_a_copy_beside_its_espn_row_is_a_duplicate(self):
        snap = _snap([SHANDONG], [COPY, ESPN])
        dup, stale = rfk.football_data_neighbours(snap, rfk.link_rows(snap, Counter()))
        assert [(m["id"], x["id"]) for m, x in dup] == [("90455957", "e77e612a")] and stale == []

    def test_a_stale_fixture_row_is_information_only(self):
        stale_row = _m("stale", _t(2026, 7, 10, 11, 35), status="scheduled", score=(None, None))
        snap = _snap([SHANDONG], [COPY, stale_row])
        dup, stale = rfk.football_data_neighbours(snap, rfk.link_rows(snap, Counter()))
        assert dup == [] and [x["id"] for _m, x in stale] == ["stale"]

    def test_alias_twins_are_found_at_the_legacy_and_the_corrected_kickoff(self):
        alias = _m("alias", _t(2026, 7, 10, 11, 35), home="t-wuh", result_source="espn_scoreboard")
        for copy in (COPY, {**COPY, "match_date": _t(2026, 7, 10, 11, 35)}):
            snap = _snap([SHANDONG], [dict(copy), dict(alias)])
            assert [t["id"] for _m, t in rfk.probable_alias_twins(snap, rfk.link_rows(snap, Counter()))] == ["alias"]


FKS = [
    rfk.FkTable("public", "odds", "match_id", "c"),
    rfk.FkTable("public", "match_stats", "match_id", "c"),
    rfk.FkTable("public", "predictions", "match_id", "c"),
    rfk.FkTable("public", "betting_recommendations", "match_id", "a"),
    rfk.FkTable("public", "betting_history", "match_id", "n"),
    rfk.FkTable("public", "features_cache", "match_id", "c"),
]


class TestSql:
    def test_guards_are_the_money_path_and_every_non_cascading_fk(self):
        weird = rfk.FkTable("public", "new_table", "match_ref", "r")
        assert rfk.guard_table_names(FKS + [weird]) == {
            "predictions",
            "betting_recommendations",
            "betting_history",
            "new_table",
        }

    def test_the_delete_is_guarded_on_each_guard_table_and_on_espn_identity(self):
        guards = [t for t in FKS if t.table_name in rfk.guard_table_names(FKS)]
        sql = rfk.delete_sql(guards)
        for t in ("predictions", "betting_recommendations", "betting_history"):
            assert f'NOT EXISTS (SELECT 1 FROM "public"."{t}" g WHERE g."match_id" = m.id)' in sql
        assert "external_ids->>'espn' IS NULL" in sql and "{guards}" not in sql

    def test_identifiers_are_quoted(self):
        assert rfk.quote_ident('we"ird') == '"we""ird"'

    def test_moves_respect_the_unique_keys(self):
        for col in ("bookmaker", "market_type", "selection", "COALESCE(x.line, -1)", "is_opening", "is_live"):
            assert col in rfk.MOVE_ODDS_SQL and col in rfk.MOVE_COUNTS_SQL
        assert "x.team_id = s.team_id" in rfk.MOVE_STATS_SQL

    def test_the_shift_is_guarded(self):
        for clause in (
            "m.match_date = v.old_kickoff",
            "m.status = v.status",
            "m.home_score IS NOT DISTINCT FROM v.home_score",
            "external_ids->>'espn' IS NULL",
            "interval '10800 seconds'",
        ):
            assert clause in rfk.SHIFT_SQL


# --- the writes -----------------------------------------------------------------------------------------------------


class _Cur:
    def __init__(self, conn):
        self.conn, self.rowcount, self._rows = conn, 0, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.log.append(("execute", sql.split()[0], sql))
        if "FOR UPDATE" in sql:
            self._rows = self.conn.locked
        elif "open_recs" in sql:
            self._rows = [{"id": i, **f} for i, f in self.conn.facts.items() if i in params[0]]
        elif "is_correct IS NOT NULL" in sql:
            self._rows = self.conn.graded.pop(0)
        elif "row_to_json" in sql:
            self._rows = [{"match_id": "90455957", "data": {"id": "x"}}]
        elif sql.lstrip().startswith("UPDATE matches"):
            self.conn.writes_after_archive.append(self.conn.archive_exists())
            if "result_adopted_from" in sql:
                self.conn.folds.append(params)
                self.rowcount = self.conn.fold_rowcount
            else:
                self.rowcount = self.conn.ht_rowcount

    def fetchall(self):
        return self._rows


class _Conn:
    def __init__(self, locked=(), ht_rowcount=1, archive=None, facts=None, graded=None, fold_rowcount=1):
        self.locked, self.ht_rowcount, self.archive = list(locked), ht_rowcount, archive
        self.facts, self.graded, self.fold_rowcount = facts or {}, list(graded or []), fold_rowcount
        self.folds: list = []
        self.log: list = []
        self.writes_after_archive: list[bool] = []
        self.committed = self.rolled_back = 0

    def archive_exists(self) -> bool:
        return bool(self.archive and self.archive.exists())

    def cursor(self, cursor_factory=None):
        return _Cur(self)

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1


@pytest.fixture
def fake_values(monkeypatch):
    """execute_values stand-in: returns `returns[kind]` rows, records whether the archive existed."""
    state = {"returns": {}, "calls": []}

    def fake(cur, sql, argslist, template=None, page_size=100, fetch=False):
        kind = sql.split()[0] + (" odds" if "UPDATE odds" in sql else " stats" if "UPDATE match_stats" in sql else "")
        state["calls"].append((kind, list(argslist), template))
        cur.conn.writes_after_archive.append(cur.conn.archive_exists())
        return state["returns"].get(kind, [])

    monkeypatch.setattr(rfk, "execute_values", fake)
    return state


def _merge_entry(ht_write=None):
    copy = {**COPY, "kickoff": COPY["match_date"]}
    twin = {**ESPN, "kickoff": ESPN["match_date"], "half_time_home": None, "half_time_away": None}
    return {
        "copy_id": copy["id"],
        "copy": copy,
        "twin": twin,
        "home_score": 4,
        "away_score": 3,
        "ht_write": ht_write,
        "action": "merge",
    }


def _locked(entry):
    return [{k: entry[r][k] for k in ("id",) + rfk.OBSERVED} for r in ("copy", "twin")]


class TestApplyDedupe:
    def test_archives_first_then_writes_and_commits(self, tmp_path, fake_values):
        entry = _merge_entry(ht_write=(3, 1))
        archive = tmp_path / "a.jsonl.gz"
        conn = _Conn(locked=_locked(entry), archive=archive)
        fake_values["returns"] = {
            "UPDATE odds": [("o1",)],
            "UPDATE stats": [("s1",), ("s2",)],
            "DELETE": [{"id": "90455957"}],
        }
        done = rfk.apply_dedupe_batch(conn, [entry], 1, FKS, archive)
        assert done == Counter(
            archived_records=2 + len(FKS), ht_copied=1, odds_moved=1, match_stats_moved=2, copies_deleted=1
        )
        assert conn.committed == 1 and conn.rolled_back == 0
        assert conn.writes_after_archive and all(conn.writes_after_archive)  # nothing written before the archive
        records = [json.loads(line) for line in gzip.open(archive, "rt")]
        assert {r["role"] for r in records} == {"copy", "espn_row_before", "references_copy"}
        assert all(r["batch"] == 1 for r in records)
        (delete_args,) = [args for kind, args, _t in fake_values["calls"] if kind == "DELETE"]
        assert delete_args == [("90455957", COPY["match_date"], "finished", 4, 3)]

    def test_a_row_that_drifted_is_caught_before_anything_is_written(self, tmp_path, fake_values):
        entry = _merge_entry()
        locked = _locked(entry)
        locked[1]["home_score"] = 9
        conn = _Conn(locked=locked, archive=tmp_path / "a.jsonl.gz")
        with pytest.raises(rfk.PlanDrift, match="twin e77e612a changed"):
            rfk.apply_dedupe_batch(conn, [entry], 1, FKS, tmp_path / "a.jsonl.gz")
        assert conn.rolled_back == 1 and not conn.committed and not (tmp_path / "a.jsonl.gz").exists()
        assert fake_values["calls"] == []

    def test_a_guard_that_keeps_the_copy_rolls_back(self, tmp_path, fake_values):
        entry = _merge_entry()
        conn = _Conn(locked=_locked(entry), archive=tmp_path / "a.jsonl.gz")
        fake_values["returns"] = {"DELETE": []}  # a rec appeared on the copy
        with pytest.raises(rfk.PlanDrift, match="not deleted by the guard"):
            rfk.apply_dedupe_batch(conn, [entry], 1, FKS, tmp_path / "a.jsonl.gz")
        assert conn.rolled_back == 1 and not conn.committed

    def test_a_half_time_that_moved_rolls_back(self, tmp_path, fake_values):
        entry = _merge_entry(ht_write=(3, 1))
        conn = _Conn(locked=_locked(entry), ht_rowcount=0, archive=tmp_path / "a.jsonl.gz")
        with pytest.raises(rfk.PlanDrift, match="half-time"):
            rfk.apply_dedupe_batch(conn, [entry], 1, FKS, tmp_path / "a.jsonl.gz")
        assert conn.rolled_back == 1

    def test_the_archive_is_write_once(self, tmp_path, fake_values):
        entry = _merge_entry()
        archive = tmp_path / "a.jsonl.gz"
        archive.write_text("earlier")
        conn = _Conn(locked=_locked(entry), archive=archive)
        with pytest.raises(FileExistsError):
            rfk.apply_dedupe_batch(conn, [entry], 1, FKS, archive)
        assert conn.rolled_back == 1 and archive.read_text() == "earlier"


def _shift_entry(id_="90455957"):
    return {
        "match_id": id_,
        "old": _t(2026, 7, 10, 12, 35),
        "new": _t(2026, 7, 10, 11, 35),
        "status": "finished",
        "home_score": 4,
        "away_score": 3,
        "action": "shift",
    }


class TestApplyShift:
    def test_commits_when_every_row_moved(self, fake_values):
        conn = _Conn()
        fake_values["returns"] = {"UPDATE": [("90455957",), ("b",)]}
        assert rfk.apply_shift_batch(conn, [_shift_entry(), _shift_entry("b")], 1) == 2
        assert conn.committed == 1
        ((_kind, args, template),) = fake_values["calls"]
        assert template == rfk.SHIFT_TEMPLATE
        assert args[0] == ("90455957", _t(2026, 7, 10, 12, 35), _t(2026, 7, 10, 11, 35), "finished", 4, 3)

    def test_a_row_that_moved_rolls_the_batch_back(self, fake_values):
        conn = _Conn()
        fake_values["returns"] = {"UPDATE": [("90455957",)]}
        with pytest.raises(rfk.PlanDrift, match=r"1 of 2 row\(s\).*\['b'\]"):
            rfk.apply_shift_batch(conn, [_shift_entry(), _shift_entry("b")], 3)
        assert conn.rolled_back == 1 and not conn.committed


# --- run / main -----------------------------------------------------------------------------------------------------


@pytest.fixture
def offline(monkeypatch):
    """run() over an in-memory snapshot; records the apply calls."""
    state = {"snap": None, "applied": []}

    state["facts"] = {}
    monkeypatch.setattr(rfk, "load_snapshot", lambda cur: state["snap"])
    monkeypatch.setattr(rfk, "load_fk_tables", lambda cur: FKS)
    monkeypatch.setattr(rfk, "load_refs", lambda cur, fks, ids: {})
    monkeypatch.setattr(
        rfk, "load_fold_facts", lambda cur, ids: {i: state["facts"][i] for i in ids if i in state["facts"]}
    )
    monkeypatch.setattr(rfk, "add_move_counts", lambda cur, entries: None)
    monkeypatch.setattr(
        rfk,
        "apply_dedupe_batch",
        lambda conn, batch, n, fks, archive: state["applied"].append(("dedupe", n, len(batch), archive))
        or Counter(copies_deleted=len(batch)),
    )
    monkeypatch.setattr(
        rfk, "apply_shift_batch", lambda conn, batch, n: state["applied"].append(("shift", n, len(batch))) or len(batch)
    )
    return state


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class TestRun:
    def test_a_dry_run_writes_the_csv_and_nothing_else(self, offline, tmp_path):
        offline["snap"] = _snap([SHANDONG], [dict(COPY), dict(ESPN)])
        counts = rfk.run(_Conn(), "dedupe", False, out_csv=str(tmp_path / "d.csv"))
        assert (counts["planned"], counts["refused"], offline["applied"]) == (1, 0, [])
        assert [(r["copy_id"], r["action"], r["espn_row_id"]) for r in _rows(tmp_path / "d.csv")] == [
            ("90455957", "merge", "e77e612a")
        ]

    def test_apply_writes_in_batches(self, offline, tmp_path):
        rows = [_raw(date=f"{d:02d}/07/2026") for d in range(1, 6)]
        offline["snap"] = _snap(rows, [_m(f"c{d}", _t(2026, 7, d, 12, 35)) for d in range(1, 6)])
        counts = rfk.run(_Conn(), "shift", True, out_csv=str(tmp_path / "s.csv"), batch_size=2)
        assert offline["applied"] == [("shift", 1, 2), ("shift", 2, 2), ("shift", 3, 1)]
        assert counts["shifted"] == 5 and [r["batch"] for r in _rows(tmp_path / "s.csv")] == ["1", "1", "2", "2", "3"]

    def test_dedupe_archives_go_beside_the_csv(self, offline, tmp_path):
        offline["snap"] = _snap([SHANDONG], [dict(COPY), dict(ESPN)])
        rfk.run(_Conn(), "dedupe", True, out_csv=str(tmp_path / "d.csv"), archive_dir=str(tmp_path))
        ((_mode, n, size, archive),) = offline["applied"]
        assert (n, size) == (1, 1) and archive.parent == tmp_path and archive.name.endswith("_b001.jsonl.gz")

    def test_a_plan_that_is_not_the_pinned_one_writes_nothing(self, offline, tmp_path, caplog):
        offline["snap"] = _snap([SHANDONG], [dict(COPY), dict(ESPN)])
        counts = rfk.run(_Conn(), "dedupe", True, out_csv=str(tmp_path / "d.csv"), expect_plan={"merge": 68})
        assert counts["plan_mismatch"] == 1 and offline["applied"] == []
        assert "1 merge, expected 68" in caplog.text and (tmp_path / "d.csv").exists()

    def test_pinned_refusals_are_accepted(self, offline, tmp_path):
        stale = _m("stale", _t(2026, 7, 10, 11, 35), status="scheduled", score=(None, None))
        offline["snap"] = _snap([SHANDONG], [dict(COPY), stale])
        offline["facts"] = {"stale": _facts(open_recs=1)}
        pin = {"merge": 0, "fold": 0, "refuse": 1}
        counts = rfk.run(_Conn(), "dedupe", True, out_csv=str(tmp_path / "d.csv"), expect_plan=pin)
        assert (counts["refused"], counts["refused_unexpected"], counts["plan_mismatch"]) == (1, 0, 0)

    def test_unpinned_refusals_count_against_the_exit(self, offline, tmp_path):
        offline["snap"] = _snap([SHANDONG], [dict(COPY), {**ESPN, "home_score": 0}])  # not a merge: scores differ
        counts = rfk.run(_Conn(), "shift", False, out_csv=str(tmp_path / "s.csv"))
        assert counts["refused_unexpected"] == 1  # the ESPN row sits on the corrected slot

    def test_the_csv_is_write_once(self, offline, tmp_path):
        offline["snap"] = _snap([SHANDONG], [dict(COPY)])
        (tmp_path / "s.csv").write_text("earlier")
        with pytest.raises(FileExistsError):
            rfk.run(_Conn(), "shift", True, out_csv=str(tmp_path / "s.csv"))
        assert offline["applied"] == []


class _ConnCtx(_Conn):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestMain:
    def test_drift_exits_1(self, monkeypatch, caplog):
        monkeypatch.setattr(rfk.psycopg2, "connect", lambda url: _ConnCtx())

        def drift(*args, **kwargs):
            raise rfk.PlanDrift("batch 2: 1 of 5000 row(s) changed")

        monkeypatch.setattr(rfk, "run", drift)
        argv = ["--shift", "--apply", "--expect-plan", "shift=5000,refuse=0", "--database-url", "postgres://fake"]
        assert rfk.main(argv) == 1
        assert "batch 2: 1 of 5000" in caplog.text

    @pytest.mark.parametrize(
        "counts, code",
        [
            (Counter(planned=3), 0),
            (Counter(planned=3, refused=2, refused_unexpected=2), 1),
            (Counter(planned=3, refused=2), 0),
            (Counter(plan_mismatch=1), 1),
            (Counter(planned=3, shift_blocked=1), 1),
            (Counter(planned=3, unsettled_unpinned=1), 1),
        ],
    )
    def test_exit_codes(self, monkeypatch, counts, code):
        monkeypatch.setattr(rfk.psycopg2, "connect", lambda url: _ConnCtx())
        monkeypatch.setattr(rfk, "run", lambda *a, **k: counts)
        assert rfk.main(["--dedupe", "--database-url", "postgres://fake"]) == code

    def test_check_is_read_only(self):
        with pytest.raises(SystemExit):
            rfk.parse_args(["--check", "--apply"])
        with pytest.raises(SystemExit):
            rfk.parse_args(["--check", "--expect-plan", "shift=1"])

    def test_expect_plan_is_parsed_per_mode(self):
        args = rfk.parse_args(["--dedupe", "--expect-plan", "merge=68,fold=26,refuse=0"])
        assert args.expect_plan == {"merge": 68, "fold": 26, "refuse": 0}
        assert rfk.parse_args(["--shift", "--expect-plan", "shift=63533,refuse=0"]).expect_plan == {
            "shift": 63533,
            "refuse": 0,
        }
        for mode, pin in (
            ("--shift", "merge=1"),
            ("--dedupe", "merge=x"),
            ("--dedupe", "merge=1,merge=2"),
            ("--dedupe", "68"),
        ):
            with pytest.raises(SystemExit):
                rfk.parse_args([mode, "--expect-plan", pin])

    @pytest.mark.parametrize(
        "argv",
        [
            ["--dedupe", "--apply"],
            ["--dedupe", "--apply", "--expect-plan", "merge=68,fold=26,refuse=0"],  # a pre-alias-merge pin
            ["--shift", "--apply", "--expect-plan", "shift=63533"],
        ],
    )
    def test_an_apply_must_pin_every_key_of_its_mode(self, argv):
        with pytest.raises(SystemExit):
            rfk.parse_args(argv)

    def test_a_full_pin_applies_and_fold_unsettled_recs_is_an_extra_dedupe_pin(self):
        args = rfk.parse_args(
            ["--dedupe", "--apply", "--expect-plan", "merge=68,alias_merge=4,fold=26,refuse=0,fold_unsettled_recs=0"]
        )
        assert args.expect_plan["fold_unsettled_recs"] == 0 and args.apply
        assert rfk.parse_args(["--shift", "--apply", "--expect-plan", "shift=1,refuse=0"]).expect_plan == {
            "shift": 1,
            "refuse": 0,
        }
        with pytest.raises(SystemExit):
            rfk.parse_args(["--shift", "--expect-plan", "shift=1,refuse=0,fold_unsettled_recs=0"])

    def test_one_mode_at_a_time(self):
        with pytest.raises(SystemExit):
            rfk.parse_args(["--dedupe", "--shift"])
        with pytest.raises(SystemExit):
            rfk.parse_args([])

    def test_no_database_url_exits_2(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert rfk.main(["--check", "--database-url", ""]) == 2


# --- the fold: a copy beside a stale 'scheduled' row at exactly its corrected kickoff ------------------------------

# Liaoning Tieren v Shanghai Port (CN1), prod 2026-10-31: the stale pre-023 fixture row sits at 11:35Z, the
# football-data copy at 12:35Z (its legacy kickoff); 31 recs on the 26 such rows, all 'void', 810 predictions.
STALE = _m("stale", _t(2026, 7, 10, 11, 35), status="scheduled", score=(None, None))


class TestFoldPlan:
    def test_the_copy_folds_into_the_stale_row(self):
        (entry,), counts = _plan_dedupe([SHANDONG], [STALE, COPY], facts={"stale": _facts(recs={"void": 2})})
        assert (entry["action"], entry["copy_id"], entry["espn_row_id"]) == ("fold", "90455957", "stale")
        assert (entry["ht_action"], entry["ht_write"], counts["dedupe_fold"]) == ("none_on_copy", None, 1)
        assert json.loads(entry["fold_recs_by_status"]) == {"void": 2} and entry["fold_open_recs"] == 0

    def test_the_copys_half_time_goes_on_whole_and_within_its_final(self):
        copy = {**COPY, "half_time_home": 2, "half_time_away": 1}
        (entry,), _ = _plan_dedupe([SHANDONG], [STALE, copy], facts={"stale": _facts()})
        assert (entry["action"], entry["ht_write"]) == ("fold", (2, 1))
        bad = {**COPY, "half_time_home": 5, "half_time_away": 1}
        (entry,), _ = _plan_dedupe([SHANDONG], [STALE, bad], facts={"stale": _facts()})
        assert (entry["action"], entry["ht_action"], entry["ht_write"]) == ("fold", "exceeds_espn_final", None)

    @pytest.mark.parametrize(
        "stale, facts, why",
        [
            (STALE, _facts(open_recs=1), "pending/placed recommendation"),
            (STALE, _facts(post_kickoff_predictions=3), "written after its kickoff"),
            ({**STALE, "match_date": _t(2026, 7, 10, 11, 0)}, _facts(), "not on the corrected kickoff"),
            ({**STALE, "home_score": 0, "away_score": 0}, _facts(), "carries a score"),
            ({**STALE, "status": "postponed"}, _facts(), "'postponed' row without ESPN identity"),
        ],
    )
    def test_refused(self, stale, facts, why):
        (entry,), _ = _plan_dedupe([SHANDONG], [stale, COPY], facts={"stale": facts})
        assert entry["action"] == "refuse" and why in entry["reason"]

    def test_a_copy_whose_score_was_edited_is_never_folded(self):
        (entry,), _ = _plan_dedupe([SHANDONG], [STALE, {**COPY, "home_score": 5}], facts={"stale": _facts()})
        assert entry["action"] == "refuse" and "not its raw row's [(4, 3)]" in entry["reason"]

    def test_a_copy_with_predictions_is_never_folded(self):
        refs = {"90455957": {"predictions": 1}}
        (entry,), _ = _plan_dedupe([SHANDONG], [STALE, COPY], refs=refs, facts={"stale": _facts()})
        assert entry["action"] == "refuse" and entry["reason"] == "copy referenced by predictions (1)"

    def test_a_stale_row_wanted_by_two_copies_is_refused(self):
        copy2 = {**COPY, "id": "copy2", "match_date": _t(2026, 7, 10, 12, 40)}
        entries, _ = _plan_dedupe([SHANDONG, _raw(time="12:40")], [STALE, COPY, copy2], facts={"stale": _facts()})
        assert [e["action"] for e in entries] == ["refuse", "refuse"]

    def test_facts_are_read_only_for_stale_rows(self):
        snap = _snap([SHANDONG], [STALE, COPY])
        cands = rfk.dedupe_candidates(snap, rfk.link_rows(snap, Counter()), Counter())
        assert rfk.fold_fact_rows(cands) == ["stale"]
        snap = _snap([SHANDONG], [COPY, ESPN])
        assert rfk.fold_fact_rows(rfk.dedupe_candidates(snap, rfk.link_rows(snap, Counter()), Counter())) == []

    def test_after_the_fold_nothing_is_planned_again(self):
        folded = {**STALE, "status": "finished", "home_score": 4, "away_score": 3}
        assert _plan_dedupe([SHANDONG], [folded], facts={"stale": _facts()})[0] == []
        entries, counts = _plan_shift([SHANDONG], [folded])
        assert entries == [] and counts["already_shifted"] == 1


def _fold_entry(ht_write=None):
    copy = {**COPY, "kickoff": COPY["match_date"]}
    twin = {**STALE, "kickoff": STALE["match_date"]}
    return {
        "copy_id": copy["id"],
        "copy": copy,
        "twin": twin,
        "home_score": 4,
        "away_score": 3,
        "ht_write": ht_write,
        "action": "fold",
    }


class TestApplyFold:
    def _conn(self, entry, tmp_path, **kw):
        kw.setdefault("facts", {"stale": _facts()})
        kw.setdefault("graded", [[], [{"prediction_type": "match_result", "graded": 2}]])
        return _Conn(locked=_locked(entry), archive=tmp_path / "a.jsonl.gz", **kw)

    def test_archives_first_then_finishes_the_stale_row_and_deletes_the_copy(self, tmp_path, fake_values):
        entry = _fold_entry(ht_write=(2, 1))
        conn = self._conn(entry, tmp_path)
        fake_values["returns"] = {"DELETE": [{"id": "90455957"}]}
        done = rfk.apply_dedupe_batch(conn, [entry], 1, FKS, tmp_path / "a.jsonl.gz")
        assert (done["folded"], done["fold_ht_copied"], done["copies_deleted"], conn.committed) == (1, 1, 1, 1)
        assert (done["trigger_graded_predictions"], done["trigger_graded:match_result"]) == (2, 2)
        assert all(conn.writes_after_archive)
        (params,) = conn.folds
        assert params == {
            "id": "stale",
            "kickoff": STALE["match_date"],
            "home": 4,
            "away": 3,
            "ht_home": 2,
            "ht_away": 1,
            "copy_id": "90455957",
        }
        roles = {(r["table"], r["role"]) for r in map(json.loads, gzip.open(tmp_path / "a.jsonl.gz", "rt"))}
        assert {("matches", "fold_row_before"), ("predictions", "fold_row_before")} <= roles
        assert ("betting_recommendations", "fold_row_before") in roles

    def test_a_rec_that_appeared_rolls_back_before_anything_is_written(self, tmp_path, fake_values):
        entry = _fold_entry()
        conn = self._conn(entry, tmp_path, facts={"stale": _facts(open_recs=1)})
        with pytest.raises(rfk.PlanDrift, match="gained 1 open rec"):
            rfk.apply_dedupe_batch(conn, [entry], 1, FKS, tmp_path / "a.jsonl.gz")
        assert conn.rolled_back == 1 and not conn.committed and not (tmp_path / "a.jsonl.gz").exists()
        assert fake_values["calls"] == [] and conn.folds == []

    def test_a_guard_miss_on_the_flip_rolls_back(self, tmp_path, fake_values):
        entry = _fold_entry()
        conn = self._conn(entry, tmp_path, fold_rowcount=0)
        with pytest.raises(rfk.PlanDrift, match="fold row stale changed"):
            rfk.apply_dedupe_batch(conn, [entry], 1, FKS, tmp_path / "a.jsonl.gz")
        assert conn.rolled_back == 1 and not conn.committed


class TestFoldSql:
    def test_the_flip_is_guarded_on_everything_the_trigger_depends_on(self):
        for clause in (
            "m.status = 'scheduled' AND m.match_date = %(kickoff)s",
            "m.home_score IS NULL AND m.away_score IS NULL",
            "m.half_time_home IS NULL AND m.half_time_away IS NULL",
            "m.external_ids->>'espn' IS NULL",
            "br.status IN ('pending', 'placed')",
            "br.prediction_id IN (SELECT p.id FROM predictions p WHERE p.match_id = m.id)",
            "p.created_at > m.match_date",
        ):
            assert clause in rfk.FOLD_SQL
        assert "status = 'finished'" in rfk.FOLD_SQL

    def test_the_facts_count_open_recs_through_predictions_and_post_kickoff_predictions(self):
        assert "br.prediction_id IN (SELECT p.id FROM predictions p WHERE p.match_id = m.id)" in rfk.FOLD_FACTS_SQL
        assert "p.created_at > m.match_date" in rfk.FOLD_FACTS_SQL  # migration 027's prediction rule


# --- --shift never puts a row on the exact kickoff of the same fixture under other team ids ----------------------


class TestAliasSlot:
    def test_a_row_sharing_a_team_at_the_new_kickoff_blocks_the_shift(self):
        # Monterrey v Santos Laguna (MX1): ESPN filed it under Monterrey v Santos (the Brazilian club's row).
        alias = _m("alias", _t(2026, 7, 10, 11, 35), home="t-sha", away="t-wuh", result_source="espn_scoreboard")
        (entry,), counts = _plan_shift([SHANDONG], [COPY, alias])
        assert (entry["action"], entry["alias_slot_row_id"]) == ("refuse", "alias")
        assert entry["reason"].startswith("alias slot") and counts["shift_refuse"] == 1

    def test_an_espn_row_with_the_same_score_and_club_names_blocks_it(self):
        alias = _m("alias", _t(2026, 7, 10, 11, 35), home="t-sha2", away="t-yun2", result_source="espn_scoreboard")
        snap = _snap([SHANDONG], [COPY, alias])
        snap.team_names.update({"t-sha2": "Shandong Taishan FC", "t-yun2": "Yunnan Yukun FC"})
        (entry,) = rfk.plan_shift(snap, rfk.link_rows(snap, Counter()), Counter())
        assert (entry["action"], entry["alias_slot_row_id"]) == ("refuse", "alias")

    def test_another_fixture_with_the_same_score_does_not(self):
        # Inter Miami v Columbus Crew 2-2 (MLS, 2026-08-01 23:30Z) beside DC United v Nashville SC 2-2 at the same
        # kickoff: same league, kickoff and score, but no club in common and nothing alike in the names.
        other = _m("other", _t(2026, 7, 10, 11, 35), home="t-ars", away="t-che", result_source="espn_scoreboard")
        (entry,), _ = _plan_shift([SHANDONG], [COPY, other])
        assert (entry["action"], entry["alias_slot_row_id"]) == ("shift", "")

    def test_a_row_dedupe_will_remove_does_not_count(self):
        alias = _m("alias", _t(2026, 7, 10, 11, 35), home="t-sha", away="t-wuh")
        (entry,), _ = _plan_shift([SHANDONG], [COPY, alias], planned_away=frozenset({"alias"}))
        assert entry["action"] == "shift"

    def test_two_football_data_rows_sharing_a_team_onto_one_kickoff_both_stay(self):
        other_raw = _raw(home="Shandong Taishan", away="Wuhan Three Towns")
        other = _m("other", _t(2026, 7, 10, 12, 35), home="t-sha", away="t-wuh")
        entries, counts = _plan_shift([SHANDONG, other_raw], [COPY, other])
        assert [e["action"] for e in entries] == ["refuse", "refuse"] and counts["shift_shift"] == 0
        assert {e["alias_slot_row_id"] for e in entries} == {"90455957", "other"}

    def test_a_copy_dedupe_still_plans_is_left_to_it(self):
        entries, counts = _plan_shift([SHANDONG], [COPY, ESPN], planned_away=frozenset({"90455957"}))
        assert entries == [] and counts["left_to_dedupe"] == 1

    @pytest.mark.parametrize(
        "a, b, same",
        [
            ("Hacken", "BK Häcken", True),
            ("Inter Miami", "Inter Miami CF", True),
            ("Santos Laguna", "Santos", True),
            ("Man City", "Manchester City", True),
            ("Inter Miami", "DC United", False),
            ("Columbus Crew", "Nashville SC", False),
            ("San Diego FC", "San Jose Earthquakes", False),
            ("Atlanta United FC", "Minnesota United", False),
        ],
    )
    def test_same_club_name(self, a, b, same):
        assert rfk.same_club_name(a, b) is same


class TestRunShiftAfterDedupe:
    def test_shift_apply_refuses_while_dedupe_has_work(self, offline, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="repair_football_data_kickoffs")
        offline["snap"] = _snap(
            [SHANDONG, _raw(date="11/07/2026")], [dict(COPY), dict(ESPN), _m("c2", _t(2026, 7, 11, 12, 35))]
        )
        counts = rfk.run(_Conn(), "shift", True, out_csv=str(tmp_path / "s.csv"))
        assert (counts["dedupe_pending"], counts["shift_blocked"], offline["applied"]) == (1, 1, [])
        assert "PLAN mode=shift shift=1 refuse=0 dedupe_pending=1" in caplog.text
        assert "Refusing to shift: --dedupe still plans to merge or fold 1 copies" in caplog.text

    def test_the_plan_line_counts_every_action(self, offline, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="repair_football_data_kickoffs")
        offline["snap"] = _snap(
            [SHANDONG, _raw(date="11/07/2026")],
            [
                dict(COPY),
                dict(ESPN),
                dict(STALE) | {"id": "s2", "match_date": _t(2026, 7, 11, 11, 35)},
                _m("c2", _t(2026, 7, 11, 12, 35)),
            ],
        )
        offline["facts"] = {"s2": _facts()}
        counts = rfk.run(
            _Conn(), "dedupe", True, out_csv=str(tmp_path / "d.csv"), expect_plan={"merge": 1, "fold": 1, "refuse": 0}
        )
        assert "PLAN mode=dedupe merge=1 alias_merge=0 fold=1 refuse=0 fold_unsettled_recs=0" in caplog.text
        assert counts["planned"] == 2 and offline["applied"][0][2] == 2  # one batch, merge and fold together


# --- --check ---------------------------------------------------------------------------------------------------------


class _CheckCur:
    """load_snapshot is faked; this answers check_legacy_kickoffs' two queries from the snapshot's rows."""

    def __init__(self, matches):
        self._matches, self._last = matches, ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._last = sql

    def fetchone(self):
        return {"TimeZone": "UTC"}

    def fetchall(self):
        return [
            {"home_id": m["home_id"], "away_id": m["away_id"], "match_date": m["match_date"]} for m in self._matches
        ]


class _CheckConn(_Conn):
    def __init__(self, matches):
        super().__init__()
        self._matches = matches

    def cursor(self, cursor_factory=None):
        return _CheckCur(self._matches)


class TestRunCheck:
    @pytest.fixture(autouse=True)
    def _info(self, caplog):
        caplog.set_level(logging.INFO, logger="repair_football_data_kickoffs")

    def _check(self, monkeypatch, matches):
        snap = _snap([SHANDONG], matches)
        monkeypatch.setattr(rfk, "load_snapshot", lambda cur: snap)
        return rfk.run_check(_CheckConn(matches))

    def test_a_row_at_the_legacy_kickoff_fails(self, monkeypatch, caplog):
        assert self._check(monkeypatch, [COPY]) == 1
        assert "CHECK legacy_kickoffs=FAIL" in caplog.text

    def test_a_copy_beside_its_espn_row_fails_twice_over(self, monkeypatch, caplog):
        assert self._check(monkeypatch, [COPY, ESPN]) == 1
        assert "CHECK legacy_kickoffs=FAIL football_data_duplicates=1" in caplog.text

    def test_the_repaired_state_passes(self, monkeypatch, caplog):
        assert self._check(monkeypatch, [ESPN]) == 0
        assert "CHECK legacy_kickoffs=PASS football_data_duplicates=0" in caplog.text

    def test_a_copy_beside_its_espn_row_off_any_legacy_kickoff_fails_on_the_duplicate_alone(self, monkeypatch, caplog):
        # Review finding (2026-10-31): --check's duplicate count was only ever tested beside a legacy failure. Here
        # the copy already sits at its corrected kickoff (shifted earlier), 10 min from the ESPN row: no row is at a
        # legacy kickoff, but the match is in the corpus twice.
        shifted = {**COPY, "match_date": _t(2026, 7, 10, 11, 35)}
        espn = {**ESPN, "match_date": _t(2026, 7, 10, 11, 25)}
        assert self._check(monkeypatch, [shifted, espn]) == 1
        assert "CHECK legacy_kickoffs=PASS football_data_duplicates=1" in caplog.text

    def test_a_gmt_copy_beside_its_espn_row_fails_on_the_duplicate_alone(self, monkeypatch, caplog):
        # 06/12/2025 08:00 UK is 08:00Z: no legacy offset, the copy is where promote_raw put it.
        raw = _raw(date="06/12/2025", time="08:00")
        copy = {**COPY, "match_date": _t(2025, 12, 6, 8, 0)}
        espn = {**ESPN, "match_date": _t(2025, 12, 6, 7, 55)}
        snap = _snap([raw], [copy, espn])
        monkeypatch.setattr(rfk, "load_snapshot", lambda cur: snap)
        assert rfk.run_check(_CheckConn([copy, espn])) == 1
        assert "CHECK legacy_kickoffs=PASS football_data_duplicates=1" in caplog.text


# --- the Liga MX rows ESPN filed under the wrong club: the reviewed re-point map and the alias merge ---------------

# Monterrey v Santos Laguna (MX1), prod 2026-10-31: ESPN's row e78e7e49 (01:05Z, 3-2) sits on Santos FC's row
# ('Santos', Brazil), with a stale 'scheduled' row a398e0fe at 01:00Z; football-data's copy df640b1b is on Santos
# Laguna at its legacy 02:05Z. repoint_team_matches refused both rows (a same-pair twin: the copy), and --shift
# refused the copy (alias slot): each waited for the other.
MX_RAW = _raw(date="19/07/2026", time="02:05", home="Monterrey", away="Santos Laguna", ft=("3", "2"), league="MX1")
MX_COPY = _m("df640b1b", _t(2026, 7, 19, 2, 5), home="t-mon", away="t-sl", score=(3, 2), league="lg-mx1")
MX_ESPN = _m(
    "e78e7e49",
    _t(2026, 7, 19, 1, 5),
    home="t-mon",
    away="t-san",
    score=(3, 2),
    ht=(3, 1),
    result_source="espn_scoreboard",
    league="lg-mx1",
)
MX_STALE = _m(
    "a398e0fe",
    _t(2026, 7, 19, 1, 0),
    home="t-mon",
    away="t-san",
    status="scheduled",
    score=(None, None),
    league="lg-mx1",
)
MX_MAP = {"e78e7e49": {"away": ("santos", "santos laguna")}, "a398e0fe": {"away": ("santos", "santos laguna")}}
MAP_HEADER = "match_id,side,from_team,to_team,league,kickoff_utc,opponent,espn_event,evidence\n"


class TestRepointMap:
    def test_the_reviewed_map_loads(self, tmp_path):
        path = tmp_path / "map.csv"
        path.write_text(
            MAP_HEADER
            + "e78e7e49,away,Santos,Santos Laguna,Liga MX,2026-07-19 01:05,Monterrey,401877038,proof\n"
            + "4a31da6c,home,Santos,Santos Laguna,Liga MX,2026-08-17 01:00,Guadalajara,401877013,proof\n"
            + "4a31da6c,away,Guadalajara,Guadalajara  Chivas,Liga MX,2026-08-17 01:00,Santos,401877013,proof\n"
        )
        assert rfk.load_repoint_map(path) == {
            "e78e7e49": {"away": ("santos", "santos laguna")},
            "4a31da6c": {"home": ("santos", "santos laguna"), "away": ("guadalajara", "guadalajara chivas")},
        }

    @pytest.mark.parametrize(
        "body, why",
        [
            ("match_id,side,from_team\nx,away,Santos\n", "missing column"),
            (MAP_HEADER + "x,both,Santos,Santos Laguna,,,,,\n", "home/away"),
            (MAP_HEADER + "x,away,,Santos Laguna,,,,,\n", "home/away"),
            (MAP_HEADER + "x,away,Santos,Santos Laguna,,,,,\nx,away,Santos,Santos Laguna,,,,,\n", "listed twice"),
        ],
    )
    def test_a_malformed_map_is_refused(self, tmp_path, body, why):
        path = tmp_path / "map.csv"
        path.write_text(body)
        with pytest.raises(ValueError, match=why):
            rfk.load_repoint_map(path)

    def test_pending_done_and_unclear_rows(self, caplog):
        moved = {**MX_ESPN, "id": "moved", "away_id": "t-sl"}
        other = {**MX_ESPN, "id": "other", "away_id": "t-cin"}
        unresolved = {**MX_ESPN, "id": "unresolved"}
        half = {**MX_ESPN, "id": "half", "home_id": "t-cin"}  # home moved already, away not: a half-moved match
        snap = _snap([], [MX_ESPN, moved, other, unresolved, half])
        counts: Counter = Counter()
        repoint_map = {
            "e78e7e49": MX_MAP["e78e7e49"],
            "moved": MX_MAP["e78e7e49"],
            "other": MX_MAP["e78e7e49"],  # on neither club: drifted since the review
            "gone": MX_MAP["e78e7e49"],
            "unresolved": {"away": ("santos", "santos de guayaquil")},  # no such team
            "half": {"home": ("monterrey", "fc cincinnati"), "away": ("santos", "santos laguna")},
        }
        assert rfk.pending_repoints(snap, repoint_map, counts) == {"e78e7e49": ("t-mon", "t-sl")}
        assert (counts["repoint_done"], counts["repoint_unclear"], counts["repoint_row_absent"]) == (1, 3, 1)
        assert "re-point map row other is drifted" in caplog.text and "row unresolved is unresolved" in caplog.text

    def test_main_reads_the_map_and_passes_it_on(self, monkeypatch, tmp_path, caplog):
        seen = {}
        monkeypatch.setattr(rfk.psycopg2, "connect", lambda url: _ConnCtx())
        monkeypatch.setattr(rfk, "run", lambda *a, **k: (seen.__setitem__("map", a[7]), Counter())[1])
        path = tmp_path / "map.csv"
        path.write_text(MAP_HEADER + "e78e7e49,away,Santos,Santos Laguna,,,,,\n")
        rfk.main(["--dedupe", "--database-url", "postgres://fake", "--repoint-map", str(path)])
        assert seen["map"] == {"e78e7e49": {"away": ("santos", "santos laguna")}}
        seen.clear()
        rfk.main(["--dedupe", "--database-url", "postgres://fake", "--repoint-map", str(tmp_path / "none.csv")])
        assert seen["map"] == {} and "No re-point map" in caplog.text


class TestAliasMerge:
    def test_the_copy_merges_into_the_espn_row_about_to_be_re_pointed(self):
        (entry,), counts = _plan_dedupe([MX_RAW], [MX_STALE, MX_ESPN, MX_COPY], repoint_map=MX_MAP)
        assert (entry["action"], entry["copy_id"], entry["espn_row_id"], entry["other_row_ids"]) == (
            "alias_merge",
            "df640b1b",
            "e78e7e49",
            "a398e0fe",
        )
        assert entry["kept_row_repoint"] == "away: Santos -> Santos Laguna"
        assert (entry["ht_action"], counts["dedupe_alias_merge"], counts["repoint_pending"]) == ("none_on_copy", 1, 2)

    def test_without_the_map_it_stays_an_alias_slot(self):
        assert _plan_dedupe([MX_RAW], [MX_STALE, MX_ESPN, MX_COPY])[0] == []
        (entry,), _ = _plan_shift([MX_RAW], [MX_STALE, MX_ESPN, MX_COPY])
        assert (entry["action"], entry["alias_slot_row_id"]) == ("refuse", "e78e7e49")

    def test_once_re_pointed_it_is_an_ordinary_merge(self):
        espn = {**MX_ESPN, "away_id": "t-sl"}
        stale = {**MX_STALE, "away_id": "t-sl"}
        (entry,), counts = _plan_dedupe([MX_RAW], [stale, espn, MX_COPY], repoint_map=MX_MAP)
        assert (entry["action"], entry["kept_row_repoint"] if "kept_row_repoint" in entry else "") == ("merge", "")
        assert counts["repoint_done"] == 2

    def test_another_score_is_refused(self):
        (entry,), _ = _plan_dedupe([MX_RAW], [MX_STALE, {**MX_ESPN, "home_score": 2}, MX_COPY], repoint_map=MX_MAP)
        assert entry["action"] == "refuse" and entry["reason"].startswith("scores differ")

    def test_never_folded_across_team_ids(self):
        # the stale row sits exactly on the corrected kickoff and its facts would allow a fold on its own pair
        stale = {**MX_STALE, "match_date": MX_ESPN["match_date"]}
        (entry,), _ = _plan_dedupe([MX_RAW], [stale, MX_COPY], repoint_map=MX_MAP, facts={"a398e0fe": _facts()})
        assert entry["action"] == "refuse" and "filed under other team ids" in entry["reason"]

    def test_the_shift_waits_for_it_and_never_lands_beside_the_re_pointed_row(self):
        entries, counts = _plan_shift(
            [MX_RAW], [MX_STALE, MX_ESPN, MX_COPY], planned_away=frozenset({"df640b1b"}), repoint_map=MX_MAP
        )
        assert entries == [] and counts["left_to_dedupe"] == 1
        # dedupe refused it, say: the row about to move onto its pair is a same-pair row for the shift too
        (entry,), _ = _plan_shift([MX_RAW], [MX_STALE, MX_ESPN, MX_COPY], repoint_map=MX_MAP)
        assert entry["action"] == "refuse" and "UNIQUE collision" in entry["reason"]

    def test_check_counts_the_copy_as_a_duplicate(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger="repair_football_data_kickoffs")
        snap = _snap([MX_RAW], [MX_ESPN, MX_COPY])
        monkeypatch.setattr(rfk, "load_snapshot", lambda cur: snap)
        assert rfk.run_check(_CheckConn([MX_ESPN, MX_COPY]), MX_MAP) == 1
        assert "football_data_duplicates=1" in caplog.text and "repoint_pending=1" in caplog.text


def _alias_entry():
    copy = {**MX_COPY, "kickoff": MX_COPY["match_date"]}
    twin = {**MX_ESPN, "kickoff": MX_ESPN["match_date"], "half_time_home": None, "half_time_away": None}
    return {
        "copy_id": copy["id"],
        "copy": copy,
        "twin": twin,
        "home_score": 3,
        "away_score": 2,
        "ht_write": (3, 1),
        "action": "alias_merge",
    }


class TestApplyAliasMerge:
    def test_archives_then_merges_like_a_merge(self, tmp_path, fake_values):
        entry = _alias_entry()
        conn = _Conn(locked=_locked(entry), archive=tmp_path / "a.jsonl.gz")
        fake_values["returns"] = {"UPDATE stats": [("s1",)], "DELETE": [{"id": "df640b1b"}]}
        done = rfk.apply_dedupe_batch(conn, [entry], 1, FKS, tmp_path / "a.jsonl.gz")
        assert (done["ht_copied"], done["match_stats_moved"], done["copies_deleted"], conn.committed) == (1, 1, 1, 1)
        assert all(conn.writes_after_archive)
        roles = {r["role"] for r in map(json.loads, gzip.open(tmp_path / "a.jsonl.gz", "rt"))}
        assert roles == {"copy", "espn_row_before", "references_copy"}

    def test_a_row_re_pointed_since_the_plan_rolls_back_before_anything_is_written(self, tmp_path, fake_values):
        entry = _alias_entry()
        locked = _locked(entry)
        locked[1]["away_id"] = "t-sl"  # repoint_team_matches moved it in between
        conn = _Conn(locked=locked, archive=tmp_path / "a.jsonl.gz")
        with pytest.raises(rfk.PlanDrift, match="twin e78e7e49 changed"):
            rfk.apply_dedupe_batch(conn, [entry], 1, FKS, tmp_path / "a.jsonl.gz")
        assert conn.rolled_back == 1 and fake_values["calls"] == [] and not (tmp_path / "a.jsonl.gz").exists()

    def test_only_a_club_in_the_kept_match_takes_the_copys_stats(self):
        for sql in (rfk.MOVE_STATS_SQL, rfk.MOVE_COUNTS_SQL):
            assert "s.team_id IN (k.home_team_id, k.away_team_id)" in sql
        assert "m.home_team_id::text AS home_id" in rfk.LOCK_SQL and {"home_id", "away_id"} <= set(rfk.OBSERVED)


class TestRunAliasMerge:
    def test_the_plan_line_and_the_next_step(self, offline, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="repair_football_data_kickoffs")
        offline["snap"] = _snap([MX_RAW], [dict(MX_STALE), dict(MX_ESPN), dict(MX_COPY)])
        counts = rfk.run(
            _Conn(),
            "dedupe",
            True,
            out_csv=str(tmp_path / "d.csv"),
            expect_plan={"merge": 0, "alias_merge": 1, "fold": 0, "refuse": 0},
            repoint_map=MX_MAP,
        )
        assert (counts["planned"], counts["plan_mismatch"], offline["applied"][0][2]) == (1, 0, 1)
        assert "PLAN mode=dedupe merge=0 alias_merge=1 fold=0 refuse=0 fold_unsettled_recs=0" in caplog.text
        assert "NEXT: 1 kept row(s) are still filed under the wrong club (e78e7e49)" in caplog.text
        ((row),) = _rows(tmp_path / "d.csv")
        assert (row["action"], row["kept_row_repoint"]) == ("alias_merge", "away: Santos -> Santos Laguna")

    def test_shift_counts_it_as_dedupe_pending(self, offline, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="repair_football_data_kickoffs")
        offline["snap"] = _snap([MX_RAW], [dict(MX_STALE), dict(MX_ESPN), dict(MX_COPY)])
        counts = rfk.run(_Conn(), "shift", False, out_csv=str(tmp_path / "s.csv"), repoint_map=MX_MAP)
        assert (counts["dedupe_pending"], counts["refused"]) == (1, 0)


class TestFoldUnsettledRecs:
    def test_counted_per_fold_and_on_the_plan_line(self, offline, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="repair_football_data_kickoffs")
        offline["snap"] = _snap([SHANDONG], [dict(STALE), dict(COPY)])
        offline["facts"] = {"stale": _facts(unsettled=2)}
        rfk.run(_Conn(), "dedupe", False, out_csv=str(tmp_path / "d.csv"))
        assert "fold_unsettled_recs=2" in caplog.text and "2 without a profit_loss" in caplog.text
        assert _rows(tmp_path / "d.csv")[0]["fold_unsettled_recs"] == "2"

    def test_an_apply_refuses_folds_holding_them_unless_pinned(self, offline, tmp_path, caplog):
        offline["snap"] = _snap([SHANDONG], [dict(STALE), dict(COPY)])
        offline["facts"] = {"stale": _facts(unsettled=2)}
        pin = {"merge": 0, "alias_merge": 0, "fold": 1, "refuse": 0}
        counts = rfk.run(_Conn(), "dedupe", True, out_csv=str(tmp_path / "d.csv"), expect_plan=pin)
        assert counts["unsettled_unpinned"] == 1 and offline["applied"] == []
        assert "pin fold_unsettled_recs=2" in caplog.text

    def test_a_wrong_unsettled_pin_is_a_plan_mismatch(self, offline, tmp_path):
        offline["snap"] = _snap([SHANDONG], [dict(STALE), dict(COPY)])
        offline["facts"] = {"stale": _facts(unsettled=2)}
        pin = {"merge": 0, "alias_merge": 0, "fold": 1, "refuse": 0, "fold_unsettled_recs": 0}
        counts = rfk.run(_Conn(), "dedupe", True, out_csv=str(tmp_path / "d.csv"), expect_plan=pin)
        assert counts["plan_mismatch"] == 1 and offline["applied"] == []

    def test_the_pinned_count_folds(self, offline, tmp_path):
        offline["snap"] = _snap([SHANDONG], [dict(STALE), dict(COPY)])
        offline["facts"] = {"stale": _facts(unsettled=2)}
        pin = {"merge": 0, "alias_merge": 0, "fold": 1, "refuse": 0, "fold_unsettled_recs": 2}
        counts = rfk.run(_Conn(), "dedupe", True, out_csv=str(tmp_path / "d.csv"), expect_plan=pin)
        assert not counts["unsettled_unpinned"] and not counts["plan_mismatch"] and len(offline["applied"]) == 1

    def test_the_fact_counts_recs_through_predictions_without_a_profit_loss(self):
        assert "br.profit_loss IS NULL AND (br.match_id = m.id OR br.prediction_id IN" in rfk.FOLD_FACTS_SQL
