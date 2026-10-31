"""Unit tests for repair_fixture_orphans — the plan (which orphan merges
where, which is deleted, cancelled or left alone, and why), the soccer no-twin
classes, the safety invariants of its SQL, and the apply loop's guards. No DB."""

from __future__ import annotations

import csv
import importlib.util
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
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


rfo = _load("repair_fixture_orphans", "repair_fixture_orphans.py")

KICK = datetime(2026, 10, 28, 18, 15, tzinfo=timezone.utc)
SOCCER = rfo.policy_for("soccer", False)
SOCCER_PH = rfo.policy_for("soccer", True)
TENNIS = rfo.policy_for("tennis", False)


def _cand(cid, hours_before=1.0, has_recs=False, pending=0, home="fritz", away="hijikata", names=None):
    return {
        "id": cid,
        "league_id": "lg",
        "home_team_id": home,
        "away_team_id": away,
        "home": (names or (home.title(), away.title()))[0],
        "away": (names or (home.title(), away.title()))[1],
        "match_date": KICK - timedelta(hours=hours_before),
        "status": "scheduled",
        "has_recs": has_recs,
        "pending_recs": pending,
    }


def _twin(tid, when=KICK, score=(1, 0)):
    return {
        "twin_id": tid,
        "home_team_id": "fritz",
        "away_team_id": "hijikata",
        "home_score": score[0],
        "away_score": score[1],
        "match_date": when,
    }


def _row(home, away, when, league="lg", names=None):
    """A finished row around a soccer no-twin candidate (NO_TWIN_CONTEXT_SQL)."""
    return {
        "id": f"{home}-{away}-{when:%H%M}",
        "league_id": league,
        "home_team_id": home,
        "away_team_id": away,
        "home": (names or (home.title(), away.title()))[0],
        "away": (names or (home.title(), away.title()))[1],
        "match_date": when,
    }


def _no_twins(c):
    return []


def _espn(twin_id):
    """twin_is_espn: the twin is an ESPN row."""
    return True


def _football_data(twin_id):
    """twin_is_espn: the twin is a football-data row (no ESPN identity)."""
    return False


class TestPlanTennis:
    def test_every_slide_of_one_fixture_merges_onto_its_finished_twin(self):
        cands = [_cand("o1", 38), _cand("o2", 1.25), _cand("o3", 0.75), _cand("o4", 0.17)]
        p = rfo.plan(cands, lambda c: [_twin("real")], TENNIS)
        assert p["groups"] == {"real": ["o1", "o2", "o3", "o4"]}
        assert p["kickoff"]["real"] == KICK
        assert p["counts"]["merge_delete"] == 4 and p["counts"]["twins"] == 1

    def test_a_pairing_that_never_played_is_deleted(self):
        p = rfo.plan([_cand("redrawn")], _no_twins, TENNIS)
        assert p["no_twin"] == ["redrawn"] and p["groups"] == {}
        assert p["counts"]["delete_no_twin"] == 1

    def test_ambiguous_twins_are_left_alone(self):
        twins = [_twin("a", score=(1, 0)), _twin("b", when=KICK + timedelta(hours=2), score=(0, 1))]
        p = rfo.plan([_cand("o")], lambda c: twins, TENNIS)
        assert p["groups"] == {} and p["no_twin"] == []
        assert p["counts"]["left_ambiguous_scores"] == 1

    def test_an_orphan_holding_recs_is_never_touched(self):
        calls = []
        p = rfo.plan([_cand("o", has_recs=True)], lambda c: calls.append(c) or [_twin("real")], TENNIS)
        assert p["groups"] == {} and p["cancel"] == {} and p["no_twin"] == [] and calls == []
        assert p["counts"]["left_has_recs"] == 1

    def test_placeholders_are_deleted_like_any_other_never_played_pairing(self):
        p = rfo.plan([_cand("tbd", names=("TBD", "Sinner"))], _no_twins, TENNIS)
        assert p["placeholders"] == ["tbd"] and p["no_twin"] == []


class TestPlanSoccer:
    def test_a_soccer_row_without_twin_is_never_deleted_and_is_classified(self):
        p = rfo.plan([_cand("never")], _no_twins, SOCCER, context_for=lambda c: [])
        assert p["no_twin"] == [] and p["groups"] == {} and p["placeholders"] == []
        assert p["entries"][0]["action"] == rfo.ACTION_LEAVE
        assert p["entries"][0]["reason"] == rfo.NO_TWIN_NEVER_RESULTED
        assert p["counts"]["left_never_resulted"] == 1

    def test_unpinned_orphan_with_twin_merges_then_deletes(self):
        p = rfo.plan([_cand("o")], lambda c: [_twin("real")], SOCCER, twin_is_espn=_espn)
        assert p["groups"] == {"real": ["o"]} and p["cancel"] == {}

    def test_pinned_orphan_with_twin_merges_then_is_cancelled_not_deleted(self):
        p = rfo.plan([_cand("o", has_recs=True)], lambda c: [_twin("real")], SOCCER, twin_is_espn=_espn)
        assert p["cancel"] == {"real": ["o"]} and p["groups"] == {}
        assert p["entries"][0]["action"] == rfo.ACTION_MERGE_CANCEL
        assert p["counts"]["merge_cancel"] == 1 and p["counts"]["twins"] == 1

    def test_a_pending_rec_blocks_the_cancel_until_repair_stranded_recs_ran(self):
        # 'cancelled' would hide the pending rec from repair_stranded_recs (it reads 'scheduled' only)
        p = rfo.plan([_cand("o", has_recs=True, pending=2)], lambda c: [_twin("real")], SOCCER, twin_is_espn=_espn)
        assert p["cancel"] == {} and p["groups"] == {}
        assert p["entries"][0]["reason"] == rfo.LEFT_PENDING_RECS
        assert "blocked_pending_recs=1" in rfo.plan_line(p["counts"])

    def test_pinned_row_without_twin_is_left_and_counted_as_pinned(self):
        p = rfo.plan([_cand("o", has_recs=True)], _no_twins, SOCCER, context_for=lambda c: [])
        assert p["cancel"] == {} and p["no_twin"] == []
        assert p["counts"]["left_pinned"] == 1 and p["counts"]["left_never_resulted"] == 1

    def test_cutoff_is_the_earlier_of_the_two_kickoffs(self):
        # the twin kicks off later: the orphan's kickoff is the real one
        late_twin = rfo.plan([_cand("o", hours_before=2)], lambda c: [_twin("t")], SOCCER)
        assert late_twin["cutoff"]["o"] == KICK - timedelta(hours=2)
        # ESPN moved the fixture earlier: the twin's kickoff is the real one
        early_twin = rfo.plan([_cand("o", hours_before=-20)], lambda c: [_twin("t")], SOCCER)
        assert early_twin["cutoff"]["o"] == KICK

    def test_observed_values_are_recorded_for_the_guards(self):
        c = _cand("o", has_recs=True)
        p = rfo.plan([c], lambda c: [_twin("t")], SOCCER, twin_is_espn=_espn)
        assert p["observed"]["o"] == ("scheduled", c["match_date"], "fritz", "hijikata", True, 0)


class TestFootballDataCopies:
    """A soccer twin with no ESPN identity exactly 1 h after the orphan is a
    football-data copy at its legacy UK-local kickoff: the orphan holds the
    real slot, and repair_football_data_kickoffs folds the copy into it."""

    @pytest.mark.parametrize("has_recs,pending", [(False, 0), (True, 0), (True, 3)])
    def test_held_with_no_write_pinned_or_not(self, has_recs, pending):
        asked = []
        p = rfo.plan(
            [_cand("o", hours_before=1, has_recs=has_recs, pending=pending)],
            lambda c: [_twin("fd")],
            SOCCER,
            twin_is_espn=lambda t: asked.append(t) or False,
        )
        assert p["groups"] == {} and p["cancel"] == {} and p["kickoff"] == {} and asked == ["fd"]
        e = p["entries"][0]
        assert e["action"] == rfo.ACTION_LEAVE and e["reason"] == rfo.HELD_FD_COPY and e["twin_id"] == "fd"
        assert p["counts"]["left_fd_copy_awaiting_repair"] == 1
        # never a pending-rec block: nothing is cancelled under it
        assert "blocked_pending_recs=0" in rfo.plan_line(p["counts"])

    def test_an_espn_twin_exactly_one_hour_late_merges_and_its_identity_is_pinned(self):
        p = rfo.plan([_cand("o", hours_before=1)], lambda c: [_twin("espn")], SOCCER, twin_is_espn=_espn)
        assert p["groups"] == {"espn": ["o"]} and p["twin_espn"] == {"espn": True}

    @pytest.mark.parametrize("hours_before", [2, 0.5, -1, 1.01])
    def test_any_other_offset_merges_without_asking(self, hours_before):
        asked = []
        p = rfo.plan(
            [_cand("o", hours_before=hours_before)],
            lambda c: [_twin("fd")],
            SOCCER,
            twin_is_espn=lambda t: asked.append(t) or False,
        )
        assert p["groups"] == {"fd": ["o"]} and asked == []

    def test_unknown_identity_holds(self):
        p = rfo.plan([_cand("o", hours_before=1)], lambda c: [_twin("fd")], SOCCER)
        assert p["entries"][0]["reason"] == rfo.HELD_FD_COPY

    def test_tennis_never_holds(self):
        p = rfo.plan([_cand("o", hours_before=1)], lambda c: [_twin("t")], TENNIS, twin_is_espn=_football_data)
        assert p["groups"] == {"t": ["o"]}

    def test_espn_identity_mirrors_promote_raw(self):
        assert rfo.has_espn_identity("401896777", None)
        assert rfo.has_espn_identity(None, "espn_scoreboard")
        assert not rfo.has_espn_identity(None, None) and not rfo.has_espn_identity("", "football_data")


class TestPlaceholders:
    @pytest.mark.parametrize(
        "names",
        [("Round of 32 1 Winner", "Group A 2nd Place"), ("Spain", "Semifinal 2 Loser"), ("TBD", "Brazil")],
    )
    def test_bracket_slots_are_placeholders(self, names):
        assert rfo.is_placeholder(_cand("x", names=names))

    def test_a_team_against_itself_is_a_placeholder(self):
        assert rfo.is_placeholder(_cand("x", home="spain", away="spain", names=("Spain", "Spain")))

    def test_a_real_fixture_is_not(self):
        assert not rfo.is_placeholder(_cand("x", names=("Atlético Nacional", "Independiente Santa Fe")))

    def test_soccer_deletes_them_only_with_the_flag_and_never_merges_them(self):
        calls = []
        ph = _cand("ph", names=("Round of 16 3 Winner", "Quarterfinal 1 Winner"))
        without = rfo.plan([ph], lambda c: calls.append(c) or [_twin("t")], SOCCER)
        assert without["placeholders"] == [] and without["groups"] == {}
        assert without["entries"][0]["reason"] == rfo.LEFT_PLACEHOLDER
        flagged = rfo.plan([ph], lambda c: calls.append(c) or [_twin("t")], SOCCER_PH)
        assert flagged["placeholders"] == ["ph"] and flagged["counts"]["delete_placeholder"] == 1
        assert calls == []

    def test_one_a_rec_references_is_kept(self):
        ph = _cand("ph", has_recs=True, names=("Final Winner", "Spain"))
        p = rfo.plan([ph], _no_twins, SOCCER_PH)
        assert p["placeholders"] == [] and p["entries"][0]["reason"] == rfo.LEFT_PLACEHOLDER_HAS_RECS


class TestClassifyNoTwin:
    C = _cand("o", hours_before=0, home="aik", away="sirius", names=("AIK", "IK Sirius"))

    def test_flipped(self):
        rows = [_row("sirius", "aik", KICK + timedelta(hours=30))]
        assert rfo.classify_no_twin(self.C, rows, 72) == rfo.NO_TWIN_FLIPPED

    def test_football_data_row_under_other_team_ids_one_hour_late(self):
        rows = [_row("aik2", "sirius2", KICK + timedelta(hours=1), names=("AIK", "Sirius"))]
        assert rfo.classify_no_twin(self.C, rows, 72) == rfo.NO_TWIN_OTHER_IDENTITY

    def test_one_side_renamed_at_the_same_kickoff(self):
        rows = [_row("aik", "ik-sirius-fc", KICK, names=("AIK", "Sirius Uppsala"))]
        assert rfo.classify_no_twin(self.C, rows, 72) == rfo.NO_TWIN_OTHER_IDENTITY

    def test_another_match_of_the_same_league_at_the_same_time_is_not_it(self):
        rows = [_row("malmo", "halmstad", KICK, names=("Malmö FF", "Halmstads BK"))]
        assert rfo.classify_no_twin(self.C, rows, 72) == rfo.NO_TWIN_NEVER_RESULTED

    def test_a_club_s_next_match_days_later_is_not_it(self):
        rows = [_row("aik", "gais", KICK + timedelta(days=3), names=("AIK", "GAIS"))]
        assert rfo.classify_no_twin(self.C, rows, 72) == rfo.NO_TWIN_NEVER_RESULTED

    def test_other_league_rows_never_count_as_another_identity(self):
        rows = [_row("aik", "x", KICK, league="cup", names=("AIK", "X"))]
        assert rfo.classify_no_twin(self.C, rows, 72) == rfo.NO_TWIN_NEVER_RESULTED

    def test_postponed_and_replayed_later(self):
        rows = [_row("aik", "sirius", KICK + timedelta(days=24))]
        assert rfo.classify_no_twin(self.C, rows, 72) == rfo.NO_TWIN_REPLAYED_LATER

    def test_a_meeting_beyond_the_look_ahead_is_not_a_replay(self):
        rows = [_row("aik", "sirius", KICK + timedelta(days=rfo.REPLAY_LOOKAHEAD_DAYS + 1))]
        assert rfo.classify_no_twin(self.C, rows, 72) == rfo.NO_TWIN_NEVER_RESULTED

    def test_an_earlier_meeting_is_not_a_replay(self):
        rows = [_row("aik", "sirius", KICK - timedelta(days=24))]
        assert rfo.classify_no_twin(self.C, rows, 72) == rfo.NO_TWIN_NEVER_RESULTED


class TestSameClub:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("Örgryte IS", "Orgryte"),
            ("Vasco da Gama", "Vasco"),
            ("Halmstads BK", "Halmstad"),
            ("SK Brann", "Brann"),
            ("Sarpsborg FK", "Sarpsborg 08"),
            ("IFK Göteborg", "Goteborg"),
        ],
    )
    def test_spellings_of_one_club(self, a, b):
        assert rfo.same_club(a, b)

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Universidad de Chile", "Universidad Católica"),
            ("Atlético Nacional", "Atlético Junior"),
            ("Malmö FF", "Halmstads BK"),
            ("", "AIK"),
        ],
    )
    def test_different_clubs(self, a, b):
        assert not rfo.same_club(a, b)


class TestSqlInvariants:
    def test_only_rows_before_each_orphans_cutoff_move(self):
        assert '"timestamp" < cut.cutoff' in rfo.MOVE_ODDS_SQL
        assert "captured_at < cut.cutoff" in rfo.MOVE_SNAPSHOTS_SQL
        assert "COALESCE(p.updated_at, p.created_at) < cut.cutoff" in rfo.MOVE_PREDICTIONS_SQL
        for sql in (rfo.MOVE_ODDS_SQL, rfo.MOVE_SNAPSHOTS_SQL, rfo.MOVE_PREDICTIONS_SQL):
            assert "unnest(%(orphans)s::uuid[], %(cutoffs)s::timestamptz[])" in sql

    def test_the_twins_own_rows_always_win(self):
        for sql in (rfo.MOVE_ODDS_SQL, rfo.MOVE_SNAPSHOTS_SQL, rfo.MOVE_PREDICTIONS_SQL):
            assert "NOT EXISTS" in sql and "t.match_id = %(twin)s::uuid" in sql

    def test_odds_keep_the_freshest_price_per_current_price_key(self):
        # the key mirrors uq_odds_current_price_key, so a move can never violate it
        assert "COALESCE(o.line, -1), COALESCE(o.is_opening, false), COALESCE(o.is_live, false)" in rfo.MOVE_ODDS_SQL
        assert 'o."timestamp" DESC' in rfo.MOVE_ODDS_SQL

    def test_a_prediction_a_rec_references_never_moves(self):
        assert "r.prediction_id = p.id" in rfo.MOVE_PREDICTIONS_SQL

    def test_the_rec_filter_runs_before_distinct_on_picks(self):
        # Mixed delete+cancel group: if the newest pre-game prediction for a key is
        # rec-referenced (on the merge_cancel sibling) and the filter ran AFTER the
        # pick, nothing would move for that key and the older movable prediction on
        # the merge_delete sibling would be cascade-deleted with its orphan.
        sql = rfo.MOVE_PREDICTIONS_SQL
        pick, update = sql.split("UPDATE predictions p", 1)
        cand = pick.split("cand AS (", 1)[1]
        rec_filter = "NOT EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.prediction_id = p.id)"
        assert rec_filter in cand and cand.index(rec_filter) < cand.index("ORDER BY")
        assert "betting_recommendations" not in update

    def test_delete_spares_recs_history_and_played_rows(self):
        sql = rfo.DELETE_ORPHANS_SQL
        assert "betting_recommendations" in sql and "betting_history" in sql
        assert "m.status IN ('scheduled', 'cancelled')" in sql

    def test_delete_is_guarded_on_the_observed_values(self):
        for guard in (
            "m.status = obs.status",
            "m.match_date = obs.match_date",
            "m.home_team_id = obs.home_team_id",
            "m.away_team_id = obs.away_team_id",
        ):
            assert guard in rfo.DELETE_ORPHANS_SQL

    def test_a_rec_whose_prediction_lives_on_the_orphan_protects_it(self):
        # prediction_id was ON DELETE CASCADE: deleting the orphan deleted a rec
        # that was already re-pointed onto the twin (2026-10-31: 45 lost)
        guard = "JOIN predictions p ON p.id = r.prediction_id"
        assert guard in rfo.DELETE_ORPHANS_SQL and "WHERE p.match_id = m.id" in rfo.DELETE_ORPHANS_SQL
        assert guard in rfo.LIST_CANDIDATES_SQL and guard in rfo.PINS_SQL

    def test_cancel_is_guarded_and_never_runs_twice_or_under_a_pending_rec(self):
        sql = rfo.CANCEL_PINNED_SQL
        assert "SET status = 'cancelled'" in sql and "'orphan_of', %(twin)s::text" in sql
        for guard in (
            "m.status = 'scheduled'",
            "m.match_date = %(match_date)s",
            "m.home_team_id = %(home)s::uuid",
            "m.away_team_id = %(away)s::uuid",
            "NOT (COALESCE(m.metadata, '{}'::jsonb) ? 'orphan_of')",
            "r.status = 'pending'",
        ):
            assert guard in sql
        assert "betting_recommendations" not in sql.split("WHERE", 1)[0]  # never writes a rec

    def test_soccer_candidates_are_scheduled_rows_only(self):
        # a soccer 'cancelled' row is an abandoned match or an orphan already retired
        assert rfo.CANDIDATE_STATUSES["soccer"] == ["scheduled"]
        assert "cancelled" in rfo.CANDIDATE_STATUSES["tennis"]

    def test_candidates_are_measured_from_the_as_of_instant(self):
        sql = rfo.LIST_CANDIDATES_SQL
        assert "m.match_date < %s::timestamptz - (%s || ' hours')::interval" in sql and "NOW()" not in sql

    def test_mark_postponed_is_guarded_and_stamps_what_the_grader_reads(self):
        sql = rfo.MARK_POSTPONED_SQL
        assert "SET status = 'postponed'" in sql
        # the keys fetch_upcoming._record_status_change stamps and grade_completed_matches reads
        assert "'postponed_kickoff', %(postponed_kickoff)s::text" in sql and "'postponed_at', %(postponed_at)s" in sql
        for guard in (
            "m.status = 'scheduled'",
            "m.match_date = %(match_date)s",
            "m.home_team_id = %(home)s::uuid",
            "m.away_team_id = %(away)s::uuid",
            "NOT (COALESCE(m.metadata, '{}'::jsonb) ? 'postponed_kickoff')",
            "t.id = %(replay)s::uuid",
            "t.status = 'finished'",
            "t.home_team_id = m.home_team_id",
            "t.away_team_id = m.away_team_id",
            "t.match_date = %(replay_kickoff)s",
        ):
            assert guard in sql

    def test_no_statement_writes_a_recommendation(self):
        for name in dir(rfo):
            if name.endswith("_SQL"):
                sql = getattr(rfo, name).upper()
                assert "UPDATE BETTING_RECOMMENDATIONS" not in sql and "DELETE FROM BETTING_RECOMMENDATIONS" not in sql


class _Cur:
    """Fake cursor: answers the lock queries from a table of current rows,
    records every statement, and returns scripted DELETE / UPDATE rowcounts."""

    def __init__(self, rows, twin=None, delete_rowcounts=(), cancel_rowcounts=(), postpone_rowcounts=()):
        self.rows = rows  # id -> current row (status, match_date, teams, has_recs, pending_recs)
        self.twin = twin
        self.delete_rowcounts = list(delete_rowcounts)
        self.cancel_rowcounts = list(cancel_rowcounts)
        self.postpone_rowcounts = list(postpone_rowcounts)
        self.executed = []
        self.rowcount = 0
        self._result = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        self.rowcount, self._result = 0, []
        if sql is rfo.LOCK_TWIN_SQL:
            self._result = [self.twin] if self.twin else []
        elif sql is rfo.LOCK_ORPHANS_SQL or sql is rfo.PINS_SQL:
            self._result = [dict(self.rows[i], id=i) for i in params[0] if i in self.rows]
        elif sql is rfo.DELETE_ORPHANS_SQL:
            self.rowcount = self.delete_rowcounts.pop(0)
        elif sql is rfo.CANCEL_PINNED_SQL:
            self.rowcount = self.cancel_rowcounts.pop(0)
        elif sql is rfo.MARK_POSTPONED_SQL:
            self.rowcount = self.postpone_rowcounts.pop(0)

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)

    def ran(self, sql):
        return [p for s, p in self.executed if s is sql]


class _Conn:
    def __init__(self, cur):
        self.cur = cur
        self.commits = 0

    def cursor(self):
        return self.cur

    def commit(self):
        self.commits += 1


# The twin as LOCK_TWIN_SQL returns it: finished, at KICK, an ESPN row.
_LOCKED_TWIN = {"status": "finished", "match_date": KICK, "espn_id": "401", "result_source": None}


def _row_state(c):
    return {k: c[k] for k in ("status", "match_date", "home_team_id", "away_team_id", "has_recs", "pending_recs")}


class TestApplyPlan:
    def _plan(self, cands, policy=SOCCER):
        return rfo.plan(cands, lambda c: [_twin("real")], policy, twin_is_espn=_espn)

    def test_merge_moves_with_per_orphan_cutoffs_then_deletes_and_cancels(self):
        a, b = _cand("a", 2), _cand("b", -3, has_recs=True)
        p = self._plan([a, b])
        cur = _Cur(
            {"a": _row_state(a), "b": _row_state(b)},
            twin=_LOCKED_TWIN,
            delete_rowcounts=[1],
            cancel_rowcounts=[1],
        )
        done = rfo.apply_plan(_Conn(cur), p, batch_size=10)
        moves = cur.ran(rfo.MOVE_ODDS_SQL)
        assert moves == [{"orphans": ["a", "b"], "cutoffs": [a["match_date"], KICK], "twin": "real"}]
        assert cur.ran(rfo.DELETE_ORPHANS_SQL)[0]["ids"] == ["a"]
        cancel = cur.ran(rfo.CANCEL_PINNED_SQL)[0]
        assert cancel["id"] == "b" and cancel["twin"] == "real" and cancel["match_date"] == b["match_date"]
        assert done["deleted_merged_orphans"] == 1 and done["cancelled_pinned_orphans"] == 1
        assert rfo.refused(done) == 0

    def test_an_orphan_that_changed_since_the_plan_is_not_touched(self, caplog):
        a, b = _cand("a", 2), _cand("b", 1)
        p = self._plan([a, b])
        moved = dict(_row_state(b), match_date=b["match_date"] + timedelta(minutes=5))
        cur = _Cur({"a": _row_state(a), "b": moved}, twin=_LOCKED_TWIN, delete_rowcounts=[1])
        done = rfo.apply_plan(_Conn(cur), p, batch_size=10)
        assert cur.ran(rfo.MOVE_ODDS_SQL)[0]["orphans"] == ["a"]
        assert cur.ran(rfo.DELETE_ORPHANS_SQL)[0]["ids"] == ["a"]
        assert done["refused_changed_since_plan"] == 1 and "changed since the plan" in caplog.text

    def test_a_rec_appearing_after_the_plan_stops_the_merge(self):
        a = _cand("a", 2)
        p = self._plan([a])
        cur = _Cur({"a": dict(_row_state(a), has_recs=True)}, twin=_LOCKED_TWIN)
        done = rfo.apply_plan(_Conn(cur), p, batch_size=10)
        assert cur.ran(rfo.MOVE_ODDS_SQL) == [] and cur.ran(rfo.DELETE_ORPHANS_SQL) == []
        assert done["refused_changed_since_plan"] == 1

    def test_a_twin_that_moved_stops_its_whole_group(self):
        a = _cand("a", 2)
        p = self._plan([a])
        cur = _Cur({"a": _row_state(a)}, twin={"status": "finished", "match_date": KICK + timedelta(hours=1)})
        done = rfo.apply_plan(_Conn(cur), p, batch_size=10)
        assert cur.ran(rfo.LOCK_ORPHANS_SQL) == [] and cur.ran(rfo.MOVE_ODDS_SQL) == []
        assert done["refused_changed_since_plan"] == 1

    def test_refused_deletes_and_cancels_are_counted_not_silent(self, caplog):
        a, b = _cand("a", 2), _cand("b", 1, has_recs=True)
        nt = [_cand(f"x{i}", names=("TBD", f"P{i}")) for i in range(3)]
        p = rfo.plan([a, b, *nt], lambda c: [_twin("real")], rfo.policy_for("soccer", True), twin_is_espn=_espn)
        rows = {c["id"]: _row_state(c) for c in [a, b, *nt]}
        cur = _Cur(rows, twin=_LOCKED_TWIN, delete_rowcounts=[0, 3], cancel_rowcounts=[0])
        done = rfo.apply_plan(_Conn(cur), p, batch_size=10)
        assert done["delete_refused_has_recs"] == 1 and done["cancel_refused"] == 1
        assert done["deleted_placeholder_orphans"] == 3
        assert rfo.refused(done) == 2
        assert "delete guard kept 1" in caplog.text and "cancel guard refused" in caplog.text

    def test_a_mixed_delete_and_cancel_group_moves_once_then_deletes_one_and_cancels_the_other(self):
        # one twin, a merge_delete and a merge_cancel sibling: ONE predictions move over both orphans
        # (the rec filter inside its candidate pick decides per key, see TestSqlInvariants), then
        # the unpinned orphan is deleted and the pinned one cancelled
        d, k = _cand("del", 3), _cand("pin", 2, has_recs=True)
        p = self._plan([d, k])
        assert p["groups"] == {"real": ["del"]} and p["cancel"] == {"real": ["pin"]}
        cur = _Cur(
            {"del": _row_state(d), "pin": _row_state(k)},
            twin=_LOCKED_TWIN,
            delete_rowcounts=[1],
            cancel_rowcounts=[1],
        )
        done = rfo.apply_plan(_Conn(cur), p, batch_size=10)
        (moves,) = cur.ran(rfo.MOVE_PREDICTIONS_SQL)
        assert moves["orphans"] == ["del", "pin"] and moves["twin"] == "real"
        assert [s for s, _ in cur.executed].index(rfo.MOVE_PREDICTIONS_SQL) < [s for s, _ in cur.executed].index(
            rfo.DELETE_ORPHANS_SQL
        )
        assert cur.ran(rfo.DELETE_ORPHANS_SQL)[0]["ids"] == ["del"]
        assert cur.ran(rfo.CANCEL_PINNED_SQL)[0]["id"] == "pin"
        assert done["deleted_merged_orphans"] == 1 and done["cancelled_pinned_orphans"] == 1

    def test_a_twin_whose_espn_identity_changed_since_the_plan_stops_its_group(self):
        a = _cand("a", 1)  # exactly 1 h before an ESPN twin: merged only because the twin is ESPN's
        p = self._plan([a])
        assert p["twin_espn"] == {"real": True}
        cur = _Cur(
            {"a": _row_state(a)},
            twin={"status": "finished", "match_date": KICK, "espn_id": None, "result_source": None},
        )
        done = rfo.apply_plan(_Conn(cur), p, batch_size=10)
        assert cur.ran(rfo.MOVE_ODDS_SQL) == [] and done["refused_changed_since_plan"] == 1
        ok = _Cur(
            {"a": _row_state(a)},
            twin={"status": "finished", "match_date": KICK, "espn_id": "401", "result_source": None},
            delete_rowcounts=[1],
        )
        assert rfo.refused(rfo.apply_plan(_Conn(ok), p, batch_size=10)) == 0


POSTPONED_CSV = SCRIPTS_DIR / "data" / "soccer_postponed_stale_rows.csv"
REPLAY_KICK = KICK + timedelta(days=18, minutes=15)


def _listed(cid="pp", kickoff=None, replay="replay", replay_kickoff=REPLAY_KICK):
    return rfo.Postponement(
        match_id=cid,
        kickoff=kickoff or KICK,
        replay_match_id=replay,
        replay_kickoff=replay_kickoff,
        replay_espn_event="401896777",
        evidence="ESPN moved the event out of its round",
    )


def _replay_row(rid="replay", when=REPLAY_KICK, home="fritz", away="hijikata"):
    return dict(_row(home, away, when), id=rid)


def _write_list(path, rows, fields=rfo.POSTPONED_LIST_FIELDS):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return path


_LIST_ROW = {
    "match_id": "pp",
    "kickoff": "2026-10-28T18:15:00+00:00",
    "replay_match_id": "replay",
    "replay_kickoff": "2026-11-15T18:30:00Z",
    "replay_espn_event": "401896777",
    "evidence": "ESPN moved the event out of its round",
}


class TestPostponedList:
    def test_the_reviewed_list_loads_and_every_row_carries_its_espn_evidence(self):
        listed = rfo.load_postponed_list(POSTPONED_CSV)
        assert len(listed) == 6
        assert (
            listed["0ea6d224-7545-4992-a897-0bb302f53a76"].replay_espn_event == "401896777"
        )  # Panathinaikos v Kifisia
        for p in listed.values():
            assert p.replay_espn_event.isdigit() and "ESPN" in p.evidence
            assert p.kickoff.utcoffset() == timedelta(0) and p.replay_kickoff - p.kickoff > timedelta(hours=72)

    def test_loads_a_valid_list(self, tmp_path):
        listed = rfo.load_postponed_list(_write_list(tmp_path / "l.csv", [_LIST_ROW]))
        assert listed["pp"] == rfo.Postponement(
            "pp",
            datetime(2026, 10, 28, 18, 15, tzinfo=timezone.utc),
            "replay",
            datetime(2026, 11, 15, 18, 30, tzinfo=timezone.utc),
            "401896777",
            "ESPN moved the event out of its round",
        )

    @pytest.mark.parametrize(
        "rows,fields",
        [
            ([_LIST_ROW], [f for f in rfo.POSTPONED_LIST_FIELDS if f != "replay_espn_event"]),
            ([dict(_LIST_ROW, evidence=" ")], rfo.POSTPONED_LIST_FIELDS),
            ([dict(_LIST_ROW, replay_espn_event="")], rfo.POSTPONED_LIST_FIELDS),
            ([_LIST_ROW, _LIST_ROW], rfo.POSTPONED_LIST_FIELDS),
            ([dict(_LIST_ROW, kickoff="2026-10-28T18:15:00")], rfo.POSTPONED_LIST_FIELDS),
            ([dict(_LIST_ROW, replay_kickoff="soon")], rfo.POSTPONED_LIST_FIELDS),
            ([dict(_LIST_ROW, replay_kickoff="2026-10-27T18:15:00Z")], rfo.POSTPONED_LIST_FIELDS),
        ],
    )
    def test_refuses_a_list_it_cannot_trust(self, tmp_path, rows, fields):
        with pytest.raises(rfo.UnsafeRequest):
            rfo.load_postponed_list(_write_list(tmp_path / "l.csv", rows, fields))

    def _plan(self, c, listed, twins=(), context=None):
        return rfo.plan(
            [c],
            lambda _: list(twins),
            SOCCER,
            context_for=lambda _: [_replay_row()] if context is None else context,
            twin_is_espn=_espn,
            postponed={c["id"]: listed},
        )

    def test_a_listed_row_with_its_replay_is_marked_postponed(self):
        c = _cand("pp", hours_before=0, has_recs=True, pending=5)
        p = self._plan(c, _listed())
        e = p["entries"][0]
        assert e["action"] == rfo.ACTION_MARK_POSTPONED and e["twin_id"] == "replay"
        assert p["postponed"] == ["pp"] and p["postponed_meta"]["pp"] == _listed()
        assert p["groups"] == {} and p["cancel"] == {} and p["mismatches"] == []
        # pending recs never block it: the grader voids them once the row is 'postponed'
        assert "mark_postponed=1" in rfo.plan_line(p["counts"]) and "blocked_pending_recs=0" in rfo.plan_line(
            p["counts"]
        )

    @pytest.mark.parametrize(
        "case",
        [
            "twin",
            "ambiguous",
            "kickoff",
            "no_replay",
            "other_pairing",
            "replay_kickoff",
            "inside_window",
            "placeholder",
        ],
    )
    def test_a_listed_row_that_no_longer_matches_is_left_and_reported(self, case):
        c = _cand("pp", hours_before=0)
        listed, twins, context = _listed(), [], None
        if case == "twin":
            twins = [_twin("fork", when=KICK + timedelta(hours=3))]
        elif case == "ambiguous":
            twins = [_twin("a", score=(1, 0)), _twin("b", when=KICK + timedelta(hours=2), score=(0, 1))]
        elif case == "kickoff":
            listed = _listed(kickoff=KICK + timedelta(minutes=15))
        elif case == "no_replay":
            context = []
        elif case == "other_pairing":
            context = [_replay_row(home="hijikata", away="fritz")]
        elif case == "replay_kickoff":
            context = [_replay_row(when=REPLAY_KICK + timedelta(hours=1))]
        elif case == "inside_window":
            listed = _listed(replay_kickoff=KICK + timedelta(hours=30))
            context = [_replay_row(when=KICK + timedelta(hours=30))]
        elif case == "placeholder":
            c = _cand("pp", hours_before=0, names=("TBD", "Brazil"))
        p = self._plan(c, listed, twins, context)
        assert p["postponed"] == [] and p["groups"] == {} and p["placeholders"] == []
        assert p["entries"][0]["reason"] == rfo.LEFT_POSTPONED_MISMATCH and len(p["mismatches"]) == 1

    def test_apply_marks_it_guarded_with_the_metadata_the_grader_reads(self):
        c = _cand("pp", hours_before=0, has_recs=True, pending=5)
        p = self._plan(c, _listed())
        cur = _Cur({"pp": _row_state(c)}, postpone_rowcounts=[1])
        done = rfo.apply_plan(_Conn(cur), p, batch_size=10)
        (params,) = cur.ran(rfo.MARK_POSTPONED_SQL)
        assert params["id"] == "pp" and params["match_date"] == KICK and params["home"] == "fritz"
        assert params["postponed_kickoff"] == KICK.isoformat() and params["postponed_kickoff"].endswith("+00:00")
        assert datetime.fromisoformat(params["postponed_at"]) > KICK - timedelta(days=1)
        assert params["replay"] == "replay" and params["replay_kickoff"] == REPLAY_KICK
        assert params["espn_event"] == "401896777" and params["evidence"]
        assert done["marked_postponed"] == 1 and rfo.refused(done) == 0
        assert cur.ran(rfo.DELETE_ORPHANS_SQL) == [] and cur.ran(rfo.MOVE_ODDS_SQL) == []

    def test_a_refused_mark_is_counted_not_silent(self, caplog):
        c = _cand("pp", hours_before=0)
        p = self._plan(c, _listed())
        done = rfo.apply_plan(_Conn(_Cur({"pp": _row_state(c)}, postpone_rowcounts=[0])), p, batch_size=10)
        assert done["postpone_refused"] == 1 and rfo.refused(done) == 1 and "postpone guard refused" in caplog.text

    def test_a_row_that_changed_since_the_plan_is_not_marked(self):
        c = _cand("pp", hours_before=0, has_recs=True, pending=5)
        p = self._plan(c, _listed())
        cur = _Cur({"pp": dict(_row_state(c), pending_recs=4)})
        done = rfo.apply_plan(_Conn(cur), p, batch_size=10)
        assert cur.ran(rfo.MARK_POSTPONED_SQL) == [] and done["refused_changed_since_plan"] == 1


class TestExpectPlan:
    def test_parses_write_actions(self):
        assert rfo.parse_expect_plan("merge_delete=203, merge_cancel=28") == {"merge_delete": 203, "merge_cancel": 28}

    @pytest.mark.parametrize("text", ["", "merge=3", "merge_delete=x", "left=153", "merge_delete"])
    def test_refuses_anything_else(self, text):
        with pytest.raises(rfo.UnsafeRequest):
            rfo.parse_expect_plan(text)

    def test_a_drifted_plan_is_refused(self):
        counts = Counter(merge_delete=204, merge_cancel=28)
        rfo.check_expected(counts, {"merge_cancel": 28})
        with pytest.raises(rfo.UnsafeRequest, match="merge_delete 203 != 204"):
            rfo.check_expected(counts, {"merge_delete": 203, "merge_cancel": 28})

    def test_plan_line_lists_every_write_action(self):
        line = rfo.plan_line(Counter(merge_delete=2, left_flipped=1, left_pinned=1, left_never_resulted=3))
        assert line == (
            "PLAN merge_delete=2 merge_cancel=0 delete_no_twin=0 delete_placeholder=0 mark_postponed=0 left=4 "
            "blocked_pending_recs=0 list_mismatch=0"
        )

    def test_expect_plan_pins_the_postponed_marks_too(self):
        assert rfo.parse_expect_plan("mark_postponed=6") == {"mark_postponed": 6}


class TestAuditFiles:
    def test_csv_is_write_once_and_in_utc(self, tmp_path):
        p = rfo.plan([_cand("o")], lambda c: [_twin("real")], SOCCER, twin_is_espn=_espn)
        out = rfo.write_csv(tmp_path / "plan.csv", "soccer", p["entries"])
        row = next(csv.DictReader(out.open()))
        assert row["action"] == "merge_delete" and row["sport"] == "soccer" and row["twin_id"] == "real"
        assert row["cutoff"] == (KICK - timedelta(hours=1)).isoformat()
        with pytest.raises(FileExistsError):
            rfo.write_csv(out, "soccer", p["entries"])

    def test_archive_is_write_once(self, tmp_path):
        existing = tmp_path / "a.jsonl.gz"
        existing.write_bytes(b"")
        with pytest.raises(FileExistsError):
            rfo.write_archive(None, existing, ["o"], {}, {})

    def test_default_paths_honour_backup_local_dir_and_name_the_mode(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BACKUP_LOCAL_DIR", str(tmp_path))
        dry_csv, archive = rfo.default_paths("soccer", apply=False)
        apply_csv, _ = rfo.default_paths("soccer", apply=True)
        assert dry_csv.parent == tmp_path and "_dryrun_" in dry_csv.name and "_apply_" in apply_csv.name
        assert archive.name.endswith(".jsonl.gz")


class TestArgs:
    def test_dry_run_is_the_default(self):
        args = rfo.parse_args(["--sport", "soccer"])
        assert args.apply is False and args.delete_placeholders is False and args.expected is None

    def test_cannot_race_the_results_look_back(self):
        with pytest.raises(rfo.UnsafeRequest):
            rfo.parse_args(["--sport", "tennis", "--stale-after-hours", "24"])

    def test_twin_window_is_bounded(self):
        with pytest.raises(rfo.UnsafeRequest):
            rfo.parse_args(["--sport", "tennis", "--twin-window-hours", "500"])

    def test_other_team_sports_are_refused(self):
        with pytest.raises(SystemExit):
            rfo.parse_args(["--sport", "nba"])

    def test_as_of_defaults_to_now_and_must_be_zoned_and_past(self):
        before = datetime.now(timezone.utc)
        assert before <= rfo.parse_args(["--sport", "soccer"]).as_of <= datetime.now(timezone.utc)
        assert rfo.parse_args(["--sport", "soccer", "--as-of", "2026-10-30T12:00:00Z"]).as_of == datetime(
            2026, 10, 30, 12, tzinfo=timezone.utc
        )
        for bad in ("2026-10-31T12:00:00", "yesterday", (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()):
            with pytest.raises(rfo.UnsafeRequest):
                rfo.parse_args(["--sport", "soccer", "--as-of", bad])

    def test_postponed_list_is_soccer_only(self, tmp_path):
        with pytest.raises(rfo.UnsafeRequest):
            rfo.parse_args(["--sport", "tennis", "--postponed-list", str(tmp_path / "x.csv")])

    def test_expect_plan_is_parsed_up_front(self):
        assert rfo.parse_args(["--sport", "soccer", "--expect-plan", "merge_cancel=28"]).expected == {
            "merge_cancel": 28
        }
        with pytest.raises(rfo.UnsafeRequest):
            rfo.parse_args(["--sport", "soccer", "--expect-plan", "everything=1"])


class _MainDB:
    """Just enough of psycopg2 for main(): the candidate list, find_twins and
    the no-twin context; any write statement fails the test."""

    def __init__(self, candidates, twins, espn=(), context=None, listed_state=()):
        self.candidates, self.twins = candidates, twins
        self.espn = set(espn)  # twin ids with ESPN identity
        self.context = context or {}  # candidate id -> NO_TWIN_CONTEXT_SQL rows
        self.listed_state = list(listed_state)  # LISTED_STATE_SQL rows
        self.cur = self
        self.rowcount = 0
        self._result = []
        self.writes = []
        self.params = {}

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.params.setdefault(sql, []).append(params)
        if sql is rfo.LIST_CANDIDATES_SQL:
            self._result = self.candidates
        elif sql is rfo.NO_TWIN_CONTEXT_SQL:
            self._result = self.context.get(params["id"], [])
        elif sql is rfo.TWIN_IDENTITY_SQL:
            espn = params[0] in self.espn
            self._result = [{"espn_id": "401" if espn else None, "result_source": None}]
        elif sql is rfo.LISTED_STATE_SQL:
            self._result = [r for r in self.listed_state if r["id"] in params[0]]
        elif "FROM matches t" in sql:  # repair_stranded_recs.find_twins
            self._result = self.twins.get(params[2], [])
        else:
            self.writes.append(sql)
            self._result = []

    def fetchall(self):
        return [dict(r) for r in self._result]

    def fetchone(self):
        return dict(self._result[0]) if self._result else None

    def rollback(self):
        pass

    def close(self):
        pass


_LIST_REPLAY = datetime(2026, 11, 15, 18, 30, tzinfo=timezone.utc)


class TestMain:
    def _run(self, monkeypatch, tmp_path, db, *argv):
        monkeypatch.setenv("BACKUP_LOCAL_DIR", str(tmp_path))
        monkeypatch.setattr(rfo.psycopg2, "connect", lambda *a, **k: db)
        return rfo.main(["--sport", "soccer", "--database-url", "x", *argv])

    def _twin_row(self):
        return {**_twin("real"), "twin_id": "real"}

    def test_dry_run_writes_the_audit_csv_and_nothing_else(self, monkeypatch, tmp_path):
        db = _MainDB([_cand("o"), _cand("never", home="a", away="b")], {"o": [self._twin_row()]}, espn={"real"})
        assert self._run(monkeypatch, tmp_path, db) == 0
        (out,) = tmp_path.glob("fixture_orphans_soccer_dryrun_*.csv")
        assert {r["match_id"]: r["action"] for r in csv.DictReader(out.open())} == {
            "o": "merge_delete",
            "never": "leave",
        }
        assert db.writes == [] and not list(tmp_path.glob("*.jsonl.gz"))

    def test_apply_refuses_while_a_pending_rec_sits_on_a_pinned_orphan(self, monkeypatch, tmp_path):
        db = _MainDB([_cand("o", has_recs=True, pending=1)], {"o": [self._twin_row()]}, espn={"real"})
        with pytest.raises(rfo.UnsafeRequest, match="repair_stranded_recs"):
            self._run(monkeypatch, tmp_path, db, "--apply")
        assert db.writes == [] and not list(tmp_path.iterdir())

    def test_apply_refuses_a_plan_that_drifted_from_the_dry_run(self, monkeypatch, tmp_path):
        db = _MainDB([_cand("o")], {"o": [self._twin_row()]}, espn={"real"})
        with pytest.raises(rfo.UnsafeRequest, match="differs from --expect-plan"):
            self._run(monkeypatch, tmp_path, db, "--apply", "--expect-plan", "merge_delete=2")
        assert db.writes == [] and not list(tmp_path.iterdir())

    def test_the_twins_espn_identity_decides_a_one_hour_twin(self, monkeypatch, tmp_path):
        for espn, action in (({"real"}, "merge_delete"), (set(), "leave")):
            out = tmp_path / ("espn" if espn else "fd")
            db = _MainDB([_cand("o", hours_before=1)], {"o": [self._twin_row()]}, espn=espn)
            assert self._run(monkeypatch, out, db) == 0
            (csv_path,) = out.glob("fixture_orphans_soccer_dryrun_*.csv")
            (row,) = csv.DictReader(csv_path.open())
            assert row["action"] == action and db.params[rfo.TWIN_IDENTITY_SQL] == [("real",)]
            if action == "leave":
                assert row["reason"] == rfo.HELD_FD_COPY

    def test_a_listed_candidate_is_planned_as_mark_postponed(self, monkeypatch, tmp_path):
        lst = _write_list(tmp_path / "list.csv", [dict(_LIST_ROW, kickoff=KICK.isoformat())])
        db = _MainDB([_cand("pp", hours_before=0)], {}, context={"pp": [_replay_row(when=_LIST_REPLAY)]})
        assert self._run(monkeypatch, tmp_path / "out", db, "--postponed-list", str(lst)) == 0
        (csv_path,) = (tmp_path / "out").glob("fixture_orphans_soccer_dryrun_*.csv")
        (row,) = csv.DictReader(csv_path.open())
        assert row["action"] == "mark_postponed" and row["twin_id"] == "replay"

    def test_a_listed_row_an_earlier_run_marked_counts_as_done(self, monkeypatch, tmp_path):
        lst = _write_list(tmp_path / "list.csv", [_LIST_ROW])
        state = {
            "id": "pp",
            "status": "postponed",
            "match_date": KICK,
            "postponed_kickoff": "2026-10-28T18:15:00+00:00",
            "replayed_as": "replay",
        }
        db = _MainDB([], {}, listed_state=[state])
        assert self._run(monkeypatch, tmp_path / "out", db, "--postponed-list", str(lst)) == 0

    @pytest.mark.parametrize("status,marked", [("scheduled", None), ("finished", None), ("postponed", "2026-10-28")])
    def test_a_listed_row_outside_the_candidates_refuses(self, monkeypatch, tmp_path, status, marked):
        lst = _write_list(tmp_path / "list.csv", [_LIST_ROW])
        state = {"id": "pp", "status": status, "match_date": KICK, "postponed_kickoff": marked, "replayed_as": None}
        db = _MainDB([], {}, listed_state=[state])
        # the dry run reports and exits 1 (after its audit CSV); --apply refuses before anything is written
        assert self._run(monkeypatch, tmp_path / "dry", db, "--postponed-list", str(lst)) == 1
        assert len(list((tmp_path / "dry").glob("*.csv"))) == 1
        with pytest.raises(rfo.UnsafeRequest, match="postponed-list"):
            self._run(monkeypatch, tmp_path / "apply", db, "--postponed-list", str(lst), "--apply")
        assert db.writes == [] and not (tmp_path / "apply").exists()

    def test_the_candidate_query_uses_the_as_of_instant(self, monkeypatch, tmp_path):
        db = _MainDB([], {})
        assert self._run(monkeypatch, tmp_path, db, "--as-of", "2026-10-30T09:15:00+00:00") == 0
        (params,) = db.params[rfo.LIST_CANDIDATES_SQL]
        assert params == ("soccer", ["scheduled"], datetime(2026, 10, 30, 9, 15, tzinfo=timezone.utc), "96")
