"""scripts/backfill_never_resulted.py: which listed rows may be healed (screen,
assess), which share one ESPN competition (fork_reasons), the scoreboard days
read, and the write-once ESPN cache the apply replays. The whole tool ran
against a replica of prod with real ESPN scoreboards
(scripts/ops/backfill_soccer_never_resulted.sh header)."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
_spec = importlib.util.spec_from_file_location("backfill_never_resulted", SCRIPTS_DIR / "backfill_never_resulted.py")
bnr = importlib.util.module_from_spec(_spec)
sys.modules["backfill_never_resulted"] = bnr
_spec.loader.exec_module(bnr)

UTC = timezone.utc
AS_OF = datetime(2026, 11, 1, 3, 0, tzinfo=UTC)
KICKOFF = datetime(2026, 6, 13, 16, 30, tzinfo=UTC)


def _row(**over):
    row = {
        "id": "aaaaaaaa-0000-0000-0000-000000000001",
        "league_id": "lg",
        "league_name": "Primera División",
        "country": "Chile",
        "sport": "soccer",
        "match_date": KICKOFF,
        "home_id": "h",
        "away_id": "a",
        "home": "Coquimbo Unido",
        "away": "O'Higgins",
        "status": "scheduled",
        "home_score": None,
        "away_score": None,
        "espn_id": None,
    }
    row.update(over)
    return row


def _comp(home=0, away=0, state="post", completed=True, name="STATUS_FULL_TIME", detail="FT", when=KICKOFF, plays=()):
    return {
        "id": "401850590",
        "date": when.strftime("%Y-%m-%dT%H:%MZ"),
        "status": {"type": {"state": state, "completed": completed, "name": name, "detail": detail}},
        "competitors": [
            {"homeAway": "home", "id": "10", "team": {"id": "10", "displayName": "Coquimbo Unido"}, "score": str(home)},
            {"homeAway": "away", "id": "20", "team": {"id": "20", "displayName": "O'Higgins"}, "score": str(away)},
        ],
        "details": [{"scoringPlay": True, "team": {"id": t}, "clock": {"displayValue": f"{m}'"}} for t, m in plays],
    }


def _assess(row, comp):
    return bnr.assess(row, comp, bnr.fu._competition_parts(comp, bnr.SOCCER))


class TestScreen:
    def test_a_stale_scheduled_soccer_row_goes_to_espn(self):
        assert bnr.screen(_row(), AS_OF) is None

    def test_a_finished_row_is_done(self):
        assert bnr.screen(_row(status="finished", home_score=1, away_score=0), AS_OF) == (bnr.DONE, "already finished")

    @pytest.mark.parametrize(
        "over, why",
        [
            ({"sport": "tennis"}, "not a soccer row"),
            ({"status": "postponed"}, "status 'postponed'"),
            ({"status": "finished"}, "status 'finished'"),  # finished without a score
            ({"match_date": datetime(2026, 10, 29, 12, 0, tzinfo=UTC)}, "within 96 h"),
        ],
    )
    def test_anything_else_is_refused(self, over, why):
        action, reason = bnr.screen(_row(**over), AS_OF)
        assert action == bnr.REFUSE and why in reason


class TestAssess:
    def test_a_completed_final_whose_plays_agree_is_healed(self):
        comp = _comp(2, 1, plays=(("10", 12), ("20", 40), ("10", 88)))
        assert _assess(_row(), comp) == (bnr.HEAL, "", (2, 1), "FT")

    def test_a_goalless_final_lists_no_plays_and_is_healed(self):
        assert _assess(_row(), _comp(0, 0))[:3] == (bnr.HEAL, "", (0, 0))

    @pytest.mark.parametrize(
        "comp, why",
        [
            (_comp(name="STATUS_POSTPONED", state="post", completed=False), "ESPN status STATUS_POSTPONED"),
            (_comp(name="STATUS_ABANDONED"), "ESPN status STATUS_ABANDONED"),
            (_comp(state="in", completed=False, name="STATUS_SECOND_HALF"), "not a completed competition"),
            (_comp(2, 1, plays=(("10", 12),)), "scoring plays add up to 1-0"),
            (_comp(when=datetime(2026, 6, 17, 16, 31, tzinfo=UTC)), "more than"),
        ],
        ids=["postponed", "abandoned", "in-play", "plays-disagree", "kickoff-too-far"],
    )
    def test_anything_else_is_refused(self, comp, why):
        action, reason, final, _ = _assess(_row(), comp)
        assert action == bnr.REFUSE and why in reason and final is None


def _found(*rows_and_kickoffs, espn="401850590"):
    """{row id: {competition key: (comp, ctx, parts)}}: every given row matched to one competition."""
    comp = _comp(when=KICKOFF)
    comp["id"] = espn
    parts = bnr.fu._competition_parts(comp, bnr.SOCCER)
    return {r["id"]: {espn: (comp, {}, parts)} for r in rows_and_kickoffs}


class TestForkReasons:
    A = _row(id="row-a")  # at ESPN's kickoff
    B = _row(id="row-b", match_date=datetime(2026, 6, 15, 0, 0, tzinfo=UTC))  # the fork, 31.5 h later

    def test_the_row_at_espns_kickoff_owns_the_competition_and_the_other_is_its_fork(self):
        reasons = bnr.fork_reasons([self.A, self.B], _found(self.A, self.B))
        assert set(reasons) == {"row-b"} and "a fork of row row-a" in reasons["row-b"]

    def test_the_row_carrying_the_espn_id_owns_it_whatever_its_kickoff(self):
        b_with_id = {**self.B, "espn_id": "401850590"}
        reasons = bnr.fork_reasons([self.A, b_with_id], _found(self.A, b_with_id))
        assert set(reasons) == {"row-a"} and "a fork of row row-b" in reasons["row-a"]

    def test_no_single_owner_refuses_them_all_as_ambiguous(self):
        c = _row(id="row-c", match_date=datetime(2026, 6, 14, 0, 0, tzinfo=UTC))
        reasons = bnr.fork_reasons([self.B, c], _found(self.B, c))
        assert set(reasons) == {"row-b", "row-c"} and all("ambiguous" in r for r in reasons.values())

    def test_one_row_per_competition_has_no_fork(self):
        assert bnr.fork_reasons([self.A], _found(self.A)) == {}


def test_scoreboard_days_are_each_kickoffs_utc_day_and_its_neighbours():
    late = datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
    assert bnr.scoreboard_days([KICKOFF, late]) == [date(2026, 6, d) for d in (12, 13, 14, 15, 16)]


class TestEspnCache:
    def test_fetch_mode_records_each_day_and_a_failed_request_as_none(self, tmp_path, monkeypatch):
        calls = []

        def fetch_day(cfg, slug, day):
            calls.append((slug, day))
            if day == date(2026, 6, 14):
                bnr.fu.FETCH_STATS["failed"] += 1
                return []
            return [{"id": f"ev-{day:%d}"}]

        monkeypatch.setattr(bnr.fu, "fetch_day", fetch_day)
        cache = bnr.EspnCache(str(tmp_path / "espn.json"))
        assert not cache.replay
        assert cache.events("chi.1", date(2026, 6, 13)) == [{"id": "ev-13"}]
        assert cache.events("chi.1", date(2026, 6, 13)) == [{"id": "ev-13"}]  # read once
        assert cache.events("chi.1", date(2026, 6, 14)) is None
        assert calls == [("chi.1", date(2026, 6, 13)), ("chi.1", date(2026, 6, 14))]
        cache.save()
        saved = json.loads((tmp_path / "espn.json").read_text())["days"]
        assert saved == {"chi.1|20260613": [{"id": "ev-13"}], "chi.1|20260614": None}

    def test_replay_mode_never_fetches_and_refuses_a_day_it_does_not_hold(self, tmp_path, monkeypatch):
        (tmp_path / "espn.json").write_text(json.dumps({"days": {"chi.1|20260613": [{"id": "ev"}]}}))
        monkeypatch.setattr(bnr.fu, "fetch_day", lambda *a: pytest.fail("a replay must not call ESPN"))
        cache = bnr.EspnCache(str(tmp_path / "espn.json"))
        assert cache.replay and cache.events("chi.1", date(2026, 6, 13)) == [{"id": "ev"}]
        with pytest.raises(bnr.Refused, match="does not hold chi.1\\|20260614"):
            cache.events("chi.1", date(2026, 6, 14))
        cache.save()  # a no-op: the file stays as the dry run wrote it
        assert json.loads((tmp_path / "espn.json").read_text())["days"] == {"chi.1|20260613": [{"id": "ev"}]}

    def test_an_unreadable_cache_refuses(self, tmp_path):
        (tmp_path / "espn.json").write_text("{not json")
        with pytest.raises(bnr.Refused, match="cannot read the ESPN cache"):
            bnr.EspnCache(str(tmp_path / "espn.json"))


class TestArguments:
    def test_expect_plan_needs_every_key(self):
        full = ",".join(f"{k}=0" for k in bnr.PLAN_KEYS)
        assert bnr.parse_expect_plan(full) == {k: 0 for k in bnr.PLAN_KEYS}
        with pytest.raises(bnr.Refused, match="missing espn_failed"):
            bnr.parse_expect_plan(full.replace(",espn_failed=0", ""))
        with pytest.raises(bnr.Refused, match="bad entry"):
            bnr.parse_expect_plan(full + ",bogus=1")

    def test_as_of_needs_a_time_zone(self):
        assert bnr.parse_as_of("2026-11-01T03:00:00Z") == AS_OF
        with pytest.raises(bnr.Refused, match="time zone"):
            bnr.parse_as_of("2026-11-01T03:00:00")

    def test_an_apply_without_a_pin_refuses(self, tmp_path):
        ids = tmp_path / "ids.txt"
        ids.write_text("aaaaaaaa-0000-0000-0000-000000000001\n")
        argv = ["--match-ids-file", str(ids), "--espn-cache", str(tmp_path / "e.json"), "--apply"]
        assert bnr.main(argv + ["--out-csv", str(tmp_path / "a.csv"), "--database-url", "x"]) == 1


# ── the write path: heal_one's savepoint checks, plan()'s ESPN-owner refusal, run()'s pin ──────────────────────────


class _HealCursor:
    """What heal_one reads back after the results path wrote, inside its savepoint."""

    def __init__(self, row_after, created=(0, 0), touched=()):
        self.row_after, self.created, self.touched = row_after, created, touched
        self.sql: list[str] = []
        self._rows: list[dict] = []

    def execute(self, sql, params=None):
        self.sql.append(" ".join(sql.split()))
        if sql is bnr.ROW_AFTER_SQL:
            self._rows = [self.row_after] if self.row_after else []
        elif sql is bnr.CREATED_SQL:
            self._rows = [{"matches": self.created[0], "teams": self.created[1]}]
        elif sql is bnr.TOUCHED_SQL:
            self._rows = [{"id": i} for i in self.touched]
        elif sql.split()[0] in ("SAVEPOINT", "ROLLBACK", "RELEASE"):
            self._rows = []
        else:
            raise AssertionError(f"unexpected SQL: {sql}")

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


MID = "aaaaaaaa-0000-0000-0000-000000000001"
FINISHED_2_1 = {"status": "finished", "home_score": 2, "away_score": 1, "match_date": KICKOFF, "espn_id": "401850590"}


def _entry():
    return {"match_id": MID, "_league_id": "lg", "_comp": {}, "_context": {}, "_final": (2, 1), "espn_final": "2-1"}


@pytest.fixture
def results_path(monkeypatch):
    """fu._record_result and the grader stubbed: `calls` records what was graded; `wrote` / `identity` steer it."""
    state = {"wrote": True, "identity": None, "raise": None, "graded": []}
    for key in list(bnr.fu.IDENTITY_STATS):
        monkeypatch.setitem(bnr.fu.IDENTITY_STATS, key, 0)

    def record_result(cur, cfg, league_id, comp, context):
        if state["raise"]:
            raise state["raise"]
        if state["identity"]:
            bnr.fu.IDENTITY_STATS[state["identity"]] += 1
        return state["wrote"]

    monkeypatch.setattr(bnr.fu, "_record_result", record_result)
    monkeypatch.setattr(bnr.gm, "finished_matches", lambda cur, ids: [{"match_id": i} for i in ids])
    monkeypatch.setattr(bnr.gm, "grade_listed", lambda cur, matches: state["graded"].extend(matches))
    return state


class TestHealOne:
    def test_a_heal_as_planned_is_released_and_graded(self, results_path):
        cur, e = _HealCursor(FINISHED_2_1, touched=(MID, "earlier-heal")), _entry()
        assert bnr.heal_one(cur, e, {"earlier-heal"}) == ""
        assert cur.sql[0] == "SAVEPOINT heal_one" and cur.sql[-1] == "RELEASE SAVEPOINT heal_one"
        assert not [q for q in cur.sql if q.startswith("ROLLBACK")]
        assert results_path["graded"] == [{"match_id": MID}]
        assert e["status_after"] == "finished" and e["score_after"] == "2-1"

    @pytest.mark.parametrize(
        "cursor_kw, steer, why",
        [
            ({"touched": (MID, "other-row")}, {}, "it touched other matches rows ['other-row']"),
            ({"created": (1, 0)}, {}, "it inserted 1 matches / 0 teams row(s)"),
            ({"created": (0, 1)}, {}, "it inserted 0 matches / 1 teams row(s)"),
            ({"row_after": {**FINISHED_2_1, "away_score": 2}}, {}, "not finished 2-1"),
            ({"row_after": {**FINISHED_2_1, "status": "scheduled"}}, {}, "the row is scheduled"),
            ({}, {"wrote": False}, "it wrote nothing"),
            ({}, {"identity": "forks_refused"}, "identity refusals {'forks_refused': 1}"),
        ],
        ids=["touched-other", "inserted-match", "inserted-team", "wrong-score", "not-finished", "nothing", "identity"],
    )
    def test_a_heal_that_did_not_write_as_planned_is_rolled_back_and_refused(self, results_path, cursor_kw, steer, why):
        results_path.update(steer)
        cur = _HealCursor(**{"row_after": FINISHED_2_1, **cursor_kw})
        reason = bnr.heal_one(cur, _entry(), set())
        assert reason.startswith("the results path did not write it as planned") and why in reason, reason
        assert "ROLLBACK TO SAVEPOINT heal_one" in cur.sql and "RELEASE SAVEPOINT heal_one" not in cur.sql
        assert results_path["graded"] == []  # nothing graded on a row that was rolled back
        assert all(v == 0 for v in bnr.fu.IDENTITY_STATS.values())  # its refusal counted nowhere

    def test_a_database_error_in_the_results_path_is_rolled_back_and_refused(self, results_path):
        results_path["raise"] = bnr.psycopg2.errors.UniqueViolation("duplicate key")
        cur = _HealCursor(FINISHED_2_1)
        assert "the results path raised UniqueViolation" in bnr.heal_one(cur, _entry(), set())
        assert cur.sql == ["SAVEPOINT heal_one", "ROLLBACK TO SAVEPOINT heal_one"] and results_path["graded"] == []


class _PlanCursor:
    """espn_owner's query: `owner` carries the competition's ESPN id already (or None)."""

    def __init__(self, owner):
        self.owner = owner
        self.params = None

    def execute(self, sql, params=None):
        assert sql is bnr.ESPN_OWNER_SQL, sql
        self.params = params

    def fetchone(self):
        return {"id": self.owner} if self.owner else None


class _Days:
    replay = True

    def events(self, slug, day):
        return []


class TestPlanEspnOwner:
    @pytest.fixture
    def matched(self, monkeypatch):
        comp = _comp(2, 1, plays=(("10", 12), ("20", 40), ("10", 88)))
        parts = bnr.fu._competition_parts(comp, bnr.SOCCER)
        monkeypatch.setattr(
            bnr, "match_competitions", lambda cur, rows, slug, events: {MID: {"401850590": (comp, {}, parts)}}
        )
        monkeypatch.setattr(bnr.fu, "build_pair_index", lambda cfg, events: {})

    def test_a_competition_another_row_already_carries_is_refused_as_its_fork(self, matched):
        cur, counts = _PlanCursor("bbbbbbbb-0000-0000-0000-000000000002"), bnr.Counter()
        (e,) = bnr.plan(cur, [_row(id=MID)], AS_OF, _Days(), counts)
        assert e["action"] == bnr.REFUSE and "already on row bbbbbbbb-0000-0000-0000-000000000002" in e["reason"]
        assert cur.params == ("401850590", MID) and "_comp" not in e

    def test_a_competition_no_other_row_carries_is_healed(self, matched):
        (e,) = bnr.plan(_PlanCursor(None), [_row(id=MID)], AS_OF, _Days(), bnr.Counter())
        assert e["action"] == bnr.HEAL and e["espn_final"] == "2-1" and e["_final"] == (2, 1)


class _RunCursor:
    """run()'s own reads: the listed rows, their lock and their teams."""

    def __init__(self, rows):
        self.rows = rows
        self._rows: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if sql is bnr.ROWS_SQL:
            self._rows = [dict(r) for r in self.rows]
        elif sql is bnr.LOCK_SQL:
            self._rows = [{k: r[k] for k in ("id", "status", "match_date", "home_id", "away_id")} for r in self.rows]
        elif sql is bnr.TEAMS_SQL:
            self._rows = [{"id": t, "external_ids": "{}"} for t in params[0]]
        else:
            raise AssertionError(f"unexpected SQL: {sql}")

    def fetchall(self):
        return list(self._rows)


class _RunConn:
    def __init__(self, rows):
        self.cur = _RunCursor(rows)
        self.commits = self.rollbacks = 0

    def cursor(self, cursor_factory=None):
        return self.cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class TestRunPin:
    @pytest.fixture
    def one_heal(self, monkeypatch):
        """One listed row, planned HEAL; the heal grades one prediction (snapshot before -> after)."""

        def plan(cur, rows, as_of, cache, counts):
            return [{**bnr.entry_for(rows[0], "chi.1"), "action": bnr.HEAL}]

        snaps = iter(
            [
                {
                    ("predictions", "p1"): {
                        "match_id": MID,
                        "is_correct": None,
                        "actual_outcome": None,
                        "model_name": "ensemble",
                        "prediction_type": "double_chance",
                    }
                },
                {
                    ("predictions", "p1"): {
                        "match_id": MID,
                        "is_correct": True,
                        "actual_outcome": "1X",
                        "model_name": "ensemble",
                        "prediction_type": "double_chance",
                    }
                },
            ]
        )
        monkeypatch.setattr(bnr, "prefetch", lambda rows, as_of, cache: None)
        monkeypatch.setattr(bnr, "plan", plan)
        monkeypatch.setattr(bnr, "write_archive", lambda cur, path, ids: 0)
        monkeypatch.setattr(bnr, "heal_one", lambda cur, e, healed: "")
        monkeypatch.setattr(bnr.gm, "snapshot", lambda cur, ids: next(snaps))
        return _RunConn([_row(id=MID)])

    PIN = {**{k: 0 for k in bnr.PLAN_KEYS}, "candidates": 1, "heal": 1, "predictions_graded": 1}

    def _run(self, conn, tmp_path, apply_changes, expected):
        (tmp_path / "espn.json").write_text(json.dumps({"days": {}}))  # a replayed cache: ESPN never called
        return bnr.run(
            conn, [MID], AS_OF, str(tmp_path / "espn.json"), apply_changes, expected, str(tmp_path / "a.csv")
        )

    def test_an_apply_at_its_pin_commits(self, one_heal, tmp_path):
        counts = self._run(one_heal, tmp_path, True, dict(self.PIN))
        assert one_heal.commits == 1 and counts["heal"] == 1 and counts["predictions_graded"] == 1

    @pytest.mark.parametrize("key", ["heal", "predictions_graded", "candidates", "refuse", "teams_stamped"])
    def test_an_apply_that_drifted_from_its_pin_rolls_back_and_writes_nothing(self, one_heal, tmp_path, key):
        expected = dict(self.PIN)
        expected[key] += 1
        with pytest.raises(bnr.Refused, match=f"drifted from --expect-plan .*'{key}'"):
            self._run(one_heal, tmp_path, True, expected)
        assert one_heal.commits == 0 and one_heal.rollbacks >= 2  # the read's, then the whole run's
        assert (tmp_path / "a.csv").exists() and (tmp_path / "a.csv.changes.csv").exists()  # written before

    def test_a_dry_run_rolls_back(self, one_heal, tmp_path):
        self._run(one_heal, tmp_path, False, None)
        assert one_heal.commits == 0 and one_heal.rollbacks == 2

    def test_an_apply_without_a_pin_rolls_back(self, one_heal, tmp_path):
        with pytest.raises(bnr.Refused):
            self._run(one_heal, tmp_path, True, None)
        assert one_heal.commits == 0
