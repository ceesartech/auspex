"""Soccer results-path correctness in fetch_upcoming.py (2026-10-31).

  * a header score contradicted by ESPN's own scoring plays is refused,
    logged and counted (exit 3), never stored — ESPN 401843358, Aalesund v
    HamKam, reports FT 0-0 while its plays are 2-2 (football-data: 2-2);
  * a tie decided in extra time or on penalties keeps its final score and
    gains metadata.regulation_home_score / regulation_away_score +
    went_extra, from the plays with minute <= 90 (stoppage included,
    shootout excluded), only when the plays reproduce the final score;
  * ESPN's STATUS_POSTPONED / STATUS_CANCELED / STATUS_ABANDONED reach the
    EXISTING row from both sweeps, guarded, never resurrecting a finished
    row or touching a cancelled one;
  * calendar-year leagues get a calendar-year season label.

The ESPN payloads in fixtures/espn_soccer_results.json are real
competitions, trimmed to the fields fetch_upcoming reads.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "espn_soccer_results.json").read_text())


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fu = _load("fetch_upcoming_soccer_results", "fetch_upcoming.py")
SOCCER = fu.SPORT_CONFIGS["soccer"]
NHL = fu.SPORT_CONFIGS["nhl"]
TENNIS = fu.SPORT_CONFIGS["tennis"]
MMA = fu.SPORT_CONFIGS["mma"]


def fixture(name: str) -> dict:
    return json.loads(json.dumps(FIXTURES[name]))  # a fresh copy per test


def sides(comp: dict):
    parts = fu._competition_parts(comp, SOCCER)
    return parts[0], parts[1]


@pytest.fixture
def stats(monkeypatch):
    for key in fu.IDENTITY_STATS:
        monkeypatch.setitem(fu.IDENTITY_STATS, key, 0)
    return fu.IDENTITY_STATS


@pytest.fixture
def recorded(monkeypatch):
    """Stub the team + match writes; capture what insert_finished_match got."""
    monkeypatch.setattr(fu, "ensure_team", lambda cur, cfg, name, league_id: f"team-{name}")
    monkeypatch.setattr(fu, "_reconcile_match_row", lambda *a, **k: True)
    monkeypatch.setattr(fu, "_stamp_identity", lambda *a, **k: True)
    calls: list[dict] = []

    def fake_insert(cur, cfg, league_id, home_id, away_id, match_dt, venue, hs, as_, meta, halftime=None, season=None):
        calls.append({"home_score": hs, "away_score": as_, "meta": meta, "halftime": halftime, "season": season})
        return 1

    monkeypatch.setattr(fu, "insert_finished_match", fake_insert)
    return calls


# ── (1) the header score must agree with the scoring plays ──────────────


class TestHeaderAgainstScoringPlays:
    def test_aalesund_plays_add_up_to_2_2_against_a_0_0_header(self):
        comp = fixture("aalesund_hamkam_header_0_0_plays_2_2")
        home, away = sides(comp)
        assert fu._final_scores(SOCCER, home, away) == (0, 0)
        assert fu.scoring_play_tally(comp, home, away) == (2, 2, 0)

    def test_aalesund_is_refused_counted_and_logged_with_both_scores(self, stats, recorded, caplog):
        comp = fixture("aalesund_hamkam_header_0_0_plays_2_2")
        with caplog.at_level(logging.ERROR):
            assert fu._record_result(MagicMock(), SOCCER, "league", comp, {"league_slug": "nor.1"}) is False
        assert recorded == []  # the header 0-0 is never stored
        assert stats["score_mismatch"] == 1
        msg = next(r.getMessage() for r in caplog.records if r.levelname == "ERROR")
        for part in ("401843358", "Aalesund", "Hamarkameratene", "header says 0-0", "add up to 2-2", "24'"):
            assert part in msg

    def test_the_refusal_fails_the_run(self, stats, recorded):
        fu._record_result(MagicMock(), SOCCER, "league", fixture("aalesund_hamkam_header_0_0_plays_2_2"))
        assert fu.identity_exit_code(SOCCER) == fu.IDENTITY_EXIT_CODE == 3

    def test_nothing_is_touched_before_the_refusal(self, stats, monkeypatch):
        # Not even a team row: the refusal comes before ensure_team.
        ensure = MagicMock()
        monkeypatch.setattr(fu, "ensure_team", ensure)
        cur = MagicMock()
        fu._record_result(cur, SOCCER, "league", fixture("aalesund_hamkam_header_0_0_plays_2_2"))
        ensure.assert_not_called()
        cur.execute.assert_not_called()

    def test_a_goal_credited_to_neither_side_is_a_contradiction(self, stats, recorded):
        comp = fixture("spain_argentina_aet_1_0")
        comp["details"][0]["team"]["id"] = "99999"
        assert fu.scoring_play_tally(comp, *sides(comp)) == (0, 0, 1)
        assert fu._record_result(MagicMock(), SOCCER, "league", comp) is False
        assert stats["score_mismatch"] == 1 and recorded == []

    def test_a_partial_play_list_is_refused(self, stats, recorded):
        comp = fixture("belgium_senegal_aet_3_2")
        comp["details"].pop()  # the 120'+5' winner missing
        assert fu._record_result(MagicMock(), SOCCER, "league", comp) is False
        assert stats["score_mismatch"] == 1

    def test_a_goalless_match_lists_no_plays_and_is_recorded(self, stats, recorded):
        comp = fixture("aalesund_hamkam_header_0_0_plays_2_2")
        comp["details"] = []
        assert fu._record_result(MagicMock(), SOCCER, "league", comp) is True
        assert (recorded[0]["home_score"], recorded[0]["away_score"]) == (0, 0)
        assert stats["score_mismatch"] == 0

    def test_shootout_kicks_are_not_counted(self, stats, recorded):
        comp = fixture("germany_paraguay_ft_pens_1_1")
        assert sum(bool(p["shootout"]) for p in comp["details"]) == 7
        assert fu.scoring_play_tally(comp, *sides(comp)) == (1, 1, 0)
        assert fu._record_result(MagicMock(), SOCCER, "league", comp) is True

    def test_other_sports_are_not_checked(self, stats, recorded):
        comp = fixture("aalesund_hamkam_header_0_0_plays_2_2")
        comp["status"]["type"]["detail"] = "Final"
        assert fu._record_result(MagicMock(), NHL, "league", comp) is True
        assert stats["score_mismatch"] == 0


# ── (3) the 90-minute score of a tie decided after 90 minutes ────────────


class TestRegulationScore:
    def test_belgium_senegal_was_2_2_after_90_minutes(self):
        # 24' A, 51' A, 86' H, 89' H, then 120'+5' H (a penalty in extra time).
        comp = fixture("belgium_senegal_aet_3_2")
        home, away = sides(comp)
        assert fu.regulation_scores(comp, home, away, (3, 2)) == (2, 2)
        assert fu.halftime_scores(comp, home, away, (3, 2)) == (0, 1)

    def test_an_extra_time_only_goal_leaves_0_0(self):
        comp = fixture("spain_argentina_aet_1_0")
        assert fu.regulation_scores(comp, *sides(comp), (1, 0)) == (0, 0)

    def test_stoppage_time_counts_and_the_shootout_does_not(self):
        # Liga de Quito v Palmeiras: 90' and 90'+3' goals, then a shootout
        # whose kicks ESPN clocks at 91'-98'.
        comp = fixture("ldu_palmeiras_ft_pens_3_2")
        assert fu.regulation_scores(comp, *sides(comp), (3, 2)) == (3, 2)

    def test_plays_that_do_not_reproduce_the_final_give_nothing(self):
        comp = fixture("belgium_senegal_aet_3_2")
        assert fu.regulation_scores(comp, *sides(comp), (4, 2)) is None
        comp["details"][0]["clock"]["displayValue"] = ""
        assert fu.regulation_scores(comp, *sides(comp), (3, 2)) is None

    def test_aet_result_carries_the_90_minute_score(self, stats, recorded):
        assert fu._record_result(MagicMock(), SOCCER, "league", fixture("belgium_senegal_aet_3_2")) is True
        got = recorded[0]
        assert (got["home_score"], got["away_score"]) == (3, 2)  # the final score stays the final score
        assert got["meta"]["regulation_home_score"] == 2 and got["meta"]["regulation_away_score"] == 2
        assert got["meta"]["went_extra"] is True
        assert got["meta"]["result_detail"] == "AET"
        assert got["halftime"] == (0, 1)

    def test_a_shootout_without_extra_time_is_stamped_too(self, stats, recorded):
        assert fu._record_result(MagicMock(), SOCCER, "league", fixture("ldu_palmeiras_ft_pens_3_2")) is True
        meta = recorded[0]["meta"]
        assert (meta["regulation_home_score"], meta["regulation_away_score"], meta["went_extra"]) == (3, 2, True)
        assert meta["result_detail"] == "FT-PENS"

    def test_goalless_shootout_is_0_0_after_90(self, stats, recorded):
        comp = fixture("germany_paraguay_ft_pens_1_1")
        comp["competitors"][0]["score"] = comp["competitors"][1]["score"] = "0"
        comp["details"] = [p for p in comp["details"] if p["shootout"]]
        assert fu._record_result(MagicMock(), SOCCER, "league", comp) is True
        meta = recorded[0]["meta"]
        assert (meta["regulation_home_score"], meta["regulation_away_score"]) == (0, 0)

    def test_without_usable_plays_no_keys_are_written_and_it_warns(self, stats, recorded, caplog):
        comp = fixture("belgium_senegal_aet_3_2")
        comp["details"] = []  # nothing to contradict the header, nothing to split it either
        with caplog.at_level(logging.WARNING):
            assert fu._record_result(MagicMock(), SOCCER, "league", comp) is True
        assert "regulation_home_score" not in recorded[0]["meta"] and "went_extra" not in recorded[0]["meta"]
        assert any("decided after 90 minutes" in r.getMessage() for r in caplog.records)

    def test_a_90_minute_result_carries_no_keys(self, stats, recorded):
        comp = fixture("spain_argentina_aet_1_0")
        comp["status"]["type"].update(name="STATUS_FULL_TIME", detail="FT", shortDetail="FT")
        comp["details"][0]["clock"]["displayValue"] = "76'"
        assert fu._record_result(MagicMock(), SOCCER, "league", comp) is True
        assert "regulation_home_score" not in recorded[0]["meta"]

    def test_nhl_overtime_is_unchanged(self, stats, recorded):
        comp = fixture("spain_argentina_aet_1_0")
        comp["status"]["type"].update(name="STATUS_FINAL", detail="Final/OT")
        assert fu._record_result(MagicMock(), NHL, "league", comp) is True
        meta = recorded[0]["meta"]
        assert meta["regulation_winner"] == "tie" and "regulation_home_score" not in meta

    @pytest.mark.parametrize(
        "name, detail, expected",
        [
            ("STATUS_FINAL_AET", "AET", True),
            ("STATUS_FINAL_PEN", "FT-PENS", True),
            ("STATUS_FULL_TIME", "FT", False),
            ("STATUS_FULL_TIME", "SUSPENDED", False),  # contains 'PEN', is not a shootout
        ],
    )
    def test_past_90_detection(self, name, detail, expected):
        assert fu._soccer_went_past_90({"name": name}, detail) is expected


# ── (2) postponed / cancelled / abandoned ───────────────────────────────

KICKOFF = datetime(2026, 10, 17, 16, 30, tzinfo=timezone.utc)
HOME, AWAY = "atromitos-uuid", "aek-uuid"


class StatusCursor:
    """The rows the status path can see, queried the way it queries them."""

    def __init__(self, rows=(), sport="soccer"):
        self.rows = [dict(r) for r in rows]
        self.sport = sport
        self.updates: list[tuple] = []
        self.dropped_ids: list = []
        self.statements: list[str] = []
        self._result: list = []

    def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        self.statements.append(sql)
        if "external_ids->>'espn' = %s" in sql and sql.startswith("SELECT"):
            espn_id, sport = params
            self._result = [r for r in self.rows if r.get("espn") == espn_id and sport == self.sport][:2]
        elif sql.startswith("SELECT id, status, home_team_id"):
            h, a, when = params
            self._result = [
                r for r in self.rows if (r["home_team_id"], r["away_team_id"], r["match_date"]) == (h, a, when)
            ]
        elif sql.startswith("UPDATE matches"):
            status, patch, drop_espn_id, rid, old = params
            assert "WHERE id = %s AND status = %s RETURNING id" in sql
            self.updates.append((status, json.loads(patch), rid, old))
            self.dropped_ids.append(drop_espn_id)
            hit = [r for r in self.rows if r["id"] == rid and r["status"] == old]
            for r in hit:
                r["status"] = status
                r["metadata"] = {**(r.get("metadata") or {}), **json.loads(patch)}
                if drop_espn_id is not None and r.get("espn") == drop_espn_id and "- 'espn'" in sql:
                    r["espn"] = None
            self._result = [{"id": rid}] if hit else []
        else:
            raise AssertionError(f"unexpected SQL: {sql[:80]}")

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)


def _row(status="scheduled", *, espn="401896749", home=HOME, away=AWAY, metadata=None, match_date=KICKOFF, rid="row-1"):
    return {
        "id": rid,
        "status": status,
        "espn": espn,
        "home_team_id": home,
        "away_team_id": away,
        "match_date": match_date,
        "metadata": metadata or {},
    }


@pytest.fixture
def teams(monkeypatch):
    found = {"Atromitos": HOME, "AEK Athens": AWAY}
    monkeypatch.setattr(fu, "find_team", lambda cur, cfg, name, league_id: found.get(name))
    monkeypatch.setattr(fu, "ensure_team", MagicMock(side_effect=AssertionError("the status path never inserts teams")))
    return found


def _postponed():
    return fixture("atromitos_aek_postponed")


class TestTerminalStatus:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("STATUS_POSTPONED", "postponed"),
            ("STATUS_CANCELED", "cancelled"),
            ("STATUS_ABANDONED", "cancelled"),
            ("STATUS_FULL_TIME", None),
            ("STATUS_SCHEDULED", None),
            ("STATUS_SUSPENDED", None),
        ],
    )
    def test_mapping(self, name, expected):
        got = fu.espn_terminal_status({"status": {"type": {"name": name}}})
        assert (got[0] if got else None) == expected

    def test_the_atromitos_payload_is_post_and_not_completed(self):
        status = _postponed()["status"]["type"]
        assert (status["name"], status["state"], status["completed"]) == ("STATUS_POSTPONED", "post", False)


class TestRecordStatusChange:
    def test_results_path_records_a_postponement(self, stats, teams, caplog):
        cur = StatusCursor([_row()])
        with caplog.at_level(logging.WARNING):
            assert fu._record_result(cur, SOCCER, "league", _postponed()) is False  # not a result
        assert len(cur.updates) == 1
        status, patch, rid, old = cur.updates[0]
        assert (status, rid, old) == ("postponed", "row-1", "scheduled")
        assert patch["espn_status"] == "STATUS_POSTPONED"
        assert datetime.fromisoformat(patch[fu.POSTPONED_KICKOFF_KEY]) == KICKOFF
        assert datetime.fromisoformat(patch[fu.POSTPONED_AT_KEY]) <= datetime.now(timezone.utc)
        assert any("as 'postponed' (was 'scheduled')" in r.getMessage() for r in caplog.records)

    def test_fixtures_path_records_it_too(self, stats, teams):
        cur = StatusCursor([_row()])
        assert fu._process_competition(cur, SOCCER, "league", _postponed()) is False
        assert cur.rows[0]["status"] == "postponed"

    def test_the_update_is_guarded_on_the_status_read(self, stats, teams):
        cur = StatusCursor([_row()])
        fu._record_result(cur, SOCCER, "league", _postponed())
        update = next(s for s in cur.statements if s.startswith("UPDATE matches"))
        assert "WHERE id = %s AND status = %s RETURNING id" in update

    def test_a_row_that_changed_meanwhile_is_left_alone(self, stats, teams, monkeypatch, caplog):
        cur = StatusCursor([_row()])
        real = fu._status_target_row

        def stale(*a, **k):
            row = dict(real(*a, **k))
            cur.rows[0]["status"] = "live"  # moved after the read
            return row

        monkeypatch.setattr(fu, "_status_target_row", stale)
        with caplog.at_level(logging.ERROR):
            fu._record_result(cur, SOCCER, "league", _postponed())
        assert cur.rows[0]["status"] == "live"
        assert any("changed while recording" in r.getMessage() for r in caplog.records)

    def test_cancelled_and_abandoned_become_cancelled(self, stats, teams):
        for name in ("STATUS_CANCELED", "STATUS_ABANDONED"):
            cur = StatusCursor([_row("postponed")])
            comp = _postponed()
            comp["status"]["type"]["name"] = name
            fu._record_result(cur, SOCCER, "league", comp)
            assert cur.rows[0]["status"] == "cancelled"
            assert fu.POSTPONED_KICKOFF_KEY not in cur.updates[0][1]

    def test_a_finished_row_is_never_resurrected(self, stats, teams, caplog):
        cur = StatusCursor([_row("finished")])
        with caplog.at_level(logging.WARNING):
            fu._record_result(cur, SOCCER, "league", _postponed())
        assert cur.updates == [] and cur.rows[0]["status"] == "finished"
        assert any("the result stands" in r.getMessage() for r in caplog.records)

    def test_a_cancelled_row_is_never_touched(self, stats, teams):
        for name in ("STATUS_POSTPONED", "STATUS_CANCELED"):
            cur = StatusCursor([_row("cancelled")])
            comp = _postponed()
            comp["status"]["type"]["name"] = name
            fu._record_result(cur, SOCCER, "league", comp)
            assert cur.updates == [] and cur.rows[0]["status"] == "cancelled"

    def test_already_postponed_is_a_no_op(self, stats, teams):
        cur = StatusCursor([_row("postponed", metadata={fu.POSTPONED_KICKOFF_KEY: "2026-10-17T16:30:00+00:00"})])
        fu._process_competition(cur, SOCCER, "league", _postponed())
        assert cur.updates == []

    def test_a_second_postponement_keeps_the_first_kickoff(self, stats, teams):
        first = {
            fu.POSTPONED_KICKOFF_KEY: "2026-10-10T16:30:00+00:00",
            fu.POSTPONED_AT_KEY: "2026-10-09T08:00:00+00:00",
        }
        cur = StatusCursor([_row("scheduled", metadata=first)])  # rescheduled once already
        fu._record_result(cur, SOCCER, "league", _postponed())
        patch = cur.updates[0][1]
        assert fu.POSTPONED_KICKOFF_KEY not in patch and fu.POSTPONED_AT_KEY not in patch
        assert cur.rows[0]["metadata"][fu.POSTPONED_KICKOFF_KEY] == first[fu.POSTPONED_KICKOFF_KEY]

    def test_no_existing_row_means_no_write_and_no_insert(self, stats, teams):
        cur = StatusCursor([])
        assert fu._record_result(cur, SOCCER, "league", _postponed()) is False
        assert cur.updates == []
        assert not any(s.startswith("INSERT") for s in cur.statements)

    def test_unknown_teams_mean_no_row_of_ours(self, stats, monkeypatch):
        monkeypatch.setattr(fu, "find_team", lambda *a: None)
        cur = StatusCursor([_row()])
        fu._record_result(cur, SOCCER, "league", _postponed())
        assert cur.statements == []

    def test_found_by_the_exact_triple_without_an_espn_id(self, stats, teams):
        cur = StatusCursor([_row(espn=None)])
        fu._record_result(cur, SOCCER, "league", _postponed())
        assert cur.rows[0]["status"] == "postponed"

    def test_an_espn_id_on_another_team_pair_is_refused_and_counted(self, stats, teams, caplog):
        cur = StatusCursor([_row(home="someone-else")])
        with caplog.at_level(logging.ERROR):
            fu._record_result(cur, SOCCER, "league", _postponed())
        assert cur.updates == [] and stats["status_refused"] == 1
        assert fu.identity_exit_code(SOCCER) == 3
        assert any("Refusing to record ESPN's status" in r.getMessage() for r in caplog.records)

    def test_an_espn_id_on_two_rows_falls_back_to_the_exact_fixture(self, stats, teams, caplog):
        twin = _row(rid="row-2", match_date=KICKOFF - timedelta(hours=2))
        cur = StatusCursor([_row(), twin])
        with caplog.at_level(logging.ERROR):
            fu._record_result(cur, SOCCER, "league", _postponed())
        assert [u[2] for u in cur.updates] == ["row-1"]  # the row at this kickoff, not a guess
        assert stats["status_refused"] == 0 and fu.identity_exit_code(SOCCER) == 0
        assert any("is on 2 matches rows" in r.getMessage() for r in caplog.records)

    def test_an_espn_id_naming_another_fixture_falls_back_to_the_exact_fixture(self, stats, teams, caplog):
        # Both sides different: a knockout placeholder filled, or a re-used id.
        # The fixture path lets the real fixture have its own row; so does this.
        placeholder = _row(rid="row-9", home="winner-sf1", away="winner-sf2")
        real = _row(rid="row-1", espn=None)
        cur = StatusCursor([placeholder, real])
        with caplog.at_level(logging.WARNING):
            fu._record_result(cur, SOCCER, "league", _postponed())
        assert [u[2] for u in cur.updates] == ["row-1"] and cur.rows[1]["status"] == "postponed"
        assert cur.rows[0]["status"] == "scheduled" and stats["status_refused"] == 0
        assert any("both teams differ" in r.getMessage() for r in caplog.records)

    def test_a_swapped_team_sport_pair_is_drift_and_refused(self, stats, teams):
        cur = StatusCursor([_row(home=AWAY, away=HOME)])
        fu._record_result(cur, SOCCER, "league", _postponed())
        assert cur.updates == [] and stats["status_refused"] == 1

    def test_a_redrawn_tennis_slot_is_logged_not_counted(self, stats, teams, caplog):
        cur = StatusCursor([_row(away="withdrawn-player")], sport="tennis")
        parts = fu._competition_parts(_postponed(), SOCCER)
        with caplog.at_level(logging.WARNING):
            assert fu._record_status_change(cur, TENNIS, "league", parts, "cancelled", "STATUS_CANCELED") is False
        assert cur.updates == [] and stats["status_refused"] == 0
        assert any("a re-drawn slot" in r.getMessage() for r in caplog.records)

    def test_a_swapped_mma_pair_is_the_same_bout(self, stats, teams):
        # MMA competitors carry no homeAway and ESPN reorders them.
        cur = StatusCursor([_row(home=AWAY, away=HOME)], sport="mma")
        parts = fu._competition_parts(_postponed(), SOCCER)
        assert fu._record_status_change(cur, MMA, "league", parts, "cancelled", "STATUS_CANCELED") is True
        assert cur.rows[0]["status"] == "cancelled" and stats["status_refused"] == 0

    def test_a_cancelled_1v1_competition_gives_up_its_espn_id(self, stats, teams):
        # ESPN re-fills the slot under the same id; that pairing needs its own row.
        cur = StatusCursor([_row()], sport="tennis")
        parts = fu._competition_parts(_postponed(), SOCCER)
        fu._record_status_change(cur, TENNIS, "league", parts, "cancelled", "STATUS_CANCELED")
        assert cur.rows[0]["espn"] is None and cur.dropped_ids == ["401896749"]
        assert cur.rows[0]["metadata"]["cancelled_espn_id"] == "401896749"

    def test_a_team_sport_cancel_keeps_its_espn_id(self, stats, teams):
        cur = StatusCursor([_row()])
        comp = _postponed()
        comp["status"]["type"]["name"] = "STATUS_CANCELED"
        fu._record_result(cur, SOCCER, "league", comp)
        assert cur.rows[0]["status"] == "cancelled" and cur.rows[0]["espn"] == "401896749"
        assert cur.dropped_ids == [None] and "cancelled_espn_id" not in cur.rows[0]["metadata"]

    def test_a_cancel_espn_reported_can_become_a_postponement(self, stats, teams):
        espn_cancel = {"status_source": "espn_scoreboard", "espn_status": "STATUS_CANCELED"}
        cur = StatusCursor([_row("cancelled", metadata=espn_cancel)])
        fu._record_result(cur, SOCCER, "league", _postponed())
        assert cur.rows[0]["status"] == "postponed"

    @pytest.mark.parametrize(
        "metadata",
        [
            {},
            {"status_source": "espn_scoreboard", "espn_status": "STATUS_CANCELED", "orphan_of": "twin"},
            {"status_source": "espn_scoreboard", "espn_status": "STATUS_POSTPONED"},
        ],
    )
    def test_a_cancel_recorded_on_purpose_is_never_reopened(self, stats, teams, metadata):
        cur = StatusCursor([_row("cancelled", metadata=metadata)])
        fu._record_result(cur, SOCCER, "league", _postponed())
        assert cur.updates == [] and cur.rows[0]["status"] == "cancelled"


# ── (4) calendar-year season labels ─────────────────────────────────────


class TestSoccerSeasonLabels:
    JULY = datetime(2026, 7, 20, tzinfo=timezone.utc)
    MARCH = datetime(2026, 3, 20, tzinfo=timezone.utc)
    # The calendar-year leagues ESPN carries (scripts/soccer_seasons.py has the evidence).
    CALENDAR_SLUGS = (
        "usa.1",
        "bra.1",
        "arg.1",
        "chn.1",
        "nor.1",
        "swe.1",
        "chi.1",
        "col.1",
        "fifa.world",
        "concacaf.gold",
        "conmebol.libertadores",
    )

    @pytest.mark.parametrize("slug", CALENDAR_SLUGS)
    def test_calendar_year_leagues_label_by_the_year(self, slug):
        assert fu._season_soccer(self.MARCH, slug) == fu._season_soccer(self.JULY, slug) == "2026"

    @pytest.mark.parametrize("slug", ["eng.1", "mex.1", "aus.1", "uefa.champions", None])
    def test_other_leagues_keep_aug_to_may(self, slug):
        assert fu._season_soccer(self.MARCH, slug) == "2025-2026"
        assert fu._season_soccer(self.JULY, slug) == "2026-2027"

    def test_j1_switched_to_aug_may_after_its_2026_special_season(self):
        assert fu._season_soccer(datetime(2026, 2, 14, 5, tzinfo=timezone.utc), "jpn.1") == "2026"
        assert fu._season_soccer(datetime(2026, 6, 6, 5, tzinfo=timezone.utc), "jpn.1") == "2026"
        assert fu._season_soccer(datetime(2026, 8, 8, 10, tzinfo=timezone.utc), "jpn.1") == "2026-2027"
        assert fu._season_soccer(datetime(2027, 3, 6, 5, tzinfo=timezone.utc), "jpn.1") == "2026-2027"

    @pytest.mark.parametrize("slug", sorted(fu.SOCCER_LEAGUES))
    def test_every_espn_league_uses_the_shared_scheme(self, slug):
        ss = _load("soccer_seasons_for_fetch", "soccer_seasons.py")
        code = fu.SOCCER_LEAGUES[slug][0]
        for when in (self.MARCH, self.JULY, datetime(2026, 12, 1, tzinfo=timezone.utc)):
            assert fu._season_soccer(when, slug) == ss.season_for_kickoff(code, when)

    def test_an_irregular_era_falls_back_loudly(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert fu._season_soccer(datetime(2019, 10, 1, tzinfo=timezone.utc), "arg.1") == "2019-2020"
        assert any("No date rule labels" in r.getMessage() for r in caplog.records)

    def test_season_label_reads_the_slug_from_the_context(self):
        assert fu.season_label(SOCCER, self.JULY, {"league_slug": "nor.1"}) == "2026"
        assert fu.season_label(SOCCER, self.JULY, {"league_slug": "eng.1"}) == "2026-2027"
        assert fu.season_label(SOCCER, self.JULY, None) == "2026-2027"
        assert fu.season_label(NHL, self.JULY, {"league_slug": "nhl"}) == NHL.season_func(self.JULY)

    def test_process_event_threads_the_slug(self):
        seen = []
        event = {"id": "e", "competitions": [{"id": "c"}]}
        fu.process_event(None, SOCCER, "lg", event, handler=lambda *a: seen.append(a[4]) or True, league_slug="bra.1")
        assert seen[0]["league_slug"] == "bra.1"

    @staticmethod
    def _sweep_event(state: str) -> dict:
        status = {"state": state, "name": "STATUS_SCHEDULED" if state == "pre" else "STATUS_FULL_TIME"}
        if state == "post":
            status.update(completed=True, detail="FT")
        return {
            "id": "e-mls",
            "competitions": [
                {
                    "id": "c-mls",
                    "date": "2026-03-21T02:30Z",  # March: '2026' calendar-year, '2025-2026' cross-year
                    "status": {"type": status},
                    "competitors": [
                        {"homeAway": "home", "id": "1", "score": "2", "team": {"id": "1", "displayName": "LA Galaxy"}},
                        {"homeAway": "away", "id": "2", "score": "1", "team": {"id": "2", "displayName": "Seattle"}},
                    ],
                }
            ],
        }

    @pytest.fixture
    def sweep(self, monkeypatch, stats):
        """fetch_all / fetch_results_all with ESPN and the database stubbed:
        what season each insert helper is handed."""
        seen: list[tuple[str, str]] = []

        class Conn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def cursor(self, cursor_factory=None):
                return self

            def commit(self):
                pass

        monkeypatch.setattr(fu.psycopg2, "connect", lambda url: Conn())
        monkeypatch.setattr(fu, "ensure_league", lambda *a: "league-mls")
        monkeypatch.setattr(fu, "ensure_team", lambda cur, cfg, name, league_id: f"team-{name}")
        monkeypatch.setattr(fu, "_reconcile_match_row", lambda *a, **k: True)
        monkeypatch.setattr(fu, "_stamp_identity", lambda *a, **k: True)
        monkeypatch.setattr(
            fu, "insert_scheduled_match", lambda *a, season=None, **k: seen.append(("scheduled", season)) or 1
        )
        monkeypatch.setattr(
            fu, "insert_finished_match", lambda *a, season=None, **k: seen.append(("finished", season)) or 1
        )
        return seen

    def test_the_fixtures_sweep_threads_the_slug_to_the_insert(self, sweep, monkeypatch):
        days = iter([[self._sweep_event("pre")]])
        monkeypatch.setattr(fu, "fetch_day", lambda cfg, slug, day: next(days, []))
        assert fu.fetch_all("postgresql://unused", SOCCER, ["usa.1"], 2) == {"usa.1": 1}
        assert sweep == [("scheduled", "2026")]

    def test_the_results_sweep_threads_the_slug_to_the_insert(self, sweep, monkeypatch):
        days = iter([[self._sweep_event("post")]])
        monkeypatch.setattr(fu, "fetch_day", lambda cfg, slug, day: next(days, []))
        assert fu.fetch_results_all("postgresql://unused", SOCCER, ["usa.1"], 1) == {"usa.1": 1}
        assert sweep == [("finished", "2026")]

    def test_the_results_path_labels_an_inserted_row(self, stats, recorded):
        comp = fixture("ldu_palmeiras_ft_pens_3_2")
        fu._record_result(MagicMock(), SOCCER, "league", comp, {"league_slug": "conmebol.libertadores"})
        assert recorded[0]["season"] == "2026"

    def test_the_insert_helpers_write_the_label_given(self):
        for insert, extra in (
            (fu.insert_scheduled_match, ()),
            (fu.insert_finished_match, (2, 1, {})),
        ):
            cur = MagicMock()
            insert(cur, SOCCER, "lg", "h", "a", self.JULY, None, *extra, season="2026")
            assert "2026" in cur.execute.call_args.args[1]
            cur = MagicMock()
            insert(cur, SOCCER, "lg", "h", "a", self.JULY, None, *extra)
            assert "2026-2027" in cur.execute.call_args.args[1]  # no label given: cfg.season_func
