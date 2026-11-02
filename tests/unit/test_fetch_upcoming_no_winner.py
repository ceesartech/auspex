"""fetch_upcoming's results path for MMA bouts that end with no winner.

Locks, against real ESPN payloads (tests/unit/fixtures/espn_ufc_no_winner.json:
every UFC competition of 2022-2026 in the fetched scoreboards that ended with
winner=false on both competitors, 11 draws and 16 no contests, plus 3 decisive
bouts, each with ESPN's core-API status result):

  * the scoreboard's own reading (judges' cards) reproduces the core API's
    'draw' / 'no-contest' on all 27, and names the kind of draw;
  * a bout is recorded finished with NULL scores and metadata.no_winner /
    no_winner_detail only when the core API confirms the reading; a winning
    method or no result leaves it for the next sweep, an unreachable core API
    is logged at ERROR, and a disagreement is refused and counted (exit 3);
  * a finished row keeps the kind of result it was recorded with;
  * tennis (no draws) and decisive bouts are untouched.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "espn_ufc_no_winner.json"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fu = _load("fetch_upcoming_no_winner", "fetch_upcoming.py")
MMA = fu.SPORT_CONFIGS["mma"]
TENNIS = fu.SPORT_CONFIGS["tennis"]
COMPS = {c["id"]: c for c in json.loads(FIXTURE.read_text())["competitions"]}
NO_WINNER_IDS = [cid for cid, c in COMPS.items() if not any(x["winner"] for x in c["competitors"])]
GATTO = "401924511"  # Melissa Gatto v Ernesta Kareckaite, UFC 2026-10-10: a majority draw
CONTEXT = {"event_id": "600061541", "league_slug": "ufc", "season_type": "regular"}


def _comp(cid: str) -> dict:
    return copy.deepcopy(COMPS[cid])


def _sides(comp: dict):
    return comp["competitors"][0], comp["competitors"][1]


@pytest.fixture(autouse=True)
def _reset_stats():
    for k in fu.IDENTITY_STATS:
        fu.IDENTITY_STATS[k] = 0
    yield
    for k in fu.IDENTITY_STATS:
        fu.IDENTITY_STATS[k] = 0


# ── how ESPN signals it ───────────────────────────────────────────────


class TestTheFixture:
    def test_it_holds_the_27_no_winner_bouts(self):
        names = [COMPS[c]["core_status_result"]["name"] for c in NO_WINNER_IDS]
        assert len(NO_WINNER_IDS) == 27
        assert names.count("draw") == 11 and names.count("no-contest") == 16

    def test_the_scoreboard_signals_draws_and_no_contests_the_same_way(self):
        # Same status, both winner flags false: the scoreboard alone names no method.
        for cid in NO_WINNER_IDS:
            st = COMPS[cid]["status"]["type"]
            assert (st["name"], st["state"], st["completed"], st["detail"]) == ("STATUS_FINAL", "post", True, "Final")
            assert [c["winner"] for c in COMPS[cid]["competitors"]] == [False, False]


class TestScoreboardReading:
    @pytest.mark.parametrize("cid", NO_WINNER_IDS)
    def test_the_judges_cards_reproduce_the_core_api_result(self, cid):
        comp = _comp(cid)
        reading = fu.scoreboard_no_winner(MMA, comp, *_sides(comp))
        assert reading is not None
        assert reading.kind == fu.core_no_winner_kind(comp["core_status_result"]["name"])

    @pytest.mark.parametrize(
        "cid,draw_type,cards",
        [
            (GATTO, "majority", ((28, 28), (27, 29), (28, 28))),
            ("401562505", "split", ((47, 48), (47, 47), (48, 47))),  # Shevchenko v Grasso, Noche UFC 2023
            ("401591072", "unanimous", ((28, 28), (28, 28), (28, 28))),
        ],
    )
    def test_the_kind_of_draw_comes_from_the_cards(self, cid, draw_type, cards):
        comp = _comp(cid)
        reading = fu.scoreboard_no_winner(MMA, comp, *_sides(comp))
        assert reading == fu.NoWinner("draw", cards, draw_type)

    @pytest.mark.parametrize("cid", ["401440755", "401581370"])  # decisions overturned to no contests
    def test_cards_a_fighter_won_are_a_no_contest(self, cid):
        comp = _comp(cid)
        reading = fu.scoreboard_no_winner(MMA, comp, *_sides(comp))
        assert reading.kind == "no_contest" and reading.cards and reading.draw_type is None

    def test_a_fight_stopped_before_the_cards_is_a_no_contest(self):
        comp = _comp("401436433")  # Munhoz v O'Malley, UFC 276: an eye poke, R2 3:09
        reading = fu.scoreboard_no_winner(MMA, comp, *_sides(comp))
        assert reading == fu.NoWinner("no_contest", (), None)

    @pytest.mark.parametrize("cid", ["401924510", "401927418", "401916277"])
    def test_a_decisive_bout_is_no_reading(self, cid):
        comp = _comp(cid)
        assert fu.scoreboard_no_winner(MMA, comp, *_sides(comp)) is None

    def test_tennis_has_no_draws(self):
        comp = _comp(GATTO)
        assert fu.scoreboard_no_winner(TENNIS, comp, *_sides(comp)) is None

    @pytest.mark.parametrize("flags", [(None, False), (False, None), (None, None), (True, True)])
    def test_a_flag_that_is_not_explicitly_false_is_no_evidence(self, flags):
        comp = _comp(GATTO)
        for side, flag in zip(comp["competitors"], flags):
            if flag is None:
                side.pop("winner")
            else:
                side["winner"] = flag
        assert fu.scoreboard_no_winner(MMA, comp, *_sides(comp)) is None

    @pytest.mark.parametrize("state,completed", [("in", False), ("post", False), ("pre", False)])
    def test_an_unfinished_bout_is_no_reading(self, state, completed):
        comp = _comp(GATTO)
        comp["status"]["type"].update(state=state, completed=completed)
        assert fu.scoreboard_no_winner(MMA, comp, *_sides(comp)) is None

    def test_unusable_cards_read_as_no_cards(self):
        comp = _comp(GATTO)
        home, away = _sides(comp)
        away["linescores"][0]["linescores"].pop()
        assert fu.judges_cards(home, away) is None
        home["linescores"][0]["linescores"][0]["value"] = "n/a"
        assert fu.judges_cards(home, home) is None
        assert fu.judges_cards({"linescores": []}, away) is None


class TestCoreResult:
    @pytest.mark.parametrize(
        "name,kind",
        [
            ("draw", "draw"),
            ("Draw", "draw"),
            ("draw---majority", "draw"),
            ("no-contest", "no_contest"),
            ("kotko", None),
            ("decision---split", None),
            ("submission", None),
            ("", None),
            (None, None),
        ],
    )
    def test_names(self, name, kind):
        assert fu.core_no_winner_kind(name) == kind

    def test_the_status_url_and_the_answer(self, monkeypatch):
        seen = {}

        def fake_get(url, params=None, timeout=None):
            seen["url"] = url
            resp = MagicMock()
            resp.json.return_value = {"result": {"id": 269, "name": "draw", "displayName": "Draw"}}
            return resp

        monkeypatch.setattr(fu.requests, "get", fake_get)
        assert fu.espn_core_result(MMA, "ufc", "600061541", GATTO) == ("draw", True)
        assert seen["url"] == (
            "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/events/600061541/competitions/401924511/status"
        )

    def test_an_answer_without_a_result(self, monkeypatch):
        resp = MagicMock()
        resp.json.return_value = {"type": {"name": "STATUS_FINAL"}}
        monkeypatch.setattr(fu.requests, "get", lambda *a, **k: resp)
        assert fu.espn_core_result(MMA, "ufc", "1", "2") == (None, True)

    def test_every_attempt_failing_is_unanswered_and_logged(self, monkeypatch, caplog):
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise fu.requests.ConnectionError("down")

        monkeypatch.setattr(fu.requests, "get", boom)
        monkeypatch.setattr(fu.time, "sleep", lambda s: None)
        with caplog.at_level(logging.ERROR):
            assert fu.espn_core_result(MMA, "ufc", "1", "2") == (None, False)
        assert len(calls) == fu.FETCH_ATTEMPTS
        assert "ESPN core API FAILED" in caplog.text

    def test_a_non_json_body_is_retried(self, monkeypatch):
        bad, good = MagicMock(), MagicMock()
        bad.json.side_effect = ValueError("not json")
        good.json.return_value = {"result": {"name": "no-contest"}}
        answers = iter([bad, good])
        monkeypatch.setattr(fu.requests, "get", lambda *a, **k: next(answers))
        monkeypatch.setattr(fu.time, "sleep", lambda s: None)
        assert fu.espn_core_result(MMA, "ufc", "1", "2") == ("no-contest", True)


# ── recording it ──────────────────────────────────────────────────────


class _Recorder:
    """Stubs the database half of _record_result: teams resolve to one id per
    name, no row is listed reversed unless asked, reconcile is a no-op, and
    insert_finished_match is captured. `existing` is the row the kind check
    reads at the triple (None: no row)."""

    def __init__(self, monkeypatch, core=("draw", True), existing=None, reversed_listing=False):
        self.captured: dict = {}
        self.core_calls: list = []
        self.cur = MagicMock()
        self.cur.fetchone.return_value = existing
        monkeypatch.setattr(fu, "ensure_team", lambda cur, cfg, name, league_id, espn_id=None, alt_names=(): name)
        monkeypatch.setattr(fu, "_listed_reversed", lambda *a: reversed_listing)
        monkeypatch.setattr(fu, "_reconcile_match_row", lambda *a, **k: True)
        monkeypatch.setattr(fu, "_stamp_identity", lambda *a: None)

        def core_result(cfg, league_slug, event_id, comp_id):
            self.core_calls.append((league_slug, event_id, comp_id))
            return core

        monkeypatch.setattr(fu, "espn_core_result", core_result)

        def fake_insert(
            cur, cfg, league_id, home_id, away_id, match_dt, venue, hs, as_, meta, halftime=None, season=None
        ):
            self.captured.update(home=home_id, away=away_id, home_score=hs, away_score=as_, meta=meta)
            return 1

        monkeypatch.setattr(fu, "insert_finished_match", fake_insert)

    def run(self, comp, cfg=MMA, context=CONTEXT):
        return fu._record_result(self.cur, cfg, "league", comp, context)


class TestRecordResult:
    def test_a_confirmed_draw_is_recorded_finished_with_null_scores(self, monkeypatch, caplog):
        rec = _Recorder(monkeypatch)
        with caplog.at_level(logging.WARNING):
            assert rec.run(_comp(GATTO)) is True
        assert rec.core_calls == [("ufc", "600061541", GATTO)]
        assert (rec.captured["home"], rec.captured["away"]) == ("Melissa Gatto", "Ernesta Kareckaite")
        assert (rec.captured["home_score"], rec.captured["away_score"]) == (None, None)
        assert rec.captured["meta"] == {
            "result_detail": "FINAL",
            "result_source": "espn_scoreboard",
            "no_winner": "draw",
            "no_winner_detail": "majority draw; judges 28-28, 27-29, 28-28; ended R3 5:00",
        }
        assert "finished with NO WINNER: draw" in caplog.text
        assert fu.IDENTITY_STATS["score_mismatch"] == 0

    def test_a_confirmed_no_contest(self, monkeypatch):
        rec = _Recorder(monkeypatch, core=("no-contest", True))
        assert rec.run(_comp("401436433")) is True
        assert rec.captured["meta"]["no_winner"] == "no_contest"
        assert rec.captured["meta"]["no_winner_detail"] == "no contest; ended R2 3:09"

    def test_a_bout_listed_the_other_way_round_stores_the_cards_in_the_rows_order(self, monkeypatch):
        rec = _Recorder(monkeypatch, reversed_listing=True)
        assert rec.run(_comp(GATTO)) is True
        assert (rec.captured["home"], rec.captured["away"]) == ("Ernesta Kareckaite", "Melissa Gatto")
        assert rec.captured["meta"]["no_winner_detail"] == "majority draw; judges 28-28, 29-27, 28-28; ended R3 5:00"

    def test_the_core_api_naming_a_winning_method_leaves_it_for_the_next_sweep(self, monkeypatch, caplog):
        rec = _Recorder(monkeypatch, core=("kotko", True))
        with caplog.at_level(logging.WARNING):
            assert rec.run(_comp(GATTO)) is False
        assert rec.captured == {} and fu.IDENTITY_STATS["score_mismatch"] == 0
        assert "not recorded yet" in caplog.text

    def test_no_result_yet_leaves_it_too(self, monkeypatch):
        rec = _Recorder(monkeypatch, core=(None, True))
        assert rec.run(_comp(GATTO)) is False and rec.captured == {}

    def test_an_unreachable_core_api_is_an_error_not_a_refusal(self, monkeypatch, caplog):
        rec = _Recorder(monkeypatch, core=(None, False))
        with caplog.at_level(logging.ERROR):
            assert rec.run(_comp(GATTO)) is False
        assert rec.captured == {} and fu.IDENTITY_STATS["score_mismatch"] == 0
        assert "did not answer" in caplog.text and "backfill_results" in caplog.text

    def test_a_disagreement_is_refused_and_counted(self, monkeypatch, caplog):
        rec = _Recorder(monkeypatch, core=("no-contest", True))
        with caplog.at_level(logging.ERROR):
            assert rec.run(_comp(GATTO)) is False
        assert rec.captured == {}
        assert fu.IDENTITY_STATS["score_mismatch"] == 1
        assert fu.identity_exit_code(MMA) == fu.IDENTITY_EXIT_CODE
        assert "the scoreboard reads draw" in caplog.text

    @pytest.mark.parametrize("missing", ["event_id", "league_slug"])
    def test_without_the_event_or_league_it_cannot_be_confirmed(self, monkeypatch, caplog, missing):
        rec = _Recorder(monkeypatch)
        with caplog.at_level(logging.ERROR):
            assert rec.run(_comp(GATTO), context={k: v for k, v in CONTEXT.items() if k != missing}) is False
        assert rec.core_calls == [] and rec.captured == {}
        assert "cannot be asked to confirm" in caplog.text

    def test_a_bout_already_recorded_as_that_kind_is_rewritten_without_asking_again(self, monkeypatch, caplog):
        # Each 15-minute sweep rewrites a bout for 3 days: the core API confirmed it when it was recorded.
        rec = _Recorder(monkeypatch, core=(None, False), existing=TestTheKindOfAResultIsKept.DRAW)
        rec.cur.fetchall.return_value = [{"status": "finished", "no_winner": "draw"}]
        with caplog.at_level(logging.INFO):
            assert rec.run(_comp(GATTO)) is True
        assert rec.core_calls == [] and rec.captured["meta"]["no_winner"] == "draw"
        assert "already finished with no winner (draw)" in caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        sql = " ".join(rec.cur.execute.call_args_list[0].args[0].split())
        assert "m.external_ids->>'espn' = %s AND l.sport = %s" in sql
        assert rec.cur.execute.call_args_list[0].args[1] == (GATTO, "mma")

    @pytest.mark.parametrize(
        "rows",
        [
            [{"status": "finished", "no_winner": "no_contest"}],  # another kind: asked, and then refused
            [{"status": "scheduled", "no_winner": None}],  # not recorded yet
            [{"status": "finished", "no_winner": "draw"}] * 2,  # the id on two rows: no shortcut
        ],
    )
    def test_anything_else_asks_the_core_api(self, monkeypatch, rows):
        rec = _Recorder(monkeypatch)
        rec.cur.fetchall.return_value = rows
        rec.run(_comp(GATTO))
        assert rec.core_calls == [("ufc", "600061541", GATTO)]

    def test_tennis_with_no_winner_is_still_not_recorded(self, monkeypatch):
        rec = _Recorder(monkeypatch)
        assert rec.run(_comp(GATTO), cfg=TENNIS) is False
        assert rec.core_calls == [] and rec.captured == {}

    def test_a_decisive_bout_never_asks_the_core_api(self, monkeypatch):
        rec = _Recorder(monkeypatch)
        assert rec.run(_comp("401924510")) is True  # Souza v Godínez: Godínez by decision
        assert rec.core_calls == []
        assert (rec.captured["home_score"], rec.captured["away_score"]) == (0, 1)
        assert "no_winner" not in rec.captured["meta"]


class TestTheKindOfAResultIsKept:
    """A finished row's predictions were graded and its recs settled on the
    kind of result it holds; ESPN reporting another kind later is refused."""

    WINNER = {"id": "row-1", "status": "finished", "home_score": 1, "away_score": 0, "metadata": {}}
    DRAW = {
        "id": "row-1",
        "status": "finished",
        "home_score": None,
        "away_score": None,
        "metadata": {"no_winner": "draw"},
    }

    def test_a_draw_over_a_recorded_winner_is_refused(self, monkeypatch, caplog):
        rec = _Recorder(monkeypatch, existing=self.WINNER)
        with caplog.at_level(logging.ERROR):
            assert rec.run(_comp(GATTO)) is False
        assert rec.captured == {} and fu.IDENTITY_STATS["score_mismatch"] == 1
        assert "finished with a winner (1-0) and ESPN now reports no winner (draw)" in caplog.text

    def test_a_winner_over_a_recorded_draw_is_refused(self, monkeypatch, caplog):
        rec = _Recorder(monkeypatch, existing=self.DRAW)
        with caplog.at_level(logging.ERROR):
            assert rec.run(_comp("401924510")) is False
        assert rec.captured == {} and fu.IDENTITY_STATS["score_mismatch"] == 1
        assert "finished with no winner (draw) and ESPN now reports a winner" in caplog.text

    def test_a_no_contest_over_a_recorded_draw_is_refused(self, monkeypatch):
        rec = _Recorder(monkeypatch, existing=self.DRAW, core=("no-contest", True))
        assert rec.run(_comp("401436433")) is False
        assert fu.IDENTITY_STATS["score_mismatch"] == 1

    def test_the_same_draw_again_is_rewritten(self, monkeypatch):
        rec = _Recorder(monkeypatch, existing=self.DRAW)
        assert rec.run(_comp(GATTO)) is True and fu.IDENTITY_STATS["score_mismatch"] == 0

    def test_a_winner_over_a_recorded_winner_is_the_upserts_business(self, monkeypatch):
        rec = _Recorder(monkeypatch, existing=self.WINNER)
        assert rec.run(_comp("401924510")) is True

    @pytest.mark.parametrize("status", ["scheduled", "postponed", "cancelled"])
    def test_an_unfinished_row_takes_any_result(self, monkeypatch, status):
        rec = _Recorder(monkeypatch, existing={**self.WINNER, "status": status})
        assert rec.run(_comp(GATTO)) is True

    def test_the_check_locks_the_row_it_reads(self):
        cur = MagicMock()
        cur.fetchone.return_value = None
        when = datetime(2026, 10, 10, 21, tzinfo=timezone.utc)
        assert fu._result_kind_allowed(cur, MMA, "h", "a", when, "draw", "x") is True
        sql = " ".join(cur.execute.call_args.args[0].split())
        assert sql.endswith("FOR UPDATE") and cur.execute.call_args.args[1] == ("h", "a", when)

    def test_other_sports_are_not_read(self):
        cur = MagicMock()
        assert fu._result_kind_allowed(cur, fu.SPORT_CONFIGS["soccer"], "h", "a", None, None, "x") is True
        cur.execute.assert_not_called()


class TestNoWinnerDetail:
    def test_without_a_round(self):
        reading = fu.NoWinner("draw", ((29.0, 28.0), (28.0, 29.0), (28.0, 28.0)), "split")
        assert fu.no_winner_detail(reading, reading.cards, {}) == "split draw; judges 29-28, 28-29, 28-28"

    def test_half_points_keep_their_half(self):
        reading = fu.NoWinner("draw", ((28.5, 28.5),), "unanimous")
        assert fu.no_winner_detail(reading, reading.cards, {}) == "unanimous draw; judges 28.5-28.5"
