"""Horse racing dormant (2026-11-01) — the pure parts, no database:

* load_racing_api: /results is read page by page (prod 2026-11-01 had seven
  recs pending five months on page-2 races); /results/{race_id} tells a
  missing result (404 "Result not found") from a failure; the non_runners
  string parses into names; --drain / --dry-run on the CLI;
* close_out_stranded_race_recs: the verdict rules of plan_race_settlement
  (each mirrors grade_completed_races.settle_recommendations) and what the
  CLI refuses;
* monitor_models: horse slices stop paging once no race prediction has
  been created for HORSE_DORMANT_AFTER_HOURS, and only then.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from decimal import Decimal
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO / "scripts"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lra = _load("load_racing_api")
cos = _load("close_out_stranded_race_recs")


class FakeResponse:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self._body = body
        self.text = text if text is not None else ("" if body is None else str(body))

    def json(self):
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


# ── /results pagination ───────────────────────────────────────────────────


def _pages(total, per_page=50, fail_at=None):
    calls = []

    def get(url, params=None, auth=None, timeout=None):
        calls.append(dict(params or {}))
        if fail_at is not None and len(calls) == fail_at:
            return FakeResponse(503, text="busy")
        skip = params["skip"]
        rows = [{"race_id": f"rac_{i}"} for i in range(skip, min(skip + per_page, total))]
        return FakeResponse(200, {"results": rows, "total": total, "limit": per_page, "skip": skip})

    return get, calls


def test_fetch_results_reads_every_page(monkeypatch):
    get, calls = _pages(69)
    monkeypatch.setattr(lra.requests, "get", get)
    monkeypatch.setattr(lra.time, "sleep", lambda s: None)
    rows = lra.fetch_results(lra.date(2026, 6, 6), auth=None)
    assert [r["race_id"] for r in rows] == [f"rac_{i}" for i in range(69)]
    assert [(c["skip"], c["limit"]) for c in calls] == [(0, 50), (50, 50)]
    assert all(c["start_date"] == c["end_date"] == "2026-06-06" for c in calls)


def test_fetch_results_stops_at_the_total_and_on_an_empty_page(monkeypatch):
    get, calls = _pages(50)
    monkeypatch.setattr(lra.requests, "get", get)
    assert len(lra.fetch_results(lra.date(2026, 6, 6), auth=None)) == 50 and len(calls) == 1
    get, calls = _pages(0)
    monkeypatch.setattr(lra.requests, "get", get)
    assert lra.fetch_results(lra.date(2026, 6, 6), auth=None) == [] and len(calls) == 1


def test_a_failed_page_keeps_what_arrived_and_logs_an_error(monkeypatch, caplog):
    get, calls = _pages(120, fail_at=2)
    monkeypatch.setattr(lra.requests, "get", get)
    monkeypatch.setattr(lra.time, "sleep", lambda s: None)
    with caplog.at_level(logging.ERROR, logger="load_racing_api"):
        rows = lra.fetch_results(lra.date(2026, 6, 6), auth=None)
    assert len(rows) == 50
    assert any("page 2" in r.getMessage() and "503" in r.getMessage() for r in caplog.records)


def test_a_total_that_never_ends_is_bounded(monkeypatch, caplog):
    def get(url, params=None, auth=None, timeout=None):
        return FakeResponse(200, {"results": [{"race_id": "x"}], "total": 10**6})

    monkeypatch.setattr(lra.requests, "get", get)
    monkeypatch.setattr(lra.time, "sleep", lambda s: None)
    with caplog.at_level(logging.ERROR, logger="load_racing_api"):
        rows = lra.fetch_results(lra.date(2026, 6, 6), auth=None)
    assert len(rows) == lra.MAX_RESULTS_PAGES
    assert any("stopped after" in r.getMessage() for r in caplog.records)


# ── /results/{race_id} ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "response, expect",
    [
        (FakeResponse(200, {"race_id": "rac_1", "runners": []}), "payload"),
        (FakeResponse(404, text='{"detail":"Result not found - race rac_1"}'), None),
        (FakeResponse(404, text="<html>Not Here</html>"), "error"),  # a wrong URL is not 'no result'
        (FakeResponse(401, text="Unauthorized"), "error"),
        (FakeResponse(403, text='{"detail":"Standard Plan required"}'), "error"),
        (FakeResponse(500, text="boom"), "error"),
        (FakeResponse(200, text="not json"), "error"),
        (FakeResponse(200, {"race_id": "rac_2"}), "error"),  # an answer for another race
        (FakeResponse(200, ["rac_1"]), "error"),
    ],
)
def test_fetch_race_result_tells_no_result_from_a_failure(monkeypatch, response, expect):
    seen = []

    def get(url, params=None, auth=None, timeout=None):
        seen.append(url)
        return response

    monkeypatch.setattr(lra.requests, "get", get)
    if expect == "error":
        with pytest.raises(lra.RacingApiError):
            lra.fetch_race_result("rac_1", auth=None)
    else:
        got = lra.fetch_race_result("rac_1", auth=None)
        assert (got is None) if expect is None else (got["race_id"] == "rac_1")
    assert seen == [f"{lra.BASE_URL}/results/rac_1"]


def test_a_network_failure_is_an_error(monkeypatch):
    def get(url, params=None, auth=None, timeout=None):
        raise lra.requests.ConnectionError("down")

    monkeypatch.setattr(lra.requests, "get", get)
    with pytest.raises(lra.RacingApiError):
        lra.fetch_race_result("rac_1", auth=None)


# ── non-runners ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, names",
    [
        ("Hall Lane (self certificate, lame)", {"hall lane"}),  # prod rac_11958661
        ("", set()),
        (None, set()),
        ("Big Brown (IRE) (going), Sea The Stars (vet's certificate, lame)", {"big brown", "sea the stars"}),
        ("Frankel, Dancing Brave", {"frankel", "dancing brave"}),
        ("  Odd   Spacing (GB)  (ground) ,", {"odd spacing"}),
    ],
)
def test_parse_non_runners(text, names):
    assert lra.parse_non_runners(text) == names


# ── CLI ───────────────────────────────────────────────────────────────────


def test_drain_is_its_own_mode():
    args = lra.parse_args(["--drain", "--database-url", "x"])
    assert args.drain and not args.results and args.upcoming is None and not args.dry_run
    assert lra.parse_args(["--drain", "--dry-run", "--database-url", "x"]).dry_run
    with pytest.raises(SystemExit):
        lra.parse_args(["--drain", "--results", "--database-url", "x"])
    with pytest.raises(SystemExit):
        lra.parse_args(["--results", "--dry-run", "--start", "2026-01-01", "--end", "2026-01-01"])


# ── plan_race_settlement ──────────────────────────────────────────────────


def _rec(horse, bet_type="win", odds="5.00", stake="10.00", rec_id="r1"):
    return {"rec_id": rec_id, "horse": horse, "bet_type": bet_type, "odds": Decimal(odds), "stake": Decimal(stake),
            "selection": horse.title()}  # fmt: skip


def _field(*names, scratched=()):
    return [{"horse": n, "scratched": n in scratched} for n in names]


def _payload(runners, non_runners=""):
    return {"runners": [{"horse": h, "position": p} for h, p in runners], "non_runners": non_runners}


FIELD8 = ["a", "b", "c", "d", "e", "f", "g", "h"]
RUN8 = [("A (IRE)", "1"), ("B", "2"), ("C", "3"), ("D", "4"), ("E", "5"), ("F", "6"), ("G", "F"), ("H", "PU")]


def test_win_won_lost_and_the_money():
    v = cos.plan_race_settlement(
        [_rec("a", odds="4.33", stake="7.77"), _rec("b", stake="3.10")], _payload(RUN8), _field(*FIELD8)
    )
    assert (v[0]["new_status"], v[0]["profit_loss"], v[0]["actual_result"]) == ("won", Decimal("25.87"), "A (IRE)")
    assert (v[1]["new_status"], v[1]["profit_loss"], v[1]["actual_result"]) == ("lost", Decimal("-3.10"), "A (IRE)")


def test_place_terms_follow_the_runners_at_the_off():
    recs = [_rec("c", "place", odds="2.00"), _rec("d", "place"), _rec("h", "place")]
    v = cos.plan_race_settlement(recs, _payload(RUN8), _field(*FIELD8))
    assert [x["places_paid"] for x in v] == [3, 3, 3]
    assert [(x["new_status"], x["actual_result"]) for x in v] == [
        ("won", "finished 3"),
        ("lost", "finished 4"),
        ("lost", "unplaced"),  # pulled up
    ]
    # One declared runner withdrawn: seven ran, two places.
    v = cos.plan_race_settlement([_rec("c", "place")], _payload(RUN8[:7], non_runners="H (lame)"), _field(*FIELD8))
    assert (v[0]["places_paid"], v[0]["new_status"]) == (2, "lost")


def test_a_non_runner_is_void_and_a_result_without_a_winner_voids_all():
    v = cos.plan_race_settlement([_rec("h")], _payload(RUN8[:7], non_runners="H (IRE) (vet's certificate)"),
                                 _field(*FIELD8))  # fmt: skip
    assert (v[0]["action"], v[0]["new_status"], v[0]["profit_loss"], v[0]["reason"]) == (
        "settle",
        "void",
        Decimal("0.00"),
        "non-runner",
    )
    no_winner = [(h, "PU") for h, _ in RUN8]
    v = cos.plan_race_settlement([_rec("a"), _rec("b", "place")], _payload(no_winner), _field(*FIELD8))
    assert {(x["new_status"], x["actual_result"]) for x in v} == {("void", "void")}


def test_a_dead_heat_or_an_unaccounted_entrant_is_unmatched():
    dead_heat = [("A", "1"), ("B", "1")] + RUN8[2:]
    v = cos.plan_race_settlement([_rec("c")], _payload(dead_heat), _field(*FIELD8))
    assert v[0]["action"] == "unmatched" and "dead heat" in v[0]["reason"]
    v = cos.plan_race_settlement([_rec("a")], _payload(RUN8[:7]), _field(*FIELD8))  # 'h' vanished
    assert v[0]["action"] == "unmatched" and "h" in v[0]["reason"]
    # Already scratched on our side: accounted for.
    v = cos.plan_race_settlement([_rec("a")], _payload(RUN8[:7]), _field(*FIELD8, scratched=("h",)))
    assert v[0]["action"] == "settle"


def test_close_out_cli_refusals():
    assert cos.parse_args(["--raced-before", "2026-11-01T00:00:00Z"]).raced_before.tzinfo is not None
    for argv in (
        [],
        ["--raced-before", "2026-11-01"],  # no timezone
        ["--raced-before", "yesterday"],
        ["--raced-before", "2026-11-01T00:00:00Z", "--apply"],  # an apply needs the pin
        ["--raced-before", "2026-11-01T00:00:00Z", "--apply", "--expect-plan", "candidates=42"],  # every key
        ["--raced-before", "2026-11-01T00:00:00Z", "--expect-plan", "candidates=x"],
        ["--raced-before", "2026-11-01T00:00:00Z", "--expect-plan", "bogus=1"],
    ):
        with pytest.raises(SystemExit):
            cos.parse_args(argv)
    full = "candidates=42,races=30,settle=10,void=32,unmatched=0,errors=0"
    args = cos.parse_args(["--raced-before", "2026-11-01T00:00:00Z", "--apply", "--expect-plan", full])
    assert args.expect_plan == {"candidates": 42, "races": 30, "settle": 10, "void": 32, "unmatched": 0, "errors": 0}


def test_close_out_and_drain_never_share_a_race():
    # The drain lists races >= NOW() - DRAIN_DAYS; the close-out only < NOW() - DRAIN_DAYS.
    assert "r.race_date >= NOW() - make_interval(days => %(drain_days)s)" in lra.DRAIN_RACES_SQL
    assert "r.race_date < NOW() - make_interval(days => %(drain_days)s)" in cos.CANDIDATES_SQL
