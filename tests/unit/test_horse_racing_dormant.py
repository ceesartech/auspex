"""Horse racing dormant (2026-11-01) — the pure parts, no database:

* load_racing_api: /results is read page by page (prod 2026-11-01 had seven
  recs pending five months on page-2 races); /results/{race_id} tells a
  missing result (404 "Result not found - race <id>") from a failure, the
  API's generic route 404 included; the non_runners string parses into
  names; the drain's field check; --drain / --dry-run on the CLI;
* close_out_stranded_race_recs: the verdict rules of plan_race_settlement
  (each mirrors grade_completed_races.settle_recommendations), what the
  CLI refuses, and its clock (--as-of never later than the database's);
* after the Racing API subscription ends: --upcoming and --results exit 1
  (missing credentials, every racecards tier refusing, a /results page
  refused with 401/402/403) instead of staying green with nothing fetched;
  a racecards outage (no answer, a 5xx) is logged, not raised;
* monitor_models: horse slices stop paging once no race prediction has
  been created for HORSE_DORMANT_AFTER_HOURS, and only then.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from datetime import datetime, timedelta, timezone
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
        (FakeResponse(404, {"detail": "Result not found - race rac_1"}), None),  # the prod per-race 404
        # A wrong path, base URL or version is the API's generic route 404 (prod probe 2026-11-01:
        # /v1/resultz/... and /v9/results/... both answer this) — never 'no result'.
        (FakeResponse(404, {"detail": "Not Found"}), "error"),
        (FakeResponse(404, {"detail": "Result not found - race rac_9"}), "error"),  # about another race
        (FakeResponse(404, text="Result not found - race rac_1"), "error"),  # not JSON
        (FakeResponse(404, ["Result not found - race rac_1"]), "error"),
        (FakeResponse(404, text="<html>Not Here</html>"), "error"),
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


# ── the drain's field check ───────────────────────────────────────────────


def _drain_field(*rows):
    """(horse, scratched, pending recs, pending place recs) -> DRAIN_FIELD_SQL rows."""
    return [{"horse": h, "scratched": s, "pending_recs": n, "pending_place_recs": p} for h, s, n, p in rows]


AYR = {  # prod, Ayr 31 Oct 13:18: the card's 'Jo Arengeot Bey Aa' is the result's 'Jo Arengeot Bey (FR)'
    "runners": [{"horse": "Jo Arengeot Bey (FR)", "position": "6"}, {"horse": "Alpha", "position": "1"}],
    "non_runners": "Gone (IRE) (going)",
}


@pytest.mark.parametrize(
    "field, problem",
    [
        (_drain_field(("alpha", False, 1, 0), ("jo arengeot bey", False, 0, 0), ("gone", False, 0, 0)), None),
        (_drain_field(("alpha", False, 1, 0), ("jo arengeot bey aa", False, 1, 0)), "carry a pending rec"),
        (_drain_field(("alpha", False, 1, 1), ("jo arengeot bey aa", False, 0, 0)), "place rec"),
        (_drain_field(("alpha", False, 1, 0), ("jo arengeot bey aa", False, 0, 0)), None),  # win only, none on it
        (_drain_field(("alpha", False, 1, 1), ("jo arengeot bey aa", True, 1, 0)), None),  # already scratched
    ],
)
def test_drain_field_problem(field, problem):
    got = lra.drain_field_problem(field, AYR)
    assert got is None if problem is None else (problem in got and "jo arengeot bey" in got)


def test_unaccounted_entrants_are_ours_unscratched_and_named_nowhere():
    field = [{"horse": h, "scratched": s} for h, s in [("alpha", False), ("jo arengeot bey aa", False),
                                                       ("gone", False), ("late", True)]]  # fmt: skip
    assert lra.unaccounted_entrants(field, AYR) == ["jo arengeot bey aa"]


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
    # The drain lists races >= NOW() - DRAIN_DAYS; the close-out only < as_of - DRAIN_DAYS, with as_of <= the
    # database's NOW() (run() refuses a later one: test_horse_racing_drain_pg).
    assert "r.race_date >= NOW() - make_interval(days => %(drain_days)s)" in lra.DRAIN_RACES_SQL
    assert "r.race_date < %(as_of)s - make_interval(days => %(drain_days)s)" in cos.CANDIDATES_SQL
    assert "r.race_date < %(as_of)s - make_interval(days => %(drain_days)s)" in cos.BEYOND_PIN_SQL
    # Recs still in the window are exactly the ones before the pin the candidates leave out.
    assert "r.race_date >= %(as_of)s - make_interval(days => %(drain_days)s)" in cos.IN_WINDOW_SQL
    assert "r.race_date < %(raced_before)s" in cos.IN_WINDOW_SQL
    assert "NOW()" not in cos.CANDIDATES_SQL + cos.BEYOND_PIN_SQL + cos.IN_WINDOW_SQL


def test_close_out_as_of_needs_a_zone():
    base = ["--raced-before", "2026-11-01T00:00:00Z"]
    assert cos.parse_args(base).as_of is None  # the database's clock
    at = cos.parse_args(base + ["--as-of", "2026-11-03T19:30:00Z"]).as_of
    assert at == datetime(2026, 11, 3, 19, 30, tzinfo=timezone.utc)
    for bad in ("2026-11-03T19:30:00", "tomorrow"):
        with pytest.raises(SystemExit):
            cos.parse_args(base + ["--as-of", bad])


def test_a_future_as_of_is_refused_against_the_database_clock_before_any_read():
    db_clock = datetime(2026, 11, 2, 6, 0, tzinfo=timezone.utc)

    class Cur:
        def __init__(self):
            self.sql = []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            self.sql.append(sql)

        def fetchone(self):
            return {"now": db_clock}

    class Conn:
        def __init__(self):
            self.cur = Cur()
            self.rolled_back = 0

        def cursor(self, cursor_factory=None):
            return self.cur

        def rollback(self):
            self.rolled_back += 1

    conn = Conn()
    with pytest.raises(cos.FutureAsOf, match="after the database's clock"):
        cos.run(conn, db_clock, fetch=None, out_csv="/nonexistent/x.csv", as_of=db_clock + timedelta(seconds=1))
    assert conn.cur.sql == ["SELECT NOW() AS now"] and conn.rolled_back == 1


# ── after the Racing API subscription ends: --upcoming / --results (the enabled branch) ─────────────────


def _refuse_every_call(status, calls):
    def get(url, params=None, auth=None, timeout=None):
        calls.append(url)
        return FakeResponse(status, {"detail": "Not authenticated"}, text='{"detail":"Not authenticated"}')

    return get


@pytest.mark.parametrize("argv", [["--upcoming", "7"], ["--results", "--start", "2026-11-01", "--end", "2026-11-01"]])
def test_upcoming_and_results_without_credentials_exit_1_without_a_call(monkeypatch, argv):
    calls = []
    monkeypatch.setattr(lra.requests, "get", _refuse_every_call(401, calls))
    monkeypatch.setattr(lra, "_auth", lambda: None)
    assert lra.main(argv + ["--database-url", "postgresql://unused"]) == 1
    assert calls == []


@pytest.mark.parametrize("status", [401, 402, 403])
def test_racecards_refused_on_every_tier_raise(monkeypatch, status):
    calls = []
    monkeypatch.setattr(lra.requests, "get", _refuse_every_call(status, calls))
    monkeypatch.setattr(lra, "_CACHED_RACECARDS_PATH", None)
    with pytest.raises(lra.RacingApiError, match="subscription"):
        lra.fetch_racecards(lra.date(2026, 11, 2), auth=None)
    assert [u.rsplit("/", 1)[1] for u in calls] == ["pro", "standard", "basic"] and lra._CACHED_RACECARDS_PATH is None


def test_a_tier_that_will_not_serve_the_request_counts_as_refused(monkeypatch):
    # 422 is a tier turning this request down on these credentials (the probe skipped it before 2026-11-02 too).
    answers = {"pro": 422, "standard": 401, "basic": 401}
    monkeypatch.setattr(lra.requests, "get", lambda url, **k: FakeResponse(answers[url.rsplit("/", 1)[1]], {}))
    monkeypatch.setattr(lra, "_CACHED_RACECARDS_PATH", None)
    with pytest.raises(lra.RacingApiError, match="every racecards tier refused"):
        lra.fetch_racecards(lra.date(2026, 11, 2), auth=None)


@pytest.mark.parametrize(
    "answers",
    [
        {"pro": 503, "standard": 503, "basic": 503},  # the API is down
        {"pro": 403, "standard": 502, "basic": 403},  # the subscribed tier is down, the others not on the plan
        {"pro": 401, "standard": 401, "basic": "timeout"},  # one tier did not answer at all
    ],
)
def test_a_racecards_outage_is_logged_not_raised(monkeypatch, caplog, answers):
    # An outage is not a lapsed subscription: no card this tick, an ERROR, and --upcoming exits 0 — so the results
    # fetch downstream of it in the DAG (all_success) still runs.
    def get(url, params=None, auth=None, timeout=None):
        answer = answers[url.rsplit("/", 1)[1]]
        if answer == "timeout":
            raise lra.requests.Timeout("read timed out")
        return FakeResponse(answer, {"detail": "x"})

    monkeypatch.setattr(lra.requests, "get", get)
    monkeypatch.setattr(lra, "_CACHED_RACECARDS_PATH", None)
    with caplog.at_level(logging.ERROR, logger="load_racing_api"):
        assert lra.fetch_racecards(lra.date(2026, 11, 2), auth=None) == []
    assert "no racecards tier answered 200" in caplog.text and lra._CACHED_RACECARDS_PATH is None


def test_an_empty_card_is_not_an_error(monkeypatch):
    monkeypatch.setattr(lra.requests, "get", lambda *a, **k: FakeResponse(200, {"racecards": []}))
    monkeypatch.setattr(lra, "_CACHED_RACECARDS_PATH", None)
    assert lra.fetch_racecards(lra.date(2026, 11, 2), auth=None) == []


@pytest.mark.parametrize("status", [401, 402, 403])
def test_a_results_page_refused_for_the_credentials_or_the_plan_raises(monkeypatch, status):
    calls = []
    monkeypatch.setattr(lra.requests, "get", _refuse_every_call(status, calls))
    with pytest.raises(lra.RacingApiError, match=f"HTTP {status}"):
        lra.fetch_results(lra.date(2026, 11, 1), auth=None)
    assert len(calls) == 1


class _Conn:
    """psycopg2 connection stand-in for run(): the fetch fails before any SQL."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self, cursor_factory=None):
        return self

    def commit(self):
        raise AssertionError("nothing to commit")


@pytest.mark.parametrize("argv", [["--upcoming", "7"], ["--results", "--start", "2026-11-01", "--end", "2026-11-02"]])
def test_a_lapsed_subscription_fails_upcoming_and_results(monkeypatch, caplog, argv):
    monkeypatch.setattr(lra.requests, "get", _refuse_every_call(403, []))
    monkeypatch.setattr(lra, "_auth", lambda: object())
    monkeypatch.setattr(lra, "_CACHED_RACECARDS_PATH", None)
    monkeypatch.setattr(lra.psycopg2, "connect", lambda dsn: _Conn())
    with caplog.at_level(logging.ERROR, logger="load_racing_api"):
        assert lra.main(argv + ["--database-url", "postgresql://unused"]) == 1
    assert "subscription" in caplog.text
