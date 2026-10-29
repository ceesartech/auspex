"""fetch_upcoming.fetch_day survives ESPN's intermittent non-JSON responses.

From 2026-10-27 ESPN's NBA scoreboard returned a 200 with a non-JSON body on
about a third of pipeline runs; r.json() raised outside the error handling and
failed the whole fetch_results task (63 failed DAG runs). fetch_day now retries
and, if the body never parses, logs at ERROR, counts the failure and returns [].
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest
import requests

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fu = _load("fetch_upcoming")


class _Resp:
    def __init__(self, body: str, status: int = 200):
        self.text = body
        self.status_code = status
        self.headers = {"content-type": "application/json" if body.startswith("{") else "text/html"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return json.loads(self.text)


@pytest.fixture(autouse=True)
def _no_sleep_and_fresh_stats(monkeypatch):
    monkeypatch.setattr(fu.time, "sleep", lambda s: None)
    monkeypatch.setitem(fu.FETCH_STATS, "ok", 0)
    monkeypatch.setitem(fu.FETCH_STATS, "failed", 0)


def _serve(monkeypatch, responses):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(fu.requests, "get", fake_get)
    return calls


CFG = fu.SPORT_CONFIGS["nba"]


def test_a_bad_body_is_retried_and_the_good_one_returned(monkeypatch):
    calls = _serve(monkeypatch, [_Resp(""), _Resp('{"events": [{"id": "1"}]}')])
    assert fu.fetch_day(CFG, "nba", date(2026, 10, 28)) == [{"id": "1"}]
    assert len(calls) == 2
    assert fu.FETCH_STATS == {"ok": 1, "failed": 0}


def test_a_body_that_never_parses_returns_empty_and_is_counted(monkeypatch, caplog):
    _serve(monkeypatch, [_Resp("<html>oops</html>")] * fu.FETCH_ATTEMPTS)
    assert fu.fetch_day(CFG, "nba", date(2026, 10, 28)) == []
    assert fu.FETCH_STATS == {"ok": 0, "failed": 1}
    assert any("FAILED" in r.message and "non-JSON" in r.message for r in caplog.records)


def test_http_errors_are_retried_too(monkeypatch):
    _serve(monkeypatch, [requests.ConnectionError("reset"), _Resp('{"events": []}')])
    assert fu.fetch_day(CFG, "nba", date(2026, 10, 28)) == []
    assert fu.FETCH_STATS == {"ok": 1, "failed": 0}
