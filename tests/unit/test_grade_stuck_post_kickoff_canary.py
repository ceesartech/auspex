"""grade_completed_matches.warn_stuck_post_kickoff_recs: the canary for recs
migration 027 leaves 'pending' on a finished match the grading window no longer
reaches. Mocked connection; the real-Postgres run is
services/data-ingestion/tests/e2e/test_match_outcome_trigger.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("grading_outcomes", "grading_outcomes.py")
gcm = _load("grade_completed_matches", "grade_completed_matches.py")


class _Cur:
    def __init__(self, rows):
        self.rows = rows
        self.calls: list[tuple[str, tuple]] = []

    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Conn:
    def __init__(self, cur):
        self.cur = cur

    def cursor(self, cursor_factory=None):
        assert cursor_factory is gcm.RealDictCursor  # rows are read by column name
        return self.cur

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _rows(n: int, shown: int, days_needed: int = 31) -> list[dict]:
    return [{"rec_id": f"rec-{i:02d}", "n": n, "days_needed": days_needed} for i in range(shown)]


def _canary(monkeypatch, rows, days=14):
    cur = _Cur(rows)
    monkeypatch.setattr(gcm.psycopg2, "connect", lambda url: _Conn(cur))
    return gcm.warn_stuck_post_kickoff_recs("postgresql://unused/db", days), cur


class TestStuckPostKickoffCanary:
    def test_logs_error_with_count_first_ids_and_the_voiding_command(self, monkeypatch, caplog):
        with caplog.at_level("INFO", logger="grade_completed_matches"):
            n, _ = _canary(monkeypatch, _rows(n=12, shown=10, days_needed=31))

        assert n == 12
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1, errors
        msg = errors[0]
        assert msg.startswith("12 pending/placed rec(s) written at or after kickoff")
        assert "--days 14 window" in msg
        assert "(first 10: rec-00, rec-01, rec-02, rec-03, rec-04, rec-05, rec-06, rec-07, rec-08, rec-09)" in msg
        assert msg.endswith("Void them with: python /app/scripts/grade_completed_matches.py --days 31")

    def test_silent_when_nothing_is_stuck(self, monkeypatch, caplog):
        with caplog.at_level("DEBUG", logger="grade_completed_matches"):
            n, cur = _canary(monkeypatch, [])
        assert n == 0
        assert len(cur.calls) == 1  # it did look
        assert [r for r in caplog.records if r.levelno >= 30] == []

    def test_query_selects_open_post_kickoff_recs_on_finished_matches_outside_the_window(self, monkeypatch):
        _, cur = _canary(monkeypatch, [], days=21)
        (sql, params), *rest = cur.calls
        assert not rest
        assert params == ("21",)
        # The grader's own rules (list_finished_matches / list_open_recs_for_match):
        for clause in (
            "FROM betting_recommendations br JOIN matches m ON m.id = br.match_id",
            "br.status IN ('pending', 'placed')",
            "m.status = 'finished'",
            "br.created_at >= m.match_date",
            # exactly the complement of list_finished_matches' window
            "m.match_date < NOW() - (%s || ' days')::interval",
            # count over every row, ids from the first 10
            "COUNT(*) OVER () AS n",
            "ORDER BY m.match_date, br.id LIMIT 10",
            # a window that reaches the oldest kickoff
            "CEIL(EXTRACT(EPOCH FROM NOW() - MIN(m.match_date) OVER ()) / 86400)::int + 1 AS days_needed",
        ):
            assert clause in sql, clause

    def test_window_is_the_complement_of_the_graders(self):
        # If list_finished_matches' window changes, the canary must follow.
        seen = []

        class Cur:
            def execute(self, sql, params=None):
                seen.append(" ".join(sql.split()))

            def fetchall(self):
                return []

        gcm.list_finished_matches(Cur(), 14, None)
        assert "m.match_date >= NOW() - (%s || ' days')::interval" in seen[0]


class TestMainAlwaysRunsTheCanary:
    def _main(self, monkeypatch, argv):
        calls = []
        monkeypatch.setattr(gcm, "run", lambda url, days, match_id: calls.append(("run", url, days, match_id)))
        monkeypatch.setattr(gcm, "warn_stuck_post_kickoff_recs", lambda url, days: calls.append(("canary", url, days)))
        rc = gcm.main(argv)
        return rc, calls

    def test_after_a_window_run(self, monkeypatch):
        rc, calls = self._main(monkeypatch, ["--database-url", "postgresql://x/db"])
        assert rc == 0
        assert calls == [("run", "postgresql://x/db", 14, None), ("canary", "postgresql://x/db", 14)]

    def test_after_a_single_match_run(self, monkeypatch):
        rc, calls = self._main(monkeypatch, ["--database-url", "postgresql://x/db", "--days", "30", "--match-id", "m1"])
        assert rc == 0
        assert calls == [("run", "postgresql://x/db", 30, "m1"), ("canary", "postgresql://x/db", 30)]
