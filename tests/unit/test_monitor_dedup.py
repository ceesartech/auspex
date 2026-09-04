"""monitor_models counts each MATCH once, not each predictions row.

Weekly retrains write a fresh predictions row for the same
(match, market) under a new model_version, and grading fills
is_correct on every one of them. The old slice queries counted them
all: the NFL 30-day slice reported n = 220 / 220 / 226 for only
47 / 49 / 49 distinct matches (4.7x inflation of near-duplicate,
strongly correlated rows), and MMA's headline "n = 444" was 121
distinct fights. That inflation affected EVERY sport, and it made the
ECE/Brier confidence implied by n badly overstated.

The fix is DISTINCT ON (match_id, prediction_type) ordered by
created_at DESC — the newest graded row per match+market, i.e. the
version actually serving — plus the shared preseason exclusion.

These tests assert BOTH halves: the SQL text (so the DISTINCT ON can't
be quietly dropped) and the resulting rows through a cursor stub that
emulates the DISTINCT ON / preseason semantics over in-memory rows.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
ML_SRC = REPO_ROOT / "services" / "ml-models" / "src"

if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from utils.training_data import is_preseason  # noqa: E402


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mm = _load("monitor_models", "monitor_models.py")


DISTINCT_ON = "SELECT DISTINCT ON (p.match_id, p.prediction_type)"
ORDER_BY = "ORDER BY p.match_id, p.prediction_type, p.created_at DESC"


def _row(match_id, ptype, model_name, prob, correct, created_at, sport="nfl", metadata=None, is_correct_null=False):
    return {
        "match_id": match_id,
        "prediction_type": ptype,
        "model_name": model_name,
        "picked_prob": prob,
        "correct": correct,
        "created_at": created_at,
        "sport": sport,
        "metadata": metadata or {},
        "is_correct_null": is_correct_null,
    }


class FakeCursor:
    """Emulates the two SQL behaviours the slice queries rely on:
    the preseason/push filters and DISTINCT ON + ORDER BY created_at
    DESC. Refuses to emulate a query that doesn't actually ask for
    them, so the stub can never be more forgiving than Postgres."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.sql: list[str] = []
        self._last: list[dict] = []

    def _deduped(self):
        kept = [r for r in self.rows if not r["is_correct_null"] and not is_preseason(r["metadata"])]
        best: dict[tuple, dict] = {}
        for r in kept:
            key = (r["match_id"], r["prediction_type"])
            if key not in best or r["created_at"] > best[key]["created_at"]:
                best[key] = r
        return list(best.values())

    def execute(self, sql, params=None):
        norm = " ".join(sql.split())
        self.sql.append(norm)
        assert " ".join(DISTINCT_ON.split()) in norm, "slice query lost its DISTINCT ON"
        assert " ".join(ORDER_BY.split()) in norm, "slice query lost its newest-row ordering"
        assert "season_type" in norm, "slice query lost the preseason exclusion"
        rows = self._deduped()
        if "COUNT(*) AS n" in norm:
            grouped: dict[tuple, int] = {}
            for r in rows:
                key = (r["sport"], r["prediction_type"], r["model_name"])
                grouped[key] = grouped.get(key, 0) + 1
            out = [
                {"sport": s, "prediction_type": t, "model_name": m, "n": n}
                for (s, t, m), n in sorted(grouped.items())
                if n >= params["min_samples"]
            ]
            self._last = out
            return
        self._last = [
            {"picked_prob": r["picked_prob"], "correct": r["correct"]}
            for r in rows
            if r["sport"] == params["sport"]
            and r["prediction_type"] == params["prediction_type"]
            and r["model_name"] == params["model_name"]
        ]

    def fetchall(self):
        return self._last


T0 = dt.datetime(2026, 8, 1, 12, 0)
T1 = dt.datetime(2026, 8, 8, 12, 0)
T2 = dt.datetime(2026, 8, 15, 12, 0)


class TestSqlShape:
    def test_slice_pairs_sql_uses_distinct_on(self):
        cur = FakeCursor([])
        mm.fetch_slice_pairs(cur, "nfl", "total", "ensemble_nfl_tot", 30)
        assert DISTINCT_ON in cur.sql[0].replace("  ", " ")

    def test_slices_sql_uses_distinct_on(self):
        cur = FakeCursor([])
        mm.fetch_slices(cur, 30, 10)
        assert DISTINCT_ON in cur.sql[0].replace("  ", " ")

    def test_pushes_stay_excluded(self):
        cur = FakeCursor([])
        mm.fetch_slices(cur, 30, 10)
        assert "p.is_correct IS NOT NULL" in cur.sql[0]

    def test_finished_and_window_filters_survive(self):
        cur = FakeCursor([])
        mm.fetch_slices(cur, 30, 10)
        assert "m.status = 'finished'" in cur.sql[0]
        assert "m.match_date >= NOW() - (%(days)s || ' days')::interval" in cur.sql[0]

    def test_no_bare_percent_in_the_shared_fragment(self):
        # psycopg2 interpolates the whole string, comments included.
        stripped = mm._DEDUPED_GRADED_PREDICTIONS_SQL.replace("%(days)s", "")
        assert "%" not in stripped


class TestDeduplication:
    def test_three_retrain_rows_for_one_match_count_once(self):
        rows = [
            _row("m1", "total", "ensemble_nfl_tot", 0.61, 1, T0),
            _row("m1", "total", "ensemble_nfl_tot", 0.58, 1, T1),
            _row("m1", "total", "ensemble_nfl_tot", 0.55, 1, T2),
        ]
        cur = FakeCursor(rows)
        predicted, actual = mm.fetch_slice_pairs(cur, "nfl", "total", "ensemble_nfl_tot", 30)
        assert len(predicted) == 1 and len(actual) == 1

    def test_the_newest_row_is_the_one_kept(self):
        rows = [
            _row("m1", "total", "ensemble_nfl_tot", 0.61, 1, T0),
            _row("m1", "total", "ensemble_nfl_tot", 0.55, 0, T2),
            _row("m1", "total", "ensemble_nfl_tot", 0.58, 1, T1),
        ]
        cur = FakeCursor(rows)
        predicted, actual = mm.fetch_slice_pairs(cur, "nfl", "total", "ensemble_nfl_tot", 30)
        assert predicted == [0.55]
        assert actual == [0]

    def test_distinct_markets_on_the_same_match_are_both_kept(self):
        rows = [
            _row("m1", "total", "ensemble_nfl_tot", 0.55, 1, T2),
            _row("m1", "spread", "ensemble_nfl_tot", 0.52, 0, T2),
        ]
        cur = FakeCursor(rows)
        totals, _ = mm.fetch_slice_pairs(cur, "nfl", "total", "ensemble_nfl_tot", 30)
        spreads, _ = mm.fetch_slice_pairs(cur, "nfl", "spread", "ensemble_nfl_tot", 30)
        assert len(totals) == 1 and len(spreads) == 1

    def test_distinct_matches_are_all_kept(self):
        rows = [
            _row("m1", "total", "ensemble_nfl_tot", 0.55, 1, T2),
            _row("m2", "total", "ensemble_nfl_tot", 0.61, 0, T2),
            _row("m3", "total", "ensemble_nfl_tot", 0.49, 1, T1),
        ]
        cur = FakeCursor(rows)
        predicted, _ = mm.fetch_slice_pairs(cur, "nfl", "total", "ensemble_nfl_tot", 30)
        assert len(predicted) == 3

    def test_slice_count_matches_the_pair_count(self):
        # The report's n and the arrays the calibration math sees must
        # never disagree — they now run through the same subquery.
        rows = [
            _row("m1", "total", "ensemble_nfl_tot", 0.55, 1, T2),
            _row("m1", "total", "ensemble_nfl_tot", 0.61, 1, T0),
            _row("m2", "total", "ensemble_nfl_tot", 0.49, 0, T1),
        ]
        cur = FakeCursor(rows)
        slices = mm.fetch_slices(cur, 30, 1)
        assert len(slices) == 1
        assert slices[0]["n"] == 2
        predicted, _ = mm.fetch_slice_pairs(cur, "nfl", "total", "ensemble_nfl_tot", 30)
        assert len(predicted) == slices[0]["n"]

    def test_min_samples_gate_sees_the_deflated_n(self):
        # 5 retrain copies of one match used to clear a min-samples=3
        # gate. One real match must not.
        rows = [_row("m1", "total", "ensemble_nfl_tot", 0.55, 1, T0 + dt.timedelta(days=i)) for i in range(5)]
        cur = FakeCursor(rows)
        assert mm.fetch_slices(cur, 30, 3) == []


class TestPreseasonAndPushExclusion:
    def test_preseason_rows_are_dropped(self):
        rows = [
            _row("m1", "total", "ensemble_nfl_tot", 0.55, 1, T2, metadata={"season_type": "preseason"}),
            _row("m2", "total", "ensemble_nfl_tot", 0.61, 0, T2, metadata={"season_type": "regular"}),
        ]
        cur = FakeCursor(rows)
        predicted, _ = mm.fetch_slice_pairs(cur, "nfl", "total", "ensemble_nfl_tot", 30)
        assert predicted == [0.61]

    def test_legacy_nhl_game_type_rows_survive(self):
        rows = [_row("m1", "total", "ensemble_nhl_tot", 0.55, 1, T2, sport="nhl", metadata={"game_type": "regular"})]
        cur = FakeCursor(rows)
        predicted, _ = mm.fetch_slice_pairs(cur, "nhl", "total", "ensemble_nhl_tot", 30)
        assert predicted == [0.55]

    def test_unmarked_rows_survive(self):
        rows = [_row("m1", "total", "ensemble_nfl_tot", 0.55, 1, T2, metadata={})]
        cur = FakeCursor(rows)
        predicted, _ = mm.fetch_slice_pairs(cur, "nfl", "total", "ensemble_nfl_tot", 30)
        assert predicted == [0.55]

    def test_pushes_are_dropped_before_dedup(self):
        # A push written AFTER the graded row must not shadow it: the
        # is_correct IS NOT NULL filter runs inside the subquery, so
        # DISTINCT ON only ever sees graded rows.
        rows = [
            _row("m1", "total", "ensemble_nfl_tot", 0.55, 1, T0),
            _row("m1", "total", "ensemble_nfl_tot", None, None, T2, is_correct_null=True),
        ]
        cur = FakeCursor(rows)
        predicted, actual = mm.fetch_slice_pairs(cur, "nfl", "total", "ensemble_nfl_tot", 30)
        assert predicted == [0.55] and actual == [1]


class TestThresholdsUntouched:
    def test_horse_racing_slices_are_unchanged(self):
        # Horse racing comes from race_predictions (per-entrant win
        # prob vs actual), which has no retrain-duplication problem to
        # collapse and no season type to filter. Left alone on purpose.
        for fn in (mm.fetch_horse_racing_slices, mm.fetch_horse_racing_pairs):
            source = inspect.getsource(fn)
            assert "DISTINCT ON" not in source
            assert "season_type" not in source

    def test_docstring_warns_that_n_changed(self):
        doc = mm.fetch_slices.__doc__ or ""
        assert "220" in doc and "444" in doc
        assert "n-sensitive" in doc
