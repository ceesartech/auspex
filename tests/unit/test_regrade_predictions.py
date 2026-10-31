"""regrade_predictions: planning (pure), the write-once CSV, and the run()
ordering guarantees against a stub connection — no Postgres needed."""

from __future__ import annotations

import csv
import importlib.util
import sys
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


_load("grading_outcomes", "grading_outcomes.py")
_load("grade_completed_matches", "grade_completed_matches.py")
rp = _load("regrade_predictions", "regrade_predictions.py")

NHL_CS_KEYS = {"3-2": 0.055, "2-2": 0.052, "4-3": 0.039, "2-3": 0.047, "other": 0.807}


def _row(pid, ptype, model_name, pick, actual, correct, *, match_id="m1", h=4, a=3, metadata=None, **extra):
    row = {
        "prediction_id": pid,
        "match_id": match_id,
        "prediction_type": ptype,
        "model_name": model_name,
        "predicted_outcome": pick,
        "probabilities": None,
        "actual_outcome": actual,
        "is_correct": correct,
        "after_kickoff": False,
        "home_score": h,
        "away_score": a,
        "metadata": metadata if metadata is not None else {"regulation_winner": "tie"},
        "half_time_home": None,
        "half_time_away": None,
    }
    row.update(extra)
    return row


def _nhl_rows():
    """The two 2026-10-31 defects on one 4-3 OT game, plus rows that must
    not move."""
    return [
        # '12' was credited on the final score; regulation was tied.
        _row("dc", "double_chance", "dixon_coles_nhl", "12", "home", True),
        # 'other' never credited; 5-2 is not among the listed keys.
        _row("cs", "correct_score", "dixon_coles_nhl", "other", "5-2", False, h=5, a=2, probabilities=NHL_CS_KEYS),
        # Already right under the current code.
        _row("rx", "regulation_1x2", "dixon_coles_nhl", "home", "draw", False),
        _row("sc", "double_chance", "ensemble", "12", "draw", False, h=1, a=1, metadata={}),
        _row("ou", "over_under", "ensemble", "over_7", "push_7", None),
    ]


# ── parse_types ──────────────────────────────────────────────────────


class TestParseTypes:
    def test_repeatable_and_comma_separated(self):
        assert rp.parse_types(["double_chance,correct_score", "btts", "correct_score"]) == [
            "btts",
            "correct_score",
            "double_chance",
        ]

    def test_absent_or_blank_means_every_type(self):
        assert rp.parse_types(None) is None
        assert rp.parse_types([" , "]) is None


# ── plan_regrade ─────────────────────────────────────────────────────


class TestPlanRegrade:
    def test_only_changed_rows_are_planned(self):
        changes, counts = rp.plan_regrade(_nhl_rows(), {})

        by_id = {c["prediction_id"]: c for c in changes}
        assert set(by_id) == {"dc", "cs"}
        assert (by_id["dc"]["old_actual_outcome"], by_id["dc"]["old_is_correct"]) == ("home", True)
        assert (by_id["dc"]["new_actual_outcome"], by_id["dc"]["new_is_correct"]) == ("draw", False)
        assert (by_id["cs"]["new_actual_outcome"], by_id["cs"]["new_is_correct"]) == ("5-2", True)
        assert counts["changed"] == 2 and counts["unchanged"] == 3

    def test_after_kickoff_rows_are_skipped(self):
        rows = [_row("late", "double_chance", "dixon_coles_nhl", "12", "home", True, after_kickoff=True)]
        changes, counts = rp.plan_regrade(rows, {})
        assert changes == [] and counts["skipped_after_kickoff"] == 1

    def test_rows_the_current_code_cannot_grade_are_never_ungraded(self):
        # regulation_winner missing: the stale verdict stays, counted, not NULLed.
        rows = [_row("dc", "double_chance", "dixon_coles_nhl", "12", "home", True, metadata={})]
        changes, counts = rp.plan_regrade(rows, {})
        assert changes == [] and counts["skipped_ungradable"] == 1

    def test_a_verdict_is_never_nulled_without_a_push(self, caplog):
        # An 'other' pick without its scoreline keys grades (actual, NULL):
        # what 'other' excluded is unknown. The old False stays, counted + logged.
        rows = [_row("cs0", "correct_score", "dixon_coles_nhl", "other", "5-2", False, h=5, a=2)]
        assert rows[0]["probabilities"] is None

        with caplog.at_level("WARNING", logger="regrade_predictions"):
            changes, counts = rp.plan_regrade(rows, {})

        assert changes == []
        assert counts["skipped_would_ungrade"] == 1 and counts["changed"] == 0
        assert any("cs0" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)

    def test_a_push_may_replace_a_verdict(self):
        # 2-1 against over_3 is a push: NULL is the right verdict there,
        # not an un-grade.
        rows = [_row("ou3", "over_under", "ensemble", "over_3", "under", False, h=2, a=1, metadata={})]
        changes, counts = rp.plan_regrade(rows, {})
        assert counts["skipped_would_ungrade"] == 0 and counts["changed"] == 1
        assert changes[0]["new_is_correct"] is None
        assert changes[0]["new_actual_outcome"].startswith("push")

    def test_line_dependent_rows_need_their_context(self):
        rows = [_row("t", "total", "ensemble_nfl_tot", "over", "over", True, h=30, a=20)]
        with pytest.raises(ValueError, match="no grading context"):
            rp.plan_regrade(rows, {})
        # With the probes: no real total line -> phantom-line skip, untouched.
        ctx = {"features": {"closing_total_line": 45.0}, "has_spread_odds": True, "has_total_odds": False}
        changes, counts = rp.plan_regrade(rows, {"m1": ctx})
        assert changes == [] and counts["skipped_phantom_line"] == 1
        assert rp.matches_needing_context(rows + _nhl_rows()) == {"m1"}
        assert rp.matches_needing_context(_nhl_rows()) == set()


# ── summary ──────────────────────────────────────────────────────────


class TestSummary:
    def test_before_and_after_accuracy_per_type(self):
        rows = _nhl_rows() + [
            _row(
                "dc2",
                "double_chance",
                "dixon_coles_nhl",
                "12",
                "away",
                True,
                match_id="m2",
                h=1,
                a=3,
                metadata={"regulation_winner": "away"},
            ),
        ]
        changes, _ = rp.plan_regrade(rows, {})
        s = rp.summarize(rows, changes)

        dc = s[("dixon_coles_nhl", "double_chance")]
        assert (dc["rows"], dc["matches"]) == (2, 2)
        assert (dc["before_correct"], dc["before_graded"]) == (2, 2)
        assert (dc["after_correct"], dc["after_graded"]) == (1, 2)
        assert (dc["changed"], dc["to_true"], dc["to_false"]) == (1, 0, 1)
        cs = s[("dixon_coles_nhl", "correct_score")]
        assert (cs["before_correct"], cs["after_correct"], cs["to_true"]) == (0, 1, 1)
        # A push row: graded on neither side.
        ou = s[("ensemble", "over_under")]
        assert (ou["before_graded"], ou["after_graded"], ou["changed"]) == (0, 0, 0)

        lines = rp.format_summary(s)
        assert lines[0].startswith("model_name/prediction_type")
        assert any(
            "dixon_coles_nhl/double_chance" in ln and "2/2 (100.0%)" in ln and "1/2 (50.0%)" in ln for ln in lines
        )


# ── write-once CSV ───────────────────────────────────────────────────


class TestCsv:
    def test_write_once(self, tmp_path):
        out = tmp_path / "sub" / "regrade.csv"
        changes, _ = rp.plan_regrade(_nhl_rows(), {})
        rp.write_csv(str(out), changes)
        first = out.read_text()

        with pytest.raises(FileExistsError, match="write-once"):
            rp.write_csv(str(out), [])
        assert out.read_text() == first
        got = list(csv.DictReader(out.open()))
        assert [r["prediction_id"] for r in got] == [c["prediction_id"] for c in changes]
        assert tuple(got[0]) == rp.CSV_FIELDS

    def test_default_path_honours_backup_local_dir(self, monkeypatch):
        monkeypatch.setenv("BACKUP_LOCAL_DIR", "/srv/bk")
        path = rp.default_out_csv()
        assert path.startswith("/srv/bk/regrade_") and path.endswith("Z.csv")
        monkeypatch.delenv("BACKUP_LOCAL_DIR")
        assert rp.default_out_csv().startswith("/app/backups/regrade_")


# ── run(): ordering + scope guarantees ───────────────────────────────


# Both halves of the plan-time guard, as GUARDED_UPDATE_SQL must spell
# them (whitespace-normalised). Dropping either lets a stale plan
# overwrite a row something else rewrote since.
UPDATE_GUARDS = (
    "WHERE id = %(prediction_id)s",
    "AND actual_outcome IS NOT DISTINCT FROM %(old_actual_outcome)s",
    "AND is_correct IS NOT DISTINCT FROM %(old_is_correct)s",
)


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rowcount = 0
        self._rows: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        sql_norm = " ".join(sql.split())
        assert "betting_recommendations" not in sql_norm, "regrade must never touch recs"
        self.conn.log.append(sql_norm[:40])
        if "FROM predictions p" in sql_norm and "JOIN matches" in sql_norm:
            self.conn.read_params = params
            self._rows = self.conn.rows
            return
        if sql_norm.startswith("UPDATE predictions"):
            assert not self.conn.readonly
            for guard in UPDATE_GUARDS:
                assert guard in sql_norm, f"UPDATE lacks its guard {guard!r}"
            # The audit CSV must already be on disk.
            assert Path(self.conn.out_csv).exists()
            if params["prediction_id"] in self.conn.boom:
                raise RuntimeError("connection lost")
            self.conn.updates.append(params)
            self.rowcount = 0 if params["prediction_id"] in self.conn.stale else 1
            return
        raise AssertionError(f"Unexpected SQL: {sql_norm[:80]}")

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, rows, out_csv, stale=(), boom=()):
        self.rows, self.out_csv, self.stale, self.boom = rows, out_csv, set(stale), set(boom)
        self.readonly = False
        self.updates: list = []
        self.log: list = []
        self.commits = 0
        self.read_params = None
        self.stale_file_at_commit = None

    def set_session(self, readonly=False):
        self.readonly = readonly

    def cursor(self, cursor_factory=None):
        return FakeCursor(self)

    def rollback(self):
        pass

    def commit(self):
        self.commits += 1
        stale_file = Path(rp.stale_ids_path(self.out_csv))
        self.stale_file_at_commit = stale_file.read_text() if stale_file.exists() else None


class TestRun:
    def test_dry_run_writes_the_csv_and_nothing_else(self, tmp_path):
        out = tmp_path / "dry.csv"
        conn = FakeConn(_nhl_rows(), str(out))

        counts = rp.run(conn, 365, "dixon_coles_nhl", ["correct_score", "double_chance"], False, str(out))

        assert conn.readonly and conn.updates == [] and conn.commits == 0
        assert conn.read_params == {
            "days": "365",
            "model_name": "dixon_coles_nhl",
            "types": ["correct_score", "double_chance"],
        }
        assert len(list(csv.DictReader(out.open()))) == 2
        assert counts["changed"] == 2 and counts["scanned"] == 5
        assert not Path(rp.stale_ids_path(str(out))).exists()

    def test_apply_updates_only_changed_rows_guarded_on_old_values(self, tmp_path):
        out = tmp_path / "apply.csv"
        conn = FakeConn(_nhl_rows(), str(out), stale={"cs"})

        counts = rp.run(conn, 30, None, None, True, str(out))

        assert {u["prediction_id"] for u in conn.updates} == {"dc", "cs"}
        dc = next(u for u in conn.updates if u["prediction_id"] == "dc")
        assert (dc["old_actual_outcome"], dc["old_is_correct"]) == ("home", True)
        assert (dc["new_actual_outcome"], dc["new_is_correct"]) == ("draw", False)
        assert counts["updated"] == 1 and counts["changed_since_plan"] == 1
        assert conn.commits == 1

    def test_stale_ids_are_logged_and_listed_before_the_commit(self, tmp_path, caplog):
        out = tmp_path / "apply.csv"
        conn = FakeConn(_nhl_rows(), str(out), stale={"cs"})

        with caplog.at_level("ERROR", logger="regrade_predictions"):
            rp.run(conn, 30, "dixon_coles_nhl", None, True, str(out))

        stale_file = Path(str(out) + ".stale.txt")
        assert stale_file.read_text() == "cs\n"
        assert conn.stale_file_at_commit == "cs\n"
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1 and "cs" in errors[0] and str(stale_file) in errors[0]
        # The rollback CSV still carries the stale row; the list says to skip it.
        assert {r["prediction_id"] for r in csv.DictReader(out.open())} == {"dc", "cs"}

    def test_a_clean_apply_leaves_an_empty_stale_list(self, tmp_path, caplog):
        out = tmp_path / "clean.csv"
        conn = FakeConn(_nhl_rows(), str(out))

        with caplog.at_level("ERROR", logger="regrade_predictions"):
            counts = rp.run(conn, 30, "dixon_coles_nhl", None, True, str(out))

        assert counts["updated"] == 2 and counts["changed_since_plan"] == 0
        assert Path(rp.stale_ids_path(str(out))).read_text() == ""
        assert not [r for r in caplog.records if r.levelname == "ERROR"]

    def test_an_existing_stale_list_refuses_before_any_update(self, tmp_path):
        out = tmp_path / "taken_stale.csv"
        stale_file = Path(rp.stale_ids_path(str(out)))
        stale_file.write_text("earlier run\n")
        conn = FakeConn(_nhl_rows(), str(out))

        with pytest.raises(FileExistsError, match="write-once"):
            rp.run(conn, 30, "dixon_coles_nhl", None, True, str(out))

        assert conn.updates == [] and conn.commits == 0
        assert stale_file.read_text() == "earlier run\n"

    def test_a_run_that_dies_mid_way_leaves_no_stale_list(self, tmp_path):
        # Nothing commits, so an empty list ("every row written") would lie.
        out = tmp_path / "boom.csv"
        conn = FakeConn(_nhl_rows(), str(out), boom={"cs"})

        with pytest.raises(RuntimeError, match="connection lost"):
            rp.run(conn, 30, "dixon_coles_nhl", None, True, str(out))

        assert conn.commits == 0
        assert not Path(rp.stale_ids_path(str(out))).exists()

    def test_the_fake_cursor_rejects_an_update_missing_a_guard(self, tmp_path, monkeypatch):
        # The guard assertions above have teeth: an UPDATE without either
        # IS NOT DISTINCT FROM clause never reaches the "database".
        for i, guard in enumerate(UPDATE_GUARDS[1:]):
            unguarded = " ".join(rp.GUARDED_UPDATE_SQL.split()).replace(guard, "")
            monkeypatch.setattr(rp, "GUARDED_UPDATE_SQL", unguarded)
            out = tmp_path / f"unguarded_{i}.csv"
            conn = FakeConn(_nhl_rows(), str(out))
            with pytest.raises(AssertionError, match="lacks its guard"):
                rp.run(conn, 30, "dixon_coles_nhl", None, True, str(out))
            assert conn.updates == [] and conn.commits == 0

    def test_existing_csv_refuses_before_any_update(self, tmp_path):
        out = tmp_path / "taken.csv"
        out.write_text("earlier run\n")
        conn = FakeConn(_nhl_rows(), str(out))
        with pytest.raises(FileExistsError):
            rp.run(conn, 30, "dixon_coles_nhl", None, True, str(out))
        assert conn.updates == [] and out.read_text() == "earlier run\n"

    def test_main_rejects_missing_url_and_bad_days(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert rp.main([]) == 2
        assert rp.main(["--database-url", "postgresql://x", "--days", "0"]) == 2

    def test_main_refuses_an_unscoped_regrade_before_connecting(self, monkeypatch):
        def no_connect(*_a, **_k):
            raise AssertionError("must refuse before connecting")

        monkeypatch.setattr(rp.psycopg2, "connect", no_connect)
        assert rp.main(["--database-url", "postgresql://x"]) == 2
        assert rp.main(["--database-url", "postgresql://x", "--prediction-type", " , "]) == 2
