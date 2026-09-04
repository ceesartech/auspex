"""Season-type (preseason) filtering across the downstream readers.

ESPN's event.season.type used to be dropped on ingest, so 147 NFL and
147 NBA preseason games sat in the corpus with no downstream filter:
they were MONEYLINE training targets, they filled NFL week-1 rolling
form (98 of 160 window slots), and — having no odds at all — their
NEUTRAL_DEFAULT closing lines (spread 0.0, total 45.0) were read back
out of features_cache and used to grade 460 predictions against a line
that never existed.

These tests pin the three-state exclusion predicate, its presence in
every query that needed it, the phantom-line grading guard, and the
backfill's date rules.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
ML_SRC = REPO_ROOT / "services" / "ml-models" / "src"

if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from utils.training_data import (  # noqa: E402
    MATCH_NOT_PRESEASON_SQL,
    NBA_MONEYLINE_TRAINING_QUERY,
    NFL_MONEYLINE_TRAINING_QUERY,
    NHL_MONEYLINE_TRAINING_QUERY,
    NHL_PUCK_LINE_TRAINING_QUERY,
    NHL_REGULATION_TRAINING_QUERY,
    NHL_TOTAL_TRAINING_QUERY,
    is_preseason,
    preseason_exclusion_sql,
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── The predicate itself ─────────────────────────────────────────────


class TestExclusionPredicate:
    """Three marker states, and the third is a deliberate choice:
    an unmarked row is UNKNOWN, not 'regular'. A COALESCE default
    would silently re-admit the 294 known historical preseason games."""

    def test_preseason_row_is_excluded(self):
        assert is_preseason({"season_type": "preseason"}) is True

    def test_regular_row_is_included(self):
        assert is_preseason({"season_type": "regular"}) is False

    def test_postseason_row_is_included(self):
        assert is_preseason({"season_type": "postseason"}) is False

    def test_legacy_nhl_game_type_regular_is_included(self):
        # The 6,551 rows load_nhl_historical.py stamped carry no
        # season_type at all — only metadata.game_type.
        assert is_preseason({"game_type": "regular"}) is False

    def test_legacy_nhl_game_type_playoff_is_included(self):
        assert is_preseason({"game_type": "playoff"}) is False

    def test_unmarked_row_is_included_as_unknown(self):
        assert is_preseason({}) is False
        assert is_preseason(None) is False

    def test_legacy_marker_wins_over_a_contradictory_season_type(self):
        # The loader verified these; a contradictory season_type would
        # be the newer, less trustworthy signal.
        assert is_preseason({"game_type": "regular", "season_type": "preseason"}) is False

    def test_unrecognised_season_type_is_included(self):
        # fetch_upcoming stores nothing for an unexpected ESPN value,
        # but a hand-edited row must not silently vanish either.
        assert is_preseason({"season_type": "friendly"}) is False


class TestExclusionSql:
    def test_sql_mirrors_the_python_semantics(self):
        sql = preseason_exclusion_sql("m")
        assert "m.metadata->>'season_type' IS DISTINCT FROM 'preseason'" in sql
        assert "m.metadata->>'game_type' IN ('regular', 'playoff')" in sql
        # Parenthesised so it can be AND-ed into any WHERE clause
        # without the OR arm leaking across the surrounding predicates.
        assert sql.startswith("(") and sql.endswith(")")

    def test_alias_is_honoured(self):
        assert "x.metadata->>'season_type'" in preseason_exclusion_sql("x")

    def test_no_bare_percent(self):
        # These fragments land inside psycopg2-parameterised SQL.
        assert "%" not in preseason_exclusion_sql("m")


# ── A. Training frames ───────────────────────────────────────────────


class TestTrainingQueriesExcludePreseason:
    """The NFL and NBA MONEYLINE queries have no odds gate, so
    preseason rows WERE training targets (147 of 1,004 NFL rows =
    14.6 percent). The NHL queries are clean today only because the
    loader filtered them; the ESPN results path will start ingesting
    NHL preseason around 2026-09-20."""

    @pytest.mark.parametrize(
        "query",
        [
            NFL_MONEYLINE_TRAINING_QUERY,
            NBA_MONEYLINE_TRAINING_QUERY,
            NHL_MONEYLINE_TRAINING_QUERY,
            NHL_REGULATION_TRAINING_QUERY,
            NHL_PUCK_LINE_TRAINING_QUERY,
            NHL_TOTAL_TRAINING_QUERY,
        ],
    )
    def test_query_carries_the_shared_predicate(self, query):
        assert MATCH_NOT_PRESEASON_SQL in query

    def test_soccer_query_is_untouched(self):
        from utils.training_data import DEFAULT_TRAINING_QUERY

        assert "season_type" not in DEFAULT_TRAINING_QUERY

    def test_tennis_and_mma_queries_are_untouched(self):
        from utils.training_data import MMA_MONEYLINE_TRAINING_QUERY, TENNIS_MONEYLINE_TRAINING_QUERY

        assert "season_type" not in TENNIS_MONEYLINE_TRAINING_QUERY
        assert "season_type" not in MMA_MONEYLINE_TRAINING_QUERY


# ── B. Rolling-form windows ──────────────────────────────────────────


class _RollingCursor:
    """Captures the SQL a fetch_* helper issues so we can assert the
    window is season-type filtered."""

    def __init__(self):
        self.sql = []

    def execute(self, sql, params=None):
        self.sql.append(" ".join(sql.split()))

    def fetchone(self):
        return {}

    def fetchall(self):
        return []


class TestRollingWindowsSkipPreseason:
    """NFL week-1 2026 rolling windows were 98/160 August preseason
    games; preseason scoring is structurally different (mean total
    37.37, sd 11.44, n=147 vs 44.85, sd 14.22, n=857), and the model
    picked UNDER on 15 of 16 week-1 totals."""

    def test_nfl_rolling_window_filters_preseason(self):
        fnfl = _load("compute_features_nfl", "compute_features_nfl.py")
        cur = _RollingCursor()
        fnfl.fetch_team_rolling(cur, "team-1", dt.date(2026, 9, 10))
        assert len(cur.sql) == 1
        assert "season_type" in cur.sql[0]
        assert "IS DISTINCT FROM 'preseason'" in cur.sql[0]

    def test_nba_rolling_window_filters_preseason(self):
        fnba = _load("compute_features_nba", "compute_features_nba.py")
        cur = _RollingCursor()
        fnba.fetch_team_rolling(cur, "team-1", dt.date(2026, 10, 25))
        assert len(cur.sql) == 1
        assert "IS DISTINCT FROM 'preseason'" in cur.sql[0]

    def test_nhl_rolling_windows_filter_preseason(self):
        fnhl = _load("compute_features_nhl", "compute_features_nhl.py")
        for fn, args in (
            (fnhl._rolling_nhl_form, ("team-1", dt.date(2026, 10, 10), "home")),
            (fnhl._rolling_pace_stats, ("team-1", dt.date(2026, 10, 10), "home")),
            (fnhl._rolling_5v5_stats, ("team-1", dt.date(2026, 10, 10), "home")),
            (fnhl._goalie_rolling_form, ("goalie-1", dt.date(2026, 10, 10), "home")),
        ):
            cur = _RollingCursor()
            fn(cur, *args)
            assert len(cur.sql) == 1, fn.__name__
            assert "IS DISTINCT FROM 'preseason'" in cur.sql[0], fn.__name__

    def test_nfl_window_size_is_unchanged(self):
        # The window reaches further BACK in time, but it is still the
        # last 5 games — the fix must not silently widen the window.
        fnfl = _load("compute_features_nfl", "compute_features_nfl.py")
        assert fnfl.WINDOW == 5


# ── D. Phantom-line grading guard ────────────────────────────────────


_load("grading_outcomes", "grading_outcomes.py")
gcm = _load("grade_completed_matches", "grade_completed_matches.py")


class _GradingCursor:
    """Cursor stub for grade_match: serves the grading-context row
    (features + odds-presence probes), the ungraded-prediction list,
    and swallows the UPDATE."""

    def __init__(self, predictions, features=None, has_spread_odds=True, has_total_odds=True):
        self.predictions = {p["id"]: dict(p) for p in predictions}
        self.features = features
        self.has_spread_odds = has_spread_odds
        self.has_total_odds = has_total_odds
        self._last = None

    def execute(self, sql, params=None):
        norm = " ".join(sql.split())
        if "FROM features_cache" in norm:
            self._last = (
                "ctx",
                {
                    "features": self.features,
                    "has_spread_odds": self.has_spread_odds,
                    "has_total_odds": self.has_total_odds,
                },
            )
            return
        if "FROM predictions" in norm and "is_correct IS NULL" in norm:
            self._last = (
                "preds",
                [
                    {
                        "prediction_id": p["id"],
                        "prediction_type": p["prediction_type"],
                        "model_name": p["model_name"],
                        "predicted_outcome": p["predicted_outcome"],
                    }
                    for p in self.predictions.values()
                    if p.get("is_correct") is None
                ],
            )
            return
        if norm.startswith("UPDATE predictions"):
            actual, correct, pred_id = params
            self.predictions[pred_id]["actual_outcome"] = actual
            self.predictions[pred_id]["is_correct"] = correct
            self._last = ("update", None)
            return
        if "FROM betting_recommendations br" in norm:
            self._last = ("recs", [])
            return
        raise AssertionError(f"Unexpected SQL: {norm[:90]}")

    def fetchone(self):
        kind, row = self._last or (None, None)
        return row if kind == "ctx" else None

    def fetchall(self):
        kind, rows = self._last or (None, [])
        return rows if kind in ("preds", "recs") else []


def _pred(pid, ptype, model_name, outcome):
    return {
        "id": pid,
        "prediction_type": ptype,
        "model_name": model_name,
        "predicted_outcome": outcome,
        "is_correct": None,
        "actual_outcome": None,
    }


class TestPhantomLineGuard:
    """compute_features_nfl.NEUTRAL_DEFAULTS filled closing_total_line
    = 45.0 and closing_spread_home = 0.0 for 49/49 August-2026 games
    that no book ever priced, and the grader read them back as "the
    closing line the model conditioned on" — 230 TOTAL and 230 SPREAD
    verdicts against a fabricated line (real preseason total averaged
    37.4)."""

    MATCH = {"match_id": "m-pre", "home_score": 17, "away_score": 20, "metadata": None}

    def test_total_is_left_ungraded_when_no_book_priced_the_market(self):
        preds = [_pred("p1", "total", "ensemble_nfl_tot", "over")]
        cur = _GradingCursor(preds, features={"closing_total_line": 45.0}, has_total_odds=False)
        stats: dict = {}

        counts = gcm.grade_match(cur, self.MATCH, stats)

        assert counts["predictions_graded"] == 0
        assert counts["predictions_skipped"] == 1
        assert stats["phantom_line_skips"] == 1
        # Never marked correct OR incorrect.
        assert cur.predictions["p1"]["is_correct"] is None
        assert cur.predictions["p1"]["actual_outcome"] is None

    def test_spread_is_left_ungraded_when_no_book_priced_the_market(self):
        preds = [_pred("p1", "spread", "ensemble_nfl_sp", "home")]
        cur = _GradingCursor(preds, features={"closing_spread_home": 0.0}, has_spread_odds=False)
        stats: dict = {}

        counts = gcm.grade_match(cur, self.MATCH, stats)

        assert counts["predictions_skipped"] == 1
        assert stats["phantom_line_skips"] == 1
        assert cur.predictions["p1"]["is_correct"] is None

    def test_grades_normally_when_the_market_has_real_odds(self):
        # 17-20 = 37 total, line 45 -> under. The pick was 'over'.
        preds = [_pred("p1", "total", "ensemble_nfl_tot", "over")]
        cur = _GradingCursor(preds, features={"closing_total_line": 45.0}, has_total_odds=True)
        stats: dict = {}

        counts = gcm.grade_match(cur, self.MATCH, stats)

        assert counts["predictions_graded"] == 1
        assert stats.get("phantom_line_skips", 0) == 0
        assert cur.predictions["p1"]["actual_outcome"] == "under"
        assert cur.predictions["p1"]["is_correct"] is False

    def test_nhl_spread_is_not_line_dependent_and_grades_without_odds(self):
        # NHL's puck line is the constant -1.5, not a book line, so an
        # odds-free NHL spread prediction is still gradable.
        preds = [_pred("p1", "spread", "ensemble_nhl_pl", "cover")]
        cur = _GradingCursor(preds, features=None, has_spread_odds=False)
        stats: dict = {}

        counts = gcm.grade_match(cur, self.MATCH, stats)

        assert counts["predictions_graded"] == 1
        assert stats.get("phantom_line_skips", 0) == 0
        assert cur.predictions["p1"]["actual_outcome"] == "no_cover"

    def test_moneyline_is_never_guarded(self):
        preds = [_pred("p1", "moneyline", "ensemble_nfl_ml", "away")]
        cur = _GradingCursor(preds, features=None, has_spread_odds=False, has_total_odds=False)

        counts = gcm.grade_match(cur, self.MATCH)

        assert counts["predictions_graded"] == 1
        assert cur.predictions["p1"]["is_correct"] is True

    def test_is_line_dependent_dispatch(self):
        assert gcm.is_line_dependent("total", "ensemble_nfl_tot") is True
        assert gcm.is_line_dependent("total", "ensemble_nba_tot") is True
        assert gcm.is_line_dependent("spread", "ensemble_nfl_sp") is True
        assert gcm.is_line_dependent("spread", "ensemble_nba_sp") is True
        assert gcm.is_line_dependent("spread", "ensemble_nhl_pl") is False
        assert gcm.is_line_dependent("total", "ensemble_nhl_tot") is False
        assert gcm.is_line_dependent("moneyline", "ensemble_nfl_ml") is False

    def test_probe_columns_absent_grades_as_before(self):
        # A cursor that predates the probe columns must not silently
        # start skipping everything — it warns (see
        # fetch_match_grading_context) and grades as before.
        class _OldCursor(_GradingCursor):
            def fetchone(self):
                kind, row = self._last or (None, None)
                if kind != "ctx":
                    return None
                return {"features": self.features}

        preds = [_pred("p1", "total", "ensemble_nfl_tot", "under")]
        cur = _OldCursor(preds, features={"closing_total_line": 45.0})
        counts = gcm.grade_match(cur, self.MATCH)
        assert counts["predictions_graded"] == 1

    def test_back_compat_features_shim(self):
        cur = _GradingCursor([], features={"closing_total_line": 41.5})
        assert gcm.fetch_match_features(cur, "m-pre") == {"closing_total_line": 41.5}


# ── E. Backfill date rules ───────────────────────────────────────────


bst = _load("backfill_season_type", "backfill_season_type.py")


class TestBackfillDateRules:
    def test_august_nfl_game_is_preseason(self):
        season_type, rule = bst.classify("nfl", dt.date(2024, 8, 17))
        assert season_type == "preseason"
        assert rule == "nfl:august"

    def test_september_nfl_game_is_regular(self):
        season_type, rule = bst.classify("nfl", dt.date(2024, 9, 8))
        assert season_type == "regular"
        assert rule == "nfl:sep-dec"

    def test_january_nfl_game_is_left_alone(self):
        # Weeks 17-18 and the Wild Card round share January.
        season_type, _ = bst.classify("nfl", dt.date(2025, 1, 12))
        assert season_type is None

    def test_nba_october_before_opener_is_preseason(self):
        # Verified on prod: 73 games on 2023-10-05..21, opener 10-24.
        assert bst.classify("nba", dt.date(2023, 10, 12))[0] == "preseason"
        assert bst.classify("nba", dt.date(2024, 10, 8))[0] == "preseason"

    def test_nba_october_on_or_after_opener_is_regular(self):
        assert bst.classify("nba", dt.date(2023, 10, 24))[0] == "regular"
        assert bst.classify("nba", dt.date(2024, 10, 23))[0] == "regular"

    def test_nba_october_without_a_known_opener_is_left_alone(self):
        season_type, rule = bst.classify("nba", dt.date(1999, 10, 12))
        assert season_type is None
        assert rule == "nba:october-no-opener-in-table"

    def test_nba_midwinter_is_regular_and_playoff_months_are_left_alone(self):
        assert bst.classify("nba", dt.date(2025, 1, 15))[0] == "regular"
        assert bst.classify("nba", dt.date(2025, 5, 2))[0] is None

    def test_nhl_stamps_nothing(self):
        # The loader already dropped gameType != 2/3 and wrote
        # metadata.game_type on 6,551 rows.
        assert bst.classify("nhl", dt.date(2025, 9, 22))[0] is None
        assert bst.classify("nhl", dt.date(2025, 12, 22))[0] is None

    def test_unknown_sport_is_loud(self):
        with pytest.raises(ValueError):
            bst.classify("soccer", dt.date(2025, 8, 1))

    def test_dry_run_is_the_default(self):
        args = bst.parse_args([])
        assert args.apply is False

    def test_sport_flag_is_repeatable(self):
        args = bst.parse_args(["--sport", "nfl", "--sport", "nba"])
        assert args.sports == ["nfl", "nba"]

    def test_unknown_sport_flag_exits_nonzero(self):
        rc = bst.main(["--sport", "cricket", "--database-url", "postgres://unused"])
        assert rc == 2

    def test_csv_name_is_scoped_and_timestamped(self):
        name = bst.default_out_csv(["nfl", "nba"])
        # ABSOLUTE, in the repo's artifact directory: the documented
        # invocation runs in the api container whose WORKDIR is
        # /app/services/api/src, so a relative path would drop the only
        # record of the backfill into the bind-mounted source tree.
        assert name.startswith("/app/backups/season_type_backfill_nba-nfl_")
        assert name.endswith(".csv")

    def test_csv_dir_follows_backup_local_dir(self, monkeypatch):
        monkeypatch.setenv("BACKUP_LOCAL_DIR", "/mnt/elsewhere")
        assert bst.default_out_csv(["nfl"]).startswith("/mnt/elsewhere/season_type_backfill_nfl_")

    def test_csv_write_is_write_once(self, tmp_path):
        target = tmp_path / "out.csv"
        bst.write_csv(str(target), [])
        with pytest.raises(FileExistsError):
            bst.write_csv(str(target), [])

    def test_csv_contains_the_affected_ids(self, tmp_path):
        target = tmp_path / "out.csv"
        bst.write_csv(
            str(target),
            [
                {
                    "match_id": "abc",
                    "sport": "nfl",
                    "match_date": dt.date(2024, 8, 17),
                    "new_season_type": "preseason",
                    "rule": "nfl:august",
                }
            ],
        )
        body = target.read_text()
        assert "match_id" in body.splitlines()[0]
        assert "abc" in body
        assert "preseason" in body


# ── Guard against the money path drifting ────────────────────────────


class TestGradingMathUnchanged:
    def test_profit_loss_math_is_untouched(self):
        status, pl = gcm.profit_loss_for(True, False, Decimal("100"), Decimal("2.50"))
        assert (status, pl) == ("won", Decimal("150.00"))
