"""Integration tests for migration 011's extended match-outcome trigger.

The trigger fires on UPDATE matches SET status='finished' and grades
every pending prediction row for that match. Before migration 011 the
trigger only handled soccer (match_result / over_under / btts);
migration 011 extends it to NHL (moneyline / regulation / puck_line /
total), NBA (moneyline / spread / total), and NFL (moneyline / spread
/ total).

For each (sport, market) combo we:
  1. Seed a league/teams/match/features_cache/prediction row
  2. UPDATE matches SET status='finished', home_score=..., away_score=...
  3. SELECT the prediction back and assert actual_outcome + is_correct

The spread/total tests JOIN to features_cache to confirm the trigger
picks up the closing line — that's the line-as-feature contract the
NBA and NFL models depend on. NHL fixed lines (puck_line -1.5,
total 5.5) are tested without any features_cache row.

Migration 027 makes the trigger skip predictions written after kickoff
and recs written at or after it (TestPostKickoffRows), so seeded
predictions default to an hour before kickoff. One test there also runs
scripts/grade_completed_matches.py against this database: the grader voids
the recs 027 leaves pending, and its canary names the ones it cannot reach.

These are integration tests — they need a real Postgres with all
migrations applied. They're auto-marked `integration` by
conftest.pytest_collection_modifyitems and run via:

    pytest -m integration services/data-ingestion/tests/e2e/test_match_outcome_trigger.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import text


@pytest.fixture
def trigger_tables(engine):
    """Truncate the tables the trigger reads + writes. Separate from
    the default clean_tables fixture so we don't break other e2e
    tests that don't want predictions/recs wiped."""
    tables = [
        "betting_recommendations",
        "predictions",
        "features_cache",
        "odds",
        "matches",
        "teams",
        "leagues",
    ]
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE " + ", ".join(tables) + " RESTART IDENTITY CASCADE"))
    yield


def _seed_match(conn, *, sport: str, metadata: dict | None = None) -> dict:
    """Insert one league/home/away/match. Returns IDs for the test."""
    league_id = str(uuid.uuid4())
    home_id = str(uuid.uuid4())
    away_id = str(uuid.uuid4())
    match_id = str(uuid.uuid4())
    match_dt = datetime.now(timezone.utc) - timedelta(hours=4)  # 4h ago

    conn.execute(
        text("INSERT INTO leagues (id, name, country, sport) VALUES (:id, :name, :country, :sport)"),
        {"id": league_id, "name": f"Test {sport} League", "country": "Testland", "sport": sport},
    )
    # normalized_name is NOT NULL with a UNIQUE(normalized_name, sport)
    # constraint — pin a unique value per test by suffixing the team id.
    conn.execute(
        text(
            "INSERT INTO teams (id, name, normalized_name, league_id, sport) "
            "VALUES (:id, :name, :norm, :lid, :sport)"
        ),
        {"id": home_id, "name": "Home FC", "norm": f"home_{home_id}", "lid": league_id, "sport": sport},
    )
    conn.execute(
        text(
            "INSERT INTO teams (id, name, normalized_name, league_id, sport) "
            "VALUES (:id, :name, :norm, :lid, :sport)"
        ),
        {"id": away_id, "name": "Away FC", "norm": f"away_{away_id}", "lid": league_id, "sport": sport},
    )
    conn.execute(
        text(
            """
            INSERT INTO matches (id, league_id, home_team_id, away_team_id,
                                 match_date, status, season, metadata)
            VALUES (:id, :lid, :hid, :aid, :dt, 'scheduled', :season, CAST(:md AS jsonb))
            """
        ),
        {
            "id": match_id,
            "lid": league_id,
            "hid": home_id,
            "aid": away_id,
            "dt": match_dt,
            "season": "2024-25",
            "md": json.dumps(metadata or {}),
        },
    )
    return {
        "match_id": match_id,
        "league_id": league_id,
        "home_id": home_id,
        "away_id": away_id,
        "kickoff": match_dt,
    }


def _seed_prediction(
    conn,
    *,
    match_id: str,
    model_name: str,
    prediction_type: str,
    predicted_outcome: str,
    model_version: str = "v1.0",
    probabilities: dict | None = None,
    created_at: datetime | None = None,
) -> str:
    """Insert one pending prediction row. Returns the prediction ID.

    Written an hour before the match's kickoff unless `created_at` says
    otherwise: since migration 027 the trigger never grades a prediction
    written after kickoff, and _seed_match puts kickoff 4h in the past, so
    a NOW() default would be one."""
    pred_id = str(uuid.uuid4())
    conn.execute(
        text(
            """
            INSERT INTO predictions
                (id, match_id, model_name, model_version, prediction_type,
                 predicted_outcome, confidence, probabilities, created_at)
            VALUES (:id, :mid, :mn, :mv, :pt, :po, :conf, CAST(:pr AS jsonb),
                    COALESCE(CAST(:ca AS timestamptz),
                             (SELECT m.match_date - INTERVAL '1 hour' FROM matches m WHERE m.id = :mid)))
            """
        ),
        {
            "id": pred_id,
            "mid": match_id,
            "mn": model_name,
            "mv": model_version,
            "pt": prediction_type,
            "po": predicted_outcome,
            "conf": 0.65,
            "pr": json.dumps(probabilities or {predicted_outcome: 0.65}),
            "ca": created_at,
        },
    )
    return pred_id


def _seed_rec(conn, *, match_id: str, prediction_id: str, selection: str, created_at: datetime | None) -> str:
    """Insert one pending recommendation written at `created_at` (None
    stores NULL). Returns the rec ID."""
    rec_id = str(uuid.uuid4())
    conn.execute(
        text(
            """
            INSERT INTO betting_recommendations
                (id, prediction_id, match_id, bet_type, selection,
                 odds_at_recommendation, bookmaker, recommended_stake, status, created_at)
            VALUES (:id, :pid, :mid, 'match_result', :sel, 2.0, 'test', 10, 'pending', :ca)
            """
        ),
        {"id": rec_id, "pid": prediction_id, "mid": match_id, "sel": selection, "ca": created_at},
    )
    return rec_id


def _read_rec(conn, rec_id: str) -> dict:
    row = (
        conn.execute(
            text("SELECT status, actual_result, settled_at FROM betting_recommendations WHERE id = :id"),
            {"id": rec_id},
        )
        .mappings()
        .one()
    )
    return dict(row)


def _seed_features(conn, *, match_id: str, feature_set: str, features: dict):
    """Insert a features_cache row so spread/total grading can read
    closing_spread_home / closing_total_line. Sets expires_at well
    in the future so the row is current."""
    conn.execute(
        text(
            """
            INSERT INTO features_cache
                (match_id, feature_set, features, feature_version, expires_at)
            VALUES (:mid, :fs, CAST(:f AS jsonb), 'v1', NOW() + INTERVAL '1 day')
            """
        ),
        {"mid": match_id, "fs": feature_set, "f": json.dumps(features)},
    )


def _finish_match(conn, *, match_id: str, home_score: int, away_score: int):
    """Flip the match to 'finished' — fires the trigger."""
    conn.execute(
        text(
            """
            UPDATE matches SET status = 'finished',
                               home_score = :hs,
                               away_score = :as
            WHERE id = :mid
            """
        ),
        {"hs": home_score, "as": away_score, "mid": match_id},
    )


def _load_grader():
    """scripts/grade_completed_matches.py as a module (it is not a package).
    It puts scripts/ on sys.path for its own grading_outcomes import; that
    entry is taken off again once it has loaded."""
    scripts = Path(__file__).resolve().parents[4] / "scripts"
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("grade_completed_matches", scripts / "grade_completed_matches.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


def _read_prediction(conn, pred_id: str) -> dict:
    row = (
        conn.execute(
            text(
                """
            SELECT predicted_outcome, actual_outcome, is_correct
            FROM predictions WHERE id = :id
            """
            ),
            {"id": pred_id},
        )
        .mappings()
        .one()
    )
    return dict(row)


# ── Soccer (regression — these worked before migration 011) ──────────


class TestSoccerGrading:
    def test_match_result_home_win(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="soccer")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_soccer_match_result",
                prediction_type="match_result",
                predicted_outcome="home",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=2, away_score=1)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] == "home"
        assert row["is_correct"] is True

    def test_match_result_draw(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="soccer")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_soccer_match_result",
                prediction_type="match_result",
                predicted_outcome="home",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=1, away_score=1)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] == "draw"
        assert row["is_correct"] is False

    def test_over_under_above_line(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="soccer")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_soccer_match_result",
                prediction_type="over_under",
                predicted_outcome="over_2.5",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=2, away_score=1)
            row = _read_prediction(conn, pred)
        # 3 > 2.5 → over → predicted_outcome 'over_2.5' is correct.
        assert row["is_correct"] is True

    def test_btts_yes_correct(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="soccer")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_soccer_match_result",
                prediction_type="btts",
                predicted_outcome="yes",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=2, away_score=1)
            row = _read_prediction(conn, pred)
        assert row["is_correct"] is True


# ── NHL (new in migration 011) ───────────────────────────────────────


class TestNhlGrading:
    def test_moneyline_home_win(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nhl")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nhl_ml",
                prediction_type="moneyline",
                predicted_outcome="home",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=4, away_score=2)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] == "home"
        assert row["is_correct"] is True

    def test_regulation_uses_metadata(self, trigger_tables, engine):
        # NHL regulation: 60-minute result. Final 3-3 went to OT and
        # away won — but regulation_winner='tie' in metadata.
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nhl", metadata={"regulation_winner": "tie"})
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nhl_reg",
                prediction_type="match_result",
                predicted_outcome="tie",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=3, away_score=4)
            row = _read_prediction(conn, pred)
        # Trigger reads regulation_winner='tie' from metadata, not the
        # final score (which would give 'away').
        assert row["actual_outcome"] == "tie"
        assert row["is_correct"] is True

    def test_puck_line_cover(self, trigger_tables, engine):
        # Home -1.5: home wins by 2 → cover.
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nhl")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nhl_pl",
                prediction_type="spread",
                predicted_outcome="cover",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=5, away_score=3)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] == "cover"
        assert row["is_correct"] is True

    def test_puck_line_no_cover_on_one_goal_win(self, trigger_tables, engine):
        # Home wins by 1 → does NOT cover -1.5.
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nhl")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nhl_pl",
                prediction_type="spread",
                predicted_outcome="cover",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=3, away_score=2)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] == "no_cover"
        assert row["is_correct"] is False

    def test_total_over_55(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nhl")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nhl_tot",
                prediction_type="total",
                predicted_outcome="over",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=4, away_score=3)
            row = _read_prediction(conn, pred)
        # 7 > 5.5 → over.
        assert row["actual_outcome"] == "over"
        assert row["is_correct"] is True


# ── NBA (new in migration 011 — variable line via features_cache) ────


class TestNbaGrading:
    def test_moneyline_away_win(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nba")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nba_ml",
                prediction_type="moneyline",
                predicted_outcome="home",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=98, away_score=110)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] == "away"
        assert row["is_correct"] is False

    def test_spread_home_covers_at_variable_line(self, trigger_tables, engine):
        # Home -5.5. Final 110-100 → margin 10 > 5.5 → home covers.
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nba")
            _seed_features(
                conn,
                match_id=ids["match_id"],
                feature_set="nba_baseline",
                features={"closing_spread_home": -5.5},
            )
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nba_sp",
                prediction_type="spread",
                predicted_outcome="home",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=110, away_score=100)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] == "home"
        assert row["is_correct"] is True

    def test_spread_push_leaves_is_correct_null(self, trigger_tables, engine):
        # Integer line -3, exact 3-point home win → push.
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nba")
            _seed_features(
                conn,
                match_id=ids["match_id"],
                feature_set="nba_baseline",
                features={"closing_spread_home": -3.0},
            )
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nba_sp",
                prediction_type="spread",
                predicted_outcome="home",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=103, away_score=100)
            row = _read_prediction(conn, pred)
        # actual_outcome='push' BUT is_correct=NULL — the Python
        # settler converts pushes to recommendations.status='void';
        # is_correct stays NULL so accuracy stats don't count the row.
        assert row["actual_outcome"] == "push"
        assert row["is_correct"] is None

    def test_total_over_at_variable_line(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nba")
            _seed_features(
                conn,
                match_id=ids["match_id"],
                feature_set="nba_baseline",
                features={"closing_total_line": 218.5},
            )
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nba_tot",
                prediction_type="total",
                predicted_outcome="over",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=120, away_score=110)
            row = _read_prediction(conn, pred)
        # 230 > 218.5 → over.
        assert row["actual_outcome"] == "over"
        assert row["is_correct"] is True

    def test_spread_missing_features_leaves_ungraded(self, trigger_tables, engine):
        # Without a features_cache row the trigger has no closing line
        # → leaves actual_outcome NULL. The Python catchup grader will
        # try again once features land.
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nba")
            # No _seed_features call.
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nba_sp",
                prediction_type="spread",
                predicted_outcome="home",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=110, away_score=100)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] is None
        assert row["is_correct"] is None


# ── NFL (new in migration 011 — same line-as-feature dispatch) ───────


class TestTennisGrading:
    """Tennis moneyline shares the prediction_type='moneyline' branch
    of the trigger. No tennis-specific dispatch was added because the
    grade_moneyline_2way SQL helper handles 1v1 final-score winners
    identically to NHL/NBA/NFL. This test locks that integration."""

    def test_player1_wins(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="tennis")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_tennis_ml",
                prediction_type="moneyline",
                predicted_outcome="home",
            )
            # Best-of-3 final, player1 won 2-1.
            _finish_match(conn, match_id=ids["match_id"], home_score=2, away_score=1)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] == "home"
        assert row["is_correct"] is True

    def test_player2_wins(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="tennis")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_tennis_ml",
                prediction_type="moneyline",
                predicted_outcome="home",
            )
            # Best-of-5 final, player2 won 2-3.
            _finish_match(conn, match_id=ids["match_id"], home_score=2, away_score=3)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] == "away"
        assert row["is_correct"] is False


class TestMmaGrading:
    """MMA moneyline shares the prediction_type='moneyline' trigger
    branch with NHL/NBA/NFL/tennis. No schema changes were needed —
    grade_moneyline_2way handles 1v1 final-score winners identically
    across all sports. This test locks that integration."""

    def test_fighter1_wins(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="mma")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_mma_ml",
                prediction_type="moneyline",
                predicted_outcome="home",
            )
            # home=1, away=0 → fighter1 wins.
            _finish_match(conn, match_id=ids["match_id"], home_score=1, away_score=0)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] == "home"
        assert row["is_correct"] is True

    def test_fighter2_wins(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="mma")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_mma_ml",
                prediction_type="moneyline",
                predicted_outcome="home",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=0, away_score=1)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] == "away"
        assert row["is_correct"] is False


class TestNflGrading:
    def test_moneyline_tie_leaves_ungraded(self, trigger_tables, engine):
        # NFL ties happen rarely (~1-2/year). 2-class model can't
        # represent them → trigger returns NULL → row stays ungraded.
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nfl")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nfl_ml",
                prediction_type="moneyline",
                predicted_outcome="home",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=16, away_score=16)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] is None
        assert row["is_correct"] is None

    def test_spread_at_key_number(self, trigger_tables, engine):
        # NFL spreads cluster on key numbers (3, 7, 10, 14). Home -7,
        # final 31-21 → margin 10 > 7 → home covers.
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nfl")
            _seed_features(
                conn,
                match_id=ids["match_id"],
                feature_set="nfl_baseline",
                features={"closing_spread_home": -7.0},
            )
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nfl_sp",
                prediction_type="spread",
                predicted_outcome="home",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=31, away_score=21)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] == "home"
        assert row["is_correct"] is True

    def test_total_under_at_variable_line(self, trigger_tables, engine):
        # NFL totals typically 40-50. Final 17+14=31 < 44.5 → under.
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nfl")
            _seed_features(
                conn,
                match_id=ids["match_id"],
                feature_set="nfl_baseline",
                features={"closing_total_line": 44.5},
            )
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nfl_tot",
                prediction_type="total",
                predicted_outcome="over",
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=17, away_score=14)
            row = _read_prediction(conn, pred)
        assert row["actual_outcome"] == "under"
        assert row["is_correct"] is False


# ── Idempotency + already-graded rows ────────────────────────────────


class TestIdempotency:
    def test_already_graded_rows_not_overwritten(self, trigger_tables, engine):
        # A row with is_correct already set (e.g. graded by an earlier
        # Python catch-up run) should NOT be re-touched by the trigger.
        # Otherwise concurrent grading paths could clobber each other.
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nhl")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nhl_ml",
                prediction_type="moneyline",
                predicted_outcome="home",
            )
            # Pre-grade the row to a wrong-but-locked value to prove
            # the trigger doesn't touch it.
            conn.execute(
                text(
                    """
                    UPDATE predictions
                       SET actual_outcome = 'sentinel',
                           is_correct = false
                     WHERE id = :id
                    """
                ),
                {"id": pred},
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=4, away_score=2)
            row = _read_prediction(conn, pred)
        # Sentinel survived — trigger respected the existing grade.
        assert row["actual_outcome"] == "sentinel"
        assert row["is_correct"] is False

    def test_status_unchanged_does_not_fire(self, trigger_tables, engine):
        # Updating an already-finished match (e.g. score correction
        # later in the day) must NOT re-run the trigger. The WHEN
        # clause gates on the status transition.
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="nhl")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_nhl_ml",
                prediction_type="moneyline",
                predicted_outcome="home",
            )
            # First flip to finished — trigger fires, grades the row.
            _finish_match(conn, match_id=ids["match_id"], home_score=4, away_score=2)
            assert _read_prediction(conn, pred)["actual_outcome"] == "home"
            # Now correct the score via UPDATE while status stays finished.
            # The trigger should NOT re-fire. To prove it, we manually
            # clear the grade first then run the UPDATE.
            conn.execute(
                text("UPDATE predictions SET actual_outcome=NULL, is_correct=NULL WHERE id=:id"),
                {"id": pred},
            )
            conn.execute(
                text("UPDATE matches SET home_score = 5 WHERE id = :id"),
                {"id": ids["match_id"]},
            )
            row = _read_prediction(conn, pred)
        # No re-grade — actual_outcome stays NULL because trigger
        # didn't fire on the score-only UPDATE.
        assert row["actual_outcome"] is None


# ── Rows written after kickoff (migration 027) ───────────────────────


class TestPostKickoffRows:
    """Since migration 027 the trigger leaves hindsight alone, by the
    Python grader's rules (scripts/grade_completed_matches.py): a
    prediction written after kickoff stays ungraded; a rec written at or
    after kickoff stays pending, for the grader to void."""

    def test_prediction_written_after_kickoff_stays_ungraded(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="soccer")
            after = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_soccer_match_result",
                prediction_type="match_result",
                predicted_outcome="home",
                created_at=ids["kickoff"] + timedelta(minutes=1),
            )
            at = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_soccer_match_result",
                model_version="v1.1",
                prediction_type="match_result",
                predicted_outcome="home",
                created_at=ids["kickoff"],
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=2, away_score=1)
            after_row = _read_prediction(conn, after)
            at_row = _read_prediction(conn, at)
        assert after_row["actual_outcome"] is None
        assert after_row["is_correct"] is None
        # Written exactly AT kickoff is not after it (the grader's `>`): graded.
        assert at_row["is_correct"] is True

    def test_rec_written_at_or_after_kickoff_stays_pending(self, trigger_tables, engine):
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="soccer")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_soccer_match_result",
                prediction_type="match_result",
                predicted_outcome="home",
            )
            # All three on the same pre-kickoff prediction, which the trigger
            # does grade: only the rec's own created_at holds two of them back.
            recs = {
                when: _seed_rec(
                    conn, match_id=ids["match_id"], prediction_id=pred, selection="home", created_at=created_at
                )
                for when, created_at in (
                    ("before", ids["kickoff"] - timedelta(minutes=30)),
                    ("at", ids["kickoff"]),
                    ("after", ids["kickoff"] + timedelta(minutes=5)),
                )
            }
            _finish_match(conn, match_id=ids["match_id"], home_score=2, away_score=1)
            pred_row = _read_prediction(conn, pred)
            rows = {when: _read_rec(conn, rec_id) for when, rec_id in recs.items()}
        assert pred_row["is_correct"] is True
        assert rows["before"]["status"] == "won"
        assert rows["before"]["settled_at"] is not None
        for when in ("at", "after"):
            assert rows[when] == {"status": "pending", "actual_result": None, "settled_at": None}, when

    def test_rec_with_no_created_at_still_settles(self, trigger_tables, engine):
        # A NULL created_at is not "after kickoff" in the grader either.
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="soccer")
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_soccer_match_result",
                prediction_type="match_result",
                predicted_outcome="home",
            )
            rec = _seed_rec(conn, match_id=ids["match_id"], prediction_id=pred, selection="home", created_at=None)
            _finish_match(conn, match_id=ids["match_id"], home_score=0, away_score=1)
            row = _read_rec(conn, rec)
        assert row["status"] == "lost"

    def test_pending_rec_the_grading_window_misses_is_reported_until_voided(self, trigger_tables, engine, caplog):
        # 027 leaves a post-kickoff rec pending for the grader, which visits
        # only matches inside --days. On a match marked finished after it aged
        # out of that window, grade_completed_matches.py's canary must name the
        # rec, and the command it prints must void it.
        grader = _load_grader()
        url = os.environ["DATABASE_URL"]
        with engine.begin() as conn:
            ids = _seed_match(conn, sport="soccer")
            kickoff = conn.execute(
                text(
                    "UPDATE matches SET match_date = match_date - INTERVAL '30 days' "
                    "WHERE id = :id RETURNING match_date"
                ),
                {"id": ids["match_id"]},
            ).scalar_one()
            pred = _seed_prediction(
                conn,
                match_id=ids["match_id"],
                model_name="ensemble_soccer_match_result",
                prediction_type="match_result",
                predicted_outcome="home",
            )
            late = _seed_rec(
                conn,
                match_id=ids["match_id"],
                prediction_id=pred,
                selection="home",
                created_at=kickoff + timedelta(minutes=5),
            )
            _finish_match(conn, match_id=ids["match_id"], home_score=2, away_score=1)
        with engine.begin() as conn:
            assert _read_rec(conn, late)["status"] == "pending"

        with caplog.at_level("INFO", logger="grade_completed_matches"):
            grader.run(url, 14, None)  # the scheduled run: the match is outside its window
            assert grader.warn_stuck_post_kickoff_recs(url, 14) == 1
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1 and late in errors[0], errors
        command = re.search(r"Void them with: python /app/scripts/grade_completed_matches.py --days (\d+)$", errors[0])
        assert command, errors[0]
        assert int(command.group(1)) == 32  # kickoff 30 days + 4 hours ago

        grader.run(url, int(command.group(1)), None)
        with engine.begin() as conn:
            row = _read_rec(conn, late)
        assert row["status"] == "void" and row["actual_result"] == grader.POST_KICKOFF_VOID_RESULT
        assert grader.warn_stuck_post_kickoff_recs(url, 14) == 0
