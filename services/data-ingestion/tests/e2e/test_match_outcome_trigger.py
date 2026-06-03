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

These are integration tests — they need a real Postgres with all
migrations applied. They're auto-marked `integration` by
conftest.pytest_collection_modifyitems and run via:

    pytest -m integration services/data-ingestion/tests/e2e/test_match_outcome_trigger.py
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

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
    return {"match_id": match_id, "league_id": league_id, "home_id": home_id, "away_id": away_id}


def _seed_prediction(
    conn,
    *,
    match_id: str,
    model_name: str,
    prediction_type: str,
    predicted_outcome: str,
    model_version: str = "v1.0",
    probabilities: dict | None = None,
) -> str:
    """Insert one pending prediction row. Returns the prediction ID."""
    pred_id = str(uuid.uuid4())
    conn.execute(
        text(
            """
            INSERT INTO predictions
                (id, match_id, model_name, model_version, prediction_type,
                 predicted_outcome, confidence, probabilities)
            VALUES (:id, :mid, :mn, :mv, :pt, :po, :conf, CAST(:pr AS jsonb))
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
        },
    )
    return pred_id


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
