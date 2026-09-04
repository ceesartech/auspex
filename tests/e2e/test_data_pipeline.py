"""E2E: Data pipeline integrity — verify DB is populated and consistent."""

import pytest


def test_matches_exist_in_db(db):
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM matches")
    count = cur.fetchone()[0]
    assert count > 0, "No matches found — run seed_dev_data.py"


def test_predictions_exist_in_db(db):
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM predictions")
    count = cur.fetchone()[0]
    assert count > 0, "No predictions found — run seed_dev_data.py"


def test_teams_exist_in_db(db):
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM teams")
    count = cur.fetchone()[0]
    assert count >= 10, f"Only {count} teams — expected at least 10"


def test_prediction_probabilities_valid(db):
    cur = db.cursor()
    cur.execute(
        """
        SELECT prediction_id,
               probability_home + probability_draw + probability_away AS total
        FROM predictions
        WHERE probability_home IS NOT NULL
        LIMIT 50
    """
    )
    for pred_id, total in cur.fetchall():
        assert abs(total - 1.0) < 0.02, f"Prediction {pred_id}: probabilities sum to {total}"


def test_finished_matches_have_outcomes(db):
    cur = db.cursor()
    cur.execute(
        """
        SELECT COUNT(*) FROM matches
        WHERE status = 'finished' AND actual_outcome IS NULL
    """
    )
    count = cur.fetchone()[0]
    assert count == 0, f"{count} finished matches are missing actual_outcome"


def test_predictions_reference_valid_matches(db):
    cur = db.cursor()
    cur.execute(
        """
        SELECT COUNT(*) FROM predictions p
        LEFT JOIN matches m ON p.match_id = m.match_id
        WHERE m.match_id IS NULL
    """
    )
    orphans = cur.fetchone()[0]
    assert orphans == 0, f"{orphans} predictions reference non-existent matches"


def test_model_performance_logs_exist(db):
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM model_performance_logs")
    count = cur.fetchone()[0]
    assert count > 0, "No model performance logs — run seed_dev_data.py"


def test_redis_reachable(redis_client):
    result = redis_client.ping()
    assert result is True


def test_redis_cache_write_read(redis_client):
    redis_client.setex("e2e:test", 10, "hello")
    val = redis_client.get("e2e:test")
    assert val == b"hello"
    redis_client.delete("e2e:test")
