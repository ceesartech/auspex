"""Async prediction generation and model retraining tasks"""

import logging
from datetime import datetime
from typing import List, Optional

from celery_app import celery_app
from database import get_db_context
from services.prediction_service import PredictionService

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.prediction_tasks.generate_predictions", bind=True)
def generate_predictions(self, match_ids: List[str]):
    """Generate predictions for a batch of matches"""

    logger.info(f"Generating predictions for {len(match_ids)} matches")

    results = {"success": [], "failed": []}

    with get_db_context() as db:
        service = PredictionService(db)

        for match_id in match_ids:
            try:
                prediction = service.predict_match(
                    match_id=match_id, include_explanation=False
                )
                results["success"].append(match_id)
            except Exception as e:
                logger.error(f"Failed to predict match {match_id}: {e}")
                results["failed"].append({"match_id": match_id, "error": str(e)})

    logger.info(
        f"Predictions complete: {len(results['success'])} success, "
        f"{len(results['failed'])} failed"
    )
    return results


@celery_app.task(name="tasks.prediction_tasks.generate_upcoming_predictions", bind=True)
def generate_upcoming_predictions(self, days_ahead: int = 3):
    """Generate predictions for all upcoming matches"""

    from sqlalchemy import text

    logger.info(f"Generating predictions for matches in next {days_ahead} days")

    with get_db_context() as db:
        query = text("""
            SELECT m.id FROM matches m
            LEFT JOIN predictions p ON m.id = p.match_id
            WHERE m.status = 'scheduled'
            AND m.match_date > NOW()
            AND m.match_date <= NOW() + make_interval(days => :days)
            AND p.id IS NULL
        """)

        results = db.execute(query, {"days": days_ahead}).fetchall()
        match_ids = [str(row.id) for row in results]

    if match_ids:
        logger.info(f"Found {len(match_ids)} matches without predictions")
        return generate_predictions(match_ids)
    else:
        logger.info("No matches need predictions")
        return {"success": [], "failed": []}


@celery_app.task(name="tasks.prediction_tasks.retrain_model", bind=True)
def retrain_model(
    self,
    model_name: str,
    trigger_reason: str = "manual",
):
    """Trigger model retraining"""

    from sqlalchemy import text
    import json

    logger.info(f"Retraining model: {model_name} (reason: {trigger_reason})")

    with get_db_context() as db:
        # Log retraining start
        log_query = text("""
            INSERT INTO model_retraining_logs
            (model_name, old_version, new_version, trigger_reason, status)
            VALUES (:model_name, :old_version, :new_version, :trigger_reason, 'started')
            RETURNING id
        """)

        result = db.execute(
            log_query,
            {
                "model_name": model_name,
                "old_version": "current",
                "new_version": f"v{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                "trigger_reason": trigger_reason,
            },
        ).fetchone()

        log_id = result.id
        db.commit()

    try:
        # Actual retraining would integrate with ml-models service
        logger.info(f"Model {model_name} retraining would be triggered here")

        # Update log to completed
        with get_db_context() as db:
            update_query = text("""
                UPDATE model_retraining_logs
                SET status = 'completed', completed_at = NOW()
                WHERE id = :log_id
            """)
            db.execute(update_query, {"log_id": str(log_id)})
            db.commit()

        return {"status": "completed", "model_name": model_name}

    except Exception as e:
        logger.error(f"Retraining failed: {e}")

        with get_db_context() as db:
            error_query = text("""
                UPDATE model_retraining_logs
                SET status = 'failed', error_message = :error, completed_at = NOW()
                WHERE id = :log_id
            """)
            db.execute(error_query, {"log_id": str(log_id), "error": str(e)})
            db.commit()

        return {"status": "failed", "model_name": model_name, "error": str(e)}
