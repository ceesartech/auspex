"""Async notification dispatch tasks"""

import logging
from typing import Any, Dict, Optional

from celery_app import celery_app
from database import get_db_context
from services.notification_service import NotificationService

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.notification_tasks.check_and_alert_high_value")
def check_and_alert_high_value(min_ev: float = 0.10):
    """Check for high-value bet opportunities and create notifications"""

    logger.info(f"Checking for high-value bets (min EV: {min_ev})")

    with get_db_context() as db:
        service = NotificationService(db)
        alerts = service.check_high_value_bets(min_ev=min_ev)

    logger.info(f"Found {len(alerts)} high-value bet alerts")
    return {"alerts_created": len(alerts)}


@celery_app.task(name="tasks.notification_tasks.check_model_drift_task")
def check_model_drift_task(threshold: float = 0.05):
    """Check for model performance drift"""

    logger.info(f"Checking model drift (threshold: {threshold})")

    with get_db_context() as db:
        service = NotificationService(db)
        drift_alerts = service.check_model_drift(threshold=threshold)

    logger.info(f"Found {len(drift_alerts)} model drift alerts")
    return {"drift_alerts": len(drift_alerts)}


@celery_app.task(name="tasks.notification_tasks.send_notification")
def send_notification(
    notification_type: str,
    title: str,
    message: str,
    source: str = "system",
    metadata: Optional[Dict[str, Any]] = None,
):
    """Create a system notification"""

    with get_db_context() as db:
        service = NotificationService(db)
        notification_id = service.create_notification(
            notification_type=notification_type,
            title=title,
            message=message,
            source=source,
            metadata=metadata,
        )

    logger.info(f"Created notification {notification_id}: {title}")
    return {"notification_id": notification_id}
