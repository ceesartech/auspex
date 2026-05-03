"""Celery application configuration"""

from celery import Celery

from config import settings

celery_app = Celery(
    "betting_system",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes
    task_soft_time_limit=540,  # 9 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    task_routes={
        "tasks.prediction_tasks.*": {"queue": "predictions"},
        "tasks.notification_tasks.*": {"queue": "notifications"},
    },
    beat_schedule={
        "check-high-value-bets": {
            "task": "tasks.notification_tasks.check_and_alert_high_value",
            "schedule": 300.0,  # Every 5 minutes
        },
        "check-model-drift": {
            "task": "tasks.notification_tasks.check_model_drift_task",
            "schedule": 3600.0,  # Every hour
        },
    },
)

celery_app.autodiscover_tasks(["tasks"])
