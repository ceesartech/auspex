"""Daily PostgreSQL backup DAG.

Runs `scripts/backup_postgres.py` at 02:00 UTC every day. The script:
  * Dumps the prod DB to a timestamped .dump file in /app/backups/
    (bind-mounted to /opt/auspex/backups/ on the host).
  * Optionally uploads to S3-compatible storage when
    BACKUP_S3_BUCKET env var is set.
  * Rotates out local backups older than 7 days.

If the dump itself fails the task fails; if the S3 upload fails the
task succeeds (local backup is the must-have) and a Telegram alert
fires from the Airflow-failure-callback infra (added separately).

02:00 UTC is chosen to land BEFORE:
  * 03:00 weather_daily_vc_fetch (so a heavy backfill doesn't run
    during pg_dump, which holds a long read transaction)
  * 04:00 retrain_models on Sundays
… and AFTER the high-volume midnight UTC odds fetches.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from alerting import notify_failure  # shared Telegram failure alerting
from airflow.operators.bash import BashOperator

DOCKER_EXEC = "docker compose -f /opt/auspex/docker-compose.yml exec -T api"

default_args = {
    "owner": "auspex",
    "on_failure_callback": notify_failure,
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="db_backup_daily",
    description="Daily pg_dump → local /opt/auspex/backups + optional S3 (02:00 UTC)",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["db", "backup", "daily"],
) as dag:

    # The script reads BACKUP_S3_BUCKET / BACKUP_S3_ENDPOINT_URL /
    # AWS_* from the api container's env (set via .env on prod),
    # so no flags need to be passed here.
    backup_postgres = BashOperator(
        task_id="backup_postgres",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/backup_postgres.py",
        execution_timeout=timedelta(minutes=30),
    )
