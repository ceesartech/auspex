"""Daily PostgreSQL backup DAG.

Runs `scripts/backup_postgres.py` at 02:00 UTC every day; every ops
wrapper also triggers it for its verified pre-write backup. The script:
  * Dumps the prod DB to a timestamped .dump file in /app/backups/
    (bind-mounted to /opt/auspex/backups/ on the host).
  * Uploads it to B2 (S3-compatible) when BACKUP_S3_BUCKET is set, and
    HEAD-verifies the upload.
  * Rotates local dumps: never the one just written; with B2, at most
    BACKUP_LOCAL_KEEP (3) and none past 7 days, each older one deleted
    only after its own B2 copy is verified; without B2, the 7-day age
    rule alone (scripts/backup_postgres.py, LOCAL ROTATION).

The task fails — and notify_failure pages Telegram — when the dump fails
(exit 1), the upload fails (exit 2, local dump kept) or rotation had to
keep a dump whose B2 copy it could not verify (exit 3). The retry takes
a fresh dump.

02:00 UTC is chosen to land BEFORE:
  * 03:00 weather_daily_vc_fetch (so a heavy backfill doesn't run
    during pg_dump, which holds a long read transaction)
  * 04:00 retrain_models on Sundays
… and AFTER the high-volume midnight UTC odds fetches.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG
from alerting import notify_failure  # shared Telegram failure alerting

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
    schedule="0 2 * * *",
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
