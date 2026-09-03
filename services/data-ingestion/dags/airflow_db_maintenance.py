"""Airflow metadata housekeeping — monthly (audit doc §6.2).

The Airflow metadata DB grows unbounded: every task instance, DAG run, log
reference, XCom and job row is retained forever. On this deployment the core
pipeline alone emits ~1,500 task-runs/day, so the metadata was ~365 MB and
compounding. `airflow db clean` deletes rows older than a cutoff across all the
history tables (task_instance, dag_run, log, job, xcom, ...).

Runs on the 1st of each month, deleting everything older than 90 days — well
beyond any window the UI or the drift monitor looks back over. `--skip-archive`
drops the rows outright rather than copying them to `_archive` tables first
(archiving just moves the bloat sideways on a single-user box).

`--yes` is required for non-interactive runs. The cutoff is computed inside the
airflow container at runtime with GNU date.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG
from alerting import notify_failure  # shared Telegram failure alerting

default_args = {
    "owner": "auspex",
    "on_failure_callback": notify_failure,
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="airflow_db_maintenance",
    description="Monthly `airflow db clean` of metadata older than 90 days",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="0 4 1 * *",  # 04:00 on the 1st of each month
    catchup=False,
    max_active_runs=1,
    tags=["auspex", "maintenance"],
) as dag:

    # `airflow db clean` needs the metadata-DB connection, and since the 2.8 -> 3.3 migration
    # task subprocesses no longer have it: Airflow 3 task isolation (AIP-72) hands tasks a
    # scrubbed env where sql_alchemy_conn is unparsable, so running the CLI directly here dies
    # in settings init ("Could not parse SQLAlchemy URL" — the 2026-09-01 failure; the 07-01 run
    # passed under 2.8). Fix: docker-exec back into the scheduler container (the pipeline DAGs'
    # established socket pattern), which lands in the FULL container env where the conn is set.
    # The 90-day cutoff is computed in the task subshell (same image, GNU date) and passed in.
    db_clean = BashOperator(
        task_id="db_clean",
        bash_command=(
            "CUTOFF=\"$(date -u -d '90 days ago' '+%Y-%m-%d %H:%M:%S%z')\" && "
            "docker compose -f /opt/auspex/docker-compose.yml exec -T airflow-scheduler "
            "airflow db clean --skip-archive --yes --clean-before-timestamp \"$CUTOFF\""
        ),
    )
