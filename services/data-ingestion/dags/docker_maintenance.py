"""Weekly Docker image/build-cache prune (audit doc §6.2).

Every CI deploy pulls fresh api/airflow/frontend images (SHA-tagged), so old
layers accumulate at ~2-4 GB per active week. The audit prescribed a weekly
host cron for this, and the 2026-07-13 hardening batch believed it installed
one — the 2026-07-14 verification pass found no crontab entry on the VM (the
guardrail was silently absent while ~10 GB of reclaimable layers piled up).

A DAG beats a host cron here: it is versioned in the repo (survives a VM
rebuild), visible in the Airflow UI, and pages Telegram on failure via the
shared notify_failure callback. The airflow containers already mount the
docker socket (the pipeline DAGs shell `docker compose exec`), so the docker
CLI works directly.

`--filter until=336h` keeps the last 14 days of images — always spanning the
current + previous deploy, so an emergency IMAGE_TAG rollback (see the
API_DOWN runbook) still finds its target locally.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
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
    dag_id="docker_maintenance",
    description="Weekly prune of Docker images older than 14 days + build cache",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 5 * * 0",  # Sundays 05:00 UTC, after the weekly retrain window opens
    catchup=False,
    max_active_runs=1,
    tags=["auspex", "maintenance"],
) as dag:

    prune_images = BashOperator(
        task_id="prune_images",
        bash_command="docker image prune -af --filter until=336h",
    )

    prune_build_cache = BashOperator(
        task_id="prune_build_cache",
        bash_command="docker builder prune -af",
    )

    prune_images >> prune_build_cache
