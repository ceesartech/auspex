"""Weekly Docker build-cache prune + B2 Modal-artifact prune (audit doc §6.2).

History: this DAG replaced a host cron the 2026-07-13 hardening batch believed
it had installed (the 2026-07-14 verification found no crontab entry while
~10 GB of reclaimable layers piled up). A DAG is versioned in the repo
(survives a VM rebuild), visible in the Airflow UI, and pages Telegram on
failure via the shared notify_failure callback. The airflow containers mount
the docker socket, so the docker CLI works directly.

The image prune and the Airflow task-log prune moved to the DAILY
disk_maintenance DAG on 2026-11-02, after the root disk reached 95%: the old
`docker image prune -af --filter until=168h` kept a week of untagged :latest
leftovers and could delete the rollback target after a quiet week (see
scripts/prune_docker_images.py), and the 30-day log prune let task logs reach
7.4 GB. What stays here is weekly by nature:

* prune_build_cache — `docker builder prune -af`. CI builds the images
  remotely, so the cache only grows after a FORCE_LOCAL_BUILD=1 deploy.
* prune_b2_modal_artifacts — B2 modal-train/<run_id>/… objects older than 14
  days (B2 storage, not the VM disk).
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
    dag_id="docker_maintenance",
    description="Weekly docker build-cache prune + B2 modal-train artifact prune (images/logs: disk_maintenance)",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="0 5 * * 0",  # Sundays 05:00 UTC, after the weekly retrain window opens
    catchup=False,
    max_active_runs=1,
    tags=["auspex", "maintenance"],
) as dag:

    prune_build_cache = BashOperator(
        task_id="prune_build_cache",
        bash_command="docker builder prune -af",
    )

    # Prune Modal training artifacts from B2 (modal-train/<run_id>/…) older than
    # 14 days — the weekly retrain + any test runs leave per-run trees there.
    # Runs in the api container, which carries the B2 creds. Safe: only touches
    # the modal-train/ prefix (intermediate artifacts), never backups or models.
    prune_b2_modal_artifacts = BashOperator(
        task_id="prune_b2_modal_artifacts",
        bash_command=(
            "docker compose -f /opt/auspex/docker-compose.yml exec -T api "
            "python /app/scripts/prune_modal_artifacts.py --days 14"
        ),
    )

    prune_build_cache >> prune_b2_modal_artifacts
