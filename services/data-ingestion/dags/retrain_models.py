"""Airflow DAG for weekly on-VM model retraining.

Runs every Sunday at 04:00 UTC. Like auspex_pipeline.py it uses the
docker-out-of-docker pattern — all heavy work runs inside the api
container via `docker compose exec api …`. The airflow scheduler just
orchestrates: docker CLI + the bind-mounted host socket let it signal
the host daemon, which already has the api container with all our
Python dependencies.

Flow:
  1. validate_data: ensures the matches table has enough rows + valid
     feature columns. Aborts the run with a clear error if not.
  2. train_models: writes new artifacts to /app/models/staging
     inside the api container (which is the host's ./models/ dir).
  3. swap_production: cp staging → production atomically. Backs up
     the previous production dir as production-prev-<timestamp>.
  4. reload_api: restart the api container so the new model files
     are picked up by load_models_into_process() at startup.
  5. cleanup_old_backups: keep only the most recent 3 backups.

Manual trigger from the Airflow UI: retrain_models DAG → Play button.
Or programmatically:
    docker compose exec airflow-scheduler airflow dags trigger retrain_models
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "auspex",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

# Common compose prefix. The compose-file path matches what's mounted
# read-only into the airflow containers at /opt/auspex.
DOCKER_EXEC = "docker compose -f /opt/auspex/docker-compose.yml exec -T api"

with DAG(
    dag_id="retrain_models",
    description="Weekly retrain of ensemble + base models (calibrated, on-VM)",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="0 4 * * 0",  # Sundays at 04:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["ml", "retraining"],
) as dag:

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=(
            f"{DOCKER_EXEC} bash -c '"
            "cd /app/services/ml-models && "
            "PYTHONPATH=src python -m validation.validate_training_data "
            '--database-url "$DATABASE_URL"'
            "'"
        ),
    )

    train_models = BashOperator(
        task_id="train_models",
        bash_command=(
            f"{DOCKER_EXEC} bash -c '"
            "cd /app/services/ml-models && "
            "PYTHONPATH=src python -m training.train_all_models "
            "--model-type all "
            '--database-url "$DATABASE_URL" '
            "--output-dir /app/models/staging "
            "--export-onnx"
            "'"
        ),
    )

    swap_production = BashOperator(
        task_id="swap_production",
        bash_command=(
            f"{DOCKER_EXEC} bash -c '"
            "set -euo pipefail; "
            "STAMP=$(date -u +%%Y%%m%%d-%%H%%M%%S); "
            "if [ -d /app/models/production ]; then "
            "mv /app/models/production /app/models/production-prev-${STAMP}; "
            "fi; "
            "mv /app/models/staging /app/models/production; "
            "ls /app/models/production"
            "'"
        ),
    )

    # Restart the api container so load_models_into_process() picks up
    # the new artifacts. Runs against the host daemon via the socket.
    reload_api = BashOperator(
        task_id="reload_api",
        bash_command="docker compose -f /opt/auspex/docker-compose.yml restart api",
    )

    cleanup_old_backups = BashOperator(
        task_id="cleanup_old_backups",
        bash_command=(
            f"{DOCKER_EXEC} bash -c '"
            "cd /app/models && "
            "ls -1dt production-prev-* 2>/dev/null | tail -n +4 | xargs -r rm -rf"
            "'"
        ),
    )

    validate_data >> train_models >> swap_production >> reload_api >> cleanup_old_backups
