"""Airflow DAG for on-VM model retraining.

For the personal-VM deployment path, retraining runs locally — no GCS,
no API reload hooks. The flow is:

  1. validate_data: ensures the matches table has enough samples + features.
  2. train_models: runs train_all_models with --export-onnx, writing artifacts
     to /models inside the VM.
  3. swap_production: atomically moves the new artifacts into
     /models/production so the next API request picks them up. The API's
     model registry is keyed by mtime, so cache invalidates automatically.

To trigger manually: `airflow dags trigger retrain_models`.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

MODEL_ROOT = Path("/models")
STAGING_DIR = MODEL_ROOT / "staging"
PRODUCTION_DIR = MODEL_ROOT / "production"

default_args = {
    "owner": "auspex",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

dag = DAG(
    "retrain_models",
    default_args=default_args,
    description="Retrain ensemble + base models weekly using on-VM Postgres",
    schedule_interval="0 4 * * 0",  # Sundays at 04:00 server time
    catchup=False,
    max_active_runs=1,
    tags=["ml", "retraining"],
)


def _validate_training_data(**_ctx) -> None:
    """Run the validation module and raise if data is insufficient."""
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app/services/ml-models/src"
    result = subprocess.run(
        ["python", "-m", "validation.validate_training_data"],
        cwd="/app/services/ml-models",
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Validation stderr: %s", result.stderr)
        raise RuntimeError(f"Training data validation failed: {result.stderr.strip()[:500]}")
    logger.info("Validation OK:\n%s", result.stdout[-1000:])


def _swap_production(**_ctx) -> None:
    """Atomically promote staging artifacts to the production directory."""
    if not STAGING_DIR.exists():
        raise FileNotFoundError(f"Staging dir missing: {STAGING_DIR}")

    backup_dir = MODEL_ROOT / f"production-prev-{datetime.utcnow():%Y%m%d-%H%M%S}"
    if PRODUCTION_DIR.exists():
        PRODUCTION_DIR.rename(backup_dir)
        logger.info("Backed up previous production -> %s", backup_dir)

    STAGING_DIR.rename(PRODUCTION_DIR)
    logger.info("Promoted staging -> production at %s", PRODUCTION_DIR)

    # Retain only the most recent 3 backups to avoid filling disk.
    backups = sorted(MODEL_ROOT.glob("production-prev-*"))
    for old in backups[:-3]:
        shutil.rmtree(old, ignore_errors=True)
        logger.info("Removed old backup %s", old)


validate_task = PythonOperator(
    task_id="validate_data",
    python_callable=_validate_training_data,
    dag=dag,
)

train_task = BashOperator(
    task_id="train_models",
    bash_command=(
        "set -euo pipefail; "
        "cd /app/services/ml-models && "
        "PYTHONPATH=src python -m training.train_all_models "
        "  --model-type all "
        "  --output-dir /models/staging "
        "  --export-onnx"
    ),
    dag=dag,
)

swap_task = PythonOperator(
    task_id="swap_production",
    python_callable=_swap_production,
    dag=dag,
)

validate_task >> train_task >> swap_task
