"""Auspex end-to-end pipeline DAG.

Runs every 15 minutes:
  1. fetch upcoming fixtures from ESPN (free API)
  2. compute features for any new scheduled matches
  3. precompute predictions + fire Telegram alerts on high-confidence

Retraining is a separate weekly DAG (retrain_models.py). Historical
ingestion (football-data, StatsBomb) is one-shot via bootstrap.sh — we
don't re-run it on a schedule.

This DAG shells out to the /app/scripts/*.py we've already smoke-tested,
which means the same logic runs whether you invoke it manually
(`docker compose exec api python /app/scripts/...`) or via Airflow.
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
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="auspex_pipeline",
    description="Fetch fixtures, compute features, predict, notify (every 15 min)",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="*/15 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["auspex", "pipeline"],
) as dag:

    # Airflow workers don't have the api code, but the api container does.
    # We exec into it via docker on the host. For this to work, the
    # docker socket must be mounted into the airflow image (which it
    # already is per docker-compose.yml's airflow-scheduler service).
    DOCKER_EXEC = "docker compose -f /opt/auspex/docker-compose.yml exec -T api"

    # ── Soccer branch (the original pipeline) ──────────────────────
    fetch_upcoming = BashOperator(
        task_id="fetch_upcoming",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/fetch_upcoming.py --sport soccer --days 14",
    )

    compute_features = BashOperator(
        task_id="compute_features",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/compute_features.py --days 14",
    )

    precompute_predictions = BashOperator(
        task_id="precompute_predictions",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/precompute_predictions.py --days 14",
    )

    fetch_upcoming >> compute_features >> precompute_predictions

    # ── NHL branch (Phase 2: ingestion + features only) ───────────
    # Prediction task lands when the NHL ensemble ships in Phase 3.
    # Until then this branch just keeps matches + features fresh so
    # the model has a target to train against the moment it's ready.
    fetch_upcoming_nhl = BashOperator(
        task_id="fetch_upcoming_nhl",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/fetch_upcoming.py --sport nhl --days 14",
    )

    compute_features_nhl = BashOperator(
        task_id="compute_features_nhl",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/compute_features_nhl.py --days 14",
    )

    fetch_upcoming_nhl >> compute_features_nhl
