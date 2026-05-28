"""Daily DAG: fetch live odds for upcoming matches from the-odds-api.com.

This DAG is intentionally separate from auspex_pipeline (which runs
every 15 min) because the-odds-api.com free tier is rate-limited to
500 requests/month. Running it once a day is enough — closing odds
don't move so much that 15-min refreshes would meaningfully improve
predictions, and the quota cost of 9 sports × 96 daily DAG runs
would blow the budget.

Schedule: once a day at 06:00 UTC, after the early-morning lines are
posted by most US books but before peak trading.

Requires THE_ODDS_API_KEY in .env. The task no-ops with a clear
error message if it's not set, so a missing key never crashes the
upstream auspex_pipeline DAG.
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
    "retry_delay": timedelta(minutes=15),
}

DOCKER_EXEC = "docker compose -f /opt/auspex/docker-compose.yml exec -T api"

with DAG(
    dag_id="fetch_live_odds",
    description="Pull pre-match odds for upcoming fixtures (the-odds-api.com, daily)",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="0 6 * * *",  # 06:00 UTC daily
    catchup=False,
    max_active_runs=1,
    tags=["odds", "live-data"],
) as dag:

    fetch_odds = BashOperator(
        task_id="fetch_odds",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/fetch_live_odds.py",
    )
