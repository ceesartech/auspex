"""Every-6h DAG: fetch live odds for upcoming matches from the-odds-api.com.

Intentionally separate from auspex_pipeline (15-min cadence) for quota
reasons. Math at 28 sports:

  every 15 min  → 28 × 96 × 30 = 80,640 quota/month (way over even paid)
  hourly        → 28 × 24 × 30 = 20,160 quota/month (right at 20k tier)
  every 6 hours → 28 ×  4 × 30 =  3,360 quota/month (4× headroom on 20k tier,
                                                     ~6.7× over free 500)
  daily         → 28 ×  1 × 30 =     840 quota/month (40% over free 500)

Closing odds shift slowly enough that 6-hour cadence is plenty for
personal-use prediction freshness, and it's the right balance between
keeping predictions current vs preserving quota for retries / manual
runs.

Requires THE_ODDS_API_KEY in .env. fetch_live_odds.py tolerates
unknown sport keys (404) and off-season sports (422) without
crashing, so a stale DEFAULT_SPORTS list never blocks the DAG.
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
    description="Pull pre-match odds for upcoming fixtures (the-odds-api.com, every 6h)",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="0 */6 * * *",  # 00:00, 06:00, 12:00, 18:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["odds", "live-data"],
) as dag:

    fetch_odds = BashOperator(
        task_id="fetch_odds",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/fetch_live_odds.py",
    )
