"""Every-90-min DAG: fetch live odds for upcoming matches from the-odds-api.com.

Intentionally separate from auspex_pipeline (15-min cadence) for quota
reasons. Math at 28 sports (the-odds-api charges 1 quota per sport call
that returns events):

  every 15 min  → 28 × 96 × 30 = 80,640 quota/month (80% of 100k tier)
  every 30 min  → 28 × 48 × 30 = 40,320 quota/month (~40% of 100k tier)
  every 90 min  → 28 × 16 × 30 = 13,440 quota/month (67% of 20k tier,
                                                     ~13% of 100k tier)
  every 6 hours → 28 ×  4 × 30 =  3,360 quota/month (4× headroom on 20k)
  daily         → 28 ×  1 × 30 =     840 quota/month (40% over free 500)

90-min cadence fits comfortably in the 20k tier and uses only ~13% of
the 100k tier — leaving plenty of room for retries, manual runs,
off-season sports that briefly come online, and new leagues added
later. Closing odds shift slowly enough that 90 min is plenty of
freshness for personal-use prediction.

Airflow can't express 90 min in cron (90 doesn't divide 60 or 24
evenly), so we use timedelta. First run anchors to start_date; every
subsequent run is exactly 90 min after the previous one.

Requires THE_ODDS_API_KEY in .env. fetch_live_odds.py tolerates
unknown sport keys (404) and off-season sports (422) without
crashing, so a stale DEFAULT_SPORTS list never blocks the DAG.
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
    "retry_delay": timedelta(minutes=15),
}

DOCKER_EXEC = "docker compose -f /opt/auspex/docker-compose.yml exec -T api"

with DAG(
    dag_id="fetch_live_odds",
    description="Pull pre-match odds for upcoming fixtures (the-odds-api.com, every 90 min)",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=timedelta(minutes=90),
    catchup=False,
    max_active_runs=1,
    tags=["odds", "live-data"],
) as dag:

    fetch_odds = BashOperator(
        task_id="fetch_odds",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/fetch_live_odds.py",
    )
