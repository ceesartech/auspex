"""Model-monitoring DAG — runs hourly, independent of the prediction pipeline.

Extracted from `auspex_pipeline` (audit doc §6.2 cadence split). It was running
every 15 min as the pipeline's terminal task, recomputing a 30-day rolling
ECE/MCE/Brier/log-loss window per (sport, market) on every tick — drift does not
move meaningfully in 15 minutes, so that was ~96 redundant recomputes/day. At
hourly cadence it is 24/day for the same signal.

It is genuinely standalone: `monitor_models.py` only READS graded predictions
from the DB and pushes drift alerts to Telegram. It has no data dependency on
the upcoming-prediction tasks (those write UPCOMING rows; this reads FINISHED,
graded ones), which is why it carried a trigger_rule=ALL_DONE in the old DAG.
The only thing lost by splitting is the cosmetic "drift alert lands after the
value-bet digest" ordering — acceptable for 4x fewer runs.
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
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="monitor_models",
    description="Rolling drift metrics per sport+market, Telegram alerts (hourly)",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="0 * * * *",  # top of every hour
    catchup=False,
    max_active_runs=1,
    tags=["auspex", "monitoring"],
) as dag:

    DOCKER_EXEC = "docker compose -f /opt/auspex/docker-compose.yml exec -T api"

    monitor_models = BashOperator(
        task_id="monitor_models",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/monitor_models.py --days 30",
    )
