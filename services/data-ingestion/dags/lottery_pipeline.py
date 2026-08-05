"""Lottery pipeline — daily draw ingestion + honest backtest bookkeeping.

Draws happen at most once per day per game (Powerball Mon/Wed/Sat, Mega
Millions Tue/Fri, both ~23:00 ET = ~03:00-04:00 UTC) and the NY Open Data
feed updates the following morning, so a single daily run at 13:00 UTC always
sees yesterday's draw. No 15-minute cadence needed — this is deliberately NOT
part of auspex_pipeline.

Tasks:
  1. fetch_lottery_draws — incremental ingest from NY Open Data (free).
  2. settle_lottery_lines — lottery_backtest.py default mode: settle stored
     lines against actual draws and print the observed-vs-theoretical
     any-prize report (strategy differences are expected to be chance; the
     report exists to prove that honestly).
  3. generate_lottery_lines — lottery_backtest.py --generate: one line per
     strategy per game for the next draw, so the backtest ledger keeps
     accumulating evidence.

Requires migration 010 (lottery_predictions) — tasks 2 and 3 fail loudly
until it is applied, which is the correct behavior.
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
    dag_id="lottery_pipeline",
    description="Daily lottery draw ingestion + backtest settle/generate",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="0 13 * * *",  # daily, after the NY feed's morning update
    catchup=False,
    max_active_runs=1,
    tags=["auspex", "lottery"],
) as dag:

    DOCKER_EXEC = "docker compose -f /opt/auspex/docker-compose.yml exec -T api"

    fetch_lottery_draws = BashOperator(
        task_id="fetch_lottery_draws",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/fetch_lottery_draws.py",
    )

    # Per-tier winner counts + jackpot amounts for new MM draws (v1.1 —
    # feeds the sales-inference + popularity-fit harness). Small --limit:
    # the daily delta is 0-1 draws; the one-shot backfill ran separately.
    fetch_lottery_winners = BashOperator(
        task_id="fetch_lottery_winners",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/fetch_lottery_winners.py --limit 20",
    )

    settle_lottery_lines = BashOperator(
        task_id="settle_lottery_lines",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/lottery_backtest.py",
    )

    generate_lottery_lines = BashOperator(
        task_id="generate_lottery_lines",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/lottery_backtest.py --generate",
    )

    fetch_lottery_draws >> fetch_lottery_winners >> settle_lottery_lines >> generate_lottery_lines
