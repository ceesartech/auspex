"""Airflow DAG for weekly on-VM model retraining.

Runs every Sunday at 04:00 UTC. Like auspex_pipeline.py it uses the
docker-out-of-docker pattern — all heavy work runs inside the api
container via `docker compose exec api …`. The airflow scheduler just
orchestrates: docker CLI + the bind-mounted host socket let it signal
the host daemon, which already has the api container with all our
Python dependencies.

Flow:
  1. validate_data: ensures the soccer matches table has enough rows
     + valid feature columns. Aborts the run with a clear error if
     not. NHL bundles do their own validation inside load_frame, so
     this gate is soccer-specific (gates train_soccer only).
  2. train_<sport>: one task per SportBundle (soccer + 4 NHL markets).
     All write to /app/models/staging. Run sequentially so the api
     container isn't running 5 ensembles' worth of training in
     parallel. If any one fails, the others still execute — that's
     `trigger_rule=all_done` on swap_production, which then MERGES
     whatever made it into staging on top of the existing production
     (instead of wholesale replacing). Partial-retrain safety.
  3. swap_production: merges staging → production, preserving any
     existing production artifacts not covered by this run. Backs up
     production to production-prev-<timestamp> first for rollback.
     The merge semantics matter: if e.g. NHL puck_line training
     fails, the previous puck_line model stays in production.
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
from airflow.utils.trigger_rule import TriggerRule

# Sports trained on every run. Matches the SPORT_BUNDLES keys in
# services/ml-models/src/training/train_all_models.py — add a new
# bundle there, add it here, and the DAG starts training it on the
# next Sunday run.
SPORTS = ["soccer", "nhl_moneyline", "nhl_regulation", "nhl_puck_line", "nhl_total"]

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

    # One BashOperator per SportBundle. Sequential (not parallel)
    # because each training run is CPU-heavy — Poisson + Dixon-Coles
    # do per-team MLE fits that can take 20-30 minutes on a full
    # corpus, and running five in parallel would saturate the api
    # container. Sequential keeps the resource profile predictable;
    # total wall-clock is still under the 90-minute per-task SLA
    # because NHL bundles are smaller than soccer.
    train_tasks = {}
    for sport in SPORTS:
        train_tasks[sport] = BashOperator(
            task_id=f"train_{sport}",
            execution_timeout=timedelta(minutes=90),
            # Don't let a single sport's data hiccup block the others —
            # all five tasks should attempt, and swap_production will
            # merge whatever made it into staging.
            trigger_rule=TriggerRule.ALL_DONE,
            bash_command=(
                f"{DOCKER_EXEC} bash -c '"
                "cd /app/services/ml-models && "
                "PYTHONPATH=src python -m training.train_all_models "
                f"--sport {sport} "
                "--model-type all "
                '--database-url "$DATABASE_URL" '
                "--output-dir /app/models/staging "
                "--export-onnx"
                "'"
            ),
        )

    # MERGE staging into production (don't wholesale replace). The
    # previous wholesale `mv staging production` clobbered any sport's
    # artifacts that weren't in staging — which was the root cause of
    # NHL models disappearing after a soccer-only retrain. `cp -aR`
    # with the `staging/.` trailing slash copies the CONTENTS of
    # staging into production, overwriting anything with the same
    # path and leaving everything else alone. Partial retrains stay
    # safe: a sport that failed to train keeps its previous model in
    # production.
    #
    # trigger_rule=ALL_DONE: run even if some train_<sport> tasks
    # failed. Without this, one finicky NHL bundle would skip the
    # whole swap and the operator would have to manually intervene.
    swap_production = BashOperator(
        task_id="swap_production",
        trigger_rule=TriggerRule.ALL_DONE,
        bash_command=(
            f"{DOCKER_EXEC} bash -c '"
            "set -euo pipefail; "
            "STAMP=$(date -u +%%Y%%m%%d-%%H%%M%%S); "
            "if [ ! -d /app/models/staging ]; then "
            'echo "No staging dir — every train_<sport> task failed. Leaving production untouched."; '
            "exit 0; "
            "fi; "
            "if [ -d /app/models/production ]; then "
            "cp -a /app/models/production /app/models/production-prev-${STAMP}; "
            "fi; "
            "mkdir -p /app/models/production; "
            "cp -aR /app/models/staging/. /app/models/production/; "
            "rm -rf /app/models/staging; "
            'echo "--- production after merge ---"; '
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

    # Validate gates SOCCER training only — NHL bundles handle their
    # own data validation inside load_frame, and a soccer corpus issue
    # shouldn't block NHL retraining.
    validate_data >> train_tasks["soccer"]

    # Chain each sport's training task to the previous one's completion
    # (not success — trigger_rule on each is ALL_DONE, so a soccer
    # training failure still lets the NHL bundles run). Sequential
    # because each training run is CPU-heavy; running them in parallel
    # would saturate the api container.
    prev = train_tasks["soccer"]
    for sport in SPORTS[1:]:
        prev >> train_tasks[sport]
        prev = train_tasks[sport]

    # The last sport feeds swap_production, which then chains through
    # reload + cleanup. (swap_production's own trigger_rule=ALL_DONE
    # makes it run even if some sports failed — see comment on the
    # swap_production BashOperator.)
    prev >> swap_production >> reload_api >> cleanup_old_backups
