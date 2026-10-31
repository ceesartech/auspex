"""Airflow DAG for the monthly soccer half-time Dixon-Coles refit.

The half-time Dixon-Coles (dixon_coles_ht_soccer) is deliberately NOT part of
the weekly retrain_models run: that DAG's swap_production copies everything
in /app/models/staging into production with no checks. The half-time trainer
writes to its own /app/models/staging-ht instead, and scripts/promote_dc_model.py
is the only way from there into production.

Runs at 05:30 UTC on the 1st of every month. Monthly is the cadence the gate
measured (scripts/ab_halftime_dixon_coles.py refits before each eval month).
05:30 also clears the Sunday retrain_models run, whose reload_api restart is
done by ~04:10 — a restart mid-fit would kill the `docker compose exec`.
Same docker-out-of-docker pattern as retrain_models.py: the work runs inside
the api container.

Flow (default ALL_SUCCESS trigger rules, so a refused fit never reaches promote):
  1. train_ht: scripts/train_halftime_dixon_coles.py --run-id <this run>.
     Fits every finished match and refuses (exit 1 → Telegram via
     notify_failure) an unconverged, level-biased, runaway-rho or unservable
     fit; otherwise writes model.bin + manifest.json — carrying this run id
     and the deployed git sha — to staging-ht atomically.
  2. promote_ht: scripts/promote_dc_model.py --model ht --expect-run-id <same>
     --rollback-on-verify-failure. Refuses a staged manifest from any other
     run, does nothing when the staged file already is production, otherwise
     checks, backs up, promotes atomically and verifies through the serve
     loader; a promotion that fails verification is restored from the backup
     (verified) before the task fails, so an unattended run never leaves an
     unverified artifact serving. Every failure's last log line is the
     rollback command. precompute_predictions serves it on the next 15-min
     tick; no api restart.

New DAGs start PAUSED here (core.dags_are_paused_at_creation=True), so after
the first deploy unpause it once:
    docker compose exec airflow-scheduler airflow dags unpause retrain_halftime_dc
Manual trigger:
    docker compose exec airflow-scheduler airflow dags trigger retrain_halftime_dc
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
    "retry_delay": timedelta(minutes=10),
}

# Null-safe run id, the same template as retrain_models.RUN_ID_TMPL (copied,
# not imported: importing another DAG file re-runs its parse-time code).
# run_id is in every task context, including Airflow 3 CLI manual runs that
# carry no logical_date. Both tasks render it identically, which is what lets
# promote_ht insist the staged manifest came from THIS run's train_ht.
RUN_ID_TMPL = "{{ run_id | replace(':', '') | replace('+', '') | replace('.', '') }}"

# Common compose prefix. The compose-file path matches what's mounted
# read-only into the airflow containers at /opt/auspex.
COMPOSE_EXEC = "docker compose -f /opt/auspex/docker-compose.yml exec -T"
DOCKER_EXEC = f"{COMPOSE_EXEC} api"

# The deployed commit, for the manifest's git_sha. Resolved in the airflow
# container (git + the read-only /opt/auspex checkout are there; the api
# container has neither) and passed in with `exec -e`. safe.directory because
# the checkout is owned by the host user, not airflow; anything failing →
# "unknown", which the manifest records rather than blocking the refit.
GIT_SHA_CMD = "git -c safe.directory=/opt/auspex -C /opt/auspex rev-parse HEAD 2>/dev/null || echo unknown"

with DAG(
    dag_id="retrain_halftime_dc",
    description="Monthly soccer half-time Dixon-Coles refit: train to staging-ht, then the gated promote",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="30 5 1 * *",  # 05:30 UTC on the 1st of every month
    catchup=False,
    max_active_runs=1,
    tags=["ml", "retraining"],
) as dag:

    train_ht = BashOperator(
        task_id="train_ht",
        # `timeout 45m` bounds the fit WHERE IT RUNS: execution_timeout only
        # SIGTERMs the host-side compose client, and a killed `docker exec`
        # client does not stop the exec'd process.
        execution_timeout=timedelta(minutes=50),
        bash_command=(
            f'GIT_SHA="$({GIT_SHA_CMD})"; '
            f'{COMPOSE_EXEC} -e GIT_SHA="$GIT_SHA" api timeout 45m '
            "python /app/scripts/train_halftime_dixon_coles.py --run-id '" + RUN_ID_TMPL + "'"
        ),
    )

    promote_ht = BashOperator(
        task_id="promote_ht",
        # No retry. A promotion that failed verification has been restored
        # from the backup (--rollback-on-verify-failure), so a retry would
        # only repeat it; and if the restore itself failed, production holds
        # the staged file and a retry would see staged == production, log
        # "nothing to promote" and exit 0 — turning a real failure green. The
        # task log ends with the rollback command.
        retries=0,
        execution_timeout=timedelta(minutes=12),
        bash_command=(
            f"{DOCKER_EXEC} timeout 10m python /app/scripts/promote_dc_model.py --model ht "
            "--expect-run-id '" + RUN_ID_TMPL + "' --rollback-on-verify-failure"
        ),
    )

    train_ht >> promote_ht
