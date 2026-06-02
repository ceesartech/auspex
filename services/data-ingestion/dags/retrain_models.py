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

# Logical sport → ordered list of SPORT_BUNDLES keys to train.
#
# We use ONE DAG task per logical sport (not per bundle) so the graph
# doesn't blow up to 4N tasks once we have NBA / NFL / etc. Each
# task's bash command iterates its bundle list sequentially inside
# the api container. Adding a new sport = adding one entry here and
# one >> wire below; no code change needed in train_all_models.py.
#
# Soccer's "match_result" ensemble + Dixon-Coles internally derive
# ~15 markets at predict time (see scripts/precompute_predictions.py),
# so soccer doesn't need a per-market bundle list. NHL's four markets
# (moneyline, regulation, puck_line, total) are direct classifiers
# with different label sets, so each gets its own bundle.
SPORT_BUNDLE_GROUPS: dict[str, list[str]] = {
    # Soccer trains ONE bundle (match_result / 1X2). The other ~15
    # soccer markets are derived from the Dixon-Coles base model at
    # predict time (see DERIVED_SOCCER_MARKETS in
    # services/api/src/services/prediction_service.py), so soccer
    # doesn't need a per-market bundle list. The bundle key follows
    # the *_soccer_<market> convention that parallels NHL.
    "soccer": ["soccer_match_result"],
    "nhl": ["nhl_moneyline", "nhl_regulation", "nhl_puck_line", "nhl_total"],
    # NBA: line-as-feature design for spread + total — one trained
    # model per market handles every line the book offers (vs NHL's
    # fixed-line approach). Moneyline is a straight 2-class winner.
    # All 3 share the nba_baseline feature_set written by
    # scripts/compute_features_nba.py.
    "nba": ["nba_moneyline", "nba_spread", "nba_total"],
}

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

    # One BashOperator per LOGICAL SPORT. Each task internally loops
    # over that sport's bundle list, training each bundle
    # sequentially in a single container invocation. This collapses
    # what would otherwise be N_sports × N_markets tasks into N_sports
    # tasks — the graph stays readable as we add NBA / NFL / etc.
    #
    # `set +e` inside the loop so one bundle's failure doesn't abort
    # the rest of that sport's training. Failures are captured in
    # `failed_bundles` and printed at the end; the task still exits 0
    # if at least one bundle wrote artifacts to staging (the swap
    # step's per-item merge handles partial output safely).
    train_tasks: dict[str, BashOperator] = {}
    for sport, bundles in SPORT_BUNDLE_GROUPS.items():
        bundle_list_bash = " ".join(bundles)
        train_tasks[sport] = BashOperator(
            task_id=f"train_{sport}",
            execution_timeout=timedelta(minutes=90),
            # Don't let one sport's data hiccup block the others —
            # all sport tasks attempt; swap merges whatever landed.
            trigger_rule=TriggerRule.ALL_DONE,
            bash_command=(
                f"{DOCKER_EXEC} bash -c '"
                "cd /app/services/ml-models; "
                'failed_bundles=""; '
                f"for bundle in {bundle_list_bash}; do "
                'echo "=== training bundle $bundle ==="; '
                "if ! PYTHONPATH=src python -m training.train_all_models "
                "--sport $bundle "
                "--model-type all "
                '--database-url "$DATABASE_URL" '
                "--output-dir /app/models/staging "
                "--export-onnx; then "
                'echo "!!! bundle $bundle FAILED"; '
                'failed_bundles="$failed_bundles $bundle"; '
                "fi; "
                "done; "
                'if [ -n "$failed_bundles" ]; then '
                'echo "Failed bundles:$failed_bundles"; '
                "fi; "
                # Exit 0 if ANY bundle succeeded (staging has its
                # artifacts), non-zero only if every single bundle
                # failed — that surfaces a real problem worth paging.
                'if [ -d /app/models/staging ] && [ -n "$(ls -A /app/models/staging 2>/dev/null)" ]; then '
                "exit 0; "
                "else "
                'echo "No artifacts in staging — every bundle failed."; '
                "exit 1; "
                "fi"
                "'"
            ),
        )

    # MERGE staging into production (don't wholesale replace). The
    # previous wholesale `mv staging production` clobbered any sport's
    # artifacts that weren't in staging — which was the root cause of
    # NHL models disappearing after a soccer-only retrain. Partial
    # retrains stay safe: a sport that failed to train keeps its
    # previous model in production.
    #
    # Why a per-item rm + cp loop (not a single `cp -aR staging/.`):
    # production may contain SYMLINKS (e.g. a manual promote of an
    # earlier training output). `cp` refuses to overwrite a symlink
    # with a real directory ("cannot overwrite non-directory"), so a
    # wholesale cp aborts mid-merge. Removing the destination first
    # handles symlinks, real dirs, and missing dirs uniformly.
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
            # Self-heal: earlier DAG versions had `%%Y%%m%%d-%%H%%M%%S`
            # in the date format. The double-percent was meant as
            # Airflow templating escape but our BashOperator is
            # Jinja2-templated and doesn't need it — bash passed
            # `%%Y...` to `date`, which printed `%Y%m%d-%H%M%S`
            # literally. Every "successful" old run then created or
            # nested into a dir named that literal string. Remove
            # any such dir before we proceed so the new run is clean.
            "rm -rf /app/models/production-prev-%Y%m%d-%H%M%S; "
            # Now the correct timestamp — no Airflow %% escape needed.
            "STAMP=$(date -u +%Y%m%d-%H%M%S); "
            "if [ ! -d /app/models/staging ]; then "
            'echo "No staging dir — every train_<sport> task failed. Leaving production untouched."; '
            "exit 0; "
            "fi; "
            "if [ -d /app/models/production ]; then "
            "cp -a /app/models/production /app/models/production-prev-${STAMP}; "
            "fi; "
            "mkdir -p /app/models/production; "
            # Per-item merge: rm-then-cp each top-level entry from
            # staging. Handles symlink-in-production → real-dir-in-
            # staging cleanly.
            "for item in /app/models/staging/*; do "
            'name=$(basename "$item"); '
            'rm -rf "/app/models/production/$name"; '
            'cp -a "$item" "/app/models/production/$name"; '
            "done; "
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

    # Validate gates SOCCER training only — non-soccer sports do their
    # own data validation inside each bundle's load_frame, and a
    # soccer corpus issue shouldn't block other sports.
    validate_data >> train_tasks["soccer"]

    # Chain sport tasks sequentially. Each sport's training runs all
    # its bundles internally (see the bash loop in each task), so the
    # DAG only has one task per logical sport — adding NBA later =
    # one entry in SPORT_BUNDLE_GROUPS + one extra link in this chain.
    sport_order = list(SPORT_BUNDLE_GROUPS.keys())
    prev = train_tasks[sport_order[0]]
    for sport in sport_order[1:]:
        prev >> train_tasks[sport]
        prev = train_tasks[sport]

    # The last sport feeds swap_production, then reload + cleanup.
    prev >> swap_production >> reload_api >> cleanup_old_backups
