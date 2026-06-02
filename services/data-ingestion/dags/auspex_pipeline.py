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
from airflow.utils.trigger_rule import TriggerRule

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

    # Turn the per-market predictions into value-bet recommendations by
    # comparing them against the latest ingested odds. Reads the most recent
    # odds snapshot — fetch_live_odds runs on its own cadence (its quota cost
    # makes it a poor fit for the 15-min loop), so recommendations are as
    # fresh as the last odds pull.
    generate_recommendations = BashOperator(
        task_id="generate_recommendations",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/generate_recommendations.py --days 14",
    )

    fetch_upcoming >> compute_features >> precompute_predictions >> generate_recommendations

    # ── NHL branch (Phase 4a: full ingest → features → predictions) ─
    # Predict task runs all 4 NHL ensembles (moneyline, regulation,
    # puck-line, total) per match via the Phase 4a TASKS registry and
    # writes one prediction row per task. High-confidence NHL picks
    # are pushed onto the same shared Redis queue as soccer picks; the
    # send_pipeline_digest task downstream drains both into one
    # combined Telegram message.
    fetch_upcoming_nhl = BashOperator(
        task_id="fetch_upcoming_nhl",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/fetch_upcoming.py --sport nhl --days 14",
    )

    compute_features_nhl = BashOperator(
        task_id="compute_features_nhl",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/compute_features_nhl.py --days 14",
    )

    precompute_predictions_nhl = BashOperator(
        task_id="precompute_predictions_nhl",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/precompute_predictions_nhl.py --days 14",
    )

    # Phase 4e: per-market NHL value-bet recommendations. Reads the
    # NHL predictions just written above, joins them to the latest
    # bookmaker odds, emits a betting_recommendations row + a digest
    # alert for every pick that clears the EV + prob thresholds.
    # Soccer has the same task at line 71.
    generate_recommendations_nhl = BashOperator(
        task_id="generate_recommendations_nhl",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/generate_recommendations_nhl.py --days 14",
    )

    fetch_upcoming_nhl >> compute_features_nhl >> precompute_predictions_nhl >> generate_recommendations_nhl

    # ── NBA branch (Phase 6a: ingest only — features/predictions/
    #    recommendations land in subsequent commits) ───────────────
    # Currently this branch JUST pulls upcoming NBA fixtures from
    # ESPN. compute_features_nba, precompute_predictions_nba, and
    # generate_recommendations_nba will be added in later commits
    # once the training corpus is in place.
    fetch_upcoming_nba = BashOperator(
        task_id="fetch_upcoming_nba",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/fetch_upcoming.py --sport nba --days 14",
    )

    # ── Phase 5: grade finished matches ───────────────────────────
    # Walks matches whose status flipped to 'finished' in the last
    # 14 days, computes actual_outcome per market, and:
    #   - updates predictions.actual_outcome / is_correct
    #   - settles betting_recommendations (status, profit_loss, settled_at)
    # Idempotent: WHERE EXISTS guards skip matches already fully graded.
    # Runs every 15 min so accuracy stats stay fresh as games complete.
    #
    # No dependency on the predict/rec tasks above — they're for
    # UPCOMING matches; grading is for FINISHED matches. The two
    # don't intersect, so this can run in parallel with the rest of
    # the pipeline.
    grade_completed_matches = BashOperator(
        task_id="grade_completed_matches",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/grade_completed_matches.py --days 14",
    )

    # ── Combined digest (fan-in) ──────────────────────────────────
    # Drains the shared Redis queue both branches push into and sends
    # ONE Telegram message with every sport's picks. Runs after both
    # branches' recommendations steps (so value-bet alerts from both
    # sports land in the same digest as the raw prediction alerts).
    #
    # trigger_rule=ALL_DONE so the digest runs whether or not a branch
    # failed. NONE_FAILED_MIN_ONE_SUCCESS (what we tried first) skips
    # the digest entirely when one branch errors, which silently drops
    # the OTHER branch's queued picks — exactly the regression we were
    # trying to avoid. Empty queue → no-op (the helper handles that).
    send_pipeline_digest = BashOperator(
        task_id="send_pipeline_digest",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/send_pipeline_digest.py",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    [generate_recommendations, generate_recommendations_nhl] >> send_pipeline_digest
