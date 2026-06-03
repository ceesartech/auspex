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

    # ── NBA branch (Phase 6b: ingestion + features) ───────────────
    # Predictions + recommendations land in subsequent commits once
    # the training corpus is in place.
    fetch_upcoming_nba = BashOperator(
        task_id="fetch_upcoming_nba",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/fetch_upcoming.py --sport nba --days 14",
    )

    compute_features_nba = BashOperator(
        task_id="compute_features_nba",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/compute_features_nba.py --days 14",
    )

    precompute_predictions_nba = BashOperator(
        task_id="precompute_predictions_nba",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/precompute_predictions_nba.py --days 14",
    )

    # NBA value-bet recommendations. Compares the model's per-market
    # probabilities (moneyline + spread + total) against the best
    # available bookmaker price and writes positive-EV picks to
    # betting_recommendations with quarter-Kelly sizing. Spread + total
    # only recommend at the closing line the model was trained on
    # (line-as-feature design — see scripts/generate_recommendations_nba.py
    # docstring for why).
    generate_recommendations_nba = BashOperator(
        task_id="generate_recommendations_nba",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/generate_recommendations_nba.py --days 14",
    )

    fetch_upcoming_nba >> compute_features_nba >> precompute_predictions_nba >> generate_recommendations_nba

    # ── NFL branch (Phase 10: end-to-end pipeline) ────────────────
    # Same chain shape as NBA: ingest → features → precompute → recs.
    # Spread + total recommendations are gated on the closing line
    # (line-as-feature design — see
    # scripts/generate_recommendations_nfl.py docstring).
    fetch_upcoming_nfl = BashOperator(
        task_id="fetch_upcoming_nfl",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/fetch_upcoming.py --sport nfl --days 14",
    )

    compute_features_nfl = BashOperator(
        task_id="compute_features_nfl",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/compute_features_nfl.py --days 14",
    )

    precompute_predictions_nfl = BashOperator(
        task_id="precompute_predictions_nfl",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/precompute_predictions_nfl.py --days 14",
    )

    generate_recommendations_nfl = BashOperator(
        task_id="generate_recommendations_nfl",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/generate_recommendations_nfl.py --days 14",
    )

    fetch_upcoming_nfl >> compute_features_nfl >> precompute_predictions_nfl >> generate_recommendations_nfl

    # ── Tennis branch (T4: end-to-end pipeline) ───────────────────
    # First 1v1 sport. Single market in v1 (moneyline) — total games
    # + set-betting land in v2 once linescore parsing is in place.
    fetch_upcoming_tennis = BashOperator(
        task_id="fetch_upcoming_tennis",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/fetch_upcoming.py --sport tennis --days 14",
    )

    compute_features_tennis = BashOperator(
        task_id="compute_features_tennis",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/compute_features_tennis.py --days 14",
    )

    precompute_predictions_tennis = BashOperator(
        task_id="precompute_predictions_tennis",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/precompute_predictions_tennis.py --days 14",
    )

    generate_recommendations_tennis = BashOperator(
        task_id="generate_recommendations_tennis",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/generate_recommendations_tennis.py --days 14",
    )

    fetch_upcoming_tennis >> compute_features_tennis >> precompute_predictions_tennis >> generate_recommendations_tennis

    # ── MMA branch (M4: end-to-end pipeline) ──────────────────────
    # Second 1v1 sport (UFC). Single market in v1 (moneyline) —
    # method-of-victory / round-group are MMA specialty markets
    # deferred to v2 (separate models).
    fetch_upcoming_mma = BashOperator(
        task_id="fetch_upcoming_mma",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/fetch_upcoming.py --sport mma --days 14",
    )

    compute_features_mma = BashOperator(
        task_id="compute_features_mma",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/compute_features_mma.py --days 14",
    )

    precompute_predictions_mma = BashOperator(
        task_id="precompute_predictions_mma",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/precompute_predictions_mma.py --days 14",
    )

    generate_recommendations_mma = BashOperator(
        task_id="generate_recommendations_mma",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/generate_recommendations_mma.py --days 14",
    )

    fetch_upcoming_mma >> compute_features_mma >> precompute_predictions_mma >> generate_recommendations_mma

    # ── Horse racing branch ───────────────────────────────────────
    # Pipeline shape per cron tick:
    #   1. upcoming racecards (next 7 days) from The Racing API.
    #      Captures the full bookmaker_odds[] array per runner
    #      (not just the morning_line_odds first-bookmaker decimal)
    #      so the recommendation step can do best-of-N pricing.
    #   2. yesterday's results (catch-up — only fires on Standard+ plan).
    #   3. compute_features_horse_racing — race-level + per-entrant.
    #   4. precompute_predictions_horse_racing — market-consensus
    #      baseline (devigged morning lines). Trained ML model
    #      replaces this once enough results data has accumulated.
    #   5. generate_recommendations_horse_racing — value bets via
    #      best-of-N bookmaker pricing vs the consensus probability.
    #      Without step 5 a consensus prediction yields zero EV
    #      against its own morning-line decimal by construction;
    #      the recs step is what unlocks any actionable signal.
    fetch_horse_racing_upcoming = BashOperator(
        task_id="fetch_horse_racing_upcoming",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/load_racing_api.py --upcoming 7",
    )
    fetch_horse_racing_results = BashOperator(
        task_id="fetch_horse_racing_results",
        bash_command=(
            f"{DOCKER_EXEC} python /app/scripts/load_racing_api.py "
            "--results --start $(date -u -d 'yesterday' +%Y-%m-%d) --end $(date -u -d 'yesterday' +%Y-%m-%d)"
        ),
    )
    compute_features_horse_racing = BashOperator(
        task_id="compute_features_horse_racing",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/compute_features_horse_racing.py --days 7",
    )
    precompute_predictions_horse_racing = BashOperator(
        task_id="precompute_predictions_horse_racing",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/precompute_predictions_horse_racing.py --days 7",
    )
    # LambdaMART ranker — beats the consensus baseline by +2.2pts
    # top-1 on the held-out test (memory: horse-racing-ml-ranker-v1).
    # Writes alongside the consensus into race_predictions under a
    # distinct model_name so the recs engine can prefer the ranker
    # when present and fall back to consensus when not. Runs AFTER
    # precompute_predictions_horse_racing so the ranker has access
    # to consensus_prob as an input feature (it's the top-importance
    # feature by a wide margin).
    precompute_predictions_horse_racing_ranker = BashOperator(
        task_id="precompute_predictions_horse_racing_ranker",
        bash_command=(f"{DOCKER_EXEC} python " "/app/scripts/precompute_predictions_horse_racing_ranker.py --days 7"),
    )
    generate_recommendations_horse_racing = BashOperator(
        task_id="generate_recommendations_horse_racing",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/generate_recommendations_horse_racing.py --days 2",
    )
    (
        fetch_horse_racing_upcoming
        >> fetch_horse_racing_results
        >> compute_features_horse_racing
        >> precompute_predictions_horse_racing
        >> precompute_predictions_horse_racing_ranker
        >> generate_recommendations_horse_racing
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

    # Parallel grader for the horse-racing schema (races,
    # race_entrants, race_predictions, race_recommendations). Same
    # split-from-the-pipeline rationale as grade_completed_matches:
    # operates on finished races independently of the upcoming-race
    # branch, so it can run alongside without ordering constraints.
    # Per-race semantics:
    #   - race_predictions: actual_outcome=1/0 per entrant (won vs
    #     lost), is_correct=TRUE iff the consensus favourite won
    #     (applied uniformly to every row in the race).
    #   - race_recommendations: status=won/lost/void with
    #     profit_loss = stake × (odds − 1) on won, −stake on lost,
    #     0 on void.
    grade_completed_races = BashOperator(
        task_id="grade_completed_races",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/grade_completed_races.py --days 14",
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

    [
        generate_recommendations,
        generate_recommendations_nhl,
        generate_recommendations_nba,
        generate_recommendations_nfl,
        generate_recommendations_tennis,
        generate_recommendations_mma,
        generate_recommendations_horse_racing,
    ] >> send_pipeline_digest

    # ── Model monitoring (Phase 9) ────────────────────────────────
    # Walks graded predictions in a 30-day rolling window, computes
    # ECE / MCE / Brier / log-loss / accuracy per (sport, market),
    # alerts via the shared Telegram digest when drift thresholds
    # are exceeded. Independent of the rec pipeline — runs after
    # the digest task so any drift alert lands AFTER the day's
    # value-bet alerts (read order matches operational priority).
    #
    # trigger_rule=ALL_DONE so a failed rec branch doesn't skip
    # monitoring — drift detection is exactly when we MOST need
    # to know if something's wrong upstream.
    monitor_models = BashOperator(
        task_id="monitor_models",
        bash_command=f"{DOCKER_EXEC} python /app/scripts/monitor_models.py --days 30",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    send_pipeline_digest >> monitor_models
