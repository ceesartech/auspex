#!/usr/bin/env bash
# One-shot operator script (2026-08-05): roll out the lottery honest-analytics
# layer on prod — apply migration 010 (lottery_predictions), backfill the full
# draw history from NY Open Data, and unpause the daily lottery_pipeline DAG.
# Idempotent: migration is IF NOT EXISTS, backfill upserts DO NOTHING.
# Delete this file once the rollout is verified.
#
# Run:  ssh auspex 'bash /opt/auspex/scripts/ops_lottery_rollout.sh'
set -euo pipefail
cd /opt/auspex

echo "== applying migration 010 (lottery_predictions) =="
docker compose cp services/data-ingestion/db/migrations/010_lottery_predictions.sql postgres:/tmp/010.sql
docker compose exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/010.sql'

echo "== backfilling full draw history (NY Open Data) =="
docker compose exec -T api python /app/scripts/fetch_lottery_draws.py --backfill

echo "== unpausing lottery_pipeline DAG =="
docker compose exec -T airflow-scheduler airflow dags unpause lottery_pipeline

echo "== post-check: draw counts by game =="
docker compose exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
SELECT game, count(*) AS draws, min(draw_date) AS first, max(draw_date) AS last
FROM lottery_draws GROUP BY game;"'
