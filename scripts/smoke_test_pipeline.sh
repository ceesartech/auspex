#!/usr/bin/env bash
# End-to-end smoke test for the auspex_pipeline DAG.
#
# Picks N recently-finished matches, temporarily flips them to status
# 'scheduled' with a near-future match_date, runs the full pipeline
# (compute_features → precompute_predictions → optional Telegram), and
# reverts the matches back to their original state.
#
# Use case: during off-season for the configured leagues (no live
# fixtures coming back from ESPN), this is the only way to exercise
# the inference path end-to-end without waiting for the season to
# resume.
#
# Usage:
#   ./scripts/smoke_test_pipeline.sh            # default: 50 matches, no Telegram
#   N=20 ./scripts/smoke_test_pipeline.sh       # smaller sample
#   NOTIFY=1 ./scripts/smoke_test_pipeline.sh   # let high-confidence predictions ping Telegram
#
# Run from the project root.

set -euo pipefail

N="${N:-50}"
NOTIFY="${NOTIFY:-0}"

log()  { printf '\033[1;34m[smoke]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[smoke]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[smoke]\033[0m %s\n' "$*" >&2; }

if [ ! -f docker-compose.yml ]; then
  err "docker-compose.yml not found in $(pwd). cd to the project root."
  exit 2
fi

COMPOSE=(docker compose -f docker-compose.yml)
[ -f docker-compose.prod.yml ] && COMPOSE+=(-f docker-compose.prod.yml)

# A scratch table tracks which match IDs we touched so we can revert
# precisely even if the pipeline takes hours and someone else added
# matches in the meantime.
SCRATCH_TABLE="smoke_test_pipeline_targets"

cleanup_on_exit() {
  local rc=$?
  log "Reverting matches to their original state..."
  "${COMPOSE[@]}" exec -T postgres psql -U betting_user -d betting_system <<SQL
    UPDATE matches m SET
      status = s.orig_status,
      match_date = s.orig_match_date
    FROM ${SCRATCH_TABLE} s
    WHERE m.id = s.match_id;
    DROP TABLE IF EXISTS ${SCRATCH_TABLE};
SQL
  if [ "$rc" -ne 0 ]; then
    err "Smoke test failed with exit code $rc. Matches reverted; check logs."
  fi
}
trap cleanup_on_exit EXIT INT TERM

log "Picking $N most-recent finished matches..."
"${COMPOSE[@]}" exec -T postgres psql -U betting_user -d betting_system <<SQL
  DROP TABLE IF EXISTS ${SCRATCH_TABLE};
  CREATE TABLE ${SCRATCH_TABLE} AS
    SELECT id AS match_id,
           status AS orig_status,
           match_date AS orig_match_date
    FROM matches
    WHERE status = 'finished'
    ORDER BY match_date DESC
    LIMIT ${N};
  SELECT COUNT(*) AS targeted FROM ${SCRATCH_TABLE};
SQL

log "Flipping targeted matches to 'scheduled' with match_date = NOW() + 2 days..."
"${COMPOSE[@]}" exec -T postgres psql -U betting_user -d betting_system <<SQL
  UPDATE matches m SET
    status = 'scheduled',
    match_date = NOW() + INTERVAL '2 days' + (random() * INTERVAL '5 days')
  FROM ${SCRATCH_TABLE} s
  WHERE m.id = s.match_id;
SQL

log "Step 1/2: compute_features"
"${COMPOSE[@]}" exec -T api python /app/scripts/compute_features.py --days 7 \
  || warn "compute_features had errors (continuing — orchestrator may complete partially)"

log "Step 2/2: precompute_predictions"
if [ "$NOTIFY" = "1" ]; then
  "${COMPOSE[@]}" exec -T api python /app/scripts/precompute_predictions.py \
      --days 7 --notify-threshold 0.65
else
  "${COMPOSE[@]}" exec -T api python /app/scripts/precompute_predictions.py \
      --days 7 --no-notify
fi

log "Predictions written:"
"${COMPOSE[@]}" exec -T postgres psql -U betting_user -d betting_system <<SQL
  SELECT COUNT(*) AS predicted_count
  FROM predictions p
  JOIN ${SCRATCH_TABLE} s ON p.match_id = s.match_id
  WHERE p.updated_at > NOW() - INTERVAL '10 minutes';
SQL

log "Sample predictions vs. actual outcomes (10):"
"${COMPOSE[@]}" exec -T postgres psql -U betting_user -d betting_system <<SQL
  SELECT
    ht.name || ' vs ' || at.name AS match,
    CASE
      WHEN m.home_score > m.away_score THEN 'home'
      WHEN m.home_score = m.away_score THEN 'draw'
      ELSE 'away'
    END AS actual,
    p.predicted_outcome AS predicted,
    ROUND(p.confidence::numeric, 3) AS confidence
  FROM ${SCRATCH_TABLE} s
  JOIN matches m ON m.id = s.match_id
  JOIN teams ht ON ht.id = m.home_team_id
  JOIN teams at ON at.id = m.away_team_id
  LEFT JOIN predictions p ON p.match_id = m.id
  WHERE p.predicted_outcome IS NOT NULL
  ORDER BY p.confidence DESC NULLS LAST
  LIMIT 10;
SQL

log "Smoke test complete. Matches will revert when this script exits."
