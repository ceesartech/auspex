#!/usr/bin/env bash
# Host-side pipeline runner. Runs every 15 minutes via cron.
#
# Three sequential steps inside the api container:
#   1. fetch_upcoming.py    — pull next 7 days of ESPN fixtures
#   2. compute_features.py  — fill features_cache for new scheduled matches
#   3. precompute_predictions.py — predict + cache + Telegram alert
#
# If any step fails, the next steps still run (set +e). Each step's stdout
# is appended to /var/log/auspex/pipeline.log so you can `tail -f` it.
#
# Suggested crontab line (edit with: `crontab -e` as the auspex user):
#   */15 * * * * /opt/auspex/scripts/run_pipeline.sh >> /var/log/auspex/cron.log 2>&1
#
# Manual invocation (works the same):
#   /opt/auspex/scripts/run_pipeline.sh

set +e

INSTALL_DIR="${INSTALL_DIR:-/opt/auspex}"
LOG_DIR="${LOG_DIR:-/var/log/auspex}"
PIPELINE_LOG="$LOG_DIR/pipeline.log"

mkdir -p "$LOG_DIR"

ts() { date -Iseconds; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$PIPELINE_LOG"; }

cd "$INSTALL_DIR" || { log "FATAL: cannot cd to $INSTALL_DIR"; exit 2; }

COMPOSE=(docker compose -f docker-compose.yml)
[ -f docker-compose.prod.yml ] && COMPOSE+=(-f docker-compose.prod.yml)

run_step() {
  local name="$1"; shift
  log "=== START $name"
  "${COMPOSE[@]}" exec -T api "$@" >>"$PIPELINE_LOG" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    log "=== OK    $name"
  else
    log "=== FAIL  $name (exit=$rc)"
  fi
  return "$rc"
}

# Skip the whole run if the api container isn't healthy. Avoids piling
# up failed runs while the stack is restarting.
if ! "${COMPOSE[@]}" exec -T api curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
  log "SKIP: api container is not healthy"
  exit 0
fi

run_step fetch_upcoming         python /app/scripts/fetch_upcoming.py --days 7
run_step compute_features       python /app/scripts/compute_features.py --days 7
run_step precompute_predictions python /app/scripts/precompute_predictions.py --days 7

log "=== DONE"
