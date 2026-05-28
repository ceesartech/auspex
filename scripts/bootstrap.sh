#!/usr/bin/env bash
# Full bootstrap of the Auspex prediction pipeline.
#
# Runs the following inside the running stack:
#   1. Load football-data.co.uk historicals (10 seasons, top-5 leagues)
#   2. Load StatsBomb open-data match aggregates
#   3. Promote raw rows -> canonical schema (leagues, teams, matches, ...)
#   4. Train all models with calibration
#   5. Promote staging models to production and reload the API
#   6. Warm the prediction cache for upcoming matches
#   7. Sanity-check by hitting /api/v1/predictions/upcoming
#
# Idempotent: re-running picks up new seasons + matches without
# duplicating anything. Safe to run on a schedule (e.g. weekly via cron
# or Airflow); for now a manual run is enough to go from empty DB to
# serving predictions.
#
# Prerequisites:
#   - Stack is up: `docker compose ps` shows api + postgres + redis healthy
#   - /opt/auspex/.env is fully populated (run regenerate_secrets.sh first
#     on a fresh deploy)
#
# Usage:
#   ./scripts/bootstrap.sh                   # full pipeline
#   SKIP_LOAD=1 ./scripts/bootstrap.sh       # skip step 1-2 (reuse raw_*)
#   SKIP_TRAIN=1 ./scripts/bootstrap.sh      # skip step 4-5 (transform-only)
#   LEAGUES=E0,D1 SEASONS=5 ./scripts/bootstrap.sh   # smaller scope

set -euo pipefail

# Tunables.
LEAGUES="${LEAGUES:-E0,E1,D1,I1,SP1,F1}"
SEASONS="${SEASONS:-10}"
SKIP_LOAD="${SKIP_LOAD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_WARM="${SKIP_WARM:-0}"
SKIP_STATSBOMB="${SKIP_STATSBOMB:-0}"

log()   { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[bootstrap]\033[0m %s\n' "$*" >&2; }
err()   { printf '\033[1;31m[bootstrap]\033[0m %s\n' "$*" >&2; }
step()  { printf '\n\033[1;36m============================================\n  %s\n============================================\033[0m\n' "$*"; }

if [ ! -f docker-compose.yml ]; then
  err "docker-compose.yml not found in $(pwd). cd to the project root."
  exit 2
fi

# Verify the API container is up before we start.
if ! docker compose ps api --status running --format json | grep -q .; then
  err "api container is not running. Start the stack first: ./scripts/restart_services.sh"
  exit 2
fi

# Wait for Postgres to be ready (it usually is, but we just restarted).
log "Waiting for Postgres..."
for _ in $(seq 1 30); do
  if docker compose exec -T postgres pg_isready -U betting_user >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

START_TS=$(date +%s)

# ─────────────────────────────────────────────────────────────────────
# Step 1 — Load football-data.co.uk historicals
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_LOAD" = "1" ]; then
  log "SKIP_LOAD=1 → skipping football-data loader"
else
  step "Step 1/7  Load football-data.co.uk ($LEAGUES, $SEASONS seasons)"
  docker compose exec -T api python /app/scripts/load_football_data.py \
      --leagues "$LEAGUES" --seasons "$SEASONS"
fi

# ─────────────────────────────────────────────────────────────────────
# Step 2 — Load StatsBomb open-data (optional, slow)
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_LOAD" = "1" ] || [ "$SKIP_STATSBOMB" = "1" ]; then
  log "Skipping StatsBomb loader"
else
  step "Step 2/7  Load StatsBomb open-data (--no-events for speed)"
  # --no-events skips the per-match event-level pulls (~10s/match) and
  # only loads match-level metadata. Drop --no-events for richer xG data
  # at the cost of much longer runtime.
  docker compose exec -T api python /app/scripts/load_statsbomb.py \
      --all-open --no-events || warn "StatsBomb load had errors (non-fatal)"
fi

# ─────────────────────────────────────────────────────────────────────
# Step 3 — Promote raw rows to canonical schema
# ─────────────────────────────────────────────────────────────────────
step "Step 3/7  Promote raw rows → canonical (leagues/teams/matches/odds)"
docker compose exec -T api python /app/scripts/promote_raw.py

# Show what landed.
log "Canonical schema row counts:"
docker compose exec -T postgres psql -U betting_user -d betting_system -t -c "
  SELECT
    (SELECT COUNT(*) FROM leagues)     AS leagues,
    (SELECT COUNT(*) FROM teams)       AS teams,
    (SELECT COUNT(*) FROM matches)     AS matches,
    (SELECT COUNT(*) FROM match_stats) AS match_stats,
    (SELECT COUNT(*) FROM odds)        AS odds;
"

# ─────────────────────────────────────────────────────────────────────
# Step 4 — Train all models
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_TRAIN" = "1" ]; then
  log "SKIP_TRAIN=1 → skipping training"
else
  step "Step 4/7  Train models (calibrated, time-series split)"
  docker compose exec -T api bash -c '
    cd /app/services/ml-models && \
    PYTHONPATH=src python -m training.train_all_models \
        --database-url "$DATABASE_URL" \
        --output-dir /models/staging
  '

  # ───────────────────────────────────────────────────────────────────
  # Step 5 — Promote staging models to production
  # ───────────────────────────────────────────────────────────────────
  step "Step 5/7  Promote staging models → production + reload API"
  docker compose exec -T api bash -c '
    mkdir -p /models/production
    cp -r /models/staging/* /models/production/ 2>/dev/null || true
    ls -la /models/production/
  '
  docker compose restart api
  # Wait for the API to come back healthy.
  log "Waiting for API to become healthy..."
  for _ in $(seq 1 30); do
    if docker compose exec -T api curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
fi

# ─────────────────────────────────────────────────────────────────────
# Step 6 — Warm the prediction cache for upcoming matches
# ─────────────────────────────────────────────────────────────────────
if [ "$SKIP_WARM" = "1" ]; then
  log "SKIP_WARM=1 → skipping cache warm"
else
  step "Step 6/7  Warm prediction cache for next 7 days of matches"
  DOMAIN=$(grep -E "^AUSPEX_DOMAIN=" .env 2>/dev/null | head -1 | cut -d= -f2-)
  if [ -z "${DOMAIN:-}" ]; then
    warn "AUSPEX_DOMAIN not set; skipping warm via public URL"
  else
    # We need a JWT to hit the predictions endpoint. Use the admin
    # account we seeded earlier. If the password changed, override via
    # ADMIN_PW=...
    ADMIN_PW="${ADMIN_PW:-Admin@1234}"
    TOKEN=$(curl -sf -X POST "https://${DOMAIN}/api/v1/user/login" \
      -H "Content-Type: application/json" \
      -d "{\"username\":\"admin\",\"password\":\"${ADMIN_PW}\"}" \
      | python3 -c "import json,sys;print(json.load(sys.stdin).get('access_token',''))")
    if [ -z "$TOKEN" ]; then
      warn "Could not get JWT for cache-warm; skipping. Login manually to verify auth."
    else
      curl -sf -H "Authorization: Bearer $TOKEN" \
        "https://${DOMAIN}/api/v1/predictions/upcoming?limit=200" \
        | python3 -c "
import json,sys
data = json.load(sys.stdin)
print(f'Warmed {len(data)} upcoming predictions')
" || warn "Warm request failed (no upcoming matches yet is OK on a historical-only DB)"
    fi
  fi
fi

# ─────────────────────────────────────────────────────────────────────
# Step 7 — Smoke test
# ─────────────────────────────────────────────────────────────────────
step "Step 7/7  Smoke test"
docker compose exec -T postgres psql -U betting_user -d betting_system -c "
  SELECT 'matches'   AS table, COUNT(*) FROM matches UNION ALL
  SELECT 'teams'     AS table, COUNT(*) FROM teams   UNION ALL
  SELECT 'odds'      AS table, COUNT(*) FROM odds    UNION ALL
  SELECT 'users'     AS table, COUNT(*) FROM users;
"

ELAPSED=$(( $(date +%s) - START_TS ))
log "Bootstrap complete in $((ELAPSED/60))m $((ELAPSED%60))s"
