#!/usr/bin/env bash
# Pull-and-redeploy. Runs ON THE VM, invoked by CI over SSH.
#
# Flow:
#   1. Stash any local changes (none should exist, but be defensive).
#   2. git pull --ff-only from the configured remote/branch.
#   3. Determine which services need rebuilding by diffing changed
#      paths against a small file→service map. (Avoids rebuilding
#      the frontend on a Python-only change, etc.)
#   4. Rebuild the affected images.
#   5. up -d --no-deps for those services.
#   6. Run /health to verify the API is still healthy. If not,
#      print recent logs.
#
# Idempotent. Safe to call repeatedly. Designed so CI can pipe its
# output back to GitHub Actions logs.
#
# Usage:
#   ./scripts/deploy_remote.sh                       # deploy from current HEAD's commit
#   FORCE_ALL=1 ./scripts/deploy_remote.sh           # rebuild every service
#   SKIP_BUILD=1 ./scripts/deploy_remote.sh          # just restart, no rebuild
#
# Run from the project root.

set -euo pipefail

REMOTE="${REMOTE:-caesar}"
BRANCH="${BRANCH:-main}"
FORCE_ALL="${FORCE_ALL:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"

log()  { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deploy]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[deploy]\033[0m %s\n' "$*" >&2; }

if [ ! -f docker-compose.yml ]; then
  err "docker-compose.yml not found in $(pwd). cd to the project root."
  exit 2
fi

COMPOSE=(docker compose -f docker-compose.yml)
if [ -f docker-compose.prod.yml ]; then
  COMPOSE+=(-f docker-compose.prod.yml)
fi

# 1. Capture the SHA BEFORE the pull so we can diff against it.
OLD_SHA="$(git rev-parse HEAD)"
log "Currently at: $OLD_SHA"

# 2. Pull.
log "Fetching $REMOTE/$BRANCH..."
git fetch --quiet "$REMOTE" "$BRANCH"
NEW_SHA="$(git rev-parse "$REMOTE/$BRANCH")"

if [ "$OLD_SHA" = "$NEW_SHA" ]; then
  log "Already up to date at $OLD_SHA. Nothing to do."
  exit 0
fi

# Be defensive: stash any local changes so the pull doesn't fail.
if ! git diff --quiet HEAD -- || ! git diff --cached --quiet; then
  warn "Local changes detected; stashing before pull."
  git stash push -m "auto-stash before deploy_remote $(date -Iseconds)" >/dev/null
fi

git pull --ff-only "$REMOTE" "$BRANCH"
log "Now at: $NEW_SHA"

# 3. Determine which services need rebuilding.
declare -a SERVICES_TO_REBUILD=()

if [ "$FORCE_ALL" = "1" ] || [ "$SKIP_BUILD" = "1" ]; then
  : # handled below
else
  # File-prefix → service name map. If any file under the prefix changed,
  # the matching service rebuilds.
  CHANGED_FILES="$(git diff --name-only "$OLD_SHA" "$NEW_SHA")"
  declare -A PREFIX_TO_SERVICE=(
    ["services/api/"]="api"
    ["services/data-ingestion/"]="api"   # api image bundles data-ingestion src
    ["services/feature-engineering/"]="api"
    ["services/ml-models/"]="api"
    ["scripts/"]="api"
    ["requirements.txt"]="api"
    ["requirements-torch.txt"]="api"
    ["docker/Dockerfile.api"]="api"
    ["services/frontend/"]="frontend"
    ["docker/Dockerfile.frontend"]="frontend"
    ["docker/Dockerfile.airflow"]="airflow-webserver airflow-scheduler"
    ["infrastructure/caddy/Caddyfile"]="caddy"
  )

  declare -A AFFECTED=()
  while IFS= read -r file; do
    [ -z "$file" ] && continue
    for prefix in "${!PREFIX_TO_SERVICE[@]}"; do
      case "$file" in
        "$prefix"*)
          for svc in ${PREFIX_TO_SERVICE[$prefix]}; do
            AFFECTED[$svc]=1
          done
          break
          ;;
      esac
    done
  done <<< "$CHANGED_FILES"

  SERVICES_TO_REBUILD=("${!AFFECTED[@]}")
fi

if [ "$FORCE_ALL" = "1" ]; then
  log "FORCE_ALL=1 — rebuilding all services."
  SERVICES_TO_REBUILD=(api frontend)
fi

# 4. Rebuild.
if [ "$SKIP_BUILD" = "1" ] || [ "${#SERVICES_TO_REBUILD[@]}" -eq 0 ]; then
  log "No service rebuilds needed."
else
  log "Rebuilding: ${SERVICES_TO_REBUILD[*]}"
  for svc in "${SERVICES_TO_REBUILD[@]}"; do
    # caddy is image-based, no build context
    [ "$svc" = "caddy" ] && continue
    log "build $svc"
    "${COMPOSE[@]}" build "$svc"
  done
fi

# 5. Up.
log "Bringing services up..."
if [ "${#SERVICES_TO_REBUILD[@]}" -eq 0 ] && [ "$SKIP_BUILD" != "1" ]; then
  "${COMPOSE[@]}" up -d
else
  # --no-deps avoids restarting healthy dependencies like postgres/redis.
  # Compose still respects depends_on healthchecks for new starts.
  "${COMPOSE[@]}" up -d --no-deps "${SERVICES_TO_REBUILD[@]}"
fi

# 6. Health check.
log "Verifying API health..."
for i in $(seq 1 30); do
  if "${COMPOSE[@]}" exec -T api curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    log "API healthy ✓"
    log "Deploy complete: $OLD_SHA → $NEW_SHA"
    exit 0
  fi
  sleep 2
done

err "API failed to become healthy after 60s. Recent logs:"
"${COMPOSE[@]}" logs api --tail=40
exit 1
