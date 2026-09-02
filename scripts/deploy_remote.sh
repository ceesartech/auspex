#!/usr/bin/env bash
# Pull-and-redeploy from ghcr.io. Runs ON THE VM, invoked by CI over SSH.
#
# Flow:
#   1. Defensive stash + git pull --ff-only to bring code (compose files,
#      scripts, DAGs) in sync with the deployed SHA.
#   2. docker login to ghcr.io with the token CI provides.
#   3. docker compose pull (layered with docker-compose.ghcr.yml so it
#      pulls images from ghcr.io/ceesartech/auspex/{api,airflow,frontend}
#      at the tag CI built).
#   4. docker compose up -d to swap in the new containers. Compose's
#      built-in rolling start handles dependency order.
#   5. Health-check the api container for up to 60s. Print recent logs
#      on failure so CI shows the smoking gun.
#
# Environment variables expected (set by CI):
#   IMAGE_TAG        — git SHA, used to tag images
#   GHCR_USER        — github actor (for docker login)
#   GHCR_TOKEN       — GITHUB_TOKEN (for docker login AND the git fetch;
#                      see the GIT_AUTH note below)
#   REMOTE / BRANCH  — git remote/branch on the VM (default: origin/main)
#
# Manual usage on the VM (without ghcr.io, fall back to local build):
#   FORCE_LOCAL_BUILD=1 ./scripts/deploy_remote.sh

set -euo pipefail

REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FORCE_LOCAL_BUILD="${FORCE_LOCAL_BUILD:-0}"

log()  { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deploy]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[deploy]\033[0m %s\n' "$*" >&2; }

if [ ! -f docker-compose.yml ]; then
  err "docker-compose.yml not found in $(pwd). cd to the project root."
  exit 2
fi

# Compose file stack. ghcr overlay is added only when not falling back
# to local build, since the overlay clears `build:` directives.
COMPOSE=(docker compose -f docker-compose.yml)
if [ -f docker-compose.prod.yml ]; then
  COMPOSE+=(-f docker-compose.prod.yml)
fi
if [ "$FORCE_LOCAL_BUILD" != "1" ] && [ -f docker-compose.ghcr.yml ]; then
  COMPOSE+=(-f docker-compose.ghcr.yml)
fi

# 1. Capture the SHA BEFORE the pull so we can diff against it.
OLD_SHA="$(git rev-parse HEAD)"
log "Currently at: $OLD_SHA"

# Defensive stash if there are any working-tree changes — including
# untracked files. The previous version only checked tracked-file
# diffs, which let untracked files (e.g. a script scp'd onto the VM
# for ad-hoc testing) block the pull when the same path arrived as
# a committed addition. --include-untracked stashes both modifications
# AND untracked files; restore via `git stash list` on the VM if
# anything operator-meaningful was caught.
if [ -n "$(git status --porcelain)" ]; then
  warn "Local changes detected; stashing (incl. untracked) before pull."
  git stash push --include-untracked -m "auto-stash before deploy_remote $(date -Iseconds)" >/dev/null
fi

# GitHub answers anonymous git-upload-pack POSTs from cloud IP ranges
# with 401 even for PUBLIC repos (CI run 33670930981, 2026-09-02: the VM's
# GET info/refs returned 200 but the POST returned 401 + www-authenticate,
# while the identical anonymous fetch succeeded from a residential IP).
# The deploying account has no repo-admin rights, so a deploy key is not
# an option; instead reuse the ephemeral GITHUB_TOKEN CI already hands us
# for the ghcr login. The header lives only in this process's `-c`
# config — nothing is written to .git/config or ~/.git-credentials.
GIT_AUTH=()
if [ -n "${GHCR_TOKEN:-}" ]; then
  GIT_AUTH=(-c "http.https://github.com/.extraheader=AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GHCR_TOKEN" | base64 | tr -d '\n')")
fi

log "Fetching $REMOTE/$BRANCH..."
git ${GIT_AUTH[@]+"${GIT_AUTH[@]}"} fetch --quiet "$REMOTE" "$BRANCH"
NEW_SHA="$(git rev-parse "$REMOTE/$BRANCH")"

if [ "$OLD_SHA" = "$NEW_SHA" ] && [ "$FORCE_LOCAL_BUILD" != "1" ]; then
  log "Already up to date at $OLD_SHA. Will still pull/swap in case IMAGE_TAG changed."
else
  git ${GIT_AUTH[@]+"${GIT_AUTH[@]}"} pull --ff-only "$REMOTE" "$BRANCH"
  log "Now at: $NEW_SHA"
fi

# 2. Login to ghcr.io (only if using prebuilt images).
if [ "$FORCE_LOCAL_BUILD" != "1" ]; then
  if [ -z "${GHCR_USER:-}" ] || [ -z "${GHCR_TOKEN:-}" ]; then
    warn "GHCR_USER / GHCR_TOKEN not set; assuming docker is already logged in."
  else
    log "Logging into ghcr.io as $GHCR_USER..."
    echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
  fi
fi

# 3. Pull / rebuild.
export IMAGE_TAG
if [ "$FORCE_LOCAL_BUILD" = "1" ]; then
  log "FORCE_LOCAL_BUILD=1 — rebuilding all images on the VM"
  "${COMPOSE[@]}" build
else
  log "Pulling images for tag=$IMAGE_TAG..."
  "${COMPOSE[@]}" pull
fi

# 4. Up.
log "Bringing services up with IMAGE_TAG=$IMAGE_TAG..."
"${COMPOSE[@]}" up -d --remove-orphans

# 5. Health check.
log "Verifying API health..."
for _ in $(seq 1 30); do
  if "${COMPOSE[@]}" exec -T api curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    log "API healthy ✓"
    log "Deploy complete: $OLD_SHA → $NEW_SHA (image tag: $IMAGE_TAG)"
    exit 0
  fi
  sleep 2
done

err "API failed to become healthy after 60s. Recent logs:"
"${COMPOSE[@]}" logs api --tail=40
exit 1
