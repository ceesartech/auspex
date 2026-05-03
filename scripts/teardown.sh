#!/usr/bin/env bash
# teardown.sh — Stop and optionally destroy all local resources
# Usage: ./scripts/teardown.sh [--volumes] [--all]
#   --volumes   Also delete Docker volumes (database data, Grafana data, etc.)
#   --all       Full wipe: volumes + venv + frontend node_modules + .env
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

YELLOW='\033[1;33m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info()   { echo -e "\033[0;34m[INFO]${NC} $*"; }
success(){ echo -e "${GREEN}[OK]${NC}   $*"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $*"; }

REMOVE_VOLUMES=false
FULL_WIPE=false
LOCAL_ONLY=false
for arg in "$@"; do
  case $arg in
    --volumes) REMOVE_VOLUMES=true ;;
    --all)     FULL_WIPE=true; REMOVE_VOLUMES=true ;;
    --local)   LOCAL_ONLY=true ;;
  esac
done

# ── Kill local (non-Docker) processes only ───────────────────────────────────
if $LOCAL_ONLY; then
  info "Killing local API, Celery, and frontend processes..."
  for pidfile in /tmp/betting-api.pid /tmp/betting-celery.pid /tmp/betting-frontend.pid; do
    if [[ -f "$pidfile" ]]; then
      PID=$(cat "$pidfile")
      kill "$PID" 2>/dev/null && success "Killed PID $PID ($(basename "$pidfile" .pid))" || true
      rm -f "$pidfile"
    fi
  done
  success "Local processes stopped. Docker infrastructure still running."
  echo "  To also stop Docker: ./scripts/teardown.sh"
  exit 0
fi

# ── Confirm destructive operations ───────────────────────────────────────────
if $REMOVE_VOLUMES; then
  echo -e "${RED}WARNING: This will delete all local data (database, Redis, Grafana).${NC}"
  read -r -p "Are you sure? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

# ── Stop all Docker Compose services ────────────────────────────────────────
info "Stopping Docker Compose services..."
if $REMOVE_VOLUMES; then
  docker-compose down -v --remove-orphans
  success "Services stopped and volumes removed"
else
  docker-compose down --remove-orphans
  success "Services stopped (data preserved)"
fi

# ── Full wipe extras ─────────────────────────────────────────────────────────
if $FULL_WIPE; then
  info "Removing Python virtual environment..."
  rm -rf venv
  success "venv removed"

  info "Removing frontend node_modules..."
  rm -rf services/frontend/node_modules services/frontend/.next
  success "node_modules removed"

  warn ".env not removed automatically. Delete it manually if required."
fi

echo ""
success "Teardown complete."
if ! $REMOVE_VOLUMES; then
  echo "  Data volumes preserved. Run './scripts/teardown.sh --volumes' to also delete data."
fi
