#!/usr/bin/env bash
# start-local.sh — Start all services for local development
# Usage: ./scripts/start-local.sh [--no-airflow] [--no-frontend] [--no-celery]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED='\033[0;31m'; BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*"; exit 1; }

NO_AIRFLOW=false
NO_FRONTEND=false
NO_CELERY=false
for arg in "$@"; do
  case $arg in
    --no-airflow)  NO_AIRFLOW=true ;;
    --no-frontend) NO_FRONTEND=true ;;
    --no-celery)   NO_CELERY=true ;;
  esac
done

[[ -f .env ]] || error "Missing .env — run ./scripts/setup.sh first"
[[ -d venv ]] || error "Python venv not found — run ./scripts/setup.sh first"

# Safe .env parser — handles unquoted values with spaces/commas
_load_env() {
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" != *=* ]] && continue
    local key="${line%%=*}"
    local value="${line#*=}"
    if [[ "$value" =~ ^\"(.*)\"$ ]]; then value="${BASH_REMATCH[1]}"
    elif [[ "$value" =~ ^\'(.*)\'$ ]]; then value="${BASH_REMATCH[1]}"; fi
    export "$key=$value"
  done < .env
}
_load_env

# Verify Docker is running (needed for infrastructure)
docker info &>/dev/null || error "Docker daemon is not running. Start Docker Desktop and retry."

# ── Infrastructure (Docker) ──────────────────────────────────────────────────
info "Starting infrastructure (postgres, redis, prometheus, grafana, mlflow)..."
docker-compose up -d postgres redis prometheus grafana mlflow

# Wait for Redis to be healthy before proceeding
info "Waiting for Redis..."
for i in $(seq 1 30); do
  docker-compose ps redis | grep -q "healthy" && break
  sleep 2
done
docker-compose ps redis | grep -q "healthy" || warn "Redis may not be healthy — check: docker-compose logs redis"

# Wait for Postgres
info "Waiting for Postgres..."
for i in $(seq 1 30); do
  docker-compose exec -T postgres pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" &>/dev/null && break
  sleep 2
done
success "Infrastructure ready"

# ── Airflow (Docker, optional) ───────────────────────────────────────────────
if ! $NO_AIRFLOW; then
  info "Starting Airflow (pass --no-airflow to skip)..."
  info "Note: first run builds the Airflow image — this can take several minutes."
  docker-compose up -d airflow-init airflow-webserver airflow-scheduler
  success "Airflow starting at http://localhost:8080 (admin/admin) — may take ~1 min to be ready"
fi

# ── API (run directly from venv — no Docker build) ───────────────────────────
# Stop any Docker api/celery-worker containers that may be holding the port.
info "Ensuring Docker API container is stopped (port ${API_PORT:-8000} must be free)..."
docker-compose stop api celery-worker 2>/dev/null || true

# Activate venv and run uvicorn in background. Logs go to /tmp/betting-api.log.
info "Starting API (uvicorn from venv)..."
source venv/bin/activate

# Kill any leftover uvicorn process from a previous run
if [[ -f /tmp/betting-api.pid ]] && kill -0 "$(cat /tmp/betting-api.pid)" 2>/dev/null; then
  kill "$(cat /tmp/betting-api.pid)" 2>/dev/null && info "Stopped previous API process"
  sleep 1
fi

MODEL_PATH="$ROOT/data/models" \
PYTHONPATH="$ROOT/services/api/src" \
  uvicorn main:app \
  --app-dir "$ROOT/services/api/src" \
  --host 0.0.0.0 \
  --port "${API_PORT:-8000}" \
  --reload \
  --log-level info \
  > /tmp/betting-api.log 2>&1 &
API_PID=$!
echo "$API_PID" > /tmp/betting-api.pid

# Wait up to 30 s for the API health endpoint
info "Waiting for API to respond..."
API_READY=false
for i in $(seq 1 15); do
  # Bail early if the process died
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo ""
    error "API process crashed. Last log output:
$(tail -30 /tmp/betting-api.log)"
  fi
  curl -sf "http://localhost:${API_PORT:-8000}/health" &>/dev/null && API_READY=true && break
  sleep 2
done

if $API_READY; then
  success "API ready at http://localhost:${API_PORT:-8000}"
else
  warn "API did not respond after 30 s. It may still be starting."
  warn "Tail logs: tail -f /tmp/betting-api.log"
fi

# ── Celery worker (run directly from venv) ───────────────────────────────────
if ! $NO_CELERY; then
  info "Starting Celery worker..."
  PYTHONPATH="$ROOT/services/api/src" celery -A celery_app worker \
    --loglevel=info \
    --concurrency=2 \
    > /tmp/betting-celery.log 2>&1 &
  CELERY_PID=$!
  echo "$CELERY_PID" > /tmp/betting-celery.pid
  # Give it 5 s to detect a crash
  sleep 5
  if kill -0 "$CELERY_PID" 2>/dev/null; then
    success "Celery worker running (PID $CELERY_PID)"
  else
    warn "Celery worker failed to start. Check /tmp/betting-celery.log"
    warn "Pass --no-celery to skip. Async tasks will be unavailable."
  fi
fi

# ── Frontend (Next.js dev server) ────────────────────────────────────────────
if ! $NO_FRONTEND; then
  info "Starting frontend (Next.js on port 3001)..."
  cd services/frontend
  NEXT_PUBLIC_API_URL="http://localhost:${API_PORT:-8000}" \
  NEXT_PUBLIC_WS_URL="ws://localhost:${API_PORT:-8000}" \
  npm run dev -- --port 3001 > /tmp/betting-frontend.log 2>&1 &
  FRONTEND_PID=$!
  echo "$FRONTEND_PID" > /tmp/betting-frontend.pid
  cd "$ROOT"
  # Wait up to 30 s for Next.js
  for i in $(seq 1 15); do
    curl -sf http://localhost:3001 &>/dev/null && break || sleep 2
  done
  if curl -sf http://localhost:3001 &>/dev/null; then
    success "Frontend ready at http://localhost:3001"
  else
    warn "Frontend not ready yet — may still be compiling. Tail: tail -f /tmp/betting-frontend.log"
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}========================================================${NC}"
echo -e "${GREEN}  Services running${NC}"
echo -e "${GREEN}========================================================${NC}"
echo ""
echo "  Frontend:   http://localhost:3001"
echo "  API:        http://localhost:${API_PORT:-8000}"
echo "  API docs:   http://localhost:${API_PORT:-8000}/docs"
echo "  Airflow:    http://localhost:8080  (admin/admin)"
echo "  Grafana:    http://localhost:${GRAFANA_PORT:-3000}  (admin/${GRAFANA_PASSWORD:-admin})"
echo "  MLflow:     http://localhost:5001"
echo "  Prometheus: http://localhost:${PROMETHEUS_PORT:-9090}"
echo ""
echo "  Logs:"
echo "    API:      tail -f /tmp/betting-api.log"
echo "    Celery:   tail -f /tmp/betting-celery.log"
echo "    Frontend: tail -f /tmp/betting-frontend.log"
echo ""
echo "  To stop all local processes:"
echo "    ./scripts/teardown.sh --local"
echo ""

# ── Keep running / tail API log ───────────────────────────────────────────────
_cleanup() {
  echo ""
  info "Shutting down local processes..."
  kill "$(cat /tmp/betting-api.pid 2>/dev/null)"      2>/dev/null || true
  kill "$(cat /tmp/betting-celery.pid 2>/dev/null)"   2>/dev/null || true
  kill "$(cat /tmp/betting-frontend.pid 2>/dev/null)" 2>/dev/null || true
  exit 0
}
trap _cleanup INT TERM

# Tail API logs so output is visible and Ctrl+C works
tail -f /tmp/betting-api.log
