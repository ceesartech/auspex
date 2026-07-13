#!/usr/bin/env bash
# run-tests.sh — Run the test suite at any level
# Usage: ./scripts/run-tests.sh [unit|integration|e2e|all] [extra pytest args]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-unit}"
shift || true

BLUE='\033[0;34m'; GREEN='\033[0;32m'; NC='\033[0m'
info(){ echo -e "${BLUE}[TEST]${NC} $*"; }

[[ -f venv/bin/activate ]] && source venv/bin/activate

PYTHON_SERVICES=(api data-ingestion feature-engineering ml-models)

run_service_pytest() {
  local marker="$1"
  shift

  for svc in "${PYTHON_SERVICES[@]}"; do
    info "Running ${svc} tests..."
    rc=0
    (
      cd "$ROOT/services/$svc"
      PYTHONPATH=src pytest tests -m "$marker" --rootdir=. "$@"
    ) || rc=$?

    if [[ "$rc" -ne 0 ]]; then
      if [[ "$marker" == "integration" && "$rc" -eq 5 ]]; then
        info "No integration tests matched for ${svc}; continuing."
      else
        exit "$rc"
      fi
    fi
  done
}

case "$MODE" in
  unit)
    info "Running unit tests..."
    run_service_pytest "not (integration or e2e or slow)" \
      --cov=src --cov-report=term-missing -x "$@"
    # Repo-root tests/unit covers the scripts/ layer (EV/Kelly math,
    # calibration metrics, graders). No DB/Redis needed.
    info "Running repo-root tests/unit..."
    pytest tests/unit -q --rootdir=. "$@"
    ;;

  integration)
    info "Running integration tests (requires Docker services)..."
    # Ensure DB + Redis are up
    docker-compose up -d postgres redis
    until docker-compose exec -T postgres pg_isready -q; do sleep 1; done
    run_service_pytest "integration" --tb=short "$@"
    ;;

  e2e)
    info "Running end-to-end tests (requires full stack on localhost)..."
    info "Make sure './scripts/start-local.sh' is running first."
    pytest tests/e2e/ -m "e2e or not unit" --tb=short -v "$@"
    ;;

  monitoring)
    info "Running monitoring e2e tests..."
    pytest tests/e2e/test_monitoring.py -v "$@"
    ;;

  all)
    info "Running all tests (unit + integration + e2e)..."
    docker-compose up -d postgres redis
    until docker-compose exec -T postgres pg_isready -q; do sleep 1; done
    run_service_pytest "not e2e" \
      --cov=src --cov-report=term-missing --tb=short "$@"
    pytest tests/e2e/ --tb=short "$@"
    ;;

  *)
    echo "Usage: $0 [unit|integration|e2e|monitoring|all] [pytest-args...]"
    exit 1
    ;;
esac

echo ""
echo -e "${GREEN}Tests complete.${NC}"
