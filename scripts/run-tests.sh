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

case "$MODE" in
  unit)
    info "Running unit tests..."
    pytest services/ -m "unit or not (integration or e2e or slow)" \
      --cov=services --cov-report=term-missing --cov-report=html:htmlcov \
      -x "$@"
    ;;

  integration)
    info "Running integration tests (requires Docker services)..."
    # Ensure DB + Redis are up
    docker-compose up -d postgres redis
    until docker-compose exec -T postgres pg_isready -q; do sleep 1; done
    pytest services/ -m "integration" --tb=short "$@"
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
    pytest services/ tests/e2e/ \
      --cov=services --cov-report=term-missing --cov-report=html:htmlcov \
      --tb=short "$@"
    ;;

  *)
    echo "Usage: $0 [unit|integration|e2e|monitoring|all] [pytest-args...]"
    exit 1
    ;;
esac

echo ""
echo -e "${GREEN}Tests complete.${NC}"
