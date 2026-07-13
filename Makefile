.PHONY: help setup test lint format docker-build docker-up docker-down clean db-migrate db-reset \
	backup restore

PYTHON_SERVICES := api data-ingestion feature-engineering ml-models
PYTHON_SERVICE_DIRS := $(addprefix services/,$(PYTHON_SERVICES))
# Baseline mypy gate. Tighten disabled error codes as services get stronger annotations.
MYPY_FLAGS := --ignore-missing-imports --implicit-optional \
	--disable-error-code=arg-type \
	--disable-error-code=assignment \
	--disable-error-code=attr-defined \
	--disable-error-code=import-untyped \
	--disable-error-code=misc \
	--disable-error-code=operator \
	--disable-error-code=return-value \
	--disable-error-code=truthy-function \
	--disable-error-code=union-attr \
	--disable-error-code=valid-type \
	--disable-error-code=var-annotated

help:
	@echo "Available commands:"
	@echo "  make setup          - Set up development environment"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make lint           - Run linters (flake8, black, isort)"
	@echo "  make format         - Format code (black, isort)"
	@echo "  make type-check     - Run type checking (mypy)"
	@echo "  make docker-build   - Build all Docker images"
	@echo "  make docker-up      - Start all services"
	@echo "  make docker-down    - Stop all services"
	@echo "  make docker-logs    - View logs from all services"
	@echo "  make db-migrate     - Run database migrations"
	@echo "  make db-reset       - Reset database (WARNING: deletes all data)"
	@echo "  make clean          - Clean temporary files"
	@echo ""
	@echo "Operations commands (single-VM docker-compose; see OPERATIONS.md):"
	@echo "  make backup              - Create a Postgres backup (scripts/backup_postgres.py)"
	@echo "  make restore FILE=<path> - Restore Postgres from a .dump (see OPERATIONS.md)"

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -r requirements-dev.txt
	cp .env.example .env
	@echo "Setup complete! Activate virtualenv with: source venv/bin/activate"

test:
	@set -e; for svc in $(PYTHON_SERVICES); do \
		echo "==> pytest services/$$svc"; \
		( cd "services/$$svc" && PYTHONPATH=src pytest tests --rootdir=. --cov=src --cov-report=term -v ); \
	done
	@echo "==> pytest tests/unit (repo root)"
	@pytest tests/unit -q --rootdir=.

test-unit:
	@set -e; for svc in $(PYTHON_SERVICES); do \
		echo "==> pytest services/$$svc (unit)"; \
		( cd "services/$$svc" && PYTHONPATH=src pytest tests -m "not integration" --rootdir=. --cov=src --cov-report=term -v ); \
	done

test-integration:
	@set -e; for svc in $(PYTHON_SERVICES); do \
		echo "==> pytest services/$$svc (integration)"; \
		rc=0; \
		( cd "services/$$svc" && PYTHONPATH=src pytest tests -m integration --rootdir=. -v ) || rc=$$?; \
		if [ "$$rc" -ne 0 ] && [ "$$rc" -ne 5 ]; then exit "$$rc"; fi; \
	done

lint:
	flake8 $(PYTHON_SERVICE_DIRS) --max-line-length=120 --extend-ignore=E203,W503
	black --check $(PYTHON_SERVICE_DIRS) --line-length=120
	isort --check-only $(PYTHON_SERVICE_DIRS) --profile black

format:
	black $(PYTHON_SERVICE_DIRS) --line-length=120
	isort $(PYTHON_SERVICE_DIRS) --profile black

type-check:
	@set -e; for svc in $(PYTHON_SERVICES); do \
		echo "==> mypy services/$$svc"; \
		( cd "services/$$svc" && PYTHONPATH=src python -m mypy src $(MYPY_FLAGS) ); \
	done

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

db-migrate:
	docker-compose exec postgres psql -U betting_user -d betting_system -f /docker-entrypoint-initdb.d/001_create_schema.sql
	docker-compose exec postgres psql -U betting_user -d betting_system -f /docker-entrypoint-initdb.d/002_create_indexes.sql
	docker-compose exec postgres psql -U betting_user -d betting_system -f /docker-entrypoint-initdb.d/003_create_functions.sql
	docker-compose exec postgres psql -U betting_user -d betting_system -f /docker-entrypoint-initdb.d/004_create_users.sql
	docker-compose exec postgres psql -U betting_user -d betting_system -f /docker-entrypoint-initdb.d/005_predictions_unique_constraint.sql
	docker-compose exec postgres psql -U betting_user -d betting_system -f /docker-entrypoint-initdb.d/006_nhl_match_stats.sql
	docker-compose exec postgres psql -U betting_user -d betting_system -f /docker-entrypoint-initdb.d/007_market_types_expansion.sql
	docker-compose exec postgres psql -U betting_user -d betting_system -f /docker-entrypoint-initdb.d/008_nhl_goalie_features.sql
	docker-compose exec postgres psql -U betting_user -d betting_system -f /docker-entrypoint-initdb.d/009_nhl_5v5_advanced_stats.sql
	docker-compose exec postgres psql -U betting_user -d betting_system -f /docker-entrypoint-initdb.d/010_lottery_predictions.sql

db-reset:
	@echo "WARNING: This will delete all data. Press Ctrl+C to cancel."
	@sleep 5
	docker-compose down -v
	docker-compose up -d postgres redis
	@sleep 5
	$(MAKE) db-migrate

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/ dist/ build/ *.egg-info

# Operations targets (single Hetzner VM + docker-compose; see OPERATIONS.md).
# The GKE-era tf-*/k8s-*/rollback/health-check targets were removed with the
# infrastructure/{terraform,kubernetes,helm} dirs (2026-07 audit §5.2).
backup:
	docker compose exec -T api python /app/scripts/backup_postgres.py

restore:
	@echo "Restore is a guarded procedure — follow OPERATIONS.md (pg_restore --clean"
	@echo "--if-exists from a .dump). FILE=$(FILE)"
