.PHONY: help setup test lint format docker-build docker-up docker-down clean db-migrate db-reset \
	tf-init tf-plan tf-apply tf-destroy k8s-deploy k8s-status backup restore rollback health-check

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
	@echo "Infrastructure commands:"
	@echo "  make tf-init ENV=dev     - Initialize Terraform for environment"
	@echo "  make tf-plan ENV=dev     - Plan Terraform changes"
	@echo "  make tf-apply ENV=dev    - Apply Terraform changes"
	@echo "  make tf-destroy ENV=dev  - Destroy Terraform resources"
	@echo "  make k8s-deploy ENV=dev  - Deploy to Kubernetes environment"
	@echo "  make k8s-status          - Check Kubernetes status"
	@echo "  make backup              - Create database backup"
	@echo "  make restore FILE=<path> - Restore database from backup"
	@echo "  make rollback SVC=betting-api - Rollback a deployment"
	@echo "  make health-check        - Run health checks"

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

# Infrastructure targets
ENV ?= dev

tf-init:
	cd infrastructure/terraform/environments/$(ENV) && terraform init

tf-plan:
	cd infrastructure/terraform/environments/$(ENV) && terraform plan

tf-apply:
	cd infrastructure/terraform/environments/$(ENV) && terraform apply

tf-destroy:
	cd infrastructure/terraform/environments/$(ENV) && terraform destroy

k8s-deploy:
	kustomize build infrastructure/kubernetes/overlays/$(ENV) | kubectl apply -f -

k8s-status:
	kubectl get all -n betting-system

backup:
	./infrastructure/scripts/backup-db.sh

restore:
	./infrastructure/scripts/restore-db.sh $(FILE)

rollback:
	./infrastructure/scripts/rollback.sh $(SVC)

health-check:
	./infrastructure/scripts/health-check.sh
