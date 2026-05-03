.PHONY: help setup test lint format docker-build docker-up docker-down clean db-migrate db-reset \
	tf-init tf-plan tf-apply tf-destroy k8s-deploy k8s-status backup restore rollback health-check

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
	pytest services/ --cov=services --cov-report=html --cov-report=term -v

test-unit:
	pytest services/ -m "not integration" --cov=services --cov-report=term -v

test-integration:
	pytest services/ -m integration -v

lint:
	flake8 services/ --max-line-length=120 --extend-ignore=E203,W503
	black --check services/ --line-length=120
	isort --check-only services/ --profile black

format:
	black services/ --line-length=120
	isort services/ --profile black

type-check:
	mypy services/ --ignore-missing-imports

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
