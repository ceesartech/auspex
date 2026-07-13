# Development Guide

## First-time Setup

```bash
git clone https://github.com/yourusername/betting-system.git
cd betting-system
./scripts/setup.sh        # creates .env, venv, installs deps, starts DB, seeds data
./scripts/start-local.sh  # starts all services including frontend on :3001
```

For a fresh wipe at any time:

```bash
./scripts/teardown.sh --all  # remove containers, volumes, venv, node_modules
./scripts/setup.sh           # start fresh
```

---

## Prerequisites

| Tool | Version | Install command |
|---|---|---|
| Python | 3.11+ | `pyenv install 3.11` or https://python.org |
| Node.js | 18+ | `nvm install 18` or https://nodejs.org |
| Docker Desktop | 24+ | https://docs.docker.com/get-docker/ |
| psql client | any | `brew install postgresql` (macOS) |

---

## Project Layout

```
services/
├── data-ingestion/
│   ├── src/
│   │   ├── core/           config.py, database.py
│   │   ├── scrapers/       one file per data source
│   │   ├── validators/     data_validator.py
│   │   ├── transformers/   data_transformer.py
│   │   └── utils/          proxy_manager.py, retry_logic.py
│   ├── dags/               Airflow DAG definitions
│   ├── db/migrations/      001_create_schema.sql, 002_indexes, 003_functions
│   └── tests/
├── feature-engineering/
│   └── src/
│       ├── categories/     base, team_performance, head_to_head, …
│       ├── core/           registry.py, cache.py
│       ├── utils/          math_helpers.py, sql_queries.py
│       ├── orchestrator.py central feature computation entry point
│       └── validator.py
├── ml-models/
│   └── src/
│       ├── models/         xgboost, lightgbm, neural_network, poisson, ensemble
│       ├── training/       calibration.py
│       ├── inference/      onnx_converter.py
│       ├── explainability/ shap_explainer.py
│       └── utils/          data_loader.py, feature_selector.py
├── api/
│   └── src/
│       ├── routes/         predictions, recommendations, matches, models, …
│       ├── auth/           JWT handlers
│       ├── middleware/     metrics.py, rate_limit.py
│       ├── services/       cache_service.py, prediction_service.py
│       └── tasks/          Celery async tasks
└── frontend/
    └── src/
        ├── app/            Next.js App Router pages
        ├── components/     ui/, dashboard/, predictions/, charts/, …
        └── lib/
            ├── api/        typed API clients
            ├── hooks/      use-predictions, use-websocket, use-auth
            └── store/      Zustand stores
```

---

## Daily Development Workflow

```bash
# Activate Python env
source venv/bin/activate

# Start just the infrastructure (no Airflow, no frontend)
docker-compose up -d postgres redis prometheus grafana

# Run the API with hot reload
uvicorn services.api.src.main:app --reload --port 8000

# In another terminal — run the frontend
cd services/frontend && npm run dev -- --port 3001

# Run tests as you work
./scripts/run-tests.sh unit        # fast, no services needed
./scripts/run-tests.sh integration # needs DB + Redis
```

---

## Adding a New Scraper

1. Create `services/data-ingestion/src/scrapers/my_source_scraper.py`

```python
from .base_scraper import BaseScraper

class MySourceScraper(BaseScraper):
    BASE_URL = "https://example.com"

    def scrape(self) -> list[dict]:
        response = self.session.get(self.BASE_URL + "/matches")
        return self.parse(response)

    def parse(self, response) -> list[dict]:
        # Return list of canonical match dicts
        return [...]

    def validate(self, records: list[dict]) -> list[dict]:
        return [r for r in records if self.validator.is_valid(r)]
```

2. Register it in `services/data-ingestion/src/core/config.py`

3. Create an Airflow DAG in `services/data-ingestion/dags/my_source_scraping.py`

4. Write tests in `services/data-ingestion/tests/unit/test_my_source_scraper.py`

---

## Adding a New Feature Category

1. Create `services/feature-engineering/src/categories/my_category.py`

```python
from .base import BaseFeatureCategory

class MyCategory(BaseFeatureCategory):
    category = "my_category"

    def compute(self, match_id: int, context: dict) -> dict[str, float]:
        return {
            "my_feature_1": ...,
            "my_feature_2": ...,
        }
```

2. Register in `services/feature-engineering/src/core/registry.py`

3. Add to the orchestrator's category list in `orchestrator.py`

4. Write tests in `services/feature-engineering/tests/unit/test_my_category.py`

---

## Training Models Locally

```bash
# Ensure you have seed data and features computed
python3 scripts/seed_dev_data.py

# Validate training data from local database
cd services/ml-models
PYTHONPATH=src python -m validation.validate_training_data

# Train all models
PYTHONPATH=src python -m training.train_all_models \
    --model-type all \
    --output-dir ../../models

# Or train a single model
PYTHONPATH=src python -m training.train_all_models \
    --model-type xgboost \
    --output-dir ../../models

# Evaluate the registered training results
PYTHONPATH=src python -m evaluation.evaluate_models \
    --model-dir ../../models \
    --output-file ../../evaluation_report.json
```

View experiments in MLflow at http://localhost:5000.

---

## Database Migrations

SQL migration files are in `services/data-ingestion/db/migrations/`, executed in filename order.

```bash
# Apply all pending migrations (local)
for f in services/data-ingestion/db/migrations/*.sql; do
  docker-compose exec -T postgres psql \
    -U betting_user -d betting_system -f /docker-entrypoint-initdb.d/$(basename $f)
done

# Create a new migration
touch services/data-ingestion/db/migrations/004_add_my_table.sql
```

---

## Testing

```bash
# Unit tests only (fast, no services required)
./scripts/run-tests.sh unit

# Integration tests (starts Docker services automatically)
./scripts/run-tests.sh integration

# End-to-end tests (requires ./scripts/start-local.sh running)
./scripts/run-tests.sh e2e

# All tests with coverage
./scripts/run-tests.sh all
open htmlcov/index.html
```

### Test structure

```
services/
  data-ingestion/tests/
    unit/     test_scrapers.py, test_validators.py
    integration/  test_scraping_pipeline.py
  feature-engineering/tests/
    unit/     test_temporal.py, test_team_performance.py, …
    integration/  test_feature_pipeline.py
  ml-models/tests/
    unit/     test_xgboost.py, test_ensemble.py, …
    integration/  test_model_pipeline.py
tests/e2e/
  test_api_health.py
  test_auth.py
  test_predictions.py
  test_recommendations.py
  test_matches.py
  test_data_pipeline.py
  test_websocket.py
  test_monitoring.py
```

---

## Code Style

| Tool | Config | Command |
|---|---|---|
| black | line-length 120 | `make lint` |
| isort | profile=black | `make lint` |
| flake8 | max-line=120 | `make lint` |
| mypy | baseline source check | `make type-check` |
| eslint | next/core-web-vitals | `cd services/frontend && npm run lint` |

Format everything:

```bash
make format
cd services/frontend && npm run lint -- --fix
```

---

## Frontend Development

The Next.js app talks to `NEXT_PUBLIC_API_URL` (default: `http://localhost:8000`).

```bash
cd services/frontend

# Development server with hot reload
NEXT_PUBLIC_API_URL=http://localhost:8000 \
NEXT_PUBLIC_WS_URL=ws://localhost:8000 \
npm run dev -- --port 3001

# Type checking
npm run type-check

# Build for production
npm run build

# Run frontend unit tests
npm test

# Run Playwright e2e tests (requires running stack)
npx playwright install chromium
npx playwright test
```

### Adding a new page

1. Create `services/frontend/src/app/(authenticated)/my-page/page.tsx`
2. Add API client methods in `services/frontend/src/lib/api/`
3. Add navigation link in `services/frontend/src/components/layout/sidebar.tsx`

---

## Useful Commands

```bash
# Tail API logs
docker-compose logs -f api

# Connect to the local database
docker-compose exec postgres psql -U betting_user betting_system

# Flush Redis cache
docker-compose exec redis redis-cli FLUSHDB

# Check Celery task queue
docker-compose exec api celery -A services.api.celery_app inspect active

# Manually trigger drift check
docker compose exec -T airflow-scheduler airflow dags trigger retrain_models

# Check model performance
python scripts/monitor_models.py --days 30

# Reload Prometheus config without restart
curl -X POST http://localhost:9090/-/reload
```
