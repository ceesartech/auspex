# Betting System

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Node](https://img.shields.io/badge/node-18%2B-green.svg)
![License](https://img.shields.io/badge/license-None-green.svg)

An enterprise-grade sports betting and lottery recommendation system built with machine learning, real-time data pipelines, and a full observability stack.

---

## Table of Contents

1. [What This System Does](#what-this-system-does)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Required Accounts & API Keys](#required-accounts--api-keys)
5. [Local Development — Quick Start](#local-development--quick-start)
6. [Running the UI Locally](#running-the-ui-locally)
7. [Environment Variables Reference](#environment-variables-reference)
8. [Testing](#testing)
9. [Production Deployment (GCP + Kubernetes)](#production-deployment-gcp--kubernetes)
10. [Monitoring](#monitoring)
11. [Project Structure](#project-structure)
12. [Documentation Index](#documentation-index)
13. [Disclaimer](#disclaimer)

---

## What This System Does

| Layer | Technology | What it does |
|---|---|---|
| Data ingestion | Airflow + custom scrapers | Scrapes odds, stats, results from Bet365, BetMGM, FBref, Understat, ESPN, NHL API, Transfermarkt |
| Feature engineering | Python / pandas | Computes 250+ features per match (team form, H2H, player metrics, weather, referee, odds movement) |
| ML models | XGBoost, LightGBM, Neural Net, Poisson, Dixon-Coles, Ensemble | Generates win/draw/loss probabilities with SHAP explainability |
| Betting strategy | Kelly Criterion | Sizes stakes optimally based on edge over bookmaker |
| API | FastAPI + Celery + WebSocket | Serves predictions, recommendations, real-time odds |
| Frontend | Next.js 14 + TypeScript | Dashboard for viewing predictions, tracking ROI, building accumulators |
| Infrastructure | GKE + Terraform + Helm | Auto-scaling Kubernetes cluster on GCP |
| Monitoring | Prometheus + Grafana + Loki + Alertmanager | Full observability with automated model retraining |

---

## Architecture Overview

```
Browser / Mobile
      │
      ▼
┌─────────────┐    WebSocket    ┌──────────────────┐
│  Next.js UI │◄───────────────►│   FastAPI + WS   │
└─────────────┘                 └────────┬─────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
             ┌─────────────┐   ┌─────────────────┐  ┌──────────────┐
             │ ML Ensemble │   │ Feature Engine  │  │  Redis Cache │
             │ (5 models)  │   │  (250+ features)│  │  + Celery    │
             └──────┬──────┘   └────────┬────────┘  └──────────────┘
                    │                   │
                    └──────────┬────────┘
                               ▼
                    ┌──────────────────────┐
                    │  PostgreSQL (30+ tbls)│
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Airflow + Scrapers   │
                    │  (live + historical)  │
                    └──────────────────────┘

Observability (runs alongside everything):
  Prometheus → Grafana → Alertmanager → Email/Telegram
  Loki (logs) ← Promtail (pod log shipper)
  MLflow (model experiments)
```

---

## Prerequisites

| Tool | Minimum version | Install |
|---|---|---|
| Docker | 24+ | https://docs.docker.com/get-docker/ |
| Docker Compose | v2+ | included with Docker Desktop |
| Python | 3.11+ | https://www.python.org/downloads/ |
| Node.js | 18+ | https://nodejs.org/ |
| `kubectl` | 1.28+ | for K8s deployment only |
| `gcloud` CLI | latest | for GCP deployment only |
| Terraform | 1.6+ | for infrastructure provisioning only |

---

## Required Accounts & API Keys

### Mandatory for any environment

| Secret | Where to get it | `.env` key |
|---|---|---|
| PostgreSQL password | Set yourself (strong password) | `POSTGRES_PASSWORD` |
| JWT secret (≥32 chars) | `python3 -c "import secrets; print(secrets.token_hex(32))"` | `JWT_SECRET` |
| Airflow Fernet key | `python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` | `AIRFLOW__CORE__FERNET_KEY` |
| Airflow webserver secret | `python3 -c "import secrets; print(secrets.token_hex(16))"` | `AIRFLOW__WEBSERVER__SECRET_KEY` |

> `./scripts/setup.sh` generates all of these automatically.

### Required for cloud deployment (GCP)

| Secret | Where to get it | `.env` key |
|---|---|---|
| GCP Project ID | [console.cloud.google.com](https://console.cloud.google.com) → select project | `GCP_PROJECT_ID` |
| GCP Region | Choose e.g. `us-central1` | `GCP_REGION` |
| Service account JSON | IAM → Service Accounts → Create → download JSON | `GOOGLE_APPLICATION_CREDENTIALS` |
| GCS bucket name | Cloud Storage → Create bucket | `GCS_BUCKET` |

**Required GCP APIs to enable:**
```bash
gcloud services enable container.googleapis.com \
  sqladmin.googleapis.com \
  redis.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com
```

**Required IAM roles for the service account:**
- Kubernetes Engine Admin
- Cloud SQL Client
- Storage Object Admin
- Artifact Registry Writer

### Optional integrations

| Integration | Purpose | Keys needed |
|---|---|---|
| Telegram | Critical alert notifications | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
| SMTP (Gmail) | Email alerts from Alertmanager | `SMTP_PASSWORD` (App Password, not account password) |
| Supabase | Alternative hosted PostgreSQL | `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY` |
| Rotating proxies | Avoid scraper IP bans | `ROTATING_PROXY_URL`, `PROXY_LIST_URL` |

**How to get a Telegram bot token:**
1. Message `@BotFather` on Telegram → `/newbot`
2. Copy the token it gives you → `TELEGRAM_BOT_TOKEN`
3. Add the bot to a group/channel → get the chat ID via `https://api.telegram.org/bot<TOKEN>/getUpdates`

**How to get a Gmail App Password (for SMTP alerts):**
1. Google Account → Security → 2-Step Verification → App passwords
2. Generate a password for "Mail" → copy it → `SMTP_PASSWORD`

---

## Local Development — Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/betting-system.git
cd betting-system

# 2. One-command setup (creates .env, venv, installs deps, starts DB, runs migrations, seeds data)
./scripts/setup.sh

# 3. Start everything
./scripts/start-local.sh
```

That's it. Services will be available at:

| Service | URL | Credentials |
|---|---|---|
| **Frontend** | http://localhost:3001 | demo / demo1234 |
| **API** | http://localhost:8000 | JWT via `/api/v1/auth/login` |
| **API docs** (Swagger) | http://localhost:8000/docs | — |
| **Grafana** | http://localhost:3000 | admin / `$GRAFANA_PASSWORD` |
| **Prometheus** | http://localhost:9090 | — |
| **MLflow** | http://localhost:5000 | — |
| **Airflow** | http://localhost:8080 | admin / admin |

### Stop everything

```bash
./scripts/teardown.sh            # stop containers, keep data
./scripts/teardown.sh --volumes  # stop + delete all data
./scripts/teardown.sh --all      # full wipe (volumes + venv + node_modules)
```

### Manual step-by-step (if setup.sh fails)

```bash
cp .env.example .env
# Edit .env — fill in generated secrets (see setup.sh for commands)

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

docker-compose up -d postgres redis prometheus grafana mlflow
# Wait ~10 s for postgres to initialise
python3 scripts/seed_dev_data.py

cd services/frontend && npm install && cd ../..
```

---

## Running the UI Locally

The Next.js frontend connects to the local API. It runs on port **3001** (to avoid conflict with Grafana on 3000).

```bash
# Option A — via start-local.sh (automatic)
./scripts/start-local.sh

# Option B — manually
cd services/frontend
NEXT_PUBLIC_API_URL=http://localhost:8000 \
NEXT_PUBLIC_WS_URL=ws://localhost:8000 \
npm run dev -- --port 3001
```

**What you'll see:**
- **Dashboard** — upcoming matches, model accuracy, recent performance
- **Predictions** — filterable list with confidence, probabilities, SHAP explanation
- **Recommendations** — Kelly-sized bets ranked by expected value
- **Accumulator builder** — multi-leg bet optimizer
- **Analytics** — ROI trend charts, win-rate by confidence band, calibration plots

**Hot reload** is enabled — any change to `services/frontend/src/` is reflected immediately.

---

## Environment Variables Reference

Copy `.env.example` to `.env` and fill in values. The table below documents every variable.

### Core infrastructure

| Variable | Default | Required | Description |
|---|---|---|---|
| `POSTGRES_HOST` | `localhost` | Yes | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | Yes | PostgreSQL port |
| `POSTGRES_USER` | `betting_user` | Yes | Database username |
| `POSTGRES_PASSWORD` | *(set this)* | **Yes** | Strong password |
| `POSTGRES_DB` | `betting_system` | Yes | Database name |
| `DATABASE_URL` | derived | Yes | Full connection string |
| `REDIS_HOST` | `localhost` | Yes | Redis host |
| `REDIS_PORT` | `6379` | Yes | Redis port |
| `REDIS_PASSWORD` | *(empty)* | No | Redis auth (set in prod) |
| `REDIS_URL` | derived | Yes | Full Redis URL |

### Security

| Variable | Default | Required | Description |
|---|---|---|---|
| `JWT_SECRET` | *(generate)* | **Yes** | Signs JWT tokens — min 32 chars |
| `JWT_ALGORITHM` | `HS256` | Yes | JWT algorithm |
| `JWT_EXPIRATION_HOURS` | `24` | Yes | Token TTL |
| `SECRET_KEY` | *(generate)* | **Yes** | General Flask/FastAPI secret |
| `AIRFLOW__CORE__FERNET_KEY` | *(generate)* | **Yes** | Encrypts Airflow connection passwords |
| `AIRFLOW__WEBSERVER__SECRET_KEY` | *(generate)* | **Yes** | Signs Airflow session cookies |

### GCP / Cloud

| Variable | Default | Required | Description |
|---|---|---|---|
| `GCP_PROJECT_ID` | — | Cloud only | Your GCP project ID |
| `GCP_REGION` | `us-central1` | Cloud only | Deployment region |
| `GCP_ZONE` | `us-central1-a` | Cloud only | Primary zone |
| `GCS_BUCKET` | — | Cloud only | GCS bucket for model artifacts |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | Cloud only | Path to service-account JSON |

### ML / MLflow

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `MLFLOW_BACKEND_STORE_URI` | derived | Metadata store (Postgres) |
| `MLFLOW_ARTIFACT_ROOT` | `/mlflow/artifacts` | Model artifact storage |

### Monitoring & Alerts

| Variable | Default | Description |
|---|---|---|
| `GRAFANA_PASSWORD` | `admin_change_me` | Grafana admin password |
| `PROMETHEUS_PORT` | `9090` | Prometheus port |
| `GRAFANA_PORT` | `3000` | Grafana port |
| `TELEGRAM_BOT_TOKEN` | — | Bot token for critical alerts |
| `TELEGRAM_CHAT_ID` | — | Target chat/group ID |
| `SMTP_PASSWORD` | — | Gmail App Password for email alerts |

### User preferences

| Variable | Default | Description |
|---|---|---|
| `USER_DOB` | — | Date of birth (for age-gating) |
| `USER_LOCATION` | — | State/country (for legal compliance checks) |
| `USER_TIMEZONE` | `America/Denver` | Display timezone |

### Feature flags

| Variable | Default | Description |
|---|---|---|
| `ENABLE_WEB_SCRAPING` | `true` | Toggle live scraping |
| `ENABLE_MODEL_TRAINING` | `true` | Toggle training jobs |
| `ENABLE_PREDICTIONS` | `true` | Toggle prediction generation |
| `ENABLE_TELEGRAM_NOTIFICATIONS` | `false` | Enable Telegram alerts |
| `USE_PROXIES` | `false` | Route scrapers through rotating proxies |

---

## Testing

### Unit tests (no services required)

```bash
./scripts/run-tests.sh unit
# or
pytest services/ -m "unit" -x
```

### Integration tests (requires Docker)

```bash
./scripts/run-tests.sh integration
```

### End-to-end tests (requires full stack running)

```bash
# Start the stack first
./scripts/start-local.sh --no-frontend &

# Seed data
python3 scripts/seed_dev_data.py

# Run e2e suite
./scripts/run-tests.sh e2e
```

E2E tests cover:
- API health and OpenAPI schema
- Auth flow (login, JWT validation, rejection of bad tokens)
- Predictions endpoint (schema, pagination, probability integrity)
- Recommendations and bet recording
- Matches endpoint
- Data pipeline integrity (DB consistency checks)
- WebSocket connection
- Monitoring stack (Prometheus, Grafana)

### All tests

```bash
./scripts/run-tests.sh all
```

---

## Production Deployment (GCP + Kubernetes)

Full instructions in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Summary:

```bash
# 1. Provision cloud infrastructure
cd infrastructure/terraform/environments/prod
terraform init
terraform apply -var="project_id=$GCP_PROJECT_ID"

# 2. Authenticate kubectl
gcloud container clusters get-credentials betting-system-cluster \
  --region $GCP_REGION --project $GCP_PROJECT_ID

# 3. Deploy application
kubectl apply -k infrastructure/kubernetes/overlays/prod/

# 4. Deploy monitoring stack
./monitoring/scripts/deploy-monitoring.sh

# 5. Validate
python3 monitoring/scripts/validate-deployment.py
```

---

## Monitoring

After `deploy-monitoring.sh` (or `docker-compose up`):

| Dashboard | URL | What it shows |
|---|---|---|
| ML Model Performance | Grafana → Betting System → ML Model Performance | Accuracy, ROI, drift score, calibration |
| API Performance | Grafana → API Performance | Latency (P50/95/99), error rate, throughput |
| Infrastructure | Grafana → Infrastructure Overview | CPU, memory, pod restarts, HPA |
| Business Metrics | Grafana → Business Metrics | Win rate, ROI trend, daily predictions |
| Scraping Status | Grafana → Scraping Status | Records/hour, error rate, odds freshness |

**Automated retraining:**

| Job | Schedule | Trigger |
|---|---|---|
| `model-retraining` | Weekly (Sun 02:00 UTC) | Scheduled |
| `drift-check` | Daily 06:00 UTC | Auto (fires retraining if drift_score > 0.3) |
| `performance-check` | Every 6 hours | Pushes Prometheus metrics |

**Manual trigger:**
```bash
python3 monitoring/scripts/trigger-retraining.py --reason "Manual: pre-season update"
```

---

## Project Structure

```
betting-system/
├── services/
│   ├── data-ingestion/         # Scrapers + Airflow DAGs
│   │   ├── src/scrapers/       # Bet365, BetMGM, FBref, Understat, ...
│   │   ├── dags/               # Airflow pipeline definitions
│   │   └── db/migrations/      # SQL schema migrations
│   ├── feature-engineering/    # 250+ feature computation
│   │   └── src/categories/     # team_performance, h2h, player_metrics, ...
│   ├── ml-models/              # Model training + inference
│   │   └── src/models/         # xgboost, neural_network, poisson, ensemble
│   ├── api/                    # FastAPI REST + WebSocket
│   │   └── src/                # routes, auth, tasks, middleware
│   └── frontend/               # Next.js 14 dashboard
│       └── src/                # app/, components/, lib/
├── monitoring/
│   ├── prometheus/             # Scrape config + alert rules + recording rules
│   ├── grafana/dashboards/     # 5 pre-built dashboards
│   ├── alertmanager/           # Routing + Telegram/email config
│   ├── loki/                   # Log aggregation config
│   ├── model-monitoring/       # Performance tracker, drift detector, auto-retrainer
│   ├── scripts/                # setup, teardown, validate, trigger-retraining
│   └── runbooks/               # API_DOWN, MODEL_DRIFT, HIGH_ERROR_RATE, DATABASE_ISSUES
├── infrastructure/
│   ├── kubernetes/             # Kustomize base + overlays (dev/staging/prod)
│   ├── terraform/              # GKE, Cloud SQL, Memorystore, GCS modules
│   └── helm/betting-system/    # Helm chart
├── tests/
│   └── e2e/                    # End-to-end integration tests
├── scripts/
│   ├── setup.sh                # One-command local setup
│   ├── teardown.sh             # Stop + optionally wipe resources
│   ├── start-local.sh          # Start all services for development
│   ├── run-tests.sh            # Test runner (unit/integration/e2e/all)
│   └── seed_dev_data.py        # Populate DB with realistic dev data
├── docker-compose.yml          # Local development stack
└── .env.example                # Template for environment configuration
```

---

## Documentation Index

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, data flow, technology choices |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Step-by-step local and cloud deployment |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | Dev workflow, adding scrapers/models, code style |
| [`docs/API.md`](docs/API.md) | Full REST API reference with request/response examples |
| [`monitoring/runbooks/API_DOWN.md`](monitoring/runbooks/API_DOWN.md) | Runbook: API down |
| [`monitoring/runbooks/MODEL_DRIFT.md`](monitoring/runbooks/MODEL_DRIFT.md) | Runbook: model drift / accuracy drop |
| [`monitoring/runbooks/HIGH_ERROR_RATE.md`](monitoring/runbooks/HIGH_ERROR_RATE.md) | Runbook: high 5xx error rate |
| [`monitoring/runbooks/DATABASE_ISSUES.md`](monitoring/runbooks/DATABASE_ISSUES.md) | Runbook: database connectivity / performance |

---

## Disclaimer

This system is for **personal and educational use only**. Sports betting involves substantial financial risk. Only risk money you can afford to lose completely. Always verify that sports betting is legal in your jurisdiction before use. The authors accept no liability for financial losses incurred through use of this software.
