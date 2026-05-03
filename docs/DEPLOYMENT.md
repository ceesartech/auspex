# Deployment Guide

## Environments

| Environment | Infrastructure | Purpose |
|---|---|---|
| Local | Docker Compose | Development and testing |
| Dev | GKE (single-node) | CI preview deploys |
| Staging | GKE (small cluster) | Pre-production validation |
| Production | GKE + Cloud SQL + Memorystore | Live system |

---

## Local Deployment

### One-command setup

```bash
./scripts/setup.sh          # First time only
./scripts/start-local.sh    # Every subsequent run
```

### Manual setup

```bash
# 1. Copy and configure env
cp .env.example .env
# Edit .env — generate required secrets:
#   Fernet key:  python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
#   JWT secret:  python3 -c "import secrets; print(secrets.token_hex(32))"
#   Secret key:  python3 -c "import secrets; print(secrets.token_hex(64))"

# 2. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

cd services/frontend && npm install && cd ../..

# 3. Start infrastructure
docker-compose up -d postgres redis prometheus grafana mlflow

# 4. Wait for postgres then run migrations
until docker-compose exec -T postgres pg_isready -U betting_user; do sleep 2; done

# 5. Seed development data
python3 scripts/seed_dev_data.py

# 6. Start API + workers
docker-compose up -d api celery-worker

# 7. Start frontend
cd services/frontend
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev -- --port 3001
```

### Service URLs (local)

| Service | URL | Default credentials |
|---|---|---|
| Frontend | http://localhost:3001 | demo / demo1234 |
| API | http://localhost:8000 | JWT auth |
| API docs | http://localhost:8000/docs | — |
| Airflow | http://localhost:8080 | admin / admin |
| Grafana | http://localhost:3000 | admin / `$GRAFANA_PASSWORD` |
| Prometheus | http://localhost:9090 | — |
| MLflow | http://localhost:5000 | — |

### Stopping local services

```bash
./scripts/teardown.sh            # Stop containers, keep volumes
./scripts/teardown.sh --volumes  # Stop + delete all data
./scripts/teardown.sh --all      # Full wipe
```

---

## GCP Prerequisites

### 1. Install tooling

```bash
# gcloud CLI
curl https://sdk.cloud.google.com | bash
gcloud init

# kubectl
gcloud components install kubectl

# Terraform >= 1.6
brew install terraform  # macOS
# or: https://developer.hashicorp.com/terraform/downloads
```

### 2. Create a GCP project

```bash
gcloud projects create betting-system-prod --name="Betting System"
gcloud config set project betting-system-prod
```

### 3. Enable required APIs

```bash
gcloud services enable \
  container.googleapis.com \
  sqladmin.googleapis.com \
  redis.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  cloudmonitoring.googleapis.com
```

### 4. Create service account

```bash
gcloud iam service-accounts create betting-system-sa \
  --display-name="Betting System SA"

# Grant roles
for role in \
  roles/container.admin \
  roles/cloudsql.client \
  roles/storage.objectAdmin \
  roles/artifactregistry.writer \
  roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding betting-system-prod \
    --member="serviceAccount:betting-system-sa@betting-system-prod.iam.gserviceaccount.com" \
    --role="$role"
done

# Download credentials
gcloud iam service-accounts keys create ~/.config/gcloud/betting-system-sa.json \
  --iam-account=betting-system-sa@betting-system-prod.iam.gserviceaccount.com

# Set in .env
echo 'GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/betting-system-sa.json' >> .env
```

### 5. Create Artifact Registry for Docker images

```bash
gcloud artifacts repositories create betting-system \
  --repository-format=docker \
  --location=us-central1

gcloud auth configure-docker us-central1-docker.pkg.dev
```

---

## Cloud Deployment (GCP + Kubernetes)

### Step 1 — Provision infrastructure with Terraform

```bash
cd infrastructure/terraform/environments/prod

# Initialise
terraform init

# Preview
terraform plan \
  -var="project_id=betting-system-prod" \
  -var="region=us-central1" \
  -var="db_password=YOUR_STRONG_DB_PASSWORD"

# Apply (takes ~15 minutes)
terraform apply \
  -var="project_id=betting-system-prod" \
  -var="region=us-central1" \
  -var="db_password=YOUR_STRONG_DB_PASSWORD"
```

Terraform creates:
- GKE Autopilot cluster: `betting-system-cluster`
- Cloud SQL PostgreSQL 15: `betting-system-db`
- Memorystore Redis: `betting-system-redis`
- GCS buckets: model artifacts + database backups
- VPC with private networking

### Step 2 — Authenticate kubectl

```bash
gcloud container clusters get-credentials betting-system-cluster \
  --region us-central1 \
  --project betting-system-prod
```

### Step 3 — Create Kubernetes secrets

```bash
# Create namespace first
kubectl apply -f infrastructure/kubernetes/base/namespace.yaml

# Create secrets (substitute real values)
kubectl create secret generic betting-secrets \
  --namespace=betting-system \
  --from-literal=DATABASE_URL="postgresql://betting_user:PASSWORD@CLOUD_SQL_IP:5432/betting_system" \
  --from-literal=REDIS_URL="redis://:PASSWORD@REDIS_IP:6379/0" \
  --from-literal=JWT_SECRET="YOUR_JWT_SECRET" \
  --from-literal=SECRET_KEY="YOUR_SECRET_KEY" \
  --from-literal=FERNET_KEY="YOUR_FERNET_KEY"
```

### Step 4 — Build and push Docker images

```bash
PROJECT_ID=betting-system-prod
REGISTRY=us-central1-docker.pkg.dev/$PROJECT_ID/betting-system
TAG=$(git rev-parse --short HEAD)

# Build all images
docker build -f docker/Dockerfile.api    -t $REGISTRY/betting-api:$TAG .
docker build -f docker/Dockerfile.airflow -t $REGISTRY/betting-airflow:$TAG .
docker build -f docker/Dockerfile.training -t $REGISTRY/betting-training:$TAG .

# Push
docker push $REGISTRY/betting-api:$TAG
docker push $REGISTRY/betting-airflow:$TAG
docker push $REGISTRY/betting-training:$TAG
```

### Step 5 — Deploy with Kustomize

```bash
# Update image tags in the overlay
cd infrastructure/kubernetes/overlays/prod
kustomize edit set image \
  betting-api=$REGISTRY/betting-api:$TAG \
  betting-airflow=$REGISTRY/betting-airflow:$TAG

# Apply
kubectl apply -k infrastructure/kubernetes/overlays/prod/

# Watch rollout
kubectl rollout status deployment/betting-api -n betting-system
kubectl get pods -n betting-system -w
```

### Step 6 — Run database migrations

```bash
kubectl exec -it -n betting-system deployment/betting-api -- \
  python -m alembic upgrade head
```

### Step 7 — Deploy monitoring stack

```bash
./monitoring/scripts/deploy-monitoring.sh --namespace monitoring

# Verify
python3 monitoring/scripts/validate-deployment.py
```

### Step 8 — Verify end-to-end

```bash
# Get external IP
kubectl get ingress -n betting-system

# Check API health
curl https://api.YOUR_DOMAIN.com/health

# Run e2e tests against production (read-only)
API_BASE_URL=https://api.YOUR_DOMAIN.com ./scripts/run-tests.sh e2e
```

---

## CI/CD Pipeline

The GitHub Actions workflows handle automated builds and deployments.

### Required GitHub Secrets

Go to **Repository → Settings → Secrets and variables → Actions** and add:

| Secret name | Value |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_SA_KEY` | Contents of the service account JSON file |
| `GKE_CLUSTER` | `betting-system-cluster` |
| `GKE_ZONE` | `us-central1` |
| `REGISTRY` | `us-central1-docker.pkg.dev/PROJECT_ID/betting-system` |
| `DATABASE_URL` | Production database URL |
| `REDIS_URL` | Production Redis URL |
| `JWT_SECRET` | Production JWT secret |
| `TELEGRAM_BOT_TOKEN` | (optional) Telegram alerts |
| `TELEGRAM_CHAT_ID` | (optional) Telegram chat ID |

### Workflow triggers

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Every push / PR | Lint, unit tests, build Docker images |
| `deploy-dev.yml` | Push to `develop` | Deploy to dev GKE namespace |
| `deploy-prod.yml` | Push to `main` | Build, test, deploy to production |
| `model-retrain.yml` | Manual / schedule | Run model retraining job |

---

## Rollback

```bash
# View rollout history
kubectl rollout history deployment/betting-api -n betting-system

# Roll back one version
kubectl rollout undo deployment/betting-api -n betting-system

# Roll back to specific revision
kubectl rollout undo deployment/betting-api --to-revision=3 -n betting-system

# Verify
kubectl rollout status deployment/betting-api -n betting-system
```

---

## Estimated Cloud Costs (GCP)

| Resource | Tier | Monthly cost |
|---|---|---|
| GKE Autopilot | ~1.5 vCPU average | ~$35 |
| Cloud SQL (db-f1-micro) | PostgreSQL 15 | ~$10 |
| Memorystore Redis (1 GB) | Basic tier | ~$25 |
| GCS storage (model artifacts + backups) | ~20 GB | ~$0.50 |
| Networking / Load Balancer | — | ~$20 |
| **Total** | | **~$90/month** |

Use `./monitoring/scripts/validate-deployment.py` to check all services are healthy after any deployment.
