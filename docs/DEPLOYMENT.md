# Deployment Guide

> The system runs on a **single Hetzner VM** with Docker Compose behind Caddy.
> There is no Kubernetes/Terraform/GCP path (removed 2026-07). See
> [Production (single Hetzner VM + Docker Compose)](#production-single-hetzner-vm--docker-compose).

## Personal-VM deployment

The fastest way to get the system live on a single VM:

```bash
# On the VM (Ubuntu 22.04+):
curl -fsSL https://raw.githubusercontent.com/ceesartech/auspex/main/scripts/provision_vm.sh | sudo -E bash
```

The script installs Docker, clones the repo to `/opt/auspex`, sets up UFW,
swap, and unattended security upgrades. Then follow the printed next-steps:

1. Fill in `/opt/auspex/.env` (secrets, domain, ACME email).
2. Point your DNS A record at the VM's public IP.
3. Bring the stack up:
   ```bash
   cd /opt/auspex
   docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```
4. Load historical training data:
   ```bash
   docker compose exec api python /app/scripts/load_football_data.py \
     --leagues E0,D1,I1,SP1,F1 --seasons 10
   docker compose exec api python /app/scripts/load_statsbomb.py --all-open
   ```
5. Train initial models (uses time-series CV + isotonic calibration + ONNX
   export by default):
   ```bash
   docker compose exec api python -m training.train_all_models \
     --output-dir /models/staging --export-onnx
   mv /models/staging /models/production  # promote
   docker compose restart api              # picks up new artifacts
   ```
6. Verify: `curl https://YOUR_DOMAIN/health`.

After that, automated retraining runs weekly via the `retrain_models`
Airflow DAG. CI auto-deploys are opt-in — see [CI/CD pipeline](#cicd-pipeline).

## Environments

| Environment | Infrastructure | Purpose |
|---|---|---|
| Local | Docker Compose | Development and testing |
| Production | Single Hetzner VM + Docker Compose + Caddy | The one live environment |

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
## Production (single Hetzner VM + Docker Compose)

Production is one Hetzner VM (`ssh auspex` → `/opt/auspex`) running the Compose
stack behind Caddy (automatic Let's Encrypt TLS). There is **no Kubernetes,
Terraform, or GCP** — the GKE-era manifests were removed in the 2026-07 audit
(`docs/SYSTEM_AUDIT_AND_ROADMAP.md` §5.2).

### First-time / manual deploy on the VM

```bash
cd /opt/auspex
# .env holds all secrets (see .env.example for the full key list).
docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.ghcr.yml pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.ghcr.yml up -d
docker compose exec -T airflow-scheduler airflow db upgrade   # migrations
```

The three compose files layer: base (`docker-compose.yml`) + prod overlay
(`docker-compose.prod.yml`: Caddy + frontend, strips published ports) + ghcr
overlay (`docker-compose.ghcr.yml`: pull images instead of build). Always pass
all three on the VM.

**Bind-mounts vs images:** `scripts/` and `services/` are bind-mounted into the
containers, so a `git pull` on the VM hot-updates batch scripts, training code,
and DAGs immediately. The **api/frontend/airflow images** only change when CI
rebuilds them, and a running uvicorn needs an `api` restart to reload Python.

## CI/CD Pipeline

`.github/workflows/ci-cd.yaml` runs on every push/PR: flake8 + black + isort +
mypy, then every test suite (service suites + repo-root `tests/unit`), then —
on `main` — it builds `ghcr.io/ceesartech/auspex/{api,airflow,frontend}` tagged
with the commit SHA and runs `scripts/deploy_remote.sh` on the VM over SSH.

`deploy_remote.sh` flow: defensive `git stash` + `git pull --ff-only` → `docker
login ghcr.io` → `docker compose pull` (ghcr overlay) → `up -d` → 60s api
health-check (prints logs on failure).

The fetch/pull is authenticated with the run's `GITHUB_TOKEN` (passed as
`GHCR_TOKEN`) via a per-process `http.extraheader` — GitHub returns 401 on
anonymous `git-upload-pack` from cloud IP ranges even for public repos (first
seen 2026-09-02, run 33670930981). Manual pulls on the VM need a PAT; see
`OPERATIONS.md` → "Manual git pull on the VM needs a token".

### Required GitHub Secrets

| Secret | Value |
|---|---|
| `SSH_PRIVATE_KEY` / `SSH_HOST` / `SSH_USER` | VM access for the deploy step |
| `GHCR_TOKEN` | `GITHUB_TOKEN` (pushes images to GHCR) |

All application secrets (DB, Redis, JWT, Telegram, API keys, B2) live in
`/opt/auspex/.env` on the VM, not in CI.

## Rollback

Images are SHA-tagged in GHCR, so rollback = redeploy an earlier tag:

```bash
cd /opt/auspex
IMAGE_TAG=<previous-good-sha> docker compose \
  -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.ghcr.yml up -d
```

For a bad migration or data issue, restore from a backup per `OPERATIONS.md`
(local `/opt/auspex/backups/*.dump` or the Backblaze B2 copy).

## Costs

Single Hetzner CPX41-class VM (~€25-28/mo) + Backblaze B2 backups (<$1/mo).
GitHub Actions + GHCR are free (public repo). See
`docs/SYSTEM_AUDIT_AND_ROADMAP.md` §6.3 for the full cost breakdown.
