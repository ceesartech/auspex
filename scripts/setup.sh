#!/usr/bin/env bash
# setup.sh — Bootstrap the betting system for local development
# Usage: ./scripts/setup.sh [--no-frontend] [--no-seed]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*"; exit 1; }

NO_FRONTEND=false
NO_SEED=false
for arg in "$@"; do
  case $arg in
    --no-frontend) NO_FRONTEND=true ;;
    --no-seed)     NO_SEED=true ;;
  esac
done

# ── 1. Prerequisite check ────────────────────────────────────────────────────
info "Checking prerequisites..."

# On macOS, several packages require a C compiler. Check for Xcode CLT.
if [[ "$(uname)" == "Darwin" ]]; then
  if ! xcode-select -p &>/dev/null; then
    warn "Xcode Command Line Tools not found."
    warn "Installing them now (you may see a system dialog)..."
    xcode-select --install 2>/dev/null || true
    warn "After installation completes, re-run this script."
    exit 1
  fi
  success "Xcode CLT found"

  # Ensure the Xcode license is accepted (required to compile C/C++ extensions).
  # xcodebuild exits non-zero if the license hasn't been agreed to.
  if ! xcodebuild -version &>/dev/null; then
    info "Xcode license not yet accepted — attempting auto-accept (may prompt for password)..."
    if sudo xcodebuild -license accept 2>/dev/null; then
      success "Xcode license accepted"
    else
      warn "Could not auto-accept Xcode license."
      warn "Run manually:  sudo xcodebuild -license accept"
      warn "Some optional packages (shap) that compile C++ extensions will be skipped."
    fi
  else
    success "Xcode license already accepted"
  fi
fi

for cmd in docker docker-compose python3 node npm; do
  if ! command -v "$cmd" &>/dev/null; then
    error "'$cmd' is required but not installed. See README.md for setup."
  fi
done

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
NODE_VERSION=$(node -e 'process.stdout.write(process.version.slice(1).split(".")[0])')

[[ "$PYTHON_VERSION" < "3.11" ]] && error "Python 3.11+ required (found $PYTHON_VERSION)"
[[ "$NODE_VERSION" -lt 18 ]]     && error "Node.js 18+ required (found $NODE_VERSION)"
success "Prerequisites OK (Python $PYTHON_VERSION, Node $NODE_VERSION)"

# ── 2. Environment file ──────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
  info "Creating .env from .env.example..."
  cp .env.example .env

  # Generate required secrets automatically
  FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || echo "REPLACE_WITH_FERNET_KEY")
  JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(64))")
  AIRFLOW_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(16))")

  # Patch .env
  sed -i.bak "s|GENERATE_WITH_COMMAND_python_-c_\"from_cryptography.fernet_import_Fernet;_print(Fernet.generate_key().decode())\"|$FERNET_KEY|" .env
  sed -i.bak "s|GENERATE_RANDOM_JWT_SECRET_AT_LEAST_32_CHARS|$JWT_SECRET|" .env
  sed -i.bak "s|GENERATE_RANDOM_SECRET_KEY_64_CHARS|$SECRET_KEY|" .env
  sed -i.bak "s|GENERATE_RANDOM_SECRET_KEY_32_CHARS|$AIRFLOW_SECRET|" .env
  rm -f .env.bak
  success ".env created with auto-generated secrets"
  warn "Review .env and fill in GCP_PROJECT_ID, TELEGRAM_BOT_TOKEN, and other optional values."
else
  success ".env already exists — skipping"
fi

# Load env vars — safe parser: handles unquoted values with spaces/commas,
# strips surrounding quotes, skips comments and blank lines.
_load_env() {
  local file="$1"
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip blank lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    # Must contain '='
    [[ "$line" != *=* ]] && continue
    local key="${line%%=*}"
    local value="${line#*=}"
    # Strip surrounding double or single quotes
    if [[ "$value" =~ ^\"(.*)\"$ ]]; then
      value="${BASH_REMATCH[1]}"
    elif [[ "$value" =~ ^\'(.*)\'$ ]]; then
      value="${BASH_REMATCH[1]}"
    fi
    export "$key=$value"
  done < "$file"
}
_load_env .env

# ── 3. Python virtual environment ────────────────────────────────────────────
if [[ ! -d venv ]]; then
  info "Creating Python virtual environment..."
  python3 -m venv venv
fi
# shellcheck source=/dev/null
source venv/bin/activate
info "Installing Python dependencies..."

# Upgrade pip, setuptools, and wheel first
pip install --quiet --upgrade pip setuptools wheel

# --prefer-binary tells pip to always use pre-built wheels and never compile
# from source. This avoids C compiler / Meson build failures on macOS and
# handles the case where the user already has a different version installed.
PIP_FLAGS="--upgrade --prefer-binary"

if ! pip install --quiet $PIP_FLAGS -r requirements.txt; then
  warn "requirements.txt install failed quietly — retrying verbosely..."
  pip install $PIP_FLAGS -r requirements.txt
fi

if ! pip install --quiet $PIP_FLAGS -r requirements-dev.txt; then
  warn "requirements-dev.txt install failed quietly — retrying verbosely..."
  pip install $PIP_FLAGS -r requirements-dev.txt
fi

# Install PyTorch separately (CPU-only, ~800 MB) — skip with --no-torch
if [[ "${NO_TORCH:-false}" == "false" ]]; then
  info "Installing PyTorch (CPU). Set NO_TORCH=true to skip..."
  pip install --quiet --prefer-binary -r requirements-torch.txt \
    || warn "PyTorch install failed — ML models requiring torch won't work. Run: pip install -r requirements-torch.txt"
fi

success "Python dependencies installed"

# ── 3b. Optional: SHAP (compiles C++ — requires Xcode license on macOS) ──────
# shap has no binary wheel for Python 3.13 on macOS arm64 — always builds from source.
# Requires the Xcode license to be accepted before clang++ can run.
if [[ "${NO_SHAP:-false}" != "true" ]]; then
  info "Attempting to install shap (optional — ML explainability)..."

  # Try to auto-accept Xcode license right before compiling (non-fatal if it fails).
  if [[ "$(uname)" == "Darwin" ]]; then
    sudo xcodebuild -license accept &>/dev/null || true
  fi

  if pip install --quiet shap==0.46.0 2>/dev/null; then
    success "shap installed"
  else
    warn "shap failed to install — SHAP explainability features will be unavailable."
    warn "To fix, run:"
    warn "  sudo xcodebuild -license accept"
    warn "  pip install shap==0.46.0"
    warn "Then set NO_SHAP=true to skip this step on future runs."
  fi
fi

# ── 4. Frontend dependencies ─────────────────────────────────────────────────
if [[ "$NO_FRONTEND" == false ]]; then
  info "Installing frontend dependencies..."
  cd services/frontend
  npm install --silent
  npx playwright install chromium >/dev/null \
    || warn "Playwright browser install failed. Run: cd services/frontend && npx playwright install chromium"
  cd "$ROOT"
  success "Frontend dependencies installed"
fi

# ── 5. Start infrastructure ──────────────────────────────────────────────────
# Verify Docker daemon is running before attempting to start containers.
if ! docker info &>/dev/null; then
  error "Docker daemon is not running. Start Docker Desktop (or 'sudo systemctl start docker' on Linux) and re-run this script."
fi

info "Starting infrastructure services (postgres, redis, prometheus, grafana, mlflow)..."
docker-compose up -d postgres redis prometheus grafana mlflow
info "Waiting for Postgres to be ready..."
until docker-compose exec -T postgres pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" &>/dev/null; do
  sleep 2
done
success "PostgreSQL ready"

# ── 6. Database migrations ───────────────────────────────────────────────────
info "Running database migrations..."
for migration in services/data-ingestion/db/migrations/*.sql; do
  docker-compose exec -T postgres psql \
    -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
    -f "/docker-entrypoint-initdb.d/$(basename "$migration")" \
    --quiet 2>/dev/null || true
done
success "Database migrations applied"

# ── 7. Seed data (optional) ──────────────────────────────────────────────────
if [[ "$NO_SEED" == false ]]; then
  info "Seeding development data..."
  python3 scripts/seed_dev_data.py 2>/dev/null || warn "Seed script not found or failed — skipping"
fi

# ── 8. Validate setup ────────────────────────────────────────────────────────
info "Validating setup..."
python3 scripts/validate_setup.py || warn "Validation found issues — check output above"

echo ""
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}  Setup complete! Run './scripts/start-local.sh'${NC}"
echo -e "${GREEN}  to launch all services.${NC}"
echo -e "${GREEN}======================================================${NC}"
echo ""
echo "  API:        http://localhost:8000"
echo "  Frontend:   http://localhost:3001  (npm run dev)"
echo "  Airflow:    http://localhost:8080  (admin / admin)"
echo "  Grafana:    http://localhost:3000  (admin / \$GRAFANA_PASSWORD)"
echo "  MLflow:     http://localhost:5001"
echo "  Prometheus: http://localhost:9090"
echo ""
