#!/usr/bin/env bash
# Wipe and re-apply every SQL migration in
# services/data-ingestion/db/migrations/.
#
# Drops Airflow's schema (if present) and the entire `public` schema in
# the configured database, then re-runs every /docker-entrypoint-initdb.d/*.sql
# inside the postgres container (which is the same mount). This is what
# you want after a Fernet-key rotation (Airflow's encrypted rows are now
# undecryptable) or any other partial-init failure.
#
# This DELETES ALL DATA. Pass --yes to skip the confirmation prompt
# (used by automation; you almost never want this interactively).
#
# Usage:
#   scripts/reset_database.sh
#   scripts/reset_database.sh --yes
#
# Run from the project root.

set -euo pipefail

CONFIRMED=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) CONFIRMED=1 ;;
    *) ;;
  esac
done

log()  { printf '\033[1;34m[reset]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[reset]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[reset]\033[0m %s\n' "$*" >&2; }

if [ ! -f docker-compose.yml ]; then
  err "docker-compose.yml not found in $(pwd). cd to the project root."
  exit 2
fi
if [ ! -f .env ]; then
  err ".env not found in $(pwd)."
  exit 2
fi

read_var() {
  grep -E "^${1}=" .env | head -1 | cut -d= -f2-
}

USER_NAME="$(read_var POSTGRES_USER)"
DB_NAME="$(read_var POSTGRES_DB)"
USER_NAME="${USER_NAME:-betting_user}"
DB_NAME="${DB_NAME:-betting_system}"

if [ "$CONFIRMED" -ne 1 ]; then
  warn "This will DROP the public schema (and any Airflow schema) in"
  warn "database \"$DB_NAME\" on the running postgres container."
  warn "ALL DATA in that database will be lost."
  printf 'Type RESET to continue: '
  read -r ANSWER
  if [ "$ANSWER" != "RESET" ]; then
    log "Aborted."
    exit 1
  fi
fi

log "Stopping airflow services so they don't reconnect during the wipe..."
docker compose stop airflow-init airflow-scheduler airflow-webserver 2>/dev/null || true

log "Dropping schemas in database \"$DB_NAME\"..."
docker compose exec -T postgres psql -U "$USER_NAME" -d "$DB_NAME" -c "
  DROP SCHEMA IF EXISTS airflow CASCADE;
  DROP SCHEMA public CASCADE;
  CREATE SCHEMA public;
  GRANT ALL ON SCHEMA public TO public;
  GRANT ALL ON SCHEMA public TO \"$USER_NAME\";
"

log "Re-applying migrations in /docker-entrypoint-initdb.d/..."
docker compose exec -T postgres sh -c "
  set -e
  for f in /docker-entrypoint-initdb.d/*.sql; do
    echo \"Applying \$f\"
    psql -U \"$USER_NAME\" -d \"$DB_NAME\" -f \"\$f\"
  done
"

log "Database reset complete. Now restart the stack:"
log "  scripts/restart_services.sh"
log ""
log "Then re-seed users:"
log "  docker compose exec api python /app/scripts/seed_users.py"
