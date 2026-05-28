#!/usr/bin/env bash
# Apply the POSTGRES_PASSWORD from .env to the running Postgres user.
#
# Use this after scripts/regenerate_secrets.sh has rotated .env but the
# Postgres volume still has the OLD password baked in. The script reads
# the old password from .env.bak (created by regenerate_secrets.sh) and
# the new one from .env, then issues an ALTER USER inside the postgres
# container.
#
# Usage:
#   scripts/apply_db_password.sh                           # uses ./.env and ./.env.bak
#   scripts/apply_db_password.sh /opt/auspex/.env          # uses specified .env, with sibling .env.bak
#   OLD_PG_PASSWORD=... scripts/apply_db_password.sh       # override the old-password source
#
# Run from the project root (where docker-compose.yml lives).

set -euo pipefail

ENV_FILE="${1:-.env}"
BACKUP_FILE="${ENV_FILE}.bak"

log()  { printf '\033[1;34m[db-pw]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[db-pw]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[db-pw]\033[0m %s\n' "$*" >&2; }

if [ ! -f "$ENV_FILE" ]; then
  err "env file not found: $ENV_FILE"
  exit 2
fi
if [ ! -f docker-compose.yml ]; then
  err "docker-compose.yml not found in $(pwd). cd to the project root."
  exit 2
fi

read_var() {
  local file="$1" key="$2"
  grep -E "^${key}=" "$file" | head -1 | cut -d= -f2-
}

NEW="$(read_var "$ENV_FILE" POSTGRES_PASSWORD)"
USER_NAME="$(read_var "$ENV_FILE" POSTGRES_USER)"
DB_NAME="$(read_var "$ENV_FILE" POSTGRES_DB)"
USER_NAME="${USER_NAME:-betting_user}"
DB_NAME="${DB_NAME:-betting_system}"

if [ -z "$NEW" ]; then
  err "POSTGRES_PASSWORD not set in $ENV_FILE"
  exit 2
fi

# The old password is needed only to authenticate the ALTER USER call.
# It can come from .env.bak (default), the OLD_PG_PASSWORD env var, or
# from inside the container if Postgres is configured to trust local
# connections (the default for the postgres user via peer/trust).
OLD="${OLD_PG_PASSWORD:-}"
if [ -z "$OLD" ] && [ -f "$BACKUP_FILE" ]; then
  OLD="$(read_var "$BACKUP_FILE" POSTGRES_PASSWORD)"
fi
if [ -z "$OLD" ]; then
  warn "No old password found in $BACKUP_FILE and OLD_PG_PASSWORD not set."
  warn "Will attempt the ALTER without explicit auth (works if postgres trusts local)."
fi

# Escape single quotes for SQL string literal: ' -> ''
escape_sql() {
  printf "%s" "$1" | sed "s/'/''/g"
}

SQL="ALTER USER \"${USER_NAME}\" WITH PASSWORD '$(escape_sql "$NEW")';"

log "Applying new password for role \"$USER_NAME\" in database \"$DB_NAME\"..."

if [ -n "$OLD" ]; then
  # PGPASSWORD inside the container authenticates the ALTER.
  docker compose exec -T -e "PGPASSWORD=$OLD" postgres \
    psql -U "$USER_NAME" -d "$DB_NAME" -c "$SQL"
else
  docker compose exec -T postgres \
    psql -U "$USER_NAME" -d "$DB_NAME" -c "$SQL"
fi

log "Done. Restart services to pick up the new password:"
log "  scripts/restart_services.sh"
