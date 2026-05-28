#!/usr/bin/env bash
# Regenerate the rotating secrets in a .env file.
#
# Flow:
#   1. Run scripts/rotate_secrets.py --print-only to dump fresh KEY=VALUE
#      lines into a mode-0600 temp file.
#   2. For each KEY=VALUE in the temp file, sed-replace the matching line
#      in the target .env (preserves comments, ordering, and all other vars).
#   3. Re-derive DATABASE_URL, AIRFLOW__DATABASE__SQL_ALCHEMY_CONN, and
#      REDIS_URL from the new secrets + existing POSTGRES_USER/POSTGRES_DB.
#   4. Shred the temp file.
#
# Usage:
#   scripts/regenerate_secrets.sh                  # rotates ./.env
#   scripts/regenerate_secrets.sh /opt/auspex/.env # rotates an absolute path
#
# After running, manually rotate TELEGRAM_BOT_TOKEN via @BotFather, paste
# the new token into .env, then run scripts/restart_services.sh.

set -euo pipefail

ENV_FILE="${1:-.env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROTATE_PY="$SCRIPT_DIR/rotate_secrets.py"

log()  { printf '\033[1;34m[regen]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[regen]\033[0m %s\n' "$*" >&2; }

if [ ! -f "$ENV_FILE" ]; then
  err "env file not found: $ENV_FILE"
  exit 2
fi

if [ ! -f "$ROTATE_PY" ]; then
  err "expected $ROTATE_PY to exist next to this script"
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  err "python3 not found in PATH (set PYTHON_BIN to override)"
  exit 2
fi

# 1. Generate fresh secrets into a mode-0600 temp file.
TMP="$(mktemp -t auspex_secrets.XXXXXX)"
chmod 600 "$TMP"
trap 'safe_shred "$TMP"' EXIT INT TERM

# Cross-platform shred: prefer GNU shred; otherwise overwrite with /dev/urandom
# before unlinking. Fine for a few-second-lived secrets file.
safe_shred() {
  local f="$1"
  [ -f "$f" ] || return 0
  if command -v shred >/dev/null 2>&1; then
    shred -u "$f" 2>/dev/null || rm -f "$f"
  else
    local size
    size=$(wc -c < "$f" | tr -d ' ')
    if [ "$size" -gt 0 ]; then
      dd if=/dev/urandom of="$f" bs=1 count="$size" conv=notrunc 2>/dev/null || true
    fi
    rm -f "$f"
  fi
}

log "Generating new secrets..."
"$PYTHON_BIN" "$ROTATE_PY" --print-only > "$TMP"
log "Wrote $(wc -l < "$TMP" | tr -d ' ') KEY=VALUE pairs to $TMP"

# 2. Back up .env before mutating it.
BACKUP="${ENV_FILE}.bak"
cp -p "$ENV_FILE" "$BACKUP"
chmod 600 "$BACKUP" || true
log "Backed up $ENV_FILE -> $BACKUP"

# 3. Apply each rotated KEY=VALUE to .env in place. If a key isn't already
# in the file (rare — only happens on a fresh seed), append it.
escape_sed_replacement() {
  printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

while IFS='=' read -r KEY VAL; do
  [ -z "${KEY// /}" ] && continue
  ESC="$(escape_sed_replacement "$VAL")"
  if grep -qE "^${KEY}=" "$ENV_FILE"; then
    sed -i.tmp "s|^${KEY}=.*|${KEY}=${ESC}|" "$ENV_FILE"
    rm -f "${ENV_FILE}.tmp"
  else
    echo "${KEY}=${VAL}" >> "$ENV_FILE"
  fi
done < "$TMP"

# 4. Re-derive URL-style fields that embed the rotated secrets.
PG_USER="$(grep -E '^POSTGRES_USER=' "$ENV_FILE" | head -1 | cut -d= -f2-)"
PG_DB="$(grep -E '^POSTGRES_DB=' "$ENV_FILE" | head -1 | cut -d= -f2-)"
PG_PW="$(grep -E '^POSTGRES_PASSWORD=' "$ENV_FILE" | head -1 | cut -d= -f2-)"
REDIS_PW="$(grep -E '^REDIS_PASSWORD=' "$ENV_FILE" | head -1 | cut -d= -f2-)"

PG_USER="${PG_USER:-betting_user}"
PG_DB="${PG_DB:-betting_system}"
PG_URL="postgresql://${PG_USER}:${PG_PW}@postgres:5432/${PG_DB}"
REDIS_URL="redis://:${REDIS_PW}@redis:6379/0"

upsert() {
  local key="$1" value="$2"
  local esc
  esc="$(escape_sed_replacement "$value")"
  if grep -qE "^${key}=" "$ENV_FILE"; then
    sed -i.tmp "s|^${key}=.*|${key}=${esc}|" "$ENV_FILE"
    rm -f "${ENV_FILE}.tmp"
  else
    echo "${key}=${value}" >> "$ENV_FILE"
  fi
}

upsert DATABASE_URL "$PG_URL"
upsert AIRFLOW__DATABASE__SQL_ALCHEMY_CONN "$PG_URL"
upsert REDIS_URL "$REDIS_URL"

chmod 600 "$ENV_FILE"

log "Rotated $ENV_FILE — old values backed up to $BACKUP"
log "Diff of changed lines:"
diff "$BACKUP" "$ENV_FILE" || true

cat <<'EOF'

Next steps:
  1. Rotate TELEGRAM_BOT_TOKEN at https://t.me/BotFather (/mybots → API Token → Revoke).
     Paste the new token over the existing TELEGRAM_BOT_TOKEN= line in the .env above.

  2. If Postgres is currently running with the OLD password (e.g. you've already
     run `docker compose up` before), apply the new password to the running DB:

       OLD=$(grep ^POSTGRES_PASSWORD= .env.bak | cut -d= -f2-)
       NEW=$(grep ^POSTGRES_PASSWORD= .env     | cut -d= -f2-)
       docker compose exec -T postgres psql -U betting_user -d betting_system \
           -c "ALTER USER betting_user WITH PASSWORD '$NEW';"

  3. Restart the stack so all containers pick up the new values:
       scripts/restart_services.sh

  4. After confirming everything works, delete the .env.bak file:
       shred -u .env.bak   # or: rm .env.bak
EOF
