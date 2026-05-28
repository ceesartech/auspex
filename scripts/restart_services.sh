#!/usr/bin/env bash
# Restart the Auspex docker-compose stack.
#
# By default, brings the prod overlay down and back up — pulls the latest
# images first, then waits for each service's health check to pass.
#
# Usage:
#   scripts/restart_services.sh             # full down/up cycle
#   scripts/restart_services.sh restart     # just `docker compose restart`
#                                           # (no rebuild, no image pull)
#   scripts/restart_services.sh api         # restart a single service
#                                           # (or any other service name)
#
# Run this from the project root (where docker-compose.yml lives).

set -euo pipefail

MODE="${1:-full}"
COMPOSE_BASE=(docker compose -f docker-compose.yml)
if [ -f docker-compose.prod.yml ]; then
  COMPOSE_BASE+=(-f docker-compose.prod.yml)
fi

log()  { printf '\033[1;34m[restart]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[restart]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[restart]\033[0m %s\n' "$*" >&2; }

if [ ! -f docker-compose.yml ]; then
  err "docker-compose.yml not found in $(pwd). cd to the project root."
  exit 2
fi

case "$MODE" in
  full)
    log "Stopping the stack..."
    "${COMPOSE_BASE[@]}" down --remove-orphans

    log "Pulling latest images..."
    "${COMPOSE_BASE[@]}" pull --quiet || warn "pull had errors (locally-built images are normal)"

    log "Starting the stack..."
    "${COMPOSE_BASE[@]}" up -d
    ;;

  restart)
    log "Restarting all services (no rebuild)..."
    "${COMPOSE_BASE[@]}" restart
    ;;

  *)
    # Restart a single named service.
    log "Restarting service: $MODE"
    "${COMPOSE_BASE[@]}" restart "$MODE"
    ;;
esac

# Give containers a moment to come up before we ask about health.
sleep 5

log "Service status:"
"${COMPOSE_BASE[@]}" ps

# Wait for any "starting" health checks to settle, up to 60s.
DEADLINE=$(( $(date +%s) + 60 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if ! "${COMPOSE_BASE[@]}" ps --format json 2>/dev/null \
       | grep -q '"Health":"starting"'; then
    break
  fi
  sleep 3
done

echo
log "Final status:"
"${COMPOSE_BASE[@]}" ps

# Quick smoke test if a domain is configured.
if [ -f .env ]; then
  DOMAIN="$(grep -E '^AUSPEX_DOMAIN=' .env | head -1 | cut -d= -f2- || true)"
fi
if [ -n "${DOMAIN:-}" ]; then
  echo
  log "Hitting https://$DOMAIN/health ..."
  if curl -fsS "https://$DOMAIN/health" >/tmp/auspex_health 2>&1; then
    cat /tmp/auspex_health
    rm -f /tmp/auspex_health
    echo
    log "Stack is up and serving."
  else
    warn "Health check failed. Show recent logs with:"
    warn "  docker compose logs api caddy --tail=40"
  fi
fi
