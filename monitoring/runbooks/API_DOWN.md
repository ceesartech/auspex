# Runbook: API Down

**Alert Name:** `APIDown`
**Severity:** Critical
**Component:** api

## Description

The `api` container is failing its health check (`/health` on `127.0.0.1:8000`).

## Impact

- Service unavailable for all users.
- No predictions served; WebSocket connections dropped.
- Potential write failures if the container crashes mid-request.

> All commands run on the VM from `/opt/auspex`. Compose needs the three
> production overlays — export once per shell:
> ```bash
> cd /opt/auspex
> alias dc='docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.ghcr.yml'
> ```

---

## Diagnosis

### 1. Container status
```bash
dc ps api
```

### 2. Logs (last 200 lines)
```bash
dc logs --tail=200 api
```

### 3. Resource usage
```bash
docker stats --no-stream api
```

### 4. Health endpoint (from the host)
```bash
curl -s http://127.0.0.1:8000/health
```

---

## Resolution

### Crash loop / OOM

1. Check logs for the root cause (DB conn, Redis conn, OOM).
2. Inspect exit reason:
   ```bash
   docker inspect auspex-api --format '{{.State.OOMKilled}} {{.State.ExitCode}}'
   ```
3. If OOM, raise the per-service `mem_limit` in `docker-compose.prod.yml`,
   then `dc up -d api`.

### Restart the container
```bash
dc restart api
# or force a clean recreate
dc up -d --force-recreate api
```

### Database connection failure
```bash
dc exec -T api \
  python -c "import psycopg2, os; psycopg2.connect(os.environ['DATABASE_URL']); print('DB OK')"
```
See also: `DATABASE_ISSUES.md`.

### Redis connection failure
```bash
dc exec -T api \
  python -c "from redis import Redis; import os; Redis.from_url(os.environ['REDIS_URL']).ping(); print('Redis OK')"
```

### Emergency rollback

Deploys are GHCR images tagged by git SHA. To roll back, set `IMAGE_TAG` to the
previous good SHA and re-pull:
```bash
IMAGE_TAG=<previous-good-sha> dc pull api && \
IMAGE_TAG=<previous-good-sha> dc up -d api
```
(Find prior tags with `docker images ghcr.io/ceesartech/auspex/api`.)

---

## Prevention

- Compose healthcheck on `/health` with `restart: unless-stopped`.
- Backups (`OPERATIONS.md`) let you recover DB state if a bad migration is the cause.
- Watch Grafana + Alertmanager (Telegram) for early memory/latency drift.

## Related Runbooks

- `HIGH_ERROR_RATE.md`
- `DATABASE_ISSUES.md`
