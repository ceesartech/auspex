# Runbook: High Error Rate

**Alert Name:** `HighErrorRate`
**Severity:** Critical
**Component:** api

## Description

The 5xx error rate on the API has exceeded 5% over a 5-minute window.

## Impact

- Users receiving errors for predictions, recommendations, or auth.
- Data may not be written correctly.

---

## Diagnosis

### 1. Identify the failing endpoint
```bash
# In Grafana: API Performance > Error Rate by Endpoint
# Or via promtool:
curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(http_requests_total{status=~\"5..\"}[5m]))by(endpoint)" \
  | python -m json.tool
```

### 2. Inspect recent errors in container logs
```bash
docker compose logs --tail=500 api | grep -E '"status": 5[0-9][0-9]'
```

### 3. Check dependent services
```bash
# Database
docker compose exec -T api python -c "import psycopg2, os; psycopg2.connect(os.environ['DATABASE_URL'])"

# Redis
docker compose exec -T api python -c "from redis import Redis; import os; Redis.from_url(os.environ['REDIS_URL']).ping()"
```

### 4. Review Celery task failures
```bash
docker compose logs --tail=200 celery-worker | grep -i "error\|exception\|traceback"
```

---

## Resolution

### Transient spike (< 15 min)
- Wait; the alert resolves automatically if the spike was transient.
- Check if a scraper job or batch task caused a burst of writes.

### Persistent errors from a specific endpoint
1. Identify the endpoint from Grafana.
2. Check whether the running image changed recently:
   ```bash
   docker inspect auspex-api --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
   docker images ghcr.io/ceesartech/auspex/api
   ```
3. Roll back to the previous good SHA if a bad deploy is suspected:
   ```bash
   IMAGE_TAG=<previous-good-sha> docker compose -f docker-compose.yml \
     -f docker-compose.prod.yml -f docker-compose.ghcr.yml up -d api
   ```

### Database overload
- Kill long-running queries and check the connection count.
- See `DATABASE_ISSUES.md`.

### Memory pressure causing 503s from uvicorn
Raise the api `mem_limit` in `docker-compose.prod.yml`, then:
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  -f docker-compose.ghcr.yml up -d api
```

---

## Prevention

- Implement request validation at API boundaries.
- Add circuit breakers for DB/Redis calls.
- Load test before each release.
- Monitor P95 latency — rising latency often precedes 5xx spikes.

## Related Runbooks

- `API_DOWN.md`
- `DATABASE_ISSUES.md`
