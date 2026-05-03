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

### 2. Inspect recent errors in Loki / pod logs
```bash
kubectl logs -n betting-system deployment/betting-api --tail=500 \
  | grep -E '"status": 5[0-9][0-9]'
```

### 3. Check dependent services
```bash
# Database
kubectl exec -it deployment/betting-api -n betting-system -- \
  python -c "import psycopg2, os; psycopg2.connect(os.environ['DATABASE_URL'])"

# Redis
kubectl exec -it deployment/betting-api -n betting-system -- \
  python -c "from redis import Redis; import os; Redis.from_url(os.environ['REDIS_URL']).ping()"
```

### 4. Review Celery task failures
```bash
kubectl logs -n betting-system deployment/celery-worker --tail=200 \
  | grep -i "error\|exception\|traceback"
```

---

## Resolution

### Transient spike (< 15 min)
- Wait; the alert resolves automatically if the spike was transient.
- Check if a scraper job or batch task caused a burst of writes.

### Persistent errors from a specific endpoint
1. Identify the endpoint from Grafana.
2. Check if a recent deployment changed that route:
   ```bash
   kubectl rollout history deployment/betting-api -n betting-system
   ```
3. Roll back if a bad deploy is suspected:
   ```bash
   kubectl rollout undo deployment/betting-api -n betting-system
   ```

### Database overload
- Scale DB read replicas or increase connection pool size.
- See `DATABASE_ISSUES.md`.

### Memory pressure causing 503s from gunicorn
```bash
kubectl set resources deployment/betting-api \
  -n betting-system --limits=memory=2Gi
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
