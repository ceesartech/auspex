# Runbook: API Down

**Alert Name:** `APIDown`
**Severity:** Critical
**Component:** api

## Description

One or more `betting-api` pod instances are failing health checks.

## Impact

- Service unavailable for all users.
- No predictions served; WebSocket connections dropped.
- Potential write failures if the pod crashes mid-request.

---

## Diagnosis

### 1. Pod status
```bash
kubectl get pods -n betting-system -l app=betting-api
```

### 2. Pod logs (last 200 lines)
```bash
kubectl logs -n betting-system deployment/betting-api --tail=200
```

### 3. Pod events
```bash
kubectl describe pod -n betting-system -l app=betting-api | grep -A 20 Events
```

### 4. Resource usage
```bash
kubectl top pod -n betting-system -l app=betting-api
```

### 5. Health endpoint (from inside cluster)
```bash
kubectl run debug --image=curlimages/curl --restart=Never --rm -it -- \
  curl -s http://betting-api:8000/health
```

---

## Resolution

### CrashLoopBackOff / OOMKilled

1. Check logs for the root cause (DB conn, Redis conn, OOM).
2. Increase memory limit temporarily:
   ```bash
   kubectl set resources deployment/betting-api \
     -n betting-system \
     --limits=memory=2Gi
   ```
3. Scale up replicas while investigating:
   ```bash
   kubectl scale deployment betting-api -n betting-system --replicas=4
   ```

### Database connection failure
```bash
# Verify connectivity from inside the pod
kubectl exec -it deployment/betting-api -n betting-system -- \
  python -c "import psycopg2, os; psycopg2.connect(os.environ['DATABASE_URL']); print('DB OK')"
```
See also: `DATABASE_ISSUES.md`.

### Redis connection failure
```bash
kubectl exec -it deployment/betting-api -n betting-system -- \
  python -c "from redis import Redis; import os; Redis.from_url(os.environ['REDIS_URL']).ping(); print('Redis OK')"
```

### Emergency rollback
```bash
kubectl rollout undo deployment/betting-api -n betting-system
kubectl rollout status deployment/betting-api -n betting-system
```

---

## Prevention

- Liveness/readiness probes configured on `/health`.
- HPA ensures minimum 2 replicas at all times.
- Circuit breakers prevent cascade failures from DB/Redis.
- Regular load testing to catch resource limits early.

## Related Runbooks

- `HIGH_ERROR_RATE.md`
- `DATABASE_ISSUES.md`
