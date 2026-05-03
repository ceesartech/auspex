# Runbook: Database Issues

**Alert Names:** `DatabaseDown`, `DatabaseConnectionPoolExhausted`, `DatabaseHighQueryTime`, `DatabaseReplicationLag`
**Severity:** Critical / Warning
**Component:** database

## Description

PostgreSQL is unreachable, connection-pool exhausted, running long queries, or
replication has fallen behind.

---

## Diagnosis

### 1. Check exporter / database status
```bash
kubectl get pods -n betting-system -l app=postgres-exporter
curl -s http://postgres-exporter:9187/metrics | grep pg_up
```

### 2. Connection counts
```sql
SELECT count(*), state
FROM pg_stat_activity
GROUP BY state;
```

### 3. Long-running queries
```sql
SELECT pid, now() - pg_stat_activity.query_start AS duration, query, state
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '30 seconds'
  AND state != 'idle'
ORDER BY duration DESC;
```

### 4. Replication lag
```sql
SELECT client_addr, state,
       pg_wal_lsn_diff(pg_current_wal_lsn(), sent_lsn)  AS send_lag,
       pg_wal_lsn_diff(sent_lsn, replay_lsn)            AS replay_lag
FROM pg_stat_replication;
```

### 5. Table bloat / disk usage
```sql
SELECT relname, pg_size_pretty(pg_total_relation_size(oid))
FROM pg_class
WHERE relkind = 'r'
ORDER BY pg_total_relation_size(oid) DESC
LIMIT 20;
```

---

## Resolution

### DatabaseDown

1. Check the Postgres pod:
   ```bash
   kubectl get pods -n betting-system -l app=postgres
   kubectl logs -n betting-system statefulset/postgres --tail=100
   ```
2. Restart if crashed:
   ```bash
   kubectl rollout restart statefulset/postgres -n betting-system
   ```
3. If using Cloud SQL, check GCP console for instance health.

### DatabaseConnectionPoolExhausted

1. Identify which service is holding connections:
   ```sql
   SELECT application_name, count(*) FROM pg_stat_activity GROUP BY 1 ORDER BY 2 DESC;
   ```
2. Terminate idle connections:
   ```sql
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE state = 'idle'
     AND state_change < now() - interval '5 minutes';
   ```
3. Increase `max_connections` in the Postgres config or reduce pool size in the app.

### DatabaseHighQueryTime

1. Identify and optionally kill long-running queries:
   ```sql
   SELECT pg_cancel_backend(pid) FROM pg_stat_activity
   WHERE (now() - query_start) > interval '2 minutes' AND state = 'active';
   ```
2. Run `EXPLAIN ANALYZE` on the offending query to find missing indexes.
3. Add indexes as needed via a new migration.

### DatabaseReplicationLag

1. Check network between primary and replica.
2. If lag is persistent, restart the replica pod:
   ```bash
   kubectl rollout restart statefulset/postgres-replica -n betting-system
   ```
3. For Cloud SQL: check the GCP console replication health tab.

---

## Backup & Restore

```bash
# Manual backup
infrastructure/scripts/backup-db.sh

# Restore from backup
infrastructure/scripts/restore-db.sh <backup-file>
```

---

## Prevention

- Connection pooling via PgBouncer.
- Regular `VACUUM ANALYZE` scheduled via Airflow.
- Index strategy reviewed on schema changes.
- Automated daily backups to GCS.

## Related Runbooks

- `API_DOWN.md`
- `HIGH_ERROR_RATE.md`
