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
docker compose ps postgres postgres-exporter
curl -s http://127.0.0.1:9187/metrics | grep pg_up
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

1. Check the Postgres container:
   ```bash
   docker compose ps postgres
   docker compose logs --tail=100 postgres
   ```
2. Restart if crashed:
   ```bash
   docker compose restart postgres
   ```
3. Confirm the data volume is intact and disk is not full (`df -h`); a full
   disk is the most common cause of Postgres refusing writes on this VM.

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

This single-VM deployment runs **one** Postgres instance — there is no replica,
so this alert should never fire here. If it does, the `pg_stat_replication`
scrape is misconfigured (a leftover rule); silence it or remove the rule from
`monitoring/prometheus/alerts/`. Durability comes from the daily backup +
Backblaze B2 offsite copy, not from streaming replication.

---

## Backup & Restore

```bash
# Manual backup
docker compose exec -T api python /app/scripts/backup_postgres.py  # see OPERATIONS.md

# Restore from backup
# restore: pg_restore --clean --if-exists from a .dump — full procedure in OPERATIONS.md
```

---

## Prevention

- Regular `VACUUM ANALYZE` (Postgres autovacuum) plus `airflow db clean` for metadata.
- Index strategy reviewed on schema changes.
- Automated daily `pg_dump` backups with local rotation + Backblaze B2 offsite copy.

## Related Runbooks

- `API_DOWN.md`
- `HIGH_ERROR_RATE.md`
