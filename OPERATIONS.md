# Operations runbook

Working playbook for routine operations + incident response. Keep
this in the repo so future-you (or anyone else) can find it. Add
new sections as patterns emerge.

## Database backups + restore

### Daily backup

The `db_backup_daily` Airflow DAG runs `scripts/backup_postgres.py`
at 02:00 UTC every day. The script:

1. Writes a compressed `pg_dump` to `/app/backups/` inside the api
   container (bind-mounted to `/opt/auspex/backups/` on the host).
   Files named `auspex_prod-YYYY-MM-DDTHHMMSSZ.dump`.
2. Optionally uploads to S3-compatible storage when
   `BACKUP_S3_BUCKET` env var is set.
3. Rotates local backups older than 7 days.

### Initial setup on a fresh host

```bash
# 1. Create the host backups dir owned by UID 1000 (appuser).
sudo mkdir -p /opt/auspex/backups
sudo chown -R 1000:1000 /opt/auspex/backups

# 2. (Optional) Configure S3 upload. Add to /opt/auspex/.env:
#      BACKUP_S3_BUCKET=auspex-backups
#      BACKUP_S3_PREFIX=postgres/                       # optional
#      BACKUP_S3_ENDPOINT_URL=https://...               # for B2/Hetzner/Wasabi (omit for AWS)
#      AWS_ACCESS_KEY_ID=...
#      AWS_SECRET_ACCESS_KEY=...
#      AWS_DEFAULT_REGION=us-east-1

# 3. Restart the api container so it picks up the new env + bind mount:
docker compose up -d --force-recreate api

# 4. Trigger the DAG manually once to verify end-to-end:
docker compose exec airflow-scheduler airflow dags trigger db_backup_daily

# 5. Verify the dump landed:
ls -la /opt/auspex/backups/
```

### S3 lifecycle policy (set on the bucket once)

Manage long-term retention server-side so the backup script stays
simple. Example for AWS S3 — apply via console or `aws s3api
put-bucket-lifecycle-configuration`:

```json
{
  "Rules": [{
    "ID": "auspex-backup-retention",
    "Status": "Enabled",
    "Filter": {"Prefix": "postgres/"},
    "Transitions": [
      {"Days": 30, "StorageClass": "GLACIER"}
    ],
    "Expiration": {"Days": 365}
  }]
}
```

Costs ~$0.10/mo for 1GB Glacier storage + tiny PUT/GET costs.

### Restore — full disaster recovery

The DB has been wiped or the host is gone. Recovery:

```bash
# 1. Fetch the most recent backup. Either:
#    a) From local disk if the host is intact:
ls -t /opt/auspex/backups/auspex_prod-*.dump | head -1
#    b) From S3 if the host is gone:
aws s3 ls s3://auspex-backups/postgres/ | tail -5
aws s3 cp s3://auspex-backups/postgres/auspex_prod-2026-06-07T020000Z.dump /tmp/

# 2. If restoring to a NEW host, bring up just postgres first so we
#    can restore into it before everything else starts using it:
docker compose up -d postgres
sleep 5  # wait for postgres to accept connections

# 3. Restore. --clean --if-exists drops existing objects first so
#    a partial DB can be replaced cleanly. -j 4 parallelises restore.
docker compose exec -T postgres pg_restore \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    --clean --if-exists \
    -j 4 \
    < /tmp/auspex_prod-2026-06-07T020000Z.dump

# 4. Bring up the rest of the stack:
docker compose up -d

# 5. Smoke-test:
docker compose exec api curl -s http://localhost:8000/health
```

### Restore — single table / partial recovery

For "oh no, I just `TRUNCATE`d the wrong table" recovery:

```bash
# Restore JUST the affected table(s) into a temporary schema, then
# INSERT INTO the live tables what you need. -t restricts pg_restore.
docker compose exec -T postgres pg_restore \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    -t race_predictions \
    --data-only \
    < /opt/auspex/backups/auspex_prod-2026-06-07T020000Z.dump
```

### Test the restore quarterly

Backups you've never restored aren't backups. Quarterly:

```bash
# 1. Spin up a throwaway postgres container on a free port:
docker run --name pg-restore-test -e POSTGRES_PASSWORD=test \
    -p 5434:5432 -d postgres:16
sleep 5

# 2. Restore the latest backup into it:
docker exec -i pg-restore-test pg_restore -U postgres -d postgres \
    --clean --if-exists < /opt/auspex/backups/$(ls -t /opt/auspex/backups/*.dump | head -1 | xargs basename)

# 3. Sanity-check a couple tables:
docker exec pg-restore-test psql -U postgres -c \
    "SELECT (SELECT COUNT(*) FROM matches) AS matches,
            (SELECT COUNT(*) FROM odds) AS odds,
            (SELECT COUNT(*) FROM race_predictions) AS race_preds;"

# 4. Tear down:
docker rm -f pg-restore-test
```

If those counts roughly match prod, the backup is valid.

## Common operations

### Restart a service

```bash
docker compose restart api          # graceful restart
docker compose up -d --force-recreate api  # full recreate (picks up new .env / mount changes)
```

### Apply a new migration on prod

```bash
ssh auspex && cd /opt/auspex
git pull
docker compose cp services/data-ingestion/db/migrations/0XX_thing.sql postgres:/tmp/0XX.sql
docker compose exec -T postgres sh -c \
    'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/0XX.sql'
```

### Roll back a model promotion

Each `promote_to_production` step backs up the prior model dir to
`/app/models/production-prev-<model>-<timestamp>`. To roll back:

```bash
docker compose exec -T api bash -c "
  cp -r /app/models/production-prev-nflml-20260604-013325/ensemble_nfl_ml \\
        /app/models/production/ensemble_nfl_ml
"
```

(Replace the timestamp + model name as appropriate.)

### Manually trigger an Airflow DAG

```bash
docker compose exec airflow-scheduler airflow dags trigger DAG_ID
```

### Drain the Redis recommendation queue

```bash
docker compose exec redis redis-cli LLEN recommendations:queue
docker compose exec redis redis-cli DEL recommendations:queue   # nuclear
```

### Enable / disable a sport for recs

Comment out the sport's `generate_recommendations_<sport>` task
dependency in `services/data-ingestion/dags/auspex_pipeline.py` and
restart the airflow scheduler:

```bash
docker compose restart airflow-scheduler
```
