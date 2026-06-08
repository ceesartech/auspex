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

# 2. Rebuild the api image (adds postgresql-client for pg_dump + boto3):
docker compose build api

# 3. (Optional but recommended) Configure offsite upload — see the
#    "Backblaze B2 setup" section below, then continue here.

# 4. Restart the api container so it picks up the new env + bind mount:
docker compose up -d --force-recreate api

# 5. Trigger the DAG manually once to verify end-to-end:
docker compose exec airflow-scheduler airflow dags trigger db_backup_daily

# 6. Verify the dump landed locally:
ls -la /opt/auspex/backups/
#    …and (if B2 configured) remotely — see the verify step in B2 setup.
```

### Backblaze B2 setup (recommended offsite backup)

B2 is the chosen offsite target: ~$0.50/yr for our backup volume,
true DR isolation (separate provider + geography from the Hetzner
prod host, unlike a same-datacenter Hetzner Storage Box), and
S3-compatible so `backup_postgres.py` works with zero code change.

**One-time B2 account + bucket + key (≈10 min, in the B2 web UI):**

1. Sign up at backblaze.com → B2 Cloud Storage (no card required to start).
2. **Buckets → Create a Bucket.** Name it `auspex-backups` (globally
   unique — pick your own if taken). Set **Files in Bucket: Private**.
   Create it.
3. On the bucket page, note the **Endpoint** value, e.g.
   `s3.us-west-004.backblazeb2.com`. The middle token (`us-west-004`)
   is your account region. **A B2 account is single-region** — every
   bucket shares it.
4. **Application Keys → Add a New Application Key.**
   - Name: `pgbackup`
   - Allow access to Bucket(s): select **only** `auspex-backups`
     (least privilege — a leaked key can't touch anything else)
   - Type of Access: **Read and Write**
   - Create. The key is shown **once** — copy both values immediately:
     - `keyID` → this is your `AWS_ACCESS_KEY_ID`
     - `applicationKey` → this is your `AWS_SECRET_ACCESS_KEY`
     (Do NOT use the account Master Key — create the scoped key above.)

**Add to `/opt/auspex/.env` on prod:**

```bash
BACKUP_S3_BUCKET=auspex-backups
BACKUP_S3_ENDPOINT_URL=https://s3.us-west-004.backblazeb2.com   # YOUR endpoint from step 3
AWS_ACCESS_KEY_ID=<B2 keyID>
AWS_SECRET_ACCESS_KEY=<B2 applicationKey>
# AWS_DEFAULT_REGION: NOT needed for B2 — the script auto-derives the
# signing region from the endpoint host. Only set it for real AWS S3.
```

> **Region gotcha (handled automatically):** B2 uses SigV4, which
> requires boto3's signing region to MATCH the region token in the
> endpoint host, or every upload fails with `SignatureDoesNotMatch`.
> `backup_postgres.py` derives the region from `BACKUP_S3_ENDPOINT_URL`
> (`s3.<region>.backblazeb2.com`) so you only have to get the endpoint
> URL right. If you ever see `SignatureDoesNotMatch`, double-check the
> endpoint host matches your bucket's actual region.

> **boto3 version gotcha (handled by pin):** `requirements.txt` pins
> `boto3==1.35.50`, which predates the botocore 1.36 change that turned
> on request checksums and broke B2 ("Unsupported header
> 'x-amz-sdk-checksum-algorithm'"). The script also exports
> `AWS_REQUEST_CHECKSUM_CALCULATION=when_required` as a forward-compat
> safeguard. If you bump boto3 to >=1.36, re-verify a B2 upload works.

**Verify end-to-end after restarting the api container:**

```bash
# Run the backup once and watch for "Upload verified: N bytes":
docker compose exec api python /app/scripts/backup_postgres.py

# Confirm the object is in B2 (uses the same creds):
docker compose exec api python -c "
import boto3, os
s3 = boto3.client('s3', endpoint_url=os.environ['BACKUP_S3_ENDPOINT_URL'])
r = s3.list_objects_v2(Bucket=os.environ['BACKUP_S3_BUCKET'], Prefix='postgres/')
for o in r.get('Contents', []):
    print(o['Key'], o['Size'])
"
```

If the offsite upload fails, the script exits non-zero (so the
`db_backup_daily` DAG goes red and — once Telegram failure alerts are
wired — pings you) while the local copy is still safely written.

### B2 retention (expire old offsite backups)

Set a **Lifecycle Rule** on the bucket in the B2 UI (Buckets →
Lifecycle Settings) to keep, e.g., the last 30 days, then hide +
delete. The local copy already rotates at 7 days
(`BACKUP_LOCAL_RETENTION`). At our backup size, even keeping a full
year on B2 costs well under $1/yr, so retention is about tidiness more
than cost.

### Restore — full disaster recovery

The DB has been wiped or the host is gone. Recovery:

```bash
# 1. Fetch the most recent backup. Either:
#    a) From local disk if the host is intact:
ls -t /opt/auspex/backups/auspex_prod-*.dump | head -1
#    b) From B2 if the host is gone (no aws CLI needed — use boto3 via
#       any machine with the B2 creds in env; download newest object):
python3 - <<'PY'
import boto3, os
s3 = boto3.client("s3", endpoint_url=os.environ["BACKUP_S3_ENDPOINT_URL"])
objs = s3.list_objects_v2(Bucket=os.environ["BACKUP_S3_BUCKET"], Prefix="postgres/").get("Contents", [])
newest = max(objs, key=lambda o: o["LastModified"])
print("downloading", newest["Key"], newest["Size"], "bytes")
s3.download_file(os.environ["BACKUP_S3_BUCKET"], newest["Key"], "/tmp/" + newest["Key"].split("/")[-1])
PY

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
