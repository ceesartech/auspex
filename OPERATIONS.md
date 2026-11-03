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
   Files named `<database>-YYYY-MM-DDTHHMMSSZ.dump`, on prod
   `betting_system-…dump`.
2. Uploads it to Backblaze B2 when `BACKUP_S3_BUCKET` is set (prod: yes)
   and HEAD-verifies the upload.
3. Rotates the local dumps: never the one it just wrote; with B2, keeps at
   most `BACKUP_LOCAL_KEEP` (3) and none older than 7 days, deleting an older
   one only after a HEAD on its own B2 copy shows the same byte size; without
   B2, only the 7-day age rule. A dump that never reached B2 (its upload
   failed, or the run used `--skip-upload`) answers the HEAD with a 404:
   rotation then uploads it, verifies the size and only then deletes it, so
   one failed upload does not turn later runs red. Anything else it cannot
   verify (B2 unreachable, an auth error, a copy of a different size, a failed
   re-upload or delete) keeps the file, logs an ERROR naming it, and exits 3.
   Every ops wrapper triggers this DAG for its pre-write backup, so a busy
   ops day used to pile up 20+ dumps under the age-only rule. On such a day an
   earlier wrapper's pre-write dump leaves the VM within hours: a reversal
   that needs a dump older than the newest 3 downloads it from B2 first
   (restore option b below, with that dump's key instead of the newest).

### When db_backup_daily fails

The task log says which exit code and why (`docker compose exec -T
airflow-scheduler airflow dags list-runs db_backup_daily -o json`, then the
task log in the Airflow UI). The DAG retries once after 10 minutes with a
fresh dump.

| Exit | Meaning | What to do |
|---|---|---|
| 1 | `pg_dump` failed (no dump kept), or `DATABASE_URL` is unset | Read the pg_dump stderr tail in the log; check postgres is up. |
| 2 | The dump is local but its upload to B2 failed | Check B2 (credentials in `.env`, endpoint, the B2 status page). Nothing else: the next run's rotation uploads this dump before it deletes it. |
| 3 | The new dump is written and uploaded, but rotation kept an older dump it would have deleted | Each `KEPT <file> (…) — …` ERROR line says why. `HEAD … failed: …` (not a 404) or `the re-upload failed`: B2 was unreachable or refused — fix that, then trigger `db_backup_daily` again. `… is N bytes, the local file M`: the B2 copy differs and is never overwritten automatically — check the local file is a readable dump (`docker compose exec -T api pg_restore --list /app/backups/<file> > /dev/null && echo READABLE`); once a newer dump is confirmed in B2 (the listing below), delete the local file (`rm -v /opt/auspex/backups/<file>`) and trigger the DAG. `the delete failed`: fix the file's ownership (`ls -l /opt/auspex/backups`). |
| 4 | `BACKUP_LOCAL_KEEP` or `BACKUP_LOCAL_RETENTION` is not a valid number (no dump was taken) | Fix the value in `.env`, recreate the api container (`export IMAGE_TAG=$(docker inspect betting-api --format '{{.Config.Image}}' \| cut -d: -f2) && docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.ghcr.yml up -d --pull never api`), trigger the DAG. |

A mistyped command-line flag in a manual run also exits 2, but argparse
prints `usage:` first; the DAG passes no flags.

The newest three dumps in B2, with sizes (read-only; it uses the script's own
client, so the signing region is derived from the endpoint):

```bash
docker compose exec -T api python -c "import os,sys; sys.path.insert(0,'/app/scripts'); import backup_postgres as b; s3=b.make_s3_client(os.environ.get('BACKUP_S3_ENDPOINT_URL')); o=[x for p in s3.get_paginator('list_objects_v2').paginate(Bucket=os.environ['BACKUP_S3_BUCKET'],Prefix=os.environ.get('BACKUP_S3_PREFIX','postgres/')) for x in p.get('Contents',[])]; [print(x['Key'],x['Size'],x['LastModified']) for x in sorted(o,key=lambda x:x['LastModified'])[-3:]]"
```

### Initial setup on a fresh host

```bash
# 1. Create the host backups dir owned by the api container's appuser.
#    NOTE: Dockerfile.api uses `useradd -r`, which allocates a SYSTEM uid
#    (999 today, NOT 1000). Hardcoding 1000 broke the first-ever backup
#    run (2026-07-13, "Permission denied") — always resolve it live:
sudo mkdir -p /opt/auspex/backups
sudo chown -R "$(docker compose exec -T api id -u | tr -d '[:space:]'):$(docker compose exec -T api id -g | tr -d '[:space:]')" /opt/auspex/backups

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
delete. The local copies rotate to the newest 3 (`BACKUP_LOCAL_KEEP`),
each older one only after its B2 copy is verified — so a B2 rule must
never expire an object younger than a week: rotation would find it missing
and upload the local file again before deleting it. At our backup size, even keeping a full
year on B2 costs well under $1/yr, so retention is about tidiness more
than cost. (2026-11-02: 148 dumps, 44 GB, the oldest from 2026-07-13 —
so no expiry rule is in effect.)

### Restore — full disaster recovery

The DB has been wiped or the host is gone. Recovery:

```bash
# 1. Fetch the most recent backup. Either:
#    a) From local disk if the host is intact (only the newest 3 are
#       local; older ones are in B2):
ls -t /opt/auspex/backups/betting_system-*.dump | head -1
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
    < /tmp/betting_system-2026-11-02T020001Z.dump

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
    < /opt/auspex/backups/betting_system-2026-11-02T020001Z.dump
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
git pull      # see "Manual git pull needs a token" below
docker compose cp services/data-ingestion/db/migrations/0XX_thing.sql postgres:/tmp/0XX.sql
docker compose exec -T postgres sh -c \
    'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/0XX.sql'
```

### Manual `git pull` on the VM needs a token

Since 2026-09-02 GitHub answers anonymous `git-upload-pack` POSTs from cloud
IP ranges with **401 even for public repos** (the ref advertisement still
returns 200, so `curl` looks fine while `git fetch` dies with "could not read
Username"). CI's `deploy_remote.sh` authenticates automatically with the
ephemeral `GITHUB_TOKEN`; a human pulling by hand needs a fine-grained PAT
(Contents: read). One-shot, nothing written to disk:

```bash
read -rs TOKEN    # paste the PAT; never echo it
git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $(printf 'x-access-token:%s' "$TOKEN" | base64 -w0)" pull --ff-only
unset TOKEN
```

(A read-only deploy key would be cleaner but needs repo-admin on the org.)

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

## Disk maintenance

The VM has one 75 GB root filesystem (4 GB of it a swapfile). On 2026-11-02 it
reached 95% used and `DiskSpaceLow` fired. Four growers had stacked up: 37 local
pg_dumps (14 GB — the rotation was by age only, and every ops wrapper takes a
backup, 21 of them that day), 7.4 GB of Airflow task logs (kept 30 days), a week
of untagged `:latest` leftovers among the docker images, and a 7.8 GB Prometheus
store (30 days, no size cap, swollen by an api label blow-up). Each now has a cap.

### Disk budget

Measured read-only on 2026-11-02 at ~23:40Z, after the one-off clean-up below.

| Grower | Now | Rate | Cap / guard | Steady state |
|---|---|---|---|---|
| Docker images (`/var/lib/containerd`; Docker 29 uses the containerd image store) | 12 GB | a deploy adds ~10-15 MB (consecutive deploys share ~4.4 GB of layers); a dependency change adds ~8 GB; re-pulling a `:latest` third-party image leaves the old one untagged (grafana: 2 GB) | daily `disk_maintenance` → `prune_docker_images`: keeps every image a container uses and the newest 3 deploys (the rollback target), removes the rest once 24 h old (`scripts/prune_docker_images.py`) | ~12-13 GB; ≤ ~22 GB while a dependency change is inside the newest 3 |
| Postgres volume | 6.6 GB (database 6.1 GB, WAL 0.2 GB with `max_wal_size` 1 GB) | the dump grew 334 → 390 MB in October (+17%) | none — it is the data. The headroom below is its runway | grows |
| Prometheus volume | 7.8 GB → ≤ 4 GiB once the deploy recreates Prometheus | ~55-60 MB/day healthy; ~700 MB/day on Oct 21-25 (the api labelled metrics with raw request paths; fixed 2026-11-02) | `--storage.tsdb.retention.time=30d` and `--storage.tsdb.retention.size=4GB` (whichever binds first; oldest blocks go first) | ~1.8 GB (30 days) |
| Airflow task logs (`services/data-ingestion/logs`) | 3.1 GB | ~200 MB/day (120-290), ~3,300 files/day, 96% `auspex_pipeline` | daily `disk_maintenance` → `prune_airflow_task_logs`: log files older than 14 days, rotated `*.log.N` included, then directories empty for an hour | ~3 GB (≤ 4 GB) |
| journald (`/var/log/journal`) | 4.0 GB | ~40 MB/day, mostly sshd brute-force noise | systemd's default cap: the smaller of 10% of the filesystem and 4 GB | 4 GB — or 1 GB with the optional drop-in below |
| Local pg_dumps (`/opt/auspex/backups/betting_system-*.dump`) | 1.1 GB (3 × ~380 MB) | one per backup run: daily 02:00, plus one per ops wrapper | `scripts/backup_postgres.py`: never the dump just written; the newest 3 (`BACKUP_LOCAL_KEEP`) and none past 7 days, each older one deleted only after its B2 copy is verified | ~1.2 GB, growing with the database |
| Other files in `backups/` | 0.75 GB: `odds_hist_cache/` 0.49 GB (the October odds backfill cache), audit CSVs and reversal archives 0.25 GB | a few MB to 60 MB per ops run | none: audit records are write-once and kept on purpose | grows with ops work |
| Container json logs | 0.15 GB | — | `max-size 50m` × `max-file 3` per container (17) | ≤ 2.5 GB |
| Grafana volume | 0.6 GB (plugins) | — | bounded | 0.6 GB |
| Models (`/opt/auspex/models`) | 0.5 GB | `modal-incoming/` +~18 MB per weekly retrain | none | +~1 GB/year |
| Swapfile, OS, `/usr`, `/boot`, other `/var` | ~7 GB | — | fixed | ~7 GB |
| Airflow metadata rows | inside Postgres | ~1,500 task runs/day | monthly `airflow_db_maintenance`: `airflow db clean` of rows older than 90 days | — |

Sizes are as `du -h` / `df -h` print them (G = GiB).

**Where the alerts fire, in `df -h /` terms.** The filesystem is 74.8G
(`df -B1` on 2026-11-03: size 80,290,492,416, used 45,670,158,336, avail
31,292,731,392 bytes); the 3.1G that is neither used nor available is ext4's
root-reserved space, so `Used` + `Avail` is only 71.7G. The alerts divide `node_filesystem_avail_bytes`
(which leaves out the reserved blocks) by the full size:

| Alert | Fires when | `df -h /` Used at that point |
|---|---|---|
| `DiskSpaceWarning` (warning, Telegram) | Avail < 20% of 74.8G = 15.0G, for 30m | ≈ 57G |
| `DiskSpaceLow` (critical, repeats every 15m) | Avail < 10% = 7.5G, for 15m | ≈ 64G |

Alertmanager mutes the warning while the critical fires. (2026-11-02 at the
incident: 68G used, 3.7G avail.)

**Totals.** Steady state ≈ 38G used (~53% in df's Use% column). The worst case — every
capped grower at its cap at once and a dependency change inside the newest 3
deploys — is ≈ 54G: only ≈ 3G under the ≈ 57G warning line. The journald drop-in
below (4G → 1G) brings that to ≈ 51G and ≈ 6G of margin, and is recommended for
that reason. The ≈ 19G between steady state and the warning line is Postgres's
runway; re-measure each quarter with the commands below.

### When DiskSpaceWarning or DiskSpaceLow fires

1. Find the grower (read-only):
   ```bash
   df -h /
   sudo du -xsh /var/lib/containerd /var/lib/docker/volumes /opt/auspex/backups \
       /opt/auspex/services/data-ingestion/logs /var/log/journal /opt/auspex/models /tmp
   docker system df
   ```
2. Check the guards ran: the `disk_maintenance` DAG (daily 05:15 UTC) and
   `db_backup_daily`. A backup run that exits 3 kept a dump whose B2 copy it
   could not verify; its task log names the file in an ERROR line, and
   "When db_backup_daily fails" above says what to do.
   ```bash
   docker compose exec -T airflow-scheduler airflow dags list-runs disk_maintenance -o json
   docker compose exec -T airflow-scheduler airflow dags list-runs db_backup_daily -o json
   ```
3. Clean up by hand only what a guard did not cover. These are the commands
   run on 2026-11-02; each is safe to repeat:
   - **Local dumps.** Normally just trigger `db_backup_daily`: its rotation
     verifies each older dump in B2 before deleting it. The blunt manual form,
     only once you have confirmed every dump is in B2:
     `ls -1t /opt/auspex/backups/betting_system-*.dump | tail -n +4 | xargs -r rm -v`
   - **Airflow task logs** older than 14 days. Simplest: trigger the guard,
     `docker compose exec -T airflow-scheduler airflow dags trigger disk_maintenance`.
     By hand, run it inside the scheduler: the log directories belong to the
     container's `airflow` user (uid 50000, mode 775), so the host's `auspex`
     user cannot delete there.
     `docker compose exec -T airflow-scheduler find /opt/airflow/logs -type f \( -name '*.log' -o -name '*.log.[0-9]*' \) -mtime +14 -delete`
   - **Docker images.** Preview the guarded prune, then apply it:
     ```bash
     docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.ghcr.yml \
         exec -T airflow-scheduler python /opt/auspex/scripts/prune_docker_images.py          # dry run
     docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.ghcr.yml \
         exec -T airflow-scheduler python /opt/auspex/scripts/prune_docker_images.py --apply
     ```
     Avoid `docker image prune -af --filter until=24h` (used on 2026-11-02)
     except in an emergency: with the containerd store `until` is the pull
     time, so it also deletes the previous deploys — the local rollback target —
     once they are a day old. GHCR still has every tag, so a rollback can re-pull.
   - **journald.** `sudo journalctl --vacuum-size=1G` frees ~3 GB at once; make
     it stick with the drop-in (recommended: it is most of the margin under the
     warning line in the worst case above):
     `sudo mkdir -p /etc/systemd/journald.conf.d && printf '[Journal]\nSystemMaxUse=1G\n' | sudo tee /etc/systemd/journald.conf.d/auspex-size.conf && sudo systemctl restart systemd-journald`
   - **Prometheus.** If the TSDB is the grower, look for a label blow-up in the
     Prometheus UI: `topk(10, count by (job, __name__) ({__name__=~".+"}))`.

> History: the audit-era hardening intended a host crontab for the image prune;
> the 2026-07-14 verification pass found no crontab entry on the VM (~10 GB of
> reclaimable layers had accumulated). It was replaced by the weekly
> `docker_maintenance` DAG, and on 2026-11-02 the image and log prunes moved to
> the daily `disk_maintenance` DAG (the weekly one keeps the build-cache and B2
> modal-artifact prunes).

Manual spot-checks:

```bash
df -h /                      # host disk
docker system df             # images / volumes / build-cache split
du -sh /opt/auspex/.git      # deploy-stash bloat (see backups note below)
```

> Backup files in `/opt/auspex/backups/` MUST stay gitignored (`/backups/*` in
> .gitignore). deploy_remote.sh auto-stashes untracked files before pulling —
> un-ignored dumps get silently stashed into `.git` on every deploy (found
> 2026-07-14: `.git` had grown to 811 MB and local dumps kept vanishing).

## GitHub costs

The repo is **public**: GitHub Actions minutes and GHCR image storage are free.
If it is ever flipped private, both start billing (Actions per-minute after the
free tier, GHCR per-GB) — CI builds 3 images per merge, so revisit the CI
matrix and image retention before going private.

## Modal training (serverless retraining)

> **Doing the cutover?** Follow the explicit step-by-step runbook in
> [`docs/MODAL_CUTOVER.md`](docs/MODAL_CUTOVER.md) — B2-key rotation, Modal
> secret creation, token wiring, smoke → shadow → cut over, and rollback, with
> every command. This section is the summary.

Model training can run on the VM (in the api container) or on **Modal** (13
bundles in parallel, each its own container). The switch is one Airflow Variable
— nothing is hardcoded, and the VM path stays as a fallback.

| | VM backend (default) | Modal backend |
|---|---|---|
| Flag | `training_backend=vm` | `training_backend=modal` |
| Where | api container, ~90 min sequential | Modal, ~90 s parallel, ~1¢/run |
| Code | `retrain_models` DAG `train_<sport>` tasks | `modal_train/train_modal.py` (13 named fns) |

**Flip the backend (live, no deploy):**
```bash
docker compose exec -T airflow-scheduler airflow variables set training_backend modal
# roll back instantly:
docker compose exec -T airflow-scheduler airflow variables set training_backend vm
```
The scheduler re-parses within ~30 s; the DAG graph shows only the active path.

**One-time setup before the first Modal run:**
1. **Rotate the B2 app key** (the trial shared one in chat). Create a fresh
   bucket-scoped B2 application key; put it in the VM `.env` (`AWS_ACCESS_KEY_ID`
   / `AWS_SECRET_ACCESS_KEY`) AND the Modal secret below; revoke the old key.
2. **Create the Modal secrets** (from your own values, run locally with the
   Modal CLI authed):
   ```bash
   modal secret create auspex-b2 \
     AWS_ACCESS_KEY_ID=<b2-keyID> AWS_SECRET_ACCESS_KEY=<b2-appKey> \
     BACKUP_S3_BUCKET=<bucket> BACKUP_S3_ENDPOINT_URL=<https://s3.<region>.backblazeb2.com>
   modal secret create auspex-telegram \
     TELEGRAM_BOT_TOKEN=<token> TELEGRAM_CHAT_ID=<chat> ENABLE_TELEGRAM_NOTIFICATIONS=true
   ```
3. **Set `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`** in the VM `.env` (read by the
   airflow-scheduler for the `modal_trigger` task) and recreate the scheduler.

**Staged rollout (do not skip):**
1. **Smoke one bundle** (from a repo checkout with the Modal CLI authed):
   `modal run modal_train/train_modal.py --run-id smoke1 --bundles soccer_match_result`
   — confirm `soccer_match_result_training` is its own Modal function, served
   Brier ≈ 0.594–0.596, artifacts under `modal-train/smoke1/`.
2. **Shadow a full run** (gate + report, NO swap): trigger all 13, then on the VM
   `docker compose exec -T api python /app/scripts/pull_modal_artifacts.py --run-id <id> --shadow`
   and eyeball `promote_decisions.json` vs live Brier from `monitor_models.py`.
   This seeds the incumbent `held_out_metrics.json` sidecars.
3. **Cut over** one real Sunday run behind `training_backend=modal`; verify the
   api reloaded, recs still generate, and `production-prev-<stamp>` exists.
4. Bake ~1 month with the VM fallback retained; then delete `modal_trial/` + its
   Modal volumes (`modal volume rm auspex-trial-data --yes`, `auspex-trial-models`).

**The promote-gate.** `scripts/pull_modal_artifacts.py` only stages a bundle if
its served held-out Brier ≤ the incumbent's + 0.009 (the noise floor). Rejected
bundles keep their current production model (never enter staging). Decisions land
in `/app/models/staging/promote_decisions.json`. **Force-promote** a rejected
bundle: delete the incumbent sidecar
(`rm /app/models/production/<ensemble_name>/held_out_metrics.json`) so the next
run treats it as un-gated (promote-to-seed), or copy the bundle's tree from
`/app/models/modal-incoming/<run_id>/<bundle>/` into `/app/models/staging/`
manually and re-run swap.

**B2 layout:** dump in at `postgres/<db>-<ISO>.dump`; artifacts out at
`modal-train/<run_id>/<bundle>/…`. A prep step caches the dump in a Modal volume
so each run pulls it once (not 13×). `modal-train/` artifacts are pruned
automatically by the weekly `docker_maintenance` DAG (`prune_b2_modal_artifacts`
task, deletes >14 days). Run it by hand anytime:
```bash
dc exec -T api python /app/scripts/prune_modal_artifacts.py --days 14 --dry-run
dc exec -T api python /app/scripts/prune_modal_artifacts.py --days 14
```
(A direct version-aware prune, not an S3 lifecycle rule — B2's versioned buckets
reject a plain `Expiration.Days` rule.)

**Rollback:** flip `training_backend=vm` (training returns to the VM with zero
code change), and `production-prev-<stamp>` in `/app/models/` restores the prior
models (stop api → restore dir → start api).
