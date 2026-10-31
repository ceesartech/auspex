#!/bin/bash
# Approved 2026-10-31 (operator: "Proceed"): apply migration 025 —
# betting_recommendations.prediction_id ON DELETE CASCADE -> NO ACTION, so a
# delete that would strand a rec fails loudly instead of deleting it.
#   1. verified backup (B2) — abort if it does not land;
#   2. apply 025 (one transaction, lock_timeout 10s; a timeout changes nothing);
#   3. verify the constraint definition;
#   4. prove it: deleting a prediction a rec references must now FAIL
#      (inside a transaction that is rolled back either way).
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/migrate_025_rec_prediction_fk.sh 2>&1 | tee -a ~/migrate_025.log
set -uo pipefail
cd /opt/auspex
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
psql_file() {
  $DC cp "$1" postgres:/tmp/"$(basename "$1")" >/dev/null 2>&1
  $DC exec -T postgres sh -c "psql -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" $2 -f /tmp/$(basename "$1")"
}
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}

log "1/4 backup"
before=$(date -u +%s)
$DC exec -T airflow-scheduler airflow dags trigger db_backup_daily >/dev/null 2>&1
landed=""
for _ in $(seq 1 40); do
  sleep 30
  l=$(latest_dump); ts=${l%% *}
  if [ -n "$ts" ] && [ "$ts" -gt "$before" ] 2>/dev/null; then landed="$l"; break; fi
done
if [ -z "$landed" ]; then log "ABORT: backup did not land in B2 within 20 min — nothing changed"; exit 1; fi
log "backup landed: $landed"

log "2/4 apply 025"
psql_file services/data-ingestion/db/migrations/025_rec_prediction_fk_no_action.sql "-v ON_ERROR_STOP=1" \
  || { log "ABORT at step 2: migration failed (one transaction — nothing changed); re-run"; exit 1; }

log "3/4 verify the constraint"
cat > /tmp/m025_verify.sql <<'SQLEOF'
SELECT pg_get_constraintdef(oid) AS def FROM pg_constraint
 WHERE conname = 'betting_recommendations_prediction_id_fkey';
SQLEOF
def=$(psql_file /tmp/m025_verify.sql "-tA")
echo "$def"
case "$def" in
  *"REFERENCES predictions(id)"*CASCADE*) log "ABORT: constraint still cascades"; exit 1 ;;
  *"REFERENCES predictions(id)"*) ;;
  *) log "ABORT: constraint missing"; exit 1 ;;
esac

log "4/4 prove a stranding delete now fails (rolled back)"
cat > /tmp/m025_prove.sql <<'SQLEOF'
BEGIN;
DELETE FROM predictions WHERE id = (SELECT prediction_id FROM betting_recommendations ORDER BY created_at LIMIT 1);
ROLLBACK;
SQLEOF
out=$(psql_file /tmp/m025_prove.sql "" 2>&1)
echo "$out"
case "$out" in
  *"violates foreign key constraint"*) log "DONE: the delete was refused" ;;
  *) log "FAILED: the delete was NOT refused — inspect now"; exit 1 ;;
esac
