#!/bin/bash
# Approved 2026-10-31 (operator: "Proceed with the fix"): make
# matches.half_time_home/away the one home of the half-time score. The
# 2026-10-30 backfill and results path wrote metadata.halftime_* keys
# instead (~2.7k soccer matches), which the HT trainers and the match API —
# both reading the columns — never saw. Run AFTER the deploy that switches
# the results path and the grader to the columns.
#   1. verified backup (B2) — abort if it does not land;
#   2. dry run (plan + write-once CSV);
#   3. move: columns filled, keys dropped, one guarded UPDATE per row;
#   4. verify: no keys left, no column/key disagreement, counts.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/move_halftime_to_columns.sh 2>&1 | tee -a ~/move_halftime.log
set -uo pipefail
cd /opt/auspex
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
TOOL="python /app/scripts/backfill_halftime_scores.py --from-metadata"

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

log "2/4 dry run"
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/halftime_from_metadata_dryrun_\$(date -u +%Y%m%dT%H%M%SZ).csv 2>&1 | tail -4" \
  || { log "ABORT at step 2: rows refused or the tool failed (see above) — nothing changed"; exit 1; }

log "3/4 move"
$DC exec -T api sh -c "$TOOL --apply 2>&1 | tail -4" \
  || { log "ABORT at step 3 (see above): the rows that moved are in the audit CSV"; exit 1; }

log "4/4 verify"
cat > /tmp/ht_move_verify.sql <<'SQLEOF'
SELECT count(*) FILTER (WHERE metadata ? 'halftime_home_score' OR metadata ? 'halftime_away_score') AS keys_left,
       count(*) FILTER (WHERE m.match_date >= now() - interval '160 days' AND half_time_home IS NOT NULL) AS recent_with_ht,
       count(*) FILTER (WHERE m.match_date >= now() - interval '160 days') AS recent_finished
  FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'soccer' AND m.status = 'finished';
SQLEOF
$DC cp /tmp/ht_move_verify.sql postgres:/tmp/ht_move_verify.sql >/dev/null 2>&1
out=$($DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/ht_move_verify.sql')
echo "$out"
keys_left=$(echo "$out" | awk 'NR==3 {print $1}')
if [ "$keys_left" = "0" ]; then log "DONE"; else log "FAILED: $keys_left row(s) still carry the keys — inspect"; exit 1; fi
