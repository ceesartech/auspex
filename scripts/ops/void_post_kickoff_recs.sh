#!/bin/bash
# Approved 2026-10-31 (operator: "fix anything that needs fixing"); data
# repairs need a verified backup first, which step 1 takes.
# Void the settled recommendations written at or after their fixture's real
# kickoff: bets placed after kickoff are hindsight, never won/lost. Verified
# read-only 2026-10-30 (the tool's own SELECT, --sport all):
#   soccer 20 rows on 5 fixtures — 8 won +476.37 / 12 lost -704.62 = -228.25 (stake 1466.36)
#   mma     2 rows               — 1 won  +31.41 /  1 lost  -40.21 =   -8.80 (stake   74.35)
#   (+1 soccer push already void at 0.00: counted, not rewritten)
# All 22 were settled 2026-09-04 17:36Z by repair_stranded_recs.py from a
# finished twin whose kickoff ESPN had moved earlier; each was written 46 min
# to 24 h after that kickoff (ESPN's scoreboard confirms every kickoff/score).
# -> status 'void', profit_loss 0, actual_result
#    'void: written after kickoff (was <status> <pl>)'; the book's net P&L
#    moves by +237.05. Deploy the settlement guard FIRST (same commit) so
#    grade_completed_matches / repair_stranded_recs settle no new ones.
#   1. verified backup (B2) — abort if it does not land;
#   2. dry run (plan + write-once CSV) — abort if the tool fails;
#   3. apply (one transaction, every UPDATE guarded on the observed values);
#   4. after-state + verify (no post-kickoff rec still carries a verdict or money).
# Re-runnable: a voided row plans as already_void and is not written again.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/void_post_kickoff_recs.sh 2>&1 | tee -a ~/void_post_kickoff_recs.log
set -uo pipefail
cd /opt/auspex
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
# Pinned to the reviewed set: any drift (more rows than the 22 reviewed) aborts
# at the dry run, before anything is written.
EXPECTED=22
TOOL="python /app/scripts/void_post_kickoff_recs.py --sport all --max-rows $EXPECTED"

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
# Exit status through a file, not a pipe: `| tail` would report tail's 0.
# The per-rec lines stay in the container log; the CSV is the record.
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/void_post_kickoff_recs_dryrun_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/void_pk_dry.log 2>&1; rc=\$?; grep -v -E ' (void|already_void) rec ' /tmp/void_pk_dry.log | tail -15; exit \$rc" \
  || { log "ABORT at step 2: the tool failed or refused (see above) — nothing changed"; exit 1; }

log "3/4 apply"
$DC exec -T api sh -c "$TOOL --apply > /tmp/void_pk.log 2>&1; rc=\$?; grep -v -E ' (void|already_void) rec ' /tmp/void_pk.log | tail -15; exit \$rc" \
  || { log "ABORT at step 3 (see above): the transaction rolled back — nothing changed"; exit 1; }

log "4/4 after-state"
cat > /tmp/void_pk_after.sql <<'SQLEOF'
SELECT l.sport, br.status, count(*) AS n, COALESCE(sum(br.profit_loss), 0) AS pl
  FROM betting_recommendations br
  JOIN matches m ON m.id = br.match_id
  JOIN leagues l ON l.id = m.league_id
 WHERE br.created_at >= m.match_date AND br.status IN ('won', 'lost', 'void')
 GROUP BY 1, 2 ORDER BY 1, 2;
SELECT count(*) AS relabelled
  FROM betting_recommendations WHERE actual_result LIKE 'void: written after kickoff (was %';
SELECT count(*) AS post_kickoff_with_money
  FROM betting_recommendations br JOIN matches m ON m.id = br.match_id
 WHERE br.created_at >= m.match_date
   AND (br.status IN ('won', 'lost') OR (br.status = 'void' AND br.profit_loss IS DISTINCT FROM 0));
SQLEOF
$DC cp /tmp/void_pk_after.sql postgres:/tmp/void_pk_after.sql >/dev/null 2>&1
out=$($DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/void_pk_after.sql')
echo "$out"
left=$(echo "$out" | awk '/post_kickoff_with_money/ {getline; getline; print $1}')
relabelled=$(echo "$out" | awk '/relabelled/ {getline; getline; print $1}')
if [ "$left" = "0" ] && [ "$relabelled" = "$EXPECTED" ]; then
  log "DONE: $relabelled rec(s) carry the post-kickoff void label; none still carries a verdict or money"
else
  log "FAILED: $left post-kickoff rec(s) still won/lost or carrying money, $relabelled relabelled (expected $EXPECTED) — inspect"
  exit 1
fi
