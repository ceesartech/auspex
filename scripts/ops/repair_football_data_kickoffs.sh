#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Move the football-data kickoffs promote_raw stored 1 h late (UK summer time
# read as UTC) back to UTC and merge the duplicates that left, with
# scripts/repair_football_data_kickoffs.py. Rehearsed 2026-10-31 on a local
# copy of a read-only prod export (all of matches, raw_football_data, leagues,
# teams; the May-Aug 2026 rows of every table that references matches):
#   --dedupe: 68 football-data copies merged into their ESPN rows — exactly the
#             68 same-team-id pairs found read-only (CN1 21, NO1 11, BR1 9, MLS 9,
#             AR1 6, MX1 6, SC0 5, SE1 1); 65 SC0 odds and 136 match_stats rows
#             move to the ESPN rows, the rest is archived (447 records); 26
#             refused: the row beside the copy is a stale 'scheduled' fixture row
#             from before migration 023 (CN1 21, NO1 3, BR1 2; 810 predictions,
#             31 recs on them — repair_fixture_orphans' business, not this one).
#   --shift:  63,718 rows move back 1 h; 26 refused (those same copies: the stale
#             row sits on the corrected kickoff, a UNIQUE collision).
#   after:    check_legacy_kickoffs passes, 0 football-data duplicates; then
#             promote_raw re-promoted all 168,077 raw rows onto the copy with 0
#             rows inserted and 0 refused.
# The 160 pairs where ESPN made its own team row are NOT merged here (team ids
# differ); --shift lands each football-data copy on its ESPN twin's kickoff and
# lists it in the CSV's alias_twin_id (176 flagged, all 160 among them; a
# heuristic: same league, kickoff and score) for the team merge.
#   1. verified backup (B2) — abort if it does not land;
#   2. before-state + dedupe dry run, pinned to the counts above — abort on any other plan;
#   3. dedupe apply (one transaction per 100 copies; every write guarded on the
#      planned values; each batch archived to a gzip JSONL in /app/backups first);
#   4. shift dry run, pinned — abort on any other plan;
#   5. shift apply (one transaction per 5,000 rows, guarded);
#   6. after-state: the tool's --check (check_legacy_kickoffs passes, 0 duplicates)
#      and exactly 68 fewer soccer matches among those that existed at step 2.
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status; a pipe through tail would report tail's 0.
# Re-runnable: a merged copy is gone and a shifted row plans as already_shifted
# (then pin the counts the new dry run reports).
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/repair_football_data_kickoffs.sh 2>&1 | tee -a ~/repair_football_data_kickoffs.log
set -uo pipefail
cd /opt/auspex
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_at() {  # psql_at <local.sql>: one unaligned value
  $DC cp "$1" postgres:/tmp/fd_kickoffs_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/fd_kickoffs_q.sql'
}
# Pinned to the rehearsed plan: any other count aborts before that step writes.
DEDUPE_PLANNED=68
DEDUPE_REFUSED=26
SHIFT_PLANNED=63718
SHIFT_REFUSED=26
TOOL="python /app/scripts/repair_football_data_kickoffs.py"
KEEP="Loaded|Planned|refused and left|ERROR|DRY-RUN|Batch|CHECK|Done:|Traceback|Error"
step() {  # step <label> <log name> <tool args...>: run the tool in the api container, exit with its status
  local label="$1" name="$2"; shift 2
  $DC exec -T api sh -c "$TOOL $* > /tmp/$name.log 2>&1; rc=\$?; grep -E '$KEEP' /tmp/$name.log | grep -v ' refused [A-Z]' | tail -25; echo \"tool exit \$rc\"; exit \$rc"
  local rc=$?
  log "$label exit $rc"
  return $rc
}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

log "1/6 backup"
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

log "2/6 before-state + dedupe dry run"
T0=$(date -u +%FT%TZ)
cat > /tmp/fd_kickoffs_count.sql <<SQLEOF
SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'soccer' AND m.created_at < '$T0';
SQLEOF
soccer_before=$(psql_at /tmp/fd_kickoffs_count.sql | tr -d '[:space:]')
if ! [[ "$soccer_before" =~ ^[0-9]+$ ]]; then log "ABORT at step 2: could not count soccer matches (got '$soccer_before') — nothing changed"; exit 1; fi
log "soccer matches created before $T0: $soccer_before"
step "before-state check (expected to FAIL now)" fd_kickoffs_check_before --check
step "dedupe dry run" fd_kickoffs_dedupe_dry --dedupe --expect-planned $DEDUPE_PLANNED --expect-refused $DEDUPE_REFUSED \
  --out-csv /app/backups/football_data_dedupe_dryrun_$STAMP.csv \
  || { log "ABORT at step 2: the dedupe plan is not the pinned one or the tool failed (see above) — nothing changed"; exit 1; }

log "3/6 dedupe apply"
step "dedupe apply" fd_kickoffs_dedupe --dedupe --apply --expect-planned $DEDUPE_PLANNED --expect-refused $DEDUPE_REFUSED \
  || { log "ABORT at step 3 (full log /tmp/fd_kickoffs_dedupe.log in the api container): a batch that raised rolled back; batches logged as committed stay merged (archives in /app/backups) — inspect before re-running"; exit 1; }

log "4/6 shift dry run"
step "shift dry run" fd_kickoffs_shift_dry --shift --expect-planned $SHIFT_PLANNED --expect-refused $SHIFT_REFUSED \
  --out-csv /app/backups/football_data_shift_dryrun_$STAMP.csv \
  || { log "ABORT at step 4: the shift plan is not the pinned one or the tool failed (see above) — the dedupe is committed, no kickoff moved"; exit 1; }

log "5/6 shift apply"
step "shift apply" fd_kickoffs_shift --shift --apply --expect-planned $SHIFT_PLANNED --expect-refused $SHIFT_REFUSED \
  || { log "ABORT at step 5 (full log /tmp/fd_kickoffs_shift.log in the api container): the failing batch rolled back; batches logged as committed stay shifted — re-run the dry run, re-pin, re-apply"; exit 1; }

log "6/6 after-state"
step "after-state check" fd_kickoffs_check_after --check
check_rc=$?
soccer_after=$(psql_at /tmp/fd_kickoffs_count.sql | tr -d '[:space:]')
expected=$((soccer_before - DEDUPE_PLANNED))
log "soccer matches created before $T0: $soccer_before -> ${soccer_after:-unknown} (expected $expected)"
rm -f /tmp/fd_kickoffs_count.sql
$DC exec -T postgres rm -f /tmp/fd_kickoffs_q.sql >/dev/null 2>&1
if [ "$check_rc" -eq 0 ] && [ "$soccer_after" = "$expected" ]; then
  log "DONE: promote_raw may run again (its check_legacy_kickoffs passes)"
else
  log "FAILED: --check exit $check_rc (needs 0: check_legacy_kickoffs passes, 0 football-data duplicates), soccer rows ${soccer_after:-unknown} vs expected $expected — inspect"
  exit 1
fi
