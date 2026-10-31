#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Move the football-data kickoffs promote_raw stored 1 h late (UK summer time
# read as UTC) back to UTC and merge the duplicates that left, with
# scripts/repair_football_data_kickoffs.py. promote_raw STAYS BLOCKED
# (check_legacy_kickoffs refuses every run while a row sits at a legacy
# kickoff) until this has run and its --check passes; then it may run again.
#
# RUN ORDER (decided 2026-10-31):
#   1. scripts/ops/merge_soccer_teams.sh   (its own approval) — FIRST. It folds
#      the 160 pairs where ESPN filed a fixture under its own team rows (it keeps
#      the ESPN row and deletes ~171 football-data rows; promote_raw then writes
#      those fixtures onto the surviving ESPN row, never beside it).
#   2. this script.
#   3. scripts/ops/cleanup_soccer_orphans.sh (its own approval) — AFTER: the 26
#      stale rows step 3 here folds are finished by then and leave its set.
# If the merge has NOT run, the 160 alias pairs are left for it: --shift refuses
# every copy whose new kickoff is the exact kickoff of the same fixture under
# other team ids (189 such refusals on the rehearsal), so step 2 aborts with
# nothing written. Never shift beside them: the merge would then collide on
# UNIQUE(home_team_id, away_team_id, match_date).
# PREREQUISITE found by the 2026-10-31 rehearsal: ESPN filed 36 Liga MX rows
# under Santos (the Brazilian club's row) and Guadalajara (the Spanish club's
# row). The team merge does not touch them (different clubs, not spellings).
# Until a team-identity repair re-points them to Santos Laguna / Guadalajara
# Chivas, --shift refuses the 6 football-data copies beside them (alias slot)
# and step 2 aborts; once re-pointed, --dedupe merges those 6 like the 68.
#
# Reviewed plan (rehearsed 2026-10-31 on a local copy of a read-only prod
# export: all soccer matches, raw_football_data, leagues, teams, soccer odds and
# match_stats, and the 2026-04-01+ soccer predictions, recs, snapshots and
# features), the bounds below:
#   before the merge:        dedupe merge=68 fold=26 refuse=0; shift 63,718 after it
#   merge + MX1 re-point:    dedupe merge=74 fold=26 refuse=0; shift 63,533 refuse=0
#   --dedupe merge: a copy into its ESPN row (same team ids, same final score;
#            65 SC0 odds and 136 match_stats move);
#   --dedupe fold: a copy into the stale pre-023 'scheduled' row at exactly its
#            corrected kickoff (CN1 21, NO1 3, BR1 2): the row takes the copy's
#            score and turns 'finished', guarded on the values observed, no
#            open rec (on it or its predictions) and no post-kickoff
#            prediction. The outcome trigger then grades 138 of its 810
#            predictions (match_result 54, over_under 42, btts 42; the same
#            under migration 011 and 027) and settles nothing: its 31 recs are
#            all 'void' already. 52 match_stats move.
#   --shift: every other BST row moves back 1 h.
#   after:   check_legacy_kickoffs passes, 0 football-data duplicates; a full
#            promote_raw re-run inserted 0 rows, refused 0, found 0 duplicates.
#   1. verified backup (B2) — abort if it does not land;
#   2. before-state, --check (expected to FAIL now), dedupe dry run and shift
#      dry run — abort unless both plans are within the reviewed bounds with 0
#      refusals, and the shift sees exactly the dedupe's merges + folds as still
#      to do (nothing is written before this passes);
#   3. dedupe apply, pinned to step 2's own counts (--expect-plan: any drift
#      refuses before a write); each batch archives (gzip JSONL in /app/backups)
#      before its first write, one transaction per 100 copies;
#   4. shift dry run again — abort unless it plans exactly step 2's shifts with
#      0 refusals and nothing left to dedupe;
#   5. shift apply, pinned to those counts (one transaction per 5,000 rows);
#   6. after-state, derived from the dry runs: --check MUST pass (exit 0),
#      soccer rows created before T0 fall by exactly merges + folds, stale
#      'scheduled' soccer rows (kickoff > 7 days before T0) by exactly folds,
#      and fresh dry runs plan nothing.
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status; a pipe through tail would report tail's 0.
# Re-runnable: merged and folded copies are gone, shifted rows plan as done.
# AFTERWARDS (optional, separate approval): the trigger grades only the market
# types it knows; the folded rows' other predictions stay ungraded until
#   docker compose exec -T api python /app/scripts/grade_completed_matches.py --match-id <id>
# runs for each (step 6 prints the ids; no rec is open on them, so it settles none).
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/repair_football_data_kickoffs.sh 2>&1 | tee -a ~/repair_football_data_kickoffs.log
set -uo pipefail
cd /opt/auspex
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_at() {  # psql_at <local.sql>: unaligned, '|'-separated
  $DC cp "$1" postgres:/tmp/fd_kickoffs_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/fd_kickoffs_q.sql'
}
plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o " $2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
# The reviewed plan's upper bounds (see header). The team merge that runs first
# lowers the shift count; the MX1 re-point raises the merges from 68 to 74.
MAX_MERGES=74
MAX_FOLDS=26
MAX_SHIFTS=63718
TOOL="python /app/scripts/repair_football_data_kickoffs.py"
KEEP="Loaded|Planned|Folds|Folded|PLAN|refused and left|ERROR|DRY-RUN|Batch|CHECK|Done:|Traceback|Error"
step() {  # step <label> <log name> <tool args...>: run the tool in the api container, exit with its status
  local label="$1" name="$2"; shift 2
  $DC exec -T api sh -c "$TOOL $* > /tmp/$name.log 2>&1; rc=\$?; grep -E '$KEEP' /tmp/$name.log | grep -v ' refused [A-Z]' | tail -25; echo \"tool exit \$rc\"; exit \$rc"
  local rc=$?
  log "$label exit $rc"
  return $rc
}
plan_of() {  # plan_of <log name>: the PLAN line that run logged
  $DC exec -T api sh -c "grep -o 'PLAN .*' /tmp/$1.log | tail -1"
}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
T0=$(date -u +%FT%TZ)
cat > /tmp/fd_kickoffs_state.sql <<SQLEOF
SELECT (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'soccer' AND m.created_at < TIMESTAMPTZ '$T0') AS soccer,
       (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'soccer' AND m.status = 'scheduled' AND m.created_at < TIMESTAMPTZ '$T0'
           AND m.match_date < TIMESTAMPTZ '$T0' - interval '7 days') AS stale;
SQLEOF

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

log "2/6 before-state + dry runs"
state=$(psql_at /tmp/fd_kickoffs_state.sql | tr -d '[:space:]')
IFS='|' read -r soccer0 stale0 <<< "$state"
if ! [[ "$soccer0" =~ ^[0-9]+$ && "$stale0" =~ ^[0-9]+$ ]]; then
  log "ABORT at step 2: the before-state query failed (got '$state') — nothing changed"; exit 1
fi
log "before (created before $T0): soccer rows=$soccer0, stale 'scheduled' soccer rows=$stale0"
step "before-state check (expected to FAIL now)" fd_kickoffs_check_before --check
step "dedupe dry run" fd_kickoffs_dedupe_dry --dedupe --out-csv /app/backups/football_data_dedupe_dryrun_$STAMP.csv
rc=$?
dplan=$(plan_of fd_kickoffs_dedupe_dry)
step "shift dry run" fd_kickoffs_shift_dry --shift --out-csv /app/backups/football_data_shift_dryrun_$STAMP.csv
src=$?
splan=$(plan_of fd_kickoffs_shift_dry)
m=$(plan_num "$dplan" merge); f=$(plan_num "$dplan" fold); dr=$(plan_num "$dplan" refuse)
s=$(plan_num "$splan" shift); sr=$(plan_num "$splan" refuse); pend=$(plan_num "$splan" dedupe_pending)
if ! [[ "$m $f $dr $s $sr $pend" =~ ^[0-9]+\ [0-9]+\ [0-9]+\ [0-9]+\ [0-9]+\ [0-9]+$ ]]; then
  log "ABORT at step 2: could not read the PLAN lines (dedupe exit $rc '$dplan'; shift exit $src '$splan') — nothing changed"; exit 1
fi
if [ "$dr" -ne 0 ] || [ "$sr" -ne 0 ]; then
  log "ABORT at step 2: $dr dedupe and $sr shift refusal(s) — see the dry-run CSVs in /app/backups (reason 'alias slot': run the team merge / the MX1 re-point first, see header) — nothing changed"; exit 1
fi
if [ "$m" -gt "$MAX_MERGES" ] || [ "$f" -gt "$MAX_FOLDS" ] || [ "$s" -gt "$MAX_SHIFTS" ] || [ "$pend" -ne $((m + f)) ]; then
  log "ABORT at step 2: plan (merge=$m fold=$f shift=$s dedupe_pending=$pend) is outside the reviewed set (merge <= $MAX_MERGES, fold <= $MAX_FOLDS, shift <= $MAX_SHIFTS, dedupe_pending = merge + fold) — nothing changed"; exit 1
fi
log "plan: merge=$m fold=$f shift=$s"

log "3/6 dedupe apply (pinned to the dry run's counts)"
step "dedupe apply" fd_kickoffs_dedupe --dedupe --apply --expect-plan merge=$m,fold=$f,refuse=0 \
  --out-csv /app/backups/football_data_dedupe_apply_$STAMP.csv \
  || { log "ABORT at step 3 (full log /tmp/fd_kickoffs_dedupe.log in the api container): if it refused before 'Batch', nothing changed; a batch that raised rolled back, batches logged as committed stay merged (archives in /app/backups) — inspect before re-running"; exit 1; }

log "4/6 shift dry run again"
step "shift dry run" fd_kickoffs_shift_dry2 --shift --out-csv /app/backups/football_data_shift_dryrun2_$STAMP.csv
splan2=$(plan_of fd_kickoffs_shift_dry2)
s2=$(plan_num "$splan2" shift); sr2=$(plan_num "$splan2" refuse); pend2=$(plan_num "$splan2" dedupe_pending)
if [ "$s2 $sr2 $pend2" != "$s 0 0" ]; then
  log "ABORT at step 4: the shift plan is now '$splan2', expected shift=$s refuse=0 dedupe_pending=0 — the dedupe is committed, no kickoff moved; inspect"; exit 1
fi

log "5/6 shift apply (pinned)"
step "shift apply" fd_kickoffs_shift --shift --apply --expect-plan shift=$s,refuse=0 \
  --out-csv /app/backups/football_data_shift_apply_$STAMP.csv \
  || { log "ABORT at step 5 (full log /tmp/fd_kickoffs_shift.log in the api container): the failing batch rolled back; batches logged as committed stay shifted — re-run this script (it re-plans)"; exit 1; }

log "6/6 after-state"
step "after-state check" fd_kickoffs_check_after --check
check_rc=$?
state=$(psql_at /tmp/fd_kickoffs_state.sql | tr -d '[:space:]')
IFS='|' read -r soccer1 stale1 <<< "$state"
want_soccer=$((soccer0 - m - f)); want_stale=$((stale0 - f))
log "after: soccer rows=${soccer1:-unknown} (expected $want_soccer), stale 'scheduled' soccer rows=${stale1:-unknown} (expected $want_stale)"
step "dedupe re-plan" fd_kickoffs_dedupe_verify --dedupe --out-csv /app/backups/football_data_dedupe_verify_$STAMP.csv
step "shift re-plan" fd_kickoffs_shift_verify --shift --out-csv /app/backups/football_data_shift_verify_$STAMP.csv
dv=$(plan_of fd_kickoffs_dedupe_verify); sv=$(plan_of fd_kickoffs_shift_verify)
replanned="$(plan_num "$dv" merge) $(plan_num "$dv" fold) $(plan_num "$dv" refuse) $(plan_num "$sv" shift) $(plan_num "$sv" refuse)"
FOLDED_PY="import csv, sys; print(' '.join(r['espn_row_id'] for r in csv.DictReader(open(sys.argv[1])) if r['action'] == 'fold'))"
folded=$($DC exec -T api python -c "$FOLDED_PY" /app/backups/football_data_dedupe_apply_$STAMP.csv 2>/dev/null)
rm -f /tmp/fd_kickoffs_state.sql
$DC exec -T postgres rm -f /tmp/fd_kickoffs_q.sql >/dev/null 2>&1
if [ "$check_rc" -eq 0 ] && [ "$soccer1" = "$want_soccer" ] && [ "$stale1" = "$want_stale" ] && [ "$replanned" = "0 0 0 0 0" ]; then
  log "DONE: promote_raw may run again (its check_legacy_kickoffs passes). Folded rows (for the optional grading in the header): ${folded:-none}"
  log "NEXT (separate approval): scripts/ops/cleanup_soccer_orphans.sh"
else
  log "FAILED after-state: --check exit $check_rc (needs 0), soccer rows ${soccer1:-unknown}/$want_soccer, stale rows ${stale1:-unknown}/$want_stale, re-plan '$replanned' (needs '0 0 0 0 0') — inspect (archives and audit CSVs are in /app/backups); promote_raw stays blocked"
  exit 1
fi
