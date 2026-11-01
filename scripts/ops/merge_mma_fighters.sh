#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Merge the 9 UFC fighters stored under two team rows into one row each, and
# fold the stale copies of their fought bouts (plus one bout stored twice the
# other way round) into the finished rows, so run B of
# scripts/ops/repair_stranded_recs_mma.sh no longer deletes them as 'no twin'.
#
# WHY (prod 2026-11-01, read only; every pair checked against ESPN athlete ids):
#   ESPN changed 9 fighters' display names ('Zachary Reese' -> 'Zach Reese',
#   'Cristian Perez' -> 'Cristian Pérez', 'Ce Liu' -> 'Liu Ce', ...), and
#   fetch_upcoming resolves an MMA fighter by name only, so each got a second
#   team row. 8 bouts were left 'scheduled' under the old spelling beside the
#   finished row under the new one; McGhee v Romero (10-03) was stored twice
#   on the same two fighters the other way round. Run B matches twins on the
#   same team ids in the same order only, so it would delete those 9 fought
#   bouts with their pre-game prices. Separately, one fighter name ('Rose
#   Conceição') is stored in Unicode NFD form; it is normalized here.
#   Reviewed lists: scripts/data/mma_fighter_merge_map.csv (9 pairs, evidence
#   per pair) and scripts/data/mma_reversed_fixture_folds.csv (1 bout).
#
# RUN ORDER (MMA; scripts/ops/soccer_repair_runbook.sh is independent of these):
#   1. scripts/ops/repair_stranded_recs_mma.sh                       (run A: the June results backfill)
#   2. THIS script                                                   (refuses unless A is done and verified)
#   3. scripts/ops/repair_stranded_recs_mma.sh --retire-stale-rows   (run B, pinned to the state this leaves;
#                                                                     refuses unless this is done and verified)
#
# Reviewed plan — a local replica of the 2026-11-01T21:06:57Z read-only export
# of every UFC match, team, rec, prediction, odds row and snapshot, with run A
# applied first (its plan did not change), this commit's code:
#   scripts/merge_mma_fighters.py, ONE transaction:
#     PLAN pairs_merge=9 pairs_done=0 pairs_refuse=0 folds=9 folds_reversed=2 folds_done=0 folds_refuse=0
#          repoints=11 left_rec_held=0 odds_moved=32 snapshots_moved=51 predictions_moved=15
#          predictions_dropped=8 espn_ids_adopted=2 athlete_ids_stamped=9 nfc_renamed=1 nfc_refuse=0
#   9 stale copies folded into their finished rows and deleted (2 the other way
#   round: Kropschot v Kunneman and McGhee v Romero, whose prices move with
#   home/away swapped and whose predictions do not move); 32 odds rows, 51
#   snapshots, 15 predictions move; 2 finished rows take the copy's ESPN
#   competition id; 11 other bouts of the merged rows re-pointed (Zachary
#   Reese's 4 finished 2023-24 bouts, Adrian Yanez's 5, Matthieu Duclos' 1,
#   'Junior Tafa v Ce Liu', which run B then retires as no-twin); each merged
#   spelling recorded as an alias of the kept row with metadata.merged_from
#   and the ESPN athlete id; the 9 merged team rows deleted; 'Rose Conceição'
#   NFC-normalized. No rec references any row it touches (left_rec_held=0).
#   A re-run plans the done plan (DONE_M below).
# Steps:
#   0. A done and verified? (repair_stranded_recs_mma.sh --verify-a: the 21
#      reviewed recs carry their reviewed outcomes, no UFC rec is stranded,
#      the 49 UFC recs are there; read only) — else abort, nothing changed.
#      Already done? The tool's read-only dry run must plan exactly the
#      reviewed plan (not done) or exactly the done plan; a done plan is
#      checked with --check-after on the plan the last apply recorded
#      ($STATE_FILE) before ALREADY DONE (exit 0, no backup); if that apply
#      recorded its follow-ups unfinished, they run now. Anything else aborts
#      before the backup;
#   1. verified backup (B2) — abort if it does not land;
#   2. dry run (write-once CSV + plan file) — abort unless it exits 0 with
#      exactly the reviewed plan;
#   3. record the plan (STATUS=applying) in $STATE_FILE, then
#      apply --expect-plan <the dry run's plan file>: it re-plans inside its
#      own transaction after locking every row it touches, refuses (exit 2,
#      nothing written) unless that is the dry run's plan to the last row id,
#      writes the gzip JSONL archive (the reversal record) before its first
#      write, and commits only once the resolver resolves every merged
#      spelling to its kept row and the after-state holds, inside it;
#   4. after-state: --check-after on the dry run's plan file (every count 0,
#      9/9 merges recorded) and a fresh dry run that plans exactly the done
#      plan; STATUS=followups;
#   5. follow-ups (derived data only): grade the finished rows that received
#      predictions (grade_completed_matches.py --match-id, the dry run's
#      GRADE_AFTER rows), then compute_features_mma --all-finished --force and
#      --days 14 --force (the kept fighters' histories changed); STATUS=done.
#      A failure exits 2; re-run them with
#        bash /opt/auspex/scripts/ops/merge_mma_fighters.sh --followups-only
#   --verify (run B's step 0 calls it; read only): the done plan, and
#      --check-after --after-later-steps on the recorded plan (B deletes one
#      re-pointed no-twin row on purpose) — exit 0 when the merge is done and
#      verified, else 1.
# Exit: 0 done (or already done, or verified); 1 aborted or an after-state
# failed; 2 the merge is verified but a follow-up failed.
# WHEN: not while a UFC card is being fought (Saturday evening US time). The
# apply's transaction is REPEATABLE READ and locks its rows: a clash fails it
# with nothing written — safe, but a wasted backup.
# Needs this commit on the VM (scripts/ is bind-mounted: a git pull is enough).
# Run on the VM host, after run A:
#   bash /opt/auspex/scripts/ops/merge_mma_fighters.sh 2>&1 | tee -a ~/merge_mma_fighters.log
# (AUSPEX_DIR and TMPDIR exist for tests and rehearsals.)
set -uo pipefail
MODE=merge
case "${1:-}" in
  "") ;;
  --verify) MODE=verify ;;
  --followups-only) MODE=followups ;;
  *) echo "usage: $0 [--verify | --followups-only]   (see the header)"; exit 2 ;;
esac
OPS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || { echo "ABORT: cannot find scripts/ops — nothing changed"; exit 1; }
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o " $2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
same_plan() {  # same_plan <PLAN line> <reviewed 'k=v ...'>: every reviewed key has exactly its count
  local k
  for k in $2; do
    [ "$(plan_num "$1" "${k%%=*}")" = "${k#*=}" ] || return 1
  done
}

MAP=scripts/data/mma_fighter_merge_map.csv
FOLDS=scripts/data/mma_reversed_fixture_folds.csv
EXPECTED_PAIRS=9
EXPECTED_FOLDS=1
TOOL="python /app/scripts/merge_mma_fighters.py"
PLAN_M="pairs_merge=9 pairs_done=0 pairs_refuse=0 folds=9 folds_reversed=2 folds_done=0 folds_refuse=0 repoints=11 left_rec_held=0 odds_moved=32 snapshots_moved=51 predictions_moved=15 predictions_dropped=8 espn_ids_adopted=2 athlete_ids_stamped=9 nfc_renamed=1 nfc_refuse=0"
DONE_M="pairs_merge=0 pairs_done=9 pairs_refuse=0 folds=0 folds_reversed=0 folds_done=1 folds_refuse=0 repoints=0 left_rec_held=0 odds_moved=0 snapshots_moved=0 predictions_moved=0 predictions_dropped=0 espn_ids_adopted=0 athlete_ids_stamped=0 nfc_renamed=0 nfc_refuse=0"
STATE_FILE=/app/backups/merge_mma_fighters_last_apply.env
# The process id too: the audit files are write-once, and run B calls this script (--verify) moments before an
# operator may run it, so two runs can start within one second.
STAMP=$(date -u +%Y%m%dT%H%M%SZ)_$$
PLAN_FILE=/app/backups/merge_mma_fighters_dryrun_$STAMP.plan.json

# run_tool <log name> <command>: the command in the api container; prints its summary lines and exit
run_tool() {
  $DC exec -T api sh -c "$2 > /tmp/$1.log 2>&1; rc=\$?; grep -E 'PLAN |PLAN_FILE|AFTER|APPLIED|Archived|WRITTEN|ROLLED BACK|REFUSED|ERROR|Traceback|Error' /tmp/$1.log | tail -30; echo \"tool exit \$rc\"; exit \$rc" </dev/null
}
# The tool's own PLAN line only (KEY=N pairs).
plan_of() { $DC exec -T api sh -c "grep -oE 'PLAN [a-z_]+=[0-9]+( [a-z_]+=[0-9]+)*' /tmp/$1.log | tail -1" </dev/null; }
# The last apply's record, in the api container: PLAN, GRADE_IDS, STATUS.
read_state_file() { REC=$($DC exec -T api sh -c "cat $STATE_FILE 2>/dev/null" </dev/null); }
rec_val() { echo "$REC" | sed -n "s/^$1=//p" | tail -1; }
write_state_file() {  # write_state_file <plan> <grade ids> <status>
  printf 'PLAN=%s\nGRADE_IDS=%s\nSTATUS=%s\n' "$1" "$2" "$3" \
    | $DC exec -T api sh -c "cat > $STATE_FILE.tmp && mv $STATE_FILE.tmp $STATE_FILE"
}

# Follow-ups (step 5): derived data only, re-runnable; sets followup_failed.
run_followups() {  # run_followups <grade ids>
  local rc
  followup_failed=""
  log "5/5 follow-ups: grade the $(echo $1 | wc -w | tr -d ' ') finished row(s) that received predictions, then recompute UFC features"
  if [ -n "${1// /}" ]; then
    $DC exec -T api sh -c "rc=0; : > /tmp/merge_mma_grade.log; for mid in $1; do python /app/scripts/grade_completed_matches.py --match-id \$mid >> /tmp/merge_mma_grade.log 2>&1 || { rc=1; echo \"grading \$mid failed\"; }; done; echo \"graded rows: \$(grep -c 'Done\.' /tmp/merge_mma_grade.log)\"; exit \$rc" </dev/null
    rc=$?
    if [ "$rc" -ne 0 ]; then
      log "ERROR step 5: grading a row failed (full log /tmp/merge_mma_grade.log in the api container)"
      followup_failed="the grading"
    fi
  fi
  $DC exec -T api sh -c "python /app/scripts/compute_features_mma.py --all-finished --force > /tmp/merge_mma_features.log 2>&1 && python /app/scripts/compute_features_mma.py --days 14 --force >> /tmp/merge_mma_features.log 2>&1; rc=\$?; grep -E 'Wrote|ERROR|Traceback' /tmp/merge_mma_features.log | tail -3; exit \$rc" </dev/null
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "ERROR step 5: compute_features_mma exited $rc (full log /tmp/merge_mma_features.log in the api container)"
    followup_failed="${followup_failed:+$followup_failed and }the feature recompute"
  fi
}
incomplete() {
  log "INCOMPLETE (exit 2): the merge is verified, but $followup_failed failed. Re-run the follow-ups only:"
  log "    bash /opt/auspex/scripts/ops/merge_mma_fighters.sh --followups-only"
  exit 2
}
# check_recorded <plan file> [--after-later-steps]: the after-state of a recorded plan
check_recorded() {
  run_tool merge_mma_after "$TOOL --check-after $1 ${2:-}"
}

if [ "$MODE" = followups ]; then
  read_state_file
  rec_plan=$(rec_val PLAN); rec_status=$(rec_val STATUS)
  if [ -z "$rec_plan" ] || [ -z "$rec_status" ]; then
    log "ABORT: no apply is recorded in $STATE_FILE (got '${REC:-nothing}') — nothing to follow up"; exit 1
  fi
  run_followups "$(rec_val GRADE_IDS)"
  [ -z "$followup_failed" ] || incomplete
  write_state_file "$rec_plan" "$(rec_val GRADE_IDS)" done || { log "FAILED: could not record STATUS=done in $STATE_FILE"; exit 1; }
  log "FOLLOW-UPS DONE (STATUS=done recorded)"
  exit 0
fi

for f in "$MAP" "$FOLDS"; do
  [ -r "$f" ] || { log "ABORT: $f is missing — pull this commit first; nothing changed"; exit 1; }
done
pairs=$(grep -v '^#' "$MAP" | tail -n +2 | grep -c .)
folds=$(grep -v '^#' "$FOLDS" | tail -n +2 | grep -c .)
if [ "$pairs" != "$EXPECTED_PAIRS" ] || [ "$folds" != "$EXPECTED_FOLDS" ]; then
  log "ABORT: $MAP has $pairs pair(s) and $FOLDS $folds fold(s); this script expects $EXPECTED_PAIRS and $EXPECTED_FOLDS — nothing changed"
  exit 1
fi

if [ "$MODE" = merge ]; then
  log "0/5 run A done and verified? (repair_stranded_recs_mma.sh --verify-a, read only)"
  if ! bash "$OPS/repair_stranded_recs_mma.sh" --verify-a; then
    log "ABORT at step 0: run A is not done and verified — run scripts/ops/repair_stranded_recs_mma.sh first (and pass); nothing changed (no backup)"
    exit 1
  fi
fi

log "0/5 already done? (read-only dry run)"
run_tool merge_mma_pre "$TOOL --out-csv /app/backups/merge_mma_fighters_precheck_$STAMP.csv --plan-out /app/backups/merge_mma_fighters_precheck_$STAMP.plan.json"
pre=$(plan_of merge_mma_pre)
if same_plan "$pre" "$DONE_M"; then
  read_state_file
  rec_plan=$(rec_val PLAN); rec_ids=$(rec_val GRADE_IDS); rec_status=$(rec_val STATUS)
  if [ -z "$rec_plan" ] || [ -z "$rec_status" ]; then
    log "FAILED: the merge reads done ('$pre') but no apply is recorded in $STATE_FILE (got '${REC:-nothing}') — inspect /app/backups/merge_mma_fighters_*; nothing changed"
    exit 1
  fi
  if ! check_recorded "$rec_plan" --after-later-steps; then
    log "FAILED: the merge reads done but the after-state of the recorded plan ($rec_plan) fails (above) — inspect; nothing changed"
    exit 1
  fi
  if [ "$MODE" = verify ]; then
    log "MERGE VERIFIED: '$pre'; the after-state of $rec_plan holds (STATUS=$rec_status)"
    exit 0
  fi
  if [ "$rec_status" = "done" ]; then
    log "ALREADY DONE: '$pre' and the after-state of $rec_plan holds — nothing to write (no backup)"
    exit 0
  fi
  log "the merge is committed and verified, but the last apply recorded STATUS=$rec_status: running its follow-ups now"
  run_followups "$rec_ids"
  [ -z "$followup_failed" ] || incomplete
  write_state_file "$rec_plan" "$rec_ids" done || { log "FAILED: could not record STATUS=done in $STATE_FILE"; exit 1; }
  log "COMPLETED: follow-ups done (STATUS=done recorded); the merge was already committed and verified"
  exit 0
fi
if [ "$MODE" = verify ]; then
  log "NOT VERIFIED: the fighter merge is not done (pre-plan '${pre:-unreadable}', done plan $DONE_M) — run scripts/ops/merge_mma_fighters.sh first"
  exit 1
fi
if ! same_plan "$pre" "$PLAN_M"; then
  log "ABORT at step 0: the pre-check plan ('${pre:-unreadable}') is neither the reviewed plan ($PLAN_M) nor the done plan ($DONE_M) — UFC rows changed since the review, a refusal (see above), or other code; nothing changed (no backup)"
  exit 1
fi
log "not done yet (pre-plan '$pre') — continuing"

log "1/5 backup"
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

log "2/5 dry run (plan file $PLAN_FILE)"
run_tool merge_mma_dry "$TOOL --out-csv /app/backups/merge_mma_fighters_dryrun_$STAMP.csv --plan-out $PLAN_FILE"
rc=$?
plan=$(plan_of merge_mma_dry)
if [ "$rc" -ne 0 ] || ! same_plan "$plan" "$PLAN_M"; then
  log "ABORT at step 2: the dry run failed (exit $rc: a refusal, see above) or its plan ('$plan') is not the reviewed one ($PLAN_M) — review the dry-run CSV; nothing changed"
  exit 1
fi
grade_ids=$($DC exec -T api sh -c "grep -o 'GRADE_AFTER [0-9a-f-]*' /tmp/merge_mma_dry.log | awk '{print \$2}' | sort -u" </dev/null | tr '\n' ' ')
log "plan: $plan; $(echo $grade_ids | wc -w | tr -d ' ') finished row(s) to grade after"

write_state_file "$PLAN_FILE" "$grade_ids" applying \
  || { log "ABORT at step 3: could not record the apply (STATUS=applying) in $STATE_FILE before it — nothing changed"; exit 1; }
log "3/5 apply (--expect-plan $PLAN_FILE; recorded STATUS=applying)"
run_tool merge_mma_apply "$TOOL --apply --expect-plan $PLAN_FILE --out-csv /app/backups/merge_mma_fighters_apply_$STAMP.csv --archive /app/backups/merge_mma_fighters_apply_$STAMP.jsonl.gz"
rc=$?
if [ "$rc" -ne 0 ]; then
  why="it rolled back"
  [ "$rc" = "2" ] && why="it refused before writing (its plan differs from the dry run's, or a refusal above)"
  log "ABORT at step 3 (exit $rc; full log /tmp/merge_mma_apply.log in the api container): $why — one transaction, nothing committed; a re-run starts from step 0"
  exit 1
fi

log "4/5 after-state (from the dry run's plan file)"
check_recorded "$PLAN_FILE"
after_rc=$?
run_tool merge_mma_verify "$TOOL --out-csv /app/backups/merge_mma_fighters_verify_$STAMP.csv --plan-out /app/backups/merge_mma_fighters_verify_$STAMP.plan.json"
replan=$(plan_of merge_mma_verify)
log "after: --check-after exit $after_rc (expected 0), re-plan '$replan' (expected $DONE_M)"
if [ "$after_rc" -ne 0 ] || ! same_plan "$replan" "$DONE_M"; then
  log "FAILED after-state at step 4 — inspect (the apply's archive /app/backups/merge_mma_fighters_apply_$STAMP.jsonl.gz and its CSV are the reversal record); STATUS stays 'applying', so a re-run checks the recorded plan again before anything else"
  exit 1
fi
write_state_file "$PLAN_FILE" "$grade_ids" followups || log "WARNING: could not record STATUS=followups in $STATE_FILE"
log "merge done: 9 fighters merged, 9 stale copies folded, 'Rose Conceição' NFC-normalized"

run_followups "$grade_ids"
[ -z "$followup_failed" ] || incomplete
write_state_file "$PLAN_FILE" "$grade_ids" done || { log "FAILED: could not record STATUS=done in $STATE_FILE (the merge and its follow-ups are through; a re-run runs the follow-ups again)"; exit 1; }
log "DONE: the 9 UFC fighters are merged and resolving; NEXT (operator): bash /opt/auspex/scripts/ops/repair_stranded_recs_mma.sh --retire-stale-rows (run B, pinned to this state)"
