#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# The MMA fighter identity clean-up the user approved on 2026-11-02 ("We can do
# the cleanups"), with scripts/mma_identity_cleanup.py, in ONE transaction:
#   - split 'Bruno Silva': one team row holds two ESPN athletes, 3895544
#     ('Bulldog', flyweight, 4 bouts; keeps the row) and 4333158 ('Blindado',
#     middleweight, 7 bouts; moves to a new row 'Bruno Silva (ESPN 4333158)',
#     id d31648b1-b3f3-4c55-9d63-e8b112a498ed). Each row is stamped with its
#     athlete id; the 2 predictions on the moved bouts stay on their bouts; no
#     rec or bet references any of the 11 bouts;
#   - 'Raffael Cerqueira' (ESPN's spelling, athlete 5212598) becomes an alias
#     of the 'Rafael Cerqueira' row;
#   - the bout-less 'José Luiz' row is deleted (archived): which ESPN athlete
#     it was cannot be established, so it is NOT aliased to José Montanha;
#   - the ESPN athlete id is stamped on every MMA fighter row ESPN links to
#     exactly one athlete id (1,299 rows; 9 already carry theirs), the id
#     fetch_upcoming resolves MMA fighters by first since this commit.
#   Reviewed lists with the ESPN evidence: scripts/data/mma_fighter_split_20261102.csv,
#   mma_fighter_aliases_20261102.csv, mma_orphan_fighters_20261102.csv,
#   mma_fighter_athlete_ids_20261102.csv.
# The parallel stream mma_draws_held owns the matches b7f16d27 (Gatto v
# Kareckaite) and 787bd68e (Valesca Machado v Melissa Amaya) and the 'Valesca
# Machado' team row; nothing here touches them (the tool refuses a list that
# names them), so the two wrappers can run in either order.
#
# Reviewed plan — a local replica of the 2026-11-02T06:22:51Z read-only export
# of every UFC team, match, rec, prediction, odds row and snapshot:
#   PLAN split=1 split_done=0 split_refuse=0 bouts_moved=7 predictions_on_moved=2 recs_on_moved=0
#        alias_add=1 alias_done=0 alias_refuse=0 orphan_delete=1 orphan_done=0 orphan_refuse=0
#        ids_stamp=1290 ids_done=9 ids_refuse=0
#   (ids_stamp + ids_done = 1,299 is pinned, not the split between them: since
#   this commit the 15-minute sweeps stamp the rows they find by name, so rows
#   move from stamp to done until this runs; the plan digest covers the id each
#   row ends up with, so that drift never refuses the apply.)
#   WRITTEN aliases_added=1 bouts_moved=7 ids_stamped=1290 orphans_deleted=1 split_row_stamped=1 teams_inserted=1
#   A re-run plans the done plan (DONE_P below).
# Steps:
#   0. refuse while another scripts/ops wrapper or the soccer runbook runs on
#      this host, while an apply of a scripts/ tool still runs in the api
#      container, or while a UFC card is on (a UFC bout kicked off within the
#      last $CARD_BEFORE_H h or starts within $CARD_AFTER_H h, or one is 'live');
#      refuse unless the fighter merge is done and verified
#      (scripts/ops/merge_mma_fighters.sh --verify, read only).
#      Already done? The tool's read-only dry run must plan exactly the
#      reviewed plan (not done) or exactly the done plan; a done plan is checked
#      with --check-after on the plan the last apply recorded ($STATE_FILE)
#      before ALREADY DONE (exit 0, no backup). If that apply recorded its
#      follow-ups unfinished, they run now. Anything else aborts before the
#      backup;
#   1. verified backup (B2) — abort if it does not land;
#   2. dry run (write-once CSV + plan file) — abort unless it exits 0 with
#      the reviewed plan;
#   3. record the plan (STATUS=applying) in $STATE_FILE, then
#      apply --expect-plan <the dry run's plan file>: it locks the teams table
#      (SHARE ROW EXCLUSIVE, lock_timeout 10s: a sweep waits seconds) and the
#      split bouts, re-plans, refuses (exit 2, nothing written) unless that is
#      the dry run's plan digest, writes the gzip JSONL archive (the reversal
#      record) before its first write, and commits only once the resolver
#      resolves every listed row by its athlete id and both Bruno Silvas to
#      their own rows, and the after-state holds, inside it;
#   4. after-state: --check-after on the dry run's plan file (every count 0,
#      1299/1299 ids) and a fresh dry run that plans exactly the done plan;
#      STATUS=followups;
#   5. follow-ups (derived data only): compute_features_mma --all-finished
#      --force and --days 14 --force (both Bruno Silvas' histories changed);
#      STATUS=done. A failure exits 2; re-run them with
#        bash /opt/auspex/scripts/ops/mma_identity_cleanup.sh --followups-only
#   --verify (read only; for a later wrapper): the done plan, a recorded apply
#      whose after-state passed (STATUS followups or done), and --check-after
#      on the recorded plan — exit 0 when done and verified, else 1.
# Exit: 0 done (or already done, or verified); 1 aborted or an after-state
# failed; 2 the clean-up is verified but a follow-up failed.
# Reversal: the apply's archive /app/backups/mma_identity_cleanup_apply_<stamp>.jsonl.gz holds
# the pre-image of every row it changed or deleted (restore the bouts first,
# then the team rows) and the row it inserted (delete it last).
# Needs this commit on the VM (scripts/ is bind-mounted: a git pull is enough).
# Run on the VM host, after scripts/ops/merge_mma_fighters.sh:
#   bash /opt/auspex/scripts/ops/mma_identity_cleanup.sh 2>&1 | tee -a ~/mma_identity_cleanup.log
# (AUSPEX_DIR, TMPDIR, AF_STATE_DIR and MMA_IDENTITY_PS exist for tests and rehearsals.)
set -uo pipefail
MODE=apply
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
# shellcheck source=_airflow_lib.sh
. "$OPS/_airflow_lib.sh" || { log "ABORT: cannot load $OPS/_airflow_lib.sh — nothing changed"; exit 1; }
W=$(mktemp -d "${TMPDIR:-/tmp}/mma_identity.XXXXXX") || { log "ABORT: mktemp failed — nothing changed"; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/mma_identity_card.sql >/dev/null 2>&1
  af_unlock
}
trap cleanup EXIT
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

SPLIT=scripts/data/mma_fighter_split_20261102.csv
ALIASES=scripts/data/mma_fighter_aliases_20261102.csv
ORPHANS=scripts/data/mma_orphan_fighters_20261102.csv
IDS=scripts/data/mma_fighter_athlete_ids_20261102.csv
EXPECTED_BOUTS=11
EXPECTED_ALIASES=1
EXPECTED_ORPHANS=1
EXPECTED_IDS=1299
TOOL="python /app/scripts/mma_identity_cleanup.py"
# Every key but ids_stamp / ids_done, which must add up to EXPECTED_IDS (see the header).
PLAN_R="split=1 split_done=0 split_refuse=0 bouts_moved=7 predictions_on_moved=2 recs_on_moved=0 alias_add=1 alias_done=0 alias_refuse=0 orphan_delete=1 orphan_done=0 orphan_refuse=0 ids_refuse=0"
DONE_P="split=0 split_done=1 split_refuse=0 bouts_moved=0 predictions_on_moved=0 recs_on_moved=0 alias_add=0 alias_done=1 alias_refuse=0 orphan_delete=0 orphan_done=1 orphan_refuse=0 ids_stamp=0 ids_done=1299 ids_refuse=0"
CARD_BEFORE_H=8
CARD_AFTER_H=3
STATE_FILE=/app/backups/mma_identity_cleanup_last_apply.env
STAMP=$(date -u +%Y%m%dT%H%M%SZ)_$$
PLAN_FILE=/app/backups/mma_identity_cleanup_dryrun_$STAMP.plan.json
PS_CMD="${MMA_IDENTITY_PS:-ps -eo pid=,args=}"

reviewed_plan() {  # reviewed_plan <PLAN line>: the reviewed plan, ids_stamp + ids_done = EXPECTED_IDS
  local s d
  same_plan "$1" "$PLAN_R" || return 1
  s=$(plan_num "$1" ids_stamp); d=$(plan_num "$1" ids_done)
  [ "$s" != missing ] && [ "$d" != missing ] && [ $((s + d)) = "$EXPECTED_IDS" ]
}
# run_tool <log name> <command>: the command in the api container; prints its summary lines and exit
run_tool() {
  $DC exec -T api sh -c "$2 > /tmp/$1.log 2>&1; rc=\$?; grep -E 'PLAN |PLAN_FILE|AFTER|APPLIED|Archived|WRITTEN|ROLLED BACK|REFUSED|ERROR|Traceback|Error' /tmp/$1.log | tail -30; echo \"tool exit \$rc\"; exit \$rc" </dev/null
}
plan_of() { $DC exec -T api sh -c "grep -oE 'PLAN [a-z_]+=[0-9]+( [a-z_]+=[0-9]+)*' /tmp/$1.log | tail -1" </dev/null; }
read_state_file() { REC=$($DC exec -T api sh -c "cat $STATE_FILE 2>/dev/null" </dev/null); }
rec_val() { echo "$REC" | sed -n "s/^$1=//p" | tail -1; }
write_state_file() {  # write_state_file <plan> <status>
  printf 'PLAN=%s\nSTATUS=%s\n' "$1" "$2" | $DC exec -T api sh -c "cat > $STATE_FILE.tmp && mv $STATE_FILE.tmp $STATE_FILE"
}
check_recorded() { run_tool mma_identity_after "$TOOL --check-after $1"; }

# Follow-ups (step 5): derived data only, re-runnable; sets followup_failed.
run_followups() {
  local rc
  followup_failed=""
  log "5/5 follow-ups: recompute UFC features (both Bruno Silvas' histories changed)"
  $DC exec -T api sh -c "python /app/scripts/compute_features_mma.py --all-finished --force > /tmp/mma_identity_features.log 2>&1 && python /app/scripts/compute_features_mma.py --days 14 --force >> /tmp/mma_identity_features.log 2>&1; rc=\$?; grep -E 'Wrote|ERROR|Traceback' /tmp/mma_identity_features.log | tail -3; exit \$rc" </dev/null
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "ERROR step 5: compute_features_mma exited $rc (full log /tmp/mma_identity_features.log in the api container)"
    followup_failed="the feature recompute"
  fi
}
incomplete() {
  log "INCOMPLETE (exit 2): the clean-up is verified, but $followup_failed failed. Re-run the follow-ups only:"
  log "    bash /opt/auspex/scripts/ops/mma_identity_cleanup.sh --followups-only"
  exit 2
}

# others_running: prints every OTHER scripts/ops wrapper or soccer runbook process on this host; 0 when there is one.
# This run and its own subshells (the same command line) are not others; a second run of this script is held off by
# its lock (af_lock), not here.
others_running() {
  local me pid args found=1
  me=$(ps -p $$ -o args= 2>/dev/null)
  while read -r pid args; do
    [ -n "$pid" ] && [ "$pid" != "$$" ] || continue
    [ -n "$me" ] && [ "$args" = "$me" ] && continue
    case "$args" in
      *scripts/ops/*.sh* | *soccer_repair_runbook.sh*) echo "  pid $pid: $args"; found=0 ;;
    esac
  done < <($PS_CMD 2>/dev/null)
  return $found
}
# ufc_card_on: prints the UFC bouts that make a card 'on' now; 0 when there is one, 1 when none, 2 unreadable.
ufc_card_on() {
  local out
  cat > "$W/card.sql" <<SQL
SELECT m.id, m.match_date, m.status FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'mma' AND m.status NOT IN ('cancelled', 'postponed')
   AND (m.status = 'live' OR m.match_date BETWEEN now() - interval '$CARD_BEFORE_H hours' AND now() + interval '$CARD_AFTER_H hours')
 ORDER BY m.match_date;
SELECT 'CARD_CHECK_OK';
SQL
  $DC cp "$W/card.sql" postgres:/tmp/mma_identity_card.sql >/dev/null 2>&1 || return 2
  out=$($DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/mma_identity_card.sql' </dev/null) || return 2
  case "$out" in *CARD_CHECK_OK*) ;; *) return 2 ;; esac
  out=$(echo "$out" | grep -v '^CARD_CHECK_OK$' | grep .)
  [ -n "$out" ] || return 1
  echo "$out" | sed 's/^/  /'
  return 0
}

if [ "$MODE" != verify ]; then
  log "0/5 nothing else running? (scripts/ops wrappers, the soccer runbook, applies in the api container, a UFC card)"
  if others_running; then
    log "ABORT at step 0: the process(es) above run a scripts/ops wrapper or the soccer runbook — wait for them to end; nothing changed (no backup)"
    exit 1
  fi
  af_lock mma_identity_cleanup || { log "ABORT at step 0: another run of this script holds its lock — nothing changed"; exit 1; }
  af_applies_running mma_identity_cleanup merge_mma_fighters repair_fixture_orphans backfill_results '*'
  case $? in
    0) log "ABORT at step 0: the apply above still runs in the api container — wait for it to end; nothing changed (no backup)"; exit 1 ;;
    1) ;;
    *) log "ABORT at step 0: could not scan the api container for a running apply — nothing changed (no backup)"; exit 1 ;;
  esac
  ufc_card_on
  case $? in
    0) log "ABORT at step 0: a UFC card is on (the bouts above kicked off within ${CARD_BEFORE_H} h, start within ${CARD_AFTER_H} h, or are live) — run it after the card; nothing changed (no backup)"; exit 1 ;;
    1) ;;
    *) log "ABORT at step 0: could not read whether a UFC card is on — nothing changed (no backup)"; exit 1 ;;
  esac
fi

if [ "$MODE" = followups ]; then
  read_state_file
  rec_plan=$(rec_val PLAN); rec_status=$(rec_val STATUS)
  if [ -z "$rec_plan" ] || [ -z "$rec_status" ]; then
    log "ABORT: no apply is recorded in $STATE_FILE (got '${REC:-nothing}') — nothing to follow up"; exit 1
  fi
  if [ "$rec_status" = applying ]; then
    log "ABORT: the recorded apply is STATUS=applying (its after-state never passed) — run scripts/ops/mma_identity_cleanup.sh without arguments: it checks the after-state, then runs the follow-ups; nothing changed"; exit 1
  fi
  run_followups
  [ -z "$followup_failed" ] || incomplete
  write_state_file "$rec_plan" done || { log "FAILED: could not record STATUS=done in $STATE_FILE"; exit 1; }
  log "FOLLOW-UPS DONE (STATUS=done recorded)"
  exit 0
fi

for f in "$SPLIT" "$ALIASES" "$ORPHANS" "$IDS"; do
  [ -r "$f" ] || { log "ABORT: $f is missing — pull this commit first; nothing changed"; exit 1; }
done
n_bouts=$(grep -v '^#' "$SPLIT" | tail -n +2 | grep -c .)
n_aliases=$(grep -v '^#' "$ALIASES" | tail -n +2 | grep -c .)
n_orphans=$(grep -v '^#' "$ORPHANS" | tail -n +2 | grep -c .)
n_ids=$(grep -v '^#' "$IDS" | tail -n +2 | grep -c .)
if [ "$n_bouts/$n_aliases/$n_orphans/$n_ids" != "$EXPECTED_BOUTS/$EXPECTED_ALIASES/$EXPECTED_ORPHANS/$EXPECTED_IDS" ]; then
  log "ABORT: the lists hold $n_bouts/$n_aliases/$n_orphans/$n_ids rows (split/aliases/orphans/ids); this script expects $EXPECTED_BOUTS/$EXPECTED_ALIASES/$EXPECTED_ORPHANS/$EXPECTED_IDS — nothing changed"
  exit 1
fi

if [ "$MODE" = apply ]; then
  log "0/5 the fighter merge done and verified? (merge_mma_fighters.sh --verify, read only)"
  if ! bash "$OPS/merge_mma_fighters.sh" --verify; then
    log "ABORT at step 0: the fighter merge is not done and verified — this clean-up is reviewed against the state it leaves; nothing changed (no backup)"
    exit 1
  fi
fi

log "0/5 already done? (read-only dry run)"
run_tool mma_identity_pre "$TOOL --out-csv /app/backups/mma_identity_cleanup_precheck_$STAMP.csv --plan-out /app/backups/mma_identity_cleanup_precheck_$STAMP.plan.json"
pre=$(plan_of mma_identity_pre)
if same_plan "$pre" "$DONE_P"; then
  read_state_file
  rec_plan=$(rec_val PLAN); rec_status=$(rec_val STATUS)
  if [ -z "$rec_plan" ] || [ -z "$rec_status" ]; then
    log "FAILED: the clean-up reads done ('$pre') but no apply is recorded in $STATE_FILE (got '${REC:-nothing}') — inspect /app/backups/mma_identity_cleanup_*; nothing changed"
    exit 1
  fi
  if [ "$MODE" = verify ] && [ "$rec_status" = applying ]; then
    log "NOT VERIFIED: the clean-up reads done ('$pre') but its recorded apply is STATUS=applying (its after-state never passed) — re-run scripts/ops/mma_identity_cleanup.sh to check it and finish the follow-ups"
    exit 1
  fi
  if ! check_recorded "$rec_plan"; then
    log "FAILED: the clean-up reads done but the after-state of the recorded plan ($rec_plan, STATUS=$rec_status) fails (above) — inspect; nothing changed"
    exit 1
  fi
  if [ "$MODE" = verify ]; then
    log "IDENTITY CLEANUP VERIFIED: '$pre'; the after-state of $rec_plan holds (STATUS=$rec_status)"
    exit 0
  fi
  if [ "$rec_status" = done ]; then
    log "ALREADY DONE: '$pre' and the after-state of $rec_plan holds — nothing to write (no backup)"
    exit 0
  fi
  [ "$rec_status" = applying ] && { write_state_file "$rec_plan" followups || log "WARNING: could not record STATUS=followups in $STATE_FILE"; }
  log "the clean-up is committed and verified, but the last apply recorded STATUS=$rec_status: running its follow-ups now"
  run_followups
  [ -z "$followup_failed" ] || incomplete
  write_state_file "$rec_plan" done || { log "FAILED: could not record STATUS=done in $STATE_FILE"; exit 1; }
  log "COMPLETED: follow-ups done (STATUS=done recorded); the clean-up was already committed and verified"
  exit 0
fi
if [ "$MODE" = verify ]; then
  log "NOT VERIFIED: the identity clean-up is not done (pre-plan '${pre:-unreadable}', done plan $DONE_P) — run scripts/ops/mma_identity_cleanup.sh first"
  exit 1
fi
if ! reviewed_plan "$pre"; then
  log "ABORT at step 0: the pre-check plan ('${pre:-unreadable}') is neither the reviewed plan ($PLAN_R, ids_stamp + ids_done = $EXPECTED_IDS) nor the done plan ($DONE_P) — UFC rows changed since the review, a refusal (see above), or other code; nothing changed (no backup)"
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
run_tool mma_identity_dry "$TOOL --out-csv /app/backups/mma_identity_cleanup_dryrun_$STAMP.csv --plan-out $PLAN_FILE"
rc=$?
plan=$(plan_of mma_identity_dry)
if [ "$rc" -ne 0 ] || ! reviewed_plan "$plan"; then
  log "ABORT at step 2: the dry run failed (exit $rc: a refusal, see above) or its plan ('$plan') is not the reviewed one ($PLAN_R, ids_stamp + ids_done = $EXPECTED_IDS) — review the dry-run CSV; nothing changed"
  exit 1
fi
log "plan: $plan"

write_state_file "$PLAN_FILE" applying \
  || { log "ABORT at step 3: could not record the apply (STATUS=applying) in $STATE_FILE before it — nothing changed"; exit 1; }
log "3/5 apply (--expect-plan $PLAN_FILE; recorded STATUS=applying)"
run_tool mma_identity_apply "$TOOL --apply --expect-plan $PLAN_FILE --out-csv /app/backups/mma_identity_cleanup_apply_$STAMP.csv --archive /app/backups/mma_identity_cleanup_apply_$STAMP.jsonl.gz"
rc=$?
if [ "$rc" -ne 0 ]; then
  why="it rolled back"
  [ "$rc" = "2" ] && why="it refused before writing (its plan differs from the dry run's, or a refusal above)"
  log "ABORT at step 3 (exit $rc; full log /tmp/mma_identity_apply.log in the api container): $why — one transaction, nothing committed; a re-run starts from step 0"
  exit 1
fi

log "4/5 after-state (from the dry run's plan file)"
check_recorded "$PLAN_FILE"
after_rc=$?
run_tool mma_identity_verify "$TOOL --out-csv /app/backups/mma_identity_cleanup_verify_$STAMP.csv --plan-out /app/backups/mma_identity_cleanup_verify_$STAMP.plan.json"
replan=$(plan_of mma_identity_verify)
log "after: --check-after exit $after_rc (expected 0), re-plan '$replan' (expected $DONE_P)"
if [ "$after_rc" -ne 0 ] || ! same_plan "$replan" "$DONE_P"; then
  log "FAILED after-state at step 4 — inspect (the apply's archive /app/backups/mma_identity_cleanup_apply_$STAMP.jsonl.gz and its CSV are the reversal record); STATUS stays 'applying', so a re-run checks the recorded plan again before anything else"
  exit 1
fi
write_state_file "$PLAN_FILE" followups || log "WARNING: could not record STATUS=followups in $STATE_FILE"
log "clean-up done: Bruno Silva split, 1 alias added, 1 orphan deleted, 1299 athlete ids on their rows"

run_followups
[ -z "$followup_failed" ] || incomplete
write_state_file "$PLAN_FILE" done || { log "FAILED: could not record STATUS=done in $STATE_FILE (the clean-up and its follow-ups are through; a re-run runs the follow-ups again)"; exit 1; }
log "DONE: the MMA fighter identities are cleaned up and every linked fighter resolves by ESPN athlete id"
