#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Settle the two UFC rows run B of scripts/ops/repair_stranded_recs_mma.sh held
# for the user's decision (scripts/data/mma_stale_rows_hold_20261101.csv):
#   * Melissa Gatto v Ernesta Kareckaite (UFC 2026-10-10, ESPN 401924511) is
#     recorded as the majority draw it was, through the results sweep's own
#     code (fetch_upcoming._record_result: finished, NULL scores,
#     metadata.no_winner 'draw'), and graded (its 2 predictions recorded
#     'draw' with is_correct NULL; it holds no rec);
#   * Valesca Machado v Melissa Amaya (UFC 2026-09-26, ESPN 401914464) is
#     deleted with its 2 predictions and 2 feature rows (it holds no odds, no
#     snapshots, no rec), archived first; nothing moves onto the finished
#     Tina Black v Melissa Amaya row (588fc028), the bout ESPN's competition
#     names now. The 'Valesca Machado' team row, which holds no other bout,
#     goes too, archived.
#
# WHY (the user's decision, 2026-11-02: "We can just make the system record it?
# But we can delete it."): fetch_upcoming now records MMA draws and no contests
# (this commit), but the pipeline's results sweep looks 3 days back and never
# reaches 2026-10-10; and the Machado row priced a matchup ESPN replaced five
# days before the card (ESPN has no athlete named Valesca Machado). Reviewed
# list with the evidence: scripts/data/mma_held_rows_settle_20261102.csv.
#
# RUN ORDER (MMA): after run A, scripts/ops/merge_mma_fighters.sh and run B
# (--retire-stale-rows), all three done on prod 2026-11-02 04:35Z. This script
# refuses unless B's done check passes (repair_stranded_recs_mma.sh
# --verify-b: A's committed state, merge_mma_fighters.sh --verify, B's done
# plan with its database state). Afterwards B's done check reads the
# 'settled' form of B's done state (left=7, 7 stale UFC rows), so a re-run of
# B, of the merge, or of this script prints ALREADY DONE / VERIFIED. B's done
# check counts only the UFC recs created before 2026-11-01 (49): the UFC recs
# the pipeline writes for the cards ahead never make it, or this script, read
# "not done".
# Safe in either order with the mma_identity wrapper (Bruno Silva split,
# aliases, ESPN athlete ids): this script touches neither the Bruno Silva,
# Raffael Cerqueira nor José Luiz / José Montanha rows, and checks that the
# results path wrote only the listed row (a resolver that stamps ESPN athlete
# ids may stamp the two fighters' external_ids; any other team change
# refuses). A deep ESPN re-fetch of these cards before this script (this
# commit's results path) may have recorded the draw already, graded or not,
# and cancelled the Machado row as a re-drawn slot: the plan then takes one of
# the two reviewed variants below (record=0 record_done=1), the end state is
# the same, and B's done check counts a cancelled Machado row as still held.
# Any other drift stops this script before the backup, nothing changed.
#
# Reviewed plan — a local replica of the 2026-11-02T07:03Z read-only export (every
# UFC match, team, prediction, rec, odds row, snapshot and feature row), this
# commit's code, ESPN live:
#   scripts/settle_mma_held_rows.py, ONE REPEATABLE READ transaction:
#     PLAN record=1 record_done=0 record_predictions=2 record_recs_void=0 delete=1 delete_done=0
#          delete_predictions=2 delete_odds=0 delete_snapshots=0 delete_features=2 delete_other_rows=0
#          teams_delete=1 refuse=0
#   Variants (rehearsed: the replica after a re-fetch of the 2026-10-10 and
#   2026-09-26 cards with this commit's code), the draw already recorded:
#     record=0 record_done=1 record_predictions=2 (a bare fetch_upcoming
#     --results run: recorded, not graded) or record_predictions=0
#     (scripts/backfill_results.py: recorded and graded); every other key as
#     above. The dry run and the apply are pinned to the one the pre-check
#     found.
#   A re-run plans the done plan (DONE_S below) and writes nothing.
# Steps:
#   0. refuse while it cannot run safely, nothing changed (no backup):
#      another run of this script (scripts/ops/_airflow_lib.sh af_lock), another
#      scripts/ops/ wrapper or the soccer runbook running on the host, any
#      tool's --apply still running in the api container (the UFC writers by
#      name, mma_identity_cleanup among them, and every other one: an apply
#      whose host wrapper was killed keeps running there, out of the host's
#      process list), a UFC card in progress (a UFC kickoff in the last 8
#      hours or the next hour), the reviewed list missing; then B's done check
#      (repair_stranded_recs_mma.sh --verify-b, read only) must pass. Already
#      done? The tool's pre-check (the whole transaction, rolled back) must
#      plan exactly the reviewed plan (or one of its reviewed variants), or
#      exactly the done plan — then --check-after must hold: ALREADY DONE,
#      exit 0, no backup;
#   1. verified backup (B2) — abort if it does not land;
#   2. dry run (the whole transaction, rolled back; write-once audit CSV) —
#      abort unless it exits 0 with exactly the plan the pre-check found;
#   3. apply --expect-plan <every key of that plan>: it re-plans after
#      locking the rows and refuses (exit 2, nothing written) unless that plan
#      is the same; it writes the gzip JSONL archive (the reversal
#      record) before its first write, and commits only once the after-state,
#      a re-plan of 0 and "nothing else changed" hold inside its transaction;
#   4. after-state: --check-after, a fresh pre-check that plans exactly the
#      done plan, and B's done check passing in its settled form.
# No follow-ups: the grading runs inside the apply, and no feature changes
# (compute_features_mma counts no bout without a winner; the deleted row was
# never finished, so no history counted it).
#   --verify (commits nothing): the tool's dry run (the whole transaction,
#   rolled back) must plan the done plan, and --check-after (read only) must
#   hold — exit 0 when the two rows are settled, else 1. Before the settle
#   has run, that dry run locks the two rows FOR UPDATE while it runs, asks
#   ESPN (the scoreboard and the core API) and runs the results path and the
#   deletes before rolling them back; every call writes an audit CSV under
#   /app/backups (empty once settled).
# Logs: each tool run writes /tmp/settle_mma_<step>.log in the api container;
# a run that exits 0 removes them, a failed one leaves them for inspection
# (the next run overwrites them).
# Exit: 0 done (or already done, or verified); 1 aborted or an after-state failed.
# WHEN: not while a UFC card is being fought (next: Saturday 2026-11-07); step 0
# refuses then.
# Needs this commit on the VM (scripts/ is bind-mounted: a git pull is enough).
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/settle_mma_held_rows.sh 2>&1 | tee -a ~/settle_mma_held_rows.log
# (AUSPEX_DIR, TMPDIR and SETTLE_PS exist for tests and rehearsals.)
set -uo pipefail
MODE=settle
case "${1:-}" in
  "") ;;
  --verify) MODE=verify ;;
  *) echo "usage: $0 [--verify]   (see the header)"; exit 2 ;;
esac
OPS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || { echo "ABORT: cannot find scripts/ops — nothing changed"; exit 1; }
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
# shellcheck source=scripts/ops/_airflow_lib.sh
. "$OPS/_airflow_lib.sh" || { log "ABORT: cannot load $OPS/_airflow_lib.sh — nothing changed"; exit 1; }
W=$(mktemp -d "${TMPDIR:-/tmp}/settle_mma.XXXXXX") || { log "ABORT: mktemp failed — nothing changed"; exit 1; }
cleanup() {
  local rc=$?
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/settle_mma_q.sql >/dev/null 2>&1
  # The tool logs in the api container: kept after a failure (the messages above point at them).
  [ "$rc" = 0 ] && $DC exec -T api sh -c 'rm -f /tmp/settle_mma_*.log' >/dev/null 2>&1
  af_unlock
}
trap cleanup EXIT
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_at() {  # psql_at <local.sql>: unaligned, '|'-separated
  $DC cp "$1" postgres:/tmp/settle_mma_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/settle_mma_q.sql'
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

LIST=scripts/data/mma_held_rows_settle_20261102.csv  # host path; /app/$LIST in the api container
N_LIST=2
TOOL="python /app/scripts/settle_mma_held_rows.py"
PLAN_S="record=1 record_done=0 record_predictions=2 record_recs_void=0 delete=1 delete_done=0 delete_predictions=2 delete_odds=0 delete_snapshots=0 delete_features=2 delete_other_rows=0 teams_delete=1 refuse=0"
# The draw already recorded by the results path (see the header): not graded, or graded.
PLAN_S_RECORDED="record=0 record_done=1 record_predictions=2 record_recs_void=0 delete=1 delete_done=0 delete_predictions=2 delete_odds=0 delete_snapshots=0 delete_features=2 delete_other_rows=0 teams_delete=1 refuse=0"
PLAN_S_GRADED="record=0 record_done=1 record_predictions=0 record_recs_void=0 delete=1 delete_done=0 delete_predictions=2 delete_odds=0 delete_snapshots=0 delete_features=2 delete_other_rows=0 teams_delete=1 refuse=0"
DONE_S="record=0 record_done=1 record_predictions=0 record_recs_void=0 delete=0 delete_done=1 delete_predictions=0 delete_odds=0 delete_snapshots=0 delete_features=0 delete_other_rows=0 teams_delete=0 refuse=0"
# Tools whose --apply writes UFC rows: none may be running in the api container meanwhile. The scan also takes
# '*' (any /app/scripts tool's --apply, as scripts/ops/mma_identity_cleanup.sh scans), passed quoted below.
UFC_WRITERS="settle_mma_held_rows backfill_results repair_fixture_orphans merge_mma_fighters repair_stranded_recs grade_matches mma_identity_cleanup"
# The host's process list (overridable for tests and rehearsals only).
SETTLE_PS="${SETTLE_PS:-ps -eo pid=,args=}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)_$$  # the process id too: the audit files are write-once

# run_tool <log name> <command>: the command in the api container; prints its summary lines and exit
run_tool() {
  $DC exec -T api sh -c "$2 > /tmp/$1.log 2>&1; rc=\$?; grep -E 'PLAN |REFUSED|AFTER|APPLIED|DRY RUN|NOTHING TO WRITE|Archived|ROLLED BACK|recorded |deleted |ERROR|Traceback|Error' /tmp/$1.log | tail -30; echo \"tool exit \$rc\"; exit \$rc" </dev/null
}
# The tool's own PLAN line only (KEY=N pairs).
plan_of() { $DC exec -T api sh -c "grep -oE 'PLAN [a-z_]+=[0-9]+( [a-z_]+=[0-9]+)*' /tmp/$1.log | tail -1" </dev/null; }

n_list=$( (grep -v '^#' "$LIST" 2>/dev/null || true) | tail -n +2 | grep -c .)
if [ "$n_list" != "$N_LIST" ]; then
  log "ABORT: $LIST holds ${n_list:-no} row(s), this script is pinned to $N_LIST — pull this commit first; nothing changed"
  exit 1
fi

if [ "$MODE" = verify ]; then
  run_tool settle_mma_verify "$TOOL --out-csv /app/backups/settle_mma_held_rows_verify_$STAMP.csv"
  pre=$(plan_of settle_mma_verify)
  if same_plan "$pre" "$DONE_S" && run_tool settle_mma_check "$TOOL --check-after"; then
    log "SETTLED VERIFIED: '$pre' and the after-state of both rows holds"
    exit 0
  fi
  log "NOT VERIFIED: the two held rows are not settled ('${pre:-unreadable}', done plan $DONE_S) — run scripts/ops/settle_mma_held_rows.sh"
  exit 1
fi

log "0/4 can it run? (one run at a time, no other ops run, no UFC card in progress)"
af_lock settle_mma_held_rows || exit 1
others=$($SETTLE_PS 2>/dev/null | grep -E 'scripts/ops/|soccer_repair_runbook' | grep -v 'settle_mma_held_rows' | grep -vE '(^|[[:space:]/])(grep|ps)( |$)')
if [ -n "$others" ]; then
  log "REFUSED: another ops run is in progress on the host — wait for it to end; nothing changed (no backup):"
  echo "$others"
  exit 1
fi
# shellcheck disable=SC2086  # the tool names split on purpose; the '*' is quoted so the shell never globs it
running=$(af_applies_running $UFC_WRITERS '*')
case $? in
  0) log "REFUSED: a tool's --apply is still running in the api container (a UFC writer, or another repair) — wait for it; nothing changed (no backup):"; echo "$running"; exit 1 ;;
  1) ;;
  *) log "REFUSED: could not scan the api container for running applies — nothing changed (no backup)"; exit 1 ;;
esac
cat > "$W/card.sql" <<'SQLEOF'
SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'mma' AND m.match_date BETWEEN NOW() - interval '8 hours' AND NOW() + interval '1 hour';
SQLEOF
card=$(psql_at "$W/card.sql" | tr -d '[:space:]')
if [ "$card" != "0" ]; then
  log "REFUSED: a UFC card is in progress or about to start (${card:-unreadable} UFC bout(s) with a kickoff in the last 8 hours or the next hour) — run it after the card; nothing changed (no backup)"
  exit 1
fi
log "0/4 run B done and verified? (repair_stranded_recs_mma.sh --verify-b: A's committed state, merge_mma_fighters.sh --verify, B's done check; read only)"
if ! bash "$OPS/repair_stranded_recs_mma.sh" --verify-b; then
  log "ABORT at step 0: run B (and the fighter merge before it) is not done and verified — see above; nothing changed (no backup)"
  exit 1
fi

log "0/4 already done? (the tool's pre-check: the whole transaction, rolled back)"
run_tool settle_mma_pre "$TOOL --out-csv /app/backups/settle_mma_held_rows_precheck_$STAMP.csv"
pre_rc=$?
pre=$(plan_of settle_mma_pre)
if [ "$pre_rc" = 0 ] && same_plan "$pre" "$DONE_S"; then
  if ! run_tool settle_mma_check "$TOOL --check-after"; then
    log "FAILED: the plan reads done ('$pre') but the after-state does not hold (above) — inspect /app/backups/settle_mma_held_rows_*; nothing changed"
    exit 1
  fi
  log "ALREADY DONE: '$pre' and the after-state of both rows holds — nothing to write (no backup)"
  exit 0
fi
REVIEWED=""
if [ "$pre_rc" = 0 ]; then
  for candidate in "$PLAN_S" "$PLAN_S_RECORDED" "$PLAN_S_GRADED"; do
    if same_plan "$pre" "$candidate"; then REVIEWED="$candidate"; break; fi
  done
fi
if [ -z "$REVIEWED" ]; then
  log "ABORT at step 0: the pre-check (exit $pre_rc) plans '${pre:-unreadable}', neither the reviewed plan ($PLAN_S, or a variant with the draw already recorded) nor the done plan ($DONE_S) — a row changed since the review, ESPN not serving or disagreeing, or a refusal (see above); nothing changed (no backup)"
  exit 1
fi
[ "$REVIEWED" = "$PLAN_S" ] || log "the draw is already recorded by the results path (a re-fetch since the review): the reviewed variant '$REVIEWED'"
log "not done yet (pre-plan '$pre') — continuing"

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

log "2/4 dry run (the whole transaction, rolled back)"
run_tool settle_mma_dry "$TOOL --out-csv /app/backups/settle_mma_held_rows_dryrun_$STAMP.csv"
rc=$?
plan=$(plan_of settle_mma_dry)
if [ "$rc" -ne 0 ] || ! same_plan "$plan" "$REVIEWED"; then
  log "ABORT at step 2: the dry run failed (exit $rc, see above) or its plan ('$plan') is not the pre-check's reviewed one ($REVIEWED) — review /app/backups/settle_mma_held_rows_dryrun_$STAMP.csv; nothing changed"
  exit 1
fi
log "plan: $plan"

pin=""
for k in $REVIEWED; do pin="$pin${pin:+,}$k"; done
log "3/4 apply (--expect-plan $pin)"
run_tool settle_mma_apply "$TOOL --apply --expect-plan $pin --out-csv /app/backups/settle_mma_held_rows_apply_$STAMP.csv --archive /app/backups/settle_mma_held_rows_apply_$STAMP.jsonl.gz"
rc=$?
if [ "$rc" -ne 0 ]; then
  why="it rolled back"
  [ "$rc" = "2" ] && why="it refused before writing (its plan differs from the reviewed one, or a refusal above)"
  log "ABORT at step 3 (exit $rc; full log /tmp/settle_mma_apply.log in the api container): $why — one transaction, nothing committed; a re-run starts from step 0"
  exit 1
fi

log "4/4 after-state"
run_tool settle_mma_after "$TOOL --check-after"
after_rc=$?
run_tool settle_mma_replan "$TOOL --out-csv /app/backups/settle_mma_held_rows_verify_$STAMP.csv"
replan=$(plan_of settle_mma_replan)
bash "$OPS/repair_stranded_recs_mma.sh" --verify-b
b_rc=$?
log "after: --check-after exit $after_rc (expected 0), re-plan '$replan' (expected $DONE_S), B's done check exit $b_rc (expected 0)"
if [ "$after_rc" -ne 0 ] || ! same_plan "$replan" "$DONE_S" || [ "$b_rc" -ne 0 ]; then
  log "FAILED after-state at step 4 — inspect (the apply's archive /app/backups/settle_mma_held_rows_apply_$STAMP.jsonl.gz and its CSV are the reversal record)"
  exit 1
fi
log "DONE: Gatto v Kareckaite is recorded as a draw (its predictions ungraded by convention), Valesca Machado v Melissa Amaya and the 'Valesca Machado' team row are deleted (archived); run B's done check reads its settled form"
