#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Close out the 21 pending UFC recommendations stranded on 21 bouts still
# 'scheduled' (kickoffs 2026-06-06 .. 2026-07-19) — the default run, A — and,
# in a SEPARATE opt-in run (--retire-stale-rows, B), retire the stale UFC
# fixture rows.
#
# WHY THEY STRANDED (prod 2026-11-01, read only):
#   * 20 bouts WERE fought, on the UFC cards of 2026-06-06, 06-14, 06-20 and
#     06-27, and ESPN still serves every winner. Results ingestion shipped
#     2026-07-13 and sweeps 3 days back, so no June bout was ever resulted:
#     all 50 June UFC rows sat 'scheduled'. There is no finished twin, so
#     repair_stranded_recs finds nothing to settle from, and voiding them
#     would destroy recs ESPN's results settle honestly. Same ESPN
#     competitions, same fighters, same order, same kickoff as our rows.
#   * 1 (Kevin Holland v Jacobe Smith, 2026-07-19 00:00Z, rec written
#     07-14) was dropped from its card: the finished UFC card of 2026-07-18
#     (ESPN event 600059599) lists neither fighter, and ESPN reports no
#     status for a dropped bout, so the row sat 'scheduled' for good.
#     fetch_upcoming's results path now cancels such rows
#     (cancel_dropped_bouts); the grader voids their open recs.
#   * Every one of the 21 recs was written before its kickoff (15 min before
#     its card segment; Holland 5 days before): nothing to void as hindsight.
#   No other sport holds stale pending recs except soccer (15, the soccer
#   runbook's) and horse racing (its own tables).
#
# Reviewed plan — a local replica of the 2026-11-01T02:42:11Z read-only
# export of every UFC match, rec, prediction, odds row and snapshot, with
# this commit's code and ESPN live:
#   A. (default) scripts/backfill_results.py --sport mma --dates $DATES, one
#      REPEATABLE READ transaction:
#        PLAN results=56 flipped=44 inserted=0 cancelled=3 stamped=12 rescored=0 other=0
#             recs_settled=20 recs_voided=1 recs_other=0 recs_left=0 fetch_failed=0 identity_refused=0
#      44 June rows take ESPN's result (the 20 with recs among them); 3 rows
#      are cancelled as dropped from their card (Holland v Smith, Jhonata
#      Diniz v Jose Luiz / v Jose Montanha, 2026-06-20); 12 finished rows of
#      the 07-18 card get their ESPN id stamped. The grader's own functions
#      settle 20 recs 6W/14L (stake 1,308.51, net -100.80) and void Holland
#      v Smith ('void: match cancelled'). Each rec's outcome is pinned in
#      scripts/data/mma_stranded_recs_20261101.csv (bout, ESPN competition,
#      winner, selection, odds, stake, P&L); the apply commits only with
#      exactly those outcomes (--expect-recs, checked inside the transaction),
#      and the dry run's audit CSV is checked against them too. A re-run
#      plans the done plan (DONE_A): results=56 and 0 for every other key.
#   B. (--retire-stale-rows, only once A is done and verified AND the fighter
#      merge, scripts/ops/merge_mma_fighters.sh, is done and verified)
#      scripts/repair_fixture_orphans.py --sport mma --as-of $AS_OF
#        --hold-list scripts/data/mma_stale_rows_hold_20261101.csv:
#        PLAN merge_delete=31 merge_cancel=0 delete_no_twin=31 delete_placeholder=4 mark_postponed=0
#             left=9 blocked_pending_recs=0 list_mismatch=0
#      Re-pinned 2026-11-01 to the state after A and the merge (a replica of
#      the 2026-11-01T22:47:06Z read-only export with A, then the merge,
#      applied): 31 duplicates merged onto 28 finished twins (strictly
#      pre-game odds, snapshots, predictions moved) and deleted; 31 rows
#      without a twin deleted, none of them fought (25 bouts whose opponent
#      was replaced, 6 dropped from their card); 4 'TBA v Opponent TBA' rows;
#      9 left: 7 because a rec references them (Holland v Smith's voided rec,
#      and 6 rows holding the prediction of a rec settled on a twin) and 2
#      HELD for the user's decision (the hold list: Gatto v Kareckaite, fought
#      to no winner — a draw or no contest, which results ingestion does not
#      record, so it never finished and has no twin; and Valesca Machado v
#      Melissa Amaya, whose ESPN competition now names Tina Black: a
#      replacement or a second spelling, undecided). Every row B may write,
#      its action and its twin are pinned in
#      scripts/data/mma_stale_rows_retire_20261101.csv with the ESPN
#      evidence: the apply refuses before its first write unless it plans
#      exactly those rows (--expect-rows), and the wrapper checks the
#      pre-check, dry-run and apply audit CSVs against them and the hold list.
#      --as-of is fixed, so rows that go stale later never enter this plan.
#      The 9 fought bouts stored under a second fighter spelling or the
#      other way round (Zachary/Zach Reese, Adrian Yanez/Yañez, Cristian
#      Perez/Pérez, Cameron/Cam Nelson, Ce Liu/Liu Ce, Gabriel
#      Lourenço/Lorenço, Sean King/Sean King III, Joseph/Joe Kropschot,
#      McGhee v Romero), which B deleted as no-twin before the merge, are
#      folded into their finished rows by the merge (user decision
#      2026-11-01). The 21 recs are closed out by A alone.
# RUN ORDER (MMA; scripts/ops/soccer_repair_runbook.sh is independent of these):
#   1. THIS script, default run (A)
#   2. scripts/ops/merge_mma_fighters.sh             (refuses unless A is done and verified)
#   3. THIS script --retire-stale-rows (B)           (refuses unless A and the merge are done and verified)
# Steps (A: 0-4 and 8; B: 0, 1, 5-8):
#   0. already done? A's dry run (it writes inside a transaction it always
#      rolls back) must plan exactly the reviewed plan (not done) or exactly
#      the done plan above — anything else (ESPN not serving, a partial
#      plan, other code) aborts before the backup. A done plan is not taken
#      on trust: the 21 recs must carry their reviewed outcomes in the
#      database, no UFC rec may be stranded, and the UFC recs created before
#      AS_OF must all be there — else abort ("A committed but not verified":
#      a run whose step 4 failed, or that died after its apply committed).
#      Default run, A done and verified: ALREADY DONE, exit 0 (no backup).
#      --retire-stale-rows: A must be done and verified, and so must the
#      fighter merge (scripts/ops/merge_mma_fighters.sh --verify, read only);
#      B's dry run (read only) must plan exactly the reviewed plan, its CSV
#      holding exactly the reviewed rows of
#      scripts/data/mma_stale_rows_retire_20261101.csv and holding back
#      exactly the rows of scripts/data/mma_stale_rows_hold_20261101.csv, or
#      exactly its done plan (0 writes, left=9, with 9 stale UFC rows and the
#      49 UFC recs in the database: ALREADY DONE, exit 0, no backup); anything
#      else aborts;
#   1. verified backup (B2) — abort if it does not land;
#   2. T0 = now; before-state + A's dry run (write-once audit CSV) — abort
#      unless the 21 reviewed recs are all still pending, the plan is exactly
#      the reviewed one, and the rec outcomes are exactly the reviewed ones;
#   3. A's apply, pinned to every key (--expect-plan) and to every rec's
#      outcome (--expect-recs); any drift rolls back;
#   4. A's after-state, EXACT: the 21 recs carry the reviewed status, P&L and
#      actual_result on a finished (settled) or cancelled (void) row; no
#      pending UFC rec is left on a stale row; the UFC recs created before T0
#      are all still there; a fresh dry run plans exactly the done plan;
#   5. B's before-state + dry run — abort unless the plan is exactly the
#      reviewed one (every key, left and blocked included) and its CSV
#      plans exactly the reviewed rows, actions and twins, and holds back
#      exactly the 2 held rows;
#   6. B's apply, pinned to the dry run's counts (--expect-plan) and to the
#      reviewed rows (--expect-rows: it refuses before its first write
#      unless it plans exactly those rows, actions and twins); the archive
#      lands before the first write;
#   7. B's after-state: the apply's own audit CSV plans exactly the reviewed
#      rows and the 2 held rows; the stale UFC rows (as of AS_OF) fell by
#      exactly merge_delete + delete_no_twin + delete_placeholder; the UFC
#      recs are all still there; a fresh dry run plans exactly B's done plan
#      (DONE_B);
#   8. follow-ups (derived data only): compute_features_mma --all-finished
#      --force (A: 44 new results change fighter histories; B: twins received
#      prices), then grade the ungraded predictions of finished UFC rows
#      (B: those moved onto twins). A failure exits 2; re-run them with
#        bash /opt/auspex/scripts/ops/repair_stranded_recs_mma.sh --followups-only
#      (a run that died before them prints ALREADY DONE on a re-run: run
#      --followups-only then).
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status.
#   --verify-a (scripts/ops/merge_mma_fighters.sh's step 0 calls it; read
#      only, no tool run, no ESPN): A's committed state in the database, as
#      step 0 verifies a done plan — exit 0 when verified, else 1.
# Exit: 0 done (or already done, or verified); 1 aborted or an after-state
# failed; 2 the repair is verified but a follow-up failed.
# WHEN: not while a UFC card is being fought (Saturday evening US time): the
# 15-min sweep then writes UFC rows. A's transaction is REPEATABLE READ, so a
# clash fails the run with a serialization error and nothing written — safe,
# but a wasted backup.
# Needs this commit on the VM (scripts/ is bind-mounted: a git pull is
# enough; the plan pins cancelled=3, which only this commit's fetch_upcoming
# produces, so older code aborts at step 0, before the backup).
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/repair_stranded_recs_mma.sh 2>&1 | tee -a ~/repair_stranded_recs_mma.log
# then scripts/ops/merge_mma_fighters.sh (see its header), then B:
#   bash /opt/auspex/scripts/ops/repair_stranded_recs_mma.sh --retire-stale-rows 2>&1 | tee -a ~/repair_stranded_recs_mma.log
# (AUSPEX_DIR and TMPDIR exist for tests and rehearsals.)
set -uo pipefail
MODE=a
case "${1:-}" in
  "") ;;
  --retire-stale-rows) MODE=b ;;
  --followups-only) MODE=followups ;;
  --verify-a) MODE=verify_a ;;
  *) echo "usage: $0 [--retire-stale-rows | --followups-only | --verify-a]   (see the header)"; exit 2 ;;
esac
OPS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || { echo "ABORT: cannot find scripts/ops — nothing changed"; exit 1; }
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
W=$(mktemp -d "${TMPDIR:-/tmp}/stranded_mma.XXXXXX") || { echo "ABORT: mktemp failed — nothing changed"; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/stranded_mma_q.sql >/dev/null 2>&1
}
trap cleanup EXIT
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_at() {  # psql_at <local.sql>: unaligned, '|'-separated
  $DC cp "$1" postgres:/tmp/stranded_mma_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/stranded_mma_q.sql'
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

DATES=2026-06-06,2026-06-14,2026-06-20,2026-06-27,2026-07-18
AS_OF=2026-11-01T00:00:00Z
REVIEWED=scripts/data/mma_stranded_recs_20261101.csv  # host path; /app/$REVIEWED in the api container
TOOL_A="python /app/scripts/backfill_results.py --sport mma --dates $DATES"
# The rows held for the user's decision (B leaves them alone: a fought draw / no contest, and a fighter swap).
HOLD_B=scripts/data/mma_stale_rows_hold_20261101.csv  # host path; /app/$HOLD_B in the api container
N_HELD_B=2
TOOL_B="python /app/scripts/repair_fixture_orphans.py --sport mma --as-of $AS_OF --hold-list /app/$HOLD_B"
PLAN_A="results=56 flipped=44 inserted=0 cancelled=3 stamped=12 rescored=0 other=0 recs_settled=20 recs_voided=1 recs_other=0 recs_left=0 fetch_failed=0 identity_refused=0"
DONE_A="results=56 flipped=0 inserted=0 cancelled=0 stamped=0 rescored=0 other=0 recs_settled=0 recs_voided=0 recs_other=0 recs_left=0 fetch_failed=0 identity_refused=0"
# Re-pinned 2026-11-01 to the state after A and scripts/ops/merge_mma_fighters.sh (it folds 9 of the 42 no-twin rows)
# with the 2 held rows left alone.
PLAN_B="merge_delete=31 merge_cancel=0 delete_no_twin=31 delete_placeholder=4 mark_postponed=0 left=9 blocked_pending_recs=0 list_mismatch=0"
# Every row B may write: match_id, action, twin_id (the CSV's other columns are the evidence).
REVIEWED_B=scripts/data/mma_stale_rows_retire_20261101.csv  # host path; /app/$REVIEWED_B in the api container
N_REVIEWED_B=66
DONE_B="merge_delete=0 merge_cancel=0 delete_no_twin=0 delete_placeholder=0 mark_postponed=0 left=9 blocked_pending_recs=0 list_mismatch=0"
# Pinned from the review (prod 2026-11-01, read only): every UFC rec was created before AS_OF (the last one
# 2026-08-29), and a rec's FKs are NO ACTION, so this count can only fall by a deliberate delete. B leaves the
# 7 stale rows that hold a rec and the 2 held rows.
RECS_BEFORE_AS_OF=49
STALE_AFTER_B=9
STAMP=$(date -u +%Y%m%dT%H%M%SZ)_$$  # the process id too: two runs within one second must not collide on write-once files
APPLY_A_CSV=/app/backups/backfill_results_mma_apply_$STAMP.csv
APPLY_B_CSV=/app/backups/fixture_orphans_mma_apply_$STAMP.csv

# run_tool <log name> <command>: the command in the api container; prints its summary lines and exit
run_tool() {
  $DC exec -T api sh -c "$2 > /tmp/$1.log 2>&1; rc=\$?; grep -E 'PLAN |candidates=|hold list|APPLIED|Archived|archived|DONE|pinned|ERROR|Traceback|Error' /tmp/$1.log | tail -25; echo \"tool exit \$rc\"; exit \$rc"
}
# The tool's own PLAN line only (KEY=N pairs), never a log line that merely mentions a plan.
plan_of() { $DC exec -T api sh -c "grep -oE 'PLAN [a-z_]+=[0-9]+( [a-z_]+=[0-9]+)*' /tmp/$1.log | tail -1"; }

# Follow-ups (step 8): derived data only, re-runnable; sets followup_failed.
run_followups() {
  local rc
  followup_failed=""
  log "8/8 follow-ups: recompute UFC features (every finished fight), then grade the ungraded predictions of finished UFC rows"
  $DC exec -T api sh -c "python /app/scripts/compute_features_mma.py --all-finished --force > /tmp/stranded_mma_features.log 2>&1; rc=\$?; tail -2 /tmp/stranded_mma_features.log; exit \$rc" </dev/null
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "ERROR step 8: compute_features_mma exited $rc (full log /tmp/stranded_mma_features.log in the api container)"
    followup_failed="the feature recompute"
  fi
  $DC exec -T api python - > "$W/grade.log" 2>&1 <<'PY'
import os, sys
sys.path.insert(0, "/app/scripts")
import psycopg2
from psycopg2.extras import RealDictCursor
import grade_completed_matches as g

# Predictions only (grade_match's own prediction loop): every rec of these
# rows is settled already. Re-runnable: only ungraded rows are read.
with psycopg2.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute(
        """SELECT DISTINCT m.id::text AS id FROM matches m JOIN leagues l ON l.id = m.league_id
            JOIN predictions p ON p.match_id = m.id
           WHERE l.sport = 'mma' AND m.status = 'finished' AND p.is_correct IS NULL"""
    )
    ids = [r["id"] for r in cur.fetchall()]
    graded = skipped = 0
    for mid in ids:
        for m in g.list_finished_matches(cur, 0, mid):
            ctx = g.fetch_match_grading_context(cur, mid)
            for pred in g.list_ungraded_predictions(cur, mid):
                skip, actual, correct = g.compute_prediction_grade(pred, m, ctx)
                if skip is not None:
                    skipped += 1
                    continue
                g.update_prediction(cur, pred["prediction_id"], actual, correct)
                graded += 1
    conn.commit()
print(f"graded {graded} predictions on {len(ids)} finished UFC rows ({skipped} skipped: after kickoff / ungradable)")
PY
  rc=$?
  tail -3 "$W/grade.log"
  if [ "$rc" -ne 0 ]; then
    log "ERROR step 8: grading the predictions exited $rc (see above)"
    followup_failed="${followup_failed:+$followup_failed and }the prediction grading"
  fi
}
incomplete() {
  log "INCOMPLETE (exit 2): the repair is verified, but $followup_failed failed. Re-run the follow-ups only:"
  log "    bash /opt/auspex/scripts/ops/repair_stranded_recs_mma.sh --followups-only"
  exit 2
}

if [ "$MODE" = followups ]; then
  run_followups
  [ -z "$followup_failed" ] || incomplete
  log "FOLLOW-UPS DONE"
  exit 0
fi

# The reviewed outcome of each rec, from the committed evidence file: id|status|profit_loss|actual_result.
if [ ! -r "$REVIEWED" ]; then log "ABORT: $REVIEWED is missing — pull this commit first; nothing changed"; exit 1; fi
if [ "$MODE" = b ]; then
  n_b=$( (grep -v '^#' "$REVIEWED_B" 2>/dev/null || true) | tail -n +2 | grep -c .)
  n_h=$( (grep -v '^#' "$HOLD_B" 2>/dev/null || true) | tail -n +2 | grep -c .)
  if [ "$n_b" != "$N_REVIEWED_B" ] || [ "$n_h" != "$N_HELD_B" ]; then
    log "ABORT: $REVIEWED_B holds ${n_b:-no} row(s) and $HOLD_B ${n_h:-no}; B is pinned to $N_REVIEWED_B and $N_HELD_B — pull this commit first; nothing changed"; exit 1
  fi
fi
reviewed=$(tail -n +2 "$REVIEWED" | awk -F, '{print $1 "|" $14 "|" $15 "|" $16}')
n_reviewed=$(echo "$reviewed" | grep -c .)
if [ "$n_reviewed" != "21" ] || echo "$reviewed" | grep -qvE '^[0-9a-f-]{36}\|(won|lost|void)\|-?[0-9]+\.[0-9]{2}\|[a-z: ]+$'; then
  log "ABORT: $REVIEWED does not hold the 21 reviewed recs in the expected shape — nothing changed"; exit 1
fi
reviewed_values=$(echo "$reviewed" | awk -F'|' '{printf "%s(%s::uuid, %s, %s::numeric, %s)", (NR > 1 ? ", " : ""), "'\''" $1 "'\''", "'\''" $2 "'\''", $3, "'\''" $4 "'\''"}')
reviewed_ids=$(echo "$reviewed" | awk -F'|' '{printf "%s%s", (NR > 1 ? ", " : ""), "'\''" $1 "'\''::uuid"}')

cat > "$W/reviewed.sql" <<SQLEOF
SELECT count(*) FROM betting_recommendations br JOIN matches m ON m.id = br.match_id
  JOIN (VALUES $reviewed_values) AS v(rec_id, status, profit_loss, actual_result) ON v.rec_id = br.id
 WHERE br.status = v.status AND br.profit_loss = v.profit_loss AND br.actual_result = v.actual_result
   AND br.settled_at IS NOT NULL
   AND m.status = CASE WHEN v.status = 'void' THEN 'cancelled' ELSE 'finished' END;
SQLEOF
cat > "$W/state_done.sql" <<SQLEOF
SELECT (SELECT count(*) FROM betting_recommendations br JOIN matches m ON m.id = br.match_id
          JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'mma' AND br.status IN ('pending', 'placed') AND m.status <> 'finished'
           AND m.match_date < NOW() - interval '24 hours') AS stranded,
       (SELECT count(*) FROM betting_recommendations br JOIN matches m ON m.id = br.match_id
          JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'mma' AND br.created_at < TIMESTAMPTZ '$AS_OF') AS recs;
SQLEOF
cat > "$W/state_b.sql" <<SQLEOF
SELECT (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'mma' AND m.status IN ('scheduled', 'cancelled')
           AND m.match_date < TIMESTAMPTZ '$AS_OF' - interval '96 hours') AS stale,
       (SELECT count(*) FROM betting_recommendations br JOIN matches m ON m.id = br.match_id
          JOIN leagues l ON l.id = m.league_id WHERE l.sport = 'mma') AS recs;
SQLEOF
read_state() {  # read_state <sql> <n fields>: the '|'-separated numbers, or empty
  local s
  s=$(psql_at "$1" | tr -d '[:space:]')
  if [[ "$s" =~ ^[0-9]+([|][0-9]+){$(($2 - 1))}$ ]]; then echo "$s"; fi
}
# A's committed state, read from the database (never inferred from a 0-write plan alone).
verify_a_done() {
  local matched state d_stranded="" d_recs=""
  matched=$(psql_at "$W/reviewed.sql" | tr -d '[:space:]')
  state=$(read_state "$W/state_done.sql" 2)
  [ -n "$state" ] && IFS='|' read -r d_stranded d_recs <<< "$state"
  log "A's committed state: reviewed recs as reviewed ${matched:-?} (expected 21), UFC recs stranded ${d_stranded:-?} (expected 0), UFC recs created before $AS_OF ${d_recs:-?} (expected $RECS_BEFORE_AS_OF)"
  [ "$matched" = "21" ] && [ "$d_stranded" = "0" ] && [ "$d_recs" = "$RECS_BEFORE_AS_OF" ]
}
not_verified() {
  log "ABORT at step 0: A looks committed ('$preA') but is not verified — a run whose step 4 failed, or one that died after its apply committed. Inspect the earlier run's /app/backups/backfill_results_mma_apply_*.csv and its backfill_results_mma_*.jsonl.gz archive (the reversal record) before anything else; nothing changed (no backup)"
  exit 1
}
# check_b_rows <B's audit CSV in the api container>: "OK <writes> held <n>" when the rows B plans to write (and their
# actions and twins) are exactly the reviewed ones and the rows it holds back are exactly the hold list, else
# "MISMATCH ...".
CHECK_B_PY="import csv, sys
writes = ('merge_delete', 'merge_cancel', 'delete_no_twin', 'delete_placeholder', 'mark_postponed')
rows = list(csv.DictReader(open(sys.argv[1])))
got = {(r['match_id'], r['action'], r['twin_id'] or '') for r in rows if r['action'] in writes}
want = {(r['match_id'], r['action'], r['twin_id'] or '') for r in csv.DictReader(l for l in open(sys.argv[2]) if not l.startswith('#'))}
held = {r['match_id'] for r in rows if r['action'] == 'leave' and r['reason'] == 'held_for_decision'}
hold = {r['match_id'] for r in csv.DictReader(l for l in open(sys.argv[3]) if not l.startswith('#'))}
extra, missing = sorted(got - want), sorted(want - got)
if extra or missing or held != hold:
    print('MISMATCH %d not reviewed %s, %d reviewed not planned %s, held %s, hold list %s' % (len(extra), extra[:3], len(missing), missing[:3], sorted(held), sorted(hold)))
else:
    print('OK %d held %d' % (len(got), len(held)))"
check_b_rows() { $DC exec -T api python -c "$CHECK_B_PY" "$1" "/app/$REVIEWED_B" "/app/$HOLD_B" </dev/null; }
ROWS_OK="OK $N_REVIEWED_B held $N_HELD_B"

if [ "$MODE" = verify_a ]; then
  if verify_a_done; then
    log "A VERIFIED: the 21 reviewed recs carry their reviewed outcomes, no UFC rec is stranded, the $RECS_BEFORE_AS_OF UFC recs are there"
    exit 0
  fi
  log "A NOT VERIFIED (above): run A (this script without arguments) has not committed, or its committed state is not the reviewed one"
  exit 1
fi

log "0/8 already done? (A's dry run rolls back$([ "$MODE" = b ] && echo "; B's only reads"))"
run_tool stranded_mma_a_pre "$TOOL_A --out-csv /app/backups/backfill_results_mma_precheck_$STAMP.csv"
preA=$(plan_of stranded_mma_a_pre)
if same_plan "$preA" "$PLAN_A"; then
  a_done=""
elif same_plan "$preA" "$DONE_A"; then
  a_done=1
  verify_a_done || not_verified
else
  log "ABORT at step 0: A's pre-check plan ('${preA:-unreadable}') is neither the reviewed plan ($PLAN_A) nor the done plan ($DONE_A) — ESPN not serving a day (fetch_failed), code other than this commit, or UFC rows changed since the review; nothing changed (no backup)"
  exit 1
fi

if [ "$MODE" = a ] && [ -n "$a_done" ]; then
  log "ALREADY DONE: A '$preA' — the 21 recs carry their reviewed outcomes; nothing to write (no backup). If an earlier run stopped before its follow-ups, run --followups-only. Retiring the stale UFC rows is the separate --retire-stale-rows run."
  exit 0
fi
if [ "$MODE" = b ]; then
  if [ -z "$a_done" ]; then
    log "ABORT at step 0: A is not done ('$preA') — run this wrapper without --retire-stale-rows first; nothing changed (no backup)"
    exit 1
  fi
  log "0/8 the fighter merge done and verified? (merge_mma_fighters.sh --verify, read only)"
  if ! bash "$OPS/merge_mma_fighters.sh" --verify; then
    log "ABORT at step 0: the fighter merge is not done and verified — run scripts/ops/merge_mma_fighters.sh first (B is pinned to the state it leaves); nothing changed (no backup)"
    exit 1
  fi
  run_tool stranded_mma_b_pre "$TOOL_B --out-csv /app/backups/fixture_orphans_mma_precheck_$STAMP.csv"
  preB=$(plan_of stranded_mma_b_pre)
  if same_plan "$preB" "$DONE_B"; then
    state=$(read_state "$W/state_b.sql" 2)
    if [ "$state" != "$STALE_AFTER_B|$RECS_BEFORE_AS_OF" ]; then
      log "ABORT at step 0: B looks done ('$preB') but the database shows stale UFC rows|UFC recs '${state:-unreadable}', not '$STALE_AFTER_B|$RECS_BEFORE_AS_OF' — inspect /app/backups/fixture_orphans_mma_*; nothing changed (no backup)"
      exit 1
    fi
    log "ALREADY DONE: A '$preA', B '$preB' (stale UFC rows|UFC recs $state) — nothing to write (no backup)"
    exit 0
  fi
  if ! same_plan "$preB" "$PLAN_B"; then
    log "ABORT at step 0: B's pre-check plan ('${preB:-unreadable}') is neither the reviewed plan ($PLAN_B) nor the done plan ($DONE_B) — a part-applied cleanup, UFC rows changed since the review, or other code; nothing changed (no backup)"
    exit 1
  fi
  rows=$(check_b_rows "/app/backups/fixture_orphans_mma_precheck_$STAMP.csv")
  if [ "$rows" != "$ROWS_OK" ]; then
    log "ABORT at step 0: B's pre-check plans other rows than the reviewed $REVIEWED_B and $HOLD_B ($rows) — nothing changed (no backup)"
    exit 1
  fi
  log "A and the fighter merge are done and verified; B is not ('$preB'; rows as reviewed: $rows) — continuing with B"
else
  log "not done yet (A pre-plan '$preA') — continuing"
fi

log "1/8 backup"
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

if [ "$MODE" = a ]; then
  T0=$(date -u +%FT%TZ)
  cat > "$W/state_a.sql" <<SQLEOF
SELECT (SELECT count(*) FROM betting_recommendations br JOIN matches m ON m.id = br.match_id
          JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'mma' AND br.status IN ('pending', 'placed') AND m.status <> 'finished'
           AND m.match_date < TIMESTAMPTZ '$T0' - interval '24 hours') AS stranded,
       (SELECT count(*) FROM betting_recommendations br JOIN matches m ON m.id = br.match_id
          JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'mma' AND br.created_at < TIMESTAMPTZ '$T0') AS recs,
       (SELECT count(*) FROM betting_recommendations WHERE id IN ($reviewed_ids) AND status = 'pending') AS reviewed_pending;
SQLEOF
  log "2/8 before-state + A's dry run (as of $T0)"
  state=$(read_state "$W/state_a.sql" 3)
  if [ -z "$state" ]; then log "ABORT at step 2: the before-state query failed — nothing changed"; exit 1; fi
  IFS='|' read -r stranded0 recs0 pending0 <<< "$state"
  log "before: UFC recs stranded on stale rows=$stranded0, UFC recs created before T0=$recs0, reviewed recs still pending=$pending0 (of 21)"
  if [ "$pending0" != "21" ]; then
    log "ABORT at step 2: only $pending0 of the 21 reviewed recs are still pending — something settled them since the review; nothing changed"
    exit 1
  fi
  run_tool stranded_mma_a_dry "$TOOL_A --expect-recs /app/$REVIEWED --out-csv /app/backups/backfill_results_mma_dryrun_$STAMP.csv"
  rc=$?
  plan=$(plan_of stranded_mma_a_dry)
  if [ "$rc" -ne 0 ] || ! same_plan "$plan" "$PLAN_A"; then
    log "ABORT at step 2: the dry run failed (exit $rc: a plan or a rec outcome not as reviewed, see above) or its plan ('$plan') is not the reviewed one ($PLAN_A) — review the dry-run CSV; nothing changed"
    exit 1
  fi
  CHECK_PY="import csv, json, sys
from decimal import Decimal
want = {r['rec_id']: r for r in csv.DictReader(open(sys.argv[2]))}
got = {r['id']: json.loads(r['after']) for r in csv.DictReader(open(sys.argv[1])) if r['kind'] == 'rec'}
bad = [i for i in want if i not in got or got[i]['status'] != want[i]['status']
       or Decimal(str(got[i]['profit_loss'])) != Decimal(want[i]['profit_loss'])
       or got[i]['actual_result'] != want[i]['actual_result']]
bad += [i for i in got if i not in want]
print('OK %d' % len(got) if not bad else 'MISMATCH ' + ' '.join(sorted(bad)))"
  check=$($DC exec -T api python -c "$CHECK_PY" "/app/backups/backfill_results_mma_dryrun_$STAMP.csv" "/app/$REVIEWED" </dev/null)
  if [ "$check" != "OK 21" ]; then
    log "ABORT at step 2: the dry run's rec outcomes are not the reviewed ones ($check) — nothing changed"; exit 1
  fi
  log "plan: $plan; rec outcomes as reviewed ($check)"

  pin=""
  for k in $PLAN_A; do pin="$pin${pin:+,}$k"; done
  log "3/8 A's apply (pinned: $pin; every rec's outcome pinned to $REVIEWED)"
  run_tool stranded_mma_a_apply "$TOOL_A --apply --expect-plan $pin --expect-recs /app/$REVIEWED --out-csv $APPLY_A_CSV"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "ABORT at step 3 (exit $rc; full log /tmp/stranded_mma_a_apply.log in the api container): a plan or a rec outcome that differs from the reviewed one rolls back — nothing committed unless the log says APPLIED; inspect before re-running (a re-run verifies a committed A from the database before anything else)"
    exit 1
  fi

  log "4/8 A's after-state"
  matched=$(psql_at "$W/reviewed.sql" | tr -d '[:space:]')
  state=$(read_state "$W/state_a.sql" 3)
  IFS='|' read -r stranded1 recs1 _ <<< "$state"
  run_tool stranded_mma_a_verify "$TOOL_A --out-csv /app/backups/backfill_results_mma_verify_$STAMP.csv"
  replan=$(plan_of stranded_mma_a_verify)
  log "after: reviewed recs as reviewed $matched (expected 21), UFC recs stranded ${stranded1:-?} (expected 0), UFC recs created before T0 ${recs1:-?} (expected $recs0), re-plan '$replan' (expected $DONE_A)"
  if [ "$matched" != "21" ] || [ "${stranded1:-}" != "0" ] || [ "${recs1:-}" != "$recs0" ] || ! same_plan "$replan" "$DONE_A"; then
    log "FAILED after-state at step 4 — inspect (the apply CSV $APPLY_A_CSV and its .jsonl.gz archive are the reversal record); a re-run checks A's committed state in the database first and refuses unless it is the reviewed one"
    exit 1
  fi
  log "A done: 20 recs settled (6W/14L), 1 voided (Holland v Smith), no UFC rec stranded"
  run_followups
  [ -z "$followup_failed" ] || incomplete
  log "DONE: the 21 stranded UFC recs are closed out. Retiring the stale UFC rows is the separate --retire-stale-rows run (see the header)."
  exit 0
fi

log "5/8 B's before-state + dry run (as of $AS_OF)"
state=$(read_state "$W/state_b.sql" 2)
if [ -z "$state" ]; then log "ABORT at step 5: the before-state query failed — nothing changed"; exit 1; fi
IFS='|' read -r stale0 recsb0 <<< "$state"
log "before: stale UFC rows (as of $AS_OF)=$stale0, UFC recs=$recsb0"
run_tool stranded_mma_b_dry "$TOOL_B --out-csv /app/backups/fixture_orphans_mma_dryrun_$STAMP.csv"
rc=$?
plan=$(plan_of stranded_mma_b_dry)
if [ "$rc" -ne 0 ] || ! same_plan "$plan" "$PLAN_B"; then
  log "ABORT at step 5: the dry run failed (exit $rc) or its plan ('$plan') is not the reviewed one ($PLAN_B) — review the dry-run CSV; nothing changed"
  exit 1
fi
rows=$(check_b_rows "/app/backups/fixture_orphans_mma_dryrun_$STAMP.csv")
if [ "$rows" != "$ROWS_OK" ]; then
  log "ABORT at step 5: the dry run plans other rows than the reviewed $REVIEWED_B and $HOLD_B ($rows) — nothing changed"
  exit 1
fi
md=$(plan_num "$plan" merge_delete); dn=$(plan_num "$plan" delete_no_twin); dp=$(plan_num "$plan" delete_placeholder)
log "plan: $plan; rows as reviewed ($rows)"

log "6/8 B's apply (pinned to the dry run's counts and to the reviewed rows, as of $AS_OF)"
run_tool stranded_mma_b_apply "$TOOL_B --apply --expect-plan merge_delete=$md,merge_cancel=0,delete_no_twin=$dn,delete_placeholder=$dp,mark_postponed=0 --expect-rows /app/$REVIEWED_B --out-csv $APPLY_B_CSV"
rc=$?
if [ "$rc" -ne 0 ]; then
  log "ABORT at step 6 (exit $rc; full log /tmp/stranded_mma_b_apply.log in the api container): if it refused before 'archived', nothing changed; otherwise the committed batches stand and the archive holds every planned row — inspect before re-running"
  exit 1
fi

log "7/8 B's after-state (as of $AS_OF)"
applied_rows=$(check_b_rows "$APPLY_B_CSV")
state=$(read_state "$W/state_b.sql" 2)
IFS='|' read -r stale1 recsb1 <<< "$state"
run_tool stranded_mma_b_verify "$TOOL_B --out-csv /app/backups/fixture_orphans_mma_verify_$STAMP.csv"
replan=$(plan_of stranded_mma_b_verify)
want_stale=$((stale0 - md - dn - dp))
log "after: the apply's rows '$applied_rows' (expected '$ROWS_OK'), stale UFC rows ${stale1:-?} (expected $want_stale), UFC recs ${recsb1:-?} (expected $recsb0), re-plan '$replan' (expected $DONE_B)"
if [ "$applied_rows" != "$ROWS_OK" ] || [ "${stale1:-}" != "$want_stale" ] || [ "${recsb1:-}" != "$recsb0" ] || ! same_plan "$replan" "$DONE_B"; then
  log "FAILED after-state at step 7 — inspect (the archive and audit CSVs are in /app/backups); a re-run refuses unless B's dry run plans exactly its done plan with $STALE_AFTER_B stale UFC rows and the $RECS_BEFORE_AS_OF UFC recs in the database"
  exit 1
fi

run_followups
[ -z "$followup_failed" ] || incomplete
log "DONE: the stale UFC fixture rows are retired, the $N_HELD_B held rows left for the user's decision ($HOLD_B); the 21 recs were closed out by the default run"
