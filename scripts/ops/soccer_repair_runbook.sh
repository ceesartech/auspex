#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval. Every step below
# changes prod data through its own wrapper, and each wrapper takes its own
# verified B2 backup before its first write (CLAUDE.md: data repairs need a
# verified backup first) and checks its own exact after-state.
#
# THE SOCCER REPAIR RUNBOOK (2026-10-31): team identity, team merge, results
# path / grading / seasons, football-data kickoffs, soccer orphans, trigger
# 027 — the whole sequence, in order, one wrapper per step. Expected counts
# are prod's (read only, 2026-10-31) and the rehearsal's: this runbook and the
# real wrappers and tools, end to end through a docker/Airflow shim, against
# a local Postgres built from the migrations (011's trigger restored) and a
# read-only export of prod taken 11:08Z (compute_features stubbed).
# Start it right after the deploy: until step 1 runs, results for the mapped
# upcoming Liga MX fixtures are refused as identity drift, and the results
# sweep only looks back 3 days.
#
#  1 scripts/ops/repoint_team_matches.sh --pass 1
#    Moves 33 Liga MX sides (31 match rows) off Santos FC's 'Santos' and CD
#    Guadalajara's 'Guadalajara' onto Santos Laguna / Guadalajara Chivas with
#    their match_stats, stamps ESPN ids 2674/225/219. Expect plan 33 repoint /
#    0 done / 6 refused; step 2 refuses unless exactly the map's 39 sides sit
#    on santos/guadalajara. After: 0/33/6, 6 sides left.
#  2 scripts/ops/migrate_027_outcome_trigger.sh
#    Function body only (calculate_match_outcome skips post-kickoff rows), no
#    rows. Expect "97384c8327efa61669af1012c06c6232|1|1|1" and
#    "m027 pre_pred=true post_pred=null pre_rec=won post_rec=pending".
#  3 scripts/ops/merge_soccer_teams.sh
#    Drains then pauses auspex_pipeline + fetch_live_odds while it runs.
#    171 pairs merge (0 refused, 0 kickoffs adopted): 2,469 match sides
#    re-pointed, 308 twin fixtures folded (Lecce v Roma and Olympiakos v OFI
#    Crete take their ESPN twins' results), 171 team rows deleted, aliases and
#    ESPN ids recorded (168 stamped), 47 odds-feed spellings held (0 unmatched
#    before the merge; any would fail the step).
#  4 scripts/ops/repair_football_data_kickoffs.sh
#    dedupe merge 68, alias_merge 4, fold 26, refuse 0, fold_unsettled_recs 0;
#    shift <= 63,533 (63,532 on the 11:08Z export). Soccer rows fall by 98,
#    stale rows by 26; --check passes. The trigger grades ~138 predictions on
#    the folded rows and settles no rec.
#  5 scripts/ops/repoint_team_matches.sh --pass 2
#    Moves the 6 sides pass 1 refused (4 rows the fd repair alias-merged + 2
#    stale rows). Expect 6 repoint / 32 done / 1 refused (cb4e21fc 'match row
#    not found': folded by step 3); step 2 refuses unless exactly 6 sides are
#    left. After: 0 Liga MX rows on santos/guadalajara.
#  6 scripts/ops/repair_stranded_recs_soccer.sh
#    12 candidates: 7 recs voided as written after kickoff (profit 0, 'void:
#    written after kickoff') and re-pointed onto their twins, 0 settled, 5 left
#    on the row without a twin (Panathinaikos v Kifisia).
#  7 scripts/ops/cleanup_soccer_orphans.sh
#    Stale soccer rows: merges <= 204 (merge_delete + merge_cancel), World Cup
#    placeholders <= 65, ESPN-verified postponements <= 6, no no-twin delete,
#    0 blocked; recs untouched. Rehearsal: merge_delete 94, merge_cancel 19,
#    placeholders 65, postponed 6 (stale rows 288 -> 104). Then features +
#    grading for the twins (107; 800 predictions graded).
#  8 scripts/ops/backfill_soccer_regulation_scores.sh, then its re-grade
#    15 AET/PENS rows get their 90-minute score (5 change basis). The
#    re-grade (prediction rows only) is applied here ONLY when its dry run
#    plans 132-138 changes (138 on the rehearsal), pinned to that exact count
#    (--expect-changes); a fresh dry run must then plan 0.
#  9 scripts/ops/backfill_season_labels.sh
#    relabel_promote_raw <= 62,430 and relabel_espn <= 2,200, reviewed leagues
#    only (rehearsal: 62,156 and 1,603); after: re-plan 0.
# 10 compute_features.py --backfill-finished soccer
#    Recomputes features_cache for every finished soccer match (replaces the
#    per-step feature commands steps 1, 3 and 7 print). Skipped as already
#    done when it completed after every data-changing step's last change
#    (this runbook's state file, below).
# 11 merge_soccer_teams.py --check-after <plan> --odds-since <MERGED_AT> --after-later-steps
#    Read-only. Waits (up to ODDS_WAIT_POLLS x AF_POLL_SECONDS, 2 h) for a
#    fetch_live_odds run that started after the merge and finished; then
#    teams_reminted, resolver_misses, odds_not_yet_observed and
#    odds_fixtures_stopped must be 0 (and the merge's other after-state counts:
#    only repoints_missing is reported, not failed — steps 4 and 7 delete some
#    re-pointed rows on purpose; 17 on the rehearsal). The plan and MERGED_AT
#    come from /app/backups/merge_soccer_teams_last_apply.env.
#
# Before step 1 (read-only) it REFUSES TO START unless:
#   * the VM HEAD contains this runbook's own commit (the file being run is
#     the one committed at HEAD, its commit is an ancestor of HEAD — git
#     merge-base --is-ancestor — and scripts/ and services/ carry no local
#     edits: the bind-mounted code is the deployed code);
#   * void_post_kickoff_recs.sh has run (>= 22 recs carry its 'void: written
#     after kickoff (was ...)' label) and fix_unplayed_soccer_rows.sh has run
#     (its 4 rows are 'cancelled');
#   * retrain_models has no run in flight. It is paused for the whole run when
#     it is unpaused (drained first; Airflow 3.3 never advances a paused DAG's
#     runs) and unpaused again on every exit (trap); a DAG paused before the
#     run stays paused.
#
# AFTERWARDS (separate approvals, not run here): the never_resulted results
# backfill in cleanup_soccer_orphans.sh's header, then grade_completed_matches
# --days N; a promote_raw run, to re-write the 4 match_stats rows step 4
# archived for the alias-merged Liga MX fixtures (it refuses nothing once step
# 5 has run); the optional grading of the 26 folded rows (step 4 prints them).
#
# It stops at the first step that exits non-zero and prints "fix, then re-run
# with --from N". Every step is safe to re-run after it succeeded: each
# wrapper first re-plans read-only and exits 0 ("ALREADY DONE") when it has
# nothing left to do, without a backup or a write. A full re-run of a finished
# sequence therefore only reads.
#
# Usage (VM host; deploy the integrated branch to main first):
#   bash /opt/auspex/scripts/ops/soccer_repair_runbook.sh 2>&1 | tee -a ~/soccer_repair_runbook.log
#   ... --from N     run steps N..11
#   ... --only N     run step N alone
# State: RUNBOOK_STATE (default ~/.soccer_repair_runbook.state) records each
# step's completion ("changed", "already_done", or "checked" for step 11);
# only step 10's skip reads it, and a missing file only means step 10 runs.
# Read-only check on prod 2026-10-31 ~12:10Z (VM HEAD a071d6a, not deployed):
# the preflight reads 22 relabelled / 4 cancelled, 39 Liga MX sides sit on
# santos/guadalajara, calculate_match_outcome is 011's.
# (AUSPEX_DIR, TMPDIR, AF_POLL_SECONDS and ODDS_WAIT_POLLS exist for tests and rehearsals.)
set -uo pipefail
FROM=1
ONLY=""
usage() { echo "usage: $0 [--from N | --only N]   (N = 1..11; see the header)"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --from) [ $# -ge 2 ] || { usage; exit 2; }; FROM="$2"; shift 2 ;;
    --only) [ $# -ge 2 ] || { usage; exit 2; }; ONLY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
for n in "$FROM" ${ONLY:+"$ONLY"}; do
  if ! [[ "$n" =~ ^([1-9]|1[01])$ ]]; then usage; exit 2; fi
done

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
OPS="$(dirname "$SELF")"
REL=scripts/ops/soccer_repair_runbook.sh
AUSPEX_DIR="${AUSPEX_DIR:-/opt/auspex}"
cd "$AUSPEX_DIR" || { echo "ABORT: cannot cd to $AUSPEX_DIR"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] runbook: $*"; }
# shellcheck source=scripts/ops/_airflow_lib.sh
. "$OPS/_airflow_lib.sh" || { log "ABORT: cannot load $OPS/_airflow_lib.sh"; exit 1; }
STATE_FILE="${RUNBOOK_STATE:-$HOME/.soccer_repair_runbook.state}"
ODDS_WAIT_POLLS="${ODDS_WAIT_POLLS:-120}"
RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/soccer_runbook.XXXXXX") || { log "ABORT: mktemp failed"; exit 1; }
finish() {
  af_resume
  rm -f "$RUN_DIR"/*.sql
  $DC exec -T postgres rm -f /tmp/soccer_runbook_q.sql >/dev/null 2>&1
  log "logs of this run: $RUN_DIR"
}
trap finish EXIT
trap 'exit 130' INT TERM HUP
should_run() { if [ -n "$ONLY" ]; then [ "$1" = "$ONLY" ]; else [ "$1" -ge "$FROM" ]; fi; }

log "start (from ${ONLY:-$FROM}${ONLY:+ only}); AUSPEX_DIR=$AUSPEX_DIR"

# ---- preflight (read-only) -------------------------------------------------
head=$(git -C "$AUSPEX_DIR" rev-parse HEAD 2>/dev/null)
head_blob=$(git -C "$AUSPEX_DIR" rev-parse "HEAD:$REL" 2>/dev/null)
self_blob=$(git -C "$AUSPEX_DIR" hash-object "$SELF" 2>/dev/null)
self_commit=$(git -C "$AUSPEX_DIR" log -1 --format=%H HEAD -- "$REL" 2>/dev/null)
if [ -z "$head" ] || [ -z "$head_blob" ] || [ "$head_blob" != "$self_blob" ] || [ -z "$self_commit" ]; then
  log "REFUSED: this runbook ($SELF) is not the one committed at the VM HEAD (${head:-no git HEAD} in $AUSPEX_DIR) — deploy the integrated branch (push to main, let CI deploy) and run /opt/auspex/$REL"
  exit 1
fi
if ! git -C "$AUSPEX_DIR" merge-base --is-ancestor "$self_commit" HEAD; then
  log "REFUSED: the runbook's commit $self_commit is not an ancestor of HEAD $head"
  exit 1
fi
if ! git -C "$AUSPEX_DIR" diff --quiet HEAD -- scripts services; then
  log "REFUSED: scripts/ or services/ differ from HEAD $head (bind-mounted code that is not the deployed commit):"
  git -C "$AUSPEX_DIR" diff --stat HEAD -- scripts services | tail -5
  exit 1
fi
log "VM HEAD $head contains the runbook's commit $self_commit; scripts/ and services/ are clean"

cat > "$RUN_DIR/preflight.sql" <<'SQLEOF'
BEGIN READ ONLY;
SELECT (SELECT count(*) FROM betting_recommendations
         WHERE actual_result LIKE 'void: written after kickoff (was %') || '|' ||
       (SELECT count(*) FROM matches
         WHERE id IN ('d7d8a9b7-39a6-4340-bb53-09982171fa8a', 'ff74d919-d543-4729-a438-5f59e826b7b9',
                      '681794ae-f1e5-4e10-a365-8e71f86ef4a1', '83ce848f-64ad-4147-ab32-4a770a5eed15')
           AND status = 'cancelled');
ROLLBACK;
SQLEOF
$DC cp "$RUN_DIR/preflight.sql" postgres:/tmp/soccer_runbook_q.sql >/dev/null 2>&1
pre=$($DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/soccer_runbook_q.sql' 2>&1 | grep -E '^[0-9]+\|[0-9]+$' | tail -1)
relabelled=${pre%%|*}; unplayed=${pre##*|}
if ! [[ "$pre" =~ ^[0-9]+\|[0-9]+$ ]]; then log "REFUSED: the preflight query failed (got '$pre')"; exit 1; fi
if [ "$relabelled" -lt 22 ]; then
  log "REFUSED: void_post_kickoff_recs.sh has not run ($relabelled recs carry its label, it relabels 22) — run it first"
  exit 1
fi
if [ "$unplayed" != "4" ]; then
  log "REFUSED: fix_unplayed_soccer_rows.sh has not run ($unplayed of its 4 rows are cancelled) — run it first"
  exit 1
fi
log "void_post_kickoff_recs has run ($relabelled relabelled) and fix_unplayed_soccer_rows has run (4/4 cancelled)"

if ! af_drain_and_pause 1 4 retrain_models; then
  log "REFUSED: retrain_models has a run in flight or cannot be paused — start the runbook after it finishes"
  exit 1
fi

# ---- steps -----------------------------------------------------------------
record() {  # record <step> <changed|already_done>
  echo "step=$1 result=$2 at=$(date -u +%s) head=$head" >> "$STATE_FILE" 2>/dev/null \
    || log "WARNING: could not write $STATE_FILE (step 10's skip then re-runs it)"
}
run_step() {  # run_step <n> <label> <command...>
  local n="$1" label="$2" rc result
  shift 2
  should_run "$n" || return 0
  log "STEP $n/11 START: $label"
  "$@" 2>&1 | tee "$RUN_DIR/step$n.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    if [ "$n" = "7" ] && [ "$rc" = "2" ]; then
      log "STEP 7 INCOMPLETE (exit 2): the cleanup is verified but its feature/grading follow-ups failed — re-run them by hand (commands above), then re-run with --from 8"
    else
      log "STEP $n FAILED (exit $rc): $label — fix, then re-run with --from $n"
    fi
    exit "$rc"
  fi
  result=changed
  grep -q "ALREADY DONE" "$RUN_DIR/step$n.log" && result=already_done
  grep -q "CHECKED (read-only)" "$RUN_DIR/step$n.log" && result=checked
  record "$n" "$result"
  log "STEP $n/11 DONE ($result)"
}

step8() {
  local out="$RUN_DIR/step8_wrapper.log" line ids types changes min max rc
  bash "$OPS/backfill_soccer_regulation_scores.sh" 2>&1 | tee "$out"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || return "$rc"
  line=$(grep '^REGRADE ' "$out" | tail -1)
  ids=$(echo "$line" | sed -n 's/.* ids_file=\([^ ]*\).*/\1/p')
  types=$(echo "$line" | sed -n 's/.* types=\([^ ]*\).*/\1/p')
  changes=$(echo "$line" | sed -n 's/.* changes=\([0-9]*\).*/\1/p')
  min=$(echo "$line" | sed -n 's/.* min=\([0-9]*\).*/\1/p')
  max=$(echo "$line" | sed -n 's/.* max=\([0-9]*\).*/\1/p')
  if [ -z "$ids" ] || [ -z "$types" ] || [ -z "$changes" ] || [ -z "$min" ] || [ -z "$max" ]; then
    log "the wrapper printed no readable REGRADE line ('$line')"; return 1
  fi
  if [ "$changes" = "0" ]; then
    log "re-grade: ALREADY DONE (the dry run plans 0 changes)"; return 0
  fi
  if [ "$changes" -lt "$min" ] || [ "$changes" -gt "$max" ]; then
    log "re-grade: $changes changes planned, outside the reviewed $min-$max — not applied"; return 1
  fi
  log "re-grade: applying $changes change(s), pinned (--expect-changes $changes)"
  $DC exec -T api sh -c "python /app/scripts/regrade_predictions.py --match-ids-file $ids --prediction-type $types --apply --expect-changes $changes --out-csv /app/backups/regrade_regulation_apply_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/runbook_regrade.log 2>&1; rc=\$?; grep -E 'Scanned|Rewrote|Pinned|ERROR|Done:|Traceback|Error' /tmp/runbook_regrade.log | tail -8; exit \$rc"
  rc=$?
  [ "$rc" -eq 0 ] || { log "re-grade apply exited $rc (a pin mismatch writes nothing)"; return "$rc"; }
  $DC exec -T api sh -c "python /app/scripts/regrade_predictions.py --match-ids-file $ids --prediction-type $types --out-csv /app/backups/regrade_regulation_verify_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/runbook_regrade_verify.log 2>&1; rc=\$?; grep -o '[0-9]* change(s) planned' /tmp/runbook_regrade_verify.log | tail -1; exit \$rc"
  rc=$?
  left=$($DC exec -T api sh -c "grep -o '[0-9]* change(s) planned' /tmp/runbook_regrade_verify.log | tail -1 | cut -d' ' -f1")
  if [ "$rc" -ne 0 ] || [ "$left" != "0" ]; then log "re-grade verify: '${left:-unreadable}' change(s) still planned (want 0)"; return 1; fi
  log "re-grade applied: $changes prediction row(s); a fresh dry run plans 0"
}

step10_done_already() {  # 0 when step 10 last completed after every data change of steps 1,3,4,5,7
  [ -r "$STATE_FILE" ] || return 1
  local last10 lastchg
  last10=$(sed -nE 's/^step=10 result=[a-z_]+ at=([0-9]+).*/\1/p' "$STATE_FILE" | sort -n | tail -1)
  lastchg=$(sed -nE 's/^step=(1|3|4|5|7) result=changed at=([0-9]+).*/\2/p' "$STATE_FILE" | sort -n | tail -1)
  [ -n "$last10" ] && [ "$last10" -gt "${lastchg:-0}" ]
}
step10() {
  if step10_done_already; then
    log "ALREADY DONE: features were recomputed after the last data-changing step (state file $STATE_FILE)"
    return 0
  fi
  $DC exec -T api sh -c "python /app/scripts/compute_features.py --backfill-finished soccer > /tmp/runbook_features.log 2>&1; rc=\$?; tail -5 /tmp/runbook_features.log; exit \$rc"
}

step11() {
  local st plan merged_at runs starts s found="" i rc
  st=$($DC exec -T api sh -c "cat /app/backups/merge_soccer_teams_last_apply.env 2>/dev/null")
  plan=$(echo "$st" | sed -n 's/^PLAN=//p'); merged_at=$(echo "$st" | sed -n 's/^MERGED_AT=//p')
  if [ -z "$plan" ] || [ -z "$merged_at" ]; then
    log "no merge apply recorded in /app/backups/merge_soccer_teams_last_apply.env (got '$st') — step 3 has not applied"; return 1
  fi
  log "merge plan $plan, merged at $merged_at; waiting for a fetch_live_odds run that started after it to finish"
  if [ "$(af_dag_paused fetch_live_odds)" = "True" ]; then log "fetch_live_odds is paused — unpause it, then re-run with --from 11"; return 1; fi
  for i in $(seq 1 "$ODDS_WAIT_POLLS"); do
    runs=$($DC exec -T airflow-scheduler sh -c "airflow dags list-runs fetch_live_odds --state success -o json 2>/dev/null | grep '^\\[' | tail -1")
    starts=$(echo "$runs" | grep -o '"start_date": "[^"]*"' | cut -d'"' -f4 | cut -c1-19)
    for s in $starts; do
      if [[ "$s" > "${merged_at:0:19}" ]]; then found="$s"; break; fi
    done
    [ -n "$found" ] && break
    sleep "$AF_POLL_SECONDS"
  done
  if [ -z "$found" ]; then log "no fetch_live_odds run has finished since $merged_at after $ODDS_WAIT_POLLS polls — re-run with --from 11 later"; return 1; fi
  log "fetch_live_odds run started $found finished; checking odds and resolution since $merged_at"
  $DC exec -T api sh -c "python /app/scripts/merge_soccer_teams.py --check-after $plan --odds-since $merged_at --after-later-steps > /tmp/runbook_odds_after.log 2>&1; rc=\$?; grep -E 'AFTER|REFUSED|ERROR|stopped|repoints_missing|Traceback' /tmp/runbook_odds_after.log | tail -15; exit \$rc"
  rc=$?
  [ "$rc" -eq 0 ] && log "CHECKED (read-only): no merged team re-minted, every merged spelling resolves, the odds feed still reaches the merged clubs' fixtures"
  return "$rc"
}

run_step 1 "repoint_team_matches --pass 1" bash "$OPS/repoint_team_matches.sh" --pass 1
run_step 2 "migrate_027_outcome_trigger" bash "$OPS/migrate_027_outcome_trigger.sh"
run_step 3 "merge_soccer_teams" bash "$OPS/merge_soccer_teams.sh"
run_step 4 "repair_football_data_kickoffs" bash "$OPS/repair_football_data_kickoffs.sh"
run_step 5 "repoint_team_matches --pass 2" bash "$OPS/repoint_team_matches.sh" --pass 2
run_step 6 "repair_stranded_recs_soccer" bash "$OPS/repair_stranded_recs_soccer.sh"
run_step 7 "cleanup_soccer_orphans" bash "$OPS/cleanup_soccer_orphans.sh"
run_step 8 "backfill_soccer_regulation_scores + re-grade" step8
run_step 9 "backfill_season_labels" bash "$OPS/backfill_season_labels.sh"
run_step 10 "compute_features --backfill-finished soccer" step10
run_step 11 "merge odds after-check" step11
log "ALL STEPS DONE (${ONLY:+step $ONLY}${ONLY:-from step $FROM})"
