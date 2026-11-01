#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Grade the rest of the predictions on the stale soccer rows
# repair_football_data_kickoffs.sh (runbook step 4) folded a football-data
# copy into, with scripts/grade_matches.py (pinned, one transaction).
# Runbook step 15. The fold turned each row 'finished', and the outcome
# trigger graded only the market types it knows (match_result, over_under,
# btts); the rows sit outside the pipeline grader's 14-day window, so their
# other predictions stay ungraded until this runs. It settles no rec: their
# recs are all 'void' already with a P&L (step 4 requires 0 without one), and
# grade_completed_matches settles only open recs and won/lost/void recs
# without a P&L.
#
# The folded rows: the soccer rows whose metadata the fold stamped
# (result_source 'football_data' with result_adopted_from); at most the fd
# repair's 26 (its MAX_FOLDS).
# Rehearsed 2026-11-01 on a local replica of a read-only prod export taken
# 02:42Z, after the runbook's steps 1-13 ran on it through a docker shim: 26
# rows, 504 predictions graded (asian_handicap, clean_sheet, correct_score,
# double_chance, draw_no_bet, odd_even, result_btts, result_over_under,
# team_total, total_goals, win_to_nil and winning_margin, 42 each), 0
# re-graded, 0 recs changed. The same on a 05:15Z export, and on a
# 06:48:46Z export after steps 1-14 (the labels repair included).
#   0. already done? A read-only dry run of the folded rows: nothing graded,
#      re-graded or changed exits 0 without a backup (a failed dry run aborts
#      here, before a backup);
#   1. verified backup (B2) — abort if it does not land;
#   2. dry run (write-once CSV of every change, old and new values) — abort
#      unless within the reviewed set: matches <= 26, predictions graded <=
#      504, none re-graded, no rec changed;
#   3. apply, pinned to the dry run's counts (--expect-plan, every key): one
#      transaction, committed only when its own plan equals the pin;
#   4. after-state, EXPECTED: a fresh dry run of the same rows grades,
#      re-grades and changes nothing.
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status; a pipe through tail would report tail's 0.
#
# RUN ORDER (scripts/ops/soccer_repair_runbook.sh runs it, each step through
# its own wrapper): 1 repoint_team_matches.sh --pass 1, 2 migrate_027_outcome_trigger.sh,
# 3 merge_soccer_teams.sh, 4 repair_football_data_kickoffs.sh,
# 5 repoint_team_matches.sh --pass 2, 6 repair_stranded_recs_soccer.sh,
# 7 cleanup_soccer_orphans.sh, 8 backfill_soccer_regulation_scores.sh,
# 9 backfill_season_labels.sh, 10 compute_features --backfill-finished soccer,
# 11 the merge's odds check, 12 backfill_soccer_never_resulted.sh (+ the
# cleanup again), 13 repair_soccer_labels.sh, 14 promote_raw_after_repair.sh,
# 15 THIS, 16 compute_features --backfill-finished soccer.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/grade_folded_soccer_rows.sh 2>&1 | tee -a ~/grade_folded_soccer_rows.log
# (AUSPEX_DIR and TMPDIR exist for tests and rehearsals.)
set -uo pipefail
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
W=$(mktemp -d "${TMPDIR:-/tmp}/grade_folded.XXXXXX") || { echo "ABORT: mktemp failed — nothing changed"; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/grade_folded_q.sql >/dev/null 2>&1
}
trap cleanup EXIT
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_at() {  # psql_at <local.sql>: unaligned, '|'-separated
  $DC cp "$1" postgres:/tmp/grade_folded_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/grade_folded_q.sql'
}
plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o " $2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
# The reviewed set (rehearsal of 2026-11-01, see the header).
MAX_FOLDED=26
MAX_GRADED=504
TOOL="python /app/scripts/grade_matches.py"
KEEP="PLAN|recs |DRY-RUN|Committed|REFUSED|ERROR|Traceback|Error"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
B=/app/backups/grade_folded_soccer_rows
IDS=${B}_ids_$STAMP.txt
step() {  # step <log name> <args...>: the tool in the api container; exits with its status
  local name="$1"; shift
  $DC exec -T api sh -c "$TOOL $* > /tmp/$name.log 2>&1; rc=\$?; grep -E '$KEEP' /tmp/$name.log | tail -12; echo \"tool exit \$rc\"; exit \$rc"
}
plan_of() { $DC exec -T api sh -c "grep -o 'PLAN .*' /tmp/$1.log | tail -1 | sed 's/ (audit CSV.*//'"; }

cat > "$W/grade_folded_ids.sql" <<'SQLEOF'
SELECT m.id FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'soccer' AND m.status = 'finished'
   AND m.metadata->>'result_source' = 'football_data' AND m.metadata ? 'result_adopted_from'
 ORDER BY m.id;
SQLEOF
log "0/4 already done? (read-only dry run of the folded rows)"
ids=$(psql_at "$W/grade_folded_ids.sql")
rc=$?
n=$(echo "$ids" | grep -cE '^[0-9a-f-]{36}$')
if [ "$rc" -ne 0 ] || [ "$n" -ne "$(echo "$ids" | grep -c .)" ]; then
  log "ABORT at step 0: the folded rows could not be read (exit $rc) — nothing changed"; exit 1
fi
if [ "$n" -eq 0 ]; then
  log "ALREADY DONE: no folded soccer row (repair_football_data_kickoffs.sh folded none) — nothing to do (no backup)"
  exit 0
fi
if [ "$n" -gt "$MAX_FOLDED" ]; then
  log "ABORT at step 0: $n folded rows, more than the fd repair's $MAX_FOLDED — inspect; nothing changed"; exit 1
fi
echo "$ids" | $DC exec -T api sh -c "cat > $IDS" || { log "ABORT at step 0: could not write $IDS — nothing changed"; exit 1; }
log "$n folded row(s) -> $IDS"
step grade_folded_pre --match-ids-file "$IDS" --out-csv ${B}_precheck_$STAMP.csv
rc=$?
pre=$(plan_of grade_folded_pre)
pre_vals="$(plan_num "$pre" predictions_graded) $(plan_num "$pre" predictions_regraded) $(plan_num "$pre" recs_changed)"
if [ "$rc" -ne 0 ] || ! [[ "$pre_vals" =~ ^[0-9]+\ [0-9]+\ [0-9]+$ ]]; then
  log "ABORT at step 0: the read-only dry run failed or printed no readable PLAN line (exit $rc, '${pre:-unreadable}'; full log /tmp/grade_folded_pre.log in the api container) — nothing changed"
  exit 1
fi
if [ "$(plan_num "$pre" predictions_graded) $(plan_num "$pre" predictions_regraded) $(plan_num "$pre" recs_changed)" = "0 0 0" ]; then
  log "ALREADY DONE: $pre — nothing left to grade on the folded rows (no backup)"
  exit 0
fi
log "not done yet (pre-plan '${pre:-unreadable}') — continuing"

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
step grade_folded_dry --match-ids-file "$IDS" --out-csv ${B}_dryrun_$STAMP.csv
rc=$?
plan=$(plan_of grade_folded_dry)
m=$(plan_num "$plan" matches); g=$(plan_num "$plan" predictions_graded)
rg=$(plan_num "$plan" predictions_regraded); rc_=$(plan_num "$plan" recs_changed)
if [ "$rc" -ne 0 ] || ! [[ "$m $g $rg $rc_" =~ ^[0-9]+(\ [0-9]+){3}$ ]]; then
  log "ABORT at step 2: the dry run failed or printed no readable PLAN line (exit $rc, '$plan'; full log /tmp/grade_folded_dry.log in the api container) — nothing changed"
  exit 1
fi
if [ "$m" -ne "$n" ] || [ "$g" -gt "$MAX_GRADED" ] || [ "$rg" -ne 0 ] || [ "$rc_" -ne 0 ]; then
  log "ABORT at step 2: plan ($plan) is outside the reviewed set (matches = the $n listed, predictions_graded <= $MAX_GRADED, predictions_regraded = recs_changed = 0) — review ${B}_dryrun_$STAMP.csv; nothing changed"
  exit 1
fi
if [ "$g" -eq 0 ]; then log "DONE: nothing to grade ($plan) — no write"; exit 0; fi

log "3/4 apply (pinned: matches=$m,predictions_graded=$g,predictions_regraded=0,recs_changed=0)"
step grade_folded_apply --match-ids-file "$IDS" --apply --expect-plan "matches=$m,predictions_graded=$g,predictions_regraded=0,recs_changed=0" --out-csv ${B}_apply_$STAMP.csv
rc=$?
log "step 3 exit $rc"
if [ "$rc" -ne 0 ]; then
  log "ABORT at step 3 (full log /tmp/grade_folded_apply.log in the api container): the apply is one transaction — a refusal or a drift rolled it back, nothing changed"
  exit 1
fi

log "4/4 after-state"
step grade_folded_verify --match-ids-file "$IDS" --out-csv ${B}_verify_$STAMP.csv
vrc=$?
vplan=$(plan_of grade_folded_verify)
v="$(plan_num "$vplan" matches) $(plan_num "$vplan" predictions_graded) $(plan_num "$vplan" predictions_regraded) $(plan_num "$vplan" recs_changed)"
log "after: re-plan '$vplan' (exit $vrc, expected matches=$n and nothing else)"
if [ "$vrc" -ne 0 ] || [ "$v" != "$n 0 0 0" ]; then
  log "FAILED after-state: a fresh dry run still grades something — inspect (${B}_apply_$STAMP.csv lists every change)"
  exit 1
fi
log "DONE: $g prediction(s) graded on the $n folded row(s); no rec changed"
