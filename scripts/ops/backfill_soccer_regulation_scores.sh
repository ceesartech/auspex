#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Stamp the 90-minute score (metadata.regulation_home_score /
# regulation_away_score + went_extra) on the finished soccer ties decided in
# extra time or on penalties, with scripts/backfill_soccer_regulation_scores.py.
# Read-only check 2026-10-31: 15 rows (5 AET, 10 FT-PENS; 2026 World Cup and
# Copa Libertadores), and a local dry run of the tool against ESPN planned all
# 15: five change basis (Belgium v Senegal 3-2 AET was 2-2 after 90 minutes,
# Argentina v Cape Verde 3-2 -> 1-1, Norway v England 1-2 -> 1-1, Argentina v
# Switzerland 3-1 -> 1-1, Spain v Argentina 1-0 -> 0-0); the ten shootouts keep
# their score. Deploy the commit that ships the tool FIRST: the graders read
# the keys only from then on.
#   1. verified backup (B2) — abort if it does not land;
#   2. before-state + dry run (plan + write-once audit CSV) — abort if the tool
#      refuses any row or fails;
#   3. apply (one transaction, every UPDATE guarded on the observed row);
#   4. after-state — expects 0 finished soccer rows whose result_detail says
#      decided after 90 minutes and that still lack the 90-minute score;
#   5. the re-grade DRY RUN of exactly the stamped matches (ids from this run's
#      apply CSV; read-only, its own write-once CSV): every soccer full-time
#      type (grading_outcomes.SOCCER_FT_TYPES), every model. EXPECTED on the
#      15 matches of 2026-10-31 (rehearsed with this script on a local replica
#      of a read-only export, the stamps planned against ESPN): 138 of the 560
#      graded rows change, all ensemble — 113 on the five AET ties (74 verdicts
#      move: match_result 9 True->False, double_chance 7, winning_margin 10
#      False->True, draw_no_bet 10 -> push, ...; 39 only rewrite the DB
#      trigger's 'over'/'under' strings) and 25 string-only rewrites on the ten
#      shootouts. 6 of those 25 are on the Libertadores ties of 2026-10-22/23,
#      which grade_completed_matches re-grades by itself once they carry the
#      score, while they sit in its 14-day window (to 2026-11-05/06), so a run
#      may plan as few as 132. The --apply command is printed; it is not run
#      here (its own approval).
#      It prints one line for the runbook:
#        REGRADE ids_file=<file> types=<list> changes=<n> min=<n> max=<n>
#      and the --apply command, pinned (--expect-changes) to that dry run.
#   0. already done? A read-only dry run first: when every candidate is
#      already stamped (0 to stamp, 0 refused, 0 left), it skips the backup
#      and the apply and only re-runs step 5 on the stamped matches: 0 changes
#      means the re-grade has been applied too (ALREADY DONE, exit 0); a count
#      inside the reviewed range means the re-grade is still to apply (exit 0,
#      the REGRADE line and command printed); anything else exits 1.
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status; a pipe through tail would report tail's 0.
# Re-runnable: a stamped row plans as already_set (step 0 above).
#
# RUN ORDER (scripts/ops/soccer_repair_runbook.sh runs it, each step through
# its own wrapper): 1 repoint_team_matches.sh --pass 1, 2 migrate_027_outcome_trigger.sh,
# 3 merge_soccer_teams.sh, 4 repair_football_data_kickoffs.sh,
# 5 repoint_team_matches.sh --pass 2, 6 repair_stranded_recs_soccer.sh,
# 7 cleanup_soccer_orphans.sh, 8 THIS (+ the re-grade it prints, which the
# runbook applies only inside the reviewed range), 9 backfill_season_labels.sh,
# 10 compute_features --backfill-finished soccer, 11 the merge's odds check.
# It does not depend on the order: no AET/PENS row has a fork, and the rows on
# twin teams match by ESPN id or by find_team with the ESPN team id.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/backfill_soccer_regulation_scores.sh 2>&1 | tee -a ~/backfill_soccer_regulation_scores.log
# (AUSPEX_DIR and TMPDIR exist for tests and rehearsals.)
set -uo pipefail
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
W=$(mktemp -d "${TMPDIR:-/tmp}/regulation_scores.XXXXXX") || { echo "ABORT: mktemp failed — nothing changed"; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/regulation_scores_q.sql >/dev/null 2>&1
}
trap cleanup EXIT
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_file() {  # psql_file <local.sql> [extra psql flags]
  local f="$1"; shift
  $DC cp "$f" postgres:/tmp/regulation_scores_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c "psql -v ON_ERROR_STOP=1 -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" $* -f /tmp/regulation_scores_q.sql"
}
TOOL="python /app/scripts/backfill_soccer_regulation_scores.py"
EXPECTED_LEFT=0
EXPECTED_REGRADE_MIN=132
EXPECTED_REGRADE_MAX=138
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
APPLY_CSV=/app/backups/soccer_regulation_scores_apply_$STAMP.csv
IDS_FILE=/app/backups/soccer_regulation_regrade_ids_$STAMP.txt
REGRADE="python /app/scripts/regrade_predictions.py"

cat > $W/regulation_scores_state.sql <<'SQLEOF'
SELECT l.external_ids->>'football_data' AS league, (m.match_date AT TIME ZONE 'UTC')::date AS day,
       ht.name AS home, at.name AS away, m.home_score || '-' || m.away_score AS final,
       m.metadata->>'result_detail' AS detail,
       (m.metadata->>'regulation_home_score') || '-' || (m.metadata->>'regulation_away_score') AS ninety,
       m.metadata->>'went_extra' AS went_extra
  FROM matches m
  JOIN leagues l ON l.id = m.league_id
  JOIN teams ht ON ht.id = m.home_team_id
  JOIN teams at ON at.id = m.away_team_id
 WHERE l.sport = 'soccer' AND m.status = 'finished'
   AND m.metadata->>'result_detail' ~* '\m(AET|PENS?)\M'
 ORDER BY m.match_date;
SQLEOF
cat > $W/regulation_scores_left.sql <<'SQLEOF'
SELECT count(*)
  FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'soccer' AND m.status = 'finished'
   AND m.metadata->>'result_detail' ~* '\m(AET|PENS?)\M'
   AND NOT (m.metadata ? 'regulation_home_score' AND m.metadata ? 'regulation_away_score');
SQLEOF

plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o " $2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
# regrade_dry <audit CSV> <ids file>: the re-grade dry run of the matches the CSV stamped or found
# stamped; prints the REGRADE line; 0 inside the reviewed range or at 0 changes, 1 otherwise. Sets CHANGES.
regrade_dry() {
  local csv="$1" ids="$2" n_ids types rc trc
  IDS_PY="import csv, sys; ids = sorted({r['match_id'] for r in csv.DictReader(open(sys.argv[1])) if r['action'] in ('stamp', 'already_set')}); open(sys.argv[2], 'x').write(''.join(i + chr(10) for i in ids)); print(len(ids))"
  n_ids=$($DC exec -T api python -c "$IDS_PY" "$csv" "$ids")
  rc=$?
  types=$($DC exec -T api python -c "import sys; sys.path.insert(0, '/app/scripts'); import grading_outcomes as g; print(','.join(sorted(g.SOCCER_FT_TYPES)))")
  trc=$?
  if [ "$rc" -ne 0 ] || [ "$trc" -ne 0 ] || ! [[ "$n_ids" =~ ^[0-9]+$ ]] || ! [[ "$types" =~ ^[a-z_,]+$ ]]; then
    log "FAILED step 5: could not read the stamped ids from $csv (got '$n_ids') or the full-time types (got '$types') — the stamps are committed and verified; build the re-grade by hand"
    CHANGES=""
    return 1
  fi
  log "$n_ids stamped match(es) in $ids; types $types"
  $DC exec -T api sh -c "$REGRADE --match-ids-file $ids --prediction-type $types --out-csv /app/backups/regrade_regulation_dryrun_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/regulation_regrade_dry.log 2>&1; rc=\$?; grep -vE ' INFO - Limited' /tmp/regulation_regrade_dry.log | grep -E 'Scanned|/|ERROR|WARNING|Traceback|Error' | tail -40; echo \"tool exit \$rc\"; exit \$rc"
  rc=$?
  CHANGES=$($DC exec -T api sh -c "grep -o '[0-9]* change(s) planned' /tmp/regulation_regrade_dry.log | tail -1 | cut -d' ' -f1")
  log "re-grade dry run exit $rc: ${CHANGES:-unknown} prediction row(s) would change (expected $EXPECTED_REGRADE_MIN-$EXPECTED_REGRADE_MAX on the 15 matches of 2026-10-31, or 0 once applied)"
  echo "REGRADE ids_file=$ids types=$types changes=${CHANGES:-unknown} min=$EXPECTED_REGRADE_MIN max=$EXPECTED_REGRADE_MAX"
  if [ "$rc" -ne 0 ] || ! [[ "$CHANGES" =~ ^[0-9]+$ ]]; then return 1; fi
  if [ "$CHANGES" -eq 0 ]; then return 0; fi
  echo "Apply it (separate approval; the runbook's step 8 runs it only inside $EXPECTED_REGRADE_MIN-$EXPECTED_REGRADE_MAX):"
  echo "  docker compose exec -T api $REGRADE --match-ids-file $ids --prediction-type $types --apply --expect-changes $CHANGES --out-csv /app/backups/regrade_regulation_apply_$STAMP.csv"
  [ "$CHANGES" -ge "$EXPECTED_REGRADE_MIN" ] && [ "$CHANGES" -le "$EXPECTED_REGRADE_MAX" ]
}

log "0/5 already done? (read-only dry run)"
PRE_CSV=/app/backups/soccer_regulation_scores_precheck_$STAMP.csv
$DC exec -T api sh -c "$TOOL --out-csv $PRE_CSV > /tmp/regulation_scores_pre.log 2>&1; echo \"tool exit \$?\""
pre=$($DC exec -T api sh -c "grep -o 'PLAN candidates=.*' /tmp/regulation_scores_pre.log | tail -1")
pre_left=$(psql_file "$W/regulation_scores_left.sql" -At | tr -d '[:space:]')
if [ -n "$pre" ] && [ "$(plan_num "$pre" stamp) $(plan_num "$pre" refuse)" = "0 0" ] && [ "$pre_left" = "$EXPECTED_LEFT" ]; then
  log "stamps already done ($pre; $pre_left left) — no backup, no apply; checking the re-grade"
  regrade_dry "$PRE_CSV" "/app/backups/soccer_regulation_regrade_ids_precheck_$STAMP.txt"
  ok=$?
  if [ "$ok" -eq 0 ] && [ "$CHANGES" = "0" ]; then
    log "ALREADY DONE: every candidate stamped and the re-grade applied (0 changes left) — nothing to do"
    exit 0
  elif [ "$ok" -eq 0 ]; then
    log "DONE (stamps already in place): the re-grade above is still to apply"
    exit 0
  fi
  log "CHECK: the re-grade dry run failed or is outside the reviewed $EXPECTED_REGRADE_MIN-$EXPECTED_REGRADE_MAX — inspect before applying"
  exit 1
fi
log "not done yet (pre-plan '${pre:-unreadable}', $pre_left left) — continuing"

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

log "2/5 before-state + dry run"
psql_file $W/regulation_scores_state.sql || { log "ABORT at step 2: the before-state query failed — nothing changed"; exit 1; }
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/soccer_regulation_scores_dryrun_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/regulation_scores_dry.log 2>&1; rc=\$?; grep -E 'stamp|already_set|refuse|candidate|ERROR|DRY-RUN|Done:|Traceback|Error' /tmp/regulation_scores_dry.log | tail -24; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 2 exit $rc"
if [ "$rc" -ne 0 ]; then log "ABORT at step 2: the dry run refused a row or failed (see above; full log /tmp/regulation_scores_dry.log in the api container) — nothing changed"; exit 1; fi

log "3/5 apply"
$DC exec -T api sh -c "$TOOL --apply --out-csv $APPLY_CSV > /tmp/regulation_scores_apply.log 2>&1; rc=\$?; grep -E 'candidate|Stamped|ERROR|Done:|Traceback|Error' /tmp/regulation_scores_apply.log | tail -12; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 3 exit $rc"
if [ "$rc" -ne 0 ]; then
  # A refusal writes nothing; a row that changed between plan and write raises
  # and rolls the whole transaction back.
  log "ABORT at step 3 (see above; full log /tmp/regulation_scores_apply.log in the api container): nothing was written"
  exit 1
fi

log "4/5 after-state"
psql_file $W/regulation_scores_state.sql
left=$(psql_file $W/regulation_scores_left.sql -At | tr -d '[:space:]')
log "finished soccer rows decided after 90 minutes still without the 90-minute score: ${left:-unknown} (expected $EXPECTED_LEFT)"
if [ "$left" != "$EXPECTED_LEFT" ]; then log "FAILED: ${left:-unknown} row(s) left — inspect the dry-run reasons above"; exit 1; fi

log "5/5 re-grade dry run of the stamped matches (read-only)"
# The stamped matches (and any a re-run finds already stamped) from this run's
# write-once apply CSV, and every soccer full-time type, from the code.
if regrade_dry "$APPLY_CSV" "$IDS_FILE" && [ "$CHANGES" != "0" ]; then
  log "DONE"
else
  log "CHECK: the re-grade dry run failed or is outside the reviewed $EXPECTED_REGRADE_MIN-$EXPECTED_REGRADE_MAX — the stamps are committed and verified; inspect the dry-run summary before applying"
  exit 1
fi
