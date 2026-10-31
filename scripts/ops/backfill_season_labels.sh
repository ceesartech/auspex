#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Relabel soccer matches.season to the one season scheme
# (scripts/soccer_seasons.py) with scripts/backfill_season_labels.py.
# DEPLOY THE COMMIT THAT SHIPS scripts/soccer_seasons.py FIRST: until then
# fetch_upcoming keeps writing the Aug->May label on calendar-year leagues and
# promote_raw the 'YYZZ' misreading, so new wrong rows keep arriving.
#
# Reviewed plan: read-only prod export 2026-10-31 07:13Z, rehearsed with this
# script on a local replica (171,677 soccer rows, 168,077 raw rows, all tied):
#   relabel_promote_raw 62,430  the old promote_raw label of the raw season:
#                               32,794 '2020-2116'-style, 26,890 'YYYY/YYYY',
#                               2,746 '2020-2021' for a 2021 calendar season
#                               (AR1 AT1 BR1 CH1 CN1 DK1 FI1 IE1 JP1 MLS MX1
#                               NO1 PL1 RO1 RU1 SE1)
#   relabel_espn         1,792  fetch_upcoming's Aug->May label on a
#                               calendar-year league: AR1 274, BR1 196, CL1
#                               119, CN1 139, CO1 249, JP1 20 (its 2026
#                               special season), LIB 38, MLS 305, NO1 131, SE1
#                               151, WC 170
#   already_correct    107,455  left alone 0 (no unknown writer, no raw
#                               disagreement, no row without a date rule)
# MAX_RELABEL_ESPN allows ~4-5 weeks of new calendar-league fixture rows
# (~80-100 a week in Oct 2026) written before the deploy; promote_raw is not
# scheduled, so its count can only shrink. Past either ceiling, re-plan
# read-only and review before raising it.
#   0. already done? A read-only dry run first: when it plans 0 relabels
#      (relabel_promote_raw and relabel_espn), this exits 0 without a backup
#      (a re-run after success);
#   1. verified backup (B2) — abort if it does not land;
#   2. before-state + dry run (write-once audit CSV) — abort if the tool fails
#      or the plan is outside the reviewed set (the ceilings below, and only
#      the reviewed leagues per kind);
#   3. apply, pinned to the dry run's counts (--expect-plan: any drift refuses
#      before a write); one transaction, every UPDATE guarded on the old
#      label, a row changed since the plan rolls everything back;
#   4. after-state, EXPECTED: every row of the apply CSV holds its new label
#      (--verify), the soccer rows with a label only the old promote_raw wrote
#      ('YYYY/YYYY', a 'YYYY-YYYY' not spanning one year) fell by exactly the
#      plan's relabel_bad_shape, and a fresh dry run plans 0 relabels with the
#      same left-alone counts.
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status; a pipe through tail would report tail's 0.
# Re-runnable: a relabelled row plans as already_correct (step 0 above).
#
# RUN ORDER (scripts/ops/soccer_repair_runbook.sh runs it, each step through
# its own wrapper): 1 repoint_team_matches.sh --pass 1, 2 migrate_027_outcome_trigger.sh,
# 3 merge_soccer_teams.sh, 4 repair_football_data_kickoffs.sh,
# 5 repoint_team_matches.sh --pass 2, 6 repair_stranded_recs_soccer.sh,
# 7 cleanup_soccer_orphans.sh, 8 backfill_soccer_regulation_scores.sh (+ re-grade),
# 9 THIS (after the deletes of steps 3, 4 and 7, which only lower its counts),
# 10 compute_features --backfill-finished soccer, 11 the merge's odds check.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/backfill_season_labels.sh 2>&1 | tee -a ~/backfill_season_labels.log
# (AUSPEX_DIR and TMPDIR exist for tests and rehearsals.)
set -uo pipefail
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
W=$(mktemp -d "${TMPDIR:-/tmp}/season_labels.XXXXXX") || { echo "ABORT: mktemp failed — nothing changed"; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/season_labels_q.sql >/dev/null 2>&1
}
trap cleanup EXIT
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_file() {  # psql_file <local.sql> [extra psql flags]
  local f="$1"; shift
  $DC cp "$f" postgres:/tmp/season_labels_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c "psql -v ON_ERROR_STOP=1 -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" $* -f /tmp/season_labels_q.sql"
}
plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o " $2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
within() {  # within <comma list> <comma list>: every item of $1 is in $2 ('-' is empty)
  local item
  [ "$1" = "-" ] && return 0
  for item in ${1//,/ }; do [[ ",$2," == *",$item,"* ]] || return 1; done
}
TOOL="python /app/scripts/backfill_season_labels.py"
MAX_RELABEL_PROMOTE_RAW=62430
MAX_RELABEL_ESPN=2200
REVIEWED_PROMOTE_RAW_LEAGUES="AR1,AT1,BR1,CH1,CN1,DK1,FI1,IE1,JP1,MLS,MX1,NO1,PL1,RO1,RU1,SE1"
REVIEWED_ESPN_LEAGUES="AR1,BR1,CL1,CN1,CO1,JP1,LIB,MLS,NO1,SE1,WC"
LEFT_KEYS="left_unknown_writer left_no_date_rule left_raw_disagree left_league_mismatch left_no_league_code"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
APPLY_CSV=/app/backups/season_labels_apply_$STAMP.csv
cat > $W/season_labels_shapes.sql <<'SQLEOF'
SELECT CASE WHEN m.season ~ '^[0-9]{4}/[0-9]{4}$' THEN 'YYYY/YYYY (old promote_raw)'
            WHEN m.season ~ '^[0-9]{4}-[0-9]{4}$'
                 AND split_part(m.season, '-', 2)::int - split_part(m.season, '-', 1)::int = 1 THEN 'YYYY-YYYY one year'
            WHEN m.season ~ '^[0-9]{4}-[0-9]{4}$' THEN 'YYYY-YYYY other span (old promote_raw)'
            WHEN m.season ~ '^[0-9]{4}$' THEN 'YYYY'
            ELSE coalesce('other: ' || m.season, 'NULL') END AS label_shape,
       count(*) AS rows
  FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'soccer'
 GROUP BY 1 ORDER BY 1;
SQLEOF
# The same test as backfill_season_labels.is_bad_shape.
cat > $W/season_labels_bad.sql <<'SQLEOF'
SELECT count(*)
  FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'soccer'
   AND (m.season ~ '^[0-9]{4}/[0-9]{4}$'
        OR (m.season ~ '^[0-9]{4}-[0-9]{4}$'
            AND split_part(m.season, '-', 2)::int - split_part(m.season, '-', 1)::int <> 1));
SQLEOF

log "0/4 already done? (read-only dry run)"
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/season_labels_precheck_$STAMP.csv > /tmp/season_labels_pre.log 2>&1; echo \"tool exit \$?\""
pre=$($DC exec -T api sh -c "grep '^PLAN ' /tmp/season_labels_pre.log | tail -1")
if [ "$(plan_num "$pre" relabel_promote_raw) $(plan_num "$pre" relabel_espn) $(plan_num "$pre" relabel)" = "0 0 0" ]; then
  log "ALREADY DONE: $pre — nothing to relabel (no backup)"
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

log "2/4 before-state + dry run"
psql_file $W/season_labels_shapes.sql || { log "ABORT at step 2: the before-state query failed — nothing changed"; exit 1; }
bad0=$(psql_file $W/season_labels_bad.sql -At | tr -d '[:space:]')
if ! [[ "$bad0" =~ ^[0-9]+$ ]]; then log "ABORT at step 2: the bad-label count failed (got '$bad0') — nothing changed"; exit 1; fi
log "before: soccer rows with a label only the old promote_raw wrote: $bad0"
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/season_labels_dryrun_$STAMP.csv > /tmp/season_labels_dry.log 2>&1; rc=\$?; grep -E '^PLAN|Loaded|Planned|  [A-Z0-9]+ +(promote_raw|espn)|ERROR|DRY-RUN|Traceback|Error' /tmp/season_labels_dry.log | tail -40; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 2 exit $rc"
if [ "$rc" -ne 0 ]; then log "ABORT at step 2: the dry run failed (see above; full log /tmp/season_labels_dry.log in the api container) — nothing changed"; exit 1; fi
plan=$($DC exec -T api sh -c "grep '^PLAN ' /tmp/season_labels_dry.log | tail -1")
leagues=$($DC exec -T api sh -c "grep '^PLAN_LEAGUES ' /tmp/season_labels_dry.log | tail -1")
rp=$(plan_num "$plan" relabel_promote_raw); re=$(plan_num "$plan" relabel_espn)
rb=$(plan_num "$plan" relabel_bad_shape); ac=$(plan_num "$plan" already_correct)
if ! [[ "$rp $re $rb $ac" =~ ^[0-9]+\ [0-9]+\ [0-9]+\ [0-9]+$ ]]; then
  log "ABORT at step 2: could not read the PLAN line (got '$plan') — nothing changed"; exit 1
fi
left0=""
for k in $LEFT_KEYS; do left0="$left0 $k=$(plan_num "$plan" "$k")"; done
lp=$(echo "$leagues" | grep -o 'promote_raw=[^ ]*' | cut -d= -f2); le=$(echo "$leagues" | grep -o 'espn=[^ ]*' | cut -d= -f2)
if [ -z "$lp" ] || [ -z "$le" ]; then log "ABORT at step 2: could not read the PLAN_LEAGUES line (got '$leagues') — nothing changed"; exit 1; fi
if [ "$rp" -gt "$MAX_RELABEL_PROMOTE_RAW" ] || [ "$re" -gt "$MAX_RELABEL_ESPN" ] \
   || ! within "$lp" "$REVIEWED_PROMOTE_RAW_LEAGUES" || ! within "$le" "$REVIEWED_ESPN_LEAGUES"; then
  log "ABORT at step 2: plan ($plan; $leagues) is outside the reviewed set (relabel_promote_raw <= $MAX_RELABEL_PROMOTE_RAW in $REVIEWED_PROMOTE_RAW_LEAGUES, relabel_espn <= $MAX_RELABEL_ESPN in $REVIEWED_ESPN_LEAGUES) — review the dry-run CSV; nothing changed"; exit 1
fi
log "plan: relabel_promote_raw=$rp relabel_espn=$re (bad-shape labels among them: $rb), already_correct=$ac;$left0"

log "3/4 apply (pinned to the dry run's counts)"
$DC exec -T api sh -c "$TOOL --apply --expect-plan relabel_promote_raw=$rp,relabel_espn=$re,relabel_bad_shape=$rb --out-csv $APPLY_CSV > /tmp/season_labels_apply.log 2>&1; rc=\$?; grep -E '^PLAN |Planned|Relabelled|ERROR|Traceback|Error' /tmp/season_labels_apply.log | tail -12; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 3 exit $rc"
if [ "$rc" -ne 0 ]; then
  # A plan drift refuses before any write; a row changed between plan and
  # write raises and rolls the one transaction back.
  log "ABORT at step 3 (see above; full log /tmp/season_labels_apply.log in the api container): nothing was written"
  exit 1
fi

log "4/4 after-state"
$DC exec -T api sh -c "$TOOL --verify $APPLY_CSV > /tmp/season_labels_verify.log 2>&1; rc=\$?; grep -E '^VERIFY|ERROR|Traceback|Error' /tmp/season_labels_verify.log; exit \$rc"
vrc=$?
verify=$($DC exec -T api sh -c "grep '^VERIFY ' /tmp/season_labels_verify.log | tail -1")
psql_file $W/season_labels_shapes.sql
bad1=$(psql_file $W/season_labels_bad.sql -At | tr -d '[:space:]')
want_bad=$((bad0 - rb)); want_rows=$((rp + re))
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/season_labels_verify_$STAMP.csv > /tmp/season_labels_replan.log 2>&1; rc=\$?; grep '^PLAN ' /tmp/season_labels_replan.log; exit \$rc"
prc=$?
replan=$($DC exec -T api sh -c "grep '^PLAN ' /tmp/season_labels_replan.log | tail -1")
left1=""
for k in $LEFT_KEYS; do left1="$left1 $k=$(plan_num "$replan" "$k")"; done
log "after: verify '$verify' (expected rows=$want_rows now_new=$want_rows), bad-shape labels $bad1 (expected $want_bad), re-plan relabel=$(plan_num "$replan" relabel) (expected 0),$left1 (expected$left0)"
if [ "$vrc" -ne 0 ] || [ "$(plan_num "$verify" rows)" != "$want_rows" ] || [ "$(plan_num "$verify" now_new)" != "$want_rows" ] \
   || [ "$bad1" != "$want_bad" ] || [ "$prc" -ne 0 ] || [ "$(plan_num "$replan" relabel)" != "0" ] || [ "$left1" != "$left0" ]; then
  log "FAILED after-state — inspect (the apply CSV $APPLY_CSV is the rollback record: old_season per match_id)"
  exit 1
fi
log "DONE"
