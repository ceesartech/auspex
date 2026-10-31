#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Move the Liga MX rows ESPN put on the wrong team rows, with
# scripts/repoint_team_matches.py and the reviewed map
# scripts/data/repoint_team_matches.csv (39 sides; read-only check 2026-10-31):
#   'Santos' (Santos FC, Brasileirao) holds 20 Liga MX rows of Santos Laguna
#     (ESPN 225, display name 'Santos') from 2026-07-19 -> 'Santos Laguna';
#   'Guadalajara' (CD Guadalajara, Segunda 2011-13) holds 19 Liga MX rows of
#     Chivas (ESPN 219) from 2026-07-19 -> 'Guadalajara Chivas'.
# and stamp the seeded ESPN ids (2674 Santos, 225 Santos Laguna, 219 Guadalajara
# Chivas). Predictions and recs keep their match_id. TWO PASSES:
#
#   --pass 1 (runbook step 1, right after the deploy; until it runs, every
#     soccer sweep meeting these clubs' ESPN fixtures exits 3): 33 sides move,
#     0 done, 6 refused — each has a football-data copy on the target pair 1 h
#     later (a same-pair twin) that repair_football_data_kickoffs alias-merges
#     first. Step 2 refuses unless exactly the map's 39 sides are the Liga MX
#     sides on santos/guadalajara (a fixture filed there that the map does not
#     name would pass the plan and fail the after-state). After: re-plan 0
#     repoint / 33 done / 6 refused, left=6, the three stamps.
#   --pass 2 (runbook step 5, after merge_soccer_teams and
#     repair_football_data_kickoffs): 6 sides move (4 ESPN rows the fd repair
#     alias-merged + the 2 stale rows that move with them: a398e0fe/e78e7e49
#     Monterrey v Santos Laguna, 9a6bcea3 Santos Laguna v Atlas, d1f660b6/
#     b03fc978 Guadalajara Chivas v Toluca, 30e7868a Puebla v Guadalajara
#     Chivas), 32 done (13 of them through an opponent the team merge renamed,
#     'América' -> 'Club America': the tool reads teams.metadata.aliases /
#     merged_from), 1 refused: cb4e21fc 'match row not found' (the merge
#     folded that stale fork into its played twin 45064ae5). Step 2 refuses
#     unless exactly those 6 sides are left on santos/guadalajara, and every
#     run pins that the 1 refusal is 'match row not found' (--expect-not-found
#     1; pass 1 pins 0). After: re-plan 0 / 38 / 1 (the not-found row), left=0,
#     stamps unchanged.
#     Rehearsed on the 11:08Z read-only export (pass 1, merge, fd repair,
#     pass 2) on 2026-10-31.
#
#   0. already done? A read-only dry run first: pass 1 is done when every side
#      but the 6 twins reads 'done' (or 'match row not found': folded away by
#      the merge) and at most those 6 are left on santos/guadalajara; pass 2 when
#      nothing plans 'repoint', every refusal is 'match row not found' and 0
#      are left. Then it exits 0 without a backup (a re-run after success);
#   1. verified backup (B2) — abort if it does not land; T0 = now;
#   2. dry run pinned to the pass's plan (write-once audit CSV) and the
#      before-state above — abort if either differs;
#   3. apply (one transaction, rows locked, every UPDATE guarded, same pins);
#   4. after-state: the pass's re-plan, the stamps, the Liga MX sides left on
#      santos/guadalajara, and the soccer rows created before T0 unchanged
#      (rows inserted after T0 — fixture inserts land at 00:00 UTC — are not
#      this tool's and never fail it);
#   then it prints the compute_features command (NOT run; the runbook's
#   step 10 recomputes every finished soccer match).
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status; a pipe through tail would report tail's 0.
# The second pass leaves 4 match_stats rows archive-only (the fd repair
# archived the other club's stats of the 4 alias-merged fixtures): a
# promote_raw run after it (separate approval) writes them again.
#
# RUN ORDER (scripts/ops/soccer_repair_runbook.sh runs it, each step through
# its own wrapper): 1 THIS --pass 1, 2 migrate_027_outcome_trigger.sh,
# 3 merge_soccer_teams.sh, 4 repair_football_data_kickoffs.sh, 5 THIS --pass 2,
# 6 repair_stranded_recs_soccer.sh, 7 cleanup_soccer_orphans.sh,
# 8 backfill_soccer_regulation_scores.sh (+ re-grade), 9 backfill_season_labels.sh,
# 10 compute_features --backfill-finished soccer, 11 the merge's odds check.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/repoint_team_matches.sh --pass 1 2>&1 | tee -a ~/repoint_team_matches.log
# (AUSPEX_DIR and TMPDIR exist for tests and rehearsals.)
set -uo pipefail
PASS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --pass) [ $# -ge 2 ] || { echo "usage: $0 --pass 1|2"; exit 2; }; PASS="$2"; shift 2 ;;
    --pass=*) PASS="${1#--pass=}"; shift ;;
    *) echo "usage: $0 --pass 1|2"; exit 2 ;;
  esac
done
case "$PASS" in 1|2) ;; *) echo "usage: $0 --pass 1|2 (see the header)"; exit 2 ;; esac
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] pass $PASS: $*"; }
W=$(mktemp -d "${TMPDIR:-/tmp}/repoint_team.XXXXXX") || { echo "ABORT: mktemp failed — nothing changed"; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/repoint_team_q.sql >/dev/null 2>&1
}
trap cleanup EXIT
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_file() {  # psql_file <local.sql> [extra psql flags]
  local f="$1"; shift
  $DC cp "$f" postgres:/tmp/repoint_team_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c "psql -v ON_ERROR_STOP=1 -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" $* -f /tmp/repoint_team_q.sql"
}
plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o " $2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
TOOL="python /app/scripts/repoint_team_matches.py"
MAP=scripts/data/repoint_team_matches.csv
STAMPS="Guadalajara Chivas:219,Santos:2674,Santos Laguna:225"
# The 6 sides pass 1 refuses (a football-data copy on the target pair).
PASS1_LEFT="a398e0fe-e7cb-4c71-9231-009f343e9c50,e78e7e49-9ce9-4584-8e35-7c5e0048150c,9a6bcea3-1740-4c89-8f43-fda6b1c5267c,d1f660b6-edce-4297-89bd-fb6993b88c6d,b03fc978-cd00-41c8-a0f8-c11c49e032a5,30e7868a-9c93-4bf7-bdf9-7c44de661c5b"
MAP_SIDES=$(tail -n +2 "$MAP" | grep -c .)
# --expect-not-found pins how many refusals are 'match row not found' (a row
# the team merge folded away): pass 1 none, pass 2 exactly cb4e21fc. Without
# it, one side that genuinely drifted while cb4e21fc read 'done' would give
# pass 2 the same 6/32/1 and pass.
if [ "$PASS" = "1" ]; then
  PINS="--expect-repoint 33 --expect-done 0 --expect-refused 6 --expect-not-found 0"
  AFTER="0 33 6 0"        # re-plan repoint done refuse not_found
  LEFT_BEFORE=$MAP_SIDES
  LEFT_AFTER=6
else
  PINS="--expect-repoint 6 --expect-done 32 --expect-refused 1 --expect-not-found 1"
  AFTER="0 38 1 1"
  LEFT_BEFORE=6
  LEFT_AFTER=0
fi

T0=$(date -u +%FT%TZ)
STATE_SQL="$W/repoint_team_state.sql"
cat > "$STATE_SQL" <<SQLEOF
SELECT 'stamps=' || string_agg(name || ':' || coalesce(external_ids->>'espn', '-'), ',' ORDER BY name)
  FROM teams WHERE sport = 'soccer' AND normalized_name IN ('santos', 'santos laguna', 'guadalajara chivas');
SELECT 'left=' || count(*)
  FROM matches m JOIN leagues l ON l.id = m.league_id JOIN teams t ON t.id IN (m.home_team_id, m.away_team_id)
 WHERE l.name = 'Liga MX' AND l.sport = 'soccer' AND t.sport = 'soccer' AND t.normalized_name IN ('santos', 'guadalajara');
SELECT 'soccer_before_t0=' || count(*)
  FROM matches m JOIN leagues l ON l.id = m.league_id WHERE l.sport = 'soccer' AND m.created_at < TIMESTAMPTZ '$T0';
SQLEOF
state_of() {  # state_of <key>: from the last state query
  echo "$STATE" | sed -n "s/^$1=//p"
}
read_state() { STATE=$(psql_file "$STATE_SQL" -At); }
plan_of() {  # plan_of <log name>: the PLAN line that run logged
  $DC exec -T api sh -c "grep -o 'PLAN repoint=.*' /tmp/$1.log | tail -1"
}
# Sides that are neither done, folded away ('match row not found'), nor (pass 1) one of the 6 twins.
UNDONE_PY="import csv, sys; left = set(sys.argv[2].split(',')) if sys.argv[2] else set(); bad = [r['match_id'] + ' ' + r['side'] + ': ' + r['action'] + ' ' + r['reason'] for r in csv.DictReader(open(sys.argv[1])) if r['kind'] == 'match' and r['action'] != 'done' and r['reason'] != 'match row not found' and r['match_id'] not in left]; print(len(bad)); print(chr(10).join(bad[:10]))"

log "0/4 already done? (read-only dry run)"
PRE_CSV=/app/backups/repoint_team_matches_p${PASS}_precheck_$(date -u +%Y%m%dT%H%M%SZ).csv
$DC exec -T api sh -c "$TOOL --out-csv $PRE_CSV > /tmp/repoint_team_pre.log 2>&1; echo \"tool exit \$?\""
pre=$(plan_of repoint_team_pre)
read_state
if [ "$PASS" = "1" ]; then left_ok=$([ "$(state_of left)" -le 6 ] 2>/dev/null && echo 1); undone_ids="$PASS1_LEFT"; else left_ok=$([ "$(state_of left)" = "0" ] && echo 1); undone_ids=""; fi
undone=$($DC exec -T api python -c "$UNDONE_PY" "$PRE_CSV" "$undone_ids" 2>/dev/null | head -1)
if [ -n "$pre" ] && [ "$undone" = "0" ] && [ "$left_ok" = "1" ] && [ "$(state_of stamps)" = "$STAMPS" ] \
   && [ "$(plan_num "$pre" stamp_stamp) $(plan_num "$pre" stamp_refuse)" = "0 0" ] \
   && { [ "$PASS" = "1" ] || [ "$(plan_num "$pre" repoint)" = "0" ]; }; then
  log "ALREADY DONE: $pre; $(state_of left) Liga MX side(s) on santos/guadalajara; stamps $(state_of stamps) — nothing to do (no backup)"
  exit 0
fi
log "not done yet (pre-plan '${pre:-unreadable}', left=$(state_of left), undone=${undone:-unreadable}) — continuing"

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

log "2/4 before-state + dry run (pins: $PINS)"
read_state
n_before=$(state_of soccer_before_t0); left0=$(state_of left)
if ! [[ "$n_before" =~ ^[0-9]+$ ]]; then log "ABORT at step 2: could not count soccer matches (got '$n_before') — nothing changed"; exit 1; fi
if [ "$left0" != "$LEFT_BEFORE" ]; then
  log "ABORT at step 2: $left0 Liga MX side(s) sit on santos/guadalajara, pass $PASS expects exactly $LEFT_BEFORE (pass 1: the map's $MAP_SIDES sides; pass 2: the 6 it refused) — a fixture the map does not name, or the passes ran out of order; nothing changed"; exit 1
fi
$DC exec -T api sh -c "$TOOL $PINS --out-csv /app/backups/repoint_team_matches_p${PASS}_dryrun_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/repoint_team_dry.log 2>&1; rc=\$?; grep -E 'refuse |PLAN |Pinned|ERROR|DRY-RUN|Done:|Traceback|Error' /tmp/repoint_team_dry.log | tail -20; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 2 exit $rc"
if [ "$rc" -ne 0 ]; then log "ABORT at step 2: the plan differs from the reviewed one or the tool failed (see above; full log /tmp/repoint_team_dry.log in the api container) — nothing changed"; exit 1; fi

log "3/4 apply (pins: $PINS)"
$DC exec -T api sh -c "$TOOL --apply $PINS --out-csv /app/backups/repoint_team_matches_p${PASS}_apply_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/repoint_team_apply.log 2>&1; rc=\$?; grep -E 'PLAN |Pinned|Moved|ERROR|Done:|Traceback|Error' /tmp/repoint_team_apply.log | tail -12; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 3 exit $rc"
if [ "$rc" -ne 0 ]; then log "ABORT at step 3 (see above; full log /tmp/repoint_team_apply.log in the api container): the transaction rolled back — nothing changed"; exit 1; fi

log "4/4 after-state"
read -r want_rp want_done want_ref want_nf <<< "$AFTER"
$DC exec -T api sh -c "$TOOL --expect-repoint $want_rp --expect-done $want_done --expect-refused $want_ref --expect-not-found $want_nf --out-csv /app/backups/repoint_team_matches_p${PASS}_after_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/repoint_team_after.log 2>&1; rc=\$?; grep -E 'PLAN |Pinned|ERROR|Traceback|Error' /tmp/repoint_team_after.log | tail -8; echo \"tool exit \$rc\"; exit \$rc"
replan=$?
read_state
echo "$STATE"
stamps=$(state_of stamps); left=$(state_of left); n_after=$(state_of soccer_before_t0)
log "NEXT (not run here; the runbook's step 10 recomputes every finished soccer match) — refresh the four teams' features:"
$DC exec -T api sh -c "grep -A1 'Refresh features' /tmp/repoint_team_apply.log | tail -1"
if [ "$replan" = "0" ] && [ "$stamps" = "$STAMPS" ] && [ "$left" = "$LEFT_AFTER" ] && [ "$n_after" = "$n_before" ]; then
  if [ "$PASS" = "2" ]; then
    log "DONE: 0 Liga MX rows left on santos/guadalajara. Separate approval: a promote_raw run re-writes the 4 match_stats rows the fd repair archived."
  else
    log "DONE: 6 sides left for --pass 2 (after merge_soccer_teams and repair_football_data_kickoffs)"
  fi
else
  log "FAILED: re-plan exit $replan (want 0: $want_rp repoint / $want_done done / $want_ref refused, $want_nf of them not found), stamps '$stamps', $left Liga MX side(s) left on santos/guadalajara (want $LEFT_AFTER), soccer rows created before $T0: $n_before -> $n_after — inspect"
  exit 1
fi
