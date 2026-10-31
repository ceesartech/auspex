#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Move the Liga MX rows ESPN put on the wrong team rows, with
# scripts/repoint_team_matches.py and the reviewed map
# scripts/data/repoint_team_matches.csv (read-only check 2026-10-31):
#   'Santos' (Santos FC, Brasileirao) holds 20 Liga MX rows of Santos Laguna
#     (ESPN 225, display name 'Santos') from 2026-07-19 -> 'Santos Laguna';
#   'Guadalajara' (CD Guadalajara, Segunda 2011-13) holds 19 Liga MX rows of
#     Chivas (ESPN 219) from 2026-07-19 -> 'Guadalajara Chivas'.
# and stamp the seeded ESPN ids (2674 Santos, 225 Santos Laguna, 219 Guadalajara
# Chivas). Predictions and recs keep their match_id. 6 sides stay refused: each
# has a football-data copy on the target pair 1 h later (a same-pair twin), to
# be merged by the duplicate repair before this can move them.
# ORDER: run right AFTER the team-identity code (fetch_upcoming seed) deploys:
# until this runs, every soccer sweep that meets one of these clubs' ESPN-
# identified fixtures exits 3 (ForkRefused, nothing written). Before the team
# merge tool (it then sees these rows on the right clubs).
#   1. verified backup (B2) — abort if it does not land;
#   2. dry run pinned to the reviewed plan (33 sides move, 6 refused; write-once
#      audit CSV) — abort if the plan differs;
#   3. apply (one transaction, rows locked, every UPDATE guarded, same pins);
#   4. after-state: the tool re-plans 33 done / 6 refused / 0 to move, the
#      stamps are on, the refused rows are all that is left on the old teams,
#      and the soccer match count is unchanged;
#   then it prints the compute_features command for the four teams (NOT run).
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status; a pipe through tail would report tail's 0.
# If prod moved since 2026-10-31 (e.g. the duplicate repair ran first), step 2
# aborts: re-review the dry-run CSV and update the pins below.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/repoint_team_matches.sh 2>&1 | tee -a ~/repoint_team_matches.log
set -uo pipefail
cd /opt/auspex
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_file() {  # psql_file <local.sql> [extra psql flags]
  local f="$1"; shift
  $DC cp "$f" postgres:/tmp/repoint_team_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c "psql -v ON_ERROR_STOP=1 -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" $* -f /tmp/repoint_team_q.sql"
}
TOOL="python /app/scripts/repoint_team_matches.py"
PINS="--expect-repoint 33 --expect-refused 6"
AFTER_PINS="--expect-repoint 0 --expect-done 33 --expect-refused 6"

cat > /tmp/repoint_team_after.sql <<'SQLEOF'
SELECT 'stamps=' || string_agg(name || ':' || coalesce(external_ids->>'espn', '-'), ',' ORDER BY name)
  FROM teams WHERE sport = 'soccer' AND normalized_name IN ('santos', 'santos laguna', 'guadalajara chivas');
SELECT 'left=' || count(*)
  FROM matches m JOIN leagues l ON l.id = m.league_id JOIN teams t ON t.id IN (m.home_team_id, m.away_team_id)
 WHERE l.name = 'Liga MX' AND l.sport = 'soccer' AND t.sport = 'soccer' AND t.normalized_name IN ('santos', 'guadalajara');
SELECT 'soccer_matches=' || count(*) FROM matches m JOIN leagues l ON l.id = m.league_id WHERE l.sport = 'soccer';
SQLEOF

log "1/4 backup"
before=$(date -u +%s)
n_before=$(psql_file /tmp/repoint_team_after.sql -At | sed -n 's/^soccer_matches=//p')
if ! [[ "$n_before" =~ ^[0-9]+$ ]]; then log "ABORT: could not count soccer matches (got '$n_before') — nothing changed"; exit 1; fi
$DC exec -T airflow-scheduler airflow dags trigger db_backup_daily >/dev/null 2>&1
landed=""
for _ in $(seq 1 40); do
  sleep 30
  l=$(latest_dump); ts=${l%% *}
  if [ -n "$ts" ] && [ "$ts" -gt "$before" ] 2>/dev/null; then landed="$l"; break; fi
done
if [ -z "$landed" ]; then log "ABORT: backup did not land in B2 within 20 min — nothing changed"; exit 1; fi
log "backup landed: $landed (soccer matches before: $n_before)"

log "2/4 dry run (pins: $PINS)"
$DC exec -T api sh -c "$TOOL $PINS --out-csv /app/backups/repoint_team_matches_dryrun_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/repoint_team_dry.log 2>&1; rc=\$?; grep -E 'refuse |Planned|Pinned|ERROR|DRY-RUN|Done:|Traceback|Error' /tmp/repoint_team_dry.log | tail -20; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 2 exit $rc"
if [ "$rc" -ne 0 ]; then log "ABORT at step 2: the plan differs from the reviewed one or the tool failed (see above; full log /tmp/repoint_team_dry.log in the api container) — nothing changed"; exit 1; fi

log "3/4 apply (pins: $PINS)"
$DC exec -T api sh -c "$TOOL --apply $PINS --out-csv /app/backups/repoint_team_matches_apply_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/repoint_team_apply.log 2>&1; rc=\$?; grep -E 'Planned|Pinned|Moved|ERROR|Done:|Traceback|Error' /tmp/repoint_team_apply.log | tail -12; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 3 exit $rc"
if [ "$rc" -ne 0 ]; then log "ABORT at step 3 (see above; full log /tmp/repoint_team_apply.log in the api container): the transaction rolled back — nothing changed"; exit 1; fi

log "4/4 after-state"
$DC exec -T api sh -c "$TOOL $AFTER_PINS --out-csv /app/backups/repoint_team_matches_after_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/repoint_team_after.log 2>&1; rc=\$?; grep -E 'Planned|Pinned|ERROR|Traceback|Error' /tmp/repoint_team_after.log | tail -8; echo \"tool exit \$rc\"; exit \$rc"
replan=$?
out=$(psql_file /tmp/repoint_team_after.sql -At)
echo "$out"
stamps=$(echo "$out" | sed -n 's/^stamps=//p')
left=$(echo "$out" | sed -n 's/^left=//p')
n_after=$(echo "$out" | sed -n 's/^soccer_matches=//p')
rm -f /tmp/repoint_team_after.sql
$DC exec -T postgres rm -f /tmp/repoint_team_q.sql >/dev/null 2>&1
log "NEXT (not run here) — refresh the four teams' features:"
$DC exec -T api sh -c "grep -A1 'Refresh features' /tmp/repoint_team_apply.log | tail -1"
if [ "$replan" = "0" ] && [ "$stamps" = "Guadalajara Chivas:219,Santos:2674,Santos Laguna:225" ] && [ "$left" = "6" ] && [ "$n_after" = "$n_before" ]; then
  log "DONE"
else
  log "FAILED: re-plan exit $replan (want 0), stamps '$stamps', $left Liga MX side(s) left on Santos/Guadalajara (want 6: the refused twins), soccer matches $n_before -> $n_after — inspect"
  exit 1
fi
