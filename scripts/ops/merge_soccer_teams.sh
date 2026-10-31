#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Merge the 171 soccer clubs ESPN minted under a second spelling into their
# established rows (scripts/data/soccer_team_merge_map.csv, reviewed
# 2026-10-31): re-point their ~1,950 matches, fold the ~290 twin fixtures the
# re-point would create or that already sat on the twin teams (keep the rec
# holder, else the finished row), record each merged spelling as an alias +
# ESPN team id on the kept team, delete the 171 history-less team rows.
# PRECONDITION: the alias resolver must be deployed FIRST (team_identity +
# fetch_upcoming + fetch_live_odds). Without it the next fetch re-inserts
# every deleted team; the tool refuses --apply (and fails the dry run) until
# the marker is there.
#   1. verified backup (B2) — abort if it does not land;
#   2. dry run (plan + write-once CSV) — abort unless all EXPECTED pairs plan
#      as merge/done with none refused;
#   3. apply (one transaction per pair; gzip JSONL archive of every row it
#      deletes or changes lands first), then grade the kept rows that adopted
#      a twin's result with open recs (2 on 2026-10-31: Olympiakos v OFI
#      Crete, Lecce v Roma);
#   4. after-state + verify: 0 matches and 0 team rows on the merged ids,
#      EXPECTED pairs recorded on their kept teams, 0 finished same-pair twins
#      within 36h on the kept teams, 0 unsettled recs on the graded rows.
# Re-runnable: a merged pair plans as 'done' and is not written again. A pair
# that rolled back (its team gained a fixture between plan and write, say) is
# merged by simply running this script again.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/merge_soccer_teams.sh 2>&1 | tee -a ~/merge_soccer_teams.log
set -uo pipefail
cd /opt/auspex
DC="docker compose"
MAP=scripts/data/soccer_team_merge_map.csv
EXPECTED=171
log() { echo "[$(date -u +%FT%TZ)] $*"; }
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
TOOL="python /app/scripts/merge_soccer_teams.py"

# The map's data rows: '#' lines are comments, the first other line the header.
rows=$(grep -v '^#' "$MAP" | tail -n +2 | grep -c .)
if [ "$rows" != "$EXPECTED" ]; then log "ABORT: $MAP has $rows pairs, this script expects $EXPECTED — nothing changed"; exit 1; fi

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
# Exit status through a file, not a pipe: `| tail` would report tail's 0.
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/merge_soccer_teams_dryrun_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/merge_teams_dry.log 2>&1; rc=\$?; grep -E 'REFUSED|ERROR|PLAN |Planned:|Audit CSV' /tmp/merge_teams_dry.log | tail -40; exit \$rc" \
  || { log "ABORT at step 2: a pair was refused, the alias resolver is missing, or the tool failed (see above) — nothing changed"; exit 1; }
plan=$($DC exec -T api sh -c "grep -o 'PLAN pairs_merge=[0-9]* pairs_done=[0-9]* pairs_refuse=[0-9]*' /tmp/merge_teams_dry.log | tail -1")
merge=$(echo "$plan" | sed -n 's/.*pairs_merge=\([0-9]*\).*/\1/p')
done_=$(echo "$plan" | sed -n 's/.*pairs_done=\([0-9]*\).*/\1/p')
refuse=$(echo "$plan" | sed -n 's/.*pairs_refuse=\([0-9]*\).*/\1/p')
if [ -z "$merge" ] || [ "$refuse" != "0" ] || [ $((merge + done_)) -ne "$EXPECTED" ]; then
  log "ABORT at step 2: plan '$plan' is not $EXPECTED pairs merge/done with 0 refused — nothing changed"; exit 1
fi
if [ "$merge" = "0" ]; then log "nothing to merge: all $EXPECTED pairs already done"; fi

log "3/4 apply ($merge pair(s))"
$DC exec -T api sh -c "$TOOL --apply > /tmp/merge_teams.log 2>&1; rc=\$?; grep -E 'REFUSED|ERROR|ROLLED BACK|Archived|Written:|WRITTEN|grade_completed_matches' /tmp/merge_teams.log | tail -60; exit \$rc" \
  || log "step 3 reported a refused or rolled-back pair (see above): each pair is its own transaction, the others are written — checking the after-state"
# A kept row that adopted its twin's result went 'finished' through the
# match-outcome trigger, which settles open recs from the PREDICTION (status
# only, no P&L; a rec on the other selection can read 'won'). The grader
# re-settles them by selection, voids post-kickoff ones and fills P&L. These
# rows are weeks old, outside its --days window, so name them.
adopted=$($DC exec -T api sh -c "grep -o 'grade_completed_matches.py --match-id [0-9a-f-]*' /tmp/merge_teams.log | awk '{print \$3}' | sort -u")
graded_ok=1
for mid in $adopted; do
  $DC exec -T api sh -c "python /app/scripts/grade_completed_matches.py --match-id $mid > /tmp/merge_teams_grade.log 2>&1; rc=\$?; tail -2 /tmp/merge_teams_grade.log; exit \$rc" \
    || { graded_ok=0; log "grading $mid failed (see above)"; }
done

log "4/4 after-state"
# new_team_id and old_team_id are the first two columns, never quoted.
values=$(grep -v '^#' "$MAP" | tail -n +2 | awk -F, -v q="'" 'NF > 1 { printf "%s(%s::uuid, %s::uuid)", (n++ ? ", " : ""), q $1 q, q $2 q }')
graded=$(for mid in $adopted; do printf "'%s'::uuid," "$mid"; done | sed 's/,$//')
[ -n "$graded" ] || graded="NULL::uuid"
cat > /tmp/merge_teams_after.sql <<SQLEOF
BEGIN READ ONLY;
WITH map(new_id, old_id) AS (VALUES $values)
SELECT 'AFTER',
       (SELECT count(DISTINCT m.id) FROM matches m JOIN map ON map.new_id IN (m.home_team_id, m.away_team_id)) AS matches_on_merged_ids,
       (SELECT count(*) FROM teams t JOIN map ON t.id = map.new_id) AS merged_team_rows_left,
       (SELECT count(*) FROM map JOIN teams o ON o.id = map.old_id
         WHERE o.metadata->'merged_from' @> jsonb_build_array(jsonb_build_object('team_id', map.new_id::text))) AS pairs_recorded,
       (SELECT count(*) FROM matches a
          JOIN matches b ON b.home_team_id = a.home_team_id AND b.away_team_id = a.away_team_id AND b.id > a.id
                        AND b.match_date BETWEEN a.match_date - interval '36 hours' AND a.match_date + interval '36 hours'
         WHERE a.status = 'finished' AND b.status = 'finished'
           AND (a.home_team_id IN (SELECT old_id FROM map) OR a.away_team_id IN (SELECT old_id FROM map))) AS finished_twins,
       (SELECT count(*) FROM betting_recommendations r
         WHERE r.match_id IN ($graded)
           AND (r.status IN ('pending', 'placed') OR r.profit_loss IS NULL)) AS unsettled_recs_on_graded;
ROLLBACK;
SQLEOF
$DC cp /tmp/merge_teams_after.sql postgres:/tmp/merge_teams_after.sql >/dev/null 2>&1
out=$($DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -q -At -F " " -v ON_ERROR_STOP=1 -f /tmp/merge_teams_after.sql')
rc=$?
$DC exec -T postgres rm -f /tmp/merge_teams_after.sql; rm -f /tmp/merge_teams_after.sql
echo "$out"
line=$(echo "$out" | grep '^AFTER ')
read -r _ on_ids teams_left recorded twins unsettled <<< "$line"
if [ "$rc" = "0" ] && [ "$on_ids" = "0" ] && [ "$teams_left" = "0" ] && [ "$recorded" = "$EXPECTED" ] \
   && [ "$twins" = "0" ] && [ "$unsettled" = "0" ] && [ "$graded_ok" = "1" ]; then
  log "DONE: $EXPECTED/$EXPECTED pairs merged, 0 matches on merged ids, 0 finished twins, adopted rows graded"
  log "NEXT (operator, not run here): recompute soccer features for the re-pointed history before the next retrain:"
  log "  docker compose exec -T api python /app/scripts/compute_features.py --backfill-finished soccer"
else
  log "FAILED: after-state rc=$rc matches_on_merged_ids=$on_ids merged_team_rows_left=$teams_left pairs_recorded=$recorded/$EXPECTED finished_twins=$twins unsettled_recs_on_graded=$unsettled graded_ok=$graded_ok — inspect /tmp/merge_teams.log in the api container; re-running this script merges the rest"
  exit 1
fi
