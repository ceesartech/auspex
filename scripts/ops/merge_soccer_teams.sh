#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Merge the 171 soccer clubs ESPN minted under a second spelling into their
# established rows (scripts/data/soccer_team_merge_map.csv, reviewed
# 2026-10-31): re-point their ~1,950 matches, fold the ~300 twin fixtures the
# re-point would create or that already sat on the twin teams (keep the rec
# holder, else the finished ESPN row), move each twin's ESPN team id onto the
# kept team, record each merged spelling as an alias, delete the 171
# history-less team rows. Each pair commits only after the deployed resolvers
# (fetch_upcoming.find_team, fetch_live_odds.find_match_id) resolve the merged
# spelling to the kept team inside its transaction.
#
# RUN ORDER (2026-10-31):
#   1. deploy the team-identity code (ESPN team ids, aliases, seed; push to main);
#   2. bash /opt/auspex/scripts/ops/repoint_team_matches.sh (identity branch);
#   3. THIS script;
#   4. bash /opt/auspex/scripts/ops/repair_football_data_kickoffs.sh —
#      promote_raw stays blocked until it has run;
#   5. bash /opt/auspex/scripts/ops/cleanup_soccer_orphans.sh.
#
#   1/5 verified backup (B2) — abort if it does not land;
#   2/5 dry run (write-once CSV + plan file) — abort unless it exits 0 with all
#       EXPECTED pairs planned as merge or done and none refused;
#   3/5 apply --expect-plan <the dry run's plan file>: refuses (nothing
#       written) unless it plans exactly what the dry run showed; one
#       transaction per pair, the gzip JSONL archive of every row it deletes
#       or changes lands first;
#   4/5 grade the kept rows the DRY RUN listed (GRADE_AFTER: predictions moved
#       onto finished rows, results adopted with open recs);
#   5/5 after-state from the dry run's plan file (--check-after): no team row or
#       match left on a merged id, every pair recorded, no dropped twin left,
#       every re-point landed, 0 finished same-pair twins on the kept teams,
#       every merged spelling resolving to its kept team, each ESPN id on its
#       kept team only, 0 teams re-minted under a merged spelling, 0 unsettled
#       recs on rows that adopted a result with open recs.
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
TS=$(date -u +%Y%m%dT%H%M%SZ)
PLAN=/app/backups/merge_soccer_teams_dryrun_$TS.plan.json

# The map's data rows: '#' lines are comments, the first other line the header.
rows=$(grep -v '^#' "$MAP" | tail -n +2 | grep -c .)
if [ "$rows" != "$EXPECTED" ]; then log "ABORT: $MAP has $rows pairs, this script expects $EXPECTED — nothing changed"; exit 1; fi

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

log "2/5 dry run"
# Exit status through a file, not a pipe: `| tail` would report tail's 0.
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/merge_soccer_teams_dryrun_$TS.csv --plan-out $PLAN > /tmp/merge_teams_dry.log 2>&1; rc=\$?; grep -E 'REFUSED|ERROR|PLAN |PLAN_FILE|Planned:|Audit CSV' /tmp/merge_teams_dry.log | tail -40; exit \$rc" \
  || { log "ABORT at step 2: a pair was refused, the resolvers are not deployed, or the tool failed (see above) — nothing changed"; exit 1; }
plan=$($DC exec -T api sh -c "grep -o 'PLAN pairs_merge=[0-9]* pairs_done=[0-9]* pairs_refuse=[0-9]*' /tmp/merge_teams_dry.log | tail -1")
merge=$(echo "$plan" | sed -n 's/.*pairs_merge=\([0-9]*\).*/\1/p')
done_=$(echo "$plan" | sed -n 's/.*pairs_done=\([0-9]*\).*/\1/p')
refuse=$(echo "$plan" | sed -n 's/.*pairs_refuse=\([0-9]*\).*/\1/p')
if [ -z "$merge" ] || [ "$refuse" != "0" ] || [ $((merge + done_)) -ne "$EXPECTED" ]; then
  log "ABORT at step 2: plan '$plan' is not $EXPECTED pairs merge/done with 0 refused — nothing changed"; exit 1
fi
# The rows to grade come from the dry run, like the after-state.
grade_ids=$($DC exec -T api sh -c "grep -o 'GRADE_AFTER [0-9a-f-]*' /tmp/merge_teams_dry.log | awk '{print \$2}' | sort -u" | tr '\n' ' ')
log "plan: $merge to merge, $done_ already done; $(echo $grade_ids | wc -w) kept row(s) to grade after; plan file $PLAN"

apply_rc=0
if [ "$merge" = "0" ]; then
  log "3/5 apply: nothing to merge, all $EXPECTED pairs already done"
else
  log "3/5 apply ($merge pair(s), --expect-plan $PLAN)"
  $DC exec -T api sh -c "$TOOL --apply --expect-plan $PLAN --out-csv /app/backups/merge_soccer_teams_apply_$TS.csv > /tmp/merge_teams.log 2>&1; rc=\$?; grep -E 'REFUSED|ERROR|ROLLED BACK|Archived|Written:|WRITTEN|matches the expected|Traceback|Error:' /tmp/merge_teams.log | tail -60; exit \$rc"
  apply_rc=$?
  if [ "$apply_rc" = "2" ]; then
    log "ABORT at step 3: the apply refused before writing (its plan differs from the dry run's, or a refusal above) — nothing changed; re-run this script"; exit 1
  fi
  [ "$apply_rc" = "0" ] || log "step 3 exited $apply_rc: a pair rolled back or the tool failed (see above); each pair is its own transaction, so whatever committed stays — grading and checking the after-state"
fi

log "4/5 grade"
# A kept row that adopted its twin's result went 'finished' through the
# match-outcome trigger, which settles open recs from the PREDICTION (status
# only, no P&L; a rec on the other selection can read 'won'); predictions
# moved onto an already finished row arrive ungraded. The grader settles both
# by selection, voids post-kickoff rows and fills P&L. These rows are weeks
# old, outside its --days window, so name them.
graded_ok=1
if [ -n "${grade_ids// /}" ]; then
  $DC exec -T api sh -c "rc=0; : > /tmp/merge_teams_grade.log; for mid in $grade_ids; do python /app/scripts/grade_completed_matches.py --match-id \$mid >> /tmp/merge_teams_grade.log 2>&1 || { rc=1; echo \"grading \$mid failed\"; }; done; grep -c 'Done\.' /tmp/merge_teams_grade.log; exit \$rc" \
    || { graded_ok=0; log "grading failed for a row (see above; /tmp/merge_teams_grade.log in the api container)"; }
fi

log "5/5 after-state (from the dry run's plan file)"
$DC exec -T api sh -c "$TOOL --check-after $PLAN > /tmp/merge_teams_after.log 2>&1; rc=\$?; grep -E 'AFTER|REFUSED|ERROR' /tmp/merge_teams_after.log | tail -10; exit \$rc"
after_rc=$?
if [ "$after_rc" = "0" ] && [ "$graded_ok" = "1" ] && [ "$apply_rc" = "0" ]; then
  log "DONE: $EXPECTED/$EXPECTED pairs merged and resolving, 0 finished twins, listed rows graded"
  log "NEXT (operator, not run here):"
  log "  1. after the next pipeline tick (15 min), re-check that no sweep re-minted a merged team:"
  log "       docker compose exec -T api python /app/scripts/merge_soccer_teams.py --check-after $PLAN"
  log "     teams_reminted and resolver_misses must stay 0;"
  log "  2. recompute soccer features for the re-pointed history before the next retrain:"
  log "       docker compose exec -T api python /app/scripts/compute_features.py --backfill-finished soccer"
  log "  3. promote_raw stays blocked until bash /opt/auspex/scripts/ops/repair_football_data_kickoffs.sh has run"
  log "     (next in the run order), then bash /opt/auspex/scripts/ops/cleanup_soccer_orphans.sh."
else
  log "FAILED: apply rc=$apply_rc graded_ok=$graded_ok after-state rc=$after_rc — inspect /tmp/merge_teams.log and /tmp/merge_teams_after.log in the api container; re-running this script merges the rest"
  exit 1
fi
