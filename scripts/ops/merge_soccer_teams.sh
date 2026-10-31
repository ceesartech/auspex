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
# spelling to the kept team inside its transaction, and every odds-feed
# spelling of the club (scripts/data/soccer_team_odds_spellings.csv) that found
# its fixture before the pair's first write still finds it.
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
#       then pause auspex_pipeline and fetch_live_odds (only those not already
#       paused; a trap unpauses them on every exit) and wait for their runs in
#       flight to finish, so no fixture, rec or price lands on a twin between the
#       dry run and the apply — abort if they do not finish within 20 min;
#   2/5 dry run (write-once CSV + plan file) — abort unless it exits 0 with all
#       EXPECTED pairs planned as merge or done, none refused, and exactly
#       EXPECTED_KICKOFFS_ADOPTED kept rows moving their kickoff (0 when this
#       was reviewed: a moved kickoff needs a human look at the CSV first);
#   3/5 apply --expect-plan <the dry run's plan file>: refuses (nothing
#       written) unless it plans exactly what the dry run showed; one
#       transaction per pair, the gzip JSONL archive of every row it deletes
#       or changes lands first (a twin holding a row written after the archive
#       rolls its pair back); then the two DAGs are unpaused;
#   4/5 grade the kept rows the DRY RUN listed (GRADE_AFTER: predictions moved
#       onto finished rows, results adopted with open recs);
#   5/5 after-state from the dry run's plan file (--check-after): no team row or
#       match left on a merged id, every pair recorded, no dropped twin left,
#       every re-point landed, 0 finished same-pair twins on the kept teams,
#       every merged spelling resolving to its kept team, each ESPN id on its
#       kept team only, 0 teams re-minted under a merged spelling, 0 unsettled
#       recs on rows that adopted a result with open recs.
# NEXT (printed at the end): after the next fetch_live_odds run (every 90 min),
# --check-after with --odds-since <when the apply finished>: every merged
# club's fixture the feed was quoting before the merge must still be quoted.
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
EXPECTED_KICKOFFS_ADOPTED=0
DAGS="auspex_pipeline fetch_live_odds"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
# Airflow 3.3 CLI (dag_id positional; -o json, structlog lines filtered out by the grep).
dag_paused() {  # True / False, or nothing when unreadable
  $DC exec -T airflow-scheduler sh -c "airflow dags details $1 -o json 2>/dev/null | grep '^\\[' | tail -1" \
    | grep -o '"is_paused": "[A-Za-z]*"' | head -1 | sed 's/.*: "\(.*\)"/\1/'
}
runs_in_flight() {  # queued + running runs of every DAG in $DAGS, or nothing when unreadable
  local n=0 c d s
  for d in $DAGS; do
    for s in queued running; do
      c=$($DC exec -T airflow-scheduler sh -c "airflow dags list-runs $d --state $s -o json 2>/dev/null | grep '^\\[' | tail -1")
      [ -n "$c" ] || return 1
      n=$((n + $(echo "$c" | grep -o '"run_id"' | wc -l)))
    done
  done
  echo "$n"
}
PAUSED_BY_US=""
resume_dags() {
  local d
  for d in $PAUSED_BY_US; do
    $DC exec -T airflow-scheduler airflow dags unpause -y "$d" >/dev/null 2>&1
    if [ "$(dag_paused "$d")" = "False" ]; then
      log "unpaused $d"
    else
      log "ALERT: $d is STILL PAUSED — unpause it now: docker compose exec -T airflow-scheduler airflow dags unpause -y $d"
    fi
  done
  PAUSED_BY_US=""
}
trap resume_dags EXIT
trap 'exit 130' INT TERM HUP
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

log "pausing $DAGS for the dry run and the apply"
for d in $DAGS; do
  st=$(dag_paused "$d")
  if [ "$st" = "True" ]; then
    log "$d is already paused — left as it is"
  elif [ "$st" = "False" ]; then
    $DC exec -T airflow-scheduler airflow dags pause -y "$d" >/dev/null 2>&1
    [ "$(dag_paused "$d")" = "True" ] || { log "ABORT: could not pause $d — nothing changed"; exit 1; }
    PAUSED_BY_US="$PAUSED_BY_US $d"
    log "paused $d"
  else
    log "ABORT: cannot read whether $d is paused — nothing changed"; exit 1
  fi
done
inflight=""
for _ in $(seq 1 40); do
  inflight=$(runs_in_flight) || inflight=""
  [ "$inflight" = "0" ] && break
  sleep 30
done
if [ "$inflight" != "0" ]; then log "ABORT: runs of $DAGS still in flight after 20 min (${inflight:-unreadable}) — nothing changed"; exit 1; fi

log "2/5 dry run"
# Exit status through a file, not a pipe: `| tail` would report tail's 0.
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/merge_soccer_teams_dryrun_$TS.csv --plan-out $PLAN > /tmp/merge_teams_dry.log 2>&1; rc=\$?; grep -E 'REFUSED|ERROR|PLAN |PLAN_FILE|Planned:|Audit CSV' /tmp/merge_teams_dry.log | tail -40; exit \$rc" \
  || { log "ABORT at step 2: a pair was refused, the resolvers are not deployed, or the tool failed (see above) — nothing changed"; exit 1; }
plan=$($DC exec -T api sh -c "grep -o 'PLAN pairs_merge=[0-9]* pairs_done=[0-9]* pairs_refuse=[0-9]* kickoffs_adopted=[0-9]*' /tmp/merge_teams_dry.log | tail -1")
merge=$(echo "$plan" | sed -n 's/.*pairs_merge=\([0-9]*\).*/\1/p')
done_=$(echo "$plan" | sed -n 's/.*pairs_done=\([0-9]*\).*/\1/p')
refuse=$(echo "$plan" | sed -n 's/.*pairs_refuse=\([0-9]*\).*/\1/p')
kickoffs=$(echo "$plan" | sed -n 's/.*kickoffs_adopted=\([0-9]*\).*/\1/p')
if [ -z "$merge" ] || [ "$refuse" != "0" ] || [ $((merge + done_)) -ne "$EXPECTED" ]; then
  log "ABORT at step 2: plan '$plan' is not $EXPECTED pairs merge/done with 0 refused — nothing changed"; exit 1
fi
if [ "$kickoffs" != "$EXPECTED_KICKOFFS_ADOPTED" ]; then
  log "ABORT at step 2: $kickoffs kept row(s) would move their kickoff, $EXPECTED_KICKOFFS_ADOPTED reviewed — read the enrich_keep rows of /app/backups/merge_soccer_teams_dryrun_$TS.csv first — nothing changed"; exit 1
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
MERGED_AT=$(date -u +%FT%TZ)
resume_dags

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
  log "  1. after the next fetch_live_odds run (every 90 min; the pipeline ticks every 15), re-check that no sweep"
  log "     re-minted a merged team and that the odds feed still reaches every merged club's fixtures:"
  log "       docker compose exec -T api python /app/scripts/merge_soccer_teams.py --check-after $PLAN --odds-since $MERGED_AT"
  log "     teams_reminted, resolver_misses and odds_fixtures_stopped must be 0 (odds_not_yet_observed=1: no odds run yet, re-run later);"
  log "  2. recompute soccer features for the re-pointed history before the next retrain:"
  log "       docker compose exec -T api python /app/scripts/compute_features.py --backfill-finished soccer"
  log "  3. promote_raw stays blocked until bash /opt/auspex/scripts/ops/repair_football_data_kickoffs.sh has run"
  log "     (next in the run order), then bash /opt/auspex/scripts/ops/cleanup_soccer_orphans.sh."
else
  log "FAILED: apply rc=$apply_rc graded_ok=$graded_ok after-state rc=$after_rc — inspect /tmp/merge_teams.log and /tmp/merge_teams_after.log in the api container; re-running this script merges the rest"
  exit 1
fi
