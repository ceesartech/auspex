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
# RUN ORDER (2026-10-31; scripts/ops/soccer_repair_runbook.sh runs it, each
# step through its own wrapper — deploy the integrated branch first):
#   1. scripts/ops/repoint_team_matches.sh --pass 1
#   2. scripts/ops/migrate_027_outcome_trigger.sh
#   3. THIS script
#   4. scripts/ops/repair_football_data_kickoffs.sh (promote_raw stays blocked until it has run)
#   5. scripts/ops/repoint_team_matches.sh --pass 2
#   6. scripts/ops/repair_stranded_recs_soccer.sh
#   7. scripts/ops/cleanup_soccer_orphans.sh
#   8. scripts/ops/backfill_soccer_regulation_scores.sh (+ the re-grade it prints)
#   9. scripts/ops/backfill_season_labels.sh
#  10. compute_features.py --backfill-finished soccer
#  11. this tool's --check-after <plan> --odds-since <MERGED_AT>, once a
#      fetch_live_odds run has FINISHED after the merge (the runbook waits for
#      one; the plan and MERGED_AT are in /app/backups/merge_soccer_teams_last_apply.env)
#
#   0/5 already done? A read-only dry run first: when it plans all EXPECTED
#       pairs as 'done' and --check-after on that plan passes, this exits 0
#       without a backup, a pause or a write (a re-run after success);
#   1/5 verified backup (B2) — abort if it does not land;
#       then DRAIN, then pause auspex_pipeline and fetch_live_odds: wait until
#       neither has a run queued or running, pause the ones not already paused
#       (a trap unpauses them on every exit), and re-check 0 runs in flight; a
#       run that slipped in between makes it unpause and try again (5 attempts,
#       each waiting up to 20 min). Airflow 3.3 never advances a paused DAG's
#       runs, so pausing first could leave a run 'running' for good. No
#       fixture, rec or price then lands on a twin between the dry run and the
#       apply;
#   2/5 dry run (write-once CSV + plan file) — abort unless it exits 0 with all
#       EXPECTED pairs planned as merge or done, none refused, and exactly
#       EXPECTED_KICKOFFS_ADOPTED kept rows moving their kickoff (0 when this
#       was reviewed: a moved kickoff needs a human look at the CSV first);
#   3/5 apply --expect-plan <the dry run's plan file>: refuses (nothing
#       written) unless it plans exactly what the dry run showed; one
#       transaction per pair, the gzip JSONL archive of every row it deletes
#       or changes lands first (a twin holding a row written after the archive
#       rolls its pair back); then the two DAGs are unpaused, and the plan file
#       and MERGED_AT go to /app/backups/merge_soccer_teams_last_apply.env.
#       The run FAILS (after steps 4-5) when the apply's WRITTEN line reports
#       odds_spellings_unmatched_before > 0: an odds-feed spelling that did not
#       find its club's fixture before the merge was not held to anything
#       (each one is logged at WARNING) — inspect before calling it done;
#   4/5 grade the kept rows the DRY RUN listed (GRADE_AFTER: predictions moved
#       onto finished rows, results adopted with open recs);
#   5/5 after-state from the dry run's plan file (--check-after): no team row or
#       match left on a merged id, every pair recorded, no dropped twin left,
#       every re-point landed, 0 finished same-pair twins on the kept teams,
#       every merged spelling resolving to its kept team, each ESPN id on its
#       kept team only, 0 teams re-minted under a merged spelling, 0 unsettled
#       recs on rows that adopted a result with open recs.
# NEXT (printed at the end): after the next fetch_live_odds run has FINISHED
# (every 90 min; confirm with `airflow dags list-runs fetch_live_odds --state
# success -o json`: a run whose start_date is after MERGED_AT), --check-after
# with --odds-since MERGED_AT: every merged club's fixture the feed was
# quoting before the merge must still be quoted. The runbook's step 11 does
# exactly that.
# Re-runnable: a merged pair plans as 'done' and is not written again; all
# 171 done exits 0 at step 0. A pair that rolled back (its team gained a
# fixture between plan and write, say) is merged by running this again.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/merge_soccer_teams.sh 2>&1 | tee -a ~/merge_soccer_teams.log
# (AUSPEX_DIR and AF_POLL_SECONDS exist for tests and rehearsals.)
set -uo pipefail
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
MAP=scripts/data/soccer_team_merge_map.csv
EXPECTED=171
EXPECTED_KICKOFFS_ADOPTED=0
DAGS="auspex_pipeline fetch_live_odds"
STATE_FILE=/app/backups/merge_soccer_teams_last_apply.env
log() { echo "[$(date -u +%FT%TZ)] $*"; }
# shellcheck source=scripts/ops/_airflow_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_airflow_lib.sh" || { log "ABORT: cannot load _airflow_lib.sh — nothing changed"; exit 1; }
trap af_resume EXIT
trap 'exit 130' INT TERM HUP
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
TOOL="python /app/scripts/merge_soccer_teams.py"
TS=$(date -u +%Y%m%dT%H%M%SZ)
PLAN=/app/backups/merge_soccer_teams_dryrun_$TS.plan.json
plan_field() {  # plan_field <PLAN line> <key>: the count, or nothing
  echo "$1" | sed -n "s/.*$2=\([0-9]*\).*/\1/p"
}

# The map's data rows: '#' lines are comments, the first other line the header.
rows=$(grep -v '^#' "$MAP" | tail -n +2 | grep -c .)
if [ "$rows" != "$EXPECTED" ]; then log "ABORT: $MAP has $rows pairs, this script expects $EXPECTED — nothing changed"; exit 1; fi

log "0/5 already done? (read-only dry run)"
PRE_PLAN=/app/backups/merge_soccer_teams_precheck_$TS.plan.json
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/merge_soccer_teams_precheck_$TS.csv --plan-out $PRE_PLAN > /tmp/merge_teams_pre.log 2>&1; echo \"tool exit \$?\""
pre=$($DC exec -T api sh -c "grep -o 'PLAN pairs_merge=[0-9]* pairs_done=[0-9]* pairs_refuse=[0-9]* kickoffs_adopted=[0-9]*' /tmp/merge_teams_pre.log | tail -1")
if [ "$(plan_field "$pre" pairs_merge)" = "0" ] && [ "$(plan_field "$pre" pairs_done)" = "$EXPECTED" ] && [ "$(plan_field "$pre" pairs_refuse)" = "0" ]; then
  $DC exec -T api sh -c "$TOOL --check-after $PRE_PLAN > /tmp/merge_teams_pre_after.log 2>&1; rc=\$?; grep -E 'AFTER|REFUSED|ERROR' /tmp/merge_teams_pre_after.log | tail -10; exit \$rc"
  if [ $? -eq 0 ]; then
    log "ALREADY DONE: all $EXPECTED pairs merged and the after-state holds — nothing to do (no backup, no pause)"
    $DC exec -T api sh -c "cat $STATE_FILE 2>/dev/null" | sed 's/^/  last apply: /'
    exit 0
  fi
  log "all $EXPECTED pairs read 'done' but the after-state check failed (above) — FAILED: inspect; nothing changed"
  exit 1
fi
log "not done yet (pre-plan '${pre:-unreadable}') — continuing"

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

log "draining, then pausing $DAGS for the dry run and the apply"
# shellcheck disable=SC2086  # $DAGS is a word list
if ! af_drain_and_pause 5 40 $DAGS; then
  log "ABORT: could not drain and pause $DAGS (see above) — nothing changed"; exit 1
fi

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
af_resume
unmatched=""
if [ "$merge" != "0" ]; then
  written=$($DC exec -T api sh -c "grep -o 'WRITTEN .*' /tmp/merge_teams.log | tail -1")
  unmatched=$(plan_field "$written" odds_spellings_unmatched_before)
  log "apply: ${written:-no WRITTEN line}"
  # The pointer the runbook's odds check (step 11) reads; rewritten only by a run that applied.
  $DC exec -T api sh -c "printf 'PLAN=%s\nMERGED_AT=%s\n' $PLAN $MERGED_AT > $STATE_FILE" \
    || log "WARNING: could not write $STATE_FILE — step 11 needs PLAN=$PLAN MERGED_AT=$MERGED_AT"
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
if [ -n "$unmatched" ] && [ "$unmatched" != "0" ]; then
  log "FAILED: the apply reports odds_spellings_unmatched_before=$unmatched — that many odds-feed spellings did not find their"
  log "  club's fixture before the merge, so the in-transaction odds check skipped them (the WARNING lines name each club"
  log "  and spelling: grep 'did not find the probe fixture' /tmp/merge_teams.log in the api container). The merge is"
  log "  committed (after-state rc=$after_rc); inspect those spellings, then re-run this script (it exits 0 once all pairs are done)."
  exit 1
fi
if [ "$after_rc" = "0" ] && [ "$graded_ok" = "1" ] && [ "$apply_rc" = "0" ]; then
  log "DONE: $EXPECTED/$EXPECTED pairs merged and resolving, 0 finished twins, listed rows graded"
  log "NEXT (operator, not run here; the runbook runs both):"
  log "  1. after the next fetch_live_odds run has FINISHED (every 90 min) — confirm with"
  log "       docker compose exec -T airflow-scheduler airflow dags list-runs fetch_live_odds --state success -o json"
  log "     (a run whose start_date is after $MERGED_AT) — re-check that no sweep re-minted a merged team and that the odds"
  log "     feed still reaches every merged club's fixtures:"
  log "       docker compose exec -T api python /app/scripts/merge_soccer_teams.py --check-after $PLAN --odds-since $MERGED_AT"
  log "     teams_reminted, resolver_misses and odds_fixtures_stopped must be 0 (odds_not_yet_observed=1: no odds run yet, re-run later);"
  log "  2. recompute soccer features for the re-pointed history before the next retrain:"
  log "       docker compose exec -T api python /app/scripts/compute_features.py --backfill-finished soccer"
  log "  promote_raw stays blocked until scripts/ops/repair_football_data_kickoffs.sh (runbook step 4) has run."
else
  log "FAILED: apply rc=$apply_rc graded_ok=$graded_ok after-state rc=$after_rc — inspect /tmp/merge_teams.log and /tmp/merge_teams_after.log in the api container; re-running this script merges the rest"
  exit 1
fi
