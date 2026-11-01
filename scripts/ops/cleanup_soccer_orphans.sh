#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Clean up the stale 'scheduled' soccer fixture rows left by the
# fixture-identity bug (pre-2026-09-04 kickoff-time forks) and the 2026 World
# Cup knockout placeholders, and mark the ESPN-verified postponed ones, with
# scripts/repair_fixture_orphans.py.
#
# RUN ORDER (2026-10-31; scripts/ops/soccer_repair_runbook.sh runs it, each
# step through its own wrapper with its own verified backup and checks):
#   1. scripts/ops/repoint_team_matches.sh --pass 1
#   2. scripts/ops/migrate_027_outcome_trigger.sh
#   3. scripts/ops/merge_soccer_teams.sh (folds the forks on twin teams)
#   4. scripts/ops/repair_football_data_kickoffs.sh (folds each 1 h-late
#      football-data copy INTO the stale row at its corrected kickoff, then
#      shifts the rest)
#   5. scripts/ops/repoint_team_matches.sh --pass 2 (BEFORE this script: run
#      after it, a398e0fe would be merge-deleted and d1f660b6 merge-cancelled
#      while still on the wrong club, and the second pass could not move them;
#      step 2 below refuses while any Liga MX side sits on santos/guadalajara)
#   6. scripts/ops/repair_stranded_recs_soccer.sh: settles/voids the pending
#      recs on orphans that have a twin and re-points them onto the twin.
#      After the team merge (which folds Lecce v Roma and Olympiakos v OFI
#      Crete into their twins): 12 pending soccer recs -> 7 voided (written
#      after the twin's kickoff) and re-pointed, 0 settled, 5 left pending on
#      the row with no twin (Panathinaikos v Kifisia, which this script marks
#      postponed). Until it has run, step 2 below aborts (blocked_pending_recs);
#   7. THIS script
#   8. scripts/ops/backfill_soccer_regulation_scores.sh (+ the re-grade it prints)
#   9. scripts/ops/backfill_season_labels.sh
#  10. compute_features.py --backfill-finished soccer
#  11. merge_soccer_teams.py --check-after ... --odds-since ... (the merge's odds check)
#
# Plan, against a fresh replica of prod 2026-10-31 07:21Z (449 stale rows),
# after the stranded-recs step but BEFORE steps 3-5 (those tools were not on
# main yet); the ceilings below only fall once they have run:
#   merge_delete        185  pre-game odds/snapshots/predictions moved onto the
#                            finished twin (cutoff LEAST(orphan, twin kickoff)),
#                            orphan deleted
#   merge_cancel         19  same merge, but recs reference the orphan (by
#                            match_id or by prediction): status 'cancelled' +
#                            metadata.orphan_of = twin, recs untouched
#   delete_placeholder   65  World Cup bracket slots / team-vs-itself rows, no rec
#   mark_postponed        6  scripts/data/soccer_postponed_stale_rows.csv: the
#                            original date of a fixture ESPN moved out of its
#                            round (Panathinaikos v Kifisia, 5 pending recs;
#                            Chicago Fire v Vancouver; 4 CSL rows of 07-11/12).
#                            status 'postponed' + metadata.postponed_kickoff /
#                            postponed_at; grade_completed_matches then voids
#                            their open recs (still 'postponed' 72 h after
#                            kickoff)
#   left alone          174  fd_copy_awaiting_repair 27 (9 pinned): the twin is
#                            a football-data row (no ESPN identity) exactly 1 h
#                            later — run-order step 4 folds it into the orphan,
#                            so after it these rows are finished and gone from
#                            this set; one still held means the fold refused it.
#                            No twin: replayed_later 59, other_identity 42,
#                            never_resulted 43, flipped 3. Of the 7
#                            replayed_later rows with recs, 6 are the listed
#                            postponements; J1 Urawa v Fagiano Okayama
#                            (2026-06-06) was played on its date (ESPN
#                            401873698, 0-2): never resulted, not postponed —
#                            the jpn.1 backfill below heals it.
#   moved: 2,369 odds rows, 23,311 snapshots, 1,754 predictions onto 197 twins.
# Rehearsed end to end (this wrapper, through a docker shim) on that replica:
# after-state exact, re-plan zero, every rec row byte-identical before/after.
# Run-order steps 3-5 should only lower the merge count (the team merge folds
# the forks it creates itself; the fold finishes the held rows); if they raise
# it past a ceiling below, step 2 aborts and the plan needs a new review.
#   0. already done? A read-only dry run first (as of now): when it plans no
#      merge, placeholder, postponement or no-twin delete, no list mismatch
#      and nothing blocked, this exits 0 without a backup (a re-run after
#      success) — unless an earlier apply's follow-ups are still pending
#      (below): then it runs steps 5-6 for them only, and exits 0 or 2;
#   1. verified backup (B2) — abort if it does not land;
#   2. START = now (one instant for every run and state query below:
#      --as-of START); before-state + dry run (write-once audit CSV) — abort
#      if a Liga MX side still sits on santos/guadalajara (repoint pass 2, run
#      order step 5, has not run or not finished: the repoint wrapper's 'left'
#      count must be 0, or a398e0fe and d1f660b6 would be merged while still on
#      the wrong club — `--from 7` after a failed step 5, or `--only 7` too
#      early, stops here), the tool fails, a pending rec blocks it, the
#      postponed list no longer matches, or the plan exceeds the reviewed set
#      (merges <= 204, placeholders <= 65, postponed <= 6, no no-twin deletes);
#   3. apply, pinned to the dry run's counts (--expect-plan: any drift refuses
#      before a write); the archive (gzip JSONL) lands before the first write;
#      batches commit separately, every write guarded on the observed values;
#   4. after-state, EXPECTED: stale rows fall by exactly merges + placeholders
#      + postponed, orphan_of rows rise by exactly merge_cancel, postponed rows
#      set by this tool rise by exactly mark_postponed, the recs on past
#      fixtures are all still there (the rec generators only ever delete
#      pending recs on UPCOMING fixtures), and a fresh dry run plans nothing;
#   5. recompute features for the twins that received prices;
#   6. grade the predictions moved onto those twins (predictions only — no rec
#      is written by any step of this script).
# Follow-ups pending: right after an apply that merged rows commits, its
# write-once apply CSV is added to
# /app/backups/cleanup_soccer_orphans_followups_pending.txt (api container),
# and the list is emptied only once steps 5-6 succeeded. Steps 5-6 take their
# twins from every CSV on the list, so a run that ended INCOMPLETE (exit 2)
# or failed its after-state (exit 1) never loses them: a re-run of this
# script, or
#   bash /opt/auspex/scripts/ops/cleanup_soccer_orphans.sh --followups-only
# (steps 5-6 for the listed CSVs, nothing else; no backup: they only refresh
# derived data), runs them.
# Exit: 0 all done (or already done); 1 aborted or the after-state failed; 2 the cleanup is
# verified (steps 1-4) but step 5 or 6 failed — re-run with --followups-only.
# Each tool step writes its output to a log file and exits with the tool's own
# status; a pipe through tail would report tail's 0.
# Re-runnable: merged rows are gone, cancelled/postponed rows leave the
# candidate set, and a listed row already marked counts as done.
#
# AFTERWARDS: the never_resulted rows (42 after run-order steps 1-7 on the
# 2026-11-01 replica) are mostly real matches ESPN has full-time results for.
# Runbook step 12 (scripts/ops/backfill_soccer_never_resulted.sh, with
# scripts/backfill_never_resulted.py) records those results on exactly these
# rows and grades them, then runs this script again: a row it heals can leave
# a fork of the same fixture with a finished twin, which this merges. Not a
# blanket `fetch_upcoming.py --results --days-back 160` (it would re-write
# every other completed event of those leagues and days too).
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/cleanup_soccer_orphans.sh 2>&1 | tee -a ~/cleanup_soccer_orphans.log
set -uo pipefail
FOLLOWUPS_ONLY=""
case "${1:-}" in
  "") ;;
  --followups-only) FOLLOWUPS_ONLY=1 ;;
  *) echo "usage: $0 [--followups-only]   (see the header)"; exit 2 ;;
esac
cd "${AUSPEX_DIR:-/opt/auspex}" || exit 1
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_file() {  # psql_file <local.sql> [extra psql flags]
  local f="$1"; shift
  $DC cp "$f" postgres:/tmp/soccer_orphans_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c "psql -v ON_ERROR_STOP=1 -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" $* -f /tmp/soccer_orphans_q.sql"
}
plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o "$2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
POSTPONED_LIST=/app/scripts/data/soccer_postponed_stale_rows.csv
TOOL="python /app/scripts/repair_fixture_orphans.py --sport soccer --delete-placeholders --postponed-list $POSTPONED_LIST"
# The reviewed set (replica of prod 2026-10-31). repair_stranded_recs can move
# a row between merge_cancel and merge_delete, so the merge ceiling is shared.
MAX_MERGES=204
MAX_PLACEHOLDERS=65
MAX_POSTPONED=6
STATE_SQL=$(mktemp "${TMPDIR:-/tmp}/soccer_orphans_state.XXXXXX") || exit 1
cleanup() {
  rm -f "$STATE_SQL"
  $DC exec -T postgres rm -f /tmp/soccer_orphans_q.sql >/dev/null 2>&1
}
trap cleanup EXIT

PENDING=/app/backups/cleanup_soccer_orphans_followups_pending.txt
pending_csvs() { $DC exec -T api sh -c "cat $PENDING 2>/dev/null" </dev/null | grep -E '^/app/backups/[^ ]+\.csv$' | sort -u; }
write_pending() {  # stdin: the apply CSVs whose follow-ups are pending (empty: none)
  $DC exec -T api sh -c "cat > $PENDING.tmp && mv $PENDING.tmp $PENDING"
}
# Steps 5-6 for the twins of every given apply CSV; sets followup_failed. The list is emptied only on success.
run_followups() {
  local twins rc
  followup_failed=""
  log "5/6 recompute features for the twins that received prices (from $*)"
  # The twins come from the write-once apply CSVs.
  TWINS_PY="import csv, sys; print(','.join(sorted({r['twin_id'] for f in sys.argv[1:] for r in csv.DictReader(open(f)) if r['action'] in ('merge_delete', 'merge_cancel')})))"
  twins=$($DC exec -T api python -c "$TWINS_PY" "$@" </dev/null)
  rc=$?
  if [ "$rc" -ne 0 ] || ! [[ "$twins" =~ ^[0-9a-f,-]+$ ]]; then
    log "ERROR step 5: could not read the twins from $* (exit $rc) — steps 5-6 not run"
    followup_failed="steps 5-6 (no twin list)"
    return
  fi
  log "$(echo "$twins" | tr ',' '\n' | grep -c .) twins"
  $DC exec -T api python /app/scripts/compute_features.py --match-ids "$twins" </dev/null > /tmp/rfo_soccer_features.log 2>&1
  rc=$?
  tail -3 /tmp/rfo_soccer_features.log
  if [ "$rc" -ne 0 ]; then
    log "ERROR step 5 exit $rc: features not recomputed (full log /tmp/rfo_soccer_features.log)"
    followup_failed="step 5"
  fi

  log "6/6 grade the predictions moved onto those twins"
  $DC exec -T -e TWINS="$twins" api python - > /tmp/rfo_soccer_grade.log 2>&1 <<'PY'
import os, sys
sys.path.insert(0, "/app/scripts")
import psycopg2
from psycopg2.extras import RealDictCursor
import grade_completed_matches as g

# Predictions only (grade_match's own prediction loop): recs on the twins
# stay the regular grader's business. Re-runnable: only ungraded rows.
twins = [t for t in os.environ["TWINS"].split(",") if t]
graded = skipped = 0
with psycopg2.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
    for mid in twins:
        for m in g.list_finished_matches(cur, 0, mid):
            ctx = g.fetch_match_grading_context(cur, mid)
            for pred in g.list_ungraded_predictions(cur, mid):
                skip, actual, correct = g.compute_prediction_grade(pred, m, ctx)
                if skip is not None:
                    skipped += 1
                    continue
                g.update_prediction(cur, pred["prediction_id"], actual, correct)
                graded += 1
    conn.commit()
print(f"graded {graded} predictions on {len(twins)} soccer twins ({skipped} skipped: after kickoff / ungradable)")
PY
  rc=$?
  tail -3 /tmp/rfo_soccer_grade.log
  if [ "$rc" -ne 0 ]; then
    log "ERROR step 6 exit $rc: grading failed (full log /tmp/rfo_soccer_grade.log)"
    followup_failed="${followup_failed:+$followup_failed and }step 6"
  fi
  if [ -z "$followup_failed" ]; then
    write_pending < /dev/null \
      || log "WARNING: could not empty $PENDING (a re-run then repeats steps 5-6, which is harmless)"
  fi
}
incomplete() {  # incomplete <what>: exit 2
  log "INCOMPLETE (exit 2): the cleanup is verified, but $followup_failed failed — $1. Re-run steps 5-6 only (the"
  log "  apply CSVs stay listed in $PENDING until they succeed):"
  log "    bash /opt/auspex/scripts/ops/cleanup_soccer_orphans.sh --followups-only"
  exit 2
}
followups_for_pending() {  # steps 5-6 for the pending list ($csvs), then exit 0 or 2
  # shellcheck disable=SC2086  # $csvs is a word list
  run_followups $csvs
  [ -z "$followup_failed" ] || incomplete "steps 5/6 for $csvs"
  log "FOLLOW-UPS DONE: features and grading for $csvs"
  exit 0
}

if [ -n "$FOLLOWUPS_ONLY" ]; then
  csvs=$(pending_csvs | tr '\n' ' ')
  if [ -z "${csvs// /}" ]; then log "--followups-only: nothing pending in $PENDING — nothing to do"; exit 0; fi
  log "--followups-only: steps 5-6 for $csvs"
  followups_for_pending
fi

log "0/6 already done? (read-only dry run)"
PRE_AT=$(date -u +%FT%TZ)
$DC exec -T api sh -c "$TOOL --as-of $PRE_AT --out-csv /app/backups/fixture_orphans_soccer_precheck_$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/rfo_soccer_pre.log 2>&1; echo \"tool exit \$?\""
pre=$($DC exec -T api sh -c "grep -o 'PLAN .*' /tmp/rfo_soccer_pre.log | tail -1")
pre_counts="$(plan_num "$pre" merge_delete) $(plan_num "$pre" merge_cancel) $(plan_num "$pre" delete_placeholder) $(plan_num "$pre" delete_no_twin) $(plan_num "$pre" mark_postponed) $(plan_num "$pre" list_mismatch) $(plan_num "$pre" blocked_pending_recs)"
if [ "$pre_counts" = "0 0 0 0 0 0 0" ]; then
  csvs=$(pending_csvs | tr '\n' ' ')
  if [ -z "${csvs// /}" ]; then
    log "ALREADY DONE: $pre — nothing to merge, delete or mark, no follow-up pending (no backup)"
    exit 0
  fi
  log "the cleanup is done ($pre) but the follow-ups of $csvs are pending (a run ended INCOMPLETE or failed its after-state): steps 5-6 only"
  followups_for_pending
fi
log "not done yet (pre-plan '${pre:-unreadable}') — continuing"

log "1/6 backup"
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

# One instant for the before-state, the dry run, the apply, the re-plan and
# the after-state: every run gets --as-of START and the state query measures
# staleness from it too, so a row crossing the 96 h line mid-run is in all or
# none of them. Taken after the backup wait, right before the first read.
START=$(date -u +%FT%TZ)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
APPLY_CSV=/app/backups/fixture_orphans_soccer_apply_$STAMP.csv
cat > "$STATE_SQL" <<SQLEOF
SELECT (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'soccer' AND m.status = 'scheduled'
           AND m.match_date < TIMESTAMPTZ '$START' - interval '96 hours') AS stale,
       (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'soccer' AND m.status = 'cancelled' AND m.metadata ? 'orphan_of') AS orphan_of,
       (SELECT count(*) FROM betting_recommendations br JOIN matches m ON m.id = br.match_id
         WHERE br.created_at < TIMESTAMPTZ '$START'
           AND m.match_date < TIMESTAMPTZ '$START' - interval '4 days') AS recs,
       (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'soccer' AND m.status = 'postponed'
           AND m.metadata->>'status_source' = 'repair_fixture_orphans') AS postponed,
       -- scripts/ops/repoint_team_matches.sh's 'left': Liga MX sides still on Santos FC / CD Guadalajara.
       (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
          JOIN teams t ON t.id IN (m.home_team_id, m.away_team_id)
         WHERE l.name = 'Liga MX' AND l.sport = 'soccer' AND t.sport = 'soccer'
           AND t.normalized_name IN ('santos', 'guadalajara')) AS liga_mx_left;
SQLEOF

log "2/6 before-state + dry run (as of $START)"
state=$(psql_file "$STATE_SQL" -At | tr -d '[:space:]')
IFS='|' read -r stale0 orphan0 recs0 postponed0 left0 <<< "$state"
if ! [[ "$stale0" =~ ^[0-9]+$ && "$orphan0" =~ ^[0-9]+$ && "$recs0" =~ ^[0-9]+$ && "$postponed0" =~ ^[0-9]+$ && "$left0" =~ ^[0-9]+$ ]]; then
  log "ABORT at step 2: the before-state query failed (got '$state') — nothing changed"; exit 1
fi
if [ "$left0" != "0" ]; then
  log "ABORT at step 2: $left0 Liga MX side(s) still sit on santos/guadalajara — repoint pass 2 (runbook step 5) has not run or did not"
  log "  finish, and this cleanup would merge a398e0fe and d1f660b6 while they are still on the wrong club. Run it first:"
  log "    bash /opt/auspex/scripts/ops/repoint_team_matches.sh --pass 2   (the runbook: --from 5) — nothing changed"
  exit 1
fi
log "before: stale soccer rows=$stale0, cancelled orphan_of rows=$orphan0, recs on past fixtures=$recs0, postponed by this tool=$postponed0"
$DC exec -T api sh -c "$TOOL --as-of $START --out-csv /app/backups/fixture_orphans_soccer_dryrun_$STAMP.csv > /tmp/rfo_soccer_dry.log 2>&1; rc=\$?; grep -E 'candidates=| PLAN |postponed list|PENDING|ERROR|Traceback|Error' /tmp/rfo_soccer_dry.log | tail -15; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 2 exit $rc"
if [ "$rc" -ne 0 ]; then log "ABORT at step 2: the dry run failed (see above; full log /tmp/rfo_soccer_dry.log in the api container) — nothing changed"; exit 1; fi
plan=$($DC exec -T api sh -c "grep -o 'PLAN .*' /tmp/rfo_soccer_dry.log | tail -1")
md=$(plan_num "$plan" merge_delete); mc=$(plan_num "$plan" merge_cancel)
dp=$(plan_num "$plan" delete_placeholder); dn=$(plan_num "$plan" delete_no_twin)
mp=$(plan_num "$plan" mark_postponed); lm=$(plan_num "$plan" list_mismatch)
blocked=$(plan_num "$plan" blocked_pending_recs)
if ! [[ "$md $mc $dp $dn $mp $lm $blocked" =~ ^[0-9]+\ [0-9]+\ [0-9]+\ [0-9]+\ [0-9]+\ [0-9]+\ [0-9]+$ ]]; then
  log "ABORT at step 2: could not read the PLAN line (got '$plan') — nothing changed"; exit 1
fi
if [ "$blocked" -ne 0 ]; then
  log "ABORT at step 2: $blocked orphan(s) with a twin still hold open recs — run scripts/ops/repair_stranded_recs_soccer.sh first (runbook step 6; see header) — nothing changed"; exit 1
fi
if [ "$lm" -ne 0 ]; then
  log "ABORT at step 2: $lm row(s) of $POSTPONED_LIST no longer match the data (see the ERROR lines above) — review the list; nothing changed"; exit 1
fi
if [ "$dn" -ne 0 ] || [ $((md + mc)) -gt "$MAX_MERGES" ] || [ "$dp" -gt "$MAX_PLACEHOLDERS" ] || [ "$mp" -gt "$MAX_POSTPONED" ]; then
  log "ABORT at step 2: plan ($plan) is outside the reviewed set (merges <= $MAX_MERGES, placeholders <= $MAX_PLACEHOLDERS, postponed <= $MAX_POSTPONED, no no-twin deletes) — review the dry-run CSV; nothing changed"; exit 1
fi
log "plan: merge_delete=$md merge_cancel=$mc delete_placeholder=$dp mark_postponed=$mp"

log "3/6 apply (pinned to the dry run's counts, as of $START)"
$DC exec -T api sh -c "$TOOL --as-of $START --apply --expect-plan merge_delete=$md,merge_cancel=$mc,delete_placeholder=$dp,mark_postponed=$mp,delete_no_twin=0 --out-csv $APPLY_CSV > /tmp/rfo_soccer_apply.log 2>&1; rc=\$?; grep -E 'candidates=| PLAN |archived|DONE|ERROR|Traceback|Error' /tmp/rfo_soccer_apply.log | tail -15; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 3 exit $rc"
if [ "$rc" -ne 0 ]; then
  # Refused before any write (plan drift, pending recs, a list mismatch, an
  # argument), or after some committed batches when a row changed under its
  # lock (it says so).
  log "ABORT at step 3 (see above; full log /tmp/rfo_soccer_apply.log in the api container): if it refused or raised before 'archived', nothing changed; otherwise the committed batches stand and the archive holds every planned row — inspect before re-running"
  exit 1
fi
if [ $((md + mc)) -gt 0 ]; then
  listed=$({ pending_csvs; echo "$APPLY_CSV"; } | sort -u)  # read in full before the write
  echo "$listed" | write_pending \
    || log "WARNING: could not list $APPLY_CSV in $PENDING — if steps 5-6 fail below, run them by hand for it"
fi

log "4/6 after-state (as of $START)"
state=$(psql_file "$STATE_SQL" -At | tr -d '[:space:]')
IFS='|' read -r stale1 orphan1 recs1 postponed1 _ <<< "$state"
want_stale=$((stale0 - md - mc - dp - mp)); want_orphan=$((orphan0 + mc)); want_postponed=$((postponed0 + mp))
log "after: stale soccer rows=$stale1 (expected $want_stale), cancelled orphan_of rows=$orphan1 (expected $want_orphan), postponed by this tool=$postponed1 (expected $want_postponed), recs on past fixtures=$recs1 (expected $recs0)"
$DC exec -T api sh -c "$TOOL --as-of $START --out-csv /app/backups/fixture_orphans_soccer_verify_$STAMP.csv > /tmp/rfo_soccer_verify.log 2>&1; rc=\$?; grep -E 'candidates=| PLAN |postponed list' /tmp/rfo_soccer_verify.log; exit \$rc"
rc=$?
replan=$($DC exec -T api sh -c "grep -o 'PLAN .*' /tmp/rfo_soccer_verify.log | tail -1")
replanned="$(plan_num "$replan" merge_delete) $(plan_num "$replan" merge_cancel) $(plan_num "$replan" delete_placeholder) $(plan_num "$replan" mark_postponed) $(plan_num "$replan" list_mismatch)"
if [ "$rc" -ne 0 ] || [ "$stale1" != "$want_stale" ] || [ "$orphan1" != "$want_orphan" ] || [ "$postponed1" != "$want_postponed" ] || [ "$recs1" != "$recs0" ] || [ "$replanned" != "0 0 0 0 0" ]; then
  log "FAILED after-state: stale $stale1/$want_stale, orphan_of $orphan1/$want_orphan, postponed $postponed1/$want_postponed, recs $recs1/$recs0, re-plan '$replan' (exit $rc) — inspect (the archive and audit CSVs are in /app/backups)"
  exit 1
fi
log "after-state OK"

# Steps 5-6 only refresh derived data; a failure there leaves the verified
# cleanup standing but must not look like success: exit 2, never DONE.
followup_failed=""
csvs=$(pending_csvs | tr '\n' ' ')
if [ $((md + mc)) -gt 0 ]; then  # this run's CSV even when it could not be listed
  case " $csvs " in *" $APPLY_CSV "*) ;; *) csvs="$csvs $APPLY_CSV" ;; esac
fi
if [ -z "${csvs// /}" ]; then
  log "5/6 + 6/6 skipped: nothing was merged, no twin received prices"
else
  # shellcheck disable=SC2086  # this run's apply CSV and any earlier one still pending
  run_followups $csvs
fi

[ -z "$followup_failed" ] || incomplete "steps 5/6 need a manual re-run (see the ERROR lines above)"
log "DONE. The never_resulted rows left alone here are runbook step 12's (scripts/ops/backfill_soccer_never_resulted.sh)."
