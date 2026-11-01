#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Move the football-data kickoffs promote_raw stored 1 h late (UK summer time
# read as UTC) back to UTC and merge the duplicates that left, with
# scripts/repair_football_data_kickoffs.py. promote_raw STAYS BLOCKED
# (check_legacy_kickoffs refuses every run while a row sits at a legacy
# kickoff) until this has run and its --check passes; then it may run again.
#
# RUN ORDER (2026-10-31; scripts/ops/soccer_repair_runbook.sh runs it, each
# step through its own wrapper with its own verified backup and checks):
#   1. scripts/ops/repoint_team_matches.sh --pass 1 — right after the
#      team-identity deploy (until then every sweep meeting Santos Laguna's or
#      Chivas' ESPN fixtures exits 3). It moves the Liga MX rows ESPN filed
#      under Santos (the Brazilian club's row) and Guadalajara (the Spanish
#      club's row): 33 sides, refusing 6 because football-data's copy sits on
#      the target pair (a same-pair twin), to be merged by THIS repair first.
#   2. scripts/ops/migrate_027_outcome_trigger.sh (before the first step that
#      fires the outcome trigger: step 3's adoptions, this repair's folds).
#   3. scripts/ops/merge_soccer_teams.sh — folds the 160 pairs where ESPN filed
#      a fixture under its own team rows (keeps the ESPN row, deletes ~171
#      football-data rows; promote_raw then writes those fixtures onto the
#      surviving ESPN row, never beside it).
#   4. THIS script. --dedupe alias-merges the 4 copies step 1 refused for:
#      each into its ESPN row, still on the wrong club, which the reviewed map
#      (scripts/data/repoint_team_matches.csv) moves onto the copy's pair.
#   5. scripts/ops/repoint_team_matches.sh --pass 2 (never a bare --apply):
#      the 6 sides step 1 refused have no twin left and move (4 ESPN rows +
#      the 2 stale rows that move with them); 32 read 'done' (the tool
#      follows the opponents step 3 renamed, 'America' -> 'Club America',
#      through teams.metadata.aliases), 1 is 'match row not found' (step 3
#      folded cb4e21fc away). Until then promote_raw refuses those 4 fixtures
#      (alias slot: no insert beside the same match under other team ids),
#      loudly, exit 3. The alias-slot guard needs a SHARED CLUB: a fixture
#      filed under two other team ids does not trip it, so step 1 must have
#      moved the map's two-sided rows (Santos v Guadalajara 2026-08-17,
#      4a31da6c / da444426) before any MX1 raw for that date is promoted.
#   6. scripts/ops/repair_stranded_recs_soccer.sh; 7. scripts/ops/cleanup_soccer_orphans.sh
#      (AFTER: the 26 stale rows step 4 folds are finished by then and leave
#      its set; the 2 stale Liga MX rows sit beside their ESPN rows once step
#      5 has run); 8. backfill_soccer_regulation_scores.sh (+ re-grade);
#      9. backfill_season_labels.sh;
#      10. compute_features --backfill-finished soccer; 11. the merge's odds check.
# If step 1 moves every side (football-data twins allowed), step 4 plans those
# 4 copies as ordinary merges (72 merges, 0 alias merges) and step 5 has
# nothing to do. If the team merge (step 3) has NOT run, its alias pairs are
# left for it: --shift refuses every copy whose new kickoff is the exact
# kickoff of the same fixture under other team ids, so this aborts at step 2
# below with nothing written (never shift beside them: the merge would then
# collide on UNIQUE(home_team_id, away_team_id, match_date)). Without the
# reviewed re-point map in scripts/data/, the 4 copies stay refused the same
# way and this aborts too.
#
# Reviewed plan (rehearsed 2026-10-31 on a local copy of a read-only prod
# export taken 08:08Z: all soccer matches, raw_football_data, leagues, teams,
# soccer odds and match_stats, the 2026-04-01+ soccer predictions, recs,
# snapshots and features), after steps 1 and 3 run with their own tools:
#   dedupe merge=68 alias_merge=4 fold=26 refuse=0; shift 63,533 refuse=0
#   (re-played on the 11:08Z export: the same dedupe, shift 63,532)
#   (step 1 moving all 39 sides instead: merge=72 alias_merge=0, the same shift)
#   --dedupe merge: a copy into its ESPN row (same team ids, same final score;
#            65 SC0 odds and 136 match_stats move);
#   --dedupe alias_merge: the same, into an ESPN row the re-point map is about
#            to move onto the copy's pair (MX1 4); only the match_stats of a
#            club in that row move (4 of 8; the other club's is archived and
#            promote_raw writes it again after step 5: runbook step 14);
#   --dedupe fold: a copy into the stale pre-023 'scheduled' row at exactly its
#            corrected kickoff (CN1 21, NO1 3, BR1 2): the row takes the copy's
#            score and turns 'finished', guarded on the values observed, no
#            open rec (on it or its predictions) and no post-kickoff
#            prediction. The outcome trigger then grades 138 of its 810
#            predictions (match_result 54, over_under 42, btts 42; the same
#            under migration 011 and 027) and settles nothing: its 31 recs are
#            all 'void' already, each with its profit_loss set. 52 match_stats
#            move. In all 98 copies go and 1,396 records are archived.
#   --shift: every other BST row moves back 1 h.
#   after:   check_legacy_kickoffs passes, 0 football-data duplicates.
# The bounds below are that plan's: the merge that runs first lowers the counts.
#   0. already done? Read-only dry runs and --check first: when both plans are
#      empty (no merge, alias merge, fold, shift or refusal, nothing pending)
#      and --check passes, this exits 0 without a backup (a re-run after
#      success);
#   1. verified backup (B2) — abort if it does not land;
#   2. before-state, --check (expected to FAIL now), dedupe dry run and shift
#      dry run — abort unless both plans are within the reviewed bounds with 0
#      refusals, no fold row holds a rec without a profit_loss, and the shift
#      sees exactly the dedupe's merges + alias merges + folds as still to do
#      (nothing is written before this passes);
#   3. dedupe apply, pinned to step 2's own counts, every key and
#      fold_unsettled_recs=0 (--expect-plan: any drift refuses before a write;
#      the tool also refuses unpinned folds holding unsettled recs); each
#      batch archives (gzip JSONL in /app/backups)
#      before its first write, one transaction per 100 copies;
#   4. shift dry run again — abort unless it plans exactly step 2's shifts with
#      0 refusals and nothing left to dedupe;
#   5. shift apply, pinned to those counts (one transaction per 5,000 rows);
#   6. after-state, derived from the dry runs: --check MUST pass (exit 0),
#      soccer rows created before T0 fall by exactly merges + alias merges +
#      folds, stale 'scheduled' soccer rows (kickoff > 7 days before T0) by
#      exactly folds, fresh dry runs plan nothing, and the folded rows still
#      hold 0 recs without a profit_loss.
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status; a pipe through tail would report tail's 0.
# Re-runnable: merged and folded copies are gone, shifted rows plan as done.
# AFTERWARDS: the trigger grades only the market types it knows; the folded
# rows' other predictions stay ungraded until runbook step 15
# (scripts/ops/grade_folded_soccer_rows.sh: scripts/grade_matches.py, pinned,
# one transaction) grades them (step 6 prints the ids; 504 on the 2026-11-01
# rehearsal). It settles no rec there: besides open recs it settles only a
# won/lost/void rec whose profit_loss is NULL, and step 6 requires 0 of those
# on the folded rows (their 31 'void' recs keep their void and their P&L).
# The alias-merged fixtures' archived match_stats are written again by
# runbook step 14 (scripts/ops/promote_raw_after_repair.sh), after the labels
# repair (step 13, scripts/ops/repair_soccer_labels.sh): until that has run,
# promote_raw commits everything else but exits 3 on the 68 mis-decoded raw
# rows its screen refuses.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/repair_football_data_kickoffs.sh 2>&1 | tee -a ~/repair_football_data_kickoffs.log
# (AUSPEX_DIR and TMPDIR exist for tests and rehearsals.)
set -uo pipefail
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
W=$(mktemp -d "${TMPDIR:-/tmp}/fd_kickoffs.XXXXXX") || { echo "ABORT: mktemp failed — nothing changed"; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/fd_kickoffs_q.sql >/dev/null 2>&1
}
trap cleanup EXIT
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_at() {  # psql_at <local.sql>: unaligned, '|'-separated
  $DC cp "$1" postgres:/tmp/fd_kickoffs_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/fd_kickoffs_q.sql'
}
plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o " $2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
# The reviewed plan's upper bounds (see header).
MAX_MERGES=72        # merges + alias merges together: 68 same-id + the 4 Liga MX copies
MAX_ALIAS_MERGES=4
MAX_FOLDS=26
MAX_SHIFTS=63533
TOOL="python /app/scripts/repair_football_data_kickoffs.py"
KEEP="Loaded|Re-point map|No re-point map|Planned|Alias merge|Folds|Folded|NEXT|PLAN|refused and left|ERROR|DRY-RUN|Batch|CHECK|Done:|Traceback|Error"
step() {  # step <label> <log name> <tool args...>: run the tool in the api container, exit with its status
  local label="$1" name="$2"; shift 2
  $DC exec -T api sh -c "$TOOL $* > /tmp/$name.log 2>&1; rc=\$?; grep -E '$KEEP' /tmp/$name.log | grep -v ' refused [A-Z]' | tail -25; echo \"tool exit \$rc\"; exit \$rc"
  local rc=$?
  log "$label exit $rc"
  return $rc
}
plan_of() {  # plan_of <log name>: the PLAN line that run logged
  $DC exec -T api sh -c "grep -o 'PLAN .*' /tmp/$1.log | tail -1"
}
ids_of() {  # ids_of <audit CSV in the api container> <action>: the kept row ids of that action
  $DC exec -T api python -c "import csv, sys; print(' '.join(r['espn_row_id'] for r in csv.DictReader(open(sys.argv[1])) if r['action'] == sys.argv[2]))" "$1" "$2" 2>/dev/null
}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

log "0/6 already done? (read-only dry runs + --check)"
step "pre-check dedupe dry run" fd_kickoffs_dedupe_pre --dedupe --out-csv /app/backups/football_data_dedupe_precheck_$STAMP.csv
step "pre-check shift dry run" fd_kickoffs_shift_pre --shift --out-csv /app/backups/football_data_shift_precheck_$STAMP.csv
step "pre-check --check" fd_kickoffs_check_pre --check
pre_check=$?
pd=$(plan_of fd_kickoffs_dedupe_pre); ps=$(plan_of fd_kickoffs_shift_pre)
pre="$(plan_num "$pd" merge) $(plan_num "$pd" alias_merge) $(plan_num "$pd" fold) $(plan_num "$pd" refuse) $(plan_num "$ps" shift) $(plan_num "$ps" refuse) $(plan_num "$ps" dedupe_pending)"
if [ "$pre_check" -eq 0 ] && [ "$pre" = "0 0 0 0 0 0 0" ]; then
  log "ALREADY DONE: nothing to merge, fold or shift and --check passes — nothing to do (no backup)"
  exit 0
fi
log "not done yet (merge alias_merge fold refuse | shift refuse pending = '$pre', --check exit $pre_check) — continuing"

T0=$(date -u +%FT%TZ)
cat > $W/fd_kickoffs_state.sql <<SQLEOF
SELECT (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'soccer' AND m.created_at < TIMESTAMPTZ '$T0') AS soccer,
       (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'soccer' AND m.status = 'scheduled' AND m.created_at < TIMESTAMPTZ '$T0'
           AND m.match_date < TIMESTAMPTZ '$T0' - interval '7 days') AS stale;
SQLEOF

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

log "2/6 before-state + dry runs"
state=$(psql_at $W/fd_kickoffs_state.sql | tr -d '[:space:]')
IFS='|' read -r soccer0 stale0 <<< "$state"
if ! [[ "$soccer0" =~ ^[0-9]+$ && "$stale0" =~ ^[0-9]+$ ]]; then
  log "ABORT at step 2: the before-state query failed (got '$state') — nothing changed"; exit 1
fi
log "before (created before $T0): soccer rows=$soccer0, stale 'scheduled' soccer rows=$stale0"
step "before-state check (expected to FAIL now)" fd_kickoffs_check_before --check
step "dedupe dry run" fd_kickoffs_dedupe_dry --dedupe --out-csv /app/backups/football_data_dedupe_dryrun_$STAMP.csv
rc=$?
dplan=$(plan_of fd_kickoffs_dedupe_dry)
step "shift dry run" fd_kickoffs_shift_dry --shift --out-csv /app/backups/football_data_shift_dryrun_$STAMP.csv
src=$?
splan=$(plan_of fd_kickoffs_shift_dry)
m=$(plan_num "$dplan" merge); a=$(plan_num "$dplan" alias_merge); f=$(plan_num "$dplan" fold)
dr=$(plan_num "$dplan" refuse); fu=$(plan_num "$dplan" fold_unsettled_recs)
s=$(plan_num "$splan" shift); sr=$(plan_num "$splan" refuse); pend=$(plan_num "$splan" dedupe_pending)
if ! [[ "$m $a $f $dr $fu $s $sr $pend" =~ ^[0-9]+(\ [0-9]+){7}$ ]]; then
  log "ABORT at step 2: could not read the PLAN lines (dedupe exit $rc '$dplan'; shift exit $src '$splan') — nothing changed"; exit 1
fi
if [ "$dr" -ne 0 ] || [ "$sr" -ne 0 ]; then
  log "ABORT at step 2: $dr dedupe and $sr shift refusal(s) — see the dry-run CSVs in /app/backups (reason 'alias slot': run the team merge / repoint_team_matches first, and check the re-point map is in scripts/data/, see header) — nothing changed"; exit 1
fi
if [ "$fu" -ne 0 ]; then
  log "ABORT at step 2: the fold rows hold $fu rec(s) without a profit_loss (column fold_unsettled_recs in the dedupe dry-run CSV): grade_completed_matches.py would settle them by the final score — review them first — nothing changed"; exit 1
fi
if [ $((m + a)) -gt "$MAX_MERGES" ] || [ "$a" -gt "$MAX_ALIAS_MERGES" ] || [ "$f" -gt "$MAX_FOLDS" ] || [ "$s" -gt "$MAX_SHIFTS" ] || [ "$pend" -ne $((m + a + f)) ]; then
  log "ABORT at step 2: plan (merge=$m alias_merge=$a fold=$f shift=$s dedupe_pending=$pend) is outside the reviewed set (merge + alias_merge <= $MAX_MERGES, alias_merge <= $MAX_ALIAS_MERGES, fold <= $MAX_FOLDS, shift <= $MAX_SHIFTS, dedupe_pending = merge + alias_merge + fold) — nothing changed"; exit 1
fi
log "plan: merge=$m alias_merge=$a fold=$f shift=$s"

log "3/6 dedupe apply (pinned to the dry run's counts)"
step "dedupe apply" fd_kickoffs_dedupe --dedupe --apply --expect-plan merge=$m,alias_merge=$a,fold=$f,refuse=0,fold_unsettled_recs=0 \
  --out-csv /app/backups/football_data_dedupe_apply_$STAMP.csv \
  || { log "ABORT at step 3 (full log /tmp/fd_kickoffs_dedupe.log in the api container): if it refused before 'Batch', nothing changed; a batch that raised rolled back, batches logged as committed stay merged (archives in /app/backups) — inspect before re-running"; exit 1; }

log "4/6 shift dry run again"
step "shift dry run" fd_kickoffs_shift_dry2 --shift --out-csv /app/backups/football_data_shift_dryrun2_$STAMP.csv
splan2=$(plan_of fd_kickoffs_shift_dry2)
s2=$(plan_num "$splan2" shift); sr2=$(plan_num "$splan2" refuse); pend2=$(plan_num "$splan2" dedupe_pending)
if [ "$s2 $sr2 $pend2" != "$s 0 0" ]; then
  log "ABORT at step 4: the shift plan is now '$splan2', expected shift=$s refuse=0 dedupe_pending=0 — the dedupe is committed, no kickoff moved; inspect"; exit 1
fi

log "5/6 shift apply (pinned)"
step "shift apply" fd_kickoffs_shift --shift --apply --expect-plan shift=$s,refuse=0 \
  --out-csv /app/backups/football_data_shift_apply_$STAMP.csv \
  || { log "ABORT at step 5 (full log /tmp/fd_kickoffs_shift.log in the api container): the failing batch rolled back; batches logged as committed stay shifted — re-run this script (it re-plans)"; exit 1; }

log "6/6 after-state"
step "after-state check" fd_kickoffs_check_after --check
check_rc=$?
state=$(psql_at $W/fd_kickoffs_state.sql | tr -d '[:space:]')
IFS='|' read -r soccer1 stale1 <<< "$state"
want_soccer=$((soccer0 - m - a - f)); want_stale=$((stale0 - f))
log "after: soccer rows=${soccer1:-unknown} (expected $want_soccer), stale 'scheduled' soccer rows=${stale1:-unknown} (expected $want_stale)"
step "dedupe re-plan" fd_kickoffs_dedupe_verify --dedupe --out-csv /app/backups/football_data_dedupe_verify_$STAMP.csv
step "shift re-plan" fd_kickoffs_shift_verify --shift --out-csv /app/backups/football_data_shift_verify_$STAMP.csv
dv=$(plan_of fd_kickoffs_dedupe_verify); sv=$(plan_of fd_kickoffs_shift_verify)
replanned="$(plan_num "$dv" merge) $(plan_num "$dv" alias_merge) $(plan_num "$dv" fold) $(plan_num "$dv" refuse) $(plan_num "$sv" shift) $(plan_num "$sv" refuse)"
folded=$(ids_of /app/backups/football_data_dedupe_apply_$STAMP.csv fold)
aliased=$(ids_of /app/backups/football_data_dedupe_apply_$STAMP.csv alias_merge)
unsettled="none folded"
if [ -n "$folded" ]; then
  if [[ "$folded" =~ ^[0-9a-f-]+(\ [0-9a-f-]+)*$ ]]; then
    arr="ARRAY['$(echo "$folded" | sed "s/ /','/g")']::uuid[]"
    cat > $W/fd_kickoffs_unsettled.sql <<SQLEOF
SELECT count(*) FROM betting_recommendations br
 WHERE br.profit_loss IS NULL
   AND (br.match_id = ANY($arr) OR br.prediction_id IN (SELECT p.id FROM predictions p WHERE p.match_id = ANY($arr)));
SQLEOF
    unsettled=$(psql_at $W/fd_kickoffs_unsettled.sql | tr -d '[:space:]')
    rm -f $W/fd_kickoffs_unsettled.sql
  else
    unsettled="unreadable ids"
  fi
fi
log "recs without a profit_loss on the folded rows (directly or through their predictions): $unsettled (needs 0)"
if [ "$check_rc" -eq 0 ] && [ "$soccer1" = "$want_soccer" ] && [ "$stale1" = "$want_stale" ] && [ "$replanned" = "0 0 0 0 0 0" ] \
   && { [ "$unsettled" = "0" ] || [ "$unsettled" = "none folded" ]; }; then
  if [ -n "$aliased" ]; then
    log "DONE: check_legacy_kickoffs passes, but promote_raw runs and refuses the $(echo $aliased | wc -w | tr -d " ") alias-merged fixture(s) (exit 3) until the second re-point pass has moved them (and screens out the 68 mis-decoded raw rows until scripts/ops/repair_soccer_labels.sh, runbook step 13, has run; the runbook runs promote_raw at step 14). Folded rows (runbook step 15 grades the rest of their predictions; it settles no rec on them): ${folded:-none}"
    log "NEXT (runbook step 5, its own approval): bash /opt/auspex/scripts/ops/repoint_team_matches.sh --pass 2 — these alias-merged rows are still on the wrong club and now have no twin: $aliased"
  else
    log "DONE: check_legacy_kickoffs passes, so promote_raw no longer refuses to start; until scripts/ops/repair_soccer_labels.sh (runbook step 13) has run it commits everything else but exits 3 on the 68 mis-decoded raw rows its screen refuses (the runbook runs promote_raw at step 14, after that repair). Folded rows (runbook step 15 grades the rest of their predictions; it settles no rec on them): ${folded:-none}"
  fi
  log "THEN (runbook steps 6-7): scripts/ops/repair_stranded_recs_soccer.sh, scripts/ops/cleanup_soccer_orphans.sh"
else
  log "FAILED after-state: --check exit $check_rc (needs 0), soccer rows ${soccer1:-unknown}/$want_soccer, stale rows ${stale1:-unknown}/$want_stale, re-plan '$replanned' (needs '0 0 0 0 0 0'), unsettled recs on folded rows '$unsettled' (needs 0) — inspect (archives and audit CSVs are in /app/backups); promote_raw stays blocked"
  exit 1
fi
