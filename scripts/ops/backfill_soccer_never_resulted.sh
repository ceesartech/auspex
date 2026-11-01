#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Record ESPN's result on the soccer rows repair_fixture_orphans reports as
# never_resulted (scripts/ops/cleanup_soccer_orphans.sh leaves them alone:
# real matches whose result never reached the row) and grade them, with
# scripts/backfill_never_resulted.py — only the rows the rehearsal reviewed
# for healing (scripts/data/soccer_never_resulted_reviewed.csv, decision
# 'heal'), never a blanket results sweep or grade_completed_matches --days N
# (the tool's docstring says why). Runbook step 12.
#
# Rehearsed 2026-11-01 on a local replica of a read-only prod export taken
# 02:42Z, after the runbook's steps 1-11 ran on it through a docker shim,
# with real ESPN scoreboards (fetched 2026-11-01 ~03:06Z): 42 never_resulted
# rows. 38 were healed (Chile 17, J1 14, Copa Libertadores 4, Belgium,
# Colombia, Champions League; 3,173 predictions graded on them, 0 recs
# changed — their 46 recs are all 'void' with a P&L — 3 teams stamped with
# their ESPN id): those 38 are the reviewed list's 'heal' rows, each with
# ESPN's final. The other 4 are its 'leave' rows, never passed to the tool:
# Coquimbo Unido v O'Higgins 2026-06-15 (a fork of the 2026-06-13 row, which
# owns ESPN 401850590 and is healed; runbook step 12 then re-runs
# cleanup_soccer_orphans.sh --after-never-resulted, which merges it),
# Queretaro v Monterrey 2026-09-05 (not on ESPN's mex.1 scoreboard that
# weekend), Rayo Vallecano v Athletic Club 2026-10-10 (not on ESPN's
# scoreboard that weekend: re-dated) and Atromitos v AEK 2026-10-17 (ESPN:
# postponed). Rehearsed again on a 05:15Z export (ESPN fetched ~05:19Z), the
# reviewed list in place: the same 42 never_resulted rows after steps 1-11;
# PLAN candidates=38 heal=38 done=0 refuse=0 predictions_graded=3173
# predictions_regraded=0 recs_changed=0 teams_stamped=3 espn_failed=0; after:
# 3 never_resulted rows (the fork has a finished twin now), 0 new.
# The never_resulted class grows: a fixture whose result never arrives joins
# it 96 h after kickoff (e.g. a Liga MX fixture the results path refuses as
# identity drift until runbook step 1 has run, once the sweep's 3-day
# look-back has passed it). Such a row is NOT in the reviewed list: it is
# listed (LEFT ... NOT in the reviewed list) and left as it is, for a
# separate review — it neither aborts the step nor gets healed unreviewed.
# Pins: the bounds below are that plan's; the run applies exactly its own
# dry run's counts (--expect-plan), with the ESPN scoreboards that dry run
# fetched (one write-once cache), so ESPN changing between the dry run and
# the apply cannot change what is written.
#   0. already done? Read-only: the never_resulted ids (a repair_fixture_orphans
#      dry run, as of now), split into the reviewed 'heal' rows and the rest
#      (listed, left alone): none of the former exits 0 without a backup;
#      then the tool's dry run of them (it fetches ESPN): nothing to heal
#      exits 0 without a backup, but a failed run or any failed ESPN request
#      aborts here (its rows would be refused, so 'nothing to heal' would not
#      be the real plan) — re-run later, nothing changed;
#   1. verified backup (B2) — abort if it does not land;
#   2. START = now; the never_resulted ids as of START (abort unless the
#      cleanup is done: nothing to merge, delete or mark, nothing blocked) and
#      the tool's dry run of the reviewed 'heal' rows among them (ESPN into a
#      new write-once cache) — abort unless the plan is within the reviewed
#      set: candidates <= 38 and = the rows passed, heal <= 38, predictions
#      graded <= 3,173, teams stamped <= 3, and no prediction re-graded, no
#      rec changed, no row already finished, no ESPN request failed;
#   3. apply, pinned to the dry run's counts (--expect-plan, the same cache):
#      one transaction, every heal checked in a savepoint (see the tool);
#   4. after-state, EXPECTED: the healed rows are finished; a re-plan of the
#      same ids with the same cache heals 0, finds the healed ones done and
#      refuses the same rows; the healed rows hold no open rec and no settled
#      rec without a P&L; the orphans re-plan (as of START) finds no
#      never_resulted row that was not one before, and at most the refused
#      reviewed rows still in that class.
# Records (/app/backups, write-once): the orphans dry-run CSVs, the ids files
# (the rows passed to the tool, and every never_resulted row), the ESPN cache, and per tool run its CSV, <csv>.changes.csv (every
# prediction and rec it changed, old and new values) and
# <csv>.archive.jsonl.gz (the healed rows, their predictions and recs before).
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status; a pipe through tail would report tail's 0.
#
# RUN ORDER (scripts/ops/soccer_repair_runbook.sh runs it, each step through
# its own wrapper): 1 repoint_team_matches.sh --pass 1, 2 migrate_027_outcome_trigger.sh,
# 3 merge_soccer_teams.sh, 4 repair_football_data_kickoffs.sh,
# 5 repoint_team_matches.sh --pass 2, 6 repair_stranded_recs_soccer.sh,
# 7 cleanup_soccer_orphans.sh, 8 backfill_soccer_regulation_scores.sh,
# 9 backfill_season_labels.sh, 10 compute_features --backfill-finished soccer,
# 11 the merge's odds check, 12 THIS (then cleanup_soccer_orphans.sh again,
# for the forks of the rows it healed), 13 repair_soccer_labels.sh,
# 14 promote_raw_after_repair.sh, 15 grade_folded_soccer_rows.sh,
# 16 compute_features --backfill-finished soccer.
# It needs steps 1-11 done: the never_resulted class is the cleanup's, and
# the rows it heals sit on the merged teams (the reviewed ids are the rows as
# they stand after steps 1-11).
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/backfill_soccer_never_resulted.sh 2>&1 | tee -a ~/backfill_soccer_never_resulted.log
# (AUSPEX_DIR and TMPDIR exist for tests and rehearsals.)
set -uo pipefail
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
W=$(mktemp -d "${TMPDIR:-/tmp}/never_resulted.XXXXXX") || { echo "ABORT: mktemp failed — nothing changed"; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/never_resulted_q.sql >/dev/null 2>&1
}
trap cleanup EXIT
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_at() {  # psql_at <local.sql>: unaligned, '|'-separated
  $DC cp "$1" postgres:/tmp/never_resulted_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/never_resulted_q.sql'
}
plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o " $2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
# The reviewed set (rehearsal of 2026-11-01, see the header): its 'heal' rows are the only ones passed to the tool.
REVIEWED=/app/scripts/data/soccer_never_resulted_reviewed.csv
MAX_CANDIDATES=38
MAX_HEAL=38
MAX_GRADED=3173
MAX_TEAMS_STAMPED=3
ORPHANS="python /app/scripts/repair_fixture_orphans.py --sport soccer --delete-placeholders --postponed-list /app/scripts/data/soccer_postponed_stale_rows.csv"
TOOL="python /app/scripts/backfill_never_resulted.py"
KEEP="listed row|PLAN|refused [0-9]|recs |DRY-RUN|Committed|REFUSED|ERROR|Traceback|Error"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
B=/app/backups/never_resulted

# never_resulted_ids <as-of> <label>: a repair_fixture_orphans dry run as of <as-of>. Writes (write-once) the
# never_resulted ids the reviewed list marks 'heal' to $B_ids_<label>_$STAMP.txt and every never_resulted id to
# $B_all_<label>_$STAMP.txt, lists the others (LEFT: reviewed and left alone, or not in the reviewed list), and sets
# IDS / ALL (those paths), N_IDS / N_ALL and ORPHAN_PLAN. 1 when the dry run, its CSV or the list cannot be read.
SPLIT_PY="import csv, sys
orphans, reviewed, ids_out, all_out = sys.argv[1:5]
decision = {r['match_id']: (r['decision'], r['evidence']) for r in csv.DictReader(open(reviewed))}
bad = sorted(m for m, d in decision.items() if d[0] not in ('heal', 'leave'))
if bad:
    sys.exit('reviewed list: a decision other than heal/leave on ' + ', '.join(bad[:3]))
rows = [r for r in csv.DictReader(open(orphans)) if r['action'] == 'leave' and r['reason'] == 'never_resulted']
ids = [r['match_id'] for r in rows if decision.get(r['match_id'], ('',))[0] == 'heal']
open(ids_out, 'x').write(''.join(i + chr(10) for i in ids))
open(all_out, 'x').write(''.join(r['match_id'] + chr(10) for r in rows))
for r in rows:
    d = decision.get(r['match_id'])
    if d is None or d[0] != 'heal':
        why = 'reviewed, left alone: ' + d[1] if d else 'NOT in the reviewed list: left as it is, for a separate review'
        print('LEFT ' + r['match_id'] + ' ' + r['match_date'][:16] + ' ' + r['home'] + ' v ' + r['away'] + ' - ' + why)
print('COUNTS %d %d' % (len(ids), len(rows)))"
never_resulted_ids() {
  local at="$1" label="$2" csv rc out counts
  csv=${B}_orphans_${label}_$STAMP.csv
  IDS=${B}_ids_${label}_$STAMP.txt
  ALL=${B}_all_${label}_$STAMP.txt
  $DC exec -T api sh -c "$ORPHANS --as-of $at --out-csv $csv > /tmp/never_resulted_orphans_$label.log 2>&1; echo \"orphans dry run exit \$?\""
  ORPHAN_PLAN=$($DC exec -T api sh -c "grep -o 'PLAN .*' /tmp/never_resulted_orphans_$label.log | tail -1")
  out=$($DC exec -T api python -c "$SPLIT_PY" "$csv" "$REVIEWED" "$IDS" "$ALL" </dev/null)
  rc=$?
  counts=$(echo "$out" | grep '^COUNTS ' | tail -1)
  N_IDS=$(echo "$counts" | cut -d' ' -f2); N_ALL=$(echo "$counts" | cut -d' ' -f3)
  if [ "$rc" -ne 0 ] || ! [[ "$N_IDS $N_ALL" =~ ^[0-9]+\ [0-9]+$ ]] || [ -z "$ORPHAN_PLAN" ]; then
    log "could not read the never_resulted rows from $csv with $REVIEWED (exit $rc, got '$counts'; orphans plan '${ORPHAN_PLAN:-unreadable}')"
    return 1
  fi
  log "$N_ALL never_resulted row(s) as of $at ($ORPHAN_PLAN): $N_IDS of the reviewed 'heal' rows -> $IDS; $((N_ALL - N_IDS)) left alone:"
  echo "$out" | grep '^LEFT ' | sed 's/^LEFT /  /'
}
tool() {  # tool <log name> <args...>: the tool in the api container; exits with its status
  local name="$1"; shift
  $DC exec -T api sh -c "$TOOL $* > /tmp/$name.log 2>&1; rc=\$?; grep -E '$KEEP' /tmp/$name.log | tail -30; echo \"tool exit \$rc\"; exit \$rc"
}
plan_of() { $DC exec -T api sh -c "grep -o 'PLAN .*' /tmp/$1.log | tail -1 | sed 's/ (audit CSV.*//'"; }

log "0/4 already done? (read-only: the never_resulted rows, and the tool's dry run of the reviewed ones)"
PRE_AT=$(date -u +%FT%TZ)
never_resulted_ids "$PRE_AT" precheck || { log "ABORT at step 0 — nothing changed"; exit 1; }
if [ "$N_IDS" = "0" ]; then
  log "ALREADY DONE: none of the reviewed never_resulted rows is left to heal ($N_ALL never_resulted row(s) left alone, listed above) — nothing to do (no backup)"
  exit 0
fi
tool nr_pre --match-ids-file "$IDS" --as-of "$PRE_AT" --espn-cache ${B}_espn_precheck_$STAMP.json --out-csv ${B}_precheck_$STAMP.csv
rc=$?
pre=$(plan_of nr_pre)
pre_heal=$(plan_num "$pre" heal); pre_failed=$(plan_num "$pre" espn_failed)
if [ "$rc" -ne 0 ] || ! [[ "$pre_heal $pre_failed" =~ ^[0-9]+\ [0-9]+$ ]]; then
  log "ABORT at step 0: the tool's read-only dry run failed or printed no readable PLAN line (exit $rc, '${pre:-unreadable}'; full log /tmp/nr_pre.log in the api container) — nothing changed"
  exit 1
fi
if [ "$pre_failed" -ne 0 ]; then
  log "ABORT at step 0: $pre_failed ESPN request(s) failed ($pre) — the rows near those days are refused, so this is not the real plan; re-run later — nothing changed"
  exit 1
fi
if [ "$pre_heal" = "0" ]; then
  log "ALREADY DONE: nothing to heal ($pre) — the rows passed are refused, reasons above; nothing to do (no backup)"
  exit 0
fi
log "not done yet (pre-plan '$pre') — continuing"

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

START=$(date -u +%FT%TZ)
CACHE=${B}_espn_$STAMP.json
log "2/4 the never_resulted rows and the dry run (as of $START)"
never_resulted_ids "$START" dryrun || { log "ABORT at step 2 — nothing changed"; exit 1; }
DRY_IDS=$IDS
DRY_ALL=$ALL
n0=$N_IDS
if [ "$n0" = "0" ]; then log "DONE: none of the reviewed never_resulted rows is left to heal as of $START — no write"; exit 0; fi
busy="$(plan_num "$ORPHAN_PLAN" merge_delete) $(plan_num "$ORPHAN_PLAN" merge_cancel) $(plan_num "$ORPHAN_PLAN" delete_placeholder) $(plan_num "$ORPHAN_PLAN" mark_postponed) $(plan_num "$ORPHAN_PLAN" blocked_pending_recs) $(plan_num "$ORPHAN_PLAN" list_mismatch)"
if [ "$busy" != "0 0 0 0 0 0" ]; then
  log "ABORT at step 2: the soccer orphan cleanup is not done ($ORPHAN_PLAN) — run scripts/ops/cleanup_soccer_orphans.sh (runbook step 7) first; nothing changed"
  exit 1
fi
tool nr_dry --match-ids-file "$DRY_IDS" --as-of "$START" --espn-cache "$CACHE" --out-csv ${B}_dryrun_$STAMP.csv
rc=$?
plan=$(plan_of nr_dry)
cand=$(plan_num "$plan" candidates); heal=$(plan_num "$plan" heal); done_=$(plan_num "$plan" done); refuse=$(plan_num "$plan" refuse)
graded=$(plan_num "$plan" predictions_graded); regraded=$(plan_num "$plan" predictions_regraded)
recs=$(plan_num "$plan" recs_changed); stamped=$(plan_num "$plan" teams_stamped); failed=$(plan_num "$plan" espn_failed)
if [ "$rc" -ne 0 ] || ! [[ "$cand $heal $done_ $refuse $graded $regraded $recs $stamped $failed" =~ ^[0-9]+(\ [0-9]+){8}$ ]]; then
  log "ABORT at step 2: the dry run failed or printed no readable PLAN line (exit $rc, '$plan'; full log /tmp/nr_dry.log in the api container) — nothing changed"
  exit 1
fi
if [ "$cand" -gt "$MAX_CANDIDATES" ] || [ "$heal" -gt "$MAX_HEAL" ] || [ "$graded" -gt "$MAX_GRADED" ] || [ "$stamped" -gt "$MAX_TEAMS_STAMPED" ] \
   || [ "$regraded" -ne 0 ] || [ "$recs" -ne 0 ] || [ "$done_" -ne 0 ] || [ "$failed" -ne 0 ] || [ "$cand" -ne "$n0" ]; then
  log "ABORT at step 2: plan ($plan) is outside the reviewed set (candidates <= $MAX_CANDIDATES and = the $n0 passed, heal <= $MAX_HEAL, predictions_graded <= $MAX_GRADED, teams_stamped <= $MAX_TEAMS_STAMPED, predictions_regraded = recs_changed = done = espn_failed = 0) — review ${B}_dryrun_$STAMP.csv and its .changes.csv; nothing changed"
  exit 1
fi
if [ "$heal" -eq 0 ]; then
  log "DONE: nothing to heal as of $START ($plan) — no write"
  exit 0
fi
log "plan: heal=$heal refuse=$refuse predictions_graded=$graded teams_stamped=$stamped"

log "3/4 apply (pinned to the dry run's counts, the same ESPN cache)"
APPLY_CSV=${B}_apply_$STAMP.csv
pin="candidates=$cand,heal=$heal,done=0,refuse=$refuse,predictions_graded=$graded,predictions_regraded=0,recs_changed=0,teams_stamped=$stamped,espn_failed=0"
tool nr_apply --match-ids-file "$DRY_IDS" --as-of "$START" --espn-cache "$CACHE" --apply --expect-plan "$pin" --out-csv "$APPLY_CSV"
rc=$?
log "step 3 exit $rc"
if [ "$rc" -ne 0 ]; then
  log "ABORT at step 3 (full log /tmp/nr_apply.log in the api container): the apply is one transaction — a refusal or a drift rolled it back, nothing changed"
  exit 1
fi

log "4/4 after-state"
tool nr_verify --match-ids-file "$DRY_IDS" --as-of "$START" --espn-cache "$CACHE" --out-csv ${B}_verify_$STAMP.csv
vrc=$?
vplan=$(plan_of nr_verify)
v="$(plan_num "$vplan" heal) $(plan_num "$vplan" done) $(plan_num "$vplan" refuse)"
HEALED_PY="import csv, sys; ids = [r['match_id'] for r in csv.DictReader(open(sys.argv[1])) if r['action'] == 'heal']; open(sys.argv[2], 'x').write(''.join(i + chr(10) for i in ids)); print(len(ids))"
healed_file=${B}_healed_$STAMP.txt
n_healed=$($DC exec -T api python -c "$HEALED_PY" "$APPLY_CSV" "$healed_file" </dev/null)
ids_sql=$($DC exec -T api sh -c "cat $healed_file 2>/dev/null" </dev/null | grep -E '^[0-9a-f-]{36}$' | sed "s/.*/'&'/" | paste -sd, -)
state="unreadable"
if [ -n "$ids_sql" ]; then
  cat > "$W/never_resulted_after.sql" <<SQLEOF
SELECT (SELECT count(*) FROM matches WHERE id IN ($ids_sql) AND status = 'finished') || '|' ||
       (SELECT count(*) FROM betting_recommendations WHERE match_id IN ($ids_sql)
           AND (status IN ('pending', 'placed') OR profit_loss IS NULL));
SQLEOF
  state=$(psql_at "$W/never_resulted_after.sql" | tr -d '[:space:]')
fi
never_resulted_ids "$START" verify || { log "FAILED after-state: the orphans re-plan could not be read — inspect"; exit 1; }
NEW_PY="import sys; before = set(open(sys.argv[1]).read().split()); after = set(open(sys.argv[2]).read().split()); print(len(after - before))"
new_rows=$($DC exec -T api python -c "$NEW_PY" "$DRY_ALL" "$ALL" </dev/null)
log "after: healed rows finished|open-or-unpaid recs on them = $state (expected $heal|0); re-plan heal/done/refuse '$v' (expected '0 $heal $refuse', exit $vrc); reviewed rows still never_resulted $N_IDS (at most the $refuse refused), never_resulted rows now $N_ALL, new ones $new_rows (expected 0)"
if [ "$n_healed" != "$heal" ] || [ "$state" != "$heal|0" ] || [ "$vrc" -ne 0 ] || [ "$v" != "0 $heal $refuse" ] \
   || [ "$new_rows" != "0" ] || ! [ "$N_IDS" -le "$refuse" ] 2>/dev/null; then
  log "FAILED after-state — inspect (the apply CSV, its .changes.csv and .archive.jsonl.gz are in /app/backups)"
  exit 1
fi
log "DONE: $heal never_resulted row(s) healed and graded ($graded predictions); $refuse refused and left as they are (reasons in $APPLY_CSV); $((N_ALL - N_IDS)) never_resulted row(s) outside the reviewed 'heal' rows left alone (listed above)"
