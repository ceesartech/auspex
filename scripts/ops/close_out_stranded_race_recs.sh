#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Close out the pending horse-racing recs stranded on races the drain will
# never get a result for (horse racing is dormant since 2026-11-01:
# services/data-ingestion/dags/sport_switches.py), with
# scripts/close_out_stranded_race_recs.py: each rec is settled from its race's
# result when The Racing API has one (/results/{race_id}), voided when it has
# none (an abandoned race: the feed drops it without a status).
#
# FIRST RUN — applied 2026-11-02 03:50Z (PLAN candidates=42 races=30 settle=10
# void=32): 10 recs settled from results that sat on page 2 of /results or
# were posted after the old two-day results window; 32 voided on abandoned
# meetings (Newbury 24 Oct, Aintree 25 Oct, Galway 27 Oct, Wexford 26 Oct,
# Bellewstown 25-26 Aug and seven single races). Its record is
# /app/backups/closed_out_race_recs_apply_20261102T035017Z.csv with its
# .reversal.jsonl and .inserted.jsonl.
#
# THIS RUN (the second) closes the last 7: Southwell (AW) 31 Oct 17:40 (4
# recs, racing_api_race_id rac_32302997831), 18:50 (2, rac_32303001705) and
# 19:25 (1, rac_32303003642). The fixture was abandoned for fog after its
# third race: the day's full, paginated /results list (59 results, 2 pages,
# read 2026-11-02 06:21Z) has Southwell 16:00, 16:35 and 17:05 only, and
# /results/{race_id} answers 404 "Result not found - race <id>" for all three.
# The drain asks about each race until it is 3 days old (load_racing_api
# DRAIN_DAYS), so each becomes a candidate here only after
# 2026-11-03 17:40Z, 18:50Z and 19:25Z.
# Reviewed plan — rehearsed end to end on a local replica of the
# 2026-11-02T06:38:41Z read-only export, through a docker shim, with the
# clock moved to 2026-11-03 19:30Z and the API's real answers for the three
# races (read 06:21Z): DONE, the after-state exact, a re-run ALREADY DONE
# with no backup and no API call:
#   PLAN candidates=7 races=3 settle=0 void=7 unmatched=0 errors=0
# A result posted late would settle that race instead (settle + void stays
# 7). Run early on the same replica: 07:00Z on 11-02 NOT YET; 18:00Z on
# 11-03 PARTIAL (4 voided, 3 left until 19:25Z); 18:30Z NOT YET; 19:00Z
# PARTIAL (2 voided, 1 left); 19:30Z DONE (1 voided); 19:40Z ALREADY DONE.
#
# WHEN: at or after 2026-11-03 19:30Z, and before the Racing API subscription
# is cancelled (the plan asks the API about every race; without it every race
# is an error and the run refuses). Run earlier, it closes only the races
# already out of the drain's window and exits 3 (PARTIAL) with the time to
# re-run; with none out yet it exits 3 (NOT YET) and changes nothing. It also
# refuses (step 0) while a UFC card is being fought, while another
# scripts/ops wrapper or the soccer runbook runs on this host, and while an
# apply of a scripts/ops tool is still running in the api container.
#
# The gate (step 0 and, binding, step 2): candidates <= 7, races <= 3,
# settle <= 7, void <= 7, unmatched = 0, errors = 0, settle + void =
# candidates, every candidate race one of the three reviewed
# (REVIEWED_RACES), and at step 2 candidates = the before-state's stranded
# count. The apply is pinned to the dry run's exact counts (--expect-plan,
# every key: any drift refuses before a write). --raced-before
# 2026-11-01T00:00:00Z keeps races from 1 Nov on out (never reviewed; no rec
# on them is pending). Every tool run of one pass measures the drain window
# from one instant (--as-of T0), so the dry run, the apply and the re-plan
# see the same set even when a race leaves the window mid-run; the tool
# refuses an --as-of later than the database's clock, so the window can
# never be moved forward from here.
# The tool also names the recs it keeps out: BEYOND_PIN (stranded on races
# at or after RACED_BEFORE — a new reviewed pin is needed) and IN_WINDOW
# (before RACED_BEFORE but still inside the drain's window). This wrapper
# never touches either, and never reports ALREADY DONE while either exists.
#   0. preconditions, read only: this script's lock (~/.auspex_ops); no other
#      scripts/ops process on this host; no apply of a scripts/ops tool still
#      running in the api container (one cut short on the host keeps
#      running there); no UFC card in progress (a UFC bout starting within
#      the last 6 h or the next hour). Then a read-only dry run: candidates =
#      0 with nothing beyond the pin or in the window exits 0 (ALREADY DONE,
#      no backup); candidates = 0 with recs in the window exits 3 (NOT YET);
#      with stranded recs beyond the pin exits 1 (NEW PIN NEEDED); a plan
#      outside the gate aborts here, before a backup;
#   1. verified backup (B2) — abort if it does not land;
#   2. T0 = now; before-state + dry run as of T0 (write-once audit CSV) —
#      the gate;
#   3. apply as of T0, pinned (one transaction; every void guarded on
#      'pending'). The reversal record, both write-once: <apply
#      csv>.reversal.jsonl, written first, holds the pre-image of every row
#      the apply can update or that sits where it can insert — the
#      candidate races' races, race_entrants, race_predictions and
#      race_recommendations rows, and the horses, jockeys, trainers and
#      leagues rows the settled races' results name; <apply
#      csv>.inserted.jsonl, written inside the transaction just before its
#      commit, holds the ids of the rows it inserted. Restore the pre-images
#      and delete those ids (race_entrants first) to reverse it;
#   4. after-state, EXACT: every rec of the apply CSV carries its planned
#      status and profit_loss; every race it settled is 'finished'; the
#      stranded count as of T0 fell by exactly the candidates; the race recs
#      created before T0 are all still there; a re-plan as of T0 plans 0.
#      Recs still in the window at T0 make it PARTIAL (exit 3), with the
#      time after which a re-run closes them.
# Exit: 0 done (or already done); 1 refused, aborted, or an after-state
# failed; 3 not finished yet — re-run after the time it names (NOT YET:
# nothing changed; PARTIAL: what was eligible is closed and verified).
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/close_out_stranded_race_recs.sh 2>&1 | tee -a ~/close_out_stranded_race_recs.log
# (AUSPEX_DIR, TMPDIR and AF_STATE_DIR exist for tests and rehearsals.)
set -uo pipefail
OPS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || { echo "ABORT: cannot find scripts/ops — nothing changed"; exit 1; }
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
# shellcheck source=scripts/ops/_airflow_lib.sh
. "$OPS/_airflow_lib.sh" || { log "ABORT: cannot load $OPS/_airflow_lib.sh — nothing changed"; exit 1; }
af_lock close_out_stranded_race_recs || exit 1
W=$(mktemp -d "${TMPDIR:-/tmp}/race_closeout.XXXXXX") || { log "ABORT: mktemp failed — nothing changed"; af_unlock; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/race_closeout_q.sql >/dev/null 2>&1
  af_unlock
}
trap cleanup EXIT
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_at() {  # psql_at <local.sql>: unaligned, '|'-separated
  $DC cp "$1" postgres:/tmp/race_closeout_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/race_closeout_q.sql'
}
plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o " $2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
RACED_BEFORE=2026-11-01T00:00:00Z
DRAIN_DAYS=3  # load_racing_api.DRAIN_DAYS (a test keeps them equal)
TOOL="python /app/scripts/close_out_stranded_race_recs.py --raced-before $RACED_BEFORE"
KEYS="candidates races settle void unmatched errors"
MAX_CANDIDATES=7
MAX_RACES=3
MAX_SETTLE=7
MAX_VOID=7
REVIEWED_RACES="rac_32302997831 rac_32303001705 rac_32303003642"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
APPLY_CSV=/app/backups/closed_out_race_recs_apply_$STAMP.csv
# run_tool <log name> <args...>: the tool in the api container; prints its PLAN line and exit
run_tool() {
  local name="$1"; shift
  $DC exec -T api sh -c "$TOOL $* > /tmp/$name.log 2>&1; rc=\$?; grep -E 'AS_OF|PLAN | settle (won|lost|void) |BEYOND_PIN|IN_WINDOW|REFUSED|pinned|Applied|ERROR|Traceback|Error' /tmp/$name.log | tail -25; echo \"tool exit \$rc\"; exit \$rc"
}
plan_of() { $DC exec -T api sh -c "grep -o 'PLAN candidates=.*' /tmp/$1.log | tail -1"; }
beyond_of() { plan_num "$($DC exec -T api sh -c "grep -o 'BEYOND_PIN recs=[0-9]*' /tmp/$1.log | tail -1")" recs; }
# window_of <log name>: sets win_recs, win_races (counts, or "missing") and win_until (an instant, "none", or "")
window_of() {
  local line
  line=$($DC exec -T api sh -c "grep -o 'IN_WINDOW recs=[0-9]* races=[0-9]* until=[^ ]*' /tmp/$1.log | tail -1")
  win_recs=$(plan_num "$line" recs)
  win_races=$(plan_num "$line" races)
  win_until=$(echo "$line" | sed -n 's/.* until=\([^ ]*\).*/\1/p')
}
read_plan() {  # read_plan <PLAN line>: sets p_<key> for every key
  local k
  for k in $KEYS; do eval "p_$k=\$(plan_num \"\$1\" $k)"; done
}
# gate <PLAN line> <its CSV> <step>: 0 when the plan is inside the reviewed set (ceilings, adding up, unmatched and
# errors 0, every race reviewed); otherwise logs why and returns 1. The caller has run read_plan on it.
RACE_IDS_PY="import csv, sys; print(' '.join(sorted({r['racing_api_race_id'] for r in csv.DictReader(open(sys.argv[1]))})))"
gate() {
  local ids outside="" r n=0
  # shellcheck disable=SC2154  # p_* are set by read_plan
  local counts="$p_candidates $p_races $p_settle $p_void $p_unmatched $p_errors"
  if ! [[ "$counts" =~ ^[0-9]+(\ [0-9]+){5}$ ]]; then
    log "ABORT at step $3: the PLAN line is unreadable ('$1') — nothing changed"; return 1
  fi
  if [ "$p_candidates" -gt "$MAX_CANDIDATES" ] || [ "$p_races" -gt "$MAX_RACES" ] || [ "$p_settle" -gt "$MAX_SETTLE" ] \
     || [ "$p_void" -gt "$MAX_VOID" ] || [ "$p_unmatched" -ne 0 ] || [ "$p_errors" -ne 0 ] \
     || [ "$p_candidates" -ne $((p_settle + p_void)) ]; then
    log "ABORT at step $3: plan ($1) is outside the reviewed set (candidates<=$MAX_CANDIDATES, races<=$MAX_RACES, settle<=$MAX_SETTLE, void<=$MAX_VOID, unmatched=0, errors=0, settle+void=candidates) — review $2; nothing changed"; return 1
  fi
  [ "$p_candidates" = 0 ] && return 0
  ids=$($DC exec -T api python -c "$RACE_IDS_PY" "$2")
  for r in $ids; do
    n=$((n + 1))
    case " $REVIEWED_RACES " in *" $r "*) ;; *) outside="$outside $r" ;; esac
  done
  if [ -n "$outside" ] || [ "$n" -ne "$p_races" ] || ! [[ "$ids" =~ ^[a-z0-9_\ ]+$ ]]; then
    log "ABORT at step $3: the candidate races (${ids:-unreadable}) are not all among the reviewed ($REVIEWED_RACES), or not the plan's $p_races${outside:+ — outside:$outside} — review $2; nothing changed"; return 1
  fi
}

log "0/4 preconditions (read only)"
# Another scripts/ops wrapper (or the soccer runbook) on this host. This script's own processes and subshells carry
# its name; a second run of it is the lock's job. The list is read from here-strings, not `echo | grep -q`: grep -q
# stops at its first match, and under pipefail (SIGPIPE is ignored: _airflow_lib.sh) the echo's write error on a long
# list would read as "cannot list".
self=$(basename "$0")
procs=$(ps -eo pid=,args= 2>/dev/null)
if ! grep -q "^ *$$ " <<< "$procs"; then
  log "REFUSED: cannot list this host's processes (ps) to rule out another scripts/ops run — nothing changed"; exit 1
fi
others=$(awk -v self="$self" '$0 ~ /scripts\/ops\/[A-Za-z0-9_.-]+\.sh/ && index($0, self) == 0' <<< "$procs")
if [ -n "$others" ]; then
  log "REFUSED: another scripts/ops process is running on this host — wait for it to end, then re-run; nothing changed:"
  echo "$others"
  exit 1
fi
# An apply of any tool the scripts/ops wrappers run, still running in the api container (docker compose exec does not
# forward an interrupt: an apply cut short on the host keeps going there) — this tool's own included.
OPS_TOOLS=$(grep -ho '/app/scripts/[a-z0-9_]*\.py' "$OPS"/*.sh 2>/dev/null | sed 's#.*/##; s#\.py$##' | sort -u | tr '\n' ' ')
running=$(af_applies_running $OPS_TOOLS)
case $? in
  0) log "REFUSED: an apply is still running in the api container — wait for it to end; nothing changed:"; echo "$running"; exit 1 ;;
  1) ;;
  *) log "REFUSED: could not scan the api container's processes for a running apply — nothing changed"; exit 1 ;;
esac
cat > "$W/ufc.sql" <<'SQLEOF'
SELECT count(*), coalesce(to_char(min(m.match_date) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI"Z"'), 'none')
  FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'mma' AND m.match_date BETWEEN NOW() - INTERVAL '6 hours' AND NOW() + INTERVAL '1 hour';
SQLEOF
ufc=$(psql_at "$W/ufc.sql" | tr -d '[:space:]')
IFS='|' read -r ufc_n ufc_first <<< "$ufc"
if ! [[ "${ufc_n:-}" =~ ^[0-9]+$ ]]; then
  log "REFUSED: could not read whether a UFC card is in progress (got '$ufc') — nothing changed"; exit 1
fi
if [ "$ufc_n" != 0 ]; then
  log "REFUSED: a UFC card is in progress ($ufc_n bout row(s) start within the last 6 h or the next hour, the first at $ufc_first) — re-run once it is over; nothing changed"
  exit 1
fi
log "no other scripts/ops run, no apply running, no UFC card in progress"

PRE_AT=$(date -u +%FT%TZ)
log "0/4 already done? (read-only dry run as of $PRE_AT)"
run_tool race_closeout_pre --as-of "$PRE_AT" --out-csv /app/backups/closed_out_race_recs_precheck_$STAMP.csv
rc=$?
pre=$(plan_of race_closeout_pre)
beyond=$(beyond_of race_closeout_pre)
window_of race_closeout_pre
read_plan "$pre"
if [ "$rc" -ne 0 ] || [ "$p_candidates" = missing ] || [ "$beyond" = missing ] || [ "$win_recs" = missing ] \
   || [ "$win_races" = missing ] || [ -z "$win_until" ] || [ "$p_unmatched" != "0" ] || [ "$p_errors" != "0" ]; then
  log "ABORT at step 0: the pre-check (exit $rc) plan '${pre:-unreadable}' (beyond the pin: $beyond; in the window: $win_recs) is unreadable or carries unmatched / errors — review the precheck CSV and /tmp/race_closeout_pre.log in the api container; nothing changed"
  exit 1
fi
if [ "$beyond" != "0" ]; then
  log "NOTE: $beyond stranded race rec(s) sit on races at or after $RACED_BEFORE — outside the reviewed set, so this wrapper leaves them pending; closing them needs a new reviewed pin: RACED_BEFORE and the ceilings in this script. They are (all in /tmp/race_closeout_pre.log in the api container):"
  $DC exec -T api sh -c "grep 'beyond pin:' /tmp/race_closeout_pre.log | head -50"
fi
if [ "$win_recs" != "0" ]; then
  log "NOTE: $win_recs rec(s) on $win_races race(s) before $RACED_BEFORE are still inside the drain's $DRAIN_DAYS-day window (the drain asks The Racing API about them every tick and settles any whose result posts); each becomes a candidate once its race is $DRAIN_DAYS days old, the last after $win_until:"
  $DC exec -T api sh -c "grep 'in window:' /tmp/race_closeout_pre.log | head -50"
fi
if [ "$p_candidates" = "0" ]; then
  if [ "$beyond" = "0" ] && [ "$win_recs" = "0" ]; then
    log "ALREADY DONE: $pre — no stranded race rec left to settle or void, none still in the drain's window (no backup)"
    exit 0
  fi
  if [ "$beyond" != "0" ]; then
    log "NEW PIN NEEDED: $pre — nothing is left before $RACED_BEFORE to close now, but the $beyond stranded rec(s) above are not closed; nothing changed"
    exit 1
  fi
  log "NOT YET: $pre — none of the $win_recs rec(s) above has left the drain's window; re-run after $win_until; nothing changed (no backup)"
  exit 3
fi
gate "$pre" /app/backups/closed_out_race_recs_precheck_$STAMP.csv 0 || exit 1
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

T0=$(date -u +%FT%TZ)
cat > "$W/state.sql" <<SQLEOF
SELECT (SELECT count(*) FROM race_recommendations rec JOIN races r ON r.id = rec.race_id
         WHERE rec.status IN ('pending', 'placed') AND r.status NOT IN ('finished', 'cancelled', 'abandoned')
           AND r.race_date < TIMESTAMPTZ '$T0' - make_interval(days => $DRAIN_DAYS)
           AND r.race_date < TIMESTAMPTZ '$RACED_BEFORE') AS stranded,
       (SELECT count(*) FROM race_recommendations WHERE created_at < TIMESTAMPTZ '$T0') AS recs;
SQLEOF
log "2/4 before-state + dry run (as of $T0)"
state=$(psql_at "$W/state.sql" | tr -d '[:space:]')
IFS='|' read -r stranded0 recs0 <<< "$state"
if ! [[ "$stranded0" =~ ^[0-9]+$ && "$recs0" =~ ^[0-9]+$ ]]; then
  log "ABORT at step 2: the before-state query failed (got '$state') — nothing changed"; exit 1
fi
log "before: stranded race recs as of T0 (the tool's candidate set)=$stranded0, race recs created before T0=$recs0"
DRY_CSV=/app/backups/closed_out_race_recs_dryrun_$STAMP.csv
run_tool race_closeout_dry --as-of "$T0" --out-csv "$DRY_CSV"
rc=$?
plan=$(plan_of race_closeout_dry)
read_plan "$plan"
if [ "$rc" -ne 0 ]; then
  log "ABORT at step 2: the dry run failed (exit $rc; '$plan') — nothing changed"; exit 1
fi
gate "$plan" "$DRY_CSV" 2 || exit 1
# shellcheck disable=SC2154
if [ "$p_candidates" -ne "$stranded0" ] || [ "$p_candidates" -eq 0 ]; then
  log "ABORT at step 2: the dry run plans $p_candidates candidate(s), the before-state as of the same T0 counts $stranded0 (and 0 is nothing to do) — review $DRY_CSV; nothing changed"; exit 1
fi
pin=""
for k in $KEYS; do pin="$pin${pin:+,}$k=$(plan_num "$plan" "$k")"; done
log "plan: $plan"

log "3/4 apply (as of $T0, pinned: $pin)"
run_tool race_closeout_apply --as-of "$T0" --apply --expect-plan "$pin" --out-csv "$APPLY_CSV"
rc=$?
if [ "$rc" -ne 0 ]; then
  log "ABORT at step 3 (exit $rc; full log /tmp/race_closeout_apply.log in the api container): a plan that differs from the dry run writes nothing, and a failed apply rolled back — inspect before re-running"
  exit 1
fi

log "4/4 after-state"
RECS_PY="import csv, sys; rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['action'] in ('settle', 'void')]; print(','.join(r['rec_id'] + ':' + r['new_status'] + ':' + r['profit_loss'] for r in rows))"
RACES_PY="import csv, sys; print(','.join(sorted({r['race_id'] for r in csv.DictReader(open(sys.argv[1])) if r['action'] == 'settle'})))"
triples=$($DC exec -T api python -c "$RECS_PY" "$APPLY_CSV")
settled_races=$($DC exec -T api python -c "$RACES_PY" "$APPLY_CSV")
as_planned="unreadable"
if [[ "$triples" =~ ^[0-9a-z:,.-]+$ ]]; then
  values=$(echo "$triples" | tr ',' '\n' | sed "s/^\([^:]*\):\([^:]*\):\(.*\)$/('\1'::uuid,'\2',\3::numeric)/" | paste -sd, -)
  cat > "$W/as_planned.sql" <<SQLEOF
SELECT count(*) FROM race_recommendations rec JOIN (VALUES $values) AS v(rec_id, status, pl) ON v.rec_id = rec.id
 WHERE rec.status = v.status AND rec.profit_loss = v.pl AND rec.settled_at IS NOT NULL;
SQLEOF
  as_planned=$(psql_at "$W/as_planned.sql" | tr -d '[:space:]')
fi
finished="none"
if [ -n "$settled_races" ]; then
  finished="unreadable"
  if [[ "$settled_races" =~ ^[0-9a-f,-]+$ ]]; then
    ids=$(echo "$settled_races" | tr ',' '\n' | sed "s/^\(.*\)$/'\1'::uuid/" | paste -sd, -)
    echo "SELECT count(*) FROM races WHERE status = 'finished' AND id IN ($ids);" > "$W/finished.sql"
    finished=$(psql_at "$W/finished.sql" | tr -d '[:space:]')
  fi
fi
want_finished=0
[ -n "$settled_races" ] && want_finished=$(echo "$settled_races" | tr ',' '\n' | grep -c .)
[ "$finished" = "none" ] && finished=0
state=$(psql_at "$W/state.sql" | tr -d '[:space:]')
IFS='|' read -r stranded1 recs1 <<< "$state"
run_tool race_closeout_verify --as-of "$T0" --out-csv /app/backups/closed_out_race_recs_verify_$STAMP.csv
replan=$(plan_of race_closeout_verify)
beyond1=$(beyond_of race_closeout_verify)
window_of race_closeout_verify
want_stranded=$((stranded0 - p_candidates))
log "after: recs as planned $as_planned (expected $p_candidates), settled races finished $finished (expected $want_finished), stranded as of T0 $stranded1 (expected $want_stranded), race recs created before T0 $recs1 (expected $recs0), re-plan as of T0 '$replan' (expected candidates=0; beyond the pin $beyond1; still in the window $win_recs)"
if [ "$as_planned" = "$p_candidates" ] && [ "$finished" = "$want_finished" ] && [ "$stranded1" = "$want_stranded" ] \
   && [ "$recs1" = "$recs0" ] && [ "$(plan_num "$replan" candidates)" = "0" ] && [ "$beyond1" != missing ] \
   && [ "$win_recs" != missing ] && [ -n "$win_until" ]; then
  log "DONE: $p_settle rec(s) settled from their race's result, $p_void voided (no result: abandoned) — the apply CSV $APPLY_CSV, its .reversal.jsonl and .inserted.jsonl are the record"
  if [ "$beyond1" != "0" ]; then
    log "NOTE: $beyond1 stranded race rec(s) on races at or after $RACED_BEFORE are still pending (not part of this run) — a re-run exits 1 (NEW PIN NEEDED) until RACED_BEFORE and the ceilings are moved to a new reviewed pin"
  fi
  if [ "$win_recs" != "0" ]; then
    log "PARTIAL: $win_recs rec(s) on $win_races race(s) before $RACED_BEFORE were still inside the drain's window at $T0 and are still pending — re-run this script after $win_until to close them (exit 3)"
    exit 3
  fi
else
  log "FAILED after-state — inspect (the apply CSV $APPLY_CSV, its .reversal.jsonl and .inserted.jsonl are the reversal record)"
  exit 1
fi
