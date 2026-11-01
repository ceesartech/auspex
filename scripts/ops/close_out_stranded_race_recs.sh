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
# Reviewed plan (prod 2026-11-01 02:5xZ, read only: the tool's own dry run,
# branch modules, PGOPTIONS default_transaction_read_only=on; re-run 04:08Z
# with the final tool: the same 42 per-rec verdicts, BEYOND_PIN recs=0):
#   PLAN candidates=42 races=30 settle=10 void=32 unmatched=0 errors=0
#   settle 10 recs / 8 races — 1 won +75.00, 9 lost -88.12 (net -13.12):
#          June 4 x3 and June 6 x4 sat on page 2 of /results, which
#          fetch_results never read; Deauville 23 Aug 15:35 x3 was posted
#          after the two-day results window had closed;
#   void   32 recs / 22 races — 404 "Result not found" and absent from the
#          day's full results list: Newbury 24 Oct 4, Aintree 25 Oct 4,
#          Galway 27 Oct 4, Wexford 26 Oct 2, Bellewstown 25-26 Aug 7,
#          Thirsk 28 Aug 3, Killarney 20 Aug 2, Le Lion-d'Angers 4 Sep 2,
#          Cork 29 Sep 2, Doncaster 26 Jun 1, Lingfield 6 Jul 1.
# The only recs that can still join are the 7 on Southwell (AW) 31 Oct 17:40 /
# 18:50 / 19:25 (no result at 02:55Z the next day): the drain asks about them
# until 2026-11-03 19:25Z and, if they are still unresolved then, they become
# candidates here (each settles or voids). So the dry run gates on ceilings —
# candidates <= 49, races <= 33, settle <= 17, void <= 39, unmatched = 0,
# errors = 0, settle + void = candidates — and the apply is pinned to the dry
# run's exact counts (--expect-plan, every key: any drift refuses before a
# write). --raced-before 2026-11-01T00:00:00Z keeps races from 1 Nov on out:
# they were never reviewed. The tool names the stranded recs it keeps out
# (BEYOND_PIN recs=N, one line per rec in its log): this wrapper never
# touches them, says so, and while any exist it does not report ALREADY DONE
# — closing them needs a new reviewed pin (RACED_BEFORE and the ceilings).
#   0. already done? A read-only dry run first: candidates = 0 with nothing
#      beyond the pin exits 0 without a backup or a write (a re-run after
#      success); candidates = 0 with stranded recs beyond it exits 1 (NEW
#      PIN NEEDED); unmatched or errors > 0 aborts here, before a backup;
#   1. verified backup (B2) — abort if it does not land;
#   2. T0 = now; before-state + dry run (write-once audit CSV) — the gate;
#   3. apply, pinned (one transaction; every void guarded on 'pending'). The
#      reversal record, both write-once: <apply csv>.reversal.jsonl, written
#      first, holds the pre-image of every row the apply can update or that
#      sits where it can insert — the candidate races' races, race_entrants,
#      race_predictions and race_recommendations rows, and the horses,
#      jockeys, trainers and leagues rows the settled races' results name;
#      <apply csv>.inserted.jsonl, written inside the transaction just before
#      its commit, holds the ids of the rows it inserted. Restore the
#      pre-images and delete those ids (race_entrants first) to reverse it;
#   4. after-state, EXACT: every rec of the apply CSV carries its planned
#      status and profit_loss; every race it settled is 'finished'; the
#      stranded count fell by exactly the candidates; the race recs created
#      before T0 are all still there; a fresh dry run plans 0.
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/close_out_stranded_race_recs.sh 2>&1 | tee -a ~/close_out_stranded_race_recs.log
# (AUSPEX_DIR and TMPDIR exist for tests and rehearsals.)
set -uo pipefail
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
W=$(mktemp -d "${TMPDIR:-/tmp}/race_closeout.XXXXXX") || { echo "ABORT: mktemp failed — nothing changed"; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/race_closeout_q.sql >/dev/null 2>&1
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
MAX_CANDIDATES=49
MAX_RACES=33
MAX_SETTLE=17
MAX_VOID=39
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
APPLY_CSV=/app/backups/closed_out_race_recs_apply_$STAMP.csv
# run_tool <log name> <args...>: the tool in the api container; prints its PLAN line and exit
run_tool() {
  local name="$1"; shift
  $DC exec -T api sh -c "$TOOL $* > /tmp/$name.log 2>&1; rc=\$?; grep -E 'PLAN | settle (won|lost|void) |BEYOND_PIN|REFUSED|pinned|Applied|ERROR|Traceback|Error' /tmp/$name.log | tail -25; echo \"tool exit \$rc\"; exit \$rc"
}
plan_of() { $DC exec -T api sh -c "grep -o 'PLAN candidates=.*' /tmp/$1.log | tail -1"; }
beyond_of() { plan_num "$($DC exec -T api sh -c "grep -o 'BEYOND_PIN recs=[0-9]*' /tmp/$1.log | tail -1")" recs; }
read_plan() {  # read_plan <PLAN line>: sets p_<key> for every key
  local k
  for k in $KEYS; do eval "p_$k=\$(plan_num \"\$1\" $k)"; done
}

log "0/4 already done? (read-only dry run)"
run_tool race_closeout_pre --out-csv /app/backups/closed_out_race_recs_precheck_$STAMP.csv
pre=$(plan_of race_closeout_pre)
beyond=$(beyond_of race_closeout_pre)
read_plan "$pre"
# shellcheck disable=SC2154  # p_* are set by read_plan
if [ "$p_candidates" = missing ] || [ "$beyond" = missing ] || [ "$p_unmatched" != "0" ] || [ "$p_errors" != "0" ]; then
  log "ABORT at step 0: pre-plan '${pre:-unreadable}' (beyond the pin: $beyond) is unreadable or carries unmatched / errors — review the precheck CSV; nothing changed"
  exit 1
fi
if [ "$beyond" != "0" ]; then
  log "NOTE: $beyond stranded race rec(s) sit on races at or after $RACED_BEFORE — outside the reviewed set, so this wrapper leaves them pending; closing them needs a new reviewed pin: RACED_BEFORE and the ceilings in this script. They are (all in /tmp/race_closeout_pre.log in the api container):"
  $DC exec -T api sh -c "grep 'beyond pin:' /tmp/race_closeout_pre.log | head -50"
fi
if [ "$p_candidates" = "0" ]; then
  if [ "$beyond" = "0" ]; then
    log "ALREADY DONE: $pre — no stranded race rec left to settle or void (no backup)"
    exit 0
  fi
  log "NEW PIN NEEDED: $pre — nothing is left before $RACED_BEFORE, but the $beyond stranded rec(s) above are not closed; nothing changed"
  exit 1
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
log "before: stranded race recs (the tool's candidate set)=$stranded0, race recs created before T0=$recs0"
run_tool race_closeout_dry --out-csv /app/backups/closed_out_race_recs_dryrun_$STAMP.csv
rc=$?
plan=$(plan_of race_closeout_dry)
read_plan "$plan"
# shellcheck disable=SC2154
counts="$p_candidates $p_races $p_settle $p_void $p_unmatched $p_errors"
if [ "$rc" -ne 0 ] || ! [[ "$counts" =~ ^[0-9]+(\ [0-9]+){5}$ ]]; then
  log "ABORT at step 2: the dry run failed (exit $rc) or its PLAN line is unreadable ('$plan') — nothing changed"; exit 1
fi
# shellcheck disable=SC2154
if [ "$p_candidates" -gt "$MAX_CANDIDATES" ] || [ "$p_races" -gt "$MAX_RACES" ] || [ "$p_settle" -gt "$MAX_SETTLE" ] \
   || [ "$p_void" -gt "$MAX_VOID" ] || [ "$p_unmatched" -ne 0 ] || [ "$p_errors" -ne 0 ] \
   || [ "$p_candidates" -ne $((p_settle + p_void)) ] || [ "$p_candidates" -ne "$stranded0" ]; then
  log "ABORT at step 2: plan ($plan) is outside the reviewed set (candidates<=$MAX_CANDIDATES and = the before-state's $stranded0, races<=$MAX_RACES, settle<=$MAX_SETTLE, void<=$MAX_VOID, unmatched=0, errors=0, settle+void=candidates) — review the dry-run CSV; nothing changed"; exit 1
fi
pin=""
for k in $KEYS; do pin="$pin${pin:+,}$k=$(plan_num "$plan" "$k")"; done
log "plan: $plan"

log "3/4 apply (pinned: $pin)"
run_tool race_closeout_apply --apply --expect-plan "$pin" --out-csv "$APPLY_CSV"
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
run_tool race_closeout_verify --out-csv /app/backups/closed_out_race_recs_verify_$STAMP.csv
replan=$(plan_of race_closeout_verify)
beyond1=$(beyond_of race_closeout_verify)
want_stranded=$((stranded0 - p_candidates))
log "after: recs as planned $as_planned (expected $p_candidates), settled races finished $finished (expected $want_finished), stranded $stranded1 (expected $want_stranded), race recs created before T0 $recs1 (expected $recs0), re-plan '$replan' (expected candidates=0; beyond the pin $beyond1)"
if [ "$as_planned" = "$p_candidates" ] && [ "$finished" = "$want_finished" ] && [ "$stranded1" = "$want_stranded" ] \
   && [ "$recs1" = "$recs0" ] && [ "$(plan_num "$replan" candidates)" = "0" ] && [ "$beyond1" != missing ]; then
  log "DONE: $p_settle rec(s) settled from their race's result, $p_void voided (no result: abandoned) — the apply CSV $APPLY_CSV, its .reversal.jsonl and .inserted.jsonl are the record"
  if [ "$beyond1" != "0" ]; then
    log "NOTE: $beyond1 stranded race rec(s) on races at or after $RACED_BEFORE are still pending (not part of this run) — a re-run exits 1 (NEW PIN NEEDED) until RACED_BEFORE and the ceilings are moved to a new reviewed pin"
  fi
else
  log "FAILED after-state — inspect (the apply CSV $APPLY_CSV, its .reversal.jsonl and .inserted.jsonl are the reversal record)"
  exit 1
fi
