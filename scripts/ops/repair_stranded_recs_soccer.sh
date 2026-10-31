#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Settle / void the pending soccer recs stranded on stale fixture rows that
# have a finished twin, with scripts/repair_stranded_recs.py --sport soccer:
# each rec is graded from the twin (or voided when it was written at or after
# the earlier of the two kickoffs) and re-pointed onto the twin, so
# cleanup_soccer_orphans can retire the stale row.
#
# Reviewed plan (prod 2026-10-31, read only; re-played on a local replica of
# the 11:08Z export AFTER the team merge, which folds Lecce v Roma (76a299dc,
# 2 pending) and Olympiakos v OFI Crete (b25078f8, 1) into their ESPN twins
# and grades them there): 12 candidates —
#   void_post_kickoff 7  Santa Clara v Rio Ave 2, Kocaelispor v Samsunspor 1,
#                        Cruz Azul v Santos Laguna 1 (fb112f26 -> f0a5a0da),
#                        Benfica v Gil Vicente 1, Genclerbirligi v Kasimpasa 2:
#                        status 'void', profit 0, 'void: written after kickoff',
#                        re-pointed onto the twin;
#   settle            0
#   no_twin           5  Panathinaikos v Kifisia (left alone here;
#                        cleanup_soccer_orphans marks the row postponed and the
#                        grader voids them 72 h after kickoff).
# The counts can only fall before this runs (a twin's results, the grader), so
# the dry run gates on ceilings and the apply is pinned to the dry run's exact
# counts (--expect-plan, every key: any drift refuses before a write).
#   0. already done? A read-only dry run first: when it plans nothing to write
#      (0 settle, 0 void_post_kickoff, 0 void_no_twin, 0 ambiguous, 0
#      ungradable) this exits 0 without a backup (a re-run after success);
#   1. verified backup (B2) — abort if it does not land;
#   2. T0 = now; before-state + dry run (write-once audit CSV) — abort unless
#      settle = 0, void_post_kickoff <= 7, no_twin <= 5, void_no_twin = 0,
#      ambiguous = 0, ungradable = 0 and the counts add up to the candidates;
#   3. apply, pinned to the dry run's counts (one transaction; every UPDATE
#      guarded on status 'pending');
#   4. after-state, EXACT: every void_post_kickoff rec of the apply CSV is
#      'void' with profit 0, the post-kickoff label and its twin's match_id;
#      the recs created before T0 are all still there; the pending soccer recs
#      on stale scheduled rows fell by exactly void_post_kickoff + settle; a
#      fresh dry run plans 0 to write.
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status (the tool exits 1 when it finds no
# candidate at all; that is read as "nothing to do", not as a failure).
#
# RUN ORDER (scripts/ops/soccer_repair_runbook.sh runs it, each step through
# its own wrapper): 1 repoint_team_matches.sh --pass 1, 2 migrate_027_outcome_trigger.sh,
# 3 merge_soccer_teams.sh, 4 repair_football_data_kickoffs.sh,
# 5 repoint_team_matches.sh --pass 2, 6 THIS, 7 cleanup_soccer_orphans.sh
# (it refuses while a stale row with a twin holds a pending rec),
# 8 backfill_soccer_regulation_scores.sh (+ re-grade), 9 backfill_season_labels.sh,
# 10 compute_features --backfill-finished soccer, 11 the merge's odds check.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/repair_stranded_recs_soccer.sh 2>&1 | tee -a ~/repair_stranded_recs_soccer.log
# (AUSPEX_DIR and TMPDIR exist for tests and rehearsals.)
set -uo pipefail
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
W=$(mktemp -d "${TMPDIR:-/tmp}/stranded_soccer.XXXXXX") || { echo "ABORT: mktemp failed — nothing changed"; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/stranded_soccer_q.sql >/dev/null 2>&1
}
trap cleanup EXIT
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_at() {  # psql_at <local.sql>: unaligned, '|'-separated
  $DC cp "$1" postgres:/tmp/stranded_soccer_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/stranded_soccer_q.sql'
}
plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o " $2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
TOOL="python /app/scripts/repair_stranded_recs.py --sport soccer"
KEYS="candidates settle void_post_kickoff void_no_twin no_twin ambiguous ungradable"
MAX_VOID_POST_KICKOFF=7
MAX_NO_TWIN=5
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
APPLY_CSV=/app/backups/repaired_rec_ids_soccer_apply_$STAMP.csv
# run_tool <log name> <args...>: the tool in the api container; prints its PLAN line and exit
run_tool() {
  local name="$1"; shift
  $DC exec -T api sh -c "$TOOL $* > /tmp/$name.log 2>&1; rc=\$?; grep -E 'PLAN |POST-KICKOFF|AMBIGUOUS|pinned|Applied|ERROR|Traceback|Error' /tmp/$name.log | tail -25; echo \"tool exit \$rc\"; exit \$rc"
}
plan_of() { $DC exec -T api sh -c "grep -o 'PLAN candidates=.*' /tmp/$1.log | tail -1"; }
writes_of() {  # writes_of <PLAN line>: settle + void_post_kickoff + void_no_twin + ambiguous + ungradable, or "missing"
  local n=0 k v
  for k in settle void_post_kickoff void_no_twin ambiguous ungradable; do
    v=$(plan_num "$1" "$k"); [ "$v" = missing ] && { echo missing; return; }
    n=$((n + v))
  done
  echo "$n"
}

log "0/4 already done? (read-only dry run)"
run_tool stranded_soccer_pre --out-csv /app/backups/repaired_rec_ids_soccer_precheck_$STAMP.csv
pre=$(plan_of stranded_soccer_pre)
if [ "$(writes_of "$pre")" = "0" ]; then
  log "ALREADY DONE: $pre — nothing to settle or void (no backup)"
  exit 0
fi
log "not done yet (pre-plan '${pre:-unreadable}') — continuing"

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
SELECT (SELECT count(*) FROM betting_recommendations br JOIN matches m ON m.id = br.match_id
          JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'soccer' AND br.status = 'pending'
           AND (m.status = 'scheduled' OR (m.status = 'cancelled' AND m.metadata ? 'orphan_of'))
           AND m.match_date < TIMESTAMPTZ '$T0' - interval '24 hours') AS stranded,
       (SELECT count(*) FROM betting_recommendations WHERE created_at < TIMESTAMPTZ '$T0') AS recs;
SQLEOF
log "2/4 before-state + dry run (as of $T0)"
state=$(psql_at "$W/state.sql" | tr -d '[:space:]')
IFS='|' read -r stranded0 recs0 <<< "$state"
if ! [[ "$stranded0" =~ ^[0-9]+$ && "$recs0" =~ ^[0-9]+$ ]]; then
  log "ABORT at step 2: the before-state query failed (got '$state') — nothing changed"; exit 1
fi
log "before: pending soccer recs on stale rows (the tool's candidate set)=$stranded0, recs created before T0=$recs0"
run_tool stranded_soccer_dry --out-csv /app/backups/repaired_rec_ids_soccer_dryrun_$STAMP.csv
rc=$?
plan=$(plan_of stranded_soccer_dry)
for k in $KEYS; do eval "p_$k=\$(plan_num \"\$plan\" $k)"; done
# shellcheck disable=SC2154  # p_* are set by the eval above
counts="$p_candidates $p_settle $p_void_post_kickoff $p_void_no_twin $p_no_twin $p_ambiguous $p_ungradable"
if [ "$rc" -ne 0 ] || ! [[ "$counts" =~ ^[0-9]+(\ [0-9]+){6}$ ]]; then
  log "ABORT at step 2: the dry run failed (exit $rc) or its PLAN line is unreadable ('$plan') — nothing changed"; exit 1
fi
# shellcheck disable=SC2154
if [ "$p_settle" -ne 0 ] || [ "$p_void_post_kickoff" -gt "$MAX_VOID_POST_KICKOFF" ] || [ "$p_no_twin" -gt "$MAX_NO_TWIN" ] \
   || [ "$p_void_no_twin" -ne 0 ] || [ "$p_ambiguous" -ne 0 ] || [ "$p_ungradable" -ne 0 ] \
   || [ "$p_candidates" -ne $((p_settle + p_void_post_kickoff + p_void_no_twin + p_no_twin + p_ambiguous + p_ungradable)) ]; then
  log "ABORT at step 2: plan ($plan) is outside the reviewed set (settle=0, void_post_kickoff<=$MAX_VOID_POST_KICKOFF, no_twin<=$MAX_NO_TWIN, void_no_twin=0, ambiguous=0, ungradable=0, adding up to the candidates) — review the dry-run CSV; nothing changed"; exit 1
fi
pin=""
for k in $KEYS; do pin="$pin${pin:+,}$k=$(plan_num "$plan" "$k")"; done
log "plan: $plan"

log "3/4 apply (pinned: $pin)"
run_tool stranded_soccer_apply --apply --expect-plan "$pin" --out-csv "$APPLY_CSV"
rc=$?
if [ "$rc" -ne 0 ]; then
  log "ABORT at step 3 (exit $rc; full log /tmp/stranded_soccer_apply.log in the api container): a plan that differs from the dry run writes nothing; a rec no longer pending at the write is left alone (see .not_applied.csv) — inspect before re-running"
  exit 1
fi

log "4/4 after-state"
CHECK_PY="import csv, sys; ids = [r['rec_id'] + ':' + r['new_match_id'] for r in csv.DictReader(open(sys.argv[1])) if r['action'] == 'void_post_kickoff']; print(','.join(ids))"
pairs=$($DC exec -T api python -c "$CHECK_PY" "$APPLY_CSV")
voided="none"
if [ -n "$pairs" ]; then
  if [[ "$pairs" =~ ^[0-9a-f:,-]+$ ]]; then
    values=$(echo "$pairs" | tr ',' '\n' | sed "s/^\(.*\):\(.*\)$/('\1'::uuid,'\2'::uuid)/" | paste -sd, -)
    cat > "$W/voided.sql" <<SQLEOF
SELECT count(*) FROM betting_recommendations br JOIN (VALUES $values) AS v(rec_id, twin) ON v.rec_id = br.id
 WHERE br.status = 'void' AND br.profit_loss = 0 AND br.actual_result = 'void: written after kickoff'
   AND br.match_id = v.twin;
SQLEOF
    voided=$(psql_at "$W/voided.sql" | tr -d '[:space:]')
  else
    voided="unreadable ids"
  fi
fi
[ "$voided" = "none" ] && voided=0
state=$(psql_at "$W/state.sql" | tr -d '[:space:]')
IFS='|' read -r stranded1 recs1 <<< "$state"
run_tool stranded_soccer_verify --out-csv /app/backups/repaired_rec_ids_soccer_verify_$STAMP.csv
replan=$(plan_of stranded_soccer_verify)
want_stranded=$((stranded0 - p_void_post_kickoff - p_settle))
log "after: void_post_kickoff recs voided onto their twin $voided (expected $p_void_post_kickoff), pending soccer recs on stale rows $stranded1 (expected $want_stranded), recs created before T0 $recs1 (expected $recs0), re-plan '$replan' (expected 0 to write)"
if [ "$voided" = "$p_void_post_kickoff" ] && [ "$stranded1" = "$want_stranded" ] && [ "$recs1" = "$recs0" ] && [ "$(writes_of "$replan")" = "0" ]; then
  log "DONE: $p_void_post_kickoff rec(s) voided as written after kickoff and re-pointed onto their twins; $p_no_twin without a twin left for cleanup_soccer_orphans (postponed) and the grader"
else
  log "FAILED after-state — inspect (the apply CSV $APPLY_CSV is the reversal record)"
  exit 1
fi
