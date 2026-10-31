#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Clean up the stale 'scheduled' soccer fixture rows left by the
# fixture-identity bug (pre-2026-09-04 kickoff-time forks) and the 2026 World
# Cup knockout placeholders, with scripts/repair_fixture_orphans.py.
#
# RUN FIRST (its own approval; it now voids recs written after kickoff):
#   docker compose exec -T api python /app/scripts/repair_stranded_recs.py --sport soccer --apply
# It settles/voids the pending recs on orphans that have a twin and re-points
# them onto the twin. Replica of prod 2026-10-31: 15 pending soccer recs ->
# 7 voided (written after the twin's kickoff) + re-pointed, 8 left pending on
# rows with no twin. Until it has run, step 2 aborts (blocked_pending_recs).
#
# Plan, read-only against a replica of prod 2026-10-31 (449 stale rows) after
# that step:
#   merge_delete        203  pre-game odds/snapshots/predictions moved onto the
#                            finished twin (cutoff LEAST(orphan, twin kickoff)),
#                            orphan deleted
#   merge_cancel         28  same merge, but recs reference the orphan (by
#                            match_id or by prediction): status 'cancelled' +
#                            metadata.orphan_of = twin, recs untouched
#   delete_placeholder   65  World Cup bracket slots / team-vs-itself rows, no rec
#   left alone          153  no twin: replayed_later 65, other_identity 42
#                            (football-data / renamed-club rows), never_resulted
#                            43, flipped 3 (33 of them pinned by recs)
#   1. verified backup (B2) — abort if it does not land;
#   2. before-state + dry run (write-once audit CSV) — abort if the tool fails,
#      a pending rec blocks it, or the plan exceeds the reviewed set
#      (merges <= 231, placeholders <= 65, no no-twin deletes);
#   3. apply, pinned to the dry run's counts (--expect-plan: any drift refuses
#      before a write); the archive (gzip JSONL) lands before the first write;
#      batches commit separately, every write guarded on the observed values;
#   4. after-state, EXPECTED: stale rows fall by exactly merges + placeholders,
#      orphan_of rows rise by exactly merge_cancel, the recs on past fixtures
#      are all still there (the rec generators only ever delete pending recs on
#      UPCOMING fixtures), and a fresh dry run plans nothing;
#   5. recompute features for the twins that received prices;
#   6. grade the predictions moved onto those twins (predictions only — no rec
#      is written by any step of this script).
# Each tool step writes its output to a log file and exits with the tool's own
# status; a pipe through tail would report tail's 0.
# Re-runnable: merged rows are gone, cancelled rows leave the candidate set.
#
# AFTERWARDS (separate approval): the 43 never_resulted rows are real matches
# ESPN has full-time results for (39 in these leagues); a results backfill
# heals and settles them and their recs:
#   docker compose exec -T api python /app/scripts/fetch_upcoming.py --sport soccer --results --days-back 160 \
#     --leagues chi.1,jpn.1,col.1,conmebol.libertadores,uefa.champions,bel.1
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/cleanup_soccer_orphans.sh 2>&1 | tee -a ~/cleanup_soccer_orphans.log
set -uo pipefail
cd /opt/auspex
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
TOOL="python /app/scripts/repair_fixture_orphans.py --sport soccer --delete-placeholders"
# The reviewed set (replica of prod 2026-10-31). repair_stranded_recs can move
# a row between merge_cancel and merge_delete, so the merge ceiling is shared.
MAX_MERGES=231
MAX_PLACEHOLDERS=65

START=$(date -u +%FT%TZ)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
APPLY_CSV=/app/backups/fixture_orphans_soccer_apply_$STAMP.csv
# One fixed cutoff for before and after, so the arithmetic is exact.
cat > /tmp/soccer_orphans_state.sql <<'SQLEOF'
SELECT (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'soccer' AND m.status = 'scheduled'
           AND m.match_date < TIMESTAMPTZ '__START__' - interval '96 hours') AS stale,
       (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'soccer' AND m.status = 'cancelled' AND m.metadata ? 'orphan_of') AS orphan_of,
       (SELECT count(*) FROM betting_recommendations br JOIN matches m ON m.id = br.match_id
         WHERE br.created_at < TIMESTAMPTZ '__START__'
           AND m.match_date < TIMESTAMPTZ '__START__' - interval '4 days') AS recs;
SQLEOF
sed -i "s/__START__/$START/g" /tmp/soccer_orphans_state.sql

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

log "2/6 before-state + dry run"
state=$(psql_file /tmp/soccer_orphans_state.sql -At | tr -d '[:space:]')
IFS='|' read -r stale0 orphan0 recs0 <<< "$state"
if ! [[ "$stale0" =~ ^[0-9]+$ && "$orphan0" =~ ^[0-9]+$ && "$recs0" =~ ^[0-9]+$ ]]; then
  log "ABORT at step 2: the before-state query failed (got '$state') — nothing changed"; exit 1
fi
log "before: stale soccer rows=$stale0, cancelled orphan_of rows=$orphan0, recs on past fixtures=$recs0"
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/fixture_orphans_soccer_dryrun_$STAMP.csv > /tmp/rfo_soccer_dry.log 2>&1; rc=\$?; grep -E 'candidates=| PLAN |PENDING|ERROR|Traceback|Error' /tmp/rfo_soccer_dry.log | tail -12; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 2 exit $rc"
if [ "$rc" -ne 0 ]; then log "ABORT at step 2: the dry run failed (see above; full log /tmp/rfo_soccer_dry.log in the api container) — nothing changed"; exit 1; fi
plan=$($DC exec -T api sh -c "grep -o 'PLAN .*' /tmp/rfo_soccer_dry.log | tail -1")
md=$(plan_num "$plan" merge_delete); mc=$(plan_num "$plan" merge_cancel)
dp=$(plan_num "$plan" delete_placeholder); dn=$(plan_num "$plan" delete_no_twin)
blocked=$(plan_num "$plan" blocked_pending_recs)
if ! [[ "$md $mc $dp $dn $blocked" =~ ^[0-9]+\ [0-9]+\ [0-9]+\ [0-9]+\ [0-9]+$ ]]; then
  log "ABORT at step 2: could not read the PLAN line (got '$plan') — nothing changed"; exit 1
fi
if [ "$blocked" -ne 0 ]; then
  log "ABORT at step 2: $blocked orphan(s) with a twin still hold pending recs — run repair_stranded_recs.py --sport soccer --apply first (see header) — nothing changed"; exit 1
fi
if [ "$dn" -ne 0 ] || [ $((md + mc)) -gt "$MAX_MERGES" ] || [ "$dp" -gt "$MAX_PLACEHOLDERS" ]; then
  log "ABORT at step 2: plan ($plan) is outside the reviewed set (merges <= $MAX_MERGES, placeholders <= $MAX_PLACEHOLDERS, no no-twin deletes) — review the dry-run CSV; nothing changed"; exit 1
fi
log "plan: merge_delete=$md merge_cancel=$mc delete_placeholder=$dp"

log "3/6 apply (pinned to the dry run's counts)"
$DC exec -T api sh -c "$TOOL --apply --expect-plan merge_delete=$md,merge_cancel=$mc,delete_placeholder=$dp,delete_no_twin=0 --out-csv $APPLY_CSV > /tmp/rfo_soccer_apply.log 2>&1; rc=\$?; grep -E 'candidates=| PLAN |archived|DONE|ERROR|Traceback|Error' /tmp/rfo_soccer_apply.log | tail -15; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 3 exit $rc"
if [ "$rc" -ne 0 ]; then
  # Refused before any write (plan drift, pending recs, an argument), or after
  # some committed batches when a row changed under its lock (it says so).
  log "ABORT at step 3 (see above; full log /tmp/rfo_soccer_apply.log in the api container): if it refused or raised before 'archived', nothing changed; otherwise the committed batches stand and the archive holds every planned row — inspect before re-running"
  exit 1
fi

log "4/6 after-state"
state=$(psql_file /tmp/soccer_orphans_state.sql -At | tr -d '[:space:]')
IFS='|' read -r stale1 orphan1 recs1 <<< "$state"
want_stale=$((stale0 - md - mc - dp)); want_orphan=$((orphan0 + mc))
log "after: stale soccer rows=$stale1 (expected $want_stale), cancelled orphan_of rows=$orphan1 (expected $want_orphan), recs on past fixtures=$recs1 (expected $recs0)"
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/fixture_orphans_soccer_verify_$STAMP.csv > /tmp/rfo_soccer_verify.log 2>&1; rc=\$?; grep -E 'candidates=| PLAN ' /tmp/rfo_soccer_verify.log; exit \$rc"
rc=$?
replan=$($DC exec -T api sh -c "grep -o 'PLAN .*' /tmp/rfo_soccer_verify.log | tail -1")
replanned="$(plan_num "$replan" merge_delete) $(plan_num "$replan" merge_cancel) $(plan_num "$replan" delete_placeholder)"
if [ "$rc" -ne 0 ] || [ "$stale1" != "$want_stale" ] || [ "$orphan1" != "$want_orphan" ] || [ "$recs1" != "$recs0" ] || [ "$replanned" != "0 0 0" ]; then
  log "FAILED after-state: stale $stale1/$want_stale, orphan_of $orphan1/$want_orphan, recs $recs1/$recs0, re-plan '$replan' (exit $rc) — inspect (the archive and audit CSVs are in /app/backups)"
  exit 1
fi
log "after-state OK"

log "5/6 recompute features for the twins that received prices"
# The twins come from this run's write-once apply CSV.
TWINS_PY="import csv, sys; print(','.join(sorted({r['twin_id'] for r in csv.DictReader(open(sys.argv[1])) if r['action'] in ('merge_delete', 'merge_cancel')})))"
twins=$($DC exec -T api python -c "$TWINS_PY" "$APPLY_CSV")
rc=$?
if [ "$rc" -ne 0 ] || ! [[ "$twins" =~ ^[0-9a-f,-]+$ ]]; then
  log "WARNING step 5: could not read the twins from $APPLY_CSV (exit $rc) — steps 5-6 skipped; the cleanup itself is verified, run them by hand"
else
  log "$(echo "$twins" | tr ',' '\n' | wc -l) twins"
  $DC exec -T api python /app/scripts/compute_features.py --match-ids "$twins" > /tmp/rfo_soccer_features.log 2>&1
  rc=$?
  tail -3 /tmp/rfo_soccer_features.log
  if [ "$rc" -ne 0 ]; then log "WARNING step 5 exit $rc: features not recomputed (full log /tmp/rfo_soccer_features.log); the cleanup itself is verified — re-run this step by hand"; fi

  log "6/6 grade the predictions moved onto those twins"
  $DC exec -T -e TWINS="$twins" api python - > /tmp/rfo_soccer_grade.log 2>&1 <<'PY'
import os, sys
sys.path.insert(0, "/app/scripts")
import psycopg2
from psycopg2.extras import RealDictCursor
import grade_completed_matches as g

# Predictions only (grade_match's own prediction loop): recs on the twins
# stay the regular grader's business.
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
  if [ "$rc" -ne 0 ]; then log "WARNING step 6 exit $rc: grading failed (full log /tmp/rfo_soccer_grade.log); the cleanup itself is verified — re-run this step by hand"; fi
fi

rm -f /tmp/soccer_orphans_state.sql
$DC exec -T postgres rm -f /tmp/soccer_orphans_q.sql >/dev/null 2>&1
log "DONE. Next (separate approval): the never_resulted rows — see the results backfill command in this script's header."
