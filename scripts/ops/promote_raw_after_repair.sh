#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# The first promote_raw run after the soccer repair, in checked mode
# (scripts/promote_raw.py --dry-run / --apply --expect-plan: one transaction,
# what it changed read back and pinned). Runbook step 14. promote_raw was
# blocked from 2026-10-31 (check_legacy_kickoffs) until
# repair_football_data_kickoffs.sh (runbook step 4) had run; that repair
# archived, for the 4 Liga MX fixtures it alias-merged, the match_stats of
# the club the re-point map moves onto them (Guadalajara Chivas / Santos
# Laguna), and promote_raw writes them again once repoint_team_matches.sh
# --pass 2 (step 5) has moved those rows (until then it refuses them).
#
# Rehearsed 2026-11-01 on a local replica of a read-only prod export taken
# 02:42Z, after the runbook's steps 1-12 ran on it through a docker shim
# (168,077 raw rows, ~100 s a run there): every one of the 168,073 promotable
# rows is already on its matches row with the same result (promote_raw no
# longer rewrites those), and the run inserts exactly
#   match_stats_inserted=4  the 4 alias-merged fixtures' second side (Puebla v
#                           Guadalajara Chivas 2026-08-01, Guadalajara Chivas v
#                           Toluca and Monterrey v Santos Laguna 2026-07-19,
#                           Santos Laguna v Atlas 2026-07-26; MX1 rows carry
#                           no shot or card columns, so the rows are empty
#                           apart from their keys, as the archived ones were)
#   odds_inserted=13        football-data's closing prices for Aberdeen v
#                           Hearts 2026-08-01, an ESPN row that holds none
# and nothing else: no match, team or league inserted or changed, no
# prediction graded, no rec settled, no raw row refused or duplicated.
# Again on 06:48:46Z and 08:24:28Z exports after steps 1-13 (the labels
# repair, step 13, included): the same plan, refused_raw=0 (the PLAN key
# since 2026-11-01: raw rows promote_raw's screen refuses); on both, the
# same dry run before step 13 screens out the 68 mis-decoded Preussen
# Muenster rows (refused_raw=68) and exits 3, which step 0 here aborts on
# before any backup.
# The checked run is one REPEATABLE READ transaction that sends the database
# only the rows it changes (promote_raw's docstring): it neither locks the
# league and team rows the pipeline's soccer sweeps rewrite every 15 min nor
# fails on them. A serialization error can only come from another session
# changing a row this run changes, meanwhile: exit 1, nothing written —
# re-run (the runbook: --from 14). Checked on the replica of a 05:15Z export
# (the same plan): a whole checked dry run beside 55 rounds of the
# pipeline's ensure_league on all 41 football-data leagues, a team stamp
# and a match_stats touch (3 s lock timeout each) — none waited, promote
# exited 0; promote_raw as it was before (e6fc14a) failed with a
# serialization error in its first seconds.
#   0. already done? A read-only checked dry run: every PLAN key 0 exits 0
#      without a backup; any non-zero exit aborts here, before a backup (1: a
#      legacy kickoff left, step 4 has not run, or a refusal; 3: a raw row
#      refused or duplicated, its ERROR lines printed);
#   1. verified backup (B2) — abort if it does not land;
#   2. checked dry run (write-once CSV of every change) — abort unless the
#      plan is within the reviewed set: match_stats_inserted <= 4,
#      odds_inserted <= 13, every other key 0;
#   3. apply, pinned to the dry run's counts (--expect-plan, every key): one
#      transaction, committed only when its own plan equals the pin (exit 1:
#      rolled back, nothing changed; exit 3 would mean committed at a pin
#      expecting refused rows, which step 2 rules out — reported as FAILED,
#      committed, inspect);
#   4. after-state, EXPECTED: a fresh dry run plans 0 for every key, and each
#      fixture the fd repair alias-merged (its newest apply CSV listing any)
#      holds a match_stats row for both its sides.
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status; a pipe through tail would report tail's 0.
#
# RUN ORDER (scripts/ops/soccer_repair_runbook.sh runs it, each step through
# its own wrapper): 1 repoint_team_matches.sh --pass 1, 2 migrate_027_outcome_trigger.sh,
# 3 merge_soccer_teams.sh, 4 repair_football_data_kickoffs.sh,
# 5 repoint_team_matches.sh --pass 2, 6 repair_stranded_recs_soccer.sh,
# 7 cleanup_soccer_orphans.sh, 8 backfill_soccer_regulation_scores.sh,
# 9 backfill_season_labels.sh, 10 compute_features --backfill-finished soccer,
# 11 the merge's odds check, 12 backfill_soccer_never_resulted.sh (+ the
# cleanup again), 13 repair_soccer_labels.sh, 14 THIS,
# 15 grade_folded_soccer_rows.sh, 16 compute_features --backfill-finished soccer.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/promote_raw_after_repair.sh 2>&1 | tee -a ~/promote_raw_after_repair.log
# (AUSPEX_DIR and TMPDIR exist for tests and rehearsals.)
set -uo pipefail
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
W=$(mktemp -d "${TMPDIR:-/tmp}/promote_raw_after.XXXXXX") || { echo "ABORT: mktemp failed — nothing changed"; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/promote_raw_after_q.sql >/dev/null 2>&1
}
trap cleanup EXIT
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_at() {  # psql_at <local.sql>: unaligned, '|'-separated
  $DC cp "$1" postgres:/tmp/promote_raw_after_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/promote_raw_after_q.sql'
}
plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o " $2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
KEYS="matches_inserted matches_changed match_stats_inserted match_stats_changed odds_inserted teams_inserted teams_changed leagues_inserted leagues_changed predictions_changed recs_changed refused duplicates refused_raw"
# The reviewed set (rehearsal of 2026-11-01, see the header): these may be up to their bound, every other key 0.
MAX_MATCH_STATS_INSERTED=4
MAX_ODDS_INSERTED=13
TOOL="python /app/scripts/promote_raw.py"
KEEP="Upserted|Checked run|PLAN|DRY-RUN|Committed|REFUSED|Refusing|ERROR|Traceback|Error"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
B=/app/backups/promote_raw_after_repair
step() {  # step <log name> <args...>: the tool in the api container; exits with its status
  local name="$1"; shift
  $DC exec -T api sh -c "$TOOL $* > /tmp/$name.log 2>&1; rc=\$?; grep -E '$KEEP' /tmp/$name.log | grep -v ' INFO - Raw team ' | tail -20; echo \"tool exit \$rc\"; exit \$rc"
}
plan_of() { $DC exec -T api sh -c "grep -o 'PLAN .*' /tmp/$1.log | tail -1"; }
values() {  # values <PLAN line>: every key's count, space-separated, in KEYS order
  local k out=""
  for k in $KEYS; do out="$out $(plan_num "$1" "$k")"; done
  echo "${out# }"
}
ZERO=$(for _ in $KEYS; do printf '0 '; done | sed 's/ $//')

log "0/4 already done? (read-only checked dry run)"
step promote_after_pre --dry-run --out-csv ${B}_precheck_$STAMP.csv
rc=$?
pre=$(plan_of promote_after_pre)
case "$rc" in
  0) ;;
  1) log "ABORT at step 0: promote_raw refused (a legacy kickoff is left — runbook step 4 first — or the refusal above; full log /tmp/promote_after_pre.log in the api container) — nothing changed"; exit 1 ;;
  3) log "ABORT at step 0: promote_raw refused, found duplicated or screened out raw row(s) (the ERROR lines above; $pre) — outside the reviewed set (refused = duplicates = refused_raw = 0; screened-out mis-decoded rows mean scripts/ops/repair_soccer_labels.sh, runbook step 13, has not run); nothing changed"; exit 1 ;;
  *) log "ABORT at step 0: the checked dry run failed (exit $rc; full log /tmp/promote_after_pre.log in the api container) — nothing changed"; exit 1 ;;
esac
if ! [[ "$(values "$pre")" =~ ^[0-9]+(\ [0-9]+){13}$ ]]; then
  log "ABORT at step 0: the checked dry run printed no readable PLAN line ('${pre:-unreadable}') — nothing changed"; exit 1
fi
if [ "$(values "$pre")" = "$ZERO" ]; then
  log "ALREADY DONE: $pre — nothing to promote (no backup)"
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

log "2/4 checked dry run"
step promote_after_dry --dry-run --out-csv ${B}_dryrun_$STAMP.csv
rc=$?
plan=$(plan_of promote_after_dry)
vals=$(values "$plan")
if [ "$rc" -ne 0 ] || ! [[ "$vals" =~ ^[0-9]+(\ [0-9]+){13}$ ]]; then
  log "ABORT at step 2: the dry run failed or printed no readable PLAN line (exit $rc, '$plan'; full log /tmp/promote_after_dry.log in the api container) — nothing changed"
  exit 1
fi
others=""
for k in $KEYS; do
  case "$k" in
    match_stats_inserted | odds_inserted) ;;
    *) others="$others $(plan_num "$plan" "$k")" ;;
  esac
done
others=${others# }
if [ "$(plan_num "$plan" match_stats_inserted)" -gt "$MAX_MATCH_STATS_INSERTED" ] || [ "$(plan_num "$plan" odds_inserted)" -gt "$MAX_ODDS_INSERTED" ] \
   || [ "$others" != "0 0 0 0 0 0 0 0 0 0 0 0" ]; then
  log "ABORT at step 2: plan ($plan) is outside the reviewed set (match_stats_inserted <= $MAX_MATCH_STATS_INSERTED, odds_inserted <= $MAX_ODDS_INSERTED, every other key 0) — review ${B}_dryrun_$STAMP.csv; nothing changed"
  exit 1
fi
if [ "$vals" = "$ZERO" ]; then log "DONE: nothing to promote ($plan) — no write"; exit 0; fi
pin=$(for k in $KEYS; do printf '%s=%s,' "$k" "$(plan_num "$plan" "$k")"; done | sed 's/,$//')
log "plan: $plan"

log "3/4 apply (pinned: $pin)"
step promote_after_apply --apply --expect-plan "$pin" --out-csv ${B}_apply_$STAMP.csv
rc=$?
log "step 3 exit $rc"
case "$rc" in
  0) ;;
  3)
    # A checked --apply exits 3 only AFTER its commit, at a pin that expected refused, duplicated or screened-out
    # rows (promote_raw's EXIT STATUS). Step 2 pins all three to 0, so this cannot happen as written; if it does,
    # the change is committed and nothing here may call it rolled back.
    log "FAILED at step 3 (full log /tmp/promote_after_apply.log in the api container): promote_raw exited 3 AFTER COMMITTING — its pin expected refused, duplicated or screened-out raw rows, which step 2's bounds (refused = duplicates = refused_raw = 0) rule out; the change IS committed — inspect (every change is in ${B}_apply_$STAMP.csv)"
    exit 1
    ;;
  *)
    log "ABORT at step 3 (full log /tmp/promote_after_apply.log in the api container): the checked apply is one transaction — a refusal or a drift rolled it back, nothing changed"
    exit 1
    ;;
esac

log "4/4 after-state"
step promote_after_verify --dry-run --out-csv ${B}_verify_$STAMP.csv
vrc=$?
vplan=$(plan_of promote_after_verify)
# The fixtures the fd repair alias-merged: its newest apply CSV that lists any.
ALIASED_PY="import csv, glob, sys
for f in sorted(glob.glob('/app/backups/football_data_dedupe_apply_*.csv'), reverse=True):
    ids = [r['espn_row_id'] for r in csv.DictReader(open(f)) if r['action'] == 'alias_merge']
    if ids:
        print(' '.join(ids))
        break"
aliased=$($DC exec -T api python -c "$ALIASED_PY" </dev/null | tail -1)
sides="no alias-merged fixture listed"
if [[ "$aliased" =~ ^[0-9a-f-]{36}(\ [0-9a-f-]{36})*$ ]]; then
  arr="ARRAY['$(echo "$aliased" | sed "s/ /','/g")']::uuid[]"
  cat > "$W/promote_raw_after_sides.sql" <<SQLEOF
SELECT count(*) FILTER (WHERE n = 2) || '/' || count(*)
  FROM (SELECT m.id, (SELECT count(*) FROM match_stats s
                       WHERE s.match_id = m.id AND s.team_id IN (m.home_team_id, m.away_team_id)) AS n
          FROM matches m WHERE m.id = ANY($arr)) t;
SQLEOF
  sides=$(psql_at "$W/promote_raw_after_sides.sql" | tr -d '[:space:]')
fi
log "after: re-plan '$vplan' (exit $vrc, expected every key 0); alias-merged fixtures with both sides' match_stats: $sides"
if [ "$vrc" -ne 0 ] || [ "$(values "$vplan")" != "$ZERO" ]; then
  log "FAILED after-state: a fresh dry run still plans changes — inspect (the apply CSV ${B}_apply_$STAMP.csv lists every change)"
  exit 1
fi
case "$sides" in
  "no alias-merged fixture listed") ;;
  */*) [ "${sides%/*}" = "${sides#*/}" ] || { log "FAILED after-state: only $sides alias-merged fixtures hold both sides' match_stats — inspect"; exit 1; } ;;
  *) log "FAILED after-state: could not count the alias-merged fixtures' match_stats (got '$sides') — inspect"; exit 1 ;;
esac
log "DONE: promote_raw ran checked ($plan); a fresh dry run plans nothing"
