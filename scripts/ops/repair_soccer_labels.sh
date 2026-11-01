#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Repair the soccer rows football-data's files filed under the wrong league or
# decoded wrongly, with scripts/repair_soccer_labels.py (its docstring has the
# investigation): ten Scottish Championship / League One fixtures of
# 2026-08-01 stored as La Liga / La Liga 2 (and the raw rows they came from),
# the twenty clubs on them that carry a Spanish league, and 'PreuÃ\x9fen
# MÃ¼nster' (one team, 68 raw rows) renamed 'Preußen Münster' — no
# correctly spelled twin exists, so a rename, not a merge.
# DEPLOY THE COMMIT THAT SHIPS scripts/text_encoding.py FIRST: until then both
# football-data loaders read UTF-8 files as latin-1 and skip the Div check, and
# promote_raw promotes such rows.
#
# Reviewed plan: this tool's dry run on prod 2026-11-01 03:27Z (read only:
# default_transaction_read_only=on), and a local replica built from the
# migrations and a read-only export (03:02Z, one REPEATABLE READ snapshot),
# both as exported and after the runbook's step 4 moves these rows' kickoffs
# (an hour earlier: 15:00Z -> 14:00Z on 2026-08-01) — the same counts each time:
#   create_league  2   SC1 Championship, SC2 League One (Scotland), created as
#                      promote_raw creates every football-data league
#   relabel_raw   10   raw_football_data SP1/SP2 2627 -> SC1/SC2 (league, Div)
#   relabel_match 10   matches.league_id La Liga / La Liga 2 -> SC1 / SC2
#   relabel_team  20   teams.league_id Spain -> SC1 / SC2 (14 clubs with only
#                      that match; Dunfermline, Inverness C, Livingston,
#                      Partick, Hamilton, Ross County with 38-448 Premiership
#                      matches each)
#   rename_team    1   b6abfa7e 'PreuÃ\x9fen MÃ¼nster' -> 'Preußen Münster'
#   rename_raw    68   raw D2 2024-25 and 2025-26 rows (34 with Preussen at
#                      home, 34 away)
#   refuse 0, mojibake_left 0; cross_country_after 2 (Guadalajara and Santos,
#   whose Liga MX rows the runbook's repoint passes move; 0 after them).
# The soccer repair runbook (scripts/ops/soccer_repair_runbook.sh) runs BEFORE
# this and touches none of it: no merge pair, seed, repoint or postponed row
# names these 21 teams (ids grepped in scripts/data/*.csv); the fd dedupe plan
# folds none of their matches; step 4 shifts their kickoffs, which this finds
# in promote_raw's own window either way; their season labels ('2026-2027',
# cross-year for SP1/SP2 and SC1/SC2 alike) and the season-label and fd-repair
# plans are the same before and after this repair (replayed). The counts can
# only fall before this runs (no football-data load or promote_raw is
# scheduled; ESPN sweeps neither the Scottish lower divisions nor the
# 2. Bundesliga), so the dry run gates on ceilings (each the reviewed count)
# and the apply is pinned to the dry run's exact counts (every key).
#   0. already done? A read-only dry run first: when it plans nothing to write,
#      refuses nothing and finds no mis-decoded value left, this exits 0
#      without a backup (ALREADY DONE); a refusal or a value it cannot repair
#      aborts here, before a backup;
#   1. verified backup (B2) — abort if it does not land;
#   2. before-state + dry run (write-once audit CSV) — abort unless every
#      count is within its ceiling, every map row is accounted for
#      (already + relabel = 10, raw and matches), refuse = 0,
#      mojibake_left = 0 and the cross-country teams are among Guadalajara
#      and Santos;
#   3. apply, pinned to the dry run's counts: ONE transaction (leagues, then
#      the write-once audit CSV and reversal SQL in /app/backups, then every
#      UPDATE guarded on the value it replaces, then a clean re-plan inside
#      the transaction before COMMIT);
#   4. after-state, EXACT: every row of the apply CSV holds its new value
#      (--verify); a fresh dry run plans nothing, refuses nothing and finds
#      all 10 map rows done in raw and matches; read independently of the
#      tool: no mapped Scottish fixture left in La Liga / La Liga 2, no
#      mis-decoded soccer team or raw name, the Scottish leagues up by exactly
#      create_league, and the soccer team, soccer match and raw row counts
#      unchanged (nothing inserted, nothing deleted).
# No DAG is paused: the write is one short transaction on rows no scheduled
# job writes (the 15-minute pipeline never touches football-data rows), and a
# retrain reads either side of it.
# AS A RUNBOOK STEP: exit 0 on success or ALREADY DONE (the line says so),
# non-zero with an ABORT / FAILED line otherwise. It CHANGES THE TRAINING
# CORPUS (matches.league_id feeds the Dixon-Coles / Poisson per-league
# baselines, and the team name keys their teams), so it is a corpus step, but
# it changes no feature (compute_features reads team ids, odds and
# match_stats, never a league or a name): before the runbook's step 10 it
# needs nothing more; after it, the features rebuild it prints is optional
# (it rewrites the same values). Re-runnable: a finished repair plans nothing.
# REVERSAL: the apply writes /app/backups/soccer_labels_apply_<stamp>.reversal.sql
# beside its CSV (one transaction; each statement raises if its row moved).
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/repair_soccer_labels.sh 2>&1 | tee -a ~/repair_soccer_labels.log
# (AUSPEX_DIR and TMPDIR exist for tests and rehearsals.)
set -uo pipefail
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
# shellcheck source=scripts/ops/_airflow_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_airflow_lib.sh" || { log "ABORT: cannot load _airflow_lib.sh — nothing changed"; exit 1; }
# One run at a time (a refused run exits here and touches nothing).
af_lock repair_soccer_labels || exit 1
W=$(mktemp -d "${TMPDIR:-/tmp}/soccer_labels.XXXXXX") || { log "ABORT: mktemp failed — nothing changed"; af_unlock; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/soccer_labels_q.sql >/dev/null 2>&1
  af_unlock
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_at() {  # psql_at <local.sql>: unaligned, '|'-separated
  $DC cp "$1" postgres:/tmp/soccer_labels_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/soccer_labels_q.sql'
}
plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o " $2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
TOOL="python /app/scripts/repair_soccer_labels.py"
KEYS="create_league relabel_raw relabel_match relabel_team rename_team rename_raw already_raw already_match refuse mojibake_left cross_country_after"
WRITE_KEYS="create_league relabel_raw relabel_match relabel_team rename_team rename_raw"
MAP_ROWS=10
# Ceilings = the reviewed counts (prod 2026-11-01 03:02Z); they can only fall.
MAX_create_league=2
MAX_relabel_raw=10
MAX_relabel_match=10
MAX_relabel_team=20
MAX_rename_team=1
MAX_rename_raw=68
ALLOWED_CROSS_COUNTRY="Guadalajara,Santos"
FEATURES_CMD="docker compose exec -T api python /app/scripts/compute_features.py --backfill-finished soccer"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
APPLY_CSV=/app/backups/soccer_labels_apply_$STAMP.csv
# run_tool <log name> <args...>: the tool in the api container; prints its key lines and exits with its status
run_tool() {
  local name="$1"; shift
  $DC exec -T api sh -c "$TOOL $* > /tmp/$name.log 2>&1; rc=\$?; grep -E '^PLAN|^VERIFY|REFUSED|ERROR|Applied|Traceback|Error' /tmp/$name.log | tail -30; echo \"tool exit \$rc\"; exit \$rc"
}
plan_of() { $DC exec -T api sh -c "grep '^PLAN ' /tmp/$1.log | tail -1"; }
cross_of() { $DC exec -T api sh -c "grep '^PLAN_CROSS_COUNTRY ' /tmp/$1.log | tail -1" | sed -n 's/^PLAN_CROSS_COUNTRY names=//p'; }
writes_of() {  # writes_of <PLAN line>: the sum of the write counts, or "missing"
  local n=0 k v
  for k in $WRITE_KEYS; do
    v=$(plan_num "$1" "$k"); [ "$v" = missing ] && { echo missing; return; }
    n=$((n + v))
  done
  echo "$n"
}
within() {  # within <comma list> <comma list>: every item of $1 is in $2 ('-' or empty is empty)
  local item
  if [ -z "$1" ] || [ "$1" = "-" ]; then return 0; fi
  for item in ${1//,/ }; do [[ ",$2," == *",$item,"* ]] || return 1; done
}
features_note() {
  log "features: compute_features reads no league and no team name, so this repair changes no feature value; the"
  log "  soccer repair runbook's step 10 rebuilds them if this ran before it; otherwise, to rebuild after this corpus"
  log "  change anyway: $FEATURES_CMD"
}

# Independent of the tool: Scottish fixtures still in La Liga / La Liga 2 (both teams carry a mapped name),
# mis-decoded soccer team names and raw names (text_encoding.MOJIBAKE_PATTERN), Scottish leagues, and the soccer
# team / soccer match / raw row totals.
cat > "$W/state.sql" <<'SQLEOF'
WITH scots(name) AS (VALUES ('ayr'), ('arbroath'), ('inverness c'), ('dunfermline'), ('livingston'), ('queens park'),
       ('morton'), ('partick'), ('raith rvs'), ('stenhousemuir'), ('alloa'), ('east fife'), ('cove rangers'),
       ('airdrie utd'), ('east kilbride'), ('queen of sth'), ('montrose'), ('hamilton'), ('peterhead'), ('ross county')),
     pat(p) AS (VALUES ('[\u0080-\u009f]|[\u00c2-\u00f4][\u00a0-\u00bf\u20ac\u201a\u0192\u201e\u2026\u2020\u2021\u02c6\u2030\u0160\u2039\u0152\u017d\u2018\u2019\u201c\u201d\u2022\u2013\u2014\u02dc\u2122\u0161\u203a\u0153\u017e\u0178]|\ufeff|\ufffd'))
SELECT (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
          JOIN teams ht ON ht.id = m.home_team_id JOIN teams at ON at.id = m.away_team_id
         WHERE l.sport = 'soccer' AND l.external_ids->>'football_data' IN ('SP1', 'SP2')
           AND ht.normalized_name IN (SELECT name FROM scots) AND at.normalized_name IN (SELECT name FROM scots))
       || '|' || (SELECT count(*) FROM teams, pat WHERE sport = 'soccer' AND (name ~ p OR normalized_name ~ p))
       || '|' || (SELECT count(*) FROM raw_football_data, pat WHERE "HomeTeam" ~ p OR "AwayTeam" ~ p)
       || '|' || (SELECT count(*) FROM leagues WHERE sport = 'soccer' AND country = 'Scotland')
       || '|' || (SELECT count(*) FROM teams WHERE sport = 'soccer')
       || '|' || (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id WHERE l.sport = 'soccer')
       || '|' || (SELECT count(*) FROM raw_football_data);
SQLEOF
read_state() {  # sets misfiled moji_teams moji_raw scot_leagues n_teams n_matches n_raw; 1 when unreadable
  local s
  s=$(psql_at "$W/state.sql" | tr -d '[:space:]')
  IFS='|' read -r misfiled moji_teams moji_raw scot_leagues n_teams n_matches n_raw <<< "$s"
  [[ "$s" =~ ^[0-9]+(\|[0-9]+){6}$ ]]
}

# A run cut short during its apply leaves the apply running in the api container (docker exec does not forward the
# interrupt); a second run would race it.
running=$(af_applies_running repair_soccer_labels)
case $? in
  0)
    log "REFUSED: a repair_soccer_labels.py --apply is still running in the api container (an earlier run ended during it):"
    echo "$running" | sed 's/^/  /'
    log "  wait until it has ended (docker compose top api), then re-run this script — nothing changed"
    exit 1
    ;;
  1) ;;
  *) log "REFUSED: could not read the api container's processes (is the api container up?) — nothing changed"; exit 1 ;;
esac

log "0/4 already done? (read-only dry run)"
run_tool soccer_labels_pre --out-csv /app/backups/soccer_labels_precheck_$STAMP.csv
pre=$(plan_of soccer_labels_pre)
pre_cross=$(cross_of soccer_labels_pre)
if [ "$(writes_of "$pre")" = "missing" ] || [ "$(plan_num "$pre" refuse)" != "0" ] || [ "$(plan_num "$pre" mojibake_left)" != "0" ]; then
  log "ABORT at step 0: the dry run refuses something, finds a mis-decoded value it cannot repair, or its PLAN line is unreadable ('${pre:-none}'; full log /tmp/soccer_labels_pre.log in the api container) — nothing changed"
  exit 1
fi
if [ "$(writes_of "$pre")" = "0" ]; then
  log "ALREADY DONE: $pre — nothing to relabel or rename (no backup)"
  within "$pre_cross" "$ALLOWED_CROSS_COUNTRY" || log "WARNING: soccer teams whose domestic matches span two countries: $pre_cross (beyond $ALLOWED_CROSS_COUNTRY) — another misfiled league or identity split; investigate"
  features_note
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

log "2/4 before-state + dry run"
if ! read_state; then log "ABORT at step 2: the before-state query failed — nothing changed"; exit 1; fi
misfiled0=$misfiled; moji_teams0=$moji_teams; moji_raw0=$moji_raw; scot0=$scot_leagues
n_teams0=$n_teams; n_matches0=$n_matches; n_raw0=$n_raw
log "before: mapped Scottish fixtures in La Liga / La Liga 2 $misfiled0, mis-decoded soccer team names $moji_teams0, mis-decoded raw rows $moji_raw0, Scottish leagues $scot0; soccer teams $n_teams0, soccer matches $n_matches0, raw rows $n_raw0"
run_tool soccer_labels_dry --out-csv /app/backups/soccer_labels_dryrun_$STAMP.csv
rc=$?
plan=$(plan_of soccer_labels_dry)
cross=$(cross_of soccer_labels_dry)
for k in $KEYS; do eval "p_$k=\$(plan_num \"\$plan\" $k)"; done
# shellcheck disable=SC2154  # p_* are set by the eval above
counts="$p_create_league $p_relabel_raw $p_relabel_match $p_relabel_team $p_rename_team $p_rename_raw $p_already_raw $p_already_match $p_refuse $p_mojibake_left $p_cross_country_after"
if [ "$rc" -ne 0 ] || ! [[ "$counts" =~ ^[0-9]+(\ [0-9]+){10}$ ]]; then
  log "ABORT at step 2: the dry run failed (exit $rc) or its PLAN line is unreadable ('$plan') — nothing changed"; exit 1
fi
over=""
for k in $WRITE_KEYS; do
  eval "v=\$p_$k; m=\$MAX_$k"
  # shellcheck disable=SC2154  # v and m are set by the eval above
  [ "$v" -le "$m" ] || over="$over $k=$v>$m"
done
# shellcheck disable=SC2154
if [ -n "$over" ] || [ "$p_refuse" -ne 0 ] || [ "$p_mojibake_left" -ne 0 ] \
   || [ $((p_already_raw + p_relabel_raw)) -ne "$MAP_ROWS" ] || [ $((p_already_match + p_relabel_match)) -ne "$MAP_ROWS" ] \
   || ! within "$cross" "$ALLOWED_CROSS_COUNTRY"; then
  log "ABORT at step 2: plan ($plan; cross-country: ${cross:-unreadable}) is outside the reviewed set (ceilings${over:+, over:$over}; refuse=0, mojibake_left=0, already+relabel=$MAP_ROWS for raw and matches, cross-country teams among $ALLOWED_CROSS_COUNTRY) — review the dry-run CSV; nothing changed"
  exit 1
fi
pin=""
for k in $KEYS; do pin="$pin${pin:+,}$k=$(plan_num "$plan" "$k")"; done
log "plan: $plan (cross-country: $cross)"

log "3/4 apply (pinned: $pin)"
run_tool soccer_labels_apply --apply --expect-plan "$pin" --out-csv "$APPLY_CSV"
rc=$?
if [ "$rc" -ne 0 ]; then
  log "ABORT at step 3 (exit $rc; full log /tmp/soccer_labels_apply.log in the api container): a plan that differs from the dry run, or a row that moved since it, rolls the one transaction back — nothing was written"
  exit 1
fi

log "4/4 after-state"
run_tool soccer_labels_verify --verify "$APPLY_CSV"
vrc=$?
verify=$($DC exec -T api sh -c "grep '^VERIFY ' /tmp/soccer_labels_verify.log | tail -1")
run_tool soccer_labels_replan --out-csv /app/backups/soccer_labels_verify_$STAMP.csv
prc=$?
replan=$(plan_of soccer_labels_replan)
if ! read_state; then log "FAILED after-state: the after-state query failed — inspect (the apply CSV $APPLY_CSV and its .reversal.sql are the record)"; exit 1; fi
want_scot=$((scot0 + p_create_league))
log "after: verify '$verify' (expected now_new = rows), re-plan '$replan' (expected 0 to write, refuse 0, already_raw=$MAP_ROWS already_match=$MAP_ROWS), mapped Scottish fixtures in La Liga / La Liga 2 $misfiled (expected 0), mis-decoded soccer team names $moji_teams and raw rows $moji_raw (expected 0 and 0), Scottish leagues $scot_leagues (expected $want_scot), soccer teams $n_teams / soccer matches $n_matches / raw rows $n_raw (expected $n_teams0 / $n_matches0 / $n_raw0)"
if [ "$vrc" -eq 0 ] && [ "$(plan_num "$verify" rows)" != missing ] && [ "$(plan_num "$verify" now_new)" = "$(plan_num "$verify" rows)" ] \
   && [ "$prc" -eq 0 ] && [ "$(writes_of "$replan")" = "0" ] && [ "$(plan_num "$replan" refuse)" = "0" ] \
   && [ "$(plan_num "$replan" mojibake_left)" = "0" ] && [ "$(plan_num "$replan" already_raw)" = "$MAP_ROWS" ] \
   && [ "$(plan_num "$replan" already_match)" = "$MAP_ROWS" ] && [ "$misfiled" = "0" ] && [ "$moji_teams" = "0" ] \
   && [ "$moji_raw" = "0" ] && [ "$scot_leagues" = "$want_scot" ] && [ "$n_teams" = "$n_teams0" ] \
   && [ "$n_matches" = "$n_matches0" ] && [ "$n_raw" = "$n_raw0" ]; then
  log "DONE: $p_relabel_match fixture(s) and $p_relabel_raw raw row(s) relabelled, $p_relabel_team team(s) moved to their Scottish league, $p_rename_team team and $p_rename_raw raw row(s) renamed, $p_create_league league(s) created (reversal: ${APPLY_CSV%.csv}.reversal.sql)"
  features_note
else
  log "FAILED after-state — inspect (the apply CSV $APPLY_CSV is the record; ${APPLY_CSV%.csv}.reversal.sql reverts it)"
  exit 1
fi
