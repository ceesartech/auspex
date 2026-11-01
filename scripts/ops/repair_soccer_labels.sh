#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Repair the soccer rows football-data's files filed under the wrong league or
# decoded wrongly, with scripts/repair_soccer_labels.py (its docstring has the
# investigation): ten Scottish Championship / League One fixtures of
# 2026-08-01 stored as La Liga / La Liga 2 (and the raw rows they came from),
# the twenty clubs on them that carry a Spanish league, and 'PreuÃ\x9fen
# MÃ¼nster' (one team, 68 raw rows) renamed 'Preußen Münster' — no
# correctly spelled twin exists, so a rename, not a merge. Runbook step 13
# (scripts/ops/soccer_repair_runbook.sh: after steps 1-12, before its
# promote_raw step 14 and its final features recompute 16).
# DEPLOY THE COMMIT THAT SHIPS scripts/text_encoding.py FIRST: until then both
# football-data loaders read UTF-8 files as latin-1 and skip the Div check, and
# promote_raw promotes such rows.
#
# Reviewed plan: this tool's dry run on prod 2026-11-01 03:27Z and again at
# 04:22Z after the review fixes (read only: default_transaction_read_only=on),
# and local replicas built from the migrations and read-only exports (03:02Z
# and 04:23Z, each one REPEATABLE READ snapshot), both as exported and after
# the runbook's step 4 moves these rows' kickoffs (an hour earlier: 15:00Z ->
# 14:00Z on 2026-08-01), and as the soccer repair runbook's step 13 on a
# replica of a 06:48:46Z export after its steps 1-12, through the docker shim
# (and again on an 08:24:28Z export)
# (cross_country_after 0 there; VERIFY rows=122 now_new=122; a re-run after
# step 14's promote_raw still ALREADY DONE) — the same counts each time:
#   create_league  2   SC1 Championship, SC2 League One (Scotland): the row
#                      promote_raw.upsert_leagues inserts for a code
#   tag_league     0   (a code-less row of that name and country — one ESPN
#                      made — gets the code instead of a second row; prod has
#                      none: Scotland holds only SC0. create + tag <= 2.)
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
# The soccer repair runbook's steps 1-12 (scripts/ops/soccer_repair_runbook.sh)
# run BEFORE this and touch none of it: no merge pair, seed, repoint or postponed row
# names these 21 teams (ids grepped in scripts/data/*.csv); the fd dedupe plan
# folds none of their matches; step 4 shifts their kickoffs, which this finds
# in promote_raw's own window either way; their season labels ('2026-2027',
# cross-year for SP1/SP2 and SC1/SC2 alike) and the season-label and fd-repair
# plans are the same before and after this repair (replayed). The counts can
# only fall before this runs (no football-data load or promote_raw is
# scheduled; ESPN sweeps neither the Scottish lower divisions nor the
# 2. Bundesliga), so the dry run gates on ceilings (each the reviewed count;
# tag_league's ceiling is the two leagues, shared with create_league: a
# league ESPN creates under that name moves one from create to tag) and the
# apply is pinned to the dry run's exact counts (every key).
#   0. already done? A read-only dry run first: when it plans nothing to write,
#      refuses nothing and finds no mis-decoded value left, and the after-state
#      read independently of the tool agrees (no mapped fixture in La Liga /
#      La Liga 2, no mis-decoded soccer team or raw name), this exits 0
#      without a backup (ALREADY DONE) — after first verifying (--verify) the
#      apply of an earlier run that ended before its step 4 (Ctrl-C or an SSH
#      drop during step 3 leaves the apply running in the container, and it
#      may commit; each run records its apply CSV in
#      /app/backups/soccer_labels_apply_pending.txt before applying and clears
#      it once step 4 passes); a refusal, a value it cannot repair, an
#      independent read that disagrees, or a pending apply that does not
#      verify aborts here, before a backup;
#   1. verified backup (B2) — abort if it does not land;
#   2. before-state + dry run (write-once audit CSV) — abort unless every
#      count is within its ceiling, every map row is accounted for
#      (already + relabel = 10, raw and matches), refuse = 0,
#      mojibake_left = 0 and the cross-country teams are among Guadalajara
#      and Santos;
#   3. apply, pinned to the dry run's counts: ONE transaction (leagues
#      created or tagged, then the write-once audit CSV and reversal SQL in
#      /app/backups, then every UPDATE guarded on the value it replaces, then
#      a clean re-plan inside the transaction before COMMIT; on a rollback
#      the CSV and SQL are renamed *.void);
#   4. after-state, EXACT: every row of the apply CSV holds its new value
#      (--verify); a fresh dry run plans nothing, refuses nothing and finds
#      all 10 map rows done in raw and matches; read independently of the
#      tool: no mapped Scottish fixture left in La Liga / La Liga 2, no
#      mis-decoded soccer team or raw name, the Scottish leagues up by exactly
#      create_league (a tagged league is not a new row), and the soccer team,
#      soccer match and raw row counts unchanged (nothing inserted, nothing
#      deleted).
# No DAG is paused: the write is one short transaction on rows no scheduled
# job writes (the 15-minute pipeline never touches football-data rows), and a
# retrain reads either side of it.
# AS A RUNBOOK STEP (13): exit 0 on success or ALREADY DONE (the line says so),
# non-zero with an ABORT / FAILED line otherwise. It CHANGES THE TRAINING
# CORPUS (matches.league_id feeds the Dixon-Coles / Poisson per-league
# baselines, and the team name keys their teams), so it is a corpus step:
# the runbook records it 'started' before it runs and keeps the retrain DAGs
# paused until its step 16 has recomputed the features after it, though it
# changes no feature value (compute_features reads team ids, odds and
# match_stats, never a league or a name). It must run before the runbook's
# promote_raw step (14): until it has, promote_raw commits everything else
# and exits 3 on the 68 mis-decoded raw rows. promote_raw keeps an existing
# team's league since 2026-11-01, so a promote_raw run after this leaves the
# twenty clubs on SC1 / SC2 and a re-run of this still reads ALREADY DONE.
# Re-runnable: a finished repair plans nothing.
#
# RUN ORDER (scripts/ops/soccer_repair_runbook.sh runs it, each step through
# its own wrapper): 1 repoint_team_matches.sh --pass 1, 2 migrate_027_outcome_trigger.sh,
# 3 merge_soccer_teams.sh, 4 repair_football_data_kickoffs.sh,
# 5 repoint_team_matches.sh --pass 2, 6 repair_stranded_recs_soccer.sh,
# 7 cleanup_soccer_orphans.sh, 8 backfill_soccer_regulation_scores.sh,
# 9 backfill_season_labels.sh, 10 compute_features --backfill-finished soccer,
# 11 the merge's odds check, 12 backfill_soccer_never_resulted.sh (+ the
# cleanup again), 13 THIS, 14 promote_raw_after_repair.sh,
# 15 grade_folded_soccer_rows.sh, 16 compute_features --backfill-finished soccer.
# REVERSAL: the apply writes /app/backups/soccer_labels_apply_<stamp>.reversal.sql
# beside its CSV (one transaction; each statement raises if its row moved; a
# tagged league only loses the code, a created one is dropped).
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
KEYS="create_league tag_league relabel_raw relabel_match relabel_team rename_team rename_raw already_raw already_match refuse mojibake_left cross_country_after"
WRITE_KEYS="create_league tag_league relabel_raw relabel_match relabel_team rename_team rename_raw"
MAP_ROWS=10
# Ceilings = the reviewed counts (prod 2026-11-01 03:27Z and 04:22Z); they can only fall. tag_league was 0: its
# ceiling is the two leagues the map needs, shared with create_league (MAX_LEAGUES).
MAX_create_league=2
MAX_tag_league=2
MAX_LEAGUES=2
MAX_relabel_raw=10
MAX_relabel_match=10
MAX_relabel_team=20
MAX_rename_team=1
MAX_rename_raw=68
ALLOWED_CROSS_COUNTRY="Guadalajara,Santos"
FEATURES_CMD="docker compose exec -T api python /app/scripts/compute_features.py --backfill-finished soccer"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
APPLY_CSV=/app/backups/soccer_labels_apply_$STAMP.csv
# The apply CSV of a run that has not passed step 4 yet (written before the apply, emptied once step 4 passes).
PENDING=/app/backups/soccer_labels_apply_pending.txt
pending_csv() { $DC exec -T api sh -c "cat $PENDING 2>/dev/null" </dev/null | grep -E '^/app/backups/soccer_labels_apply_[0-9TZ]+\.csv$' | tail -1; }
write_pending() { $DC exec -T api sh -c "cat > $PENDING.tmp && mv $PENDING.tmp $PENDING"; }  # stdin: the CSV, or nothing
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
  log "features: compute_features reads no league and no team name, so this repair changes no feature value; run as"
  log "  the soccer repair runbook's step 13, its step 16 rebuilds them; run on its own, to rebuild after this corpus"
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

verified() {  # verified <VERIFY line> <exit status>: every row of the apply CSV holds its new value
  [ "$2" -eq 0 ] && [ "$(plan_num "$1" rows)" != missing ] && [ "$(plan_num "$1" now_new)" = "$(plan_num "$1" rows)" ]
}
verify_line() { $DC exec -T api sh -c "grep '^VERIFY ' /tmp/soccer_labels_verify.log | tail -1"; }

log "0/4 already done? (read-only dry run)"
run_tool soccer_labels_pre --out-csv /app/backups/soccer_labels_precheck_$STAMP.csv
pre=$(plan_of soccer_labels_pre)
pre_cross=$(cross_of soccer_labels_pre)
if [ "$(writes_of "$pre")" = "missing" ] || [ "$(plan_num "$pre" refuse)" != "0" ] || [ "$(plan_num "$pre" mojibake_left)" != "0" ]; then
  log "ABORT at step 0: the dry run refuses something, finds a mis-decoded value it cannot repair, or its PLAN line is unreadable ('${pre:-none}'; full log /tmp/soccer_labels_pre.log in the api container) — nothing changed"
  exit 1
fi
if [ "$(writes_of "$pre")" = "0" ]; then
  if ! read_state; then log "ABORT at step 0: the state query failed — nothing changed"; exit 1; fi
  if [ "$misfiled" != "0" ] || [ "$moji_teams" != "0" ] || [ "$moji_raw" != "0" ]; then
    log "FAILED at step 0: the tool plans nothing, but read independently there are mapped Scottish fixtures in La Liga / La Liga 2 $misfiled, mis-decoded soccer team names $moji_teams and raw rows $moji_raw (expected 0, 0 and 0) — inspect; nothing changed"
    exit 1
  fi
  pending=$(pending_csv)
  if [ -n "$pending" ]; then
    log "an earlier run's apply has not passed its step 4 ($pending: that run ended during its apply, or its after-state FAILED) — verifying it (read only)"
    run_tool soccer_labels_verify --verify "$pending"
    vrc=$?
    verify=$(verify_line)
    if ! verified "$verify" "$vrc"; then
      log "FAILED at step 0: the earlier run's apply CSV $pending does not verify ('${verify:-unreadable}'; expected now_new = rows) — inspect it (${pending%.csv}.reversal.sql reverts it); nothing changed"
      exit 1
    fi
    write_pending < /dev/null || log "WARNING: could not empty $PENDING (a re-run verifies that apply again, which is harmless)"
    log "the earlier run's apply verifies: $verify"
  fi
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
counts="$p_create_league $p_tag_league $p_relabel_raw $p_relabel_match $p_relabel_team $p_rename_team $p_rename_raw $p_already_raw $p_already_match $p_refuse $p_mojibake_left $p_cross_country_after"
if [ "$rc" -ne 0 ] || ! [[ "$counts" =~ ^[0-9]+(\ [0-9]+){11}$ ]]; then
  log "ABORT at step 2: the dry run failed (exit $rc) or its PLAN line is unreadable ('$plan') — nothing changed"; exit 1
fi
over=""
for k in $WRITE_KEYS; do
  eval "v=\$p_$k; m=\$MAX_$k"
  # shellcheck disable=SC2154  # v and m are set by the eval above
  [ "$v" -le "$m" ] || over="$over $k=$v>$m"
done
# shellcheck disable=SC2154
[ $((p_create_league + p_tag_league)) -le "$MAX_LEAGUES" ] || over="$over create_league+tag_league=$((p_create_league + p_tag_league))>$MAX_LEAGUES"
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
if ! echo "$APPLY_CSV" | write_pending; then
  log "ABORT at step 3: could not record the apply CSV in $PENDING (a re-run could not check an apply this run left unchecked) — nothing changed"
  exit 1
fi
run_tool soccer_labels_apply --apply --expect-plan "$pin" --out-csv "$APPLY_CSV"
rc=$?
if [ "$rc" -ne 0 ]; then
  # $PENDING keeps $APPLY_CSV: a re-run that finds the work left applies afresh (and ignores it), and one that finds
  # it done — the apply committed after all, its exit lost — verifies it before any ALREADY DONE.
  log "ABORT at step 3 (exit $rc; full log /tmp/soccer_labels_apply.log in the api container): a plan that differs from the dry run, or a row that moved since it, rolls the one transaction back — nothing was written to the database (an audit CSV or reversal SQL the apply had already written is renamed ${APPLY_CSV}.void / ${APPLY_CSV%.csv}.reversal.sql.void: they list changes that never happened)"
  exit 1
fi

log "4/4 after-state"
run_tool soccer_labels_verify --verify "$APPLY_CSV"
vrc=$?
verify=$(verify_line)
run_tool soccer_labels_replan --out-csv /app/backups/soccer_labels_verify_$STAMP.csv
prc=$?
replan=$(plan_of soccer_labels_replan)
if ! read_state; then log "FAILED after-state: the after-state query failed — inspect (the apply CSV $APPLY_CSV and its .reversal.sql are the record)"; exit 1; fi
want_scot=$((scot0 + p_create_league))
log "after: verify '$verify' (expected now_new = rows), re-plan '$replan' (expected 0 to write, refuse 0, already_raw=$MAP_ROWS already_match=$MAP_ROWS), mapped Scottish fixtures in La Liga / La Liga 2 $misfiled (expected 0), mis-decoded soccer team names $moji_teams and raw rows $moji_raw (expected 0 and 0), Scottish leagues $scot_leagues (expected $want_scot), soccer teams $n_teams / soccer matches $n_matches / raw rows $n_raw (expected $n_teams0 / $n_matches0 / $n_raw0)"
if verified "$verify" "$vrc" \
   && [ "$prc" -eq 0 ] && [ "$(writes_of "$replan")" = "0" ] && [ "$(plan_num "$replan" refuse)" = "0" ] \
   && [ "$(plan_num "$replan" mojibake_left)" = "0" ] && [ "$(plan_num "$replan" already_raw)" = "$MAP_ROWS" ] \
   && [ "$(plan_num "$replan" already_match)" = "$MAP_ROWS" ] && [ "$misfiled" = "0" ] && [ "$moji_teams" = "0" ] \
   && [ "$moji_raw" = "0" ] && [ "$scot_leagues" = "$want_scot" ] && [ "$n_teams" = "$n_teams0" ] \
   && [ "$n_matches" = "$n_matches0" ] && [ "$n_raw" = "$n_raw0" ]; then
  write_pending < /dev/null || log "WARNING: could not empty $PENDING (a re-run verifies $APPLY_CSV again, which is harmless)"
  log "DONE: $p_relabel_match fixture(s) and $p_relabel_raw raw row(s) relabelled, $p_relabel_team team(s) moved to their Scottish league, $p_rename_team team and $p_rename_raw raw row(s) renamed, $p_create_league league(s) created and $p_tag_league tagged (reversal: ${APPLY_CSV%.csv}.reversal.sql)"
  features_note
else
  log "FAILED after-state — inspect (the apply CSV $APPLY_CSV is the record; ${APPLY_CSV%.csv}.reversal.sql reverts it; a re-run verifies it again before any ALREADY DONE)"
  exit 1
fi
