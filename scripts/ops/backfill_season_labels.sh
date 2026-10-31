#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Relabel the soccer matches.season values promote_raw mislabelled, with
# scripts/backfill_season_labels.py. Read-only check 2026-10-31: 32,794 rows
# carry a 'YYYY-YYYY' that does not span one year ('2020-2116' for a 2016
# calendar season) and 26,890 carry 'YYYY/YYYY'. All 59,684 tie back to a
# raw_football_data row at the kickoff promote_raw wrote. Calendar-year
# leagues (promote_raw.CALENDAR_SEASON_LEAGUES) get the single year ('2016');
# cross-year leagues get 'YYYY-YYYY'.
#   1. verified backup (B2) — abort if it does not land;
#   2. before-state + dry run (plan + write-once audit CSV) — abort if the tool fails;
#   3. apply (one transaction, every UPDATE guarded on the old value);
#   4. after-state — expects 0 soccer rows with a 'YYYY/YYYY' label, a
#      'YYYY-YYYY' label that does not span exactly one year, or a single-year
#      label outside the calendar-year leagues (where it is the right label).
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status; a pipe through tail would report tail's 0.
# Re-runnable: a relabelled row plans as already_correct.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/backfill_season_labels.sh 2>&1 | tee -a ~/backfill_season_labels.log
set -uo pipefail
cd /opt/auspex
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_file() {  # psql_file <local.sql> [extra psql flags]
  local f="$1"; shift
  $DC cp "$f" postgres:/tmp/season_labels_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c "psql -v ON_ERROR_STOP=1 -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" $* -f /tmp/season_labels_q.sql"
}
TOOL="python /app/scripts/backfill_season_labels.py"

# The calendar-year leagues come from the code that writes the labels, so this
# check cannot drift from it.
CAL=$($DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import promote_raw as pr; print(','.join(sorted(pr.CALENDAR_SEASON_LEAGUES)))" 2>/dev/null | tail -1)
if ! [[ "$CAL" =~ ^[A-Z0-9]+(,[A-Z0-9]+)+$ ]]; then log "ABORT: could not read promote_raw.CALENDAR_SEASON_LEAGUES (got '$CAL') — nothing changed"; exit 1; fi
log "calendar-year leagues: $CAL"

cat > /tmp/season_labels_shapes.sql <<'SQLEOF'
SELECT CASE WHEN m.season ~ '^[0-9]{4}/[0-9]{4}$' THEN 'YYYY/YYYY (bad)'
            WHEN m.season ~ '^[0-9]{4}-[0-9]{4}$'
                 AND split_part(m.season, '-', 2)::int - split_part(m.season, '-', 1)::int = 1 THEN 'YYYY-YYYY one year'
            WHEN m.season ~ '^[0-9]{4}-[0-9]{4}$' THEN 'YYYY-YYYY other span (bad)'
            WHEN m.season ~ '^[0-9]{4}$' AND l.external_ids->>'football_data' = ANY (string_to_array('__CAL__', ','))
                 THEN 'YYYY, calendar-year league'
            WHEN m.season ~ '^[0-9]{4}$' THEN 'YYYY, cross-year league (bad)'
            ELSE coalesce('other: ' || m.season, 'NULL') END AS label_shape,
       count(*) AS rows
  FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'soccer'
 GROUP BY 1 ORDER BY 1;
SQLEOF
cat > /tmp/season_labels_bad.sql <<'SQLEOF'
SELECT count(*)
  FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'soccer'
   AND (m.season ~ '^[0-9]{4}/[0-9]{4}$'
        OR (m.season ~ '^[0-9]{4}-[0-9]{4}$'
            AND split_part(m.season, '-', 2)::int - split_part(m.season, '-', 1)::int <> 1)
        OR (m.season ~ '^[0-9]{4}$'
            AND NOT coalesce(l.external_ids->>'football_data' = ANY (string_to_array('__CAL__', ',')), false)));
SQLEOF
sed -i "s/__CAL__/$CAL/g" /tmp/season_labels_shapes.sql /tmp/season_labels_bad.sql

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
psql_file /tmp/season_labels_shapes.sql || { log "ABORT at step 2: the before-state query failed — nothing changed"; exit 1; }
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/season_labels_dryrun_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/season_labels_dry.log 2>&1; rc=\$?; grep -E 'Loaded|Planned|ERROR|DRY-RUN|Done:|Traceback|Error' /tmp/season_labels_dry.log | tail -12; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 2 exit $rc"
if [ "$rc" -ne 0 ]; then log "ABORT at step 2: the dry run failed (see above; full log /tmp/season_labels_dry.log in the api container) — nothing changed"; exit 1; fi

log "3/4 apply"
$DC exec -T api sh -c "$TOOL --apply > /tmp/season_labels_apply.log 2>&1; rc=\$?; grep -E 'Planned|ERROR|Done:|Traceback|Error' /tmp/season_labels_apply.log | tail -12; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 3 exit $rc"
if [ "$rc" -ne 0 ]; then
  # An exception rolls the whole transaction back; exit 1 after a commit means
  # some rows changed between plan and write and were left as they were.
  log "ABORT at step 3 (see above; full log /tmp/season_labels_apply.log in the api container): if the tool raised, nothing changed; if it reported changed_since_plan, the other relabels are committed — inspect before re-running"
  exit 1
fi

log "4/4 after-state"
psql_file /tmp/season_labels_shapes.sql
bad=$(psql_file /tmp/season_labels_bad.sql -At | tr -d '[:space:]')
log "soccer rows still carrying a bad season label: ${bad:-unknown}"
rm -f /tmp/season_labels_shapes.sql /tmp/season_labels_bad.sql
$DC exec -T postgres rm -f /tmp/season_labels_q.sql >/dev/null 2>&1
if [ "$bad" = "0" ]; then log "DONE"; else log "FAILED: ${bad:-unknown} soccer row(s) still carry a bad season label — the dry-run counters above (raw_row_unresolved, no_matches_row, raw_duplicates_disagree, season_not_from_this_raw_row) say why; inspect"; exit 1; fi
