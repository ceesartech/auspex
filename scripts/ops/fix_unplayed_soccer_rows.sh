#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Retire the four soccer rows football-data wrote as 'finished' with no
# full-time score (verified read-only 2026-10-31; none has predictions or
# recommendations):
#   d7d8a9b7 Panathinaikos v Olympiakos 2019-03-17 (G1)  abandoned, ESPN 524152, bogus HT 0-3
#   ff74d919 Bastia v Red Star          2025-12-05 (F2)  abandoned at 58', ESPN 747597, bogus HT 0-3
#   681794ae Niki Volos v OFI           2015-05-10 (G1)  never played, not on ESPN
#   83ce848f Chapecoense v Atletico-MG  2016-12-11 (BR1) cancelled, ESPN 445952
# -> status 'cancelled', the bogus HT pair NULLed, external_ids.espn and
#    metadata.result_note set. Deploy the promote_raw fix FIRST (same commit):
#    a re-run of the old loader would rewrite all four as 'finished'.
#   1. verified backup (B2) — abort if it does not land;
#   2. dry run (plan + write-once CSV) — abort if any row is refused;
#   3. apply (one transaction, every UPDATE guarded on the observed values);
#   4. after-state + verify.
# Re-runnable: a fixed row plans as 'done' and is not written again.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/fix_unplayed_soccer_rows.sh 2>&1 | tee -a ~/fix_unplayed_soccer_rows.log
set -uo pipefail
cd /opt/auspex
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
TOOL="python /app/scripts/fix_unplayed_soccer_rows.py"

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

log "2/4 dry run"
# Exit status through a file, not a pipe: `| tail` would report tail's 0.
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/fix_unplayed_soccer_rows_dryrun_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/fix_unplayed_dry.log 2>&1; rc=\$?; tail -8 /tmp/fix_unplayed_dry.log; exit \$rc" \
  || { log "ABORT at step 2: a row was refused or the tool failed (see above) — nothing changed"; exit 1; }

log "3/4 apply"
$DC exec -T api sh -c "$TOOL --apply > /tmp/fix_unplayed.log 2>&1; rc=\$?; tail -8 /tmp/fix_unplayed.log; exit \$rc" \
  || { log "ABORT at step 3 (see above): the transaction rolled back — nothing changed"; exit 1; }

log "4/4 after-state"
cat > /tmp/fix_unplayed_after.sql <<'SQLEOF'
SELECT m.id, ht.name AS home, at.name AS away, m.match_date::date AS day, m.status,
       m.home_score, m.away_score, m.half_time_home, m.half_time_away,
       m.external_ids->>'espn' AS espn, left(m.metadata->>'result_note', 60) AS note
  FROM matches m JOIN teams ht ON ht.id = m.home_team_id JOIN teams at ON at.id = m.away_team_id
 WHERE m.id IN ('d7d8a9b7-39a6-4340-bb53-09982171fa8a', 'ff74d919-d543-4729-a438-5f59e826b7b9',
                '681794ae-f1e5-4e10-a365-8e71f86ef4a1', '83ce848f-64ad-4147-ab32-4a770a5eed15')
 ORDER BY m.match_date;
SELECT count(*) AS soccer_finished_without_ft
  FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'soccer' AND m.status = 'finished' AND (m.home_score IS NULL OR m.away_score IS NULL);
SQLEOF
$DC cp /tmp/fix_unplayed_after.sql postgres:/tmp/fix_unplayed_after.sql >/dev/null 2>&1
out=$($DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/fix_unplayed_after.sql')
echo "$out"
cancelled=$(echo "$out" | grep -c '| cancelled |')
left=$(echo "$out" | awk '/soccer_finished_without_ft/ {getline; getline; print $1}')
if [ "$cancelled" = "4" ] && [ "$left" = "0" ]; then log "DONE"; else log "FAILED: $cancelled/4 rows cancelled, $left soccer row(s) still finished without a score — inspect"; exit 1; fi
