#!/bin/bash
# Approved 2026-10-30 (operator: "proceed with all of them"): clean up the
# ~19.5k stale 'scheduled' tennis fixture rows left by the fixture-identity
# bug (pre-2026-09-04) and by re-drawn draw slots (fixed in 2337db1).
#   1. verified backup (B2) — abort if it does not land;
#   2. repair_stranded_recs.py --sport tennis --void-no-twin --apply
#      (dry run 2026-10-30: 80 pending recs -> 50 settled 18W/32L on 37
#      distinct fixtures + re-pointed onto the twin, 30 voided: no twin);
#   3. repair_fixture_orphans.py --sport tennis --apply
#      (dry run: 16,003 orphans merge onto 3,200 finished twins — strictly
#      pre-kickoff odds / CLV snapshots / predictions — and are deleted;
#      3,386 never-played pairings deleted; archive written first);
#   4. recompute tennis features (the twins now carry the moved prices);
#   5. grade the predictions moved onto finished twins (tennis only — a wide
#      --days run would also touch ~3k soccer/NHL/NFL matches);
#   6. print the after-state.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/cleanup_tennis_orphans.sh 2>&1 | tee -a ~/cleanup_tennis_orphans.log
set -uo pipefail
cd /opt/auspex
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}

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

log "2/6 settle / void the stranded tennis recs"
$DC exec -T api sh -c "cd /app/scripts && python repair_stranded_recs.py --sport tennis --void-no-twin --apply > /tmp/rsr.log 2>&1; rc=\$?; grep -v 'FAR TWIN' /tmp/rsr.log | tail -8; exit \$rc" || { log "ABORT at step 2 (see output above)"; exit 1; }

log "3/6 merge + delete the orphan fixture rows"
$DC exec -T api sh -c "cd /app/scripts && python repair_fixture_orphans.py --sport tennis --apply > /tmp/rfo.log 2>&1; rc=\$?; grep -E 'candidates=|archived|DONE|Error|Traceback' /tmp/rfo.log; exit \$rc" || { log "ABORT at step 3 (see output above)"; exit 1; }

log "4/6 recompute tennis features"
$DC exec -T api sh -c "cd /app/scripts && python compute_features_tennis.py --all-finished --force 2>&1 | tail -1"

log "5/6 grade the predictions moved onto finished twins"
$DC exec -T api python - <<'PY'
import os, sys
sys.path.insert(0, "/app/scripts")
import psycopg2
from psycopg2.extras import RealDictCursor
import grade_completed_matches as g

with psycopg2.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute("""
        SELECT m.id::text AS match_id, m.home_score, m.away_score, m.status, m.metadata
          FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'tennis' AND m.status = 'finished' AND m.match_date >= '2026-05-01'
           AND EXISTS (SELECT 1 FROM predictions p WHERE p.match_id = m.id AND p.is_correct IS NULL)
         ORDER BY m.match_date""")
    matches = [dict(r) for r in cur.fetchall()]
    stats, graded = {"phantom_line_skips": 0}, 0
    for m in matches:
        graded += g.grade_match(cur, m, stats)["predictions_graded"]
    conn.commit()
print(f"graded {graded} predictions on {len(matches)} tennis matches")
PY

log "6/6 after-state"
cat > /tmp/cleanup_after.sql <<'SQLEOF'
SELECT count(*) AS stale_tennis_rows_left FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'tennis' AND m.status IN ('scheduled','cancelled') AND m.match_date < now() - interval '4 days';
SELECT count(*) AS finished_tennis_since_june, count(*) FILTER (WHERE EXISTS (SELECT 1 FROM odds o WHERE o.match_id = m.id)) AS priced
  FROM matches m JOIN leagues l ON l.id = m.league_id WHERE l.sport = 'tennis' AND m.status = 'finished' AND m.match_date >= '2026-06-01';
SQLEOF
$DC cp /tmp/cleanup_after.sql postgres:/tmp/cleanup_after.sql >/dev/null 2>&1
$DC exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/cleanup_after.sql'
log "DONE"
