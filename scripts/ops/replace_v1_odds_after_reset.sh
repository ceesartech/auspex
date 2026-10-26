#!/bin/bash
# Approved 2026-10-26: replace the remaining v1 historical odds
# (NBA/NFL 2022-24, NHL 2022-25) once the-odds-api quota resets on Nov 1, then
# give tennis its first market prices from the same provider:
#   * load the missing Jan 2025 - Apr 2026 tennis matches from ESPN (free,
#     additive; stops before the ESPN-keyed live era to avoid duplicates);
#   * backfill tennis match-winner odds per tournament (--sport tennis,
#     ~40k credits at 3 h windows) — ONLY if the balance then still covers it
#     plus a month of live fetching (TENNIS_MIN_REMAINING);
#   * recompute features for every sport touched.
# Verified backup first; every replaced row archived to CSV; quota floor 15k.
# Run on the VM host after the Nov 1 reset (it waits for the reset itself):
#   nohup bash /opt/auspex/scripts/ops/replace_v1_odds_after_reset.sh >> ~/replace_v1_odds.log 2>&1 &
# or schedule it once:  5 0 1 11 * /bin/bash /opt/auspex/scripts/ops/replace_v1_odds_after_reset.sh
# Removes any cron line naming it when it finishes (success or give-up).
set -u
cd /opt/auspex
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
cleanup_cron() { crontab -l 2>/dev/null | grep -v replace_v1_odds_after_reset.sh | crontab -; log "cron entry removed"; }
remaining() {
  $DC exec -T api python -c "import os,requests; r=requests.get('https://api.the-odds-api.com/v4/sports', params={'apiKey': os.environ['THE_ODDS_API_KEY']}, timeout=20); print(r.headers.get('x-requests-remaining','0'))" 2>/dev/null | tail -1
}
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(o['LastModified'].strftime('%s'))" 2>/dev/null | tail -1
}
backup_and_wait() {
  local before; before=$(date -u +%s)
  $DC exec -T airflow-scheduler airflow dags trigger db_backup_daily >/dev/null 2>&1
  for _ in $(seq 1 40); do
    sleep 30
    local ts; ts=$(latest_dump)
    if [ -n "$ts" ] && [ "$ts" -gt "$before" ]; then log "backup landed in B2 (dump epoch $ts)"; return 0; fi
  done
  log "ERROR backup did not land in B2 within 20 min"; return 1
}

log "start; waiting for the quota reset"
deadline=$(( $(date -u +%s) + 12*3600 ))
while :; do
  r=$(remaining); log "remaining=$r"
  if [ -n "$r" ] && [ "${r%%.*}" -ge 70000 ] 2>/dev/null; then break; fi
  if [ "$(date -u +%s)" -ge "$deadline" ]; then log "GAVE UP: quota never reset within 12 h — nothing replaced"; cleanup_cron; exit 1; fi
  sleep 600
done

backup_and_wait || { log "ABORT: no verified backup, nothing replaced"; cleanup_cron; exit 1; }

for spec in "americanfootball_nfl 2022-09-01 2024-02-29" "basketball_nba 2022-10-01 2024-06-30" "icehockey_nhl 2022-10-01 2025-06-30"; do
  set -- $spec
  log "replace $1 $2..$3"
  $DC exec -T api sh -c "cd /app/scripts && python backfill_historical_odds.py --sport $1 --start $2 --end $3 --mode replace --confirm-replace --cache-dir /app/backups/odds_hist_cache --archive-dir /app/backups --quota-floor 15000 2>&1 | grep -E 'Plan:|Totals:|Quota floor|archiving|Traceback|Error'"
done

log "tennis: load the missing Jan 2025 - Apr 2026 matches from ESPN (free, additive)"
$DC exec -T api sh -c "cd /app/scripts && python load_tennis_historical.py --start-date 2025-01-01 --end-date 2026-04-30 --tour both 2>&1 | tail -2"

TENNIS_MIN_REMAINING=${TENNIS_MIN_REMAINING:-85000}
r=$(remaining)
if [ -n "$r" ] && [ "${r%%.*}" -ge "$TENNIS_MIN_REMAINING" ] 2>/dev/null; then
  log "tennis odds backfill (remaining=$r)"
  $DC exec -T api sh -c "cd /app/scripts && python backfill_historical_odds.py --sport tennis --start 2021-09-01 --end 2026-10-31 --cache-dir /app/backups/odds_hist_cache --quota-floor 45000 2>&1 | grep -E 'Tennis plan|Totals:|Quota floor|Traceback|Error' | cut -c1-600"
else
  log "SKIP tennis odds: remaining=$r < $TENNIS_MIN_REMAINING (needs ~40k plus a month of live fetching); re-run this step after raising the plan"
fi

for s in nfl nba nhl tennis; do
  log "recompute features $s"
  $DC exec -T api sh -c "cd /app/scripts && python compute_features_$s.py --all-finished --force 2>&1 | tail -1"
done

backup_and_wait || log "WARNING post-replace backup did not land; Sunday retrain uses the previous dump"
hour=$(date -u +%H)
if [ "$(date -u +%u)" = "7" ] && [ "$hour" -ge 4 ]; then
  log "finished after the 04:00 scheduled retrain — triggering retrain_models"
  $DC exec -T airflow-scheduler airflow dags trigger retrain_models >/dev/null 2>&1
fi
log "DONE"
cleanup_cron
