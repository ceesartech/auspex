#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Stamp the 90-minute score (metadata.regulation_home_score /
# regulation_away_score + went_extra) on the finished soccer ties decided in
# extra time or on penalties, with scripts/backfill_soccer_regulation_scores.py.
# Read-only check 2026-10-31: 15 rows (5 AET, 10 FT-PENS; 2026 World Cup and
# Copa Libertadores), and a local dry run of the tool against ESPN planned all
# 15: five change basis (Belgium v Senegal 3-2 AET was 2-2 after 90 minutes,
# Argentina v Cape Verde 3-2 -> 1-1, Norway v England 1-2 -> 1-1, Argentina v
# Switzerland 3-1 -> 1-1, Spain v Argentina 1-0 -> 0-0); the ten shootouts keep
# their score. Deploy the commit that ships the tool FIRST: the graders read
# the keys only from then on.
#   1. verified backup (B2) — abort if it does not land;
#   2. before-state + dry run (plan + write-once audit CSV) — abort if the tool
#      refuses any row or fails;
#   3. apply (one transaction, every UPDATE guarded on the observed row);
#   4. after-state — expects 0 finished soccer rows whose result_detail says
#      decided after 90 minutes and that still lack the 90-minute score.
# Then re-grade the predictions on those matches (printed at the end; not run
# here: it has its own dry run and write-once CSV).
# Each tool step writes its output to a log file inside the api container and
# exits with the tool's own status; a pipe through tail would report tail's 0.
# Re-runnable: a stamped row plans as already_set.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/backfill_soccer_regulation_scores.sh 2>&1 | tee -a ~/backfill_soccer_regulation_scores.log
set -uo pipefail
cd /opt/auspex
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
psql_file() {  # psql_file <local.sql> [extra psql flags]
  local f="$1"; shift
  $DC cp "$f" postgres:/tmp/regulation_scores_q.sql >/dev/null 2>&1 || return 1
  $DC exec -T postgres sh -c "psql -v ON_ERROR_STOP=1 -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" $* -f /tmp/regulation_scores_q.sql"
}
TOOL="python /app/scripts/backfill_soccer_regulation_scores.py"
EXPECTED_LEFT=0

cat > /tmp/regulation_scores_state.sql <<'SQLEOF'
SELECT l.external_ids->>'football_data' AS league, (m.match_date AT TIME ZONE 'UTC')::date AS day,
       ht.name AS home, at.name AS away, m.home_score || '-' || m.away_score AS final,
       m.metadata->>'result_detail' AS detail,
       (m.metadata->>'regulation_home_score') || '-' || (m.metadata->>'regulation_away_score') AS ninety,
       m.metadata->>'went_extra' AS went_extra
  FROM matches m
  JOIN leagues l ON l.id = m.league_id
  JOIN teams ht ON ht.id = m.home_team_id
  JOIN teams at ON at.id = m.away_team_id
 WHERE l.sport = 'soccer' AND m.status = 'finished'
   AND m.metadata->>'result_detail' ~* '\m(AET|PENS?)\M'
 ORDER BY m.match_date;
SQLEOF
cat > /tmp/regulation_scores_left.sql <<'SQLEOF'
SELECT count(*)
  FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'soccer' AND m.status = 'finished'
   AND m.metadata->>'result_detail' ~* '\m(AET|PENS?)\M'
   AND NOT (m.metadata ? 'regulation_home_score' AND m.metadata ? 'regulation_away_score');
SQLEOF

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
psql_file /tmp/regulation_scores_state.sql || { log "ABORT at step 2: the before-state query failed — nothing changed"; exit 1; }
$DC exec -T api sh -c "$TOOL --out-csv /app/backups/soccer_regulation_scores_dryrun_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/regulation_scores_dry.log 2>&1; rc=\$?; grep -E 'stamp|already_set|refuse|candidate|ERROR|DRY-RUN|Done:|Traceback|Error' /tmp/regulation_scores_dry.log | tail -24; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 2 exit $rc"
if [ "$rc" -ne 0 ]; then log "ABORT at step 2: the dry run refused a row or failed (see above; full log /tmp/regulation_scores_dry.log in the api container) — nothing changed"; exit 1; fi

log "3/4 apply"
$DC exec -T api sh -c "$TOOL --apply > /tmp/regulation_scores_apply.log 2>&1; rc=\$?; grep -E 'candidate|Stamped|ERROR|Done:|Traceback|Error' /tmp/regulation_scores_apply.log | tail -12; echo \"tool exit \$rc\"; exit \$rc"
rc=$?
log "step 3 exit $rc"
if [ "$rc" -ne 0 ]; then
  # A refusal writes nothing; a row that changed between plan and write raises
  # and rolls the whole transaction back.
  log "ABORT at step 3 (see above; full log /tmp/regulation_scores_apply.log in the api container): nothing was written"
  exit 1
fi

log "4/4 after-state"
psql_file /tmp/regulation_scores_state.sql
left=$(psql_file /tmp/regulation_scores_left.sql -At | tr -d '[:space:]')
log "finished soccer rows decided after 90 minutes still without the 90-minute score: ${left:-unknown} (expected $EXPECTED_LEFT)"
rm -f /tmp/regulation_scores_state.sql /tmp/regulation_scores_left.sql
$DC exec -T postgres rm -f /tmp/regulation_scores_q.sql >/dev/null 2>&1
if [ "$left" != "$EXPECTED_LEFT" ]; then log "FAILED: ${left:-unknown} row(s) left — inspect the dry-run reasons above"; exit 1; fi
log "DONE. Now re-grade their predictions (dry run; inspect the per-type summary, then repeat with --apply):"
cat <<'EOF'
  docker compose exec -T api python /app/scripts/regrade_predictions.py --model-name ensemble --days 150 \
      --prediction-type match_result,btts,over_under,double_chance,draw_no_bet,correct_score \
      --prediction-type asian_handicap,clean_sheet,win_to_nil,team_total,odd_even,total_goals \
      --prediction-type winning_margin,result_btts,result_over_under,ht_ft_double_result
  # Expected on the five AET matches (local plan from a read-only export, 160 rows): 113 rows change,
  # e.g. match_result 9 True->False, draw_no_bet 10 -> push. over_under also rewrites the DB trigger's
  # 'over'/'under' strings to 'over_0.5'/'under_5.5' on every graded row in the window, verdicts unchanged.
EOF
