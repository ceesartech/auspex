#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md:
# migrations need a verified backup first, which step 2 takes).
# Apply migration 027: the match-outcome trigger (calculate_match_outcome)
# stops marking recs written at or after kickoff won/lost and stops grading
# predictions written after kickoff. The Python grader voids / skips those.
#   1. pre-check (read-only): the live function must be 011's (or already
#      027's). Anything else is an unknown edit 027 would overwrite;
#   2. verified backup (B2) — abort if it does not land;
#   3. dry run: 027 plus the proof from step 6 in ONE transaction, rolled
#      back, so the new body is proven on prod before it is committed;
#   4. apply 027 (one transaction; CREATE OR REPLACE FUNCTION locks no table);
#   5. after-state: the live body is 027's byte for byte (md5), carries each
#      guard exactly once, and trg_match_outcome still fires it, enabled.
#      EXPECTED "<md5 027>|1|1|1";
#   6. prove it (rolled back): flip a scratch match to finished holding a
#      pre-kickoff rec, a rec written AT kickoff (the >= boundary, on the same
#      graded prediction), and a pre- and a post-kickoff prediction. EXPECTED
#      "m027 pre_pred=true post_pred=null pre_rec=won post_rec=pending";
#      pre_rec=won proves the trigger fired at all.
# Re-runnable: re-applying 027 is a no-op.
# Run on the VM host:
#   bash /opt/auspex/scripts/ops/migrate_027_outcome_trigger.sh 2>&1 | tee -a ~/migrate_027.log
set -uo pipefail
cd /opt/auspex
DC="docker compose"
MIG=services/data-ingestion/db/migrations/027_outcome_trigger_skips_post_kickoff.sql
MD5_011=24b0d94b2d5b9ad6e6f8b04550e9d9f7 # md5(pg_get_functiondef) after 011 — prod's, 2026-10-31
MD5_027=97384c8327efa61669af1012c06c6232 # after 027 (scratch Postgres, every migration in order)
EXPECT_STATE="$MD5_027|1|1|1"
EXPECT_PROOF="m027 pre_pred=true post_pred=null pre_rec=won post_rec=pending"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
# run_sql <host file> <log file> [psql opts]: output to the log; exit status
# is psql's (ON_ERROR_STOP, so any SQL error is non-zero) — never a pipe's.
run_sql() {
  $DC cp "$1" postgres:/tmp/"$(basename "$1")" >/dev/null 2>&1
  $DC exec -T postgres sh -c "psql -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -X -q -v ON_ERROR_STOP=1 ${3:-} -f /tmp/$(basename "$1")" > "$2" 2>&1
}
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
cleanup() {
  rm -f /tmp/m027_*.sql
  $DC exec -T postgres sh -c "rm -f /tmp/m027_*.sql /tmp/$(basename "$MIG")" >/dev/null 2>&1
}
trap cleanup EXIT

# The proof, without BEGIN/ROLLBACK: step 3 runs it inside 027's transaction,
# step 6 inside its own. A scratch match (no league or teams; kickoff 3h ago)
# and its rows exist only inside that transaction; stakes are 0.
cat > /tmp/m027_prove_body.sql <<'SQLEOF'
SET LOCAL lock_timeout = '10s';
INSERT INTO matches (match_date, status, season)
VALUES (date_trunc('second', NOW()) - interval '3 hours', 'scheduled', 'm027')
RETURNING id AS mid, match_date AS ko \gset
INSERT INTO predictions (match_id, model_name, model_version, prediction_type, predicted_outcome,
                         confidence, probabilities, created_at)
VALUES (:'mid', 'm027_pre', 'm027', 'match_result', 'home', 0.5, '{}', :'ko'::timestamptz - interval '1 hour')
RETURNING id AS pre_pred \gset
INSERT INTO predictions (match_id, model_name, model_version, prediction_type, predicted_outcome,
                         confidence, probabilities, created_at)
VALUES (:'mid', 'm027_post', 'm027', 'match_result', 'home', 0.5, '{}', :'ko'::timestamptz + interval '1 minute')
RETURNING id AS post_pred \gset
INSERT INTO betting_recommendations (prediction_id, match_id, bet_type, selection, odds_at_recommendation,
                                     bookmaker, recommended_stake, status, created_at)
VALUES (:'pre_pred', :'mid', 'match_result', 'home', 2.0, 'm027', 0, 'pending',
        :'ko'::timestamptz - interval '30 minutes')
RETURNING id AS pre_rec \gset
INSERT INTO betting_recommendations (prediction_id, match_id, bet_type, selection, odds_at_recommendation,
                                     bookmaker, recommended_stake, status, created_at)
VALUES (:'pre_pred', :'mid', 'match_result', 'home', 2.0, 'm027', 0, 'pending', :'ko'::timestamptz)
RETURNING id AS post_rec \gset
UPDATE matches SET status = 'finished', home_score = 2, away_score = 1 WHERE id = :'mid';
SELECT 'm027 pre_pred=' || COALESCE((SELECT is_correct::text FROM predictions WHERE id = :'pre_pred'), 'null')
    || ' post_pred=' || COALESCE((SELECT is_correct::text FROM predictions WHERE id = :'post_pred'), 'null')
    || ' pre_rec=' || (SELECT status FROM betting_recommendations WHERE id = :'pre_rec')
    || ' post_rec=' || (SELECT status FROM betting_recommendations WHERE id = :'post_rec');
SQLEOF

log "1/6 pre-check (read-only)"
cat > /tmp/m027_fn.sql <<'SQLEOF'
SELECT md5(pg_get_functiondef('calculate_match_outcome()'::regprocedure));
SQLEOF
run_sql /tmp/m027_fn.sql /tmp/m027_fn.log -tA
rc=$?
live=$(tr -d '[:space:]' < /tmp/m027_fn.log)
if [ $rc -ne 0 ]; then cat /tmp/m027_fn.log; log "ABORT at step 1: could not read calculate_match_outcome — nothing changed"; exit 1; fi
case "$live" in
  "$MD5_011") log "live function is 011's, as expected" ;;
  "$MD5_027") log "live function is already 027's — re-applying is a no-op" ;;
  *) log "ABORT at step 1: live calculate_match_outcome (md5 $live) is neither 011's nor 027's — an unknown edit 027 would overwrite; inspect. Nothing changed"; exit 1 ;;
esac

log "2/6 backup"
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

log "3/6 dry run: 027 + proof in one transaction, rolled back"
sed '/^COMMIT;$/d' "$MIG" > /tmp/m027_dry.sql
cat /tmp/m027_prove_body.sql >> /tmp/m027_dry.sql
echo 'ROLLBACK;' >> /tmp/m027_dry.sql
if grep -qiE '^[[:space:]]*COMMIT' /tmp/m027_dry.sql; then log "ABORT at step 3: the dry-run file still commits — nothing changed"; exit 1; fi
run_sql /tmp/m027_dry.sql /tmp/m027_dry.log -tA
rc=$?
cat /tmp/m027_dry.log
if [ $rc -ne 0 ]; then log "ABORT at step 3: the dry run failed (rolled back) — nothing changed"; exit 1; fi
if ! grep -qFx "$EXPECT_PROOF" /tmp/m027_dry.log; then
  log "ABORT at step 3: under 027 the proof did not read '$EXPECT_PROOF' (rolled back) — nothing changed"; exit 1
fi

log "4/6 apply 027"
run_sql "$MIG" /tmp/m027_apply.log
rc=$?
cat /tmp/m027_apply.log
if [ $rc -ne 0 ]; then log "ABORT at step 4: migration failed (one transaction — nothing changed); re-run"; exit 1; fi

log "5/6 after-state"
cat > /tmp/m027_verify.sql <<'SQLEOF'
WITH f AS (SELECT pg_get_functiondef('calculate_match_outcome()'::regprocedure) AS d)
SELECT md5(d)
    || '|' || (cardinality(string_to_array(d, 'AND (p.created_at > NEW.match_date) IS NOT TRUE')) - 1)
    || '|' || (cardinality(string_to_array(d, 'AND (br.created_at >= NEW.match_date) IS NOT TRUE')) - 1)
    || '|' || (SELECT count(*) FROM pg_trigger
                WHERE tgname = 'trg_match_outcome' AND tgrelid = 'matches'::regclass
                  AND tgfoid = 'calculate_match_outcome()'::regprocedure AND tgenabled <> 'D')
  FROM f;
SQLEOF
run_sql /tmp/m027_verify.sql /tmp/m027_verify.log -tA
rc=$?
state=$(tr -d '[:space:]' < /tmp/m027_verify.log)
echo "md5|prediction guard|rec guard|trigger: $state"
if [ $rc -ne 0 ] || [ "$state" != "$EXPECT_STATE" ]; then
  log "FAILED at step 5: expected $EXPECT_STATE — inspect now (revert: re-apply 011)"; exit 1
fi

log "6/6 prove: a post-kickoff rec stays pending (rolled back)"
echo 'BEGIN;' > /tmp/m027_prove.sql
cat /tmp/m027_prove_body.sql >> /tmp/m027_prove.sql
echo 'ROLLBACK;' >> /tmp/m027_prove.sql
run_sql /tmp/m027_prove.sql /tmp/m027_prove.log -tA
rc=$?
cat /tmp/m027_prove.log
if [ $rc -eq 0 ] && grep -qFx "$EXPECT_PROOF" /tmp/m027_prove.log; then
  log "DONE: post-kickoff rec left pending, post-kickoff prediction ungraded, pre-kickoff rec settled"
else
  log "FAILED at step 6: expected '$EXPECT_PROOF' — inspect now (revert: re-apply 011)"; exit 1
fi
