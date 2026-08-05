#!/usr/bin/env bash
# One-shot operator script (2026-08-05): re-grade the two soccer markets whose
# grading was broken until 6c54a11 — correct_score (predicted_outcome was the
# ungradable 'other' bucket) and double_chance (single-label equality on a
# coverage market). Deterministic: recomputes from stored probabilities JSONB
# + final scores, then re-runs the (fixed) grader over the affected window.
# Safe to re-run; delete this file once the re-grade is verified.
#
# Run:  ssh auspex 'bash /opt/auspex/scripts/ops_regrade_soccer_markets.sh'
set -euo pipefail
cd /opt/auspex

cat > /tmp/regrade_soccer_markets.sql <<'SQLEOF'
-- 1. correct_score: recompute predicted_outcome/confidence from the stored
--    JSONB (argmax excluding 'other' and '*_push'), reset the grade.
WITH best AS (
  SELECT p.id, kv.key AS new_pred, kv.value::float AS new_conf
  FROM predictions p
  CROSS JOIN LATERAL (
    SELECT key, value FROM jsonb_each_text(p.probabilities)
    WHERE key <> 'other' AND key !~ '_push$'
    ORDER BY value::float DESC
    LIMIT 1
  ) kv
  WHERE p.prediction_type = 'correct_score'
)
UPDATE predictions p
SET predicted_outcome = b.new_pred,
    confidence = b.new_conf,
    actual_outcome = NULL,
    is_correct = NULL,
    updated_at = NOW()
FROM best b
WHERE p.id = b.id;

-- 2. double_chance: reset grades so the coverage-membership grader re-runs.
UPDATE predictions
SET actual_outcome = NULL,
    is_correct = NULL,
    updated_at = NOW()
WHERE prediction_type = 'double_chance'
  AND is_correct IS NOT NULL;
SQLEOF

echo "== applying re-grade SQL (single transaction) =="
docker compose cp /tmp/regrade_soccer_markets.sql postgres:/tmp/rg.sql
docker compose exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" --single-transaction -f /tmp/rg.sql'

echo "== re-grading with the fixed grader (40-day window) =="
docker compose exec -T api python /app/scripts/grade_completed_matches.py --days 40

echo "== post-check: accuracy of the re-graded markets =="
cat > /tmp/regrade_check.sql <<'SQLEOF'
SELECT prediction_type,
       count(*) FILTER (WHERE is_correct IS NOT NULL) AS graded,
       round(avg(is_correct::int) FILTER (WHERE is_correct IS NOT NULL), 4) AS accuracy
FROM predictions
WHERE prediction_type IN ('correct_score', 'double_chance')
GROUP BY prediction_type;
SQLEOF
docker compose cp /tmp/regrade_check.sql postgres:/tmp/regrade_check.sql
docker compose exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/regrade_check.sql'
