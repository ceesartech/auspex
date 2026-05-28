-- Make ON CONFLICT (match_id, model_name, model_version) work on the
-- predictions table.
--
-- precompute_predictions.py and the predict_match route both upsert
-- predictions keyed by (match_id, model_name, model_version) so a re-run
-- of the pipeline replaces stale rows instead of accumulating one row
-- per execution. The original schema (001_create_schema.sql) didn't
-- create a unique constraint on that triple, so the upserts crash with:
--
--   psycopg2.errors.InvalidColumnReference: there is no unique or
--   exclusion constraint matching the ON CONFLICT specification
--
-- This migration deduplicates any rows that violate the constraint
-- (keeps the most recent), then adds the constraint.

WITH ranked AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY match_id, model_name, model_version
            ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST, id DESC
        ) AS rn
    FROM predictions
)
DELETE FROM predictions
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);

ALTER TABLE predictions
    ADD CONSTRAINT predictions_match_model_version_key
    UNIQUE (match_id, model_name, model_version);
