-- 020: De-duplicate races + fix the dedup key (racing_api_race_id)
--
-- BUG: scripts/load_racing_api.py upserted races with
--   ON CONFLICT (track_name, race_date, race_number)
-- but the /racecards/standard API feed leaves race_number NULL. In
-- Postgres, NULLs are DISTINCT in a unique index, so that conflict
-- target NEVER matched for these rows. The auspex_pipeline DAG runs
-- the upcoming-races ingest every 15 minutes and re-fetches the same
-- same-day cards each time, so every run INSERTED fresh duplicate race
-- rows. One real race (e.g. racing_api_race_id rac_11986871) became
-- dozens of rows; the races table held ~70k rows for ~13k real races.
--
-- IMPACT: recommendations attached to whichever duplicate the recs
-- pipeline happened to write, those duplicates never matched the
-- results-ingest pass so they never transitioned scheduled->finished,
-- and the recs therefore never settled (all pending). Horse-racing
-- recs were effectively invisible + ungradeable.
--
-- FIX: (1) collapse duplicates to one survivor per racing_api_race_id;
--      (2) add a unique index on racing_api_race_id so future upserts
--      dedup correctly (the code's ON CONFLICT now targets it).
--
-- A pre-migration backup was taken (OPERATIONS.md / backups/).
-- All race children (race_entrants, race_predictions,
-- race_recommendations, race feature rows) are ON DELETE CASCADE on
-- races(id), so deleting a duplicate race row removes its children
-- automatically — no manual FK re-pointing needed.

BEGIN;

-- Step 1 — collapse duplicates. Survivor preference per
-- racing_api_race_id, chosen to PRESERVE graded history:
--   1. finished status (carries results, gradeable) over scheduled
--   2. most graded predictions  (keep the row the grader wrote to)
--   3. most predictions overall  (keep the most complete model row)
--   4. most-recently-updated, then created_at, then id (determinism)
-- Verified on prod before running: under this ordering, 0 of the
-- 11,552 race_ids that have graded prediction history would lose it
-- (the "lost" graded preds are all redundant copies of the same race
-- graded on multiple duplicates). Pending recommendations (all
-- unsettled, on scheduled duplicates) are dropped by the cascade and
-- regenerate on the next recs DAG run against the surviving row.
-- Everything else cascades (race_entrants / race_predictions /
-- race_recommendations / race feature rows are ON DELETE CASCADE).
-- Races with no racing_api_race_id are left untouched (the API always
-- provides one; this only guards a theoretical edge case).
WITH pred_stats AS (
    SELECT
        race_id,
        COUNT(*) FILTER (WHERE actual_outcome IS NOT NULL) AS graded_cnt,
        COUNT(*) AS pred_cnt
    FROM race_predictions
    GROUP BY race_id
),
ranked AS (
    SELECT
        r.id,
        ROW_NUMBER() OVER (
            PARTITION BY r.external_ids->>'racing_api_race_id'
            ORDER BY
                (r.status = 'finished') DESC,
                COALESCE(ps.graded_cnt, 0) DESC,
                COALESCE(ps.pred_cnt, 0) DESC,
                r.updated_at DESC, r.created_at DESC, r.id
        ) AS rn
    FROM races r
    LEFT JOIN pred_stats ps ON ps.race_id = r.id
    WHERE r.external_ids->>'racing_api_race_id' IS NOT NULL
)
DELETE FROM races
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);

-- Step 2 — unique index on the stable racing-API race id. Partial so
-- it only covers rows that actually have the id (matches the code's
-- ON CONFLICT inference predicate). After step 1 there is exactly one
-- row per racing_api_race_id, so this builds cleanly.
CREATE UNIQUE INDEX IF NOT EXISTS idx_races_racing_api_id
    ON races ((external_ids->>'racing_api_race_id'))
    WHERE external_ids->>'racing_api_race_id' IS NOT NULL;

COMMIT;
