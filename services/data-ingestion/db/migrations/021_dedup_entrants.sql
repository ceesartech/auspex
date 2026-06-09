-- 021: De-duplicate race_entrants + fix the dedup key (race_id, horse_id)
--
-- BUG (same class as migration 020's race duplication): the entrant
-- upsert in scripts/load_racing_api.py used
--   ON CONFLICT (race_id, program_number)
-- but program_number is `_safe_int(...) or 0`, so a runner the
-- /racecards feed omits a number for lands under program_number 0.
-- The same horse could land under 0 in one ingest pass and its real
-- number in another → two race_entrants rows for one horse in one
-- race (e.g. "Minzazi" appeared at program 0 and program 3).
--
-- Small blast radius: only 16 duplicate (race_id, horse_id) groups
-- and 82 program_number=0 rows out of ~151k entrants — but worth
-- fixing so it can't accumulate and so a horse never shows twice in
-- the race-card UI / field_size count.
--
-- FIX: (1) collapse duplicates to one survivor per (race_id, horse_id);
--      (2) replace the unreliable UNIQUE(race_id, program_number) with
--      a UNIQUE index on the stable (race_id, horse_id). The code's
--      ON CONFLICT now targets (race_id, horse_id).
--
-- All entrant children (race_predictions, race_recommendations) are
-- ON DELETE CASCADE on race_entrants(id), so deleting a duplicate
-- entrant removes its (redundant) predictions/recs automatically.

BEGIN;

-- Step 1 — collapse duplicate entrants. Survivor preference per
-- (race_id, horse_id): a real (non-zero) program_number wins over the
-- 0 placeholder; then the row with the most graded predictions; then
-- most predictions; then lower post_position; then id. The dropped
-- duplicates' predictions are redundant copies of the same horse's
-- consensus, so no unique grading is lost.
WITH pred_stats AS (
    SELECT
        entrant_id,
        COUNT(*) FILTER (WHERE actual_outcome IS NOT NULL) AS graded_cnt,
        COUNT(*) AS pred_cnt
    FROM race_predictions
    GROUP BY entrant_id
),
ranked AS (
    SELECT
        e.id,
        ROW_NUMBER() OVER (
            PARTITION BY e.race_id, e.horse_id
            ORDER BY
                (e.program_number > 0) DESC,
                COALESCE(ps.graded_cnt, 0) DESC,
                COALESCE(ps.pred_cnt, 0) DESC,
                e.post_position NULLS LAST,
                e.id
        ) AS rn
    FROM race_entrants e
    LEFT JOIN pred_stats ps ON ps.entrant_id = e.id
)
DELETE FROM race_entrants
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);

-- Step 2 — swap the unique key. The old UNIQUE(race_id, program_number)
-- is unreliable (program_number 0 collides distinct no-number horses);
-- the stable identity is (race_id, horse_id) — a horse runs once per
-- race. After step 1 there is exactly one row per (race_id, horse_id).
ALTER TABLE race_entrants DROP CONSTRAINT IF EXISTS race_entrants_race_id_program_number_key;
CREATE UNIQUE INDEX IF NOT EXISTS uq_race_entrants_race_horse
    ON race_entrants (race_id, horse_id);

COMMIT;
