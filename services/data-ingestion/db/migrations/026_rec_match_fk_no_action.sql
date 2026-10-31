-- 026: deleting a match no longer deletes the recommendations on it
--
-- WHY: the second half of the 2026-10-31 tennis incident (see 025). After 025,
-- betting_recommendations.match_id was the last FK through which a match
-- delete could silently remove money-path rows: ON DELETE CASCADE meant any
-- DELETE FROM matches -- a dedupe, an orphan cleanup with a broken guard, a
-- by-hand fix -- took that match's recs, settled P&L included, with it.
-- scripts/repair_fixture_orphans.py guards this in code; this makes the
-- schema refuse it for every writer. A tool that really means to remove a
-- fixture with recs must now re-point or delete those recs FIRST, explicitly.
--
-- After 026 deleting a match that holds a rec fails (025's note that such a
-- delete "still works" is superseded). The other tables keyed on match_id
-- (odds, odds_snapshots, predictions, features_cache, stats, weather) keep
-- cascading -- they are derived or re-fetchable; betting_history already uses
-- SET NULL and holds no rows.
--
-- NO ACTION (checked at end of statement), consistent with 025. Rewrites no
-- row; re-validation scans betting_recommendations (1,819 rows on
-- 2026-10-31) under a brief lock; lock_timeout keeps it from queueing behind
-- a long pipeline transaction -- if it times out, nothing changed; re-run.
--
-- REVERSIBLE:
--   ALTER TABLE betting_recommendations
--       DROP CONSTRAINT betting_recommendations_match_id_fkey,
--       ADD CONSTRAINT betting_recommendations_match_id_fkey
--           FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE;

BEGIN;
SET LOCAL lock_timeout = '10s';

ALTER TABLE betting_recommendations
    DROP CONSTRAINT IF EXISTS betting_recommendations_match_id_fkey,
    ADD CONSTRAINT betting_recommendations_match_id_fkey
        FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE NO ACTION;

COMMIT;
