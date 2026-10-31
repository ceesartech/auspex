-- 025: deleting a prediction no longer deletes the recommendations on it
--
-- WHY (2026-10-31 tennis incident): betting_recommendations.prediction_id was
-- ON DELETE CASCADE. scripts/repair_stranded_recs.py re-pointed 50 settled
-- tennis recs' match_id onto the finished twin fixture, but their
-- prediction_id still named a prediction on the orphan row; when
-- scripts/repair_fixture_orphans.py deleted the orphans, the cascade deleted
-- the predictions and 45 of the recs with them -- silently, money-path rows,
-- found only by an after-state check by rec id (restored the same night by
-- scripts/restore_cascaded_recs.py). The tool's guard is fixed (5682c90); this
-- makes the whole class fail LOUDLY for every future writer instead.
--
-- NO ACTION, not RESTRICT: NO ACTION is checked at the END of the statement,
-- so a delete that removes a rec and its prediction together (deleting a
-- match cascades to both through their match_id FKs) still works; only a
-- statement that would leave a rec pointing at a deleted prediction fails.
-- RESTRICT checks at the moment each prediction row goes, before the sibling
-- match_id cascade may have removed the rec, so whether it fired would
-- depend on trigger order.
--
-- betting_recommendations.match_id stays ON DELETE CASCADE (deleting a match
-- still deletes its recs); scripts/repair_fixture_orphans.py guards that one.
--
-- Rewrites no row. Re-validating the FK scans betting_recommendations (1,817
-- rows on 2026-10-31) under a brief ACCESS EXCLUSIVE lock; lock_timeout keeps
-- it from queueing behind a long pipeline transaction -- if it times out,
-- nothing changed; re-run it.
--
-- REVERSIBLE:
--   ALTER TABLE betting_recommendations
--       DROP CONSTRAINT betting_recommendations_prediction_id_fkey,
--       ADD CONSTRAINT betting_recommendations_prediction_id_fkey
--           FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE;

BEGIN;
SET LOCAL lock_timeout = '10s';

ALTER TABLE betting_recommendations
    DROP CONSTRAINT IF EXISTS betting_recommendations_prediction_id_fkey,
    ADD CONSTRAINT betting_recommendations_prediction_id_fkey
        FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE NO ACTION;

COMMIT;
