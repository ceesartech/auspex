-- 027: the match-outcome trigger no longer settles hindsight
--
-- WHY: calculate_match_outcome (migration 011) fires once, when a match flips
-- to 'finished'. It graded EVERY ungraded prediction on the match and marked
-- EVERY pending/placed rec won/lost from its prediction -- including rows
-- written at or after kickoff. Those are hindsight, not picks or bets:
-- scripts/grade_completed_matches.py skips such predictions and voids such
-- recs (actual_result 'void: written after kickoff'). But the trigger runs
-- first, and the grader only revisits a rec the trigger already flipped when
-- its match is inside the grader's --days window (14 in auspex_pipeline);
-- outside it, a post-kickoff rec stayed won/lost (P&L NULL) and a
-- post-kickoff prediction stayed graded, both counted in the rollups.
-- Latent on prod as of 2026-10-31: the post-kickoff rows there trace to
-- backfills and fixture repairs, not to this trigger. Any rec or prediction
-- written while a match is in play would have been settled by it.
--
-- Now the trigger leaves both alone: a post-kickoff rec stays 'pending' for
-- the Python grader to void, a post-kickoff prediction stays ungraded. Same
-- comparisons as the grader, so the two never disagree about a row:
--   predictions  created_at >  match_date  (list_ungraded_predictions)
--   recs         created_at >= match_date  (list_open_recs_for_match)
-- IS NOT TRUE rather than <= / <: a NULL created_at is not "after kickoff"
-- in the grader either (its after_kickoff is falsy), so it still settles.
--
-- Everything else in the function body is migration 011's, verbatim (prod's
-- live definition matched 011 byte for byte -- pg_get_functiondef md5
-- 24b0d94b... -- on 2026-10-31). Only the function body changes: CREATE OR
-- REPLACE keeps its OID, so trg_match_outcome keeps firing it, and the
-- trigger is NOT re-declared -- that would lock matches for no reason.
-- Idempotent; one transaction; locks no table. lock_timeout only stops it
-- queueing on the catalog row -- if it times out, nothing changed; re-run.
--
-- Rows the old trigger already settled are untouched here:
-- scripts/ops/void_post_kickoff_recs.sh re-settles those recs.
--
-- KNOWN GAP (accepted -- pending beats wrongly settled): the grader voids a
-- post-kickoff rec only when it visits the match, i.e. when the match's
-- kickoff is inside its --days window. A match that flips to 'finished'
-- after aging out of that window (a results backlog after an outage, a
-- repair flipping old rows) leaves its post-kickoff recs 'pending' with no
-- scheduled run to void them; they stay in the dashboard's active_recs, and
-- void_post_kickoff_recs.py selects only settled recs. Under 011 the same
-- recs were marked won/lost instead. grade_completed_matches.py checks for
-- them after every run (warn_stuck_post_kickoff_recs) and logs ERROR with
-- the count, the first ids and the command that voids them: an operator run
--   python /app/scripts/grade_completed_matches.py --days N
-- with N reaching the oldest such kickoff (the ERROR line gives N).
--
-- REVERSIBLE: re-apply 011_extend_match_outcome_trigger.sql (idempotent:
-- CREATE OR REPLACE of the helpers and this function, then re-declares the
-- trigger).

BEGIN;
SET LOCAL lock_timeout = '10s';

CREATE OR REPLACE FUNCTION calculate_match_outcome()
RETURNS TRIGGER AS $$
DECLARE
    closing_spread NUMERIC;
    closing_total  NUMERIC;
BEGIN
    -- Only fire on the transition into 'finished'.
    IF NEW.status <> 'finished' OR OLD.status = 'finished' THEN
        RETURN NEW;
    END IF;

    -- Cache the closing lines for this match once. The same line
    -- applies to every NBA/NFL spread/total prediction row, so a
    -- single lookup beats N JOINs in the UPDATE below.
    SELECT
        (features->>'closing_spread_home')::NUMERIC,
        (features->>'closing_total_line')::NUMERIC
    INTO closing_spread, closing_total
    FROM features_cache
    WHERE match_id = NEW.id
      AND feature_set IN ('nba_baseline', 'nfl_baseline')
    ORDER BY computed_at DESC
    LIMIT 1;

    UPDATE predictions p
    SET
        actual_outcome = CASE
            -- Soccer 1X2 OR NHL regulation (both prediction_type='match_result').
            WHEN p.prediction_type = 'match_result' AND p.model_name = 'ensemble_nhl_reg' THEN
                grade_nhl_regulation(NEW.metadata)
            WHEN p.prediction_type = 'match_result' THEN
                grade_match_result(NEW.home_score, NEW.away_score)

            -- Moneyline: all sports use final-score winner.
            WHEN p.prediction_type = 'moneyline' THEN
                grade_moneyline_2way(NEW.home_score, NEW.away_score)

            -- Spread: NBA/NFL use variable line from features_cache;
            -- NHL uses fixed puck-line -1.5.
            WHEN p.prediction_type = 'spread' AND p.model_name IN ('ensemble_nba_sp', 'ensemble_nfl_sp') THEN
                grade_spread_with_line(NEW.home_score, NEW.away_score, closing_spread)
            WHEN p.prediction_type = 'spread' THEN
                grade_nhl_puck_line(NEW.home_score, NEW.away_score)

            -- Total: NBA/NFL use variable line from features_cache;
            -- NHL uses fixed over/under 5.5.
            WHEN p.prediction_type = 'total' AND p.model_name IN ('ensemble_nba_tot', 'ensemble_nfl_tot') THEN
                grade_total_with_line(NEW.home_score, NEW.away_score, closing_total)
            WHEN p.prediction_type = 'total' THEN
                grade_nhl_total(NEW.home_score, NEW.away_score)

            -- Soccer over_under (legacy, line embedded in predicted_outcome).
            WHEN p.prediction_type = 'over_under' AND p.predicted_outcome LIKE 'over_%' THEN
                CASE WHEN (NEW.home_score + NEW.away_score)::NUMERIC
                          > SPLIT_PART(p.predicted_outcome, '_', 2)::NUMERIC
                     THEN 'over' ELSE 'under' END
            WHEN p.prediction_type = 'over_under' AND p.predicted_outcome LIKE 'under_%' THEN
                CASE WHEN (NEW.home_score + NEW.away_score)::NUMERIC
                          < SPLIT_PART(p.predicted_outcome, '_', 2)::NUMERIC
                     THEN 'under' ELSE 'over' END

            -- Soccer BTTS.
            WHEN p.prediction_type = 'btts' THEN
                CASE WHEN NEW.home_score > 0 AND NEW.away_score > 0
                     THEN 'yes' ELSE 'no' END

            ELSE NULL  -- ungraded — Python script picks it up next tick
        END,

        is_correct = CASE
            WHEN p.prediction_type = 'match_result' AND p.model_name = 'ensemble_nhl_reg' THEN
                p.predicted_outcome = grade_nhl_regulation(NEW.metadata)
            WHEN p.prediction_type = 'match_result' THEN
                p.predicted_outcome = grade_match_result(NEW.home_score, NEW.away_score)

            WHEN p.prediction_type = 'moneyline' THEN
                p.predicted_outcome = grade_moneyline_2way(NEW.home_score, NEW.away_score)

            WHEN p.prediction_type = 'spread' AND p.model_name IN ('ensemble_nba_sp', 'ensemble_nfl_sp') THEN
                CASE
                    -- Push leaves is_correct NULL (handled by Python settler for rec status='void').
                    WHEN grade_spread_with_line(NEW.home_score, NEW.away_score, closing_spread) = 'push' THEN NULL
                    ELSE p.predicted_outcome = grade_spread_with_line(NEW.home_score, NEW.away_score, closing_spread)
                END
            WHEN p.prediction_type = 'spread' THEN
                p.predicted_outcome = grade_nhl_puck_line(NEW.home_score, NEW.away_score)

            WHEN p.prediction_type = 'total' AND p.model_name IN ('ensemble_nba_tot', 'ensemble_nfl_tot') THEN
                CASE
                    WHEN grade_total_with_line(NEW.home_score, NEW.away_score, closing_total) = 'push' THEN NULL
                    ELSE p.predicted_outcome = grade_total_with_line(NEW.home_score, NEW.away_score, closing_total)
                END
            WHEN p.prediction_type = 'total' THEN
                p.predicted_outcome = grade_nhl_total(NEW.home_score, NEW.away_score)

            WHEN p.prediction_type = 'over_under' AND p.predicted_outcome LIKE 'over_%' THEN
                (NEW.home_score + NEW.away_score)::NUMERIC
                    > SPLIT_PART(p.predicted_outcome, '_', 2)::NUMERIC
            WHEN p.prediction_type = 'over_under' AND p.predicted_outcome LIKE 'under_%' THEN
                (NEW.home_score + NEW.away_score)::NUMERIC
                    < SPLIT_PART(p.predicted_outcome, '_', 2)::NUMERIC

            WHEN p.prediction_type = 'btts' THEN
                (p.predicted_outcome = 'yes' AND NEW.home_score > 0 AND NEW.away_score > 0)
                OR (p.predicted_outcome = 'no' AND (NEW.home_score = 0 OR NEW.away_score = 0))

            ELSE NULL
        END,
        updated_at = NOW()
    WHERE p.match_id = NEW.id
      AND p.is_correct IS NULL  -- never re-grade already-graded rows
      -- 027: written after kickoff = hindsight, never graded (the grader's
      -- after_kickoff rule; see the migration header).
      AND (p.created_at > NEW.match_date) IS NOT TRUE;

    -- Settle betting_recommendations whose underlying prediction just
    -- got graded. won/lost only — Push handling stays with the Python
    -- settler since it needs to set status='void' explicitly.
    UPDATE betting_recommendations br
    SET status = CASE
            WHEN p.is_correct = true  THEN 'won'
            WHEN p.is_correct = false THEN 'lost'
            ELSE br.status
        END,
        actual_result = p.actual_outcome,
        settled_at = NOW(),
        updated_at = NOW()
    FROM predictions p
    WHERE br.prediction_id = p.id
      AND p.match_id = NEW.id
      AND br.status IN ('pending', 'placed')
      AND p.is_correct IS NOT NULL
      -- 027: a rec written at or after kickoff is not a bet. Leave it
      -- pending: the grader voids it (POST_KICKOFF_VOID_RESULT).
      AND (br.created_at >= NEW.match_date) IS NOT TRUE;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

COMMIT;
