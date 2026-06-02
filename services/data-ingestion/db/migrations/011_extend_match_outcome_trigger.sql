-- Phase 10b: Extend calculate_match_outcome trigger to cover all
-- currently-supported sport markets (NHL, NBA, NFL) — not just soccer.
--
-- Before this migration, the trigger only graded soccer markets
-- (match_result, over_under, btts). Everything else relied on the
-- 15-minute Python pipeline (scripts/grade_completed_matches.py) for
-- grading. That meant NHL/NBA/NFL picks sat with is_correct=NULL for
-- up to 15 minutes after the game finished — usually fine but
-- annoying when verifying that a pick won/lost during a live demo.
--
-- This migration extends the trigger to grade every market the Python
-- script knows about:
--   soccer: match_result, over_under, btts (unchanged)
--   NHL:    moneyline, regulation, puck_line, total
--   NBA:    moneyline, spread, total (line-as-feature via features_cache)
--   NFL:    moneyline, spread, total (line-as-feature via features_cache)
--
-- Per-sport SQL helpers keep the dispatch readable. The Python script
-- in grade_completed_matches.py remains the comprehensive backup —
-- it also handles re-grading predictions written AFTER the status
-- flip (the trigger only fires on the status transition itself).
--
-- Reversibility: revert by dropping the new helpers + trigger and
-- re-installing the previous (soccer-only) calculate_match_outcome
-- from migration 003. The trigger is a pure optimization on top of
-- grade_completed_matches.py; reverting just adds back the 15-min
-- grading lag.

-- ============================================================
-- Per-sport, per-market grading helper functions
-- ============================================================
--
-- All helpers return TEXT for actual_outcome and accept NULLs
-- defensively — a missing feature or score means "leave ungraded"
-- (NULL), which the trigger interprets as "skip is_correct update,
-- try again next pipeline tick". The Python script handles the
-- catch-up grading.

-- Soccer: 1X2 three-way result.
CREATE OR REPLACE FUNCTION grade_match_result(home_score INT, away_score INT)
RETURNS TEXT AS $$
BEGIN
    IF home_score IS NULL OR away_score IS NULL THEN
        RETURN NULL;
    END IF;
    IF home_score > away_score THEN RETURN 'home'; END IF;
    IF home_score < away_score THEN RETURN 'away'; END IF;
    RETURN 'draw';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- NHL/NBA/NFL moneyline: 2-way winner. Ties → NULL (leave ungraded
-- since 2-class classifier can't represent them). NHL never ties in
-- the moneyline market post-OT/SO; NFL rarely does (~1-2/year);
-- NBA never does.
CREATE OR REPLACE FUNCTION grade_moneyline_2way(home_score INT, away_score INT)
RETURNS TEXT AS $$
BEGIN
    IF home_score IS NULL OR away_score IS NULL THEN
        RETURN NULL;
    END IF;
    IF home_score > away_score THEN RETURN 'home'; END IF;
    IF home_score < away_score THEN RETURN 'away'; END IF;
    RETURN NULL;  -- tie: ungradable
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- NHL regulation: 3-way (home/away/tie) read from
-- matches.metadata.regulation_winner — written by the NHL ingest
-- loader, captures the 60-minute result before OT/SO.
CREATE OR REPLACE FUNCTION grade_nhl_regulation(metadata JSONB)
RETURNS TEXT AS $$
BEGIN
    RETURN metadata->>'regulation_winner';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- NHL puck line: fixed -1.5 home / +1.5 away. Cover = home wins by
-- ≥ 2 goals. ±1.5 never pushes.
CREATE OR REPLACE FUNCTION grade_nhl_puck_line(home_score INT, away_score INT)
RETURNS TEXT AS $$
BEGIN
    IF home_score IS NULL OR away_score IS NULL THEN
        RETURN NULL;
    END IF;
    IF (home_score - away_score) >= 2 THEN
        RETURN 'cover';
    END IF;
    RETURN 'no_cover';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- NHL total: fixed over/under 5.5 goals. 5.5 never pushes.
CREATE OR REPLACE FUNCTION grade_nhl_total(home_score INT, away_score INT)
RETURNS TEXT AS $$
BEGIN
    IF home_score IS NULL OR away_score IS NULL THEN
        RETURN NULL;
    END IF;
    IF (home_score + away_score) > 5.5 THEN
        RETURN 'over';
    END IF;
    RETURN 'under';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- NBA/NFL spread (line-as-feature): home covers iff margin > -line.
-- `line` is the home-perspective spread (-7 = home favored by 7).
-- Integer lines can push; half-point lines never do. Same math as
-- scripts/grading_outcomes.py::nba_spread_outcome.
CREATE OR REPLACE FUNCTION grade_spread_with_line(home_score INT, away_score INT, line NUMERIC)
RETURNS TEXT AS $$
DECLARE
    margin INT;
BEGIN
    IF home_score IS NULL OR away_score IS NULL OR line IS NULL THEN
        RETURN NULL;
    END IF;
    margin := home_score - away_score;
    IF margin::NUMERIC > -line THEN RETURN 'home'; END IF;
    IF margin::NUMERIC < -line THEN RETURN 'away'; END IF;
    RETURN 'push';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- NBA/NFL total (line-as-feature): over iff (home + away) > line.
-- Integer lines (rare in NBA, common in NFL on key numbers) can push.
CREATE OR REPLACE FUNCTION grade_total_with_line(home_score INT, away_score INT, line NUMERIC)
RETURNS TEXT AS $$
DECLARE
    total INT;
BEGIN
    IF home_score IS NULL OR away_score IS NULL OR line IS NULL THEN
        RETURN NULL;
    END IF;
    total := home_score + away_score;
    IF total::NUMERIC > line THEN RETURN 'over'; END IF;
    IF total::NUMERIC < line THEN RETURN 'under'; END IF;
    RETURN 'push';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================
-- New calculate_match_outcome trigger function
-- ============================================================
--
-- Per-row grading inside an UPDATE statement on predictions. The
-- per-prediction dispatch picks the right helper based on
-- (prediction_type, model_name). model_name disambiguates shared
-- prediction_types — 'spread' is NHL puck line OR NBA/NFL covering
-- the closing line depending on which ensemble wrote the row.
--
-- For line-as-feature markets (NBA/NFL spread/total) we JOIN to
-- features_cache. The freshest non-expired row by computed_at wins,
-- matching the Python script's fetch_match_features helper.

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
      AND p.is_correct IS NULL;  -- never re-grade already-graded rows

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
      AND p.is_correct IS NOT NULL;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- The trigger itself stays the same shape — only the function body
-- changed. Re-declare for clarity / idempotency.
DROP TRIGGER IF EXISTS trg_match_outcome ON matches;
CREATE TRIGGER trg_match_outcome
    AFTER UPDATE ON matches
    FOR EACH ROW
    WHEN (NEW.status = 'finished' AND OLD.status <> 'finished')
    EXECUTE FUNCTION calculate_match_outcome();
