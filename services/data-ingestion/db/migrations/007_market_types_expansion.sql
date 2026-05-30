-- Widen market_type / prediction_type vocabularies for the analytic
-- soccer markets derived from the Dixon-Coles scoreline distribution,
-- and allow one prediction row PER MARKET per (match, model, version).
--
-- Background: until now soccer produced a single 1x2 prediction
-- (prediction_type='match_result'). The market-derivation engine
-- (services/ml-models/src/predictors/market_derivation.py) now turns the
-- Dixon-Coles joint score matrix into ~15 markets — double chance,
-- draw-no-bet, over/under (all lines), BTTS, correct score, Asian
-- handicap, team totals, clean sheet, win-to-nil, odd/even, total-goals
-- bands, winning margin, and result+BTTS / result+O/U combos.
--
-- Three changes:
--   (a) odds.market_type CHECK — accept the new bookmaker-sourced +
--       derived market names (so fetch_live_odds can store them and the
--       feature/recommendation paths can reference them).
--   (b) predictions.prediction_type CHECK — same vocabulary the engine
--       emits. The 1x2 headline stays 'match_result' for backward
--       compatibility with the auto-grade trigger + vw_active_recommendations.
--   (c) Replace the (match_id, model_name, model_version) unique
--       constraint from 005 with one that also keys on prediction_type,
--       so the per-market upserts in precompute_predictions.py don't
--       clobber each other. precompute_predictions.store_prediction and
--       api .../prediction_service._store_prediction are updated in the
--       same change to ON CONFLICT on the new 4-column key.
--
-- Postgres can't ALTER a CHECK in place, so we drop and re-add. The 001
-- constraints are inline (unnamed) → Postgres auto-named them
-- <table>_<column>_check; DROP ... IF EXISTS tolerates either name.

BEGIN;

-- (a) odds.market_type
ALTER TABLE odds DROP CONSTRAINT IF EXISTS odds_market_type_check;
ALTER TABLE odds ADD CONSTRAINT odds_market_type_check
    CHECK (market_type IN (
        -- existing
        '1x2', 'over_under', 'btts', 'asian_handicap', 'correct_score',
        'moneyline', 'spread', 'total', 'player_prop', 'first_scorer', 'half_time',
        -- new (derived soccer markets + secondary bookmaker markets)
        'double_chance', 'draw_no_bet', 'team_total', 'clean_sheet',
        'win_to_nil', 'odd_even', 'winning_margin', 'total_goals',
        'result_btts', 'result_over_under'
    ));

-- (b) predictions.prediction_type
ALTER TABLE predictions DROP CONSTRAINT IF EXISTS predictions_prediction_type_check;
ALTER TABLE predictions ADD CONSTRAINT predictions_prediction_type_check
    CHECK (prediction_type IN (
        -- existing
        'match_result', 'over_under', 'btts', 'correct_score', 'player_prop',
        'moneyline', 'spread', 'total', 'lottery',
        -- new
        'double_chance', 'draw_no_bet', 'asian_handicap', 'team_total',
        'clean_sheet', 'win_to_nil', 'odd_even', 'winning_margin',
        'total_goals', 'result_btts', 'result_over_under'
    ));

-- (c) Unique constraint now includes prediction_type. Dedupe any rows
--     that would violate the new key first (keep the most recent),
--     mirroring the 005 pattern.
WITH ranked AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY match_id, model_name, model_version, prediction_type
            ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST, id DESC
        ) AS rn
    FROM predictions
)
DELETE FROM predictions
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);

ALTER TABLE predictions
    DROP CONSTRAINT IF EXISTS predictions_match_model_version_key;
ALTER TABLE predictions
    ADD CONSTRAINT predictions_match_model_version_type_key
    UNIQUE (match_id, model_name, model_version, prediction_type);

COMMIT;
