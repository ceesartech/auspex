-- 015: Halftime/fulltime double-result market
--
-- The joint HT × FT outcome market derived from convolving the HT
-- Dixon-Coles scoreline matrix (migration 014) with a SECOND HALF
-- Dixon-Coles scoreline matrix (model artifact at
-- /app/models/production/dixon_coles_h2_soccer/). 9 selections:
-- (home/draw/away at HT) × (home/draw/away at FT).
--
-- The bookmaker odds CHECK is NOT extended here — Path 3 (HT odds
-- ingestion) will add the matching odds.market_type entries when
-- that lands.

BEGIN;

ALTER TABLE predictions DROP CONSTRAINT IF EXISTS predictions_prediction_type_check;
ALTER TABLE predictions ADD CONSTRAINT predictions_prediction_type_check
    CHECK (prediction_type IN (
        -- existing
        'match_result', 'over_under', 'btts', 'correct_score', 'player_prop',
        'moneyline', 'spread', 'total', 'lottery',
        'double_chance', 'draw_no_bet', 'asian_handicap', 'team_total',
        'clean_sheet', 'win_to_nil', 'odd_even', 'winning_margin',
        'total_goals', 'result_btts', 'result_over_under',
        -- halftime soccer markets (migration 014)
        'match_result_ht', 'over_under_ht', 'btts_ht',
        -- halftime/fulltime joint market (migration 015)
        'ht_ft_double_result'
    ));

COMMIT;
