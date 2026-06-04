-- 017: Hockey analytic market types from derive_hockey_markets
--
-- The new NHL Dixon-Coles model + derive_hockey_markets engine (PR
-- #20) produces analytic markets the existing trained ensemble
-- doesn't cover. Adds the prediction_type values they store under.
-- Moneyline / spread / total already exist in the CHECK list; the
-- new ones are regulation_1x2 and the soccer-shared "shape" markets
-- that hockey reuses (clean_sheet, win_to_nil, double_chance,
-- correct_score, total_goals — already in the CHECK from migration
-- 007, so nothing to add for those).

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
        'match_result_ht', 'over_under_ht', 'btts_ht',
        'ht_ft_double_result',
        -- hockey analytic markets (migration 017)
        'regulation_1x2', 'puck_line'
    ));

COMMIT;
