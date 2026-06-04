-- 014: Halftime soccer markets — match_result_ht, over_under_ht, btts_ht
--
-- A halftime Dixon-Coles model (trained on matches.half_time_home /
-- matches.half_time_away) produces a HT scoreline matrix. The market
-- derivation engine derives three HT markets per match: 1x2 at HT,
-- over/under HT goals, and BTTS in 1H. Each lands in `predictions`
-- with a HT-specific prediction_type.
--
-- The bookmaker odds CHECK on `odds.market_type` does NOT need updating
-- here — we don't ingest separate HT odds rows from The Odds API in
-- this phase (recs are derived from FT odds against FT predictions).
-- The HT predictions stand alone for prediction quality + future HT
-- odds integration.

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
        'match_result_ht', 'over_under_ht', 'btts_ht'
    ));

COMMIT;
