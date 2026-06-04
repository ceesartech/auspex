-- 016: Halftime soccer odds market types
--
-- Extends `odds.market_type` CHECK with the three HT market types so
-- the-odds-api's per-event "additional markets" endpoint can store
-- HT odds rows. Matches the naming in `predictions.prediction_type`
-- (migration 014) so the recs engine's market-type join works
-- unchanged.
--
-- Fetching HT odds is OPT-IN at the script layer
-- (scripts/fetch_live_odds.py --additional-markets includes h2h_h1,
-- totals_h1, btts_h1 in the per-event call list) because each
-- additional market costs API quota per (event × region × market).

BEGIN;

ALTER TABLE odds DROP CONSTRAINT IF EXISTS odds_market_type_check;
ALTER TABLE odds ADD CONSTRAINT odds_market_type_check
    CHECK (market_type IN (
        -- existing
        '1x2', 'over_under', 'btts', 'asian_handicap', 'correct_score',
        'moneyline', 'spread', 'total', 'player_prop', 'first_scorer', 'half_time',
        'double_chance', 'draw_no_bet', 'team_total', 'clean_sheet',
        'win_to_nil', 'odd_even', 'winning_margin', 'total_goals',
        'result_btts', 'result_over_under',
        -- halftime soccer markets (migration 016)
        'match_result_ht', 'over_under_ht', 'btts_ht'
    ));

COMMIT;
