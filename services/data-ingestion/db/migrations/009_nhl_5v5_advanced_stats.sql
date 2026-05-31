-- NHL 5v5 advanced stats per team per game.
--
-- Phase 3d showed puck-line and total models can't break through the
-- naive marginal baseline with our v2 feature set (goalies + pace).
-- The canonical signal NHL analytics communities use to predict scoring
-- environments is 5v5 shot-attempt differential (Corsi). Knowing that
-- Team A out-attempted Team B 60-40 at even strength over their last
-- 10 games is a much stronger predictor of future goal margins than
-- raw shot-on-goal averages (which include special teams + empty net).
--
-- This table holds per-team aggregates restricted to 5v5 even-strength
-- play (situationCode='1551' in the NHL play-by-play feed):
--   * shot_attempts_5v5   = Corsi  (SOG + missed + blocked)
--   * unblocked_shots_5v5 = Fenwick (SOG + missed)
--   * shots_on_goal_5v5   = SOG only
--   * goals_5v5           = goals scored at 5v5
--   * xg_5v5              = expected goals at 5v5 from a distance-based
--                           xG approximation (v1 — a properly trained
--                           shot-location model lands in v3.1)
--
-- Separated from nhl_match_stats so the boxscore-derived columns stay
-- compact and the heavier play-by-play backfill can run independently.

CREATE TABLE IF NOT EXISTS nhl_match_advanced_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    -- Raw 5v5 totals
    shot_attempts_5v5 INTEGER DEFAULT 0,
    unblocked_shots_5v5 INTEGER DEFAULT 0,
    shots_on_goal_5v5 INTEGER DEFAULT 0,
    goals_5v5 INTEGER DEFAULT 0,
    -- Expected goals at 5v5 (sum of per-shot xG from the approximation
    -- model — see backfill_nhl_5v5_stats.shot_xg for the formula).
    xg_5v5 DECIMAL(6,3) DEFAULT 0.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(match_id, team_id)
);

CREATE INDEX IF NOT EXISTS idx_nhl_advanced_match
    ON nhl_match_advanced_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_nhl_advanced_team
    ON nhl_match_advanced_stats(team_id);
