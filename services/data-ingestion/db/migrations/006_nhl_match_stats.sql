-- NHL-specific per-team match statistics.
--
-- The existing `match_stats` table is soccer-shaped (possession, xG, fouls,
-- cards, PPDA, etc.) and reusing it for hockey would either pollute it with
-- nullable hockey columns or force us to overload `metadata` JSONB for
-- structured analytics. A parallel `nhl_match_stats` table keeps both
-- sport pipelines independent until we extract a shared abstraction.
--
-- Game-level tiebreak state (regulation/OT/SO winners) lives in
-- `matches.metadata` as JSON keys: regulation_winner, period_type_final,
-- ot_winner, so_winner — kept off the columnar schema since it's only
-- read by the NHL feature pipeline.

CREATE TABLE IF NOT EXISTS nhl_match_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    -- Shooting
    shots_on_goal INTEGER DEFAULT 0,
    shot_attempts INTEGER DEFAULT 0,
    blocked_shots INTEGER DEFAULT 0,
    missed_shots INTEGER DEFAULT 0,
    goals INTEGER DEFAULT 0,
    -- Goaltending
    saves INTEGER DEFAULT 0,
    save_pct DECIMAL(5,4),
    goals_against INTEGER DEFAULT 0,
    -- Special teams
    power_play_goals INTEGER DEFAULT 0,
    power_play_opportunities INTEGER DEFAULT 0,
    power_play_pct DECIMAL(5,4),
    short_handed_goals INTEGER DEFAULT 0,
    penalty_kill_goals_against INTEGER DEFAULT 0,
    penalty_kill_opportunities INTEGER DEFAULT 0,
    penalty_kill_pct DECIMAL(5,4),
    -- Physical / possession
    hits INTEGER DEFAULT 0,
    faceoffs_won INTEGER DEFAULT 0,
    faceoffs_taken INTEGER DEFAULT 0,
    faceoff_win_pct DECIMAL(5,4),
    giveaways INTEGER DEFAULT 0,
    takeaways INTEGER DEFAULT 0,
    pim INTEGER DEFAULT 0,
    -- Goalie
    starting_goalie_id UUID REFERENCES players(id) ON DELETE SET NULL,
    -- Period-by-period scoring (e.g., {"1": 1, "2": 0, "3": 2, "OT": 1})
    period_scores JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(match_id, team_id)
);

CREATE INDEX IF NOT EXISTS idx_nhl_match_stats_match
    ON nhl_match_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_nhl_match_stats_team
    ON nhl_match_stats(team_id);
