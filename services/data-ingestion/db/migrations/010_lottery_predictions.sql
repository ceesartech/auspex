-- Lottery combination predictions + backtest results.
--
-- Lottery lines don't fit the `predictions` table (which FKs `matches`), so
-- generated combinations get their own table. Each row is one combination the
-- lottery engine (services/api/src/services/lottery_analysis.py) produced for a
-- given game + strategy, optionally tied to a target draw date so it can be
-- settled against the actual draw and the strategy honestly backtested.
--
-- IMPORTANT: Powerball / Mega Millions draws are uniform and independent — these
-- rows are NOT forecasts of winning numbers. The engine optimizes statistical-
-- profile fit and expected value via jackpot-share avoidance; backtest hit-rates
-- are expected to track pure chance. The `score` is an internal ranking signal,
-- not a probability of winning.

CREATE TABLE IF NOT EXISTS lottery_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    game VARCHAR(50) NOT NULL
        CHECK (game IN ('powerball', 'mega_millions')),
    strategy VARCHAR(30) NOT NULL,
    numbers INTEGER[] NOT NULL,
    bonus_number INTEGER NOT NULL,
    score DECIMAL(6, 4),
    features JSONB DEFAULT '{}',
    rationale TEXT,
    -- The draw this line is "for" (typically the next scheduled draw). NULL for
    -- ad-hoc generations the user didn't tie to a draw.
    target_draw_date DATE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    -- Settlement — filled by scripts/lottery_backtest.py once target_draw_date
    -- has an actual lottery_draws row.
    matched_main INTEGER,
    matched_bonus BOOLEAN,
    prize_tier VARCHAR(30),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    settled_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_lottery_predictions_game_target
    ON lottery_predictions (game, target_draw_date);
CREATE INDEX IF NOT EXISTS idx_lottery_predictions_game_created
    ON lottery_predictions (game, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lottery_predictions_strategy
    ON lottery_predictions (game, strategy);
