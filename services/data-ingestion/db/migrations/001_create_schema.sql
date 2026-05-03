-- Betting System Database Schema
-- Migration 001: Create core tables
-- PostgreSQL 15+

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- ============= CORE TABLES =============

-- Leagues table
CREATE TABLE IF NOT EXISTS leagues (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    country VARCHAR(100) NOT NULL,
    sport VARCHAR(50) NOT NULL DEFAULT 'soccer',
    tier INTEGER DEFAULT 1,
    external_ids JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(name, country, sport)
);

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    normalized_name VARCHAR(255) NOT NULL,
    league_id UUID REFERENCES leagues(id) ON DELETE SET NULL,
    country VARCHAR(100),
    sport VARCHAR(50) NOT NULL DEFAULT 'soccer',
    external_ids JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(normalized_name, sport)
);

-- Matches table
CREATE TABLE IF NOT EXISTS matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    league_id UUID REFERENCES leagues(id) ON DELETE CASCADE,
    home_team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    away_team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    match_date TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(20) DEFAULT 'scheduled'
        CHECK (status IN ('scheduled', 'live', 'finished', 'postponed', 'cancelled')),
    home_score INTEGER,
    away_score INTEGER,
    half_time_home INTEGER,
    half_time_away INTEGER,
    season VARCHAR(10) NOT NULL,
    round VARCHAR(50),
    venue VARCHAR(255),
    attendance INTEGER,
    referee_id UUID,
    external_ids JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(home_team_id, away_team_id, match_date)
);

-- Match statistics table
CREATE TABLE IF NOT EXISTS match_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    possession DECIMAL(5,2),
    shots INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    expected_goals DECIMAL(5,3),
    expected_goals_against DECIMAL(5,3),
    passes INTEGER DEFAULT 0,
    pass_accuracy DECIMAL(5,2),
    corners INTEGER DEFAULT 0,
    fouls INTEGER DEFAULT 0,
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    offsides INTEGER DEFAULT 0,
    tackles INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    saves INTEGER DEFAULT 0,
    clearances INTEGER DEFAULT 0,
    blocked_shots INTEGER DEFAULT 0,
    crosses INTEGER DEFAULT 0,
    cross_accuracy DECIMAL(5,2),
    long_balls INTEGER DEFAULT 0,
    through_balls INTEGER DEFAULT 0,
    aerials_won INTEGER DEFAULT 0,
    dribbles_completed INTEGER DEFAULT 0,
    progressive_carries INTEGER DEFAULT 0,
    progressive_passes INTEGER DEFAULT 0,
    deep_completions INTEGER DEFAULT 0,
    ppda DECIMAL(5,2),
    oppda DECIMAL(5,2),
    high_press_success DECIMAL(5,2),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(match_id, team_id)
);

-- ============= ODDS TABLES =============

-- Odds table
CREATE TABLE IF NOT EXISTS odds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) ON DELETE CASCADE,
    bookmaker VARCHAR(50) NOT NULL,
    market_type VARCHAR(50) NOT NULL
        CHECK (market_type IN ('1x2', 'over_under', 'btts', 'asian_handicap',
                               'correct_score', 'moneyline', 'spread', 'total',
                               'player_prop', 'first_scorer', 'half_time')),
    selection VARCHAR(100) NOT NULL,
    odds_decimal DECIMAL(10,4) NOT NULL,
    odds_american INTEGER,
    line DECIMAL(5,2),
    implied_probability DECIMAL(5,4),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_opening BOOLEAN DEFAULT false,
    is_live BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============= PLAYER TABLES =============

-- Players table
CREATE TABLE IF NOT EXISTS players (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    normalized_name VARCHAR(255) NOT NULL,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,
    position VARCHAR(50),
    nationality VARCHAR(100),
    date_of_birth DATE,
    height_cm INTEGER,
    weight_kg INTEGER,
    preferred_foot VARCHAR(10),
    market_value DECIMAL(15,2),
    external_ids JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Player availability (injuries, suspensions)
CREATE TABLE IF NOT EXISTS player_availability (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    player_id UUID REFERENCES players(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL
        CHECK (status IN ('available', 'injured', 'suspended', 'doubtful', 'international_duty', 'personal')),
    reason TEXT,
    start_date DATE NOT NULL,
    expected_return DATE,
    source VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Player match statistics
CREATE TABLE IF NOT EXISTS player_match_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    player_id UUID REFERENCES players(id) ON DELETE CASCADE,
    match_id UUID REFERENCES matches(id) ON DELETE CASCADE,
    minutes_played INTEGER DEFAULT 0,
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    shots INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    key_passes INTEGER DEFAULT 0,
    expected_goals DECIMAL(5,3) DEFAULT 0,
    expected_assists DECIMAL(5,3) DEFAULT 0,
    passes_completed INTEGER DEFAULT 0,
    passes_attempted INTEGER DEFAULT 0,
    tackles INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    clearances INTEGER DEFAULT 0,
    dribbles_completed INTEGER DEFAULT 0,
    dribbles_attempted INTEGER DEFAULT 0,
    fouls_committed INTEGER DEFAULT 0,
    fouls_drawn INTEGER DEFAULT 0,
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    rating DECIMAL(4,2),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(player_id, match_id)
);

-- ============= TRANSFER/MANAGER TABLES =============

-- Transfers table
CREATE TABLE IF NOT EXISTS transfers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    player_id UUID REFERENCES players(id) ON DELETE CASCADE,
    from_team_id UUID REFERENCES teams(id) ON DELETE SET NULL,
    to_team_id UUID REFERENCES teams(id) ON DELETE SET NULL,
    transfer_date DATE NOT NULL,
    fee DECIMAL(15,2),
    transfer_type VARCHAR(50)
        CHECK (transfer_type IN ('permanent', 'loan', 'free', 'swap', 'loan_return')),
    season VARCHAR(10),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Managers table
CREATE TABLE IF NOT EXISTS managers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    normalized_name VARCHAR(255) NOT NULL,
    nationality VARCHAR(100),
    date_of_birth DATE,
    external_ids JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Team managers (current and historical)
CREATE TABLE IF NOT EXISTS team_managers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    manager_id UUID REFERENCES managers(id) ON DELETE CASCADE,
    start_date DATE NOT NULL,
    end_date DATE,
    is_current BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============= REFEREE TABLES =============

-- Referees table
CREATE TABLE IF NOT EXISTS referees (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    normalized_name VARCHAR(255) NOT NULL,
    nationality VARCHAR(100),
    external_ids JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Referee statistics per season/league
CREATE TABLE IF NOT EXISTS referee_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    referee_id UUID REFERENCES referees(id) ON DELETE CASCADE,
    league_id UUID REFERENCES leagues(id) ON DELETE CASCADE,
    season VARCHAR(10) NOT NULL,
    matches_officiated INTEGER DEFAULT 0,
    avg_fouls_per_match DECIMAL(5,2),
    avg_yellow_cards DECIMAL(5,2),
    avg_red_cards DECIMAL(5,2),
    avg_penalties DECIMAL(5,2),
    home_win_pct DECIMAL(5,2),
    draw_pct DECIMAL(5,2),
    away_win_pct DECIMAL(5,2),
    avg_goals DECIMAL(5,2),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(referee_id, league_id, season)
);

-- ============= WEATHER TABLE =============

CREATE TABLE IF NOT EXISTS weather (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) ON DELETE CASCADE,
    temperature_celsius DECIMAL(5,2),
    humidity_pct DECIMAL(5,2),
    wind_speed_kmh DECIMAL(5,2),
    wind_direction VARCHAR(10),
    precipitation_mm DECIMAL(5,2),
    weather_condition VARCHAR(50),
    visibility_km DECIMAL(5,2),
    pressure_hpa DECIMAL(7,2),
    source VARCHAR(100),
    recorded_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(match_id)
);

-- ============= LOTTERY TABLES =============

CREATE TABLE IF NOT EXISTS lottery_draws (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    game VARCHAR(50) NOT NULL
        CHECK (game IN ('powerball', 'mega_millions')),
    draw_date DATE NOT NULL,
    draw_number INTEGER,
    numbers INTEGER[] NOT NULL,
    bonus_number INTEGER NOT NULL,
    multiplier INTEGER,
    jackpot_amount DECIMAL(15,2),
    winners_by_tier JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(game, draw_date)
);

-- ============= PREDICTION TABLES =============

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL
        CHECK (prediction_type IN ('match_result', 'over_under', 'btts',
                                    'correct_score', 'player_prop', 'moneyline',
                                    'spread', 'total', 'lottery')),
    predicted_outcome VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    probabilities JSONB NOT NULL,
    features_used JSONB DEFAULT '{}',
    feature_importance JSONB DEFAULT '{}',
    expected_value DECIMAL(10,4),
    kelly_fraction DECIMAL(10,6),
    recommended_stake DECIMAL(10,2),
    actual_outcome VARCHAR(100),
    is_correct BOOLEAN,
    profit_loss DECIMAL(10,2),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============= BETTING TABLES =============

-- Betting recommendations
CREATE TABLE IF NOT EXISTS betting_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id UUID REFERENCES predictions(id) ON DELETE CASCADE,
    match_id UUID REFERENCES matches(id) ON DELETE CASCADE,
    bet_type VARCHAR(50) NOT NULL,
    selection VARCHAR(100) NOT NULL,
    odds_at_recommendation DECIMAL(10,4) NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    confidence_rating VARCHAR(10)
        CHECK (confidence_rating IN ('low', 'medium', 'high', 'very_high')),
    expected_value DECIMAL(10,4),
    kelly_stake DECIMAL(10,2),
    recommended_stake DECIMAL(10,2),
    max_stake DECIMAL(10,2),
    reasoning TEXT,
    risk_factors JSONB DEFAULT '[]',
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'placed', 'won', 'lost', 'void', 'expired')),
    actual_result VARCHAR(100),
    profit_loss DECIMAL(10,2),
    placed_at TIMESTAMP WITH TIME ZONE,
    settled_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Accumulators (parlays)
CREATE TABLE IF NOT EXISTS accumulators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255),
    total_odds DECIMAL(15,4),
    stake DECIMAL(10,2),
    potential_return DECIMAL(15,2),
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'placed', 'won', 'lost', 'partial', 'void')),
    confidence_rating VARCHAR(10),
    reasoning TEXT,
    actual_return DECIMAL(15,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Accumulator legs
CREATE TABLE IF NOT EXISTS accumulator_legs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    accumulator_id UUID REFERENCES accumulators(id) ON DELETE CASCADE,
    recommendation_id UUID REFERENCES betting_recommendations(id) ON DELETE CASCADE,
    leg_order INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'won', 'lost', 'void')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(accumulator_id, leg_order)
);

-- ============= USER TABLES =============

-- User preferences
CREATE TABLE IF NOT EXISTS user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    preference_key VARCHAR(100) NOT NULL UNIQUE,
    preference_value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Betting history (manual tracking)
CREATE TABLE IF NOT EXISTS betting_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recommendation_id UUID REFERENCES betting_recommendations(id) ON DELETE SET NULL,
    match_id UUID REFERENCES matches(id) ON DELETE SET NULL,
    bookmaker VARCHAR(50) NOT NULL,
    bet_type VARCHAR(50) NOT NULL,
    selection VARCHAR(100) NOT NULL,
    odds DECIMAL(10,4) NOT NULL,
    stake DECIMAL(10,2) NOT NULL,
    potential_return DECIMAL(15,2),
    actual_return DECIMAL(15,2),
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'won', 'lost', 'void', 'cashout')),
    notes TEXT,
    placed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    settled_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============= SYSTEM TABLES =============

-- Model performance logs
CREATE TABLE IF NOT EXISTS model_performance_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    sport VARCHAR(50) NOT NULL,
    league_id UUID REFERENCES leagues(id) ON DELETE SET NULL,
    evaluation_date DATE NOT NULL,
    metrics JSONB NOT NULL,
    sample_size INTEGER,
    date_range_start DATE,
    date_range_end DATE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model retraining logs
CREATE TABLE IF NOT EXISTS model_retraining_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    old_version VARCHAR(50),
    new_version VARCHAR(50) NOT NULL,
    trigger_reason VARCHAR(50)
        CHECK (trigger_reason IN ('scheduled', 'performance_degradation', 'new_data', 'manual', 'config_change')),
    training_data_start DATE,
    training_data_end DATE,
    training_samples INTEGER,
    validation_metrics JSONB,
    test_metrics JSONB,
    hyperparameters JSONB,
    training_duration_seconds INTEGER,
    status VARCHAR(20)
        CHECK (status IN ('started', 'training', 'evaluating', 'deploying', 'completed', 'failed', 'rolled_back')),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Data quality logs
CREATE TABLE IF NOT EXISTS data_quality_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source VARCHAR(100) NOT NULL,
    check_type VARCHAR(50) NOT NULL,
    table_name VARCHAR(100),
    check_date DATE NOT NULL DEFAULT CURRENT_DATE,
    records_checked INTEGER,
    records_passed INTEGER,
    records_failed INTEGER,
    failure_rate DECIMAL(5,4),
    details JSONB DEFAULT '{}',
    severity VARCHAR(20)
        CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Scraping logs
CREATE TABLE IF NOT EXISTS scraping_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scraper_name VARCHAR(100) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20)
        CHECK (status IN ('running', 'completed', 'failed', 'partial')),
    records_scraped INTEGER DEFAULT 0,
    records_inserted INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    error_message TEXT,
    error_traceback TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Features cache
CREATE TABLE IF NOT EXISTS features_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES matches(id) ON DELETE CASCADE,
    feature_set VARCHAR(100) NOT NULL,
    features JSONB NOT NULL,
    feature_version VARCHAR(50) NOT NULL,
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(match_id, feature_set, feature_version)
);

-- API request logs
CREATE TABLE IF NOT EXISTS api_request_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    request_body JSONB,
    response_summary JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System notifications
CREATE TABLE IF NOT EXISTS system_notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type VARCHAR(50) NOT NULL
        CHECK (type IN ('info', 'warning', 'error', 'alert', 'recommendation')),
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    source VARCHAR(100),
    is_read BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add foreign key for matches.referee_id after referees table exists
ALTER TABLE matches ADD CONSTRAINT fk_matches_referee
    FOREIGN KEY (referee_id) REFERENCES referees(id) ON DELETE SET NULL;
