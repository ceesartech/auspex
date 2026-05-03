-- Additional performance indexes for complex queries

-- ============= COMPOSITE INDEXES =============

-- Matches - common query patterns
CREATE INDEX idx_matches_league_status_date ON matches(league_id, status, match_date DESC);
CREATE INDEX idx_matches_season_league ON matches(season, league_id);
CREATE INDEX idx_matches_home_team_date ON matches(home_team_id, match_date DESC);
CREATE INDEX idx_matches_away_team_date ON matches(away_team_id, match_date DESC);

-- Odds - lookup by match and bookmaker
CREATE INDEX idx_odds_match_bookmaker_market ON odds(match_id, bookmaker, market_type);
CREATE INDEX idx_odds_match_timestamp ON odds(match_id, timestamp DESC);

-- Player stats - performance lookups
CREATE INDEX idx_player_match_stats_player ON player_match_stats(player_id, match_id);
CREATE INDEX idx_player_availability_player_status ON player_availability(player_id, status);

-- Predictions - model analysis
CREATE INDEX idx_predictions_match_model ON predictions(match_id, model_name, model_version);
CREATE INDEX idx_predictions_created ON predictions(created_at DESC);

-- Betting recommendations - status tracking
CREATE INDEX idx_recommendations_status ON betting_recommendations(status, created_at DESC);
CREATE INDEX idx_recommendations_match ON betting_recommendations(match_id, status);

-- ============= PARTIAL INDEXES =============

-- Only index upcoming/live matches (most queried)
CREATE INDEX idx_matches_upcoming ON matches(match_date)
    WHERE status IN ('scheduled', 'live');

-- Only active player availability
CREATE INDEX idx_player_availability_active ON player_availability(player_id, expected_return)
    WHERE status != 'available';

-- Pending recommendations (most frequently queried)
CREATE INDEX idx_recommendations_pending ON betting_recommendations(created_at DESC)
    WHERE status = 'pending';

-- Unread notifications
CREATE INDEX idx_notifications_unread ON system_notifications(created_at DESC)
    WHERE is_read = false;

-- ============= GIN INDEXES FOR JSONB =============

CREATE INDEX idx_teams_external_ids ON teams USING GIN(external_ids);
CREATE INDEX idx_matches_external_ids ON matches USING GIN(external_ids);
CREATE INDEX idx_matches_metadata ON matches USING GIN(metadata);
CREATE INDEX idx_odds_metadata ON odds USING GIN(metadata);
CREATE INDEX idx_players_external_ids ON players USING GIN(external_ids);
CREATE INDEX idx_predictions_probabilities ON predictions USING GIN(probabilities);
CREATE INDEX idx_predictions_features ON predictions USING GIN(features_used);
CREATE INDEX idx_features_cache_features ON features_cache USING GIN(features);
CREATE INDEX idx_match_stats_metadata ON match_stats USING GIN(metadata);
CREATE INDEX idx_lottery_draws_winners ON lottery_draws USING GIN(winners_by_tier);
CREATE INDEX idx_model_performance_metrics ON model_performance_logs USING GIN(metrics);

-- ============= COVERING INDEXES =============

-- Cover common odds queries without table lookups
CREATE INDEX idx_odds_covering ON odds(match_id, market_type, bookmaker)
    INCLUDE (odds_decimal, implied_probability, timestamp);

-- Cover match result lookups
CREATE INDEX idx_matches_result_covering ON matches(league_id, match_date DESC)
    INCLUDE (home_team_id, away_team_id, home_score, away_score, status);

-- ============= EXPRESSION INDEXES =============

-- Index on date part of match_date for daily grouping. We anchor to UTC
-- because `match_date::date` depends on the session timezone and
-- therefore isn't IMMUTABLE — Postgres rejects it as an index expression.
CREATE INDEX idx_matches_date_only ON matches(((match_date AT TIME ZONE 'UTC')::date));

-- Index on lower case for case-insensitive team search
CREATE INDEX idx_teams_name_lower ON teams(lower(name));
CREATE INDEX idx_players_name_lower ON players(lower(name));

-- ============= BTREE-GIN FOR ARRAY TYPES =============

-- Lottery number searches
CREATE INDEX idx_lottery_numbers ON lottery_draws USING GIN(numbers);

-- ============= FULL TEXT SEARCH =============

-- Search across team names
CREATE INDEX idx_teams_name_trgm ON teams USING GIN(name gin_trgm_ops);
CREATE INDEX idx_players_name_trgm ON players USING GIN(name gin_trgm_ops);

-- ============= TIME-SERIES OPTIMIZATION =============

-- Scraping logs by time (for monitoring dashboards)
CREATE INDEX idx_scraping_logs_time ON scraping_logs(start_time DESC);
CREATE INDEX idx_scraping_logs_scraper_time ON scraping_logs(scraper_name, start_time DESC);

-- API request logs by time
CREATE INDEX idx_api_logs_time ON api_request_logs(created_at DESC);
CREATE INDEX idx_api_logs_endpoint_time ON api_request_logs(endpoint, created_at DESC);

-- Data quality logs
CREATE INDEX idx_data_quality_source_date ON data_quality_logs(source, check_date DESC);

-- Model performance tracking
CREATE INDEX idx_model_performance_model_date ON model_performance_logs(model_name, evaluation_date DESC);
CREATE INDEX idx_model_retraining_model ON model_retraining_logs(model_name, created_at DESC);
