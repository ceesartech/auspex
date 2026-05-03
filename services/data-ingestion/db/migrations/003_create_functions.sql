-- Database functions, triggers, and views

-- ============= TRIGGER FUNCTIONS =============

-- Auto-calculate match outcome
CREATE OR REPLACE FUNCTION calculate_match_outcome()
RETURNS TRIGGER AS $$
BEGIN
    -- Update predictions when match finishes
    IF NEW.status = 'finished' AND OLD.status != 'finished' THEN
        -- Update match result predictions
        UPDATE predictions
        SET actual_outcome = CASE
                WHEN NEW.home_score > NEW.away_score THEN 'home'
                WHEN NEW.home_score < NEW.away_score THEN 'away'
                ELSE 'draw'
            END,
            is_correct = CASE
                WHEN prediction_type = 'match_result' THEN
                    predicted_outcome = CASE
                        WHEN NEW.home_score > NEW.away_score THEN 'home'
                        WHEN NEW.home_score < NEW.away_score THEN 'away'
                        ELSE 'draw'
                    END
                WHEN prediction_type = 'over_under' THEN
                    CASE
                        WHEN predicted_outcome LIKE 'over_%' THEN
                            (NEW.home_score + NEW.away_score) > CAST(SPLIT_PART(predicted_outcome, '_', 2) AS DECIMAL)
                        WHEN predicted_outcome LIKE 'under_%' THEN
                            (NEW.home_score + NEW.away_score) < CAST(SPLIT_PART(predicted_outcome, '_', 2) AS DECIMAL)
                    END
                WHEN prediction_type = 'btts' THEN
                    (predicted_outcome = 'yes' AND NEW.home_score > 0 AND NEW.away_score > 0)
                    OR (predicted_outcome = 'no' AND (NEW.home_score = 0 OR NEW.away_score = 0))
                ELSE NULL
            END,
            updated_at = NOW()
        WHERE match_id = NEW.id;

        -- Update betting recommendations
        UPDATE betting_recommendations
        SET status = CASE
                WHEN br.is_correct = true THEN 'won'
                WHEN br.is_correct = false THEN 'lost'
                ELSE status
            END,
            settled_at = NOW(),
            updated_at = NOW()
        FROM (
            SELECT p.id as pred_id, p.is_correct
            FROM predictions p
            WHERE p.match_id = NEW.id
        ) br
        WHERE betting_recommendations.prediction_id = br.pred_id
        AND betting_recommendations.status = 'placed';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_match_outcome
    AFTER UPDATE ON matches
    FOR EACH ROW
    WHEN (NEW.status = 'finished' AND OLD.status != 'finished')
    EXECUTE FUNCTION calculate_match_outcome();

-- Auto-update updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at trigger to relevant tables
CREATE TRIGGER trg_leagues_updated_at BEFORE UPDATE ON leagues
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER trg_teams_updated_at BEFORE UPDATE ON teams
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER trg_matches_updated_at BEFORE UPDATE ON matches
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER trg_players_updated_at BEFORE UPDATE ON players
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER trg_player_availability_updated_at BEFORE UPDATE ON player_availability
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER trg_predictions_updated_at BEFORE UPDATE ON predictions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER trg_recommendations_updated_at BEFORE UPDATE ON betting_recommendations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER trg_accumulators_updated_at BEFORE UPDATE ON accumulators
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER trg_user_preferences_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER trg_betting_history_updated_at BEFORE UPDATE ON betting_history
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============= UTILITY FUNCTIONS =============

-- Get team form (last N matches)
CREATE OR REPLACE FUNCTION get_team_form(
    p_team_id UUID,
    p_num_matches INTEGER DEFAULT 5,
    p_before_date TIMESTAMP WITH TIME ZONE DEFAULT NOW()
)
RETURNS TABLE (
    match_id UUID,
    match_date TIMESTAMP WITH TIME ZONE,
    opponent_id UUID,
    is_home BOOLEAN,
    goals_for INTEGER,
    goals_against INTEGER,
    result CHAR(1),
    points INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id as match_id,
        m.match_date,
        CASE WHEN m.home_team_id = p_team_id THEN m.away_team_id ELSE m.home_team_id END as opponent_id,
        (m.home_team_id = p_team_id) as is_home,
        CASE WHEN m.home_team_id = p_team_id THEN m.home_score ELSE m.away_score END as goals_for,
        CASE WHEN m.home_team_id = p_team_id THEN m.away_score ELSE m.home_score END as goals_against,
        CASE
            WHEN m.home_team_id = p_team_id AND m.home_score > m.away_score THEN 'W'
            WHEN m.away_team_id = p_team_id AND m.away_score > m.home_score THEN 'W'
            WHEN m.home_score = m.away_score THEN 'D'
            ELSE 'L'
        END::CHAR(1) as result,
        CASE
            WHEN m.home_team_id = p_team_id AND m.home_score > m.away_score THEN 3
            WHEN m.away_team_id = p_team_id AND m.away_score > m.home_score THEN 3
            WHEN m.home_score = m.away_score THEN 1
            ELSE 0
        END as points
    FROM matches m
    WHERE (m.home_team_id = p_team_id OR m.away_team_id = p_team_id)
        AND m.status = 'finished'
        AND m.match_date < p_before_date
    ORDER BY m.match_date DESC
    LIMIT p_num_matches;
END;
$$ LANGUAGE plpgsql;

-- Get head-to-head stats
CREATE OR REPLACE FUNCTION get_h2h_stats(
    p_team1_id UUID,
    p_team2_id UUID,
    p_num_matches INTEGER DEFAULT 10
)
RETURNS TABLE (
    match_id UUID,
    match_date TIMESTAMP WITH TIME ZONE,
    home_team_id UUID,
    away_team_id UUID,
    home_score INTEGER,
    away_score INTEGER,
    league_name VARCHAR(255)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id as match_id,
        m.match_date,
        m.home_team_id,
        m.away_team_id,
        m.home_score,
        m.away_score,
        l.name as league_name
    FROM matches m
    JOIN leagues l ON m.league_id = l.id
    WHERE m.status = 'finished'
        AND (
            (m.home_team_id = p_team1_id AND m.away_team_id = p_team2_id)
            OR (m.home_team_id = p_team2_id AND m.away_team_id = p_team1_id)
        )
    ORDER BY m.match_date DESC
    LIMIT p_num_matches;
END;
$$ LANGUAGE plpgsql;

-- Get league standings
CREATE OR REPLACE FUNCTION get_league_standings(
    p_league_id UUID,
    p_season VARCHAR(10)
)
RETURNS TABLE (
    team_id UUID,
    team_name VARCHAR(255),
    played INTEGER,
    won INTEGER,
    drawn INTEGER,
    lost INTEGER,
    goals_for BIGINT,
    goals_against BIGINT,
    goal_difference BIGINT,
    points BIGINT
) AS $$
BEGIN
    RETURN QUERY
    WITH team_results AS (
        SELECT
            t.id as tid,
            t.name as tname,
            COUNT(*) as matches_played,
            SUM(CASE
                WHEN (m.home_team_id = t.id AND m.home_score > m.away_score)
                    OR (m.away_team_id = t.id AND m.away_score > m.home_score) THEN 1 ELSE 0
            END) as wins,
            SUM(CASE WHEN m.home_score = m.away_score THEN 1 ELSE 0 END) as draws,
            SUM(CASE
                WHEN (m.home_team_id = t.id AND m.home_score < m.away_score)
                    OR (m.away_team_id = t.id AND m.away_score < m.home_score) THEN 1 ELSE 0
            END) as losses,
            SUM(CASE WHEN m.home_team_id = t.id THEN m.home_score ELSE m.away_score END) as gf,
            SUM(CASE WHEN m.home_team_id = t.id THEN m.away_score ELSE m.home_score END) as ga
        FROM teams t
        JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
        WHERE m.league_id = p_league_id
            AND m.season = p_season
            AND m.status = 'finished'
        GROUP BY t.id, t.name
    )
    SELECT
        tr.tid as team_id,
        tr.tname as team_name,
        tr.matches_played::INTEGER as played,
        tr.wins::INTEGER as won,
        tr.draws::INTEGER as drawn,
        tr.losses::INTEGER as lost,
        tr.gf as goals_for,
        tr.ga as goals_against,
        (tr.gf - tr.ga) as goal_difference,
        (tr.wins * 3 + tr.draws) as points
    FROM team_results tr
    ORDER BY (tr.wins * 3 + tr.draws) DESC, (tr.gf - tr.ga) DESC, tr.gf DESC;
END;
$$ LANGUAGE plpgsql;

-- Get player season statistics
CREATE OR REPLACE FUNCTION get_player_season_stats(
    p_player_id UUID,
    p_season VARCHAR(10)
)
RETURNS TABLE (
    appearances INTEGER,
    total_minutes INTEGER,
    goals BIGINT,
    assists BIGINT,
    total_xg DECIMAL,
    total_xa DECIMAL,
    shots BIGINT,
    key_passes BIGINT,
    avg_rating DECIMAL,
    yellow_cards BIGINT,
    red_cards BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER as appearances,
        SUM(pms.minutes_played)::INTEGER as total_minutes,
        SUM(pms.goals) as goals,
        SUM(pms.assists) as assists,
        ROUND(SUM(pms.expected_goals)::DECIMAL, 2) as total_xg,
        ROUND(SUM(pms.expected_assists)::DECIMAL, 2) as total_xa,
        SUM(pms.shots) as shots,
        SUM(pms.key_passes) as key_passes,
        ROUND(AVG(pms.rating), 2) as avg_rating,
        SUM(pms.yellow_cards) as yellow_cards,
        SUM(pms.red_cards) as red_cards
    FROM player_match_stats pms
    JOIN matches m ON pms.match_id = m.id
    WHERE pms.player_id = p_player_id
        AND m.season = p_season
        AND m.status = 'finished';
END;
$$ LANGUAGE plpgsql;

-- Check data completeness for a match
CREATE OR REPLACE FUNCTION check_data_completeness(p_match_id UUID)
RETURNS TABLE (
    has_match_stats BOOLEAN,
    has_odds BOOLEAN,
    has_player_stats BOOLEAN,
    has_weather BOOLEAN,
    has_referee BOOLEAN,
    completeness_score DECIMAL
) AS $$
DECLARE
    v_has_stats BOOLEAN;
    v_has_odds BOOLEAN;
    v_has_players BOOLEAN;
    v_has_weather BOOLEAN;
    v_has_referee BOOLEAN;
    v_score DECIMAL;
BEGIN
    SELECT EXISTS(SELECT 1 FROM match_stats WHERE match_id = p_match_id) INTO v_has_stats;
    SELECT EXISTS(SELECT 1 FROM odds WHERE match_id = p_match_id) INTO v_has_odds;
    SELECT EXISTS(SELECT 1 FROM player_match_stats pms JOIN matches m ON pms.match_id = m.id WHERE m.id = p_match_id) INTO v_has_players;
    SELECT EXISTS(SELECT 1 FROM weather WHERE match_id = p_match_id) INTO v_has_weather;
    SELECT EXISTS(SELECT 1 FROM matches WHERE id = p_match_id AND referee_id IS NOT NULL) INTO v_has_referee;

    v_score := (
        CASE WHEN v_has_stats THEN 25 ELSE 0 END +
        CASE WHEN v_has_odds THEN 25 ELSE 0 END +
        CASE WHEN v_has_players THEN 25 ELSE 0 END +
        CASE WHEN v_has_weather THEN 15 ELSE 0 END +
        CASE WHEN v_has_referee THEN 10 ELSE 0 END
    )::DECIMAL;

    RETURN QUERY SELECT v_has_stats, v_has_odds, v_has_players, v_has_weather, v_has_referee, v_score;
END;
$$ LANGUAGE plpgsql;

-- ============= VIEWS =============

-- Upcoming matches with latest odds
CREATE OR REPLACE VIEW vw_upcoming_matches_with_odds AS
SELECT
    m.id as match_id,
    m.match_date,
    l.name as league_name,
    l.country as league_country,
    ht.name as home_team,
    at.name as away_team,
    m.venue,
    latest_odds.home_odds,
    latest_odds.draw_odds,
    latest_odds.away_odds,
    latest_odds.bookmaker
FROM matches m
JOIN leagues l ON m.league_id = l.id
JOIN teams ht ON m.home_team_id = ht.id
JOIN teams at ON m.away_team_id = at.id
LEFT JOIN LATERAL (
    SELECT
        MAX(CASE WHEN o.selection = 'home' THEN o.odds_decimal END) as home_odds,
        MAX(CASE WHEN o.selection = 'draw' THEN o.odds_decimal END) as draw_odds,
        MAX(CASE WHEN o.selection = 'away' THEN o.odds_decimal END) as away_odds,
        o.bookmaker
    FROM odds o
    WHERE o.match_id = m.id
        AND o.market_type = '1x2'
    GROUP BY o.bookmaker
    ORDER BY MAX(o.timestamp) DESC
    LIMIT 1
) latest_odds ON true
WHERE m.status = 'scheduled'
    AND m.match_date > NOW()
ORDER BY m.match_date ASC;

-- Active betting recommendations
CREATE OR REPLACE VIEW vw_active_recommendations AS
SELECT
    br.id as recommendation_id,
    m.match_date,
    l.name as league_name,
    ht.name as home_team,
    at.name as away_team,
    br.bet_type,
    br.selection,
    br.odds_at_recommendation,
    br.bookmaker,
    br.confidence_rating,
    br.expected_value,
    br.recommended_stake,
    br.reasoning,
    br.status,
    p.model_name,
    p.confidence as model_confidence
FROM betting_recommendations br
JOIN predictions p ON br.prediction_id = p.id
JOIN matches m ON br.match_id = m.id
JOIN leagues l ON m.league_id = l.id
JOIN teams ht ON m.home_team_id = ht.id
JOIN teams at ON m.away_team_id = at.id
WHERE br.status IN ('pending', 'placed')
    AND m.match_date > NOW()
ORDER BY br.confidence_rating DESC, br.expected_value DESC;
