"""SQL query constants for feature engineering."""

# ---------- Team Performance Queries ----------

TEAM_RECENT_MATCHES = """
    SELECT
        m.match_id,
        m.match_date,
        m.home_team_id,
        m.away_team_id,
        m.home_score,
        m.away_score,
        m.season,
        ms.possession_home,
        ms.possession_away,
        ms.shots_home,
        ms.shots_away,
        ms.shots_on_target_home,
        ms.shots_on_target_away,
        ms.corners_home,
        ms.corners_away,
        ms.fouls_home,
        ms.fouls_away,
        ms.yellow_cards_home,
        ms.yellow_cards_away,
        ms.red_cards_home,
        ms.red_cards_away,
        ms.xg_home,
        ms.xg_away
    FROM matches m
    LEFT JOIN match_stats ms ON m.match_id = ms.match_id
    WHERE (m.home_team_id = %s OR m.away_team_id = %s)
      AND m.status = 'completed'
      AND m.match_date < %s
    ORDER BY m.match_date DESC
    LIMIT %s
"""

TEAM_HOME_MATCHES = """
    SELECT
        m.match_id,
        m.match_date,
        m.home_score,
        m.away_score,
        ms.possession_home AS possession,
        ms.shots_home AS shots,
        ms.shots_on_target_home AS shots_on_target,
        ms.xg_home AS xg,
        ms.corners_home AS corners
    FROM matches m
    LEFT JOIN match_stats ms ON m.match_id = ms.match_id
    WHERE m.home_team_id = %s
      AND m.status = 'completed'
      AND m.match_date < %s
    ORDER BY m.match_date DESC
    LIMIT %s
"""

TEAM_AWAY_MATCHES = """
    SELECT
        m.match_id,
        m.match_date,
        m.away_score AS goals_for,
        m.home_score AS goals_against,
        ms.possession_away AS possession,
        ms.shots_away AS shots,
        ms.shots_on_target_away AS shots_on_target,
        ms.xg_away AS xg,
        ms.corners_away AS corners
    FROM matches m
    LEFT JOIN match_stats ms ON m.match_id = ms.match_id
    WHERE m.away_team_id = %s
      AND m.status = 'completed'
      AND m.match_date < %s
    ORDER BY m.match_date DESC
    LIMIT %s
"""

# ---------- Head-to-Head Queries ----------

H2H_MATCHES = """
    SELECT
        m.match_id,
        m.match_date,
        m.home_team_id,
        m.away_team_id,
        m.home_score,
        m.away_score,
        ms.possession_home,
        ms.possession_away,
        ms.shots_home,
        ms.shots_away,
        ms.xg_home,
        ms.xg_away
    FROM matches m
    LEFT JOIN match_stats ms ON m.match_id = ms.match_id
    WHERE ((m.home_team_id = %s AND m.away_team_id = %s)
        OR (m.home_team_id = %s AND m.away_team_id = %s))
      AND m.status = 'completed'
      AND m.match_date < %s
    ORDER BY m.match_date DESC
    LIMIT %s
"""

# ---------- Player Queries ----------

TEAM_SQUAD = """
    SELECT
        p.player_id,
        p.name,
        p.position,
        p.market_value,
        p.date_of_birth
    FROM players p
    WHERE p.team_id = %s
      AND p.is_active = true
"""

PLAYER_AVAILABILITY = """
    SELECT
        pa.player_id,
        pa.status,
        pa.injury_type,
        pa.expected_return
    FROM player_availability pa
    JOIN players p ON pa.player_id = p.player_id
    WHERE p.team_id = %s
      AND pa.status IN ('injured', 'suspended', 'doubtful')
"""

PLAYER_RECENT_STATS = """
    SELECT
        pms.player_id,
        pms.minutes_played,
        pms.goals,
        pms.assists,
        pms.xg,
        pms.xa,
        pms.rating,
        pms.key_passes,
        pms.shots,
        m.match_date
    FROM player_match_stats pms
    JOIN matches m ON pms.match_id = m.match_id
    WHERE pms.player_id = %s
      AND m.match_date < %s
    ORDER BY m.match_date DESC
    LIMIT %s
"""

# ---------- Odds Queries ----------

MATCH_ODDS = """
    SELECT
        o.bookmaker,
        o.market_type,
        o.selection,
        o.odds_decimal,
        o.timestamp,
        o.is_opening
    FROM odds o
    WHERE o.match_id = %s
    ORDER BY o.timestamp ASC
"""

OPENING_ODDS = """
    SELECT
        o.bookmaker,
        o.market_type,
        o.selection,
        o.odds_decimal
    FROM odds o
    WHERE o.match_id = %s
      AND o.is_opening = true
"""

LATEST_ODDS = """
    SELECT DISTINCT ON (o.bookmaker, o.market_type, o.selection)
        o.bookmaker,
        o.market_type,
        o.selection,
        o.odds_decimal,
        o.timestamp
    FROM odds o
    WHERE o.match_id = %s
    ORDER BY o.bookmaker, o.market_type, o.selection, o.timestamp DESC
"""

# ---------- Contextual Queries ----------

MATCH_DETAILS = """
    SELECT
        m.match_id,
        m.match_date,
        m.home_team_id,
        m.away_team_id,
        m.league_id,
        m.season,
        m.venue,
        m.referee,
        m.attendance,
        l.name AS league_name,
        l.country,
        ht.name AS home_team_name,
        at.name AS away_team_name
    FROM matches m
    JOIN leagues l ON m.league_id = l.league_id
    JOIN teams ht ON m.home_team_id = ht.team_id
    JOIN teams at ON m.away_team_id = at.team_id
    WHERE m.match_id = %s
"""

REFEREE_STATS = """
    SELECT
        COUNT(*) AS matches_refereed,
        AVG(ms.yellow_cards_home + ms.yellow_cards_away) AS avg_yellows,
        AVG(ms.red_cards_home + ms.red_cards_away) AS avg_reds,
        AVG(ms.fouls_home + ms.fouls_away) AS avg_fouls,
        AVG(m.home_score + m.away_score) AS avg_goals
    FROM matches m
    JOIN match_stats ms ON m.match_id = ms.match_id
    WHERE m.referee = %s
      AND m.status = 'completed'
"""

TEAM_LAST_MATCH_DATE = """
    SELECT MAX(m.match_date) AS last_match_date
    FROM matches m
    WHERE (m.home_team_id = %s OR m.away_team_id = %s)
      AND m.status = 'completed'
      AND m.match_date < %s
"""

LEAGUE_SEASON_STATS = """
    SELECT
        AVG(m.home_score + m.away_score) AS avg_goals,
        AVG(m.home_score) AS avg_home_goals,
        AVG(m.away_score) AS avg_away_goals,
        STDDEV(m.home_score + m.away_score) AS std_goals,
        SUM(CASE WHEN m.home_score > m.away_score THEN 1 ELSE 0 END)::FLOAT
            / COUNT(*) AS home_win_rate,
        SUM(CASE WHEN m.home_score = m.away_score THEN 1 ELSE 0 END)::FLOAT
            / COUNT(*) AS draw_rate
    FROM matches m
    WHERE m.league_id = %s
      AND m.season = %s
      AND m.status = 'completed'
"""

# ---------- Cache Queries ----------

CACHE_GET = """
    SELECT feature_data, computed_at
    FROM features_cache
    WHERE cache_key = %s
      AND computed_at > NOW() - INTERVAL '%s seconds'
"""

CACHE_SET = """
    INSERT INTO features_cache (cache_key, feature_data, computed_at)
    VALUES (%s, %s, NOW())
    ON CONFLICT (cache_key)
    DO UPDATE SET feature_data = EXCLUDED.feature_data,
                  computed_at = EXCLUDED.computed_at
"""

CACHE_DELETE = """
    DELETE FROM features_cache
    WHERE cache_key = %s
"""

CACHE_CLEANUP = """
    DELETE FROM features_cache
    WHERE computed_at < NOW() - INTERVAL '%s seconds'
"""
