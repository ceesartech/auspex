"""SQL query constants for feature engineering.

All queries target the canonical schema defined in
services/data-ingestion/db/migrations/001_create_schema.sql:

  - PK columns are named `id` everywhere (matches.id, teams.id, etc).
    FK columns keep their natural names (home_team_id, league_id, ...).
  - matches.status uses {'scheduled','live','finished','postponed','cancelled'}.
    Use 'finished' for completed matches.
  - match_stats has ONE ROW PER TEAM PER MATCH. To get home/away
    side-by-side, JOIN match_stats twice, aliased msh + msa.
  - matches.referee_id is a UUID FK (nullable) into a future
    referees table; treat as opaque identifier here.
  - features_cache uniqueness is (match_id, feature_set, feature_version);
    there is no `cache_key` column. cache.py passes the 3-tuple.

Output columns are intentionally aliased so the existing computer
modules (categories/*.py) continue to consume `shots_home`,
`shots_away`, `xg_home`, `feature_data`, etc. without modification.
"""

# ---------- Team Performance Queries ----------

TEAM_RECENT_MATCHES = """
    SELECT
        m.id                AS match_id,
        m.match_date,
        m.home_team_id,
        m.away_team_id,
        m.home_score,
        m.away_score,
        m.season,
        msh.possession      AS possession_home,
        msa.possession      AS possession_away,
        msh.shots           AS shots_home,
        msa.shots           AS shots_away,
        msh.shots_on_target AS shots_on_target_home,
        msa.shots_on_target AS shots_on_target_away,
        msh.corners         AS corners_home,
        msa.corners         AS corners_away,
        msh.fouls           AS fouls_home,
        msa.fouls           AS fouls_away,
        msh.yellow_cards    AS yellow_cards_home,
        msa.yellow_cards    AS yellow_cards_away,
        msh.red_cards       AS red_cards_home,
        msa.red_cards       AS red_cards_away,
        msh.expected_goals  AS xg_home,
        msa.expected_goals  AS xg_away
    FROM matches m
    LEFT JOIN match_stats msh
        ON msh.match_id = m.id AND msh.team_id = m.home_team_id
    LEFT JOIN match_stats msa
        ON msa.match_id = m.id AND msa.team_id = m.away_team_id
    WHERE (m.home_team_id = %s OR m.away_team_id = %s)
      AND m.status = 'finished'
      AND m.match_date < %s
    ORDER BY m.match_date DESC
    LIMIT %s
"""

# Not currently consumed (categories/team_performance.py uses TEAM_RECENT_MATCHES)
# but kept for completeness — they would join match_stats once, by side.
TEAM_HOME_MATCHES = """
    SELECT
        m.id              AS match_id,
        m.match_date,
        m.home_score,
        m.away_score,
        ms.possession     AS possession,
        ms.shots          AS shots,
        ms.shots_on_target AS shots_on_target,
        ms.expected_goals AS xg,
        ms.corners        AS corners
    FROM matches m
    LEFT JOIN match_stats ms
        ON ms.match_id = m.id AND ms.team_id = m.home_team_id
    WHERE m.home_team_id = %s
      AND m.status = 'finished'
      AND m.match_date < %s
    ORDER BY m.match_date DESC
    LIMIT %s
"""

TEAM_AWAY_MATCHES = """
    SELECT
        m.id              AS match_id,
        m.match_date,
        m.away_score      AS goals_for,
        m.home_score      AS goals_against,
        ms.possession     AS possession,
        ms.shots          AS shots,
        ms.shots_on_target AS shots_on_target,
        ms.expected_goals AS xg,
        ms.corners        AS corners
    FROM matches m
    LEFT JOIN match_stats ms
        ON ms.match_id = m.id AND ms.team_id = m.away_team_id
    WHERE m.away_team_id = %s
      AND m.status = 'finished'
      AND m.match_date < %s
    ORDER BY m.match_date DESC
    LIMIT %s
"""

# ---------- Head-to-Head Queries ----------

H2H_MATCHES = """
    SELECT
        m.id                AS match_id,
        m.match_date,
        m.home_team_id,
        m.away_team_id,
        m.home_score,
        m.away_score,
        msh.possession      AS possession_home,
        msa.possession      AS possession_away,
        msh.shots           AS shots_home,
        msa.shots           AS shots_away,
        msh.expected_goals  AS xg_home,
        msa.expected_goals  AS xg_away
    FROM matches m
    LEFT JOIN match_stats msh
        ON msh.match_id = m.id AND msh.team_id = m.home_team_id
    LEFT JOIN match_stats msa
        ON msa.match_id = m.id AND msa.team_id = m.away_team_id
    WHERE ((m.home_team_id = %s AND m.away_team_id = %s)
        OR (m.home_team_id = %s AND m.away_team_id = %s))
      AND m.status = 'finished'
      AND m.match_date < %s
    ORDER BY m.match_date DESC
    LIMIT %s
"""

# ---------- Player Queries ----------

TEAM_SQUAD = """
    SELECT
        p.id           AS player_id,
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
        pa.reason       AS injury_type,
        pa.expected_return
    FROM player_availability pa
    JOIN players p ON pa.player_id = p.id
    WHERE p.team_id = %s
      AND pa.status IN ('injured', 'suspended', 'doubtful')
"""

PLAYER_RECENT_STATS = """
    SELECT
        pms.player_id,
        pms.minutes_played,
        pms.goals,
        pms.assists,
        pms.expected_goals    AS xg,
        pms.expected_assists  AS xa,
        pms.rating,
        pms.key_passes,
        pms.shots,
        m.match_date
    FROM player_match_stats pms
    JOIN matches m ON pms.match_id = m.id
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
        m.id              AS match_id,
        m.match_date,
        m.home_team_id,
        m.away_team_id,
        m.league_id,
        m.season,
        m.venue,
        m.referee_id      AS referee,
        m.attendance,
        l.name            AS league_name,
        l.country,
        ht.name           AS home_team_name,
        at.name           AS away_team_name
    FROM matches m
    JOIN leagues l ON m.league_id = l.id
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
    WHERE m.id = %s
"""

# Sum yellow/red/foul cards across both teams. Pre-aggregated per match
# in a CTE so the outer SELECT only averages once.
REFEREE_STATS = """
    WITH per_match AS (
        SELECT
            m.id AS match_id,
            SUM(ms.yellow_cards) AS yellows,
            SUM(ms.red_cards)    AS reds,
            SUM(ms.fouls)        AS fouls
        FROM matches m
        JOIN match_stats ms ON ms.match_id = m.id
        WHERE m.referee_id = %s
          AND m.status = 'finished'
        GROUP BY m.id
    )
    SELECT
        COUNT(*)                          AS matches_refereed,
        AVG(yellows)                      AS avg_yellows,
        AVG(reds)                         AS avg_reds,
        AVG(fouls)                        AS avg_fouls,
        AVG((SELECT m2.home_score + m2.away_score
             FROM matches m2 WHERE m2.id = per_match.match_id)) AS avg_goals
    FROM per_match
"""

TEAM_LAST_MATCH_DATE = """
    SELECT MAX(m.match_date) AS last_match_date
    FROM matches m
    WHERE (m.home_team_id = %s OR m.away_team_id = %s)
      AND m.status = 'finished'
      AND m.match_date < %s
"""

LEAGUE_SEASON_STATS = """
    SELECT
        AVG(m.home_score + m.away_score) AS avg_goals,
        AVG(m.home_score)                AS avg_home_goals,
        AVG(m.away_score)                AS avg_away_goals,
        STDDEV(m.home_score + m.away_score) AS std_goals,
        SUM(CASE WHEN m.home_score > m.away_score THEN 1 ELSE 0 END)::FLOAT
            / NULLIF(COUNT(*), 0) AS home_win_rate,
        SUM(CASE WHEN m.home_score = m.away_score THEN 1 ELSE 0 END)::FLOAT
            / NULLIF(COUNT(*), 0) AS draw_rate
    FROM matches m
    WHERE m.league_id = %s
      AND m.season = %s
      AND m.status = 'finished'
"""

# ---------- Cache Queries ----------
#
# features_cache uniqueness is (match_id, feature_set, feature_version).
# There is no `cache_key` column. cache.py passes the composite as a
# tuple in the parameter list.

CACHE_GET = """
    SELECT features AS feature_data, computed_at
    FROM features_cache
    WHERE match_id = %s
      AND feature_set = %s
      AND feature_version = %s
      AND (expires_at IS NULL OR expires_at > NOW())
    ORDER BY computed_at DESC
    LIMIT 1
"""

CACHE_SET = """
    INSERT INTO features_cache
        (match_id, feature_set, feature_version, features, computed_at, expires_at)
    VALUES
        (%s, %s, %s, %s::jsonb, NOW(), NOW() + (%s || ' seconds')::interval)
    ON CONFLICT (match_id, feature_set, feature_version) DO UPDATE
        SET features = EXCLUDED.features,
            computed_at = NOW(),
            expires_at = EXCLUDED.expires_at
"""

CACHE_DELETE = """
    DELETE FROM features_cache
    WHERE match_id = %s
      AND feature_set = %s
      AND feature_version = %s
"""

CACHE_CLEANUP = """
    DELETE FROM features_cache
    WHERE expires_at IS NOT NULL
      AND expires_at < NOW()
"""
