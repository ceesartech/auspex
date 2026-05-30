-- NHL goalie-features prerequisite: unique index on the NHL player ID
-- inside players.external_ids so the goalie loader can ON-CONFLICT
-- upsert by NHL's stable player id.
--
-- The original schema's UNIQUE constraint on (normalized_name, sport)
-- doesn't include sport='nhl' for goalies (we never wrote those before)
-- AND multiple players across history can share a normalized name. The
-- NHL-API player id is unique-and-stable, so we key on that instead.
--
-- We use a partial expression index restricted to rows that actually
-- carry the key (every NHL goalie/skater we ingest will, but other
-- player sources won't). The WHERE clause keeps the index small and
-- avoids forcing nhl_player_id on soccer/non-NHL inserts.

CREATE UNIQUE INDEX IF NOT EXISTS idx_players_nhl_player_id
    ON players ((external_ids ->> 'nhl_player_id'))
    WHERE external_ids ? 'nhl_player_id';
