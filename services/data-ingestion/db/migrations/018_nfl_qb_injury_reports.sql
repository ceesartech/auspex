-- 018: NFL QB injury reports
--
-- Memory's `nfl-spread-total-efficient` calls out "QB injury / inactive
-- list" as the biggest single signal for both spread and total, and
-- earlier this session a DB audit found `player_availability` was
-- soccer-shaped + empty. The Phase D scope for NFL needs:
--   1. NFL-shaped per-player roster table (separate from the
--      goals/assists/xG shape `players` uses for NHL/soccer).
--   2. Per-game starting-QB attribution + injury / inactive status.
--   3. Historical backfill for the 3 in-corpus seasons (2022-23
--      through 2024-25), then forward-running ingestion.
--
-- This migration creates the schema. The scraper / backfill /
-- feature integration are out of scope (separate scripts and PRs).

BEGIN;

CREATE TABLE IF NOT EXISTS nfl_players (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Stable identifier from the source (pro-football-reference,
    -- nfl.com player ID, etc.). Unique so re-ingest is idempotent.
    source_player_id TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    position TEXT NOT NULL,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,
    -- Source identifier so we know which scraper populated this
    -- row. The Phase D scraper writes 'pro_football_reference';
    -- a future ingest can write 'nfl_official' etc.
    source TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS nfl_players_team_idx ON nfl_players (team_id);
CREATE INDEX IF NOT EXISTS nfl_players_position_idx ON nfl_players (position);

-- One row per (match, player) injury report. The injury report is
-- published Wed/Thu/Fri pre-game; we snapshot the FINAL Friday
-- report (highest signal — gives time for status to firm up but
-- before the Sunday inactive list).
CREATE TABLE IF NOT EXISTS nfl_injury_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    player_id UUID NOT NULL REFERENCES nfl_players(id) ON DELETE CASCADE,
    -- 'Q' = questionable, 'D' = doubtful, 'O' = out, 'IR' =
    -- injured reserve, 'PUP' = physically unable to perform,
    -- 'NSI' = not on injury report (full go). Captured verbatim
    -- from the source so we don't lose nuance.
    status TEXT NOT NULL,
    injury_type TEXT,
    is_starter BOOLEAN NOT NULL DEFAULT FALSE,
    -- Snapshot timestamp — usually the Friday before the game.
    -- Distinguished from match_id's match_date so future weekly
    -- updates can be appended without overwriting earlier rows.
    snapshot_at TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- One report row per (match, player, snapshot) — multiple
    -- snapshots per match are allowed for the weekly Wed/Thu/Fri
    -- updates. Idempotent re-ingest hits the unique constraint
    -- and ON CONFLICT updates the row.
    UNIQUE (match_id, player_id, snapshot_at)
);

CREATE INDEX IF NOT EXISTS nfl_injury_reports_match_idx
    ON nfl_injury_reports (match_id);
CREATE INDEX IF NOT EXISTS nfl_injury_reports_player_idx
    ON nfl_injury_reports (player_id);
-- For "who's the starter for this match" lookups the feature
-- compute does.
CREATE INDEX IF NOT EXISTS nfl_injury_reports_match_starter_idx
    ON nfl_injury_reports (match_id, is_starter)
    WHERE is_starter = TRUE;

COMMIT;
