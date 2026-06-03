-- Phase 13: weather data for outdoor sports (NFL + soccer + Grand Slam tennis).
--
-- Two new tables:
--   * venue_coords — one row per known venue, with lat/lon for the
--     Open-Meteo lookup. Populated by a one-time seed script
--     (scripts/seed_venue_coords.py) for NFL stadiums + Grand Slam
--     tournament venues + top-5 soccer league grounds.
--   * match_weather — one row per (match_id, fetched_at) snapshot.
--     Stores raw + derived weather metrics that the per-sport
--     compute_features_*.py scripts pick up.
--
-- Why a separate table per concern (instead of stuffing into
-- matches.metadata):
--   * venue_coords is queryable independently and reused across
--     many matches at the same venue.
--   * match_weather supports both FORECAST (for upcoming matches)
--     and ACTUAL (for finished matches) — the dual-pass model
--     needs a row per snapshot, not a single overwrite.
--   * JSONB in matches.metadata works for ad-hoc fields but loses
--     the type information + foreign key relationships we want for
--     analytics ("which finished games had >15mph wind").
--
-- Reversibility: drop both tables. The compute_features scripts
-- treat missing weather as "feature absent" via NEUTRAL_DEFAULTS,
-- so removal degrades model quality but doesn't break inference.

-- ============================================================
-- venue_coords: stadium-level lat/lon lookup
-- ============================================================
--
-- Key design choices:
--   * normalized_venue_name (TEXT, UNIQUE) is the lookup key —
--     same shape as teams.normalized_name. Avoids cascading
--     migrations when a vendor renames "MetLife Stadium" to
--     "MetLife Stadium (East Rutherford)".
--   * (latitude, longitude) DOUBLE PRECISION — 6 decimal places
--     resolves to ~10cm, way more than weather APIs need (they
--     typically interpolate to ~5km grids).
--   * timezone IANA string — needed to convert UTC match_date to
--     local time when querying historical weather (some APIs
--     expect local time for the venue).
--   * sport scoping is NOT enforced — same venue can host multiple
--     sports (MetLife = NFL Giants/Jets, sometimes soccer too).
--     The lookup is by venue name, not sport.

CREATE TABLE IF NOT EXISTS venue_coords (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    normalized_venue_name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    -- Indoor venues (NFL domes like AT&T Stadium, Caesars Superdome,
    -- Allegiant Stadium) bypass weather feature computation since
    -- conditions inside are climate-controlled. Flagged here so the
    -- fetch script can skip them entirely.
    is_indoor BOOLEAN NOT NULL DEFAULT false,
    -- Source of the coords (manual seed, geocoding API, etc.) — kept
    -- for auditability when coords drift between sources.
    source TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_venue_coords_normalized
    ON venue_coords(normalized_venue_name);

-- ============================================================
-- match_weather: per-match weather snapshots
-- ============================================================
--
-- One row per (match_id, fetched_at). For upcoming matches the
-- fetcher writes FORECAST data; for finished matches it writes
-- ACTUAL (historical archive). Multiple snapshots per match are
-- expected as forecasts converge to actuals.
--
-- The compute_features script reads the freshest row per match
-- (highest fetched_at) and treats null fields as missing features
-- (NEUTRAL_DEFAULTS fill in).
--
-- All numeric fields use sensible units:
--   temperature_c: Celsius (Open-Meteo native)
--   wind_kmh:      km/h (Open-Meteo native)
--   precipitation_mm: mm of rain/snow accumulated over the match window
--   humidity_pct:  0-100 relative humidity
--   conditions:    short text label (clear / cloudy / rain / snow / fog)

CREATE TABLE IF NOT EXISTS match_weather (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    -- The venue_coords row that drove the lookup. NULL when the
    -- match's venue couldn't be resolved (the fetcher records the
    -- attempt anyway with conditions='unknown' so we don't retry
    -- repeatedly).
    venue_coords_id UUID REFERENCES venue_coords(id) ON DELETE SET NULL,
    fetched_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    -- 'forecast' for pre-match queries, 'actual' for post-match
    -- archive lookups. Forecast can land repeatedly as kickoff
    -- approaches; actual lands once.
    data_kind TEXT NOT NULL CHECK (data_kind IN ('forecast', 'actual', 'unknown')),
    temperature_c NUMERIC(5, 2),
    wind_kmh NUMERIC(5, 2),
    precipitation_mm NUMERIC(5, 2),
    humidity_pct NUMERIC(4, 1),
    conditions TEXT,
    -- Raw API response — kept for debugging / future feature
    -- extraction without re-fetching.
    raw JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_match_weather_match
    ON match_weather(match_id, fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_match_weather_data_kind
    ON match_weather(data_kind);

-- Convenience view: latest weather snapshot per match. compute_features
-- scripts use this directly to avoid duplicating the
-- DISTINCT-ON-fetched_at logic per sport.
CREATE OR REPLACE VIEW match_weather_latest AS
    SELECT DISTINCT ON (match_id)
        match_id,
        venue_coords_id,
        fetched_at,
        data_kind,
        temperature_c,
        wind_kmh,
        precipitation_mm,
        humidity_pct,
        conditions
    FROM match_weather
    ORDER BY match_id, fetched_at DESC;

-- ============================================================
-- updated_at trigger reuse — venue_coords keeps a fresh
-- updated_at when coords get corrected.
-- ============================================================

CREATE TRIGGER trg_venue_coords_updated_at BEFORE UPDATE ON venue_coords
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
