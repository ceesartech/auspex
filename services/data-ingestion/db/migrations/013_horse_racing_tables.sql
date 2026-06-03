-- Phase 14: horse racing schema.
--
-- Horse racing is structurally different from the team / 1v1 sports
-- already in the system:
--   * A race has N runners (typically 6-14, sometimes 20+).
--   * Outcome is a finish ORDER, not a binary winner. Multiple
--     markets are layered on top: win, place, show, exacta, etc.
--   * Horses, jockeys, and trainers all carry stats that travel
--     across races — they're first-class entities, not derived
--     from a single race's row.
--
-- The existing `matches` schema (home_team_id / away_team_id) can't
-- cleanly represent multi-runner races. This migration adds a
-- parallel `races` + `race_entrants` shape that downstream
-- horse-racing code (ingest, features, training, recs) keys off,
-- while leaving the matches-based sports untouched.
--
-- Horses / jockeys / trainers get their own tables so that:
--   * Rolling form (last N races a horse ran in) is a JOIN, not a
--     JSON dive into matches.metadata.
--   * Same jockey appearing on multiple horses across multiple
--     races deduplicates to one row.
--   * Trainer / jockey win-rate features for future races become
--     a simple aggregate query.
--
-- The trade-off: every horse / jockey / trainer name we ingest
-- gets a row, populated lazily. Roster size grows monotonically
-- (~80k active racehorses worldwide, ~5k jockeys, ~10k trainers).
-- All manageable.
--
-- Reversibility: drop in reverse FK order
--   race_entrants → races → horses / jockeys / trainers → leagues.

-- ============================================================
-- Horses, jockeys, trainers — long-lived entities
-- ============================================================
--
-- normalized_name (UNIQUE) is the dedup key, same shape as
-- teams.normalized_name. Vendor name variants ("Big Brown" vs "BIG
-- BROWN" vs "Big Brown (USA)") collapse to one row at lookup time.

CREATE TABLE IF NOT EXISTS horses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    -- Country suffix from racing IDs ("(USA)", "(GB)", "(IRE)").
    -- Stored separately so the normalized_name doesn't carry it
    -- (lets "Big Brown (USA)" and "Big Brown" match).
    country TEXT,
    birth_year INT,
    -- Sex categories per North American racing convention:
    -- colt, filly, gelding, mare, horse (entire 5+yo male),
    -- rig (cryptorchid), ridgling (rare).
    sex TEXT,
    sire TEXT,
    dam TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_horses_normalized ON horses(normalized_name);

CREATE TRIGGER trg_horses_updated_at BEFORE UPDATE ON horses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


CREATE TABLE IF NOT EXISTS jockeys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    country TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jockeys_normalized ON jockeys(normalized_name);

CREATE TRIGGER trg_jockeys_updated_at BEFORE UPDATE ON jockeys
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


CREATE TABLE IF NOT EXISTS trainers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    country TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trainers_normalized ON trainers(normalized_name);

CREATE TRIGGER trg_trainers_updated_at BEFORE UPDATE ON trainers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- ============================================================
-- races — one row per race
-- ============================================================
--
-- The race row is the rough equivalent of a `matches` row for team
-- sports. We DON'T link to the matches table (would force 1-to-1
-- mapping when conceptually races are independent).
--
-- league_id reuses the existing leagues table — each track is one
-- "league" with sport='horse_racing' (e.g., Churchill Downs is one
-- league row, Royal Ascot another). This lets the existing
-- season-rollover / venue helpers keep working without a parallel
-- track table.
--
-- Surface + going (turf condition like 'good to soft') are
-- first-class because they're high-signal features for the model:
-- a turf horse on dirt is a different bet than the same horse on
-- its preferred surface.

CREATE TABLE IF NOT EXISTS races (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    league_id UUID REFERENCES leagues(id) ON DELETE SET NULL,
    -- External vendor IDs for cross-referencing (Racing API
    -- race_id, Equibase race_id, etc.). JSONB keyed by source.
    external_ids JSONB DEFAULT '{}',
    race_date TIMESTAMP WITH TIME ZONE NOT NULL,
    track_name TEXT NOT NULL,
    -- Race number within the day's card (typically 1-12 in the US).
    race_number INT,
    -- Distance in meters. We use the SI unit so different countries'
    -- furlongs / yards / metric variants normalise. Caller converts:
    --   1 furlong = 201.168 m
    --   1 yard = 0.9144 m
    distance_meters INT,
    -- 'turf' | 'dirt' | 'synthetic' | 'all_weather'. Free text so
    -- vendor variations ('polytrack', 'tapeta', 'fibresand') can be
    -- preserved without a new ENUM each time.
    surface TEXT,
    -- Track condition / going (varies by surface):
    --   dirt:  fast | sloppy | muddy | slow | wet_fast
    --   turf:  firm | good | yielding | soft | heavy
    --   syn:   fast | standard | sealed
    -- Free text, normalised case-insensitively at read time.
    track_condition TEXT,
    -- Race class / grade. North America: G1, G2, G3, Listed, Allowance,
    -- Claiming $X, Maiden, MSW. UK/Ireland: Group 1, etc.
    race_class TEXT,
    purse_currency TEXT DEFAULT 'USD',
    purse_amount NUMERIC,
    -- Number of horses entered (denormalised from race_entrants for
    -- cheap reads; updated by the ingest path).
    field_size INT,
    status TEXT NOT NULL CHECK (status IN (
        'scheduled', 'live', 'finished', 'cancelled', 'abandoned'
    )),
    venue_coords_id UUID REFERENCES venue_coords(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Same race never gets ingested twice: (track, date, race_number)
    -- is the unique key. Multiple data sources writing the same race
    -- collapse via this constraint.
    UNIQUE(track_name, race_date, race_number)
);

CREATE INDEX IF NOT EXISTS idx_races_date ON races(race_date);
CREATE INDEX IF NOT EXISTS idx_races_status ON races(status);
CREATE INDEX IF NOT EXISTS idx_races_league_date ON races(league_id, race_date);

CREATE TRIGGER trg_races_updated_at BEFORE UPDATE ON races
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- ============================================================
-- race_entrants — one row per horse-in-race
-- ============================================================
--
-- Same horse appearing in 50 different races over a career
-- generates 50 race_entrants rows but only 1 horses row.
-- finish_position is NULL until the race is finished; scratched
-- horses are kept (scratched=true) so the ingest path is
-- idempotent and post-race we still have the field-size context.
--
-- starting_price is the closing pari-mutuel; morning_line_odds
-- is the track handicapper's published opening number. Both are
-- useful — sharp money is the move FROM morning line TO closing.

CREATE TABLE IF NOT EXISTS race_entrants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    race_id UUID NOT NULL REFERENCES races(id) ON DELETE CASCADE,
    horse_id UUID NOT NULL REFERENCES horses(id) ON DELETE RESTRICT,
    jockey_id UUID REFERENCES jockeys(id) ON DELETE SET NULL,
    trainer_id UUID REFERENCES trainers(id) ON DELETE SET NULL,
    -- Program number / saddlecloth number / post position.
    -- Same horse name appearing twice in different cloths is a
    -- coupled entry (rare in modern racing); each gets its own row.
    program_number INT NOT NULL,
    -- Post position (gate draw). Can differ from program number
    -- in some jurisdictions; null if not yet drawn.
    post_position INT,
    -- Weight assigned, in pounds. UK racing uses stones+pounds —
    -- caller converts (1 stone = 14 lb).
    weight_carried_lbs NUMERIC(5, 2),
    -- Morning line (track handicapper's pre-betting opening odds)
    -- and starting price (final closing pari-mutuel). Decimal
    -- format throughout the system. NULL on scratched entries.
    morning_line_odds NUMERIC(8, 2),
    starting_price NUMERIC(8, 2),
    -- 1-based finish position. NULL until results post. Disqualified
    -- horses keep their crossing-line position here; the DQ flag is
    -- separate so the model can choose how to handle DQs.
    finish_position INT,
    disqualified BOOLEAN DEFAULT false,
    scratched BOOLEAN DEFAULT false,
    -- Per-entrant numeric stats from the data source:
    --   speed_figure (Beyer / Timeform / Topspeed depending on source)
    --   margin_lengths (distance behind winner)
    --   final_time_seconds (full race time for THIS horse)
    -- Stored as JSONB to avoid schema churn when adding new metrics.
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(race_id, program_number)
);

CREATE INDEX IF NOT EXISTS idx_race_entrants_race ON race_entrants(race_id);
CREATE INDEX IF NOT EXISTS idx_race_entrants_horse ON race_entrants(horse_id);
CREATE INDEX IF NOT EXISTS idx_race_entrants_jockey ON race_entrants(jockey_id);
CREATE INDEX IF NOT EXISTS idx_race_entrants_trainer ON race_entrants(trainer_id);

CREATE TRIGGER trg_race_entrants_updated_at BEFORE UPDATE ON race_entrants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- ============================================================
-- Race-aware version of predictions
-- ============================================================
--
-- The existing predictions table is keyed on match_id. Horse-racing
-- predictions are per-(race, runner) — we need P(this horse wins
-- this race) for every horse in every race. Adding a race_id +
-- entrant_id column to predictions would force every other sport to
-- carry NULLs.
--
-- New parallel race_predictions table mirrors predictions' columns
-- but with the right keys. Same unique-constraint shape so the
-- store/upsert path is symmetric.

CREATE TABLE IF NOT EXISTS race_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    race_id UUID NOT NULL REFERENCES races(id) ON DELETE CASCADE,
    entrant_id UUID NOT NULL REFERENCES race_entrants(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    -- 'win' | 'place' | 'show' | 'exacta' | 'trifecta'. For multi-
    -- horse markets (exacta, trifecta) the entrant_id points at the
    -- FIRST horse in the combo and the rest live in metadata.
    prediction_type TEXT NOT NULL CHECK (prediction_type IN (
        'win', 'place', 'show', 'exacta', 'trifecta', 'superfecta'
    )),
    -- For win: P(this horse wins). For place: P(top 2). etc.
    confidence NUMERIC(6, 5) NOT NULL,
    -- Full per-horse distribution (when applicable), JSONB keyed by
    -- entrant_id::text. Mirrors the predictions.probabilities shape.
    probabilities JSONB DEFAULT '{}',
    -- Actual outcome (for grading). For 'win': 1 if this entrant
    -- finished first, else 0.
    actual_outcome NUMERIC(3, 2),
    is_correct BOOLEAN,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- One row per (race, entrant, model, version, type). Re-runs
    -- upsert; different markets get their own rows per entrant.
    UNIQUE(race_id, entrant_id, model_name, model_version, prediction_type)
);

CREATE INDEX IF NOT EXISTS idx_race_predictions_race ON race_predictions(race_id);
CREATE INDEX IF NOT EXISTS idx_race_predictions_model ON race_predictions(model_name);

CREATE TRIGGER trg_race_predictions_updated_at BEFORE UPDATE ON race_predictions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- ============================================================
-- Race-aware betting recommendations
-- ============================================================
--
-- Same justification as race_predictions: betting_recommendations is
-- keyed on match_id. Parallel race_recommendations gives us per-
-- entrant rec rows without polluting the match-based path.

CREATE TABLE IF NOT EXISTS race_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    race_prediction_id UUID NOT NULL REFERENCES race_predictions(id) ON DELETE CASCADE,
    race_id UUID NOT NULL REFERENCES races(id) ON DELETE CASCADE,
    entrant_id UUID NOT NULL REFERENCES race_entrants(id) ON DELETE CASCADE,
    bet_type TEXT NOT NULL CHECK (bet_type IN (
        'win', 'place', 'show', 'exacta', 'trifecta', 'superfecta'
    )),
    selection TEXT NOT NULL,  -- "horse name" for win/place/show; "H1-H2-H3" for exotics
    odds_at_recommendation NUMERIC(8, 2) NOT NULL,
    bookmaker TEXT,
    confidence_rating TEXT,
    expected_value NUMERIC(6, 4),
    kelly_stake NUMERIC(6, 4),
    recommended_stake NUMERIC(10, 2),
    reasoning TEXT,
    risk_factors JSONB DEFAULT '[]',
    status TEXT DEFAULT 'pending' CHECK (status IN (
        'pending', 'placed', 'won', 'lost', 'void', 'cancelled'
    )),
    actual_result TEXT,
    profit_loss NUMERIC(10, 2),
    settled_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_race_recs_race ON race_recommendations(race_id);
CREATE INDEX IF NOT EXISTS idx_race_recs_status ON race_recommendations(status);
CREATE INDEX IF NOT EXISTS idx_race_recs_created ON race_recommendations(created_at DESC);

CREATE TRIGGER trg_race_recs_updated_at BEFORE UPDATE ON race_recommendations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- ============================================================
-- Race features cache
-- ============================================================
--
-- features_cache is per-match. For horse racing we cache per-race
-- and per-entrant features separately (race-level features apply
-- to all runners; entrant-level features differ per horse).
--
-- A v1 simplification: store one row per race with the full feature
-- matrix (race-level scalars + per-entrant arrays) in JSONB. The
-- training pipeline unpacks it. Avoids N feature_cache rows per
-- race when N=14.

CREATE TABLE IF NOT EXISTS race_features_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    race_id UUID NOT NULL REFERENCES races(id) ON DELETE CASCADE,
    feature_set TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    -- {race_level: {...}, entrants: [{entrant_id: ..., features: {...}}, ...]}
    features JSONB NOT NULL,
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(race_id, feature_set, feature_version)
);

CREATE INDEX IF NOT EXISTS idx_race_features_race ON race_features_cache(race_id);
CREATE INDEX IF NOT EXISTS idx_race_features_expires ON race_features_cache(expires_at);
