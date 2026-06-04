-- 019: widen match_weather.data_kind CHECK to include vc_* values
--
-- The original migration 012 CHECK only allows ('forecast', 'actual',
-- 'unknown') — the Open-Meteo writer's set. PR #24 added the
-- Visual Crossing writer (scripts/fetch_weather_visual_crossing.py)
-- which emits 'vc_forecast', 'vc_actual', 'vc_unknown' so VC rows can
-- be distinguished from Open-Meteo rows during the side-by-side A/B.
-- Without this migration, every VC INSERT fails the check constraint.
--
-- Backward compatibility: pre-existing rows are untouched; the new
-- constraint is a SUPERSET of the old one.

BEGIN;

ALTER TABLE match_weather DROP CONSTRAINT IF EXISTS match_weather_data_kind_check;
ALTER TABLE match_weather ADD CONSTRAINT match_weather_data_kind_check
    CHECK (data_kind IN (
        -- Open-Meteo (migration 012)
        'forecast', 'actual', 'unknown',
        -- Visual Crossing (PR #24 / migration 019)
        'vc_forecast', 'vc_actual', 'vc_unknown'
    ));

COMMIT;
