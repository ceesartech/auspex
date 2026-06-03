"""Unit tests for compute_features_horse_racing — pure helpers.

The DB-touching paths (rolling form, course form, etc.) are
integration territory. These tests cover the pure helpers: surface
bucketing, going-severity collapse, race-class tier, with_defaults
backfill, and the NEUTRAL_DEFAULTS shape.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fhr = _load("compute_features_horse_racing", "compute_features_horse_racing.py")


# ── Constants lockdown ──────────────────────────────────────────────


class TestConstants:
    def test_feature_set_pinned(self):
        # Training queries + predict-time lookups all key on this
        # exact pair.
        assert fhr.FEATURE_SET == "horse_racing_baseline"
        assert fhr.FEATURE_VERSION == "v1"

    def test_horse_window_is_5(self):
        # Last 5 starts is the standard "recent form" window in
        # racing form pages. Longer windows include too much old
        # info; shorter windows are too noisy.
        assert fhr.HORSE_WINDOW == 5

    def test_jockey_trainer_window_30_days(self):
        # 30-day strike rate is the standard "current form" stat
        # for jockeys + trainers in racing.
        assert fhr.JOCKEY_DAYS == 30
        assert fhr.TRAINER_DAYS == 30


# ── Surface bucketing ──────────────────────────────────────────────


class TestBucketSurface:
    def test_turf_variants(self):
        assert fhr._bucket_surface("Turf") == fhr.SURFACE_TURF
        assert fhr._bucket_surface("turf") == fhr.SURFACE_TURF
        assert fhr._bucket_surface("Grass") == fhr.SURFACE_TURF

    def test_dirt(self):
        assert fhr._bucket_surface("Dirt") == fhr.SURFACE_DIRT
        assert fhr._bucket_surface("dirt") == fhr.SURFACE_DIRT

    def test_synthetic_variants(self):
        # Polytrack / Tapeta / "all weather" / "synthetic" all map
        # to the same bucket — they share modeling characteristics
        # (consistent, weather-independent).
        assert fhr._bucket_surface("Synthetic") == fhr.SURFACE_SYNTHETIC
        assert fhr._bucket_surface("polytrack") == fhr.SURFACE_SYNTHETIC
        assert fhr._bucket_surface("Tapeta") == fhr.SURFACE_SYNTHETIC
        assert fhr._bucket_surface("All Weather") == fhr.SURFACE_SYNTHETIC
        assert fhr._bucket_surface("all_weather") == fhr.SURFACE_SYNTHETIC

    def test_unknown_for_missing_or_empty(self):
        assert fhr._bucket_surface(None) == fhr.SURFACE_UNKNOWN
        assert fhr._bucket_surface("") == fhr.SURFACE_UNKNOWN
        assert fhr._bucket_surface("Unrecognised Surface") == fhr.SURFACE_UNKNOWN


# ── Going severity ─────────────────────────────────────────────────


class TestGoingSeverity:
    def test_firm_fast_is_zero(self):
        # Dry / firm conditions = 0 severity. Track plays fast.
        assert fhr._going_severity("Fast") == 0
        assert fhr._going_severity("Firm") == 0
        assert fhr._going_severity("Good to Firm") == 0

    def test_good_standard_is_one(self):
        # Modal / "good" conditions.
        assert fhr._going_severity("Good") == 1
        assert fhr._going_severity("Standard") == 1

    def test_soft_sloppy_is_two(self):
        assert fhr._going_severity("Soft") == 2
        assert fhr._going_severity("Sloppy") == 2
        assert fhr._going_severity("Yielding") == 2

    def test_heavy_muddy_is_three(self):
        assert fhr._going_severity("Heavy") == 3
        assert fhr._going_severity("Muddy") == 3

    def test_handles_case_and_whitespace(self):
        assert fhr._going_severity("  GOOD  ") == 1
        assert fhr._going_severity("good   to   firm") == 0

    def test_unknown_returns_none(self):
        # Unrecognised vendor values return None so caller falls
        # back to NEUTRAL_DEFAULTS.
        assert fhr._going_severity("alien terrain") is None
        assert fhr._going_severity(None) is None
        assert fhr._going_severity("") is None


# ── Race class tier ────────────────────────────────────────────────


class TestRaceClassTier:
    def test_north_america_grades(self):
        assert fhr._race_class_tier("G1") == 1
        assert fhr._race_class_tier("Grade 1") == 1
        assert fhr._race_class_tier("Grade I") == 1
        assert fhr._race_class_tier("G2") == 2
        assert fhr._race_class_tier("G3") == 3

    def test_uk_groups(self):
        assert fhr._race_class_tier("Group 1") == 1
        assert fhr._race_class_tier("Group 2") == 2

    def test_listed_is_4(self):
        assert fhr._race_class_tier("Listed") == 4

    def test_allowance_handicap_5_to_6(self):
        assert fhr._race_class_tier("Allowance") == 5
        assert fhr._race_class_tier("Claiming $50k") == 6
        assert fhr._race_class_tier("Maiden Special Weight") == 6
        assert fhr._race_class_tier("MSW") == 6

    def test_uk_classes(self):
        # UK Class 1 ≈ G2/G3 tier; Class 7 ≈ maiden tier.
        assert fhr._race_class_tier("Class 1") == 2
        assert fhr._race_class_tier("Class 7") == 6

    def test_unknown_returns_none(self):
        assert fhr._race_class_tier("alien race") is None
        assert fhr._race_class_tier(None) is None


# ── Neutral defaults ────────────────────────────────────────────────


class TestNeutralDefaults:
    def test_every_default_is_finite_number(self):
        for k, v in fhr.NEUTRAL_DEFAULTS.items():
            assert isinstance(v, (int, float)), f"{k} is {type(v).__name__}"
            assert v == v, f"{k} is NaN"  # NaN != NaN

    def test_morning_line_default_matches_uniform_prior(self):
        # 1 / field_size = uniform; field_size=10 → 0.10.
        assert fhr.NEUTRAL_DEFAULTS["morning_line_implied_prob"] == 0.10
        assert fhr.NEUTRAL_DEFAULTS["field_size"] == 10.0

    def test_surface_one_hot_sums_to_one(self):
        # Exactly one surface_* default should be 1.0.
        keys = ("surface_turf", "surface_dirt", "surface_synthetic", "surface_unknown")
        s = sum(fhr.NEUTRAL_DEFAULTS[k] for k in keys)
        assert s == 1.0

    def test_horse_debut_defaults_to_no_starts(self):
        # First-time starter has 0 prior runs — defaults reflect
        # that explicitly so the model learns "0 starts" is a real
        # signal, not missing data.
        assert fhr.NEUTRAL_DEFAULTS["horse_starts_last_5"] == 0.0
        assert fhr.NEUTRAL_DEFAULTS["horse_win_rate_last_5"] == 0.0
        assert fhr.NEUTRAL_DEFAULTS["horse_place_rate_last_5"] == 0.0

    def test_jockey_trainer_modal_rates(self):
        # ~10% win rate is the tour modal for both jockeys and
        # trainers across the population.
        assert fhr.NEUTRAL_DEFAULTS["jockey_30d_win_rate"] == 0.10
        assert fhr.NEUTRAL_DEFAULTS["trainer_30d_win_rate"] == 0.10


# ── _with_defaults ─────────────────────────────────────────────────


class TestWithDefaults:
    def test_missing_keys_filled(self):
        out = fhr._with_defaults({})
        for k in fhr.NEUTRAL_DEFAULTS:
            assert k in out

    def test_provided_values_override(self):
        out = fhr._with_defaults({"horse_win_rate_last_5": 0.25})
        assert out["horse_win_rate_last_5"] == 0.25

    def test_none_replaced_with_default(self):
        out = fhr._with_defaults({"jockey_30d_win_rate": None})
        assert out["jockey_30d_win_rate"] == fhr.NEUTRAL_DEFAULTS["jockey_30d_win_rate"]

    def test_extra_keys_preserved(self):
        # Forward-compat for v2 features.
        out = fhr._with_defaults({"experimental_speed_figure": 92.0})
        assert out["experimental_speed_figure"] == 92.0


# ── Race-level feature assembly ────────────────────────────────────


class TestRaceLevelFeatures:
    def test_field_size_distance_one_hot(self):
        meta = {
            "field_size": 8,
            "distance_meters": 1609,
            "surface": "Turf",
            "track_condition": "Good",
            "race_class": "Group 1",
        }
        out = fhr.race_level_features(meta)
        assert out["field_size"] == 8.0
        assert out["distance_meters"] == 1609.0
        assert out["surface_turf"] == 1.0
        assert out["surface_dirt"] == 0.0
        assert out["going_severity"] == 1.0
        assert out["race_class_tier"] == 1.0

    def test_missing_fields_omit(self):
        # Empty meta produces a sparse dict that NEUTRAL_DEFAULTS
        # will fill in downstream.
        out = fhr.race_level_features({})
        assert out["surface_unknown"] == 1.0
        assert "distance_meters" not in out  # caller's _with_defaults supplies
        assert "going_severity" not in out


# ── Argparse plumbing ──────────────────────────────────────────────


class TestCli:
    def test_default_days_is_seven(self):
        args = fhr.parse_args(["--database-url", "x"])
        assert args.days == 7
        assert args.force is False
        assert args.all_finished is False
        assert args.race_ids is None

    def test_race_ids_parses(self):
        args = fhr.parse_args(["--race-ids", "a,b,c", "--database-url", "x"])
        assert args.race_ids == "a,b,c"

    def test_all_finished_flag(self):
        args = fhr.parse_args(["--all-finished", "--database-url", "x"])
        assert args.all_finished is True
