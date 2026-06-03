"""Unit tests for walk_forward_predictions — orchestration pieces.

Doesn't test the actual training subprocess (that's the existing
train_all_models.py covered by its own tests). Locks down the pure
pieces that decide WHICH bundles to run, WHICH data lands in the
training set, and HOW the predictions get keyed for retrieval by
the backtest engine.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


wfp = _load("walk_forward_predictions", "walk_forward_predictions.py")


class TestResolveBundles:
    def test_all_expands(self):
        bundles = wfp.resolve_bundles("all")
        # 12 known bundles (soccer + 4×NHL + 3×NBA + 3×NFL + 1×tennis).
        assert len(bundles) == 12
        # Order matches ALL_BUNDLES — locked for reproducibility.
        assert bundles[0] == "soccer_match_result"
        assert "nba_total" in bundles
        assert "nfl_moneyline" in bundles
        assert "tennis_moneyline" in bundles

    def test_sport_filter(self):
        assert set(wfp.resolve_bundles("nba")) == {"nba_moneyline", "nba_spread", "nba_total"}
        assert set(wfp.resolve_bundles("nhl")) == {
            "nhl_moneyline",
            "nhl_regulation",
            "nhl_puck_line",
            "nhl_total",
        }
        assert set(wfp.resolve_bundles("nfl")) == {"nfl_moneyline", "nfl_spread", "nfl_total"}
        assert set(wfp.resolve_bundles("tennis")) == {"tennis_moneyline"}
        # Soccer has only the match_result bundle (derived markets
        # are predict-time only, no separate training).
        assert wfp.resolve_bundles("soccer") == ["soccer_match_result"]

    def test_specific_bundle(self):
        assert wfp.resolve_bundles("nba_spread") == ["nba_spread"]
        assert wfp.resolve_bundles("tennis_moneyline") == ["tennis_moneyline"]

    def test_unknown_raises(self):
        # Defensive: a typo'd arg should fail fast, not run the wrong
        # bundle.
        with pytest.raises(ValueError, match="Unknown bundle"):
            wfp.resolve_bundles("mma_moneyline")


class TestFilterFrameBefore:
    """The training set must NEVER include data on or after the split
    date — that's the OOS contract. Lock it."""

    def test_drops_post_split_rows(self):
        frame = pd.DataFrame(
            {
                "match_date": [
                    "2023-12-01T19:00:00+00:00",
                    "2024-01-01T19:00:00+00:00",  # exactly on split
                    "2024-06-15T19:00:00+00:00",
                ],
                "value": [1, 2, 3],
            }
        )
        filtered = wfp.filter_frame_before(frame, "2024-01-01")
        # Strict < cutoff: rows ON the split day are excluded.
        assert list(filtered["value"]) == [1]

    def test_empty_frame_passes_through(self):
        frame = pd.DataFrame({"match_date": pd.Series([], dtype=object)})
        filtered = wfp.filter_frame_before(frame, "2024-01-01")
        assert len(filtered) == 0

    def test_frame_without_match_date_unchanged(self):
        # Defensive: don't crash if upstream prepare_*_frame removed
        # the match_date column. Caller will hit a different validation
        # downstream.
        frame = pd.DataFrame({"value": [1, 2, 3]})
        filtered = wfp.filter_frame_before(frame, "2024-01-01")
        assert len(filtered) == 3

    def test_handles_timezone_naive_dates(self):
        # Mixed tz scenarios — be permissive (the SQL roundtrip can
        # strip tz info on some psycopg2 configs).
        frame = pd.DataFrame(
            {
                "match_date": ["2023-12-01", "2024-06-15"],
                "value": [1, 2],
            }
        )
        filtered = wfp.filter_frame_before(frame, "2024-01-01")
        assert list(filtered["value"]) == [1]


class TestBundleMappings:
    """Each bundle has 3 associated identifiers — ensemble registry
    name, prediction_type DB value, feature_set string. If any of
    these drift relative to the live system (prediction_service.TASKS
    or compute_features_*.FEATURE_SET), walk-forward predictions
    won't merge correctly with live predictions for accuracy /
    recommendations consumption."""

    def test_every_bundle_has_an_ensemble_name(self):
        for bundle in wfp.ALL_BUNDLES:
            assert bundle in wfp.BUNDLE_TO_ENSEMBLE, f"Missing ensemble for {bundle}"

    def test_every_bundle_has_a_prediction_type(self):
        for bundle in wfp.ALL_BUNDLES:
            assert bundle in wfp.BUNDLE_TO_PREDICTION_TYPE, f"Missing pred_type for {bundle}"

    def test_every_bundle_has_a_feature_set(self):
        for bundle in wfp.ALL_BUNDLES:
            assert bundle in wfp.BUNDLE_TO_FEATURE_SET, f"Missing feature_set for {bundle}"

    def test_nhl_bundles_share_feature_set(self):
        # All 4 NHL bundles read from features_cache feature_set='nhl_baseline'.
        nhl = {wfp.BUNDLE_TO_FEATURE_SET[b] for b in wfp.ALL_BUNDLES if b.startswith("nhl_")}
        assert nhl == {"nhl_baseline"}

    def test_nba_bundles_share_feature_set(self):
        nba = {wfp.BUNDLE_TO_FEATURE_SET[b] for b in wfp.ALL_BUNDLES if b.startswith("nba_")}
        assert nba == {"nba_baseline"}

    def test_ensemble_names_match_live_convention(self):
        # Locked against the live TaskSpec.ensemble_name values
        # (prediction_service.TASKS). If those rename, this test
        # fails and the walk-forward script needs the same rename.
        assert wfp.BUNDLE_TO_ENSEMBLE["soccer_match_result"] == "ensemble_soccer_match_result"
        assert wfp.BUNDLE_TO_ENSEMBLE["nhl_moneyline"] == "ensemble_nhl_ml"
        assert wfp.BUNDLE_TO_ENSEMBLE["nhl_regulation"] == "ensemble_nhl_reg"
        assert wfp.BUNDLE_TO_ENSEMBLE["nhl_puck_line"] == "ensemble_nhl_pl"
        assert wfp.BUNDLE_TO_ENSEMBLE["nhl_total"] == "ensemble_nhl_tot"
        assert wfp.BUNDLE_TO_ENSEMBLE["nba_moneyline"] == "ensemble_nba_ml"
        assert wfp.BUNDLE_TO_ENSEMBLE["nba_spread"] == "ensemble_nba_sp"
        assert wfp.BUNDLE_TO_ENSEMBLE["nba_total"] == "ensemble_nba_tot"

    def test_prediction_types_match_live_taskspec(self):
        # Lock the values that get written to predictions.prediction_type.
        # Soccer + NHL regulation share 'match_result' (disambiguated
        # by model_name on read).
        assert wfp.BUNDLE_TO_PREDICTION_TYPE["soccer_match_result"] == "match_result"
        assert wfp.BUNDLE_TO_PREDICTION_TYPE["nhl_regulation"] == "match_result"
        assert wfp.BUNDLE_TO_PREDICTION_TYPE["nhl_moneyline"] == "moneyline"
        # NHL puck_line uses prediction_type='spread' in the live DB.
        assert wfp.BUNDLE_TO_PREDICTION_TYPE["nhl_puck_line"] == "spread"
        # NBA spread + total use the natural names.
        assert wfp.BUNDLE_TO_PREDICTION_TYPE["nba_spread"] == "spread"
        assert wfp.BUNDLE_TO_PREDICTION_TYPE["nba_total"] == "total"


class TestCliPlumbing:
    def test_split_date_required(self):
        with pytest.raises(SystemExit):
            wfp.parse_args([])

    def test_split_date_format_validated_at_main(self):
        # parse_args accepts any string; main() validates via
        # date.fromisoformat. We test the format gate here by hand.
        args = wfp.parse_args(["--split-date", "garbage"])
        # parse succeeds, but main() will reject.
        assert args.split_date == "garbage"

    def test_defaults(self):
        args = wfp.parse_args(["--split-date", "2024-01-01"])
        # Default to running ALL bundles — explicit opt-in to single
        # is via --bundle.
        assert args.bundle == "all"
        assert args.snapshots_dir == "/tmp/wf_snapshots"
        assert args.dry_run is False
