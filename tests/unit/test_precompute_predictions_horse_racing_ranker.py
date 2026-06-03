"""Unit tests for the LambdaMART precompute script.

Focuses on the parts of the script that don't require a live DB:
  * CLI shape (defaults + flag parsing)
  * MODEL_NAME / MODEL_VERSION lockdown (race_predictions UNIQUE
    constraint includes these — renaming silently creates a parallel
    set of rows)
  * predict_for_races: shape of the returned {race_id: {entrant_id:
    prob}} dict, ordering vs the input frame, per-race softmax
    invariants.

The DB-touching paths (load_target_races, store_prediction, run)
are smoke-tested on prod after deploy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
ML_SRC = Path(__file__).resolve().parents[2] / "services" / "ml-models" / "src"
sys.path.insert(0, str(ML_SRC))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pred = _load("precompute_predictions_horse_racing_ranker", "precompute_predictions_horse_racing_ranker.py")


# ── Constants lockdown ─────────────────────────────────────────────


class TestConstants:
    def test_model_name(self):
        # race_predictions UNIQUE constraint includes model_name +
        # model_version. Renaming without bumping the version means
        # the predictor writes a parallel set of rows that nothing
        # reads — silent data drift.
        assert pred.MODEL_NAME == "lightgbm_ranker_v1"
        assert pred.MODEL_VERSION == "1.0.0"


# ── CLI plumbing ───────────────────────────────────────────────────


class TestCli:
    def test_defaults(self):
        args = pred.parse_args(["--database-url", "postgresql://x"])
        # 2-day window matches the consensus precompute defaults so
        # the two predictors stay scoped to the same window.
        assert args.days == 2
        assert args.race_ids is None

    def test_race_ids_parses(self):
        args = pred.parse_args(["--race-ids", "a,b,c", "--database-url", "x"])
        assert args.race_ids == "a,b,c"

    def test_days_can_be_overridden(self):
        args = pred.parse_args(["--days", "7", "--database-url", "x"])
        assert args.days == 7


# ── predict_for_races: race-grouped dict + softmax shape ───────────


class TestPredictForRaces:
    def _frame(self):
        # Two races, 3 + 2 entrants. Order matches what the loader
        # returns (race_date ASC, race_id, program_number).
        return pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r1", "r2", "r2"],
                "entrant_id": ["e1", "e2", "e3", "e4", "e5"],
                "feat_a": [0.2, 0.4, 0.6, 0.3, 0.5],
                "feat_b": [1.0, 2.0, 3.0, 1.5, 2.5],
            }
        )

    def _trained_ranker(self):
        # Tiny synthetic train just enough to get a fitted model
        # whose predict_probabilities works. Real signal isn't the
        # point — the unit under test is the dict-shape projection.
        from predictors.horse_racing_ranker import HorseRacingRanker, HorseRacingRankerConfig

        np.random.seed(0)
        train = pd.DataFrame(
            {
                "race_id": [f"trace{i // 4}" for i in range(40)],
                "feat_a": np.random.rand(40),
                "feat_b": np.random.rand(40),
                "target": ([1, 0, 0, 0] * 10),
            }
        )
        groups = train["race_id"].value_counts(sort=False).values
        model = HorseRacingRanker(HorseRacingRankerConfig(n_estimators=20, early_stopping_rounds=10, learning_rate=0.1))
        model.fit(
            X_train=train[["feat_a", "feat_b"]],
            y_train=train["target"].to_numpy(dtype=np.int64),
            groups_train=groups,
        )
        return model

    def test_returns_one_dict_per_race(self):
        model = self._trained_ranker()
        out = pred.predict_for_races(model, self._frame(), ["feat_a", "feat_b"])
        assert set(out.keys()) == {"r1", "r2"}

    def test_each_race_dict_keys_are_entrant_ids(self):
        model = self._trained_ranker()
        out = pred.predict_for_races(model, self._frame(), ["feat_a", "feat_b"])
        assert set(out["r1"].keys()) == {"e1", "e2", "e3"}
        assert set(out["r2"].keys()) == {"e4", "e5"}

    def test_per_race_probabilities_sum_to_one(self):
        # Critical load-bearing invariant — if the softmax is
        # mis-scoped to the WRONG entrants (e.g. groups desync) the
        # sums won't be 1.0 and downstream EV math breaks.
        model = self._trained_ranker()
        out = pred.predict_for_races(model, self._frame(), ["feat_a", "feat_b"])
        for race_id, probs in out.items():
            assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)

    def test_empty_frame_returns_empty_dict(self):
        # Safe short-circuit so the caller doesn't have to special-
        # case empty inputs.
        model = self._trained_ranker()
        empty = pd.DataFrame(columns=["race_id", "entrant_id", "feat_a", "feat_b"])
        assert pred.predict_for_races(model, empty, ["feat_a", "feat_b"]) == {}

    def test_handles_extra_columns_in_input(self):
        # Real scoring frames carry identifier columns (race_id,
        # entrant_id, program_number, etc) that aren't in
        # feature_cols. The reindex inside predict_for_races picks
        # only the model's feature_cols and ignores the rest.
        model = self._trained_ranker()
        frame = self._frame().assign(program_number=[1, 2, 3, 1, 2], horse_name=["A", "B", "C", "D", "E"])
        out = pred.predict_for_races(model, frame, ["feat_a", "feat_b"])
        assert set(out.keys()) == {"r1", "r2"}
        for probs in out.values():
            assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
