"""Unit tests for the LightGBM ranker adapter.

Two layers:
  * Pure helpers (_softmax, _race_nll) — fast invariants.
  * End-to-end fit + predict on a synthetic 4-feature dataset where
    one feature monotonically predicts the winner. The model should
    learn to rank by that feature; the test catches groups-shape
    breakage + a regression where the ranker silently ignores the
    signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from predictors.horse_racing_ranker import HorseRacingRanker, HorseRacingRankerConfig, _race_nll, _softmax  # noqa: E402

# ── _softmax ───────────────────────────────────────────────────────


class TestSoftmax:
    def test_sums_to_one(self):
        # Vanilla softmax across a small race.
        probs = _softmax(np.array([1.0, 2.0, 3.0]), temperature=1.0)
        assert probs.sum() == pytest.approx(1.0)

    def test_temperature_above_one_flattens(self):
        # Higher temperature = more uniform. Test that the
        # max-probability shrinks toward 1/N as T increases.
        scores = np.array([5.0, 1.0, 1.0, 1.0])
        sharp = _softmax(scores, temperature=0.5).max()
        warm = _softmax(scores, temperature=5.0).max()
        assert warm < sharp

    def test_temperature_below_one_sharpens(self):
        # Lower temperature → favourite gets more mass. The mass at
        # the argmax should grow as T decreases.
        scores = np.array([3.0, 1.0, 1.0])
        warm = _softmax(scores, temperature=1.0).max()
        sharp = _softmax(scores, temperature=0.1).max()
        assert sharp > warm

    def test_numerical_stability_with_large_scores(self):
        # Without the max-subtract trick this overflows np.exp.
        scores = np.array([1000.0, 999.0, 998.0])
        probs = _softmax(scores, temperature=1.0)
        # Sums to 1, no NaNs.
        assert probs.sum() == pytest.approx(1.0)
        assert not np.isnan(probs).any()

    def test_empty_returns_empty(self):
        # The corpus filter should never produce a race with zero
        # entrants but defensive — predict-time shouldn't crash.
        assert len(_softmax(np.array([]), temperature=1.0)) == 0


# ── _race_nll ──────────────────────────────────────────────────────


class TestRaceNll:
    def test_lower_when_winner_has_higher_score(self):
        # 2-race setup, winner clearly indicated.
        scores = np.array([5.0, 1.0, 1.0, 5.0])  # race1 then race2
        groups = np.array([2, 2], dtype=np.int64)
        winners = [0, 1]  # race1 winner at local idx 0; race2 at 1
        nll = _race_nll(scores, groups, winners, temperature=1.0)
        # Should be small: the model picked the right horse both times.
        assert nll < 0.5

    def test_higher_when_winner_has_lower_score(self):
        # Same shape but the model picked the wrong horse.
        scores = np.array([1.0, 5.0, 5.0, 1.0])
        groups = np.array([2, 2], dtype=np.int64)
        winners = [0, 1]
        nll = _race_nll(scores, groups, winners, temperature=1.0)
        # NLL roughly = -log(softmax_at_winner) ≈ -log(0.018) ≈ 4
        assert nll > 2.0

    def test_skips_races_without_winner(self):
        # Race with winner=None contributes nothing.
        scores = np.array([1.0, 1.0])
        groups = np.array([2], dtype=np.int64)
        winners = [None]
        assert _race_nll(scores, groups, winners, temperature=1.0) == float("inf")


# ── End-to-end fit + predict ───────────────────────────────────────


def _synthetic_dataset(n_races=200, runners_per_race=8, seed=0):
    """Synthesise a corpus where one feature monotonically predicts
    the winner. Each race has `runners_per_race` entrants; the
    entrant with the highest `signal` wins. A few noise features
    make the test realistic (model shouldn't pick those up if it's
    actually learning)."""
    rng = np.random.RandomState(seed)
    rows = []
    for race_id in range(n_races):
        signal_scores = rng.uniform(0, 1, runners_per_race)
        winner_idx = int(np.argmax(signal_scores))
        for j in range(runners_per_race):
            rows.append(
                {
                    "race_id": f"r{race_id}",
                    "signal": float(signal_scores[j]),
                    "noise_a": float(rng.normal()),
                    "noise_b": float(rng.normal()),
                    "noise_c": float(rng.normal()),
                    "target": 1 if j == winner_idx else 0,
                }
            )
    df = pd.DataFrame(rows)
    return df


class TestFitAndPredict:
    def _split(self, df, n_train_races):
        train_ids = set(f"r{i}" for i in range(n_train_races))
        train = df[df["race_id"].isin(train_ids)].reset_index(drop=True)
        test = df[~df["race_id"].isin(train_ids)].reset_index(drop=True)
        return train, test

    def _groups(self, df):
        return df["race_id"].value_counts(sort=False).values

    def test_learns_the_signal(self):
        # If the ranker is wired correctly, top1 accuracy on a corpus
        # where `signal` is the perfect predictor should be high
        # (>80%) — there's some noise from feature subsampling +
        # learning rate but the signal dominates.
        df = _synthetic_dataset(n_races=300, runners_per_race=8, seed=0)
        train, test = self._split(df, 240)
        feature_cols = ["signal", "noise_a", "noise_b", "noise_c"]

        ranker = HorseRacingRanker(
            HorseRacingRankerConfig(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=15,
                early_stopping_rounds=10,
            )
        )
        ranker.fit(
            X_train=train[feature_cols],
            y_train=train["target"].to_numpy(dtype=np.int64),
            groups_train=self._groups(train),
            X_val=test[feature_cols],
            y_val=test["target"].to_numpy(dtype=np.int64),
            groups_val=self._groups(test),
        )
        assert ranker.is_fitted
        assert ranker.validation_metrics["top1_accuracy"] >= 0.8
        # The signal feature should dominate gain-based importance.
        top_feature = max(ranker.feature_importance.items(), key=lambda kv: kv[1])[0]
        assert top_feature == "signal"

    def test_predict_probabilities_sums_to_one_per_race(self):
        # Race-level softmax invariant: every race's probabilities
        # must sum to 1.0. If the group array desyncs from the
        # frame, this fails because the cursor walks into the next
        # race's rows.
        df = _synthetic_dataset(n_races=120, runners_per_race=6, seed=1)
        train, test = self._split(df, 100)
        feature_cols = ["signal", "noise_a", "noise_b", "noise_c"]
        ranker = HorseRacingRanker(
            HorseRacingRankerConfig(
                n_estimators=50,
                learning_rate=0.1,
                early_stopping_rounds=10,
            )
        )
        ranker.fit(
            X_train=train[feature_cols],
            y_train=train["target"].to_numpy(dtype=np.int64),
            groups_train=self._groups(train),
            X_val=test[feature_cols],
            y_val=test["target"].to_numpy(dtype=np.int64),
            groups_val=self._groups(test),
        )
        probs = ranker.predict_probabilities(test[feature_cols], self._groups(test))
        for race_probs in probs:
            assert race_probs.sum() == pytest.approx(1.0, abs=1e-6)

    def test_groups_mismatch_raises(self):
        # Defensive check: a mismatched group array is a silent
        # killer — the model would train on misaligned races without
        # erroring. Catch at the boundary.
        df = _synthetic_dataset(n_races=50, runners_per_race=6, seed=2)
        feature_cols = ["signal", "noise_a", "noise_b", "noise_c"]
        ranker = HorseRacingRanker()
        with pytest.raises(ValueError, match="groups_train"):
            ranker.fit(
                X_train=df[feature_cols],
                y_train=df["target"].to_numpy(dtype=np.int64),
                groups_train=np.array([5, 5, 5]),  # wrong sum
            )

    def test_predict_before_fit_raises(self):
        ranker = HorseRacingRanker()
        with pytest.raises(ValueError, match="not fitted"):
            ranker.predict_scores(pd.DataFrame({"x": [1.0]}))


class TestLoadFromArtifacts:
    """The precompute path loads a model from /app/models/ rather than
    training fresh on every cron tick — training a model on the full
    155k-row corpus OOM'd the api container. These tests pin the
    save → load → predict round-trip so a future refactor can't
    silently break the artefact format the DAG depends on."""

    def _train_and_save(self, tmp_path):
        import json

        from predictors.horse_racing_ranker import HorseRacingRankerConfig

        rng = np.random.RandomState(0)
        rows = []
        for race in range(80):
            signal = rng.uniform(0, 1, 6)
            for j in range(6):
                rows.append(
                    {
                        "race_id": f"r{race}",
                        "signal": float(signal[j]),
                        "noise": float(rng.normal()),
                        "target": 1 if j == int(np.argmax(signal)) else 0,
                    }
                )
        df = pd.DataFrame(rows)
        groups = df["race_id"].value_counts(sort=False).values
        ranker = HorseRacingRanker(
            HorseRacingRankerConfig(n_estimators=30, learning_rate=0.1, early_stopping_rounds=10)
        )
        ranker.fit(
            X_train=df[["signal", "noise"]],
            y_train=df["target"].to_numpy(dtype=np.int64),
            groups_train=groups,
            X_val=df[["signal", "noise"]],
            y_val=df["target"].to_numpy(dtype=np.int64),
            groups_val=groups,
        )
        out_dir = tmp_path / "ranker_v1"
        out_dir.mkdir()
        ranker.model.booster_.save_model(str(out_dir / "model.bin"))
        with open(out_dir / "feature_names.json", "w") as f:
            json.dump(ranker.feature_names, f)
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(
                {"fit_result": {"temperature": ranker.temperature}, "test_metrics": {}},
                f,
            )
        return ranker, out_dir, df, groups

    def test_load_reconstructs_predictions(self, tmp_path):
        # Save a trained model, load it back, predict — scores from
        # the loaded model should match what the original model
        # produces, otherwise the DAG-deployed predictions drift
        # from what training measured.
        ranker, out_dir, df, groups = self._train_and_save(tmp_path)

        loaded = HorseRacingRanker.load(out_dir)
        assert loaded.is_fitted is True
        assert loaded.feature_names == ranker.feature_names
        assert loaded.temperature == pytest.approx(ranker.temperature)

        # Same scoring frame through both — outputs should match
        # within numerical tolerance.
        X = df[["signal", "noise"]]
        original = ranker.predict_scores(X)
        loaded_scores = loaded.predict_scores(X)
        np.testing.assert_allclose(loaded_scores, original, rtol=1e-6, atol=1e-9)

    def test_load_preserves_softmax_temperature(self, tmp_path):
        # Per-race probabilities must round-trip too — temperature
        # is the load-bearing calibration knob. If we lose it on
        # load, the recs engine sees too-sharp probs and the
        # false-positive value-bet rate explodes (verified
        # empirically: 27 → 1,223 picks when temperature defaulted
        # to 1.0 instead of the tuned ~0.35).
        ranker, out_dir, df, groups = self._train_and_save(tmp_path)
        loaded = HorseRacingRanker.load(out_dir)
        X = df[["signal", "noise"]]
        original_probs = ranker.predict_probabilities(X, groups)
        loaded_probs = loaded.predict_probabilities(X, groups)
        assert len(original_probs) == len(loaded_probs)
        for a, b in zip(original_probs, loaded_probs):
            assert a.sum() == pytest.approx(1.0, abs=1e-6)
            assert b.sum() == pytest.approx(1.0, abs=1e-6)
            for orig, lod in zip(a, b):
                assert orig == pytest.approx(lod, abs=1e-6)

    def test_load_raises_when_artifacts_missing(self, tmp_path):
        # Empty directory: should raise rather than silently load a
        # half-initialised model — the precompute uses that error to
        # log a clear bootstrap message pointing to the training
        # script.
        with pytest.raises(FileNotFoundError):
            HorseRacingRanker.load(tmp_path / "does-not-exist")
