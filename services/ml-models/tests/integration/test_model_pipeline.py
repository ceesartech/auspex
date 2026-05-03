"""Integration tests for the full model pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.model_config import ModelType, PredictionTask
from models.model_registry import ModelRegistry, ModelVersion
from models.poisson_models import PoissonMatchPredictor
from models.xgboost_model import XGBoostMatchPredictor
from training.calibration import ProbabilityCalibrator
from training.cross_validation import TimeSeriesCV


class TestModelRegistryIntegration:
    """Test model registry with real model save/load."""

    def test_register_and_retrieve(self, xgboost_config, train_val_test_split, feature_columns):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            # Train model
            model = XGBoostMatchPredictor(xgboost_config)
            model.train(
                train_val_test_split["train"],
                val_df=train_val_test_split["val"],
                features=feature_columns,
            )

            # Register
            mv = registry.register_model(
                model=model,
                name="xgboost",
                version="1.0.0",
                metrics={"accuracy": 0.55},
            )

            assert isinstance(mv, ModelVersion)
            assert mv.name == "xgboost"
            assert mv.version == "1.0.0"

            # Retrieve
            latest = registry.get_latest_version("xgboost")
            assert latest is not None
            assert latest.version == "1.0.0"

    def test_promote_and_get_active(self, xgboost_config, train_val_test_split, feature_columns):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            model = XGBoostMatchPredictor(xgboost_config)
            model.train(
                train_val_test_split["train"],
                val_df=train_val_test_split["val"],
                features=feature_columns,
            )

            registry.register_model(model, "xgboost", "1.0.0", {"accuracy": 0.5})
            registry.register_model(model, "xgboost", "1.1.0", {"accuracy": 0.6})

            registry.promote_model("xgboost", "1.1.0")

            active = registry.get_active_version("xgboost")
            assert active is not None
            assert active.version == "1.1.0"
            assert active.is_active is True

    def test_delete_version(self, xgboost_config, train_val_test_split, feature_columns):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            model = XGBoostMatchPredictor(xgboost_config)
            model.train(
                train_val_test_split["train"],
                val_df=train_val_test_split["val"],
                features=feature_columns,
            )

            registry.register_model(model, "xgboost", "1.0.0", {"accuracy": 0.5})
            registry.register_model(model, "xgboost", "1.1.0", {"accuracy": 0.6})

            registry.delete_version("xgboost", "1.0.0")

            versions = registry.get_model_versions("xgboost")
            assert len(versions) == 1
            assert versions[0].version == "1.1.0"

    def test_list_models(self, xgboost_config, train_val_test_split, feature_columns):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            model = XGBoostMatchPredictor(xgboost_config)
            model.train(
                train_val_test_split["train"],
                val_df=train_val_test_split["val"],
                features=feature_columns,
            )

            registry.register_model(model, "xgboost", "1.0.0", {})
            registry.register_model(model, "xgboost", "1.1.0", {})

            models = registry.list_models()
            assert models == {"xgboost": 2}


class TestCVPipelineIntegration:
    """Test cross-validation with real models."""

    def test_poisson_cv(self, poisson_config, sample_match_data):
        cv = TimeSeriesCV(n_splits=3, date_column="match_date")
        result = cv.cross_validate(
            model_class=PoissonMatchPredictor,
            config=poisson_config,
            df=sample_match_data,
            target="match_outcome",
        )

        assert result["n_folds"] == 3
        assert "mean_accuracy" in result
        assert "std_accuracy" in result
        assert 0 <= result["mean_accuracy"] <= 1


class TestCalibrationIntegration:
    """Test calibration with real model predictions."""

    def test_xgboost_calibration(self, xgboost_config, train_val_test_split, feature_columns):
        model = XGBoostMatchPredictor(xgboost_config)
        model.train(
            train_val_test_split["train"],
            val_df=train_val_test_split["val"],
            features=feature_columns,
        )

        val_df = train_val_test_split["val"]
        X_val = val_df[feature_columns].fillna(val_df[feature_columns].median())
        y_val = model.label_encoder.transform(val_df["match_outcome"])

        y_proba = model.predict_proba(X_val)

        # Fit calibrator
        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(y_proba, y_val)

        calibrated = cal.calibrate(y_proba)
        assert calibrated.shape == y_proba.shape
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=0.01)

        # Calibration error should be calculable
        error = cal.calibration_error(y_proba, y_val)
        assert error["ece"] >= 0
        assert error["mce"] >= 0
