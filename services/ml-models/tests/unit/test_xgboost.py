"""Tests for XGBoost model."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from models.xgboost_model import XGBoostMatchPredictor


class TestXGBoostMatchPredictor:
    """Tests for XGBoostMatchPredictor."""

    def test_init(self, xgboost_config):
        model = XGBoostMatchPredictor(xgboost_config)
        assert model.config == xgboost_config
        assert model.is_fitted is False
        assert model.model is None

    def test_prepare_data(self, xgboost_config, sample_match_data, feature_columns):
        model = XGBoostMatchPredictor(xgboost_config)
        X, y = model.prepare_data(sample_match_data, "match_outcome", feature_columns)

        assert X.shape == (len(sample_match_data), len(feature_columns))
        assert len(y) == len(sample_match_data)
        assert set(np.unique(y)).issubset({0, 1, 2})

    def test_prepare_data_auto_features(self, xgboost_config, sample_match_data):
        model = XGBoostMatchPredictor(xgboost_config)
        X, y = model.prepare_data(sample_match_data, "match_outcome")

        # Should auto-detect numeric columns
        assert X.shape[0] == len(sample_match_data)
        assert X.shape[1] > 0

    def test_train(self, xgboost_config, train_val_test_split, feature_columns):
        model = XGBoostMatchPredictor(xgboost_config)
        result = model.train(
            train_val_test_split["train"],
            val_df=train_val_test_split["val"],
            features=feature_columns,
            target="match_outcome",
        )

        assert model.is_fitted is True
        assert model.model is not None
        assert "training_history" in result
        assert "feature_importance" in result
        assert len(model.feature_names) == len(feature_columns)

    def test_predict(self, xgboost_config, train_val_test_split, feature_columns):
        model = XGBoostMatchPredictor(xgboost_config)
        model.train(
            train_val_test_split["train"],
            val_df=train_val_test_split["val"],
            features=feature_columns,
        )

        test_df = train_val_test_split["test"]
        preds = model.predict(test_df[feature_columns])
        assert len(preds) == len(test_df)
        assert all(p in [0, 1, 2] for p in preds)

    def test_predict_proba(self, xgboost_config, train_val_test_split, feature_columns):
        model = XGBoostMatchPredictor(xgboost_config)
        model.train(
            train_val_test_split["train"],
            val_df=train_val_test_split["val"],
            features=feature_columns,
        )

        test_df = train_val_test_split["test"]
        proba = model.predict_proba(test_df[feature_columns])
        assert proba.shape == (len(test_df), 3)
        # Probabilities should sum to ~1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_predict_not_fitted(self, xgboost_config, sample_match_data, feature_columns):
        model = XGBoostMatchPredictor(xgboost_config)
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(sample_match_data[feature_columns])

    def test_predict_single(self, xgboost_config, train_val_test_split, feature_columns):
        model = XGBoostMatchPredictor(xgboost_config)
        model.train(
            train_val_test_split["train"],
            val_df=train_val_test_split["val"],
            features=feature_columns,
        )

        row = train_val_test_split["test"].iloc[0]
        features = {col: row[col] for col in feature_columns}
        result = model.predict_single(features, explain=False)

        assert "predicted_class" in result
        assert "probabilities" in result
        assert "confidence" in result
        assert result["predicted_class"] in [0, 1, 2]
        assert 0 < result["confidence"] <= 1.0

    def test_feature_importance(self, xgboost_config, train_val_test_split, feature_columns):
        model = XGBoostMatchPredictor(xgboost_config)
        model.train(
            train_val_test_split["train"],
            val_df=train_val_test_split["val"],
            features=feature_columns,
        )

        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_save_load(self, xgboost_config, train_val_test_split, feature_columns):
        model = XGBoostMatchPredictor(xgboost_config)
        model.train(
            train_val_test_split["train"],
            val_df=train_val_test_split["val"],
            features=feature_columns,
        )

        test_df = train_val_test_split["test"][feature_columns]
        original_preds = model.predict_proba(test_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.bin")
            model.save(path)

            # Verify files exist
            assert Path(path).exists()
            metadata_path = Path(tmpdir) / "model_metadata.json"
            assert metadata_path.exists()

            # Load in new model
            model2 = XGBoostMatchPredictor(xgboost_config)
            model2.load(path)

            assert model2.is_fitted is True
            assert model2.feature_names == model.feature_names

            loaded_preds = model2.predict_proba(test_df)
            np.testing.assert_allclose(original_preds, loaded_preds, atol=1e-6)

    def test_evaluate(self, xgboost_config, train_val_test_split, feature_columns):
        model = XGBoostMatchPredictor(xgboost_config)
        model.train(
            train_val_test_split["train"],
            val_df=train_val_test_split["val"],
            features=feature_columns,
        )

        test_df = train_val_test_split["test"]
        X_test = test_df[feature_columns]
        y_test = model.label_encoder.transform(test_df["match_outcome"])
        metrics = model.evaluate(X_test, y_test)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert "log_loss" in metrics
