"""Tests for Neural Network model."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from models.neural_network import (
    MatchDataset,
    MatchPredictorNN,
    NeuralNetworkMatchPredictor,
)


class TestMatchDataset:
    """Tests for the PyTorch Dataset."""

    def test_dataset_creation(self):
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randint(0, 3, 50)
        dataset = MatchDataset(X, y)

        assert len(dataset) == 50

    def test_dataset_getitem(self):
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randint(0, 3, 50)
        dataset = MatchDataset(X, y)

        x_item, y_item = dataset[0]
        assert isinstance(x_item, torch.Tensor)
        assert isinstance(y_item, torch.Tensor)
        assert x_item.shape == (10,)


class TestMatchPredictorNN:
    """Tests for the PyTorch Module."""

    def test_forward_pass(self):
        model = MatchPredictorNN(
            input_size=20,
            hidden_layers=[32, 16],
            dropout_rate=0.2,
            num_classes=3,
            batch_norm=True,
        )
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8, 3)

    def test_forward_pass_no_batchnorm(self):
        model = MatchPredictorNN(
            input_size=10,
            hidden_layers=[16, 8],
            dropout_rate=0.1,
            num_classes=3,
            batch_norm=False,
        )
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)


class TestNeuralNetworkMatchPredictor:
    """Tests for the full NN predictor."""

    def test_init(self, nn_config):
        model = NeuralNetworkMatchPredictor(nn_config)
        assert model.is_fitted is False
        assert model.config == nn_config

    def test_prepare_data(self, nn_config, sample_match_data, feature_columns):
        model = NeuralNetworkMatchPredictor(nn_config)
        X, y = model.prepare_data(sample_match_data, "match_outcome", feature_columns)

        assert X.shape == (len(sample_match_data), len(feature_columns))
        assert len(y) == len(sample_match_data)

    def test_train(self, nn_config, train_val_test_split, feature_columns):
        model = NeuralNetworkMatchPredictor(nn_config)
        result = model.train(
            train_val_test_split["train"],
            val_df=train_val_test_split["val"],
            features=feature_columns,
            target="match_outcome",
        )

        assert model.is_fitted is True
        assert model.model is not None
        assert "training_history" in result
        assert len(result["training_history"]["train_loss"]) > 0

    def test_predict(self, nn_config, train_val_test_split, feature_columns):
        model = NeuralNetworkMatchPredictor(nn_config)
        model.train(
            train_val_test_split["train"],
            val_df=train_val_test_split["val"],
            features=feature_columns,
        )

        test_df = train_val_test_split["test"]
        preds = model.predict(test_df)
        assert len(preds) == len(test_df)
        assert all(p in [0, 1, 2] for p in preds)

    def test_predict_proba(self, nn_config, train_val_test_split, feature_columns):
        model = NeuralNetworkMatchPredictor(nn_config)
        model.train(
            train_val_test_split["train"],
            val_df=train_val_test_split["val"],
            features=feature_columns,
        )

        test_df = train_val_test_split["test"]
        proba = model.predict_proba(test_df)
        assert proba.shape == (len(test_df), 3)
        # Softmax output should sum to ~1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=0.01)
        # All probabilities should be non-negative
        assert (proba >= 0).all()

    def test_predict_not_fitted(self, nn_config, sample_match_data):
        model = NeuralNetworkMatchPredictor(nn_config)
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict_proba(sample_match_data)

    def test_save_load(self, nn_config, train_val_test_split, feature_columns):
        model = NeuralNetworkMatchPredictor(nn_config)
        model.train(
            train_val_test_split["train"],
            val_df=train_val_test_split["val"],
            features=feature_columns,
        )

        test_df = train_val_test_split["test"]
        original_preds = model.predict_proba(test_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "nn_model.pt")
            model.save(path)

            assert Path(path).exists()

            model2 = NeuralNetworkMatchPredictor(nn_config)
            model2.load(path)

            assert model2.is_fitted is True
            loaded_preds = model2.predict_proba(test_df)
            np.testing.assert_allclose(original_preds, loaded_preds, atol=1e-5)

    def test_training_with_early_stopping(self, nn_config, train_val_test_split, feature_columns):
        """Ensure early stopping works (training may stop before max epochs)."""
        model = NeuralNetworkMatchPredictor(nn_config)
        result = model.train(
            train_val_test_split["train"],
            val_df=train_val_test_split["val"],
            features=feature_columns,
        )

        # With only 5 epochs and patience 3, may or may not trigger
        n_epochs_run = len(result["training_history"]["train_loss"])
        assert n_epochs_run >= 1
        assert n_epochs_run <= nn_config.hyperparameters["epochs"]
