"""Shared test fixtures for ML models service."""

# Import torch early to avoid PyTorch/XGBoost native library conflicts
# when running the full test suite
import torch  # noqa: F401

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============= SAMPLE DATA =============

@pytest.fixture
def sample_match_data():
    """Generate sample match data for training/testing."""
    np.random.seed(42)
    n = 200

    teams = ["TeamA", "TeamB", "TeamC", "TeamD", "TeamE", "TeamF"]
    dates = pd.date_range("2023-01-01", periods=n, freq="3D")

    data = {
        "match_date": dates,
        "home_team": np.random.choice(teams, n),
        "away_team": np.random.choice(teams, n),
        "home_score": np.random.poisson(1.4, n),
        "away_score": np.random.poisson(1.1, n),
    }

    df = pd.DataFrame(data)

    # Ensure home != away
    mask = df["home_team"] == df["away_team"]
    df.loc[mask, "away_team"] = "TeamZ"

    # Derive target
    df["match_outcome"] = np.where(
        df["home_score"] > df["away_score"], 0,
        np.where(df["home_score"] == df["away_score"], 1, 2)
    )

    # Generate numeric features
    n_features = 20
    for i in range(n_features):
        df[f"feature_{i}"] = np.random.randn(n)

    # Add some odds columns
    df["home_odds"] = np.random.uniform(1.3, 4.0, n)
    df["draw_odds"] = np.random.uniform(2.5, 5.0, n)
    df["away_odds"] = np.random.uniform(1.5, 6.0, n)

    return df


@pytest.fixture
def feature_columns():
    """Feature column names."""
    return [f"feature_{i}" for i in range(20)]


@pytest.fixture
def train_val_test_split(sample_match_data):
    """Split sample data into train/val/test sets chronologically."""
    df = sample_match_data.sort_values("match_date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    return {
        "train": df.iloc[:train_end].copy(),
        "val": df.iloc[train_end:val_end].copy(),
        "test": df.iloc[val_end:].copy(),
    }


# ============= MODEL CONFIGS =============

@pytest.fixture
def xgboost_config():
    """XGBoost config for testing (fast settings)."""
    from models.model_config import ModelConfig, ModelType, PredictionTask

    return ModelConfig(
        name="test_xgboost",
        model_type=ModelType.XGBOOST,
        prediction_task=PredictionTask.MATCH_OUTCOME,
        version="test",
        hyperparameters={
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "tree_method": "hist",
            "random_state": 42,
        },
        features=[],
        target_column="match_outcome",
        loss_function="multi:softprob",
        metrics=["accuracy", "log_loss"],
        training_config={
            "early_stopping_rounds": 5,
            "eval_metric": "mlogloss",
            "verbose": 0,
        },
    )


@pytest.fixture
def lightgbm_config():
    """LightGBM config for testing (fast settings)."""
    from models.model_config import ModelConfig, ModelType, PredictionTask

    return ModelConfig(
        name="test_lightgbm",
        model_type=ModelType.LIGHTGBM,
        prediction_task=PredictionTask.MATCH_OUTCOME,
        version="test",
        hyperparameters={
            "objective": "multiclass",
            "num_class": 3,
            "boosting_type": "gbdt",
            "num_leaves": 16,
            "learning_rate": 0.1,
            "n_estimators": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 5,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "random_state": 42,
            "verbose": -1,
        },
        features=[],
        target_column="match_outcome",
        loss_function="multiclass",
        metrics=["accuracy", "log_loss"],
        training_config={"early_stopping_rounds": 5, "verbose": 0},
    )


@pytest.fixture
def nn_config():
    """Neural network config for testing (fast settings)."""
    from models.model_config import ModelConfig, ModelType, PredictionTask

    return ModelConfig(
        name="test_nn",
        model_type=ModelType.NEURAL_NETWORK,
        prediction_task=PredictionTask.MATCH_OUTCOME,
        version="test",
        hyperparameters={
            "hidden_layers": [32, 16],
            "dropout_rate": 0.2,
            "learning_rate": 0.01,
            "batch_size": 64,
            "epochs": 5,
            "batch_norm": True,
        },
        features=[],
        target_column="match_outcome",
        loss_function="categorical_crossentropy",
        metrics=["accuracy"],
        training_config={
            "early_stopping_patience": 3,
            "reduce_lr_patience": 2,
            "verbose": 0,
        },
    )


@pytest.fixture
def poisson_config():
    """Poisson model config for testing."""
    from models.model_config import ModelConfig, ModelType, PredictionTask

    return ModelConfig(
        name="test_poisson",
        model_type=ModelType.POISSON,
        prediction_task=PredictionTask.MATCH_OUTCOME,
        version="test",
        hyperparameters={
            "max_goals": 4,
            "home_advantage": 0.25,
            "regularization": 0.01,
            "max_iterations": 50,
            "convergence_threshold": 1e-4,
        },
        features=[],
        target_column="match_outcome",
        loss_function="poisson_nll",
        metrics=["accuracy"],
        training_config={},
    )


@pytest.fixture
def dixon_coles_config():
    """Dixon-Coles model config for testing."""
    from models.model_config import ModelConfig, ModelType, PredictionTask

    return ModelConfig(
        name="test_dixon_coles",
        model_type=ModelType.DIXON_COLES,
        prediction_task=PredictionTask.MATCH_OUTCOME,
        version="test",
        hyperparameters={
            "max_goals": 4,
            "rho_init": -0.1,
            "time_decay": 0.001,
            "home_advantage": 0.25,
            "regularization": 0.01,
            "max_iterations": 50,
            "convergence_threshold": 1e-4,
        },
        features=[],
        target_column="match_outcome",
        loss_function="dixon_coles_nll",
        metrics=["accuracy"],
        training_config={},
    )


@pytest.fixture
def ensemble_config():
    """Ensemble config for testing."""
    from models.model_config import ModelConfig, ModelType, PredictionTask

    return ModelConfig(
        name="test_ensemble",
        model_type=ModelType.ENSEMBLE,
        prediction_task=PredictionTask.MATCH_OUTCOME,
        version="test",
        hyperparameters={
            "combination_method": "weighted_average",
            "optimize_weights": True,
            "weight_optimization_metric": "log_loss",
            "min_weight": 0.05,
        },
        features=[],
        target_column="match_outcome",
        loss_function="ensemble",
        metrics=["accuracy", "log_loss"],
        training_config={},
    )
