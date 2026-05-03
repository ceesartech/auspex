"""Abstract base model class for all prediction models."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .model_config import ModelConfig

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.training_history: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame, target: str) -> tuple:
        """Prepare data for training."""
        pass

    @abstractmethod
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions (class labels)."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data."""
        from evaluation.metrics import compute_all_metrics

        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)

        return compute_all_metrics(
            y_true=y,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            task=self.config.prediction_task,
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance
