"""LightGBM model implementation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .base_model import BaseModel
from .model_config import ModelConfig, PredictionTask

logger = logging.getLogger(__name__)


class LightGBMMatchPredictor(BaseModel):
    """LightGBM model for match prediction (outcome or over/under)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.label_encoder = LabelEncoder()
        self._is_binary = config.prediction_task in (
            PredictionTask.OVER_UNDER,
            PredictionTask.BTTS,
        )

    def prepare_data(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
    ) -> tuple:
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in features:
                features.remove(target)

        self.feature_names = features

        X = df[features].copy().fillna(df[features].median())
        y = self.label_encoder.fit_transform(df[target])

        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        features: Optional[List[str]] = None,
        target: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        target = target or self.config.target_column
        logger.info("Starting LightGBM training...")

        X_train, y_train = self.prepare_data(train_df, target, features)

        params = {
            k: v
            for k, v in self.config.hyperparameters.items()
            if k not in ("n_estimators", "random_state")
        }
        params["verbose"] = -1

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)

        valid_sets = [dtrain]
        valid_names = ["train"]

        callbacks = [lgb.log_evaluation(period=self.config.training_config.get("verbose", 100))]

        if val_df is not None:
            X_val = val_df[self.feature_names].fillna(
                val_df[self.feature_names].median()
            )
            y_val = self.label_encoder.transform(val_df[target])
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("val")
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.config.training_config.get(
                        "early_stopping_rounds", 50
                    )
                )
            )

        self.model = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=self.config.hyperparameters.get("n_estimators", 400),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self.is_fitted = True
        self.training_history = dict(self.model.evals_result_) if hasattr(self.model, "evals_result_") else {}

        # Feature importance
        importance = self.model.feature_importance(importance_type="gain")
        self.feature_importance = dict(
            sorted(
                zip(self.feature_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        validation_metrics = {}
        if val_df is not None:
            validation_metrics = self.evaluate(X_val, y_val)

        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")
        return {
            "training_history": self.training_history,
            "feature_importance": self.feature_importance,
            "validation_metrics": validation_metrics,
            "best_iteration": self.model.best_iteration,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        proba = self.predict_proba(X)
        if self._is_binary:
            return (proba >= 0.5).astype(int)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        if isinstance(X, pd.DataFrame):
            X_vals = X[self.feature_names].fillna(X[self.feature_names].median())
        else:
            X_vals = X

        raw = self.model.predict(X_vals)

        if self._is_binary and raw.ndim == 1:
            return raw
        return raw

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path))

        metadata = {
            "feature_names": self.feature_names,
            "label_classes": self.label_encoder.classes_.tolist(),
            "config": self.config.to_dict(),
            "feature_importance": self.feature_importance,
            "is_fitted": self.is_fitted,
        }
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        path = Path(path)

        self.model = lgb.Booster(model_file=str(path))

        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.feature_names = metadata["feature_names"]
        self.label_encoder.classes_ = np.array(metadata["label_classes"])
        self.feature_importance = metadata["feature_importance"]
        self.is_fitted = metadata["is_fitted"]

        logger.info(f"Model loaded from {path}")
