"""XGBoost model implementation with SHAP explainability."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from .base_model import BaseModel
from .model_config import ModelConfig

logger = logging.getLogger(__name__)


class XGBoostMatchPredictor(BaseModel):
    """XGBoost model for match prediction."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.label_encoder = LabelEncoder()
        self.shap_explainer = None

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

        X = df[features].copy()
        X = X.fillna(X.median())

        y = self.label_encoder.fit_transform(df[target])

        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        return X.values, y

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        features: Optional[List[str]] = None,
        target: str = "match_outcome",
        **kwargs,
    ) -> Dict[str, Any]:
        logger.info("Starting XGBoost training...")

        X_train, y_train = self.prepare_data(train_df, target, features)

        dtrain = xgb.DMatrix(
            X_train, label=y_train, feature_names=self.feature_names
        )

        params = {
            k: v
            for k, v in self.config.hyperparameters.items()
            if k != "n_estimators"
        }

        evals = [(dtrain, "train")]
        if val_df is not None:
            X_val = val_df[self.feature_names].fillna(
                val_df[self.feature_names].median()
            ).values
            y_val = self.label_encoder.transform(val_df[target])
            dval = xgb.DMatrix(
                X_val, label=y_val, feature_names=self.feature_names
            )
            evals.append((dval, "val"))

        evals_result: Dict = {}
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.config.hyperparameters.get("n_estimators", 500),
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=self.config.training_config.get(
                "early_stopping_rounds", 50
            ),
            verbose_eval=self.config.training_config.get("verbose", 100),
        )

        self.is_fitted = True
        self.training_history = evals_result

        # Feature importance
        importance_dict = self.model.get_score(importance_type="gain")
        self.feature_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        # Initialize SHAP
        try:
            import shap

            self.shap_explainer = shap.TreeExplainer(self.model)
        except ImportError:
            logger.warning("shap not installed, explainability disabled")

        # Validation metrics
        validation_metrics = {}
        if val_df is not None:
            validation_metrics = self.evaluate(
                pd.DataFrame(X_val, columns=self.feature_names), y_val
            )

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
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        if isinstance(X, pd.DataFrame):
            X_vals = X[self.feature_names].fillna(
                X[self.feature_names].median()
            )
            dmatrix = xgb.DMatrix(X_vals.values, feature_names=self.feature_names)
        else:
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)

        return self.model.predict(dmatrix)

    def predict_single(
        self, features: Dict[str, Any], explain: bool = True
    ) -> Dict[str, Any]:
        """Predict single match with optional SHAP explanation."""
        X = pd.DataFrame([features])[self.feature_names]
        X = X.fillna(X.median())

        proba = self.predict_proba(X)[0]
        predicted_class = int(np.argmax(proba))

        result = {
            "predicted_class": predicted_class,
            "predicted_label": self.label_encoder.inverse_transform(
                [predicted_class]
            )[0],
            "probabilities": {
                "home_win": float(proba[0]),
                "draw": float(proba[1]),
                "away_win": float(proba[2]),
            },
            "confidence": float(np.max(proba)),
        }

        if explain and self.shap_explainer is not None:
            shap_values = self.shap_explainer.shap_values(X.values)
            explanation = self._format_shap_explanation(
                X.values[0], shap_values[predicted_class][0]
            )
            result["explanation"] = explanation

        return result

    def _format_shap_explanation(
        self,
        feature_values: np.ndarray,
        shap_values: np.ndarray,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        top_indices = np.argsort(np.abs(shap_values))[-top_n:][::-1]

        explanations = []
        for idx in top_indices:
            name = self.feature_names[idx]
            val = feature_values[idx]
            sv = shap_values[idx]
            impact = "increases" if sv > 0 else "decreases"
            explanations.append({
                "feature": name,
                "value": float(val),
                "impact": float(sv),
                "description": f"{name} ({val:.2f}) {impact} probability",
            })

        return {"top_features": explanations}

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

        self.model = xgb.Booster()
        self.model.load_model(str(path))

        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.feature_names = metadata["feature_names"]
        self.label_encoder.classes_ = np.array(metadata["label_classes"])
        self.feature_importance = metadata["feature_importance"]
        self.is_fitted = metadata["is_fitted"]

        try:
            import shap

            self.shap_explainer = shap.TreeExplainer(self.model)
        except ImportError:
            pass

        logger.info(f"Model loaded from {path}")
