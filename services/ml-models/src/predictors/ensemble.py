"""Ensemble predictor combining multiple model predictions."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base_model import BaseModel
from .model_config import ModelConfig

logger = logging.getLogger(__name__)


class EnsemblePredictor(BaseModel):
    """Weighted ensemble of multiple models."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.models: Dict[str, BaseModel] = {}
        self.weights: Dict[str, float] = {}
        self.calibrators: Dict[str, Any] = {}
        # Isotonic calibrator fit on the BLENDED ensemble output (the
        # served probability). None until fit_calibrator() runs at the
        # end of training; applied automatically in predict_proba and
        # round-tripped through save()/load() via the metadata JSON.
        self.calibrator: Optional[Any] = None

    def add_model(self, name: str, model: BaseModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self.models[name] = model
        self.weights[name] = weight
        logger.info(f"Added model '{name}' with weight {weight:.3f}")

    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        self.models.pop(name, None)
        self.weights.pop(name, None)

    def prepare_data(self, df: pd.DataFrame, target: str) -> tuple:
        return df, df[target].values if target in df.columns else None

    def train(
        self,
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Optimize ensemble weights on validation data.

        `train_df` is accepted for API consistency with the base models
        but the ensemble fits its blend weights from val_df only.
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        if val_df is None:
            logger.warning("No validation data; using equal weights")
            n = len(self.models)
            self.weights = {name: 1.0 / n for name in self.models}
            self.is_fitted = True
            return {"weights": self.weights}

        optimize = self.config.hyperparameters.get("optimize_weights", True)
        if not optimize:
            n = len(self.models)
            self.weights = {name: 1.0 / n for name in self.models}
            self.is_fitted = True
            return {"weights": self.weights}

        logger.info("Optimizing ensemble weights...")

        target = kwargs.get("target", self.config.target_column)

        # Collect predictions from each model
        model_names = list(self.models.keys())
        model_predictions = {}
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(val_df)
                model_predictions[name] = proba
            except Exception as e:
                logger.error(f"Model '{name}' prediction failed: {e}")

        if not model_predictions:
            raise ValueError("No model produced valid predictions")

        model_names = list(model_predictions.keys())
        y_true = val_df[target].values if target in val_df.columns else None
        if y_true is None:
            raise ValueError(f"Target column '{target}' not in validation data")

        # Encode labels if needed
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_true)

        def neg_log_loss(w):
            w_normalized = np.exp(w) / np.sum(np.exp(w))  # Softmax normalization
            blended = np.zeros_like(model_predictions[model_names[0]])
            for i, name in enumerate(model_names):
                blended += w_normalized[i] * model_predictions[name]

            # Clip to avoid log(0)
            blended = np.clip(blended, 1e-10, 1.0)

            # Log loss
            n_samples = len(y_encoded)
            loss = 0.0
            for j in range(n_samples):
                loss -= np.log(blended[j, y_encoded[j]])
            return loss / n_samples

        # Optimize
        n_models = len(model_names)
        x0 = np.zeros(n_models)

        result = minimize(
            neg_log_loss,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 1000},
        )

        # Convert to normalized weights
        optimized_w = np.exp(result.x) / np.sum(np.exp(result.x))
        min_weight = self.config.hyperparameters.get("min_weight", 0.05)

        # Apply minimum weight constraint
        for i, w in enumerate(optimized_w):
            if w < min_weight:
                optimized_w[i] = min_weight
        optimized_w /= optimized_w.sum()

        self.weights = {name: float(optimized_w[i]) for i, name in enumerate(model_names)}
        self.is_fitted = True

        logger.info(f"Optimized weights: {self.weights}")

        # Evaluate ensemble
        validation_metrics = self.evaluate(val_df, y_encoded)

        return {
            "weights": self.weights,
            "optimization_loss": float(result.fun),
            "validation_metrics": validation_metrics,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def fit_calibrator(self, y_proba: np.ndarray, y_true: np.ndarray, method: str = "isotonic") -> None:
        """Fit the post-blend calibrator on validation ensemble output.
        Pass the RAW blended probabilities (predict_proba(..., apply_calibration=False))
        and the encoded validation labels."""
        from training.calibration import ProbabilityCalibrator

        cal = ProbabilityCalibrator(method=method)
        cal.fit(y_proba, y_true)
        self.calibrator = cal

    # A member failure drops its weight from the blend. Below this surviving
    # fraction we refuse to serve: a majority-degraded blend is not "the
    # ensemble, slightly worse" — it silently becomes whichever members
    # survived. This exact failure mode served a constant Poisson/DC prior
    # for every soccer match for a month (July-2026 audit §1.1) because the
    # learned members KeyError'd on renamed features and were swallowed here.
    MIN_SURVIVING_WEIGHT_FRACTION = 0.5

    def predict_proba(self, X: pd.DataFrame, apply_calibration: bool = True) -> np.ndarray:
        if not self.models:
            raise ValueError("No models in ensemble")

        blended = None
        total_weight = 0.0
        attempted_weight = 0.0
        failed: list[str] = []

        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w <= 0:
                continue
            attempted_weight += w
            try:
                proba = model.predict_proba(X)
                if blended is None:
                    blended = np.zeros_like(proba)
                blended += w * proba
                total_weight += w
            except Exception as e:
                failed.append(name)
                logger.error("Ensemble member %r failed to predict: %s", name, e)

        if blended is None:
            raise ValueError(f"No model produced valid predictions (all members failed: {failed})")

        if failed and attempted_weight > 0 and total_weight < self.MIN_SURVIVING_WEIGHT_FRACTION * attempted_weight:
            raise ValueError(
                f"Ensemble degraded: members {failed} failed; surviving weight "
                f"{total_weight:.3f} of {attempted_weight:.3f} is below the "
                f"{self.MIN_SURVIVING_WEIGHT_FRACTION:.0%} serving threshold — refusing to serve"
            )

        if total_weight > 0:
            blended /= total_weight

        # Apply the served calibration map. apply_calibration=False is used
        # at training time to fit the calibrator on RAW blended output and
        # to report raw-vs-calibrated metrics on the held-out test set.
        if apply_calibration and self.calibrator is not None and self.calibrator.is_fitted:
            try:
                blended = self.calibrator.calibrate(blended)
            except Exception as e:
                logger.error("Ensemble calibration failed, serving raw proba: %s", e)

        return blended

    def predict_with_disagreement(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Predict with individual model outputs and disagreement measure."""
        individual = {}
        for name, model in self.models.items():
            try:
                individual[name] = model.predict_proba(X)
            except Exception as e:
                logger.error(f"Model '{name}' failed: {e}")

        ensemble_proba = self.predict_proba(X)

        # Compute disagreement (std across models)
        if len(individual) > 1:
            all_preds = np.array(list(individual.values()))
            disagreement = np.mean(np.std(all_preds, axis=0), axis=1)
        else:
            disagreement = np.zeros(len(X))

        return {
            "ensemble_proba": ensemble_proba,
            "individual_predictions": individual,
            "disagreement": disagreement,
        }

    def save(self, path: str) -> None:
        import json
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "weights": self.weights,
            "model_names": list(self.models.keys()),
            "config": self.config.to_dict(),
            # Embed the calibrator IN the metadata (not a sidecar) so it
            # survives the registry -> promotion -> serve copy of model.bin
            # as a single file. None when training ran --no-calibration.
            "calibrator": self.calibrator.to_dict() if self.calibrator is not None else None,
        }
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Ensemble metadata saved to {path}")

    def load(self, path: str) -> None:
        import json
        from pathlib import Path

        path = Path(path)
        with open(path, "r") as f:
            metadata = json.load(f)

        self.weights = metadata["weights"]
        cal_meta = metadata.get("calibrator")
        if cal_meta:
            from training.calibration import ProbabilityCalibrator

            self.calibrator = ProbabilityCalibrator.from_dict(cal_meta)
        self.is_fitted = True
        logger.info(f"Ensemble metadata loaded from {path}")
