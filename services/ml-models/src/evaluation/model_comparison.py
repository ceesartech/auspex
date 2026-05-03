"""Model comparison and A/B testing framework."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.base_model import BaseModel
from .metrics import compute_all_metrics, compute_roi

logger = logging.getLogger(__name__)


class ModelComparison:
    """Compare multiple models on the same dataset."""

    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.results: Dict[str, Dict[str, float]] = {}

    def add_model(self, name: str, model: BaseModel) -> None:
        self.models[name] = model

    def evaluate_all(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        odds: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Evaluate all models and return comparison table."""
        rows = []
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)
                metrics = compute_all_metrics(
                    y_true=y,
                    y_pred=y_pred,
                    y_pred_proba=y_proba,
                    task=model.config.prediction_task,
                )

                if odds is not None:
                    roi_metrics = compute_roi(y, y_pred, odds, y_pred_proba=y_proba)
                    metrics["roi"] = roi_metrics["roi"]
                    metrics["profit"] = roi_metrics["profit"]

                metrics["model"] = name
                rows.append(metrics)
                self.results[name] = metrics
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")

        df = pd.DataFrame(rows)
        if "model" in df.columns:
            df = df.set_index("model")
        return df

    def statistical_test(
        self,
        model_a: str,
        model_b: str,
        X: pd.DataFrame,
        y: np.ndarray,
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]:
        """Bootstrap test for significant difference between two models."""
        if model_a not in self.models or model_b not in self.models:
            raise ValueError("Both models must be added first")

        proba_a = self.models[model_a].predict_proba(X)
        proba_b = self.models[model_b].predict_proba(X)

        n = len(y)
        metric_diffs = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            y_boot = y[idx]

            pred_a = np.argmax(proba_a[idx], axis=1)
            pred_b = np.argmax(proba_b[idx], axis=1)

            acc_a = np.mean(pred_a == y_boot)
            acc_b = np.mean(pred_b == y_boot)
            metric_diffs.append(acc_a - acc_b)

        metric_diffs = np.array(metric_diffs)
        p_value = float(np.mean(metric_diffs <= 0)) if np.mean(metric_diffs) > 0 else float(np.mean(metric_diffs >= 0))

        return {
            "model_a": model_a,
            "model_b": model_b,
            "mean_diff": float(np.mean(metric_diffs)),
            "std_diff": float(np.std(metric_diffs)),
            "p_value": p_value,
            "significant_at_05": p_value < 0.05,
            "ci_95": (float(np.percentile(metric_diffs, 2.5)), float(np.percentile(metric_diffs, 97.5))),
        }


class ABTestTracker:
    """Track A/B test results between model versions."""

    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []

    def record_prediction(
        self,
        test_name: str,
        model_name: str,
        match_id: str,
        predicted: int,
        actual: Optional[int] = None,
        probability: Optional[float] = None,
        odds: Optional[float] = None,
    ) -> None:
        self.test_results.append({
            "test_name": test_name,
            "model_name": model_name,
            "match_id": match_id,
            "predicted": predicted,
            "actual": actual,
            "probability": probability,
            "odds": odds,
        })

    def get_results(self, test_name: str) -> pd.DataFrame:
        filtered = [r for r in self.test_results if r["test_name"] == test_name]
        return pd.DataFrame(filtered)

    def compute_test_metrics(self, test_name: str) -> Dict[str, Dict[str, float]]:
        """Compute metrics per model for a given test."""
        df = self.get_results(test_name)
        df = df.dropna(subset=["actual"])

        metrics_by_model = {}
        for model_name, group in df.groupby("model_name"):
            y_true = group["actual"].values
            y_pred = group["predicted"].values
            accuracy = float(np.mean(y_true == y_pred))
            n = len(group)

            result = {"accuracy": accuracy, "n_predictions": n}

            if "odds" in group.columns and group["odds"].notna().any():
                roi_data = compute_roi(y_true.astype(int), y_pred.astype(int), group["odds"].values)
                result["roi"] = roi_data["roi"]

            metrics_by_model[model_name] = result

        return metrics_by_model
