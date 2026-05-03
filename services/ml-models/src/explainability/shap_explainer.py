"""SHAP-based model explainability."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Unified SHAP explainer for all model types.

    Supports TreeExplainer (XGBoost/LightGBM), DeepExplainer (NN),
    and KernelExplainer (any model).
    """

    def __init__(self, model, model_type: str = "tree"):
        """
        Args:
            model: Trained model object.
            model_type: 'tree' (XGBoost/LightGBM), 'deep' (NN), or 'kernel' (generic).
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.expected_value = None

    def fit(self, background_data: Optional[np.ndarray] = None) -> None:
        """Initialize the SHAP explainer."""
        try:
            import shap

            if self.model_type == "tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "deep":
                if background_data is None:
                    raise ValueError("background_data required for DeepExplainer")
                import torch

                self.explainer = shap.DeepExplainer(
                    self.model, torch.FloatTensor(background_data)
                )
            elif self.model_type == "kernel":
                if background_data is None:
                    raise ValueError("background_data required for KernelExplainer")
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba
                    if hasattr(self.model, "predict_proba")
                    else self.model.predict,
                    background_data,
                )
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            self.expected_value = self.explainer.expected_value
            logger.info(f"Initialized {self.model_type} SHAP explainer")

        except ImportError:
            logger.warning("shap not installed, explainability unavailable")

    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute SHAP values for given inputs.

        Returns dict with shap_values, expected_value, and feature_names.
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call fit() first.")

        shap_values = self.explainer.shap_values(X)

        return {
            "shap_values": shap_values,
            "expected_value": self.expected_value,
            "feature_names": feature_names,
        }

    def explain_single(
        self,
        x: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Explain a single prediction with top contributing features.

        Args:
            x: Single sample, shape (1, n_features) or (n_features,).
            feature_names: Names of the features.
            top_n: Number of top features to return.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        result = self.explain(x, feature_names)
        shap_values = result["shap_values"]

        # Handle multiclass (list of arrays) vs binary (single array)
        if isinstance(shap_values, list):
            # Pick the class with highest probability
            sv_abs = [np.abs(sv[0]).sum() for sv in shap_values]
            dominant_class = int(np.argmax(sv_abs))
            sv = shap_values[dominant_class][0]
        else:
            sv = shap_values[0]

        top_indices = np.argsort(np.abs(sv))[-top_n:][::-1]

        explanations = []
        for idx in top_indices:
            name = feature_names[idx] if feature_names else f"feature_{idx}"
            explanations.append({
                "feature": name,
                "value": float(x[0, idx]),
                "shap_value": float(sv[idx]),
                "impact": "positive" if sv[idx] > 0 else "negative",
                "abs_impact": float(abs(sv[idx])),
            })

        return {
            "top_features": explanations,
            "expected_value": result["expected_value"],
        }

    def feature_importance(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute global feature importance via mean |SHAP|.

        Args:
            X: Dataset to compute importance over, shape (n_samples, n_features).
            feature_names: Feature names.

        Returns:
            Dict mapping feature name to mean absolute SHAP value.
        """
        result = self.explain(X, feature_names)
        shap_values = result["shap_values"]

        # Aggregate across classes if multiclass
        if isinstance(shap_values, list):
            combined = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            combined = np.abs(shap_values)

        mean_abs = np.mean(combined, axis=0)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(mean_abs))]

        importance = dict(
            sorted(
                zip(feature_names, mean_abs.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        return importance

    def interaction_values(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute SHAP interaction values (tree models only)."""
        if self.model_type != "tree":
            raise ValueError("Interaction values only supported for tree models")

        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call fit() first.")

        interaction = self.explainer.shap_interaction_values(X)

        return {
            "interaction_values": interaction,
            "feature_names": feature_names,
        }
