"""Feature selection utilities for model training."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Select relevant features using various methods."""

    def __init__(self):
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}

    def select_by_variance(
        self,
        X: pd.DataFrame,
        threshold: float = 0.01,
    ) -> List[str]:
        """Remove features with low variance.

        Args:
            X: Feature DataFrame.
            threshold: Minimum variance threshold.
        """
        variances = X.var()
        selected = variances[variances > threshold].index.tolist()

        removed = set(X.columns) - set(selected)
        if removed:
            logger.info(f"Removed {len(removed)} low-variance features: {removed}")

        self.selected_features = selected
        return selected

    def select_by_correlation(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95,
    ) -> List[str]:
        """Remove highly correlated features (keep the first of each pair).

        Args:
            X: Feature DataFrame.
            threshold: Maximum correlation threshold.
        """
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = set()
        for col in upper.columns:
            highly_correlated = upper.index[upper[col] > threshold].tolist()
            to_drop.update(highly_correlated)

        selected = [c for c in X.columns if c not in to_drop]

        if to_drop:
            logger.info(f"Removed {len(to_drop)} highly correlated features")

        self.selected_features = selected
        return selected

    def select_by_importance(
        self,
        feature_importance: Dict[str, float],
        top_n: Optional[int] = None,
        min_importance: float = 0.0,
    ) -> List[str]:
        """Select features by model-derived importance scores.

        Args:
            feature_importance: Dict mapping feature name to importance.
            top_n: Keep top N features.
            min_importance: Minimum importance threshold.
        """
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        # Apply threshold
        filtered = [(f, s) for f, s in sorted_features if s > min_importance]

        # Apply top_n
        if top_n is not None:
            filtered = filtered[:top_n]

        selected = [f for f, _ in filtered]
        self.feature_scores = dict(filtered)
        self.selected_features = selected

        logger.info(
            f"Selected {len(selected)} features from {len(feature_importance)} by importance"
        )
        return selected

    def select_by_mutual_information(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        top_n: Optional[int] = None,
    ) -> List[str]:
        """Select features using mutual information with the target.

        Args:
            X: Feature DataFrame.
            y: Target array.
            top_n: Number of top features to keep.
        """
        from sklearn.feature_selection import mutual_info_classif

        X_filled = X.fillna(0)
        mi_scores = mutual_info_classif(X_filled, y, random_state=42)
        mi_dict = dict(zip(X.columns, mi_scores))

        sorted_features = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)

        if top_n is not None:
            sorted_features = sorted_features[:top_n]

        selected = [f for f, _ in sorted_features]
        self.feature_scores = dict(sorted_features)
        self.selected_features = selected

        logger.info(f"Selected {len(selected)} features by mutual information")
        return selected

    def select_recursive(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        n_features: int = 50,
    ) -> List[str]:
        """Recursive Feature Elimination (RFE).

        Args:
            model: sklearn-compatible estimator with feature_importances_.
            X: Feature DataFrame.
            y: Target array.
            n_features: Number of features to select.
        """
        from sklearn.feature_selection import RFE

        X_filled = X.fillna(0)
        selector = RFE(model, n_features_to_select=n_features, step=10)
        selector.fit(X_filled, y)

        selected = X.columns[selector.support_].tolist()
        self.selected_features = selected

        rankings = dict(zip(X.columns, selector.ranking_))
        self.feature_scores = {f: 1.0 / r for f, r in rankings.items()}

        logger.info(f"RFE selected {len(selected)} features")
        return selected

    def apply_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the current feature selection to a DataFrame."""
        if not self.selected_features:
            logger.warning("No features selected, returning all columns")
            return df
        available = [f for f in self.selected_features if f in df.columns]
        return df[available]

    def get_report(self) -> Dict[str, any]:
        """Get a summary report of the current selection."""
        return {
            "n_selected": len(self.selected_features),
            "features": self.selected_features,
            "scores": self.feature_scores,
        }
