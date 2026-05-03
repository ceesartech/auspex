"""Batch prediction for multiple matches."""

import logging
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class BatchPredictor:
    """Predict outcomes for batches of matches efficiently."""

    def __init__(self, model: BaseModel, batch_size: int = 256):
        self.model = model
        self.batch_size = batch_size

    def predict_batch(
        self,
        df: pd.DataFrame,
        include_probabilities: bool = True,
    ) -> pd.DataFrame:
        """Predict for a batch of matches.

        Returns DataFrame with predictions, probabilities, and confidence.
        """
        start = time.time()
        n = len(df)
        logger.info(f"Batch predicting {n} matches...")

        all_preds = []
        all_probas = []

        for i in range(0, n, self.batch_size):
            batch = df.iloc[i : i + self.batch_size]
            preds = self.model.predict(batch)
            all_preds.append(preds)

            if include_probabilities:
                probas = self.model.predict_proba(batch)
                all_probas.append(probas)

        y_pred = np.concatenate(all_preds)

        result = df.copy()
        result["prediction"] = y_pred

        if include_probabilities and all_probas:
            y_proba = np.vstack(all_probas)
            if y_proba.ndim > 1:
                for c in range(y_proba.shape[1]):
                    labels = ["home_win", "draw", "away_win"]
                    col_name = labels[c] if c < len(labels) else f"class_{c}"
                    result[f"prob_{col_name}"] = y_proba[:, c]
            result["confidence"] = np.max(y_proba, axis=1) if y_proba.ndim > 1 else y_proba

        duration = time.time() - start
        logger.info(f"Batch prediction complete: {n} matches in {duration:.2f}s")
        return result

    def predict_with_value(
        self,
        df: pd.DataFrame,
        odds_columns: Optional[Dict[int, str]] = None,
        min_value: float = 0.05,
    ) -> pd.DataFrame:
        """Predict and identify value bets (predicted prob > implied prob).

        Args:
            df: Match data with features and odds columns.
            odds_columns: Mapping of class index to odds column name.
                e.g., {0: 'home_odds', 1: 'draw_odds', 2: 'away_odds'}
            min_value: Minimum edge required to flag as value bet.
        """
        result = self.predict_batch(df, include_probabilities=True)

        if odds_columns is None:
            return result

        # Calculate value for each outcome
        for class_idx, odds_col in odds_columns.items():
            if odds_col not in df.columns:
                continue

            labels = ["home_win", "draw", "away_win"]
            label = labels[class_idx] if class_idx < len(labels) else f"class_{class_idx}"
            prob_col = f"prob_{label}"

            if prob_col in result.columns:
                implied_prob = 1.0 / df[odds_col].replace(0, np.nan)
                result[f"value_{label}"] = result[prob_col] - implied_prob
                result[f"is_value_{label}"] = result[f"value_{label}"] > min_value

        return result
