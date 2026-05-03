"""Time-series cross-validation to prevent data leakage."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class TimeSeriesCV:
    """Time-series cross-validation for match prediction models.

    Ensures training data always precedes validation data chronologically.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: Optional[int] = None,
        gap: int = 0,
        date_column: str = "match_date",
    ):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.gap = gap
        self.date_column = date_column

    def split(
        self, df: pd.DataFrame
    ) -> List[Dict[str, pd.DataFrame]]:
        """Split data chronologically into train/val folds."""
        df_sorted = df.sort_values(self.date_column).reset_index(drop=True)
        n = len(df_sorted)

        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            gap=self.gap,
        )

        folds = []
        for train_idx, val_idx in tscv.split(df_sorted):
            if self.min_train_size and len(train_idx) < self.min_train_size:
                continue
            folds.append({
                "train": df_sorted.iloc[train_idx],
                "val": df_sorted.iloc[val_idx],
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            })

        logger.info(f"Created {len(folds)} time-series CV folds from {n} samples")
        return folds

    def cross_validate(
        self,
        model_class: type,
        config: Any,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        target: str = "match_outcome",
    ) -> Dict[str, Any]:
        """Run full cross-validation and aggregate metrics."""
        folds = self.split(df)

        fold_metrics = []
        for i, fold in enumerate(folds):
            logger.info(
                f"Fold {i + 1}/{len(folds)}: "
                f"train={fold['train_size']}, val={fold['val_size']}"
            )

            model = model_class(config)
            model.train(
                fold["train"],
                val_df=fold["val"],
                features=features,
                target=target,
            )

            if features:
                X_val = fold["val"][features].fillna(fold["val"][features].median())
            else:
                X_val = fold["val"]

            y_val_col = fold["val"][target]
            if hasattr(model, "label_encoder") and hasattr(model.label_encoder, "classes_"):
                y_val = model.label_encoder.transform(y_val_col)
            else:
                y_val = y_val_col.values

            metrics = model.evaluate(X_val, y_val)
            fold_metrics.append(metrics)

            logger.info(
                f"Fold {i + 1} metrics: "
                + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            )

        # Aggregate
        all_metric_names = set()
        for fm in fold_metrics:
            all_metric_names.update(fm.keys())

        aggregated = {}
        for metric_name in all_metric_names:
            values = [fm[metric_name] for fm in fold_metrics if metric_name in fm]
            if values:
                aggregated[f"mean_{metric_name}"] = float(np.mean(values))
                aggregated[f"std_{metric_name}"] = float(np.std(values))

        aggregated["n_folds"] = len(folds)
        aggregated["fold_metrics"] = fold_metrics

        logger.info(
            f"CV complete: "
            + ", ".join(
                f"{k}={v:.4f}"
                for k, v in aggregated.items()
                if isinstance(v, float) and k.startswith("mean_")
            )
        )

        return aggregated
