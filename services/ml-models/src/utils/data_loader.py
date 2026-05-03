"""Data loading utilities for ML model training."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare data for model training and evaluation."""

    def __init__(self, db_connection_string: Optional[str] = None):
        self.db_connection_string = db_connection_string

    def load_from_csv(
        self,
        filepath: str,
        date_column: str = "match_date",
        parse_dates: bool = True,
    ) -> pd.DataFrame:
        """Load match data from CSV file."""
        df = pd.read_csv(filepath)
        if parse_dates and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df

    def load_from_db(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Load data from PostgreSQL database."""
        if self.db_connection_string is None:
            raise ValueError("db_connection_string not configured")

        import psycopg2

        with psycopg2.connect(self.db_connection_string) as conn:
            df = pd.read_sql(query, conn, params=params)

        logger.info(f"Loaded {len(df)} rows from database")
        return df

    def train_test_split_temporal(
        self,
        df: pd.DataFrame,
        date_column: str = "match_date",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data chronologically into train/val/test sets.

        Preserves temporal ordering to prevent data leakage.
        """
        df_sorted = df.sort_values(date_column).reset_index(drop=True)
        n = len(df_sorted)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]

        logger.info(
            f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        return train_df, val_df, test_df

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        fill_strategy: str = "median",
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract and clean features and target.

        Args:
            df: Full DataFrame.
            feature_columns: Columns to use as features.
            target_column: Column to use as target.
            fill_strategy: How to fill NaN values ('median', 'mean', 'zero').
        """
        X = df[feature_columns].copy()

        if fill_strategy == "median":
            X = X.fillna(X.median())
        elif fill_strategy == "mean":
            X = X.fillna(X.mean())
        elif fill_strategy == "zero":
            X = X.fillna(0)

        y = df[target_column].values

        logger.info(
            f"Prepared {X.shape[0]} samples with {X.shape[1]} features, "
            f"NaN remaining: {X.isna().sum().sum()}"
        )
        return X, y

    def encode_target(
        self,
        y: np.ndarray,
        mapping: Optional[Dict[Any, int]] = None,
    ) -> Tuple[np.ndarray, Dict[Any, int]]:
        """Encode categorical target to integers.

        If mapping is provided, use it. Otherwise, create a new mapping.
        """
        if mapping is None:
            unique_vals = sorted(set(y))
            mapping = {val: i for i, val in enumerate(unique_vals)}

        encoded = np.array([mapping.get(v, -1) for v in y])

        if (encoded == -1).any():
            n_unknown = (encoded == -1).sum()
            logger.warning(f"{n_unknown} unknown target values encoded as -1")

        return encoded, mapping

    def get_feature_columns(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None,
    ) -> List[str]:
        """Auto-detect numeric feature columns, excluding specified columns."""
        exclude = set(exclude_columns or [])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]
