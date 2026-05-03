"""Tests for utility modules: DataLoader and FeatureSelector."""

import numpy as np
import pandas as pd
import pytest

from utils.data_loader import DataLoader
from utils.feature_selector import FeatureSelector


class TestDataLoader:
    """Tests for DataLoader."""

    def test_train_test_split_temporal(self):
        df = pd.DataFrame({
            "match_date": pd.date_range("2023-01-01", periods=100, freq="D"),
            "feature": np.random.randn(100),
            "target": np.random.randint(0, 3, 100),
        })

        loader = DataLoader()
        train, val, test = loader.train_test_split_temporal(
            df, train_ratio=0.6, val_ratio=0.2
        )

        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

        # Check temporal ordering
        assert train["match_date"].max() <= val["match_date"].min()
        assert val["match_date"].max() <= test["match_date"].min()

    def test_prepare_features_median(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, np.nan, 4.0],
            "f2": [5.0, np.nan, 7.0, 8.0],
            "target": [0, 1, 2, 0],
        })

        loader = DataLoader()
        X, y = loader.prepare_features(df, ["f1", "f2"], "target", fill_strategy="median")

        assert X.shape == (4, 2)
        assert X.isna().sum().sum() == 0
        assert len(y) == 4

    def test_prepare_features_zero(self):
        df = pd.DataFrame({
            "f1": [1.0, np.nan],
            "target": [0, 1],
        })

        loader = DataLoader()
        X, y = loader.prepare_features(df, ["f1"], "target", fill_strategy="zero")
        assert X.iloc[1, 0] == 0.0

    def test_encode_target(self):
        loader = DataLoader()
        y = np.array(["home_win", "draw", "away_win", "home_win"])
        encoded, mapping = loader.encode_target(y)

        assert len(encoded) == 4
        assert set(encoded) == {0, 1, 2}
        assert len(mapping) == 3

    def test_encode_target_with_mapping(self):
        loader = DataLoader()
        mapping = {"home_win": 0, "draw": 1, "away_win": 2}
        y = np.array(["home_win", "draw"])
        encoded, _ = loader.encode_target(y, mapping=mapping)

        assert encoded[0] == 0
        assert encoded[1] == 1

    def test_get_feature_columns(self):
        df = pd.DataFrame({
            "f1": [1.0], "f2": [2.0], "name": ["team"],
            "target": [0], "match_id": ["abc"],
        })

        loader = DataLoader()
        cols = loader.get_feature_columns(df, exclude_columns=["target", "match_id"])
        assert "f1" in cols
        assert "f2" in cols
        assert "target" not in cols
        assert "match_id" not in cols
        assert "name" not in cols  # Not numeric


class TestFeatureSelector:
    """Tests for FeatureSelector."""

    def test_select_by_variance(self):
        df = pd.DataFrame({
            "constant": [1.0] * 100,
            "low_var": np.random.uniform(0, 0.01, 100),
            "normal": np.random.randn(100),
        })

        selector = FeatureSelector()
        selected = selector.select_by_variance(df, threshold=0.01)

        assert "normal" in selected
        assert "constant" not in selected

    def test_select_by_correlation(self):
        np.random.seed(42)
        f1 = np.random.randn(100)
        df = pd.DataFrame({
            "f1": f1,
            "f2": f1 + np.random.randn(100) * 0.01,  # Highly correlated with f1
            "f3": np.random.randn(100),  # Independent
        })

        selector = FeatureSelector()
        selected = selector.select_by_correlation(df, threshold=0.95)

        assert "f3" in selected
        # One of f1/f2 should be removed (they're highly correlated)
        assert len(selected) == 2
        assert ("f1" in selected) or ("f2" in selected)
        assert not ("f1" in selected and "f2" in selected)

    def test_select_by_importance(self):
        importance = {"f1": 0.5, "f2": 0.3, "f3": 0.1, "f4": 0.01, "f5": 0.0}

        selector = FeatureSelector()
        selected = selector.select_by_importance(importance, top_n=3)

        assert len(selected) == 3
        assert selected[0] == "f1"
        assert selected[1] == "f2"
        assert selected[2] == "f3"

    def test_select_by_importance_threshold(self):
        importance = {"f1": 0.5, "f2": 0.3, "f3": 0.01}

        selector = FeatureSelector()
        selected = selector.select_by_importance(importance, min_importance=0.1)

        assert len(selected) == 2
        assert "f3" not in selected

    def test_apply_selection(self):
        df = pd.DataFrame({"f1": [1], "f2": [2], "f3": [3]})

        selector = FeatureSelector()
        selector.selected_features = ["f1", "f3"]

        result = selector.apply_selection(df)
        assert list(result.columns) == ["f1", "f3"]

    def test_apply_selection_empty(self):
        df = pd.DataFrame({"f1": [1], "f2": [2]})
        selector = FeatureSelector()

        result = selector.apply_selection(df)
        assert len(result.columns) == 2  # Returns all when empty

    def test_get_report(self):
        selector = FeatureSelector()
        selector.selected_features = ["f1", "f2"]
        selector.feature_scores = {"f1": 0.5, "f2": 0.3}

        report = selector.get_report()
        assert report["n_selected"] == 2
        assert report["features"] == ["f1", "f2"]
