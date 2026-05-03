"""Tests for feature validator."""

import pytest
from src.core.registry import FeatureMetadata, FeatureRegistry
from src.validator import FeatureValidator


class TestFeatureValidator:

    @pytest.fixture
    def registry(self):
        reg = FeatureRegistry()
        reg.register_many(
            [
                FeatureMetadata(name="f1", category="test", description="test feature 1", min_value=0, max_value=1),
                FeatureMetadata(name="f2", category="test", description="test feature 2", min_value=-10, max_value=10),
                FeatureMetadata(name="f3", category="test", description="test feature 3"),
            ]
        )
        return reg

    @pytest.fixture
    def validator(self, mock_config, registry):
        return FeatureValidator(mock_config, registry)

    def test_valid_features(self, validator):
        features = {"f1": 0.5, "f2": 5.0, "f3": 100.0}
        result = validator.validate(features)
        assert result.is_valid
        assert result.missing_count == 0

    def test_missing_features(self, validator):
        features = {"f1": None, "f2": None, "f3": 1.0}
        result = validator.validate(features)
        assert result.missing_count == 2
        assert result.missing_rate == pytest.approx(2 / 3)

    def test_high_missing_rate_warning(self, validator):
        features = {"f1": None, "f2": None, "f3": None}
        result = validator.validate(features)
        assert result.missing_rate == 1.0
        assert not result.is_valid  # >80% missing -> invalid
        assert len(result.warnings) > 0

    def test_out_of_range(self, validator):
        features = {"f1": 2.0, "f2": 5.0, "f3": 1.0}  # f1 > max 1
        result = validator.validate(features)
        assert not result.is_valid
        assert len(result.out_of_range) == 1

    def test_below_minimum(self, validator):
        features = {"f1": -0.5, "f2": 5.0, "f3": 1.0}  # f1 < min 0
        result = validator.validate(features)
        assert not result.is_valid
        assert len(result.out_of_range) == 1

    def test_nan_detection(self, validator):
        features = {"f1": float("nan"), "f2": 5.0, "f3": 1.0}
        result = validator.validate(features)
        assert not result.is_valid
        assert len(result.out_of_range) >= 1

    def test_inf_detection(self, validator):
        features = {"f1": float("inf"), "f2": 5.0, "f3": 1.0}
        result = validator.validate(features)
        assert not result.is_valid

    def test_outlier_detection(self, validator):
        historical = {
            "f2": {"mean": 5.0, "std": 1.0},
        }
        features = {"f1": 0.5, "f2": 20.0, "f3": 1.0}  # f2 is 15 std away
        result = validator.validate(features, historical_stats=historical)
        assert len(result.outliers) >= 1

    def test_empty_features(self, validator):
        result = validator.validate({})
        assert len(result.warnings) == 1
        assert "Empty" in result.warnings[0]

    def test_summary_string(self, validator):
        features = {"f1": 0.5, "f2": None, "f3": 1.0}
        result = validator.validate(features)
        assert "missing=1" in result.summary
