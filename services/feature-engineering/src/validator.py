"""Feature validation — checks missing rates, value ranges, and outliers."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .core.config import FeatureConfig
from .core.registry import FeatureRegistry

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of feature validation."""

    is_valid: bool = True
    missing_count: int = 0
    missing_rate: float = 0.0
    out_of_range: List[str] = field(default_factory=list)
    outliers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [f"valid={self.is_valid}"]
        parts.append(f"missing={self.missing_count} ({self.missing_rate:.1%})")
        if self.out_of_range:
            parts.append(f"out_of_range={len(self.out_of_range)}")
        if self.outliers:
            parts.append(f"outliers={len(self.outliers)}")
        return ", ".join(parts)


class FeatureValidator:
    """Validate computed feature vectors."""

    def __init__(self, config: FeatureConfig, registry: FeatureRegistry):
        self.config = config
        self.registry = registry

    def validate(
        self,
        features: Dict[str, Optional[float]],
        historical_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> ValidationResult:
        """Validate a feature vector.

        Args:
            features: The computed feature dict.
            historical_stats: Optional dict of {feature_name: {mean, std}}
                for outlier detection.

        Returns:
            ValidationResult with details.
        """
        result = ValidationResult()

        total = len(features)
        if total == 0:
            result.warnings.append("Empty feature vector")
            return result

        # Missing rate
        null_count = sum(1 for v in features.values() if v is None)
        result.missing_count = null_count
        result.missing_rate = null_count / total

        if result.missing_rate > self.config.max_missing_rate:
            result.warnings.append(
                f"High missing rate: {result.missing_rate:.1%} " f"(threshold: {self.config.max_missing_rate:.1%})"
            )

        # Range validation
        for name, value in features.items():
            if value is None:
                continue

            meta = self.registry.get(name)
            if meta is None:
                continue

            if meta.min_value is not None and value < meta.min_value:
                result.out_of_range.append(f"{name}={value} < min={meta.min_value}")
            if meta.max_value is not None and value > meta.max_value:
                result.out_of_range.append(f"{name}={value} > max={meta.max_value}")

        # NaN/Inf check
        for name, value in features.items():
            if value is None:
                continue
            if not np.isfinite(value):
                result.out_of_range.append(f"{name}={value} (non-finite)")

        # Outlier detection (if historical stats available)
        if historical_stats:
            threshold = self.config.outlier_std_threshold
            for name, value in features.items():
                if value is None:
                    continue
                stats = historical_stats.get(name)
                if stats is None:
                    continue
                mean = stats.get("mean", 0)
                std = stats.get("std", 0)
                if std > 0:
                    z = abs(value - mean) / std
                    if z > threshold:
                        result.outliers.append(f"{name}={value} (z={z:.1f}, mean={mean:.2f}, std={std:.2f})")

        # Determine validity
        if result.out_of_range:
            result.is_valid = False
        if result.missing_rate > 0.8:
            result.is_valid = False

        if result.warnings or result.out_of_range or result.outliers:
            logger.warning(f"Feature validation: {result.summary}")

        return result
