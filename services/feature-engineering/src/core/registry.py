"""Feature registry for tracking and validating feature metadata."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for a single feature."""

    name: str
    category: str
    description: str
    dtype: str = "float"  # float, int, bool
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    nullable: bool = True
    tags: List[str] = field(default_factory=list)


class FeatureRegistry:
    """Central registry of all known features with metadata."""

    def __init__(self):
        self._features: Dict[str, FeatureMetadata] = {}
        self._categories: Dict[str, Set[str]] = {}

    def register(self, metadata: FeatureMetadata) -> None:
        """Register a feature with its metadata."""
        if metadata.name in self._features:
            logger.warning(f"Feature '{metadata.name}' already registered, overwriting")

        self._features[metadata.name] = metadata

        if metadata.category not in self._categories:
            self._categories[metadata.category] = set()
        self._categories[metadata.category].add(metadata.name)

    def register_many(self, features: List[FeatureMetadata]) -> None:
        """Register multiple features at once."""
        for feat in features:
            self.register(feat)

    def get(self, name: str) -> Optional[FeatureMetadata]:
        """Get metadata for a feature by name."""
        return self._features.get(name)

    def get_by_category(self, category: str) -> List[FeatureMetadata]:
        """Get all features in a category."""
        names = self._categories.get(category, set())
        return [self._features[n] for n in sorted(names)]

    def get_all_names(self) -> List[str]:
        """Get all registered feature names."""
        return sorted(self._features.keys())

    def get_categories(self) -> List[str]:
        """Get all registered categories."""
        return sorted(self._categories.keys())

    @property
    def count(self) -> int:
        """Total number of registered features."""
        return len(self._features)

    def category_counts(self) -> Dict[str, int]:
        """Get feature count per category."""
        return {cat: len(names) for cat, names in sorted(self._categories.items())}

    def validate_output(self, features: Dict[str, Optional[float]]) -> List[str]:
        """Validate computed features against registry.

        Returns list of validation warnings.
        """
        warnings = []
        registered = set(self._features.keys())
        computed = set(features.keys())

        missing = registered - computed
        extra = computed - registered

        if missing:
            warnings.append(
                f"Missing {len(missing)} registered features: "
                f"{sorted(list(missing))[:5]}..."
            )

        if extra:
            warnings.append(
                f"Found {len(extra)} unregistered features: "
                f"{sorted(list(extra))[:5]}..."
            )

        for name, value in features.items():
            meta = self._features.get(name)
            if meta is None or value is None:
                continue

            if meta.min_value is not None and value < meta.min_value:
                warnings.append(
                    f"Feature '{name}' value {value} below min {meta.min_value}"
                )
            if meta.max_value is not None and value > meta.max_value:
                warnings.append(
                    f"Feature '{name}' value {value} above max {meta.max_value}"
                )

        return warnings


# Global registry instance
global_registry = FeatureRegistry()
