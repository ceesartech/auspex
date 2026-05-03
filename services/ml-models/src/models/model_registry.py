"""Model version management and registry."""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class ModelVersion:
    """Represents a versioned model artifact."""

    def __init__(
        self,
        name: str,
        version: str,
        model_type: str,
        path: str,
        metrics: Dict[str, float],
        created_at: str,
        is_active: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.version = version
        self.model_type = model_type
        self.path = path
        self.metrics = metrics
        self.created_at = created_at
        self.is_active = is_active
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "path": self.path,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        return cls(**data)


class ModelRegistry:
    """Local file-based model registry for versioning and management."""

    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.registry_dir / "registry_index.json"
        self._index: Dict[str, List[Dict]] = self._load_index()

    def _load_index(self) -> Dict[str, List[Dict]]:
        if self._index_path.exists():
            with open(self._index_path, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def register_model(
        self,
        model: BaseModel,
        name: str,
        version: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """Register and save a trained model."""
        model_dir = self.registry_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.bin"
        model.save(str(model_path))

        # Create version entry
        mv = ModelVersion(
            name=name,
            version=version,
            model_type=model.config.model_type.value,
            path=str(model_path),
            metrics=metrics,
            created_at=datetime.utcnow().isoformat(),
            is_active=False,
            metadata=metadata or {},
        )

        # Add to index
        if name not in self._index:
            self._index[name] = []
        self._index[name].append(mv.to_dict())
        self._save_index()

        logger.info(f"Registered model '{name}' version {version}")
        return mv

    def get_model_versions(self, name: str) -> List[ModelVersion]:
        """Get all versions of a model."""
        entries = self._index.get(name, [])
        return [ModelVersion.from_dict(e) for e in entries]

    def get_latest_version(self, name: str) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        versions = self.get_model_versions(name)
        if not versions:
            return None
        return versions[-1]

    def get_active_version(self, name: str) -> Optional[ModelVersion]:
        """Get the currently active version of a model."""
        versions = self.get_model_versions(name)
        for v in reversed(versions):
            if v.is_active:
                return v
        return None

    def promote_model(self, name: str, version: str) -> None:
        """Set a model version as the active/production version."""
        if name not in self._index:
            raise ValueError(f"Model '{name}' not found")

        for entry in self._index[name]:
            entry["is_active"] = entry["version"] == version

        self._save_index()
        logger.info(f"Promoted model '{name}' version {version} to active")

    def compare_models(self, name: str) -> pd.DataFrame:
        """Compare all versions of a model by metrics."""
        import pandas as pd

        versions = self.get_model_versions(name)
        if not versions:
            return pd.DataFrame()

        rows = []
        for v in versions:
            row = {"version": v.version, "created_at": v.created_at, "is_active": v.is_active}
            row.update(v.metrics)
            rows.append(row)

        return pd.DataFrame(rows)

    def delete_version(self, name: str, version: str) -> None:
        """Delete a specific model version."""
        if name not in self._index:
            return

        self._index[name] = [
            e for e in self._index[name] if e["version"] != version
        ]

        # Remove files
        model_dir = self.registry_dir / name / version
        if model_dir.exists():
            shutil.rmtree(model_dir)

        self._save_index()
        logger.info(f"Deleted model '{name}' version {version}")

    def list_models(self) -> Dict[str, int]:
        """List all registered models and their version counts."""
        return {name: len(versions) for name, versions in self._index.items()}
