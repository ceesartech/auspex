"""Track model performance over time."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track and monitor model performance over time.

    Detects performance drift and alerts when models degrade.
    """

    def __init__(self, storage_path: str = "performance_logs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}

    def log_metrics(
        self,
        model_name: str,
        metrics: Dict[str, float],
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a set of metrics for a model."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "dataset_info": dataset_info or {},
        }

        if model_name not in self.metrics_history:
            self.metrics_history[model_name] = []
        self.metrics_history[model_name].append(entry)

        # Persist
        log_path = self.storage_path / f"{model_name}_metrics.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_metric_trend(self, model_name: str, metric_name: str, last_n: int = 20) -> List[float]:
        """Get recent values of a metric."""
        entries = self.metrics_history.get(model_name, [])
        values = []
        for e in entries[-last_n:]:
            v = e["metrics"].get(metric_name)
            if v is not None:
                values.append(v)
        return values

    def detect_drift(
        self,
        model_name: str,
        metric_name: str = "accuracy",
        window: int = 10,
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Detect performance drift by comparing recent vs historical performance.

        Returns drift information if recent performance drops significantly.
        """
        values = self.get_metric_trend(model_name, metric_name, last_n=window * 2)

        if len(values) < window * 2:
            return {"has_drift": False, "reason": "Not enough data"}

        historical = values[:window]
        recent = values[window:]

        hist_mean = float(np.mean(historical))
        recent_mean = float(np.mean(recent))
        diff = recent_mean - hist_mean

        # For metrics where higher is better (accuracy, ROI)
        has_drift = diff < -threshold

        return {
            "has_drift": has_drift,
            "historical_mean": hist_mean,
            "recent_mean": recent_mean,
            "difference": diff,
            "threshold": threshold,
        }

    def get_summary(self, model_name: str) -> Dict[str, Any]:
        """Get summary statistics for a model."""
        entries = self.metrics_history.get(model_name, [])
        if not entries:
            return {}

        latest = entries[-1]["metrics"]
        all_metrics = {}
        for e in entries:
            for k, v in e["metrics"].items():
                if k not in all_metrics:
                    all_metrics[k] = []
                if v is not None:
                    all_metrics[k].append(v)

        summary = {
            "n_evaluations": len(entries),
            "latest": latest,
        }
        for k, values in all_metrics.items():
            if values:
                summary[f"{k}_mean"] = float(np.mean(values))
                summary[f"{k}_std"] = float(np.std(values))
                summary[f"{k}_best"] = float(max(values))

        return summary
