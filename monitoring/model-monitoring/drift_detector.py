"""Detect data drift in input feature distributions."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import ks_2samp
from prometheus_client import Gauge

logger = logging.getLogger(__name__)

drift_score_gauge = Gauge('drift_score', 'Overall feature drift score (0-1)')
drifted_features_gauge = Gauge('drift_features_count', 'Number of features with statistically significant drift')


class DriftDetector:
    """Detect distribution shift in features using the Kolmogorov-Smirnov test."""

    def __init__(self, db_connection, redis_client):
        self.db = db_connection
        self.redis = redis_client

    def detect_drift(
        self,
        baseline_period_days: int = 90,
        current_period_days: int = 7,
        significance_level: float = 0.05,
    ) -> Optional[Dict[str, Any]]:
        """Run drift detection and return a summary dict, or None if data is insufficient."""
        baseline = self._get_features(
            lookback_days=baseline_period_days + current_period_days,
            offset_days=current_period_days,
        )
        current = self._get_features(lookback_days=current_period_days, offset_days=0)

        if not baseline or not current:
            logger.warning("Insufficient feature data for drift detection")
            return None

        drift_results: Dict[str, Any] = {}
        for feature_name, baseline_values in baseline.items():
            if feature_name not in current:
                continue
            current_values = current[feature_name]

            statistic, p_value = ks_2samp(baseline_values, current_values)
            drift_results[feature_name] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'has_drift': p_value < significance_level,
                'baseline_mean': float(np.mean(baseline_values)),
                'current_mean': float(np.mean(current_values)),
                'baseline_std': float(np.std(baseline_values)),
                'current_std': float(np.std(current_values)),
            }

        drift_score = self._calculate_drift_score(drift_results)
        drifted = [name for name, r in drift_results.items() if r['has_drift']]

        self._store_drift_results(drift_results, drift_score)

        # Update Prometheus
        drift_score_gauge.set(drift_score)
        drifted_features_gauge.set(len(drifted))

        logger.info(
            "Drift detection complete: score=%.4f drifted_features=%d/%d",
            drift_score, len(drifted), len(drift_results),
        )
        return {
            'drift_score': drift_score,
            'features_with_drift': drifted,
            'total_features': len(drift_results),
            'drift_results': drift_results,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_features(self, lookback_days: int, offset_days: int = 0) -> Dict[str, np.ndarray]:
        """Fetch feature vectors from the features_cache table for a time window."""
        query = """
            SELECT features_json
            FROM features_cache
            WHERE computed_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
              AND computed_at <  CURRENT_TIMESTAMP - INTERVAL '%s days'
              AND is_valid = true
        """
        cursor = self.db.cursor()
        cursor.execute(query, (lookback_days, offset_days))
        rows = cursor.fetchall()

        if not rows:
            return {}

        aggregated: Dict[str, list] = {}
        for (features_json,) in rows:
            feature_data = json.loads(features_json)
            for name, value in feature_data.items():
                if isinstance(value, (int, float)):
                    aggregated.setdefault(name, []).append(float(value))

        # Require at least 30 samples for a meaningful KS test
        return {
            name: np.array(values)
            for name, values in aggregated.items()
            if len(values) >= 30
        }

    def _calculate_drift_score(self, drift_results: Dict[str, Any]) -> float:
        """Aggregate KS p-values into a single drift score in [0, 1]."""
        if not drift_results:
            return 0.0
        p_values = np.array([r['p_value'] for r in drift_results.values()])
        return float(np.mean(1 - p_values))

    def _store_drift_results(self, drift_results: Dict[str, Any], drift_score: float) -> None:
        """Cache latest drift results in Redis for quick API access."""
        payload = json.dumps({
            'timestamp': datetime.utcnow().isoformat(),
            'drift_score': drift_score,
            'features_with_drift': [
                name for name, r in drift_results.items() if r['has_drift']
            ],
        })
        self.redis.setex('drift:latest', 86400, payload)
