"""Real-time prediction with caching and latency optimization."""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from redis import Redis

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class RealTimePredictor:
    """Low-latency predictor for single match predictions.

    Features:
    - Redis caching for repeated predictions
    - ONNX runtime fallback for speed
    - Latency tracking
    """

    def __init__(
        self,
        model: BaseModel,
        redis_client: Optional[Redis] = None,
        cache_ttl: int = 300,
        onnx_session: Optional[Any] = None,
    ):
        self.model = model
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self.onnx_session = onnx_session
        self.latency_history: list = []

    def predict(
        self,
        features: Dict[str, float],
        match_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Predict single match with caching and timing.

        Returns prediction dict with probabilities, class, confidence, and latency.
        """
        start = time.perf_counter()

        # Check cache
        if use_cache and self.redis and match_id:
            cached = self._get_cached(match_id)
            if cached:
                elapsed = (time.perf_counter() - start) * 1000
                cached["latency_ms"] = elapsed
                cached["cache_hit"] = True
                return cached

        # Predict
        X = pd.DataFrame([features])
        proba = self._predict_fast(X)
        predicted_class = int(np.argmax(proba[0]))

        result = {
            "predicted_class": predicted_class,
            "probabilities": {
                "home_win": float(proba[0][0]),
                "draw": float(proba[0][1]),
                "away_win": float(proba[0][2]),
            },
            "confidence": float(np.max(proba[0])),
            "cache_hit": False,
        }

        # Cache result
        if use_cache and self.redis and match_id:
            self._set_cached(match_id, result)

        elapsed = (time.perf_counter() - start) * 1000
        result["latency_ms"] = elapsed
        self.latency_history.append(elapsed)

        return result

    def _predict_fast(self, X: pd.DataFrame) -> np.ndarray:
        """Use ONNX session if available, otherwise fall back to model."""
        if self.onnx_session:
            try:
                input_name = self.onnx_session.get_inputs()[0].name
                feature_values = X[self.model.feature_names].fillna(0).values.astype(np.float32)
                result = self.onnx_session.run(None, {input_name: feature_values})
                return result[0]
            except Exception as e:
                logger.warning(f"ONNX inference failed, falling back: {e}")

        return self.model.predict_proba(X)

    def _get_cached(self, match_id: str) -> Optional[Dict]:
        try:
            key = f"prediction:{match_id}"
            data = self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
        return None

    def _set_cached(self, match_id: str, result: Dict) -> None:
        try:
            key = f"prediction:{match_id}"
            cacheable = {k: v for k, v in result.items() if k not in ("latency_ms", "cache_hit")}
            self.redis.setex(key, self.cache_ttl, json.dumps(cacheable))
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics from prediction history."""
        if not self.latency_history:
            return {}
        arr = np.array(self.latency_history)
        return {
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "n_predictions": len(arr),
        }
