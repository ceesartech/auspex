"""Prediction service"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings
from models.responses import MatchInfo, PredictionResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Process-wide model registry. Populated once during app startup
# (see services/api/src/main.py lifespan) and shared across requests.
# Loading joblib pickles on every request adds ~10-50ms; this avoids that.
_MODELS: Dict[str, Any] = {}
_MODEL_VERSION: str = "ensemble_v1.0"


def get_loaded_models() -> Dict[str, Any]:
    """Return the process-wide loaded model registry."""
    return _MODELS


def get_model_version() -> str:
    """Version string for the currently loaded ensemble. Used in cache keys."""
    return _MODEL_VERSION


def _predict_one(model, features: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter from the (features dict) call shape used by the route + DAG
    to EnsemblePredictor.predict_proba(DataFrame).

    The previous code called ensemble.predict_single() which doesn't exist
    on EnsemblePredictor or its base class — it was a non-existent API.
    """
    import pandas as pd

    X = pd.DataFrame([features])
    proba = model.predict_proba(X)[0]  # shape (3,) — [home, draw, away]
    labels = ["home", "draw", "away"]
    idx = int(proba.argmax())
    return {
        "predicted_label": labels[idx],
        "confidence": float(proba[idx]),
        "probabilities": {labels[i]: float(proba[i]) for i in range(3)},
    }


def _latest_model_bin(root: Path) -> Optional[Path]:
    """Return the most recent `*/model.bin` under `root` (by mtime), or None."""
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*/model.bin"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _build_ensemble_from_registry(model_path: Path):
    """Reconstitute the EnsemblePredictor + its base models from the
    ModelRegistry layout under `model_path`.

    The training pipeline writes:
      {model_path}/ensemble/{version}/model.bin   (JSON: weights + model_names)
      {model_path}/xgboost/{version}/model.bin    (XGBoost UBJSON)
      {model_path}/lightgbm/{version}/model.bin   (LightGBM text)
      {model_path}/neural_network/{version}/model.bin (PyTorch pickle)
      {model_path}/poisson/{version}/model.bin    (JSON)
      {model_path}/dixon_coles/{version}/model.bin (JSON)

    EnsemblePredictor.save() only persists weights + base-model names —
    not the base models themselves — so reloading requires us to walk
    the registry, instantiate each base model class, load its
    artifact, and re-attach via add_model() before setting weights.
    """
    import json
    import sys

    # The ml-models predictor classes live under `predictors.*` (renamed
    # from the old `models.*` to avoid collision with api/src/models/,
    # which is the pydantic schema package). PYTHONPATH from
    # Dockerfile.api includes /app/services/ml-models/src so these
    # resolve directly.
    ml_src = "/app/services/ml-models/src"
    if ml_src not in sys.path:
        sys.path.insert(0, ml_src)

    from predictors.ensemble import EnsemblePredictor
    from predictors.lightgbm_model import LightGBMMatchPredictor
    from predictors.model_config import (
        DIXON_COLES_CONFIG,
        ENSEMBLE_CONFIG,
        LIGHTGBM_MATCH_OUTCOME,
        NEURAL_NETWORK_CONFIG,
        POISSON_CONFIG,
        XGBOOST_MATCH_OUTCOME,
    )
    from predictors.neural_network import NeuralNetworkMatchPredictor
    from predictors.poisson_models import DixonColesPredictor, PoissonMatchPredictor
    from predictors.xgboost_model import XGBoostMatchPredictor

    ensemble_meta = _latest_model_bin(model_path / "ensemble")
    if ensemble_meta is None:
        return None, None

    with open(ensemble_meta) as f:
        meta = json.load(f)

    # name -> (class, config) for every base-model type the ensemble
    # might reference. Anything else in meta["model_names"] is skipped
    # with a warning.
    klass_for: Dict[str, tuple] = {
        "xgboost": (XGBoostMatchPredictor, XGBOOST_MATCH_OUTCOME),
        "lightgbm": (LightGBMMatchPredictor, LIGHTGBM_MATCH_OUTCOME),
        "neural_network": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_CONFIG),
        "poisson": (PoissonMatchPredictor, POISSON_CONFIG),
        "dixon_coles": (DixonColesPredictor, DIXON_COLES_CONFIG),
    }

    ensemble = EnsemblePredictor(ENSEMBLE_CONFIG)
    loaded_names = []
    for name in meta.get("model_names", []):
        if name not in klass_for:
            logger.warning("Ensemble references unknown base model %r; skipping", name)
            continue
        artifact = _latest_model_bin(model_path / name)
        if artifact is None:
            logger.warning("Ensemble references %r but no artifact under %s/%s/", name, model_path, name)
            continue
        cls, cfg = klass_for[name]
        try:
            model = cls(cfg)
            model.load(str(artifact))
            ensemble.add_model(name, model)
            loaded_names.append(name)
        except Exception as e:
            logger.error("Failed to load base model %r from %s: %s", name, artifact, e)

    if not loaded_names:
        logger.error("Could not reconstitute any base models for the ensemble")
        return None, None

    # ensemble.load() restores the blend weights from JSON metadata.
    ensemble.load(str(ensemble_meta))
    return ensemble, ensemble_meta


def load_models_into_process(model_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load ML models into the process-wide registry. Idempotent.

    Called once from FastAPI's lifespan startup. Returns the loaded models.
    Subsequent calls are no-ops if models are already loaded.
    """
    global _MODEL_VERSION

    if _MODELS:
        return _MODELS

    model_path = model_dir or (Path(settings.MODEL_PATH) / "production")
    try:
        ensemble, ensemble_meta_path = _build_ensemble_from_registry(model_path)
        if ensemble is not None:
            _MODELS["ensemble"] = ensemble
            mtime = int(ensemble_meta_path.stat().st_mtime)
            version_label = ensemble_meta_path.parent.name
            _MODEL_VERSION = f"ensemble_{version_label}+{mtime}"
            logger.info(
                "Loaded ensemble model from %s (version=%s, base_models=%s)",
                ensemble_meta_path,
                _MODEL_VERSION,
                list(ensemble.models.keys()),
            )
        else:
            logger.warning(
                "Ensemble model not found under %s/ensemble/*/model.bin; serving from DB fallback",
                model_path,
            )
    except Exception as e:
        logger.error("Failed to load models: %s", e, exc_info=True)
    return _MODELS


class PredictionService:
    """Service for generating predictions"""

    def __init__(self, db: Optional[Session] = None, models: Optional[Dict[str, Any]] = None):
        self.db = db
        # Default to the process-wide registry. Tests can pass their own.
        self.models: Dict[str, Any] = models if models is not None else _MODELS

    def _require_db(self) -> Session:
        if self.db is None:
            raise RuntimeError("PredictionService requires a database session for this operation")
        return self.db

    def load_models(self) -> None:
        """Compatibility shim. Real loading happens in load_models_into_process()."""
        load_models_into_process()
        self.models = _MODELS

    def predict_match(
        self,
        match_id: str,
        include_explanation: bool = True,
        include_alternate_models: bool = False,
    ) -> PredictionResponse:
        """Generate prediction for a match"""

        match_data = self._get_match_data(match_id)
        if not match_data:
            raise ValueError(f"Match {match_id} not found")

        features = self._get_match_features(match_id)

        if "ensemble" in self.models:
            prediction = _predict_one(self.models["ensemble"], features)
        else:
            # Fallback: return stored prediction from DB
            prediction = self._get_stored_prediction(match_id)
            if not prediction:
                raise ValueError("No model available for predictions")

        response = PredictionResponse(
            match_info=MatchInfo(**match_data),
            predicted_outcome=prediction["predicted_label"],
            probabilities=prediction["probabilities"],
            confidence=prediction["confidence"],
            model_version=_MODEL_VERSION,
            timestamp=datetime.utcnow(),
            explanation=prediction.get("explanation") if include_explanation else None,
        )

        self._store_prediction(match_id, prediction)

        return response

    def get_upcoming_predictions(
        self,
        sport: Optional[str] = None,
        league: Optional[str] = None,
        limit: int = 20,
    ) -> List[PredictionResponse]:
        """Get predictions for upcoming matches"""

        query = text(
            """
            SELECT p.id, p.match_id, p.predicted_outcome, p.confidence,
                   p.probabilities, p.model_name, p.model_version,
                   p.features_used, p.feature_importance,
                   m.match_date, m.venue,
                   l.name as league_name,
                   ht.name as home_team, at.name as away_team
            FROM predictions p
            JOIN matches m ON p.match_id = m.id
            JOIN leagues l ON m.league_id = l.id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.status = 'scheduled' AND m.match_date > NOW()
            AND (:sport IS NULL OR l.sport = :sport)
            AND (:league IS NULL OR l.name = :league)
            ORDER BY m.match_date ASC
            LIMIT :limit
        """
        )

        db = self._require_db()
        results = db.execute(query, {"sport": sport, "league": league, "limit": limit}).fetchall()

        predictions = []
        for row in results:
            predictions.append(
                PredictionResponse(
                    match_info=MatchInfo(
                        match_id=str(row.match_id),
                        league_name=row.league_name,
                        home_team=row.home_team,
                        away_team=row.away_team,
                        match_date=row.match_date,
                        venue=row.venue,
                    ),
                    predicted_outcome=row.predicted_outcome,
                    probabilities=row.probabilities or {},
                    confidence=float(row.confidence),
                    model_version=row.model_version,
                    timestamp=datetime.utcnow(),
                )
            )

        return predictions

    def get_live_predictions(self) -> List[PredictionResponse]:
        """Get predictions for live matches"""

        query = text(
            """
            SELECT p.id, p.match_id, p.predicted_outcome, p.confidence,
                   p.probabilities, p.model_version,
                   m.match_date, m.venue,
                   l.name as league_name,
                   ht.name as home_team, at.name as away_team
            FROM predictions p
            JOIN matches m ON p.match_id = m.id
            JOIN leagues l ON m.league_id = l.id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.status = 'live'
            ORDER BY m.match_date ASC
        """
        )

        db = self._require_db()
        results = db.execute(query).fetchall()

        predictions = []
        for row in results:
            predictions.append(
                PredictionResponse(
                    match_info=MatchInfo(
                        match_id=str(row.match_id),
                        league_name=row.league_name,
                        home_team=row.home_team,
                        away_team=row.away_team,
                        match_date=row.match_date,
                        venue=row.venue,
                    ),
                    predicted_outcome=row.predicted_outcome,
                    probabilities=row.probabilities or {},
                    confidence=float(row.confidence),
                    model_version=row.model_version,
                    timestamp=datetime.utcnow(),
                )
            )

        return predictions

    def _get_match_data(self, match_id: str) -> Optional[Dict]:
        """Get match information from database"""

        query = text(
            """
            SELECT m.id, l.name as league_name,
                   ht.name as home_team, at.name as away_team,
                   m.match_date, m.venue
            FROM matches m
            JOIN leagues l ON m.league_id = l.id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.id = :match_id
        """
        )

        db = self._require_db()
        result = db.execute(query, {"match_id": match_id}).fetchone()

        if result:
            return {
                "match_id": str(result.id),
                "league_name": result.league_name,
                "home_team": result.home_team,
                "away_team": result.away_team,
                "match_date": result.match_date,
                "venue": result.venue,
            }
        return None

    def _get_match_features(self, match_id: str) -> Dict[str, Any]:
        """Get computed features for a match from cache"""

        query = text(
            """
            SELECT features FROM features_cache
            WHERE match_id = :match_id
            AND expires_at > NOW()
            ORDER BY computed_at DESC LIMIT 1
        """
        )

        db = self._require_db()
        result = db.execute(query, {"match_id": match_id}).fetchone()
        if result:
            return result.features
        return {}

    def _get_stored_prediction(self, match_id: str) -> Optional[Dict]:
        """Get existing prediction from database"""

        query = text(
            """
            SELECT predicted_outcome, confidence, probabilities
            FROM predictions
            WHERE match_id = :match_id
            ORDER BY created_at DESC LIMIT 1
        """
        )

        db = self._require_db()
        result = db.execute(query, {"match_id": match_id}).fetchone()
        if result:
            return {
                "predicted_label": result.predicted_outcome,
                "confidence": float(result.confidence),
                "probabilities": result.probabilities or {},
            }
        return None

    def _store_prediction(self, match_id: str, prediction: Dict):
        """Store prediction in database"""

        query = text(
            """
            INSERT INTO predictions
            (match_id, model_name, model_version, prediction_type,
             predicted_outcome, confidence, probabilities)
            VALUES (:match_id, :model_name, :model_version, :prediction_type,
                    :predicted_outcome, :confidence, CAST(:probabilities AS jsonb))
            ON CONFLICT (match_id, model_name, model_version, prediction_type)
            DO UPDATE SET
                predicted_outcome = EXCLUDED.predicted_outcome,
                confidence = EXCLUDED.confidence,
                probabilities = EXCLUDED.probabilities,
                updated_at = NOW()
        """
        )

        try:
            import json

            db = self._require_db()
            db.execute(
                query,
                {
                    "match_id": match_id,
                    "model_name": "ensemble",
                    "model_version": "v1.0",
                    "prediction_type": "match_result",
                    "predicted_outcome": prediction["predicted_label"],
                    "confidence": prediction["confidence"],
                    "probabilities": json.dumps(prediction["probabilities"]),
                },
            )
            db.commit()
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
            if self.db is not None:
                self.db.rollback()
