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
        import joblib

        ensemble_path = model_path / "ensemble_model.pkl"
        if ensemble_path.exists():
            _MODELS["ensemble"] = joblib.load(ensemble_path)
            # Derive version from mtime to invalidate caches when artifacts change.
            mtime = int(ensemble_path.stat().st_mtime)
            _MODEL_VERSION = f"ensemble_v1.0+{mtime}"
            logger.info("Loaded ensemble model (version=%s)", _MODEL_VERSION)
        else:
            logger.warning("Ensemble model not found at %s; serving from DB fallback", ensemble_path)
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
            prediction = self.models["ensemble"].predict_single(features, explain=include_explanation)
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

        query = text("""
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
        """)

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

        query = text("""
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
        """)

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

        query = text("""
            SELECT m.id, l.name as league_name,
                   ht.name as home_team, at.name as away_team,
                   m.match_date, m.venue
            FROM matches m
            JOIN leagues l ON m.league_id = l.id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.id = :match_id
        """)

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

        query = text("""
            SELECT features FROM features_cache
            WHERE match_id = :match_id
            AND expires_at > NOW()
            ORDER BY computed_at DESC LIMIT 1
        """)

        db = self._require_db()
        result = db.execute(query, {"match_id": match_id}).fetchone()
        if result:
            return result.features
        return {}

    def _get_stored_prediction(self, match_id: str) -> Optional[Dict]:
        """Get existing prediction from database"""

        query = text("""
            SELECT predicted_outcome, confidence, probabilities
            FROM predictions
            WHERE match_id = :match_id
            ORDER BY created_at DESC LIMIT 1
        """)

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

        query = text("""
            INSERT INTO predictions
            (match_id, model_name, model_version, prediction_type,
             predicted_outcome, confidence, probabilities)
            VALUES (:match_id, :model_name, :model_version, :prediction_type,
                    :predicted_outcome, :confidence, CAST(:probabilities AS jsonb))
            ON CONFLICT (match_id, model_name, model_version)
            DO UPDATE SET
                predicted_outcome = EXCLUDED.predicted_outcome,
                confidence = EXCLUDED.confidence,
                probabilities = EXCLUDED.probabilities,
                updated_at = NOW()
        """)

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
