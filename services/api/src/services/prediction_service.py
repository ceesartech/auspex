"""Prediction service"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from models.responses import MatchInfo, PredictionResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── Multi-sport task registry ─────────────────────────────────────────
# Each entry maps a (sport, market) pair to the registry directory, label
# set, DB prediction_type, base-model list, and features_cache key needed
# to compute and persist that task's predictions. Adding a new sport or
# market means adding a TaskSpec entry — the orchestration code is
# task-agnostic. Soccer's match_result is the legacy task and uses the
# unsuffixed "ensemble" registry name for backwards compatibility with
# the pre-NHL deployment.


@dataclass(frozen=True)
class TaskSpec:
    """Configuration for one prediction task (sport + market combo)."""

    sport: str  # leagues.sport value: 'soccer' or 'nhl'
    market: str  # human-readable market label
    ensemble_name: str  # registry directory name (e.g. 'ensemble', 'ensemble_nhl_ml')
    labels: List[str]  # output label order — index matches predict_proba columns
    prediction_type: str  # value persisted to predictions.prediction_type (CHECK constraint)
    base_models: List[str]  # registry directories of base models to load alongside the ensemble
    feature_set: str  # features_cache.feature_set filter for compute / lookup


# Task registry. The key format is "{sport}:{market}" so call sites can
# look up a task by composite key. Insertion order is stable so iteration
# at load time is deterministic.
TASKS: Dict[str, TaskSpec] = {
    "soccer:match_result": TaskSpec(
        sport="soccer",
        market="match_result",
        ensemble_name="ensemble",
        labels=["home", "draw", "away"],
        prediction_type="match_result",
        base_models=["xgboost", "lightgbm", "neural_network", "poisson", "dixon_coles"],
        feature_set="baseline",
    ),
    "nhl:moneyline": TaskSpec(
        sport="nhl",
        market="moneyline",
        ensemble_name="ensemble_nhl_ml",
        labels=["home", "away"],
        prediction_type="moneyline",
        base_models=["xgboost_nhl_ml", "lightgbm_nhl_ml", "neural_network_nhl_ml", "hockey_poisson_nhl_ml"],
        feature_set="nhl_baseline",
    ),
    "nhl:regulation": TaskSpec(
        sport="nhl",
        market="regulation",
        ensemble_name="ensemble_nhl_reg",
        # 3-class: home reg win / regulation tie / away reg win
        labels=["home", "tie", "away"],
        # 'match_result' is the closest existing CHECK-allowed value; a
        # follow-up migration could add 'regulation_winner' explicitly.
        prediction_type="match_result",
        base_models=["xgboost_nhl_reg", "lightgbm_nhl_reg", "neural_network_nhl_reg", "hockey_poisson_nhl_reg"],
        feature_set="nhl_baseline",
    ),
    "nhl:puck_line": TaskSpec(
        sport="nhl",
        market="puck_line",
        ensemble_name="ensemble_nhl_pl",
        labels=["cover", "no_cover"],
        prediction_type="spread",
        base_models=["xgboost_nhl_pl", "lightgbm_nhl_pl", "neural_network_nhl_pl", "hockey_poisson_nhl_pl"],
        feature_set="nhl_baseline",
    ),
    "nhl:total": TaskSpec(
        sport="nhl",
        market="total",
        ensemble_name="ensemble_nhl_tot",
        labels=["over", "under"],
        prediction_type="total",
        base_models=["xgboost_nhl_tot", "lightgbm_nhl_tot", "neural_network_nhl_tot", "hockey_poisson_nhl_tot"],
        feature_set="nhl_baseline",
    ),
}


def tasks_for_sport(sport: str) -> List[TaskSpec]:
    """All registered tasks for a given sport, in registry order."""
    return [spec for spec in TASKS.values() if spec.sport == sport]


# Process-wide model registry. Populated once during app startup
# (see services/api/src/main.py lifespan) and shared across requests.
# Loading joblib pickles on every request adds ~10-50ms; this avoids that.
# Keys are the TaskSpec composite key ("soccer:match_result", "nhl:moneyline",
# etc.). Soccer's "ensemble" entry is also mirrored at the legacy key
# "ensemble" for backwards compatibility with any callers that still
# reach in directly.
_MODELS: Dict[str, Any] = {}
_MODEL_VERSIONS: Dict[str, str] = {}
_MODEL_VERSION: str = "ensemble_v1.0"  # Legacy global; soccer version


def get_loaded_models() -> Dict[str, Any]:
    """Return the process-wide loaded model registry."""
    return _MODELS


def get_model_version(task_key: Optional[str] = None) -> str:
    """Version string for a task's ensemble. Defaults to soccer match_result
    for backwards compatibility with cache-key callers."""
    if task_key and task_key in _MODEL_VERSIONS:
        return _MODEL_VERSIONS[task_key]
    return _MODEL_VERSION


def _predict_one(model, features: Dict[str, Any], labels: List[str]) -> Dict[str, Any]:
    """Adapter from the (features dict) call shape used by the route + DAG
    to EnsemblePredictor.predict_proba(DataFrame).

    `labels` defines the output label order — must match the model's
    predict_proba column order (which in turn matches the LabelEncoder
    that was fit at training time). 2-class tasks (NHL moneyline/PL/total)
    pass 2 labels; 3-class tasks (soccer match_result, NHL regulation)
    pass 3.
    """
    import pandas as pd

    X = pd.DataFrame([features])
    proba = model.predict_proba(X)[0]
    if len(proba) != len(labels):
        raise ValueError(
            f"predict_proba returned {len(proba)} classes but {len(labels)} labels were provided. " f"Labels: {labels}"
        )
    idx = int(proba.argmax())
    return {
        "predicted_label": labels[idx],
        "confidence": float(proba[idx]),
        "probabilities": {labels[i]: float(proba[i]) for i in range(len(labels))},
    }


def _latest_model_bin(root: Path) -> Optional[Path]:
    """Return the most recent `*/model.bin` under `root` (by mtime), or None."""
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*/model.bin"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _klass_registry():
    """Build the {base_model_name → (predictor class, ModelConfig)} map.

    Pulled into a helper so we can call it once per process and reuse
    across multiple ensemble loads. Each NHL ensemble references its
    own per-task config (e.g. XGBOOST_NHL_MONEYLINE) so the LabelEncoder
    and num_class settings match what was used at training time.
    """
    import sys

    # The ml-models predictor classes live under `predictors.*` (renamed
    # from the old `models.*` to avoid collision with api/src/models/,
    # which is the pydantic schema package). PYTHONPATH from
    # Dockerfile.api includes /app/services/ml-models/src so these
    # resolve directly.
    ml_src = "/app/services/ml-models/src"
    if ml_src not in sys.path:
        sys.path.insert(0, ml_src)

    from predictors.lightgbm_model import LightGBMMatchPredictor
    from predictors.model_config import (
        DIXON_COLES_CONFIG,
        HOCKEY_POISSON_NHL_MONEYLINE,
        HOCKEY_POISSON_NHL_PUCK_LINE,
        HOCKEY_POISSON_NHL_REGULATION,
        HOCKEY_POISSON_NHL_TOTAL,
        LIGHTGBM_MATCH_OUTCOME,
        LIGHTGBM_NHL_MONEYLINE,
        LIGHTGBM_NHL_PUCK_LINE,
        LIGHTGBM_NHL_REGULATION,
        LIGHTGBM_NHL_TOTAL,
        NEURAL_NETWORK_CONFIG,
        NEURAL_NETWORK_NHL_MONEYLINE,
        NEURAL_NETWORK_NHL_PUCK_LINE,
        NEURAL_NETWORK_NHL_REGULATION,
        NEURAL_NETWORK_NHL_TOTAL,
        POISSON_CONFIG,
        XGBOOST_MATCH_OUTCOME,
        XGBOOST_NHL_MONEYLINE,
        XGBOOST_NHL_PUCK_LINE,
        XGBOOST_NHL_REGULATION,
        XGBOOST_NHL_TOTAL,
    )
    from predictors.neural_network import NeuralNetworkMatchPredictor
    from predictors.poisson_models import DixonColesPredictor, HockeyPoissonPredictor, PoissonMatchPredictor
    from predictors.xgboost_model import XGBoostMatchPredictor

    return {
        # Soccer base models (legacy match_result ensemble)
        "xgboost": (XGBoostMatchPredictor, XGBOOST_MATCH_OUTCOME),
        "lightgbm": (LightGBMMatchPredictor, LIGHTGBM_MATCH_OUTCOME),
        "neural_network": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_CONFIG),
        "poisson": (PoissonMatchPredictor, POISSON_CONFIG),
        "dixon_coles": (DixonColesPredictor, DIXON_COLES_CONFIG),
        # NHL moneyline
        "xgboost_nhl_ml": (XGBoostMatchPredictor, XGBOOST_NHL_MONEYLINE),
        "lightgbm_nhl_ml": (LightGBMMatchPredictor, LIGHTGBM_NHL_MONEYLINE),
        "neural_network_nhl_ml": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NHL_MONEYLINE),
        "hockey_poisson_nhl_ml": (HockeyPoissonPredictor, HOCKEY_POISSON_NHL_MONEYLINE),
        # NHL regulation
        "xgboost_nhl_reg": (XGBoostMatchPredictor, XGBOOST_NHL_REGULATION),
        "lightgbm_nhl_reg": (LightGBMMatchPredictor, LIGHTGBM_NHL_REGULATION),
        "neural_network_nhl_reg": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NHL_REGULATION),
        "hockey_poisson_nhl_reg": (HockeyPoissonPredictor, HOCKEY_POISSON_NHL_REGULATION),
        # NHL puck line
        "xgboost_nhl_pl": (XGBoostMatchPredictor, XGBOOST_NHL_PUCK_LINE),
        "lightgbm_nhl_pl": (LightGBMMatchPredictor, LIGHTGBM_NHL_PUCK_LINE),
        "neural_network_nhl_pl": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NHL_PUCK_LINE),
        "hockey_poisson_nhl_pl": (HockeyPoissonPredictor, HOCKEY_POISSON_NHL_PUCK_LINE),
        # NHL total
        "xgboost_nhl_tot": (XGBoostMatchPredictor, XGBOOST_NHL_TOTAL),
        "lightgbm_nhl_tot": (LightGBMMatchPredictor, LIGHTGBM_NHL_TOTAL),
        "neural_network_nhl_tot": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NHL_TOTAL),
        "hockey_poisson_nhl_tot": (HockeyPoissonPredictor, HOCKEY_POISSON_NHL_TOTAL),
    }


def _build_ensemble_for_task(
    model_path: Path,
    task: TaskSpec,
    klass_for: Dict[str, tuple],
) -> Tuple[Optional[Any], Optional[Path]]:
    """Reconstitute the EnsemblePredictor + its base models for one task.

    The training pipeline writes:
      {model_path}/{task.ensemble_name}/{version}/model.bin (JSON: weights + model_names)
      {model_path}/{base_model_name}/{version}/model.bin (per base model)

    EnsemblePredictor.save() only persists weights + base-model names —
    not the base models themselves — so reloading requires us to walk
    the registry, instantiate each referenced base model class, load
    its artifact, and re-attach via add_model() before applying weights.

    Returns (ensemble, meta_path) or (None, None) if the task can't be
    loaded (missing artifacts, etc.) — caller logs the warning and
    continues.
    """
    import json
    import sys

    ml_src = "/app/services/ml-models/src"
    if ml_src not in sys.path:
        sys.path.insert(0, ml_src)

    from predictors.ensemble import EnsemblePredictor
    from predictors.model_config import (
        ENSEMBLE_CONFIG,
        ENSEMBLE_NHL_MONEYLINE,
        ENSEMBLE_NHL_PUCK_LINE,
        ENSEMBLE_NHL_REGULATION,
        ENSEMBLE_NHL_TOTAL,
    )

    # Pick the task-specific ensemble config so the EnsemblePredictor's
    # internal LabelEncoder and prediction_task match what was used at
    # training time. Fall back to ENSEMBLE_CONFIG for soccer compatibility.
    ensemble_cfg_for = {
        "ensemble": ENSEMBLE_CONFIG,
        "ensemble_nhl_ml": ENSEMBLE_NHL_MONEYLINE,
        "ensemble_nhl_reg": ENSEMBLE_NHL_REGULATION,
        "ensemble_nhl_pl": ENSEMBLE_NHL_PUCK_LINE,
        "ensemble_nhl_tot": ENSEMBLE_NHL_TOTAL,
    }
    ensemble_cfg = ensemble_cfg_for.get(task.ensemble_name, ENSEMBLE_CONFIG)

    ensemble_meta = _latest_model_bin(model_path / task.ensemble_name)
    if ensemble_meta is None:
        return None, None

    with open(ensemble_meta) as f:
        meta = json.load(f)

    ensemble = EnsemblePredictor(ensemble_cfg)
    loaded_names: List[str] = []
    for name in meta.get("model_names", []):
        if name not in klass_for:
            logger.warning("Task %s ensemble references unknown base model %r; skipping", task.ensemble_name, name)
            continue
        artifact = _latest_model_bin(model_path / name)
        if artifact is None:
            logger.warning(
                "Task %s ensemble references %r but no artifact under %s/%s/",
                task.ensemble_name,
                name,
                model_path,
                name,
            )
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
        logger.error("Could not reconstitute any base models for task %s", task.ensemble_name)
        return None, None

    # ensemble.load() restores the blend weights from JSON metadata.
    ensemble.load(str(ensemble_meta))
    return ensemble, ensemble_meta


def load_models_into_process(model_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load ML models into the process-wide registry. Idempotent.

    Called once from FastAPI's lifespan startup. Walks every TaskSpec in
    the TASKS registry and tries to load the corresponding ensemble +
    base models from disk. Tasks without on-disk artifacts (e.g. NHL
    ensembles on a fresh deployment where Phase 3 hasn't run yet) get
    a single warning and are skipped — the rest still load. Returns
    the loaded model registry keyed by task composite key.
    """
    global _MODEL_VERSION

    if _MODELS:
        return _MODELS

    model_path = model_dir or (Path(settings.MODEL_PATH) / "production")
    try:
        klass_for = _klass_registry()
    except Exception as e:
        logger.error("Could not import predictor classes: %s", e, exc_info=True)
        return _MODELS

    for task_key, task in TASKS.items():
        try:
            ensemble, meta_path = _build_ensemble_for_task(model_path, task, klass_for)
        except Exception as e:
            logger.error("Failed to load task %s: %s", task_key, e, exc_info=True)
            continue
        if ensemble is None:
            logger.warning(
                "Task %s: ensemble not found under %s/%s/*/model.bin; skipping",
                task_key,
                model_path,
                task.ensemble_name,
            )
            continue
        _MODELS[task_key] = ensemble
        mtime = int(meta_path.stat().st_mtime)
        version_label = meta_path.parent.name
        _MODEL_VERSIONS[task_key] = f"{task.ensemble_name}_{version_label}+{mtime}"
        logger.info(
            "Loaded task %s (version=%s, base_models=%s)",
            task_key,
            _MODEL_VERSIONS[task_key],
            list(ensemble.models.keys()),
        )

    # Legacy compatibility: pre-NHL callers reach for _MODELS["ensemble"]
    # directly and _MODEL_VERSION as a global. Mirror soccer's task into
    # both so the existing predict_match path keeps working unchanged.
    soccer_key = "soccer:match_result"
    if soccer_key in _MODELS:
        _MODELS["ensemble"] = _MODELS[soccer_key]
        _MODEL_VERSION = _MODEL_VERSIONS[soccer_key]

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
        market: Optional[str] = None,
    ) -> PredictionResponse:
        """Generate prediction(s) for a match. Routes based on the match's
        sport: soccer matches predict 1 task (match_result); NHL matches
        predict 4 tasks (moneyline, regulation, puck_line, total) and
        return the moneyline as the headline response (since it's the
        most-marketable single prediction).

        `market` lets the caller pin to a specific task instead of the
        headline default — useful for the API when a user explicitly
        wants e.g. NHL totals.
        """

        match_data = self._get_match_data(match_id)
        if not match_data:
            raise ValueError(f"Match {match_id} not found")

        sport = match_data.get("sport") or "soccer"
        features = self._get_match_features(match_id, feature_set=None)

        # Pick which task(s) to run for this match.
        applicable = tasks_for_sport(sport)
        if not applicable:
            # Unknown sport — fall through to stored-prediction lookup.
            prediction = self._get_stored_prediction(match_id)
            if not prediction:
                raise ValueError(f"No tasks registered for sport={sport!r}")
            return self._response_from_prediction(match_data, prediction, model_version=_MODEL_VERSION)

        # Determine the headline task (the one whose response we return).
        headline_task: Optional[TaskSpec] = None
        if market:
            headline_task = next((t for t in applicable if t.market == market), None)
            if headline_task is None:
                raise ValueError(
                    f"No task for sport={sport!r} market={market!r}. "
                    f"Available markets: {[t.market for t in applicable]}"
                )
        else:
            # Default headline: match_result for soccer, moneyline for NHL.
            headline_task = applicable[0]

        # Run + persist every applicable task; remember the headline
        # prediction for the response object.
        headline_prediction: Optional[Dict[str, Any]] = None
        for task in applicable:
            task_key = f"{task.sport}:{task.market}"
            model = self.models.get(task_key)
            if model is None:
                logger.warning("Skipping task %s: model not loaded", task_key)
                continue
            try:
                prediction = _predict_one(model, features, task.labels)
            except Exception as e:
                logger.error("predict failed for task %s match %s: %s", task_key, match_id, e)
                continue
            self._store_prediction(match_id, prediction, task)
            if task is headline_task:
                headline_prediction = prediction

        if headline_prediction is None:
            # Final fallback: stored prediction from DB so the request
            # still returns something usable while we wait for models
            # to populate.
            headline_prediction = self._get_stored_prediction(match_id)
            if not headline_prediction:
                raise ValueError(f"No prediction available for match {match_id}")

        headline_version = _MODEL_VERSIONS.get(
            f"{headline_task.sport}:{headline_task.market}",
            _MODEL_VERSION,
        )
        response = self._response_from_prediction(
            match_data,
            headline_prediction,
            model_version=headline_version,
            include_explanation=include_explanation,
        )
        return response

    def _response_from_prediction(
        self,
        match_data: Dict,
        prediction: Dict,
        model_version: str,
        include_explanation: bool = True,
    ) -> PredictionResponse:
        """Build a PredictionResponse from a prediction dict. Filters
        out the 'sport' key from match_data before passing to MatchInfo
        since MatchInfo doesn't accept it."""
        match_info_kwargs = {k: v for k, v in match_data.items() if k != "sport"}
        return PredictionResponse(
            match_info=MatchInfo(**match_info_kwargs),
            predicted_outcome=prediction["predicted_label"],
            probabilities=prediction["probabilities"],
            confidence=prediction["confidence"],
            model_version=model_version,
            timestamp=datetime.utcnow(),
            explanation=prediction.get("explanation") if include_explanation else None,
        )

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
        """Get match information from database. Includes sport so
        predict_match can route to the right task ensemble."""

        query = text(
            """
            SELECT m.id, l.name as league_name, l.sport as sport,
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
                "sport": result.sport,
                "home_team": result.home_team,
                "away_team": result.away_team,
                "match_date": result.match_date,
                "venue": result.venue,
            }
        return None

    def _get_match_features(
        self,
        match_id: str,
        feature_set: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get computed features for a match from cache. Without
        feature_set, returns the most-recent fresh row regardless of
        which feature set wrote it — fine for soccer where there's
        only one set ('baseline') but ambiguous for NHL once both
        soccer and NHL feature sets coexist. Passing feature_set pins
        the lookup."""

        if feature_set is None:
            query = text(
                """
                SELECT features FROM features_cache
                WHERE match_id = :match_id
                AND expires_at > NOW()
                ORDER BY computed_at DESC LIMIT 1
            """
            )
            params = {"match_id": match_id}
        else:
            query = text(
                """
                SELECT features FROM features_cache
                WHERE match_id = :match_id
                AND feature_set = :feature_set
                AND expires_at > NOW()
                ORDER BY computed_at DESC LIMIT 1
            """
            )
            params = {"match_id": match_id, "feature_set": feature_set}

        db = self._require_db()
        result = db.execute(query, params).fetchone()
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

    def _store_prediction(
        self,
        match_id: str,
        prediction: Dict,
        task: Optional[TaskSpec] = None,
    ):
        """Store prediction in database. With `task`, uses the task's
        ensemble_name + prediction_type so soccer and NHL rows coexist
        without colliding on the (match_id, model_name, model_version,
        prediction_type) unique constraint. Without `task`, falls back
        to the legacy soccer match_result values for backwards compat."""

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

        if task is not None:
            task_key = f"{task.sport}:{task.market}"
            model_name = task.ensemble_name
            model_version = _MODEL_VERSIONS.get(task_key, "v1.0")
            prediction_type = task.prediction_type
        else:
            model_name = "ensemble"
            model_version = "v1.0"
            prediction_type = "match_result"

        try:
            import json

            db = self._require_db()
            db.execute(
                query,
                {
                    "match_id": match_id,
                    "model_name": model_name,
                    "model_version": model_version,
                    "prediction_type": prediction_type,
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
