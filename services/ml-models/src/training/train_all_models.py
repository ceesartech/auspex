"""Training orchestrator — trains all models and registers results."""

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from models.base_model import BaseModel
from models.ensemble import EnsemblePredictor
from models.lightgbm_model import LightGBMMatchPredictor
from models.model_config import (
    DIXON_COLES_CONFIG,
    ENSEMBLE_CONFIG,
    LIGHTGBM_MATCH_OUTCOME,
    NEURAL_NETWORK_CONFIG,
    POISSON_CONFIG,
    XGBOOST_MATCH_OUTCOME,
    ModelConfig,
)
from models.model_registry import ModelRegistry
from models.neural_network import NeuralNetworkMatchPredictor
from models.poisson_models import DixonColesPredictor, PoissonMatchPredictor
from models.xgboost_model import XGBoostMatchPredictor
from training.calibration import ProbabilityCalibrator

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Orchestrates training of all models."""

    def __init__(
        self,
        registry_dir: str = "model_registry",
        run_cv: bool = True,
        calibrate: bool = True,
    ):
        self.registry = ModelRegistry(registry_dir)
        self.run_cv = run_cv
        self.calibrate = calibrate
        self.trained_models: Dict[str, BaseModel] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

    def train_all(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        features: List[str],
        target: str = "match_outcome",
    ) -> Dict[str, Any]:
        """Train all individual models and then the ensemble."""
        start = time.time()
        logger.info("Starting training of all models...")

        # 1. Train XGBoost
        self._train_model(
            "xgboost",
            XGBoostMatchPredictor,
            XGBOOST_MATCH_OUTCOME,
            train_df, val_df, features, target,
        )

        # 2. Train LightGBM
        self._train_model(
            "lightgbm",
            LightGBMMatchPredictor,
            LIGHTGBM_MATCH_OUTCOME,
            train_df, val_df, features, target,
        )

        # 3. Train Neural Network
        self._train_model(
            "neural_network",
            NeuralNetworkMatchPredictor,
            NEURAL_NETWORK_CONFIG,
            train_df, val_df, features, target,
        )

        # 4. Train Poisson (needs team columns)
        if "home_team" in train_df.columns:
            self._train_model(
                "poisson",
                PoissonMatchPredictor,
                POISSON_CONFIG,
                train_df, val_df, None, target,
            )

            # 5. Train Dixon-Coles
            self._train_model(
                "dixon_coles",
                DixonColesPredictor,
                DIXON_COLES_CONFIG,
                train_df, val_df, None, target,
            )

        # 6. Train Ensemble
        if len(self.trained_models) >= 2:
            self._train_ensemble(val_df, target)

        duration = time.time() - start
        logger.info(f"All training completed in {duration:.1f}s")

        return self.results

    def _train_model(
        self,
        name: str,
        model_class: type,
        config: ModelConfig,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        features: Optional[List[str]],
        target: str,
    ) -> None:
        logger.info(f"Training {name}...")
        start = time.time()

        try:
            model = model_class(config)
            result = model.train(
                train_df, val_df=val_df, features=features, target=target
            )

            self.trained_models[name] = model
            self.results[name] = result

            # Calibrate if requested
            if self.calibrate and features:
                try:
                    X_val = val_df[features].fillna(val_df[features].median())
                    y_proba = model.predict_proba(X_val)
                    if hasattr(model, "label_encoder"):
                        y_val = model.label_encoder.transform(val_df[target])
                    else:
                        y_val = val_df[target].values

                    calibrator = ProbabilityCalibrator(method="isotonic")
                    calibrator.fit(y_proba, y_val)
                    cal_error = calibrator.calibration_error(y_proba, y_val)
                    self.results[name]["calibration"] = cal_error
                except Exception as e:
                    logger.warning(f"Calibration failed for {name}: {e}")

            # Register model
            metrics = result.get("validation_metrics", {})
            self.registry.register_model(
                model=model,
                name=name,
                version=config.version,
                metrics=metrics,
            )

            duration = time.time() - start
            logger.info(f"{name} training completed in {duration:.1f}s")

        except Exception as e:
            logger.error(f"Failed to train {name}: {e}", exc_info=True)
            self.results[name] = {"error": str(e)}

    def _train_ensemble(self, val_df: pd.DataFrame, target: str) -> None:
        logger.info("Training ensemble...")

        ensemble = EnsemblePredictor(ENSEMBLE_CONFIG)
        for name, model in self.trained_models.items():
            ensemble.add_model(name, model)

        try:
            result = ensemble.train(val_df=val_df, target=target)
            self.trained_models["ensemble"] = ensemble
            self.results["ensemble"] = result

            self.registry.register_model(
                model=ensemble,
                name="ensemble",
                version=ENSEMBLE_CONFIG.version,
                metrics=result.get("validation_metrics", {}),
            )
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}", exc_info=True)
            self.results["ensemble"] = {"error": str(e)}

    def get_best_model(self, metric: str = "accuracy") -> Optional[str]:
        """Get the name of the best model by a metric."""
        best_name = None
        best_score = -float("inf")

        for name, result in self.results.items():
            metrics = result.get("validation_metrics", {})
            score = metrics.get(metric)
            if score is not None and score > best_score:
                best_score = score
                best_name = name

        return best_name
