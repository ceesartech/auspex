"""Training orchestrator for model retraining."""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from inference.onnx_converter import ONNXConverter
from predictors.base_model import BaseModel
from predictors.ensemble import EnsemblePredictor
from predictors.lightgbm_model import LightGBMMatchPredictor
from predictors.model_config import (
    DIXON_COLES_CONFIG,
    ENSEMBLE_CONFIG,
    ENSEMBLE_MMA_MONEYLINE,
    ENSEMBLE_NBA_MONEYLINE,
    ENSEMBLE_NBA_SPREAD,
    ENSEMBLE_NBA_TOTAL,
    ENSEMBLE_NFL_MONEYLINE,
    ENSEMBLE_NFL_SPREAD,
    ENSEMBLE_NFL_TOTAL,
    ENSEMBLE_NHL_MONEYLINE,
    ENSEMBLE_NHL_PUCK_LINE,
    ENSEMBLE_NHL_REGULATION,
    ENSEMBLE_NHL_TOTAL,
    ENSEMBLE_TENNIS_MONEYLINE,
    HOCKEY_POISSON_NHL_MONEYLINE,
    HOCKEY_POISSON_NHL_PUCK_LINE,
    HOCKEY_POISSON_NHL_REGULATION,
    HOCKEY_POISSON_NHL_TOTAL,
    LIGHTGBM_MATCH_OUTCOME,
    LIGHTGBM_MMA_MONEYLINE,
    LIGHTGBM_NBA_MONEYLINE,
    LIGHTGBM_NBA_SPREAD,
    LIGHTGBM_NBA_TOTAL,
    LIGHTGBM_NFL_MONEYLINE,
    LIGHTGBM_NFL_SPREAD,
    LIGHTGBM_NFL_TOTAL,
    LIGHTGBM_NHL_MONEYLINE,
    LIGHTGBM_NHL_PUCK_LINE,
    LIGHTGBM_NHL_REGULATION,
    LIGHTGBM_NHL_TOTAL,
    LIGHTGBM_TENNIS_MONEYLINE,
    NEURAL_NETWORK_CONFIG,
    NEURAL_NETWORK_MMA_MONEYLINE,
    NEURAL_NETWORK_NBA_MONEYLINE,
    NEURAL_NETWORK_NBA_SPREAD,
    NEURAL_NETWORK_NBA_TOTAL,
    NEURAL_NETWORK_NFL_MONEYLINE,
    NEURAL_NETWORK_NFL_SPREAD,
    NEURAL_NETWORK_NFL_TOTAL,
    NEURAL_NETWORK_NHL_MONEYLINE,
    NEURAL_NETWORK_NHL_PUCK_LINE,
    NEURAL_NETWORK_NHL_REGULATION,
    NEURAL_NETWORK_NHL_TOTAL,
    NEURAL_NETWORK_TENNIS_MONEYLINE,
    POISSON_CONFIG,
    XGBOOST_MATCH_OUTCOME,
    XGBOOST_MMA_MONEYLINE,
    XGBOOST_NBA_MONEYLINE,
    XGBOOST_NBA_SPREAD,
    XGBOOST_NBA_TOTAL,
    XGBOOST_NFL_MONEYLINE,
    XGBOOST_NFL_SPREAD,
    XGBOOST_NFL_TOTAL,
    XGBOOST_NHL_MONEYLINE,
    XGBOOST_NHL_PUCK_LINE,
    XGBOOST_NHL_REGULATION,
    XGBOOST_NHL_TOTAL,
    XGBOOST_TENNIS_MONEYLINE,
    ModelConfig,
)
from predictors.model_registry import ModelRegistry
from predictors.neural_network import NeuralNetworkMatchPredictor
from predictors.poisson_models import DixonColesPredictor, HockeyPoissonPredictor, PoissonMatchPredictor
from predictors.xgboost_model import XGBoostMatchPredictor
from training.calibration import ProbabilityCalibrator
from utils.training_data import (
    MMA_MONEYLINE_TARGET,
    NBA_MONEYLINE_TARGET,
    NBA_SPREAD_TARGET,
    NBA_TOTAL_TARGET,
    NFL_MONEYLINE_TARGET,
    NFL_SPREAD_TARGET,
    NFL_TOTAL_TARGET,
    NHL_MONEYLINE_TARGET,
    NHL_PUCK_LINE_TARGET,
    NHL_REGULATION_TARGET,
    NHL_TOTAL_TARGET,
    TARGET_COLUMN,
    TENNIS_MONEYLINE_TARGET,
    get_feature_columns,
    get_mma_moneyline_feature_columns,
    get_nba_moneyline_feature_columns,
    get_nba_spread_feature_columns,
    get_nba_total_feature_columns,
    get_nfl_moneyline_feature_columns,
    get_nfl_spread_feature_columns,
    get_nfl_total_feature_columns,
    get_nhl_feature_columns,
    get_nhl_puck_line_feature_columns,
    get_nhl_regulation_feature_columns,
    get_nhl_total_feature_columns,
    get_tennis_moneyline_feature_columns,
    load_mma_moneyline_frame,
    load_nba_moneyline_frame,
    load_nba_spread_frame,
    load_nba_total_frame,
    load_nfl_moneyline_frame,
    load_nfl_spread_frame,
    load_nfl_total_frame,
    load_nhl_moneyline_frame,
    load_nhl_puck_line_frame,
    load_nhl_regulation_frame,
    load_nhl_total_frame,
    load_tennis_moneyline_frame,
    load_training_frame,
    validate_training_frame,
)

logger = logging.getLogger(__name__)


# ── Sport bundles ─────────────────────────────────────────────────────
# Each bundle wires up the four sport-specific pieces of a training run:
#   * how to load the frame (different SQL queries / target columns)
#   * which feature-selection helper to use
#   * which (model_name, model_class, ModelConfig) triples to train
#   * which default target column to pass through
# Adding a new sport/task = adding a SportBundle entry. The orchestrator
# itself stays sport-agnostic.


@dataclass(frozen=True)
class ModelSpec:
    """One model in a sport bundle: registry name + class + config."""

    name: str
    cls: type
    config: ModelConfig
    needs_team_columns: bool = False  # True only for Poisson / Dixon-Coles


@dataclass(frozen=True)
class SportBundle:
    sport: str  # CLI value: 'soccer', 'nhl_moneyline'
    target_column: str
    load_frame: Callable[..., pd.DataFrame]
    feature_columns: Callable[[pd.DataFrame, str], List[str]]
    base_models: List[ModelSpec]
    ensemble_config: ModelConfig
    ensemble_name: str  # registry name for the ensemble


# Soccer trains EXACTLY ONE ensemble for match_result (1X2). The other
# ~15 soccer markets (BTTS, over/under variants, double-chance, draw-no-bet,
# Asian handicaps, correct score) are DERIVED at predict time from this
# ensemble's Dixon-Coles base model — see scripts/precompute_predictions.py
# and the DERIVED_SOCCER_MARKETS list in
# services/api/src/services/prediction_service.py.
#
# The "_soccer_match_result" suffix on every artifact name is intentional:
# it mirrors the NHL naming (xgboost_nhl_ml etc.) so the on-disk layout
# under /app/models/production/ explicitly tells you what each model
# trained for. Ensembles trained before this rename still load via the
# legacy_ensemble_name / legacy_base_models fallback in TaskSpec.
SOCCER_BUNDLE = SportBundle(
    sport="soccer_match_result",
    target_column=TARGET_COLUMN,
    load_frame=load_training_frame,
    feature_columns=get_feature_columns,
    base_models=[
        ModelSpec("xgboost_soccer_match_result", XGBoostMatchPredictor, XGBOOST_MATCH_OUTCOME),
        ModelSpec("lightgbm_soccer_match_result", LightGBMMatchPredictor, LIGHTGBM_MATCH_OUTCOME),
        ModelSpec("neural_network_soccer_match_result", NeuralNetworkMatchPredictor, NEURAL_NETWORK_CONFIG),
        ModelSpec(
            "poisson_soccer_match_result",
            PoissonMatchPredictor,
            POISSON_CONFIG,
            needs_team_columns=True,
        ),
        ModelSpec(
            "dixon_coles_soccer_match_result",
            DixonColesPredictor,
            DIXON_COLES_CONFIG,
            needs_team_columns=True,
        ),
    ],
    ensemble_config=ENSEMBLE_CONFIG,
    ensemble_name="ensemble_soccer_match_result",
)


NHL_MONEYLINE_BUNDLE = SportBundle(
    sport="nhl_moneyline",
    target_column=NHL_MONEYLINE_TARGET,
    load_frame=load_nhl_moneyline_frame,
    feature_columns=get_nhl_feature_columns,
    base_models=[
        ModelSpec("xgboost_nhl_ml", XGBoostMatchPredictor, XGBOOST_NHL_MONEYLINE),
        ModelSpec("lightgbm_nhl_ml", LightGBMMatchPredictor, LIGHTGBM_NHL_MONEYLINE),
        ModelSpec("neural_network_nhl_ml", NeuralNetworkMatchPredictor, NEURAL_NETWORK_NHL_MONEYLINE),
        # Hockey-Poisson (Phase 3d): joint goal distribution → P(home win
        # incl OT/SO). Needs home_team/away_team columns (not feature cols)
        # for its team-attack/defense MLE fit.
        ModelSpec(
            "hockey_poisson_nhl_ml",
            HockeyPoissonPredictor,
            HOCKEY_POISSON_NHL_MONEYLINE,
            needs_team_columns=True,
        ),
    ],
    ensemble_config=ENSEMBLE_NHL_MONEYLINE,
    ensemble_name="ensemble_nhl_ml",
)


NHL_REGULATION_BUNDLE = SportBundle(
    sport="nhl_regulation",
    target_column=NHL_REGULATION_TARGET,
    load_frame=load_nhl_regulation_frame,
    feature_columns=get_nhl_regulation_feature_columns,
    base_models=[
        ModelSpec("xgboost_nhl_reg", XGBoostMatchPredictor, XGBOOST_NHL_REGULATION),
        ModelSpec("lightgbm_nhl_reg", LightGBMMatchPredictor, LIGHTGBM_NHL_REGULATION),
        ModelSpec("neural_network_nhl_reg", NeuralNetworkMatchPredictor, NEURAL_NETWORK_NHL_REGULATION),
        ModelSpec(
            "hockey_poisson_nhl_reg",
            HockeyPoissonPredictor,
            HOCKEY_POISSON_NHL_REGULATION,
            needs_team_columns=True,
        ),
    ],
    ensemble_config=ENSEMBLE_NHL_REGULATION,
    ensemble_name="ensemble_nhl_reg",
)


NHL_PUCK_LINE_BUNDLE = SportBundle(
    sport="nhl_puck_line",
    target_column=NHL_PUCK_LINE_TARGET,
    load_frame=load_nhl_puck_line_frame,
    feature_columns=get_nhl_puck_line_feature_columns,
    base_models=[
        ModelSpec("xgboost_nhl_pl", XGBoostMatchPredictor, XGBOOST_NHL_PUCK_LINE),
        ModelSpec("lightgbm_nhl_pl", LightGBMMatchPredictor, LIGHTGBM_NHL_PUCK_LINE),
        ModelSpec("neural_network_nhl_pl", NeuralNetworkMatchPredictor, NEURAL_NETWORK_NHL_PUCK_LINE),
        # Hockey-Poisson is the principled answer for puck-line: derive
        # P(home covers -1.5) from the joint goal matrix rather than ask
        # a classifier to learn margin from coarse features.
        ModelSpec(
            "hockey_poisson_nhl_pl",
            HockeyPoissonPredictor,
            HOCKEY_POISSON_NHL_PUCK_LINE,
            needs_team_columns=True,
        ),
    ],
    ensemble_config=ENSEMBLE_NHL_PUCK_LINE,
    ensemble_name="ensemble_nhl_pl",
)


NHL_TOTAL_BUNDLE = SportBundle(
    sport="nhl_total",
    target_column=NHL_TOTAL_TARGET,
    load_frame=load_nhl_total_frame,
    feature_columns=get_nhl_total_feature_columns,
    base_models=[
        ModelSpec("xgboost_nhl_tot", XGBoostMatchPredictor, XGBOOST_NHL_TOTAL),
        ModelSpec("lightgbm_nhl_tot", LightGBMMatchPredictor, LIGHTGBM_NHL_TOTAL),
        ModelSpec("neural_network_nhl_tot", NeuralNetworkMatchPredictor, NEURAL_NETWORK_NHL_TOTAL),
        # Hockey-Poisson directly integrates P(total >= 6) from the joint
        # goal distribution — the natural framing for over/under markets.
        ModelSpec(
            "hockey_poisson_nhl_tot",
            HockeyPoissonPredictor,
            HOCKEY_POISSON_NHL_TOTAL,
            needs_team_columns=True,
        ),
    ],
    ensemble_config=ENSEMBLE_NHL_TOTAL,
    ensemble_name="ensemble_nhl_tot",
)


# ── NBA bundles ───────────────────────────────────────────────────────
# Three markets, line-as-feature design: spread/total models consume
# the actual closing line as an input column so ONE trained model
# handles every line the book offers per game. No Poisson/Dixon-Coles
# — basketball scoring is too high-variance / continuous for a
# discrete prior to add signal over GBDT + neural net.
#
# Suffix convention: _nba_ml / _nba_sp / _nba_tot (matches the
# concise NHL _nhl_ml / _nhl_pl / _nhl_tot pattern). The features_cache
# pin is feature_set='nba_baseline', set at compute time by
# scripts/compute_features_nba.py.

NBA_MONEYLINE_BUNDLE = SportBundle(
    sport="nba_moneyline",
    target_column=NBA_MONEYLINE_TARGET,
    load_frame=load_nba_moneyline_frame,
    feature_columns=get_nba_moneyline_feature_columns,
    base_models=[
        ModelSpec("xgboost_nba_ml", XGBoostMatchPredictor, XGBOOST_NBA_MONEYLINE),
        ModelSpec("lightgbm_nba_ml", LightGBMMatchPredictor, LIGHTGBM_NBA_MONEYLINE),
        ModelSpec("neural_network_nba_ml", NeuralNetworkMatchPredictor, NEURAL_NETWORK_NBA_MONEYLINE),
    ],
    ensemble_config=ENSEMBLE_NBA_MONEYLINE,
    ensemble_name="ensemble_nba_ml",
)


NBA_SPREAD_BUNDLE = SportBundle(
    sport="nba_spread",
    target_column=NBA_SPREAD_TARGET,
    load_frame=load_nba_spread_frame,
    feature_columns=get_nba_spread_feature_columns,
    base_models=[
        ModelSpec("xgboost_nba_sp", XGBoostMatchPredictor, XGBOOST_NBA_SPREAD),
        ModelSpec("lightgbm_nba_sp", LightGBMMatchPredictor, LIGHTGBM_NBA_SPREAD),
        ModelSpec("neural_network_nba_sp", NeuralNetworkMatchPredictor, NEURAL_NETWORK_NBA_SPREAD),
    ],
    ensemble_config=ENSEMBLE_NBA_SPREAD,
    ensemble_name="ensemble_nba_sp",
)


NBA_TOTAL_BUNDLE = SportBundle(
    sport="nba_total",
    target_column=NBA_TOTAL_TARGET,
    load_frame=load_nba_total_frame,
    feature_columns=get_nba_total_feature_columns,
    base_models=[
        ModelSpec("xgboost_nba_tot", XGBoostMatchPredictor, XGBOOST_NBA_TOTAL),
        ModelSpec("lightgbm_nba_tot", LightGBMMatchPredictor, LIGHTGBM_NBA_TOTAL),
        ModelSpec("neural_network_nba_tot", NeuralNetworkMatchPredictor, NEURAL_NETWORK_NBA_TOTAL),
    ],
    ensemble_config=ENSEMBLE_NBA_TOTAL,
    ensemble_name="ensemble_nba_tot",
)


# NFL bundles — same shape as NBA (3 base models + ensemble per market,
# no Poisson/Dixon-Coles). NFL_SPREAD / NFL_TOTAL use the closing line
# as a model feature, mirroring NBA's line-as-feature design.
NFL_MONEYLINE_BUNDLE = SportBundle(
    sport="nfl_moneyline",
    target_column=NFL_MONEYLINE_TARGET,
    load_frame=load_nfl_moneyline_frame,
    feature_columns=get_nfl_moneyline_feature_columns,
    base_models=[
        ModelSpec("xgboost_nfl_ml", XGBoostMatchPredictor, XGBOOST_NFL_MONEYLINE),
        ModelSpec("lightgbm_nfl_ml", LightGBMMatchPredictor, LIGHTGBM_NFL_MONEYLINE),
        ModelSpec("neural_network_nfl_ml", NeuralNetworkMatchPredictor, NEURAL_NETWORK_NFL_MONEYLINE),
    ],
    ensemble_config=ENSEMBLE_NFL_MONEYLINE,
    ensemble_name="ensemble_nfl_ml",
)


NFL_SPREAD_BUNDLE = SportBundle(
    sport="nfl_spread",
    target_column=NFL_SPREAD_TARGET,
    load_frame=load_nfl_spread_frame,
    feature_columns=get_nfl_spread_feature_columns,
    base_models=[
        ModelSpec("xgboost_nfl_sp", XGBoostMatchPredictor, XGBOOST_NFL_SPREAD),
        ModelSpec("lightgbm_nfl_sp", LightGBMMatchPredictor, LIGHTGBM_NFL_SPREAD),
        ModelSpec("neural_network_nfl_sp", NeuralNetworkMatchPredictor, NEURAL_NETWORK_NFL_SPREAD),
    ],
    ensemble_config=ENSEMBLE_NFL_SPREAD,
    ensemble_name="ensemble_nfl_sp",
)


NFL_TOTAL_BUNDLE = SportBundle(
    sport="nfl_total",
    target_column=NFL_TOTAL_TARGET,
    load_frame=load_nfl_total_frame,
    feature_columns=get_nfl_total_feature_columns,
    base_models=[
        ModelSpec("xgboost_nfl_tot", XGBoostMatchPredictor, XGBOOST_NFL_TOTAL),
        ModelSpec("lightgbm_nfl_tot", LightGBMMatchPredictor, LIGHTGBM_NFL_TOTAL),
        ModelSpec("neural_network_nfl_tot", NeuralNetworkMatchPredictor, NEURAL_NETWORK_NFL_TOTAL),
    ],
    ensemble_config=ENSEMBLE_NFL_TOTAL,
    ensemble_name="ensemble_nfl_tot",
)


# Tennis bundles — first 1v1 sport. One market in v1 (moneyline);
# total games + set-betting come in v2 once we parse linescores.
# Reuses NBA hyperparameter shape since the tennis training corpus
# (~12-15k ATP+WTA matches across 3 seasons) is comparable to NBA.
TENNIS_MONEYLINE_BUNDLE = SportBundle(
    sport="tennis_moneyline",
    target_column=TENNIS_MONEYLINE_TARGET,
    load_frame=load_tennis_moneyline_frame,
    feature_columns=get_tennis_moneyline_feature_columns,
    base_models=[
        ModelSpec("xgboost_tennis_ml", XGBoostMatchPredictor, XGBOOST_TENNIS_MONEYLINE),
        ModelSpec("lightgbm_tennis_ml", LightGBMMatchPredictor, LIGHTGBM_TENNIS_MONEYLINE),
        ModelSpec("neural_network_tennis_ml", NeuralNetworkMatchPredictor, NEURAL_NETWORK_TENNIS_MONEYLINE),
    ],
    ensemble_config=ENSEMBLE_TENNIS_MONEYLINE,
    ensemble_name="ensemble_tennis_ml",
)


# MMA bundle — same shape as tennis (XGB+LGB+NN+Ensemble, no team
# columns). UFC corpus ~1500 fights/3 yrs, NBA-shape hyperparameters.
MMA_MONEYLINE_BUNDLE = SportBundle(
    sport="mma_moneyline",
    target_column=MMA_MONEYLINE_TARGET,
    load_frame=load_mma_moneyline_frame,
    feature_columns=get_mma_moneyline_feature_columns,
    base_models=[
        ModelSpec("xgboost_mma_ml", XGBoostMatchPredictor, XGBOOST_MMA_MONEYLINE),
        ModelSpec("lightgbm_mma_ml", LightGBMMatchPredictor, LIGHTGBM_MMA_MONEYLINE),
        ModelSpec("neural_network_mma_ml", NeuralNetworkMatchPredictor, NEURAL_NETWORK_MMA_MONEYLINE),
    ],
    ensemble_config=ENSEMBLE_MMA_MONEYLINE,
    ensemble_name="ensemble_mma_ml",
)


# --sport value → SportBundle. Every key follows the
# <sport>_<market> convention.
SPORT_BUNDLES: Dict[str, SportBundle] = {
    "soccer_match_result": SOCCER_BUNDLE,
    "nhl_moneyline": NHL_MONEYLINE_BUNDLE,
    "nhl_regulation": NHL_REGULATION_BUNDLE,
    "nhl_puck_line": NHL_PUCK_LINE_BUNDLE,
    "nhl_total": NHL_TOTAL_BUNDLE,
    "nba_moneyline": NBA_MONEYLINE_BUNDLE,
    "nba_spread": NBA_SPREAD_BUNDLE,
    "nba_total": NBA_TOTAL_BUNDLE,
    "nfl_moneyline": NFL_MONEYLINE_BUNDLE,
    "nfl_spread": NFL_SPREAD_BUNDLE,
    "nfl_total": NFL_TOTAL_BUNDLE,
    "tennis_moneyline": TENNIS_MONEYLINE_BUNDLE,
    "mma_moneyline": MMA_MONEYLINE_BUNDLE,
}


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
        model_types: Optional[List[str]] = None,
        skip: Optional[List[str]] = None,
        bundle: Optional[SportBundle] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Train all individual models and then the ensemble.

        `bundle` selects which sport's model registry to iterate over
        (defaults to SOCCER_BUNDLE for backwards compatibility). `skip`
        is a list of model names to explicitly exclude even when
        `model_types=['all']` or `['ensemble']` would otherwise include them.
        """
        bundle = bundle or SOCCER_BUNDLE
        start = time.time()
        logger.info("Starting training for sport=%s...", bundle.sport)
        requested = set(model_types or ["all"])
        skip_set = set(skip or [])
        train_everything = "all" in requested
        train_for_ensemble = "ensemble" in requested

        def should_train(spec: ModelSpec) -> bool:
            # CLI --model-type aliases match the model TYPE (xgboost,
            # lightgbm, neural_network, poisson, dixon_coles), not the
            # sport-specific registry name. Otherwise --model-type
            # xgboost would only train the soccer config.
            type_alias = spec.config.model_type.value
            if spec.name in skip_set or type_alias in skip_set:
                logger.info("Skipping %s (in --skip-models)", spec.name)
                return False
            if train_everything or train_for_ensemble:
                return True
            return spec.name in requested or type_alias in requested

        team_cols_present = "home_team" in train_df.columns

        for spec in bundle.base_models:
            if not should_train(spec):
                continue
            if spec.needs_team_columns and not team_cols_present:
                logger.warning(
                    "Skipping %s because team columns are missing from the frame",
                    spec.name,
                )
                continue
            # Poisson-family models ignore `features` and read team
            # columns directly; pass None for those.
            features_for_model = None if spec.needs_team_columns else features
            self._train_model(
                spec.name,
                spec.cls,
                spec.config,
                train_df,
                val_df,
                features_for_model,
                target,
            )

        # Ensemble — needs ≥2 base models trained successfully.
        if (train_everything or train_for_ensemble) and len(self.trained_models) >= 2:
            self._train_ensemble(
                val_df,
                target,
                ensemble_config=bundle.ensemble_config,
                ensemble_name=bundle.ensemble_name,
                test_df=test_df,
            )

        duration = time.time() - start
        logger.info(f"All training for sport={bundle.sport} completed in {duration:.1f}s")

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
            result = model.train(train_df, val_df=val_df, features=features, target=target)

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

    def _train_ensemble(
        self,
        val_df: pd.DataFrame,
        target: str,
        ensemble_config: ModelConfig = ENSEMBLE_CONFIG,
        ensemble_name: str = "ensemble",
        test_df: Optional[pd.DataFrame] = None,
    ) -> None:
        logger.info("Training ensemble (%s)...", ensemble_name)

        ensemble = EnsemblePredictor(ensemble_config)
        for name, model in self.trained_models.items():
            ensemble.add_model(name, model)

        try:
            result = ensemble.train(val_df=val_df, target=target)

            # Fit the calibrator on val, then KEEP it only if it improves
            # the held-out TEST set (self-gating). Must run before register
            # so the saved artifact reflects the decision. Some sports
            # (e.g. soccer) are already well-calibrated and isotonic
            # overfits val — there the gate drops the calibrator and the
            # ensemble serves raw probabilities, exactly as before.
            if self.calibrate:
                self._fit_and_gate_calibrator(ensemble, val_df, test_df, target, ensemble_name, result)

            self.trained_models[ensemble_name] = ensemble
            self.results[ensemble_name] = result

            self.registry.register_model(
                model=ensemble,
                name=ensemble_name,
                version=ensemble_config.version,
                metrics=result.get("validation_metrics", {}),
            )
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}", exc_info=True)
            self.results[ensemble_name] = {"error": str(e)}

    def _holdout_metrics(
        self, ensemble: "EnsemblePredictor", df: pd.DataFrame, target: str
    ) -> Optional[Dict[str, Any]]:
        """Raw-vs-calibrated ECE/MCE/Brier/accuracy on `df`. Requires a
        FITTED calibrator on the ensemble. Returns None on shape mismatch."""
        from sklearn.preprocessing import LabelEncoder

        from training.calibration import brier_multiclass

        y = LabelEncoder().fit_transform(df[target].values)
        raw = ensemble.predict_proba(df, apply_calibration=False)
        if raw.ndim != 2 or raw.shape[1] != len(np.unique(y)):
            return None
        cal = ensemble.calibrator.calibrate(raw)
        cerr = ensemble.calibrator.calibration_error

        def _m(p: np.ndarray) -> Dict[str, float]:
            e = cerr(p, y)
            return {
                "ece": round(e["ece"], 5),
                "mce": round(e["mce"], 5),
                "brier": round(brier_multiclass(p, y), 5),
                "accuracy": round(float((np.argmax(p, axis=1) == y).mean()), 5),
            }

        return {"n": int(len(df)), "raw": _m(raw), "calibrated": _m(cal)}

    def _fit_and_gate_calibrator(
        self,
        ensemble: "EnsemblePredictor",
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        target: str,
        ensemble_name: str,
        result: Dict[str, Any],
    ) -> None:
        """Fit the isotonic calibrator on raw ensemble VAL output, then
        keep it only if it lowers Brier on the held-out TEST set. Encoding
        mirrors EnsemblePredictor.train (LabelEncoder.fit_transform) so
        calibrator class indices align with predict_proba columns. On any
        failure or a non-improving gate, the calibrator is dropped and the
        ensemble serves RAW probabilities (no regression vs before)."""
        from sklearn.preprocessing import LabelEncoder

        try:
            raw = ensemble.predict_proba(val_df, apply_calibration=False)
            y_enc = LabelEncoder().fit_transform(val_df[target].values)
            if raw.ndim != 2 or raw.shape[1] != len(np.unique(y_enc)):
                logger.warning(
                    "Skipping calibration for %s: %s proba cols vs %d classes",
                    ensemble_name,
                    raw.shape,
                    len(np.unique(y_enc)),
                )
                return

            ensemble.fit_calibrator(raw, y_enc)

            decision: Dict[str, Any] = {"kept": True, "reason": "no held-out test set to gate on"}
            if test_df is not None and not test_df.empty and target in test_df.columns:
                metrics = self._holdout_metrics(ensemble, test_df, target)
                if metrics is not None:
                    improves = metrics["calibrated"]["brier"] < metrics["raw"]["brier"] - 1e-4
                    decision = {
                        "kept": improves,
                        "reason": (
                            "calibration lowers held-out Brier"
                            if improves
                            else "calibration does NOT improve held-out Brier — serving raw"
                        ),
                        **metrics,
                    }

            if not decision["kept"]:
                ensemble.calibrator = None  # serve raw

            result["calibration"] = decision
            logger.info(
                "Ensemble %s calibration gate: kept=%s (%s)",
                ensemble_name,
                decision["kept"],
                decision["reason"],
            )
        except Exception as e:
            logger.warning("Calibrator fit/gate failed for %s: %s", ensemble_name, e)
            ensemble.calibrator = None

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sport",
        default=os.getenv("TRAIN_SPORT", "soccer_match_result"),
        choices=sorted(SPORT_BUNDLES.keys()),
        help="Which sport/task bundle to train. Picks the data loader, "
        "feature selector, model list, and ensemble config.",
    )
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--input-csv")
    parser.add_argument(
        "--model-type",
        default=os.getenv("MODEL_TYPE", "all"),
        choices=["all", "xgboost", "lightgbm", "neural_network", "poisson", "dixon_coles", "ensemble"],
    )
    parser.add_argument(
        "--skip-models",
        default=os.getenv("SKIP_MODELS", ""),
        help=(
            "Comma-separated list of model names (or type aliases) to skip when "
            "--model-type=all (or ensemble). Useful for slow models — e.g. "
            "'poisson,dixon_coles' to bypass the per-team MLE fits."
        ),
    )
    parser.add_argument("--output-dir", default="./models")
    parser.add_argument(
        "--target",
        default=None,
        help="Target column. Defaults to the bundle's target (match_outcome "
        "for soccer, nhl_moneyline for nhl_moneyline).",
    )
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--min-feature-count", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Convert XGBoost/LightGBM/Neural-Network models to ONNX after training. "
        "ONNX inference is 10-50x faster than the Python equivalents.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    bundle = SPORT_BUNDLES[args.sport]
    target = args.target or bundle.target_column

    try:
        frame = bundle.load_frame(database_url=args.database_url, input_csv=args.input_csv)
        quality = validate_training_frame(
            frame,
            target=target,
            min_samples=args.min_samples,
            min_feature_count=args.min_feature_count,
        )
        train_df, val_df, test_df = _split_temporally(frame, args.train_ratio, args.val_ratio)
        features = bundle.feature_columns(frame, target)
        orchestrator = TrainingOrchestrator(
            registry_dir=args.output_dir,
            calibrate=not args.no_calibration,
        )
        skip_list = [s.strip() for s in (args.skip_models or "").split(",") if s.strip()]
        results = orchestrator.train_all(
            train_df=train_df,
            val_df=val_df,
            features=features,
            target=target,
            model_types=[args.model_type],
            skip=skip_list,
            bundle=bundle,
            test_df=test_df,
        )
    except Exception as exc:
        logger.error("Model training failed: %s", exc, exc_info=True)
        return 1

    # The ensemble calibrator was fit on val and self-gated on the held-out
    # test (kept only where it lowers test Brier). Surface that decision +
    # the raw-vs-calibrated numbers in the report.
    holdout = results.get(bundle.ensemble_name, {}).get("calibration")
    if holdout:
        logger.info("CALIBRATION GATE (%s): %s", bundle.ensemble_name, json.dumps(holdout))

    onnx_paths: Dict[str, str] = {}
    if args.export_onnx:
        onnx_paths = _export_onnx(
            orchestrator.trained_models,
            features=features,
            output_dir=Path(args.output_dir) / "onnx",
        )

    report = {
        "data_quality": quality.to_dict(),
        "trained_models": sorted(orchestrator.trained_models.keys()),
        "best_model": orchestrator.get_best_model(),
        "results": results,
        "onnx_models": onnx_paths,
        "holdout_test": holdout,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default),
        encoding="utf-8",
    )

    errored = {name: result["error"] for name, result in results.items() if "error" in result}
    if errored:
        logger.error("One or more requested models failed: %s", errored)
        return 1

    if not orchestrator.trained_models:
        logger.error("No models were trained")
        return 1

    print(json.dumps(report, indent=2, default=_json_default))
    return 0


def _export_onnx(
    trained_models: Dict[str, BaseModel],
    features: List[str],
    output_dir: Path,
) -> Dict[str, str]:
    """Best-effort ONNX export. Skips models whose backend isn't supported.

    Failures are logged but don't fail the training run — ONNX is a
    nice-to-have for latency, not a correctness requirement.
    """
    converter = ONNXConverter(output_dir=str(output_dir))
    paths: Dict[str, str] = {}
    for name, model in trained_models.items():
        try:
            backend = getattr(model, "model", None)
            if backend is None:
                continue
            module = type(backend).__module__
            if module.startswith("xgboost"):
                p = converter.convert_xgboost(backend, features, name)
            elif module.startswith("lightgbm"):
                p = converter.convert_lightgbm(backend, features, name)
            elif "torch" in module:
                p = converter.convert_pytorch(backend, len(features), name)
            else:
                logger.info("ONNX export skipped for %s (backend=%s)", name, module)
                continue
            if p:
                paths[name] = p
        except Exception as exc:
            logger.warning("ONNX export failed for %s: %s", name, exc)
    return paths


def _split_temporally(
    frame: pd.DataFrame, train_ratio: float, val_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal train / val / test split. The tail after
    (train_ratio + val_ratio) is the HELD-OUT TEST set — previously
    discarded — used for honest raw-vs-calibrated metrics on data that
    neither the base models, the ensemble weights, nor the calibrator
    ever saw. Empty when the ratios sum to 1.0 (caller tolerates that)."""
    ordered = frame.sort_values("match_date").reset_index(drop=True)
    train_end = int(len(ordered) * train_ratio)
    val_end = int(len(ordered) * (train_ratio + val_ratio))
    train_df = ordered.iloc[:train_end].copy()
    val_df = ordered.iloc[train_end:val_end].copy()
    test_df = ordered.iloc[val_end:].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Training and validation splits must both contain rows")
    return train_df, val_df, test_df


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
