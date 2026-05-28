"""Training orchestrator for model retraining."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from inference.onnx_converter import ONNXConverter
from predictors.base_model import BaseModel
from predictors.ensemble import EnsemblePredictor
from predictors.lightgbm_model import LightGBMMatchPredictor
from predictors.model_config import (
    DIXON_COLES_CONFIG,
    ENSEMBLE_CONFIG,
    LIGHTGBM_MATCH_OUTCOME,
    NEURAL_NETWORK_CONFIG,
    POISSON_CONFIG,
    XGBOOST_MATCH_OUTCOME,
    ModelConfig,
)
from predictors.model_registry import ModelRegistry
from predictors.neural_network import NeuralNetworkMatchPredictor
from predictors.poisson_models import DixonColesPredictor, PoissonMatchPredictor
from predictors.xgboost_model import XGBoostMatchPredictor
from training.calibration import ProbabilityCalibrator
from utils.training_data import get_feature_columns, load_training_frame, validate_training_frame

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
        model_types: Optional[List[str]] = None,
        skip: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Train all individual models and then the ensemble.

        `skip` is a list of model names to explicitly exclude even when
        `model_types=['all']` or `['ensemble']` would otherwise include them.
        """
        start = time.time()
        logger.info("Starting training of all models...")
        requested = set(model_types or ["all"])
        skip_set = set(skip or [])
        train_everything = "all" in requested
        train_for_ensemble = "ensemble" in requested

        def should_train(name: str) -> bool:
            if name in skip_set:
                logger.info("Skipping %s (in --skip-models)", name)
                return False
            return train_everything or train_for_ensemble or name in requested

        # 1. Train XGBoost
        if should_train("xgboost"):
            self._train_model(
                "xgboost",
                XGBoostMatchPredictor,
                XGBOOST_MATCH_OUTCOME,
                train_df,
                val_df,
                features,
                target,
            )

        # 2. Train LightGBM
        if should_train("lightgbm"):
            self._train_model(
                "lightgbm",
                LightGBMMatchPredictor,
                LIGHTGBM_MATCH_OUTCOME,
                train_df,
                val_df,
                features,
                target,
            )

        # 3. Train Neural Network
        if should_train("neural_network"):
            self._train_model(
                "neural_network",
                NeuralNetworkMatchPredictor,
                NEURAL_NETWORK_CONFIG,
                train_df,
                val_df,
                features,
                target,
            )

        # 4. Train Poisson (needs team columns)
        if "home_team" in train_df.columns:
            if should_train("poisson"):
                self._train_model(
                    "poisson",
                    PoissonMatchPredictor,
                    POISSON_CONFIG,
                    train_df,
                    val_df,
                    None,
                    target,
                )

            # 5. Train Dixon-Coles
            if should_train("dixon_coles"):
                self._train_model(
                    "dixon_coles",
                    DixonColesPredictor,
                    DIXON_COLES_CONFIG,
                    train_df,
                    val_df,
                    None,
                    target,
                )
        elif should_train("poisson") or should_train("dixon_coles"):
            logger.warning("Skipping Poisson-family models because team columns are missing")

        # 6. Train Ensemble
        if (train_everything or "ensemble" in requested) and len(self.trained_models) >= 2:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
            "Comma-separated list of model names to skip when --model-type=all "
            "(or ensemble). Useful for slow models — e.g. 'poisson,dixon_coles' "
            "to bypass the per-team MLE fits that can take 30+ minutes."
        ),
    )
    parser.add_argument("--output-dir", default="./models")
    parser.add_argument("--target", default="match_outcome")
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

    try:
        frame = load_training_frame(database_url=args.database_url, input_csv=args.input_csv)
        quality = validate_training_frame(
            frame,
            target=args.target,
            min_samples=args.min_samples,
            min_feature_count=args.min_feature_count,
        )
        train_df, val_df = _split_temporally(frame, args.train_ratio, args.val_ratio)
        features = get_feature_columns(frame, target=args.target)
        orchestrator = TrainingOrchestrator(
            registry_dir=args.output_dir,
            calibrate=not args.no_calibration,
        )
        skip_list = [s.strip() for s in (args.skip_models or "").split(",") if s.strip()]
        results = orchestrator.train_all(
            train_df=train_df,
            val_df=val_df,
            features=features,
            target=args.target,
            model_types=[args.model_type],
            skip=skip_list,
        )
    except Exception as exc:
        logger.error("Model training failed: %s", exc, exc_info=True)
        return 1

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


def _split_temporally(frame: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("match_date").reset_index(drop=True)
    train_end = int(len(ordered) * train_ratio)
    val_end = int(len(ordered) * (train_ratio + val_ratio))
    train_df = ordered.iloc[:train_end].copy()
    val_df = ordered.iloc[train_end:val_end].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Training and validation splits must both contain rows")
    return train_df, val_df


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
