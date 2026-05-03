"""ONNX model conversion for optimized inference (<100ms)."""

import logging
import time
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class ONNXConverter:
    """Convert trained models to ONNX format for fast inference."""

    def __init__(self, output_dir: str = "onnx_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_xgboost(self, model, feature_names: list, output_name: str) -> str:
        """Convert XGBoost Booster to ONNX."""
        try:
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType

            initial_type = [("features", FloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_xgboost(model, initial_types=initial_type)

            output_path = self.output_dir / f"{output_name}.onnx"
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"XGBoost model converted to ONNX: {output_path}")
            return str(output_path)
        except ImportError:
            logger.warning("onnxmltools not installed, skipping ONNX conversion")
            return ""

    def convert_lightgbm(self, model, feature_names: list, output_name: str) -> str:
        """Convert LightGBM model to ONNX."""
        try:
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType

            initial_type = [("features", FloatTensorType([None, len(feature_names)]))]
            onnx_model = convert_lightgbm(model, initial_types=initial_type)

            output_path = self.output_dir / f"{output_name}.onnx"
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"LightGBM model converted to ONNX: {output_path}")
            return str(output_path)
        except ImportError:
            logger.warning("onnxmltools not installed, skipping ONNX conversion")
            return ""

    def convert_pytorch(self, model, input_size: int, output_name: str) -> str:
        """Convert PyTorch model to ONNX."""
        try:
            import torch

            model.eval()
            dummy_input = torch.randn(1, input_size)
            output_path = self.output_dir / f"{output_name}.onnx"

            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                input_names=["features"],
                output_names=["probabilities"],
                dynamic_axes={
                    "features": {0: "batch_size"},
                    "probabilities": {0: "batch_size"},
                },
                opset_version=13,
            )

            logger.info(f"PyTorch model converted to ONNX: {output_path}")
            return str(output_path)
        except ImportError:
            logger.warning("torch not available for ONNX export")
            return ""

    def benchmark(self, onnx_path: str, input_data: np.ndarray, n_runs: int = 100) -> Dict[str, float]:
        """Benchmark ONNX inference latency."""
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name

            # Warmup
            for _ in range(10):
                session.run(None, {input_name: input_data.astype(np.float32)})

            # Benchmark
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                session.run(None, {input_name: input_data.astype(np.float32)})
                times.append((time.perf_counter() - start) * 1000)  # ms

            return {
                "mean_ms": float(np.mean(times)),
                "p50_ms": float(np.percentile(times, 50)),
                "p95_ms": float(np.percentile(times, 95)),
                "p99_ms": float(np.percentile(times, 99)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
            }
        except ImportError:
            logger.warning("onnxruntime not installed, cannot benchmark")
            return {}
