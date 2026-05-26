"""Evaluate registered model artifacts from a retraining run."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--output-file", default="./evaluation_report.json")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--min-metric", type=float)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        report = build_evaluation_report(Path(args.model_dir), metric=args.metric, min_metric=args.min_metric)
    except Exception as exc:
        logger.error("Model evaluation failed: %s", exc)
        return 1

    Path(args.output_file).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def build_evaluation_report(
    model_dir: Path,
    *,
    metric: str = "accuracy",
    min_metric: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a report from the file-backed model registry index."""
    index_path = model_dir / "registry_index.json"
    if not index_path.exists():
        raise ValueError(f"Model registry index not found at {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    models = _flatten_registry(index)
    if not models:
        raise ValueError("Model registry contains no model versions")

    metric_candidates = [model for model in models if isinstance(model["metrics"].get(metric), (int, float))]
    if not metric_candidates:
        raise ValueError(f"No registered model has metric '{metric}'")

    best_model = max(metric_candidates, key=lambda item: item["metrics"][metric])
    passed = min_metric is None or best_model["metrics"][metric] >= min_metric

    return {
        "passed": passed,
        "metric": metric,
        "minimum_required": min_metric,
        "best_model": best_model,
        "models": models,
    }


def _flatten_registry(index: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    for name, versions in index.items():
        for version in versions:
            models.append(
                {
                    "name": name,
                    "version": version.get("version"),
                    "model_type": version.get("model_type"),
                    "metrics": version.get("metrics", {}),
                    "created_at": version.get("created_at"),
                    "is_active": version.get("is_active", False),
                    "path": version.get("path"),
                }
            )
    return models


if __name__ == "__main__":
    sys.exit(main())
