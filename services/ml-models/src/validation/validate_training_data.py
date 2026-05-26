"""Validate that training data is present and suitable for retraining."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from utils.training_data import load_training_frame, validate_training_frame

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--input-csv")
    parser.add_argument("--target", default="match_outcome")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--min-feature-count", type=int, default=5)
    parser.add_argument("--output-file")
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
    except Exception as exc:
        logger.error("Training data validation failed: %s", exc)
        return 1

    report = {"quality": "passed", **quality.to_dict()}
    if args.output_file:
        Path(args.output_file).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
