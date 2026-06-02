"""Drain the shared Telegram alert queue and send ONE combined digest.

The DAG runs this AFTER both soccer (`precompute_predictions`) and NHL
(`precompute_predictions_nhl`) finish enqueueing their high-confidence
picks. The pop+delete is atomic so a concurrent enqueue can't race in
between the LRANGE and DEL — those late picks just stay in the queue
and ride the next digest.

Why a separate task instead of having one of the precompute scripts
"finalize" the digest? Two reasons:
  1. Either precompute branch could fail mid-run. If the failing
     branch is also the one supposed to send, we'd lose the other
     branch's queued picks. A separate task guarded with the right
     Airflow trigger_rule keeps that simple.
  2. The shape mirrors the standard "fan-in" pattern any future sport
     extension (NFL, MLB, …) just needs to enqueue; the digest task
     is sport-agnostic.

Usage (inside the api container):
    python /app/scripts/send_pipeline_digest.py
    python /app/scripts/send_pipeline_digest.py --header "Custom title"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Make the shared telegram_notify importable.
sys.path.insert(0, os.path.dirname(__file__))

from telegram_notify import drain_and_send_digest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("send_pipeline_digest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--header",
        default=None,
        help="Optional digest header. Defaults to 'Auspex picks · N high-confidence'.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = drain_and_send_digest(header=args.header)
    logger.info(
        "Drained %d picks; sent %d Telegram message(s)",
        result.get("drained", 0),
        result.get("sent", 0),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
