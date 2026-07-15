"""Prune old Modal training artifacts from B2.

Every Modal retrain leaves a per-run tree under ``modal-train/<run_id>/`` in the
B2 bucket (weekly runs + any smoke/shadow/test runs). They're consumed by the
VM pull within minutes of a run; after that they're just history. This deletes
``modal-train/`` objects older than ``--days``, permanently (all versions +
delete markers), so they don't accumulate.

Why a direct prune and not an S3 lifecycle rule: B2 buckets are versioned and
its S3 lifecycle API rejects a plain Expiration.Days rule ("no
ExpiredObjectDeleteMarker rule with the exact same prefix"). Listing + deleting
versions is reliable and reclaims storage immediately.

SAFE: only touches the ``modal-train/`` prefix (intermediate artifacts) — never
``postgres/`` (backups) or the local production models. Runs weekly via the
docker_maintenance DAG; also runnable by hand:

    docker compose exec -T api python /app/scripts/prune_modal_artifacts.py --days 14
    docker compose exec -T api python /app/scripts/prune_modal_artifacts.py --days 14 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))

import b2_io  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("prune_modal_artifacts")

PREFIX = "modal-train/"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=14, help="Delete modal-train/ objects older than this (default 14).")
    p.add_argument("--dry-run", action="store_true", help="List what would be deleted without deleting.")
    args = p.parse_args(argv)

    if PREFIX in ("", "/"):  # guard: never sweep the whole bucket
        raise SystemExit("refusing to prune an empty prefix")

    s3 = b2_io.s3_client()
    bkt = b2_io.bucket()
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    # Enumerate ALL versions + delete markers under the prefix (B2 buckets are
    # versioned), collect the ones older than the cutoff.
    stale: list[dict] = []
    runs_touched: set[str] = set()
    for page in s3.get_paginator("list_object_versions").paginate(Bucket=bkt, Prefix=PREFIX):
        for v in page.get("Versions", []) + page.get("DeleteMarkers", []):
            if v["LastModified"] < cutoff:
                stale.append({"Key": v["Key"], "VersionId": v["VersionId"]})
                parts = v["Key"][len(PREFIX) :].split("/", 1)
                if parts and parts[0]:
                    runs_touched.add(parts[0])

    logger.info(
        "%s %d object-versions under %s older than %dd (runs: %s)",
        "would delete" if args.dry_run else "deleting",
        len(stale),
        PREFIX,
        args.days,
        ", ".join(sorted(runs_touched)) or "none",
    )
    if args.dry_run or not stale:
        return 0

    deleted = 0
    for i in range(0, len(stale), 1000):  # delete_objects caps at 1000/call
        resp = s3.delete_objects(Bucket=bkt, Delete={"Objects": stale[i : i + 1000], "Quiet": True})
        errs = resp.get("Errors", [])
        if errs:
            logger.warning("delete errors (%d), first: %s", len(errs), errs[0])
        deleted += len(stale[i : i + 1000]) - len(errs)
    logger.info("Deleted %d/%d object-versions from %s", deleted, len(stale), PREFIX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
