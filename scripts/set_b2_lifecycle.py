"""Set a B2 lifecycle rule to auto-expire Modal training artifacts.

Every Modal retrain leaves a per-run tree under ``modal-train/<run_id>/`` in the
B2 bucket (the weekly runs + any smoke/shadow/schedtest runs). Without expiry
they accumulate forever. This sets an S3 lifecycle rule that deletes
``modal-train/`` objects after ``--days``, while PRESERVING any existing bucket
rules (e.g. the backups retention rule the backup script relies on).

Idempotent — safe to re-run. Run inside the api container (it has the B2 creds):

    docker compose exec -T api python /app/scripts/set_b2_lifecycle.py --days 14
    docker compose exec -T api python /app/scripts/set_b2_lifecycle.py --days 14 --dry-run

If B2 rejects the S3 lifecycle call (support varies), set the rule from the
Backblaze console instead: Bucket → Lifecycle Settings → custom rule, file prefix
``modal-train/``, "Keep only the last version … for N days". See OPERATIONS.md.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import b2_io  # noqa: E402

RULE_ID = "expire-modal-train"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=14, help="Delete modal-train/ objects older than this (default 14).")
    p.add_argument("--prefix", default="modal-train/", help="Object prefix to expire (default modal-train/).")
    p.add_argument("--dry-run", action="store_true", help="Show what would be set without applying.")
    args = p.parse_args(argv)

    s3 = b2_io.s3_client()
    bkt = b2_io.bucket()

    # Preserve existing rules; drop any prior copy of ours so re-runs are clean.
    existing = []
    try:
        cfg = s3.get_bucket_lifecycle_configuration(Bucket=bkt)
        existing = [r for r in cfg.get("Rules", []) if r.get("ID") != RULE_ID]
    except Exception as exc:  # noqa: BLE001
        if "NoSuchLifecycleConfiguration" not in str(exc):
            print(f"note: could not read existing lifecycle ({exc}); starting fresh")

    # B2 buckets are versioned: Expiration.Days HIDES the object (B2's
    # daysFromUploadingToHiding); the hidden version must also be deleted, or B2
    # rejects with "no ExpiredObjectDeleteMarker rule". NoncurrentVersionExpiration
    # maps to B2's daysFromHidingToDeleting — so the object is gone ~N+1 days out.
    rule = {
        "ID": RULE_ID,
        "Filter": {"Prefix": args.prefix},
        "Status": "Enabled",
        "Expiration": {"Days": args.days},
        "NoncurrentVersionExpiration": {"NoncurrentDays": 1},
    }
    rules = existing + [rule]
    print(f"bucket={bkt} | preserving {len(existing)} existing rule(s), " f"expiring {args.prefix} after {args.days}d")
    if args.dry_run:
        print("dry-run: not applied")
        return 0

    try:
        s3.put_bucket_lifecycle_configuration(Bucket=bkt, LifecycleConfiguration={"Rules": rules})
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to set lifecycle via S3 API: {exc}")
        print("Set it from the Backblaze console instead (see this file's docstring / OPERATIONS.md).")
        return 1

    print("applied. Current rules:")
    for r in s3.get_bucket_lifecycle_configuration(Bucket=bkt).get("Rules", []):
        print("  -", r.get("ID"), "|", r.get("Filter", {}).get("Prefix"), "|", r.get("Expiration"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
