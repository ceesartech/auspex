"""Backblaze B2 (S3-compatible) helpers shared by the Modal training path.

Mirrors scripts/backup_postgres.py's boto3 setup EXACTLY — same region
derivation from the endpoint host (the #1 B2 footgun), the same boto3>=1.36
checksum-header safeguards, and HEAD-verify-after-write. Used by:

  - modal_train/train_modal.py     : download_latest(dump) + upload_tree(artifacts)
  - scripts/pull_modal_artifacts.py: download_prefix(artifacts) back onto the VM

Env (identical to the backup script): BACKUP_S3_BUCKET, BACKUP_S3_ENDPOINT_URL,
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY. The signing region is DERIVED from the
endpoint host, so the operator only sets the endpoint (copied from the B2 bucket
page) — no AWS_DEFAULT_REGION needed for B2.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("b2_io")


def _derive_region(endpoint_url: Optional[str]) -> Optional[str]:
    """s3.us-west-004.backblazeb2.com → us-west-004. Returns None for real AWS."""
    if not endpoint_url:
        return None
    m = re.search(r"s3[.\-]([a-z]{2}-[a-z]+-\d+)\.", endpoint_url)
    return m.group(1) if m else None


def s3_client():
    """boto3 S3 client configured for B2, identical to backup_postgres.py."""
    import boto3

    endpoint = os.environ.get("BACKUP_S3_ENDPOINT_URL")
    kwargs: dict = {}
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    region = _derive_region(endpoint) or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    if region:
        kwargs["region_name"] = region
    # boto3 >= 1.36 defaults request checksums ON, which broke B2. We pin
    # 1.35.50, but set the env safeguard too (no-op on 1.35.x) so a future bump
    # doesn't silently break B2.
    os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
    os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")
    return boto3.client("s3", **kwargs)


def bucket() -> str:
    b = os.environ.get("BACKUP_S3_BUCKET")
    if not b:
        raise RuntimeError("BACKUP_S3_BUCKET not set — cannot reach B2.")
    return b


def _list(s3, prefix: str) -> list[dict]:
    """All non-'directory' objects under a prefix (paginated)."""
    out: list[dict] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket(), Prefix=prefix):
        out.extend(o for o in page.get("Contents", []) if not o["Key"].endswith("/"))
    return out


def download_latest(
    prefix: str, dest: Path, *, max_age_hours: float = 30.0, name_contains: Optional[str] = None
) -> dict:
    """Download the most-recent object under `prefix` to `dest`, then verify
    size + freshness. Guards (design risk 'B2 dump freshness / wrong-object'):
    the newest object must be < max_age_hours old, and if name_contains is set
    the key must contain it (so a stale or wrong-DB dump is caught loudly)."""
    s3 = s3_client()
    objs = _list(s3, prefix)
    if name_contains:
        objs = [o for o in objs if name_contains in o["Key"]]
    if not objs:
        raise RuntimeError(
            f"No objects under s3://{bucket()}/{prefix}" + (f" matching {name_contains!r}" if name_contains else "")
        )
    latest = max(objs, key=lambda o: o["LastModified"])
    age = datetime.now(timezone.utc) - latest["LastModified"]
    if age > timedelta(hours=max_age_hours):
        raise RuntimeError(
            f"Latest object s3://{bucket()}/{latest['Key']} is {age} old "
            f"(> {max_age_hours}h) — refusing to train on a stale dump."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading s3://%s/%s (%d bytes) → %s", bucket(), latest["Key"], latest["Size"], dest)
    s3.download_file(bucket(), latest["Key"], str(dest))
    local = dest.stat().st_size
    if local != latest["Size"]:
        raise RuntimeError(f"Download size mismatch for {latest['Key']}: local={local} remote={latest['Size']}")
    return {"key": latest["Key"], "size": latest["Size"], "last_modified": latest["LastModified"].isoformat()}


def upload_tree(local_dir: Path, key_prefix: str) -> int:
    """Upload every file under local_dir to <key_prefix>/<relpath>, HEAD-verifying
    each (size match). Returns the file count. key_prefix should end without a
    trailing slash; it is joined with '/'."""
    s3 = s3_client()
    bkt = bucket()
    key_prefix = key_prefix.rstrip("/")
    n = 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = f"{key_prefix}/{rel}"
        s3.upload_file(str(path), bkt, key)
        remote = s3.head_object(Bucket=bkt, Key=key).get("ContentLength")
        if remote != path.stat().st_size:
            raise RuntimeError(f"Upload size mismatch for {key}: local={path.stat().st_size} remote={remote}")
        n += 1
    logger.info("Uploaded %d files → s3://%s/%s/", n, bkt, key_prefix)
    return n


def download_prefix(key_prefix: str, dest_dir: Path) -> int:
    """Download every object under key_prefix into dest_dir, preserving the
    relative layout, HEAD/size-verifying each (the backup script only verifies
    on upload; we verify on download too, closing that gap). Returns file count."""
    s3 = s3_client()
    bkt = bucket()
    key_prefix = key_prefix.rstrip("/") + "/"
    objs = _list(s3, key_prefix)
    if not objs:
        raise RuntimeError(f"No objects under s3://{bkt}/{key_prefix}")
    n = 0
    for o in objs:
        rel = o["Key"][len(key_prefix) :]
        local = dest_dir / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bkt, o["Key"], str(local))
        if local.stat().st_size != o["Size"]:
            raise RuntimeError(
                f"Download size mismatch for {o['Key']}: local={local.stat().st_size} remote={o['Size']}"
            )
        n += 1
    logger.info("Downloaded %d files from s3://%s/%s → %s", n, bkt, key_prefix, dest_dir)
    return n
