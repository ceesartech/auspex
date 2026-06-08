"""Daily PostgreSQL backup → local disk + optional S3 upload.

Two-tier backup strategy:
  * Local (always): writes a compressed `pg_dump` to /app/backups/
    inside the api container, which is bind-mounted to
    /opt/auspex/backups/ on the host. Keeps the last
    --local-retention-days files for quick restore (default 7).
  * Remote (optional): if BACKUP_S3_BUCKET env var is set, also
    uploads the dump to S3-compatible storage. Backblaze B2 is the
    recommended target (set BACKUP_S3_ENDPOINT_URL to the B2 endpoint;
    the SigV4 signing region is auto-derived from it). The upload is
    HEAD-verified (size match) and a failed offsite upload exits the
    process non-zero so the Airflow DAG turns red — a broken remote
    backup never hides behind a green run. Long-term retention is a
    B2/S3 lifecycle rule on the bucket side. See OPERATIONS.md for
    the full B2 setup + DR restore procedure.

Dump format: pg_dump custom format (-Fc), which is compressed by
default and supports parallel restore via pg_restore -j. Single
.dump file per day, named with ISO timestamp:
    auspex-prod-2026-06-07T020000Z.dump

Restore (full DR, see OPERATIONS.md for details):
    pg_restore -h HOST -U USER -d auspex_prod --clean --if-exists \\
        auspex-prod-YYYY-MM-DDTHHMMSSZ.dump

Env vars consumed:
    DATABASE_URL              — postgres connection string
    BACKUP_LOCAL_DIR          — local dump directory (default /app/backups)
    BACKUP_LOCAL_RETENTION    — days of local backups to keep (default 7)
    BACKUP_S3_BUCKET          — optional; enables S3/B2 upload
    BACKUP_S3_PREFIX          — optional; key prefix (default "postgres/")
    BACKUP_S3_ENDPOINT_URL    — optional; B2/Hetzner/Wasabi endpoint
                                (region auto-derived from the host)
    AWS_ACCESS_KEY_ID         — B2 keyID (or AWS access key)
    AWS_SECRET_ACCESS_KEY     — B2 applicationKey (or AWS secret)
    AWS_DEFAULT_REGION        — only used as a fallback when the region
                                can't be derived from the endpoint (i.e.
                                real AWS S3); not needed for B2

Run on prod via:
    docker compose exec api python /app/scripts/backup_postgres.py

Wired into Airflow as daily 02:00 UTC by services/data-ingestion/
dags/db_backup.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backup_postgres")


def parse_database_url(url: str) -> dict:
    """Parse postgresql://user:pass@host:port/dbname into pg_dump args."""
    p = urlparse(url)
    if p.scheme not in ("postgresql", "postgres"):
        raise ValueError(f"Expected postgres URL, got scheme {p.scheme!r}")
    return {
        "host": p.hostname or "localhost",
        "port": str(p.port or 5432),
        "user": p.username or "postgres",
        "password": p.password or "",
        "dbname": (p.path or "/").lstrip("/") or "postgres",
    }


def run_pg_dump(database_url: str, output_path: Path) -> None:
    """Invoke pg_dump via subprocess. Custom format (-Fc) is
    compressed by default and supports parallel pg_restore -j."""
    parts = parse_database_url(database_url)
    cmd = [
        "pg_dump",
        "-h",
        parts["host"],
        "-p",
        parts["port"],
        "-U",
        parts["user"],
        "-d",
        parts["dbname"],
        "-F",
        "c",  # custom format (compressed)
        "-Z",
        "9",  # max compression
        "-f",
        str(output_path),
        "--verbose",
    ]
    env = os.environ.copy()
    if parts["password"]:
        env["PGPASSWORD"] = parts["password"]
    logger.info("Running pg_dump → %s", output_path)
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # pg_dump writes its progress to stderr. Show the last few
        # lines on failure so the cause is visible in the log.
        tail = "\n".join(proc.stderr.splitlines()[-20:])
        raise RuntimeError(f"pg_dump exited {proc.returncode}:\n{tail}")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("pg_dump complete: %.1f MB", size_mb)


def _derive_region(endpoint_url: Optional[str]) -> Optional[str]:
    """Derive the SigV4 signing region from an S3-compatible endpoint
    host when it embeds the region (Backblaze B2 + most providers do:
    ``s3.<region>.backblazeb2.com``). This eliminates the #1 B2 setup
    footgun: B2 requires the boto3 region to MATCH the region token in
    the endpoint host, and a wrong/ambient region (e.g. the compose
    default ``us-east-1``) yields ``SignatureDoesNotMatch``. By deriving
    from the endpoint, the operator only has to set the endpoint URL
    correctly (copied verbatim from the B2 bucket page) — the region
    follows automatically. Returns None when the pattern doesn't match
    (e.g. real AWS S3, where region comes from AWS_DEFAULT_REGION)."""
    if not endpoint_url:
        return None
    # B2: s3.us-west-004.backblazeb2.com → us-west-004.
    # Generic S3-compatible: s3.<region>.<provider>.com → <region>.
    m = re.search(r"s3[.\-]([a-z]{2}-[a-z]+-\d+)\.", endpoint_url)
    return m.group(1) if m else None


def upload_to_s3(local_path: Path, bucket: str, key: str, endpoint_url: Optional[str]) -> None:
    """Upload via boto3, then HEAD the object to confirm it landed.
    Imports lazy so backups still run when boto3 isn't installed
    (local-only mode). Raises on any failure so the caller can flag a
    broken offsite backup."""
    try:
        import boto3
    except ImportError:
        logger.error(
            "boto3 not installed — install with `pip install boto3` "
            "to enable S3 uploads, or unset BACKUP_S3_BUCKET to skip."
        )
        raise

    client_kwargs: dict = {}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
        logger.info("Using custom S3 endpoint: %s", endpoint_url)

    # Region resolution order: derive from the endpoint host (most
    # robust for B2 — guarantees signing region == endpoint region),
    # else fall back to AWS_DEFAULT_REGION / AWS_REGION. Passing it
    # explicitly avoids NoRegionError when ambient AWS config is absent.
    region = _derive_region(endpoint_url) or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    if region:
        client_kwargs["region_name"] = region
        logger.info("S3 signing region: %s", region)

    # boto3 >= 1.36 (botocore Jan 2025) defaults request checksums ON,
    # which historically broke B2 with "Unsupported header
    # 'x-amz-sdk-checksum-algorithm'". We pin boto3==1.35.50 which
    # predates that change, but set the env-var safeguard too (read
    # only by botocore >= 1.36; a harmless no-op on 1.35.x) so a
    # future boto3 bump doesn't silently break B2 uploads. Done via
    # env rather than botocore Config kwargs because those kwargs
    # don't exist on 1.35.x and would raise TypeError.
    os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
    os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")

    s3 = boto3.client("s3", **client_kwargs)
    logger.info("Uploading → s3://%s/%s", bucket, key)
    s3.upload_file(str(local_path), bucket, key)

    # Verify the object actually landed (size match) before declaring
    # success — turns a silently-partial upload into a loud failure.
    head = s3.head_object(Bucket=bucket, Key=key)
    remote_size = head.get("ContentLength")
    local_size = local_path.stat().st_size
    if remote_size != local_size:
        raise RuntimeError(f"S3 object size mismatch: local={local_size} remote={remote_size}")
    logger.info("Upload verified: %d bytes in s3://%s/%s", remote_size, bucket, key)


def rotate_local_backups(local_dir: Path, retention_days: int) -> int:
    """Delete .dump files older than retention_days. Returns count
    of files deleted."""
    if retention_days <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    deleted = 0
    for f in local_dir.glob("*.dump"):
        if datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc) < cutoff:
            logger.info("Rotating out old backup: %s", f.name)
            f.unlink()
            deleted += 1
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--local-dir",
        default=os.environ.get("BACKUP_LOCAL_DIR", "/app/backups"),
        help="Local directory for dumps. Bind-mounted to host /opt/auspex/backups.",
    )
    parser.add_argument(
        "--local-retention-days",
        type=int,
        default=int(os.environ.get("BACKUP_LOCAL_RETENTION", "7")),
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.environ.get("BACKUP_S3_BUCKET"),
        help="Optional S3 bucket for remote backup. Unset = local-only.",
    )
    parser.add_argument(
        "--s3-prefix",
        default=os.environ.get("BACKUP_S3_PREFIX", "postgres/"),
    )
    parser.add_argument(
        "--s3-endpoint-url",
        default=os.environ.get("BACKUP_S3_ENDPOINT_URL"),
        help="For non-AWS S3 providers (B2, Hetzner, etc).",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Local backup only; skip S3 upload even if configured.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if not args.database_url:
        logger.error("DATABASE_URL not set.")
        return 1

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Filename includes UTC timestamp + DB name from URL — makes
    # multi-DB setups distinguish naturally.
    db_name = parse_database_url(args.database_url)["dbname"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    filename = f"{db_name}-{ts}.dump"
    local_path = local_dir / filename

    try:
        run_pg_dump(args.database_url, local_path)
    except Exception as e:
        logger.error("Backup failed during pg_dump: %s", e)
        # Don't leave a partial dump file behind on failure.
        if local_path.exists():
            local_path.unlink()
        return 1

    # Upload to S3 if configured. Track failure so a broken OFFSITE
    # backup surfaces as a non-zero exit — otherwise the Airflow DAG
    # would show green while only the local copy exists, defeating the
    # entire point of remote DR. The local backup + rotation still run
    # regardless, so the must-have (local) is never lost, and the
    # DAG's retry will re-dump + re-attempt the upload (good for
    # transient B2 outages).
    s3_failed = False
    if args.s3_bucket and not args.skip_upload:
        s3_key = f"{args.s3_prefix.rstrip('/')}/{filename}"
        try:
            upload_to_s3(local_path, args.s3_bucket, s3_key, args.s3_endpoint_url)
        except Exception as e:
            logger.error("S3 upload FAILED (local backup retained): %s", e)
            s3_failed = True

    # Rotate old local backups (always — local retention is independent
    # of the remote-upload outcome).
    deleted = rotate_local_backups(local_dir, args.local_retention_days)
    logger.info("Rotation: %d old backup(s) deleted.", deleted)

    if s3_failed:
        logger.error(
            "Backup run completed with a FAILED offsite upload — local "
            "copy is safe but the remote DR copy did not land. "
            "Exiting non-zero so the failure is visible."
        )
        return 2

    logger.info("Backup run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
