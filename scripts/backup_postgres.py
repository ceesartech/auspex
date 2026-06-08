"""Daily PostgreSQL backup → local disk + optional S3 upload.

Two-tier backup strategy:
  * Local (always): writes a compressed `pg_dump` to /app/backups/
    inside the api container, which is bind-mounted to
    /opt/auspex/backups/ on the host. Keeps the last
    --local-retention-days files for quick restore (default 7).
  * Remote (optional): if BACKUP_S3_BUCKET env var is set, also
    uploads the dump to S3 (or any S3-compatible storage like
    Backblaze B2, Hetzner Object Storage — set
    BACKUP_S3_ENDPOINT_URL for non-AWS providers). Long-term
    retention is managed by S3 lifecycle rules on the bucket
    side (transition to Glacier after 30 days, expire after 1
    year — see OPERATIONS.md for the lifecycle JSON to apply).

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
    BACKUP_S3_BUCKET          — optional; enables S3 upload
    BACKUP_S3_PREFIX          — optional; key prefix (default "postgres/")
    BACKUP_S3_ENDPOINT_URL    — optional; for non-AWS S3 (B2, Hetzner, etc.)
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY — standard AWS creds
    AWS_DEFAULT_REGION        — S3 region (default us-east-1)

Run on prod via:
    docker compose exec api python /app/scripts/backup_postgres.py

Wired into Airflow as daily 02:00 UTC by services/data-ingestion/
dags/db_backup.py.
"""

from __future__ import annotations

import argparse
import logging
import os
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


def upload_to_s3(local_path: Path, bucket: str, key: str, endpoint_url: Optional[str]) -> None:
    """Upload via boto3. Imports lazy so backups still run when
    boto3 isn't installed (local-only mode)."""
    try:
        import boto3
    except ImportError:
        logger.error(
            "boto3 not installed — install with `pip install boto3` "
            "to enable S3 uploads, or unset BACKUP_S3_BUCKET to skip."
        )
        raise

    client_kwargs = {}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
        logger.info("Using custom S3 endpoint: %s", endpoint_url)

    s3 = boto3.client("s3", **client_kwargs)
    logger.info("Uploading → s3://%s/%s", bucket, key)
    s3.upload_file(str(local_path), bucket, key)
    logger.info("Upload complete.")


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

    # Upload to S3 if configured.
    if args.s3_bucket and not args.skip_upload:
        s3_key = f"{args.s3_prefix.rstrip('/')}/{filename}"
        try:
            upload_to_s3(local_path, args.s3_bucket, s3_key, args.s3_endpoint_url)
        except Exception as e:
            logger.error("S3 upload failed (local backup retained): %s", e)
            # Don't fail the whole run — local backup succeeded.
            # The next run will retry the upload.

    # Rotate old local backups.
    deleted = rotate_local_backups(local_dir, args.local_retention_days)
    logger.info("Rotation: %d old backup(s) deleted.", deleted)

    logger.info("Backup run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
