"""Daily PostgreSQL backup → local disk + optional S3 upload.

Two-tier backup strategy:
  * Local (always): writes a compressed `pg_dump` to /app/backups/
    inside the api container, which is bind-mounted to
    /opt/auspex/backups/ on the host, for quick restore. See LOCAL
    ROTATION below for how many are kept.
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
default and supports parallel restore via pg_restore -j. One .dump
file per run, named <database>-<UTC stamp>.dump:
    betting_system-2026-11-02T020001Z.dump

LOCAL ROTATION (after every run, only this database's
<database>-<stamp>.dump files; audit CSVs, archives and any other
file in the directory are never touched):
  * The dump this run just wrote is never deleted, so the newest dump
    always stays on disk.
  * With BACKUP_S3_BUCKET set: keep at most --local-keep dumps (the
    newest N, default 3) and none older than --local-retention-days
    (default 7). An older dump is deleted ONLY after a HEAD on its B2
    copy (same key as its upload) shows the same byte size. If B2
    cannot be reached, or the copy is missing or a different size,
    the file is kept and an ERROR names it (fail closed), and the run
    exits 3 so the DAG turns red.
  * Without BACKUP_S3_BUCKET the local copy is the only copy: only the
    age rule applies (--local-keep is ignored), as before.
Why 3: every older dump is already in B2 before its local copy goes,
so local copies are a convenience for a fast restore, not the disaster
copy. A restore after a bad write wants the dump taken just before
that write (every ops wrapper takes one), so the newest three cover the
last write, the one before it and the previous run, at ~380 MB each
(2026-11). Ops days took 21 dumps in one day under the age-only rule
and filled the disk.

Exit codes: 0 ok; 1 pg_dump failed (no dump kept); 2 the offsite upload
failed (local dump kept); 3 the dump and upload succeeded but rotation
kept a dump it would have deleted (see the ERROR lines).

Restore (full DR, see OPERATIONS.md for details):
    pg_restore -h HOST -U USER -d betting_system --clean --if-exists \\
        betting_system-YYYY-MM-DDTHHMMSSZ.dump

Env vars consumed:
    DATABASE_URL              — postgres connection string
    BACKUP_LOCAL_DIR          — local dump directory (default /app/backups)
    BACKUP_LOCAL_RETENTION    — days of local backups to keep (default 7)
    BACKUP_LOCAL_KEEP         — max local dumps to keep when B2 is set
                                (default 3; ignored without B2)
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


def make_s3_client(endpoint_url: Optional[str]):
    """boto3 S3 client for B2 (or any S3-compatible store). Imports lazy
    so backups still run when boto3 isn't installed (local-only mode).
    Raises on any failure; callers decide what that means."""
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

    return boto3.client("s3", **client_kwargs)


def upload_to_s3(local_path: Path, bucket: str, key: str, endpoint_url: Optional[str], s3=None) -> None:
    """Upload via boto3, then HEAD the object to confirm it landed.
    Raises on any failure so the caller can flag a broken offsite
    backup."""
    if s3 is None:
        s3 = make_s3_client(endpoint_url)
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


DUMP_STAMP_FORMAT = "%Y-%m-%dT%H%M%SZ"


def s3_key_for(prefix: str, filename: str) -> str:
    """The object key a dump is uploaded under (and verified at)."""
    return f"{prefix.rstrip('/')}/{filename}"


def local_dumps(local_dir: Path, db_name: str) -> list[tuple[datetime, Path]]:
    """This database's dumps in local_dir, newest first, ordered by the
    UTC stamp in the name. Only files named exactly
    <db_name>-YYYY-MM-DDTHHMMSSZ.dump count: audit CSVs, archives and
    other databases' dumps are never rotation candidates."""
    pattern = re.compile(re.escape(db_name) + r"-(\d{4}-\d{2}-\d{2}T\d{6}Z)\.dump")
    out: list[tuple[datetime, Path]] = []
    for f in local_dir.iterdir():
        m = pattern.fullmatch(f.name)
        if not m or not f.is_file():
            continue
        stamp = datetime.strptime(m.group(1), DUMP_STAMP_FORMAT).replace(tzinfo=timezone.utc)
        out.append((stamp, f))
    out.sort(key=lambda t: (t[0], t[1].name), reverse=True)
    return out


def plan_rotation(
    dumps: list[tuple[datetime, Path]],
    *,
    current: Path,
    now: datetime,
    retention_days: int,
    keep: Optional[int],
) -> list[tuple[Path, str]]:
    """(dump, reason) for every dump the rules would delete, oldest
    last. `dumps` is newest first (local_dumps). keep=None means no
    count cap (no B2: the age rule alone). The current dump is never
    planned, whatever its stamp."""
    cutoff = now - timedelta(days=retention_days) if retention_days > 0 else None
    planned: list[tuple[Path, str]] = []
    for rank, (stamp, f) in enumerate(dumps):
        if f.resolve() == current.resolve():
            continue
        reasons = []
        if keep is not None and rank >= keep:
            reasons.append(f"beyond the newest {keep}")
        if cutoff is not None and stamp < cutoff:
            reasons.append(f"older than {retention_days} days")
        if reasons:
            planned.append((f, " and ".join(reasons)))
    return planned


def remote_copy_problem(s3, bucket: str, key: str, local_size: int) -> Optional[str]:
    """None when s3://bucket/key exists with exactly local_size bytes,
    else why not. Any failure (missing object, auth, network) is a
    problem: the caller then keeps the local file."""
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except Exception as e:  # botocore raises several unrelated types; every one means "not verified"
        return f"HEAD s3://{bucket}/{key} failed: {type(e).__name__}: {e}"
    remote_size = head.get("ContentLength") if isinstance(head, dict) else None
    if remote_size != local_size:
        return f"s3://{bucket}/{key} is {remote_size} bytes, the local file {local_size}"
    return None


def rotate_local_backups(
    local_dir: Path,
    db_name: str,
    *,
    current: Path,
    retention_days: int,
    keep: int,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "postgres/",
    s3_endpoint_url: Optional[str] = None,
    s3_client_factory=None,
    now: Optional[datetime] = None,
) -> tuple[list[Path], list[Path]]:
    """Apply the LOCAL ROTATION rules (module docstring). Returns
    (deleted, kept_with_error): kept_with_error are dumps the rules
    wanted gone but that stayed because their B2 copy was not verified
    or the delete failed; each was logged at ERROR by name."""
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    now = now or datetime.now(timezone.utc)
    dumps = local_dumps(local_dir, db_name)
    if s3_bucket:
        planned = plan_rotation(dumps, current=current, now=now, retention_days=retention_days, keep=keep)
    else:
        logger.info(
            "No BACKUP_S3_BUCKET: the local dumps are the only copies — applying only the %d-day age rule "
            "(--local-keep is ignored).",
            retention_days,
        )
        planned = plan_rotation(dumps, current=current, now=now, retention_days=retention_days, keep=None)
    logger.info("Rotation: %d local dump(s) of %s, %d to rotate out.", len(dumps), db_name, len(planned))

    deleted: list[Path] = []
    kept: list[Path] = []
    s3 = None
    client_error: Optional[str] = None
    if planned and s3_bucket:
        try:
            s3 = s3_client_factory() if s3_client_factory else make_s3_client(s3_endpoint_url)
        except Exception as e:  # no boto3, bad creds/config: nothing can be verified
            client_error = f"could not create the S3 client: {type(e).__name__}: {e}"

    for f, reason in planned:
        try:
            size = f.stat().st_size
        except OSError as e:
            logger.error("KEPT %s (%s) — could not stat it: %s", f, reason, e)
            kept.append(f)
            continue
        if s3_bucket:
            key = s3_key_for(s3_prefix, f.name)
            problem = client_error or remote_copy_problem(s3, s3_bucket, key, size)
            if problem:
                logger.error("KEPT %s (%s) — its B2 copy is not verified: %s", f, reason, problem)
                kept.append(f)
                continue
            why = f"{reason}; B2 copy verified, {size} bytes at s3://{s3_bucket}/{key}"
        else:
            why = reason
        try:
            f.unlink()
        except OSError as e:
            logger.error("KEPT %s (%s) — the delete failed: %s", f, why, e)
            kept.append(f)
            continue
        logger.info("Rotated out %s (%s)", f.name, why)
        deleted.append(f)
    return deleted, kept


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        help="Delete this database's local dumps older than this many days (default 7, env "
        "BACKUP_LOCAL_RETENTION; 0 disables the age rule). With B2 set, only after the B2 copy is verified.",
    )
    parser.add_argument(
        "--local-keep",
        type=_positive_int,
        default=os.environ.get("BACKUP_LOCAL_KEEP", "3"),
        help="With BACKUP_S3_BUCKET set: keep at most this many local dumps of this database, the newest "
        "(default 3, env BACKUP_LOCAL_KEEP; >= 1). An older one is deleted only after its B2 copy is "
        "confirmed with the same byte size; when that cannot be confirmed the file is kept and the run "
        "exits 3. Ignored without BACKUP_S3_BUCKET: the local copy is then the only copy, so only the "
        "age rule applies.",
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
    args = parser.parse_args(argv)
    logger.setLevel(args.log_level)

    if not args.database_url:
        logger.error("DATABASE_URL not set.")
        return 1

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Filename includes UTC timestamp + DB name from URL — makes
    # multi-DB setups distinguish naturally.
    db_name = parse_database_url(args.database_url)["dbname"]
    ts = datetime.now(timezone.utc).strftime(DUMP_STAMP_FORMAT)
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
        s3_key = s3_key_for(args.s3_prefix, filename)
        try:
            upload_to_s3(local_path, args.s3_bucket, s3_key, args.s3_endpoint_url)
        except Exception as e:
            logger.error("S3 upload FAILED (local backup retained): %s", e)
            s3_failed = True

    # Rotate old local backups (always — even after a failed upload: an
    # older dump goes only once its own B2 copy is verified, and this
    # run's dump is never a candidate).
    deleted, kept = rotate_local_backups(
        local_dir,
        db_name,
        current=local_path,
        retention_days=args.local_retention_days,
        keep=args.local_keep,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_endpoint_url=args.s3_endpoint_url,
    )
    logger.info("Rotation: %d old backup(s) deleted, %d kept on an ERROR.", len(deleted), len(kept))

    if s3_failed:
        logger.error(
            "Backup run completed with a FAILED offsite upload — local "
            "copy is safe but the remote DR copy did not land. "
            "Exiting non-zero so the failure is visible."
        )
        return 2
    if kept:
        logger.error(
            "Backup and upload OK, but rotation kept %d dump(s) it would have deleted (ERROR lines above: "
            "%s). Exiting 3 so the disk does not fill silently.",
            len(kept),
            ", ".join(f.name for f in kept),
        )
        return 3

    logger.info("Backup run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
