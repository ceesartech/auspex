"""Rotate all secret values in an Auspex .env file.

Regenerates the following keys (and re-derives DATABASE_URL, REDIS_URL,
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN from them):

    POSTGRES_PASSWORD          token_urlsafe(32)
    REDIS_PASSWORD             token_urlsafe(24)
    JWT_SECRET                 token_hex(32)
    SECRET_KEY                 token_hex(64)
    AIRFLOW__CORE__FERNET_KEY  base64(token_bytes(32))  -- valid Fernet key
    AIRFLOW__WEBSERVER__SECRET_KEY  token_hex(32)
    GRAFANA_PASSWORD           token_urlsafe(20)

Does NOT touch any other key — TELEGRAM_*, USER_*, AUSPEX_*, CORS_ORIGINS,
NEXT_PUBLIC_*, feature flags, etc. all stay as you set them.

Usage:
    # Rotate in place, write a .env.bak first
    python scripts/rotate_secrets.py

    # Print the new .env to stdout without writing
    python scripts/rotate_secrets.py --dry-run

    # Just print the regenerated KEY=VALUE pairs (don't touch any file)
    python scripts/rotate_secrets.py --print-only

    # Operate on a different file
    python scripts/rotate_secrets.py --env-file /opt/auspex/.env

Re-running is safe: secrets are regenerated each run, other lines are
preserved exactly. Comments and ordering are kept.

This script uses stdlib only (secrets + base64) so it runs anywhere
python3 is available — no pip install required.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import secrets
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("rotate_secrets")


def gen_fernet_key() -> str:
    """A Fernet key is exactly 32 random bytes, urlsafe-base64 encoded."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")


def generate_new_secrets() -> dict[str, str]:
    return {
        "POSTGRES_PASSWORD": secrets.token_urlsafe(32),
        "REDIS_PASSWORD": secrets.token_urlsafe(24),
        "JWT_SECRET": secrets.token_hex(32),
        "SECRET_KEY": secrets.token_hex(64),
        "AIRFLOW__CORE__FERNET_KEY": gen_fernet_key(),
        "AIRFLOW__WEBSERVER__SECRET_KEY": secrets.token_hex(32),
        "GRAFANA_PASSWORD": secrets.token_urlsafe(20),
    }


def parse_env_file(path: Path) -> tuple[list[str], dict[str, int]]:
    """Return (lines, key->index map). Only includes plain KEY=VALUE lines
    in the index — comments and blanks aren't mapped but stay in the list."""
    lines = path.read_text().splitlines()
    index: dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key.isidentifier() or all(c.isalnum() or c == "_" for c in key):
            index[key] = i
    return lines, index


def upsert(lines: list[str], index: dict[str, int], key: str, value: str) -> None:
    """Replace value of `key` if present, otherwise append `KEY=VALUE`."""
    new_line = f"{key}={value}"
    if key in index:
        lines[index[key]] = new_line
    else:
        lines.append(new_line)
        index[key] = len(lines) - 1


def derive_urls(new: dict[str, str], existing: dict[str, str]) -> dict[str, str]:
    """Build derived URLs from the new secrets + existing user/db/host values.

    Falls back to safe defaults when the host file doesn't already have
    POSTGRES_USER / POSTGRES_DB set.
    """
    user = existing.get("POSTGRES_USER", "betting_user")
    db = existing.get("POSTGRES_DB", "betting_system")
    pg_pw = new["POSTGRES_PASSWORD"]
    redis_pw = new["REDIS_PASSWORD"]
    pg_url = f"postgresql://{user}:{pg_pw}@postgres:5432/{db}"
    return {
        "DATABASE_URL": pg_url,
        "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN": pg_url,
        "REDIS_URL": f"redis://:{redis_pw}@redis:6379/0",
    }


def current_values(lines: list[str], index: dict[str, int]) -> dict[str, str]:
    """Return a snapshot of current KEY=VALUE pairs for use in derive_urls."""
    out = {}
    for k, i in index.items():
        line = lines[i]
        if "=" in line:
            out[k] = line.split("=", 1)[1]
    return out


def rotate(env_path: Path, backup: bool, dry_run: bool) -> str:
    lines, index = parse_env_file(env_path)
    existing = current_values(lines, index)
    new = generate_new_secrets()

    for k, v in new.items():
        upsert(lines, index, k, v)
    for k, v in derive_urls(new, existing).items():
        upsert(lines, index, k, v)

    new_content = "\n".join(lines)
    if not new_content.endswith("\n"):
        new_content += "\n"

    if dry_run:
        return new_content

    if backup:
        backup_path = env_path.with_suffix(env_path.suffix + ".bak")
        shutil.copy2(env_path, backup_path)
        os.chmod(backup_path, 0o600)
        logger.info("Backed up %s -> %s", env_path, backup_path)

    # Atomic write: temp file in the same dir, then rename.
    tmp = env_path.with_suffix(env_path.suffix + ".tmp")
    tmp.write_text(new_content)
    os.chmod(tmp, 0o600)
    os.replace(tmp, env_path)
    logger.info("Wrote %d lines to %s", len(lines), env_path)
    return new_content


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--env-file", type=Path, default=Path(".env"), help="Path to the .env file (default: ./.env).")
    p.add_argument("--no-backup", action="store_true", help="Skip writing a .env.bak before overwriting.")
    p.add_argument("--dry-run", action="store_true", help="Print the new .env content to stdout, don't write.")
    p.add_argument("--print-only", action="store_true", help="Print just the rotated KEY=VALUE pairs and exit.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.print_only:
        for k, v in generate_new_secrets().items():
            print(f"{k}={v}")
        return 0

    if not args.env_file.exists():
        logger.error("env file not found: %s", args.env_file)
        return 2

    out = rotate(args.env_file, backup=not args.no_backup, dry_run=args.dry_run)
    if args.dry_run:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
