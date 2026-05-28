"""Seed pre-defined user accounts with bcrypt-hashed passwords.

Personal-use Auspex doesn't have a signup flow — accounts are created via
this script. Re-running is idempotent: existing usernames are updated
with the new password/role, missing ones are inserted.

Usage:
    # Seed the default three (admin + the two configured owner emails)
    docker compose exec api python /app/scripts/seed_users.py

    # Add or rotate a single user
    docker compose exec api python /app/scripts/seed_users.py \\
        --username someone --email someone@example.com \\
        --password 'StrongPass1!' --role user
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass

import psycopg2

# `passlib` ships with Auspex (requirements.txt). The hash_password helper
# is reused so this script stays consistent with the runtime.
sys.path.insert(0, "/app/services/api/src")
try:
    from auth.jwt_handler import hash_password
except ImportError:
    # Fallback for running outside the api container.
    from passlib.context import CryptContext

    _pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash_password(password: str) -> str:  # type: ignore[no-redef]
        return _pwd.hash(password)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("seed_users")


@dataclass
class SeedUser:
    username: str
    email: str | None
    password: str
    role: str = "user"


# These three are seeded by default. Password is the same for all three at
# the user's request; rotate per-account via --username after first login.
DEFAULT_USERS: list[SeedUser] = [
    SeedUser(username="admin", email=None, password="Admin@1234", role="admin"),
    SeedUser(
        username="chijiokekechi",
        email="chijiokekechi@gmail.com",
        password="Admin@1234",
        role="admin",
    ),
    SeedUser(
        username="ceesartech25",
        email="ceesartech25@gmail.com",
        password="Admin@1234",
        role="user",
    ),
]


UPSERT_SQL = """
    INSERT INTO users (username, email, password_hash, role, is_active)
    VALUES (%(username)s, %(email)s, %(password_hash)s, %(role)s, true)
    ON CONFLICT (username) DO UPDATE SET
        email = EXCLUDED.email,
        password_hash = EXCLUDED.password_hash,
        role = EXCLUDED.role,
        is_active = true
"""


def seed(users: list[SeedUser], database_url: str) -> int:
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            for u in users:
                cur.execute(
                    UPSERT_SQL,
                    {
                        "username": u.username,
                        "email": u.email,
                        "password_hash": hash_password(u.password),
                        "role": u.role,
                    },
                )
                logger.info("Upserted user=%s role=%s", u.username, u.role)
        conn.commit()
    return len(users)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--username", help="Seed a single user with this username.")
    p.add_argument("--email", help="Email for the single-user mode.")
    p.add_argument("--password", help="Password for the single-user mode.")
    p.add_argument("--role", default="user", choices=["admin", "user"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set and --database-url not provided")
        return 2

    if args.username:
        if not args.password:
            logger.error("--password is required when --username is given")
            return 2
        users = [SeedUser(args.username, args.email, args.password, args.role)]
    else:
        users = DEFAULT_USERS

    n = seed(users, args.database_url)
    logger.info("Seeded %d user(s)", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
