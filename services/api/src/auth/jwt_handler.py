"""JWT + password handling"""

from datetime import datetime, timedelta
from typing import Dict, Optional

import bcrypt
import jwt
from config import settings
from fastapi import HTTPException, status

# bcrypt has a hard 72-byte limit on the input. Truncate longer inputs
# rather than letting bcrypt raise, since the alternative is asking users
# to remember which characters got dropped.
_BCRYPT_MAX_BYTES = 72


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

    return encoded_jwt


def verify_token(token: str) -> Dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def _encode(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    """Return a bcrypt hash suitable for storage in users.password_hash."""
    return bcrypt.hashpw(_encode(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time bcrypt verification. Returns False on any malformed input."""
    try:
        return bcrypt.checkpw(_encode(password), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False
