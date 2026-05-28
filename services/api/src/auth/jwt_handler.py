"""JWT + password handling"""

from datetime import datetime, timedelta
from typing import Dict, Optional

import jwt
from config import settings
from fastapi import HTTPException, status
from passlib.context import CryptContext

# bcrypt is the only scheme we accept. Cost is left at passlib's default (12).
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


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


def hash_password(password: str) -> str:
    """Return a bcrypt hash suitable for storage in users.password_hash."""
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time bcrypt verification."""
    return _pwd_context.verify(password, password_hash)
