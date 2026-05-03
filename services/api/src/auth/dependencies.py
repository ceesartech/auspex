"""Authentication dependencies for routes"""

from typing import Dict

from auth.jwt_handler import verify_token
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict:
    """Get current authenticated user"""
    token = credentials.credentials
    payload = verify_token(token)
    return payload


def require_auth(payload: Dict = Depends(get_current_user)) -> Dict:
    """Require authentication"""
    if not payload.get("user_id"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication",
        )
    return payload
