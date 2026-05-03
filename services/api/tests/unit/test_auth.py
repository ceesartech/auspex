"""Tests for JWT authentication and DOB verification"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
import jwt as pyjwt

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all auth tests"""
    with patch("config.settings") as mock:
        mock.JWT_SECRET = "test-secret-key-for-testing-only"
        mock.JWT_ALGORITHM = "HS256"
        mock.JWT_EXPIRATION_HOURS = 24
        mock.USER_DOB = "1994-05-09"
        mock.DATABASE_URL = "postgresql://test:test@localhost:5432/test_db"
        mock.REDIS_URL = "redis://localhost:6379/15"
        yield mock


class TestCreateAccessToken:
    """Test JWT token creation"""

    def test_create_token_with_default_expiry(self, mock_settings):
        from auth.jwt_handler import create_access_token

        token = create_access_token(data={"user_id": "owner"})
        assert isinstance(token, str)
        assert len(token) > 0

        # Decode and verify
        payload = pyjwt.decode(
            token, "test-secret-key-for-testing-only", algorithms=["HS256"]
        )
        assert payload["user_id"] == "owner"
        assert "exp" in payload

    def test_create_token_with_custom_expiry(self, mock_settings):
        from auth.jwt_handler import create_access_token

        delta = timedelta(hours=1)
        token = create_access_token(data={"user_id": "owner"}, expires_delta=delta)

        payload = pyjwt.decode(
            token, "test-secret-key-for-testing-only", algorithms=["HS256"]
        )
        assert payload["user_id"] == "owner"

    def test_create_token_preserves_data(self, mock_settings):
        from auth.jwt_handler import create_access_token

        data = {"user_id": "owner", "username": "ceesar", "role": "admin"}
        token = create_access_token(data=data)

        payload = pyjwt.decode(
            token, "test-secret-key-for-testing-only", algorithms=["HS256"]
        )
        assert payload["user_id"] == "owner"
        assert payload["username"] == "ceesar"
        assert payload["role"] == "admin"


class TestVerifyToken:
    """Test JWT token verification"""

    def test_verify_valid_token(self, mock_settings):
        from auth.jwt_handler import create_access_token, verify_token

        token = create_access_token(data={"user_id": "owner"})
        payload = verify_token(token)

        assert payload["user_id"] == "owner"

    def test_verify_expired_token_raises(self, mock_settings):
        from auth.jwt_handler import verify_token
        from fastapi import HTTPException

        expired_token = pyjwt.encode(
            {"user_id": "owner", "exp": datetime.utcnow() - timedelta(hours=1)},
            "test-secret-key-for-testing-only",
            algorithm="HS256",
        )

        with pytest.raises(HTTPException) as exc_info:
            verify_token(expired_token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_verify_invalid_token_raises(self, mock_settings):
        from auth.jwt_handler import verify_token
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            verify_token("invalid.token.here")
        assert exc_info.value.status_code == 401

    def test_verify_wrong_secret_raises(self, mock_settings):
        from auth.jwt_handler import verify_token
        from fastapi import HTTPException

        token = pyjwt.encode(
            {"user_id": "owner", "exp": datetime.utcnow() + timedelta(hours=1)},
            "wrong-secret",
            algorithm="HS256",
        )

        with pytest.raises(HTTPException) as exc_info:
            verify_token(token)
        assert exc_info.value.status_code == 401


class TestVerifyDateOfBirth:
    """Test DOB verification"""

    def test_correct_dob(self, mock_settings):
        from auth.jwt_handler import verify_date_of_birth

        assert verify_date_of_birth("1994-05-09") is True

    def test_incorrect_dob(self, mock_settings):
        from auth.jwt_handler import verify_date_of_birth

        assert verify_date_of_birth("1990-01-01") is False

    def test_empty_dob(self, mock_settings):
        from auth.jwt_handler import verify_date_of_birth

        assert verify_date_of_birth("") is False


class TestAuthDependencies:
    """Test authentication dependencies"""

    @pytest.mark.asyncio
    async def test_get_current_user_valid(self, mock_settings):
        from auth.dependencies import get_current_user
        from auth.jwt_handler import create_access_token

        token = create_access_token(data={"user_id": "owner", "username": "ceesar"})

        credentials = MagicMock()
        credentials.credentials = token

        result = await get_current_user(credentials)
        assert result["user_id"] == "owner"
        assert result["username"] == "ceesar"

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, mock_settings):
        from auth.dependencies import get_current_user
        from fastapi import HTTPException

        credentials = MagicMock()
        credentials.credentials = "invalid-token"

        with pytest.raises(HTTPException):
            await get_current_user(credentials)

    def test_require_auth_valid(self, mock_settings):
        from auth.dependencies import require_auth

        payload = {"user_id": "owner", "username": "ceesar"}
        result = require_auth(payload)
        assert result["user_id"] == "owner"

    def test_require_auth_missing_user_id(self, mock_settings):
        from auth.dependencies import require_auth
        from fastapi import HTTPException

        payload = {"username": "ceesar"}  # No user_id

        with pytest.raises(HTTPException) as exc_info:
            require_auth(payload)
        assert exc_info.value.status_code == 401
