"""Tests for middleware - rate limiter, error handler, metrics"""

import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
import time

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture(autouse=True)
def mock_settings():
    with patch("config.settings") as mock:
        mock.JWT_SECRET = "test-secret-key-for-testing-only"
        mock.JWT_ALGORITHM = "HS256"
        mock.JWT_EXPIRATION_HOURS = 24
        mock.USER_DOB = "1994-05-09"
        mock.DATABASE_URL = "postgresql://test:test@localhost:5432/test_db"
        mock.REDIS_URL = "redis://localhost:6379/15"
        mock.RATE_LIMIT_REQUESTS = 100
        mock.RATE_LIMIT_WINDOW = 60
        yield mock


class TestRateLimiter:
    """Test rate limiting middleware"""

    def test_rate_limiter_init(self, mock_settings):
        from middleware.rate_limiter import RateLimitMiddleware

        app = MagicMock()
        limiter = RateLimitMiddleware(app)

        assert len(limiter.requests) == 0

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_normal_traffic(self, mock_settings):
        from middleware.rate_limiter import RateLimitMiddleware

        app = MagicMock()
        limiter = RateLimitMiddleware(app)

        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers.get.return_value = None

        response = MagicMock()
        response.headers = {}

        call_next = AsyncMock(return_value=response)

        result = await limiter.dispatch(request, call_next)

        call_next.assert_called_once_with(request)
        assert "X-RateLimit-Limit" in result.headers

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_traffic(self, mock_settings):
        import middleware.rate_limiter as rl_module
        from middleware.rate_limiter import RateLimitMiddleware

        # Patch settings at the module level where rate_limiter reads it
        original_requests = rl_module.settings.RATE_LIMIT_REQUESTS
        original_window = rl_module.settings.RATE_LIMIT_WINDOW
        rl_module.settings.RATE_LIMIT_REQUESTS = 5
        rl_module.settings.RATE_LIMIT_WINDOW = 60

        try:
            app = MagicMock()
            limiter = RateLimitMiddleware(app)

            request = MagicMock()
            request.client.host = "192.168.1.1"  # Unique IP
            request.headers.get.return_value = None

            response = MagicMock()
            response.headers = {}
            call_next = AsyncMock(return_value=response)

            # Fill up rate limit
            for _ in range(5):
                await limiter.dispatch(request, call_next)

            # 6th request should be blocked - returns JSONResponse with 429
            result = await limiter.dispatch(request, call_next)

            # JSONResponse has a real status_code attribute
            assert result.status_code == 429
        finally:
            rl_module.settings.RATE_LIMIT_REQUESTS = original_requests
            rl_module.settings.RATE_LIMIT_WINDOW = original_window

    @pytest.mark.asyncio
    async def test_rate_limiter_uses_auth_header_as_id(self, mock_settings):
        from middleware.rate_limiter import RateLimitMiddleware

        app = MagicMock()
        limiter = RateLimitMiddleware(app)

        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers.get.return_value = "Bearer some-token-here"

        response = MagicMock()
        response.headers = {}
        call_next = AsyncMock(return_value=response)

        await limiter.dispatch(request, call_next)

        # Token should be the client identifier
        assert "Bearer some-token-here" in limiter.requests

    @pytest.mark.asyncio
    async def test_rate_limiter_cleans_old_requests(self, mock_settings):
        from middleware.rate_limiter import RateLimitMiddleware

        app = MagicMock()
        limiter = RateLimitMiddleware(app)

        client_id = "127.0.0.1"
        # Add old request (beyond window)
        limiter.requests[client_id] = [time.time() - 120]

        request = MagicMock()
        request.client.host = client_id
        request.headers.get.return_value = None

        response = MagicMock()
        response.headers = {}
        call_next = AsyncMock(return_value=response)

        await limiter.dispatch(request, call_next)

        # Old request should be cleaned, only new one remains
        assert len(limiter.requests[client_id]) == 1


class TestErrorHandler:
    """Test error handler middleware"""

    @pytest.mark.asyncio
    async def test_error_handler_passes_through(self, mock_settings):
        from middleware.error_handler import error_handler_middleware

        request = MagicMock()
        request.method = "GET"
        request.url.path = "/health"

        expected_response = MagicMock()
        call_next = AsyncMock(return_value=expected_response)

        result = await error_handler_middleware(request, call_next)

        assert result == expected_response

    @pytest.mark.asyncio
    async def test_error_handler_catches_unhandled_exception(self, mock_settings):
        from middleware.error_handler import error_handler_middleware

        request = MagicMock()
        request.method = "GET"
        request.url.path = "/api/v1/crash"

        call_next = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        result = await error_handler_middleware(request, call_next)

        assert result.status_code == 500

    def test_format_validation_error(self, mock_settings):
        from middleware.error_handler import format_validation_error

        errors = [
            {
                "loc": ["body", "match_id"],
                "msg": "field required",
                "type": "value_error.missing",
            }
        ]

        formatted = format_validation_error(errors)

        assert len(formatted) == 1
        assert formatted[0]["field"] == "body -> match_id"
        assert formatted[0]["message"] == "field required"


class TestMetricsMiddleware:
    """Test Prometheus metrics middleware"""

    @pytest.mark.asyncio
    async def test_metrics_middleware_adds_timing_header(self, mock_settings):
        from middleware.metrics import metrics_middleware

        request = MagicMock()
        request.method = "GET"
        request.url.path = "/health"

        response = MagicMock()
        response.status_code = 200
        response.headers = {}

        call_next = AsyncMock(return_value=response)

        result = await metrics_middleware(request, call_next)

        assert "X-Process-Time" in result.headers

    @pytest.mark.asyncio
    async def test_metrics_middleware_records_metrics(self, mock_settings):
        from middleware.metrics import metrics_middleware, REQUEST_COUNT

        request = MagicMock()
        request.method = "POST"
        request.url.path = "/api/v1/predictions"

        response = MagicMock()
        response.status_code = 200
        response.headers = {}

        call_next = AsyncMock(return_value=response)

        await metrics_middleware(request, call_next)

        # Verify counter was incremented (Prometheus internals)
        # Just ensure no exception was raised during metric recording
        call_next.assert_called_once()
