"""Rate limiting middleware"""

import time
from collections import defaultdict

from config import settings
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""

    def __init__(self, app):
        super().__init__(app)
        self.requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        """Check rate limit before processing request"""

        # Get client identifier (IP or user ID)
        client_id = request.client.host if request.client else "unknown"

        # Get user from auth header if available
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            client_id = auth_header  # Use token as identifier

        # Check rate limit
        now = time.time()
        window_start = now - settings.RATE_LIMIT_WINDOW

        # Clean old requests
        self.requests[client_id] = [req_time for req_time in self.requests[client_id] if req_time > window_start]

        # Check limit
        if len(self.requests[client_id]) >= settings.RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": (
                        f"Maximum {settings.RATE_LIMIT_REQUESTS} requests " f"per {settings.RATE_LIMIT_WINDOW} seconds"
                    ),
                },
            )

        # Add current request
        self.requests[client_id].append(now)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(settings.RATE_LIMIT_REQUESTS - len(self.requests[client_id]))
        response.headers["X-RateLimit-Reset"] = str(int(window_start + settings.RATE_LIMIT_WINDOW))

        return response
