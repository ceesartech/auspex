"""Global exception handlers and error formatting"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
import logging
import time
import traceback

logger = logging.getLogger(__name__)


async def error_handler_middleware(request: Request, call_next):
    """Global error handler middleware"""
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}: {exc}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred. Please try again later.",
                "timestamp": time.time(),
            },
        )


def format_validation_error(errors: list) -> list:
    """Format Pydantic validation errors for API response"""
    formatted = []
    for error in errors:
        formatted.append(
            {
                "field": " -> ".join(str(loc) for loc in error.get("loc", [])),
                "message": error.get("msg", ""),
                "type": error.get("type", ""),
            }
        )
    return formatted
