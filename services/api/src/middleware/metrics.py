"""Prometheus metrics middleware"""

import time

from prometheus_client import Counter, Histogram

# Metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
)


async def metrics_middleware(request, call_next):
    """Collect Prometheus metrics"""

    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()

    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    # Add timing header
    response.headers["X-Process-Time"] = str(duration)

    return response
