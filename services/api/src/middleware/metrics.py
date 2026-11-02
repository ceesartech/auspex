"""Prometheus metrics middleware.

The `endpoint` label is the matched route TEMPLATE ("/api/v1/matches/{match_id}"),
never the raw request path. With the raw path every distinct URL became its own
label set — each match id, and every probe a vulnerability scanner sends
("/.env", "/.git/HEAD", "/../../root/.config/…") — so the series count grew
without bound until the api restarted: 7,247 api series on 2026-10-23 against
~2,200 a week earlier, and Prometheus wrote ~700 MB/day instead of ~60 MB/day
(prometheus_client also keeps every label set in the api's memory). Requests
that match no route (404s, slash redirects) share one label, UNMATCHED, and an
unknown HTTP method is labelled OTHER.
"""

import time

from prometheus_client import Counter, Histogram

UNMATCHED = "<unmatched>"
# Any other method token (clients may send arbitrary ones) is labelled OTHER.
KNOWN_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})

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


def endpoint_label(request) -> str:
    """Route template the router matched for this request, else UNMATCHED.

    Read after the app has handled the request: the router writes the match
    into the shared ASGI scope. FastAPI routes put themselves in
    scope["route"]; plain Starlette routes and mounts (/docs, /metrics) only
    leave scope["endpoint"], which is looked up among the app's routes.
    Never raises — a metrics label must not break a request."""
    scope = getattr(request, "scope", None)
    if not isinstance(scope, dict):
        return UNMATCHED
    path = getattr(scope.get("route"), "path", None)
    if isinstance(path, str) and path:
        return path
    endpoint = scope.get("endpoint")
    if endpoint is None:
        return UNMATCHED
    routes = getattr(getattr(scope.get("app"), "router", None), "routes", None)
    if not isinstance(routes, list):
        return UNMATCHED
    for route in routes:
        if getattr(route, "endpoint", None) is endpoint or getattr(route, "app", None) is endpoint:
            path = getattr(route, "path", None)
            if isinstance(path, str) and path:
                return path
    return UNMATCHED


async def metrics_middleware(request, call_next):
    """Collect Prometheus metrics"""

    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    endpoint = endpoint_label(request)
    method = request.method if request.method in KNOWN_METHODS else "OTHER"

    # Record metrics
    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status=response.status_code,
    ).inc()

    REQUEST_DURATION.labels(
        method=method,
        endpoint=endpoint,
    ).observe(duration)

    # Add timing header
    response.headers["X-Process-Time"] = str(duration)

    return response
