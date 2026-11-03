"""The api's Prometheus labels stay bounded (middleware/metrics.py).

A small FastAPI app wired like main.py (an included router with a prefix and
path parameters, a plain Starlette route, the /metrics mount, the metrics
middleware outermost) is driven through TestClient. Pinned: a request is
labelled with its route TEMPLATE, never the raw path; every unmatched path
(scanner probes included) shares one label; an unknown method is OTHER; the
label code never fails a request.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

fastapi = pytest.importorskip("fastapi")
from fastapi import APIRouter, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from middleware.metrics import UNMATCHED, endpoint_label, metrics_middleware  # noqa: E402
from prometheus_client import REGISTRY, make_asgi_app  # noqa: E402
from starlette.responses import PlainTextResponse  # noqa: E402


def build_app() -> FastAPI:
    app = FastAPI()
    router = APIRouter()

    @router.get("/{match_id}")
    async def get_match(match_id: int):
        return {"id": match_id}

    @router.get("/{match_id}/odds/{market}")
    async def get_odds(match_id: int, market: str):
        return {"id": match_id, "market": market}

    app.include_router(router, prefix="/api/v1/matches")

    @app.get("/health")
    async def health():
        return {"ok": True}

    async def plain(request):
        return PlainTextResponse("plain")

    app.add_route("/plain", plain)
    app.mount("/metrics", make_asgi_app())
    app.middleware("http")(metrics_middleware)
    return app


def count(method: str, endpoint: str, status: str) -> float:
    v = REGISTRY.get_sample_value("http_requests_total", {"method": method, "endpoint": endpoint, "status": status})
    return v or 0.0


def endpoints_seen() -> set[str]:
    return {
        s.labels["endpoint"]
        for m in REGISTRY.collect()
        if m.name == "http_requests"
        for s in m.samples
        if s.name == "http_requests_total"
    }


@pytest.fixture(scope="module")
def client():
    return TestClient(build_app())


def test_path_parameters_collapse_to_the_template(client):
    before = count("GET", "/api/v1/matches/{match_id}", "200")
    for i in (1, 2, 3, 98765):
        assert client.get(f"/api/v1/matches/{i}").status_code == 200
    assert count("GET", "/api/v1/matches/{match_id}", "200") == before + 4
    assert client.get("/api/v1/matches/7/odds/btts").status_code == 200
    assert count("GET", "/api/v1/matches/{match_id}/odds/{market}", "200") >= 1
    assert not any(e.startswith("/api/v1/matches/") and "{" not in e for e in endpoints_seen())


def test_scanner_probes_share_one_label(client):
    before_series = len(endpoints_seen())
    before = count("GET", UNMATCHED, "404")
    probes = ["/.env", "/.git/HEAD", "/../../root/.config/gcloud/credentials.db", "/wp-login.php", "/x" * 40]
    for p in probes:
        assert client.get(p).status_code == 404
    assert count("GET", UNMATCHED, "404") == before + len(probes)
    assert len(endpoints_seen()) <= before_series + 1  # at most UNMATCHED is new
    assert not any(e in endpoints_seen() for e in probes)


def test_mounts_and_plain_routes_keep_their_path(client):
    assert client.get("/metrics/").status_code == 200
    assert client.get("/plain").status_code == 200
    assert client.get("/health").status_code == 200
    seen = endpoints_seen()
    assert {"/metrics", "/plain", "/health"} <= seen


def test_wrong_method_on_a_real_route_is_labelled_by_route(client):
    assert client.post("/health").status_code == 405
    assert count("POST", "/health", "405") >= 1


def test_unknown_methods_are_other(client):
    before = count("OTHER", UNMATCHED, "404")
    client.request("PROPFIND", "/nowhere")
    client.request("XYZZY", "/also/nowhere")
    assert count("OTHER", UNMATCHED, "404") == before + 2


def test_timing_header_still_set(client):
    assert "X-Process-Time" in client.get("/health").headers


@pytest.mark.parametrize(
    "scope",
    [None, "not a dict", {}, {"route": object()}, {"endpoint": object()}, {"endpoint": object(), "app": object()}],
)
def test_endpoint_label_never_raises(scope):
    class Req:
        pass

    r = Req()
    if scope is not None:
        r.scope = scope
    assert endpoint_label(r) == UNMATCHED
