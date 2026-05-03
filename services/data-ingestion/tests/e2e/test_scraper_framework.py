"""Integration: BaseScraper infrastructure (HTTP fetch + Redis dedup).

These tests build a tiny concrete scraper and run it through `_fetch_page`,
`_calculate_checksum`, and `_is_duplicate` against a real Redis. The HTTP
calls are intercepted with `responses`.
"""

import pytest
import responses


@pytest.fixture
def fetcher_scraper(scraper_config, db_manager, redis_client):
    """A minimal concrete scraper exposing the BaseScraper machinery."""
    from src.scrapers.base_scraper import BaseScraper

    class _Fetcher(BaseScraper):
        def scrape(self) -> int:  # required by ABC
            return 0

    return _Fetcher(scraper_config, db_manager, redis_client)


@responses.activate
def test_fetch_page_returns_response_body(fetcher_scraper):
    responses.add(
        responses.GET,
        "https://example.test/page",
        body="<html><body><h1>OK</h1></body></html>",
        status=200,
    )

    html = fetcher_scraper._fetch_page("https://example.test/page")
    assert "<h1>OK</h1>" in html


@responses.activate
def test_fetch_page_retries_on_5xx(fetcher_scraper):
    """Two 503s then a 200 — retry_with_backoff should win."""
    responses.add(responses.GET, "https://flaky.test/", status=503)
    responses.add(responses.GET, "https://flaky.test/", status=503)
    responses.add(responses.GET, "https://flaky.test/", body="ok", status=200)

    html = fetcher_scraper._fetch_page("https://flaky.test/")
    assert html == "ok"
    assert len(responses.calls) == 3


def test_dedup_checksum_persists_in_redis(fetcher_scraper, redis_client):
    """Same checksum twice in a row → second call reports duplicate."""
    payload = {"src": "test", "id": 42, "value": "first-write"}
    checksum = fetcher_scraper._calculate_checksum(payload)

    assert fetcher_scraper._is_duplicate(checksum) is False
    # The checksum should now be cached in Redis with the scraper-name namespace.
    key = f"checksum:{fetcher_scraper.scraper_name}:{checksum}"
    assert redis_client.exists(key) == 1
    assert fetcher_scraper._is_duplicate(checksum) is True


def test_dedup_distinct_payloads_have_different_checksums(fetcher_scraper):
    a = fetcher_scraper._calculate_checksum({"id": 1, "v": "a"})
    b = fetcher_scraper._calculate_checksum({"id": 1, "v": "b"})
    assert a != b


def test_dedup_payload_order_independent(fetcher_scraper):
    """`json.dumps(..., sort_keys=True)` makes ordering irrelevant."""
    a = fetcher_scraper._calculate_checksum({"x": 1, "y": 2})
    b = fetcher_scraper._calculate_checksum({"y": 2, "x": 1})
    assert a == b
