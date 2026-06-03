"""Unit tests for geocode_soccer_venues — pure helpers + CLI.

We don't hit Open-Meteo from tests. The HTTP fetcher geocode() is a
thin wrapper over requests + parse_geocode_result; the parsing
branches are covered exhaustively here, and the URL shape is
locked via build_search_url. The DB upsert path is covered by an
in-memory cursor stub so we exercise the insert/update/skip
fan-out without needing a real psycopg connection.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


geo = _load("geocode_soccer_venues", "geocode_soccer_venues.py")


# ── Constants lockdown ──────────────────────────────────────────────


class TestConstants:
    def test_geocode_url(self):
        # Open-Meteo's geocoding endpoint — free, no API key. Locked
        # so any future move to a paid geocoder forces an explicit
        # test update.
        assert geo.GEOCODE_URL == "https://geocoding-api.open-meteo.com/v1/search"

    def test_throttle_matches_fetch_weather(self):
        # 0.20s/call keeps us well under the 10k/day free quota even
        # with sustained sequential lookups.
        assert geo.REQUEST_DELAY_SEC == 0.20


# ── normalize_venue: matches seed_venue_coords shape ────────────────


class TestNormalizeVenue:
    def test_lowercases(self):
        # Mirror of seed_venue_coords.normalize_venue — same shape
        # ensures manually-seeded rows and geocoded rows collide on
        # the same UNIQUE key.
        assert geo.normalize_venue("Allianz Parque") == "allianz parque"

    def test_collapses_whitespace(self):
        assert geo.normalize_venue("  Camp   Nou  ") == "camp nou"

    def test_handles_empty(self):
        assert geo.normalize_venue("") == ""
        assert geo.normalize_venue(None) == ""


# ── build_search_url: param shape ───────────────────────────────────


class TestBuildSearchUrl:
    def test_includes_name_param(self):
        url = geo.build_search_url("Anfield")
        assert "name=Anfield" in url

    def test_caps_count_at_one(self):
        # We only ever consume the first hit (top match by population).
        # If a future tweak wants alternatives, the parse side needs
        # an update too — locked here so the contract stays explicit.
        url = geo.build_search_url("Anfield")
        assert "count=1" in url

    def test_quotes_special_characters(self):
        # Stadium names with spaces, accents, ampersands need URL
        # encoding so the API doesn't see malformed query strings.
        url = geo.build_search_url("Estádio do Maracanã")
        # %20 = space, %C3%A1 = á, %C3%A3 = ã.
        assert " " not in url
        assert "%C3%A1" in url or "%C3%83" in url


# ── parse_geocode_result: response shape branches ───────────────────


class TestParseGeocodeResult:
    def test_returns_first_hit(self):
        # Modal Open-Meteo response: { results: [hit, ...] }.
        out = geo.parse_geocode_result(
            {
                "results": [
                    {"latitude": -22.9519, "longitude": -43.2105, "timezone": "America/Sao_Paulo", "country_code": "BR"},
                    {"latitude": 0.0, "longitude": 0.0},  # Second hit ignored.
                ]
            }
        )
        assert out == {
            "latitude": -22.9519,
            "longitude": -43.2105,
            "timezone": "America/Sao_Paulo",
            "country_code": "BR",
        }

    def test_returns_none_for_empty_results(self):
        # API returns {results: []} when nothing matches the query.
        assert geo.parse_geocode_result({"results": []}) is None

    def test_returns_none_when_results_key_missing(self):
        # Defensive: Open-Meteo's docs say results is always
        # present, but a 200 with no results key has been observed
        # historically in similar APIs.
        assert geo.parse_geocode_result({}) is None

    def test_returns_none_when_payload_is_not_dict(self):
        # Caller passes raw decoded JSON; a string / list shouldn't
        # crash the parser.
        assert geo.parse_geocode_result("not a dict") is None
        assert geo.parse_geocode_result(None) is None

    def test_returns_none_when_lat_missing(self):
        # Partial hit (no coords) is unusable — skip it.
        out = geo.parse_geocode_result({"results": [{"longitude": 0.0, "timezone": "UTC"}]})
        assert out is None

    def test_returns_none_when_lon_missing(self):
        out = geo.parse_geocode_result({"results": [{"latitude": 0.0, "timezone": "UTC"}]})
        assert out is None

    def test_falls_back_to_utc_when_timezone_missing(self):
        # Open-Meteo populates timezone for every populated-place
        # result we've seen, but a missing field shouldn't blow up.
        # Better to anchor weather lookups to UTC than skip the venue.
        out = geo.parse_geocode_result({"results": [{"latitude": 0.0, "longitude": 0.0}]})
        assert out is not None
        assert out["timezone"] == "UTC"


# ── upsert path: insert vs update vs skip ───────────────────────────


class _FakeCursor:
    """Stub for upsert_geocoded tests. Tracks the executed SQL +
    parameter tuples + lets the test seed fetchone() / fetchall()
    responses."""

    def __init__(self):
        self.executions: list[tuple[str, tuple]] = []
        self._responses: list = []

    def queue(self, *responses):
        # Test helper to seed fetchone()/fetchall() return values.
        self._responses.extend(responses)

    def execute(self, sql, params=None):
        self.executions.append((sql, params or ()))

    def fetchone(self):
        return self._responses.pop(0) if self._responses else None


class TestUpsertGeocoded:
    def _hit(self):
        return {"latitude": -22.95, "longitude": -43.21, "timezone": "America/Sao_Paulo"}

    def test_inserts_when_row_absent_no_update_flag(self):
        cur = _FakeCursor()
        # First query: existence check returns None (no existing row).
        # Second query: INSERT...RETURNING inserted=True.
        cur.queue(None, {"inserted": True})
        result = geo.upsert_geocoded(cur, "Maracanã", self._hit(), "src", update=False)
        assert result == "inserted"
        # Two SQL calls: existence check, then INSERT.
        assert len(cur.executions) == 2
        assert "SELECT 1 FROM venue_coords" in cur.executions[0][0]
        assert "INSERT INTO venue_coords" in cur.executions[1][0]

    def test_skips_when_row_exists_no_update_flag(self):
        cur = _FakeCursor()
        # Existence check finds a row → return 'skipped', no INSERT.
        cur.queue({"_": 1})
        result = geo.upsert_geocoded(cur, "Maracanã", self._hit(), "src", update=False)
        assert result == "skipped"
        # Only the existence check ran.
        assert len(cur.executions) == 1

    def test_updates_when_update_flag_set_and_row_exists(self):
        cur = _FakeCursor()
        # --update bypasses the existence check entirely.
        # INSERT...ON CONFLICT DO UPDATE returns inserted=False
        # because xmax != 0 (the row already existed pre-conflict).
        cur.queue({"inserted": False})
        result = geo.upsert_geocoded(cur, "Maracanã", self._hit(), "src", update=True)
        assert result == "updated"
        # No existence check fired — straight to INSERT...ON CONFLICT.
        assert len(cur.executions) == 1
        assert "INSERT INTO venue_coords" in cur.executions[0][0]

    def test_normalizes_venue_name_for_unique_key(self):
        # The UNIQUE constraint is on normalized_venue_name, so the
        # upsert MUST pass the lower-cased + whitespace-collapsed
        # form as the key. Without that, "Camp Nou" and "camp nou"
        # would race and one would error on the constraint.
        cur = _FakeCursor()
        cur.queue(None, {"inserted": True})
        geo.upsert_geocoded(cur, "  Camp   Nou  ", self._hit(), "src", update=False)
        # The first param of the second execute (INSERT) is the
        # normalized name; the second is the original display_name.
        _, insert_params = cur.executions[1]
        assert insert_params[0] == "camp nou"
        assert insert_params[1] == "  Camp   Nou  "


# ── CLI ─────────────────────────────────────────────────────────────


class TestCli:
    def test_defaults(self):
        args = geo.parse_args(["--database-url", "postgresql://x"])
        assert args.limit is None
        assert args.update is False
        assert args.dry_run is False

    def test_limit_parses_as_int(self):
        args = geo.parse_args(["--limit", "10", "--database-url", "x"])
        assert args.limit == 10

    def test_update_flag(self):
        args = geo.parse_args(["--update", "--database-url", "x"])
        assert args.update is True

    def test_dry_run_flag(self):
        args = geo.parse_args(["--dry-run", "--database-url", "x"])
        assert args.dry_run is True
