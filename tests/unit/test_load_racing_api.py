"""Unit tests for load_racing_api — pure parser helpers.

HTTP + DB layers are integration territory. These tests cover the
pure helpers: name normalization (country-suffix stripping), unit
conversion (furlongs → meters), prize parsing (multi-currency +
K/M suffixes), and odds-payload shape extraction.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lra = _load("load_racing_api", "load_racing_api.py")


# ── normalize_name ──────────────────────────────────────────────────


class TestNormalizeName:
    def test_lowercases_and_trims(self):
        assert lra.normalize_name("  Big Brown  ") == "big brown"

    def test_collapses_whitespace(self):
        assert lra.normalize_name("Frankel   the  Champion") == "frankel the champion"

    def test_strips_country_suffix(self):
        # "(USA)" / "(GB)" / "(IRE)" suffixes are racing-bureau
        # identifiers, not part of the horse's racing name. Stripping
        # them lets "Big Brown (USA)" and "Big Brown" collapse to
        # the same horses row.
        assert lra.normalize_name("Big Brown (USA)") == "big brown"
        assert lra.normalize_name("Frankel (GB)") == "frankel"
        assert lra.normalize_name("Sea The Stars (IRE)") == "sea the stars"

    def test_handles_empty(self):
        assert lra.normalize_name("") == ""
        assert lra.normalize_name(None) == ""

    def test_handles_no_country_suffix(self):
        # Lower-case parens (not country codes) are preserved.
        assert lra.normalize_name("Just A Horse (special)") == "just a horse (special)"


class TestParseCountrySuffix:
    def test_extracts_three_letter_code(self):
        assert lra.parse_country_suffix("Big Brown (USA)") == "USA"
        assert lra.parse_country_suffix("Frankel (GB)") == "GB"
        assert lra.parse_country_suffix("Sea The Stars (IRE)") == "IRE"

    def test_returns_none_when_no_suffix(self):
        assert lra.parse_country_suffix("Just A Horse") is None
        assert lra.parse_country_suffix("Horse (special)") is None  # lowercase = not a code

    def test_returns_none_for_empty(self):
        assert lra.parse_country_suffix("") is None
        assert lra.parse_country_suffix(None) is None


# ── furlongs_to_meters ─────────────────────────────────────────────


class TestFurlongsToMeters:
    def test_one_furlong_is_201m(self):
        # 1 furlong = 201.168 m exactly.
        assert lra.furlongs_to_meters(1) == 201

    def test_five_furlongs_is_about_1006m(self):
        # 5f = 1005.84 m → rounds to 1006.
        assert lra.furlongs_to_meters(5) == 1006

    def test_classic_mile_and_a_half(self):
        # 12f = 1 mile 4f = the Epsom Derby distance.
        # 12 × 201.168 = 2414.016 → rounds to 2414.
        assert lra.furlongs_to_meters(12) == 2414

    def test_handles_string_input(self):
        # Racing API sometimes ships numbers as strings.
        assert lra.furlongs_to_meters("8") == 1609

    def test_handles_none(self):
        assert lra.furlongs_to_meters(None) is None

    def test_handles_invalid(self):
        assert lra.furlongs_to_meters("abc") is None
        assert lra.furlongs_to_meters("") is None


# ── parse_offdt ────────────────────────────────────────────────────


class TestParseOffdt:
    def test_iso_with_z(self):
        out = lra.parse_offdt("2024-09-08T14:30:00Z")
        assert out == datetime(2024, 9, 8, 14, 30, tzinfo=timezone.utc)

    def test_iso_with_offset(self):
        out = lra.parse_offdt("2024-09-08T14:30:00+00:00")
        assert out == datetime(2024, 9, 8, 14, 30, tzinfo=timezone.utc)

    def test_iso_naive_assumed_utc(self):
        # Some Racing API responses drop the tz offset. Naive
        # datetimes get tagged UTC so downstream JOINs stay
        # timezone-correct.
        out = lra.parse_offdt("2024-09-08T14:30:00")
        assert out is not None
        assert out.tzinfo is not None

    def test_returns_none_on_bad_input(self):
        assert lra.parse_offdt("not-a-date") is None
        assert lra.parse_offdt(None) is None
        assert lra.parse_offdt("") is None


# ── _parse_prize ───────────────────────────────────────────────────


class TestParsePrize:
    def test_pound_thousands(self):
        # UK racing typical: '£1,000,000' → 1000000.0
        assert lra._parse_prize("£1,000,000") == 1_000_000.0

    def test_dollar_thousands(self):
        assert lra._parse_prize("$500,000") == 500_000.0

    def test_euro_thousands(self):
        assert lra._parse_prize("€250,000") == 250_000.0

    def test_k_suffix_expands(self):
        # '$500K' → 500,000
        assert lra._parse_prize("$500K") == 500_000.0
        assert lra._parse_prize("£50k") == 50_000.0

    def test_m_suffix_expands(self):
        assert lra._parse_prize("$2M") == 2_000_000.0

    def test_handles_none(self):
        assert lra._parse_prize(None) is None
        assert lra._parse_prize("") is None

    def test_handles_unparseable(self):
        assert lra._parse_prize("NOT A NUMBER") is None


# ── _strip_odds_prefix ─────────────────────────────────────────────


class TestStripOddsPrefix:
    def test_extracts_decimal_from_array(self):
        # Racing API racecards format: odds is a list of bookmaker
        # quotes; take the first one as the morning-line proxy.
        payload = [{"bookmaker": "bet365", "decimal": "1.5"}]
        assert lra._strip_odds_prefix(payload) == "1.5"

    def test_extracts_price_key_fallback(self):
        # Some endpoints use 'price' instead of 'decimal'.
        payload = [{"bookmaker": "bet365", "price": 2.5}]
        assert lra._strip_odds_prefix(payload) == 2.5

    def test_handles_raw_number(self):
        assert lra._strip_odds_prefix(1.5) == 1.5
        assert lra._strip_odds_prefix("3.2") == "3.2"

    def test_returns_none_when_empty(self):
        assert lra._strip_odds_prefix(None) is None
        assert lra._strip_odds_prefix([]) is None
        assert lra._strip_odds_prefix({}) is None


# ── _looks_uk ──────────────────────────────────────────────────────


class TestLooksUk:
    def test_uk_courses(self):
        assert lra._looks_uk("Ascot") is True
        assert lra._looks_uk("Newmarket") is True
        assert lra._looks_uk("Cheltenham") is True

    def test_irish_courses(self):
        # Irish racing shares the UK lookup since both use £/€ but
        # Racing API tags Irish courses as "Ireland" country.
        assert lra._looks_uk("Curragh") is True
        assert lra._looks_uk("Leopardstown") is True

    def test_handles_case(self):
        assert lra._looks_uk("ascot") is True
        assert lra._looks_uk("ASCOT") is True

    def test_returns_false_for_unknown(self):
        # US tracks (Churchill Downs, Belmont) shouldn't match the UK
        # list. Currency defaults to USD elsewhere.
        assert lra._looks_uk("Churchill Downs") is False
        assert lra._looks_uk("Belmont Park") is False

    def test_handles_empty(self):
        assert lra._looks_uk("") is False
        assert lra._looks_uk(None) is False


# ── Safe numeric helpers ───────────────────────────────────────────


class TestSafeNumerics:
    def test_safe_int_parses_string(self):
        assert lra._safe_int("5") == 5
        assert lra._safe_int("5.7") == 5  # truncates via int(float(.))

    def test_safe_int_handles_empty(self):
        assert lra._safe_int(None) is None
        assert lra._safe_int("") is None
        assert lra._safe_int("abc") is None

    def test_safe_float_parses(self):
        assert lra._safe_float("3.14") == 3.14
        assert lra._safe_float(2) == 2.0

    def test_safe_float_handles_empty(self):
        assert lra._safe_float(None) is None
        assert lra._safe_float("") is None
        assert lra._safe_float("abc") is None


# ── iter_dates ──────────────────────────────────────────────────────


class TestIterDates:
    def test_inclusive_range(self):
        from datetime import date

        days = list(lra.iter_dates(date(2024, 9, 5), date(2024, 9, 7)))
        assert days == [date(2024, 9, 5), date(2024, 9, 6), date(2024, 9, 7)]


# ── Argparse ───────────────────────────────────────────────────────


class TestCli:
    def test_upcoming_or_results_required(self):
        with pytest.raises(SystemExit):
            lra.parse_args([])

    def test_upcoming_mode(self):
        args = lra.parse_args(["--upcoming", "7"])
        assert args.upcoming == 7
        assert args.results is False

    def test_results_mode(self):
        args = lra.parse_args(["--results", "--start", "2024-01-01", "--end", "2024-12-31"])
        assert args.results is True
        assert args.start == "2024-01-01"
        assert args.end == "2024-12-31"

    def test_mutually_exclusive(self):
        # --upcoming and --results can't both be set.
        with pytest.raises(SystemExit):
            lra.parse_args(["--upcoming", "7", "--results"])
