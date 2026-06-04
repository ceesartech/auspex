"""Unit tests for scripts/scrape_fbref_venues.py — HTML parser +
team-name normalisation + comp_id mapping. No network or DB."""

from __future__ import annotations

import importlib.util
import sys
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


sf = _load("scrape_fbref_venues", "scrape_fbref_venues.py")


@pytest.mark.unit
class TestNormalize:
    def test_lowercase_and_strip(self):
        assert sf._normalize_team_name("  Manchester United  ") == "manchester united"

    def test_collapses_multiple_spaces(self):
        assert sf._normalize_team_name("Real  Madrid") == "real madrid"

    def test_empty(self):
        assert sf._normalize_team_name("") == ""
        assert sf._normalize_team_name(None) == ""  # type: ignore[arg-type]


@pytest.mark.unit
class TestCompIds:
    def test_premier_league_known(self):
        assert "Premier League" in sf.FBREF_COMP_IDS
        assert isinstance(sf.FBREF_COMP_IDS["Premier League"], int)

    def test_top5_leagues_present(self):
        # The big 5 European leagues all need comp_ids — they're the
        # bulk of the NULL-venue corpus.
        for name in ("Premier League", "Serie A", "La Liga", "Ligue 1", "Bundesliga"):
            assert name in sf.FBREF_COMP_IDS


@pytest.mark.unit
class TestParseScheduleHtml:
    """Mock the FBRef schedule table shape (stats_table class, data-stat
    attributes per cell) so we exercise the parser logic without HTTP."""

    @staticmethod
    def _build_html(rows: list[dict], extra_trs: str = "") -> str:
        # Each row provides {home, away, date, venue}; other cells filled.
        body_rows = []
        for r in rows:
            body_rows.append(
                f"<tr>"
                f'<td data-stat="home_team">{r["home"]}</td>'
                f'<td data-stat="away_team">{r["away"]}</td>'
                f'<td data-stat="date">{r["date"]}</td>'
                f'<td data-stat="venue">{r["venue"]}</td>'
                f"</tr>"
            )
        return (
            "<html><body>"
            '<table class="stats_table sortable">'
            "<thead><tr>"
            '<th data-stat="home_team">Home</th>'
            '<th data-stat="away_team">Away</th>'
            '<th data-stat="date">Date</th>'
            '<th data-stat="venue">Venue</th>'
            "</tr></thead>"
            "<tbody>" + extra_trs + "".join(body_rows) + "</tbody></table>"
            "</body></html>"
        )

    def test_single_row(self):
        html = self._build_html(
            [
                {"home": "Arsenal", "away": "Chelsea", "date": "2024-08-17", "venue": "Emirates Stadium"},
            ]
        )
        rows = sf.parse_schedule_html(html)
        assert rows == [
            {
                "home": "Arsenal",
                "away": "Chelsea",
                "date": "2024-08-17",
                "venue": "Emirates Stadium",
            }
        ]

    def test_multiple_rows(self):
        html = self._build_html(
            [
                {"home": "Liverpool", "away": "Man United", "date": "2024-09-01", "venue": "Anfield"},
                {"home": "Chelsea", "away": "Tottenham", "date": "2024-09-08", "venue": "Stamford Bridge"},
            ]
        )
        rows = sf.parse_schedule_html(html)
        assert len(rows) == 2
        assert rows[0]["venue"] == "Anfield"
        assert rows[1]["venue"] == "Stamford Bridge"

    def test_skips_thead_rows(self):
        # FBRef inserts "thead" rows mid-table for new sections.
        extra = (
            '<tr class="thead">'
            '<td data-stat="home_team">section header</td>'
            '<td data-stat="away_team">x</td>'
            '<td data-stat="date">x</td>'
            '<td data-stat="venue">x</td>'
            "</tr>"
        )
        html = self._build_html(
            [{"home": "Arsenal", "away": "Chelsea", "date": "2024-08-17", "venue": "Emirates Stadium"}],
            extra_trs=extra,
        )
        rows = sf.parse_schedule_html(html)
        assert len(rows) == 1
        assert rows[0]["home"] == "Arsenal"

    def test_skips_spacer_rows(self):
        extra = (
            '<tr class="spacer">'
            '<td data-stat="home_team"></td>'
            '<td data-stat="away_team"></td>'
            '<td data-stat="date"></td>'
            '<td data-stat="venue"></td>'
            "</tr>"
        )
        html = self._build_html(
            [{"home": "Arsenal", "away": "Chelsea", "date": "2024-08-17", "venue": "Emirates Stadium"}],
            extra_trs=extra,
        )
        rows = sf.parse_schedule_html(html)
        assert len(rows) == 1

    def test_skips_row_with_empty_venue(self):
        # Future / postponed matches list home/away/date but no
        # venue yet — drop those rather than write empty strings.
        html = self._build_html(
            [
                {"home": "Arsenal", "away": "Chelsea", "date": "2024-08-17", "venue": ""},
                {"home": "Liverpool", "away": "Man United", "date": "2024-09-01", "venue": "Anfield"},
            ]
        )
        rows = sf.parse_schedule_html(html)
        assert len(rows) == 1
        assert rows[0]["home"] == "Liverpool"

    def test_skips_row_with_missing_team(self):
        # Defensive: a malformed row missing home/away should be
        # skipped, not exception out.
        body = (
            "<tr>"
            '<td data-stat="away_team">Chelsea</td>'
            '<td data-stat="date">2024-08-17</td>'
            '<td data-stat="venue">Stadium</td>'
            "</tr>"
        )
        html = '<html><body><table class="stats_table">' "<tbody>" + body + "</tbody></table></body></html>"
        rows = sf.parse_schedule_html(html)
        assert rows == []

    def test_no_table_returns_empty(self):
        # If FBRef changes their markup and the table can't be found
        # we should not raise — just warn (logged) and return empty.
        rows = sf.parse_schedule_html("<html><body>No table here.</body></html>")
        assert rows == []
