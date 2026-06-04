"""Unit tests for scripts/scrape_nfl_injury_reports.py — parser
behaviour on mocked PFR-shape HTML. No network or DB."""

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


snir = _load("scrape_nfl_injury_reports", "scrape_nfl_injury_reports.py")


def _build_table(rows: list[dict], table_id: str = "injuries") -> str:
    body_rows = []
    for r in rows:
        body_rows.append(
            f"<tr>"
            f'<td data-stat="team">{r.get("team", "")}</td>'
            f'<td data-stat="player_name">{r.get("player", "")}</td>'
            f'<td data-stat="position">{r.get("position", "")}</td>'
            f'<td data-stat="status">{r.get("status", "")}</td>'
            f'<td data-stat="injury">{r.get("injury", "")}</td>'
            f"</tr>"
        )
    return (
        "<html><body>"
        f'<table id="{table_id}" class="stats_table">'
        "<thead><tr>"
        '<th data-stat="team">Team</th>'
        '<th data-stat="player_name">Player</th>'
        '<th data-stat="position">Pos</th>'
        '<th data-stat="status">Status</th>'
        '<th data-stat="injury">Injury</th>'
        "</tr></thead>"
        "<tbody>" + "".join(body_rows) + "</tbody></table>"
        "</body></html>"
    )


@pytest.mark.unit
class TestStatusCodes:
    def test_known_codes_set(self):
        # The 6 codes PFR uses across active injury reports.
        assert snir.STATUS_CODES == {"Q", "D", "O", "IR", "PUP", "NSI"}


@pytest.mark.unit
class TestParseInjuryTable:
    def test_single_row(self):
        rows = snir.parse_injury_table(
            _build_table(
                [
                    {"team": "NWE", "player": "Drake Maye", "position": "QB", "status": "Q", "injury": "shoulder"},
                ]
            )
        )
        assert rows == [
            {
                "team": "NWE",
                "player": "Drake Maye",
                "position": "QB",
                "status": "Q",
                "injury_type": "shoulder",
            }
        ]

    def test_multiple_rows_filtered_by_known_status(self):
        rows = snir.parse_injury_table(
            _build_table(
                [
                    {"team": "BUF", "player": "Josh Allen", "position": "QB", "status": "Q", "injury": "throwing-hand"},
                    {
                        "team": "MIA",
                        "player": "Tua Tagovailoa",
                        "position": "QB",
                        "status": "O",
                        "injury": "concussion",
                    },
                    # Unknown status — should be skipped.
                    {
                        "team": "MIA",
                        "player": "Random Backup",
                        "position": "QB",
                        "status": "PROBABLE",
                        "injury": "rest",
                    },
                ]
            )
        )
        assert len(rows) == 2
        assert rows[0]["player"] == "Josh Allen"
        assert rows[1]["player"] == "Tua Tagovailoa"

    def test_status_case_insensitive(self):
        # PFR is normally uppercase but the parser uppercases
        # defensively so 'q' inputs don't slip through.
        rows = snir.parse_injury_table(
            _build_table(
                [
                    {"team": "NYJ", "player": "Aaron Rodgers", "position": "QB", "status": "q", "injury": "achilles"},
                ]
            )
        )
        assert len(rows) == 1
        assert rows[0]["status"] == "Q"

    def test_skips_thead_and_spacer_rows(self):
        body = (
            '<tr class="thead">'
            '<td data-stat="team">section header</td>'
            '<td data-stat="player_name">x</td>'
            '<td data-stat="position">x</td>'
            '<td data-stat="status">x</td>'
            '<td data-stat="injury">x</td>'
            "</tr>"
            '<tr class="spacer">'
            '<td data-stat="team"></td>'
            '<td data-stat="player_name"></td>'
            '<td data-stat="position"></td>'
            '<td data-stat="status"></td>'
            '<td data-stat="injury"></td>'
            "</tr>"
            "<tr>"
            '<td data-stat="team">BUF</td>'
            '<td data-stat="player_name">Josh Allen</td>'
            '<td data-stat="position">QB</td>'
            '<td data-stat="status">Q</td>'
            '<td data-stat="injury">hand</td>'
            "</tr>"
        )
        html = (
            "<html><body>"
            '<table id="injuries" class="stats_table">'
            "<tbody>" + body + "</tbody></table></body></html>"
        )
        rows = snir.parse_injury_table(html)
        assert len(rows) == 1
        assert rows[0]["player"] == "Josh Allen"

    def test_skips_row_missing_player(self):
        body = (
            "<tr>"
            '<td data-stat="team">BUF</td>'
            '<td data-stat="position">QB</td>'
            '<td data-stat="status">Q</td>'
            '<td data-stat="injury">hand</td>'
            "</tr>"
        )
        html = (
            "<html><body>"
            '<table id="injuries" class="stats_table">'
            "<tbody>" + body + "</tbody></table></body></html>"
        )
        rows = snir.parse_injury_table(html)
        assert rows == []

    def test_falls_back_to_first_stats_table(self):
        # If PFR renames the id, the parser falls back to ANY
        # stats_table — the data-stat cell attributes still work.
        rows = snir.parse_injury_table(
            _build_table(
                [{"team": "DAL", "player": "Dak Prescott", "position": "QB", "status": "Q", "injury": "calf"}],
                table_id="some_other_id",
            )
        )
        assert len(rows) == 1
        assert rows[0]["player"] == "Dak Prescott"

    def test_no_table_returns_empty(self):
        rows = snir.parse_injury_table("<html><body>No table.</body></html>")
        assert rows == []

    def test_missing_status_skipped(self):
        body = (
            "<tr>"
            '<td data-stat="team">BUF</td>'
            '<td data-stat="player_name">Josh Allen</td>'
            '<td data-stat="position">QB</td>'
            '<td data-stat="status"></td>'
            '<td data-stat="injury">hand</td>'
            "</tr>"
        )
        html = (
            "<html><body>"
            '<table id="injuries" class="stats_table">'
            "<tbody>" + body + "</tbody></table></body></html>"
        )
        rows = snir.parse_injury_table(html)
        # Empty status fails both the truthy guard and the
        # STATUS_CODES whitelist.
        assert rows == []

    def test_qb_starter_status_codes(self):
        # All 6 codes a starter QB might appear under should parse.
        rows = snir.parse_injury_table(
            _build_table(
                [
                    {"team": "TM1", "player": "QB1", "position": "QB", "status": "Q", "injury": ""},
                    {"team": "TM2", "player": "QB2", "position": "QB", "status": "D", "injury": ""},
                    {"team": "TM3", "player": "QB3", "position": "QB", "status": "O", "injury": ""},
                    {"team": "TM4", "player": "QB4", "position": "QB", "status": "IR", "injury": ""},
                    {"team": "TM5", "player": "QB5", "position": "QB", "status": "PUP", "injury": ""},
                    {"team": "TM6", "player": "QB6", "position": "QB", "status": "NSI", "injury": ""},
                ]
            )
        )
        assert len(rows) == 6
        for r in rows:
            assert r["status"] in snir.STATUS_CODES
