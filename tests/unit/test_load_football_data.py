"""load_football_data.parse_csv / main: a season file is decoded strictly, its
Div column must name the division it was fetched as, and a mis-decoded team
name refuses the file. On 2026-08-06 football-data served the Scottish
Championship and League One at the 2026-27 SP1/SP2 URLs; the byte-order mark
of the UTF-8 file (read as latin-1) hid the Div column, and ten Scottish
fixtures were filed under La Liga / La Liga 2."""

from __future__ import annotations

import codecs
import importlib.util
import logging
import re
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lfd = _load("load_football_data")

HEADER = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS,AS"
# The first rows of football-data's 2026-27 SC1.csv (fetched 2026-11-01): what 2627/SP1.csv held on 2026-08-06.
SC1_ROWS = [
    "SC1,01/08/2026,15:00,Ayr,Arbroath,2,0,H,1,0,H,R Hardie,6,6",
    "SC1,01/08/2026,15:00,Raith Rvs,Stenhousemuir,1,1,D,1,1,D,C Scott,14,3",
]
SP1_ROWS = ["SP1,14/08/2026,20:00,Ath Bilbao,Sevilla,2,1,H,1,0,H,X,10,8"]


def _file(rows: list[str], bom: bool = True, encoding: str = "utf-8") -> bytes:
    body = ("\n".join([HEADER, *rows]) + "\n").encode(encoding)
    return (codecs.BOM_UTF8 if bom else b"") + body


class TestParseCsv:
    def test_a_bom_file_keeps_its_div_column_and_loads(self):
        df = lfd.parse_csv(_file(SP1_ROWS), "SP1", "2627", "2627/SP1.csv")
        assert list(df["Div"]) == ["SP1"] and list(df["league"]) == ["SP1"] and list(df["season"]) == ["2627"]

    def test_another_divisions_file_at_the_url_is_refused(self):
        with pytest.raises(
            lfd.RefusedFile, match=r"fetched as SP1 2627 but 2 of its 2 row\(s\) name division\(s\) \['SC1'\]"
        ):
            lfd.parse_csv(_file(SC1_ROWS), "SP1", "2627", "2627/SP1.csv")

    def test_one_foreign_row_refuses_the_whole_file(self):
        with pytest.raises(lfd.RefusedFile, match=r"1 of its 3 row"):
            lfd.parse_csv(_file(SP1_ROWS + SC1_ROWS[:1] + SP1_ROWS[:1]), "SP1", "2627", "x")

    def test_a_file_without_a_div_column_is_refused(self):
        raw = b"Date,HomeTeam,AwayTeam,FTHG,FTAG\n14/08/2026,Ath Bilbao,Sevilla,2,1\n"
        with pytest.raises(lfd.RefusedFile, match="has no Div column"):
            lfd.parse_csv(raw, "SP1", "2627", "x")

    def test_the_utf_8_preussen_muenster_loads_spelled_right(self):
        rows = ["D2,31/10/2025,17:30,Preußen Münster,Holstein Kiel,1,1,D,0,0,D,X,10,9"]
        df = lfd.parse_csv(_file(rows), "D2", "2526", "2526/D2.csv")
        assert df["HomeTeam"].iloc[0] == "Preußen Münster"

    def test_a_name_mis_decoded_upstream_refuses_the_file(self):
        garbled = "Preußen Münster".encode("utf-8").decode("latin-1")
        rows = [f"D2,31/10/2025,17:30,{garbled},Holstein Kiel,1,1,D,0,0,D,X,10,9"]
        with pytest.raises(lfd.RefusedFile, match="look mis-decoded"):
            lfd.parse_csv(_file(rows), "D2", "2526", "x")

    def test_a_windows_1252_file_still_loads(self):
        # 2018-19 SC0: ASCII names, one 0xA0 (no-break space) in an odds column.
        raw = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,BbAHh\nSC0,04/08/2018,Celtic,Livingston,3,1,8\xa0\n"
        df = lfd.parse_csv(raw, "SC0", "1819", "x")
        assert list(df["HomeTeam"]) == ["Celtic"]

    def test_undecodable_bytes_are_refused(self):
        with pytest.raises(lfd.RefusedFile, match="neither UTF-8 nor Windows-1252"):
            lfd.parse_csv(b"Div,HomeTeam\nSP1,\x81\x8d\n", "SP1", "2627", "x")


class TestMain:
    def test_an_unknown_league_code_is_refused_before_anything_is_fetched(self, monkeypatch, caplog):
        fetched: list = []
        monkeypatch.setattr(lfd, "fetch_csv", lambda league, season: fetched.append(league))
        with caplog.at_level(logging.ERROR, logger="load_football_data"):
            assert lfd.main(["--leagues", "SP1,XX9", "--seasons", "1", "--database-url", ""]) == 2
        assert fetched == [] and "Unknown league code(s) ['XX9']" in caplog.text

    def test_every_documented_code_is_one_promote_raw_files(self):
        section = lfd.__doc__.split("League codes")[1].split("A code promote_raw")[0]
        documented = re.findall(r"^    ([A-Z]+[0-9])\s", section, re.M)
        assert {"SP1", "SP2", "SC0", "SC1", "SC2", "SC3"} <= set(documented)
        assert lfd.unknown_league_codes(documented) == []

    def test_a_refused_file_exits_3_after_the_others_are_written(self, monkeypatch, caplog):
        def fetch(league, season):
            if league == "SP2":
                return lfd.parse_csv(_file(SC1_ROWS), league, season, f"{season}/{league}.csv")
            return lfd.parse_csv(_file(SP1_ROWS), league, season, f"{season}/{league}.csv")

        written: list = []
        monkeypatch.setattr(lfd, "fetch_csv", fetch)
        monkeypatch.setattr(lfd, "write_database", lambda frames, url: written.extend(frames) or 1)
        with caplog.at_level(logging.ERROR, logger="load_football_data"):
            rc = lfd.main(["--leagues", "SP1,SP2", "--seasons", "1", "--end-year", "2027", "--database-url", "pg://x"])
        assert rc == lfd.REFUSED_EXIT_CODE == 3
        assert [df["league"].iloc[0] for df in written] == ["SP1"]
        assert "REFUSED SP2 2627" in caplog.text and "REFUSED 1 file(s)" in caplog.text

    def test_only_refused_files_exits_3_not_1(self, monkeypatch):
        monkeypatch.setattr(
            lfd, "fetch_csv", lambda league, season: lfd.parse_csv(_file(SC1_ROWS), league, season, "x")
        )
        assert lfd.main(["--leagues", "SP1", "--seasons", "1", "--end-year", "2027", "--database-url", ""]) == 3

    def test_a_missing_season_is_still_only_a_warning(self, monkeypatch):
        def fetch(league, season):
            if season == "2526":
                raise FileNotFoundError("404")
            return lfd.parse_csv(_file(SP1_ROWS), league, season, "x")

        monkeypatch.setattr(lfd, "fetch_csv", fetch)
        assert lfd.main(["--leagues", "SP1", "--seasons", "2", "--end-year", "2027", "--database-url", ""]) == 0
