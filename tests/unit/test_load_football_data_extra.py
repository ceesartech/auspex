"""load_football_data_extra.COLUMN_MAP: a /new/ country-file column lands only
in the raw_football_data column of the same market. Until 2026-10-31
AvgCAHH/AvgCAHA (closing Asian-handicap average) mapped onto Avg>2.5/Avg<2.5
(over/under 2.5)."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


extra = _load("load_football_data_extra", "load_football_data_extra.py")

# football-data's Asian-handicap columns: AHh/AHCh (the line), B365AHH, PAHH, MaxAHH, AvgAHH, and the
# closing C variants (AvgCAHH, PCAHA...).
ASIAN_HANDICAP = re.compile(r"AH", re.IGNORECASE)
OVER_UNDER = re.compile(r"[<>][\d.]+")  # '>2.5' in 'Avg>2.5': side and line


def _over_under(column: str) -> str | None:
    m = OVER_UNDER.search(column)
    return m.group(0) if m else None


class TestColumnMap:
    def test_no_asian_handicap_column_maps_to_an_over_under_column(self):
        wrong = {src: dst for src, dst in extra.COLUMN_MAP.items() if ASIAN_HANDICAP.search(src) and _over_under(dst)}
        assert wrong == {}

    def test_every_over_under_target_comes_from_the_same_side_and_line(self):
        for src, dst in extra.COLUMN_MAP.items():
            if _over_under(dst):
                assert _over_under(src) == _over_under(dst), (src, dst)

    def test_the_old_mapping_is_gone(self):
        assert "AvgCAHH" not in extra.COLUMN_MAP and "AvgCAHA" not in extra.COLUMN_MAP

    def test_a_file_with_asian_handicap_columns_writes_no_over_under(self):
        df = pd.DataFrame(
            {
                "Home": ["Malmo FF"],
                "Away": ["Goteborg"],
                "Date": ["12/07/2026"],
                "Time": ["13:00"],
                "HG": [4],
                "AG": [0],
                "AvgCAHH": [1.91],
                "AvgCAHA": [1.99],
                "Season": ["2026"],
            }
        )
        out = extra.normalize(df, "SWE")
        assert "Avg>2.5" not in out.columns and "Avg<2.5" not in out.columns
        assert out.iloc[0][["HomeTeam", "AwayTeam", "FTHG", "FTAG", "league"]].tolist() == [
            "Malmo FF",
            "Goteborg",
            4,
            0,
            "SE1",
        ]


# --- parse_country_csv: strict decoding, the Country check, mis-decoded names (2026-11-01) -----------------------

NEW_HEADER = "Country,League,Season,Date,Time,Home,Away,HG,AG,Res"


def _country_file(rows: list[str], bom: bool = True) -> bytes:
    import codecs

    return (codecs.BOM_UTF8 if bom else b"") + ("\n".join([NEW_HEADER, *rows]) + "\n").encode("utf-8")


class TestParseCountryCsv:
    def test_a_bom_file_with_its_country_loads(self):
        # Every /new/ file is UTF-8 with a BOM; a few rows carry a trailing space ('Sweden ').
        rows = [
            "Sweden,Allsvenskan,2026,12/07/2026,13:00,Malmo FF,Goteborg,4,0,H",
            "Sweden ,Allsvenskan,2026,13/07/2026,13:00,AIK,Hacken,1,1,D",
        ]
        df = extra.parse_country_csv(_country_file(rows), "SWE", "new/SWE.csv")
        assert list(df["Home"]) == ["Malmo FF", "AIK"]

    def test_another_countrys_file_at_the_url_is_refused(self):
        rows = ["Norway,Eliteserien,2026,12/07/2026,13:00,Bodo/Glimt,Molde,2,0,H"]
        with pytest.raises(
            extra.RefusedFile, match=r"fetched as SWE \(Sweden\) but 1 of its 1 row\(s\) name \['Norway'\]"
        ):
            extra.parse_country_csv(_country_file(rows), "SWE", "new/SWE.csv")

    def test_a_file_without_a_country_column_is_refused(self):
        raw = b"League,Season,Date,Home,Away,HG,AG\nAllsvenskan,2026,12/07/2026,Malmo FF,Goteborg,4,0\n"
        with pytest.raises(extra.RefusedFile, match="has no Country column"):
            extra.parse_country_csv(raw, "SWE", "x")

    def test_a_mis_decoded_name_is_refused(self):
        garbled = "Malmö FF".encode("utf-8").decode("latin-1")
        rows = [f"Sweden,Allsvenskan,2026,12/07/2026,13:00,{garbled},Goteborg,4,0,H"]
        with pytest.raises(extra.RefusedFile, match="look mis-decoded"):
            extra.parse_country_csv(_country_file(rows), "SWE", "x")

    def test_undecodable_bytes_are_refused(self):
        with pytest.raises(extra.RefusedFile, match="neither UTF-8 nor Windows-1252"):
            extra.parse_country_csv(b"Country,Home\nSweden,\x81\x8d\n", "SWE", "x")


class TestExtraMain:
    def test_an_unknown_country_code_is_refused_before_anything_is_fetched(self, monkeypatch):
        fetched: list = []
        monkeypatch.setattr(extra, "fetch_country", lambda code: fetched.append(code))
        assert extra.main(["--countries", "SWE,XYZ", "--database-url", ""]) == 2
        assert fetched == []

    def test_a_refused_file_exits_3_after_the_others(self, monkeypatch):
        good = ["Sweden,Allsvenskan,2026,12/07/2026,13:00,Malmo FF,Goteborg,4,0,H"]
        wrong = ["Sweden,Allsvenskan,2026,12/07/2026,13:00,Malmo FF,Goteborg,4,0,H"]

        def fetch(code):
            return extra.parse_country_csv(_country_file(good if code == "SWE" else wrong), code, code)

        monkeypatch.setattr(extra, "fetch_country", fetch)
        assert extra.main(["--countries", "SWE,NOR", "--database-url", ""]) == extra.REFUSED_EXIT_CODE == 3
