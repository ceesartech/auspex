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
