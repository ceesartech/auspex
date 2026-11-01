"""scripts/text_encoding.py: strict decoding of football-data CSVs (UTF-8 with or
without a byte-order mark, else Windows-1252, else a loud refusal) and the
mojibake test the loaders, promote_raw and repair_soccer_labels share. Until
2026-11-01 every file was read as latin-1: the BOM hid the Div column and
'Preußen Münster' became 'PreuÃ\\x9fen MÃ¼nster'."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("text_encoding", ROOT / "scripts" / "text_encoding.py")
te = importlib.util.module_from_spec(_spec)
sys.modules["text_encoding"] = te
_spec.loader.exec_module(te)

PROD_MOJIBAKE = "Preu\u00c3\u009fen M\u00c3\u00bcnster"  # teams b6abfa7e on prod, 2026-11-01


class TestDecode:
    def test_a_bom_file_reads_its_first_column_name_clean(self):
        text, enc = te.decode_csv_bytes("\ufeffDiv,HomeTeam\nD2,Preußen Münster\n".encode("utf-8"), "D2 2526")
        assert enc == "utf-8" and text.splitlines()[0] == "Div,HomeTeam"
        assert "Preußen Münster" in text

    def test_the_old_latin_1_read_is_what_broke(self):
        raw = "\ufeffDiv,HomeTeam\nD2,Preußen Münster\n".encode("utf-8")
        old = raw.decode("latin-1")
        assert old.startswith("ï»¿Div") and PROD_MOJIBAKE in old

    def test_a_windows_1252_file_with_a_no_break_space_reads(self):
        # 2018-19 I2 and SC0 carry one 0xA0 in an odds column.
        text, enc = te.decode_csv_bytes(b"Div,BbAHh\nSC0,8\xa0\n", "SC0 1819")
        assert enc == "cp1252" and text.endswith("8\u00a0\n")

    def test_an_ascii_file_is_utf_8(self):
        assert te.decode_csv_bytes(b"Div,HomeTeam\nE0,Man City\n", "E0") == ("Div,HomeTeam\nE0,Man City\n", "utf-8")

    def test_bytes_neither_encoding_reads_are_refused_loudly(self):
        with pytest.raises(te.UndecodableText, match=r"E0 9900: neither UTF-8 nor Windows-1252 \(byte 0x81"):
            te.decode_csv_bytes(b"Div\n\x81\x8d\n", "E0 9900")


class TestMojibake:
    @pytest.mark.parametrize(
        "text",
        [
            PROD_MOJIBAKE,
            "Atlético".encode("utf-8").decode("cp1252"),  # 'AtlÃ©tico'
            "Kärnten’s".encode("utf-8").decode("cp1252"),  # three-byte quote: 'â€™'
            "Málaga".encode("utf-8").decode("cp1252").encode("utf-8").decode("cp1252"),  # twice
            "\ufeffDiv",
            "M\ufffdnchen",
        ],
    )
    def test_wrong_decodes_are_caught(self, text):
        assert te.looks_mojibake(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Málaga",
            "São Paulo",
            "Borussia Mönchengladbach",
            "Åsane",
            "Çorum FK",
            "Ñublense",
            "Djurgården",
            "Västerås SK",
            "Örgryte IS",
            "Águilas Doradas",
            "Grêmio",
            "Deportivo La Coruña",
            "Türkiye",
            "Queen of Sth",
            "",
            None,
        ],
    )
    def test_every_non_ascii_soccer_name_on_prod_is_clean(self, text):
        # The 55 non-ASCII soccer team names on prod (2026-11-01) include these; none but Preussen is mojibake.
        assert not te.looks_mojibake(text)

    @pytest.mark.parametrize(
        "bad, good",
        [
            (PROD_MOJIBAKE, "Preußen Münster"),
            ("Atlético".encode("utf-8").decode("cp1252"), "Atlético"),
            ("Málaga".encode("utf-8").decode("cp1252").encode("utf-8").decode("cp1252"), "Málaga"),
        ],
    )
    def test_repair_undoes_the_misreading(self, bad, good):
        assert te.repair_mojibake(bad) == good

    def test_repair_leaves_clean_text_and_unrepairable_text_alone(self):
        assert te.repair_mojibake("Preußen Münster") is None
        assert te.repair_mojibake("M\ufffdnchen") is None  # the bytes are gone

    def test_the_pattern_is_ascii_and_reads_the_same_in_postgres(self):
        # \uXXXX escapes only: the same text is a Python `re` pattern and a Postgres `~` one.
        assert all(ord(c) < 128 for c in te.MOJIBAKE_PATTERN)
        assert re.compile(te.MOJIBAKE_PATTERN).search(PROD_MOJIBAKE)
