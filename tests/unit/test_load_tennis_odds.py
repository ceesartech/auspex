"""Unit tests for scripts/load_tennis_odds.py — the pure parts: name folding,
'Surname I.' parsing, ESPN <-> tennis-data player matching, the stdlib xlsx
reader, row parsing, match pairing (nearest date, ambiguity, winner
agreement) and the odds rows written per pairing."""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
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


lto = _load("load_tennis_odds_under_test", "load_tennis_odds.py")


# ── names ─────────────────────────────────────────────────────────────


def test_fold_strips_accents_apostrophes_and_punctuation():
    assert lto.fold("Félix Auger-Aliassime") == "felix auger aliassime"
    assert lto.fold("O'Connell C.") == "oconnell c"
    assert lto.fold("Jil  Teichmann ") == "jil teichmann"


@pytest.mark.parametrize(
    "raw,surname,initials,dotted",
    [
        ("Auger-Aliassime F.", ("auger", "aliassime"), "f", False),
        ("De Minaur A.", ("de", "minaur"), "a", False),
        ("Wang Xiy.", ("wang",), "xiy", False),
        ("Cerundolo J.M.", ("cerundolo",), "jm", True),
        ("Munar J", ("munar",), "j", False),
        ("Bautista Agut R.", ("bautista", "agut"), "r", False),
    ],
)
def test_parse_td_name(raw, surname, initials, dotted):
    n = lto.parse_td_name(raw)
    assert n is not None
    assert n.surname == surname
    assert n.initials == initials
    assert n.dotted is dotted


def test_parse_td_name_rejects_bare_surnames_and_blanks():
    assert lto.parse_td_name("Djokovic") is None
    assert lto.parse_td_name("") is None
    assert lto.parse_td_name("Nadal 1.") is None


@pytest.mark.parametrize(
    "espn,td,rule",
    [
        ("Felix Auger-Aliassime", "Auger-Aliassime F.", "suffix"),
        ("Alex de Minaur", "De Minaur A.", "suffix"),
        ("Ho Ching Wu", "Wu H.", "suffix"),
        ("Juan Manuel Cerundolo", "Cerundolo J.M.", "suffix"),
        ("Botic van de Zandschulp", "Van De Zandschulp B.", "suffix"),
        ("Xiyu Wang", "Wang Xiy.", "suffix"),
        ("J.J. Wolf", "Wolf J.J.", "suffix"),
        ("Zhang Shuai", "Zhang S.", "prefix"),
    ],
)
def test_match_rule_positive(espn, td, rule):
    assert lto.match_rule(lto.name_tokens(espn), lto.parse_td_name(td)) == rule


@pytest.mark.parametrize(
    "espn,td",
    [
        ("Xinyu Wang", "Wang Xiy."),  # multi-letter initials separate Xinyu from Xiyu
        ("Novak Djokovic", "Djokovic M."),  # wrong initial
        ("Andy Murray", "Murray J."),
        ("Sinner", "Sinner J."),  # ESPN name with no given name
        ("Daniil Medvedev", "Medvedeva D."),
    ],
)
def test_match_rule_negative(espn, td):
    assert lto.match_rule(lto.name_tokens(espn), lto.parse_td_name(td)) is None


# ── dates, odds sanity ────────────────────────────────────────────────


def test_excel_serial_and_string_dates():
    assert lto.excel_serial_to_date(45292.0) == date(2024, 1, 1)
    assert lto.excel_serial_to_date("2024-01-01") == date(2024, 1, 1)
    assert lto.excel_serial_to_date("31/12/2023") == date(2023, 12, 31)
    assert lto.excel_serial_to_date(None) is None
    assert lto.excel_serial_to_date("not a date") is None


def test_book_pair_requires_both_sides_and_sane_overround():
    assert lto.book_pair(1.50, 2.70) == (1.50, 2.70)
    assert lto.book_pair(1.50, None) is None
    assert lto.book_pair(1.0, 5.0) is None  # no price
    assert lto.book_pair(2.2, 2.2) is None  # negative overround (arb / bad data)
    assert lto.book_pair(1.2, 1.9) is None  # ~36% overround


def test_american_from_decimal():
    assert lto.american_from_decimal(2.5) == 150
    assert lto.american_from_decimal(1.5) == -200


# ── sheet parsing ─────────────────────────────────────────────────────


HEADER = ["ATP", "Location", "Tournament", "Date", "Winner", "Loser", "Comment", "B365W", "B365L", "PSW", "PSL",
          "MaxW", "MaxL", "AvgW", "AvgL"]  # fmt: skip


def test_rows_from_table_maps_books_and_skips_walkovers_and_unpriced():
    blank = [None] * 8
    table = [
        HEADER,
        [1, "Doha", "Qatar Open", 45292.0, "Sinner J.", "Medvedev D.", "Completed"]
        + [1.6, 2.3, 1.62, 2.45, 1.7, 2.6, 1.6, 2.35],
        [1, "Doha", "Qatar Open", 45292.0, "Rune H.", "Ruud C.", "Walkover"] + blank,
        [1, "Doha", "Qatar Open", 45293.0, "Fils A.", "Paul T.", "Completed"] + blank,
    ]
    rows, skips = lto.rows_from_table(table, "atp_2024.xlsx")
    assert len(rows) == 1
    r = rows[0]
    assert r.tour == "ATP" and r.day == date(2024, 1, 1)
    assert set(r.odds) == {"Average", "Pinnacle", "Bet365"}  # Max is never read
    assert r.odds["Pinnacle"] == (1.62, 2.45)
    assert skips["walkover"] == 1 and skips["no_odds"] == 1


def test_rows_from_table_reads_wta_and_rejects_unknown_headers():
    rows, _ = lto.rows_from_table(
        [["WTA", "Date", "Winner", "Loser", "AvgW", "AvgL"], [1, 45292.0, "Swiatek I.", "Gauff C.", 1.4, 3.0]], "w.xlsx"
    )
    assert rows[0].tour == "WTA"
    with pytest.raises(ValueError):
        lto.rows_from_table([["Foo", "Date", "Winner", "Loser"]], "bad.xlsx")


def _write_xlsx(path: Path, rows: list[list]) -> None:
    """Minimal valid xlsx: shared strings for text, numbers inline."""
    strings: list[str] = []
    sheet_rows = []
    for ri, row in enumerate(rows, start=1):
        cells = []
        for ci, v in enumerate(row):
            ref = f"{chr(65 + ci)}{ri}"
            if v is None:
                continue
            if isinstance(v, str):
                if v not in strings:
                    strings.append(v)
                cells.append(f'<c r="{ref}" t="s"><v>{strings.index(v)}</v></c>')
            else:
                cells.append(f'<c r="{ref}"><v>{v}</v></c>')
        sheet_rows.append(f'<row r="{ri}">{"".join(cells)}</row>')
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rns = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "xl/workbook.xml",
            f'<workbook {ns} {rns}><sheets><sheet name="2024" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        sst = "".join(f"<si><t>{s}</t></si>" for s in strings)
        zf.writestr("xl/sharedStrings.xml", f'<sst {ns} count="{len(strings)}">{sst}</sst>')
        zf.writestr(
            "xl/worksheets/sheet1.xml", f'<worksheet {ns}><sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
        )


def test_read_xlsx_rows_and_load_td_files(tmp_path):
    _write_xlsx(
        tmp_path / "anything.xlsx",
        [
            ["WTA", "Tournament", "Date", "Winner", "Loser", "PSW", "PSL", "AvgW", "AvgL"],
            [1, "Brisbane", 45292, "Sabalenka A.", "Andreeva M.", 1.3, 3.8, 1.28, 3.6],
            [1, "Brisbane", 45293, "Rybakina E.", None, 1.3, 3.8, 1.28, 3.6],  # missing loser
        ],
    )
    table = lto.read_xlsx_rows(tmp_path / "anything.xlsx")
    assert table[0][0] == "WTA" and table[1][3] == "Sabalenka A." and table[1][5] == 1.3
    assert table[2][4] is None  # empty cell keeps its column position
    rows, summary = lto.load_td_files(tmp_path)
    assert len(rows) == 1 and rows[0].tour == "WTA" and set(rows[0].odds) == {"Pinnacle", "Average"}
    assert summary["anything.xlsx"]["skipped"]["unparseable_name"] == 1


def test_load_td_files_without_files_explains_manual_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="alldata.php"):
        lto.load_td_files(tmp_path)


# ── pairing ───────────────────────────────────────────────────────────

UTC = timezone.utc


def _row(day, w, lo, tour="ATP", odds=None):
    return lto.TdRow(
        tour=tour,
        day=day,
        winner=lto.parse_td_name(w),
        loser=lto.parse_td_name(lo),
        tournament="T",
        odds=odds or {"Average": (1.5, 2.6), "Pinnacle": (1.52, 2.7)},
        source_file="atp_2024.xlsx",
    )


def _match(mid, when, home, away, home_won, tour="ATP", has_odds=False):
    return lto.OurMatch(
        id=mid,
        match_date=when,
        tour=tour,
        home=lto.name_tokens(home),
        away=lto.name_tokens(away),
        home_won=home_won,
        has_odds=has_odds,
        home_name=home,
        away_name=away,
    )


def test_match_rows_orients_by_name_and_checks_winner():
    m = _match("m1", datetime(2024, 1, 2, 10, tzinfo=UTC), "Daniil Medvedev", "Jannik Sinner", home_won=False)
    res = lto.match_rows([_row(date(2024, 1, 1), "Sinner J.", "Medvedev D.")], [m])
    assert len(res.pairings) == 1
    p = res.pairings[0]
    assert p.winner_is_home is False and p.winner_agrees and p.day_diff == 1


def test_match_rows_flags_winner_disagreement():
    m = _match("m1", datetime(2024, 1, 1, 10, tzinfo=UTC), "Jannik Sinner", "Daniil Medvedev", home_won=False)
    res = lto.match_rows([_row(date(2024, 1, 1), "Sinner J.", "Medvedev D.")], [m])
    assert len(res.pairings) == 1 and not res.pairings[0].winner_agrees
    assert res.agreeing == []
    rep = lto.precision_report(res)
    assert rep["suffix"]["pairs"] == 1 and rep["suffix"]["winner_agree"] == 0


def test_match_rows_respects_tour_and_date_window():
    m = _match("m1", datetime(2024, 1, 10, tzinfo=UTC), "Jannik Sinner", "Daniil Medvedev", True)
    assert lto.match_rows([_row(date(2024, 1, 1), "Sinner J.", "Medvedev D.")], [m]).unmatched == 1
    w = _match("w1", datetime(2024, 1, 1, tzinfo=UTC), "Jannik Sinner", "Daniil Medvedev", True, tour="WTA")
    assert lto.match_rows([_row(date(2024, 1, 1), "Sinner J.", "Medvedev D.")], [w]).unmatched == 1


def test_match_rows_prefers_nearest_date_and_drops_ties():
    near = _match("near", datetime(2024, 1, 1, 12, tzinfo=UTC), "Jannik Sinner", "Daniil Medvedev", True)
    far = _match("far", datetime(2024, 1, 3, 12, tzinfo=UTC), "Jannik Sinner", "Daniil Medvedev", True)
    res = lto.match_rows([_row(date(2024, 1, 1), "Sinner J.", "Medvedev D.")], [near, far])
    assert [p.match.id for p in res.pairings] == ["near"]
    twin = _match("twin", datetime(2024, 1, 1, 18, tzinfo=UTC), "Jannik Sinner", "Daniil Medvedev", True)
    res2 = lto.match_rows([_row(date(2024, 1, 1), "Sinner J.", "Medvedev D.")], [near, twin])
    assert res2.pairings == [] and res2.ambiguous_row == 1


def test_match_rows_drops_a_match_claimed_twice():
    m = _match("m1", datetime(2024, 1, 2, tzinfo=UTC), "Jannik Sinner", "Daniil Medvedev", True)
    rows = [_row(date(2024, 1, 1), "Sinner J.", "Medvedev D."), _row(date(2024, 1, 3), "Sinner J.", "Medvedev D.")]
    res = lto.match_rows(rows, [m])
    assert res.pairings == [] and res.ambiguous_match == 2


def test_match_rows_refuses_double_orientation():
    # Two players who both read as 'Pliskova K.' cannot be oriented.
    m = _match("m1", datetime(2024, 1, 1, tzinfo=UTC), "Karolina Pliskova", "Kristyna Pliskova", True, tour="WTA")
    res = lto.match_rows([_row(date(2024, 1, 1), "Pliskova K.", "Pliskova K.", tour="WTA")], [m])
    assert res.pairings == [] and res.unmatched == 1


# ── rows written ──────────────────────────────────────────────────────


def test_odds_rows_for_maps_sides_orders_books_and_tags_source():
    when = datetime(2024, 1, 2, 10, tzinfo=UTC)
    m = _match("m1", when, "Daniil Medvedev", "Jannik Sinner", home_won=False)
    p = lto.match_rows([_row(date(2024, 1, 2), "Sinner J.", "Medvedev D.")], [m]).pairings[0]
    rows = lto.odds_rows_for(p, "run1")
    assert len(rows) == 4  # Average + Pinnacle, two sides each
    avg = {r["selection"]: r for r in rows if r["bookmaker"] == "Average"}
    assert avg["away"]["odds_decimal"] == 1.5 and avg["home"]["odds_decimal"] == 2.6  # winner (Sinner) is away
    ts = {r["bookmaker"]: r["timestamp"] for r in rows}
    assert ts["Average"] > ts["Pinnacle"]  # the feature builder reads the newest row: Average first
    assert all(r["timestamp"] < when for r in rows)
    meta = json.loads(rows[0]["metadata"])
    assert meta["source"] == "tennis-data.co.uk" and meta["run_id"] == "run1"
    assert {r["bookmaker"] for r in rows} <= {"Average", "Pinnacle", "Bet365"}
    assert ts["Average"] - ts["Pinnacle"] == timedelta(minutes=1)


def test_accepted_pairings_gates_prefix_on_measured_precision():
    when = datetime(2024, 1, 1, 10, tzinfo=UTC)
    m = _match("m1", when, "Zhang Shuai", "Jil Teichmann", home_won=True, tour="WTA")
    res = lto.match_rows([_row(date(2024, 1, 1), "Zhang S.", "Teichmann J.", tour="WTA")], [m])
    assert res.pairings and "prefix" in res.pairings[0].rules
    prec = lto.precision_report(res)
    kept, gates = lto.accepted_pairings(res, prec)
    assert kept == [] and gates["prefix_included"] is False  # 1 pair < MIN_PREFIX_PAIRS
    kept, gates = lto.accepted_pairings(res, prec, allow_prefix=True)
    assert len(kept) == 1 and gates["prefix_included_by"] == "flag"
    big = {"prefix": {"pairs": 50, "agree_rate": 0.99}, "suffix": {"agree_rate": 0.995}}
    kept, gates = lto.accepted_pairings(res, big)
    assert len(kept) == 1 and gates["prefix_included_by"] == "measured" and gates["suffix_gate_passed"]
    low = {"suffix": {"agree_rate": 0.9}}
    assert lto.accepted_pairings(res, low)[1]["suffix_gate_passed"] is False
