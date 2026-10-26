"""Load historical tennis CLOSING odds from tennis-data.co.uk into `odds`.

Why: the tennis moneyline model trains on NEUTRAL_DEFAULT odds for ~99% of
its corpus (27,243 / 27,517 finished matches on 2026-10-25) — ESPN gives us
results, and the-odds-api only ever priced the matches the live system saw.
The model has therefore never seen a market price, which is the binding
constraint the 2026-09 audit found for both 1v1 sports. tennis-data.co.uk
publishes one workbook per tour per year (ATP main tour, WTA main tour) with
per-match closing odds: B365W/L (Bet365), PSW/L (Pinnacle), MaxW/L, AvgW/L.

Source files. tennis-data.co.uk sits behind a Cloudflare bot check, so an
automated download usually gets HTTP 403 (verified 2026-10-25). Do NOT work
around that. Download the workbooks in a browser from
http://www.tennis-data.co.uk/alldata.php and drop them in one directory;
any file name works because the tour is read from the header ('ATP' or 'WTA'
column) and the year from the dates. `--download` tries a plain, honestly
identified fetch first and fails loudly with these instructions on a block.

Matching (the hard part — ESPN and tennis-data name players differently):
  * tennis-data names a player 'Surname I.' ('Auger-Aliassime F.',
    'De Minaur A.', 'Wang Xiy.', 'Cerundolo J.M.'); ESPN uses 'Given Surname'
    ('Felix Auger-Aliassime', 'Alex de Minaur', 'Xiyu Wang').
  * A player matches when the tennis-data surname tokens equal a SUFFIX of
    the ESPN tokens (accent/case/punctuation folded, hyphens split) and the
    first initial equals the first given-name letter. A multi-letter initial
    without dots ('Xiy') must prefix the joined given names, which separates
    'Wang Xiy.' from 'Wang Xin.'. A surname-first fallback ('Zhang Shuai')
    is reported separately so its precision can be judged on its own.
  * A tennis-data row matches one of OUR finished matches when both players
    match (either orientation), the tours agree, and the dates are within
    [-1, +2] days (tennis-data dates are local and record the scheduled day).
    The nearest date wins; a tie, or one of our matches claimed by two rows,
    is dropped as ambiguous.
  * PRECISION CHECK: the stored winner must agree with the file's winner.
    A wrong pairing agrees by chance half the time, so the disagreement rate
    among name-matched pairs estimates the false-match rate among accepted
    ones. It is printed on every run; rows that disagree are never loaded.

Writes (FINISHED matches only, additive only):
  * market_type='moneyline', selection home/away, line NULL, is_live=false,
    is_opening=false, one row per book per side for Average (AvgW/L),
    Pinnacle (PSW/L) and Bet365 (B365W/L). Max is NOT stored — an average
    over books that includes the best price is biased long.
  * A book is written only when it prices BOTH sides with a sane overround
    (0 <= 1/w + 1/l - 1 <= MAX_OVERROUND).
  * Matches that already carry any pre-match moneyline row (the live system
    priced them) are skipped entirely, so live prices are never mixed with or
    shadowed by these.
  * compute_features_tennis.fetch_closing_odds reads ONE row per selection —
    the newest `timestamp`. Rows are stamped match_date - 1 min (Average),
    - 2 min (Pinnacle), - 3 min (Bet365), so features read the Average book
    consistently for both sides and fall back to Pinnacle, then Bet365. The
    training query's AVG(odds_decimal) averages the three.
  * Every row carries metadata {"source": "tennis-data.co.uk", ...}.
    ROLLBACK:  DELETE FROM odds WHERE metadata->>'source' = 'tennis-data.co.uk';
    then recompute tennis features (compute_features_tennis.py --all-finished).

Usage (inside the api container):

    python /app/scripts/load_tennis_odds.py --files-dir /tmp/tennis_data --dry-run
    python /app/scripts/load_tennis_odds.py --files-dir /tmp/tennis_data --report-csv /tmp/td_matches.csv
    python /app/scripts/load_tennis_odds.py --files-dir /tmp/tennis_data --recompute-features
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import random
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("load_tennis_odds")

SOURCE_TAG = "tennis-data.co.uk"
BASE_URL = "http://www.tennis-data.co.uk"
DOWNLOAD_HELP = (
    "tennis-data.co.uk refused the automated download (it sits behind a Cloudflare bot check). "
    "Download the yearly ATP and WTA workbooks in a browser from http://www.tennis-data.co.uk/alldata.php, "
    "put them in one directory (any names; the tour is read from each file's header) and re-run with "
    "--files-dir <dir>."
)

# (bookmaker name stored in `odds`, winner column, loser column, minutes before match_date).
# The smallest offset is the NEWEST row, which is what compute_features_tennis reads.
BOOKS: tuple[tuple[str, str, str, int], ...] = (
    ("Average", "AvgW", "AvgL", 1),
    ("Pinnacle", "PSW", "PSL", 2),
    ("Bet365", "B365W", "B365L", 3),
)
DATE_WINDOW_DAYS = (-1, 2)
MAX_OVERROUND = 0.30
# Precision gates, measured on every run from winner agreement (see docstring).
# Below MIN_SUFFIX_AGREE the run refuses to write at all; surname-first
# ('prefix') pairings are loaded only when their own measured agreement clears
# MIN_PREFIX_AGREE on at least MIN_PREFIX_PAIRS pairs (or --allow-prefix).
MIN_SUFFIX_AGREE = 0.97
MIN_PREFIX_AGREE = 0.98
MIN_PREFIX_PAIRS = 30
MIN_DECIMAL_ODDS = 1.001

# ---------------------------------------------------------------------------
# Name handling (pure)
# ---------------------------------------------------------------------------

_APOSTROPHES = "'`’‘´"


def fold(text: str) -> str:
    """Lowercase, strip accents and apostrophes, turn '.', '-' and other
    punctuation into spaces, collapse whitespace."""
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    for a in _APOSTROPHES:
        s = s.replace(a, "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def name_tokens(full_name: str) -> tuple[str, ...]:
    """'Felix Auger-Aliassime' -> ('felix', 'auger', 'aliassime')."""
    return tuple(fold(full_name).split())


@dataclass(frozen=True)
class TdName:
    """A tennis-data 'Surname I.' name, parsed."""

    raw: str
    surname: tuple[str, ...]
    initials: str  # folded letters of the initials token, '' when absent
    dotted: bool  # True for 'J.L.'-style initials (one letter per given name)

    @property
    def first_initial(self) -> str:
        return self.initials[:1]


def parse_td_name(raw: str) -> Optional[TdName]:
    """Parse 'Auger-Aliassime F.' / 'Wang Xiy.' / 'Cerundolo J.M.' / 'Munar J'.

    The LAST whitespace token is the initials; everything before it is the
    surname. Returns None for an empty or single-token name (a bare surname
    cannot be matched with measurable precision)."""
    parts = (raw or "").strip().split()
    if len(parts) < 2:
        return None
    init_token = parts[-1]
    surname = tuple(fold(" ".join(parts[:-1])).split())
    letters = fold(init_token).replace(" ", "")
    if not surname or not letters or not letters.isalpha():
        return None
    inner = init_token.rstrip(".")
    dotted = "." in inner
    return TdName(raw=raw.strip(), surname=surname, initials=letters, dotted=dotted)


def _initials_agree(td: TdName, given: tuple[str, ...]) -> bool:
    if not given:
        return False
    if td.first_initial != given[0][:1]:
        return False
    if not td.dotted and len(td.initials) > 1:
        # 'Xiy' must prefix the joined given names ('xiyu'), not just share 'x'.
        return "".join(given).startswith(td.initials)
    return True


def match_rule(full: tuple[str, ...], td: TdName) -> Optional[str]:
    """Return 'suffix' / 'prefix' when the ESPN tokens `full` name the same
    player as `td`, else None.

    suffix: Western order — surname tokens are the tail of the ESPN name.
    prefix: surname-first order ('Zhang Shuai') — reported separately."""
    n, k = len(full), len(td.surname)
    if n <= k:
        return None
    if full[n - k :] == td.surname and _initials_agree(td, full[: n - k]):
        return "suffix"
    if full[:k] == td.surname and _initials_agree(td, full[k:]):
        return "prefix"
    return None


# ---------------------------------------------------------------------------
# Minimal stdlib .xlsx reader (the api image has no openpyxl)
# ---------------------------------------------------------------------------

_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}
_EXCEL_EPOCH = date(1899, 12, 30)


def _col_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def _first_sheet_path(zf: zipfile.ZipFile) -> str:
    try:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet = wb.find("m:sheets/m:sheet", _NS)
        rid = sheet.get(f"{{{_NS['r']}}}id") if sheet is not None else None
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        for rel in rels.findall("pr:Relationship", _NS):
            if rel.get("Id") == rid:
                target = rel.get("Target", "")
                target = target.lstrip("/")
                return target if target.startswith("xl/") else f"xl/{target}"
    except (KeyError, ET.ParseError, AttributeError):
        pass
    return "xl/worksheets/sheet1.xml"


def read_xlsx_rows(path: str | Path) -> list[list]:
    """Rows of the first worksheet as lists (None for empty cells). Numbers
    come back as float, strings as str. Dates stay Excel serials — callers
    convert with `excel_serial_to_date`."""
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sst.findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        sheet = ET.fromstring(zf.read(_first_sheet_path(zf)))
    rows: list[list] = []
    for row in sheet.iter(f"{{{_NS['m']}}}row"):
        values: dict[int, object] = {}
        for c in row.findall("m:c", _NS):
            ref = c.get("r", "")
            col = _col_index(ref) if ref else len(values)
            t = c.get("t")
            v = c.find("m:v", _NS)
            if t == "inlineStr":
                values[col] = "".join(x.text or "" for x in c.iter(f"{{{_NS['m']}}}t"))
            elif v is None or v.text is None:
                continue
            elif t == "s":
                values[col] = shared[int(v.text)]
            elif t in ("str", "e"):
                values[col] = v.text
            elif t == "b":
                values[col] = v.text == "1"
            else:
                try:
                    values[col] = float(v.text)
                except ValueError:
                    values[col] = v.text
        if values:
            width = max(values) + 1
            rows.append([values.get(i) for i in range(width)])
    return rows


def excel_serial_to_date(value) -> Optional[date]:
    """Excel serial (1900 system) or an ISO-ish / dd/mm/yyyy string -> date."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return _EXCEL_EPOCH + timedelta(days=int(value))
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# tennis-data rows
# ---------------------------------------------------------------------------


@dataclass
class TdRow:
    tour: str  # 'ATP' / 'WTA'
    day: date
    winner: TdName
    loser: TdName
    tournament: str
    odds: dict[str, tuple[float, float]]  # book -> (winner odds, loser odds)
    source_file: str
    comment: str = ""


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN guard


def book_pair(w: Optional[float], l_: Optional[float]) -> Optional[tuple[float, float]]:
    """(w, l) when both prices are real and the overround is sane, else None."""
    if w is None or l_ is None or w < MIN_DECIMAL_ODDS or l_ < MIN_DECIMAL_ODDS:
        return None
    overround = 1.0 / w + 1.0 / l_ - 1.0
    if overround < 0.0 or overround > MAX_OVERROUND:
        return None
    return (w, l_)


def rows_from_table(table: list[list], source_file: str) -> tuple[list[TdRow], Counter]:
    """Turn a tennis-data sheet (header row first) into TdRows. Returns the
    rows plus a Counter of skip reasons."""
    skips: Counter = Counter()
    if not table:
        return [], skips
    header = [str(h).strip() if h is not None else "" for h in table[0]]
    col = {h: i for i, h in enumerate(header) if h}
    if "ATP" in col:
        tour = "ATP"
    elif "WTA" in col:
        tour = "WTA"
    else:
        raise ValueError(f"{source_file}: header has neither an 'ATP' nor a 'WTA' column: {header[:8]}")
    for needed in ("Date", "Winner", "Loser"):
        if needed not in col:
            raise ValueError(f"{source_file}: missing column {needed!r}")

    def cell(r: list, name: str):
        i = col.get(name)
        return r[i] if i is not None and i < len(r) else None

    out: list[TdRow] = []
    for r in table[1:]:
        day = excel_serial_to_date(cell(r, "Date"))
        w = parse_td_name(str(cell(r, "Winner") or ""))
        lo = parse_td_name(str(cell(r, "Loser") or ""))
        if day is None:
            skips["no_date"] += 1
            continue
        if w is None or lo is None:
            skips["unparseable_name"] += 1
            continue
        comment = str(cell(r, "Comment") or "").strip()
        if comment.lower().startswith("walkover"):
            skips["walkover"] += 1
            continue
        odds: dict[str, tuple[float, float]] = {}
        for book, wc, lc, _ in BOOKS:
            pair = book_pair(_to_float(cell(r, wc)), _to_float(cell(r, lc)))
            if pair:
                odds[book] = pair
        if not odds:
            skips["no_odds"] += 1
            continue
        out.append(
            TdRow(
                tour=tour,
                day=day,
                winner=w,
                loser=lo,
                tournament=str(cell(r, "Tournament") or ""),
                odds=odds,
                source_file=source_file,
                comment=comment,
            )
        )
    return out, skips


def load_td_files(files_dir: str | Path, years: Optional[set[int]] = None) -> tuple[list[TdRow], dict]:
    """Read every .xlsx / .csv under files_dir. Returns rows (optionally
    filtered to `years`) and a per-file summary."""
    rows: list[TdRow] = []
    summary: dict = {}
    paths = sorted(p for p in Path(files_dir).rglob("*") if p.suffix.lower() in (".xlsx", ".csv"))
    if not paths:
        raise FileNotFoundError(f"no .xlsx/.csv files under {files_dir}. {DOWNLOAD_HELP}")
    for p in paths:
        if p.suffix.lower() == ".csv":
            with open(p, newline="", encoding="utf-8-sig", errors="replace") as fh:
                table = [list(r) for r in csv.reader(fh)]
        else:
            table = read_xlsx_rows(p)
        file_rows, skips = rows_from_table(table, p.name)
        if years:
            file_rows = [r for r in file_rows if r.day.year in years]
        rows.extend(file_rows)
        summary[p.name] = {"rows_with_odds": len(file_rows), "skipped": dict(skips)}
        logger.info("%s: %d rows with odds (%s)", p.name, len(file_rows), dict(skips))
    return rows, summary


# ---------------------------------------------------------------------------
# Matching (pure given the match list)
# ---------------------------------------------------------------------------


@dataclass
class OurMatch:
    id: str
    match_date: datetime
    tour: str
    home: tuple[str, ...]
    away: tuple[str, ...]
    home_won: bool
    has_odds: bool
    home_name: str = ""
    away_name: str = ""


@dataclass
class Pairing:
    row: TdRow
    match: OurMatch
    winner_is_home: bool  # orientation implied by NAMES
    rules: tuple[str, str]
    day_diff: int

    @property
    def winner_agrees(self) -> bool:
        return self.winner_is_home == self.match.home_won


@dataclass
class MatchResult:
    pairings: list[Pairing] = field(default_factory=list)
    unmatched: int = 0
    ambiguous_row: int = 0
    ambiguous_match: int = 0

    @property
    def agreeing(self) -> list[Pairing]:
        return [p for p in self.pairings if p.winner_agrees]


def tour_of_league(league_name: str) -> str:
    n = (league_name or "").upper()
    return "WTA" if "WTA" in n else "ATP"


def _orient(m: OurMatch, row: TdRow) -> Optional[tuple[bool, tuple[str, str]]]:
    """(winner_is_home, (winner rule, loser rule)), or None when the names do
    not fit — or fit BOTH ways (e.g. two players sharing surname + initial),
    which cannot be oriented safely."""
    hw, al = match_rule(m.home, row.winner), match_rule(m.away, row.loser)
    hl, aw = match_rule(m.home, row.loser), match_rule(m.away, row.winner)
    straight, crossed = bool(hw and al), bool(hl and aw)
    if straight and not crossed:
        return True, (hw, al)
    if crossed and not straight:
        return False, (aw, hl)
    return None


def match_rows(rows: Iterable[TdRow], matches: Iterable[OurMatch]) -> MatchResult:
    """Pair tennis-data rows with our finished matches (see module docstring)."""
    by_day: dict[tuple[str, date], list[OurMatch]] = defaultdict(list)
    for m in matches:
        by_day[(m.tour, m.match_date.date())].append(m)
    result = MatchResult()
    claimed: dict[str, list[Pairing]] = defaultdict(list)
    lo, hi = DATE_WINDOW_DAYS
    for row in rows:
        found: list[Pairing] = []
        for off in range(lo, hi + 1):
            for m in by_day.get((row.tour, row.day + timedelta(days=off)), ()):
                o = _orient(m, row)
                if o:
                    found.append(Pairing(row=row, match=m, winner_is_home=o[0], rules=o[1], day_diff=abs(off)))
        if not found:
            result.unmatched += 1
            continue
        best = min(p.day_diff for p in found)
        nearest = [p for p in found if p.day_diff == best]
        if len({p.match.id for p in nearest}) > 1:
            result.ambiguous_row += 1
            continue
        claimed[nearest[0].match.id].append(nearest[0])
    for mid, ps in claimed.items():
        if len(ps) > 1:
            result.ambiguous_match += len(ps)
            continue
        result.pairings.append(ps[0])
    return result


def precision_report(result: MatchResult) -> dict:
    """Winner-agreement by rule. A wrong pairing agrees with probability ~0.5,
    so est_false_accepted ~= disagreeing pairs."""
    by_rule: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in result.pairings:
        key = "prefix" if "prefix" in p.rules else "suffix"
        by_rule[key][0] += 1
        by_rule[key][1] += int(p.winner_agrees)
    rep = {}
    for k, (n, agree) in by_rule.items():
        dis = n - agree
        rep[k] = {
            "pairs": n,
            "winner_agree": agree,
            "agree_rate": round(agree / n, 4) if n else None,
            "est_false_among_accepted": dis,
            "est_precision_accepted": round(1 - dis / agree, 4) if agree else None,
        }
    return rep


def accepted_pairings(result: MatchResult, prec: dict, allow_prefix: bool = False) -> tuple[list[Pairing], dict]:
    """Winner-agreeing pairings that clear the precision gates. Returns
    (pairings, gate decisions)."""
    suffix = prec.get("suffix") or {}
    prefix = prec.get("prefix") or {}
    prefix_ok = (prefix.get("pairs") or 0) >= MIN_PREFIX_PAIRS and (prefix.get("agree_rate") or 0) >= MIN_PREFIX_AGREE
    use_prefix = bool(allow_prefix or prefix_ok)
    gates = {
        "suffix_agree_rate": suffix.get("agree_rate"),
        "suffix_gate_passed": (suffix.get("agree_rate") or 0) >= MIN_SUFFIX_AGREE,
        "prefix_included": use_prefix,
        "prefix_included_by": "flag" if allow_prefix else ("measured" if prefix_ok else None),
    }
    return [p for p in result.agreeing if use_prefix or "prefix" not in p.rules], gates


# ---------------------------------------------------------------------------
# Odds rows (pure)
# ---------------------------------------------------------------------------


def american_from_decimal(d: float) -> int:
    return int(round((d - 1.0) * 100)) if d >= 2.0 else int(round(-100.0 / (d - 1.0)))


def odds_rows_for(p: Pairing, run_id: str) -> list[dict]:
    """Two rows (home, away) per book that priced both sides."""
    out: list[dict] = []
    for book, _, _, minutes in BOOKS:
        pair = p.row.odds.get(book)
        if not pair:
            continue
        w, lo = pair
        home_odds, away_odds = (w, lo) if p.winner_is_home else (lo, w)
        ts = p.match.match_date - timedelta(minutes=minutes)
        meta = {
            "source": SOURCE_TAG,
            "loader": "load_tennis_odds",
            "run_id": run_id,
            "file": p.row.source_file,
            "tournament": p.row.tournament,
            "td_date": p.row.day.isoformat(),
        }
        for sel, o in (("home", home_odds), ("away", away_odds)):
            out.append(
                {
                    "match_id": p.match.id,
                    "bookmaker": book,
                    "selection": sel,
                    "odds_decimal": round(float(o), 4),
                    "odds_american": american_from_decimal(float(o)),
                    "implied_probability": round(1.0 / float(o), 4),
                    "timestamp": ts,
                    "metadata": json.dumps(meta),
                }
            )
    return out


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------

MATCHES_SQL = """
    SELECT m.id::text AS id, m.match_date, l.name AS league,
           ht.name AS home_name, at.name AS away_name,
           (m.home_score > m.away_score) AS home_won,
           EXISTS (
               SELECT 1 FROM odds o
               WHERE o.match_id = m.id AND o.market_type = 'moneyline'
                 AND COALESCE(o.is_live, false) = false
           ) AS has_odds
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON ht.id = m.home_team_id
    JOIN teams at ON at.id = m.away_team_id
    WHERE l.sport = 'tennis' AND m.status = 'finished'
      AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
      AND m.home_score <> m.away_score
"""

INSERT_SQL = """
    INSERT INTO odds (match_id, bookmaker, market_type, selection, odds_decimal, odds_american,
                      line, implied_probability, timestamp, is_opening, is_live, metadata, last_seen_at)
    SELECT %(match_id)s, %(bookmaker)s, 'moneyline', %(selection)s, %(odds_decimal)s, %(odds_american)s,
           NULL, %(implied_probability)s, %(timestamp)s, false, false, %(metadata)s::jsonb, %(timestamp)s
    WHERE NOT EXISTS (
        SELECT 1 FROM odds
        WHERE match_id = %(match_id)s AND bookmaker = %(bookmaker)s
          AND market_type = 'moneyline' AND selection = %(selection)s
          AND line IS NULL
          AND COALESCE(is_opening, false) = false
          AND COALESCE(is_live, false) = false
    )
"""


def fetch_our_matches(cur) -> list[OurMatch]:
    cur.execute(MATCHES_SQL)
    out = []
    for r in cur.fetchall():
        md = r["match_date"]
        if md.tzinfo is None:
            md = md.replace(tzinfo=timezone.utc)
        out.append(
            OurMatch(
                id=r["id"],
                match_date=md.astimezone(timezone.utc),
                tour=tour_of_league(r["league"]),
                home=name_tokens(r["home_name"]),
                away=name_tokens(r["away_name"]),
                home_won=bool(r["home_won"]),
                has_odds=bool(r["has_odds"]),
                home_name=r["home_name"],
                away_name=r["away_name"],
            )
        )
    return out


def insert_rows(cur, rows: list[dict]) -> int:
    from psycopg2.extras import execute_batch

    before = cur.connection.cursor()
    before.execute("SELECT count(*) FROM odds WHERE metadata->>'source' = %s", (SOURCE_TAG,))
    n0 = before.fetchone()[0]
    execute_batch(cur, INSERT_SQL, rows, page_size=500)
    before.execute("SELECT count(*) FROM odds WHERE metadata->>'source' = %s", (SOURCE_TAG,))
    n1 = before.fetchone()[0]
    return int(n1 - n0)


def default_odds_rate(cur) -> dict:
    """Share of finished tennis matches whose newest tennis_baseline row still
    carries the NEUTRAL_DEFAULT moneyline (odds_home_ml = 1.54, away 2.60)."""
    cur.execute(
        """
        WITH f AS (
            SELECT DISTINCT ON (fc.match_id) fc.match_id, fc.features
            FROM features_cache fc
            JOIN matches m ON m.id = fc.match_id
            JOIN leagues l ON l.id = m.league_id
            WHERE l.sport = 'tennis' AND m.status = 'finished' AND fc.feature_set = 'tennis_baseline'
            ORDER BY fc.match_id, fc.computed_at DESC
        )
        SELECT count(*) AS n,
               count(*) FILTER (WHERE (features->>'odds_home_ml')::float = 1.54
                                  AND (features->>'odds_away_ml')::float = 2.60) AS at_default
        FROM f
        """
    )
    r = cur.fetchone()
    n, d = int(r["n"]), int(r["at_default"])
    return {"finished_with_features": n, "at_default": d, "rate": round(d / n, 4) if n else None}


def _load_compute_features_tennis():
    path = Path(__file__).resolve().parent / "compute_features_tennis.py"
    spec = importlib.util.spec_from_file_location("_compute_features_tennis", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_compute_features_tennis"] = mod
    spec.loader.exec_module(mod)
    return mod


def try_download(files_dir: Path, years: list[int]) -> None:
    """Plain, honestly identified fetch. A Cloudflare block is NOT worked
    around — the run aborts with manual-download instructions."""
    import requests

    files_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "auspex-load-tennis-odds/1.0 (+personal research)"}
    for y in years:
        for tour, path in (("atp", f"{y}/{y}.xlsx"), ("wta", f"{y}w/{y}.xlsx")):
            dest = files_dir / f"{tour}_{y}.xlsx"
            if dest.exists():
                continue
            r = requests.get(f"{BASE_URL}/{path}", headers=headers, timeout=60)
            if r.status_code != 200 or not r.content.startswith(b"PK"):
                raise RuntimeError(f"{BASE_URL}/{path}: HTTP {r.status_code}. {DOWNLOAD_HELP}")
            dest.write_bytes(r.content)
            logger.info("downloaded %s (%d bytes)", dest, len(r.content))


def write_report(path: str, result: MatchResult) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "match_id",
                "match_date",
                "tour",
                "home",
                "away",
                "td_date",
                "td_winner",
                "td_loser",
                "tournament",
                "winner_is_home",
                "winner_agrees",
                "rules",
                "day_diff",
                "has_live_odds",
                "books",
            ]
        )
        for p in result.pairings:
            w.writerow(
                [
                    p.match.id,
                    p.match.match_date.isoformat(),
                    p.match.tour,
                    p.match.home_name,
                    p.match.away_name,
                    p.row.day.isoformat(),
                    p.row.winner.raw,
                    p.row.loser.raw,
                    p.row.tournament,
                    p.winner_is_home,
                    p.winner_agrees,
                    "/".join(p.rules),
                    p.day_diff,
                    p.match.has_odds,
                    "/".join(sorted(p.row.odds)),
                ]
            )


def run(args: argparse.Namespace) -> dict:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    years = sorted({int(y) for y in args.years.split(",")}) if args.years else None
    files_dir = Path(args.files_dir)
    if args.download:
        try_download(files_dir, years or list(range(2021, datetime.now(timezone.utc).year + 1)))
    td_rows, file_summary = load_td_files(files_dir, set(years) if years else None)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    with psycopg2.connect(args.database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            ours = fetch_our_matches(cur)
            logger.info(
                "our finished tennis matches: %d (%d already carry moneyline odds)",
                len(ours),
                sum(m.has_odds for m in ours),
            )
            result = match_rows(td_rows, ours)
            prec = precision_report(result)
            accepted, gates = accepted_pairings(result, prec, args.allow_prefix)
            to_load = [p for p in accepted if not p.match.has_odds]
            rows = [r for p in to_load for r in odds_rows_for(p, run_id)]

            by_year: Counter = Counter(p.match.match_date.year for p in to_load)
            ours_by_year: Counter = Counter(m.match_date.year for m in ours)
            summary = {
                "run_id": run_id,
                "dry_run": bool(args.dry_run),
                "td_rows_with_odds": len(td_rows),
                "our_finished_matches": len(ours),
                "paired": len(result.pairings),
                "unmatched_td_rows": result.unmatched,
                "ambiguous_td_rows": result.ambiguous_row,
                "ambiguous_our_matches_dropped": result.ambiguous_match,
                "precision": prec,
                "gates": gates,
                "accepted_winner_agrees": len(accepted),
                "skipped_already_priced": len(accepted) - len(to_load),
                "matches_to_load": len(to_load),
                "odds_rows_to_insert": len(rows),
                "coverage_by_year": {
                    str(y): {"ours": ours_by_year[y], "loaded": by_year.get(y, 0)} for y in sorted(ours_by_year)
                },
                "files": file_summary,
            }
            if args.sample:
                sample = random.Random(7).sample(result.pairings, min(args.sample, len(result.pairings)))
                summary["sample"] = [
                    f"{p.row.day} {p.row.winner.raw} d. {p.row.loser.raw} -> "
                    f"{p.match.match_date.date()} {p.match.home_name} v {p.match.away_name} "
                    f"(winner_is_home={p.winner_is_home}, agrees={p.winner_agrees}, rules={p.rules})"
                    for p in sample
                ]
            if args.report_csv:
                write_report(args.report_csv, result)
                summary["report_csv"] = args.report_csv

            summary["default_odds_rate_before"] = default_odds_rate(cur)
            if args.dry_run:
                conn.rollback()
                return summary
            if not gates["suffix_gate_passed"] and not args.force_low_precision:
                raise RuntimeError(
                    f"precision gate: winner agreement {gates['suffix_agree_rate']} < {MIN_SUFFIX_AGREE} — "
                    "refusing to write. Inspect --report-csv; override only with --force-low-precision."
                )

            inserted = insert_rows(cur, rows)
            conn.commit()
            summary["odds_rows_inserted"] = inserted

    if args.recompute_features and to_load:
        cft = _load_compute_features_tennis()
        ids = sorted({p.match.id for p in to_load})
        summary["features_recomputed"] = cft.run(args.database_url, 0, False, False, ids)
        with psycopg2.connect(args.database_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                summary["default_odds_rate_after"] = default_odds_rate(cur)
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files-dir", required=True, help="directory holding the tennis-data .xlsx/.csv workbooks")
    p.add_argument("--years", help="comma-separated match years to load (default: every year in the files)")
    p.add_argument("--download", action="store_true", help="try a plain download first (fails loudly when blocked)")
    p.add_argument("--dry-run", action="store_true", help="match and report; write nothing")
    p.add_argument("--allow-prefix", action="store_true", help="also load surname-first ('prefix') pairings")
    p.add_argument("--force-low-precision", action="store_true", help="write even below MIN_SUFFIX_AGREE")
    p.add_argument("--recompute-features", action="store_true", help="recompute tennis features for loaded matches")
    p.add_argument("--report-csv", help="write every pairing to this CSV")
    p.add_argument("--sample", type=int, default=20, help="random pairings to print for eyeballing")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    summary = run(args)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
