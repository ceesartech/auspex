"""Promote raw_football_data rows into the canonical schema.

raw_football_data is what scripts/load_football_data.py writes — flat
CSV-style rows direct from football-data.co.uk. This script translates
each row into the canonical schema the training pipeline expects:

    leagues       <- one row per football-data league code
    teams         <- one row per unique home/away team string (normalized)
    matches       <- one row per fixture (home_team, away_team, match_date)
    match_stats   <- two rows per fixture (home + away shooting/discipline)
    odds          <- closing 1x2 + over/under 2.5 from B365, Pinnacle, Avg

Idempotent: re-running only inserts rows that aren't already present.
Uses ON CONFLICT on the existing UNIQUE constraints in 001_create_schema.

A raw row without a full-time score is not promoted (see raw_result): it
is an abandoned, awarded or unplayed fixture, never a finished match, and
its HTHG/HTAG can hold the awarded score rather than a half-time one. A
re-run never downgrades a row already written (see INSERT_MATCH_SQL).

A fixture that already has a row for the same home and away team ids within
3 h, on football-data's date (the ESPN results path's, at ESPN's kickoff),
is written onto that row, never beside it (find_existing_matches,
update_existing_match), and an ESPN row's score is never overwritten. Nor
is a stale fixture row (unfinished, no ESPN identity) ever finished here:
that would grade its predictions and settle its recs from a run nobody
reviewed. A raw row refused that way, or written onto an ESPN row while
football-data's own finished row for the pair still sits beside it (a
duplicate), is logged at ERROR, and the run exits 3 once everything else is
committed.

The run refuses to start while any row promote_raw wrote at the legacy
kickoff (football-data's UK time read as UTC, before 2026-10-31) is left
(check_legacy_kickoffs): scripts/ops/repair_football_data_kickoffs.sh moves
or merges them.

A raw row is never promoted, and never decides a team's league, when its
league code is not in LEAGUE_MAP, when its Div (football-data's own division
code) names another division than the code it was loaded under, or when a
team name looks mojibake (screen_raw_rows, before any league or team is
written): each is logged at ERROR, counted (refused_raw), and the run exits
3 once everything else is committed. Until 2026-11-01 an unknown code was a
warning and the rest was promoted as filed: ten Scottish fixtures loaded as
SP1/SP2 2026-27 (their Div was hidden by a byte-order mark) became La Liga /
La Liga 2 rows and set twenty clubs' league to Spain, and 'PreuÃ\\x9fen
MÃ¼nster' became a team (scripts/ops/repair_soccer_labels.sh repairs both;
the loaders now refuse such files, load_football_data.parse_csv). The screen
catches the 68 mis-decoded Preussen Muenster rows, but not those ten: their
Div is NULL (the byte-order mark hid it at load time), so nothing in the row
contradicts its SP1/SP2 code until the repair relabels it. Until the repair
has run, a run therefore commits everything else and exits 3 on the 68
(the soccer repair runbook runs the repair, step 13, before its promote_raw
step, 14).

A re-run writes only what changes: a matches row whose status, scores and
half-time already equal football-data's is not updated (until 2026-11-01
every run rewrote all ~168k of them, and their match_stats and teams rows,
for nothing), and an equal match_stats row, a league that already carries
its football-data code and a team that already exists are not even sent to
the database, so a run neither rewrites nor locks them (an INSERT ... ON
CONFLICT DO UPDATE locks the row it conflicts with even when its WHERE
skips the update, and under REPEATABLE READ it fails on a row another
session changed after the snapshot: the pipeline's ensure_league rewrites
the football-data league rows on every soccer sweep). An existing team
keeps its league: only a missing one is filled (as fetch_upcoming's
INSERT_TEAM_SQL does). The league a raw name is first seen under in one
run is arbitrary (an unordered scan), and the five Scottish Championship
rows football-data filed as SP1 (above: the screen cannot see them) would
otherwise re-file their clubs under La Liga on every run, and after the
repair move the six established clubs from SC1/SC2 to SC0.

CHECKED MODE (--dry-run, or --apply --expect-plan): the whole run is one
REPEATABLE READ transaction, and what it changed is read back before the
end (rows created or updated in this transaction, and match_stats compared
with a snapshot taken first): the PLAN line counts matches, match_stats,
odds, teams and leagues inserted or changed, the predictions and
recommendations the outcome trigger graded or settled (a scheduled row
finished), the raw rows refused or found duplicated, and the raw rows the
screen refused (refused_raw). --dry-run then rolls everything back;
--apply commits only when every PLAN key equals --expect-plan (the reviewed
dry run's counts: a screened row the pin does not expect refuses the apply
like any other drift), otherwise it rolls back and exits 1. Every inserted
or changed row goes to a write-once CSV (--out-csv) first, its old values
beside the new: the reversal record.
scripts/ops/promote_raw_after_repair.sh (runbook step 14) drives it.

EXIT STATUS: 0 done; 1 refused before writing (a legacy kickoff left) or a
checked run that wrote nothing (refused, rolled back, or failed: its one
transaction is never committed) — but an unexpected error in a plain run
(Python's own exit 1) leaves the batches it committed before the error in
place (it commits every --batch-size matches); 2 a usage error; 3 raw rows
were refused, found duplicated or screened out (refused, duplicates,
refused_raw > 0) — in a plain run and a checked --apply everything else IS
committed (a checked apply only at a pin that expected those counts), and
a checked --dry-run wrote nothing; the run exits 3 until they are resolved.

Usage (inside the api container or anywhere DATABASE_URL is set):
    python /app/scripts/promote_raw.py
    python /app/scripts/promote_raw.py --database-url postgres://...
    python /app/scripts/promote_raw.py --dry-run --out-csv /app/backups/promote_raw_dry.csv
    python /app/scripts/promote_raw.py --apply --expect-plan matches_inserted=0,...,refused_raw=0 \\
        --out-csv /app/backups/promote_raw_apply.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import Json, RealDictCursor, execute_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soccer_seasons  # noqa: E402
import team_identity  # noqa: E402
import text_encoding  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("promote_raw")

# League code → canonical (name, country).
# Codes here MUST stay aligned with fetch_upcoming.SOCCER_LEAGUES and
# the country codes used by load_football_data_extra.py.
LEAGUE_MAP: dict[str, tuple[str, str]] = {
    # ── European top flights (from load_football_data.py) ───────────
    "E0": ("Premier League", "England"),
    "E1": ("Championship", "England"),
    "E2": ("League One", "England"),
    "E3": ("League Two", "England"),
    "EC": ("National League", "England"),
    "D1": ("Bundesliga", "Germany"),
    "D2": ("2. Bundesliga", "Germany"),
    "I1": ("Serie A", "Italy"),
    "I2": ("Serie B", "Italy"),
    "SP1": ("La Liga", "Spain"),
    "SP2": ("La Liga 2", "Spain"),
    "F1": ("Ligue 1", "France"),
    "F2": ("Ligue 2", "France"),
    "N1": ("Eredivisie", "Netherlands"),
    "B1": ("Pro League", "Belgium"),
    "P1": ("Primeira Liga", "Portugal"),
    "SC0": ("Premiership", "Scotland"),
    "SC1": ("Championship", "Scotland"),
    "SC2": ("League One", "Scotland"),
    "SC3": ("League Two", "Scotland"),
    "T1": ("Süper Lig", "Turkey"),
    "G1": ("Super League", "Greece"),
    # ── Summer / non-European leagues (from load_football_data_extra.py) ──
    "MLS": ("MLS", "USA"),
    "BR1": ("Brasileirão Série A", "Brazil"),
    "AR1": ("Primera División", "Argentina"),
    "MX1": ("Liga MX", "Mexico"),
    "NO1": ("Eliteserien", "Norway"),
    "SE1": ("Allsvenskan", "Sweden"),
    "JP1": ("J1 League", "Japan"),
    "AT1": ("Bundesliga", "Austria"),
    "CH1": ("Super League", "Switzerland"),
    "CN1": ("Chinese Super League", "China"),
    "DK1": ("Superliga", "Denmark"),
    "FI1": ("Veikkausliiga", "Finland"),
    "IE1": ("Premier Division", "Ireland"),
    "PL1": ("Ekstraklasa", "Poland"),
    "RO1": ("Liga I", "Romania"),
    "RU1": ("Premier League", "Russia"),
    # ── Additional leagues from fetch_upcoming (ESPN-only, no
    # football-data historical CSVs yet). Kept here so promote_raw +
    # ensure_team can resolve their league_id correctly.
    "CL1": ("Primera División", "Chile"),
    "CO1": ("Primera A", "Colombia"),
    "KR1": ("K League 1", "South Korea"),
    "AU1": ("A-League", "Australia"),
    # ── International competitions (from load_international.py) ─────
    # martj42/international_results dataset. No odds data — these rows
    # will have NULL closing-odds columns and the model trains on
    # match outcome + team identity alone.
    "WC": ("FIFA World Cup", "International"),
    "LIB": ("Copa Libertadores", "International"),
    "WCQ": ("FIFA World Cup Qualifiers", "International"),
    "EURO": ("UEFA Euro", "International"),
    "EUROQ": ("UEFA Euro Qualifiers", "International"),
    "UNL": ("UEFA Nations League", "International"),
    "COPA": ("Copa América", "International"),
    "GOLD": ("CONCACAF Gold Cup", "International"),
    "AFCON": ("Africa Cup of Nations", "International"),
    "AFCASIAN": ("AFC Asian Cup", "International"),
    "CONFED": ("FIFA Confederations Cup", "International"),
}

# football-data labels a season two ways. The per-season European files
# (load_football_data.py) use a 'YYZZ' code: '1617' is 2016-17. The /new/
# country files (load_football_data_extra.py) use the calendar year for a
# league that plays within one ('2016': Brazil, MLS, Norway...) and
# 'YYYY/YYYY' for one that straddles two (Austria, Mexico...); the
# international loader uses the year too. '2021' reads both ways, so the
# league decides: a 4-digit season from a league whose files name years
# (YEAR_CODED_LEAGUES) is a year. Which leagues those are, and which leagues
# play calendar-year seasons (what fetch_upcoming labels by), is decided in
# one place, scripts/soccer_seasons.py; YEAR_CODED_LEAGUES must cover every
# league code in load_football_data_extra.COUNTRY_MAP (pinned by
# tests/unit/test_promote_raw.py). Until 2026-10-31 this set was named
# CALENDAR_SEASON_LEAGUES, although Liga MX and Austria are cross-year.
YEAR_CODED_LEAGUES: frozenset[str] = soccer_seasons.FOOTBALL_DATA_YEAR_CODED
_CROSS_YEAR_SEASON = re.compile(r"(\d{4})/(\d{4})")


# football-data's Time column is UK local time, the /new/ country files
# included. Verified on prod 2026-10-31: in July/August each of the 65 soccer
# fixtures that both the ESPN results path and promote_raw wrote is exactly
# 1 h later on the promote_raw row (CN1, BR1, NO1, MLS, AR1, MX1, SE1, SC0).
FOOTBALL_DATA_TZ = ZoneInfo("Europe/London")


def _kickoff_time(raw_time: str | None) -> tuple[int, int] | None:
    """(hour, minute) from football-data's HH:MM Time, or None."""
    if not raw_time:
        return None
    try:
        t = datetime.strptime(raw_time, "%H:%M")
    except ValueError:
        return None
    return t.hour, t.minute


def parse_wall_clock(raw_date: str, raw_time: str | None) -> datetime | None:
    """football-data's Date (+ Time) as the NAIVE UK wall clock it spells.
    Date is DD/MM/YYYY (newer) or DD/MM/YY (older); a Time that is not HH:MM
    is ignored. Until 2026-10-31 promote_raw stored exactly this, and
    Postgres read it in the session TimeZone (UTC on prod), so every
    summer-time kickoff landed 1 h late. Kept for finding those rows."""
    if not raw_date:
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            d = datetime.strptime(raw_date, fmt)
        except ValueError:
            continue
        hm = _kickoff_time(raw_time)
        return d.replace(hour=hm[0], minute=hm[1]) if hm else d
    return None


def parse_match_date(raw_date: str, raw_time: str | None) -> datetime | None:
    """The kickoff of a football-data row.

    A timed row is UK local time, returned as an aware UTC datetime: 15:00
    on a BST date is 14:00Z, and on a GMT date it is 15:00Z. In the repeated
    hour of the October change the first (BST) reading wins, and a time in
    the skipped March hour is read as GMT (zoneinfo's fold=0). A date-only
    row (or one whose Time is not HH:MM) stays a naive midnight, which
    Postgres reads in its session TimeZone exactly as before.

    Rows written before this change sit at the naive wall clock
    (parse_wall_clock). promote() refuses to run while any of them would get a
    corrected twin (check_legacy_kickoffs); moving them is an operator task."""
    d = parse_wall_clock(raw_date, raw_time)
    if d is None or _kickoff_time(raw_time) is None:
        return d
    return d.replace(tzinfo=FOOTBALL_DATA_TZ).astimezone(timezone.utc)


def to_int(v: str | None) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def to_float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalize_team_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


@dataclass
class Promotion:
    leagues: int = 0
    teams: int = 0
    matches: int = 0
    match_stats: int = 0
    odds: int = 0
    skipped_no_result: int = 0  # raw rows without a full-time score (not promoted)
    ht_dropped: int = 0  # promoted without the half-time value the raw row carried
    left_cancelled: int = 0  # existing row is 'cancelled' — not resurrected
    merged: int = 0  # written onto an existing row at another kickoff (would have been a second row)
    unchanged: int = 0  # the existing row already holds this result: not updated
    refused: int = 0  # not written: an ESPN row's score disagrees, a stale fixture row, or ambiguous
    duplicates: int = 0  # written onto an ESPN row while football-data's own finished row for the pair sits beside it
    ht_conflicts: int = 0  # an ESPN row kept its half-time where football-data's differs
    refused_raw: int = 0  # raw rows screened out before anything (screen_raw_rows): unknown code, wrong Div, mojibake


@dataclass(frozen=True)
class RawResult:
    """The scores one football-data row may write (see raw_result)."""

    home_score: int
    away_score: int
    ht_home: int | None = None
    ht_away: int | None = None
    ht_dropped: str | None = None  # why a half-time value present in the row was not used


def raw_result(row: dict) -> RawResult | None:
    """Scores from a raw row, or None when it has no full-time score.

    A blank FTHG or FTAG is not a finished match: football-data leaves both
    blank for abandoned, cancelled and unplayed fixtures, and for an awarded
    one puts the awarded score in HTHG/HTAG with HTR 'A' (Panathinaikos v
    Olympiakos 2019-03-17 and Bastia v Red Star 2025-12-05 carry "HT" 0-3;
    neither was 0-3 at half-time). Such a row's half-time columns are never
    used — the caller skips the whole row. The half-time pair is used only
    whole and only when neither half exceeds the final score.
    """
    home, away = to_int(row.get("FTHG")), to_int(row.get("FTAG"))
    if home is None or away is None:
        return None
    ht_home, ht_away = to_int(row.get("HTHG")), to_int(row.get("HTAG"))
    if ht_home is None and ht_away is None:
        return RawResult(home, away)
    if ht_home is None or ht_away is None:
        return RawResult(home, away, ht_dropped=f"half-time pair incomplete ({ht_home!r}/{ht_away!r})")
    if ht_home > home or ht_away > away:
        return RawResult(home, away, ht_dropped=f"half-time {ht_home}-{ht_away} exceeds full-time {home}-{away}")
    return RawResult(home, away, ht_home, ht_away)


def fixture_label(row: dict) -> str:
    """'F2 05/12/2025 Bastia v Red Star' — names a raw row in log lines."""
    return f"{row.get('league')} {row.get('Date')} {row.get('HomeTeam')} v {row.get('AwayTeam')}"


# Every soccer kickoff already in matches, for check_legacy_kickoffs.
EXISTING_SOCCER_KICKOFFS_SQL = """
    SELECT m.home_team_id::text AS home_id, m.away_team_id::text AS away_id, m.match_date
      FROM matches m
      JOIN leagues l ON l.id = m.league_id
     WHERE l.sport = 'soccer'
"""


class LegacyKickoffsPresent(RuntimeError):
    """promote() would insert a corrected-kickoff twin of a row it wrote at the legacy kickoff."""


def stranded_legacy_kickoffs(
    rows: Iterable[dict],
    league_ids: dict[str, str],
    team_ids: dict[str, str],
    existing: set[tuple[str, str, datetime]],
    db_tz: tzinfo,
) -> list[dict]:
    """Raw rows whose legacy row is still there: the row promote_raw wrote
    for the same pair at the legacy kickoff (the UK wall clock read in
    `db_tz`) exists, on a date where that differs from the corrected UTC
    kickoff. promote() would then either insert a second matches row at the
    corrected kickoff (when nothing is near it), write onto the legacy row
    and leave it 1 h late for good (when it is the only row near), or write
    onto the ESPN row at the corrected kickoff and leave the legacy row
    beside it as a duplicate. No pair plays twice within the hour, so a row
    at the legacy kickoff is always one of these. `existing` holds
    (home_id, away_id, UTC kickoff). Pure."""
    stranded = []
    for r in rows:
        if r.get("league") not in league_ids or raw_result(r) is None:
            continue
        home_id = team_ids.get(normalize_team_name(r.get("HomeTeam") or ""))
        away_id = team_ids.get(normalize_team_name(r.get("AwayTeam") or ""))
        wall = parse_wall_clock(r.get("Date") or "", r.get("Time"))
        corrected = parse_match_date(r.get("Date") or "", r.get("Time"))
        if not (home_id and away_id) or wall is None or corrected is None or corrected.tzinfo is None:
            continue  # unresolved, or date-only: its kickoff did not change
        legacy = wall.replace(tzinfo=db_tz).astimezone(timezone.utc)
        if legacy == corrected:
            continue
        if (str(home_id), str(away_id), legacy) in existing:
            stranded.append(r)
    return stranded


def check_legacy_kickoffs(cur, rows: list[dict], league_ids: dict[str, str], team_ids: dict[str, str]) -> None:
    """Raise LegacyKickoffsPresent, before promote() writes a single match,
    while any raw row's legacy row is still there (stranded_legacy_kickoffs).
    Every run re-promotes the whole raw table, and on 2026-10-31 about
    63,800 timed summer-time rows still sat at the legacy kickoff: one run
    after the timezone fix would have inserted that many duplicate finished
    matches into the training corpus, and since find_existing_matches it
    would instead update each legacy row and leave it 1 h late for good, or,
    where the ESPN row sits at the corrected kickoff, write onto that and
    leave the legacy copy beside it, all without a word. This keeps
    promote() refusing until scripts/ops/repair_football_data_kickoffs.sh
    has merged (--dedupe) and moved (--shift) every one of them; the tool's
    --check runs this."""
    cur.execute("SHOW TimeZone")
    db_tz = ZoneInfo(cur.fetchone()["TimeZone"])  # loud if Postgres reports a zone Python cannot load
    cur.execute(EXISTING_SOCCER_KICKOFFS_SQL)
    existing = {(r["home_id"], r["away_id"], r["match_date"].astimezone(timezone.utc)) for r in cur.fetchall()}
    stranded = stranded_legacy_kickoffs(rows, league_ids, team_ids, existing, db_tz)
    if stranded:
        sample = "; ".join(f"{fixture_label(r)} {r.get('Time')}" for r in stranded[:5])
        raise LegacyKickoffsPresent(
            f"{len(stranded)} raw row(s) still have their legacy matches row: promote_raw wrote them at the UK "
            f"wall clock read as {db_tz} (before 2026-10-31), 1 h after their UTC kickoff. Merge and move those "
            f"rows first (scripts/ops/repair_football_data_kickoffs.sh: --dedupe, then --shift), then re-run. "
            f"First {min(5, len(stranded))}: {sample}"
        )


RAW_UNKNOWN_LEAGUE = "unknown_league"
RAW_DIV_MISMATCH = "div_mismatch"
RAW_MOJIBAKE = "mojibake_name"
RAW_PROBLEM_TEXT = {
    RAW_UNKNOWN_LEAGUE: "league code not in promote_raw.LEAGUE_MAP",
    RAW_DIV_MISMATCH: "Div names another division: the file held another league",
    RAW_MOJIBAKE: "a team name looks mis-decoded (scripts/ops/repair_soccer_labels.sh)",
}


def raw_row_problem(row: dict) -> str | None:
    """Why a raw row must not be promoted, or None. RAW_UNKNOWN_LEAGUE: its
    code is not in LEAGUE_MAP (nothing says which league it is).
    RAW_DIV_MISMATCH: its Div — football-data's division code, read since
    2026-11-01 (a byte-order mark hid it on the newer files) — names another
    division: the file at that URL held another league. RAW_MOJIBAKE: a team
    name looks mis-decoded (text_encoding.looks_mojibake) and would mint a
    twin of the club. Pure."""
    code = row.get("league")
    if code not in LEAGUE_MAP:
        return RAW_UNKNOWN_LEAGUE
    div = (row.get("Div") or "").strip()
    if div and div != code:
        return RAW_DIV_MISMATCH
    if any(text_encoding.looks_mojibake(row.get(col)) for col in ("HomeTeam", "AwayTeam")):
        return RAW_MOJIBAKE
    return None


def screen_raw_rows(rows: list[dict]) -> tuple[list[dict], list[tuple[str, dict]]]:
    """(the rows promote() may use, [(problem, row)] for the ones it refuses);
    each (problem, league, season) group is logged at ERROR with its count and
    a sample (raw_row_problem)."""
    keep: list[dict] = []
    groups: dict[tuple[str, str, str], list[dict]] = {}
    refused: list[tuple[str, dict]] = []
    for r in rows:
        problem = raw_row_problem(r)
        if problem is None:
            keep.append(r)
            continue
        refused.append((problem, r))
        groups.setdefault((problem, str(r.get("league")), str(r.get("season"))), []).append(r)
    for (problem, code, season), group in sorted(groups.items()):
        logger.error(
            "REFUSED %d raw row(s) of %s %s — %s: not promoted, and they decide no team's league (e.g. %s, Div %r)",
            len(group),
            code,
            season,
            RAW_PROBLEM_TEXT[problem],
            fixture_label(group[0]),
            group[0].get("Div"),
        )
    return keep, refused


# The soccer leagues as they stand, with the football-data code each carries (NULL: none).
LEAGUE_CODES_SQL = "SELECT name, country, external_ids->>'football_data' AS code FROM leagues WHERE sport = 'soccer'"


def upsert_leagues(cur, codes: set[str]) -> dict[str, str]:
    """Insert leagues for each code; return {code: league_id}. promote_rows()
    passes LEAGUE_MAP codes only (screen_raw_rows refuses the rest). A league
    that already carries its code is not sent (see the module docstring: not
    rewritten, not locked)."""
    rows = []
    for code in sorted(codes):
        if code not in LEAGUE_MAP:
            logger.error("unknown league code %s — no league for it", code)
            continue
        name, country = LEAGUE_MAP[code]
        rows.append((name, country, "soccer", code))
    if not rows:
        return {}

    cur.execute(LEAGUE_CODES_SQL)
    have = {(r["name"], r["country"]): r["code"] for r in cur.fetchall()}
    missing = [r for r in rows if have.get((r[0], r[1])) != r[3]]
    if missing:
        execute_values(
            cur,
            """
            INSERT INTO leagues (name, country, sport, external_ids)
            VALUES %s
            ON CONFLICT (name, country, sport) DO UPDATE
                SET external_ids = leagues.external_ids || EXCLUDED.external_ids
                WHERE leagues.external_ids IS DISTINCT FROM leagues.external_ids || EXCLUDED.external_ids
            """,
            [(name, country, sport, Json({"football_data": code})) for name, country, sport, code in missing],
        )
    cur.execute(
        "SELECT id, external_ids->>'football_data' AS code " "FROM leagues WHERE external_ids ? 'football_data'"
    )
    return {r["code"]: r["id"] for r in cur.fetchall()}


def upsert_teams(cur, team_names: set[str], league_id_by_team: dict[str, str]) -> dict[str, str]:
    """Insert the teams referenced; return {normalized_name: team_id} for
    every soccer team, plus each raw name resolved onto another row.

    A raw name that is no team's normalized_name may still be a known club:
    an alias (teams.metadata.aliases, the resolver fetch_upcoming uses too)
    or the same name with accents ("Alaves" / "Alavés"). Such a name maps
    onto that row instead of minting a twin with no history. Only a name
    that matches nothing is inserted, and each one is logged as a canary
    (team_identity.log_new_soccer_team).

    An existing team keeps its league; only a missing one is filled, and a
    team that needs neither is not sent (see the module docstring)."""
    if not team_names:
        return {}
    cur.execute("SELECT id, normalized_name, league_id FROM teams WHERE sport = 'soccer'")
    existing = {r["normalized_name"]: r["league_id"] for r in cur.fetchall()}
    aliases = team_identity.load_alias_index(cur, "soccer")
    # Dedupe on the conflict key (normalized_name, sport): two RAW names
    # can normalize identically ("Alaves"/"Alavés"), and ON CONFLICT DO
    # UPDATE cannot touch the same row twice within one command
    # (CardinalityViolation) — keep the first raw spelling.
    seen: set[str] = set()
    rows: list[tuple[str, str, str | None, str]] = []
    resolved: dict[str, str] = {}
    minted: list[str] = []
    for name in sorted(team_names):
        norm = normalize_team_name(name)
        if norm in seen:
            continue
        seen.add(norm)
        league_id = league_id_by_team.get(name)
        if norm in existing:
            if existing[norm] is None and league_id is not None:
                rows.append((name, norm, league_id, "soccer"))  # its missing league filled
            continue
        team_id = team_identity.resolve_alias(aliases, "soccer", norm) or (
            team_identity.find_team_accent_insensitive(cur, "soccer", norm)
        )
        if team_id:
            resolved[norm] = team_id
            logger.info("Raw team %r resolves to existing soccer team %s (alias or accent fold)", name, team_id)
            continue
        minted.append(name)
        rows.append((name, norm, league_id, "soccer"))
    if rows:
        # The conflict branch is a row inserted between the read above and this statement; it too keeps its league.
        execute_values(
            cur,
            """
            INSERT INTO teams (name, normalized_name, league_id, sport)
            VALUES %s
            ON CONFLICT (normalized_name, sport) DO UPDATE
                SET league_id = COALESCE(teams.league_id, EXCLUDED.league_id)
                WHERE teams.league_id IS NULL AND EXCLUDED.league_id IS NOT NULL
            """,
            rows,
        )
    for name in minted:
        team_identity.log_new_soccer_team(cur, name, league_id_by_team.get(name), "football-data promote_raw")
    cur.execute("SELECT id, normalized_name FROM teams WHERE sport = 'soccer'")
    team_ids = {r["normalized_name"]: r["id"] for r in cur.fetchall()}
    team_ids.update(resolved)
    return team_ids


# promote() inserts only where find_existing_matches found no row, so this
# conflict branch catches a row written in between. It never downgrades the
# row already there: each score pair is replaced only by a complete pair (a
# load without half-time columns — the /new/ country files — used to NULL a
# half-time the ESPN results path had written), 'finished' is set only with
# a full-time pair in hand, season is left alone, and a 'cancelled' row is
# not touched at all (RETURNING then yields no row): that status is recorded
# on purpose, e.g. by scripts/fix_unplayed_soccer_rows.py.
INSERT_MATCH_SQL = """
    INSERT INTO matches (
        league_id, home_team_id, away_team_id, match_date, status,
        home_score, away_score, half_time_home, half_time_away, season
    )
    VALUES (%s, %s, %s, %s, 'finished', %s, %s, %s, %s, %s)
    ON CONFLICT (home_team_id, away_team_id, match_date) DO UPDATE
        SET home_score = CASE WHEN EXCLUDED.home_score IS NOT NULL AND EXCLUDED.away_score IS NOT NULL
                              THEN EXCLUDED.home_score ELSE matches.home_score END,
            away_score = CASE WHEN EXCLUDED.home_score IS NOT NULL AND EXCLUDED.away_score IS NOT NULL
                              THEN EXCLUDED.away_score ELSE matches.away_score END,
            half_time_home = CASE WHEN EXCLUDED.half_time_home IS NOT NULL AND EXCLUDED.half_time_away IS NOT NULL
                                  THEN EXCLUDED.half_time_home ELSE matches.half_time_home END,
            half_time_away = CASE WHEN EXCLUDED.half_time_home IS NOT NULL AND EXCLUDED.half_time_away IS NOT NULL
                                  THEN EXCLUDED.half_time_away ELSE matches.half_time_away END,
            status = CASE WHEN EXCLUDED.home_score IS NOT NULL AND EXCLUDED.away_score IS NOT NULL
                          THEN 'finished' ELSE matches.status END,
            updated_at = NOW()
        WHERE matches.status <> 'cancelled'
    RETURNING id
"""


# A fixture already in matches under the same home and away team ids, within
# EXISTING_MATCH_WINDOW of the football-data kickoff and on football-data's
# (UK) date, is the same fixture: no pair meets twice in a few hours. ESPN's
# kickoff can differ from football-data's by minutes, and until 2026-10-31
# promote_raw read football-data's an hour late; the exact-triple upsert
# missed both, and 228 finished matches came to exist twice
# (scripts/repair_football_data_kickoffs.py merges them).
EXISTING_MATCH_WINDOW = timedelta(hours=3)
# main()'s exit code when write_match refused a raw row or found a duplicate
# (everything else committed), as fetch_upcoming exits on a refused fork.
REFUSED_EXIT_CODE = 3

EXISTING_MATCH_SQL = """
    SELECT id::text AS id, match_date, status, home_score, away_score, half_time_home, half_time_away,
           external_ids->>'espn' AS espn_id, metadata->>'result_source' AS result_source
      FROM matches
     WHERE home_team_id = %s AND away_team_id = %s
       AND match_date >= %s AND match_date <= %s
"""

# Rows for ANOTHER team pair that share a club with the fixture, in the same
# window: no club plays twice in a few hours, so such a row is this fixture
# filed under other team ids (a club ESPN minted a second row for, or a row it
# filed under the wrong club), and inserting would put the match in the
# corpus twice (find_alias_slot_rows).
ALIAS_SLOT_SQL = """
    SELECT id::text AS id, home_team_id::text AS home_id, away_team_id::text AS away_id, match_date, status,
           home_score, away_score
      FROM matches
     WHERE (home_team_id IN (%s, %s) OR away_team_id IN (%s, %s))
       AND NOT (home_team_id = %s AND away_team_id = %s)
       AND match_date >= %s AND match_date <= %s
"""

# Guarded on every value update_existing_match decided from: a row that moved
# after the lookup is not written (RETURNING yields no row).
UPDATE_EXISTING_MATCH_SQL = """
    UPDATE matches
       SET home_score = %(home)s, away_score = %(away)s,
           half_time_home = %(ht_home)s, half_time_away = %(ht_away)s,
           status = 'finished', updated_at = NOW()
     WHERE id = %(id)s
       AND status = %(old_status)s
       AND home_score IS NOT DISTINCT FROM %(old_home)s AND away_score IS NOT DISTINCT FROM %(old_away)s
       AND half_time_home IS NOT DISTINCT FROM %(old_ht_home)s
       AND half_time_away IS NOT DISTINCT FROM %(old_ht_away)s
    RETURNING id
"""


def has_espn_identity(espn_id: str | None, result_source: str | None) -> bool:
    """A matches row the ESPN path owns: it carries the ESPN competition id
    (fetch_upcoming, since migration 023) or its result came off the ESPN
    scoreboard (rows the results path wrote before that)."""
    return bool(espn_id) or result_source == "espn_scoreboard"


def existing_match_window(match_dt: datetime, fixture_day: date) -> tuple[datetime, datetime]:
    """The kickoffs find_existing_matches looks at, as aware UTC bounds, both
    inclusive: within EXISTING_MATCH_WINDOW of a timed kickoff and on
    `fixture_day`, football-data's Date, read as the UK date it is (an MLS
    19:30 ET kickoff is 00:30 the next day there, and on ESPN's row too). A
    date-only row (naive midnight) has no kickoff to be near: its whole UK
    date, which holds its own naive midnight in a UTC session (prod's)."""
    day_start = datetime.combine(fixture_day, time(), tzinfo=FOOTBALL_DATA_TZ).astimezone(timezone.utc)
    next_day = datetime.combine(fixture_day + timedelta(days=1), time(), tzinfo=FOOTBALL_DATA_TZ)
    day_end = next_day.astimezone(timezone.utc) - timedelta(microseconds=1)
    if match_dt.tzinfo is None:
        return day_start, day_end
    kickoff = match_dt.astimezone(timezone.utc)
    return max(kickoff - EXISTING_MATCH_WINDOW, day_start), min(kickoff + EXISTING_MATCH_WINDOW, day_end)


def choose_existing(candidates: list[dict]) -> tuple[dict | None, str | None]:
    """(row, None) for the row a football-data fixture belongs on, (None, None)
    when there is none, (None, why) when it is ambiguous. An ESPN row beats
    any other (its identity is the one predictions, odds and recommendations
    hang off), then a finished row an unfinished one; two rows still level
    are ambiguous. Pure."""
    if not candidates:
        return None, None

    def rank(c: dict) -> tuple[bool, bool]:
        return not has_espn_identity(c.get("espn_id"), c.get("result_source")), c.get("status") != "finished"

    best = min(rank(c) for c in candidates)
    top = [c for c in candidates if rank(c) == best]
    if len(top) > 1:
        rows = ", ".join(f"{c['id']} {c['status']} {c['match_date']}" for c in top)
        return None, f"{len(top)} existing rows for this team pair could each be it ({rows})"
    return top[0], None


def same_kickoff(stored: datetime, match_dt: datetime) -> bool:
    """Whether a stored timestamptz is the kickoff promote() would write.
    psycopg2 hands it back in the session TimeZone, the zone a naive
    (date-only) kickoff is read in."""
    if match_dt.tzinfo is None:
        return stored.replace(tzinfo=None) == match_dt
    return stored == match_dt


def find_existing_matches(cur, home_id, away_id, match_dt: datetime, fixture_day: date) -> list[dict]:
    """This team pair's rows in existing_match_window (choose_existing picks)."""
    lo, hi = existing_match_window(match_dt, fixture_day)
    cur.execute(EXISTING_MATCH_SQL, (home_id, away_id, lo, hi))
    return [dict(r) for r in cur.fetchall()]


def find_alias_slot_rows(cur, home_id, away_id, match_dt: datetime, fixture_day: date) -> list[dict]:
    """Rows for another team pair sharing a club with this fixture in
    existing_match_window (ALIAS_SLOT_SQL): write_match inserts nothing beside
    one. After scripts/repair_football_data_kickoffs.py --dedupe has merged a
    copy into an ESPN row still filed under the wrong club, this keeps every
    promote run from inserting the copy again until scripts/
    repoint_team_matches.py has moved that row onto the right club; it does
    the same for a club ESPN minted a second row for that merge_soccer_teams
    has not folded yet. It needs a SHARED CLUB: a fixture filed under two
    other team ids (both sides aliased) does not trip it, and promote_raw
    would insert it beside that row. The re-point map's two-sided rows (Santos
    v Guadalajara, 2026-08-17: 4a31da6c, da444426) are that shape, which is why
    scripts/ops/repoint_team_matches.sh --pass 1 (runbook step 1) must have
    moved them before any MX1 raw row for that date is promoted; prod had no
    football-data copy of them on 2026-10-31."""
    lo, hi = existing_match_window(match_dt, fixture_day)
    cur.execute(ALIAS_SLOT_SQL, (home_id, away_id, home_id, away_id, home_id, away_id, lo, hi))
    return [dict(r) for r in cur.fetchall()]


def duplicates_beside(chosen: dict, candidates: list[dict]) -> list[dict]:
    """Finished rows without ESPN identity (football-data's own) beside the
    ESPN row a fixture is written onto: the same match twice in the corpus,
    which only scripts/repair_football_data_kickoffs.py --dedupe merges. Pure."""
    if not has_espn_identity(chosen.get("espn_id"), chosen.get("result_source")):
        return []
    return [
        c
        for c in candidates
        if c["id"] != chosen["id"]
        and c.get("status") == "finished"
        and not has_espn_identity(c.get("espn_id"), c.get("result_source"))
    ]


def update_existing_match(cur, existing: dict, result: RawResult) -> tuple[str | None, str]:
    """Write a football-data result onto the row choose_existing picked;
    returns (match_id or None, outcome).

    'cancelled': the row is 'cancelled' and stays as it is (see
    INSERT_MATCH_SQL). 'stale_fixture': the row is unfinished and has no ESPN
    identity, so it is a fixture row from before migration 023 that never got
    its result (football-data's own rows are always finished); finishing it
    would grade its predictions and settle its recs through the outcome
    trigger, which is scripts/repair_football_data_kickoffs.py's reviewed
    --dedupe fold or repair_fixture_orphans' business, never a promote run's;
    nothing written. 'score_conflict': an ESPN row holds a different final
    score, which is never overwritten; nothing written. 'ht_conflict':
    written, but an ESPN row keeps a half-time that differs from
    football-data's. 'score_changed': a row without ESPN identity
    (football-data's own) took football-data's new score, as the upsert always
    let it. 'changed': the row moved after the lookup; nothing written.
    'unchanged': a finished row that already holds exactly what would be
    written; nothing written (an 'ht_conflict' on such a row is still
    reported as one). Otherwise 'updated'. A half-time pair is only ever
    replaced by a complete one, an ESPN row's only when it has none; the
    kickoff, season and identity stay the row's own."""
    if existing["status"] == "cancelled":
        return None, "cancelled"
    espn = has_espn_identity(existing.get("espn_id"), existing.get("result_source"))
    if not espn and existing["status"] != "finished":
        return None, "stale_fixture"
    old_ft = (existing["home_score"], existing["away_score"])
    old_ht = (existing["half_time_home"], existing["half_time_away"])
    new_ft = (result.home_score, result.away_score)
    raw_ht = (result.ht_home, result.ht_away) if result.ht_home is not None and result.ht_away is not None else None
    outcome = "updated"
    if espn:
        if old_ft != (None, None) and old_ft != new_ft:
            return None, "score_conflict"
        new_ht = raw_ht if raw_ht is not None and old_ht == (None, None) else old_ht
        if raw_ht is not None and old_ht != (None, None) and old_ht != raw_ht:
            outcome = "ht_conflict"
    else:
        new_ht = raw_ht or old_ht
        if old_ft != (None, None) and old_ft != new_ft:
            outcome = "score_changed"
    if existing["status"] == "finished" and old_ft == new_ft and old_ht == new_ht:
        return existing["id"], ("unchanged" if outcome == "updated" else outcome)
    cur.execute(
        UPDATE_EXISTING_MATCH_SQL,
        {
            "id": existing["id"],
            "home": new_ft[0],
            "away": new_ft[1],
            "ht_home": new_ht[0],
            "ht_away": new_ht[1],
            "old_status": existing["status"],
            "old_home": old_ft[0],
            "old_away": old_ft[1],
            "old_ht_home": old_ht[0],
            "old_ht_away": old_ht[1],
        },
    )
    r = cur.fetchone()
    if r is None:
        return None, "changed"
    return r["id"], outcome


def insert_match(
    cur, *, league_id, home_id, away_id, match_dt, season, home_score, away_score, ht_home, ht_away
) -> str | None:
    """Upsert one finished fixture; returns its id, or None when the existing
    row is 'cancelled' and was left as it is.

    Refuses (ValueError) a fixture without a full-time score — not a
    finished match, see raw_result — and a split half-time pair, rather
    than writing either.
    """
    if home_score is None or away_score is None:
        raise ValueError(f"insert_match needs a full-time score, got {home_score!r}-{away_score!r}")
    if (ht_home is None) != (ht_away is None):
        raise ValueError(f"insert_match got a split half-time pair {ht_home!r}-{ht_away!r}")
    cur.execute(
        INSERT_MATCH_SQL,
        (league_id, home_id, away_id, match_dt, home_score, away_score, ht_home, ht_away, season),
    )
    r = cur.fetchone()
    return r["id"] if r else None


def write_match(
    cur, stats: Promotion, row: dict, *, league_id, home_id, away_id, match_dt, result: RawResult
) -> str | None:
    """The matches row a raw row's result goes on: the existing row for the
    pair (find_existing_matches, choose_existing, update_existing_match) or a new one
    (insert_match), never a new one beside the same fixture under other team
    ids (find_alias_slot_rows: refused). Returns its id, or None when nothing
    was written (counted in `stats` and logged; the caller then writes no
    stats or odds)."""
    label = fixture_label(row)
    fixture_day = parse_wall_clock(row.get("Date") or "", None)
    if fixture_day is None:
        raise ValueError(f"write_match needs a football-data Date, got {row.get('Date')!r} ({label})")
    candidates = find_existing_matches(cur, home_id, away_id, match_dt, fixture_day.date())
    existing, ambiguity = choose_existing(candidates)
    if ambiguity:
        stats.refused += 1
        logger.error("Not promoting %s: %s — merge them first", label, ambiguity)
        return None
    if existing is None:
        aliases = find_alias_slot_rows(cur, home_id, away_id, match_dt, fixture_day.date())
        if aliases:
            stats.refused += 1
            logger.error(
                "Not promoting %s (%s): no row for this team pair, but %s share a club with it within %s of that "
                "kickoff — no club plays twice in a few hours, so that is this match under other team ids; nothing "
                "inserted. Unify the teams first (scripts/ops/merge_soccer_teams.sh, or "
                "scripts/repoint_team_matches.py for a row filed under the wrong club), then re-run",
                label,
                match_dt,
                ", ".join(
                    f"{a['id']} at {a['match_date']} ({a['status']} {a['home_score']}-{a['away_score']})"
                    for a in aliases
                ),
                EXISTING_MATCH_WINDOW,
            )
            return None
        match_id = insert_match(
            cur,
            league_id=league_id,
            home_id=home_id,
            away_id=away_id,
            match_dt=match_dt,
            season=season_label(row.get("season", ""), row.get("league")),
            home_score=result.home_score,
            away_score=result.away_score,
            ht_home=result.ht_home,
            ht_away=result.ht_away,
        )
        outcome = "inserted" if match_id else "cancelled"
    else:
        match_id, outcome = update_existing_match(cur, existing, result)
    where = f"row {existing['id']} at {existing['match_date']}" if existing else "the existing row"
    ex: dict = existing or {}  # the row the outcome is about (None only on an insert)
    if outcome == "cancelled":
        stats.left_cancelled += 1
        logger.warning(
            "Left %s as it is: %s is 'cancelled' but football-data now has a full-time score for it — check "
            "which is right",
            label,
            where,
        )
        return None
    if outcome == "stale_fixture":
        stats.refused += 1
        logger.error(
            "Not promoting %s: %s is a '%s' fixture row without ESPN identity (from before migration 023; "
            "football-data's own rows are always finished) — nothing written: finishing it would grade its "
            "predictions and settle its recs. Fold it with scripts/repair_football_data_kickoffs.py --dedupe or "
            "clean it up with repair_fixture_orphans, then re-run",
            label,
            where,
            ex["status"],
        )
        return None
    if outcome in ("score_conflict", "changed"):
        stats.refused += 1
        logger.error(
            "Not promoting %s: %s %s (football-data %d-%d) — nothing written; check which score is right",
            label,
            where,
            (
                f"is an ESPN row with final score {ex['home_score']}-{ex['away_score']}"
                if outcome == "score_conflict"
                else "changed after the lookup"
            ),
            result.home_score,
            result.away_score,
        )
        return None
    if outcome == "ht_conflict":
        stats.ht_conflicts += 1
        logger.error(
            "%s: kept the half-time %s-%s on ESPN %s; football-data says %s-%s — check which is right",
            label,
            ex["half_time_home"],
            ex["half_time_away"],
            where,
            result.ht_home,
            result.ht_away,
        )
    elif outcome == "score_changed":
        logger.warning(
            "%s: football-data changed the score on %s from %s-%s to %d-%d",
            label,
            where,
            ex["home_score"],
            ex["away_score"],
            result.home_score,
            result.away_score,
        )
    if outcome == "unchanged":
        stats.unchanged += 1
    if existing is not None and not same_kickoff(existing["match_date"], match_dt):
        stats.merged += 1
        logger.info(
            "%s %s %s (football-data kickoff %s)",
            label,
            "already on" if outcome == "unchanged" else "written onto",
            where,
            match_dt,
        )
    beside = duplicates_beside(existing, candidates) if existing is not None else []
    if beside:
        stats.duplicates += 1
        logger.error(
            "%s: written onto ESPN %s, but football-data's own finished row(s) for the pair sit beside it (%s) — "
            "the match is in the corpus twice; merge them with scripts/repair_football_data_kickoffs.py --dedupe",
            label,
            where,
            ", ".join(f"{c['id']} at {c['match_date']}" for c in beside),
        )
    return match_id


# A row that already holds these values is not inserted at all (NOT EXISTS): no conflict, so the run neither rewrites
# nor locks it (see the module docstring). Only a missing or differing row reaches ON CONFLICT.
INSERT_MATCH_STATS_SQL = """
    INSERT INTO match_stats (
        match_id, team_id, shots, shots_on_target, corners, fouls,
        yellow_cards, red_cards
    )
    SELECT v.match_id, v.team_id, v.shots, v.shots_on_target, v.corners, v.fouls, v.yellow_cards, v.red_cards
      FROM (VALUES (%s::uuid, %s::uuid, %s::integer, %s::integer, %s::integer, %s::integer, %s::integer,
                    %s::integer))
           AS v (match_id, team_id, shots, shots_on_target, corners, fouls, yellow_cards, red_cards)
     WHERE NOT EXISTS (
           SELECT 1 FROM match_stats s
            WHERE s.match_id = v.match_id AND s.team_id = v.team_id
              AND (s.shots, s.shots_on_target, s.corners, s.fouls, s.yellow_cards, s.red_cards)
                  IS NOT DISTINCT FROM (v.shots, v.shots_on_target, v.corners, v.fouls, v.yellow_cards, v.red_cards)
     )
    ON CONFLICT (match_id, team_id) DO UPDATE
        SET shots = EXCLUDED.shots,
            shots_on_target = EXCLUDED.shots_on_target,
            corners = EXCLUDED.corners,
            fouls = EXCLUDED.fouls,
            yellow_cards = EXCLUDED.yellow_cards,
            red_cards = EXCLUDED.red_cards
        WHERE (match_stats.shots, match_stats.shots_on_target, match_stats.corners, match_stats.fouls,
               match_stats.yellow_cards, match_stats.red_cards)
              IS DISTINCT FROM (EXCLUDED.shots, EXCLUDED.shots_on_target, EXCLUDED.corners, EXCLUDED.fouls,
                                EXCLUDED.yellow_cards, EXCLUDED.red_cards)
"""


def insert_match_stats(
    cur, match_id, team_id, *, shots, shots_on_target, corners, fouls, yellow_cards, red_cards
) -> None:
    cur.execute(
        INSERT_MATCH_STATS_SQL,
        (match_id, team_id, shots, shots_on_target, corners, fouls, yellow_cards, red_cards),
    )


def insert_odds(cur, match_id, rows: list[tuple]) -> int:
    """rows: (bookmaker, market_type, selection, odds_decimal, line, is_opening)"""
    if not rows:
        return 0
    # We don't have a unique constraint on odds, so duplicate-protect with
    # a NOT EXISTS via a CTE. Insert only if no row with the same key already
    # exists.
    inserted = 0
    for bookmaker, market_type, selection, odds_decimal, line, is_opening in rows:
        if odds_decimal is None:
            continue
        cur.execute(
            """
            INSERT INTO odds (match_id, bookmaker, market_type, selection,
                              odds_decimal, line, is_opening, is_live)
            SELECT %s, %s, %s, %s, %s, %s, %s, false
            WHERE NOT EXISTS (
                SELECT 1 FROM odds
                WHERE match_id = %s AND bookmaker = %s
                  AND market_type = %s AND selection = %s
                  AND COALESCE(line, -1) = COALESCE(%s, -1)
                  AND is_opening = %s
            )
            """,
            (
                match_id,
                bookmaker,
                market_type,
                selection,
                odds_decimal,
                line,
                is_opening,
                match_id,
                bookmaker,
                market_type,
                selection,
                line,
                is_opening,
            ),
        )
        inserted += cur.rowcount
    return inserted


def odds_rows_for(row: dict) -> list[tuple]:
    """Build the odds-table rows for a single raw match. Closing odds only."""
    out: list[tuple] = []
    # 1x2 (home / draw / away)
    odds_specs = [
        ("Bet365", "B365H", "B365D", "B365A"),
        ("Pinnacle", "PSCH", "PSCD", "PSCA"),
        ("Average", "AvgH", "AvgD", "AvgA"),
        ("Max", "MaxH", "MaxD", "MaxA"),
    ]
    for book, h_col, d_col, a_col in odds_specs:
        out += [
            (book, "1x2", "home", to_float(row.get(h_col)), None, False),
            (book, "1x2", "draw", to_float(row.get(d_col)), None, False),
            (book, "1x2", "away", to_float(row.get(a_col)), None, False),
        ]
    # Over/Under 2.5
    for book, over_col, under_col in [
        ("Bet365", "B365>2.5", "B365<2.5"),
        ("Average", "Avg>2.5", "Avg<2.5"),
    ]:
        out += [
            (book, "over_under", "over", to_float(row.get(over_col)), 2.5, False),
            (book, "over_under", "under", to_float(row.get(under_col)), 2.5, False),
        ]
    return out


def season_label(code: str, league: str | None = None) -> str:
    """Convert a football-data season to the stored label.

    'YYZZ' code (per-season European files): '2425' -> '2024-2025'. Y2K
    rollover: codes 50-99 → 19xx; codes 00-49 → 20xx. football-data has
    been publishing since 1993 so the 1990s codes are real.
    'YYYY/YYYY' (cross-year /new/ files): '2016/2017' -> '2016-2017', the
    spelling every other loader writes.
    Calendar year (/new/ files, internationals): '2016' stays '2016'. Read
    as a 'YYZZ' code it became '2020-2116' (32,794 soccer rows by
    2026-10-31; scripts/backfill_season_labels.py corrects them).

    '2021' is both the 2020-21 code and a calendar year, so `league`
    decides (YEAR_CODED_LEAGUES). Without one, a 4-digit code expands
    only when its halves are consecutive years; anything else is a year.
    """
    code = (code or "").strip()
    cross_year = _CROSS_YEAR_SEASON.fullmatch(code)
    if cross_year and int(cross_year.group(2)) - int(cross_year.group(1)) == 1:
        return f"{cross_year.group(1)}-{cross_year.group(2)}"
    if len(code) != 4 or not code.isdigit():
        return code
    if league in YEAR_CODED_LEAGUES:
        return code
    start, end = int(code[:2]), int(code[2:])
    if end != (start + 1) % 100:
        return code  # not a 'YYZZ' code: a calendar year
    start_year = (1900 if start >= 50 else 2000) + start
    return f"{start_year}-{start_year + 1}"


def promote_rows(conn, cur, stats: Promotion, *, batch_size: int | None) -> None:
    """Promote every raw row through `cur`. `batch_size` commits every that
    many matches (the plain run); None never commits (checked mode: the
    caller decides)."""
    cur.execute("SELECT * FROM raw_football_data")
    rows = cur.fetchall()

    if not rows:
        logger.info("raw_football_data is empty — nothing to promote")
        return
    rows, refused_raw = screen_raw_rows(rows)
    stats.refused_raw = len(refused_raw)

    league_codes = {r["league"] for r in rows if r.get("league")}
    league_ids = upsert_leagues(cur, league_codes)
    stats.leagues = len(league_ids)
    logger.info("Upserted %d leagues", stats.leagues)

    # Collect teams and their first-seen league (for league_id assignment)
    team_to_league: dict[str, str] = {}
    for r in rows:
        code = r.get("league")
        if code in league_ids:
            lid = league_ids[code]
            for col in ("HomeTeam", "AwayTeam"):
                name = r.get(col)
                if name and name not in team_to_league:
                    team_to_league[name] = lid
    team_ids = upsert_teams(cur, set(team_to_league.keys()), team_to_league)
    stats.teams = len(team_ids)
    logger.info("Upserted %d teams", stats.teams)

    # Before any match is written: the loop below commits in batches.
    check_legacy_kickoffs(cur, rows, league_ids, team_ids)

    # Insert matches one-by-one (we need the returned id for stats/odds)
    for r in rows:
        code = r.get("league")
        if code not in league_ids:
            continue
        home_norm = normalize_team_name(r.get("HomeTeam", ""))
        away_norm = normalize_team_name(r.get("AwayTeam", ""))
        home_id = team_ids.get(home_norm)
        away_id = team_ids.get(away_norm)
        if not (home_id and away_id):
            continue
        match_dt = parse_match_date(r.get("Date", ""), r.get("Time"))
        if match_dt is None:
            continue
        result = raw_result(r)
        if result is None:
            stats.skipped_no_result += 1
            logger.warning(
                "Not promoting %s: no full-time score (FTHG=%r FTAG=%r) — abandoned, awarded or "
                "unplayed, not a finished match; its HTHG/HTAG (%r/%r, HTR %r) are not a half-time score",
                fixture_label(r),
                r.get("FTHG"),
                r.get("FTAG"),
                r.get("HTHG"),
                r.get("HTAG"),
                r.get("HTR"),
            )
            continue
        if result.ht_dropped:
            stats.ht_dropped += 1
            logger.warning("Promoting %s without its half-time: %s", fixture_label(r), result.ht_dropped)
        match_id = write_match(
            cur,
            stats,
            r,
            league_id=league_ids[code],
            home_id=home_id,
            away_id=away_id,
            match_dt=match_dt,
            result=result,
        )
        if not match_id:
            continue
        stats.matches += 1

        # Per-team stats (home then away)
        for tid, prefix in [(home_id, "H"), (away_id, "A")]:
            insert_match_stats(
                cur,
                match_id,
                tid,
                shots=to_int(r.get(f"{prefix}S")),
                shots_on_target=to_int(r.get(f"{prefix}ST")),
                corners=to_int(r.get(f"{prefix}C")),
                fouls=to_int(r.get(f"{prefix}F")),
                yellow_cards=to_int(r.get(f"{prefix}Y")),
                red_cards=to_int(r.get(f"{prefix}R")),
            )
        stats.match_stats += 2

        stats.odds += insert_odds(cur, match_id, odds_rows_for(r))

        if batch_size and stats.matches % batch_size == 0:
            conn.commit()
            logger.info("Committed batch: matches=%d odds=%d", stats.matches, stats.odds)


def promote(database_url: str, *, batch_size: int = 1000) -> Promotion:
    stats = Promotion()
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            promote_rows(conn, cur, stats, batch_size=batch_size)
            conn.commit()
    return stats


# ── checked mode: one transaction, what changed read back, pinned ──────────

# The PLAN keys, in order. Every one is pinned by --expect-plan.
PLAN_KEYS = (
    "matches_inserted",
    "matches_changed",
    "match_stats_inserted",
    "match_stats_changed",
    "odds_inserted",
    "teams_inserted",
    "teams_changed",
    "leagues_inserted",
    "leagues_changed",
    "predictions_changed",
    "recs_changed",
    "refused",
    "duplicates",
    "refused_raw",
)

# Taken first, in the same REPEATABLE READ transaction: the soccer rows a run can update, with the values it can
# change. Rows a run inserts carry created_at = now() (the transaction's start, the column default); rows it updates
# carry updated_at = now() (the updated_at triggers), except match_stats, which has no updated_at: it is compared with
# its snapshot. Under REPEATABLE READ no other session's write is visible here, so all of it is this run's.
CHECKED_SNAPSHOT_SQL = (
    """
    CREATE TEMP TABLE promote_before_matches ON COMMIT DROP AS
    SELECT m.id, m.status, m.home_score, m.away_score, m.half_time_home, m.half_time_away
      FROM matches m JOIN leagues l ON l.id = m.league_id
     WHERE l.sport = 'soccer'
    """,
    "ALTER TABLE promote_before_matches ADD PRIMARY KEY (id)",
    """
    CREATE TEMP TABLE promote_before_stats ON COMMIT DROP AS
    SELECT s.match_id, s.team_id, s.shots, s.shots_on_target, s.corners, s.fouls, s.yellow_cards, s.red_cards
      FROM match_stats s JOIN matches m ON m.id = s.match_id JOIN leagues l ON l.id = m.league_id
     WHERE l.sport = 'soccer'
    """,
    "ALTER TABLE promote_before_stats ADD PRIMARY KEY (match_id, team_id)",
    """
    CREATE TEMP TABLE promote_before_teams ON COMMIT DROP AS
    SELECT t.id, t.name, t.league_id FROM teams t WHERE t.sport = 'soccer'
    """,
    """
    CREATE TEMP TABLE promote_before_leagues ON COMMIT DROP AS
    SELECT l.id, l.external_ids FROM leagues l WHERE l.sport = 'soccer'
    """,
)

# Each: (PLAN key, CSV table, query returning key / old / new as text). Run after promote_rows, before the end.
CHECKED_CHANGES_SQL = (
    (
        "matches_inserted",
        "matches",
        """
        SELECT m.id::text AS key, '' AS old,
               concat_ws(' ', m.match_date, m.status, m.home_score || '-' || m.away_score,
                         'ht ' || m.half_time_home || '-' || m.half_time_away, m.season) AS new
          FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'soccer' AND m.created_at = now()
         ORDER BY m.id
        """,
    ),
    (
        "matches_changed",
        "matches",
        """
        SELECT m.id::text AS key,
               concat_ws(' ', b.status, b.home_score || '-' || b.away_score,
                         'ht ' || b.half_time_home || '-' || b.half_time_away) AS old,
               concat_ws(' ', m.status, m.home_score || '-' || m.away_score,
                         'ht ' || m.half_time_home || '-' || m.half_time_away) AS new
          FROM matches m JOIN promote_before_matches b ON b.id = m.id
         WHERE (m.status, m.home_score, m.away_score, m.half_time_home, m.half_time_away)
               IS DISTINCT FROM (b.status, b.home_score, b.away_score, b.half_time_home, b.half_time_away)
         ORDER BY m.id
        """,
    ),
    (
        "match_stats_inserted",
        "match_stats",
        """
        SELECT s.match_id || '/' || s.team_id AS key, '' AS old,
               concat_ws(' ', s.shots, s.shots_on_target, s.corners, s.fouls, s.yellow_cards, s.red_cards) AS new
          FROM match_stats s
         WHERE s.created_at = now()
         ORDER BY 1
        """,
    ),
    (
        "match_stats_changed",
        "match_stats",
        """
        SELECT s.match_id || '/' || s.team_id AS key,
               concat_ws(' ', b.shots, b.shots_on_target, b.corners, b.fouls, b.yellow_cards, b.red_cards) AS old,
               concat_ws(' ', s.shots, s.shots_on_target, s.corners, s.fouls, s.yellow_cards, s.red_cards) AS new
          FROM match_stats s JOIN promote_before_stats b ON b.match_id = s.match_id AND b.team_id = s.team_id
         WHERE (s.shots, s.shots_on_target, s.corners, s.fouls, s.yellow_cards, s.red_cards)
               IS DISTINCT FROM (b.shots, b.shots_on_target, b.corners, b.fouls, b.yellow_cards, b.red_cards)
         ORDER BY 1
        """,
    ),
    (
        "odds_inserted",
        "odds",
        """
        SELECT o.id::text AS key, '' AS old,
               concat_ws(' ', o.match_id, o.bookmaker, o.market_type, o.selection, o.line, o.odds_decimal) AS new
          FROM odds o
         WHERE o.created_at = now()
         ORDER BY o.id
        """,
    ),
    (
        "teams_inserted",
        "teams",
        "SELECT t.id::text AS key, '' AS old, t.name AS new FROM teams t WHERE t.created_at = now() ORDER BY t.id",
    ),
    (
        "teams_changed",
        "teams",
        """
        SELECT t.id::text AS key, concat_ws(' ', b.name, b.league_id) AS old,
               concat_ws(' ', t.name, t.league_id) AS new
          FROM teams t JOIN promote_before_teams b ON b.id = t.id
         WHERE (t.name, t.league_id) IS DISTINCT FROM (b.name, b.league_id)
         ORDER BY t.id
        """,
    ),
    (
        "leagues_inserted",
        "leagues",
        """
        SELECT l.id::text AS key, '' AS old, concat_ws(' ', l.name, l.country, l.external_ids) AS new
          FROM leagues l WHERE l.created_at = now() ORDER BY l.id
        """,
    ),
    (
        "leagues_changed",
        "leagues",
        """
        SELECT l.id::text AS key, b.external_ids::text AS old, l.external_ids::text AS new
          FROM leagues l JOIN promote_before_leagues b ON b.id = l.id
         WHERE l.external_ids IS DISTINCT FROM b.external_ids
         ORDER BY l.id
        """,
    ),
    (
        # The outcome trigger, when a run finished a row that was not finished.
        "predictions_changed",
        "predictions",
        """
        SELECT p.id::text AS key, '' AS old,
               concat_ws(' ', p.match_id, p.prediction_type, p.actual_outcome, p.is_correct) AS new
          FROM predictions p
         WHERE p.updated_at = now()
         ORDER BY p.id
        """,
    ),
    (
        "recs_changed",
        "betting_recommendations",
        """
        SELECT r.id::text AS key, '' AS old, concat_ws(' ', r.match_id, r.status, r.profit_loss) AS new
          FROM betting_recommendations r
         WHERE r.updated_at = now()
         ORDER BY r.id
        """,
    ),
)
CHECKED_CSV_FIELDS = ("plan_key", "table", "key", "old", "new")


class PlanRefused(RuntimeError):
    """A checked run whose plan cannot be committed as asked: rolled back."""


def parse_expect_plan(text: str) -> dict[str, int]:
    """'matches_inserted=0,...' -> dict: every PLAN key exactly once, nothing else."""
    out: dict[str, int] = {}
    for part in (text or "").split(","):
        key, sep, val = part.strip().partition("=")
        if not sep or key not in PLAN_KEYS or not val.isdigit() or key in out:
            raise PlanRefused(f"--expect-plan: bad entry {part!r} (keys: {', '.join(PLAN_KEYS)})")
        out[key] = int(val)
    missing = [k for k in PLAN_KEYS if k not in out]
    if missing:
        raise PlanRefused(f"--expect-plan must pin every key; missing {', '.join(missing)}")
    return out


def plan_line(counts: Counter) -> str:
    return "PLAN " + " ".join(f"{k}={counts[k]}" for k in PLAN_KEYS)


def read_changes(cur, stats: Promotion) -> tuple[Counter, list[dict]]:
    """The PLAN counts and the CSV rows of what this transaction changed (CHECKED_CHANGES_SQL), plus the raw rows
    promote_rows refused or found duplicated, and the ones its screen refused (screen_raw_rows: refused_raw)."""
    counts: Counter = Counter({k: 0 for k in PLAN_KEYS})
    rows: list[dict] = []
    for key, table, sql in CHECKED_CHANGES_SQL:
        cur.execute(sql)
        found = cur.fetchall()
        counts[key] = len(found)
        rows += [{"plan_key": key, "table": table, "key": r["key"], "old": r["old"], "new": r["new"]} for r in found]
    counts["refused"] = stats.refused
    counts["duplicates"] = stats.duplicates
    counts["refused_raw"] = stats.refused_raw
    return counts, rows


def write_changes_csv(path: str, rows: list[dict]) -> None:
    """Write-once (mode 'x'): a re-run can never overwrite an earlier run's record."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = out.open("x", newline="")
    except FileExistsError:
        raise PlanRefused(f"{out} exists — the audit CSV is write-once") from None
    with fh:
        w = csv.DictWriter(fh, fieldnames=CHECKED_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def promote_checked(
    database_url: str, *, apply: bool, expected: dict[str, int] | None, out_csv: str
) -> tuple[Promotion, Counter]:
    """Checked mode (see the module docstring): one REPEATABLE READ transaction, rolled back unless `apply` and the
    plan equals `expected`. Raises LegacyKickoffsPresent (nothing written) or PlanRefused (rolled back)."""
    stats = Promotion()
    conn = psycopg2.connect(database_url)
    try:
        conn.set_session(isolation_level="REPEATABLE READ")
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for sql in CHECKED_SNAPSHOT_SQL:
                cur.execute(sql)
            promote_rows(conn, cur, stats, batch_size=None)
            counts, rows = read_changes(cur, stats)
        write_changes_csv(out_csv, rows)
        logger.info(
            "Checked run: leagues=%d teams=%d matches=%d (unchanged %d) merged=%d skipped_no_result=%d ht_dropped=%d "
            "left_cancelled=%d ht_conflicts=%d refused=%d duplicates=%d refused_raw=%d; audit CSV %s",
            stats.leagues,
            stats.teams,
            stats.matches,
            stats.unchanged,
            stats.merged,
            stats.skipped_no_result,
            stats.ht_dropped,
            stats.left_cancelled,
            stats.ht_conflicts,
            stats.refused,
            stats.duplicates,
            stats.refused_raw,
            out_csv,
        )
        logger.info("%s", plan_line(counts))
        if not apply:
            conn.rollback()
            logger.info("DRY-RUN: rolled back, nothing written")
            return stats, counts
        drift = {k: (expected[k], counts[k]) for k in PLAN_KEYS if expected is not None and expected[k] != counts[k]}
        if expected is None or drift:
            conn.rollback()
            raise PlanRefused(f"the plan drifted from --expect-plan (expected, now): {drift} — rolled back")
        conn.commit()
        logger.info("Committed: %s", plan_line(counts)[5:])
        return stats, counts
    finally:
        conn.close()  # an open transaction is rolled back


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL"), help="Postgres URL. Defaults to $DATABASE_URL."
    )
    p.add_argument("--batch-size", type=int, default=1000)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Checked mode, rolled back (see CHECKED MODE).")
    mode.add_argument("--apply", action="store_true", help="Checked mode, committed only at --expect-plan.")
    p.add_argument("--expect-plan", default=None, help="With --apply: every PLAN key, from the reviewed dry run.")
    p.add_argument("--out-csv", default=None, help="Checked mode: the write-once audit CSV of every change.")
    return p.parse_args(argv)


def main_checked(args: argparse.Namespace) -> int:
    if not args.out_csv:
        logger.error("--dry-run / --apply need --out-csv (the write-once record of every change)")
        return 2
    if args.apply and not args.expect_plan:
        logger.error("--apply needs --expect-plan (every PLAN key, from the reviewed dry run)")
        return 2
    if args.expect_plan and not args.apply:
        logger.error("--expect-plan goes with --apply")
        return 2
    try:
        expected = parse_expect_plan(args.expect_plan) if args.apply else None
        stats, counts = promote_checked(args.database_url, apply=args.apply, expected=expected, out_csv=args.out_csv)
    except LegacyKickoffsPresent as e:
        logger.error("Refusing to promote — nothing written: %s", e)
        return 1
    except PlanRefused as e:
        logger.error("REFUSED: %s", e)
        return 1
    if stats.refused or stats.duplicates or stats.refused_raw:
        # The plain run's exit 3, in both checked modes (module docstring, EXIT STATUS): the dry run wrote nothing;
        # the apply committed, but only because its pin expected exactly these counts.
        done = (
            "the --apply committed at a pin that expects them; everything else is committed"
            if args.apply
            else "nothing written (dry run); a checked --apply commits only when they are pinned too"
        )
        logger.error(
            "REFUSED %d raw row(s), screened out %d and found %d duplicate(s) (ERROR lines above): %s; exiting %d "
            "until they are resolved",
            stats.refused,
            stats.refused_raw,
            stats.duplicates,
            done,
            REFUSED_EXIT_CODE,
        )
        return REFUSED_EXIT_CODE
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set and --database-url not provided")
        return 2
    if args.dry_run or args.apply or args.expect_plan or args.out_csv:
        return main_checked(args)
    try:
        stats = promote(args.database_url, batch_size=args.batch_size)
    except LegacyKickoffsPresent as e:
        logger.error("Refusing to promote — nothing written: %s", e)
        return 1
    logger.info(
        "Promotion complete: leagues=%d teams=%d matches=%d match_stats=%d odds=%d "
        "skipped_no_result=%d ht_dropped=%d left_cancelled=%d merged=%d unchanged=%d refused=%d duplicates=%d "
        "ht_conflicts=%d refused_raw=%d",
        stats.leagues,
        stats.teams,
        stats.matches,
        stats.match_stats,
        stats.odds,
        stats.skipped_no_result,
        stats.ht_dropped,
        stats.left_cancelled,
        stats.merged,
        stats.unchanged,
        stats.refused,
        stats.duplicates,
        stats.ht_conflicts,
        stats.refused_raw,
    )
    if stats.refused or stats.duplicates or stats.refused_raw:
        logger.error(
            "REFUSED %d raw row(s), screened out %d and found %d duplicate(s) (ERROR lines above): an existing "
            "row's score disagrees, it is a stale fixture row or ambiguous, the match sits under other team ids "
            "(alias slot), or football-data's own row sits beside the ESPN row; screened out: an unknown league "
            "code, a Div naming another division, or a mis-decoded team name. Everything else is committed; "
            "exiting %d until they are resolved.",
            stats.refused,
            stats.refused_raw,
            stats.duplicates,
            REFUSED_EXIT_CODE,
        )
        return REFUSED_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
