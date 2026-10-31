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
is written onto that row, never beside it (find_existing_match,
update_existing_match), and an ESPN row's score is never overwritten. A raw
row refused that way is logged at ERROR, and the run exits 3 once
everything else is committed.

Usage (inside the api container or anywhere DATABASE_URL is set):
    python /app/scripts/promote_raw.py
    python /app/scripts/promote_raw.py --database-url postgres://...
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Iterable
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import Json, RealDictCursor, execute_values

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
# league decides: a 4-digit season from one of these leagues is a calendar
# year. Must cover every league code in load_football_data_extra.COUNTRY_MAP
# (pinned by tests/unit/test_promote_raw.py).
CALENDAR_SEASON_LEAGUES: frozenset[str] = frozenset(
    {"AR1", "AT1", "BR1", "CH1", "CN1", "DK1", "FI1", "IE1", "JP1", "MLS", "MX1", "NO1", "PL1", "RO1", "RU1", "SE1"}
    | {code for code, (_name, country) in LEAGUE_MAP.items() if country == "International"}
)
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
    refused: int = 0  # not written: an ESPN row's score disagrees, or which row is meant is ambiguous
    ht_conflicts: int = 0  # an ESPN row kept its half-time where football-data's differs


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
    """Raw rows that promote() would now insert as a SECOND matches row: the
    corrected (UTC) kickoff is free, while the row promote_raw wrote for the
    same pair at the legacy kickoff (the UK wall clock read in `db_tz`)
    exists. `existing` holds (home_id, away_id, UTC kickoff). Pure."""
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
        key_home, key_away = str(home_id), str(away_id)
        if (key_home, key_away, legacy) in existing and (key_home, key_away, corrected) not in existing:
            stranded.append(r)
    return stranded


def check_legacy_kickoffs(cur, rows: list[dict], league_ids: dict[str, str], team_ids: dict[str, str]) -> None:
    """Raise LegacyKickoffsPresent, before promote() writes a single match,
    when any raw row would get a corrected-kickoff twin
    (stranded_legacy_kickoffs). Every run re-promotes the whole raw table,
    and on 2026-10-31 about 63,800 timed summer-time rows still sat at the
    legacy kickoff, so one run after the timezone fix would have inserted
    that many duplicate finished matches into the training corpus. Since
    find_existing_match the run would update the legacy row instead, leaving
    it 1 h late for good and without a word; this keeps promote() refusing
    until scripts/repair_football_data_kickoffs.py has moved them (its
    --check runs this)."""
    cur.execute("SHOW TimeZone")
    db_tz = ZoneInfo(cur.fetchone()["TimeZone"])  # loud if Postgres reports a zone Python cannot load
    cur.execute(EXISTING_SOCCER_KICKOFFS_SQL)
    existing = {(r["home_id"], r["away_id"], r["match_date"].astimezone(timezone.utc)) for r in cur.fetchall()}
    stranded = stranded_legacy_kickoffs(rows, league_ids, team_ids, existing, db_tz)
    if stranded:
        sample = "; ".join(f"{fixture_label(r)} {r.get('Time')}" for r in stranded[:5])
        raise LegacyKickoffsPresent(
            f"{len(stranded)} raw row(s) would be promoted as a second matches row: promote_raw wrote them at the "
            f"UK wall clock read as {db_tz} (before 2026-10-31) and their corrected UTC kickoff is free. Move "
            f"those rows' match_date from UK local time to UTC first (scripts/ops/repair_football_data_kickoffs.sh: "
            f"--dedupe, then --shift), then re-run. First {min(5, len(stranded))}: {sample}"
        )


def upsert_leagues(cur, codes: set[str]) -> dict[str, str]:
    """Insert leagues for each code; return {code: league_id}."""
    rows = []
    for code in codes:
        if code not in LEAGUE_MAP:
            logger.warning("unknown league code %s — skipping", code)
            continue
        name, country = LEAGUE_MAP[code]
        rows.append((name, country, "soccer", code))
    if not rows:
        return {}

    execute_values(
        cur,
        """
        INSERT INTO leagues (name, country, sport, external_ids)
        VALUES %s
        ON CONFLICT (name, country, sport) DO UPDATE
            SET external_ids = leagues.external_ids || EXCLUDED.external_ids
        """,
        [(name, country, sport, Json({"football_data": code})) for name, country, sport, code in rows],
    )
    cur.execute(
        "SELECT id, external_ids->>'football_data' AS code " "FROM leagues WHERE external_ids ? 'football_data'"
    )
    return {r["code"]: r["id"] for r in cur.fetchall()}


def upsert_teams(cur, team_names: set[str], league_id_by_team: dict[str, str]) -> dict[str, str]:
    """Insert all teams referenced; return {normalized_name: team_id}."""
    if not team_names:
        return {}
    # Dedupe on the conflict key (normalized_name, sport): two RAW names
    # can normalize identically ("Alaves"/"Alavés"), and ON CONFLICT DO
    # UPDATE cannot touch the same row twice within one command
    # (CardinalityViolation) — keep the first raw spelling.
    seen: set[str] = set()
    rows = []
    for name in sorted(team_names):
        norm = normalize_team_name(name)
        if norm in seen:
            continue
        seen.add(norm)
        rows.append((name, norm, league_id_by_team.get(name), "soccer"))
    execute_values(
        cur,
        """
        INSERT INTO teams (name, normalized_name, league_id, sport)
        VALUES %s
        ON CONFLICT (normalized_name, sport) DO UPDATE
            SET league_id = COALESCE(EXCLUDED.league_id, teams.league_id)
        """,
        rows,
    )
    cur.execute("SELECT id, normalized_name FROM teams WHERE sport = 'soccer'")
    return {r["normalized_name"]: r["id"] for r in cur.fetchall()}


# promote() inserts only where find_existing_match found no row, so this
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
# main()'s exit code when write_match refused a raw row (everything else committed),
# as fetch_upcoming exits on a refused fork.
REFUSED_EXIT_CODE = 3

EXISTING_MATCH_SQL = """
    SELECT id::text AS id, match_date, status, home_score, away_score, half_time_home, half_time_away,
           external_ids->>'espn' AS espn_id, metadata->>'result_source' AS result_source
      FROM matches
     WHERE home_team_id = %s AND away_team_id = %s
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
    """The kickoffs find_existing_match looks at, as aware UTC bounds, both
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


def find_existing_match(cur, home_id, away_id, match_dt: datetime, fixture_day: date) -> tuple[dict | None, str | None]:
    """choose_existing over this team pair's rows in existing_match_window."""
    lo, hi = existing_match_window(match_dt, fixture_day)
    cur.execute(EXISTING_MATCH_SQL, (home_id, away_id, lo, hi))
    return choose_existing([dict(r) for r in cur.fetchall()])


def update_existing_match(cur, existing: dict, result: RawResult) -> tuple[str | None, str]:
    """Write a football-data result onto the row choose_existing picked;
    returns (match_id or None, outcome).

    'cancelled': the row is 'cancelled' and stays as it is (see
    INSERT_MATCH_SQL). 'score_conflict': an ESPN row holds a different final
    score, which is never overwritten; nothing written. 'ht_conflict':
    written, but an ESPN row keeps a half-time that differs from
    football-data's. 'score_changed': a row without ESPN identity
    (football-data's own) took football-data's new score, as the upsert always
    let it. 'changed': the row moved after the lookup; nothing written.
    Otherwise 'updated'. A half-time pair is only ever replaced by a complete
    one, an ESPN row's only when it has none; the kickoff, season and
    identity stay the row's own."""
    if existing["status"] == "cancelled":
        return None, "cancelled"
    old_ft = (existing["home_score"], existing["away_score"])
    old_ht = (existing["half_time_home"], existing["half_time_away"])
    new_ft = (result.home_score, result.away_score)
    raw_ht = (result.ht_home, result.ht_away) if result.ht_home is not None and result.ht_away is not None else None
    outcome = "updated"
    if has_espn_identity(existing.get("espn_id"), existing.get("result_source")):
        if old_ft != (None, None) and old_ft != new_ft:
            return None, "score_conflict"
        new_ht = raw_ht if raw_ht is not None and old_ht == (None, None) else old_ht
        if raw_ht is not None and old_ht != (None, None) and old_ht != raw_ht:
            outcome = "ht_conflict"
    else:
        new_ht = raw_ht or old_ht
        if old_ft != (None, None) and old_ft != new_ft:
            outcome = "score_changed"
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
    pair (find_existing_match, update_existing_match) or a new one
    (insert_match). Returns its id, or None when nothing was written (counted
    in `stats` and logged; the caller then writes no stats or odds)."""
    label = fixture_label(row)
    fixture_day = parse_wall_clock(row.get("Date") or "", None)
    if fixture_day is None:
        raise ValueError(f"write_match needs a football-data Date, got {row.get('Date')!r} ({label})")
    existing, ambiguity = find_existing_match(cur, home_id, away_id, match_dt, fixture_day.date())
    if ambiguity:
        stats.refused += 1
        logger.error("Not promoting %s: %s — merge them first", label, ambiguity)
        return None
    if existing is None:
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
    if outcome == "cancelled":
        stats.left_cancelled += 1
        logger.warning(
            "Left %s as it is: %s is 'cancelled' but football-data now has a full-time score for it — check "
            "which is right",
            label,
            where,
        )
        return None
    if outcome in ("score_conflict", "changed"):
        stats.refused += 1
        logger.error(
            "Not promoting %s: %s %s (football-data %d-%d) — nothing written; check which score is right",
            label,
            where,
            (
                f"is an ESPN row with final score {existing['home_score']}-{existing['away_score']}"
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
            existing["half_time_home"],
            existing["half_time_away"],
            where,
            result.ht_home,
            result.ht_away,
        )
    elif outcome == "score_changed":
        logger.warning(
            "%s: football-data changed the score on %s from %s-%s to %d-%d",
            label,
            where,
            existing["home_score"],
            existing["away_score"],
            result.home_score,
            result.away_score,
        )
    if existing is not None and not same_kickoff(existing["match_date"], match_dt):
        stats.merged += 1
        logger.info("%s written onto %s (football-data kickoff %s)", label, where, match_dt)
    return match_id


def insert_match_stats(
    cur, match_id, team_id, *, shots, shots_on_target, corners, fouls, yellow_cards, red_cards
) -> None:
    cur.execute(
        """
        INSERT INTO match_stats (
            match_id, team_id, shots, shots_on_target, corners, fouls,
            yellow_cards, red_cards
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (match_id, team_id) DO UPDATE
            SET shots = EXCLUDED.shots,
                shots_on_target = EXCLUDED.shots_on_target,
                corners = EXCLUDED.corners,
                fouls = EXCLUDED.fouls,
                yellow_cards = EXCLUDED.yellow_cards,
                red_cards = EXCLUDED.red_cards
        """,
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
    decides (CALENDAR_SEASON_LEAGUES). Without one, a 4-digit code expands
    only when its halves are consecutive years; anything else is a year.
    """
    code = (code or "").strip()
    cross_year = _CROSS_YEAR_SEASON.fullmatch(code)
    if cross_year and int(cross_year.group(2)) - int(cross_year.group(1)) == 1:
        return f"{cross_year.group(1)}-{cross_year.group(2)}"
    if len(code) != 4 or not code.isdigit():
        return code
    if league in CALENDAR_SEASON_LEAGUES:
        return code
    start, end = int(code[:2]), int(code[2:])
    if end != (start + 1) % 100:
        return code  # not a 'YYZZ' code: a calendar year
    start_year = (1900 if start >= 50 else 2000) + start
    return f"{start_year}-{start_year + 1}"


def promote(database_url: str, *, batch_size: int = 1000) -> Promotion:
    stats = Promotion()
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM raw_football_data")
            rows = cur.fetchall()

            if not rows:
                logger.info("raw_football_data is empty — nothing to promote")
                return stats

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

                if stats.matches % batch_size == 0:
                    conn.commit()
                    logger.info("Committed batch: matches=%d odds=%d", stats.matches, stats.odds)

            conn.commit()
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL"), help="Postgres URL. Defaults to $DATABASE_URL."
    )
    p.add_argument("--batch-size", type=int, default=1000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set and --database-url not provided")
        return 2
    try:
        stats = promote(args.database_url, batch_size=args.batch_size)
    except LegacyKickoffsPresent as e:
        logger.error("Refusing to promote — nothing written: %s", e)
        return 1
    logger.info(
        "Promotion complete: leagues=%d teams=%d matches=%d match_stats=%d odds=%d "
        "skipped_no_result=%d ht_dropped=%d left_cancelled=%d merged=%d refused=%d ht_conflicts=%d",
        stats.leagues,
        stats.teams,
        stats.matches,
        stats.match_stats,
        stats.odds,
        stats.skipped_no_result,
        stats.ht_dropped,
        stats.left_cancelled,
        stats.merged,
        stats.refused,
        stats.ht_conflicts,
    )
    if stats.refused:
        logger.error(
            "REFUSED %d raw row(s) (ERROR lines above): an existing row's score disagrees or the row is ambiguous. "
            "Everything else is committed; exiting %d until they are resolved.",
            stats.refused,
            REFUSED_EXIT_CODE,
        )
        return REFUSED_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
