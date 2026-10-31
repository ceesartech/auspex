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
from datetime import datetime, timezone, tzinfo
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
    that many duplicate finished matches into the training corpus."""
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
            f"those rows' match_date from UK local time to UTC first (an operator task), then re-run. "
            f"First {min(5, len(stranded))}: {sample}"
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


# The conflict branch never downgrades the row already there: each score
# pair is replaced only by a complete pair (a load without half-time
# columns — the /new/ country files — used to NULL a half-time the ESPN
# results path had written), 'finished' is set only with a full-time pair
# in hand, season is left alone, and a 'cancelled' row is not touched at
# all (RETURNING then yields no row): that status is recorded on purpose,
# e.g. by scripts/fix_unplayed_soccer_rows.py.
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
                match_id = insert_match(
                    cur,
                    league_id=league_ids[code],
                    home_id=home_id,
                    away_id=away_id,
                    match_dt=match_dt,
                    season=season_label(r.get("season", ""), code),
                    home_score=result.home_score,
                    away_score=result.away_score,
                    ht_home=result.ht_home,
                    ht_away=result.ht_away,
                )
                if not match_id:
                    stats.left_cancelled += 1
                    logger.warning(
                        "Left %s as it is: the existing row is 'cancelled' but football-data now has a "
                        "full-time score for it — check which is right",
                        fixture_label(r),
                    )
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
        "skipped_no_result=%d ht_dropped=%d left_cancelled=%d",
        stats.leagues,
        stats.teams,
        stats.matches,
        stats.match_stats,
        stats.odds,
        stats.skipped_no_result,
        stats.ht_dropped,
        stats.left_cancelled,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
