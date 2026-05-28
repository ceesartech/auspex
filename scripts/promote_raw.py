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

Usage (inside the api container or anywhere DATABASE_URL is set):
    python /app/scripts/promote_raw.py
    python /app/scripts/promote_raw.py --database-url postgres://...
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import psycopg2
from psycopg2.extras import Json, RealDictCursor, execute_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("promote_raw")

# League code → canonical (name, country).
# Codes here MUST stay aligned with fetch_upcoming.LEAGUE_MAP and the
# country codes used by load_football_data_extra.py.
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


def parse_match_date(raw_date: str, raw_time: str | None) -> datetime | None:
    """football-data uses DD/MM/YYYY (newer) or DD/MM/YY (older)."""
    if not raw_date:
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            d = datetime.strptime(raw_date, fmt)
            if raw_time:
                try:
                    t = datetime.strptime(raw_time, "%H:%M")
                    d = d.replace(hour=t.hour, minute=t.minute)
                except ValueError:
                    pass
            return d
        except ValueError:
            continue
    return None


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
    rows = [(name, normalize_team_name(name), league_id_by_team.get(name), "soccer") for name in team_names]
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


def insert_match(
    cur, *, league_id, home_id, away_id, match_dt, season, home_score, away_score, ht_home, ht_away
) -> str | None:
    cur.execute(
        """
        INSERT INTO matches (
            league_id, home_team_id, away_team_id, match_date, status,
            home_score, away_score, half_time_home, half_time_away, season
        )
        VALUES (%s, %s, %s, %s, 'finished', %s, %s, %s, %s, %s)
        ON CONFLICT (home_team_id, away_team_id, match_date) DO UPDATE
            SET home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score,
                half_time_home = EXCLUDED.half_time_home,
                half_time_away = EXCLUDED.half_time_away,
                status = 'finished',
                updated_at = NOW()
        RETURNING id
        """,
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


def season_label(code: str) -> str:
    """Convert football-data season code (e.g. '2425') to '2024-2025'.

    Y2K rollover: codes 50-99 → 19xx; codes 00-49 → 20xx. football-data
    has been publishing since 1993 so the 1990s codes are real.
    """
    if not code or len(code) != 4 or not code.isdigit():
        return code or ""
    start = int(code[:2])
    end = int(code[2:])
    start_year = (1900 if start >= 50 else 2000) + start
    end_year = (1900 if end >= 50 else 2000) + end
    if end_year < start_year:
        end_year += 100
    return f"{start_year}-{end_year}"


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
                match_id = insert_match(
                    cur,
                    league_id=league_ids[code],
                    home_id=home_id,
                    away_id=away_id,
                    match_dt=match_dt,
                    season=season_label(r.get("season", "")),
                    home_score=to_int(r.get("FTHG")),
                    away_score=to_int(r.get("FTAG")),
                    ht_home=to_int(r.get("HTHG")),
                    ht_away=to_int(r.get("HTAG")),
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
    stats = promote(args.database_url, batch_size=args.batch_size)
    logger.info(
        "Promotion complete: leagues=%d teams=%d matches=%d match_stats=%d odds=%d",
        stats.leagues,
        stats.teams,
        stats.matches,
        stats.match_stats,
        stats.odds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
