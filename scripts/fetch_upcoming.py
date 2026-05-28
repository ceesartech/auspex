"""Fetch upcoming fixtures from ESPN's free public API.

Populates the `matches` table with future (status='scheduled') fixtures
for the configured leagues of the chosen sport. Matched to existing
teams by normalized name; new teams are inserted on the fly.

ESPN's scoreboard endpoint returns the current scoring window (a few
days) plus you can pass ?dates=YYYYMMDD to walk forward up to ~2 weeks.

Usage:
    python scripts/fetch_upcoming.py                               # soccer, 14 days
    python scripts/fetch_upcoming.py --sport nhl                   # NHL, 14 days
    python scripts/fetch_upcoming.py --sport soccer --leagues eng.1,ger.1
    python scripts/fetch_upcoming.py --sport nhl --days 7
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from typing import Callable

import psycopg2
import requests
from psycopg2.extras import Json, RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_upcoming")

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"


def _season_soccer(dt: datetime) -> str:
    """Soccer season runs Aug→May. Aug-Dec → YYYY-YYYY+1; Jan-Jul → YYYY-1-YYYY."""
    if dt.month >= 7:
        return f"{dt.year}-{dt.year + 1}"
    return f"{dt.year - 1}-{dt.year}"


def _season_nhl(dt: datetime) -> str:
    """NHL season runs Oct→Jun. Sep-Dec → YYYY-YYYY+1; Jan-Aug → YYYY-1-YYYY.
    Preseason starts late September so the cutoff is month 9."""
    if dt.month >= 9:
        return f"{dt.year}-{dt.year + 1}"
    return f"{dt.year - 1}-{dt.year}"


@dataclass(frozen=True)
class SportConfig:
    """Per-sport ingestion config consumed by the generic ESPN fetcher.

    Adding a new sport here is the minimum work to wire up upcoming-fixture
    ingestion: define the ESPN path, the league slug → (code, name,
    country) map, the team-name normalisation rules, and the season string
    convention. Everything else (HTTP, upserts, fuzzy matching) is shared.
    """

    sport: str  # value stored in leagues.sport / teams.sport
    espn_path: str  # ESPN scoreboard segment after /apis/.../sports/
    leagues: dict[str, tuple[str, str, str]]  # slug -> (code, name, country)
    season_func: Callable[[datetime], str]
    # Club name tokens stripped only for fuzzy matching. Soccer has many
    # ("FC", "AFC", "BK"); NHL teams are unambiguous and need none.
    club_suffixes: frozenset[str] = frozenset()
    club_prefixes: frozenset[str] = frozenset()


# ── Soccer config ─────────────────────────────────────────────────────
# ESPN league slug -> our canonical (football-data code, league name, country).
# Keep aligned with promote_raw.LEAGUE_MAP so the same `leagues` row is reused
# regardless of which loader wrote it first.
#
# European leagues run Aug-May; summer-season leagues (MLS, Brasileirão,
# Eliteserien, Allsvenskan) plus international tournaments fill the May-Aug
# gap so the pipeline always has fixtures to predict on. The football-data
# codes for non-European leagues are best-effort labels — football-data.co.uk
# doesn't publish CSVs for these, so models trained on EU data won't be
# accurate on them until we add additional historical sources.
SOCCER_LEAGUES: dict[str, tuple[str, str, str]] = {
    # ── European top flights (active Aug-May) ───────────────────────
    "eng.1": ("E0", "Premier League", "England"),
    "eng.2": ("E1", "Championship", "England"),
    "ger.1": ("D1", "Bundesliga", "Germany"),
    "ita.1": ("I1", "Serie A", "Italy"),
    "esp.1": ("SP1", "La Liga", "Spain"),
    "fra.1": ("F1", "Ligue 1", "France"),
    "ned.1": ("N1", "Eredivisie", "Netherlands"),
    "por.1": ("P1", "Primeira Liga", "Portugal"),
    "bel.1": ("B1", "Pro League", "Belgium"),
    "tur.1": ("T1", "Süper Lig", "Turkey"),
    "gre.1": ("G1", "Super League", "Greece"),
    "sco.1": ("SC0", "Premiership", "Scotland"),
    # ── Americas (active Feb-Dec / Apr-Nov, varies) ─────────────────
    "usa.1": ("MLS", "MLS", "USA"),
    "bra.1": ("BR1", "Brasileirão Série A", "Brazil"),
    "arg.1": ("AR1", "Primera División", "Argentina"),
    "mex.1": ("MX1", "Liga MX", "Mexico"),
    "chi.1": ("CL1", "Primera División", "Chile"),
    "col.1": ("CO1", "Primera A", "Colombia"),
    # ── Asia-Pacific (varies) ───────────────────────────────────────
    # ESPN doesn't carry K League — `kor.1` returns HTTP 400. Removed.
    "jpn.1": ("JP1", "J1 League", "Japan"),
    "chn.1": ("CN1", "Chinese Super League", "China"),
    "aus.1": ("AU1", "A-League", "Australia"),
    # ── Nordic (summer-active filler) ───────────────────────────────
    "nor.1": ("NO1", "Eliteserien", "Norway"),
    "swe.1": ("SE1", "Allsvenskan", "Sweden"),
    # ── International competitions ─────────────────────────────────
    "fifa.world": ("WC", "FIFA World Cup", "International"),
    "concacaf.gold": ("GOLD", "CONCACAF Gold Cup", "International"),
    "uefa.champions": ("UCL", "UEFA Champions League", "International"),
    "uefa.europa": ("UEL", "UEFA Europa League", "International"),
    "conmebol.libertadores": ("LIB", "Copa Libertadores", "International"),
}


SOCCER_CLUB_SUFFIXES = frozenset(
    {"fc", "bk", "sk", "ac", "cf", "sc", "afc", "rfc", "cfc", "fk", "if", "tf", "se", "kc", "bc", "ks"}
)
SOCCER_CLUB_PREFIXES = frozenset({"fc", "afc", "as", "sc", "ks"})


# ── NHL config ────────────────────────────────────────────────────────
# ESPN exposes the NHL under hockey/nhl with a single "league" slug.
# We keep the registry-shape consistent with soccer (slug → triple) even
# though NHL has just one entry, so callers can do --leagues nhl just as
# they would --leagues eng.1.
NHL_LEAGUES: dict[str, tuple[str, str, str]] = {
    "nhl": ("NHL", "NHL", "USA"),
}


SPORT_CONFIGS: dict[str, SportConfig] = {
    "soccer": SportConfig(
        sport="soccer",
        espn_path="soccer",
        leagues=SOCCER_LEAGUES,
        season_func=_season_soccer,
        club_suffixes=SOCCER_CLUB_SUFFIXES,
        club_prefixes=SOCCER_CLUB_PREFIXES,
    ),
    "nhl": SportConfig(
        sport="nhl",
        espn_path="hockey/nhl",
        leagues=NHL_LEAGUES,
        season_func=_season_nhl,
    ),
}


def normalize_team(name: str) -> str:
    """Loose normalisation for the teams.normalized_name unique key."""
    return " ".join(name.strip().lower().split())


def _strip_club_tokens(norm: str, cfg: SportConfig) -> str:
    """Aggressive normalisation: drop common club abbreviation tokens so
    'rosenborg bk' and 'rosenborg' compare equal. Used only for fuzzy
    matching, not as the persisted normalized_name. No-op for sports
    without club-token conventions (NHL)."""
    if not cfg.club_suffixes and not cfg.club_prefixes:
        return norm
    tokens = norm.split()
    while tokens and tokens[-1] in cfg.club_suffixes:
        tokens.pop()
    while tokens and tokens[0] in cfg.club_prefixes:
        tokens.pop(0)
    return " ".join(tokens) if tokens else norm


def fetch_day(cfg: SportConfig, league_slug: str, day: date) -> list[dict]:
    url = f"{ESPN_BASE}/{cfg.espn_path}/{league_slug}/scoreboard"
    params = {"dates": day.strftime("%Y%m%d")}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning("ESPN fetch failed for %s/%s %s: %s", cfg.sport, league_slug, day, e)
        return []
    return r.json().get("events", []) or []


def ensure_league(cur, cfg: SportConfig, code: str, name: str, country: str) -> str | None:
    cur.execute(
        """
        INSERT INTO leagues (name, country, sport, external_ids)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (name, country, sport) DO UPDATE
            SET external_ids = leagues.external_ids || EXCLUDED.external_ids
        RETURNING id
        """,
        (name, country, cfg.sport, Json({"football_data": code} if cfg.sport == "soccer" else {"espn": code})),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def ensure_team(cur, cfg: SportConfig, name: str, league_id: str) -> str | None:
    """Resolve `name` to a team_id, reusing an existing row when possible.

    The match strategy is three-tier so vendor team names (e.g. ESPN's
    "Rosenborg BK") reuse the same row as football-data's "Rosenborg"
    instead of creating an orphan that has no historical matches:

      1. Exact normalised match on (normalized_name, sport).
      2. Fuzzy match against teams in the same league: drop common club
         abbreviation tokens (BK, FC, etc.) from both sides and compare
         via SequenceMatcher. Reuse if ratio >= 0.85. Soccer-only — NHL
         team names are unambiguous so we skip the fuzzy step.
      3. INSERT a new row only if neither lookup found a candidate.
    """
    norm = normalize_team(name)

    # 1. Exact normalised match
    cur.execute(
        "SELECT id FROM teams WHERE normalized_name = %s AND sport = %s",
        (norm, cfg.sport),
    )
    row = cur.fetchone()
    if row:
        return row["id"]

    # 2. Fuzzy match within the same league, after stripping club tokens.
    if league_id and (cfg.club_suffixes or cfg.club_prefixes):
        cur.execute(
            "SELECT id, normalized_name FROM teams WHERE league_id = %s AND sport = %s",
            (league_id, cfg.sport),
        )
        candidates = cur.fetchall()
        if candidates:
            target = _strip_club_tokens(norm, cfg)
            best_id = None
            best_score = 0.0
            for c in candidates:
                c_stripped = _strip_club_tokens(c["normalized_name"], cfg)
                if not target or not c_stripped:
                    continue
                score = SequenceMatcher(None, c_stripped, target).ratio()
                if score > best_score:
                    best_score = score
                    best_id = c["id"]
            if best_id and best_score >= 0.85:
                return best_id

    # 3. Insert new team. ON CONFLICT guard for race conditions on the
    # unique (normalized_name, sport) constraint.
    cur.execute(
        """
        INSERT INTO teams (name, normalized_name, league_id, sport)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (normalized_name, sport) DO UPDATE
            SET league_id = COALESCE(teams.league_id, EXCLUDED.league_id)
        RETURNING id
        """,
        (name, norm, league_id, cfg.sport),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def insert_scheduled_match(cur, cfg: SportConfig, league_id, home_id, away_id, match_dt, venue) -> int:
    cur.execute(
        """
        INSERT INTO matches (league_id, home_team_id, away_team_id, match_date,
                             status, venue, season)
        VALUES (%s, %s, %s, %s, 'scheduled', %s, %s)
        ON CONFLICT (home_team_id, away_team_id, match_date) DO UPDATE
            SET status = CASE
                  WHEN matches.status = 'finished' THEN matches.status
                  ELSE EXCLUDED.status
                END,
                venue = COALESCE(matches.venue, EXCLUDED.venue),
                updated_at = NOW()
        """,
        (league_id, home_id, away_id, match_dt, venue, cfg.season_func(match_dt)),
    )
    return cur.rowcount


def process_event(cur, cfg: SportConfig, league_id: str, event: dict) -> bool:
    comp = (event.get("competitions") or [{}])[0]
    competitors = comp.get("competitors") or []
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not (home and away):
        return False

    home_name = home.get("team", {}).get("displayName") or home.get("team", {}).get("name")
    away_name = away.get("team", {}).get("displayName") or away.get("team", {}).get("name")
    if not (home_name and away_name):
        return False

    try:
        match_dt = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return False

    state = (event.get("status", {}).get("type", {}).get("state") or "pre").lower()
    if state != "pre":
        return False  # skip in-progress and finished events here

    venue = (comp.get("venue") or {}).get("fullName")

    home_id = ensure_team(cur, cfg, home_name, league_id)
    away_id = ensure_team(cur, cfg, away_name, league_id)
    if not (home_id and away_id):
        return False

    return bool(insert_scheduled_match(cur, cfg, league_id, home_id, away_id, match_dt, venue))


def fetch_all(database_url: str, cfg: SportConfig, leagues: list[str], days: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    today = date.today()
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for slug in leagues:
                if slug not in cfg.leagues:
                    logger.warning("Unknown %s league slug: %s — skipping", cfg.sport, slug)
                    continue
                code, name, country = cfg.leagues[slug]
                league_id = ensure_league(cur, cfg, code, name, country)
                if not league_id:
                    continue
                n = 0
                for offset in range(days):
                    day = today + timedelta(days=offset)
                    for ev in fetch_day(cfg, slug, day):
                        if process_event(cur, cfg, league_id, ev):
                            n += 1
                counts[slug] = n
                conn.commit()
                logger.info("Fetched %d upcoming %s fixtures for %s (%s)", n, cfg.sport, name, slug)
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sport",
        default="soccer",
        choices=sorted(SPORT_CONFIGS.keys()),
        help="Which sport's fixtures to fetch (default: soccer).",
    )
    p.add_argument(
        "--leagues",
        default=None,
        help="Comma-separated ESPN league slugs (default: all known for --sport).",
    )
    p.add_argument(
        "--days",
        type=int,
        default=14,
        help="How many days forward to look (default: 14 — wide enough "
        "to start capturing tournament-format competitions like the "
        "World Cup as group stages approach, narrow enough to keep "
        "per-run feature compute cost low).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set and --database-url not provided")
        return 2
    cfg = SPORT_CONFIGS[args.sport]
    leagues_arg = args.leagues if args.leagues is not None else ",".join(cfg.leagues.keys())
    leagues = [s.strip() for s in leagues_arg.split(",") if s.strip()]
    counts = fetch_all(args.database_url, cfg, leagues, args.days)
    total = sum(counts.values())
    logger.info("Fetched %d upcoming %s fixtures across %d leagues", total, cfg.sport, len(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
