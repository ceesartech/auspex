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
import json
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


def _season_nba(dt: datetime) -> str:
    """NBA season runs Oct→Jun. Same cutoff month as NHL — preseason
    starts in early October, so Oct-Dec → YYYY-YYYY+1; Jan-Sep → YYYY-1-YYYY.
    We use month 10 as the cutoff (vs NHL's 9) because NBA preseason is
    later than NHL's by about a month."""
    if dt.month >= 10:
        return f"{dt.year}-{dt.year + 1}"
    return f"{dt.year - 1}-{dt.year}"


def _season_nfl(dt: datetime) -> str:
    """NFL season runs Sep→Feb (regular season Sep-Jan, playoffs Jan,
    Super Bowl early Feb). Aug-Dec → YYYY-YYYY+1 (the new season);
    Jan-Jul → YYYY-1-YYYY (the season that ended that Feb). Cutoff
    month 8 catches preseason (early August)."""
    if dt.month >= 8:
        return f"{dt.year}-{dt.year + 1}"
    return f"{dt.year - 1}-{dt.year}"


def _season_tennis(dt: datetime) -> str:
    """ATP/WTA tour year is a calendar year (Jan-Nov regular season +
    Nov Finals). Pre-season exhibitions in December roll into the next
    season for the ATP rankings rollover. Single-year string so
    cross-year tournaments don't fragment the season label."""
    return str(dt.year)


def _season_mma(dt: datetime) -> str:
    """UFC events span the full calendar year (~40-45 cards/yr). No
    cross-year season — each year is its own self-contained 'season'
    for stats purposes."""
    return str(dt.year)


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
    # True for 1v1 sports (tennis, MMA, boxing) where ESPN's competitors
    # don't carry homeAway and use 'athlete' instead of 'team'. Triggers
    # the positional fallback in process_event and the per-player team
    # rows (one "team" per player in the teams table — pragmatic reuse
    # of the team-sports schema for individual sports).
    is_individual: bool = False


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


# ── NBA config ────────────────────────────────────────────────────────
# ESPN exposes the NBA under basketball/nba with a single league slug.
# Mirror NHL's shape so the same registry conventions work without
# special-casing.
NBA_LEAGUES: dict[str, tuple[str, str, str]] = {
    "nba": ("NBA", "NBA", "USA"),
}


# ── NHL config ────────────────────────────────────────────────────────
# ESPN exposes the NHL under hockey/nhl with a single "league" slug.
# We keep the registry-shape consistent with soccer (slug → triple) even
# though NHL has just one entry, so callers can do --leagues nhl just as
# they would --leagues eng.1.
NHL_LEAGUES: dict[str, tuple[str, str, str]] = {
    "nhl": ("NHL", "NHL", "USA"),
}


# ── NFL config ────────────────────────────────────────────────────────
# ESPN exposes the NFL under football/nfl with a single league slug.
# Same shape as NBA/NHL — single league, no slug variations to worry
# about. NCAA / preseason / Super Bowl all roll up under the same nfl
# slug on ESPN's scoreboard.
NFL_LEAGUES: dict[str, tuple[str, str, str]] = {
    "nfl": ("NFL", "NFL", "USA"),
}


# ── Tennis config ─────────────────────────────────────────────────────
# Two tours, both 1v1: ATP (men) and WTA (women). ESPN path is
# tennis/{atp,wta}/scoreboard. Country = "World" since both tours are
# global; the league row's external_ids.espn carries the tour code.
TENNIS_LEAGUES: dict[str, tuple[str, str, str]] = {
    "atp": ("ATP", "ATP Tour", "World"),
    "wta": ("WTA", "WTA Tour", "World"),
}


# ── MMA config ────────────────────────────────────────────────────────
# 1v1, professional mixed martial arts. ESPN's path is /mma/ufc/
# scoreboard for UFC events. PFL / Bellator / ONE could layer in as
# additional slugs once the dataset stabilises.
MMA_LEAGUES: dict[str, tuple[str, str, str]] = {
    "ufc": ("UFC", "UFC", "USA"),
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
        # ESPN's hockey API path: /sports/hockey/{league_slug}/scoreboard.
        # The slug "nhl" supplies the second segment, so espn_path here
        # is just "hockey" (NOT "hockey/nhl" — that would double up to
        # /sports/hockey/nhl/nhl/scoreboard and 404).
        espn_path="hockey",
        leagues=NHL_LEAGUES,
        season_func=_season_nhl,
    ),
    "nba": SportConfig(
        sport="nba",
        # ESPN's basketball API path: /sports/basketball/{league_slug}/scoreboard.
        # Same shape gotcha as NHL — espn_path is JUST "basketball" so
        # the slug "nba" doesn't double up to /basketball/nba/nba/.
        espn_path="basketball",
        leagues=NBA_LEAGUES,
        season_func=_season_nba,
    ),
    "nfl": SportConfig(
        sport="nfl",
        # ESPN's football API path: /sports/football/{league_slug}/scoreboard.
        # Same shape gotcha as NBA/NHL — espn_path is JUST "football" so
        # the slug "nfl" doesn't double up to /football/nfl/nfl/.
        espn_path="football",
        leagues=NFL_LEAGUES,
        season_func=_season_nfl,
    ),
    "tennis": SportConfig(
        sport="tennis",
        # ESPN's tennis API path: /sports/tennis/{atp|wta}/scoreboard.
        # Same shape gotcha as NHL/NBA/NFL — espn_path is JUST "tennis"
        # so the slug "atp" doesn't double up.
        espn_path="tennis",
        leagues=TENNIS_LEAGUES,
        season_func=_season_tennis,
        # ESPN tennis competitors use 'athlete' not 'team' and lack
        # the homeAway field — flip on positional / athlete handling
        # in process_event.
        is_individual=True,
    ),
    "mma": SportConfig(
        sport="mma",
        # ESPN MMA endpoint: /sports/mma/ufc/scoreboard. UFC cards
        # arrive as single events containing competitions[] (12-15
        # fights per card). Same espn_path-vs-slug split as the
        # team sports — espn_path is just "mma".
        espn_path="mma",
        leagues=MMA_LEAGUES,
        season_func=_season_mma,
        # MMA competitors use 'athlete' (no team) and lack homeAway;
        # positional fallback gives a consistent fighter1/fighter2
        # labeling.
        is_individual=True,
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


def insert_finished_match(
    cur,
    cfg: SportConfig,
    league_id,
    home_id,
    away_id,
    match_dt,
    venue,
    home_score: int,
    away_score: int,
    result_meta: dict,
) -> int:
    """Record a final result (audit doc §2.1 — the missing half of the
    feedback loop). Updates the row fetch_upcoming created — callers run
    _reconcile_match_row first so that row's match_date already matches
    ESPN's current time — or inserts it as finished when we never saw it
    as a fixture (grows the corpus). grade_completed_matches
    (already on the 15-min cadence) settles predictions + recs from here.
    Score columns follow the corpus conventions: real scores for team
    sports, winner-flag 1/0 for tennis/MMA."""
    cur.execute(
        """
        INSERT INTO matches (league_id, home_team_id, away_team_id, match_date,
                             status, venue, season, home_score, away_score, metadata)
        VALUES (%s, %s, %s, %s, 'finished', %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (home_team_id, away_team_id, match_date) DO UPDATE
            SET status = 'finished',
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score,
                metadata = COALESCE(matches.metadata, '{}'::jsonb) || EXCLUDED.metadata,
                venue = COALESCE(matches.venue, EXCLUDED.venue),
                updated_at = NOW()
        """,
        (
            league_id,
            home_id,
            away_id,
            match_dt,
            venue,
            cfg.season_func(match_dt),
            home_score,
            away_score,
            json.dumps(result_meta),
        ),
    )
    return cur.rowcount


def _competitor_name(competitor: dict, is_individual: bool) -> str | None:
    """Pull the canonical name from a competitor entry. Team sports use
    .team.displayName; individual sports use .athlete.displayName.
    Falls back to .name on either nested dict if the displayName is
    missing (ESPN occasionally omits it for less-tracked athletes)."""
    if is_individual:
        athlete = competitor.get("athlete") or {}
        return athlete.get("displayName") or athlete.get("shortName") or athlete.get("name")
    team = competitor.get("team") or {}
    return team.get("displayName") or team.get("name")


# ESPN's event.season.type enum. 1 = preseason, 2 = regular season,
# 3 = post-season. Verified live 2026-09-04 against the NFL, NBA and NHL
# scoreboards (NFL 2026-08-15 -> 7 type-1 events; NFL 2026-09-13 -> 13
# type-2; NBA 2023-10-05 / 2024-10-04 and NHL 2025-09-22 -> type 1).
_ESPN_SEASON_TYPES: dict[int, str] = {1: "preseason", 2: "regular", 3: "postseason"}

# Unknown enum values are logged ONCE per process (ESPN occasionally adds
# codes — e.g. all-star / exhibition windows). Storing nothing is the safe
# choice: readers treat a MISSING marker as "unknown", never as "regular".
_UNKNOWN_SEASON_TYPES_SEEN: set[str] = set()


def _event_season_type(event: dict | None) -> str | None:
    """Map ESPN `event.season.type` to our stored season_type label.

    Returns 'preseason' | 'regular' | 'postseason', or None when ESPN
    omitted the field or sent a code we don't recognise. None means
    "unknown" and is never written — a COALESCE(..., 'regular') anywhere
    downstream would silently re-admit preseason games into training.
    """
    raw = ((event or {}).get("season") or {}).get("type")
    if raw is None:
        return None
    try:
        mapped = _ESPN_SEASON_TYPES.get(int(raw))
    except (TypeError, ValueError):
        mapped = None
    if mapped is None:
        key = repr(raw)
        if key not in _UNKNOWN_SEASON_TYPES_SEEN:
            _UNKNOWN_SEASON_TYPES_SEEN.add(key)
            logger.warning(
                "Unrecognised ESPN season.type %s on event %s — storing no season_type "
                "(rows without the marker are treated as unknown downstream)",
                key,
                (event or {}).get("id"),
            )
    return mapped


def _event_context(event: dict | None) -> dict:
    """The slice of the ESPN *event* that individual competitions need but
    don't carry themselves. season.type lives on the event, while the
    competition is the unit of a match — for tennis the two are separated
    by two levels of nesting (event.groupings[].competitions[])."""
    return {"season_type": _event_season_type(event), "event_id": (event or {}).get("id")}


# Healing window: how far either side of ESPN's kickoff we look for the
# row a slid fixture already owns. Prod's measured twin shifts: min
# 0.08 h, p25 0.58 h, median 2 h, p75 24 h, max 70.5 h.
IDENTITY_WINDOW_HOURS = 72

# A heal that drags a row further than this is past the measured p75 and
# is worth an ERROR line even though the payload guard has cleared it.
LARGE_SHIFT_HOURS = 24


def _pair_key(home_name: str, away_name: str) -> tuple[str, str]:
    """Index key for a fixture's ORDERED team pair.

    Keyed on the ESPN display names rather than our team ids: the index is
    built from the raw payload before any team row is touched, ensure_team
    maps a name onto an id deterministically within a run, and the healing
    query is itself keyed on the ordered (home, away) pair."""
    return (home_name, away_name)


def build_pair_index(cfg: SportConfig, events: list[dict]) -> dict[tuple[str, str], list[datetime]]:
    """Map every ordered team pair in a sweep to the kickoff times ESPN
    published for it. This is what makes step-3 healing payload-aware.

    Healing resolves an incoming competition onto "the one unidentified
    'scheduled' row for this pair within +/- 72 h". That is wrong whenever
    the SAME pair plays twice inside the window and only one of the two
    rows exists: the surviving row belongs to the OTHER game, and healing
    would drag it onto this kickoff and then overwrite it with this game's
    result — silently settling one fixture's predictions and recs against
    another fixture's outcome, and erasing it from the corpus.

    Same-pair repeats inside 72 h are routine, not exotic: 21 NBA pairs
    41-65 h apart in 2026-04-18..05-12, 13 NHL pairs in 2026-04-22..05-11,
    and back-to-backs in the regular season too. Both games are always in
    the same sweep, so counting the payload blocks every such merge."""
    index: dict[tuple[str, str], list[datetime]] = {}
    for event in events or []:
        for comp, _context in _iter_event_competitions(event, cfg):
            parts = _competition_parts(comp, cfg)
            if parts is None:
                continue
            _home, _away, home_name, away_name, match_dt, _state, _venue, _espn = parts
            index.setdefault(_pair_key(home_name, away_name), []).append(match_dt)
    return index


def healing_allowed(
    pair_index: dict[tuple[str, str], list[datetime]] | None,
    home_name: str,
    away_name: str,
    match_dt: datetime,
    window_hours: int = IDENTITY_WINDOW_HOURS,
) -> bool:
    """False when THIS sweep published more than one competition for this
    ordered pair inside the healing window — i.e. when "the single
    unidentified row for this pair" is ambiguous by construction, whether
    or not both rows happen to exist in `matches`.

    A None index means "no payload knowledge" (a caller holding a bare
    competition) and leaves healing enabled."""
    if pair_index is None:
        return True
    times = pair_index.get(_pair_key(home_name, away_name)) or []
    lo = match_dt - timedelta(hours=window_hours)
    hi = match_dt + timedelta(hours=window_hours)
    near = [t for t in times if lo <= t <= hi]
    if len(near) <= 1:
        return True
    logger.warning(
        "Refusing fixture healing for %s vs %s at %s: ESPN published %d competitions for this "
        "pair within %dh (%s) — healing cannot tell them apart, so this competition gets its "
        "own row rather than risking a merge of two different fixtures",
        home_name,
        away_name,
        match_dt,
        len(near),
        window_hours,
        ", ".join(sorted(t.isoformat() for t in near)),
    )
    return False


def _resolve_match_id(
    cur,
    cfg: SportConfig,
    espn_id,
    home_id,
    away_id,
    match_dt,
    window_hours: int = IDENTITY_WINDOW_HOURS,
    allow_heal: bool = True,
):
    """Resolve an ESPN competition to an EXISTING matches row, or None.

    matches is keyed UNIQUE (home_team_id, away_team_id, match_date) and
    ESPN slides scheduled kickoff times, so one real fixture used to become
    several rows: predictions/odds/recommendations attached to the early
    row while the result landed on a later one (prod: 19,008 stale tennis
    rows, 1,727 recs that could never settle). Resolution order:

      1. external_ids->>'espn' — the stable identity, scoped to this sport
         (an ESPN id is only unique within a sport, so an unscoped lookup
         could collide with e.g. an NFL competition of the same number).
      2. the exact (home, away, match_date) triple — the legacy identity,
         still correct whenever ESPN hasn't moved the time.
      3. HEALING — exactly ONE 'scheduled' row for the same team pair
         within +/- `window_hours` that carries no ESPN id yet. This is
         what re-attaches the pre-fix orphans instead of forking a new row.

    Step 3 refuses to resolve when two or more rows qualify: a same-pair
    rematch inside 72 h is routine (playoff series, back-to-backs,
    two-legged ties) and guessing would corrupt a fixture. It logs the
    candidates so the ambiguity is visible rather than silent. That check
    only sees rows that EXIST, though — when the second fixture has no row
    yet, the first fixture's row looks like a lone healing candidate. The
    caller therefore passes allow_heal=False (see healing_allowed) whenever
    the sweep's own payload shows two competitions for the pair inside the
    window.

    Returns (match_id, source) where source is one of "espn" / "date" /
    "healed" / None; the public resolve_match_id() wrapper returns just the
    id. Callers use the source to tell a trusted realignment from a healed
    one.
    """
    if espn_id:
        # ORDER BY (not a bare LIMIT 1): a pre-fix duplicate can leave the
        # same ESPN id on two rows, and an unordered LIMIT 1 would answer
        # differently run to run, making the fragmentation permanent
        # instead of self-healing. LIMIT 2 so the duplicate is detectable.
        cur.execute(
            """
            SELECT m.id
              FROM matches m
              JOIN leagues l ON l.id = m.league_id
             WHERE m.external_ids->>'espn' = %s
               AND l.sport = %s
             ORDER BY m.match_date DESC
             LIMIT 2
            """,
            (str(espn_id), cfg.sport),
        )
        rows = cur.fetchall() or []
        if len(rows) > 1:
            logger.error(
                "ESPN %s id %s is on %d matches rows — the fixture is fragmented; resolving the "
                "most recent (%s). Repair the duplicates: predictions/odds/recs are split across them.",
                cfg.sport,
                espn_id,
                len(rows),
                rows[0]["id"],
            )
        if rows:
            return rows[0]["id"], "espn"

    cur.execute(
        "SELECT id FROM matches WHERE home_team_id = %s AND away_team_id = %s AND match_date = %s",
        (home_id, away_id, match_dt),
    )
    row = cur.fetchone()
    if row:
        return row["id"], "date"

    if not allow_heal:
        return None, None

    cur.execute(
        """
        SELECT id, match_date
          FROM matches
         WHERE home_team_id = %s
           AND away_team_id = %s
           AND status = 'scheduled'
           AND external_ids->>'espn' IS NULL
           AND match_date BETWEEN %s AND %s
         ORDER BY match_date
        """,
        (
            home_id,
            away_id,
            match_dt - timedelta(hours=window_hours),
            match_dt + timedelta(hours=window_hours),
        ),
    )
    candidates = cur.fetchall() or []
    if len(candidates) == 1:
        return candidates[0]["id"], "healed"
    if len(candidates) > 1:
        logger.warning(
            "Ambiguous fixture healing for %s vs %s at %s: %d unidentified scheduled rows within %dh (%s) "
            "— resolving nothing rather than guessing (same-pair rematch?)",
            home_id,
            away_id,
            match_dt,
            len(candidates),
            window_hours,
            ", ".join(f"{c['id']}@{c['match_date']}" for c in candidates),
        )
    return None, None


def resolve_match_id(
    cur,
    cfg: SportConfig,
    espn_id,
    home_id,
    away_id,
    match_dt,
    window_hours: int = IDENTITY_WINDOW_HOURS,
    allow_heal: bool = True,
):
    """Public wrapper: the resolved match id, or None. See _resolve_match_id."""
    return _resolve_match_id(cur, cfg, espn_id, home_id, away_id, match_dt, window_hours, allow_heal)[0]


def _realign_match_date(cur, match_id, home_id, away_id, match_dt, healed: bool = False) -> bool:
    """Point a resolved row at ESPN's CURRENT kickoff time.

    This is the de-fragmentation step: without it a moved kickoff makes the
    upsert's ON CONFLICT (home, away, match_date) target miss and insert a
    twin.

    Returns True when the row SITS AT `match_dt` afterwards — it was
    already there, or it moved — and False when it does not (pair drift, a
    refused move, a vanished row). The caller needs exactly that
    distinction: if the resolved row could not be put at (home, away,
    match_dt), then that slot belongs to a DIFFERENT row and stamping the
    ESPN id there would put one id on two rows.

    Both the guard and the UPDATE are keyed on the RESOLVED ROW's own team
    pair, never the incoming one. Step 1 resolves by ESPN id alone, so the
    row it returns can carry a different pair than the current parse: MMA
    competitors carry no homeAway at all (verified live — every competitor
    on UFC comps 401913130 / 401902948 / 401911287 returns None), so
    _competition_parts orders them by array POSITION and ESPN reorders that
    array freely; ensure_team's fuzzy branch can likewise re-resolve a
    vendor name onto another team. Moving such a row would silently rewrite
    a fixture we did not parse, and a guard checking a different key than
    the UPDATE writes cannot prevent the unique violation that would abort
    the whole league's ingest.
    """
    cur.execute(
        "SELECT home_team_id, away_team_id, match_date FROM matches WHERE id = %s",
        (match_id,),
    )
    row = cur.fetchone()
    if row is None:
        logger.error("Resolved match %s disappeared before realignment — not touching it", match_id)
        return False
    if row["home_team_id"] != home_id or row["away_team_id"] != away_id:
        logger.error(
            "Resolved match %s is %s vs %s but this ESPN competition parses as %s vs %s — refusing "
            "to realign or stamp it (team pair drifted; check competitor ordering / team matching)",
            match_id,
            row["home_team_id"],
            row["away_team_id"],
            home_id,
            away_id,
        )
        return False
    if row["match_date"] == match_dt:
        return True
    cur.execute(
        """
        UPDATE matches
           SET match_date = %s,
               updated_at = NOW()
         WHERE id = %s
           AND NOT EXISTS (
                 SELECT 1 FROM matches dup
                  WHERE dup.home_team_id = %s
                    AND dup.away_team_id = %s
                    AND dup.match_date = %s
                    AND dup.id <> %s)
        """,
        (match_dt, match_id, row["home_team_id"], row["away_team_id"], match_dt, match_id),
    )
    if cur.rowcount == 0:
        logger.warning(
            "Refusing to realign match %s from %s to %s — another row already holds that slot; "
            "the fixture stays fragmented until the duplicate is repaired",
            match_id,
            row["match_date"],
            match_dt,
        )
        return False
    shift_h = abs((match_dt - row["match_date"]).total_seconds()) / 3600.0
    if healed and shift_h > LARGE_SHIFT_HOURS:
        # Loud on purpose: a heal is an inference, and this one moved the
        # row further than the p75 shift measured on prod (24 h).
        logger.error(
            "Healed match %s moved %.1f h (%s -> %s), past the %dh p75 shift measured on prod — "
            "verify this is the same fixture and not two games for the same pair",
            match_id,
            shift_h,
            row["match_date"],
            match_dt,
            LARGE_SHIFT_HOURS,
        )
    else:
        logger.info("Realigned match %s: %s -> %s (ESPN moved the kickoff)", match_id, row["match_date"], match_dt)
    return True


def _stamp_identity(cur, home_id, away_id, match_dt, espn_id, season_type) -> bool:
    """Merge the ESPN competition id and the season-type marker onto the
    row now living at (home, away, match_dt).

    Both columns are MERGED (`||`), never replaced, so nothing else already
    stored in external_ids / metadata (result_detail, result_source,
    regulation_winner, the legacy NHL game_type marker) is dropped. A no-op
    when ESPN gave us neither value.
    """
    external = {"espn": str(espn_id)} if espn_id else {}
    meta = {"season_type": season_type} if season_type else {}
    if not external and not meta:
        return False
    cur.execute(
        """
        UPDATE matches
           SET external_ids = COALESCE(external_ids, '{}'::jsonb) || %s::jsonb,
               metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb,
               updated_at = NOW()
         WHERE home_team_id = %s AND away_team_id = %s AND match_date = %s
        """,
        (json.dumps(external), json.dumps(meta), home_id, away_id, match_dt),
    )
    return cur.rowcount > 0


def _reconcile_match_row(cur, cfg: SportConfig, espn_id, home_id, away_id, match_dt, allow_heal: bool = True) -> bool:
    """Pre-insert half of the identity fix, shared by both insert paths:
    find the row this competition already owns and drag its match_date to
    ESPN's current time so the upsert that follows lands ON that row
    instead of forking a twin.

    Returns True when the upsert that follows will land on the row this
    competition owns — i.e. when it is safe to stamp the ESPN id onto
    (home, away, match_dt). False means a row was resolved but could not be
    put at that slot, so the slot belongs to someone else."""
    match_id, source = _resolve_match_id(cur, cfg, espn_id, home_id, away_id, match_dt, allow_heal=allow_heal)
    if match_id is None:
        # Nothing to reconcile: step 2 already proved the slot is empty, so
        # the upsert inserts a fresh row there and the stamp is safe.
        return True
    return _realign_match_date(cur, match_id, home_id, away_id, match_dt, healed=(source == "healed"))


def _iter_event_competitions(event: dict, cfg: SportConfig):
    """Yield (competition, event_context) for each match in an ESPN event.

    For team sports, an ESPN event IS a match — its `competitions[0]`
    holds the competitors. We yield that single competition.

    For tennis (and other individual-sport tournaments), the ESPN
    event is a TOURNAMENT and matches are nested two layers deep:
        event.groupings[].competitions[]
    We walk that structure and yield each leaf competition. Each one
    is treated as a standalone match downstream.

    The event context travels alongside because `season.type` (preseason /
    regular / postseason) lives on the EVENT, not the competition, and
    dropping it is what let 147 NFL and 147 NBA preseason games into the
    training corpus untagged.
    """
    context = _event_context(event)
    if cfg.is_individual and event.get("groupings"):
        for grouping in event["groupings"]:
            for competition in grouping.get("competitions") or []:
                yield competition, context
    else:
        for competition in event.get("competitions") or []:
            yield competition, context


def _competition_parts(comp: dict, cfg: SportConfig):
    """Shared resolution for a competition object: (home, away competitor
    dicts, names, match datetime, state, venue, espn competition id) — or
    None when the shape is unusable. Used by both the fixtures path and the
    results path so team/date identity can never drift between them.

    The trailing espn id is the COMPETITION id, not the event id: a
    competition is the unit of a match (a tennis event is a whole
    tournament, an MMA event a whole fight card)."""
    competitors = comp.get("competitors") or []

    if cfg.is_individual:
        # Tennis competitors carry homeAway (set by ESPN seed ordering).
        # Fall back to positional ordering if missing — arbitrary but
        # consistent labeling of player1 / player2.
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if home is None or away is None:
            if len(competitors) < 2:
                return None
            home, away = competitors[0], competitors[1]
    else:
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not (home and away):
            return None

    home_name = _competitor_name(home, cfg.is_individual)
    away_name = _competitor_name(away, cfg.is_individual)
    if not (home_name and away_name):
        return None

    raw_date = comp.get("date") or comp.get("startDate")
    if not raw_date:
        return None
    try:
        match_dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None

    state = (comp.get("status", {}).get("type", {}).get("state") or "pre").lower()
    venue = (comp.get("venue") or {}).get("fullName")
    espn_id = comp.get("id")
    espn_id = str(espn_id) if espn_id is not None else None
    return home, away, home_name, away_name, match_dt, state, venue, espn_id


def _process_competition(cur, cfg: SportConfig, league_id: str, comp: dict, event_context: dict | None = None) -> bool:
    """Insert one scheduled match from a single competition object.

    `event_context` carries the season_type that lives on the parent ESPN
    event; it defaults to None so a caller with only a competition in hand
    still works (the marker is simply not written, i.e. "unknown")."""
    parts = _competition_parts(comp, cfg)
    if parts is None:
        return False
    home, away, home_name, away_name, match_dt, state, venue, espn_id = parts

    if state != "pre":
        return False  # results path (--results) handles post events

    home_id = ensure_team(cur, cfg, home_name, league_id)
    away_id = ensure_team(cur, cfg, away_name, league_id)
    if not (home_id and away_id):
        return False

    season_type = (event_context or {}).get("season_type")
    identity_safe = _reconcile_match_row(
        cur,
        cfg,
        espn_id,
        home_id,
        away_id,
        match_dt,
        allow_heal=healing_allowed((event_context or {}).get("pair_index"), home_name, away_name, match_dt),
    )
    written = bool(insert_scheduled_match(cur, cfg, league_id, home_id, away_id, match_dt, venue))
    _stamp_identity(cur, home_id, away_id, match_dt, espn_id if identity_safe else None, season_type)
    if espn_id and not identity_safe:
        logger.error(
            "Not stamping ESPN id %s onto %s vs %s at %s: the row that owns this fixture could not "
            "be realigned onto that slot, so stamping would put one ESPN id on two rows",
            espn_id,
            home_name,
            away_name,
            match_dt,
        )
    return written


def _final_scores(cfg: SportConfig, home: dict, away: dict) -> tuple[int, int] | None:
    """Final (home_score, away_score) per the corpus conventions:
    team sports use the real score; tennis/MMA use winner-flag 1/0
    (matches how the historical loaders populated the corpus)."""
    if cfg.is_individual:
        hw = home.get("winner")
        aw = away.get("winner")
        if hw is True:
            return 1, 0
        if aw is True:
            return 0, 1
        return None  # walkover / unknown — don't guess
    try:
        return int(float(home.get("score"))), int(float(away.get("score")))
    except (TypeError, ValueError):
        return None


_EXTRA_TIME_TOKENS = ("/OT", "/SO", "/2OT", "OVERTIME", "SHOOTOUT", "AET", "PEN")


def _record_result(cur, cfg: SportConfig, league_id: str, comp: dict, event_context: dict | None = None) -> bool:
    """Record one completed competition (state == 'post'). Grading needs:
    scores + status='finished' (all sports), and for NHL additionally
    metadata.regulation_winner ('draw' when the game went past regulation
    — grading_outcomes.nhl_regulation_outcome reads it). ESPN's status
    detail ('Final/OT', 'Final/SO') tells us that.

    Known limitation (documented, accepted): soccer cup ties decided in
    extra time / on penalties store the post-ET score, so a 1x2 grade on
    such a match reflects the ET result rather than the 90-minute result.
    League play (the overwhelming bulk) is unaffected; result_detail is
    stored in metadata for a later refinement."""
    parts = _competition_parts(comp, cfg)
    if parts is None:
        return False
    home, away, home_name, away_name, match_dt, state, venue, espn_id = parts

    status_type = comp.get("status", {}).get("type", {})
    if state != "post" or not status_type.get("completed", False):
        return False

    scores = _final_scores(cfg, home, away)
    if scores is None:
        return False
    home_score, away_score = scores

    detail = (status_type.get("detail") or status_type.get("shortDetail") or "").upper()
    went_extra = any(tok in detail for tok in _EXTRA_TIME_TOKENS)

    result_meta: dict = {"result_detail": detail or "FINAL", "result_source": "espn_scoreboard"}
    if cfg.sport == "nhl":
        result_meta["regulation_winner"] = "draw" if went_extra else ("home" if home_score > away_score else "away")

    home_id = ensure_team(cur, cfg, home_name, league_id)
    away_id = ensure_team(cur, cfg, away_name, league_id)
    if not (home_id and away_id):
        return False

    # The RESULTS path fragments fixtures too: 20 of the 119 prod twins were
    # inserted here, after kickoff, because ESPN had moved the time since we
    # ingested the fixture. Reconcile first so this writes to the row that
    # already carries the predictions/odds/recs.
    season_type = (event_context or {}).get("season_type")
    identity_safe = _reconcile_match_row(
        cur,
        cfg,
        espn_id,
        home_id,
        away_id,
        match_dt,
        allow_heal=healing_allowed((event_context or {}).get("pair_index"), home_name, away_name, match_dt),
    )
    written = bool(
        insert_finished_match(
            cur, cfg, league_id, home_id, away_id, match_dt, venue, home_score, away_score, result_meta
        )
    )
    _stamp_identity(cur, home_id, away_id, match_dt, espn_id if identity_safe else None, season_type)
    if espn_id and not identity_safe:
        logger.error(
            "Not stamping ESPN id %s onto %s vs %s at %s: the row that owns this fixture could not "
            "be realigned onto that slot, so stamping would put one ESPN id on two rows",
            espn_id,
            home_name,
            away_name,
            match_dt,
        )
    return written


def process_event(cur, cfg: SportConfig, league_id: str, event: dict, handler=None, pair_index=None) -> int:
    """Process every match in an ESPN event with `handler` (default: the
    scheduled-fixture path; the results path passes _record_result).
    Returns the count of matches handled. For team sports this is 0 or 1
    (one match per event); for tennis tournaments it can be dozens (one
    event = full bracket of matches in progress on the same day)."""
    handler = handler or _process_competition
    inserted = 0
    for comp, context in _iter_event_competitions(event, cfg):
        if pair_index is not None:
            context = {**context, "pair_index": pair_index}
        if handler(cur, cfg, league_id, comp, context):
            inserted += 1
    return inserted


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
                # Buffer the whole sweep BEFORE writing anything: step-3
                # healing has to know when ESPN published two competitions
                # for the same pair inside the window (see build_pair_index),
                # and both are always in the same sweep.
                events = []
                for offset in range(days):
                    day = today + timedelta(days=offset)
                    events.extend(fetch_day(cfg, slug, day))
                pair_index = build_pair_index(cfg, events)
                n = 0
                for ev in events:
                    # process_event returns the count of matches inserted
                    # (>=1 for tennis tournaments that contain multiple
                    # matches per ESPN event).
                    n += process_event(cur, cfg, league_id, ev, pair_index=pair_index)
                counts[slug] = n
                conn.commit()
                logger.info("Fetched %d upcoming %s fixtures for %s (%s)", n, cfg.sport, name, slug)
    return counts


def fetch_results_all(database_url: str, cfg: SportConfig, leagues: list[str], days_back: int) -> dict[str, int]:
    """Walk recent days BACKWARD and record final results for completed
    events (audit doc §2.1). Same endpoints, leagues, and identity as the
    fixtures path — only the competition handler differs. Downstream,
    grade_completed_matches (15-min cadence) settles predictions + recs
    for anything that flips to finished here."""
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
                # Same sweep buffering as the fixtures path — the results
                # path fragments fixtures too (20 of the 119 prod twins).
                events = []
                for offset in range(days_back + 1):  # today back through days_back
                    day = today - timedelta(days=offset)
                    events.extend(fetch_day(cfg, slug, day))
                pair_index = build_pair_index(cfg, events)
                n = 0
                for ev in events:
                    n += process_event(cur, cfg, league_id, ev, handler=_record_result, pair_index=pair_index)
                counts[slug] = n
                conn.commit()
                if n:
                    logger.info("Recorded %d %s result(s) for %s (%s)", n, cfg.sport, name, slug)
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
    p.add_argument(
        "--results",
        action="store_true",
        help="Results mode: walk recent days backward and record final "
        "scores for completed events (flips matches to 'finished' so the "
        "graders settle predictions + recommendations).",
    )
    p.add_argument(
        "--days-back",
        type=int,
        default=3,
        help="Results mode: how many days back to sweep (default 3 — "
        "catches late finals + ESPN corrections without rescanning weeks).",
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
    if args.results:
        counts = fetch_results_all(args.database_url, cfg, leagues, args.days_back)
        total = sum(counts.values())
        logger.info("Recorded %d %s result(s) across %d leagues", total, cfg.sport, len(counts))
        return 0
    counts = fetch_all(args.database_url, cfg, leagues, args.days)
    total = sum(counts.values())
    logger.info("Fetched %d upcoming %s fixtures across %d leagues", total, cfg.sport, len(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
