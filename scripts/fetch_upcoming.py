"""Fetch upcoming fixtures from ESPN's free public API.

Populates the `matches` table with future (status='scheduled') fixtures
for the configured leagues of the chosen sport. Matched to existing
teams by ESPN team id, then name (see find_team); new teams are inserted
on the fly.

ESPN's scoreboard endpoint returns the current scoring window (a few
days) plus you can pass ?dates=YYYYMMDD to walk forward up to ~2 weeks.

Usage:
    python scripts/fetch_upcoming.py                               # soccer, 14 days
    python scripts/fetch_upcoming.py --sport nhl                   # NHL, 14 days
    python scripts/fetch_upcoming.py --sport soccer --leagues eng.1,ger.1
    python scripts/fetch_upcoming.py --sport nhl --days 7

Exit status: 0 ok; 1 every ESPN request failed (results mode, an outage);
2 no database URL; 3 the sweep refused at least one competition because
writing it would corrupt a fixture (its identity, its score, or its
status), inserted a club as its own row because another ESPN team holds its
name, or met an ESPN team id conflict (see IDENTITY_STATS). Everything else
in the sweep is committed before the script exits 3.

ESPN's STATUS_POSTPONED / STATUS_CANCELED / STATUS_ABANDONED are recorded on
the EXISTING row ('postponed' / 'cancelled'), never inserted, from either
mode (see _record_status_change); grade_completed_matches voids the recs of a
match postponed more than 72 h, or cancelled with a kickoff 72 h old. A cancel
ESPN itself reported stays correctable (espn_cancelled); one recorded on
purpose (an operator, a duplicate, a re-drawn slot) is final. ESPN reports no
status at all for a bout it drops from a fight card, so results mode cancels
the stale rows a finished card no longer lists (cancel_dropped_bouts, MMA).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable, NamedTuple

import psycopg2
import requests
from psycopg2.extras import Json, RealDictCursor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soccer_seasons  # noqa: E402
import team_identity  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_upcoming")

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"


def _season_soccer(dt: datetime, league_slug: str | None = None) -> str:
    """The season label of a soccer fixture of ESPN league `league_slug`, by
    the scheme every soccer loader shares (soccer_seasons.season_for_kickoff,
    keyed by the league's football-data code): '2026' for a calendar-year
    league, Jul-Dec -> YYYY-YYYY+1 / Jan-Jun -> YYYY-1-YYYY otherwise. Without
    a slug (a caller that does not know the league) the cross-year rule
    applies. A kickoff in an era the scheme cannot label by date (none in
    ESPN's live window) falls back to the cross-year rule, loudly."""
    code = SOCCER_LEAGUES[league_slug][0] if league_slug in SOCCER_LEAGUES else None
    label = soccer_seasons.season_for_kickoff(code, dt)
    if label is None:
        label = soccer_seasons.cross_year_label(dt)
        logger.warning(
            "No date rule labels a %s (%s) fixture at %s (an irregular season, see soccer_seasons); labelled %s",
            code,
            league_slug,
            dt,
            label,
        )
    return label


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
    # Club name tokens ("FC", "AFC", "BK"). Soccer has many; NHL teams are
    # unambiguous and need none. find_team runs its fuzzy step only for a
    # sport that has them.
    club_suffixes: frozenset[str] = frozenset()
    club_prefixes: frozenset[str] = frozenset()
    # True for 1v1 sports (tennis, MMA, boxing) where ESPN's competitors
    # don't carry homeAway and use 'athlete' instead of 'team'. Triggers
    # the positional fallback in process_event and the per-player team
    # rows (one "team" per player in the teams table — pragmatic reuse
    # of the team-sports schema for individual sports).
    is_individual: bool = False
    # True when an ESPN event is a fight CARD: every bout of it is listed in
    # one event and a finished card lists every bout that took place, so a
    # bout ESPN removed from it was not fought on it (cancel_dropped_bouts).
    # Not tennis: a tournament event spans weeks and a day's scoreboard
    # carries only some of its matches.
    card_events: bool = False


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
        card_events=True,
    ),
}


def normalize_team(name: str) -> str:
    """Loose normalisation for the teams.normalized_name unique key, in
    Unicode NFC (composed) form: ESPN sends some names decomposed ('Rose
    Conceição' as 'c' + U+0327, 'a' + U+0303), and an exact match against a
    composed spelling of the same name would miss it and mint a twin."""
    return unicodedata.normalize("NFC", " ".join(name.strip().lower().split()))


def _strip_club_tokens(norm: str, cfg: SportConfig) -> str:
    """Aggressive normalisation: drop common club abbreviation tokens so
    'rosenborg bk' and 'rosenborg' compare equal. Not the persisted
    normalized_name, and no longer on the team-resolution path (soccer's
    fuzzy step is team_identity.match_club_name; club_suffixes/club_prefixes
    still mark the sports that run it). No-op for sports without club-token
    conventions (NHL)."""
    if not cfg.club_suffixes and not cfg.club_prefixes:
        return norm
    tokens = norm.split()
    while tokens and tokens[-1] in cfg.club_suffixes:
        tokens.pop()
    while tokens and tokens[0] in cfg.club_prefixes:
        tokens.pop(0)
    return " ".join(tokens) if tokens else norm


# ESPN's scoreboard API intermittently answers 200 with an empty or non-JSON
# body. From 2026-10-27 the NBA endpoint did so on about a third of pipeline
# runs; r.json() sat outside the error handling, so one bad body crashed the
# whole fetch_results task, skipped every sport chained after NBA, and blocked
# grading and NHL enrichment behind it (63 failed DAG runs in two days).
# Every request now gets FETCH_ATTEMPTS tries with backoff; a day that still
# fails is logged at ERROR and counted, never raised. FETCH_STATS lets main()
# tell a blip (some days failed) from an outage (every request failed).
FETCH_ATTEMPTS = 3
FETCH_BACKOFF_SECONDS = (2.0, 6.0)
FETCH_STATS = {"ok": 0, "failed": 0}


def fetch_day(cfg: SportConfig, league_slug: str, day: date) -> list[dict]:
    url = f"{ESPN_BASE}/{cfg.espn_path}/{league_slug}/scoreboard"
    params = {"dates": day.strftime("%Y%m%d")}
    last_error = ""
    for attempt in range(FETCH_ATTEMPTS):
        if attempt:
            time.sleep(FETCH_BACKOFF_SECONDS[min(attempt - 1, len(FETCH_BACKOFF_SECONDS) - 1)])
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            last_error = str(e)
            continue
        try:
            payload = r.json()
        except ValueError:  # requests' JSONDecodeError subclasses ValueError
            last_error = (
                f"non-JSON body (HTTP {r.status_code}, {r.headers.get('content-type')}, "
                f"{len(r.text)} bytes: {r.text[:80]!r})"
            )
            continue
        FETCH_STATS["ok"] += 1
        return (payload or {}).get("events", []) or []
    FETCH_STATS["failed"] += 1
    logger.error(
        "ESPN fetch FAILED after %d attempts for %s/%s %s: %s", FETCH_ATTEMPTS, cfg.sport, league_slug, day, last_error
    )
    return []


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


# Inserted with an empty external_ids: the ESPN id goes on afterwards through
# team_identity.stamp_espn_id, whose guard never puts one id on two rows. The
# conflict branch is the race where another sweep inserted the same name
# between our lookup and this insert.
INSERT_TEAM_SQL = """
    INSERT INTO teams (name, normalized_name, league_id, sport)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (normalized_name, sport) DO UPDATE
        SET league_id = COALESCE(teams.league_id, EXCLUDED.league_id)
    RETURNING id, (xmax = 0) AS inserted
"""


def ensure_team(
    cur,
    cfg: SportConfig,
    name: str,
    league_id: str,
    espn_id: str | None = None,
    alt_names: tuple[str, ...] = (),
) -> str | None:
    """Resolve `name` (and, for a team sport, ESPN team id `espn_id`) to a
    team_id via find_team's tiers, inserting a new team row only when nothing
    matches.

    The ESPN id is stamped onto a row found by name and onto an inserted row
    (guarded: team_identity.stamp_espn_id), so the name decision is made once
    and every later sweep resolves by id. A new soccer team is logged as a
    canary: in a league with established teams it is usually a split of one.

    A club whose name belongs to ANOTHER ESPN team (Santos Laguna arriving as
    'Santos' after Santos FC stamped the 'Santos' row) is never merged and
    never dropped: when no other tier resolves it, it gets its own row under
    a distinct name (namesake_name), stamped with its id, and the run counts
    IDENTITY_STATS['team_namesakes'] so it exits 3 once; later sweeps resolve
    it by id. None only on an ESPN-id conflict (the id on two rows, the
    seeded row holding another id, a stamp refused on a new row), counted in
    IDENTITY_STATS['team_conflicts'] — the competition is then skipped."""
    espn_id = str(espn_id) if espn_id else None
    label = f"{cfg.sport} {name!r}"
    found = _find_team(cur, cfg, name, league_id, espn_id)
    if found.team_id:
        if espn_id and found.how != "espn" and not team_identity.stamp_espn_id(cur, found.team_id, espn_id, label):
            # Only a race reaches this (each tier skips rows of other ESPN
            # teams, and a holder of this id resolves first): the name
            # decision stands, the conflict is logged above and counted.
            IDENTITY_STATS["team_conflicts"] += 1
        return found.team_id
    if found.how == "refused":
        IDENTITY_STATS["team_conflicts"] += 1
        return None

    insert_name = name
    if found.how == "seed-new" and found.insert_name:
        insert_name = found.insert_name
    elif found.how == "namesake":
        insert_name = namesake_name(cur, cfg, name, espn_id, alt_names)
    insert_name = unicodedata.normalize("NFC", insert_name)  # stored composed, like normalized_name
    cur.execute(INSERT_TEAM_SQL, (insert_name, normalize_team(insert_name), league_id, cfg.sport))
    row = cur.fetchone()
    if not row:
        IDENTITY_STATS["team_conflicts"] += 1
        logger.error("Inserting %s team %r returned no row — the competition is skipped", cfg.sport, insert_name)
        return None
    if espn_id and not team_identity.stamp_espn_id(cur, row["id"], espn_id, label):
        IDENTITY_STATS["team_conflicts"] += 1
        logger.error(
            "Not using %s team %s (%r) for ESPN team %s: the guarded stamp refused it (see above), so this side is "
            "unresolved and the competition is skipped; this run exits %d",
            cfg.sport,
            row["id"],
            insert_name,
            espn_id,
            IDENTITY_EXIT_CODE,
        )
        return None
    if found.how == "namesake":
        IDENTITY_STATS["team_namesakes"] += 1
        logger.error(
            "%s ESPN team %s %r: that name belongs to %s — a different club under the same name. Inserted it as "
            "its own team %s (%r) stamped with ESPN id %s; later sweeps resolve it by that id. This run exits %d. "
            "If the two ARE one club, one of the stamped ESPN ids is wrong: pin the right row in "
            "scripts/data/soccer_team_espn_seed.csv and move this team's matches (scripts/repoint_team_matches.py).",
            cfg.sport,
            espn_id,
            name,
            "; ".join(f"team {tid} ({tier} match, ESPN team {held})" for tier, tid, held in found.blocked),
            row["id"],
            insert_name,
            espn_id,
            IDENTITY_EXIT_CODE,
        )
    if cfg.sport == "soccer" and row.get("inserted", True):
        source = f"ESPN sweep, ESPN team id {espn_id}" if espn_id else "ESPN sweep"
        if found.how == "seed-new":
            source += ", pinned by scripts/data/soccer_team_espn_seed.csv"
        team_identity.log_new_soccer_team(cur, insert_name, league_id, source, seeded=found.how == "seed-new")
    return row["id"]


def namesake_name(cur, cfg: SportConfig, name: str, espn_id: str | None, alt_names: tuple[str, ...] = ()) -> str:
    """A name for the row of a club whose display name another ESPN team
    holds, free under UNIQUE(normalized_name, sport) and in every name tier.

    ESPN's own full names come first (competitor team.name / team.location,
    e.g. 'Athletico Paranaense' for 'Athletico-PR'). Its abbreviation and
    shortDisplayName are never used: the collisions this exists for share
    them ('SAN' for both Santos clubs, 'LIV' for both Liverpools), and short
    forms like 'Estudiantes' name several clubs. Otherwise the deterministic
    '<name> (ESPN <id>)'."""
    for alt in alt_names:
        alt = " ".join((alt or "").split())
        if alt and normalize_team(alt) != normalize_team(name) and _name_is_free(cur, cfg, alt):
            return alt
    return f"{name} (ESPN {espn_id})"


def _name_is_free(cur, cfg: SportConfig, name: str) -> bool:
    """True when no exact, alias or accent-folded name tier resolves `name`."""
    norm = normalize_team(name)
    cur.execute("SELECT id FROM teams WHERE normalized_name = %s AND sport = %s", (norm, cfg.sport))
    if cur.fetchone():
        return False
    if team_identity.find_team_by_alias(cur, cfg.sport, norm):
        return False
    return not team_identity.find_team_accent_insensitive(cur, cfg.sport, norm)


def find_team(cur, cfg: SportConfig, name: str, league_id: str | None, espn_id: str | None = None) -> str | None:
    """Resolve `name` to an EXISTING team_id, or None. Never writes —
    backfills that must not mint teams use this instead of ensure_team.

    Vendor names differ from the history's (ESPN "Tottenham Hotspur",
    football-data "Tottenham"), and a miss mints a twin with no history, so
    the lookup runs in tiers (the shared rules live in team_identity):

      1. The ESPN team id, when the caller has one (teams.external_ids.espn).
         The id on two rows is corruption: refused, never resolved by name.
      1b. Soccer: the seed (team_identity.seed_entry) — ESPN ids whose names
         collide, pinned to their rows ('Santos' FC vs Santos Laguna).
      2. Exact normalised name on (normalized_name, sport).
      3. An alias (teams.metadata.aliases).
      4. The same name with accents folded (ESPN "Montreal" must reuse the
         NHL API's "Montréal" row).
      5. Soccer only: a club-name match among the teams of the swept league's
         country (team_identity.match_club_name) — never in an international
         competition, where clubs of every country meet.
      6. Soccer only: the in-league spelling ratio, the pre-2026-10 step kept
         last (team_identity.match_league_ratio): 'Espanyol' -> 'Espanol'.
      7. (ensure_team only) INSERT a new row.

    With an ESPN id, tiers 2-6 skip every row that carries a DIFFERENT ESPN
    id: that row is another club under the same (or a similar) name. If one
    was skipped and nothing else resolves, ensure_team inserts the club under
    a distinct name (see ensure_team); find_team returns None.
    """
    return _find_team(cur, cfg, name, league_id, str(espn_id) if espn_id else None).team_id


class TeamLookup(NamedTuple):
    """_find_team's verdict. how: 'espn' | 'seed' | 'exact' | 'alias' |
    'accent' | 'fuzzy' | 'ratio' for a hit (team_id set); for a miss,
    '' (insert `name`), 'seed-new' (insert insert_name, the seeded name),
    'namesake' (a name tier hit rows of other ESPN teams, listed in blocked
    as (tier, team_id, their ESPN id): insert under a distinct name) or
    'refused' (an ESPN id conflict: write nothing)."""

    team_id: str | None
    how: str
    insert_name: str | None = None
    blocked: tuple[tuple[str, str, str], ...] = ()


def _find_team(cur, cfg: SportConfig, name: str, league_id: str | None, espn_id: str | None) -> TeamLookup:
    """find_team's tiers (see find_team), with the verdict ensure_team acts on."""
    norm = normalize_team(name)

    if espn_id:
        holders = team_identity.espn_id_holders(cur, cfg.sport, espn_id)
        seed = team_identity.seed_entry(cfg.sport, espn_id)
        if len(holders) > 1:
            logger.error(
                "Refusing %s team %r: ESPN team id %s is on %d teams (%s) — one ESPN team must be one row. The "
                "competition is skipped until the rows are repaired.",
                cfg.sport,
                name,
                espn_id,
                len(holders),
                ", ".join(holders),
            )
            return TeamLookup(None, "refused")
        if holders:
            if seed and _team_normalized_name(cur, holders[0]) != seed.normalized_name:
                logger.error(
                    "Refusing %s team %r: ESPN team id %s is stamped on team %s, but the seed pins it to %r (%s). "
                    "One of the two is wrong; the competition is skipped until they agree.",
                    cfg.sport,
                    name,
                    espn_id,
                    holders[0],
                    seed.team_name,
                    seed.evidence,
                )
                return TeamLookup(None, "refused")
            return TeamLookup(holders[0], "espn")
        if seed:
            return _seeded_team(cur, cfg, name, espn_id, seed)

    blocked: list[tuple[str, str, str]] = []

    def usable(team_id: str | None, tier: str) -> bool:
        """A name-tier hit counts unless it belongs to another ESPN team."""
        if not team_id:
            return False
        held = team_identity.espn_id_of(cur, team_id) if espn_id else None
        if held and held != espn_id:
            if all(str(team_id) != tid for _, tid, _ in blocked):
                blocked.append((tier, str(team_id), held))
            return False
        return True

    cur.execute(
        "SELECT id FROM teams WHERE normalized_name = %s AND sport = %s",
        (norm, cfg.sport),
    )
    row = cur.fetchone()
    if row and usable(row["id"], "exact"):
        return TeamLookup(row["id"], "exact")
    alias_id = team_identity.find_team_by_alias(cur, cfg.sport, norm)
    if usable(alias_id, "alias"):
        return TeamLookup(alias_id, "alias")
    accent_id = team_identity.find_team_accent_insensitive(cur, cfg.sport, norm)
    if usable(accent_id, "accent"):
        return TeamLookup(accent_id, "accent")

    if league_id and (cfg.club_suffixes or cfg.club_prefixes):
        # Both fuzzy tiers leave out rows stamped with another ESPN id.
        fuzzy_id = team_identity.find_team_fuzzy(cur, cfg.sport, name, league_id, espn_id)
        if fuzzy_id:
            return TeamLookup(fuzzy_id, "fuzzy")
        ratio_id = _find_team_by_league_ratio(cur, cfg, name, league_id, espn_id)
        if ratio_id:
            return TeamLookup(ratio_id, "ratio")
    if blocked:
        return TeamLookup(None, "namesake", blocked=tuple(blocked))
    return TeamLookup(None, "")


def _team_normalized_name(cur, team_id) -> str | None:
    cur.execute("SELECT normalized_name FROM teams WHERE id = %s", (team_id,))
    row = cur.fetchone()
    return row["normalized_name"] if row else None


def _seeded_team(cur, cfg: SportConfig, name: str, espn_id: str, seed) -> TeamLookup:
    """The seed's row for this ESPN id: found by its exact name, 'seed-new'
    when no such row exists yet (ensure_team creates it under the seeded
    name), refused when that row already carries another ESPN id."""
    cur.execute(
        "SELECT id FROM teams WHERE normalized_name = %s AND sport = %s",
        (seed.normalized_name, cfg.sport),
    )
    row = cur.fetchone()
    if not row:
        logger.info("%s ESPN team %s %r is seeded as new team %r", cfg.sport, espn_id, name, seed.team_name)
        return TeamLookup(None, "seed-new", insert_name=seed.team_name)
    held = team_identity.espn_id_of(cur, row["id"])
    if held and held != espn_id:
        logger.error(
            "Refusing %s team %r (ESPN team id %s): the seed pins it to team %s (%r), which carries ESPN id %s. "
            "One of the two is wrong; the competition is skipped until they agree.",
            cfg.sport,
            name,
            espn_id,
            row["id"],
            seed.team_name,
            held,
        )
        return TeamLookup(None, "refused")
    logger.info("%s ESPN team %s %r resolves to %r by the seed", cfg.sport, espn_id, name, seed.team_name)
    return TeamLookup(row["id"], "seed")


def _find_team_by_league_ratio(cur, cfg: SportConfig, name: str, league_id, espn_id: str | None) -> str | None:
    """The last tier: team_identity.match_league_ratio among the teams whose
    first league is `league_id` (the pre-2026-10 fuzzy step), skipping rows
    stamped with another ESPN id when the caller has one. A match is logged at
    WARNING, like the club-name tier: it is the decision to review if two
    clubs were merged."""
    cur.execute(
        "SELECT id, name, normalized_name, external_ids->>'espn' AS espn FROM teams "
        "WHERE league_id = %s AND sport = %s",
        (league_id, cfg.sport),
    )
    candidates = [
        (c, _strip_club_tokens(c["normalized_name"], cfg)) for c in cur.fetchall() if not (espn_id and c.get("espn"))
    ]
    verdict = team_identity.match_league_ratio(name, _strip_club_tokens(normalize_team(name), cfg), candidates)
    if verdict.outcome == "matched":
        logger.warning(
            "Resolved %s team %r onto existing %r by the in-league spelling ratio %.2f (runner-up %r %.2f) — if "
            "they are different clubs, split them and pin both in scripts/data/soccer_team_espn_seed.csv",
            cfg.sport,
            name,
            verdict.name,
            verdict.score,
            verdict.runner_up,
            verdict.runner_up_score,
        )
        return verdict.team_id
    if verdict.outcome == "ambiguous":
        logger.warning(
            "In-league spelling ratio for %s team %r is ambiguous: %r %.2f vs %r %.2f — not merging",
            cfg.sport,
            name,
            verdict.name,
            verdict.score,
            verdict.runner_up,
            verdict.runner_up_score,
        )
    return None


def season_label(cfg: SportConfig, match_dt: datetime, event_context: dict | None = None) -> str:
    """matches.season for a fixture: the sport's convention, and for soccer
    the league's (event_context['league_slug'], set by the sweeps)."""
    if cfg.sport == "soccer":
        return _season_soccer(match_dt, (event_context or {}).get("league_slug"))
    return cfg.season_func(match_dt)


def insert_scheduled_match(
    cur, cfg: SportConfig, league_id, home_id, away_id, match_dt, venue, season: str | None = None
) -> int:
    """`season` labels an INSERTED row (an existing row keeps its label);
    None falls back to cfg.season_func. On the existing row of the triple a
    'finished' status stays, and so does a 'cancelled' one recorded on purpose
    (an operator's, repair_fixture_orphans' orphan_of, a re-drawn slot's): the
    results path's rule (INSERT_FINISHED_MATCH_SQL), so the two paths never
    disagree about a cancel. Only ESPN's own cancel (ESPN_CANCELLED_SQL) and
    any other status go back to 'scheduled'."""
    cur.execute(
        INSERT_SCHEDULED_MATCH_SQL,
        (league_id, home_id, away_id, match_dt, venue, season if season is not None else cfg.season_func(match_dt)),
    )
    return cur.rowcount


# A 'cancelled' row ESPN's own status put there (_record_status_change stamps
# status_source='espn_scoreboard' and espn_status) stays open to correction:
# ESPN cancels and abandons matches it later resumes, awards or replays under
# the same id. A cancel recorded on purpose stays final — an operator's
# (fix_unplayed_soccer_rows: no status_source), a duplicate's
# (repair_fixture_orphans: metadata.orphan_of), a re-drawn tennis/MMA slot's
# (_release_redrawn_slot: metadata.redrawn_espn_id) — and so does a row ESPN
# only postponed before someone else cancelled it (espn_status
# STATUS_POSTPONED). A bout ESPN dropped from a finished fight card
# (cancel_dropped_bouts: espn_status DROPPED_FROM_CARD) is read off ESPN's
# scoreboard too, and stays correctable the same way: a bout a later sweep
# finds on the card after all gets its result.
DROPPED_FROM_CARD = "DROPPED_FROM_CARD"
ESPN_CANCEL_STATUSES = ("STATUS_CANCELED", "STATUS_CANCELLED", "STATUS_ABANDONED", DROPPED_FROM_CARD)
ESPN_CANCELLED_SQL = (
    "({t}.metadata->>'status_source' = 'espn_scoreboard'"
    " AND {t}.metadata->>'espn_status' IN (" + ", ".join(f"'{s}'" for s in ESPN_CANCEL_STATUSES) + ")"
    " AND {t}.metadata->>'orphan_of' IS NULL"
    " AND {t}.metadata->>'redrawn_espn_id' IS NULL)"
)


# The fixtures path's upsert (insert_scheduled_match). COALESCE: on a row
# without those metadata keys ESPN_CANCELLED_SQL is NULL, which must read
# "not ESPN's cancel" (keep it), never fall through to 'scheduled'.
INSERT_SCHEDULED_MATCH_SQL = (
    """
        INSERT INTO matches (league_id, home_team_id, away_team_id, match_date,
                             status, venue, season)
        VALUES (%s, %s, %s, %s, 'scheduled', %s, %s)
        ON CONFLICT (home_team_id, away_team_id, match_date) DO UPDATE
            SET status = CASE
                  WHEN matches.status = 'finished' THEN matches.status
                  WHEN matches.status = 'cancelled' AND NOT COALESCE("""
    + ESPN_CANCELLED_SQL.format(t="matches")
    + """, false) THEN matches.status
                  ELSE EXCLUDED.status
                END,
                venue = COALESCE(matches.venue, EXCLUDED.venue),
                updated_at = NOW()
    """
)


def espn_cancelled(status: str | None, metadata: dict | None) -> bool:
    """True for a 'cancelled' row whose cancel came from ESPN's scoreboard
    and nothing else has claimed since (ESPN_CANCELLED_SQL, in Python)."""
    meta = metadata or {}
    return (
        status == "cancelled"
        and meta.get("status_source") == "espn_scoreboard"
        and meta.get("espn_status") in ESPN_CANCEL_STATUSES
        and not meta.get("orphan_of")
        and not meta.get("redrawn_espn_id")
    )


# The DO UPDATE's WHERE leaves a final 'cancelled' row alone (see
# insert_finished_match).
INSERT_FINISHED_MATCH_SQL = (
    """
        INSERT INTO matches (league_id, home_team_id, away_team_id, match_date,
                             status, venue, season, home_score, away_score, metadata,
                             half_time_home, half_time_away)
        VALUES (%s, %s, %s, %s, 'finished', %s, %s, %s, %s, %s::jsonb, %s, %s)
        ON CONFLICT (home_team_id, away_team_id, match_date) DO UPDATE
            SET status = 'finished',
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score,
                metadata = COALESCE(matches.metadata, '{}'::jsonb) || EXCLUDED.metadata,
                half_time_home = CASE WHEN matches.half_time_home IS NULL
                                      THEN EXCLUDED.half_time_home ELSE matches.half_time_home END,
                half_time_away = CASE WHEN matches.half_time_home IS NULL
                                      THEN EXCLUDED.half_time_away ELSE matches.half_time_away END,
                venue = COALESCE(matches.venue, EXCLUDED.venue),
                updated_at = NOW()
            WHERE matches.status <> 'cancelled' OR """
    + ESPN_CANCELLED_SQL.format(t="matches")
    + """
        RETURNING id
    """
)


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
    halftime: tuple[int, int] | None = None,
    season: str | None = None,
) -> int:
    """Record a final result (audit doc §2.1 — the missing half of the
    feedback loop). Updates the row fetch_upcoming created — callers run
    _reconcile_match_row first so that row's match_date already matches
    ESPN's current time — or inserts it as finished when we never saw it
    as a fixture (grows the corpus). grade_completed_matches
    (already on the 15-min cadence) settles predictions + recs from here.
    Score columns follow the corpus conventions: real scores for team
    sports, winner-flag 1/0 for tennis/MMA. `halftime` (soccer) fills
    half_time_home/away only while that pair is empty: a football-data load
    may already hold it, and the pair is never split. `season` labels an
    inserted row, as in insert_scheduled_match.

    A 'cancelled' row is never touched (the same rule as promote_raw's
    upsert) when that status was set on purpose, by an operator, by
    repair_fixture_orphans or by _release_redrawn_slot: a deep results
    backfill must not flip it to 'finished'. A cancel taken from ESPN's own
    scoreboard (ESPN_CANCELLED_SQL) is not final: an abandoned match ESPN
    later reports as final (resumed, awarded) gets its result. Returns 1 when
    a row was written and 0 when the existing row is a final 'cancelled' (the
    DO UPDATE's WHERE is false, so no row comes back)."""
    ht_home, ht_away = halftime if halftime is not None else (None, None)
    cur.execute(
        INSERT_FINISHED_MATCH_SQL,
        (
            league_id,
            home_id,
            away_id,
            match_dt,
            venue,
            season if season is not None else cfg.season_func(match_dt),
            home_score,
            away_score,
            json.dumps(result_meta),
            ht_home,
            ht_away,
        ),
    )
    return 0 if cur.fetchone() is None else 1


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


def _competitor_espn_id(competitor: dict, is_individual: bool) -> str | None:
    """ESPN's id for a team-sport competitor's team (.team.id, else the
    competitor's own .id, which ESPN sets to the same value), as a string.
    None for an individual sport: athlete rows are not keyed by ESPN id."""
    if is_individual:
        return None
    raw = (competitor.get("team") or {}).get("id") or competitor.get("id")
    return str(raw) if raw not in (None, "") else None


def _competitor_alt_names(competitor: dict, is_individual: bool) -> tuple[str, ...]:
    """ESPN's other full names for a team-sport competitor (team.name, then
    team.location), for naming a namesake row (see namesake_name). Empty for
    an individual sport."""
    if is_individual:
        return ()
    team = competitor.get("team") or {}
    names: list[str] = []
    for key in ("name", "location"):
        value = " ".join(str(team.get(key) or "").split())
        if value and value not in names:
            names.append(value)
    return tuple(names)


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

# What a sweep did that needs a person, counted so main() exits
# IDENTITY_EXIT_CODE and the DAG's failure callback alerts someone:
#   forks_refused   competitions not written: the write would fork an
#                   ESPN-identified fixture (see ForkRefused);
#   same_team       competitions not written: both sides resolved to one team
#                   (see _resolve_teams);
#   score_mismatch  results not written: ESPN's header score is contradicted
#                   by its own scoring plays (see _record_result);
#   status_refused  postponements / cancels not written: the ESPN id names a
#                   row for another team pair (see _record_status_change);
#   team_namesakes  a club whose name another ESPN team holds was inserted as
#                   its own row (see ensure_team) — written, but a person must
#                   confirm it is not a mis-stamped id; once only, since later
#                   sweeps resolve it by its id;
#   team_conflicts  ESPN team id conflicts (the id on two rows, the seed and a
#                   stamp disagreeing, a guarded stamp refused); a side left
#                   unresolved skips its competition.
# The rest of the sweep is still written and committed. An ERROR line alone
# was not enough: from 2026-09-29 to 2026-10-10 the soccer fixtures path
# logged the same refusal on about 1,075 runs while it forked FC Juarez v
# Tijuana into two rows.
IDENTITY_STATS = {
    "forks_refused": 0,
    "same_team": 0,
    "score_mismatch": 0,
    "status_refused": 0,
    "team_namesakes": 0,
    "team_conflicts": 0,
}
IDENTITY_EXIT_CODE = 3


class ForkRefused(Exception):
    """The ESPN id names an existing row for THIS fixture under another team
    pair. That row cannot be realigned, and an upsert at the parsed (home,
    away, match_date) would insert a second row for the same fixture."""

    def __init__(self, match_id, owner: dict):
        super().__init__(f"match {match_id} owns this ESPN competition under another team pair")
        self.match_id = match_id
        self.owner = owner


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


_PLACEHOLDER_NAMES = {"tbd", "tba", "to be determined", "to be announced", "bye", "qualifier"}

# Knockout-bracket slots ESPN publishes before the teams are known. Each name
# is reused from tournament to tournament and fuzzy-matches its siblings
# ("Round of 32 1 Winner" vs "Round of 32 2 Winner" scores above 0.85), so the
# 2026 World Cup's slots collapsed into 8 team rows and 20 fixtures of a team
# against itself. Matched against the whole, whitespace-collapsed name, so a
# real club that merely contains one of these words is kept.
_PLACEHOLDER_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # "Group A 2nd Place", "Group C Winner", "Group B Runner-up"
        r"group [a-z]{1,2} (?:winners?|runners?[- ]?up|\d+(?:st|nd|rd|th)(?: place)?)",
        # "Third Place Group C/E/F/H/I", "3rd Place Group A/B/C/D/F"
        r"(?:third|3rd|fourth|4th)(?: place)? group [a-z](?: ?[/,] ?[a-z])*",
        # "Round of 32 1 Winner", "Quarterfinal 2 Winner", "Semifinal 1 Loser", "Match 73 Winner"
        r"(?:round of \d+|(?:quarter|semi)[- ]?finals?|play[- ]?offs?|match|game)(?: \d+)? (?:winners?|losers?)",
        # "Winner Match 73", "Loser of Semifinal 1", "Winner Group A"
        r"(?:winners?|losers?|runners?[- ]?up) (?:of )?(?:match|game|group|round|quarter|semi|final|play)\b.*",
        # "Final ...": the final's own slot, however the rest is spelled
        r"finals?\b.*",
    )
)


def _is_placeholder_name(name: str) -> bool:
    """True for ESPN's draw-slot placeholders, which are not real competitors.

    'Qualifier' is included: ESPN uses it for an unfilled qualifying slot, and
    like TBD it is reused across every such slot in the draw, so two unrelated
    fixtures would collide on the same (home, away) identity. The knockout
    slots in _PLACEHOLDER_PATTERNS are refused for the same reason.
    """
    norm = " ".join((name or "").split()).lower()
    return norm in _PLACEHOLDER_NAMES or any(p.fullmatch(norm) for p in _PLACEHOLDER_PATTERNS)


def _sample_times(times: list[datetime], limit: int = 6) -> str:
    """Compact rendering of a timestamp list for a log line."""
    ordered = sorted(t.isoformat() for t in times)
    if len(ordered) <= limit:
        return ", ".join(ordered)
    return "%s, ... (%d more)" % (", ".join(ordered[:limit]), len(ordered) - limit)


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
    # The refusal itself is the useful signal; the full list is not. A tennis
    # sweep can find 190+ competitions for one placeholder pair, and dumping
    # every timestamp buries real errors in the log (non-negotiable 3).
    logger.warning(
        "Refusing fixture healing for %s vs %s at %s: ESPN published %d competitions for this "
        "pair within %dh (%s) — healing cannot tell them apart, so this competition gets its "
        "own row rather than risking a merge of two different fixtures",
        home_name,
        away_name,
        match_dt,
        len(near),
        window_hours,
        _sample_times(near),
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

    Step 1 has one exception, for 1v1 sports: ESPN keeps a competition id
    when it re-fills a draw slot (a withdrawal, a lucky loser, a replacement
    opponent), so the id can point at a row for a pairing that will never be
    played (prod 2026-10-28: comp 185138 stayed on "Fritz v Tsitsipas" after
    Hijikata replaced Tsitsipas; every later time change of the real match
    then forked a new unidentified row, ~25-50 a week). When the id's row is
    still 'scheduled' (or cancelled by ESPN's own status, espn_cancelled)
    and shares not even the same two players, the slot was re-drawn: the
    stale row is released (id removed, status 'cancelled', the re-draw
    recorded in metadata) and resolution continues by pair. A
    swapped pair (same two players) is NOT a re-draw: both handlers first
    turn the parse into the row's order (_listed_reversed), and only a pair
    still reversed here (a fork already holds the parse's slot) meets the
    old refuse-and-log behaviour in _realign_match_date.

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
            SELECT m.id, m.home_team_id, m.away_team_id, m.status, m.metadata
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
        if rows and not (
            cfg.is_individual
            and _is_redrawn_slot(rows[0], home_id, away_id)
            and _release_redrawn_slot(cur, cfg, espn_id, rows[0])
        ):
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


def _is_redrawn_slot(row: dict, home_id, away_id) -> bool:
    """True when the row an ESPN id resolved to is for a DIFFERENT pairing
    than this parse — not merely the same two players with sides swapped."""
    if row.get("home_team_id") is None or row.get("away_team_id") is None:
        return False
    stored = {str(row["home_team_id"]), str(row["away_team_id"])}
    return stored != {str(home_id), str(away_id)}


def _release_redrawn_slot(cur, cfg: SportConfig, espn_id, row: dict) -> bool:
    """Detach an ESPN id from a scheduled row whose draw slot ESPN re-filled
    with another player, and cancel that pairing (it will never be played).
    A row ESPN's own status cancelled (espn_cancelled) is released the same
    way. Finished/live rows, and rows cancelled on purpose, are never
    touched. Returns True when released, so the caller resolves the new
    pairing by pair instead of by the stale row."""
    if row.get("status") != "scheduled" and not espn_cancelled(row.get("status"), row.get("metadata")):
        logger.error(
            "ESPN %s id %s resolves to %s row %s for a different pairing — not releasing a non-scheduled row; "
            "this competition stays unidentified",
            cfg.sport,
            espn_id,
            row.get("status"),
            row["id"],
        )
        return False
    cur.execute(
        """
        UPDATE matches
           SET external_ids = COALESCE(external_ids, '{}'::jsonb) - 'espn',
               status = 'cancelled',
               metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb,
               updated_at = NOW()
         WHERE id = %s
           AND (status = 'scheduled' OR (status = 'cancelled' AND """
        + ESPN_CANCELLED_SQL.format(t="matches")
        + """))
        """,
        (json.dumps({"redrawn_espn_id": str(espn_id)}), row["id"]),
    )
    if cur.rowcount == 0:
        return False
    logger.warning(
        "ESPN %s competition %s was re-drawn (different players): released it from row %s and cancelled that pairing",
        cfg.sport,
        espn_id,
        row["id"],
    )
    return True


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


# id::text: never a cast error inside the sweep's transaction, whatever the ids.
TEAM_NAMES_SQL = "SELECT id::text AS id, name FROM teams WHERE id::text = ANY(%s)"


def _team_names(cur, *team_ids) -> dict[str, str]:
    """{team id: name} of the team rows a refusal names, so the log line shows
    which club each id is (ESPN calls Santos FC and Santos Laguna both
    'Santos'). A row that is not found is left out (the line prints '?')."""
    ids = sorted({str(t) for t in team_ids if t})
    if not ids:
        return {}
    cur.execute(TEAM_NAMES_SQL, (ids,))
    return {str(r["id"]): r["name"] for r in (cur.fetchall() or [])}


def _match_teams(cur, match_id) -> dict | None:
    """A row's team pair, with names for the log line."""
    cur.execute(
        """
        SELECT m.home_team_id, m.away_team_id, th.name AS home_name, ta.name AS away_name
          FROM matches m
          JOIN teams th ON th.id = m.home_team_id
          JOIN teams ta ON ta.id = m.away_team_id
         WHERE m.id = %s
        """,
        (match_id,),
    )
    return cur.fetchone()


def _reconcile_match_row(cur, cfg: SportConfig, espn_id, home_id, away_id, match_dt, allow_heal: bool = True) -> bool:
    """Pre-insert half of the identity fix, shared by both insert paths:
    find the row this competition already owns and drag its match_date to
    ESPN's current time so the upsert that follows lands ON that row
    instead of forking a twin.

    Returns True when the upsert that follows will land on the row this
    competition owns — i.e. when it is safe to stamp the ESPN id onto
    (home, away, match_dt). False means a row was resolved but could not be
    put at that slot, so the slot belongs to someone else.

    Raises ForkRefused, for team sports, when the ESPN id resolved a row
    whose team pair is not this parse's. A team-sport ESPN competition id
    is the fixture's stable identity, so a different pair means OUR team
    resolution moved, not that the match changed. Prod example: ESPN's
    "FC Juárez" began resolving to an accented duplicate of the stored
    "FC Juarez" team, and every sweep from 2026-09-29 to 2026-10-10 upserted
    a second, unidentified row. That row took the result while the original,
    holding 57 predictions, stayed 'scheduled' for good. The caller writes
    nothing. 1v1 sports are exempt: there a different pair is a re-drawn
    slot (see _resolve_match_id), and the new pairing's own row is the right
    place for the write."""
    match_id, source = _resolve_match_id(cur, cfg, espn_id, home_id, away_id, match_dt, allow_heal=allow_heal)
    if match_id is None:
        # Nothing to reconcile: step 2 already proved the slot is empty, so
        # the upsert inserts a fresh row there and the stamp is safe.
        return True
    if _realign_match_date(cur, match_id, home_id, away_id, match_dt, healed=(source == "healed")):
        return True
    if source == "espn" and not cfg.is_individual:
        owner = _match_teams(cur, match_id)
        if owner is not None and _is_identity_drift(owner, home_id, away_id):
            raise ForkRefused(match_id, owner)
    return False


# The row an ESPN competition id names, when it holds the given ordered pair.
REVERSED_ROW_SQL = """
    SELECT m.id
      FROM matches m
      JOIN leagues l ON l.id = m.league_id
     WHERE m.external_ids->>'espn' = %s
       AND l.sport = %s
       AND m.home_team_id = %s
       AND m.away_team_id = %s
"""


def _listed_reversed(cur, cfg: SportConfig, espn_id, home_id, away_id, match_dt) -> bool:
    """True when, in a 1v1 sport, the row ESPN competition `espn_id` names
    holds these two players the other way round and no row sits at this
    parse's own (home, away, match_dt). The caller then writes in that row's
    order (both handlers swap the sides, the results path the scores too), so
    the write lands on the row that owns the fixture.

    ESPN orders MMA competitors freely (see _realign_match_date), and the
    realignment refuses a row whose pair is reversed, so the upsert at the
    parse's slot used to fork a reversed twin: UFC 2026-10-03, McGhee v
    Romero (ESPN 401927906, 1 prediction, 14 odds rows) stayed 'scheduled'
    for good while the fixtures path inserted Romero v McGhee and the result
    landed there. A row already at the parse's slot is that fork: nothing is
    swapped (two rows for one bout need a person) and it is logged."""
    if not (cfg.is_individual and espn_id):
        return False
    cur.execute(REVERSED_ROW_SQL, (str(espn_id), cfg.sport, away_id, home_id))
    rows = cur.fetchall() or []
    if len(rows) != 1:
        return False
    cur.execute(
        "SELECT id FROM matches WHERE home_team_id = %s AND away_team_id = %s AND match_date = %s",
        (home_id, away_id, match_dt),
    )
    fork = cur.fetchone()
    if fork:
        logger.error(
            "ESPN %s competition %s lists %s v %s, the reverse of its row %s, and row %s already holds that order at "
            "%s — one bout on two rows; not swapping. Repair the duplicate (its predictions/odds/recs are split).",
            cfg.sport,
            espn_id,
            home_id,
            away_id,
            rows[0]["id"],
            fork["id"],
            match_dt,
        )
        return False
    logger.info(
        "ESPN %s competition %s lists its players the other way round from row %s — writing in the row's order",
        cfg.sport,
        espn_id,
        rows[0]["id"],
    )
    return True


def _is_identity_drift(owner: dict, home_id, away_id) -> bool:
    """True when the row an ESPN id owns differs from the parse by DRIFT —
    the two pairs share a team, or are the same pair swapped — which is our
    team resolution moving under a fixed fixture (FC Juárez: Tijuana stayed,
    the home side re-resolved). Both sides different is a knockout slot
    being filled ("Semifinal 1 Winner" → a real club) or a re-used id: the
    real fixture must still get its own row, so that is not refused."""
    owner_pair = (owner["home_team_id"], owner["away_team_id"])
    if owner_pair == (home_id, away_id):
        return False
    return bool({home_id, away_id} & set(owner_pair))


def _log_refused_fork(
    cur, cfg: SportConfig, espn_id, refusal: ForkRefused, home_name, home_id, away_name, away_id, match_dt
) -> None:
    """Count a ForkRefused and log both team pairs, so the repair is obvious:
    the parse's sides as ESPN names them AND as the team rows they resolved
    to (ESPN's 'Santos' is two clubs; only the row names tell them apart)."""
    IDENTITY_STATS["forks_refused"] += 1
    owner = refusal.owner
    names = _team_names(cur, home_id, away_id)
    logger.error(
        "Refusing %s ESPN competition %s at %s: it already owns match %s, %s [%s] vs %s [%s], but now parses as "
        "%s -> team %r [%s] vs %s -> team %r [%s]. Writing it would insert a second row for the same fixture, "
        "splitting predictions/recs from the result. Nothing written. Repair the team mapping (a duplicate team "
        "row?) or the fixture row; this run exits %d.",
        cfg.sport,
        espn_id,
        match_dt,
        refusal.match_id,
        owner.get("home_name"),
        owner["home_team_id"],
        owner.get("away_name"),
        owner["away_team_id"],
        home_name,
        names.get(str(home_id), "?"),
        home_id,
        away_name,
        names.get(str(away_id), "?"),
        away_id,
        IDENTITY_EXIT_CODE,
    )


def _resolve_teams(
    cur,
    cfg: SportConfig,
    league_id: str,
    home_name: str,
    away_name: str,
    espn_id,
    match_dt,
    home_espn: str | None = None,
    away_espn: str | None = None,
    home_alt: tuple[str, ...] = (),
    away_alt: tuple[str, ...] = (),
):
    """(home_id, away_id) for a competition, or None when a side cannot be
    resolved (ensure_team counted why) or both sides resolve to ONE team. A
    team playing itself is a team-matching error (a fuzzy match, or a
    placeholder slot the filter missed: the 2026 World Cup left 20 such
    rows), never a fixture, so it is refused and counted. home_espn /
    away_espn are the sides' ESPN team ids (_competitor_espn_id) and
    home_alt / away_alt their other ESPN names (_competitor_alt_names);
    espn_id is the competition's."""
    home_id = ensure_team(cur, cfg, home_name, league_id, espn_id=home_espn, alt_names=home_alt)
    away_id = ensure_team(cur, cfg, away_name, league_id, espn_id=away_espn, alt_names=away_alt)
    if not (home_id and away_id):
        return None
    if home_id == away_id:
        IDENTITY_STATS["same_team"] += 1
        logger.error(
            "Refusing %s competition %s at %s: %r and %r both resolve to team %s — a team cannot play itself; "
            "fix the team matching (fuzzy match or an unfiltered placeholder slot). This run exits %d.",
            cfg.sport,
            espn_id,
            match_dt,
            home_name,
            away_name,
            home_id,
            IDENTITY_EXIT_CODE,
        )
        return None
    return home_id, away_id


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
    # Placeholder draw slots carry no identity. ESPN publishes future tennis
    # rounds as "TBD vs TBD"; each one becomes a matches row that can never be
    # updated when the players are known (the team ids change, so a NEW row is
    # created and the placeholder is orphaned forever). Prod 2026-09-04 carried
    # 3,061 such rows, 3,022 of them stale, holding 3,528 predictions and zero
    # recommendations — pure junk that also makes every pair ambiguous for the
    # healing check. Skip them at the door.
    if _is_placeholder_name(home_name) or _is_placeholder_name(away_name):
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
    still works (the marker is simply not written, i.e. "unknown").

    A fixture ESPN has postponed or cancelled (state 'post', not completed)
    goes to _record_status_change: this sweep sees the 14 days ahead, where
    a postponement is usually announced."""
    parts = _competition_parts(comp, cfg)
    if parts is None:
        return False
    home, away, home_name, away_name, match_dt, state, venue, espn_id = parts

    terminal = espn_terminal_status(comp)
    if terminal is not None:
        _record_status_change(cur, cfg, league_id, parts, *terminal)
        return False

    if state != "pre":
        return False  # results path (--results) handles post events

    teams = _resolve_teams(
        cur,
        cfg,
        league_id,
        home_name,
        away_name,
        espn_id,
        match_dt,
        home_espn=_competitor_espn_id(home, cfg.is_individual),
        away_espn=_competitor_espn_id(away, cfg.is_individual),
        home_alt=_competitor_alt_names(home, cfg.is_individual),
        away_alt=_competitor_alt_names(away, cfg.is_individual),
    )
    if teams is None:
        return False
    home_id, away_id = teams
    # The names in the ids' order, for the logs below; home_name / away_name
    # stay in ESPN's order (the healing key).
    log_home, log_away = home_name, away_name
    if _listed_reversed(cur, cfg, espn_id, home_id, away_id, match_dt):
        home_id, away_id = away_id, home_id  # the row's order
        log_home, log_away = away_name, home_name

    season_type = (event_context or {}).get("season_type")
    try:
        identity_safe = _reconcile_match_row(
            cur,
            cfg,
            espn_id,
            home_id,
            away_id,
            match_dt,
            allow_heal=healing_allowed((event_context or {}).get("pair_index"), home_name, away_name, match_dt),
        )
    except ForkRefused as refusal:
        _log_refused_fork(cur, cfg, espn_id, refusal, log_home, home_id, log_away, away_id, match_dt)
        return False
    written = bool(
        insert_scheduled_match(
            cur, cfg, league_id, home_id, away_id, match_dt, venue, season=season_label(cfg, match_dt, event_context)
        )
    )
    _stamp_identity(cur, home_id, away_id, match_dt, espn_id if identity_safe else None, season_type)
    if espn_id and not identity_safe:
        logger.error(
            "Not stamping ESPN id %s onto %s vs %s at %s: the row that owns this fixture could not "
            "be realigned onto that slot, so stamping would put one ESPN id on two rows",
            espn_id,
            log_home,
            log_away,
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

# A soccer match decided after 90 minutes: ESPN's STATUS_FINAL_AET ('AET')
# and STATUS_FINAL_PEN ('FT-Pens'). A two-legged tie that goes straight to a
# shootout is STATUS_FINAL_PEN with no extra time (Copa Libertadores).
_SOCCER_PAST_90_STATUSES = frozenset({"STATUS_FINAL_AET", "STATUS_FINAL_PEN"})
_SOCCER_PAST_90_DETAIL = re.compile(r"\b(?:AET|PENS?)\b", re.IGNORECASE)

_PLAY_MINUTE = re.compile(r"\s*(\d+)'")


def _competitor_id(competitor: dict) -> str:
    """ESPN's id for a side; a play's team.id is the team id, which the
    competitor id repeats."""
    return str(competitor.get("id") or (competitor.get("team") or {}).get("id") or "")


def _goal_plays(comp: dict) -> list[dict]:
    """The scoreboard's goals: scoring plays that are not shootout kicks."""
    return [p for p in comp.get("details") or [] if p.get("scoringPlay") and not p.get("shootout")]


def soccer_goals(comp: dict, home: dict, away: dict) -> list[tuple[str, int]] | None:
    """Every goal in a soccer scoreboard's scoring plays as (side, minute).

    ESPN's soccer scoreboard has no linescores, but `details` lists every
    goal with its clock and the credited team (an own goal is credited to
    the side it benefits). `side` is 'home' or 'away'; `minute` is the base
    minute of the clock, so stoppage time counts in the period it ends
    ("45'+2'" -> 45, "90'+7'" -> 90, "120'+5'" -> 120). Shootout kicks are
    not goals. None when a goal names neither side or has no readable clock:
    such a list cannot be trusted for any per-period score."""
    home_id, away_id = _competitor_id(home), _competitor_id(away)
    if not home_id or not away_id or home_id == away_id:
        return None
    goals: list[tuple[str, int]] = []
    for play in _goal_plays(comp):
        team = str((play.get("team") or {}).get("id") or "")
        minute = _PLAY_MINUTE.match((play.get("clock") or {}).get("displayValue") or "")
        if team not in (home_id, away_id) or minute is None:
            return None
        goals.append(("home" if team == home_id else "away", int(minute.group(1))))
    return goals


def _goals_until(
    comp: dict, home: dict, away: dict, final: tuple[int, int], last_minute: int
) -> tuple[int, int] | None:
    """(home, away) goals scored by `last_minute` (stoppage time included),
    or None unless the plays reproduce `final` exactly: a partial play list
    must not grade a market."""
    goals = soccer_goals(comp, home, away)
    if goals is None:
        return None
    total = (sum(s == "home" for s, _ in goals), sum(s == "away" for s, _ in goals))
    if total != tuple(final):
        return None
    return (
        sum(s == "home" and m <= last_minute for s, m in goals),
        sum(s == "away" and m <= last_minute for s, m in goals),
    )


def halftime_scores(comp: dict, home: dict, away: dict, final: tuple[int, int]) -> tuple[int, int] | None:
    """Soccer half-time (home, away) from the scoreboard's scoring plays:
    base minute <= 45, first-half stoppage included. None unless the plays
    reproduce the final score exactly — a partial play list must not grade
    four half-time markets."""
    return _goals_until(comp, home, away, final, 45)


def regulation_scores(comp: dict, home: dict, away: dict, final: tuple[int, int]) -> tuple[int, int] | None:
    """Soccer 90-minute (home, away) from the scoreboard's scoring plays:
    base minute <= 90, second-half stoppage included ("90'+3'"), extra time
    and the shootout excluded. Books settle every full-time market on it.
    None unless the plays reproduce the final, post-extra-time score."""
    return _goals_until(comp, home, away, final, 90)


def scoring_play_tally(comp: dict, home: dict, away: dict) -> tuple[int, int, int] | None:
    """(home, away, unattributed) goals counted from the scoring plays, or
    None when the scoreboard lists no goal at all (nothing to check the
    header score against). Needs only the credited team, not the clock."""
    plays = _goal_plays(comp)
    if not plays:
        return None
    home_id, away_id = _competitor_id(home), _competitor_id(away)
    teams = [str((p.get("team") or {}).get("id") or "") for p in plays]
    h = sum(t == home_id for t in teams) if home_id else 0
    a = sum(t == away_id for t in teams) if away_id and away_id != home_id else 0
    return h, a, len(teams) - h - a


def _soccer_went_past_90(status_type: dict, detail: str) -> bool:
    """ESPN says this soccer match was decided after 90 minutes."""
    return status_type.get("name") in _SOCCER_PAST_90_STATUSES or bool(_SOCCER_PAST_90_DETAIL.search(detail))


def _log_score_mismatch(
    cfg: SportConfig, espn_id, home_name, away_name, match_dt, header: tuple[int, int], tally, comp: dict
) -> None:
    """Count and log a result refused because ESPN's header score is
    contradicted by its own scoring plays."""
    IDENTITY_STATS["score_mismatch"] += 1
    plays = ", ".join(
        f"{(p.get('clock') or {}).get('displayValue') or '?'} team {(p.get('team') or {}).get('id')}"
        f"{' (own goal)' if p.get('ownGoal') else ''}"
        for p in _goal_plays(comp)
    )
    logger.error(
        "Refusing %s result %s vs %s at %s (ESPN %s): the header says %d-%d but its scoring plays add up to %d-%d"
        "%s [%s]. Nothing written: the header score is not stored when ESPN's own plays contradict it. This run "
        "exits %d until ESPN corrects it or the row is repaired by hand.",
        cfg.sport,
        home_name,
        away_name,
        match_dt,
        espn_id,
        header[0],
        header[1],
        tally[0],
        tally[1],
        f" with {tally[2]} goal(s) credited to neither side" if tally[2] else "",
        plays,
        IDENTITY_EXIT_CODE,
    )


# ESPN statuses that end a fixture without a result, and the matches.status
# each is recorded as. Until 2026-10-31 neither mode wrote them, so a
# postponed fixture sat 'scheduled' for good (Atromitos v AEK, gre.1
# 401896749, 2026-10-17: STATUS_POSTPONED, state 'post', completed false).
_ESPN_STATUS_RECORDED_AS = {
    "STATUS_POSTPONED": "postponed",
    "STATUS_CANCELED": "cancelled",
    "STATUS_CANCELLED": "cancelled",
    "STATUS_ABANDONED": "cancelled",
}

# The statuses each may replace. 'finished' is never touched (a result
# stands). A 'cancelled' row is replaced only when ESPN's own status put it
# there (espn_cancelled: a cancel ESPN turns into a postponement); one set on
# purpose — by an operator (an awarded match), by repair_fixture_orphans or by
# _release_redrawn_slot — stays final.
_STATUS_REPLACES = {"postponed": ("scheduled", "live"), "cancelled": ("scheduled", "live", "postponed")}

# Stamped on the first postponement only (never overwritten): the kickoff
# the recs were priced for and when the postponement was recorded.
# grade_completed_matches (POSTPONED_KICKOFF_KEY / POSTPONED_AT_KEY) voids a
# rec written before it when the match is played more than 72 h later.
POSTPONED_KICKOFF_KEY = "postponed_kickoff"
POSTPONED_AT_KEY = "postponed_at"

STATUS_ROW_BY_ESPN_SQL = """
    SELECT m.id, m.status, m.home_team_id, m.away_team_id, m.match_date, m.metadata
      FROM matches m
      JOIN leagues l ON l.id = m.league_id
     WHERE m.external_ids->>'espn' = %s
       AND l.sport = %s
     ORDER BY m.match_date DESC
     LIMIT 2
"""

STATUS_ROW_BY_TRIPLE_SQL = """
    SELECT id, status, home_team_id, away_team_id, match_date, metadata
      FROM matches
     WHERE home_team_id = %s AND away_team_id = %s AND match_date = %s
"""

# Guarded on the status the row was read with. The CASE drops the ESPN id
# when it is the one given (a cancelled tennis/MMA competition, see
# _record_status_change); NULL keeps every id.
STATUS_UPDATE_SQL = """
    UPDATE matches
       SET status = %s,
           metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb,
           external_ids = CASE WHEN external_ids->>'espn' = %s
                               THEN external_ids - 'espn' ELSE external_ids END,
           updated_at = NOW()
     WHERE id = %s
       AND status = %s
    RETURNING id
"""


def espn_terminal_status(comp: dict) -> tuple[str, str] | None:
    """(matches.status, ESPN status name) when ESPN says this competition
    was postponed, cancelled or abandoned, else None."""
    name = (comp.get("status", {}).get("type", {}).get("name") or "").upper()
    target = _ESPN_STATUS_RECORDED_AS.get(name)
    return (target, name) if target else None


def _status_target_row(cur, cfg: SportConfig, espn_id, home_id, away_id, match_dt, home_name, away_name):
    """The existing row a status change applies to, or None.

    By ESPN id (scoped to the sport) when the id names exactly one row and
    that row is this fixture: the same pair, or in a 1v1 sport the same two
    players in either order (ESPN orders MMA competitors freely, see
    _realign_match_date). Otherwise the exact (home, away, kickoff) triple —
    no healing: a status change never moves a row or guesses one.

    The id naming a row whose pair DRIFTED from this parse — one team shared,
    or the pair swapped (team sports) — is refused and counted
    (status_refused, exit 3), the same rule the fixture path refuses a fork
    by (_is_identity_drift): our team resolution moved under a fixed fixture.
    Both sides different (a knockout placeholder filled, a re-used id, a
    re-drawn tennis/MMA slot) or the id on two rows is only logged, and the
    triple decides, as the fixture path lets the real fixture have its row."""
    if espn_id:
        cur.execute(STATUS_ROW_BY_ESPN_SQL, (str(espn_id), cfg.sport))
        rows = cur.fetchall() or []
        if len(rows) == 1:
            row = rows[0]
            stored = (row["home_team_id"], row["away_team_id"])
            if stored == (home_id, away_id):
                return row
            if cfg.is_individual and {str(t) for t in stored} == {str(home_id), str(away_id)}:
                return row  # the same two players, listed the other way round
            if not cfg.is_individual and _is_identity_drift(row, home_id, away_id):
                IDENTITY_STATS["status_refused"] += 1
                names = _team_names(cur, row["home_team_id"], row["away_team_id"], home_id, away_id)
                logger.error(
                    "Refusing to record ESPN's status for %s %s vs %s at %s (ESPN %s): the id names row %s "
                    "(team %r [%s] v team %r [%s]), which shares a team with this parse (team %r [%s] v team %r "
                    "[%s]) — team resolution drifted under the fixture. Nothing written; repair the team mapping or "
                    "the row. This run exits %d.",
                    cfg.sport,
                    home_name,
                    away_name,
                    match_dt,
                    espn_id,
                    row["id"],
                    names.get(str(row["home_team_id"]), "?"),
                    row["home_team_id"],
                    names.get(str(row["away_team_id"]), "?"),
                    row["away_team_id"],
                    names.get(str(home_id), "?"),
                    home_id,
                    names.get(str(away_id), "?"),
                    away_id,
                    IDENTITY_EXIT_CODE,
                )
                return None
            logger.warning(
                "ESPN %s id %s names row %s (%s v %s), another fixture than %s vs %s (%s v %s: %s) — looking it "
                "up by its teams and kickoff %s",
                cfg.sport,
                espn_id,
                row["id"],
                row["home_team_id"],
                row["away_team_id"],
                home_name,
                away_name,
                home_id,
                away_id,
                "a re-drawn slot" if cfg.is_individual else "both teams differ: a filled slot or a re-used id",
                match_dt,
            )
        elif len(rows) > 1:
            logger.error(
                "ESPN %s id %s is on %d matches rows (%s) — the fixture is fragmented; repair the duplicates. "
                "Looking %s vs %s up by its teams and kickoff %s",
                cfg.sport,
                espn_id,
                len(rows),
                ", ".join(str(r["id"]) for r in rows),
                home_name,
                away_name,
                match_dt,
            )
    cur.execute(STATUS_ROW_BY_TRIPLE_SQL, (home_id, away_id, match_dt))
    return cur.fetchone()


def _existing_side(cur, cfg: SportConfig, name: str, league_id: str, competitor: dict) -> str | None:
    """One side of a competition the status path records, resolved as
    ensure_team resolves it (find_team's tiers, the side's ESPN team id
    first) but never inserted. By name alone ESPN's 'Santos' (Santos Laguna,
    ESPN 225) is Santos FC's row, so a Liga MX postponement found its own
    fixture row 'drifted' and was refused every sweep. An ESPN-id conflict
    (the id on two rows, the seed and a stamp disagreeing) leaves the side
    unresolved and is counted in IDENTITY_STATS['team_conflicts'], as
    ensure_team counts it: a postponed competition never reaches ensure_team."""
    found = _find_team(cur, cfg, name, league_id, _competitor_espn_id(competitor, cfg.is_individual))
    if found.how == "refused":
        IDENTITY_STATS["team_conflicts"] += 1
    return found.team_id


def _record_status_change(cur, cfg: SportConfig, league_id: str, parts: tuple, target: str, espn_status: str) -> bool:
    """Record ESPN's postponement / cancellation on the EXISTING row for
    this competition. Never inserts a row or a team (teams resolve through
    find_team's tiers, ESPN team id first: _existing_side), never touches a
    'finished' row or one cancelled on purpose (a cancel ESPN recorded may
    still become a postponement), and writes only when the status changes,
    guarded on the status read. Returns True when a row was written.

    A cancel stamps status_source / espn_status (what keeps it correctable,
    espn_cancelled). In a 1v1 sport it also drops the ESPN id, as
    _release_redrawn_slot does, recording it as cancelled_espn_id: ESPN
    re-fills a cancelled draw slot under the same id, and that pairing must
    resolve to its own row."""
    home, away, home_name, away_name, match_dt, _state, _venue, espn_id = parts
    home_id = _existing_side(cur, cfg, home_name, league_id, home)
    away_id = _existing_side(cur, cfg, away_name, league_id, away)
    if not (home_id and away_id) or home_id == away_id:
        return False  # no row of ours can be this fixture
    row = _status_target_row(cur, cfg, espn_id, home_id, away_id, match_dt, home_name, away_name)
    if row is None or row["status"] == target:
        return False
    old = row["status"]
    reopened = target == "postponed" and espn_cancelled(old, row.get("metadata"))
    if old not in _STATUS_REPLACES[target] and not reopened:
        if old == "finished":
            logger.warning(
                "ESPN reports %s for %s %s vs %s at %s (ESPN %s) but row %s is 'finished' — the result stands; "
                "check which is right",
                espn_status,
                cfg.sport,
                home_name,
                away_name,
                match_dt,
                espn_id,
                row["id"],
            )
        return False
    meta = dict(row.get("metadata") or {})
    patch: dict = {"espn_status": espn_status, "status_source": "espn_scoreboard"}
    if target == "postponed" and POSTPONED_KICKOFF_KEY not in meta:
        patch[POSTPONED_KICKOFF_KEY] = row["match_date"].isoformat()
        patch[POSTPONED_AT_KEY] = datetime.now(timezone.utc).isoformat()
    drop_espn_id = str(espn_id) if (target == "cancelled" and cfg.is_individual and espn_id) else None
    if drop_espn_id:
        patch["cancelled_espn_id"] = drop_espn_id
    cur.execute(STATUS_UPDATE_SQL, (target, json.dumps(patch), drop_espn_id, row["id"], old))
    if cur.fetchone() is None:
        logger.error(
            "Row %s (%s %s vs %s at %s) changed while recording ESPN's %s — left as it is",
            row["id"],
            cfg.sport,
            home_name,
            away_name,
            match_dt,
            espn_status,
        )
        return False
    logger.warning(
        "Recorded %s %s vs %s at %s (ESPN %s, row %s) as '%s' (was '%s'): ESPN reports %s",
        cfg.sport,
        home_name,
        away_name,
        match_dt,
        espn_id,
        row["id"],
        target,
        old,
        espn_status,
    )
    return True


def _record_result(cur, cfg: SportConfig, league_id: str, comp: dict, event_context: dict | None = None) -> bool:
    """Record one completed competition (state == 'post'). Grading needs:
    scores + status='finished' (all sports), and for NHL additionally
    metadata.regulation_winner ('tie' when the game went past regulation
    — grading_outcomes.nhl_regulation_outcome reads it). ESPN's status
    detail ('Final/OT', 'Final/SO') tells us that.

    Soccer, two guards on the score:
      * the header score must agree with ESPN's own scoring plays. ESPN
        401843358 (Aalesund v HamKam, 2026-05-29) reports FT 0-0 while its
        plays are 24' own goal (Aalesund), 44' and 54' HamKam, 57' Aalesund
        — 2-2, as football-data has it. Such a result is refused, logged and
        counted (exit 3), never stored. No plays at all is not a
        contradiction: a goalless match lists none.
      * a tie decided in extra time or on penalties stores the final,
        post-extra-time score in home_score/away_score and its 90-minute
        score in metadata.regulation_home_score / regulation_away_score
        with went_extra=true, when the plays reproduce the final score.
        Soccer full-time markets and recs settle on the 90-minute score
        (grading_outcomes.settlement_scores). Until 2026-10-31 they graded
        on the post-extra-time score.

    ESPN's postponed / cancelled / abandoned statuses go to
    _record_status_change instead (the existing row only)."""
    parts = _competition_parts(comp, cfg)
    if parts is None:
        return False
    home, away, home_name, away_name, match_dt, state, venue, espn_id = parts

    terminal = espn_terminal_status(comp)
    if terminal is not None:
        _record_status_change(cur, cfg, league_id, parts, *terminal)
        return False

    status_type = comp.get("status", {}).get("type", {})
    if state != "post" or not status_type.get("completed", False):
        return False

    scores = _final_scores(cfg, home, away)
    if scores is None:
        return False
    home_score, away_score = scores

    if cfg.sport == "soccer":
        tally = scoring_play_tally(comp, home, away)
        if tally is not None and tally != (home_score, away_score, 0):
            _log_score_mismatch(cfg, espn_id, home_name, away_name, match_dt, scores, tally, comp)
            return False

    detail = (status_type.get("detail") or status_type.get("shortDetail") or "").upper()
    went_extra = any(tok in detail for tok in _EXTRA_TIME_TOKENS)

    result_meta: dict = {"result_detail": detail or "FINAL", "result_source": "espn_scoreboard"}
    if cfg.sport == "nhl":
        # 'tie', not 'draw': the regulation model's labels, its training
        # query (regulation_winner IN ('home','tie','away')), the NHL loader
        # and the DB outcome trigger all use 'tie'. Writing 'draw' (until
        # 2026-10) graded every correct tie pick as a miss and silently
        # dropped every live-season regulation tie from the training frame.
        result_meta["regulation_winner"] = "tie" if went_extra else ("home" if home_score > away_score else "away")
    if cfg.sport == "soccer" and _soccer_went_past_90(status_type, detail):
        regulation = regulation_scores(comp, home, away, scores)
        if regulation is None:
            logger.warning(
                "%s %s vs %s at %s (ESPN %s) was decided after 90 minutes (%s) but its scoring plays do not give "
                "the 90-minute score; its full-time markets will grade on the final %d-%d",
                cfg.sport,
                home_name,
                away_name,
                match_dt,
                espn_id,
                detail,
                home_score,
                away_score,
            )
        else:
            result_meta.update(
                {"regulation_home_score": regulation[0], "regulation_away_score": regulation[1], "went_extra": True}
            )
    # Soccer half-time goes to matches.half_time_home/away, where the HT
    # trainers, the match API and the HT-market grader read it; None when
    # the plays don't reproduce the final score.
    ht = halftime_scores(comp, home, away, scores) if cfg.sport == "soccer" else None

    teams = _resolve_teams(
        cur,
        cfg,
        league_id,
        home_name,
        away_name,
        espn_id,
        match_dt,
        home_espn=_competitor_espn_id(home, cfg.is_individual),
        away_espn=_competitor_espn_id(away, cfg.is_individual),
        home_alt=_competitor_alt_names(home, cfg.is_individual),
        away_alt=_competitor_alt_names(away, cfg.is_individual),
    )
    if teams is None:
        return False
    home_id, away_id = teams
    # The names in the ids' order, for the logs below; home_name / away_name
    # stay in ESPN's order (the healing key).
    log_home, log_away = home_name, away_name
    if _listed_reversed(cur, cfg, espn_id, home_id, away_id, match_dt):
        # The row's order, and the winner flags with it (1v1: no half-time score).
        home_id, away_id = away_id, home_id
        home_score, away_score = away_score, home_score
        log_home, log_away = away_name, home_name

    # The RESULTS path fragments fixtures too: 20 of the 119 prod twins were
    # inserted here, after kickoff, because ESPN had moved the time since we
    # ingested the fixture. Reconcile first so this writes to the row that
    # already carries the predictions/odds/recs.
    season_type = (event_context or {}).get("season_type")
    try:
        identity_safe = _reconcile_match_row(
            cur,
            cfg,
            espn_id,
            home_id,
            away_id,
            match_dt,
            allow_heal=healing_allowed((event_context or {}).get("pair_index"), home_name, away_name, match_dt),
        )
    except ForkRefused as refusal:
        _log_refused_fork(cur, cfg, espn_id, refusal, log_home, home_id, log_away, away_id, match_dt)
        return False
    written = bool(
        insert_finished_match(
            cur,
            cfg,
            league_id,
            home_id,
            away_id,
            match_dt,
            venue,
            home_score,
            away_score,
            result_meta,
            halftime=ht,
            season=season_label(cfg, match_dt, event_context),
        )
    )
    if not written:
        # insert_finished_match leaves a 'cancelled' row alone when that
        # status was recorded on purpose (an abandoned or awarded match an
        # operator cancelled; a cancel ESPN itself reported is finished
        # instead), and ESPN can still report a "full time" score for it:
        # fra.2 747597, Bastia v Red Star, shows 0-0 STATUS_FULL_TIME although
        # the match was abandoned at 58' and awarded.
        logger.warning(
            "Left %s %s vs %s at %s (ESPN %s) as it is: the existing row is 'cancelled' but ESPN reports a final "
            "%d-%d (%s) — check which is right",
            cfg.sport,
            log_home,
            log_away,
            match_dt,
            espn_id,
            home_score,
            away_score,
            result_meta["result_detail"],
        )
        return False
    _stamp_identity(cur, home_id, away_id, match_dt, espn_id if identity_safe else None, season_type)
    if espn_id and not identity_safe:
        logger.error(
            "Not stamping ESPN id %s onto %s vs %s at %s: the row that owns this fixture could not "
            "be realigned onto that slot, so stamping would put one ESPN id on two rows",
            espn_id,
            log_home,
            log_away,
            match_dt,
        )
    return written


def process_event(
    cur, cfg: SportConfig, league_id: str, event: dict, handler=None, pair_index=None, league_slug: str | None = None
) -> int:
    """Process every match in an ESPN event with `handler` (default: the
    scheduled-fixture path; the results path passes _record_result).
    Returns the count of matches handled. For team sports this is 0 or 1
    (one match per event); for tennis tournaments it can be dozens (one
    event = full bracket of matches in progress on the same day).
    `league_slug` reaches the handlers in the event context (season_label
    reads it)."""
    handler = handler or _process_competition
    inserted = 0
    for comp, context in _iter_event_competitions(event, cfg):
        if pair_index is not None:
            context = {**context, "pair_index": pair_index}
        if league_slug is not None:
            context = {**context, "league_slug": league_slug}
        if handler(cur, cfg, league_id, comp, context):
            inserted += 1
    return inserted


def record_results(cur, cfg: SportConfig, league_id: str, events: list[dict], league_slug: str) -> int:
    """The results path over one league's buffered sweep: every competition
    through _record_result, then (fight cards) cancel_dropped_bouts. Returns
    the count of results written. fetch_results_all and
    scripts/backfill_results.py both run exactly this, so a backfill writes
    what the 15-min sweep would have written."""
    pair_index = build_pair_index(cfg, events)
    n = 0
    for ev in events:
        n += process_event(
            cur, cfg, league_id, ev, handler=_record_result, pair_index=pair_index, league_slug=league_slug
        )
    cancel_dropped_bouts(cur, cfg, league_id, events)
    return n


# A bout ESPN dropped from a fight card simply disappears from the event — an
# opponent replaced under a new competition id, a fighter withdrawn, the bout
# moved to another card. ESPN never reports a status for it, so its row sat
# 'scheduled' for good: its predictions never graded and a rec on it never
# settled (Holland v Smith, UFC 2026-07-18: one rec stranded since July; prod
# 2026-11-01: 8 of the 18 stale UFC rows of September/October). Once a card is
# finished, a stale row of its window that the card does not list is
# cancelled here; grade_completed_matches voids its open recs 72 h after
# kickoff ('void: match cancelled').
#
# CARD_WINDOW_HOURS bounds the rows a card speaks for: a UFC card's bouts sit
# at its segment start times (prelims, main card), a few hours apart, and
# cards are days apart.
CARD_WINDOW_HOURS = 6

DROPPED_CANDIDATES_SQL = """
    SELECT m.id, m.home_team_id, m.away_team_id, m.match_date, m.metadata,
           m.external_ids->>'espn' AS espn_id
      FROM matches m
     WHERE m.league_id = %s
       AND m.status = 'scheduled'
       AND m.match_date BETWEEN %s AND %s
       AND m.match_date < NOW()
     ORDER BY m.match_date, m.id
"""


def _finished_card(event: dict, cfg: SportConfig):
    """(event id, first bout time, last bout time, competitions) for an ESPN
    fight card whose every competition is over (state 'post': fought, or
    cancelled / postponed by ESPN's own status), else None."""
    comps = [comp for comp, _context in _iter_event_competitions(event, cfg)]
    times = []
    for comp in comps:
        if (comp.get("status", {}).get("type", {}).get("state") or "pre").lower() != "post":
            return None
        raw = comp.get("date") or comp.get("startDate")
        try:
            times.append(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
        except ValueError:
            return None
    if not comps:
        return None
    return event.get("id"), min(times), max(times), comps


def cancel_dropped_bouts(cur, cfg: SportConfig, league_id: str, events: list[dict]) -> int:
    """Cancel the stale 'scheduled' rows a finished fight card no longer lists
    (why: the comment above CARD_WINDOW_HOURS). Fight-card sports only
    (cfg.card_events); results mode only, after every competition of the sweep
    went through _record_result. Returns the count of rows cancelled.

    A row of a finished card's window (its first to its last bout, +/-
    CARD_WINDOW_HOURS, kickoff past) is left alone when the sweep lists its
    ESPN competition id, or when its two fighters meet on the card (a
    duplicate of a fought bout: repair_fixture_orphans / repair_stranded_recs
    settle it from its twin). Otherwise it is cancelled when it carries an
    ESPN id (ESPN no longer lists that competition anywhere in the sweep), or
    — a row from before ESPN ids were stored — when neither fighter is on the
    card at all. An unidentified row with one fighter on the card against
    someone else is NOT cancelled: an opponent change and a fighter stored
    under two names ('Zachary Reese' / 'Zach Reese') look the same, and only
    the latter was fought. It is logged and left for repair_fixture_orphans.
    A card one of whose competitions cannot be parsed is skipped whole: it
    cannot prove a bout absent.

    The cancel is _record_status_change's for a 1v1 sport: status
    'cancelled', espn_status DROPPED_FROM_CARD / status_source
    espn_scoreboard (correctable: ESPN_CANCEL_STATUSES), the ESPN id moved to
    cancelled_espn_id, the card's event id in dropped_from_event; guarded on
    status 'scheduled'."""
    if not cfg.card_events:
        return 0
    listed = {str(comp.get("id")) for ev in events for comp, _ in _iter_event_competitions(ev, cfg) if comp.get("id")}
    cancelled = 0
    for ev in events:
        card = _finished_card(ev, cfg)
        if card is None:
            continue
        event_id, first, last, comps = card
        pairs: set[frozenset] = set()
        fighters: set[str] = set()
        unparsed = False
        for comp in comps:
            parts = _competition_parts(comp, cfg)
            if parts is None:
                unparsed = True
                break
            home_id = find_team(cur, cfg, parts[2], league_id)
            away_id = find_team(cur, cfg, parts[3], league_id)
            ids = {str(t) for t in (home_id, away_id) if t}
            fighters |= ids
            if len(ids) == 2:
                pairs.add(frozenset(ids))
        if unparsed:
            logger.warning(
                "%s card %s: a competition could not be parsed, so the card cannot prove a bout absent — "
                "no dropped bout is cancelled from it",
                cfg.sport,
                event_id,
            )
            continue
        window = timedelta(hours=CARD_WINDOW_HOURS)
        cur.execute(DROPPED_CANDIDATES_SQL, (league_id, first - window, last + window))
        for row in cur.fetchall() or []:
            espn_id = row.get("espn_id")
            pair = {str(row["home_team_id"]), str(row["away_team_id"])}
            if (espn_id and espn_id in listed) or frozenset(pair) in pairs:
                continue
            if not espn_id and pair & fighters:
                logger.warning(
                    "%s row %s (%s v %s at %s) is not on finished card %s, but one of its fighters is, against "
                    "someone else — an opponent change or one fighter under two names; left for "
                    "repair_fixture_orphans",
                    cfg.sport,
                    row["id"],
                    row["home_team_id"],
                    row["away_team_id"],
                    row["match_date"],
                    event_id,
                )
                continue
            patch = {
                "espn_status": DROPPED_FROM_CARD,
                "status_source": "espn_scoreboard",
                "dropped_from_event": str(event_id),
            }
            if espn_id:
                patch["cancelled_espn_id"] = espn_id
            cur.execute(STATUS_UPDATE_SQL, ("cancelled", json.dumps(patch), espn_id, row["id"], "scheduled"))
            if cur.fetchone() is None:
                logger.error(
                    "%s row %s changed while cancelling it as dropped from card %s — left as it is",
                    cfg.sport,
                    row["id"],
                    event_id,
                )
                continue
            cancelled += 1
            logger.warning(
                "Cancelled %s row %s (%s v %s at %s, ESPN %s): finished card %s does not list it — the bout was "
                "dropped from the card",
                cfg.sport,
                row["id"],
                row["home_team_id"],
                row["away_team_id"],
                row["match_date"],
                espn_id or "-",
                event_id,
            )
    return cancelled


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
                    n += process_event(cur, cfg, league_id, ev, pair_index=pair_index, league_slug=slug)
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
                n = record_results(cur, cfg, league_id, events, slug)
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
        if FETCH_STATS["failed"] and not FETCH_STATS["ok"]:
            logger.error(
                "Every ESPN request for %s failed (%d) — treating as an outage", cfg.sport, FETCH_STATS["failed"]
            )
            return 1
        total = sum(counts.values())
        logger.info("Recorded %d %s result(s) across %d leagues", total, cfg.sport, len(counts))
        return identity_exit_code(cfg)
    counts = fetch_all(args.database_url, cfg, leagues, args.days)
    total = sum(counts.values())
    logger.info("Fetched %d upcoming %s fixtures across %d leagues", total, cfg.sport, len(counts))
    return identity_exit_code(cfg)


def identity_exit_code(cfg: SportConfig) -> int:
    """0, or IDENTITY_EXIT_CODE when the sweep refused a competition or did
    something to a team that needs a person (see IDENTITY_STATS). The rest
    of the sweep has already been committed."""
    forks, same_team = IDENTITY_STATS["forks_refused"], IDENTITY_STATS["same_team"]
    scores, statuses = IDENTITY_STATS["score_mismatch"], IDENTITY_STATS["status_refused"]
    namesakes, conflicts = IDENTITY_STATS["team_namesakes"], IDENTITY_STATS["team_conflicts"]
    refused = forks + same_team + scores + statuses
    if not (refused or namesakes or conflicts):
        return 0
    logger.error(
        "%s sweep REFUSED %d write(s): %d would have forked an ESPN-identified fixture, %d resolved both sides to "
        "one team, %d carried a header score its own scoring plays contradict, %d status change(s) named a row "
        "for another team pair. %d club(s) were inserted as their own row because another ESPN team holds their "
        "name; %d ESPN team id conflict(s). Everything else was written and committed; exiting %d so this run "
        "fails until they are repaired, ESPN corrects them, or the team mapping is checked (details in the ERROR "
        "lines above).",
        cfg.sport,
        refused,
        forks,
        same_team,
        scores,
        statuses,
        namesakes,
        conflicts,
        IDENTITY_EXIT_CODE,
    )
    return IDENTITY_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
