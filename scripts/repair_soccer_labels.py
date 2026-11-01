"""Repair soccer rows that a football-data file filed under the wrong league
or decoded wrongly.

THE BUGS (prod, read only, 2026-11-01; scripts/text_encoding.py has the root
cause, fixed in the loaders the same day)
-----------------------------------------------------------------------
Both football-data loaders read every file as latin-1. football-data's files
from 2024-25 on are UTF-8 with a byte-order mark, so:

  * The BOM hid the Div column (it read as 'ï»¿Div'), the one field that
    names a file's division. On 2026-08-06 football-data served the Scottish
    Championship and League One openers of 2026-08-01 at the 2026-27 SP1 and
    SP2 URLs; nothing checked, and promote_raw wrote ten Scottish fixtures as
    La Liga / La Liga 2 matches. promote_raw gives a team the league of the
    first raw row naming it, so twenty clubs got a Spanish league: fourteen
    that have no other match (Arbroath, Ayr, Morton, Queens Park, Raith Rvs,
    Stenhousemuir; Airdrie Utd, Alloa, Cove Rangers, East Fife, East Kilbride,
    Montrose, Peterhead, Queen of Sth) and six with 38-448 Premiership
    matches each (Dunfermline, Inverness C, Livingston, Partick; Hamilton,
    Ross County). The matches are wrong AND the teams are wrong. Downstream:
    team_identity.country_candidates puts a club in a country by its own
    league or its latest match's league, so all twenty are Spanish candidates
    for ESPN's La Liga names and none is a Scottish one — the next Premiership
    name variant of the six ('Partick Thistle', 'Inverness Caledonian
    Thistle') would mint a history-less twin instead of resolving; the
    Dixon-Coles / Poisson per-league baselines of La Liga (7,348 matches) and
    La Liga 2 (8,767) carry five Scottish lower-division games each (shrunk at
    200 effective matches: negligible, but wrong); the UI lists them under La
    Liga. No prediction, recommendation or odds snapshot exists on the ten (the
    loader ran after they were played); each has 13 football-data closing-odds
    rows, 2 match_stats rows and 1 features_cache row, none of which holds a
    league. Season labels do not change (SP1/SP2 and SC1/SC2 are all
    cross-year: '2026-2027').
  * The UTF-8 'Preußen Münster' (2. Bundesliga 2024-25 and 2025-26) became
    'PreuÃ\\x9fen MÃ¼nster': 68 raw rows (34 home, 34 away) and one team with
    68 matches. No correctly spelled twin exists (2. Bundesliga is not an ESPN
    league), so it is a RENAME, not a merge. A sweep of every text and json
    column of the 44 non-Airflow tables (C1 controls, UTF-8 read as latin-1 or
    Windows-1252, BOM, U+FFFD, '?' inside a word) found nothing else.
  * Every other raw row was checked against football-data's files of
    2026-11-01: the 12,018 whose Div was hidden are in their own league's file
    with their own Div, the 93,629 with a Div name their league, and the
    62,430 /new/ rows are in their country's file.

WHAT IT PLANS (from the database at run time; counts on the PLAN line)
------------------------------------------------------------------------
  create_league  a league the relabel needs that does not exist yet, by
                 code or by (name, country) (SC1 Championship and SC2 League
                 One, Scotland): the row promote_raw.upsert_leagues inserts
                 for a football-data code (LEAGUE_MAP's name and country,
                 sport soccer, external_ids {football_data: code}).
  tag_league     a league the relabel needs whose row exists under that name
                 and country without a football_data code (one ESPN created,
                 say): the code is added to its external_ids, as
                 upsert_leagues' ON CONFLICT (name, country, sport) would do
                 — never a second row, and never a DELETE on reversal.
  relabel_raw    a raw_football_data row of the reviewed map
                 (scripts/data/soccer_misfiled_division_rows.csv) still under
                 its wrong code: league and Div set to the right code. Its
                 kickoff and score must agree with the map.
  relabel_match  the matches row promote_raw wrote from it — the same home and
                 away team ids (exact name or alias, as promote_raw resolves
                 them), a kickoff within promote_raw's window of the
                 football-data kickoff on its UK date (so it is found before
                 and after repair_football_data_kickoffs moved it), still in
                 the wrong league: league_id set.
  relabel_team   a team on one of those matches whose league is in another
                 country than the right league: league_id set to the right
                 league of its latest mapped match. promote_raw, re-run, sets
                 a team's league from the first raw row naming it; once the
                 raw rows are relabelled that is a Scottish league either way.
  rename_team    a soccer team whose name looks mis-decoded
                 (text_encoding.looks_mojibake) and decodes cleanly
                 (text_encoding.repair_mojibake): name and normalized_name
                 set to the decoded spelling — only when no other team holds
                 that spelling as its name, an alias or its accent fold. A
                 twin is a MERGE (scripts/merge_soccer_teams.py and its
                 reviewed map), refused here.
  rename_raw     a raw_football_data row with a mis-decoded HomeTeam/AwayTeam:
                 set to the decoded spelling, so the next promote_raw run
                 resolves it to the renamed team (it refuses mis-decoded names
                 now) and the next load, decoding right, meets its key (ON
                 CONFLICT DO NOTHING) instead of adding a duplicate row.
Reported, never written: already_raw / already_match (map rows already
right), mojibake_left (mis-decoded values the plan does not repair: soccer
team names, aliases and metadata, soccer league names and countries, raw
names, Div, league and season), cross_country_after (soccer teams whose
domestic matches would still span leagues of two countries, International
excluded; PLAN_CROSS_COUNTRY names them — Santos and Guadalajara until
repoint_team_matches has moved their Liga MX rows, none after).

REFUSED (counted as refuse=N, each logged at ERROR; --apply writes nothing):
a league the map needs whose name and country a row with another
football_data code holds (tagging would overwrite that code); a map row
whose raw row is missing, or whose kickoff or score disagrees; a
mapped match row missing, ambiguous (two candidates) or in a third league; a
mapped team name that resolves to no team; a raw row both relabelled and
renamed; a mis-decoded name that does not decode cleanly; a rename whose
spelling another team holds (a merge); a raw rename whose decoded key already
exists (a duplicate row a reload wrote); a mapped match whose team would end
up with domestic matches in two countries.

--apply re-plans in ONE transaction: --expect-plan pins it to the dry run's
counts (any drift refuses before a write) and a plan that refuses anything
or leaves a mis-decoded value writes nothing; the leagues are created first
(an INSERT that meets an existing row rolls everything back instead of
merging into it) and tagged (guarded on the row still having no code), then
the write-once audit CSV and the reversal SQL beside it are written (the
CSV's old/new value per row and column; the SQL restores every old value,
guarded on the new one, removes the code it tagged and drops the leagues it
created once nothing references them), then every UPDATE runs guarded on
the value it replaces (a row changed since the plan rolls everything back),
and a re-plan inside the transaction must find nothing left to write,
nothing refused and no mis-decoded value left before it commits. On a
rollback the audit CSV and reversal SQL are renamed *.void: they describe
changes that never happened. --verify <apply CSV> checks, read-only, that
every row of that CSV holds its new value.

It changes the training corpus (matches.league_id of the mapped fixtures, the
renamed team's name, which keys the Dixon-Coles / Poisson members) but no
feature: compute_features reads team ids, odds and match_stats, never a league
or a name. scripts/ops/repair_soccer_labels.sh runs it (backup, pinned
apply, after-state) and prints the features rebuild.

Usage (inside the api container):
    python /app/scripts/repair_soccer_labels.py                    # dry run
    python /app/scripts/repair_soccer_labels.py --apply --expect-plan create_league=2,relabel_raw=10,...
    python /app/scripts/repair_soccer_labels.py --verify /app/backups/soccer_labels_apply_<utc>.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import promote_raw as pr  # noqa: E402
import team_identity  # noqa: E402
import text_encoding  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("repair_soccer_labels")

MAP_PATH = Path(__file__).resolve().parent / "data" / "soccer_misfiled_division_rows.csv"
MAP_FIELDS = (
    "season",
    "date",
    "time",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "from_league",
    "to_league",
    "evidence",
)

WRITE_COUNTERS = (
    "create_league",
    "tag_league",
    "relabel_raw",
    "relabel_match",
    "relabel_team",
    "rename_team",
    "rename_raw",
)
PLAN_COUNTERS = (*WRITE_COUNTERS, "already_raw", "already_match", "refuse", "mojibake_left", "cross_country_after")
CSV_FIELDS = ("kind", "table", "row_key", "column", "old_value", "new_value", "detail")

LEAGUES_SQL = """
    SELECT id::text AS id, name, country, external_ids->>'football_data' AS code
      FROM leagues WHERE sport = 'soccer'
"""
TEAMS_SQL = """
    SELECT id::text AS id, name, normalized_name, league_id::text AS league_id, metadata::text AS metadata_text
      FROM teams WHERE sport = 'soccer'
"""
RAW_SQL = 'SELECT "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Div", league, season FROM raw_football_data'
CANDIDATES_SQL = """
    SELECT id::text AS id, league_id::text AS league_id, match_date
      FROM matches
     WHERE home_team_id = %s AND away_team_id = %s AND match_date >= %s AND match_date <= %s
"""
# Soccer teams whose domestic (non-International) matches span leagues of more than one country.
CROSS_COUNTRY_SQL = """
    WITH side AS (
        SELECT m.home_team_id AS team_id, l.country FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'soccer' AND l.country <> 'International'
        UNION ALL
        SELECT m.away_team_id, l.country FROM matches m JOIN leagues l ON l.id = m.league_id
         WHERE l.sport = 'soccer' AND l.country <> 'International'
    )
    SELECT team_id::text AS team_id FROM side GROUP BY team_id HAVING count(DISTINCT country) > 1
"""
# (team, match, country) of every domestic match of the teams given.
TEAM_SIDES_SQL = """
    SELECT x.team_id::text AS team_id, x.match_id::text AS match_id, l.country
      FROM (SELECT id AS match_id, league_id, home_team_id AS team_id FROM matches WHERE home_team_id = ANY(%s::uuid[])
            UNION ALL
            SELECT id, league_id, away_team_id FROM matches WHERE away_team_id = ANY(%s::uuid[])) x
      JOIN leagues l ON l.id = x.league_id
     WHERE l.sport = 'soccer' AND l.country <> 'International'
"""


class PlanDrift(RuntimeError):
    """--expect-plan does not match the plan computed at apply time."""


class Refused(RuntimeError):
    """The plan refuses something: nothing is written."""


@dataclass(frozen=True)
class MapRow:
    season: str
    date: str
    time: str
    home: str
    away: str
    home_goals: int
    away_goals: int
    from_league: str
    to_league: str
    evidence: str

    @property
    def raw_key(self) -> tuple[str, str, str]:
        return (self.date, self.home, self.away)

    @property
    def label(self) -> str:
        return f"{self.from_league}->{self.to_league} {self.date} {self.home} v {self.away}"


def load_map(path: Path | str | None = None) -> list[MapRow]:
    """The reviewed map; a malformed one raises ValueError (nothing is planned from it)."""
    p = Path(path) if path else MAP_PATH
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    rows: list[MapRow] = []
    seen: set[tuple[str, str, str]] = set()
    for i, rec in enumerate(csv.DictReader(lines), start=1):
        if set(rec) != set(MAP_FIELDS):
            raise ValueError(f"{p}: columns {list(rec)} are not {list(MAP_FIELDS)}")
        try:
            row = MapRow(
                **{k: rec[k].strip() for k in MAP_FIELDS if k not in ("home_goals", "away_goals")},
                home_goals=int(rec["home_goals"]),
                away_goals=int(rec["away_goals"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{p} row {i}: {exc}") from None
        for code in (row.from_league, row.to_league):
            if code not in pr.LEAGUE_MAP:
                raise ValueError(f"{p} row {i}: league code {code!r} is not in promote_raw.LEAGUE_MAP")
        if row.from_league == row.to_league:
            raise ValueError(f"{p} row {i}: from_league equals to_league ({row.from_league})")
        if row.raw_key in seen:
            raise ValueError(f"{p} row {i}: {row.raw_key} is listed twice")
        seen.add(row.raw_key)
        rows.append(row)
    return rows


@dataclass
class State:
    """One snapshot of what the plan reads."""

    leagues_by_code: dict[str, dict]
    leagues_by_id: dict[str, dict]
    leagues_by_name: dict[tuple[str, str], dict]  # (name, country): the soccer key upsert_leagues conflicts on
    teams: dict[str, dict]
    team_by_norm: dict[str, str]
    aliases: dict[str, set[str]]
    raw: dict[tuple[str, str, str], dict]
    candidates: dict[tuple[str, str, str], list[dict]] = field(default_factory=dict)
    cross_country_now: set[str] = field(default_factory=set)
    sides: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # team -> [(match_id, country)]


def resolve_team(state: State, name: str) -> Optional[str]:
    """The team id a raw name promotes onto: exact normalized_name, then an alias (promote_raw.upsert_teams)."""
    norm = pr.normalize_team_name(name)
    return state.team_by_norm.get(norm) or team_identity.resolve_alias(state.aliases, "soccer", norm)


def mapped_kickoff_window(row: MapRow) -> tuple[datetime, datetime]:
    """Where promote_raw looks for the row it wrote from this raw row (promote_raw.existing_match_window of the
    corrected kickoff on the UK date); it holds the legacy kickoff (UK wall clock read as UTC) too."""
    kickoff = pr.parse_match_date(row.date, row.time)
    day = pr.parse_wall_clock(row.date, None)
    if kickoff is None or day is None:
        raise ValueError(f"map row {row.label}: unreadable date/time {row.date!r} {row.time!r}")
    return pr.existing_match_window(kickoff, day.date())


@dataclass
class Plan:
    counts: Counter
    entries: list[dict]
    refusals: list[str]
    creates: list[str]  # league codes to create
    tags: list[tuple[str, str]]  # (league code, id of the code-less row with LEAGUE_MAP's name and country)
    relabel_raw: list[tuple[MapRow, dict]]
    relabel_match: list[tuple[MapRow, dict, str, Optional[str]]]  # (row, match, old league id, new league id|None)
    relabel_team: list[tuple[str, Optional[str], str]]  # (team id, old league id, new code)
    rename_team: list[tuple[str, str, str, str, str]]  # (team id, old name, old norm, new name, new norm)
    rename_raw: list[tuple[tuple[str, str, str], tuple[str, str]]]  # (old key, (new home, new away))
    cross_country_names: list[str]


def _league_id(state: State, code: str) -> Optional[str]:
    lg = state.leagues_by_code.get(code)
    return lg["id"] if lg else None


def _league_country(state: State, league_id: Optional[str]) -> Optional[str]:
    lg = state.leagues_by_id.get(league_id or "")
    return lg["country"] if lg else None


def _entry(kind: str, table: str, row_key: str, column: str, old, new, detail: str = "") -> dict:
    return {
        "kind": kind,
        "table": table,
        "row_key": row_key,
        "column": column,
        "old_value": "" if old is None else str(old),
        "new_value": "" if new is None else str(new),
        "detail": detail,
    }


def raw_key_text(key: tuple[str, str, str]) -> str:
    return " | ".join(key)


def plan(state: State, map_rows: list[MapRow]) -> Plan:
    """Everything to write, refuse and report, from one snapshot. Pure."""
    counts: Counter = Counter({k: 0 for k in PLAN_COUNTERS})
    p = Plan(counts, [], [], [], [], [], [], [], [], [], [])

    def refuse(msg: str) -> None:
        p.refusals.append(msg)

    # Leagues the map needs: by code, else by the (name, country) upsert_leagues conflicts on.
    target_ids: dict[str, Optional[str]] = {}  # code -> its league id once the plan is applied (None: created)
    for code in sorted({r.to_league for r in map_rows}):
        if code in state.leagues_by_code:
            target_ids[code] = state.leagues_by_code[code]["id"]
            continue
        name, country = pr.LEAGUE_MAP[code]
        same = state.leagues_by_name.get((name, country))
        target_ids[code] = None
        if same is None:
            p.creates.append(code)
            p.entries.append(_entry("create_league", "leagues", code, "row", None, f"{name} | {country} | soccer"))
        elif same.get("code"):
            refuse(
                f"league {name!r} ({country}) {same['id']} carries football_data code {same['code']!r}: giving it "
                f"{code} would overwrite that — fix the league by hand"
            )
        else:
            target_ids[code] = same["id"]
            p.tags.append((code, same["id"]))
            p.entries.append(
                _entry(
                    "tag_league",
                    "leagues",
                    same["id"],
                    "football_data",
                    None,
                    code,
                    f"{name} | {country} | soccer: the existing row has no football_data code",
                )
            )

    def target_country(code: str) -> str:
        return pr.LEAGUE_MAP[code][1]

    # Raw rows and their matches.
    new_league_of_match: dict[str, str] = {}  # match id -> country after the plan
    team_targets: dict[str, tuple[str, str]] = {}  # team id -> (date key, to code) of its latest mapped match
    for row in map_rows:
        raw = state.raw.get(row.raw_key)
        if raw is None:
            refuse(f"{row.label}: no raw_football_data row with that Date/HomeTeam/AwayTeam")
            continue
        observed = ((raw.get("Time") or ""), pr.to_int(raw.get("FTHG")), pr.to_int(raw.get("FTAG")))
        if observed != (row.time, row.home_goals, row.away_goals):
            refuse(
                f"{row.label}: the raw row says {raw.get('Time')} {raw.get('FTHG')}-{raw.get('FTAG')}, the map "
                f"{row.time} {row.home_goals}-{row.away_goals}"
            )
            continue
        if raw.get("league") == row.to_league:
            counts["already_raw"] += 1
        elif raw.get("league") == row.from_league and (raw.get("season") or "") == row.season:
            p.relabel_raw.append((row, raw))
            rkey = raw_key_text(row.raw_key)
            p.entries.append(_entry("relabel_raw", "raw_football_data", rkey, "league", raw["league"], row.to_league))
            p.entries.append(_entry("relabel_raw", "raw_football_data", rkey, "Div", raw.get("Div"), row.to_league))
        else:
            refuse(f"{row.label}: the raw row is filed as {raw.get('league')} {raw.get('season')}, not {row.season}")
            continue
        home_id, away_id = resolve_team(state, row.home), resolve_team(state, row.away)
        if not (home_id and away_id):
            refuse(f"{row.label}: team name(s) resolve to no team (home {home_id}, away {away_id})")
            continue
        cands = state.candidates.get(row.raw_key, [])
        if len(cands) != 1:
            refuse(
                f"{row.label}: {len(cands)} matches row(s) for that pair near the kickoff "
                f"({', '.join(c['id'] for c in cands) or 'none'}) — expected exactly one"
            )
            continue
        match = cands[0]
        from_id, to_id = _league_id(state, row.from_league), target_ids.get(row.to_league)
        if to_id and match["league_id"] == to_id:
            counts["already_match"] += 1
        elif match["league_id"] == from_id:
            p.relabel_match.append((row, match, match["league_id"], to_id))
            p.entries.append(
                _entry(
                    "relabel_match",
                    "matches",
                    match["id"],
                    "league_id",
                    match["league_id"],
                    to_id or f"(new {row.to_league})",
                    f"{row.label} at {match['match_date']}",
                )
            )
        else:
            league = state.leagues_by_id.get(match["league_id"], {})
            refuse(f"{row.label}: matches row {match['id']} is in {league.get('code')} {league.get('name')}")
            continue
        new_league_of_match[match["id"]] = target_country(row.to_league)
        for tid in (home_id, away_id):
            sort_key = row.date[6:] + row.date[3:5] + row.date[:2] + row.time
            if tid not in team_targets or team_targets[tid][0] < sort_key:
                team_targets[tid] = (sort_key, row.to_league)

    # Teams on mapped matches whose league is in another country.
    for tid, (_k, code) in sorted(team_targets.items()):
        team = state.teams[tid]
        if _league_country(state, team["league_id"]) != target_country(code):
            p.relabel_team.append((tid, team["league_id"], code))
            p.entries.append(
                _entry(
                    "relabel_team",
                    "teams",
                    tid,
                    "league_id",
                    team["league_id"],
                    target_ids.get(code) or f"(new {code})",
                    f"{team['name']}: {_league_country(state, team['league_id'])} -> {code} {target_country(code)}",
                )
            )

    # Teams whose domestic matches would span countries after the plan.
    after: list[str] = []
    for tid in sorted(state.cross_country_now | set(team_targets)):
        countries = {new_league_of_match.get(mid, country) for mid, country in state.sides.get(tid, [])}
        if len(countries) > 1:
            after.append(tid)
            if tid in team_targets:
                refuse(
                    f"team {state.teams[tid]['name']} ({tid}) would still play domestic matches in "
                    f"{sorted(countries)} after the relabel — is the map right?"
                )
    p.cross_country_names = sorted(state.teams[t]["name"] for t in after if t in state.teams)
    counts["cross_country_after"] = len(after)

    # Mis-decoded team names.
    renamed: set[str] = set()
    for tid, team in sorted(state.teams.items()):
        if not (text_encoding.looks_mojibake(team["name"]) or text_encoding.looks_mojibake(team["normalized_name"])):
            continue
        new_name = text_encoding.repair_mojibake(team["name"])
        if new_name is None:
            refuse(f"team {tid} {team['name']!r}: looks mis-decoded but does not decode cleanly — fix it by hand")
            continue
        new_norm = pr.normalize_team_name(new_name)
        holders = {state.team_by_norm.get(new_norm), team_identity.resolve_alias(state.aliases, "soccer", new_norm)}
        holders |= {
            other
            for other, t in state.teams.items()
            if team_identity.strip_accents(t["normalized_name"]) == team_identity.strip_accents(new_norm)
        }
        holders -= {None, tid}
        if holders:
            refuse(
                f"team {tid} {team['name']!r} decodes to {new_name!r}, which team(s) {sorted(h for h in holders if h)} "
                "already hold (name, alias or accent fold): that is a MERGE — add the pair to "
                "scripts/data/soccer_team_merge_map.csv and run scripts/ops/merge_soccer_teams.sh first"
            )
            continue
        renamed.add(tid)
        p.rename_team.append((tid, team["name"], team["normalized_name"], new_name, new_norm))
        p.entries.append(_entry("rename_team", "teams", tid, "name", team["name"], new_name))
        p.entries.append(_entry("rename_team", "teams", tid, "normalized_name", team["normalized_name"], new_norm))

    # Mis-decoded raw names.
    relabel_keys = {row.raw_key for row, _raw in p.relabel_raw}
    for key, raw in sorted(state.raw.items()):
        home, away = raw["HomeTeam"] or "", raw["AwayTeam"] or ""
        if not (text_encoding.looks_mojibake(home) or text_encoding.looks_mojibake(away)):
            continue
        new_home = text_encoding.repair_mojibake(home) if text_encoding.looks_mojibake(home) else home
        new_away = text_encoding.repair_mojibake(away) if text_encoding.looks_mojibake(away) else away
        if new_home is None or new_away is None:
            refuse(f"raw row {raw_key_text(key)}: a name looks mis-decoded but does not decode cleanly")
            continue
        if key in relabel_keys:
            refuse(f"raw row {raw_key_text(key)} is both relabelled and renamed — fix the map")
            continue
        new_key = (key[0], new_home, new_away)
        if new_key in state.raw:
            refuse(
                f"raw row {raw_key_text(key)} decodes to {raw_key_text(new_key)}, which already exists (a reload "
                "wrote it): delete one of the two by hand after comparing them"
            )
            continue
        p.rename_raw.append((key, (new_home, new_away)))
        for column, old, new in (("HomeTeam", home, new_home), ("AwayTeam", away, new_away)):
            if old != new:
                p.entries.append(_entry("rename_raw", "raw_football_data", raw_key_text(key), column, old, new))

    # Mis-decoded values nothing above repairs.
    left = 0
    for tid, team in state.teams.items():
        if tid not in renamed and (
            text_encoding.looks_mojibake(team["name"]) or text_encoding.looks_mojibake(team["normalized_name"])
        ):
            left += 1
        elif text_encoding.looks_mojibake(team.get("metadata_text")):
            left += 1
    for lg in state.leagues_by_id.values():
        left += int(text_encoding.looks_mojibake(lg["name"]) or text_encoding.looks_mojibake(lg["country"]))
    renamed_raw = {k for k, _ in p.rename_raw}
    for key, raw in state.raw.items():
        names_bad = text_encoding.looks_mojibake(raw["HomeTeam"]) or text_encoding.looks_mojibake(raw["AwayTeam"])
        other_bad = any(text_encoding.looks_mojibake(raw.get(c)) for c in ("Div", "league", "season"))
        left += int((names_bad and key not in renamed_raw) or other_bad)
    counts["mojibake_left"] = left

    counts["create_league"] = len(p.creates)
    counts["tag_league"] = len(p.tags)
    counts["relabel_raw"] = len(p.relabel_raw)
    counts["relabel_match"] = len(p.relabel_match)
    counts["relabel_team"] = len(p.relabel_team)
    counts["rename_team"] = len(p.rename_team)
    counts["rename_raw"] = len(p.rename_raw)
    counts["refuse"] = len(p.refusals)
    return p


def load_state(cur, map_rows: list[MapRow]) -> State:
    """Read everything plan() needs, in the caller's transaction."""
    cur.execute(LEAGUES_SQL)
    leagues = [dict(r) for r in cur.fetchall()]
    cur.execute(TEAMS_SQL)
    teams = {r["id"]: dict(r) for r in cur.fetchall()}
    cur.execute(RAW_SQL)
    raw = {(r["Date"], r["HomeTeam"], r["AwayTeam"]): dict(r) for r in cur.fetchall()}
    state = State(
        leagues_by_code={lg["code"]: lg for lg in leagues if lg["code"]},
        leagues_by_id={lg["id"]: lg for lg in leagues},
        leagues_by_name={(lg["name"], lg["country"]): lg for lg in leagues},
        teams=teams,
        team_by_norm={t["normalized_name"]: tid for tid, t in teams.items()},
        aliases=team_identity.load_alias_index(cur, "soccer"),
        raw=raw,
    )
    involved: set[str] = set()
    for row in map_rows:
        home_id, away_id = resolve_team(state, row.home), resolve_team(state, row.away)
        if not (home_id and away_id):
            continue
        lo, hi = mapped_kickoff_window(row)
        cur.execute(CANDIDATES_SQL, (home_id, away_id, lo, hi))
        state.candidates[row.raw_key] = [dict(r) for r in cur.fetchall()]
        involved |= {home_id, away_id}
    cur.execute(CROSS_COUNTRY_SQL)
    state.cross_country_now = {r["team_id"] for r in cur.fetchall()}
    ids = sorted(involved | state.cross_country_now)
    if ids:
        cur.execute(TEAM_SIDES_SQL, (ids, ids))
        for r in cur.fetchall():
            state.sides.setdefault(r["team_id"], []).append((r["match_id"], r["country"]))
    return state


def plan_line(counts: Counter) -> str:
    """The one line a wrapper parses: 'PLAN key=n ...'."""
    return "PLAN " + " ".join(f"{k}={counts[k]}" for k in PLAN_COUNTERS)


def parse_expect_plan(spec: str) -> dict[str, int]:
    """'create_league=2,relabel_raw=10,...' -> {counter: n}."""
    out: dict[str, int] = {}
    for part in (spec or "").split(","):
        key, sep, value = part.strip().partition("=")
        if not sep or key not in PLAN_COUNTERS or not value.isdigit():
            raise ValueError(f"--expect-plan: bad item {part!r} (want key=N, key one of {PLAN_COUNTERS})")
        out[key] = int(value)
    return out


def default_out_csv(kind: str = "dryrun") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base = os.environ.get("BACKUP_LOCAL_DIR", "/app/backups")
    return f"{base}/soccer_labels_{kind}_{stamp}.csv"


def _open_once(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return path.open("x", newline="", encoding="utf-8")
    except FileExistsError:
        raise FileExistsError(f"{path} exists — the audit files are write-once") from None


def write_csv(path: str, entries: list[dict]) -> None:
    with _open_once(Path(path)) as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(entries)


def reversal_path(csv_path: str) -> str:
    return str(Path(csv_path).with_suffix("")) + ".reversal.sql"


def reversal_sql(cur, p: Plan, league_ids: dict[str, str]) -> str:
    """SQL restoring every old value, newest change first, in one transaction. Each statement is guarded on the
    new value and raises when it changes no row (the row moved since the apply), so nothing is half reverted."""

    def guarded(sql: str, params: tuple, what: str) -> str:
        stmt = cur.mogrify(sql, params).decode()
        label = cur.mogrify("%s", (f"reversal: no row for {what}",)).decode()
        return f"DO $rev$ BEGIN {stmt}; IF NOT FOUND THEN RAISE EXCEPTION '%', {label}; END IF; END $rev$;"

    out = [
        "-- Reversal of a scripts/repair_soccer_labels.py apply. Review it, then on the VM host:",
        "--   docker compose cp <this file> postgres:/tmp/soccer_labels_reversal.sql",
        '--   docker compose exec -T postgres sh -c \'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" '
        '-d "$POSTGRES_DB" -f /tmp/soccer_labels_reversal.sql\'',
        "BEGIN;",
    ]
    for (date, home, away), (new_home, new_away) in reversed(p.rename_raw):
        out.append(
            guarded(
                'UPDATE raw_football_data SET "HomeTeam" = %s, "AwayTeam" = %s '
                'WHERE "Date" = %s AND "HomeTeam" = %s AND "AwayTeam" = %s',
                (home, away, date, new_home, new_away),
                f"raw {date} {new_home} v {new_away}",
            )
        )
    for tid, old_name, old_norm, new_name, new_norm in reversed(p.rename_team):
        out.append(
            guarded(
                "UPDATE teams SET name = %s, normalized_name = %s WHERE id = %s AND name = %s AND normalized_name = %s",
                (old_name, old_norm, tid, new_name, new_norm),
                f"team {tid}",
            )
        )
    for tid, prev_team_league, code in reversed(p.relabel_team):
        out.append(
            guarded(
                "UPDATE teams SET league_id = %s WHERE id = %s AND league_id = %s",
                (prev_team_league, tid, league_ids[code]),
                f"team {tid}",
            )
        )
    for row, match, prev_league, _new in reversed(p.relabel_match):
        out.append(
            guarded(
                "UPDATE matches SET league_id = %s WHERE id = %s AND league_id = %s",
                (prev_league, match["id"], league_ids[row.to_league]),
                f"match {match['id']}",
            )
        )
    for row, raw in reversed(p.relabel_raw):
        out.append(
            guarded(
                'UPDATE raw_football_data SET league = %s, "Div" = %s '
                'WHERE "Date" = %s AND "HomeTeam" = %s AND "AwayTeam" = %s AND league = %s',
                (raw["league"], raw.get("Div"), row.date, row.home, row.away, row.to_league),
                f"raw {row.label}",
            )
        )
    for code, lid in reversed(p.tags):
        out.append(
            guarded(
                "UPDATE leagues SET external_ids = external_ids - 'football_data' "
                "WHERE id = %s AND external_ids->>'football_data' = %s",
                (lid, code),
                f"league {lid} tagged {code}",
            )
        )
    for code in reversed(p.creates):
        out.append(
            guarded(
                "DELETE FROM leagues l WHERE l.id = %s AND NOT EXISTS (SELECT 1 FROM matches WHERE league_id = l.id) "
                "AND NOT EXISTS (SELECT 1 FROM teams WHERE league_id = l.id)",
                (league_ids[code],),
                f"league {code} (or a row still references it)",
            )
        )
    out.append("COMMIT;")
    return "\n".join(out) + "\n"


def _one(cur, sql: str, params: tuple, what: str) -> None:
    cur.execute(sql, params)
    if cur.rowcount != 1:
        raise RuntimeError(f"{what}: {cur.rowcount} row(s) changed, expected 1 — rolled back, nothing written")


def create_leagues(cur, p: Plan) -> dict[str, str]:
    """{code: id} of the leagues the plan creates or tags. A created league is the row promote_raw.upsert_leagues
    inserts (LEAGUE_MAP name and country, sport soccer, external_ids {football_data: code}), but an INSERT that meets
    an existing (name, country, sport) row raises instead of merging into it: the reversal would drop a row this
    never created. A tag is guarded on the row still having no football_data code."""
    out: dict[str, str] = {}
    for code in p.creates:
        name, country = pr.LEAGUE_MAP[code]
        cur.execute(
            "INSERT INTO leagues (name, country, sport, external_ids) VALUES (%s, %s, 'soccer', %s) "
            "ON CONFLICT (name, country, sport) DO NOTHING RETURNING id::text AS id",
            (name, country, Json({"football_data": code})),
        )
        got = cur.fetchone()
        if got is None:
            raise RuntimeError(
                f"league {code} ({name!r}, {country}) exists since the plan — not created, rolled back, nothing written"
            )
        out[code] = got["id"]
    for code, lid in p.tags:
        _one(
            cur,
            "UPDATE leagues SET external_ids = COALESCE(external_ids, '{}'::jsonb) || %s "
            "WHERE id = %s AND sport = 'soccer' AND NOT (COALESCE(external_ids, '{}'::jsonb) ? 'football_data')",
            (Json({"football_data": code}), lid),
            f"league {lid} (tag {code})",
        )
        out[code] = lid
    return out


def apply_plan(cur, p: Plan, league_ids: dict[str, str]) -> None:
    """Every write of the plan, each guarded on the value it replaces."""
    for row, _raw in p.relabel_raw:
        _one(
            cur,
            'UPDATE raw_football_data SET league = %s, "Div" = %s '
            'WHERE "Date" = %s AND "HomeTeam" = %s AND "AwayTeam" = %s AND league = %s AND season = %s',
            (row.to_league, row.to_league, row.date, row.home, row.away, row.from_league, row.season),
            f"raw row {row.label}",
        )
    for row, match, old_league, _new in p.relabel_match:
        _one(
            cur,
            "UPDATE matches SET league_id = %s WHERE id = %s AND league_id = %s",
            (league_ids[row.to_league], match["id"], old_league),
            f"matches row {match['id']} ({row.label})",
        )
    for tid, old_team_league, code in p.relabel_team:
        _one(
            cur,
            "UPDATE teams SET league_id = %s WHERE id = %s AND league_id IS NOT DISTINCT FROM %s",
            (league_ids[code], tid, old_team_league),
            f"team {tid}",
        )
    for tid, old_name, old_norm, new_name, new_norm in p.rename_team:
        _one(
            cur,
            "UPDATE teams SET name = %s, normalized_name = %s WHERE id = %s AND name = %s AND normalized_name = %s",
            (new_name, new_norm, tid, old_name, old_norm),
            f"team {tid} {old_name!r}",
        )
    for (date, home, away), (new_home, new_away) in p.rename_raw:
        _one(
            cur,
            'UPDATE raw_football_data SET "HomeTeam" = %s, "AwayTeam" = %s '
            'WHERE "Date" = %s AND "HomeTeam" = %s AND "AwayTeam" = %s',
            (new_home, new_away, date, home, away),
            f"raw row {date} {home!r} v {away!r}",
        )


def log_plan(p: Plan) -> None:
    for code in p.creates:
        logger.info("create_league  %s %s", code, pr.LEAGUE_MAP[code])
    for code, lid in p.tags:
        logger.info("tag_league     %s %s: existing row %s gets the football_data code", code, pr.LEAGUE_MAP[code], lid)
    for row, _raw in p.relabel_raw:
        logger.info("relabel_raw    %s", row.label)
    for row, match, _prev_league, _new in p.relabel_match:
        logger.info("relabel_match  %s %s at %s", match["id"], row.label, match["match_date"])
    for tid, _prev, code in p.relabel_team:
        logger.info("relabel_team   %s -> %s", tid, code)
    for tid, old_name, _on, new_name, _nn in p.rename_team:
        logger.info("rename_team    %s %r -> %r", tid, old_name, new_name)
    if p.rename_raw:
        logger.info(
            "rename_raw     %d raw row(s), e.g. %s -> %s", len(p.rename_raw), p.rename_raw[0][0], p.rename_raw[0][1]
        )
    for msg in p.refusals:
        logger.error("REFUSED: %s", msg)
    if p.counts["mojibake_left"]:
        logger.error("%d mis-decoded value(s) this tool does not repair are left", p.counts["mojibake_left"])
    if p.cross_country_names:
        logger.warning(
            "soccer teams whose domestic matches would still span two countries: %s", ", ".join(p.cross_country_names)
        )


def emit(p: Plan) -> None:
    print(plan_line(p.counts), flush=True)
    print("PLAN_CROSS_COUNTRY names=" + (",".join(p.cross_country_names) or "-"), flush=True)


def void_files(paths: list[Path]) -> None:
    """Rename audit files of a rolled-back apply to *.void: they describe changes that never happened."""
    for path in paths:
        try:
            path.rename(path.with_name(path.name + ".void"))
            logger.error("VOID: %s.void — the apply rolled back, nothing it lists was written", path)
        except OSError as exc:
            logger.error("could not rename %s to *.void (%s): it describes changes that were rolled back", path, exc)


def run(conn, apply_changes: bool, out_csv: Optional[str], expect: Optional[dict[str, int]], map_rows) -> Plan:
    written: list[Path] = []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        state = load_state(cur, map_rows)
        p = plan(state, map_rows)
        log_plan(p)
        emit(p)
        if not apply_changes:
            conn.rollback()
            path = out_csv or default_out_csv("dryrun")
            write_csv(path, p.entries)
            logger.info("DRY-RUN: audit CSV %s; re-run with --apply to write", path)
            return p
        try:
            if expect is not None:
                got = {k: p.counts[k] for k in expect}
                if got != expect:
                    raise PlanDrift(f"plan {got} differs from --expect-plan {expect} — nothing written")
            if p.refusals or p.counts["mojibake_left"]:
                raise Refused(
                    f"{len(p.refusals)} refusal(s), {p.counts['mojibake_left']} mis-decoded value(s) left — nothing "
                    "written"
                )
            league_ids = {code: lg["id"] for code, lg in state.leagues_by_code.items()}
            league_ids.update(create_leagues(cur, p))
            path = out_csv or default_out_csv("apply")
            entries = [
                {**e, "new_value": league_ids[e["new_value"][5:-1]]} if e["new_value"].startswith("(new ") else e
                for e in p.entries
            ]
            write_csv(path, entries)
            written.append(Path(path))
            rev = reversal_path(path)
            with _open_once(Path(rev)) as fh:
                written.append(Path(rev))
                fh.write(reversal_sql(cur, p, league_ids))
            logger.info("audit CSV %s, reversal SQL %s", path, rev)
            apply_plan(cur, p, league_ids)
            after = plan(load_state(cur, map_rows), map_rows)
            if any(after.counts[k] for k in WRITE_COUNTERS) or after.refusals or after.counts["mojibake_left"]:
                raise RuntimeError(f"the re-plan inside the transaction is not clean ({plan_line(after.counts)})")
        except Exception:
            conn.rollback()
            void_files(written)
            raise
    # A COMMIT that raises leaves the audit files as they are: whether it landed is unknown (--verify tells).
    conn.commit()
    logger.info("Applied: %s", {k: p.counts[k] for k in WRITE_COUNTERS})
    return p


def verify(conn, csv_path: str) -> Counter:
    """Read-only: does every row of an apply CSV hold its new value?"""
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    counts: Counter = Counter(rows=len(rows))
    renamed: dict[str, dict[str, str]] = {}
    for r in rows:
        if r["kind"] == "rename_raw":
            renamed.setdefault(r["row_key"], {})[r["column"]] = r["new_value"]
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        for r in rows:
            now: object = "__missing__"
            if r["table"] == "leagues" and r["column"] == "football_data":
                cur.execute("SELECT external_ids->>'football_data' AS v FROM leagues WHERE id = %s", (r["row_key"],))
            elif r["table"] == "leagues":
                cur.execute(
                    "SELECT name || ' | ' || country || ' | ' || sport AS v FROM leagues "
                    "WHERE sport = 'soccer' AND external_ids->>'football_data' = %s",
                    (r["row_key"],),
                )
            elif r["table"] == "matches":
                cur.execute("SELECT league_id::text AS v FROM matches WHERE id = %s", (r["row_key"],))
            elif r["table"] == "teams":
                col = {"league_id": "league_id::text", "name": "name", "normalized_name": "normalized_name"}[
                    r["column"]
                ]
                cur.execute(f"SELECT {col} AS v FROM teams WHERE id = %s", (r["row_key"],))
            else:
                date, home, away = r["row_key"].split(" | ")
                new = renamed.get(r["row_key"], {})
                col = {"league": "league", "Div": '"Div"', "HomeTeam": '"HomeTeam"', "AwayTeam": '"AwayTeam"'}[
                    r["column"]
                ]
                cur.execute(
                    f"SELECT {col} AS v FROM raw_football_data "
                    'WHERE "Date" = %s AND "HomeTeam" = %s AND "AwayTeam" = %s',
                    (date, new.get("HomeTeam", home), new.get("AwayTeam", away)),
                )
            got = cur.fetchone()
            now = got["v"] if got else "__missing__"
            if now == "__missing__":
                counts["missing"] += 1
            elif (now or "") == r["new_value"]:
                counts["now_new"] += 1
            elif (now or "") == r["old_value"]:
                counts["still_old"] += 1
            else:
                counts["other"] += 1
    conn.rollback()
    print(
        "VERIFY " + " ".join(f"{k}={counts[k]}" for k in ("rows", "now_new", "still_old", "other", "missing")),
        flush=True,
    )
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="Write (default: dry run).")
    p.add_argument("--expect-plan", default=None, help="With --apply: the dry run's counts; any drift refuses.")
    p.add_argument("--verify", default=None, metavar="CSV", help="Read-only check of an apply CSV; writes nothing.")
    p.add_argument("--out-csv", default=None, help="Audit CSV (default $BACKUP_LOCAL_DIR or /app/backups, stamped).")
    p.add_argument("--map", default=None, help=f"The reviewed map (default {MAP_PATH}).")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    try:
        expect = parse_expect_plan(args.expect_plan) if args.expect_plan else None
        map_rows = load_map(args.map)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    if expect is not None and not args.apply:
        logger.error("--expect-plan pins an --apply; it means nothing on a dry run")
        return 2
    with psycopg2.connect(args.database_url) as conn:
        if args.verify:
            counts = verify(conn, args.verify)
            return 0 if counts["now_new"] == counts["rows"] else 1
        try:
            p = run(conn, args.apply, args.out_csv, expect, map_rows)
        except (PlanDrift, Refused) as exc:
            logger.error("%s", exc)
            return 1
    if p.refusals or p.counts["mojibake_left"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
