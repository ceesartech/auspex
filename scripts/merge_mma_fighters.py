"""Merge UFC fighters stored under two team rows into one row, and fold the
stale copies of their fought bouts into the finished rows.

THE BUG THIS REPAIRS
--------------------
MMA fighters are team rows (sport 'mma', one row per fighter), and
fetch_upcoming resolves an ESPN competitor by name: an individual sport's ESPN
athlete id is not read (_competitor_espn_id returns None for one). When ESPN
changed a fighter's display name ('Zachary Reese' -> 'Zach Reese', 'Cristian
Perez' -> 'Cristian Pérez', 'Ce Liu' -> 'Liu Ce'), the next sweep found no row
of that name and inserted a second one. The bout scheduled under the old name
stayed 'scheduled' beside the finished row the results sweep wrote under the
new name, and the fighter's history split across two rows (rolling form and
head-to-head read one of them). Prod 2026-11-01 (read only, every pair checked
against ESPN athlete ids): 9 fighters on two rows, 8 of them with a stale copy
of a fought bout; and one bout stored twice on the same two fighters the other
way round (McGhee v Romero, a reversed-listing fork, which fetch_upcoming
stops since 2026-11-01). Run B of scripts/ops/repair_stranded_recs_mma.sh
(repair_fixture_orphans) matches a twin on the same team ids in the same
order only, so it would delete those 9 stale copies as 'no twin', with the
bouts' pre-game prices.

THE LISTS (reviewed by hand; the tool touches nothing else)
-----------------------------------------------------------
  scripts/data/mma_fighter_merge_map.csv      one row per split fighter:
      merged_team_id (the row that goes) -> kept_team_id (the row carrying
      ESPN's current display name), the ESPN athlete id both rows' bouts
      carry, and the evidence.
  scripts/data/mma_reversed_fixture_folds.csv one row per bout stored twice
      the other way round on the same two fighter rows: the stale copy and the
      finished row.
And every MMA team name not in Unicode NFC form is NFC-normalized (prod: one,
'Rose Conceição', stored decomposed; ESPN still sends it decomposed, and
fetch_upcoming.normalize_team now composes what it compares and inserts).

WHAT A RUN DOES (ONE transaction: any failure rolls all of it back)
-------------------------------------------------------------------
1. Fold the twin fixtures first. Re-pointing a merged fighter's bouts onto
   the kept row puts some of them beside a row of the same bout: the same two
   fighters (either order) within TWIN_WINDOW. Each such cluster keeps its ONE
   finished row and folds every stale copy into it ('scheduled' or
   'cancelled', kickoff more than STALE_AFTER old: the 3-day results sweep can
   no longer write it), then deletes the copy:
     odds            the freshest pre-game price per current-price key (book,
                     market, selection, line, opening, live) the finished row
                     lacks; in-play rows never move;
     odds_snapshots  pre-game captures the finished row lacks;
     predictions     per (model, version, type) the latest one written before
                     kickoff, when the finished row has none under that key
                     and no rec references it.
   "Pre-game" is before the earlier of the two kickoffs. A copy listed the
   other way round moves its odds and snapshots with the selection swapped
   (home <-> away: a moneyline price belongs to a fighter, not to a slot; any
   other market refuses), and moves NO prediction: a prediction was computed
   from features in its own row's order and the model is not symmetric, so it
   is archived and deleted with the copy (counted predictions_dropped). The
   copy's ESPN competition id moves to the finished row when that row has none,
   both share the kickoff and no other row holds it. Everything else on the
   copy (later predictions, features_cache) goes with it, archived first.
   A cluster with no finished row is left as it is (run B judges those rows);
   two finished rows, a row in play, or a copy not yet stale refuse the pair.
   A copy that a recommendation or a betting_history row references (directly
   or through one of its predictions) is never folded: it is LEFT in place,
   re-pointed like any other row, and reported (left_rec_held; the wrapper
   pins 0). A settled rec is never re-pointed by this tool.
2. Re-point every other bout of the merged row: matches.home_team_id /
   away_team_id -> the kept row, each UPDATE guarded on the row's observed
   teams, kickoff and status. A re-point that would collide on UNIQUE
   (home_team_id, away_team_id, match_date) refuses the pair.
3. Record the merge on the kept row: the merged spelling joins
   metadata.aliases (team_identity's contract: fetch_upcoming.find_team and
   fetch_live_odds.find_match_id read it) and metadata.merged_from gets
   {team_id, name, normalized_name, espn_athlete_id, merged_at}. The ESPN
   athlete id is stamped on the kept row (external_ids.espn, through
   team_identity.stamp_espn_id's guard: the row holds none and no other MMA
   row holds it). Today's MMA resolver does not read it (see above); it is the
   identity a future athlete-id resolver keys on, and the record of the merge.
4. Delete the merged row (guarded: same name, nothing references it under any
   foreign key to teams; matches -> teams cascades, so this runs after the
   re-point and a reference check).
5. NFC-normalize the planned names (guarded on the observed spelling and on
   no other row holding the composed normalized_name).
6. Before COMMIT, the deployed resolver must resolve every merged spelling
   (as ESPN may send it, composed and decomposed) and every kept name to the
   kept row, and every normalized name (both forms) to its row, inside this
   transaction: fetch_upcoming.find_team(cur, SPORT_CONFIGS['mma'], ...).
   Deleting the 'zachary reese' row frees that name: had the resolver missed
   it, the next 15-minute sweep would insert the fighter again (the soccer
   lesson, audit §7 2026-10-26). Then --check-after's after-state must hold
   exactly inside the transaction, or everything rolls back.
The dry run predicts the resolver from the database (no other row holds the
merged name, no other row claims the alias, the athlete id is on no third
row) and refuses a pair the resolver would fail.

SAFETY
------
Every foreign key to teams and matches is read from pg_constraint first; one
the tool does not know refuses the run (a delete could cascade into it
unarchived). The dry run (the default) runs in a READ ONLY transaction and
writes only its write-once audit CSV and plan file. --apply re-plans inside
its own REPEATABLE READ transaction after locking every team row and match
row it touches (FOR UPDATE), and refuses (nothing written, exit 2) unless that
plan has exactly the digest of --expect-plan <the dry run's plan file>: every
fold, every moved odds / snapshot / prediction id, every re-point, alias,
ESPN id and name. Before its first write it writes a write-once gzip JSONL
archive of every row it deletes or changes (the reversal record), and refuses
to delete a stale copy holding a child row the archive lacks. Every UPDATE and
DELETE is guarded on the observed values and must touch exactly the planned
number of rows. Re-runnable: a merged pair plans as 'done', a folded listed
bout as done, and a run with nothing left writes nothing.

--check-after <plan file> (read-only) checks the after-state the plan implies:
no merged team row or reference left, every merge recorded on its kept row
(alias, merged_from, ESPN athlete id on that row only), no team re-minted
under a merged spelling, every stale copy gone and its finished row holding
exactly the moved rows (selections swapped where planned) and the adopted
ESPN id, no stale row of a folded bout left within the window, every re-point
landed, every name NFC, and the resolver resolving every spelling to its kept
row. --after-later-steps (run B has deleted some re-pointed no-twin rows since)
reports repoints_missing instead of failing on it.

RUN ORDER (MMA, 2026-11-01; scripts/ops/soccer_repair_runbook.sh is
independent of these)
  1. scripts/ops/repair_stranded_recs_mma.sh                       (run A: the June results backfill)
  2. scripts/ops/merge_mma_fighters.sh                             (this tool; refuses unless A is done and verified)
  3. scripts/ops/repair_stranded_recs_mma.sh --retire-stale-rows   (run B, pinned to the post-merge state; refuses
                                                                    unless this merge is done and verified)

Usage (inside the api container; dry run is the default):
    python /app/scripts/merge_mma_fighters.py --plan-out /app/backups/merge_mma.plan.json
    python /app/scripts/merge_mma_fighters.py --apply --expect-plan /app/backups/merge_mma.plan.json
    python /app/scripts/merge_mma_fighters.py --check-after /app/backups/merge_mma.plan.json [--after-later-steps]
Afterwards (scripts/ops/merge_mma_fighters.sh runs them): grade_completed_matches.py
--match-id <id> for every row the plan lists under GRADE_AFTER (predictions
moved onto finished rows), then compute_features_mma.py --all-finished --force
and --days 14 --force (the kept fighters' histories changed).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("merge_mma_fighters")

SCRIPTS_DIR = Path(__file__).resolve().parent
MAP_PATH = SCRIPTS_DIR / "data" / "mma_fighter_merge_map.csv"
FOLDS_PATH = SCRIPTS_DIR / "data" / "mma_reversed_fixture_folds.csv"
MAP_FIELDS = ("merged_team_id", "merged_name", "kept_team_id", "kept_name", "espn_athlete_id", "method", "evidence")
FOLD_FIELDS = (
    "stale_match_id",
    "finished_match_id",
    "kickoff",
    "stale_home",
    "stale_away",
    "espn_competition_id",
    "evidence",
)
SPORT = "mma"
TWIN_WINDOW = timedelta(hours=36)  # one fight card; a fighter never fights twice inside it
STALE_AFTER = timedelta(hours=96)  # past the results sweep's 3-day look-back (repair_fixture_orphans' floor)
LOCK_TIMEOUT = "10s"
FINISHED = "finished"
STALE_STATUSES = ("scheduled", "cancelled")
SWAP = {"home": "away", "away": "home"}
SWAPPABLE_MARKET = "moneyline"
ALIASES_KEY = "aliases"
MERGED_FROM_KEY = "merged_from"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# Every foreign key the tool knows, as (table, column). One that pg_constraint
# reports and that is not here refuses the run: a delete could cascade into it
# unarchived.
TEAM_REFERENCES = (
    ("matches", "home_team_id"),
    ("matches", "away_team_id"),
    ("match_stats", "team_id"),
    ("players", "team_id"),
    ("team_managers", "team_id"),
    ("transfers", "from_team_id"),
    ("transfers", "to_team_id"),
    ("nhl_match_stats", "team_id"),
    ("nhl_match_advanced_stats", "team_id"),
    ("nfl_players", "team_id"),
)
# Rows hanging off a match that a deleted stale copy takes with it (archived first).
MATCH_CHILDREN = (
    "odds",
    "odds_snapshots",
    "predictions",
    "features_cache",
    "match_stats",
    "match_weather",
    "weather",
    "player_match_stats",
    "nhl_match_stats",
    "nhl_match_advanced_stats",
    "nfl_injury_reports",
)
# Rows that make a match a rec holder: such a copy is left in place, never deleted.
MATCH_HOLDERS = ("betting_recommendations", "betting_history")

PLAN_KEYS = (
    "pairs_merge",
    "pairs_done",
    "pairs_refuse",
    "folds",
    "folds_reversed",
    "folds_done",
    "folds_refuse",
    "repoints",
    "left_rec_held",
    "odds_moved",
    "snapshots_moved",
    "predictions_moved",
    "predictions_dropped",
    "espn_ids_adopted",
    "athlete_ids_stamped",
    "nfc_renamed",
    "nfc_refuse",
)


class ListError(ValueError):
    """A reviewed list is malformed or unsafe to run."""


class GuardError(RuntimeError):
    """A guarded write found the database different from the plan."""


# ---------------------------------------------------------------------------
# The reviewed lists
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergePair:
    merged_team_id: str
    merged_name: str
    kept_team_id: str
    kept_name: str
    espn_athlete_id: str
    method: str
    evidence: str


@dataclass(frozen=True)
class ListedFold:
    stale_match_id: str
    finished_match_id: str
    kickoff: datetime
    stale_home: str
    stale_away: str
    espn_competition_id: str
    evidence: str


def _rows(path: Path, fields: tuple[str, ...]) -> list[dict]:
    """The data rows of a reviewed CSV ('#' lines are comments), every column
    present and every field filled."""
    with path.open(newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    missing = [f for f in fields if f not in (reader.fieldnames or [])]
    if missing:
        raise ListError(f"{path}: missing column(s) {', '.join(missing)}")
    out = []
    for n, row in enumerate(reader, start=2):
        vals = {f: (row.get(f) or "").strip() for f in fields}
        empty = [f for f, v in vals.items() if not v]
        if empty:
            raise ListError(f"{path}: data row {n - 1}: empty {', '.join(empty)} — only reviewed rows with evidence")
        out.append(vals)
    return out


def load_map(path: Path) -> list[MergePair]:
    """The fighter merge map, validated: ids are UUIDs, no team appears twice
    (a kept row is never also merged), names differ, athlete ids are digits
    and unique."""
    pairs = [MergePair(**r) for r in _rows(path, MAP_FIELDS)]
    seen: dict[str, str] = {}
    athletes: set[str] = set()
    for p in pairs:
        for tid in (p.merged_team_id, p.kept_team_id):
            if not UUID_RE.match(tid):
                raise ListError(f"{path}: {tid!r} is not a team id")
        if p.merged_team_id == p.kept_team_id or p.merged_name == p.kept_name:
            raise ListError(f"{path}: {p.merged_name!r} -> {p.kept_name!r} merges a row into itself")
        for tid, role in ((p.merged_team_id, "merged"), (p.kept_team_id, "kept")):
            if tid in seen:
                raise ListError(f"{path}: team {tid} is listed twice ({seen[tid]} and {role})")
            seen[tid] = role
        if not p.espn_athlete_id.isdigit() or p.espn_athlete_id in athletes:
            raise ListError(f"{path}: ESPN athlete id {p.espn_athlete_id!r} is not digits or is listed twice")
        athletes.add(p.espn_athlete_id)
    return pairs


def load_folds(path: Path) -> list[ListedFold]:
    """The reviewed reversed-copy list, validated: UUIDs, a zoned kickoff,
    no match listed twice."""
    out: list[ListedFold] = []
    seen: set[str] = set()
    for r in _rows(path, FOLD_FIELDS):
        for mid in (r["stale_match_id"], r["finished_match_id"]):
            if not UUID_RE.match(mid):
                raise ListError(f"{path}: {mid!r} is not a match id")
            if mid in seen:
                raise ListError(f"{path}: match {mid} is listed twice")
            seen.add(mid)
        try:
            kickoff = datetime.fromisoformat(r["kickoff"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ListError(f"{path}: kickoff {r['kickoff']!r} is not ISO-8601") from exc
        if kickoff.tzinfo is None:
            raise ListError(f"{path}: kickoff {r['kickoff']!r} has no time zone")
        out.append(ListedFold(**{**r, "kickoff": kickoff}))
    return out


# ---------------------------------------------------------------------------
# The deployed resolver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolvers:
    """The resolution code the pipeline runs (scripts/ is bind-mounted, so this
    is what the next sweep uses)."""

    normalize: Callable[[str], str]  # fetch_upcoming.normalize_team: teams.normalized_name
    alias_form: Callable[[str], str]
    find_team: Callable[[Any, str, Optional[str]], Optional[str]]  # cur, name, league_id
    find_team_by_alias: Callable[[Any, str], Optional[str]]  # cur, name
    stamp_espn_id: Callable[..., bool]


def load_resolvers(scripts_dir: Path = SCRIPTS_DIR) -> tuple[Optional[Resolvers], Optional[str]]:
    """(Resolvers, None), or (None, why not): an import error or a missing
    function is a refusal, never a crash."""
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    mods = {}
    api = {
        "team_identity": ("alias_form", "find_team_by_alias", "stamp_espn_id"),
        "fetch_upcoming": ("find_team", "normalize_team", "SPORT_CONFIGS"),
    }
    for name, attrs in api.items():
        try:
            mods[name] = importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - the refusal names it
            return None, f"cannot import scripts/{name}.py: {type(e).__name__}: {e}"
        missing = [a for a in attrs if not hasattr(mods[name], a)]
        if missing:
            return None, f"scripts/{name}.py lacks {', '.join(missing)}"
    fu, ti = mods["fetch_upcoming"], mods["team_identity"]
    cfg = fu.SPORT_CONFIGS.get(SPORT)
    if cfg is None:
        return None, "fetch_upcoming.SPORT_CONFIGS has no mma config"
    return (
        Resolvers(
            normalize=fu.normalize_team,
            alias_form=ti.alias_form,
            find_team=lambda cur, name, league_id: fu.find_team(cur, cfg, name, league_id, None),
            find_team_by_alias=lambda cur, name: ti.find_team_by_alias(cur, SPORT, name),
            stamp_espn_id=ti.stamp_espn_id,
        ),
        None,
    )


def spellings(name: str) -> list[str]:
    """`name` as ESPN may send it: composed (NFC) and decomposed (NFD), once each."""
    out: list[str] = []
    for s in (unicodedata.normalize("NFC", name), unicodedata.normalize("NFD", name)):
        if s not in out:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

FOREIGN_KEYS_SQL = """
    SELECT c.conrelid::regclass::text AS tbl, a.attname AS col, c.confrelid::regclass::text AS ref
      FROM pg_constraint c
      CROSS JOIN LATERAL unnest(c.conkey) AS k(attnum)
      JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
     WHERE c.contype = 'f' AND c.confrelid IN ('teams'::regclass, 'matches'::regclass)
"""


@dataclass(frozen=True)
class Schema:
    team_refs: tuple[tuple[str, str], ...]
    match_children: tuple[str, ...]


def read_schema(rows: Iterable[dict]) -> tuple[Optional[Schema], list[str]]:
    """(Schema, []) from FOREIGN_KEYS_SQL rows, or (None, the unknown keys). Pure."""
    known_team = set(TEAM_REFERENCES)
    known_match = set(MATCH_CHILDREN) | set(MATCH_HOLDERS)
    team_refs, children, unknown = set(), set(), []
    for r in rows:
        tbl, col, ref = r["tbl"], r["col"], r["ref"]
        if ref == "teams" and (tbl, col) in known_team:
            team_refs.add((tbl, col))
        elif ref == "matches" and col == "match_id" and tbl in known_match:
            if tbl in MATCH_CHILDREN:
                children.add(tbl)
        else:
            unknown.append(f"{tbl}.{col} -> {ref}")
    if unknown:
        return None, sorted(unknown)
    missing = {("matches", "home_team_id"), ("matches", "away_team_id")} - team_refs
    if missing:
        return None, [f"expected foreign key {t}.{c} -> teams is missing" for t, c in sorted(missing)]
    return (
        Schema(
            team_refs=tuple(t for t in TEAM_REFERENCES if t in team_refs),
            match_children=tuple(t for t in MATCH_CHILDREN if t in children),
        ),
        [],
    )


def team_refs_sql(schema: Schema) -> str:
    """References to one team (%(t)s) under every foreign key, one count per
    table. Names come from read_schema, i.e. from the TEAM_REFERENCES whitelist."""
    by_table: dict[str, list[str]] = defaultdict(list)
    for tbl, col in schema.team_refs:
        by_table[tbl].append(col)
    cols = [
        f'(SELECT count(*) FROM {tbl} x WHERE {" OR ".join(f"x.{c} = %(t)s::uuid" for c in cs)}) AS "{tbl}"'
        for tbl, cs in by_table.items()
    ]
    return "SELECT " + ", ".join(cols)


def team_delete_sql(schema: Schema) -> str:
    """The guarded team delete: same name, sport mma, and nothing references it."""
    by_table: dict[str, list[str]] = defaultdict(list)
    for tbl, col in schema.team_refs:
        by_table[tbl].append(col)
    guards = "\n".join(
        f"       AND NOT EXISTS (SELECT 1 FROM {tbl} x WHERE {' OR '.join(f'x.{c} = t.id' for c in cs)})"
        for tbl, cs in by_table.items()
    )
    return (
        "DELETE FROM teams t\n"
        "     WHERE t.id = %(id)s::uuid AND t.sport = 'mma' AND t.name = %(name)s\n"
        "       AND t.normalized_name = %(normalized_name)s\n" + guards
    )


def children_sql(schema: Schema) -> str:
    """Every row hanging off one match (%(m)s), per child table (whitelisted names)."""
    return "\nUNION ALL\n".join(
        f"SELECT '{t}' AS tbl, id::text AS id FROM {t} WHERE match_id = %(m)s::uuid" for t in schema.match_children
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class Deps:
    odds: list[dict] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)
    predictions: list[dict] = field(default_factory=list)
    recs: int = 0
    history: int = 0
    pred_recs: int = 0

    @property
    def holds_recs(self) -> bool:
        return bool(self.recs or self.history or self.pred_recs)


@dataclass
class State:
    teams: dict[str, dict]  # the listed rows (merged, kept), by id
    sport_teams: list[dict]  # every MMA team: id, name, normalized_name, espn, aliases
    matches: dict[str, dict]  # every match of a listed fighter, and the listed folds' rows
    deps: dict[str, Deps]
    espn_holders: dict[str, set[str]]  # ESPN competition id -> match ids holding it
    team_refs: dict[str, dict[str, int]]  # merged team id -> table -> count (matches excluded)


TEAMS_SQL = """
    SELECT id::text AS id, name, normalized_name, sport, league_id::text AS league_id, external_ids, metadata
      FROM teams WHERE id = ANY(%s::uuid[])
"""
SPORT_TEAMS_SQL = """
    SELECT id::text AS id, name, normalized_name, COALESCE(external_ids->>'espn', '') AS espn,
           metadata->'aliases' AS aliases
      FROM teams WHERE sport = 'mma'
"""
MATCHES_SQL = """
    SELECT m.id::text AS id, m.league_id::text AS league_id,
           m.home_team_id::text AS home_team_id, m.away_team_id::text AS away_team_id,
           m.match_date, m.status, m.home_score, m.away_score, COALESCE(m.external_ids->>'espn', '') AS espn
      FROM matches m
     WHERE m.home_team_id = ANY(%(t)s::uuid[]) OR m.away_team_id = ANY(%(t)s::uuid[]) OR m.id = ANY(%(m)s::uuid[])
"""
ODDS_SQL = """
    SELECT id::text AS id, match_id::text AS match_id, bookmaker, market_type, selection, line,
           is_opening, is_live, "timestamp", created_at
      FROM odds WHERE match_id = ANY(%s::uuid[])
"""
SNAPSHOTS_SQL = """
    SELECT id::text AS id, match_id::text AS match_id, bookmaker, market_type, selection, line, captured_at
      FROM odds_snapshots WHERE match_id = ANY(%s::uuid[])
"""
PREDICTIONS_SQL = """
    SELECT p.id::text AS id, p.match_id::text AS match_id, p.model_name, p.model_version, p.prediction_type,
           p.created_at, p.updated_at,
           EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.prediction_id = p.id) AS rec_referenced
      FROM predictions p WHERE p.match_id = ANY(%s::uuid[])
"""
HOLDERS_SQL = """
    SELECT m.id::text AS id,
           (SELECT count(*) FROM betting_recommendations r WHERE r.match_id = m.id) AS recs,
           (SELECT count(*) FROM betting_history h WHERE h.match_id = m.id) AS history,
           (SELECT count(*) FROM betting_recommendations r JOIN predictions p ON p.id = r.prediction_id
             WHERE p.match_id = m.id) AS pred_recs
      FROM matches m WHERE m.id = ANY(%s::uuid[])
"""
ESPN_HOLDERS_SQL = (
    "SELECT id::text AS id, external_ids->>'espn' AS espn FROM matches WHERE external_ids->>'espn' = ANY(%s)"
)


def listed_team_ids(pairs: list[MergePair]) -> list[str]:
    return sorted({t for p in pairs for t in (p.merged_team_id, p.kept_team_id)})


def load_schema(cur) -> tuple[Optional[Schema], list[str]]:
    cur.execute(FOREIGN_KEYS_SQL)
    return read_schema(cur.fetchall())


def load_state(cur, pairs: list[MergePair], folds: list[ListedFold], schema: Schema, lock: bool = False) -> State:
    """Everything the plan reads. lock=True (the apply) takes FOR UPDATE on every
    team row and match row the run can touch, in id order."""
    team_ids = listed_team_ids(pairs)
    fold_ids = sorted({m for f in folds for m in (f.stale_match_id, f.finished_match_id)})
    suffix = " ORDER BY id FOR UPDATE" if lock else ""
    cur.execute(TEAMS_SQL + suffix, (team_ids,))
    teams = {r["id"]: dict(r) for r in cur.fetchall()}
    cur.execute(SPORT_TEAMS_SQL)
    sport_teams = [dict(r) for r in cur.fetchall()]
    cur.execute(MATCHES_SQL + (" ORDER BY m.id FOR UPDATE OF m" if lock else ""), {"t": team_ids, "m": fold_ids})
    matches = {r["id"]: dict(r) for r in cur.fetchall()}
    ids = sorted(matches)
    deps: dict[str, Deps] = {mid: Deps() for mid in ids}
    if ids:
        cur.execute(ODDS_SQL, (ids,))
        for r in cur.fetchall():
            deps[r["match_id"]].odds.append(dict(r))
        cur.execute(SNAPSHOTS_SQL, (ids,))
        for r in cur.fetchall():
            deps[r["match_id"]].snapshots.append(dict(r))
        cur.execute(PREDICTIONS_SQL, (ids,))
        for r in cur.fetchall():
            deps[r["match_id"]].predictions.append(dict(r))
        cur.execute(HOLDERS_SQL, (ids,))
        for r in cur.fetchall():
            d = deps[r["id"]]
            d.recs, d.history, d.pred_recs = int(r["recs"]), int(r["history"]), int(r["pred_recs"])
    espn_ids = sorted({m["espn"] for m in matches.values() if m["espn"]})
    holders: dict[str, set[str]] = defaultdict(set)
    if espn_ids:
        cur.execute(ESPN_HOLDERS_SQL, (espn_ids,))
        for r in cur.fetchall():
            holders[r["espn"]].add(r["id"])
    refs: dict[str, dict[str, int]] = {}
    sql = team_refs_sql(schema)
    for p in pairs:
        if p.merged_team_id in teams:
            cur.execute(sql, {"t": p.merged_team_id})
            refs[p.merged_team_id] = {k: int(v) for k, v in dict(cur.fetchone()).items() if k != "matches" and v}
    return State(teams, sport_teams, matches, deps, dict(holders), refs)


# ---------------------------------------------------------------------------
# Planning (pure: state in, plan out)
# ---------------------------------------------------------------------------


@dataclass
class Fold:
    stale_id: str
    keep_id: str
    reversed: bool
    cutoff: datetime
    source: str  # 'pair <i>' | 'listed <i>'
    stale: dict  # the copy as observed
    keep: dict  # the finished row as observed
    odds: list[list[str]] = field(default_factory=list)  # [odds id, selection after the move]
    snapshots: list[list[str]] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)
    predictions_dropped: int = 0
    adopt_espn: str = ""
    keep_teams: list[str] = field(default_factory=list)  # the finished row's fighters once the merge is done

    def doc(self) -> dict:
        return {
            "stale_id": self.stale_id,
            "keep_id": self.keep_id,
            "reversed": self.reversed,
            "cutoff": self.cutoff.isoformat(),
            "odds": sorted(self.odds),
            "snapshots": sorted(self.snapshots),
            "predictions": sorted(self.predictions),
            "predictions_dropped": self.predictions_dropped,
            "adopt_espn": self.adopt_espn,
            "keep_teams": self.keep_teams or [self.keep["home_team_id"], self.keep["away_team_id"]],
            "keep_kickoff": self.keep["match_date"].isoformat(),
        }


@dataclass
class Repoint:
    match_id: str
    side: str  # 'home' | 'away'
    merged_id: str
    kept_id: str
    observed: dict  # home_team_id, away_team_id, match_date, status before THIS update


@dataclass
class PairPlan:
    index: int
    pair: MergePair
    action: str  # 'merge' | 'done' | 'refuse'
    reason: str = ""
    merged_team: dict = field(default_factory=dict)
    kept_team: dict = field(default_factory=dict)
    folds: list[Fold] = field(default_factory=list)
    repoints: list[Repoint] = field(default_factory=list)
    left: list[str] = field(default_factory=list)  # stale copies a rec holds: left in place
    team_patch: dict = field(default_factory=dict)


@dataclass
class ListedPlan:
    index: int
    listed: ListedFold
    action: str  # 'fold' | 'done' | 'refuse'
    reason: str = ""
    fold: Optional[Fold] = None


@dataclass
class NfcRename:
    team_id: str
    name: str
    normalized_name: str
    new_name: str
    new_normalized_name: str


@dataclass
class RunPlan:
    pairs: list[PairPlan]
    listed: list[ListedPlan]
    nfc: list[NfcRename]
    nfc_refused: list[str]

    def folds(self) -> list[Fold]:
        out = [f for p in self.pairs if p.action == "merge" for f in p.folds]
        return out + [lp.fold for lp in self.listed if lp.action == "fold" and lp.fold is not None]


def _team_espn(t: Optional[dict]) -> str:
    return str(((t or {}).get("external_ids") or {}).get("espn") or "")


def _aliases(t: Optional[dict]) -> Optional[list[str]]:
    """metadata.aliases as a list of strings ([] when absent); None when malformed."""
    v = ((t or {}).get("metadata") or {}).get(ALIASES_KEY)
    if v is None:
        return []
    if isinstance(v, list) and all(isinstance(a, str) for a in v):
        return list(v)
    return None


def _merged_from(t: Optional[dict]) -> Optional[list[dict]]:
    v = ((t or {}).get("metadata") or {}).get(MERGED_FROM_KEY)
    if v is None:
        return []
    if isinstance(v, list) and all(isinstance(x, dict) for x in v):
        return list(v)
    return None


def _written(p: dict) -> datetime:
    """When a prediction's content was last written: COALESCE(updated_at, created_at)."""
    return p.get("updated_at") or p["created_at"]


def _line(v: Any) -> Decimal:
    return Decimal(-1) if v is None else Decimal(str(v))


def _price_key(o: dict, selection: str) -> tuple:
    return (
        o["bookmaker"],
        o["market_type"],
        selection,
        _line(o.get("line")),
        bool(o.get("is_opening")),
        bool(o.get("is_live")),
    )


def _snap_key(s: dict, selection: str) -> tuple:
    return (s["bookmaker"], s["market_type"], selection, _line(s.get("line")), s["captured_at"])


def _pred_key(p: dict) -> tuple:
    return (p["model_name"], p.get("model_version"), p["prediction_type"])


def _selection_after(row: dict, reverse: bool) -> Optional[str]:
    """The selection a moved price carries on the finished row; None when a
    reversed copy holds a price this tool cannot re-orient."""
    if not reverse:
        return row["selection"]
    if row.get("market_type") != SWAPPABLE_MARKET or row.get("selection") not in SWAP:
        return None
    return SWAP[row["selection"]]


def plan_fold(
    stale: dict, keep: dict, deps: dict[str, Deps], reverse: bool, source: str, espn_holders: dict[str, set[str]]
) -> Fold | str:
    """What moves from a stale copy onto its finished row (a Fold), or why it
    cannot be folded (a refusal string). Mutates deps[keep] so a second copy of
    the same bout sees what the first one moved. Pure apart from that."""
    cutoff = min(stale["match_date"], keep["match_date"])
    sd, kd = deps.get(stale["id"], Deps()), deps.setdefault(keep["id"], Deps())
    held = {_price_key(o, o["selection"]) for o in kd.odds}
    freshest: dict[tuple, tuple[dict, str]] = {}
    for o in sd.odds:
        if o.get("is_live") or o.get("timestamp") is None or o["timestamp"] >= cutoff:
            continue
        sel = _selection_after(o, reverse)
        if sel is None:
            return f"the reversed copy {stale['id']} holds a {o.get('market_type')}/{o.get('selection')} price"
        k = _price_key(o, sel)
        best = freshest.get(k)
        rank = (o["timestamp"], o.get("created_at") or o["timestamp"], o["id"])
        if best is None or rank > (
            best[0]["timestamp"],
            best[0].get("created_at") or best[0]["timestamp"],
            best[0]["id"],
        ):
            freshest[k] = (o, sel)
    odds = sorted(([o["id"], sel] for k, (o, sel) in freshest.items() if k not in held), key=lambda x: x[0])
    have = {_snap_key(s, s["selection"]) for s in kd.snapshots}
    first: dict[tuple, tuple[dict, str]] = {}
    for s in sd.snapshots:
        if s["captured_at"] >= cutoff:
            continue
        sel = _selection_after(s, reverse)
        if sel is None:
            return f"the reversed copy {stale['id']} holds a {s.get('market_type')}/{s.get('selection')} snapshot"
        k = _snap_key(s, sel)
        if k not in first or s["id"] < first[k][0]["id"]:
            first[k] = (s, sel)
    snaps = sorted(([s["id"], sel] for k, (s, sel) in first.items() if k not in have), key=lambda x: x[0])
    preds: list[str] = []
    if not reverse:
        taken = {_pred_key(p) for p in kd.predictions}
        by_key: dict[tuple, list[dict]] = defaultdict(list)
        for p in sd.predictions:
            if _written(p) < cutoff and not p.get("rec_referenced"):
                by_key[_pred_key(p)].append(p)
        latest = {k: min(ps, key=lambda p: (-_written(p).timestamp(), p["id"])) for k, ps in by_key.items()}
        preds = sorted(p["id"] for k, p in latest.items() if k not in taken)
    adopt = ""
    if stale["espn"] and not keep["espn"] and stale["match_date"] == keep["match_date"]:
        if espn_holders.get(stale["espn"], set()) <= {stale["id"]}:
            adopt = stale["espn"]
    odds_sel, snap_sel = dict((i, sel) for i, sel in odds), dict((i, sel) for i, sel in snaps)
    kd.odds = kd.odds + [dict(o, selection=odds_sel[o["id"]]) for o in sd.odds if o["id"] in odds_sel]
    kd.snapshots = kd.snapshots + [dict(s, selection=snap_sel[s["id"]]) for s in sd.snapshots if s["id"] in snap_sel]
    kd.predictions = kd.predictions + [p for p in sd.predictions if p["id"] in preds]
    return Fold(
        stale_id=stale["id"],
        keep_id=keep["id"],
        reversed=reverse,
        cutoff=cutoff,
        source=source,
        stale=dict(stale),
        keep=dict(keep),
        odds=odds,
        snapshots=snaps,
        predictions=preds,
        predictions_dropped=len(sd.predictions) - len(preds),
        adopt_espn=adopt,
    )


def find_clusters(rows: list[dict], canon: dict[str, str], merged: set[str]) -> list[list[dict]]:
    """Rows of the same bout once each merged fighter reads as its kept row:
    the same two fighters in EITHER order within TWIN_WINDOW, grouped
    transitively; only groups holding a merged fighter's row and more than one
    row. Pure."""

    def c(t: str) -> str:
        return canon.get(t, t)

    by_pair: dict[frozenset, list[dict]] = defaultdict(list)
    for m in rows:
        by_pair[frozenset((c(m["home_team_id"]), c(m["away_team_id"])))].append(m)
    out = []
    for group in by_pair.values():
        group = sorted(group, key=lambda m: (m["match_date"], m["id"]))
        clusters: list[list[dict]] = []
        for m in group:
            if clusters and m["match_date"] - clusters[-1][-1]["match_date"] <= TWIN_WINDOW:
                clusters[-1].append(m)
            else:
                clusters.append([m])
        for cl in clusters:
            if len(cl) > 1 and any(t in merged for m in cl for t in (m["home_team_id"], m["away_team_id"])):
                out.append(cl)
    return sorted(out, key=lambda cl: (cl[0]["match_date"], cl[0]["id"]))


def assess_pair(p: MergePair, state: State, after_names: dict[str, set[str]], resolvers: Resolvers) -> tuple[str, str]:
    """('merge'|'done'|'refuse', reason) for one pair, before its fixtures are
    looked at. after_names: normalized_name -> team ids once the run is done
    (merged rows gone, NFC renames applied)."""
    merged, kept = state.teams.get(p.merged_team_id), state.teams.get(p.kept_team_id)
    if kept is None:
        return "refuse", f"kept team {p.kept_team_id} ({p.kept_name!r}) does not exist"
    if kept.get("sport") != SPORT or kept.get("name") != p.kept_name:
        return (
            "refuse",
            f"kept team {p.kept_team_id} is {kept.get('sport')} {kept.get('name')!r}, not mma {p.kept_name!r}",
        )
    if merged is None:
        recorded = _merged_from(kept) or []
        if any(str(x.get("team_id")) == p.merged_team_id for x in recorded):
            return "done", f"merged into {p.kept_name!r} already (metadata.merged_from)"
        return "refuse", f"merged team {p.merged_team_id} is gone but {p.kept_name!r} does not record merging it"
    if merged.get("sport") != SPORT or merged.get("name") != p.merged_name:
        return (
            "refuse",
            f"merged team {p.merged_team_id} is {merged.get('sport')} {merged.get('name')!r}, "
            f"not mma {p.merged_name!r}",
        )
    refs = state.team_refs.get(p.merged_team_id) or {}
    if refs:
        return "refuse", f"{p.merged_name!r} is referenced outside matches ({refs}); this tool moves matches only"
    if _aliases(kept) is None or _merged_from(kept) is None:
        return "refuse", f"{p.kept_name!r}'s metadata.aliases / merged_from is malformed"
    for label, t in (("kept", kept), ("merged", merged)):
        held = _team_espn(t)
        if held and held != p.espn_athlete_id:
            return "refuse", f"the {label} row holds ESPN id {held}, the map says {p.espn_athlete_id}"
    third = sorted(
        t["id"]
        for t in state.sport_teams
        if t["espn"] == p.espn_athlete_id and t["id"] not in (p.merged_team_id, p.kept_team_id)
    )
    if third:
        return "refuse", f"ESPN athlete id {p.espn_athlete_id} is on another MMA row ({', '.join(third)})"
    for s in spellings(p.merged_name):
        holders = after_names.get(resolvers.normalize(s), set()) - {p.kept_team_id}
        if holders:
            return (
                "refuse",
                f"after the merge {s!r} still names team(s) {sorted(holders)}: the resolver would pick them",
            )
    form = resolvers.alias_form(p.merged_name)
    claim = sorted(
        t["id"]
        for t in state.sport_teams
        if t["id"] not in (p.kept_team_id, p.merged_team_id)
        and isinstance(t.get("aliases"), list)
        and any(isinstance(a, str) and resolvers.alias_form(a) == form for a in t["aliases"])
    )
    if claim:
        return "refuse", f"alias {form!r} is already claimed by {claim}: it would resolve to neither"
    return "merge", ""


def plan_nfc(state: State, deleting: set[str], normalize: Callable[[str], str]) -> tuple[list[NfcRename], list[str]]:
    """Every MMA name not in NFC form (rows this run deletes excepted), and the
    refusals: a composed normalized_name another row already holds. Pure."""
    renames, refused = [], []
    norms = defaultdict(set)
    for t in state.sport_teams:
        if t["id"] not in deleting:
            norms[t["normalized_name"]].add(t["id"])
    for t in sorted(state.sport_teams, key=lambda t: t["id"]):
        if t["id"] in deleting:
            continue
        name, norm = t["name"], t["normalized_name"]
        new_name, new_norm = unicodedata.normalize("NFC", name), unicodedata.normalize("NFC", norm)
        if new_name == name and new_norm == norm:
            continue
        if new_norm != normalize(new_name):
            refused.append(f"team {t['id']} {name!r}: its normalized_name {norm!r} is not its name's")
            continue
        clash = norms.get(new_norm, set()) - {t["id"]}
        if clash:
            refused.append(f"team {t['id']} {name!r}: the composed name {new_norm!r} is held by {sorted(clash)}")
            continue
        renames.append(NfcRename(t["id"], name, norm, new_name, new_norm))
    return renames, refused


def plan_run(
    pairs: list[MergePair], listed: list[ListedFold], state: State, now: datetime, resolvers: Resolvers
) -> RunPlan:
    """The whole run, planned on one state. Pure."""
    nfc0, _ = plan_nfc(state, {p.merged_team_id for p in pairs}, resolvers.normalize)
    renamed = {r.team_id: r.new_normalized_name for r in nfc0}
    after_names: dict[str, set[str]] = defaultdict(set)
    merged_ids = {p.merged_team_id for p in pairs}
    for t in state.sport_teams:
        if t["id"] not in merged_ids:
            after_names[renamed.get(t["id"], t["normalized_name"])].add(t["id"])
    plans: list[PairPlan] = []
    for i, p in enumerate(pairs, start=1):
        action, reason = assess_pair(p, state, after_names, resolvers)
        plans.append(
            PairPlan(
                i,
                p,
                action,
                reason,
                merged_team=dict(state.teams.get(p.merged_team_id) or {}),
                kept_team=dict(state.teams.get(p.kept_team_id) or {}),
            )
        )
    merging = {pl.pair.merged_team_id: pl for pl in plans if pl.action == "merge"}
    canon = {mid: pl.pair.kept_team_id for mid, pl in merging.items()}
    kept_ids = {pl.pair.kept_team_id for pl in merging.values()}
    deps = {
        k: Deps(list(v.odds), list(v.snapshots), list(v.predictions), v.recs, v.history, v.pred_recs)
        for k, v in state.deps.items()
    }
    rows = [m for m in state.matches.values() if {m["home_team_id"], m["away_team_id"]} & (set(canon) | kept_ids)]
    deleted: set[str] = set()
    cluster_rows: set[str] = set()

    def owner(cl: list[dict]) -> PairPlan:
        ids = {t for m in cl for t in (m["home_team_id"], m["away_team_id"]) if t in merging}
        return min((merging[t] for t in ids), key=lambda pl: pl.index)

    for cl in find_clusters(rows, canon, set(canon)):
        pl = owner(cl)
        cluster_rows.update(m["id"] for m in cl)
        statuses = {m["status"] for m in cl}
        finished = [m for m in cl if m["status"] == FINISHED]
        tag = ", ".join(f"{m['id']} {m['status']} {m['match_date'].isoformat()}" for m in cl)
        if "live" in statuses:
            pl.action, pl.reason = "refuse", f"a row of the bout is in play ({tag})"
            continue
        if not finished:
            continue  # no fold: run B judges these rows (re-pointed below)
        if len(finished) > 1:
            pl.action, pl.reason = "refuse", f"two finished rows for one bout ({tag})"
            continue
        keep = finished[0]
        for d in cl:
            if d["id"] == keep["id"]:
                continue
            if d["status"] not in STALE_STATUSES:
                pl.action, pl.reason = "refuse", f"{d['id']} is {d['status']!r} beside the finished row {keep['id']}"
                break
            if d["match_date"] >= now - STALE_AFTER:
                pl.action, pl.reason = (
                    "refuse",
                    f"{d['id']} kicked off less than {STALE_AFTER} ago: the results sweep may still write it",
                )
                break
            if deps.get(d["id"], Deps()).holds_recs:
                pl.left.append(d["id"])
                continue
            keep_teams = [
                canon.get(keep["home_team_id"], keep["home_team_id"]),
                canon.get(keep["away_team_id"], keep["away_team_id"]),
            ]
            reverse = [
                canon.get(d["home_team_id"], d["home_team_id"]),
                canon.get(d["away_team_id"], d["away_team_id"]),
            ] != keep_teams
            fold = plan_fold(d, keep, deps, reverse, f"pair {pl.index}", state.espn_holders)
            if isinstance(fold, str):
                pl.action, pl.reason = "refuse", fold
                break
            fold.keep_teams = keep_teams
            pl.folds.append(fold)
            deleted.add(d["id"])
    listed_plans: list[ListedPlan] = []
    for i, lf in enumerate(listed, start=1):
        lp = ListedPlan(i, lf, "refuse")
        listed_plans.append(lp)
        stale, fin = state.matches.get(lf.stale_match_id), state.matches.get(lf.finished_match_id)
        if fin is None or fin["status"] != FINISHED:
            lp.reason = f"the finished row {lf.finished_match_id} is " + (
                "gone" if fin is None else repr(fin["status"])
            )
            continue
        if stale is None:
            lp.action, lp.reason = "done", "the stale copy is gone"
            continue
        if lf.stale_match_id in cluster_rows or lf.finished_match_id in cluster_rows:
            lp.reason = "a fighter merge folds this bout already"
            continue
        if (stale["home_team_id"], stale["away_team_id"]) != (fin["away_team_id"], fin["home_team_id"]):
            lp.reason = "the two rows are not the same two fighters the other way round"
            continue
        if stale["match_date"] != lf.kickoff or abs(stale["match_date"] - fin["match_date"]) > TWIN_WINDOW:
            lp.reason = (
                f"the stale copy kicks off at {stale['match_date'].isoformat()}, the list says {lf.kickoff.isoformat()}"
            )
            continue
        if stale["status"] not in STALE_STATUSES or stale["match_date"] >= now - STALE_AFTER:
            lp.reason = f"the stale copy is {stale['status']!r} at {stale['match_date'].isoformat()}: not a stale row"
            continue
        if deps.get(stale["id"], Deps()).holds_recs:
            lp.reason = "a rec or betting_history row references the stale copy: left in place"
            continue
        if stale["espn"] and stale["espn"] != lf.espn_competition_id:
            lp.reason = f"the stale copy carries ESPN id {stale['espn']}, the list says {lf.espn_competition_id}"
            continue
        fold = plan_fold(stale, fin, deps, True, f"listed {i}", state.espn_holders)
        if isinstance(fold, str):
            lp.reason = fold
            continue
        lp.action, lp.reason, lp.fold = "fold", "", fold
        deleted.add(stale["id"])
    # Re-point what stays, then check UNIQUE (home_team_id, away_team_id, match_date) over the result.
    final: dict[str, tuple[str, str, datetime]] = {}
    for m in sorted(rows, key=lambda m: m["id"]):
        if m["id"] in deleted:
            continue
        home: str = m["home_team_id"]
        away: str = m["away_team_id"]
        for side in ("home", "away"):
            tid = home if side == "home" else away
            if tid in merging and merging[tid].action == "merge":
                pl = merging[tid]
                pl.repoints.append(
                    Repoint(
                        m["id"],
                        side,
                        tid,
                        canon[tid],
                        {
                            "home_team_id": home,
                            "away_team_id": away,
                            "match_date": m["match_date"],
                            "status": m["status"],
                        },
                    )
                )
                if side == "home":
                    home = canon[tid]
                else:
                    away = canon[tid]
        final[m["id"]] = (home, away, m["match_date"])
    seen: dict[tuple, str] = {}
    for mid, key in sorted(final.items()):
        if key in seen:
            for tid in (
                state.matches[mid]["home_team_id"],
                state.matches[mid]["away_team_id"],
                state.matches[seen[key]]["home_team_id"],
                state.matches[seen[key]]["away_team_id"],
            ):
                if tid in merging and merging[tid].action == "merge":
                    merging[tid].action = "refuse"
                    merging[tid].reason = (
                        f"re-pointing {mid} would duplicate {seen[key]} (same fighters, same order, same kickoff)"
                    )
        seen.setdefault(key, mid)
    for pl in plans:
        if pl.action != "merge":
            pl.folds, pl.repoints, pl.left = [], [], []
            continue
        kept = pl.kept_team
        aliases = _aliases(kept) or []
        pl.team_patch = {
            "observed_aliases": aliases,
            "observed_merged_from": _merged_from(kept) or [],
            "observed_espn": _team_espn(kept),
            "aliases": aliases if pl.pair.merged_name in aliases else aliases + [pl.pair.merged_name],
            "espn": pl.pair.espn_athlete_id,
            "stamp": _team_espn(kept) != pl.pair.espn_athlete_id,
            "clear_merged_espn": _team_espn(pl.merged_team) == pl.pair.espn_athlete_id,
        }
    deleting = {pl.pair.merged_team_id for pl in plans if pl.action == "merge"}
    nfc, nfc_refused = plan_nfc(state, deleting, resolvers.normalize)
    return RunPlan(plans, listed_plans, nfc, nfc_refused)


def summarize(plan: RunPlan) -> dict[str, int]:
    c: Counter = Counter({k: 0 for k in PLAN_KEYS})
    for pl in plan.pairs:
        c[f"pairs_{pl.action}"] += 1
        if pl.action == "merge":
            c["repoints"] += len(pl.repoints)
            c["left_rec_held"] += len(pl.left)
            c["athlete_ids_stamped"] += int(bool(pl.team_patch.get("stamp")))
    for lp in plan.listed:
        if lp.action == "done":
            c["folds_done"] += 1
        elif lp.action == "refuse":
            c["folds_refuse"] += 1
    for f in plan.folds():
        c["folds"] += 1
        c["folds_reversed"] += int(f.reversed)
        c["odds_moved"] += len(f.odds)
        c["snapshots_moved"] += len(f.snapshots)
        c["predictions_moved"] += len(f.predictions)
        c["predictions_dropped"] += f.predictions_dropped
        c["espn_ids_adopted"] += int(bool(f.adopt_espn))
    c["nfc_renamed"] = len(plan.nfc)
    c["nfc_refuse"] = len(plan.nfc_refused)
    return {k: int(c[k]) for k in PLAN_KEYS}


def refusals(plan: RunPlan) -> list[str]:
    out = [
        f"pair {pl.index} {pl.pair.merged_name!r} -> {pl.pair.kept_name!r}: {pl.reason}"
        for pl in plan.pairs
        if pl.action == "refuse"
    ]
    out += [
        f"listed fold {lp.index} {lp.listed.stale_match_id}: {lp.reason}" for lp in plan.listed if lp.action == "refuse"
    ]
    return out + [f"NFC: {r}" for r in plan.nfc_refused]


def grade_after(plan: RunPlan) -> list[str]:
    """Finished rows that receive predictions: the grader grades them by --match-id."""
    return sorted({f.keep_id for f in plan.folds() if f.predictions})


def plan_line(totals: dict[str, int]) -> str:
    return "PLAN " + " ".join(f"{k}={totals[k]}" for k in PLAN_KEYS)


# ---------------------------------------------------------------------------
# Plan file, audit CSV, archive
# ---------------------------------------------------------------------------


def _jsonable(v: Any) -> Any:
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return str(v)
    if v is None or isinstance(v, (str, int, float, bool, list, dict)):
        return v
    return str(v)


def _j(v: Any) -> str:
    return json.dumps(v, default=_jsonable, sort_keys=True, ensure_ascii=False)


def plan_document(plan: RunPlan, mode: str) -> dict:
    """The plan as data: what --expect-plan compares (the digest covers every
    decision and every row id) and --check-after checks."""
    pairs = []
    for pl in plan.pairs:
        e: dict[str, Any] = {
            "index": pl.index,
            "merged_team_id": pl.pair.merged_team_id,
            "merged_name": pl.pair.merged_name,
            "kept_team_id": pl.pair.kept_team_id,
            "kept_name": pl.pair.kept_name,
            "espn_athlete_id": pl.pair.espn_athlete_id,
            "action": pl.action,
            "reason": pl.reason,
        }
        if pl.action == "merge":
            e.update(
                league_id=pl.kept_team.get("league_id") or pl.merged_team.get("league_id"),
                merged_normalized_name=pl.merged_team.get("normalized_name"),
                folds=[f.doc() for f in pl.folds],
                repoints=sorted([r.match_id, r.side] for r in pl.repoints),
                left=sorted(pl.left),
                aliases=pl.team_patch["aliases"],
                stamp=pl.team_patch["stamp"],
                clear_merged_espn=pl.team_patch["clear_merged_espn"],
            )
        pairs.append(e)
    listed = [
        {
            "index": lp.index,
            "stale_match_id": lp.listed.stale_match_id,
            "finished_match_id": lp.listed.finished_match_id,
            "action": lp.action,
            "reason": lp.reason,
            "fold": lp.fold.doc() if lp.fold else None,
        }
        for lp in plan.listed
    ]
    nfc = [
        {
            "team_id": r.team_id,
            "name": r.name,
            "normalized_name": r.normalized_name,
            "new_name": r.new_name,
            "new_normalized_name": r.new_normalized_name,
        }
        for r in plan.nfc
    ]
    structural = {
        "pairs": [{k: v for k, v in e.items() if k not in ("index", "reason")} for e in pairs],
        "listed": [{k: v for k, v in e.items() if k not in ("index", "reason")} for e in listed],
        "nfc": nfc,
    }
    return {
        "tool": "merge_mma_fighters",
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "digest": hashlib.sha256(_j(structural).encode()).hexdigest(),
        "summary": summarize(plan),
        "pairs": pairs,
        "listed": listed,
        "nfc": nfc,
        "nfc_refused": plan.nfc_refused,
        "grade_after": grade_after(plan),
    }


def plan_difference(expected: dict, doc: dict) -> Optional[str]:
    """None when `doc` is the plan `expected` (a dry run's plan file) showed,
    else the first difference found. Pure."""
    if expected.get("tool") != "merge_mma_fighters" or "digest" not in expected:
        return "the expected plan file is not a merge_mma_fighters plan"
    if expected["digest"] == doc["digest"]:
        return None
    if expected.get("summary") != doc["summary"]:
        return f"the counts differ: expected {expected.get('summary')}, this run plans {doc['summary']}"
    for section, key in (("pairs", "merged_team_id"), ("listed", "stale_match_id"), ("nfc", "team_id")):
        want = {e[key]: e for e in expected.get(section, [])}
        for e in doc[section]:
            w = want.get(e[key])
            if w is None:
                return f"{section} {e[key]} is not in the expected plan"
            for k, v in e.items():
                if k not in ("index", "reason") and w.get(k) != v:
                    return f"{section} {e[key]}: {k} differs from the expected plan"
    return "the plan digest differs from the expected plan"


CSV_FIELDS = ("unit", "action", "table", "row_id", "keep_id", "before", "after", "reason")


def csv_rows(plan: RunPlan) -> list[dict]:
    """One row per row the run touches (or per unit it refuses / finds done)."""
    out: list[dict] = []

    def add(unit: str, action: str, table: str, row_id: str, **kw: Any) -> None:
        out.append({"unit": unit, "action": action, "table": table, "row_id": row_id, **kw})

    def add_fold(unit: str, f: Fold) -> None:
        add(
            unit,
            "fold_stale_reversed" if f.reversed else "fold_stale",
            "matches",
            f.stale_id,
            keep_id=f.keep_id,
            before=_j(f.stale),
        )
        for table, rows in (("odds", f.odds), ("odds_snapshots", f.snapshots)):
            for rid, sel in rows:
                add(
                    unit,
                    "move_swapped" if f.reversed else "move",
                    table,
                    rid,
                    keep_id=f.keep_id,
                    after=_j({"match_id": f.keep_id, "selection": sel}),
                )
        for rid in f.predictions:
            add(unit, "move", "predictions", rid, keep_id=f.keep_id, after=_j({"match_id": f.keep_id}))
        if f.adopt_espn:
            add(unit, "adopt_espn_id", "matches", f.keep_id, before=_j(f.keep), after=_j({"espn": f.adopt_espn}))

    for pl in plan.pairs:
        unit = f"pair {pl.index} {pl.pair.merged_name} -> {pl.pair.kept_name}"
        if pl.action != "merge":
            add(unit, f"pair_{pl.action}", "teams", pl.pair.merged_team_id, reason=pl.reason)
            continue
        for f in pl.folds:
            add_fold(unit, f)
        for mid in pl.left:
            add(unit, "left_rec_held", "matches", mid, reason="a rec references this stale copy: left in place")
        for r in pl.repoints:
            add(
                unit,
                f"repoint_{r.side}",
                "matches",
                r.match_id,
                before=_j(r.observed),
                after=_j({f"{r.side}_team_id": r.kept_id}),
            )
        t = pl.team_patch
        add(
            unit,
            "record_merge",
            "teams",
            pl.pair.kept_team_id,
            before=_j({"aliases": t["observed_aliases"], "espn": t["observed_espn"]}),
            after=_j({"aliases": t["aliases"], "espn": t["espn"]}),
        )
        add(unit, "delete_team", "teams", pl.pair.merged_team_id, before=_j(pl.merged_team))
    for lp in plan.listed:
        unit = f"listed {lp.index} {lp.listed.stale_home} v {lp.listed.stale_away}"
        if lp.action == "fold" and lp.fold:
            add_fold(unit, lp.fold)
        else:
            add(unit, f"fold_{lp.action}", "matches", lp.listed.stale_match_id, reason=lp.reason)
    for rn in plan.nfc:
        add(
            "nfc",
            "nfc_rename",
            "teams",
            rn.team_id,
            before=_j([rn.name, rn.normalized_name]),
            after=_j([rn.new_name, rn.new_normalized_name]),
        )
    for why in plan.nfc_refused:
        add("nfc", "nfc_refuse", "teams", "", reason=why)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as fh:  # exclusive create: write-once
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_plan(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True, ensure_ascii=False)


def write_archive(cur, path: Path, plan: RunPlan, schema: Schema) -> tuple[int, dict[str, set[str]]]:
    """gzip JSONL of every row the run deletes or changes, written before the
    first write: team rows (merged, kept, renamed), each stale copy with every
    child row, each finished row a fold changes, each re-pointed row. Returns
    (records, child table -> archived ids)."""
    merges = [pl for pl in plan.pairs if pl.action == "merge"]
    folds = plan.folds()
    team_ids = sorted(
        {t for pl in merges for t in (pl.pair.merged_team_id, pl.pair.kept_team_id)} | {r.team_id for r in plan.nfc}
    )
    stale_ids = sorted(f.stale_id for f in folds)
    keep_ids = sorted({f.keep_id for f in folds})
    repointed = sorted({r.match_id for pl in merges for r in pl.repoints})
    held: dict[str, set[str]] = defaultdict(set)
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "xt", encoding="utf-8") as fh:

        def emit(table: str, role: str, rows: Iterable[dict]) -> None:
            nonlocal n
            for r in rows:
                if role == "stale_child":
                    held[table].add(str(r["id"]))
                fh.write(
                    json.dumps(
                        {"table": table, "role": role, "row": {k: _jsonable(v) for k, v in r.items()}},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1

        cur.execute("SELECT * FROM teams WHERE id = ANY(%s::uuid[])", (team_ids,))
        emit("teams", "team_before", cur.fetchall())
        for role, ids in (("stale_copy", stale_ids), ("finished_before", keep_ids), ("repointed_before", repointed)):
            if ids:
                cur.execute("SELECT * FROM matches WHERE id = ANY(%s::uuid[])", (ids,))
                emit("matches", role, cur.fetchall())
        if stale_ids:
            for table in schema.match_children:
                cur.execute(f"SELECT * FROM {table} WHERE match_id = ANY(%s::uuid[])", (stale_ids,))
                emit(table, "stale_child", cur.fetchall())
    return n, dict(held)


# ---------------------------------------------------------------------------
# Writing (inside the one transaction)
# ---------------------------------------------------------------------------

MOVE_ODDS_SQL = """
    UPDATE odds o SET match_id = %(keep)s::uuid
     WHERE o.id = ANY(%(ids)s::uuid[]) AND o.match_id = %(stale)s::uuid
       AND o."timestamp" < %(cutoff)s AND NOT COALESCE(o.is_live, false)
       AND NOT EXISTS (
             SELECT 1 FROM odds t
              WHERE t.match_id = %(keep)s::uuid AND t.bookmaker = o.bookmaker AND t.market_type = o.market_type
                AND t.selection = o.selection AND COALESCE(t.line, -1) = COALESCE(o.line, -1)
                AND COALESCE(t.is_opening, false) = COALESCE(o.is_opening, false)
                AND COALESCE(t.is_live, false) = COALESCE(o.is_live, false))
"""
# A reversed copy: the price of the fighter listed first on the copy is the
# price of the fighter listed second on the finished row.
MOVE_ODDS_SWAPPED_SQL = """
    UPDATE odds o SET match_id = %(keep)s::uuid,
                      selection = CASE o.selection WHEN 'home' THEN 'away' WHEN 'away' THEN 'home' END
     WHERE o.id = ANY(%(ids)s::uuid[]) AND o.match_id = %(stale)s::uuid
       AND o."timestamp" < %(cutoff)s AND NOT COALESCE(o.is_live, false)
       AND o.market_type = 'moneyline' AND o.selection IN ('home', 'away')
       AND NOT EXISTS (
             SELECT 1 FROM odds t
              WHERE t.match_id = %(keep)s::uuid AND t.bookmaker = o.bookmaker AND t.market_type = o.market_type
                AND t.selection = CASE o.selection WHEN 'home' THEN 'away' ELSE 'home' END
                AND COALESCE(t.line, -1) = COALESCE(o.line, -1)
                AND COALESCE(t.is_opening, false) = COALESCE(o.is_opening, false)
                AND COALESCE(t.is_live, false) = COALESCE(o.is_live, false))
"""
MOVE_SNAPSHOTS_SQL = """
    UPDATE odds_snapshots s SET match_id = %(keep)s::uuid
     WHERE s.id = ANY(%(ids)s::uuid[]) AND s.match_id = %(stale)s::uuid AND s.captured_at < %(cutoff)s
       AND NOT EXISTS (
             SELECT 1 FROM odds_snapshots t
              WHERE t.match_id = %(keep)s::uuid AND t.bookmaker = s.bookmaker AND t.market_type = s.market_type
                AND t.selection = s.selection AND COALESCE(t.line, -1) = COALESCE(s.line, -1)
                AND t.captured_at = s.captured_at)
"""
MOVE_SNAPSHOTS_SWAPPED_SQL = """
    UPDATE odds_snapshots s SET match_id = %(keep)s::uuid,
                                selection = CASE s.selection WHEN 'home' THEN 'away' WHEN 'away' THEN 'home' END
     WHERE s.id = ANY(%(ids)s::uuid[]) AND s.match_id = %(stale)s::uuid AND s.captured_at < %(cutoff)s
       AND s.market_type = 'moneyline' AND s.selection IN ('home', 'away')
       AND NOT EXISTS (
             SELECT 1 FROM odds_snapshots t
              WHERE t.match_id = %(keep)s::uuid AND t.bookmaker = s.bookmaker AND t.market_type = s.market_type
                AND t.selection = CASE s.selection WHEN 'home' THEN 'away' ELSE 'home' END
                AND COALESCE(t.line, -1) = COALESCE(s.line, -1) AND t.captured_at = s.captured_at)
"""
MOVE_PREDICTIONS_SQL = """
    UPDATE predictions p SET match_id = %(keep)s::uuid
     WHERE p.id = ANY(%(ids)s::uuid[]) AND p.match_id = %(stale)s::uuid
       AND COALESCE(p.updated_at, p.created_at) < %(cutoff)s
       AND NOT EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.prediction_id = p.id)
       AND NOT EXISTS (
             SELECT 1 FROM predictions t
              WHERE t.match_id = %(keep)s::uuid AND t.model_name = p.model_name
                AND t.model_version IS NOT DISTINCT FROM p.model_version AND t.prediction_type = p.prediction_type)
"""
# A rec on one of the copy's predictions makes this DELETE fail (migration 025,
# NO ACTION): that FK error is the last guard; the plan leaves such a copy first.
DELETE_STALE_SQL = """
    DELETE FROM matches m
     WHERE m.id = %(id)s::uuid AND m.home_team_id = %(home_team_id)s::uuid AND m.away_team_id = %(away_team_id)s::uuid
       AND m.match_date = %(match_date)s AND m.status = %(status)s AND m.status IN ('scheduled', 'cancelled')
       AND COALESCE(m.external_ids->>'espn', '') = %(espn)s
       AND NOT EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.match_id = m.id)
       AND NOT EXISTS (SELECT 1 FROM betting_history h WHERE h.match_id = m.id)
"""
ADOPT_ESPN_SQL = """
    UPDATE matches m
       SET external_ids = COALESCE(m.external_ids, '{}'::jsonb) || jsonb_build_object('espn', %(espn)s::text),
                         updated_at = NOW()
     WHERE m.id = %(id)s::uuid AND m.status = 'finished' AND m.match_date = %(match_date)s
       AND COALESCE(m.external_ids->>'espn', '') = ''
       AND NOT EXISTS (SELECT 1 FROM matches o WHERE o.id <> m.id AND o.external_ids->>'espn' = %(espn)s)
"""
REPOINT_SQL = {
    "home": """
        UPDATE matches SET home_team_id = %(kept)s::uuid, updated_at = NOW()
         WHERE id = %(id)s::uuid AND home_team_id = %(merged)s::uuid AND away_team_id = %(away_team_id)s::uuid
           AND match_date = %(match_date)s AND status = %(status)s
    """,
    "away": """
        UPDATE matches SET away_team_id = %(kept)s::uuid, updated_at = NOW()
         WHERE id = %(id)s::uuid AND away_team_id = %(merged)s::uuid AND home_team_id = %(home_team_id)s::uuid
           AND match_date = %(match_date)s AND status = %(status)s
    """,
}
CLEAR_MERGED_ESPN_SQL = """
    UPDATE teams SET external_ids = external_ids - 'espn', updated_at = NOW()
     WHERE id = %(id)s::uuid AND sport = 'mma' AND external_ids->>'espn' = %(espn)s
"""
TEAM_RECORD_SQL = """
    UPDATE teams
       SET metadata = jsonb_set(jsonb_set(COALESCE(metadata, '{}'::jsonb), '{aliases}', %(aliases)s::jsonb),
                                '{merged_from}', %(merged_from)s::jsonb),
           updated_at = NOW()
     WHERE id = %(id)s::uuid AND sport = 'mma' AND name = %(name)s
       AND COALESCE(metadata->'aliases', '[]'::jsonb) = %(observed_aliases)s::jsonb
       AND COALESCE(metadata->'merged_from', '[]'::jsonb) = %(observed_merged_from)s::jsonb
       AND COALESCE(external_ids->>'espn', '') = %(observed_espn)s
"""
NFC_RENAME_SQL = """
    UPDATE teams t SET name = %(new_name)s, normalized_name = %(new_normalized_name)s, updated_at = NOW()
     WHERE t.id = %(id)s::uuid AND t.sport = 'mma' AND t.name = %(name)s AND t.normalized_name = %(normalized_name)s
       AND NOT EXISTS (SELECT 1 FROM teams o WHERE o.sport = 'mma' AND o.id <> t.id
                                               AND o.normalized_name = %(new_normalized_name)s)
"""


def _expect(cur, sql: str, params: Any, n: int, what: str) -> None:
    cur.execute(sql, params)
    if cur.rowcount != n:
        raise GuardError(f"{what}: {cur.rowcount} row(s) changed, the plan expected {n}")


def apply_fold(cur, f: Fold, schema: Schema, held: dict[str, set[str]], done: Counter) -> None:
    params = {"keep": f.keep_id, "stale": f.stale_id, "cutoff": f.cutoff}
    odds_sql, snaps_sql = (
        (MOVE_ODDS_SWAPPED_SQL, MOVE_SNAPSHOTS_SWAPPED_SQL) if f.reversed else (MOVE_ODDS_SQL, MOVE_SNAPSHOTS_SQL)
    )
    for sql, ids, label in (
        (odds_sql, [o[0] for o in f.odds], "odds"),
        (snaps_sql, [s[0] for s in f.snapshots], "odds_snapshots"),
        (MOVE_PREDICTIONS_SQL, f.predictions, "predictions"),
    ):
        if ids:
            _expect(cur, sql, {**params, "ids": ids}, len(ids), f"move {label} {f.stale_id} -> {f.keep_id}")
            done[f"moved_{label}"] += len(ids)
    if schema.match_children:
        # The copy is locked FOR UPDATE: no child row can arrive before COMMIT; one written between the archive
        # and the lock would cascade away unarchived, so it refuses.
        cur.execute(children_sql(schema), {"m": f.stale_id})
        extra = sorted(f"{r['tbl']}:{r['id']}" for r in cur.fetchall() if r["id"] not in held.get(r["tbl"], set()))
        if extra:
            raise GuardError(f"stale copy {f.stale_id} holds {len(extra)} row(s) the archive lacks, e.g. {extra[:3]}")
    s = f.stale
    _expect(
        cur,
        DELETE_STALE_SQL,
        {
            "id": f.stale_id,
            "home_team_id": s["home_team_id"],
            "away_team_id": s["away_team_id"],
            "match_date": s["match_date"],
            "status": s["status"],
            "espn": s["espn"],
        },
        1,
        f"delete stale copy {f.stale_id}",
    )
    done["stale_copies_deleted"] += 1
    if f.adopt_espn:
        _expect(
            cur,
            ADOPT_ESPN_SQL,
            {"id": f.keep_id, "espn": f.adopt_espn, "match_date": f.keep["match_date"]},
            1,
            f"adopt ESPN id {f.adopt_espn} on {f.keep_id}",
        )
        done["espn_ids_adopted"] += 1


def apply_writes(
    cur, plan: RunPlan, schema: Schema, resolvers: Resolvers, held: dict[str, set[str]], now: datetime
) -> Counter:
    done: Counter = Counter()
    for f in plan.folds():
        apply_fold(cur, f, schema, held, done)
    merges = [pl for pl in plan.pairs if pl.action == "merge"]
    for pl in merges:
        for r in pl.repoints:
            o = r.observed
            _expect(
                cur,
                REPOINT_SQL[r.side],
                {"id": r.match_id, "kept": r.kept_id, "merged": r.merged_id, **o},
                1,
                f"re-point {r.side} of {r.match_id}",
            )
            done["matches_repointed"] += 1
    refs_sql = team_refs_sql(schema)
    for pl in merges:
        p, t = pl.pair, pl.team_patch
        cur.execute(refs_sql, {"t": p.merged_team_id})
        refs = {k: int(v) for k, v in dict(cur.fetchone()).items() if v}
        if refs:
            raise GuardError(f"{p.merged_name!r} is still referenced after the re-point: {refs}")
        if t["clear_merged_espn"]:
            _expect(
                cur,
                CLEAR_MERGED_ESPN_SQL,
                {"id": p.merged_team_id, "espn": p.espn_athlete_id},
                1,
                f"clear ESPN id off {p.merged_name!r}",
            )
        merged_from = t["observed_merged_from"] + [
            {
                "team_id": p.merged_team_id,
                "name": p.merged_name,
                "normalized_name": pl.merged_team.get("normalized_name"),
                "espn_athlete_id": p.espn_athlete_id,
                "merged_at": now.isoformat(),
                "source": "scripts/merge_mma_fighters.py (scripts/data/mma_fighter_merge_map.csv)",
            }
        ]
        _expect(
            cur,
            TEAM_RECORD_SQL,
            {
                "id": p.kept_team_id,
                "name": p.kept_name,
                "aliases": json.dumps(t["aliases"], ensure_ascii=False),
                "merged_from": json.dumps(merged_from, ensure_ascii=False),
                "observed_aliases": json.dumps(t["observed_aliases"], ensure_ascii=False),
                "observed_merged_from": json.dumps(t["observed_merged_from"], ensure_ascii=False, default=_jsonable),
                "observed_espn": t["observed_espn"],
            },
            1,
            f"record the merge on {p.kept_name!r}",
        )
        if t["stamp"]:
            if not resolvers.stamp_espn_id(cur, p.kept_team_id, p.espn_athlete_id, f"mma {p.kept_name!r} (merge)"):
                raise GuardError(f"the guarded stamp refused ESPN athlete id {p.espn_athlete_id} on {p.kept_name!r}")
            done["athlete_ids_stamped"] += 1
        _expect(
            cur,
            team_delete_sql(schema),
            {"id": p.merged_team_id, "name": p.merged_name, "normalized_name": pl.merged_team.get("normalized_name")},
            1,
            f"delete team {p.merged_name!r}",
        )
        done["teams_deleted"] += 1
    for rn in plan.nfc:
        _expect(
            cur,
            NFC_RENAME_SQL,
            {
                "id": rn.team_id,
                "name": rn.name,
                "normalized_name": rn.normalized_name,
                "new_name": rn.new_name,
                "new_normalized_name": rn.new_normalized_name,
            },
            1,
            f"NFC-normalize team {rn.team_id}",
        )
        done["nfc_renamed"] += 1
    return done


# ---------------------------------------------------------------------------
# After-state (read-only; also run inside the apply before COMMIT)
# ---------------------------------------------------------------------------

AFTER_INFO = ("pairs_expected", "pairs_recorded")
# Every count --check-after prints; each one but AFTER_INFO must be 0.
AFTER_KEYS = AFTER_INFO + (
    "merged_rows_left",
    "merged_refs_left",
    "merges_unrecorded",
    "athlete_ids_missing",
    "athlete_ids_elsewhere",
    "teams_reminted",
    "resolver_misses",
    "alias_misses",
    "repoints_missing",
    "repoints_wrong",
    "stale_copies_left",
    "finished_rows_missing",
    "espn_ids_missing",
    "odds_missing",
    "odds_snapshots_missing",
    "predictions_missing",
    "stale_twins_left",
    "nfc_missing",
    "names_not_nfc",
)
AFTER_LATER_STEPS_INFO = ("repoints_missing",)


def _merge_entries(doc: dict) -> list[dict]:
    return [e for e in doc.get("pairs", []) if e.get("action") == "merge"]


def _fold_entries(doc: dict) -> list[dict]:
    out = [f for e in _merge_entries(doc) for f in e.get("folds", [])]
    return out + [e["fold"] for e in doc.get("listed", []) if e.get("action") == "fold" and e.get("fold")]


def check_after(cur, doc: dict, resolvers: Resolvers) -> dict[str, int]:
    """The after-state `doc` (a plan file) implies, as problem counts (every one
    must be 0) and the info counts in AFTER_INFO."""
    out: Counter = Counter({k: 0 for k in AFTER_KEYS})
    merges = _merge_entries(doc)
    out["pairs_expected"] = len(merges)
    if merges:
        cur.execute(TEAMS_SQL, ([t for e in merges for t in (e["merged_team_id"], e["kept_team_id"])],))
        teams = {r["id"]: dict(r) for r in cur.fetchall()}
        cur.execute(SPORT_TEAMS_SQL)
        sport = [dict(r) for r in cur.fetchall()]
    else:
        teams, sport = {}, []
    for e in merges:
        mid, kid = e["merged_team_id"], e["kept_team_id"]
        if mid in teams:
            out["merged_rows_left"] += 1
        cur.execute(
            "SELECT count(*) AS n FROM matches WHERE home_team_id = %(t)s::uuid OR away_team_id = %(t)s::uuid",
            {"t": mid},
        )
        out["merged_refs_left"] += int(cur.fetchone()["n"])
        kept = teams.get(kid)
        recorded = (
            kept is not None
            and e["merged_name"] in (_aliases(kept) or [])
            and any(str(x.get("team_id")) == mid for x in (_merged_from(kept) or []))
        )
        if recorded:
            out["pairs_recorded"] += 1
        else:
            out["merges_unrecorded"] += 1
        if _team_espn(kept) != e["espn_athlete_id"]:
            out["athlete_ids_missing"] += 1
        out["athlete_ids_elsewhere"] += sum(1 for t in sport if t["espn"] == e["espn_athlete_id"] and t["id"] != kid)
        merged_norms = {resolvers.normalize(s) for s in spellings(e["merged_name"])}
        out["teams_reminted"] += sum(1 for t in sport if t["normalized_name"] in merged_norms and t["id"] != kid)
        for name in spellings(e["merged_name"]) + spellings(e["kept_name"]):
            if str(resolvers.find_team(cur, name, e.get("league_id")) or "") != kid:
                out["resolver_misses"] += 1
        if str(resolvers.find_team_by_alias(cur, e["merged_name"]) or "") != kid:
            out["alias_misses"] += 1
        for match_id, side in e.get("repoints", []):
            cur.execute(
                "SELECT home_team_id::text AS h, away_team_id::text AS a FROM matches WHERE id = %s::uuid", (match_id,)
            )
            row = cur.fetchone()
            if row is None:
                out["repoints_missing"] += 1
            elif (row["h"] if side == "home" else row["a"]) != kid:
                out["repoints_wrong"] += 1
    for f in _fold_entries(doc):
        cur.execute("SELECT 1 FROM matches WHERE id = %s::uuid", (f["stale_id"],))
        if cur.fetchone():
            out["stale_copies_left"] += 1
        cur.execute(
            "SELECT status, COALESCE(external_ids->>'espn', '') AS espn FROM matches WHERE id = %s::uuid",
            (f["keep_id"],),
        )
        keep = cur.fetchone()
        if keep is None or keep["status"] != FINISHED:
            out["finished_rows_missing"] += 1
        elif f.get("adopt_espn") and keep["espn"] != f["adopt_espn"]:
            out["espn_ids_missing"] += 1
        for table, rows in (("odds", f["odds"]), ("odds_snapshots", f["snapshots"])):
            if rows:
                cur.execute(
                    f"SELECT count(*) AS n FROM {table} t JOIN unnest(%(ids)s::uuid[], %(sels)s::text[]) AS w(id, sel) "
                    f"ON w.id = t.id WHERE t.match_id = %(keep)s::uuid AND t.selection = w.sel",
                    {"ids": [r[0] for r in rows], "sels": [r[1] for r in rows], "keep": f["keep_id"]},
                )
                out[f"{table}_missing"] += len(rows) - int(cur.fetchone()["n"])
        if f["predictions"]:
            cur.execute(
                "SELECT count(*) AS n FROM predictions WHERE id = ANY(%(ids)s::uuid[]) AND match_id = %(keep)s::uuid",
                {"ids": f["predictions"], "keep": f["keep_id"]},
            )
            out["predictions_missing"] += len(f["predictions"]) - int(cur.fetchone()["n"])
        h, a = f["keep_teams"]
        cur.execute(
            """SELECT count(*) AS n FROM matches
                WHERE id <> %(keep)s::uuid AND status IN ('scheduled', 'cancelled')
                  AND ((home_team_id = %(h)s::uuid AND away_team_id = %(a)s::uuid)
                       OR (home_team_id = %(a)s::uuid AND away_team_id = %(h)s::uuid))
                  AND match_date BETWEEN %(k)s::timestamptz - %(w)s::interval
                                     AND %(k)s::timestamptz + %(w)s::interval""",
            {
                "keep": f["keep_id"],
                "h": h,
                "a": a,
                "k": f["keep_kickoff"],
                "w": f"{int(TWIN_WINDOW.total_seconds() // 3600)} hours",
            },
        )
        out["stale_twins_left"] += int(cur.fetchone()["n"])
    for r in doc.get("nfc", []):
        cur.execute("SELECT name, normalized_name FROM teams WHERE id = %s::uuid", (r["team_id"],))
        row = cur.fetchone()
        if row is None or row["name"] != r["new_name"] or row["normalized_name"] != r["new_normalized_name"]:
            out["nfc_missing"] += 1
        for s in spellings(r["new_name"]):
            if str(resolvers.find_team(cur, s, None) or "") != r["team_id"]:
                out["resolver_misses"] += 1
    cur.execute("SELECT id::text AS id, name, normalized_name FROM teams WHERE sport = 'mma'")
    out["names_not_nfc"] = sum(
        1
        for t in cur.fetchall()
        if unicodedata.normalize("NFC", t["name"]) != t["name"]
        or unicodedata.normalize("NFC", t["normalized_name"]) != t["normalized_name"]
    )
    return dict(out)


def after_state_ok(out: dict[str, int], after_later_steps: bool = False) -> bool:
    info = AFTER_INFO + (AFTER_LATER_STEPS_INFO if after_later_steps else ())
    return out.get("pairs_recorded") == out.get("pairs_expected") and all(
        v == 0 for k, v in out.items() if k not in info
    )


def after_line(out: dict[str, int]) -> str:
    """The AFTER counts, every key in AFTER_KEYS order (one fixed-format line for the wrapper)."""
    return " ".join(f"{k}={out.get(k, 0)}" for k in AFTER_KEYS)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def default_out(kind: str) -> Path:
    base = Path(os.environ.get("BACKUP_LOCAL_DIR", "/app/backups"))
    ext = {"csv": "csv", "archive": "jsonl.gz"}[kind]
    return base / f"merge_mma_fighters_{_stamp()}.{ext}"


def log_plan(plan: RunPlan) -> None:
    for pl in plan.pairs:
        tag = f"[pair {pl.index}/{len(plan.pairs)}] {pl.pair.merged_name!r} -> {pl.pair.kept_name!r}"
        if pl.action == "refuse":
            logger.error("%s: REFUSED — %s", tag, pl.reason)
        elif pl.action == "done":
            logger.info("%s: done (%s)", tag, pl.reason)
        else:
            logger.info(
                "%s: fold %d stale cop(ies) (%d reversed), re-point %d side(s), leave %d rec-held; ESPN athlete %s",
                tag,
                len(pl.folds),
                sum(f.reversed for f in pl.folds),
                len(pl.repoints),
                len(pl.left),
                pl.pair.espn_athlete_id,
            )
            for mid in pl.left:
                logger.warning("%s: stale copy %s is held by a rec — LEFT in place (re-pointed, not folded)", tag, mid)
    for lp in plan.listed:
        tag = f"[listed {lp.index}/{len(plan.listed)}] {lp.listed.stale_home} v {lp.listed.stale_away}"
        if lp.action == "refuse":
            logger.error("%s: REFUSED — %s", tag, lp.reason)
        elif lp.action == "done":
            logger.info("%s: done (%s)", tag, lp.reason)
        else:
            logger.info("%s: fold %s into %s (reversed)", tag, lp.listed.stale_match_id, lp.listed.finished_match_id)
    for r in plan.nfc:
        logger.info("[nfc] team %s %r -> NFC %r", r.team_id, r.name, r.new_name)
    for why in plan.nfc_refused:
        logger.error("[nfc] REFUSED — %s", why)


def _plan_on(cur, pairs, folds, schema, resolvers, now, lock: bool) -> tuple[RunPlan, dict]:
    state = load_state(cur, pairs, folds, schema, lock=lock)
    plan = plan_run(pairs, folds, state, now, resolvers)
    return plan, plan_document(plan, "apply" if lock else "dry-run")


def run(
    conn,
    apply_changes: bool,
    pairs: list[MergePair],
    folds: list[ListedFold],
    out_csv: Path,
    archive: Path,
    resolvers: Optional[Resolvers],
    resolver_reason: Optional[str] = None,
    plan_out: Optional[Path] = None,
    expect_plan: Optional[dict] = None,
    now: Optional[datetime] = None,
) -> int:
    """Exit status: 0 done (dry run: nothing refused); 1 a refusal in the dry
    run, or the apply rolled back after it began writing; 2 refused before any
    write (no resolver, unknown foreign key, a plan that is not the expected
    one, a refusal)."""
    if resolvers is None or resolver_reason:
        logger.error("REFUSED: %s — nothing read or written", resolver_reason)
        return 2
    now = now or datetime.now(timezone.utc)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if apply_changes:
            cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cur.execute("SET LOCAL lock_timeout = %s", (LOCK_TIMEOUT,))
        else:
            cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        schema, unknown = load_schema(cur)
        if schema is None:
            conn.rollback()
            logger.error("REFUSED: foreign keys this tool does not know: %s — nothing written", "; ".join(unknown))
            return 2
        try:
            plan, doc = _plan_on(cur, pairs, folds, schema, resolvers, now, lock=apply_changes)
        except psycopg2.Error as e:
            conn.rollback()
            logger.error("REFUSED: reading the state failed (%s) — nothing written", str(e).strip().splitlines()[0])
            return 2
        write_csv(out_csv, csv_rows(plan))
        plan_path = plan_out or out_csv.with_suffix(".plan.json")
        write_plan(plan_path, doc)
        log_plan(plan)
        logger.info("Audit CSV: %s", out_csv)
        logger.info("PLAN_FILE %s", plan_path)
        logger.info("%s", plan_line(doc["summary"]))
        logger.info("PLAN_DIGEST %s", doc["digest"])
        for mid in doc["grade_after"]:
            logger.info("GRADE_AFTER %s (grade_completed_matches.py --match-id %s)", mid, mid)
        refused = refusals(plan)
        if not apply_changes:
            conn.rollback()
            for why in refused:
                logger.error("REFUSED %s", why)
            logger.info("DRY-RUN: nothing written")
            return 1 if refused else 0
        if expect_plan is None:
            conn.rollback()
            logger.error("REFUSED: --apply needs --expect-plan <the dry run's plan file> — nothing written")
            return 2
        diff = plan_difference(expect_plan, doc)
        if diff:
            conn.rollback()
            logger.error("REFUSED: this plan is not the one the dry run showed — %s; nothing written", diff)
            return 2
        if refused:
            conn.rollback()
            for why in refused:
                logger.error("REFUSED %s", why)
            logger.error("REFUSED: the plan refuses %d unit(s) — nothing written", len(refused))
            return 2
        logger.info("Plan matches the expected plan (digest %s)", doc["digest"])
        try:
            n, held = write_archive(cur, archive, plan, schema)
            logger.info("Archived %d row(s) to %s", n, archive)
            written = apply_writes(cur, plan, schema, resolvers, held, now)
            after = check_after(cur, doc, resolvers)
            logger.info("AFTER (in the transaction) %s", after_line(after))
            if not after_state_ok(after):
                raise GuardError("the after-state does not hold inside the transaction (above)")
        except (GuardError, psycopg2.Error, OSError) as e:
            conn.rollback()
            logger.error("ROLLED BACK — nothing committed: %s", str(e).strip().splitlines()[0])
            return 1
        conn.commit()
    logger.info("WRITTEN %s", " ".join(f"{k}={v}" for k, v in sorted(written.items())))
    logger.info("APPLIED: committed (plan digest %s)", doc["digest"])
    return 0


def run_check_after(
    conn, plan_path: Path, resolvers: Optional[Resolvers], resolver_reason: Optional[str], after_later_steps: bool
) -> int:
    if resolvers is None or resolver_reason:
        logger.error("REFUSED: %s", resolver_reason)
        return 2
    doc = json.loads(plan_path.read_text(encoding="utf-8"))
    if doc.get("tool") != "merge_mma_fighters":
        logger.error("REFUSED: %s is not a merge_mma_fighters plan file", plan_path)
        return 2
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        out = check_after(cur, doc, resolvers)
    conn.rollback()
    logger.info("AFTER %s", after_line(out))
    if after_state_ok(out, after_later_steps):
        if after_later_steps and out.get("repoints_missing"):
            logger.info("repoints_missing=%d reported, not failed (--after-later-steps)", out["repoints_missing"])
        logger.info("AFTER-STATE OK: %d/%d merges recorded and resolving", out["pairs_recorded"], out["pairs_expected"])
        return 0
    logger.error("AFTER-STATE FAILED: every count above must be 0 and pairs_recorded=pairs_expected")
    return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="Write (default: a read-only dry run).")
    p.add_argument("--expect-plan", type=Path, default=None, help="--apply only if the plan equals this dry run's.")
    p.add_argument("--map", type=Path, default=MAP_PATH, help=argparse.SUPPRESS)
    p.add_argument("--folds", type=Path, default=FOLDS_PATH, help=argparse.SUPPRESS)
    p.add_argument("--out-csv", type=Path, default=None, help="Audit CSV (default $BACKUP_LOCAL_DIR or /app/backups).")
    p.add_argument("--plan-out", type=Path, default=None, help="Plan file (default: the audit CSV's .plan.json).")
    p.add_argument("--archive", type=Path, default=None, help="gzip JSONL archive written before --apply writes.")
    p.add_argument("--check-after", type=Path, default=None, help="Read-only: check the after-state of a plan file.")
    p.add_argument(
        "--after-later-steps",
        action="store_true",
        help="With --check-after: run B has run since (it deletes some re-pointed no-twin rows on purpose).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    resolvers, reason = load_resolvers()
    if args.after_later_steps and not args.check_after:
        logger.error("REFUSED: --after-later-steps only goes with --check-after")
        return 2
    if args.check_after:
        conn = psycopg2.connect(args.database_url)
        try:
            return run_check_after(conn, args.check_after, resolvers, reason, args.after_later_steps)
        finally:
            conn.close()
    if args.expect_plan and not args.apply:
        logger.error("REFUSED: --expect-plan only goes with --apply")
        return 2
    try:
        pairs = load_map(args.map)
        folds = load_folds(args.folds)
        expect = json.loads(args.expect_plan.read_text(encoding="utf-8")) if args.expect_plan else None
    except (ListError, OSError, ValueError) as e:
        logger.error("REFUSED: %s", e)
        return 2
    out_csv = args.out_csv or default_out("csv")
    conn = psycopg2.connect(args.database_url)
    try:
        return run(
            conn,
            args.apply,
            pairs,
            folds,
            out_csv,
            args.archive or default_out("archive"),
            resolvers,
            reason,
            plan_out=args.plan_out,
            expect_plan=expect,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
