"""MMA fighter identity clean-up: split the one team row that holds two ESPN
athletes, add the spelling ESPN uses for one fighter as an alias, delete one
bout-less orphan row, and stamp every unambiguously linked fighter row with its
ESPN athlete id (teams.external_ids.espn) — the id fetch_upcoming resolves MMA
fighters by first since 2026-11-02 (SportConfig.athlete_ids).

THE PROBLEMS (prod read-only export 2026-11-02T06:22:51Z, checked against ESPN)
------------------------------------------------------------------------------
fetch_upcoming used to resolve an MMA fighter by name only. That split 9
fighters across two rows when ESPN changed their display names (merged by
scripts/merge_mma_fighters.py on 2026-11-02), and it put two different ESPN
athletes named 'Bruno Silva' on ONE row: 4333158 ('Blindado', middleweight, 7
bouts) and 3895544 ('Bulldog', flyweight, 4 bouts), so each one's rolling form
and record mixed in the other's fights. ESPN now spells 'Rafael Cerqueira' as
'Raffael Cerqueira' (athlete 5212598), so his next bout would mint a second
row. 'José Luiz' is a row with no bout left (run B deleted its one bout, which
ESPN had dropped), and which ESPN athlete it was cannot be established any
more. And only the 9 merged rows carried their athlete id, so the new
athlete-id resolver would have had nothing to key on for the other ~1,290.

THE LISTS (reviewed by hand, with ESPN evidence; the tool touches nothing else)
------------------------------------------------------------------------------
  scripts/data/mma_fighter_split_20261102.csv     every bout of the split row:
      the athlete ESPN names on it, keep (stays on the row) or move (goes to the
      new row, whose id and name the list pins).
  scripts/data/mma_fighter_aliases_20261102.csv   row -> alias (+ athlete id).
  scripts/data/mma_orphan_fighters_20261102.csv   bout-less rows to delete.
  scripts/data/mma_fighter_athlete_ids_20261102.csv  row -> ESPN athlete id,
      for every row ESPN links to exactly one athlete id that no other row
      links to (1,299 rows; the split row and the orphan are not in it).
The parallel stream mma_draws_held owns two match rows (b7f16d27 Gatto v
Kareckaite, 787bd68e Valesca Machado v Melissa Amaya) and the 'Valesca
Machado' team row: no list may name them (a list that does is refused), so
this tool and that stream's wrapper can run in either order.

WHAT A RUN DOES (ONE transaction: any failure rolls all of it back)
-------------------------------------------------------------------
1. Split: insert the new row (the listed id and name, the old row's league,
   external_ids.espn = the moved athlete, metadata.aliases = [the old name]
   so the odds feed's alias tier finds his bouts under ESPN's plain name, and
   metadata.split_from); stamp the old row with the kept athlete's id and
   append metadata.split_into; re-point every 'move' bout's side to the new
   row, each UPDATE guarded on the bout's observed teams, kickoff and status.
   Predictions stay on their bouts (a bout is the same bout; the follow-up
   feature recompute rebuilds both fighters' histories); a recommendation or
   a betting_history row on a moved bout (directly or through one of its
   predictions) refuses the split — today there are none.
2. Aliases: append the alias to metadata.aliases (guarded on the observed
   list). Refused when another row carries that name or claims that alias.
3. Orphans: delete the row, guarded on its name, no ESPN id, and no reference
   under ANY foreign key to teams (read from pg_constraint: matches -> teams
   cascades, so a delete must never find a bout).
4. Athlete ids: stamp each listed row through team_identity.STAMP_ESPN_SQL
   (the row holds no ESPN id and no other MMA row holds this one). A row that
   already carries its listed id is done (a sweep stamps the rows it finds by
   name since the resolver went live); one that carries another id, or whose
   id another row holds, refuses the run.
5. Before COMMIT, inside the transaction, the deployed resolver
   (fetch_upcoming.find_team on SPORT_CONFIGS['mma']) must resolve every
   listed row by its athlete id, both Bruno Silvas to their own rows (and the
   plain name to the kept row), the alias to its row; and the after-state of
   --check-after must hold exactly.

SAFETY
------
The dry run (the default) runs in a READ ONLY transaction and writes only its
write-once audit CSV and plan file. --apply takes SHARE ROW EXCLUSIVE on teams
(no sweep can stamp or insert a fighter meanwhile; lock_timeout 10s), locks the
moved bouts FOR UPDATE, re-plans, and refuses (nothing written, exit 2) unless
the plan digest equals --expect-plan <the dry run's plan file>. The digest
covers the split (every moved bout with its observed side, kickoff and
status), the aliases, the orphans and the athlete id each listed row ends up
with, so a row a sweep stamped between the dry run and the apply (done instead
of stamp) does not change it. Before its first write the apply writes a
write-once gzip JSONL archive of every row it changes or deletes (the reversal
record). Every write is guarded on the observed values and must touch exactly
one row. Re-runnable: a run with everything done writes nothing.

--check-after <plan file> (read-only) checks the after-state a plan implies.

RUN ORDER: after scripts/ops/merge_mma_fighters.sh (its wrapper refuses unless
merge_mma_fighters.sh --verify passes). Independent of mma_draws_held.

Usage (inside the api container; dry run is the default):
    python /app/scripts/mma_identity_cleanup.py --plan-out /app/backups/mma_identity.plan.json
    python /app/scripts/mma_identity_cleanup.py --apply --expect-plan /app/backups/mma_identity.plan.json
    python /app/scripts/mma_identity_cleanup.py --check-after /app/backups/mma_identity.plan.json
Afterwards (scripts/ops/mma_identity_cleanup.sh runs them): compute_features_mma.py
--all-finished --force and --days 14 --force (the split fighters' histories changed).
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
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("mma_identity_cleanup")

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA = SCRIPTS_DIR / "data"
SPLIT_PATH = DATA / "mma_fighter_split_20261102.csv"
ALIAS_PATH = DATA / "mma_fighter_aliases_20261102.csv"
ORPHAN_PATH = DATA / "mma_orphan_fighters_20261102.csv"
IDS_PATH = DATA / "mma_fighter_athlete_ids_20261102.csv"
SPLIT_FIELDS = (
    "team_id",
    "team_name",
    "new_team_id",
    "new_team_name",
    "match_id",
    "kickoff",
    "status",
    "side",
    "opponent",
    "espn_competition_id",
    "espn_event_id",
    "espn_athlete_id",
    "assign",
    "evidence",
)
ALIAS_FIELDS = ("team_id", "team_name", "alias", "espn_athlete_id", "evidence")
ORPHAN_FIELDS = ("team_id", "team_name", "created_at", "evidence")
IDS_FIELDS = ("team_id", "team_name", "espn_athlete_id", "espn_names", "bouts", "espn_competition_ids", "method")
SPORT = "mma"
LOCK_TIMEOUT = "10s"
SOURCE = "scripts/mma_identity_cleanup.py"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# The parallel stream mma_draws_held owns these rows; no list may name them.
HELD_MATCHES = frozenset({"b7f16d27-42a1-41b0-a6f6-6e8e9837da47", "787bd68e-2ce8-4beb-b346-c9f422c42574"})
HELD_TEAMS = frozenset({"108423d1-7f54-43ba-8039-43e5ef23b8bf"})  # 'Valesca Machado'
REC_HOLDERS = ("betting_recommendations", "betting_history")

PLAN_KEYS = (
    "split",
    "split_done",
    "split_refuse",
    "bouts_moved",
    "predictions_on_moved",
    "recs_on_moved",
    "alias_add",
    "alias_done",
    "alias_refuse",
    "orphan_delete",
    "orphan_done",
    "orphan_refuse",
    "ids_stamp",
    "ids_done",
    "ids_refuse",
)


class ListError(ValueError):
    """A reviewed list is malformed or unsafe to run."""


class GuardError(RuntimeError):
    """A guarded write found the database different from the plan."""


# ---------------------------------------------------------------------------
# The reviewed lists
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitBout:
    match_id: str
    kickoff: datetime
    status: str
    side: str
    opponent: str
    espn_competition_id: str
    espn_athlete_id: str
    assign: str


@dataclass(frozen=True)
class SplitList:
    team_id: str
    team_name: str
    new_team_id: str
    new_team_name: str
    keep_athlete: str
    move_athlete: str
    bouts: tuple[SplitBout, ...]

    def of(self, assign: str) -> list[SplitBout]:
        return [b for b in self.bouts if b.assign == assign]


@dataclass(frozen=True)
class AliasRow:
    team_id: str
    team_name: str
    alias: str
    espn_athlete_id: str


@dataclass(frozen=True)
class OrphanRow:
    team_id: str
    team_name: str


@dataclass(frozen=True)
class IdRow:
    team_id: str
    team_name: str
    espn_athlete_id: str


@dataclass(frozen=True)
class Lists:
    split: SplitList
    aliases: tuple[AliasRow, ...]
    orphans: tuple[OrphanRow, ...]
    ids: tuple[IdRow, ...]


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
    for n, row in enumerate(reader, start=1):
        vals = {f: (row.get(f) or "").strip() for f in fields}
        empty = [f for f, v in vals.items() if not v]
        if empty:
            raise ListError(f"{path}: data row {n}: empty {', '.join(empty)} — only reviewed rows with evidence")
        out.append(vals)
    if not out:
        raise ListError(f"{path}: no data rows")
    return out


def _uuid(path: Path, value: str, what: str) -> str:
    if not UUID_RE.match(value):
        raise ListError(f"{path}: {what} {value!r} is not a UUID")
    if value in HELD_MATCHES or value in HELD_TEAMS:
        raise ListError(f"{path}: {what} {value} belongs to the parallel stream mma_draws_held — never listed here")
    return value


def _athlete(path: Path, value: str) -> str:
    if not value.isdigit():
        raise ListError(f"{path}: ESPN athlete id {value!r} is not digits")
    return value


def load_split(path: Path) -> SplitList:
    """The split list, validated: one team, one new row, every match once, a
    zoned kickoff, side home|away, assign keep|move, exactly one athlete id per
    assignment and two different ones, at least one bout of each."""
    rows = _rows(path, SPLIT_FIELDS)
    heads = {(r["team_id"], r["team_name"], r["new_team_id"], r["new_team_name"]) for r in rows}
    if len(heads) != 1:
        raise ListError(f"{path}: the rows name {len(heads)} different (team, new team) heads; one split per list")
    team_id, team_name, new_team_id, new_team_name = heads.pop()
    _uuid(path, team_id, "team_id")
    _uuid(path, new_team_id, "new_team_id")
    if team_id == new_team_id or team_name == new_team_name:
        raise ListError(f"{path}: the new row must differ from the split row in id and name")
    bouts: list[SplitBout] = []
    seen: set[str] = set()
    athletes: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        mid = _uuid(path, r["match_id"], "match_id")
        if mid in seen:
            raise ListError(f"{path}: match {mid} is listed twice")
        seen.add(mid)
        try:
            kickoff = datetime.fromisoformat(r["kickoff"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ListError(f"{path}: kickoff {r['kickoff']!r} is not ISO-8601") from exc
        if kickoff.tzinfo is None:
            raise ListError(f"{path}: kickoff {r['kickoff']!r} has no time zone")
        if r["side"] not in ("home", "away") or r["assign"] not in ("keep", "move"):
            raise ListError(f"{path}: match {mid}: side must be home|away and assign keep|move")
        athletes[r["assign"]].add(_athlete(path, r["espn_athlete_id"]))
        bouts.append(
            SplitBout(
                mid,
                kickoff,
                r["status"],
                r["side"],
                r["opponent"],
                r["espn_competition_id"],
                r["espn_athlete_id"],
                r["assign"],
            )
        )
    if len(athletes["keep"]) != 1 or len(athletes["move"]) != 1:
        raise ListError(f"{path}: keep and move must each name exactly one ESPN athlete id (got {dict(athletes)})")
    (keep,), (move,) = athletes["keep"], athletes["move"]
    if keep == move:
        raise ListError(f"{path}: keep and move name the same athlete {keep}")
    return SplitList(team_id, team_name, new_team_id, new_team_name, keep, move, tuple(bouts))


def load_aliases(path: Path) -> tuple[AliasRow, ...]:
    out: list[AliasRow] = []
    seen: set[str] = set()
    for r in _rows(path, ALIAS_FIELDS):
        a = AliasRow(
            _uuid(path, r["team_id"], "team_id"), r["team_name"], r["alias"], _athlete(path, r["espn_athlete_id"])
        )
        if a.alias.casefold() in seen or a.alias.strip().casefold() == a.team_name.strip().casefold():
            raise ListError(f"{path}: alias {a.alias!r} is listed twice or is the row's own name")
        seen.add(a.alias.casefold())
        out.append(a)
    return tuple(out)


def load_orphans(path: Path) -> tuple[OrphanRow, ...]:
    out: list[OrphanRow] = []
    seen: set[str] = set()
    for r in _rows(path, ORPHAN_FIELDS):
        o = OrphanRow(_uuid(path, r["team_id"], "team_id"), r["team_name"])
        if o.team_id in seen:
            raise ListError(f"{path}: team {o.team_id} is listed twice")
        seen.add(o.team_id)
        out.append(o)
    return tuple(out)


def load_ids(path: Path) -> tuple[IdRow, ...]:
    out: list[IdRow] = []
    teams: set[str] = set()
    athletes: set[str] = set()
    for r in _rows(path, IDS_FIELDS):
        row = IdRow(_uuid(path, r["team_id"], "team_id"), r["team_name"], _athlete(path, r["espn_athlete_id"]))
        if row.team_id in teams or row.espn_athlete_id in athletes:
            raise ListError(f"{path}: team {row.team_id} or ESPN athlete {row.espn_athlete_id} is listed twice")
        teams.add(row.team_id)
        athletes.add(row.espn_athlete_id)
        out.append(row)
    return tuple(out)


def load_lists(
    split_path: Path = SPLIT_PATH,
    alias_path: Path = ALIAS_PATH,
    orphan_path: Path = ORPHAN_PATH,
    ids_path: Path = IDS_PATH,
) -> Lists:
    """All four lists, cross-checked: the athlete-id list names neither split
    row, no orphan and no split athlete; an orphan is in no other list; an
    aliased row listed for an athlete id names the alias row's id."""
    lists = Lists(load_split(split_path), load_aliases(alias_path), load_orphans(orphan_path), load_ids(ids_path))
    s = lists.split
    ids_by_team = {r.team_id: r for r in lists.ids}
    orphan_ids = {o.team_id for o in lists.orphans}
    for tid in (s.team_id, s.new_team_id):
        if tid in ids_by_team or tid in orphan_ids:
            raise ListError(f"split row {tid} is also in the athlete-id or orphan list; the split stamps both rows")
    taken = {r.espn_athlete_id for r in lists.ids}
    if {s.keep_athlete, s.move_athlete} & taken:
        raise ListError("a split athlete id is also in the athlete-id list")
    for o in lists.orphans:
        if o.team_id in ids_by_team or any(a.team_id == o.team_id for a in lists.aliases):
            raise ListError(f"orphan {o.team_id} is also in the athlete-id or alias list")
    for a in lists.aliases:
        r = ids_by_team.get(a.team_id)
        if r is not None and r.espn_athlete_id != a.espn_athlete_id:
            raise ListError(f"alias row {a.team_id} names athlete {a.espn_athlete_id}, the id list {r.espn_athlete_id}")
    return lists


# ---------------------------------------------------------------------------
# The deployed resolver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolvers:
    """The resolution code the pipeline runs (scripts/ is bind-mounted)."""

    normalize: Callable[[str], str]  # fetch_upcoming.normalize_team: teams.normalized_name
    alias_form: Callable[[str], str]
    find_team: Callable[[Any, str, Optional[str]], Optional[str]]  # cur, name, ESPN athlete id
    stamp_sql: str  # team_identity.STAMP_ESPN_SQL


def load_resolvers(scripts_dir: Path = SCRIPTS_DIR) -> tuple[Optional[Resolvers], Optional[str]]:
    """(Resolvers, None), or (None, why not). Refused unless the deployed
    fetch_upcoming resolves MMA by athlete id: the after-state this tool
    checks is that resolver's."""
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    mods = {}
    api = {
        "team_identity": ("alias_form", "STAMP_ESPN_SQL"),
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
    if not getattr(cfg, "athlete_ids", False):
        return None, "the deployed fetch_upcoming does not resolve MMA fighters by ESPN athlete id (athlete_ids)"
    return (
        Resolvers(
            normalize=fu.normalize_team,
            alias_form=ti.alias_form,
            find_team=lambda cur, name, espn: fu.find_team(cur, cfg, name, None, espn),
            stamp_sql=ti.STAMP_ESPN_SQL,
        ),
        None,
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

TEAM_FKS_SQL = """
    SELECT c.conrelid::regclass::text AS tbl, a.attname AS col
      FROM pg_constraint c
      CROSS JOIN LATERAL unnest(c.conkey) AS k(attnum)
      JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
     WHERE c.contype = 'f' AND c.confrelid = 'teams'::regclass
     ORDER BY 1, 2
"""
SPORT_TEAMS_SQL = """
    SELECT id::text AS id, name, normalized_name, league_id::text AS league_id, country, is_active,
           COALESCE(external_ids->>'espn', '') AS espn, external_ids, metadata
      FROM teams WHERE sport = 'mma'
"""
SPLIT_MATCHES_SQL = """
    SELECT id::text AS id, home_team_id::text AS home, away_team_id::text AS away, match_date, status
      FROM matches WHERE home_team_id = ANY(%(t)s::uuid[]) OR away_team_id = ANY(%(t)s::uuid[])
     ORDER BY match_date, id
"""
LOCK_MATCHES_SQL = "SELECT id FROM matches WHERE id = ANY(%s::uuid[]) ORDER BY id FOR UPDATE"
HOLDERS_SQL = """
    SELECT (SELECT count(*) FROM betting_recommendations r
             WHERE r.match_id = ANY(%(m)s::uuid[])
                OR r.prediction_id IN (SELECT p.id FROM predictions p WHERE p.match_id = ANY(%(m)s::uuid[]))) AS recs,
           (SELECT count(*) FROM betting_history h WHERE h.match_id = ANY(%(m)s::uuid[])) AS history,
           (SELECT count(*) FROM predictions p WHERE p.match_id = ANY(%(m)s::uuid[])) AS predictions
"""


@dataclass
class State:
    teams: dict[str, dict]  # every MMA team, by id
    split_matches: dict[str, dict]  # every bout of the split row and the new row
    holders: dict[str, int]  # recs / history / predictions on the bouts the split moves
    team_fks: tuple[tuple[str, str], ...]
    orphan_refs: dict[str, dict[str, int]]  # orphan id -> "table.column" -> count

    def holders_of(self, espn: str) -> list[str]:
        return sorted(t["id"] for t in self.teams.values() if espn and t["espn"] == espn)

    def by_normalized(self, norm: str) -> list[str]:
        return sorted(t["id"] for t in self.teams.values() if t["normalized_name"] == norm)

    def alias_claims(self, form: str, alias_form: Callable[[str], str]) -> list[str]:
        out = []
        for t in self.teams.values():
            aliases = (t.get("metadata") or {}).get("aliases")
            if isinstance(aliases, list) and any(isinstance(a, str) and alias_form(a) == form for a in aliases):
                out.append(t["id"])
        return sorted(out)


def _team_refs_sql(fks: Iterable[tuple[str, str]]) -> sql.Composed:
    """One count per foreign key to teams, for team %(t)s (identifiers quoted)."""
    parts = [
        sql.SQL("(SELECT count(*) FROM {} x WHERE x.{} = %(t)s::uuid) AS {}").format(
            sql.Identifier(*tbl.split(".")), sql.Identifier(col), sql.Identifier(f"{tbl}.{col}")
        )
        for tbl, col in fks
    ]
    return sql.SQL("SELECT ") + sql.SQL(", ").join(parts)


def load_state(cur, lists: Lists, lock: bool = False) -> State:
    cur.execute(TEAM_FKS_SQL)
    fks = tuple((r["tbl"], r["col"]) for r in cur.fetchall())
    if ("matches", "home_team_id") not in fks or ("matches", "away_team_id") not in fks:
        raise GuardError("pg_constraint lists no matches -> teams foreign keys: not the schema this tool knows")
    cur.execute(SPORT_TEAMS_SQL)
    teams = {r["id"]: dict(r) for r in cur.fetchall()}
    s = lists.split
    move_ids = [b.match_id for b in s.of("move")]
    if lock:
        cur.execute(LOCK_MATCHES_SQL, (sorted(b.match_id for b in s.bouts),))
    cur.execute(SPLIT_MATCHES_SQL, {"t": [s.team_id, s.new_team_id]})
    split_matches = {r["id"]: dict(r) for r in cur.fetchall()}
    cur.execute(HOLDERS_SQL, {"m": move_ids})
    holders = dict(cur.fetchone())
    orphan_refs: dict[str, dict[str, int]] = {}
    for o in lists.orphans:
        cur.execute(_team_refs_sql(fks), {"t": o.team_id})
        orphan_refs[o.team_id] = {k: int(v) for k, v in dict(cur.fetchone()).items()}
    return State(teams, split_matches, {k: int(v) for k, v in holders.items()}, fks, orphan_refs)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def _aliases(t: Optional[dict]) -> Optional[list[str]]:
    """metadata.aliases as a list of strings ([] when absent); None when malformed."""
    a = ((t or {}).get("metadata") or {}).get("aliases")
    if a is None:
        return []
    if isinstance(a, list) and all(isinstance(x, str) for x in a):
        return list(a)
    return None


@dataclass
class SplitPlan:
    action: str  # split | done | refuse
    reason: str
    moves: list[dict] = field(default_factory=list)  # match_id, side, kickoff, status, other
    observed_espn: str = ""
    observed_metadata: dict = field(default_factory=dict)
    new_row: dict = field(default_factory=dict)
    predictions: int = 0
    recs: int = 0


@dataclass
class UnitPlan:
    kind: str  # alias | orphan | ids
    team_id: str
    action: str  # alias: add|done|refuse; orphan: delete|done|refuse; ids: stamp|done|refuse
    reason: str
    detail: dict = field(default_factory=dict)


@dataclass
class RunPlan:
    split: SplitPlan
    aliases: list[UnitPlan]
    orphans: list[UnitPlan]
    ids: list[UnitPlan]

    def units(self) -> list[UnitPlan]:
        return self.aliases + self.orphans + self.ids


def _side_of(m: dict, team_id: str) -> Optional[str]:
    if m["home"] == team_id and m["away"] != team_id:
        return "home"
    if m["away"] == team_id and m["home"] != team_id:
        return "away"
    return None


def plan_split(s: SplitList, state: State, resolvers: Resolvers) -> SplitPlan:  # noqa: C901 - one check per state
    t = state.teams.get(s.team_id)
    n = state.teams.get(s.new_team_id)
    if t is None:
        return SplitPlan("refuse", f"split row {s.team_id} ({s.team_name!r}) is not an MMA team row")
    if t["name"] != s.team_name:
        return SplitPlan("refuse", f"split row {s.team_id} is named {t['name']!r}, the list says {s.team_name!r}")
    listed = {b.match_id: b for b in s.bouts}
    on_t = {mid for mid, m in state.split_matches.items() if s.team_id in (m["home"], m["away"])}
    on_n = {mid for mid, m in state.split_matches.items() if s.new_team_id in (m["home"], m["away"])}
    missing = sorted(set(listed) - set(state.split_matches))
    if missing:
        return SplitPlan("refuse", f"listed bout(s) {', '.join(missing)} are not on either row any more")
    for b in s.bouts:
        m = state.split_matches[b.match_id]
        owner = s.new_team_id if (b.assign == "move" and b.match_id in on_n) else s.team_id
        if _side_of(m, owner) != b.side or m["match_date"] != b.kickoff or m["status"] != b.status:
            return SplitPlan(
                "refuse",
                f"bout {b.match_id} is not as reviewed (side {b.side}, {b.kickoff.isoformat()}, {b.status}): "
                f"{m['home']} v {m['away']}, {m['match_date']}, {m['status']}",
            )
    keep_on_n = sorted(b.match_id for b in s.of("keep") if b.match_id in on_n)
    if keep_on_n:
        return SplitPlan("refuse", f"kept bout(s) {', '.join(keep_on_n)} sit on the new row")
    holders = state.holders
    if n is not None:
        moved_on_t = sorted(b.match_id for b in s.of("move") if b.match_id in on_t)
        ok = (
            n["name"] == s.new_team_name
            and n["espn"] == s.move_athlete
            and t["espn"] == s.keep_athlete
            and not moved_on_t
            and s.team_name in (_aliases(n) or [])
        )
        if ok:
            return SplitPlan(
                "done", f"{s.new_team_name!r} holds athlete {s.move_athlete} and his {len(s.of('move'))} bouts"
            )
        return SplitPlan(
            "refuse",
            f"new row {s.new_team_id} exists but the split is not complete (name {n['name']!r}, "
            f"ESPN {n['espn'] or '-'}, split row ESPN {t['espn'] or '-'}, {len(moved_on_t)} moved bout(s) still on "
            "the split row)",
        )
    unlisted = sorted(on_t - set(listed))
    if unlisted:
        return SplitPlan(
            "refuse", f"the split row has bout(s) the list does not assign: {', '.join(unlisted)} — re-review"
        )
    if t["espn"] not in ("", s.keep_athlete):
        return SplitPlan("refuse", f"the split row carries ESPN athlete {t['espn']}, not the kept {s.keep_athlete}")
    for athlete, own in ((s.keep_athlete, s.team_id), (s.move_athlete, None)):
        others = [h for h in state.holders_of(athlete) if h != own]
        if others:
            return SplitPlan("refuse", f"ESPN athlete {athlete} is already on team(s) {', '.join(others)}")
    new_norm = resolvers.normalize(s.new_team_name)
    if state.by_normalized(new_norm):
        return SplitPlan("refuse", f"a row already carries the new name {s.new_team_name!r}")
    claims = state.alias_claims(resolvers.alias_form(s.team_name), resolvers.alias_form)
    if claims:
        return SplitPlan("refuse", f"alias {s.team_name!r} is already claimed by {', '.join(claims)}")
    if _aliases(t) is None:
        return SplitPlan("refuse", "the split row's metadata.aliases is malformed")
    if holders["recs"] or holders["history"]:
        return SplitPlan(
            "refuse",
            f"{holders['recs']} rec(s) and {holders['history']} betting_history row(s) reference a bout to move — "
            "a person decides what happens to them",
            recs=holders["recs"] + holders["history"],
        )
    moves = []
    for b in s.of("move"):
        m = state.split_matches[b.match_id]
        moves.append(
            {
                "match_id": b.match_id,
                "side": b.side,
                "kickoff": m["match_date"],
                "status": m["status"],
                "other": m["away"] if b.side == "home" else m["home"],
            }
        )
    meta = dict(t.get("metadata") or {})
    new_row = {
        "id": s.new_team_id,
        "name": s.new_team_name,
        "normalized_name": new_norm,
        "league_id": t.get("league_id"),
        "country": t.get("country"),
        "is_active": t.get("is_active"),
        "espn": s.move_athlete,
        "aliases": [s.team_name],
    }
    return SplitPlan(
        "split",
        f"move {len(moves)} bout(s) of athlete {s.move_athlete} to {s.new_team_name!r}; "
        f"{s.team_name!r} keeps athlete {s.keep_athlete}",
        moves=sorted(moves, key=lambda x: x["match_id"]),
        observed_espn=t["espn"],
        observed_metadata=meta,
        new_row=new_row,
        predictions=holders["predictions"],
        recs=0,
    )


def plan_alias(a: AliasRow, state: State, resolvers: Resolvers) -> UnitPlan:
    t = state.teams.get(a.team_id)
    if t is None or t["name"] != a.team_name:
        return UnitPlan("alias", a.team_id, "refuse", f"{a.team_name!r} ({a.team_id}) is not an MMA row of that name")
    aliases = _aliases(t)
    if aliases is None:
        return UnitPlan("alias", a.team_id, "refuse", f"{a.team_name!r}'s metadata.aliases is malformed")
    form = resolvers.alias_form(a.alias)
    named = [x for x in state.by_normalized(resolvers.normalize(a.alias)) if x != a.team_id]
    folded = [
        x["id"]
        for x in state.teams.values()
        if x["id"] != a.team_id and resolvers.alias_form(x["normalized_name"]) == form
    ]
    if named or folded:
        return UnitPlan(
            "alias",
            a.team_id,
            "refuse",
            f"row(s) {', '.join(sorted(set(named + folded)))} already carry the name {a.alias!r} — a merge, "
            "not an alias",
        )
    claims = [x for x in state.alias_claims(form, resolvers.alias_form) if x != a.team_id]
    if claims:
        return UnitPlan("alias", a.team_id, "refuse", f"alias {a.alias!r} is already claimed by {', '.join(claims)}")
    holders = [h for h in state.holders_of(a.espn_athlete_id) if h != a.team_id]
    if holders or t["espn"] not in ("", a.espn_athlete_id):
        return UnitPlan(
            "alias",
            a.team_id,
            "refuse",
            f"ESPN athlete {a.espn_athlete_id} is on {holders or '-'}; the row carries {t['espn'] or '-'}",
        )
    if any(resolvers.alias_form(x) == form for x in aliases):
        return UnitPlan("alias", a.team_id, "done", f"{a.alias!r} is already an alias of {a.team_name!r}")
    return UnitPlan(
        "alias",
        a.team_id,
        "add",
        f"{a.alias!r} becomes an alias of {a.team_name!r}",
        {"alias": a.alias, "observed_aliases": aliases, "aliases": aliases + [a.alias]},
    )


def plan_orphan(o: OrphanRow, state: State) -> UnitPlan:
    t = state.teams.get(o.team_id)
    if t is None:
        return UnitPlan("orphan", o.team_id, "done", f"{o.team_name!r} is gone")
    if t["name"] != o.team_name:
        return UnitPlan("orphan", o.team_id, "refuse", f"{o.team_id} is named {t['name']!r}, not {o.team_name!r}")
    refs = {k: v for k, v in state.orphan_refs.get(o.team_id, {}).items() if v}
    if refs:
        return UnitPlan("orphan", o.team_id, "refuse", f"{o.team_name!r} is referenced: {refs}")
    if t["espn"] or _aliases(t):
        return UnitPlan(
            "orphan", o.team_id, "refuse", f"{o.team_name!r} carries an ESPN id or aliases now — an identity, re-review"
        )
    return UnitPlan(
        "orphan",
        o.team_id,
        "delete",
        f"delete bout-less {o.team_name!r}",
        {"name": t["name"], "normalized_name": t["normalized_name"]},
    )


def plan_id(r: IdRow, state: State) -> UnitPlan:
    t = state.teams.get(r.team_id)
    if t is None or t["name"] != r.team_name:
        return UnitPlan("ids", r.team_id, "refuse", f"{r.team_name!r} ({r.team_id}) is not an MMA row of that name")
    others = [h for h in state.holders_of(r.espn_athlete_id) if h != r.team_id]
    if others:
        return UnitPlan(
            "ids",
            r.team_id,
            "refuse",
            f"ESPN athlete {r.espn_athlete_id} is already on team(s) {', '.join(others)} — a second row for "
            f"{r.team_name!r} minted since the review? merge it first",
        )
    if t["espn"] == r.espn_athlete_id:
        return UnitPlan("ids", r.team_id, "done", "", {"espn": r.espn_athlete_id})
    if t["espn"]:
        return UnitPlan(
            "ids", r.team_id, "refuse", f"{r.team_name!r} carries ESPN id {t['espn']}, the list {r.espn_athlete_id}"
        )
    return UnitPlan("ids", r.team_id, "stamp", "", {"espn": r.espn_athlete_id})


def plan_run(lists: Lists, state: State, resolvers: Resolvers) -> RunPlan:
    return RunPlan(
        plan_split(lists.split, state, resolvers),
        [plan_alias(a, state, resolvers) for a in lists.aliases],
        [plan_orphan(o, state) for o in lists.orphans],
        [plan_id(r, state) for r in lists.ids],
    )


def summarize(plan: RunPlan) -> dict[str, int]:
    out = {k: 0 for k in PLAN_KEYS}
    sp = plan.split
    out[{"split": "split", "done": "split_done", "refuse": "split_refuse"}[sp.action]] += 1
    out["bouts_moved"] = len(sp.moves)
    out["predictions_on_moved"] = sp.predictions if sp.action == "split" else 0
    out["recs_on_moved"] = sp.recs
    keys = {
        "alias": {"add": "alias_add", "done": "alias_done", "refuse": "alias_refuse"},
        "orphan": {"delete": "orphan_delete", "done": "orphan_done", "refuse": "orphan_refuse"},
        "ids": {"stamp": "ids_stamp", "done": "ids_done", "refuse": "ids_refuse"},
    }
    for u in plan.units():
        out[keys[u.kind][u.action]] += 1
    return out


def refusals(plan: RunPlan) -> list[str]:
    out = [f"split {plan.split.reason}"] if plan.split.action == "refuse" else []
    return out + [f"{u.kind} {u.team_id}: {u.reason}" for u in plan.units() if u.action == "refuse"]


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


def plan_document(plan: RunPlan, lists: Lists, mode: str) -> dict:
    """The plan as data. The digest covers every decision and row id, and for
    the athlete ids the id each listed row ENDS UP with (stamp and done alike:
    a sweep stamping a row between the dry run and the apply changes nothing)."""
    sp, s = plan.split, lists.split
    split = {
        "team_id": s.team_id,
        "team_name": s.team_name,
        "new_team_id": s.new_team_id,
        "new_team_name": s.new_team_name,
        "keep_athlete": s.keep_athlete,
        "move_athlete": s.move_athlete,
        "keep_bouts": sorted(b.match_id for b in s.of("keep")),
        "move_bouts": sorted(b.match_id for b in s.of("move")),
        "action": sp.action,
        "reason": sp.reason,
        "moves": sp.moves,
        "observed_espn": sp.observed_espn,
        "observed_metadata": sp.observed_metadata,
        "new_row": sp.new_row,
    }
    units = {
        kind: [{"team_id": u.team_id, "action": u.action, "reason": u.reason, **u.detail} for u in getattr(plan, attr)]
        for kind, attr in (("aliases", "aliases"), ("orphans", "orphans"), ("ids", "ids"))
    }
    names = {r.team_id: r.team_name for r in lists.ids}
    alias_names = {a.team_id: (a.team_name, a.espn_athlete_id) for a in lists.aliases}
    for e in units["ids"]:
        e["team_name"] = names[e["team_id"]]
    for e in units["aliases"]:
        e["team_name"], e["espn_athlete_id"] = alias_names[e["team_id"]]
        e.setdefault("alias", next(a.alias for a in lists.aliases if a.team_id == e["team_id"]))
    structural = {
        "split": {k: v for k, v in split.items() if k != "reason"},
        "aliases": [{k: v for k, v in e.items() if k != "reason"} for e in units["aliases"]],
        "orphans": [{k: v for k, v in e.items() if k != "reason"} for e in units["orphans"]],
        "ids_target": sorted([e["team_id"], e["espn"]] for e in units["ids"] if e["action"] in ("stamp", "done")),
        "ids_refused": sorted(e["team_id"] for e in units["ids"] if e["action"] == "refuse"),
    }
    return {
        "tool": "mma_identity_cleanup",
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "digest": hashlib.sha256(_j(structural).encode()).hexdigest(),
        "summary": summarize(plan),
        "split": split,
        **units,
    }


def plan_difference(expected: dict, doc: dict) -> Optional[str]:
    """None when `doc` is the plan `expected` (a dry run's plan file) showed,
    else what differs. Pure."""
    if expected.get("tool") != "mma_identity_cleanup" or "digest" not in expected:
        return "the expected plan file is not an mma_identity_cleanup plan"
    if expected["digest"] == doc["digest"]:
        return None
    if expected.get("split", {}).get("action") != doc["split"]["action"]:
        return (
            f"the split is {doc['split']['action']}, the expected plan says {expected.get('split', {}).get('action')}"
        )
    for section in ("aliases", "orphans", "ids"):
        want = {e["team_id"]: e for e in expected.get(section, [])}
        for e in doc[section]:
            w = want.get(e["team_id"], {})
            if section == "ids":
                if w.get("espn") != e.get("espn") or (w.get("action") == "refuse") != (e["action"] == "refuse"):
                    return f"ids {e['team_id']}: the planned athlete id differs from the expected plan"
            elif w.get("action") != e["action"]:
                return f"{section} {e['team_id']}: {e['action']}, the expected plan says {w.get('action')}"
    return "the plan digest differs from the expected plan"


CSV_FIELDS = ("unit", "action", "table", "row_id", "before", "after", "reason")


def csv_rows(plan: RunPlan, lists: Lists) -> list[dict]:
    out: list[dict] = []
    sp, s = plan.split, lists.split
    out.append(
        {
            "unit": "split",
            "action": sp.action,
            "table": "teams",
            "row_id": s.team_id,
            "before": _j({"espn": sp.observed_espn}),
            "after": _j({"espn": s.keep_athlete}),
            "reason": sp.reason,
        }
    )
    if sp.action == "split":
        out.append(
            {
                "unit": "split",
                "action": "insert_team",
                "table": "teams",
                "row_id": s.new_team_id,
                "after": _j(sp.new_row),
            }
        )
        for m in sp.moves:
            out.append(
                {
                    "unit": "split",
                    "action": f"repoint_{m['side']}",
                    "table": "matches",
                    "row_id": m["match_id"],
                    "before": s.team_id,
                    "after": s.new_team_id,
                }
            )
    names = {r.team_id: r.team_name for r in lists.ids}
    names.update({a.team_id: a.team_name for a in lists.aliases})
    names.update({o.team_id: o.team_name for o in lists.orphans})
    for u in plan.units():
        out.append(
            {
                "unit": u.kind,
                "action": u.action,
                "table": "teams",
                "row_id": u.team_id,
                "before": _j(u.detail.get("observed_aliases")) if u.kind == "alias" else "",
                "after": _j(u.detail.get("aliases") if u.kind == "alias" else u.detail.get("espn", "")),
                "reason": u.reason or names.get(u.team_id, ""),
            }
        )
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
        json.dump(doc, fh, indent=1, sort_keys=True, ensure_ascii=False, default=_jsonable)


def write_archive(cur, path: Path, plan: RunPlan, lists: Lists) -> int:
    """gzip JSONL, written before the first write: the pre-image of every row
    the run changes or deletes (the split row, every moved bout, every aliased,
    deleted and stamped team row) and the row it inserts. To reverse: restore
    the pre-images (bouts first), then delete the inserted row."""
    team_ids = set()
    match_ids: list[str] = []
    if plan.split.action == "split":
        team_ids.add(lists.split.team_id)
        match_ids = [m["match_id"] for m in plan.split.moves]
    team_ids |= {u.team_id for u in plan.units() if u.action in ("add", "delete", "stamp")}
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "xt", encoding="utf-8") as fh:

        def emit(table: str, role: str, rows: Iterable[dict]) -> None:
            nonlocal n
            for r in rows:
                fh.write(_j({"table": table, "role": role, "row": dict(r)}) + "\n")
                n += 1

        cur.execute("SELECT * FROM teams WHERE id = ANY(%s::uuid[]) ORDER BY id", (sorted(team_ids),))
        emit("teams", "pre_image", cur.fetchall())
        if match_ids:
            cur.execute("SELECT * FROM matches WHERE id = ANY(%s::uuid[]) ORDER BY id", (sorted(match_ids),))
            emit("matches", "pre_image", cur.fetchall())
            emit("teams", "inserted", [plan.split.new_row])
    return n


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

INSERT_TEAM_SQL = """
    INSERT INTO teams (id, name, normalized_name, league_id, country, sport, external_ids, metadata, is_active)
    SELECT %(id)s::uuid, %(name)s, %(normalized_name)s, %(league_id)s::uuid, %(country)s, 'mma',
           jsonb_build_object('espn', %(espn)s::text), %(metadata)s::jsonb, %(is_active)s
     WHERE NOT EXISTS (SELECT 1 FROM teams o
                        WHERE o.sport = 'mma' AND o.external_ids @> jsonb_build_object('espn', %(espn)s::text))
    RETURNING id
"""
STAMP_SPLIT_ROW_SQL = """
    UPDATE teams t
       SET external_ids = COALESCE(t.external_ids, '{}'::jsonb) || jsonb_build_object('espn', %(espn)s::text),
           metadata = %(metadata)s::jsonb
     WHERE t.id = %(id)s::uuid AND t.sport = 'mma' AND t.name = %(name)s
       AND COALESCE(t.external_ids->>'espn', '') = %(observed_espn)s
       AND COALESCE(t.metadata, '{}'::jsonb) = %(observed_metadata)s::jsonb
       AND NOT EXISTS (SELECT 1 FROM teams o WHERE o.sport = 'mma' AND o.id <> t.id
                         AND o.external_ids @> jsonb_build_object('espn', %(espn)s::text))
    RETURNING t.id
"""
REPOINT_SQL = {
    "home": """
        UPDATE matches SET home_team_id = %(new)s::uuid
         WHERE id = %(id)s::uuid AND home_team_id = %(old)s::uuid AND away_team_id = %(other)s::uuid
           AND match_date = %(kickoff)s AND status = %(status)s
        RETURNING id""",
    "away": """
        UPDATE matches SET away_team_id = %(new)s::uuid
         WHERE id = %(id)s::uuid AND away_team_id = %(old)s::uuid AND home_team_id = %(other)s::uuid
           AND match_date = %(kickoff)s AND status = %(status)s
        RETURNING id""",
}
ALIAS_SQL = """
    UPDATE teams SET metadata = jsonb_set(COALESCE(metadata, '{}'::jsonb), '{aliases}', %(aliases)s::jsonb)
     WHERE id = %(id)s::uuid AND sport = 'mma' AND name = %(name)s
       AND COALESCE(metadata->'aliases', '[]'::jsonb) = %(observed)s::jsonb
    RETURNING id
"""


def _orphan_delete_sql(fks: Iterable[tuple[str, str]]) -> sql.Composed:
    guards = [
        sql.SQL(" AND NOT EXISTS (SELECT 1 FROM {} x WHERE x.{} = t.id)").format(
            sql.Identifier(*tbl.split(".")), sql.Identifier(col)
        )
        for tbl, col in fks
    ]
    return (
        sql.SQL(
            "DELETE FROM teams t WHERE t.id = %(id)s::uuid AND t.sport = 'mma' AND t.name = %(name)s"
            " AND t.normalized_name = %(normalized_name)s AND NOT (COALESCE(t.external_ids, '{}'::jsonb) ? 'espn')"
        )
        + sql.SQL("").join(guards)
        + sql.SQL(" RETURNING t.id")
    )


def _expect(cur, query: Any, params: Any, what: str) -> None:
    cur.execute(query, params)
    got = len(cur.fetchall())
    if got != 1:
        raise GuardError(f"{what}: touched {got} row(s), expected 1 — the row changed since the plan")


def apply_writes(cur, plan: RunPlan, lists: Lists, state: State, resolvers: Resolvers) -> dict[str, int]:
    done: dict[str, int] = defaultdict(int)
    sp, s = plan.split, lists.split
    now = datetime.now(timezone.utc).isoformat()
    if sp.action == "split":
        moved = [m["match_id"] for m in sp.moves]
        nr = sp.new_row
        meta = {
            "aliases": nr["aliases"],
            "split_from": {
                "team_id": s.team_id,
                "name": s.team_name,
                "espn_athlete_id": s.move_athlete,
                "moved_matches": moved,
                "split_at": now,
                "source": f"{SOURCE} (scripts/data/{SPLIT_PATH.name})",
            },
        }
        _expect(cur, INSERT_TEAM_SQL, {**nr, "metadata": _j(meta)}, f"insert the new row {nr['id']}")
        done["teams_inserted"] += 1
        t_meta = dict(sp.observed_metadata)
        t_meta["split_into"] = list(t_meta.get("split_into") or []) + [
            {
                "team_id": s.new_team_id,
                "name": s.new_team_name,
                "espn_athlete_id": s.move_athlete,
                "moved_matches": moved,
                "kept_espn_athlete_id": s.keep_athlete,
                "split_at": now,
                "source": f"{SOURCE} (scripts/data/{SPLIT_PATH.name})",
            }
        ]
        _expect(
            cur,
            STAMP_SPLIT_ROW_SQL,
            {
                "id": s.team_id,
                "name": s.team_name,
                "espn": s.keep_athlete,
                "metadata": _j(t_meta),
                "observed_espn": sp.observed_espn,
                "observed_metadata": _j(sp.observed_metadata),
            },
            f"stamp the split row {s.team_id}",
        )
        done["split_row_stamped"] += 1
        for m in sp.moves:
            _expect(
                cur,
                REPOINT_SQL[m["side"]],
                {"id": m["match_id"], "new": s.new_team_id, "old": s.team_id, **m},
                f"re-point bout {m['match_id']}",
            )
            done["bouts_moved"] += 1
    for u in plan.aliases:
        if u.action == "add":
            name = next(a.team_name for a in lists.aliases if a.team_id == u.team_id)
            _expect(
                cur,
                ALIAS_SQL,
                {
                    "id": u.team_id,
                    "name": name,
                    "aliases": _j(u.detail["aliases"]),
                    "observed": _j(u.detail["observed_aliases"]),
                },
                f"alias on {u.team_id}",
            )
            done["aliases_added"] += 1
    for u in plan.orphans:
        if u.action == "delete":
            _expect(
                cur, _orphan_delete_sql(state.team_fks), {"id": u.team_id, **u.detail}, f"delete orphan {u.team_id}"
            )
            done["orphans_deleted"] += 1
    for u in plan.ids:
        if u.action == "stamp":
            _expect(cur, resolvers.stamp_sql, {"id": u.team_id, "espn": u.detail["espn"]}, f"stamp {u.team_id}")
            done["ids_stamped"] += 1
    return dict(done)


# ---------------------------------------------------------------------------
# After-state
# ---------------------------------------------------------------------------

AFTER_KEYS = (
    "split_missing",
    "bouts_misplaced",
    "split_resolver_misses",
    "alias_missing",
    "alias_resolver_misses",
    "orphans_left",
    "ids_missing",
    "ids_resolver_misses",
    "athlete_ids_on_two_rows",
    "ids_expected",
    "ids_recorded",
)
AFTER_INFO = ("ids_expected", "ids_recorded")


def check_after(cur, doc: dict, resolvers: Resolvers) -> dict[str, int]:
    """The after-state the plan implies (read-only): counts that must be 0,
    plus ids_expected / ids_recorded."""
    out = {k: 0 for k in AFTER_KEYS}
    cur.execute(SPORT_TEAMS_SQL)
    teams = {r["id"]: dict(r) for r in cur.fetchall()}
    sp = doc["split"]
    if sp["action"] in ("split", "done"):
        t, n = teams.get(sp["team_id"]), teams.get(sp["new_team_id"])
        if (
            t is None
            or n is None
            or t["espn"] != sp["keep_athlete"]
            or n["espn"] != sp["move_athlete"]
            or n["name"] != sp["new_team_name"]
            or t["name"] != sp["team_name"]
            or sp["team_name"] not in (_aliases(n) or [])
        ):
            out["split_missing"] = 1
        cur.execute(SPLIT_MATCHES_SQL, {"t": [sp["team_id"], sp["new_team_id"]]})
        rows = {r["id"]: r for r in cur.fetchall()}
        for mid in sp["move_bouts"]:
            m = rows.get(mid)
            out["bouts_misplaced"] += int(m is None or sp["new_team_id"] not in (m["home"], m["away"]))
        for mid in sp["keep_bouts"]:
            m = rows.get(mid)
            out["bouts_misplaced"] += int(m is None or sp["team_id"] not in (m["home"], m["away"]))
        for name, espn, want in (
            (sp["team_name"], sp["keep_athlete"], sp["team_id"]),
            (sp["team_name"], sp["move_athlete"], sp["new_team_id"]),
            (sp["team_name"], None, sp["team_id"]),
            (sp["new_team_name"], None, sp["new_team_id"]),
        ):
            out["split_resolver_misses"] += int(str(resolvers.find_team(cur, name, espn) or "") != want)
    for e in doc.get("aliases", []):
        if e["action"] not in ("add", "done"):
            continue
        t = teams.get(e["team_id"])
        form = resolvers.alias_form(e["alias"])
        if t is None or not any(resolvers.alias_form(a) == form for a in (_aliases(t) or [])):
            out["alias_missing"] += 1
        for espn in (None, e["espn_athlete_id"]):
            out["alias_resolver_misses"] += int(str(resolvers.find_team(cur, e["alias"], espn) or "") != e["team_id"])
    for e in doc.get("orphans", []):
        if e["action"] in ("delete", "done"):
            out["orphans_left"] += int(e["team_id"] in teams)
    for e in doc.get("ids", []):
        if e["action"] not in ("stamp", "done"):
            continue
        out["ids_expected"] += 1
        t = teams.get(e["team_id"])
        if t is None or t["espn"] != e["espn"]:
            out["ids_missing"] += 1
            continue
        out["ids_recorded"] += 1
        out["ids_resolver_misses"] += int(
            str(resolvers.find_team(cur, e["team_name"], e["espn"]) or "") != e["team_id"]
        )
    per_id: dict[str, int] = defaultdict(int)
    for t in teams.values():
        if t["espn"]:
            per_id[t["espn"]] += 1
    out["athlete_ids_on_two_rows"] = sum(1 for v in per_id.values() if v > 1)
    return out


def after_state_ok(out: dict[str, int]) -> bool:
    return out.get("ids_recorded") == out.get("ids_expected") and all(
        v == 0 for k, v in out.items() if k not in AFTER_INFO
    )


def after_line(out: dict[str, int]) -> str:
    return " ".join(f"{k}={out.get(k, 0)}" for k in AFTER_KEYS)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def default_out(kind: str) -> Path:
    base = Path(os.environ.get("BACKUP_LOCAL_DIR", "/app/backups"))
    ext = {"csv": "csv", "archive": "jsonl.gz"}[kind]
    return base / f"mma_identity_cleanup_{_stamp()}.{ext}"


def log_plan(plan: RunPlan, lists: Lists) -> None:
    sp = plan.split
    tag = f"[split] {lists.split.team_name!r}"
    if sp.action == "refuse":
        logger.error("%s: REFUSED — %s", tag, sp.reason)
    else:
        logger.info(
            "%s: %s — %s (predictions on the moved bouts: %d, staying on them)",
            tag,
            sp.action,
            sp.reason,
            sp.predictions,
        )
    for u in plan.aliases + plan.orphans:
        (logger.error if u.action == "refuse" else logger.info)(
            "[%s] %s: %s — %s", u.kind, u.team_id, u.action, u.reason
        )
    for u in plan.ids:
        if u.action == "refuse":
            logger.error("[ids] %s: REFUSED — %s", u.team_id, u.reason)


def run(
    conn,
    apply_changes: bool,
    lists: Lists,
    out_csv: Path,
    archive: Path,
    resolvers: Optional[Resolvers],
    resolver_reason: Optional[str] = None,
    plan_out: Optional[Path] = None,
    expect_plan: Optional[dict] = None,
) -> int:
    """Exit status: 0 done (dry run: nothing refused); 1 a refusal in the dry
    run, or the apply rolled back after it began writing; 2 refused before any
    write (no resolver, a plan that is not the expected one, a refusal)."""
    if resolvers is None or resolver_reason:
        logger.error("REFUSED: %s — nothing read or written", resolver_reason)
        return 2
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        try:
            if apply_changes:
                cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                cur.execute("SET LOCAL lock_timeout = %s", (LOCK_TIMEOUT,))
                # No sweep stamps or inserts a team while this runs (seconds); taken before the snapshot.
                cur.execute("LOCK TABLE teams IN SHARE ROW EXCLUSIVE MODE")
            else:
                cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            state = load_state(cur, lists, lock=apply_changes)
        except (psycopg2.Error, GuardError) as e:
            conn.rollback()
            logger.error("REFUSED: reading the state failed (%s) — nothing written", str(e).strip().splitlines()[0])
            return 2
        plan = plan_run(lists, state, resolvers)
        doc = plan_document(plan, lists, "apply" if apply_changes else "dry-run")
        write_csv(out_csv, csv_rows(plan, lists))
        plan_path = plan_out or out_csv.with_suffix(".plan.json")
        write_plan(plan_path, doc)
        log_plan(plan, lists)
        logger.info("Audit CSV: %s", out_csv)
        logger.info("PLAN_FILE %s", plan_path)
        logger.info("%s", plan_line(doc["summary"]))
        logger.info("PLAN_DIGEST %s", doc["digest"])
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
            n = write_archive(cur, archive, plan, lists)
            logger.info("Archived %d row(s) to %s", n, archive)
            written = apply_writes(cur, plan, lists, state, resolvers)
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


def run_check_after(conn, plan_path: Path, resolvers: Optional[Resolvers], resolver_reason: Optional[str]) -> int:
    if resolvers is None or resolver_reason:
        logger.error("REFUSED: %s", resolver_reason)
        return 2
    doc = json.loads(plan_path.read_text(encoding="utf-8"))
    if doc.get("tool") != "mma_identity_cleanup":
        logger.error("REFUSED: %s is not an mma_identity_cleanup plan file", plan_path)
        return 2
    if doc["split"]["action"] == "refuse" or any(
        e["action"] == "refuse" for k in ("aliases", "orphans", "ids") for e in doc.get(k, [])
    ):
        logger.error("REFUSED: %s plans a refusal — it was never applied; no after-state to check", plan_path)
        return 2
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        out = check_after(cur, doc, resolvers)
    conn.rollback()
    logger.info("AFTER %s", after_line(out))
    if after_state_ok(out):
        logger.info(
            "AFTER-STATE OK: the split, %d alias(es), %d orphan(s) and %d/%d athlete ids hold and resolve",
            len(doc.get("aliases", [])),
            len(doc.get("orphans", [])),
            out["ids_recorded"],
            out["ids_expected"],
        )
        return 0
    logger.error("AFTER-STATE FAILED: every count above must be 0 and ids_recorded=ids_expected")
    return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="Write (default: a read-only dry run).")
    p.add_argument("--expect-plan", type=Path, default=None, help="--apply only if the plan equals this dry run's.")
    p.add_argument("--split", type=Path, default=SPLIT_PATH, help=argparse.SUPPRESS)
    p.add_argument("--aliases", type=Path, default=ALIAS_PATH, help=argparse.SUPPRESS)
    p.add_argument("--orphans", type=Path, default=ORPHAN_PATH, help=argparse.SUPPRESS)
    p.add_argument("--ids", type=Path, default=IDS_PATH, help=argparse.SUPPRESS)
    p.add_argument("--out-csv", type=Path, default=None, help="Audit CSV (default $BACKUP_LOCAL_DIR or /app/backups).")
    p.add_argument("--plan-out", type=Path, default=None, help="Plan file (default: the audit CSV's .plan.json).")
    p.add_argument("--archive", type=Path, default=None, help="gzip JSONL archive written before --apply writes.")
    p.add_argument("--check-after", type=Path, default=None, help="Read-only: check the after-state of a plan file.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    resolvers, reason = load_resolvers()
    if args.check_after:
        conn = psycopg2.connect(args.database_url)
        try:
            return run_check_after(conn, args.check_after, resolvers, reason)
        finally:
            conn.close()
    if args.expect_plan and not args.apply:
        logger.error("REFUSED: --expect-plan only goes with --apply")
        return 2
    try:
        lists = load_lists(args.split, args.aliases, args.orphans, args.ids)
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
            lists,
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
