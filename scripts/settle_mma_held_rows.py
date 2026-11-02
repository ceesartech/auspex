"""Settle the two UFC rows run B held for the user's decision: record Melissa
Gatto v Ernesta Kareckaite (UFC 2026-10-10) as the draw it was, and delete
Valesca Machado v Melissa Amaya (UFC 2026-09-26), a matchup ESPN replaced.

THE DECISION
------------
Run B of scripts/ops/repair_stranded_recs_mma.sh left two stale UFC rows alone
for the user (scripts/data/mma_stale_rows_hold_20261101.csv). The user,
2026-11-02: "We can just make the system record it? But we can delete it."
  * Gatto v Kareckaite was fought and ended with no winner. fetch_upcoming
    now records MMA draws and no contests (finished, NULL scores,
    metadata.no_winner; grading_outcomes.NO_WINNER_KEY), so this row is
    recorded through that same code path. The pipeline's results sweep looks
    3 days back, so it never reaches a bout of 2026-10-10 on its own.
  * Valesca Machado v Melissa Amaya is deleted: ESPN kept the competition but
    replaced Valesca Machado with Tina Black five days before the card, and
    the finished row Tina Black v Melissa Amaya is the bout that was fought.

THE LIST (scripts/data/mma_held_rows_settle_20261102.csv: reviewed, with the
evidence; the tool touches nothing else and refuses any other row)
  record_no_winner  the row's ESPN competition is fetched from ESPN's live
                    scoreboard and written by fetch_upcoming._record_result,
                    the 15-minute sweep's own function (ESPN's core API must
                    confirm the result's kind, as in the sweep); then
                    grade_completed_matches.grade_match grades it: its
                    predictions are recorded 'draw' with is_correct NULL, an
                    open rec would be voided ('void: draw', profit 0).
  delete_replaced   ESPN's scoreboard must list the competition with the
                    replacing fighter and not the replaced one, and the
                    replacement row must be finished at the same kickoff
                    against the same opponent. The row and every row under it
                    (predictions, features_cache, any odds or snapshots) are
                    archived and deleted; nothing moves onto the replacement
                    row (its prices priced another matchup). Then the
                    replaced fighter's team row, archived, when no other bout
                    references it. A rec or betting_history row referencing
                    the row refuses the run: those foreign keys do not
                    cascade, and a rec must keep its match.
  prediction_ids pins every prediction on each row.

WHAT A RUN DOES (ONE REPEATABLE READ transaction: any failure rolls back all of it)
  1. every foreign key to matches and teams is read from pg_constraint; one
     the tool does not know refuses (merge_mma_fighters.read_schema);
  2. the listed rows, the replacement row and the team row are locked FOR
     UPDATE and planned; while anything is left to write, ESPN's scoreboard
     for each row's day is checked (read only). PLAN keys:
       record              rows to record (still 'scheduled')
       record_done         rows already finished with the expected kind
       record_predictions  pinned predictions still to record
       record_recs_void    open recs to void
       delete              rows to delete
       delete_done         rows already gone
       delete_predictions / delete_odds / delete_snapshots / delete_features /
       delete_other_rows   the rows that go with them, per table
       teams_delete        replaced fighters' team rows to delete
       refuse              rows the tool refuses (any refusal: exit 2)
  3. --apply refuses (exit 2, nothing written) unless the plan equals
     --expect-plan, every key; then it writes the write-once gzip JSONL
     archive (the reversal record: every row it changes or deletes, before)
     ahead of its first write;
  4. the writes (record, grade, delete), each checked: the results path must
     write the listed row and no other (no matches or teams row minted, no
     identity refusal), the grader must record exactly the planned
     predictions and voids, each delete must remove exactly its row;
  5. inside the transaction, the after-state (--check-after's checks) and a
     re-plan of 0 must hold, and no other UFC match, prediction, team or any
     rec may have changed (fingerprints before and after);
  6. the write-once audit CSV (each changed row, before and after) is written;
     a dry run (the default) then rolls back, --apply commits.
A dry run therefore runs the whole transaction and commits nothing (the
backfill_results.py pattern): its PLAN line and audit CSV are the apply's.
Re-runnable: once done, the plan is the done plan, ESPN is not asked, and
nothing is written.

--check-after (read only): every listed row is in its after-state (the
recorded bout finished with no winner of its kind, NULL scores, the ESPN id
kept, every pinned prediction recorded with is_correct NULL, no open rec, any
rec void 'void: <kind>' with profit 0; the deleted row gone with every row
under it and its pinned predictions, the replacement row still finished, the
team row gone unless another bout references it) and the plan is the done
plan. Exit 0 when it holds, else 1.

Usage (inside the api container; scripts/ops/settle_mma_held_rows.sh pins it):
    python /app/scripts/settle_mma_held_rows.py --out-csv /app/backups/settle_dry.csv
    python /app/scripts/settle_mma_held_rows.py --apply --expect-plan record=1,record_done=0,... \\
        --out-csv /app/backups/settle_apply.csv --archive /app/backups/settle_apply.jsonl.gz
    python /app/scripts/settle_mma_held_rows.py --check-after
Exit: 0 done (or the after-state holds); 1 a failed check or write (rolled
back); 2 refused (a refusal, a plan other than --expect-plan, bad arguments).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_REPEATABLE_READ
from psycopg2.extras import RealDictCursor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_upcoming as fu  # noqa: E402
import grade_completed_matches as gcm  # noqa: E402
import merge_mma_fighters as mmf  # noqa: E402
from grading_outcomes import NO_WINNER_KEY, NO_WINNER_KINDS, NO_WINNER_VOID_RESULTS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("settle_mma_held_rows")

SCRIPTS_DIR = Path(__file__).resolve().parent
LIST_PATH = SCRIPTS_DIR / "data" / "mma_held_rows_settle_20261102.csv"
LIST_FIELDS = (
    "match_id",
    "action",
    "expect",
    "espn_event_id",
    "espn_competition_id",
    "scoreboard_date",
    "kickoff",
    "home",
    "away",
    "replaced_side",
    "replaced_team_id",
    "replaced_by",
    "replacement_match_id",
    "prediction_ids",
    "evidence",
)
SPORT = "mma"
LEAGUE_SLUG = "ufc"
LOCK_TIMEOUT = "10s"
ACTION_RECORD = "record_no_winner"
ACTION_DELETE = "delete_replaced"
DELETABLE_STATUSES = ("scheduled", "cancelled")
OPEN_REC_STATUSES = ("pending", "placed")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
DELETE_KEYS = {"predictions": "delete_predictions", "odds": "delete_odds", "odds_snapshots": "delete_snapshots"}

PLAN_KEYS = (
    "record",
    "record_done",
    "record_predictions",
    "record_recs_void",
    "delete",
    "delete_done",
    "delete_predictions",
    "delete_odds",
    "delete_snapshots",
    "delete_features",
    "delete_other_rows",
    "teams_delete",
    "refuse",
)
CSV_FIELDS = ("match_id", "action", "table", "row_id", "change", "before", "after")


class ListError(ValueError):
    """The reviewed list is malformed or unsafe to run."""


class GuardError(RuntimeError):
    """A write or a check found the database other than the plan."""


# ---------------------------------------------------------------------------
# The reviewed list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    match_id: str
    action: str
    expect: str
    espn_event_id: str
    espn_competition_id: str
    scoreboard_date: date
    kickoff: datetime
    home: str
    away: str
    replaced_side: str
    replaced_team_id: str
    replaced_by: str
    replacement_match_id: str
    prediction_ids: tuple[str, ...]

    @property
    def replaced_name(self) -> str:
        return self.home if self.replaced_side == "home" else self.away

    @property
    def opponent_name(self) -> str:
        return self.away if self.replaced_side == "home" else self.home


def _uuid(value: str, what: str) -> str:
    v = (value or "").strip().lower()
    if not UUID_RE.match(v):
        raise ListError(f"{what}: {value!r} is not a uuid")
    return v


def load_list(path: Path = LIST_PATH) -> list[Entry]:
    """The reviewed rows. Every field is checked; anything off raises ListError."""
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    if not rows:
        raise ListError(f"{path}: no rows")
    entries: list[Entry] = []
    for n, r in enumerate(rows, start=1):
        where = f"{path.name} row {n}"
        missing = [f for f in LIST_FIELDS if f not in r]
        if missing:
            raise ListError(f"{where}: missing column(s) {missing}")
        action = r["action"].strip()
        if action not in (ACTION_RECORD, ACTION_DELETE):
            raise ListError(f"{where}: unknown action {action!r}")
        expect = r["expect"].strip()
        if action == ACTION_RECORD and expect not in NO_WINNER_KINDS:
            raise ListError(f"{where}: record_no_winner expects one of {NO_WINNER_KINDS}, not {expect!r}")
        if not r["evidence"].strip():
            raise ListError(f"{where}: no evidence")
        try:
            kickoff = datetime.fromisoformat(r["kickoff"].strip())
            day = date.fromisoformat(r["scoreboard_date"].strip())
        except ValueError as e:
            raise ListError(f"{where}: {e}") from None
        if kickoff.tzinfo is None:
            raise ListError(f"{where}: kickoff {r['kickoff']!r} has no time zone")
        preds = tuple(sorted(_uuid(p, f"{where} prediction_ids") for p in r["prediction_ids"].split(";") if p.strip()))
        for f in ("espn_event_id", "espn_competition_id", "home", "away"):
            if not r[f].strip():
                raise ListError(f"{where}: empty {f}")
        side = r["replaced_side"].strip()
        if action == ACTION_DELETE:
            if side not in ("home", "away") or not r["replaced_by"].strip():
                raise ListError(f"{where}: delete_replaced needs replaced_side (home/away) and replaced_by")
            if _norm(r["replaced_by"]) in (_norm(r["home"]), _norm(r["away"])):
                raise ListError(f"{where}: replaced_by {r['replaced_by']!r} is one of the row's own fighters")
            replaced_team = _uuid(r["replaced_team_id"], f"{where} replaced_team_id")
            replacement = _uuid(r["replacement_match_id"], f"{where} replacement_match_id")
        else:
            if side or r["replaced_team_id"].strip() or r["replaced_by"].strip() or r["replacement_match_id"].strip():
                raise ListError(f"{where}: record_no_winner takes no replaced_* / replacement columns")
            replaced_team = replacement = ""
        entries.append(
            Entry(
                match_id=_uuid(r["match_id"], f"{where} match_id"),
                action=action,
                expect=expect,
                espn_event_id=r["espn_event_id"].strip(),
                espn_competition_id=r["espn_competition_id"].strip(),
                scoreboard_date=day,
                kickoff=kickoff,
                home=r["home"].strip(),
                away=r["away"].strip(),
                replaced_side=side,
                replaced_team_id=replaced_team,
                replaced_by=r["replaced_by"].strip(),
                replacement_match_id=replacement,
                prediction_ids=preds,
            )
        )
    ids = [e.match_id for e in entries]
    if len(set(ids)) != len(ids):
        raise ListError(f"{path}: a match is listed twice")
    return entries


# ---------------------------------------------------------------------------
# Reading the database
# ---------------------------------------------------------------------------

ROW_SQL = """
    SELECT m.id::text AS id, m.league_id::text AS league_id, m.status, m.match_date,
           m.home_team_id::text AS home_team_id, m.away_team_id::text AS away_team_id,
           m.home_score, m.away_score, m.metadata, m.external_ids,
           ht.name AS home_name, at.name AS away_name, l.sport, l.name AS league_name
      FROM matches m
      JOIN leagues l ON l.id = m.league_id
      JOIN teams ht ON ht.id = m.home_team_id
      JOIN teams at ON at.id = m.away_team_id
     WHERE m.id = %s
"""
LOCK_SQL = "SELECT id FROM matches WHERE id = ANY(%s::uuid[]) ORDER BY id FOR UPDATE"
LOCK_TEAMS_SQL = "SELECT id FROM teams WHERE id = ANY(%s::uuid[]) ORDER BY id FOR UPDATE"
PREDICTIONS_SQL = """
    SELECT p.id::text AS id, p.actual_outcome, p.is_correct, (p.created_at > m.match_date) AS after_kickoff
      FROM predictions p JOIN matches m ON m.id = p.match_id
     WHERE p.match_id = %s
     ORDER BY p.id
"""
RECS_SQL = """
    SELECT br.id::text AS id, br.status, br.actual_result, br.profit_loss
      FROM betting_recommendations br
     WHERE br.match_id = %(m)s::uuid
        OR br.prediction_id IN (SELECT id FROM predictions WHERE match_id = %(m)s::uuid)
     ORDER BY br.id
"""
HISTORY_SQL = "SELECT count(*) AS n FROM betting_history WHERE match_id = %s"
TEAM_SQL = "SELECT id::text AS id, name, normalized_name, sport FROM teams WHERE id = %s"


def _norm(name: Optional[str]) -> str:
    return fu.normalize_team(name or "")


@dataclass
class EntryPlan:
    entry: Entry
    refusal: str = ""
    row: Optional[dict] = None
    record: bool = False
    record_done: bool = False
    predictions: list[dict] = field(default_factory=list)  # every prediction on the row
    to_record: list[str] = field(default_factory=list)  # pinned predictions still unrecorded
    recs: list[dict] = field(default_factory=list)
    open_recs: list[str] = field(default_factory=list)
    delete: bool = False
    delete_done: bool = False
    children: dict[str, list[str]] = field(default_factory=dict)  # table -> row ids
    team: Optional[dict] = None
    team_delete: bool = False

    def refuse(self, why: str) -> "EntryPlan":
        self.refusal = why
        return self


def _children(cur, schema: mmf.Schema, match_id: str) -> dict[str, list[str]]:
    cur.execute(mmf.children_sql(schema), {"m": match_id})
    out: dict[str, list[str]] = {}
    for r in cur.fetchall():
        out.setdefault(r["tbl"], []).append(r["id"])
    return out


def _team_refs(cur, schema: mmf.Schema, team_id: str) -> dict[str, int]:
    cur.execute(mmf.team_refs_sql(schema), {"t": team_id})
    row = cur.fetchone() or {}
    return {k: int(v) for k, v in row.items() if v}


def _names_match(row: dict, home: str, away: str) -> bool:
    """The row's two fighters are these two, in either order (ESPN orders MMA
    competitors freely)."""
    return {_norm(row["home_name"]), _norm(row["away_name"])} == {_norm(home), _norm(away)}


def _espn_ids_of(row: dict) -> set[str]:
    meta, ext = row.get("metadata") or {}, row.get("external_ids") or {}
    return {str(v) for v in (ext.get("espn"), meta.get("cancelled_espn_id"), meta.get("redrawn_espn_id")) if v}


def _common(cur, e: Entry, p: EntryPlan) -> Optional[dict]:
    """The checks both actions share. Returns the row, or None (refused or gone)."""
    cur.execute(ROW_SQL, (e.match_id,))
    row = cur.fetchone()
    if row is None:
        return None
    row = dict(row)
    p.row = row
    if row["sport"] != SPORT or row["league_name"] != "UFC":
        p.refuse(f"the row is {row['sport']}/{row['league_name']}, not a UFC bout")
        return None
    if not _names_match(row, e.home, e.away):
        p.refuse(f"the row is {row['home_name']} v {row['away_name']}, the list says {e.home} v {e.away}")
        return None
    if row["match_date"] != e.kickoff:
        p.refuse(f"the row's kickoff is {row['match_date']}, the list says {e.kickoff}")
        return None
    if e.espn_competition_id not in _espn_ids_of(row):
        p.refuse(f"the row carries ESPN id(s) {sorted(_espn_ids_of(row))}, not {e.espn_competition_id}")
        return None
    cur.execute(PREDICTIONS_SQL, (e.match_id,))
    p.predictions = [dict(r) for r in cur.fetchall()]
    got = tuple(sorted(r["id"] for r in p.predictions))
    if got != e.prediction_ids:
        p.refuse(f"the row's predictions {list(got)} are not the reviewed {list(e.prediction_ids)}")
        return None
    cur.execute(RECS_SQL, {"m": e.match_id})
    p.recs = [dict(r) for r in cur.fetchall()]
    return row


def plan_record(cur, e: Entry) -> EntryPlan:
    p = EntryPlan(e)
    row = _common(cur, e, p)
    if p.refusal:
        return p
    if row is None:
        return p.refuse("the row is gone")
    if any(r["status"] not in OPEN_REC_STATUSES + ("void",) for r in p.recs):
        return p.refuse(f"a rec on the row is already settled: {[(r['id'], r['status']) for r in p.recs]}")
    p.open_recs = [r["id"] for r in p.recs if r["status"] in OPEN_REC_STATUSES]
    meta = row.get("metadata") or {}
    stored = meta.get(NO_WINNER_KEY)
    scores = (row["home_score"], row["away_score"])
    if row["status"] == "scheduled" and scores == (None, None) and stored is None:
        p.record = True
    elif row["status"] == "finished" and scores == (None, None) and stored == e.expect:
        p.record_done = True
    else:
        return p.refuse(
            f"the row is {row['status']} with scores {scores} and no_winner={stored!r}: neither still to record "
            f"nor recorded as a {e.expect}"
        )
    p.to_record = [
        r["id"]
        for r in p.predictions
        if r["actual_outcome"] is None and r["is_correct"] is None and not r["after_kickoff"]
    ]
    for r in p.predictions:
        if (
            r["id"] not in p.to_record
            and not r["after_kickoff"]
            and (r["actual_outcome"], r["is_correct"])
            != (
                e.expect,
                None,
            )
        ):
            return p.refuse(f"prediction {r['id']} holds ({r['actual_outcome']!r}, {r['is_correct']!r})")
    return p


def plan_delete(cur, e: Entry, schema: mmf.Schema) -> EntryPlan:
    p = EntryPlan(e)
    row = _common(cur, e, p)
    if p.refusal:
        return p
    cur.execute(TEAM_SQL, (e.replaced_team_id,))
    team = cur.fetchone()
    p.team = dict(team) if team else None
    if row is None:
        p.delete_done = True
    else:
        if row["status"] not in DELETABLE_STATUSES:
            return p.refuse(f"the row is {row['status']!r}, not one of {DELETABLE_STATUSES}")
        side_team = row["home_team_id"] if e.replaced_side == "home" else row["away_team_id"]
        side_name = row["home_name"] if e.replaced_side == "home" else row["away_name"]
        if side_team != e.replaced_team_id or _norm(side_name) != _norm(e.replaced_name):
            return p.refuse(f"the {e.replaced_side} side is {side_name} ({side_team}), not {e.replaced_name}")
        if p.recs:
            return p.refuse(
                f"{len(p.recs)} rec(s) reference the row or its predictions ({[r['id'] for r in p.recs]}): the "
                "foreign key does not cascade and a rec keeps its match — review them first"
            )
        cur.execute(HISTORY_SQL, (e.match_id,))
        if int(cur.fetchone()["n"]):
            return p.refuse("a betting_history row references the row")
        cur.execute(ROW_SQL, (e.replacement_match_id,))
        rep = cur.fetchone()
        opponent_team = row["away_team_id"] if e.replaced_side == "home" else row["home_team_id"]
        if rep is None:
            return p.refuse(f"the replacement row {e.replacement_match_id} is gone")
        rep = dict(rep)
        rep_teams = {rep["home_team_id"], rep["away_team_id"]}
        rep_names = {_norm(rep["home_name"]), _norm(rep["away_name"])}
        if (
            rep["status"] != "finished"
            or rep["match_date"] != e.kickoff
            or opponent_team not in rep_teams
            or e.replaced_team_id in rep_teams
            or rep_names != {_norm(e.replaced_by), _norm(e.opponent_name)}
        ):
            return p.refuse(
                f"the replacement row {rep['id']} is {rep['status']} {rep['home_name']} v {rep['away_name']} at "
                f"{rep['match_date']}, not the finished {e.replaced_by} v {e.opponent_name} at {e.kickoff}"
            )
        p.delete = True
        p.children = _children(cur, schema, e.match_id)
    if p.team is not None:
        refs = _team_refs(cur, schema, e.replaced_team_id)
        own = 1 if p.delete else 0  # the row being deleted references it once (matches)
        others = {k: v for k, v in refs.items() if k != "matches"}
        if refs.get("matches", 0) == own and not others:
            p.team_delete = True
    return p


def plan_entries(cur, entries: list[Entry], schema: mmf.Schema, lock: bool) -> list[EntryPlan]:
    """Plan every listed row; with lock, the rows (and the replacement rows and
    team rows) are locked FOR UPDATE first."""
    if lock:
        cur.execute(
            LOCK_SQL, (sorted({e.match_id for e in entries} | {e.replacement_match_id for e in entries} - {""}),)
        )
        cur.execute(LOCK_TEAMS_SQL, (sorted({e.replaced_team_id for e in entries} - {""}),))
    return [plan_record(cur, e) if e.action == ACTION_RECORD else plan_delete(cur, e, schema) for e in entries]


def summarize(plans: list[EntryPlan]) -> dict[str, int]:
    t = {k: 0 for k in PLAN_KEYS}
    for p in plans:
        if p.refusal:
            t["refuse"] += 1
            continue
        t["record"] += p.record
        t["record_done"] += p.record_done
        t["record_predictions"] += len(p.to_record)
        t["record_recs_void"] += len(p.open_recs)
        t["delete"] += p.delete
        t["delete_done"] += p.delete_done
        for table, ids in p.children.items():
            if table in DELETE_KEYS:
                t[DELETE_KEYS[table]] += len(ids)
            elif table == "features_cache":
                t["delete_features"] += len(ids)
            else:
                t["delete_other_rows"] += len(ids)
        t["teams_delete"] += p.team_delete
    return t


def is_done(totals: dict[str, int], entries: list[Entry]) -> bool:
    n_record = sum(e.action == ACTION_RECORD for e in entries)
    n_delete = len(entries) - n_record
    work = ("record", "record_predictions", "record_recs_void", "delete", "teams_delete", "refuse")
    return all(totals[k] == 0 for k in work) and totals["record_done"] == n_record and totals["delete_done"] == n_delete


def counts_text(totals: dict[str, int]) -> str:
    return " ".join(f"{k}={totals[k]}" for k in PLAN_KEYS)


def plan_line(totals: dict[str, int]) -> str:
    """THE plan line of a run: the wrapper reads the last 'PLAN k=v ...' of the log, so a run prints exactly
    one, and every other count line (the re-plan, --check-after's) uses counts_text without the prefix."""
    return "PLAN " + counts_text(totals)


def parse_expect(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in (text or "").split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        if k.strip() not in PLAN_KEYS or not v.strip().isdigit():
            raise ValueError(f"--expect-plan: bad entry {part!r} (keys: {', '.join(PLAN_KEYS)})")
        out[k.strip()] = int(v)
    if set(out) != set(PLAN_KEYS):
        raise ValueError(f"--expect-plan must give every key: missing {sorted(set(PLAN_KEYS) - set(out))}")
    return out


# ---------------------------------------------------------------------------
# ESPN (read only), while anything is left to write
# ---------------------------------------------------------------------------


def _find_competition(events: list[dict], comp_id: str):
    cfg = fu.SPORT_CONFIGS[SPORT]
    for ev in events:
        for comp, _ctx in fu._iter_event_competitions(ev, cfg):
            if str(comp.get("id")) == comp_id:
                return ev, comp
    return None, None


def check_espn(plans: list[EntryPlan], fetch=None) -> dict[str, tuple[list[dict], dict, dict]]:
    """For each row still to record or delete: ESPN's scoreboard for its day
    must list its competition as the list says. Refuses the row otherwise.
    Returns match_id -> (the day's events, the event, the competition)."""
    cfg = fu.SPORT_CONFIGS[SPORT]
    fetch = fetch or fu.fetch_day
    days: dict[date, list[dict]] = {}
    found: dict[str, tuple[list[dict], dict, dict]] = {}
    for p in plans:
        if p.refusal or not (p.record or p.delete):
            continue
        e = p.entry
        if e.scoreboard_date not in days:
            failed_before = fu.FETCH_STATS["failed"]
            days[e.scoreboard_date] = fetch(cfg, LEAGUE_SLUG, e.scoreboard_date)
            if fu.FETCH_STATS["failed"] > failed_before:
                p.refuse(f"ESPN did not serve the scoreboard of {e.scoreboard_date}")
                continue
        events = days[e.scoreboard_date]
        ev, comp = _find_competition(events, e.espn_competition_id)
        if comp is None:
            p.refuse(f"ESPN's scoreboard of {e.scoreboard_date} does not list competition {e.espn_competition_id}")
            continue
        if str(ev.get("id")) != e.espn_event_id:
            p.refuse(f"competition {e.espn_competition_id} is on ESPN event {ev.get('id')}, not {e.espn_event_id}")
            continue
        parts = fu._competition_parts(comp, cfg)
        if parts is None:
            p.refuse(f"ESPN competition {e.espn_competition_id} cannot be parsed")
            continue
        home, away, home_name, away_name, match_dt = parts[:5]
        names = {_norm(home_name), _norm(away_name)}
        if match_dt != e.kickoff:
            p.refuse(f"ESPN lists competition {e.espn_competition_id} at {match_dt}, not {e.kickoff}")
            continue
        if p.record:
            if names != {_norm(e.home), _norm(e.away)}:
                p.refuse(f"ESPN lists {home_name} v {away_name}, not {e.home} v {e.away}")
                continue
            reading = fu.scoreboard_no_winner(cfg, comp, home, away)
            if reading is None or reading.kind != e.expect:
                p.refuse(
                    f"ESPN's scoreboard reads {reading.kind if reading else 'a winner or no result'}, not {e.expect}"
                )
                continue
        else:
            # The replacing fighter is never one of the row's own (load_list), so this also says ESPN no longer
            # lists the replaced one.
            if names != {_norm(e.replaced_by), _norm(e.opponent_name)}:
                p.refuse(f"ESPN lists {home_name} v {away_name}, not {e.replaced_by} v {e.opponent_name}")
                continue
        found[e.match_id] = (events, ev, comp)
    return found


# ---------------------------------------------------------------------------
# Fingerprints: nothing outside the listed rows may change
# ---------------------------------------------------------------------------

FINGERPRINT_SQL = """
    SELECT
      (SELECT md5(COALESCE(string_agg(concat_ws('|', m.id, m.status, m.home_score, m.away_score, m.match_date,
                                                m.home_team_id, m.away_team_id, m.external_ids::text,
                                                m.metadata::text, m.updated_at), ',' ORDER BY m.id), ''))
         FROM matches m JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'mma' AND m.id <> ALL(%(rows)s::uuid[])) AS matches,
      (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'mma' AND m.id <> ALL(%(rows)s::uuid[])) AS n_matches,
      (SELECT md5(COALESCE(string_agg(concat_ws('|', p.id, p.match_id, p.actual_outcome, p.is_correct, p.updated_at),
                                      ',' ORDER BY p.id), ''))
         FROM predictions p JOIN matches m ON m.id = p.match_id JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'mma' AND p.match_id <> ALL(%(rows)s::uuid[])) AS predictions,
      (SELECT md5(COALESCE(string_agg(concat_ws('|', br.id, br.match_id, br.prediction_id, br.status,
                                                br.actual_result, br.profit_loss), ',' ORDER BY br.id), ''))
         FROM betting_recommendations br
        WHERE br.match_id <> ALL(%(rows)s::uuid[])) AS recs,
      (SELECT md5(COALESCE(string_agg(concat_ws('|', t.id, t.name, t.normalized_name,
                                                CASE WHEN t.id = ANY(%(fighters)s::uuid[]) THEN ''
                                                     ELSE t.external_ids::text || t.metadata::text END),
                                      ',' ORDER BY t.id), ''))
         FROM teams t WHERE t.sport = 'mma' AND t.id <> ALL(%(teams)s::uuid[])) AS teams,
      (SELECT count(*) FROM teams t WHERE t.sport = 'mma' AND t.id <> ALL(%(teams)s::uuid[])) AS n_teams
"""


def fingerprint(cur, plans: list[EntryPlan]) -> dict:
    """Everything the run must leave alone: every other UFC match and
    prediction, every rec not on a listed row, every UFC team row but the
    replaced fighters' (the recorded fighters' identity columns only: a
    resolver that reads ESPN athlete ids may stamp their external_ids)."""
    rows = sorted(p.entry.match_id for p in plans)
    teams = sorted({p.entry.replaced_team_id for p in plans} - {""})
    fighters = sorted(
        {
            t
            for p in plans
            if p.entry.action == ACTION_RECORD and p.row
            for t in (p.row["home_team_id"], p.row["away_team_id"])
        }
    )
    cur.execute(FINGERPRINT_SQL, {"rows": rows, "teams": teams, "fighters": fighters})
    return dict(cur.fetchone())


# ---------------------------------------------------------------------------
# Archive and audit CSV (write-once)
# ---------------------------------------------------------------------------


def _j(v: Any) -> str:
    return json.dumps(v, default=mmf._jsonable, sort_keys=True, ensure_ascii=False)


def write_archive(cur, path: Path, plans: list[EntryPlan], schema: mmf.Schema) -> int:
    """gzip JSONL of every row the run changes or deletes, before its first
    write: each listed row with every row under it (every child table), every
    rec on it, and each replaced fighter's team row."""
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "xt", encoding="utf-8") as fh:

        def emit(table: str, role: str, rows) -> None:
            nonlocal n
            for r in rows:
                fh.write(_j({"table": table, "role": role, "row": dict(r)}) + "\n")
                n += 1

        for p in plans:
            if not (p.record or p.to_record or p.open_recs or p.delete or p.team_delete):
                continue
            mid = p.entry.match_id
            cur.execute("SELECT * FROM matches WHERE id = %s", (mid,))
            emit("matches", f"{p.entry.action}_before", cur.fetchall())
            for table in schema.match_children:
                cur.execute(f"SELECT * FROM {table} WHERE match_id = %s", (mid,))
                emit(table, "child_before", cur.fetchall())
            cur.execute(
                "SELECT * FROM betting_recommendations WHERE match_id = %(m)s::uuid "
                "OR prediction_id IN (SELECT id FROM predictions WHERE match_id = %(m)s::uuid)",
                {"m": mid},
            )
            emit("betting_recommendations", "rec_before", cur.fetchall())
            if p.team_delete:
                cur.execute("SELECT * FROM teams WHERE id = %s", (p.entry.replaced_team_id,))
                emit("teams", "team_before", cur.fetchall())
    return n


def snapshot(cur, plans: list[EntryPlan]) -> dict[tuple[str, str], dict]:
    """(table, id) -> row (JSON-ready), for every row the audit CSV reports on:
    each listed row, its pinned predictions, the recs on it, every row under a
    row to delete, and each replaced fighter's team row. A row that no longer
    exists is absent."""
    out: dict[tuple[str, str], dict] = {}

    def take(table: str, sql: str, params) -> None:
        cur.execute(sql, params)
        for r in cur.fetchall():
            out[(table, str(r["id"]))] = json.loads(_j(dict(r)))

    for p in plans:
        mid, preds = p.entry.match_id, list(p.entry.prediction_ids)
        take("matches", "SELECT * FROM matches WHERE id = %s", (mid,))
        take("predictions", "SELECT * FROM predictions WHERE id = ANY(%s::uuid[])", (preds,))
        take(
            "betting_recommendations",
            "SELECT * FROM betting_recommendations WHERE match_id = %(m)s::uuid OR prediction_id = ANY(%(p)s::uuid[])",
            {"m": mid, "p": preds},
        )
        for table, ids in p.children.items():  # table names come from mmf.read_schema's whitelist
            take(table, f"SELECT * FROM {table} WHERE id = ANY(%s::uuid[])", (ids,))
        if p.entry.replaced_team_id:
            take("teams", "SELECT * FROM teams WHERE id = %s", (p.entry.replaced_team_id,))
    return out


def audit_rows(plans: list[EntryPlan], before: dict, after: dict) -> list[dict]:
    owner: dict[tuple[str, str], EntryPlan] = {}
    for p in plans:
        owner[("matches", p.entry.match_id)] = p
        for i in p.entry.prediction_ids:
            owner[("predictions", i)] = p
        for table, ids in p.children.items():
            for i in ids:
                owner[(table, i)] = p
        if p.entry.replaced_team_id:
            owner[("teams", p.entry.replaced_team_id)] = p
    rows = []
    for key in sorted(set(before) | set(after)):
        b, a = before.get(key), after.get(key)
        if b == a:
            continue
        whose = owner.get(key)
        change = "deleted" if a is None else ("inserted" if b is None else "updated")
        rows.append(
            {
                "match_id": whose.entry.match_id if whose else "",
                "action": whose.entry.action if whose else "",
                "table": key[0],
                "row_id": key[1],
                "change": change,
                "before": _j(b) if b is not None else "",
                "after": _j(a) if a is not None else "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as fh:  # exclusive create: write-once
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Writing (inside the one transaction)
# ---------------------------------------------------------------------------

DELETE_MATCH_SQL = """
    DELETE FROM matches
     WHERE id = %(id)s::uuid AND status = %(status)s AND match_date = %(kickoff)s
       AND home_team_id = %(home)s::uuid AND away_team_id = %(away)s::uuid
    RETURNING id
"""


def apply_record(cur, p: EntryPlan, espn: tuple[list[dict], dict, dict]) -> None:
    """The results sweep's own write of ESPN's competition, then the checks
    that it wrote exactly the listed row."""
    cfg = fu.SPORT_CONFIGS[SPORT]
    events, ev, comp = espn
    before_stats = dict(fu.IDENTITY_STATS)
    ctx = {**fu._event_context(ev), "pair_index": fu.build_pair_index(cfg, events), "league_slug": LEAGUE_SLUG}
    assert p.row is not None
    if not fu._record_result(cur, cfg, p.row["league_id"], comp, ctx):
        raise GuardError(f"{p.entry.match_id}: the results path did not record the bout (see the log above)")
    if fu.IDENTITY_STATS != before_stats:
        raise GuardError(
            f"{p.entry.match_id}: the results path counted a refusal or a team problem: {fu.IDENTITY_STATS}"
        )
    cur.execute(ROW_SQL, (p.entry.match_id,))
    row = dict(cur.fetchone() or {})
    meta = row.get("metadata") or {}
    if (
        row.get("status") != "finished"
        or (row.get("home_score"), row.get("away_score")) != (None, None)
        or meta.get(NO_WINNER_KEY) != p.entry.expect
        or (row.get("external_ids") or {}).get("espn") != p.entry.espn_competition_id
    ):
        raise GuardError(f"{p.entry.match_id}: after the results path the row reads {row}")
    logger.info("recorded %s as finished with no winner: %s", p.entry.match_id, meta.get("no_winner_detail"))


def apply_grade(cur, p: EntryPlan) -> None:
    """grade_completed_matches.grade_match on the recorded row: exactly the
    planned predictions recorded and recs voided."""
    matches = gcm.list_finished_matches(cur, 0, p.entry.match_id)
    if len(matches) != 1:
        raise GuardError(f"{p.entry.match_id}: the grader does not see one finished match ({len(matches)})")
    stats: dict = {}
    gcm.grade_match(cur, matches[0], stats)
    voids = stats.get("no_winner_rec_voids", 0) + stats.get("post_kickoff_rec_voids", 0)
    if stats.get("no_winner_predictions", 0) != len(p.to_record) or voids != len(p.open_recs):
        raise GuardError(
            f"{p.entry.match_id}: the grader recorded {stats.get('no_winner_predictions', 0)} prediction(s) and voided "
            f"{voids} rec(s); the plan says {len(p.to_record)} and {len(p.open_recs)} ({stats})"
        )


def apply_delete(cur, p: EntryPlan, schema: mmf.Schema) -> None:
    assert p.row is not None
    cur.execute(
        DELETE_MATCH_SQL,
        {
            "id": p.entry.match_id,
            "status": p.row["status"],
            "kickoff": p.row["match_date"],
            "home": p.row["home_team_id"],
            "away": p.row["away_team_id"],
        },
    )
    if len(cur.fetchall()) != 1:
        raise GuardError(f"{p.entry.match_id}: the guarded delete did not remove exactly the row")
    left = _children(cur, schema, p.entry.match_id)
    if left:
        raise GuardError(f"{p.entry.match_id}: rows under it survived the delete: {left}")
    logger.info(
        "deleted %s (%s) and %d row(s) under it", p.entry.match_id, p.entry.action, sum(map(len, p.children.values()))
    )


def apply_team_delete(cur, p: EntryPlan, schema: mmf.Schema) -> None:
    assert p.team is not None
    cur.execute(
        mmf.team_delete_sql(schema),
        {"id": p.team["id"], "name": p.team["name"], "normalized_name": p.team["normalized_name"]},
    )
    if cur.rowcount != 1:
        raise GuardError(f"team {p.team['id']} ({p.team['name']}): the guarded delete removed {cur.rowcount} row(s)")
    logger.info("deleted team %s (%s): no bout references it", p.team["id"], p.team["name"])


# ---------------------------------------------------------------------------
# After-state
# ---------------------------------------------------------------------------


def after_state_problems(cur, entries: list[Entry], schema: mmf.Schema) -> list[str]:
    """Every listed row in its after-state (read only)."""
    problems: list[str] = []
    for e in entries:
        cur.execute(ROW_SQL, (e.match_id,))
        row = cur.fetchone()
        if e.action == ACTION_RECORD:
            if row is None:
                problems.append(f"{e.match_id}: the recorded row is gone")
                continue
            meta = row["metadata"] or {}
            if (
                row["status"] != "finished"
                or (row["home_score"], row["away_score"]) != (None, None)
                or meta.get(NO_WINNER_KEY) != e.expect
                or not meta.get("no_winner_detail")
                or (row["external_ids"] or {}).get("espn") != e.espn_competition_id
            ):
                problems.append(
                    f"{e.match_id}: {row['status']} {row['home_score']}-{row['away_score']} metadata {meta} "
                    f"external_ids {row['external_ids']}"
                )
            cur.execute(PREDICTIONS_SQL, (e.match_id,))
            preds = [dict(r) for r in cur.fetchall()]
            if tuple(sorted(r["id"] for r in preds)) != e.prediction_ids:
                problems.append(f"{e.match_id}: predictions {[r['id'] for r in preds]}")
            for r in preds:
                want = (None, None) if r["after_kickoff"] else (e.expect, None)
                if (r["actual_outcome"], r["is_correct"]) != want:
                    problems.append(f"prediction {r['id']}: ({r['actual_outcome']!r}, {r['is_correct']!r}), not {want}")
            cur.execute(RECS_SQL, {"m": e.match_id})
            for r in cur.fetchall():
                if r["status"] != "void" or r["profit_loss"] is None or r["profit_loss"] != 0:
                    problems.append(f"rec {r['id']}: {r['status']} {r['profit_loss']}")
                elif r["actual_result"] not in (NO_WINNER_VOID_RESULTS[e.expect], gcm.POST_KICKOFF_VOID_RESULT):
                    problems.append(f"rec {r['id']}: actual_result {r['actual_result']!r}")
        else:
            if row is not None:
                problems.append(f"{e.match_id}: the replaced row is still there ({row['status']})")
            left = _children(cur, schema, e.match_id)
            if left:
                problems.append(f"{e.match_id}: rows still reference it: {left}")
            cur.execute("SELECT id::text AS id FROM predictions WHERE id = ANY(%s::uuid[])", (list(e.prediction_ids),))
            survivors = [r["id"] for r in cur.fetchall()]
            if survivors:
                problems.append(f"{e.match_id}: its predictions survive: {survivors}")
            cur.execute(ROW_SQL, (e.replacement_match_id,))
            rep = cur.fetchone()
            if rep is None or rep["status"] != "finished":
                problems.append(f"the replacement row {e.replacement_match_id} is {rep['status'] if rep else 'gone'}")
            cur.execute(TEAM_SQL, (e.replaced_team_id,))
            if cur.fetchone() is not None:
                refs = _team_refs(cur, schema, e.replaced_team_id)
                if not refs:
                    problems.append(f"team {e.replaced_team_id} ({e.replaced_name}) is left with no bout")
    return problems


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def _schema(cur) -> mmf.Schema:
    cur.execute(mmf.FOREIGN_KEYS_SQL)
    schema, unknown = mmf.read_schema([dict(r) for r in cur.fetchall()])
    if schema is None:
        raise GuardError(f"unknown foreign key(s) to teams/matches: {unknown}")
    return schema


def run(
    conn,
    entries: list[Entry],
    apply: bool,
    out_csv: Path,
    archive: Path,
    expect: Optional[dict[str, int]] = None,
    fetch=None,
) -> int:
    conn.set_isolation_level(ISOLATION_LEVEL_REPEATABLE_READ)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'")
            schema = _schema(cur)
            plans = plan_entries(cur, entries, schema, lock=True)
            espn = check_espn(plans, fetch)
            totals = summarize(plans)
            line = plan_line(totals)
            logger.info("%s", line)
            for p in plans:
                if p.refusal:
                    logger.error("REFUSED %s (%s): %s", p.entry.match_id, p.entry.action, p.refusal)
            if totals["refuse"]:
                conn.rollback()
                logger.error("REFUSED: %d row(s) refused (above) — nothing written", totals["refuse"])
                return 2
            if apply and expect != totals:
                diff = {
                    k: (expect.get(k) if expect else None, totals[k])
                    for k in PLAN_KEYS
                    if not expect or expect.get(k) != totals[k]
                }
                conn.rollback()
                logger.error(
                    "REFUSED: the plan is not --expect-plan (key: (expected, planned)) %s — nothing written", diff
                )
                return 2
            if is_done(totals, entries):
                conn.rollback()
                write_csv(out_csv, [])
                logger.info("NOTHING TO WRITE: every listed row is settled (audit CSV %s, empty)", out_csv)
                return 0
            before_fp = fingerprint(cur, plans)
            before = snapshot(cur, plans)
            if apply:
                n = write_archive(cur, archive, plans, schema)
                logger.info("Archived %d row(s) to %s", n, archive)
            for p in plans:
                if p.record:
                    apply_record(cur, p, espn[p.entry.match_id])
                if p.record or p.to_record or p.open_recs:
                    apply_grade(cur, p)
                if p.delete:
                    apply_delete(cur, p, schema)
                if p.team_delete:
                    apply_team_delete(cur, p, schema)
            after = snapshot(cur, plans)
            problems = after_state_problems(cur, entries, schema)
            replan = summarize(plan_entries(cur, entries, schema, lock=False))
            if not is_done(replan, entries):
                problems.append(f"the re-plan is not done: {counts_text(replan)}")
            after_fp = fingerprint(cur, plans)
            changed = {k: (before_fp[k], after_fp[k]) for k in before_fp if before_fp[k] != after_fp[k]}
            if changed:
                problems.append(f"rows outside the listed ones changed: {sorted(changed)}")
            rows = audit_rows(plans, before, after)
            if problems:
                conn.rollback()
                for pr in problems:
                    logger.error("AFTER (in the transaction) FAILED: %s", pr)
                logger.error("ROLLED BACK: the after-state did not hold — nothing committed")
                return 1
            logger.info("AFTER (in the transaction): every listed row settled; re-plan: %s", counts_text(replan))
            write_csv(out_csv, rows)
            logger.info("audit CSV %s: %d changed row(s)", out_csv, len(rows))
            if not apply:
                conn.rollback()
                logger.info("DRY RUN: the whole transaction ran and was ROLLED BACK — nothing committed")
                return 0
        conn.commit()
        logger.info("APPLIED: committed (%s)", line)
        return 0
    except (GuardError, psycopg2.Error) as e:
        conn.rollback()
        logger.error("ROLLED BACK (%s): %s", type(e).__name__, e)
        return 1


def run_check_after(conn, entries: list[Entry]) -> int:
    conn.set_session(readonly=True)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            schema = _schema(cur)
            problems = after_state_problems(cur, entries, schema)
            totals = summarize(plan_entries(cur, entries, schema, lock=False))
            logger.info("now: %s", counts_text(totals))
            if not is_done(totals, entries):
                problems.append(f"the plan is not done: {counts_text(totals)}")
        conn.rollback()
    except GuardError as e:
        conn.rollback()
        problems = [str(e)]
    if problems:
        for pr in problems:
            logger.error("AFTER-STATE FAILED: %s", pr)
        return 1
    logger.info(
        "AFTER-STATE OK: %d row(s) recorded with no winner, %d replaced row(s) deleted, the plan is done",
        sum(e.action == ACTION_RECORD for e in entries),
        sum(e.action == ACTION_DELETE for e in entries),
    )
    return 0


def default_out(kind: str, apply: bool) -> Path:
    base = Path(os.environ.get("BACKUP_LOCAL_DIR", "/app/backups"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    mode = "apply" if apply else "dryrun"
    return base / (
        f"settle_mma_held_rows_{mode}_{stamp}.csv" if kind == "csv" else f"settle_mma_held_rows_{stamp}.jsonl.gz"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", type=Path, default=LIST_PATH, help="The reviewed list (default: %(default)s).")
    p.add_argument("--apply", action="store_true", help="Commit (default: run the transaction and roll it back).")
    p.add_argument("--expect-plan", default=None, help="--apply: every PLAN key, k=v,... (the reviewed dry run's).")
    p.add_argument("--check-after", action="store_true", help="Read only: the after-state of every listed row.")
    p.add_argument("--out-csv", type=Path, default=None, help="Audit CSV, write-once (default under /app/backups).")
    p.add_argument("--archive", type=Path, default=None, help="--apply: gzip JSONL reversal record, write-once.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    try:
        entries = load_list(args.list)
    except (ListError, OSError) as e:
        logger.error("REFUSED: %s", e)
        return 2
    if args.check_after:
        if args.apply or args.expect_plan:
            logger.error("REFUSED: --check-after takes no --apply / --expect-plan")
            return 2
        conn = psycopg2.connect(args.database_url)
        try:
            return run_check_after(conn, entries)
        finally:
            conn.close()
    if bool(args.apply) != bool(args.expect_plan):
        logger.error("REFUSED: --apply needs --expect-plan (every PLAN key), and --expect-plan only goes with --apply")
        return 2
    try:
        expect = parse_expect(args.expect_plan) if args.expect_plan else None
    except ValueError as e:
        logger.error("REFUSED: %s", e)
        return 2
    out_csv = args.out_csv or default_out("csv", args.apply)
    archive = args.archive or default_out("archive", args.apply)
    for path in (out_csv, archive) if args.apply else (out_csv,):
        if path.exists():
            logger.error("REFUSED: %s exists — the audit files are write-once", path)
            return 2
    conn = psycopg2.connect(args.database_url)
    try:
        return run(conn, entries, args.apply, out_csv, archive, expect)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
