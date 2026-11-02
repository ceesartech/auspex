"""Delete the eight bout-less MMA fighter rows the user approved on 2026-11-02
("Yes" to adding the 8 empty MMA fighter records, including 'TBA' and
'Opponent TBA', to the clean-up).

THE ROWS (scripts/data/mma_orphan_fighters_20261103.csv, reviewed 2026-11-02)
------------------------------------------------------------------------------
Rodrigo Vera, Max Gimenis, Ramiro Jimenez, Uran Satybaldiev, Henrique da Silva
Lopes, Ethyn Ewing, and the placeholders 'TBA' and 'Opponent TBA'. Each was
created by the ESPN sweep for a bout ESPN later dropped or re-drew; run B of
scripts/ops/repair_stranded_recs_mma.sh deleted those 10 bouts on 2026-11-02
(archive /app/backups/fixture_orphans_mma_20261102T043520Z.jsonl.gz), which left
the rows with no bout. None carries an ESPN athlete id, an alias or any other
metadata; nothing references them under any foreign key or in any JSON column;
ESPN lists none of them on a UFC card through 2026-12-17. fetch_upcoming skips
'TBA' and 'Opponent TBA' as placeholders, so nothing resolves to those two.

WHY A TOOL OF ITS OWN (and not a new mode of scripts/mma_identity_cleanup.py)
------------------------------------------------------------------------------
That tool's apply is one pinned transaction over four lists, its wrapper's
re-run and --verify read its recorded plan file, and it has run on prod. A new
list or mode there would change the plan it re-checks. This tool reuses its
pieces unchanged (the foreign-key discovery, the per-key reference counts, the
exactly-one-row guard, the write-once audit CSV and plan file) and adds the
guards this delete needs:
  * every listed row is planned against the row as it is now: delete, done
    (gone) or refuse. It is refused when its name, sport or ESPN id differ from
    the list, when it carries aliases, when any foreign key to teams references
    it, when another team row's metadata or external_ids, or an MMA match's (or
    its predictions', features', odds' or recs' JSON), names its id, or when
    another MMA row claims its name as an alias;
  * the list may not name a row any other MMA list holds (the identity
    clean-up's split, alias, orphan and athlete-id lists, the fighter merge
    map), so their --verify keeps holding after this runs;
  * each DELETE is guarded on the row exactly as the plan saw it: id, sport,
    name, normalized_name, league, country, is_active, external_ids, metadata,
    created_at and updated_at, and NOT EXISTS under every foreign key to teams
    (matches -> teams cascades: a delete must never take a bout with it).

WHAT --apply DOES (ONE transaction: any failure rolls all of it back)
--------------------------------------------------------------------
1. LOCK TABLE teams IN EXCLUSIVE MODE (lock_timeout 10s) before its snapshot.
   EXCLUSIVE, not SHARE ROW EXCLUSIVE: inserting or re-pointing a bout checks
   its foreign key with SELECT ... FOR KEY SHARE on teams, which takes ROW
   SHARE, and only EXCLUSIVE (or stronger) blocks that. So until it commits no
   sweep can insert a bout onto any team, nor insert, stamp or rename a team;
   plain reads go on. A sweep that tries waits seconds; one that wanted a bout
   for a deleted row then fails loudly on the foreign key and the next sweep
   creates a fresh row by ESPN athlete id.
2. Plan again under the lock; refuse (exit 2, nothing written) unless the plan
   digest equals --expect-plan <the dry run's plan file>, or any row refuses.
3. Write a write-once gzip JSONL archive of every row it deletes (the reversal
   record) before the first write.
4. Delete each row, guarded as above; each must touch exactly one row.
5. Commit only if, inside the transaction: every listed row is gone, teams
   fell by exactly the deletes, and no table with a foreign key to teams lost a
   row (no cascade).

The dry run (the default) runs in a READ ONLY transaction and writes only its
write-once audit CSV and plan file. --check-after <plan file> (read only)
checks the after-state a plan implies. Re-runnable: once done, the plan is
'done' for every row and nothing is written.

REVERSAL: re-insert the archived rows (table teams, role pre_image) with their
own ids. To restore any of run B's deleted bouts of these fighters from run B's
archive, restore these rows first (the bouts reference them).

Usage (inside the api container; dry run is the default):
    python /app/scripts/mma_orphan_cleanup.py --plan-out /app/backups/mma_orphan_cleanup.plan.json
    python /app/scripts/mma_orphan_cleanup.py --apply --expect-plan /app/backups/mma_orphan_cleanup.plan.json
    python /app/scripts/mma_orphan_cleanup.py --check-after /app/backups/mma_orphan_cleanup.plan.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mma_identity_cleanup as mic  # noqa: E402 - the reused machinery, unchanged
from team_identity import alias_form  # noqa: E402

logger = logging.getLogger("mma_orphan_cleanup")

DATA = SCRIPTS_DIR / "data"
LIST_PATH = DATA / "mma_orphan_fighters_20261103.csv"
MERGE_MAP_PATH = DATA / "mma_fighter_merge_map.csv"
FIELDS = ("team_id", "team_name", "espn_athlete_id", "why")
SPORT = "mma"
LOCK_TIMEOUT = "10s"
TOOL = "mma_orphan_cleanup"
PLAN_KEYS = ("listed", "delete", "done", "refuse")
GuardError = mic.GuardError
ListError = mic.ListError


# ---------------------------------------------------------------------------
# The reviewed list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Orphan:
    team_id: str
    team_name: str
    espn: str  # '' when the reviewed row carries no ESPN athlete id


def load_list(path: Path) -> tuple[Orphan, ...]:
    """The reviewed rows ('#' lines are comments): team_id a UUID, team_name and
    why filled, espn_athlete_id digits or empty; no id or name twice."""
    with path.open(newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    missing = [f for f in FIELDS if f not in (reader.fieldnames or [])]
    if missing:
        raise ListError(f"{path}: missing column(s) {', '.join(missing)}")
    out: list[Orphan] = []
    ids: set[str] = set()
    names: set[str] = set()
    for n, row in enumerate(reader, start=1):
        vals = {f: (row.get(f) or "").strip() for f in FIELDS}
        empty = [f for f in ("team_id", "team_name", "why") if not vals[f]]
        if empty:
            raise ListError(f"{path}: data row {n}: empty {', '.join(empty)} — only reviewed rows with evidence")
        if not mic.UUID_RE.match(vals["team_id"]):
            raise ListError(f"{path}: data row {n}: team_id {vals['team_id']!r} is not a UUID")
        if vals["espn_athlete_id"] and not vals["espn_athlete_id"].isdigit():
            raise ListError(f"{path}: data row {n}: espn_athlete_id {vals['espn_athlete_id']!r} is not digits")
        if vals["team_id"] in ids or vals["team_name"].casefold() in names:
            raise ListError(f"{path}: data row {n}: {vals['team_id']} / {vals['team_name']!r} is listed twice")
        ids.add(vals["team_id"])
        names.add(vals["team_name"].casefold())
        out.append(Orphan(vals["team_id"], vals["team_name"], vals["espn_athlete_id"]))
    if not out:
        raise ListError(f"{path}: no data rows")
    return tuple(out)


def other_list_ids(merge_map: Path = MERGE_MAP_PATH) -> dict[str, str]:
    """{team id: which list} for every row another MMA list holds: the identity
    clean-up's four lists (split row and its new row, aliases, orphans, athlete
    ids) and the fighter merge map (merged and kept rows)."""
    lists = mic.load_lists()
    out: dict[str, str] = {}
    out[lists.split.team_id] = "the identity clean-up's split row"
    out[lists.split.new_team_id] = "the identity clean-up's new split row"
    out.update({a.team_id: "the identity clean-up's alias list" for a in lists.aliases})
    out.update({o.team_id: "the identity clean-up's orphan list" for o in lists.orphans})
    out.update({r.team_id: "the identity clean-up's athlete-id list" for r in lists.ids})
    with merge_map.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(ln for ln in fh if not ln.startswith("#")):
            for col in ("merged_team_id", "kept_team_id"):
                if not (row.get(col) or "").strip():
                    raise ListError(f"{merge_map}: a row without {col}")
                out[row[col].strip()] = "the fighter merge map"
    return out


def check_no_overlap(orphans: Iterable[Orphan], others: dict[str, str]) -> None:
    for o in orphans:
        if o.team_id in others:
            raise ListError(
                f"{o.team_name!r} ({o.team_id}) is in {others[o.team_id]}: deleting it would break that list's done "
                "state — never listed here"
            )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

ROWS_SQL = """
    SELECT id::text AS id, name, normalized_name, sport, league_id::text AS league_id, country, is_active,
           external_ids, metadata, created_at, updated_at
      FROM teams WHERE id = ANY(%s::uuid[])
"""
# Another row or an MMA match (or a row hanging off one) that names the id in its JSON: merge, split and alias
# records, a re-point or fold note. The pattern is the bare uuid (it holds no LIKE wildcard).
SOFT_REFS_SQL = """
    SELECT
      (SELECT count(*) FROM teams o WHERE o.id <> %(t)s::uuid
          AND (COALESCE(o.metadata::text, '') LIKE %(pat)s OR COALESCE(o.external_ids::text, '') LIKE %(pat)s))
        AS "teams.json",
      (SELECT count(*) FROM matches m JOIN leagues l ON l.id = m.league_id WHERE l.sport = 'mma'
          AND (COALESCE(m.metadata::text, '') LIKE %(pat)s OR COALESCE(m.external_ids::text, '') LIKE %(pat)s))
        AS "matches.json",
      (SELECT count(*) FROM predictions p JOIN matches m ON m.id = p.match_id JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'mma' AND (COALESCE(p.metadata::text, '') || COALESCE(p.features_used::text, '')
                                   || COALESCE(p.feature_importance::text, '')) LIKE %(pat)s)
        AS "predictions.json",
      (SELECT count(*) FROM features_cache f JOIN matches m ON m.id = f.match_id JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'mma' AND COALESCE(f.features::text, '') LIKE %(pat)s)
        AS "features_cache.json",
      (SELECT count(*) FROM odds o JOIN matches m ON m.id = o.match_id JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'mma' AND COALESCE(o.metadata::text, '') LIKE %(pat)s)
        AS "odds.json",
      (SELECT count(*) FROM betting_recommendations r WHERE COALESCE(r.risk_factors::text, '') LIKE %(pat)s)
        AS "betting_recommendations.json"
"""
ALIASES_SQL = (
    "SELECT id::text AS id, metadata->'aliases' AS aliases FROM teams WHERE sport = 'mma' AND metadata ? 'aliases'"
)
OTHER_BOUTLESS_SQL = """
    SELECT t.id::text AS id, t.name FROM teams t
     WHERE t.sport = 'mma' AND NOT (t.id = ANY(%s::uuid[]))
       AND NOT EXISTS (SELECT 1 FROM matches m WHERE m.home_team_id = t.id OR m.away_team_id = t.id)
     ORDER BY t.name
"""


@dataclass
class State:
    rows: dict[str, dict]  # the listed rows that exist, by id (any sport)
    team_fks: tuple[tuple[str, str], ...]
    refs: dict[str, dict[str, int]]  # id -> "table.column" (foreign keys) or "table.json" (soft) -> count
    alias_claims: dict[str, list[str]]  # id -> other MMA rows whose metadata.aliases holds its name
    other_boutless: list[dict]  # MMA rows with no bout that the list does not name (reported, never touched)


def load_state(cur, orphans: tuple[Orphan, ...]) -> State:
    cur.execute(mic.TEAM_FKS_SQL)
    fks = tuple((r["tbl"], r["col"]) for r in cur.fetchall())
    if ("matches", "home_team_id") not in fks or ("matches", "away_team_id") not in fks:
        raise GuardError("pg_constraint lists no matches -> teams foreign keys: not the schema this tool knows")
    ids = [o.team_id for o in orphans]
    cur.execute(ROWS_SQL, (ids,))
    rows = {r["id"]: dict(r) for r in cur.fetchall()}
    refs: dict[str, dict[str, int]] = {}
    for o in orphans:
        if o.team_id not in rows:
            continue
        cur.execute(mic._team_refs_sql(fks), {"t": o.team_id})
        counts = {k: int(v) for k, v in dict(cur.fetchone()).items()}
        cur.execute(SOFT_REFS_SQL, {"t": o.team_id, "pat": f"%{o.team_id}%"})
        counts.update({k: int(v) for k, v in dict(cur.fetchone()).items()})
        refs[o.team_id] = counts
    cur.execute(ALIASES_SQL)
    claims: dict[str, list[str]] = {o.team_id: [] for o in orphans}
    forms = {o.team_id: alias_form(o.team_name) for o in orphans}
    for r in cur.fetchall():
        aliases = r["aliases"] if isinstance(r["aliases"], list) else []
        held = {alias_form(a) for a in aliases if isinstance(a, str)}
        for tid, form in forms.items():
            if r["id"] != tid and form in held:
                claims[tid].append(r["id"])
    cur.execute(OTHER_BOUTLESS_SQL, (ids,))
    other = [dict(r) for r in cur.fetchall()]
    return State(rows, fks, refs, {k: sorted(v) for k, v in claims.items()}, other)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

OBSERVED = (
    "name",
    "normalized_name",
    "sport",
    "league_id",
    "country",
    "is_active",
    "external_ids",
    "metadata",
    "created_at",
    "updated_at",
)


@dataclass
class RowPlan:
    team_id: str
    team_name: str
    action: str  # delete | done | refuse
    reason: str
    observed: dict = field(default_factory=dict)  # the row as the plan saw it (delete only)


def _espn(row: dict) -> str:
    ext = row.get("external_ids")
    return str(ext.get("espn") or "") if isinstance(ext, dict) else ""


def plan_row(o: Orphan, state: State) -> RowPlan:
    t = state.rows.get(o.team_id)
    if t is None:
        return RowPlan(o.team_id, o.team_name, "done", f"{o.team_name!r} is gone")
    if t["sport"] != SPORT:
        return RowPlan(o.team_id, o.team_name, "refuse", f"{o.team_id} is a {t['sport']} row, not MMA")
    if t["name"] != o.team_name:
        return RowPlan(o.team_id, o.team_name, "refuse", f"{o.team_id} is named {t['name']!r}, not {o.team_name!r}")
    if _espn(t) != o.espn:
        return RowPlan(
            o.team_id,
            o.team_name,
            "refuse",
            f"{o.team_name!r} carries ESPN athlete id {_espn(t) or '-'}, the list {o.espn or '-'} — an identity "
            "changed since the review: re-review",
        )
    meta, ext = t.get("metadata"), t.get("external_ids")
    extra = sorted(set(ext) - {"espn"}) if isinstance(ext, dict) else ["<not an object>"] if ext is not None else []
    if meta not in (None, {}) or extra:
        keys = sorted(meta) if isinstance(meta, dict) else ["<not an object>"] if meta is not None else []
        return RowPlan(
            o.team_id,
            o.team_name,
            "refuse",
            f"{o.team_name!r} carries metadata {keys} / external ids {extra} the review did not see (aliases, "
            "merge or split records, other ids) — re-review",
        )
    refs = {k: v for k, v in state.refs.get(o.team_id, {}).items() if v}
    if refs:
        return RowPlan(o.team_id, o.team_name, "refuse", f"{o.team_name!r} is referenced: {refs}")
    claims = state.alias_claims.get(o.team_id) or []
    if claims:
        return RowPlan(
            o.team_id,
            o.team_name,
            "refuse",
            f"MMA row(s) {', '.join(claims)} claim {o.team_name!r} as an alias — re-review",
        )
    return RowPlan(
        o.team_id, o.team_name, "delete", f"delete bout-less {o.team_name!r}", {k: t.get(k) for k in OBSERVED}
    )


def plan_run(orphans: tuple[Orphan, ...], state: State) -> list[RowPlan]:
    return [plan_row(o, state) for o in orphans]


def summarize(plan: list[RowPlan]) -> dict[str, int]:
    out = {k: 0 for k in PLAN_KEYS}
    out["listed"] = len(plan)
    for p in plan:
        out[p.action] += 1
    return out


def plan_line(totals: dict[str, int]) -> str:
    return "PLAN " + " ".join(f"{k}={totals[k]}" for k in PLAN_KEYS)


def plan_document(plan: list[RowPlan], state: State, mode: str) -> dict:
    """The plan as data; the digest covers every row's id, action and the row as
    observed (a delete), not the reasons, timestamps or the other bout-less rows."""
    rows = [
        {"team_id": p.team_id, "team_name": p.team_name, "action": p.action, "reason": p.reason, "observed": p.observed}
        for p in plan
    ]
    structural = sorted(
        ([r["team_id"], r["team_name"], r["action"], r["observed"]] for r in rows), key=lambda x: str(x[0])
    )
    return {
        "tool": TOOL,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "digest": hashlib.sha256(mic._j(structural).encode()).hexdigest(),
        "summary": summarize(plan),
        "rows": rows,
        "other_boutless": state.other_boutless,
    }


def plan_difference(expected: dict, doc: dict) -> Optional[str]:
    """None when `doc` is the plan `expected` (a dry run's plan file) showed, else what differs. Pure."""
    if expected.get("tool") != TOOL or "digest" not in expected:
        return f"the expected plan file is not a {TOOL} plan"
    if expected["digest"] == doc["digest"]:
        return None
    want = {r.get("team_id"): r for r in expected.get("rows", [])}
    for r in doc["rows"]:
        w = want.get(r["team_id"])
        if w is None:
            return f"{r['team_id']} is not in the expected plan"
        if w.get("action") != r["action"]:
            return f"{r['team_id']} ({r['team_name']!r}): {r['action']}, the expected plan says {w.get('action')}"
        if mic._j(w.get("observed")) != mic._j(r["observed"]):
            return f"{r['team_id']} ({r['team_name']!r}) changed since the dry run"
    return "the plan digest differs from the expected plan"


def refusals(plan: list[RowPlan]) -> list[str]:
    return [f"{p.team_id} ({p.team_name!r}): {p.reason}" for p in plan if p.action == "refuse"]


# ---------------------------------------------------------------------------
# Audit CSV, archive, writes
# ---------------------------------------------------------------------------


def csv_rows(plan: list[RowPlan]) -> list[dict]:
    return [
        {
            "unit": "orphan",
            "action": p.action,
            "table": "teams",
            "row_id": p.team_id,
            "before": mic._j(p.observed) if p.observed else "",
            "after": "",
            "reason": p.reason,
        }
        for p in plan
    ]


def write_archive(cur, path: Path, plan: list[RowPlan]) -> int:
    """gzip JSONL, written (exclusive create) before the first write: the full
    pre-image of every row the run deletes."""
    ids = sorted(p.team_id for p in plan if p.action == "delete")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "xt", encoding="utf-8") as fh:
        cur.execute("SELECT * FROM teams WHERE id = ANY(%s::uuid[]) ORDER BY id", (ids,))
        for r in cur.fetchall():
            fh.write(mic._j({"table": "teams", "role": "pre_image", "row": dict(r)}) + "\n")
            n += 1
    if n != len(ids):
        raise GuardError(f"archived {n} row(s), the plan deletes {len(ids)}")
    return n


def delete_sql(fks: Iterable[tuple[str, str]]) -> sql.Composed:
    """DELETE one team row exactly as the plan saw it, and only while nothing
    references it under any foreign key to teams."""
    guards = [
        sql.SQL(" AND NOT EXISTS (SELECT 1 FROM {} x WHERE x.{} = t.id)").format(
            sql.Identifier(*tbl.split(".")), sql.Identifier(col)
        )
        for tbl, col in fks
    ]
    return (
        sql.SQL(
            "DELETE FROM teams t WHERE t.id = %(id)s::uuid AND t.sport = 'mma' AND t.name = %(name)s"
            " AND t.normalized_name IS NOT DISTINCT FROM %(normalized_name)s"
            " AND t.league_id IS NOT DISTINCT FROM %(league_id)s::uuid"
            " AND t.country IS NOT DISTINCT FROM %(country)s"
            " AND t.is_active IS NOT DISTINCT FROM %(is_active)s"
            " AND t.external_ids IS NOT DISTINCT FROM %(external_ids)s::jsonb"
            " AND t.metadata IS NOT DISTINCT FROM %(metadata)s::jsonb"
            " AND t.created_at IS NOT DISTINCT FROM %(created_at)s"
            " AND t.updated_at IS NOT DISTINCT FROM %(updated_at)s"
        )
        + sql.SQL("").join(guards)
        + sql.SQL(" RETURNING t.id")
    )


def _json_param(v: Any) -> Optional[str]:
    return None if v is None else mic._j(v)


def delete_params(p: RowPlan) -> dict:
    ob = p.observed
    return {
        "id": p.team_id,
        "name": ob["name"],
        "normalized_name": ob["normalized_name"],
        "league_id": ob["league_id"],
        "country": ob["country"],
        "is_active": ob["is_active"],
        "external_ids": _json_param(ob["external_ids"]),
        "metadata": _json_param(ob["metadata"]),
        "created_at": ob["created_at"],
        "updated_at": ob["updated_at"],
    }


def table_counts(cur, fks: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Row counts of teams and of every table with a foreign key to it, and the non-NULL count of each such
    column: an ON DELETE CASCADE would lower a table's count, an ON DELETE SET NULL a column's."""
    out: dict[str, int] = {}
    cur.execute("SELECT count(*) AS n FROM teams")
    out["teams"] = int(cur.fetchone()["n"])
    for tbl, col in sorted(set(fks)):
        cur.execute(
            sql.SQL("SELECT count(*) AS n, count({}) AS nn FROM {}").format(
                sql.Identifier(col), sql.Identifier(*tbl.split("."))
            )
        )
        r = cur.fetchone()
        out[tbl] = int(r["n"])
        out[f"{tbl}.{col}"] = int(r["nn"])
    return out


# ---------------------------------------------------------------------------
# After-state
# ---------------------------------------------------------------------------


def check_after(cur, doc: dict) -> dict[str, int]:
    """Read-only: of the rows the plan deletes (or found gone), how many are still there."""
    ids = [r["team_id"] for r in doc.get("rows", []) if r.get("action") in ("delete", "done")]
    cur.execute("SELECT count(*) AS n FROM teams WHERE id = ANY(%s::uuid[])", (ids,))
    return {"orphans_expected_gone": len(ids), "orphans_left": int(cur.fetchone()["n"])}


def after_line(out: dict[str, int]) -> str:
    return " ".join(f"{k}={v}" for k, v in out.items())


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def default_out(kind: str) -> Path:
    base = Path(os.environ.get("BACKUP_LOCAL_DIR", "/app/backups"))
    ext = {"csv": "csv", "archive": "jsonl.gz"}[kind]
    return base / f"{TOOL}_{_stamp()}.{ext}"


def log_plan(plan: list[RowPlan], state: State) -> None:
    for p in plan:
        (logger.error if p.action == "refuse" else logger.info)(
            "[orphan] %s %r: %s — %s", p.team_id, p.team_name, p.action, p.reason
        )
    names = ", ".join(f"{r['name']!r} ({r['id']})" for r in state.other_boutless) or "none"
    logger.info(
        "OTHER_BOUTLESS %d MMA row(s) with no bout that the list does not name: %s (never touched)",
        len(state.other_boutless),
        names,
    )


def run(
    conn,
    apply_changes: bool,
    orphans: tuple[Orphan, ...],
    out_csv: Path,
    archive: Path,
    plan_out: Optional[Path] = None,
    expect_plan: Optional[dict] = None,
) -> int:
    """Exit status: 0 done (dry run: nothing refused); 1 a refusal in the dry
    run, or the apply rolled back after it began writing; 2 refused before any
    write (the state unreadable or locked, a plan that is not the expected one,
    a refusal)."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        try:
            if apply_changes:
                cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
                cur.execute("SET LOCAL lock_timeout = %s", (LOCK_TIMEOUT,))
                # Before the snapshot: no team insert/update and no bout insert or re-point (its foreign-key check
                # takes ROW SHARE on teams) until this commits; plain reads go on.
                cur.execute("LOCK TABLE teams IN EXCLUSIVE MODE")
            else:
                cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            state = load_state(cur, orphans)
        except (psycopg2.Error, GuardError) as e:
            conn.rollback()
            logger.error("REFUSED: reading the state failed (%s) — nothing written", str(e).strip().splitlines()[0])
            return 2
        plan = plan_run(orphans, state)
        doc = plan_document(plan, state, "apply" if apply_changes else "dry-run")
        mic.write_csv(out_csv, csv_rows(plan))
        plan_path = plan_out or out_csv.with_suffix(".plan.json")
        mic.write_plan(plan_path, doc)
        log_plan(plan, state)
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
            logger.error("REFUSED: the plan refuses %d row(s) — nothing written", len(refused))
            return 2
        logger.info("Plan matches the expected plan (digest %s)", doc["digest"])
        deletes = [p for p in plan if p.action == "delete"]
        try:
            n = write_archive(cur, archive, plan)
            logger.info("Archived %d row(s) to %s", n, archive)
            before = table_counts(cur, state.team_fks)
            query = delete_sql(state.team_fks)
            for p in deletes:
                mic._expect(cur, query, delete_params(p), f"delete {p.team_name!r} ({p.team_id})")
            after = check_after(cur, doc)
            now = table_counts(cur, state.team_fks)
            lost = {t: before[t] - now[t] for t in before if t != "teams" and now[t] != before[t]}
            teams_removed = before["teams"] - now["teams"]
            logger.info(
                "AFTER (in the transaction) %s teams_removed=%d cascaded=%d",
                after_line(after),
                teams_removed,
                sum(lost.values()),
            )
            if after["orphans_left"] or teams_removed != len(deletes) or lost:
                raise GuardError(
                    f"the after-state does not hold inside the transaction: {after['orphans_left']} listed row(s) "
                    f"left, teams fell by {teams_removed} (planned {len(deletes)}), rows lost elsewhere {lost or 0}"
                )
        except (GuardError, psycopg2.Error, OSError) as e:
            conn.rollback()
            logger.error("ROLLED BACK — nothing committed: %s", str(e).strip().splitlines()[0])
            return 1
        conn.commit()
    logger.info("WRITTEN orphans_deleted=%d", len(deletes))
    logger.info("APPLIED: committed (plan digest %s)", doc["digest"])
    return 0


def run_check_after(conn, plan_path: Path) -> int:
    doc = json.loads(plan_path.read_text(encoding="utf-8"))
    if doc.get("tool") != TOOL:
        logger.error("REFUSED: %s is not a %s plan file", plan_path, TOOL)
        return 2
    if any(r.get("action") == "refuse" for r in doc.get("rows", [])):
        logger.error("REFUSED: %s plans a refusal — it was never applied; no after-state to check", plan_path)
        return 2
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        out = check_after(cur, doc)
    conn.rollback()
    logger.info("AFTER %s", after_line(out))
    if out["orphans_left"] == 0 and out["orphans_expected_gone"] == len(doc.get("rows", [])):
        logger.info("AFTER-STATE OK: the %d listed bout-less MMA fighter rows are gone", out["orphans_expected_gone"])
        return 0
    logger.error("AFTER-STATE FAILED: every listed row must be planned delete/done and be gone")
    return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="Write (default: a read-only dry run).")
    p.add_argument("--expect-plan", type=Path, default=None, help="--apply only if the plan equals this dry run's.")
    p.add_argument("--list", type=Path, default=LIST_PATH, help=argparse.SUPPRESS)
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
    if args.check_after:
        conn = psycopg2.connect(args.database_url)
        try:
            return run_check_after(conn, args.check_after)
        finally:
            conn.close()
    if args.expect_plan and not args.apply:
        logger.error("REFUSED: --expect-plan only goes with --apply")
        return 2
    try:
        orphans = load_list(args.list)
        check_no_overlap(orphans, other_list_ids())
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
            orphans,
            out_csv,
            args.archive or default_out("archive"),
            plan_out=args.plan_out,
            expect_plan=expect,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
