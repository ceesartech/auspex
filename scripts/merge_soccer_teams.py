"""Merge soccer teams that ESPN minted under a second spelling into the
established team row.

THE BUG THIS REPAIRS
--------------------
fetch_upcoming resolves an ESPN competitor to a team by its display name
(exact, accent-folded, then a 0.85 fuzzy ratio within the team's first
league) and inserts a new team when nothing matches. "Manchester City",
"Tottenham Hotspur", "BK Häcken" and ~170 other clubs failed every step, so
since 2026-05-28 their fixtures have landed on a history-less twin of the
football-data row ("Man City", "Tottenham", "Hacken"). The name-keyed
Dixon-Coles and Poisson members know none of the twins, the eligibility gate
counts a club's games per team_id, and where both sources wrote the same
summer games the corpus holds the same match twice (prod 2026-10-31, read
only: 1,953 matches on the 171 twin teams, 290 twin clusters of which 171
are two finished rows of one game). Investigation: audit §7, 2026-10-31
"team identity".

THE MAP
-------
scripts/data/soccer_team_merge_map.csv, reviewed by hand: one row per split
club (new_team_id -> old_team_id), with the evidence and the ESPN team id
the new spelling carries. The tool merges ONLY pairs listed there, and
--only can narrow a run to some of them (an id not in the map is refused).

WHAT ONE PAIR DOES (one transaction per pair; a failing pair rolls back
alone and is reported, the others go ahead)
-----------------------------------------------------------------------
1. Twin fixtures first. Re-pointing the new team's matches onto the old
   team puts some of them next to a row for the same fixture: same home and
   away team within 36 h (TWIN_WINDOW). Every such cluster keeps ONE row:
     * the row holding recommendations (recs on the match, recs on one of
       its predictions, or betting_history) — two holders refuse the pair;
     * else a finished row: one with an ESPN competition id first, then the
       one with more predictions, then more odds rows. Finished rows that
       disagree on the score refuse the pair.
   A kept row that is not finished while a twin is (the rec holder is the
   stale scheduled row) ADOPTS the twin's result — status, score, half-time
   — but only when its recs are all settled or both rows share the kickoff,
   so a pending rec is never graded against a kickoff nobody vouched for;
   grade_completed_matches.py --match-id then settles it. The kept row also
   takes a dropped twin's half-time pair and ESPN id when it has none.
   Each dropped twin hands the kept row, before it is deleted:
     odds            the freshest pre-game price per current-price key
                     (book, market, selection, line, opening, live) the kept
                     row does not already hold; in-play rows never move;
     odds_snapshots  captures before kickoff the kept row lacks;
     match_stats     a team's stats row when the kept row has none for it.
   "Before kickoff" is before the EARLIEST kickoff in the cluster:
   football-data summer kickoffs sit an hour late, so the later time could
   let in-play prices through. Everything else on the dropped twin
   (predictions, later prices, features_cache, weather) goes with it. A
   prediction a rec points at is never dropped: the plan refuses that
   pair, and migration 025 makes the DELETE fail if one appears anyway.
   A cluster whose rows already share both team ids is an ESPN kickoff
   fork older than the merge (a stale 'scheduled' row next to the played
   one): the same rules fold it, but one they would refuse is left alone
   and reported instead of blocking the pair (repair_stranded_recs owns it).
2. Re-point matches.home_team_id / away_team_id from the new team to the
   old one, each UPDATE guarded on the row's observed teams and kickoff.
3. Assert nothing references the new team any more (matches, match_stats,
   players, team_managers, transfers, nhl_*) and that no same-pair twin
   within 36 h touches a re-pointed row, except the forks left alone.
4. Record the merge on the kept team: the new spelling's normalised name
   joins metadata.aliases, external_ids.espn takes the ESPN team id, and
   metadata.merged_from gets {team_id, name, espn, merged_at}.
5. Delete the new team row (guarded: zero references, same name). matches
   -> teams is ON DELETE CASCADE, so the delete runs only after step 3.

PRECONDITION: THE ALIAS RESOLVER
--------------------------------
Deleting "manchester city" frees that name: unless fetch_upcoming resolves
ESPN's spelling through the alias recorded in step 4, the next 15-minute
fetch inserts the team again (the Montreal lesson, audit §7 2026-10-26), and
fetch_live_odds would lose the odds-API spelling for every merged club.
--apply therefore refuses unless scripts/team_identity.py exposes a public
callable with "alias" in its name, reads the 'aliases' metadata key this
tool writes, and both fetch_upcoming.py and fetch_live_odds.py call it. A
dry run still plans, but exits 3 with the refusal.

SAFETY
------
The whole run is planned up front (pairs in map order, each planned on the
state the earlier ones leave), and the write-once audit CSV of every row the
run touches lands BEFORE the first write, dry run included. --apply also
writes a write-once gzip JSONL archive of every row it will delete or
change (dropped twins with all their dependents, kept rows it enriches, both
team rows) before the first write. Each pair locks its two team rows, and
its plan is re-checked against what it locks: a pair whose new team gained
or lost a match since the plan, or any guarded statement that hits a row
that moved, rolls that pair back and is reported. Re-runnable: a merged
pair plans as 'done'.

Usage (inside the api container; dry run is the default):
    python /app/scripts/merge_soccer_teams.py
    python /app/scripts/merge_soccer_teams.py --apply
    python /app/scripts/merge_soccer_teams.py --only <new_team_id>[,<new_team_id>...] --apply
Afterwards: compute_features.py --backfill-finished soccer (every re-pointed
finished match carries form computed on the thin twin's history), and
grade_completed_matches.py --match-id <id> for each kept row that adopted a
result with pending recs (the summary lists them).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gzip
import importlib.util
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("merge_soccer_teams")

SCRIPTS_DIR = Path(__file__).resolve().parent
MAP_PATH = SCRIPTS_DIR / "data" / "soccer_team_merge_map.csv"
MAP_FIELDS = ("new_team_id", "old_team_id", "new_name", "old_name", "espn_team_id", "espn_league", "method", "evidence")
TWIN_WINDOW = timedelta(hours=36)
ALIASES_KEY = "aliases"
MERGED_FROM_KEY = "merged_from"
RESOLVER_HINT = "alias"
RESOLVER_CALLERS = ("fetch_upcoming.py", "fetch_live_odds.py")
LOCK_TIMEOUT = "10s"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
FINISHED = "finished"


class MapError(ValueError):
    """The merge map is malformed or unsafe to run."""


class GuardError(RuntimeError):
    """A guarded write found the database different from the plan."""


@dataclass(frozen=True)
class MapRow:
    new_team_id: str
    old_team_id: str
    new_name: str
    old_name: str
    espn_team_id: str  # '' when unknown
    method: str
    evidence: str


@dataclass
class Deps:
    """What hangs off one match row, as far as a twin decision needs it."""

    holds_recs: bool = False
    open_recs: int = 0
    n_preds: int = 0
    odds: list[dict] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)
    match_stats: list[dict] = field(default_factory=list)


@dataclass
class Drop:
    match_id: str
    keep_id: str
    observed: dict  # the row as planned: teams, kickoff, status, scores
    odds_ids: list[str]
    snapshot_ids: list[str]
    match_stats_ids: list[str]
    cutoff: datetime  # earliest kickoff in the cluster: only prices from before it move
    fork: bool = False  # folds a pre-merge ESPN fork rather than a merge-made twin


@dataclass
class Enrich:
    match_id: str
    observed: dict
    sets: dict  # status / home_score / away_score / half_time_home / half_time_away
    external_patch: dict
    metadata_patch: dict
    reason: str
    open_recs: int = 0


@dataclass
class PairPlan:
    index: int
    pair: MapRow
    action: str  # 'merge' | 'done' | 'refuse'
    reason: str = ""
    new_match_ids: list[str] = field(default_factory=list)  # every match on the new team at plan time
    repoints: list[dict] = field(default_factory=list)
    drops: list[Drop] = field(default_factory=list)
    enrich: list[Enrich] = field(default_factory=list)
    forks: list[dict] = field(default_factory=list)  # pre-merge forks left alone: {'ids', 'reason'}
    new_team: dict = field(default_factory=dict)
    old_team: dict = field(default_factory=dict)
    team_patch: dict = field(default_factory=dict)  # aliases / merged_from / espn after the merge


def normalize_name(name: str) -> str:
    """teams.normalized_name as fetch_upcoming.normalize_team builds it."""
    return " ".join((name or "").strip().lower().split())


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------


def load_map(path: Path) -> list[MapRow]:
    """Parse and validate the merge map ('#' lines are comments). Raises
    MapError on anything that would make a run ambiguous."""
    with path.open(newline="") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    if tuple(reader.fieldnames or ()) != MAP_FIELDS:
        raise MapError(f"{path}: header {reader.fieldnames} != {list(MAP_FIELDS)}")
    rows = [
        MapRow(
            new_team_id=r["new_team_id"].strip(),
            old_team_id=r["old_team_id"].strip(),
            new_name=r["new_name"],
            old_name=r["old_name"],
            espn_team_id=(r["espn_team_id"] or "").strip(),
            method=r["method"],
            evidence=r["evidence"],
        )
        for r in reader
    ]
    if not rows:
        raise MapError(f"{path}: no pairs")
    for r in rows:
        for v in (r.new_team_id, r.old_team_id):
            if not UUID_RE.match(v):
                raise MapError(f"{path}: {v!r} is not a lowercase uuid ({r.new_name} -> {r.old_name})")
        if r.new_team_id == r.old_team_id:
            raise MapError(f"{path}: {r.new_name} maps onto itself")
        if r.espn_team_id and not r.espn_team_id.isdigit():
            raise MapError(f"{path}: espn_team_id {r.espn_team_id!r} is not numeric ({r.new_name})")
        if not r.new_name.strip() or not r.old_name.strip() or not r.evidence.strip():
            raise MapError(f"{path}: {r.new_team_id} needs both names and its evidence")
    dup = [k for k, n in Counter(r.new_team_id for r in rows).items() if n > 1]
    if dup:
        raise MapError(f"{path}: new_team_id listed more than once: {dup}")
    chained = {r.old_team_id for r in rows} & {r.new_team_id for r in rows}
    if chained:
        raise MapError(f"{path}: a kept team is itself merged away (chain): {sorted(chained)}")
    espn_by_old: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.espn_team_id:
            espn_by_old[r.old_team_id].add(r.espn_team_id)
    clash = {k: sorted(v) for k, v in espn_by_old.items() if len(v) > 1}
    if clash:
        raise MapError(f"{path}: two ESPN team ids for one kept team: {clash}")
    return rows


def select_pairs(rows: list[MapRow], only: Optional[str]) -> list[MapRow]:
    """The map, or the --only subset of it; an id outside the map is refused."""
    if not only:
        return rows
    wanted = [s.strip() for s in only.split(",") if s.strip()]
    known = {r.new_team_id for r in rows}
    unknown = [w for w in wanted if w not in known]
    if unknown:
        raise MapError(f"--only names team(s) that are not in the merge map: {unknown}")
    return [r for r in rows if r.new_team_id in set(wanted)]


# ---------------------------------------------------------------------------
# The precondition: the alias resolver must be live before any merge
# ---------------------------------------------------------------------------


def resolver_refusal(scripts_dir: Path = SCRIPTS_DIR) -> Optional[str]:
    """None when the deployed code resolves a merged spelling through the
    alias this tool records, else why not. Reads the files the pipeline runs
    (scripts/ is bind-mounted into every container)."""
    try:
        spec = importlib.util.spec_from_file_location("_merge_probe_team_identity", scripts_dir / "team_identity.py")
        assert spec is not None and spec.loader is not None
        team_identity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(team_identity)
    except Exception as e:  # an import error here is the refusal, not a crash
        return f"cannot import scripts/team_identity.py: {e}"
    names = sorted(
        n
        for n in dir(team_identity)
        if RESOLVER_HINT in n.lower() and not n.startswith("_") and callable(getattr(team_identity, n))
    )
    if not names:
        return "scripts/team_identity.py exposes no alias resolver (no public callable named *alias*)"
    source = (scripts_dir / "team_identity.py").read_text()
    if f"'{ALIASES_KEY}'" not in source and f'"{ALIASES_KEY}"' not in source:
        return f"scripts/team_identity.py never reads the {ALIASES_KEY!r} metadata key this tool writes"
    for caller in RESOLVER_CALLERS:
        path = scripts_dir / caller
        text = path.read_text() if path.exists() else ""
        if not any(n in text for n in names):
            return f"scripts/{caller} does not call the alias resolver ({', '.join(names)})"
    return None


# ---------------------------------------------------------------------------
# Planning (pure: rows in, plan out)
# ---------------------------------------------------------------------------


def _espn(m: dict) -> str:
    return str((m.get("external_ids") or {}).get("espn") or "")


def _finished(m: dict) -> bool:
    return m.get("status") == FINISHED


def _observed(m: dict) -> dict:
    return {
        "home_team_id": m["home_team_id"],
        "away_team_id": m["away_team_id"],
        "match_date": m["match_date"],
        "status": m.get("status"),
        "home_score": m.get("home_score"),
        "away_score": m.get("away_score"),
        "half_time_home": m.get("half_time_home"),
        "half_time_away": m.get("half_time_away"),
        "espn": _espn(m),
    }


def _rank(m: dict, deps: dict[str, Deps]) -> tuple:
    d = deps.get(m["id"], Deps())
    return (_finished(m), bool(_espn(m)), d.n_preds, len(d.odds) + len(d.snapshots), m["id"])


def find_clusters(new_id: str, old_id: str, state: dict[str, dict]) -> list[list[dict]]:
    """Rows that share (home, away) within TWIN_WINDOW once `new_id` reads as
    `old_id`, grouped transitively; only groups touching the new team and
    holding more than one row. Pure."""

    def canon(t: str) -> str:
        return old_id if t == new_id else t

    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in state.values():
        by_pair[(canon(m["home_team_id"]), canon(m["away_team_id"]))].append(m)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for m in state.values():
        if new_id not in (m["home_team_id"], m["away_team_id"]):
            continue
        parent.setdefault(m["id"], m["id"])
        for o in by_pair[(canon(m["home_team_id"]), canon(m["away_team_id"]))]:
            if o["id"] != m["id"] and abs(o["match_date"] - m["match_date"]) <= TWIN_WINDOW:
                parent.setdefault(o["id"], o["id"])
                parent[find(o["id"])] = find(m["id"])
    groups: dict[str, list[dict]] = defaultdict(list)
    for mid in parent:
        groups[find(mid)].append(state[mid])
    return [sorted(g, key=lambda m: (m["match_date"], m["id"])) for g in groups.values() if len(g) > 1]


def reversed_twins(new_id: str, old_id: str, state: dict[str, dict]) -> list[tuple[str, str]]:
    """New-team rows that would sit within TWIN_WINDOW of a row with the
    SAME two teams the other way round, where the merge creates that
    adjacency (rows already on the same two ids are a pre-existing data
    issue). Orientation is a human call. Pure."""

    def canon(t: str) -> str:
        return old_id if t == new_id else t

    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for o in state.values():
        by_pair[(canon(o["home_team_id"]), canon(o["away_team_id"]))].append(o)
    out = []
    for m in state.values():
        if new_id not in (m["home_team_id"], m["away_team_id"]):
            continue
        for o in by_pair[(canon(m["away_team_id"]), canon(m["home_team_id"]))]:
            if {o["home_team_id"], o["away_team_id"]} == {m["home_team_id"], m["away_team_id"]}:
                continue  # already the same two ids: not created by this merge
            if abs(o["match_date"] - m["match_date"]) <= TWIN_WINDOW:
                out.append((m["id"], o["id"]))
    return out


def decide_cluster(cluster: list[dict], deps: dict[str, Deps]) -> dict:
    """Which row of a twin cluster survives and what it takes from the rest.
    Returns {'refuse': reason} or {'keep', 'drops', 'adopt_from', 'ht_from',
    'espn_from'}. Pure."""
    span = cluster[-1]["match_date"] - cluster[0]["match_date"]
    if span > TWIN_WINDOW:
        return {"refuse": f"twin cluster spans {span} (> {TWIN_WINDOW}): more than one fixture?"}
    statuses = {m.get("status") for m in cluster}
    if "live" in statuses:
        return {"refuse": "a twin is in play"}
    if "cancelled" in statuses:
        return {"refuse": "a twin is 'cancelled' (recorded on purpose) next to another row of the fixture"}
    finished = [m for m in cluster if _finished(m)]
    if any(m.get("home_score") is None or m.get("away_score") is None for m in finished):
        return {"refuse": "a finished twin has no full-time score"}
    scores = {(m["home_score"], m["away_score"]) for m in finished}
    if len(scores) > 1:
        return {"refuse": f"finished twins disagree on the score: {sorted(scores)}"}
    espn_ids = {_espn(m) for m in cluster} - {""}
    if len(espn_ids) > 1:
        return {"refuse": f"twins carry different ESPN competition ids: {sorted(espn_ids)}"}
    holders = [m for m in cluster if deps.get(m["id"], Deps()).holds_recs]
    if len(holders) > 1:
        return {"refuse": f"{len(holders)} twins hold recommendations"}
    ranked = sorted(cluster, key=lambda m: _rank(m, deps), reverse=True)
    keep = holders[0] if holders else ranked[0]
    adopt_from = None
    if not _finished(keep) and finished:
        adopt_from = max(finished, key=lambda m: _rank(m, deps))
        if deps.get(keep["id"], Deps()).open_recs and keep["match_date"] != adopt_from["match_date"]:
            return {
                "refuse": f"the rec holder {keep['id']} is {keep.get('status')} with open recs and its kickoff "
                f"{keep['match_date']} differs from the finished twin's {adopt_from['match_date']}: settle it with "
                f"repair_stranded_recs.py after the merge, not here"
            }
    ht_from = None
    if (keep.get("half_time_home") is None or keep.get("half_time_away") is None) and finished:
        for m in sorted(finished, key=lambda m: _rank(m, deps), reverse=True):
            if m["id"] != keep["id"] and m.get("half_time_home") is not None and m.get("half_time_away") is not None:
                ht_from = m
                break
    espn_from = None
    if not _espn(keep):
        espn_from = next((m for m in ranked if _espn(m)), None)
    return {
        "keep": keep,
        "drops": [m for m in cluster if m["id"] != keep["id"]],
        "adopt_from": adopt_from,
        "ht_from": ht_from,
        "espn_from": espn_from,
    }


def _price_key(o: dict) -> tuple:
    return (
        o["bookmaker"],
        o["market_type"],
        o["selection"],
        Decimal(-1) if o.get("line") is None else Decimal(str(o["line"])),
        bool(o.get("is_opening")),
        bool(o.get("is_live")),
    )


def _snap_key(s: dict) -> tuple:
    line = Decimal(-1) if s.get("line") is None else Decimal(str(s["line"]))
    return (s["bookmaker"], s["market_type"], s["selection"], line, s["captured_at"])


def plan_moves(drop: Deps, keep: Deps, cutoff: datetime, keep_teams: set[str]) -> tuple[list, list, list]:
    """Rows of a dropped twin that move to the kept row: (odds, snapshots,
    match_stats). Mutates `keep` so a later twin of the same cluster sees
    them as already there. Pure apart from that."""
    held = {_price_key(o) for o in keep.odds}
    freshest: dict[tuple, dict] = {}
    for o in drop.odds:
        if o.get("is_live") or o.get("timestamp") is None or o["timestamp"] >= cutoff:
            continue
        k = _price_key(o)
        best = freshest.get(k)
        if best is None or (o["timestamp"], o.get("created_at") or o["timestamp"], o["id"]) > (
            best["timestamp"],
            best.get("created_at") or best["timestamp"],
            best["id"],
        ):
            freshest[k] = o
    odds = sorted((o for k, o in freshest.items() if k not in held), key=lambda o: o["id"])
    have = {_snap_key(s) for s in keep.snapshots}
    first: dict[tuple, dict] = {}
    for s in drop.snapshots:
        if s["captured_at"] >= cutoff:
            continue
        k = _snap_key(s)
        if k not in first or s["id"] < first[k]["id"]:
            first[k] = s
    snaps = sorted((s for k, s in first.items() if k not in have), key=lambda s: s["id"])
    stat_teams = {s["team_id"] for s in keep.match_stats}
    stats = sorted(
        (s for s in drop.match_stats if s["team_id"] in keep_teams and s["team_id"] not in stat_teams),
        key=lambda s: s["id"],
    )
    keep.odds = keep.odds + odds
    keep.snapshots = keep.snapshots + snaps
    keep.match_stats = keep.match_stats + stats
    return odds, snaps, stats


def assess_teams(pair: MapRow, new: Optional[dict], old: Optional[dict], refs: dict) -> tuple[str, str]:
    """('merge' | 'done' | 'refuse', reason) from the two team rows. Pure."""
    alias = normalize_name(pair.new_name)
    if old is None:
        return "refuse", "kept team row not found"
    if old.get("sport") != "soccer" or old.get("name") != pair.old_name:
        return "refuse", f"kept team is {old.get('sport')!r} {old.get('name')!r}, the map says soccer {pair.old_name!r}"
    old_espn = str((old.get("external_ids") or {}).get("espn") or "")
    meta = old.get("metadata") or {}
    if new is None:
        merged = {str(e.get("team_id")) for e in meta.get(MERGED_FROM_KEY) or [] if isinstance(e, dict)}
        if pair.new_team_id in merged and alias in (meta.get(ALIASES_KEY) or []):
            return "done", "already merged"
        return "refuse", "new team row is gone but the kept team does not record the merge"
    if new.get("sport") != "soccer" or new.get("name") != pair.new_name:
        return "refuse", f"new team is {new.get('sport')!r} {new.get('name')!r}, the map says soccer {pair.new_name!r}"
    new_espn = str((new.get("external_ids") or {}).get("espn") or "")
    if pair.espn_team_id and new_espn and new_espn != pair.espn_team_id:
        return "refuse", f"new team carries ESPN team {new_espn}, the map says {pair.espn_team_id}"
    espn = pair.espn_team_id or new_espn
    if espn and old_espn and old_espn != espn:
        return "refuse", f"kept team already carries ESPN team {old_espn}, this pair brings {espn}"
    busy = {k: v for k, v in (refs or {}).items() if v}
    if busy:
        return "refuse", f"the new team is referenced outside matches: {busy}"
    return "merge", ""


def plan_pair(
    index: int,
    pair: MapRow,
    teams: dict[str, dict],
    refs: dict[str, dict],
    state: dict[str, dict],
    deps: dict[str, Deps],
    now: datetime,
) -> PairPlan:
    """Plan one pair against the current `state` (earlier pairs applied).
    Pure, except that a merging pair records its planned moves in `deps`."""
    new, old = teams.get(pair.new_team_id), teams.get(pair.old_team_id)
    action, reason = assess_teams(pair, new, old, refs.get(pair.new_team_id, {}))
    plan = PairPlan(index=index, pair=pair, action=action, reason=reason)
    if action != "merge":
        return plan
    assert new is not None and old is not None
    plan.new_team, plan.old_team = dict(new), dict(old)
    nid, oid = pair.new_team_id, pair.old_team_id
    mine = sorted((m for m in state.values() if nid in (m["home_team_id"], m["away_team_id"])), key=lambda m: m["id"])
    plan.new_match_ids = [m["id"] for m in mine]
    selfie = [m["id"] for m in mine if oid in (m["home_team_id"], m["away_team_id"])]
    if selfie:
        return _refuse(plan, f"{len(selfie)} match(es) would pit the club against itself: {selfie[:3]}")
    flipped = reversed_twins(nid, oid, state)
    if flipped:
        return _refuse(plan, f"{len(flipped)} twin(s) with home and away swapped, e.g. {flipped[0]}")
    dropped: set[str] = set()
    scratch: dict[str, Deps] = {}  # kept rows' dependents after the planned moves; kept only if the pair merges
    for cluster in find_clusters(nid, oid, state):
        fork = len({(m["home_team_id"], m["away_team_id"]) for m in cluster}) == 1
        verdict = decide_cluster(cluster, deps)
        if "refuse" in verdict:
            if fork:  # older than the merge: fold it only when it is safe, never block the pair on it
                plan.forks.append({"ids": tuple(m["id"] for m in cluster), "reason": verdict["refuse"]})
                continue
            return _refuse(plan, f"twin cluster {[m['id'] for m in cluster]}: {verdict['refuse']}")
        keep = verdict["keep"]
        cutoff = min(m["match_date"] for m in cluster)
        keep_teams = {oid if t == nid else t for t in (keep["home_team_id"], keep["away_team_id"])}
        keep_deps = scratch.setdefault(keep["id"], dataclasses.replace(deps.get(keep["id"], Deps())))
        for d in verdict["drops"]:
            odds, snaps, stats = plan_moves(deps.get(d["id"], Deps()), keep_deps, cutoff, keep_teams)
            plan.drops.append(
                Drop(
                    match_id=d["id"],
                    keep_id=keep["id"],
                    observed=_observed(d),
                    odds_ids=[o["id"] for o in odds],
                    snapshot_ids=[s["id"] for s in snaps],
                    match_stats_ids=[s["id"] for s in stats],
                    cutoff=cutoff,
                    fork=fork,
                )
            )
            dropped.add(d["id"])
        enrich = _enrichment(keep, verdict, keep_deps)
        if enrich:
            plan.enrich.append(enrich)
    for m in mine:
        if m["id"] in dropped:
            continue
        side = "home" if m["home_team_id"] == nid else "away"
        plan.repoints.append({"match_id": m["id"], "side": side, "observed": _observed(m)})
    plan.team_patch = _team_patch(pair, new, old, now)
    deps.update(scratch)
    return plan


def _refuse(plan: PairPlan, reason: str) -> PairPlan:
    plan.action, plan.reason = "refuse", reason
    plan.repoints, plan.drops, plan.enrich, plan.forks = [], [], [], []
    return plan


def _enrichment(keep: dict, verdict: dict, keep_deps: Deps) -> Optional[Enrich]:
    sets: dict[str, Any] = {}
    external: dict[str, str] = {}
    metadata: dict[str, Any] = {}
    why = []
    src = verdict["adopt_from"]
    if src is not None:
        sets.update(status=FINISHED, home_score=src["home_score"], away_score=src["away_score"])
        metadata["result_adopted_from"] = src["id"]
        if (src.get("metadata") or {}).get("result_source"):
            metadata["result_source"] = src["metadata"]["result_source"]
        why.append(f"adopt result {src['home_score']}-{src['away_score']} from {src['id']}")
    ht = verdict["ht_from"]
    if ht is not None:
        sets.update(half_time_home=ht["half_time_home"], half_time_away=ht["half_time_away"])
        why.append(f"half-time {ht['half_time_home']}-{ht['half_time_away']} from {ht['id']}")
    espn = verdict["espn_from"]
    if espn is not None:
        external["espn"] = _espn(espn)
        why.append(f"ESPN id {external['espn']} from {espn['id']}")
    if not why:
        return None
    return Enrich(
        match_id=keep["id"],
        observed=_observed(keep),
        sets=sets,
        external_patch=external,
        metadata_patch=metadata,
        reason="; ".join(why),
        open_recs=keep_deps.open_recs if src is not None else 0,
    )


def _team_patch(pair: MapRow, new: dict, old: dict, now: datetime) -> dict:
    meta = old.get("metadata") or {}
    aliases = list(meta.get(ALIASES_KEY) or [])
    alias = normalize_name(pair.new_name)
    if alias not in aliases:
        aliases.append(alias)
    new_espn = str((new.get("external_ids") or {}).get("espn") or "")
    espn = pair.espn_team_id or new_espn
    merged_from = list(meta.get(MERGED_FROM_KEY) or [])
    merged_from.append(
        {"team_id": pair.new_team_id, "name": pair.new_name, "espn": espn or None, "merged_at": now.isoformat()}
    )
    return {
        "observed_aliases": list(meta.get(ALIASES_KEY) or []),
        "observed_merged_from": list(meta.get(MERGED_FROM_KEY) or []),
        "observed_espn": str((old.get("external_ids") or {}).get("espn") or ""),
        "aliases": sorted(aliases),
        "merged_from": merged_from,
        "external_patch": {"espn": espn} if espn else {},
    }


def apply_to_state(plan: PairPlan, teams: dict[str, dict], state: dict[str, dict]) -> None:
    """Advance the in-memory state by one planned pair, so the next pair is
    planned on what this one leaves behind."""
    if plan.action != "merge":
        return
    for d in plan.drops:
        state.pop(d.match_id, None)
    for e in plan.enrich:
        row = state[e.match_id]
        row.update(e.sets)
        row["external_ids"] = {**(row.get("external_ids") or {}), **e.external_patch}
        row["metadata"] = {**(row.get("metadata") or {}), **e.metadata_patch}
    for r in plan.repoints:
        state[r["match_id"]][f"{r['side']}_team_id"] = plan.pair.old_team_id
    old = teams[plan.pair.old_team_id]
    old["metadata"] = {
        **(old.get("metadata") or {}),
        ALIASES_KEY: plan.team_patch["aliases"],
        MERGED_FROM_KEY: plan.team_patch["merged_from"],
    }
    old["external_ids"] = {**(old.get("external_ids") or {}), **plan.team_patch["external_patch"]}
    teams.pop(plan.pair.new_team_id, None)


def plan_all(
    pairs: list[MapRow],
    teams: dict[str, dict],
    refs: dict[str, dict],
    matches: dict[str, dict],
    deps: dict[str, Deps],
    now: datetime,
) -> list[PairPlan]:
    """Plan every pair in map order on a private copy of the state."""
    teams = {k: dict(v) for k, v in teams.items()}
    state = {k: dict(v) for k, v in matches.items()}
    deps = {k: dataclasses.replace(v) for k, v in deps.items()}  # plan_moves re-binds the lists, never edits them
    plans = []
    for i, pair in enumerate(pairs, 1):
        plan = plan_pair(i, pair, teams, refs, state, deps, now)
        apply_to_state(plan, teams, state)
        plans.append(plan)
    return plans


def cluster_member_ids(pairs: list[MapRow], matches: dict[str, dict]) -> set[str]:
    """Every row a twin search can reach, whichever pairs merge or refuse —
    the rows whose dependents the plan must load. Folding ALL the pairs at
    once is the coarsest identity any run can reach, so its same-pair chains
    within TWIN_WINDOW (holding a row of a merged team) cover every cluster."""
    canon = {p.new_team_id: p.old_team_id for p in pairs}
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in matches.values():
        by_pair[
            (canon.get(m["home_team_id"], m["home_team_id"]), canon.get(m["away_team_id"], m["away_team_id"]))
        ].append(m)
    out: set[str] = set()

    def flush(chain: list[dict]) -> None:
        if len(chain) > 1 and any(t in canon for r in chain for t in (r["home_team_id"], r["away_team_id"])):
            out.update(r["id"] for r in chain)

    for rows in by_pair.values():
        rows = sorted(rows, key=lambda m: m["match_date"])
        chain = [rows[0]]
        for m in rows[1:]:
            if m["match_date"] - chain[-1]["match_date"] <= TWIN_WINDOW:
                chain.append(m)
            else:
                flush(chain)
                chain = [m]
        flush(chain)
    return out


def summarize(plans: list[PairPlan]) -> Counter:
    c: Counter = Counter()
    for p in plans:
        c[f"pairs_{p.action}"] += 1
        if p.action != "merge":
            continue
        c["matches_repointed"] += len(p.repoints)
        c["twins_dropped"] += len(p.drops)
        c["twins_dropped_finished"] += sum(1 for d in p.drops if d.observed["status"] == FINISHED)
        c["kept_rows_enriched"] += len(p.enrich)
        c["results_adopted"] += sum(1 for e in p.enrich if "status" in e.sets)
        c["odds_moved"] += sum(len(d.odds_ids) for d in p.drops)
        c["snapshots_moved"] += sum(len(d.snapshot_ids) for d in p.drops)
        c["match_stats_moved"] += sum(len(d.match_stats_ids) for d in p.drops)
        c["forks_folded"] += len({d.keep_id for d in p.drops if d.fork})
        c["forks_left_alone"] += len(p.forks)
        c["teams_deleted"] += 1
    return c


# ---------------------------------------------------------------------------
# Audit CSV + archive (write-once, before any write)
# ---------------------------------------------------------------------------

CSV_FIELDS = (
    "pair",
    "new_team_id",
    "new_name",
    "old_team_id",
    "old_name",
    "action",
    "table",
    "row_id",
    "keep_id",
    "before",
    "after",
    "reason",
)


def _j(v: Any) -> str:
    return json.dumps(v, default=_jsonable, sort_keys=True)


def _jsonable(v: Any) -> Any:
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return str(v)
    return str(v)


def csv_rows(plans: list[PairPlan]) -> list[dict]:
    """One row per row the run touches (or per pair it refuses / skips)."""
    out: list[dict] = []
    for p in plans:
        pair = p.pair

        def add(action: str, table: str, row_id: str, **kw: Any) -> None:
            out.append(
                {
                    "pair": p.index,
                    "new_team_id": pair.new_team_id,
                    "new_name": pair.new_name,
                    "old_team_id": pair.old_team_id,
                    "old_name": pair.old_name,
                    "action": action,
                    "table": table,
                    "row_id": row_id,
                    **kw,
                }
            )

        if p.action != "merge":
            add(f"pair_{p.action}", "teams", pair.new_team_id, reason=p.reason)
            continue
        for d in p.drops:
            why = "pre-merge ESPN fork of keep_id" if d.fork else "same fixture as keep_id within 36h"
            add("drop_twin", "matches", d.match_id, keep_id=d.keep_id, before=_j(d.observed), reason=why)
            for table, ids in (
                ("odds", d.odds_ids),
                ("odds_snapshots", d.snapshot_ids),
                ("match_stats", d.match_stats_ids),
            ):
                for rid in ids:
                    add(
                        "move",
                        table,
                        rid,
                        keep_id=d.keep_id,
                        before=_j({"match_id": d.match_id}),
                        after=_j({"match_id": d.keep_id}),
                    )
        for e in p.enrich:
            after = {**e.sets, "external_ids": e.external_patch, "metadata": e.metadata_patch}
            add("enrich_keep", "matches", e.match_id, before=_j(e.observed), after=_j(after), reason=e.reason)
        for r in p.repoints:
            add(
                f"repoint_{r['side']}",
                "matches",
                r["match_id"],
                before=_j(r["observed"]),
                after=_j({f"{r['side']}_team_id": pair.old_team_id}),
            )
        for f in p.forks:
            add("leave_fork", "matches", ";".join(f["ids"]), reason=f"pre-merge ESPN fork left alone: {f['reason']}")
        t = p.team_patch
        add(
            "record_alias",
            "teams",
            pair.old_team_id,
            before=_j({"aliases": t["observed_aliases"], "espn": t["observed_espn"]}),
            after=_j({"aliases": t["aliases"], "external_ids": t["external_patch"]}),
        )
        add(
            "delete_team",
            "teams",
            pair.new_team_id,
            before=_j({k: p.new_team.get(k) for k in ("name", "normalized_name", "external_ids", "metadata")}),
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as fh:  # exclusive create: the audit trail is write-once
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


ARCHIVE_CHILDREN = (
    "odds",
    "odds_snapshots",
    "predictions",
    "features_cache",
    "match_stats",
    "match_weather",
    "weather",
    "player_match_stats",
)


def write_archive(cur, path: Path, plans: list[PairPlan]) -> int:
    """gzip JSONL of every row the run deletes or changes, written before the
    first write. Returns the record count."""
    merges = [p for p in plans if p.action == "merge"]
    drop_ids = [d.match_id for p in merges for d in p.drops]
    keep_of = {d.match_id: d.keep_id for p in merges for d in p.drops}
    enrich_ids = [e.match_id for p in merges for e in p.enrich]
    team_ids = [t for p in merges for t in (p.pair.new_team_id, p.pair.old_team_id)]
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "xt") as fh:

        def emit(table: str, role: str, rows: Iterable[dict], **extra: Any) -> None:
            nonlocal n
            for r in rows:
                rec = {"table": table, "role": role, **extra, "row": {k: _jsonable_row(v) for k, v in r.items()}}
                if table == "matches" and role == "dropped_twin":
                    rec["keep_id"] = keep_of.get(str(r["id"]))
                fh.write(json.dumps(rec) + "\n")
                n += 1

        cur.execute("SELECT * FROM teams WHERE id = ANY(%s::uuid[])", (team_ids,))
        emit("teams", "team_before", cur.fetchall())
        if enrich_ids:
            cur.execute("SELECT * FROM matches WHERE id = ANY(%s::uuid[])", (enrich_ids,))
            emit("matches", "kept_before", cur.fetchall())
        for i in range(0, len(drop_ids), 500):
            chunk = drop_ids[i : i + 500]
            cur.execute("SELECT * FROM matches WHERE id = ANY(%s::uuid[])", (chunk,))
            emit("matches", "dropped_twin", cur.fetchall())
            for table in ARCHIVE_CHILDREN:
                cur.execute(f"SELECT * FROM {table} WHERE match_id = ANY(%s::uuid[])", (chunk,))
                emit(table, "dropped_twin_child", cur.fetchall())
    return n


def _jsonable_row(v: Any) -> Any:
    if isinstance(v, (datetime, date, Decimal)):
        return _jsonable(v)
    if v is None or isinstance(v, (str, int, float, bool, list, dict)):
        return v
    return str(v)  # uuid and friends


# ---------------------------------------------------------------------------
# Reading the database
# ---------------------------------------------------------------------------

TEAMS_SQL = """
    SELECT id::text AS id, name, normalized_name, sport, external_ids, metadata
      FROM teams WHERE id = ANY(%s::uuid[])
"""

TEAM_REFS_SQL = """
    SELECT t.id::text AS id,
           (SELECT count(*) FROM match_stats x WHERE x.team_id = t.id) AS match_stats,
           (SELECT count(*) FROM players x WHERE x.team_id = t.id) AS players,
           (SELECT count(*) FROM team_managers x WHERE x.team_id = t.id) AS team_managers,
           (SELECT count(*) FROM transfers x WHERE x.from_team_id = t.id OR x.to_team_id = t.id) AS transfers,
           (SELECT count(*) FROM nhl_match_stats x WHERE x.team_id = t.id) AS nhl_match_stats,
           (SELECT count(*) FROM nhl_match_advanced_stats x WHERE x.team_id = t.id) AS nhl_match_advanced_stats
      FROM teams t WHERE t.id = ANY(%s::uuid[])
"""

# Every match of a merged team, plus the kept teams' rows from TWIN_WINDOW
# before the first of those on: a twin can only sit that close.
MATCHES_SQL = """
    WITH lo AS (
        SELECT min(match_date) - %(window)s::interval AS since
          FROM matches
         WHERE home_team_id = ANY(%(new_ids)s::uuid[]) OR away_team_id = ANY(%(new_ids)s::uuid[])
    )
    SELECT m.id::text AS id, m.home_team_id::text AS home_team_id, m.away_team_id::text AS away_team_id,
           m.match_date, m.status, m.home_score, m.away_score, m.half_time_home, m.half_time_away,
           m.external_ids, m.metadata
      FROM matches m, lo
     WHERE m.home_team_id = ANY(%(new_ids)s::uuid[]) OR m.away_team_id = ANY(%(new_ids)s::uuid[])
        OR ((m.home_team_id = ANY(%(old_ids)s::uuid[]) OR m.away_team_id = ANY(%(old_ids)s::uuid[]))
            AND m.match_date >= lo.since)
"""

DEPS_SQL = """
    SELECT m.id::text AS id,
           EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.match_id = m.id)
             OR EXISTS (SELECT 1 FROM betting_history h WHERE h.match_id = m.id)
             OR EXISTS (SELECT 1 FROM betting_recommendations r
                          JOIN predictions p ON p.id = r.prediction_id
                         WHERE p.match_id = m.id) AS holds_recs,
           (SELECT count(*) FROM betting_recommendations r
             WHERE r.status IN ('pending', 'placed')
               AND (r.match_id = m.id
                    OR r.prediction_id IN (SELECT p.id FROM predictions p WHERE p.match_id = m.id))) AS open_recs,
           (SELECT count(*) FROM predictions p WHERE p.match_id = m.id) AS n_preds
      FROM matches m WHERE m.id = ANY(%s::uuid[])
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

MATCH_STATS_SQL = """
    SELECT id::text AS id, match_id::text AS match_id, team_id::text AS team_id
      FROM match_stats WHERE match_id = ANY(%s::uuid[])
"""


def load_state(cur, pairs: list[MapRow]) -> tuple[dict, dict, dict, dict]:
    """(teams, refs, matches, deps) for the pairs — read-only."""
    new_ids = [p.new_team_id for p in pairs]
    old_ids = sorted({p.old_team_id for p in pairs})
    cur.execute(TEAMS_SQL, (new_ids + old_ids,))
    teams = {r["id"]: dict(r) for r in cur.fetchall()}
    cur.execute(TEAM_REFS_SQL, (new_ids,))
    refs = {r.pop("id"): {k: int(v) for k, v in r.items()} for r in (dict(x) for x in cur.fetchall())}
    cur.execute(
        MATCHES_SQL,
        {"window": f"{int(TWIN_WINDOW.total_seconds() // 3600)} hours", "new_ids": new_ids, "old_ids": old_ids},
    )
    matches = {r["id"]: dict(r) for r in cur.fetchall()}
    members = sorted(cluster_member_ids(pairs, matches))
    deps: dict[str, Deps] = {mid: Deps() for mid in members}
    for i in range(0, len(members), 500):
        chunk = members[i : i + 500]
        cur.execute(DEPS_SQL, (chunk,))
        for r in cur.fetchall():
            d = deps[r["id"]]
            d.holds_recs, d.open_recs, d.n_preds = bool(r["holds_recs"]), int(r["open_recs"]), int(r["n_preds"])
        for sql, attr in ((ODDS_SQL, "odds"), (SNAPSHOTS_SQL, "snapshots"), (MATCH_STATS_SQL, "match_stats")):
            cur.execute(sql, (chunk,))
            for r in cur.fetchall():
                getattr(deps[r["match_id"]], attr).append(dict(r))
    return teams, refs, matches, deps


# ---------------------------------------------------------------------------
# Writing one pair (one transaction)
# ---------------------------------------------------------------------------

LOCK_TEAMS_SQL = """
    SELECT id::text AS id, name, normalized_name, sport, external_ids, metadata
      FROM teams WHERE id = ANY(%s::uuid[]) FOR UPDATE
"""

LOCK_NEW_TEAM_MATCHES_SQL = """
    SELECT id::text AS id FROM matches WHERE home_team_id = %(t)s::uuid OR away_team_id = %(t)s::uuid FOR UPDATE
"""

LOCK_ROWS_SQL = "SELECT id FROM matches WHERE id = ANY(%s::uuid[]) FOR UPDATE"

MOVE_ODDS_SQL = """
    UPDATE odds o SET match_id = %(keep)s::uuid
     WHERE o.id = ANY(%(ids)s::uuid[])
       AND o.match_id = %(drop)s::uuid
       AND o."timestamp" < %(cutoff)s
       AND NOT COALESCE(o.is_live, false)
       AND NOT EXISTS (
             SELECT 1 FROM odds t
              WHERE t.match_id = %(keep)s::uuid
                AND t.bookmaker = o.bookmaker
                AND t.market_type = o.market_type
                AND t.selection = o.selection
                AND COALESCE(t.line, -1) = COALESCE(o.line, -1)
                AND COALESCE(t.is_opening, false) = COALESCE(o.is_opening, false)
                AND COALESCE(t.is_live, false) = COALESCE(o.is_live, false))
"""

MOVE_SNAPSHOTS_SQL = """
    UPDATE odds_snapshots s SET match_id = %(keep)s::uuid
     WHERE s.id = ANY(%(ids)s::uuid[])
       AND s.match_id = %(drop)s::uuid
       AND s.captured_at < %(cutoff)s
       AND NOT EXISTS (
             SELECT 1 FROM odds_snapshots t
              WHERE t.match_id = %(keep)s::uuid
                AND t.bookmaker = s.bookmaker
                AND t.market_type = s.market_type
                AND t.selection = s.selection
                AND COALESCE(t.line, -1) = COALESCE(s.line, -1)
                AND t.captured_at = s.captured_at)
"""

MOVE_MATCH_STATS_SQL = """
    UPDATE match_stats s SET match_id = %(keep)s::uuid
     WHERE s.id = ANY(%(ids)s::uuid[])
       AND s.match_id = %(drop)s::uuid
       AND NOT EXISTS (SELECT 1 FROM match_stats t WHERE t.match_id = %(keep)s::uuid AND t.team_id = s.team_id)
"""

# A rec on one of the twin's predictions makes this DELETE fail (migration
# 025, NO ACTION): that FK error is the last guard, the plan refuses first.
DROP_TWIN_SQL = """
    DELETE FROM matches m
     WHERE m.id = %(id)s::uuid
       AND m.home_team_id = %(home_team_id)s::uuid
       AND m.away_team_id = %(away_team_id)s::uuid
       AND m.match_date = %(match_date)s
       AND m.status = %(status)s
       AND m.home_score IS NOT DISTINCT FROM %(home_score)s
       AND m.away_score IS NOT DISTINCT FROM %(away_score)s
       AND NOT EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.match_id = m.id)
       AND NOT EXISTS (SELECT 1 FROM betting_history h WHERE h.match_id = m.id)
"""

ENRICH_SQL = """
    UPDATE matches
       SET status = %(status)s,
           home_score = %(home_score)s,
           away_score = %(away_score)s,
           half_time_home = %(half_time_home)s,
           half_time_away = %(half_time_away)s,
           external_ids = COALESCE(external_ids, '{}'::jsonb) || %(external_patch)s::jsonb,
           metadata = COALESCE(metadata, '{}'::jsonb) || %(metadata_patch)s::jsonb,
           updated_at = NOW()
     WHERE id = %(id)s::uuid
       AND status = %(o_status)s
       AND home_score IS NOT DISTINCT FROM %(o_home_score)s
       AND away_score IS NOT DISTINCT FROM %(o_away_score)s
       AND half_time_home IS NOT DISTINCT FROM %(o_half_time_home)s
       AND half_time_away IS NOT DISTINCT FROM %(o_half_time_away)s
       AND COALESCE(external_ids->>'espn', '') = %(o_espn)s
"""

REPOINT_SQL = {
    "home": """
        UPDATE matches SET home_team_id = %(old)s::uuid, updated_at = NOW()
         WHERE id = %(id)s::uuid AND home_team_id = %(new)s::uuid
           AND away_team_id = %(away_team_id)s::uuid AND match_date = %(match_date)s
    """,
    "away": """
        UPDATE matches SET away_team_id = %(old)s::uuid, updated_at = NOW()
         WHERE id = %(id)s::uuid AND away_team_id = %(new)s::uuid
           AND home_team_id = %(home_team_id)s::uuid AND match_date = %(match_date)s
    """,
}

NEW_TEAM_REFS_SQL = """
    SELECT (SELECT count(*) FROM matches WHERE home_team_id = %(t)s::uuid OR away_team_id = %(t)s::uuid) AS matches,
           (SELECT count(*) FROM match_stats WHERE team_id = %(t)s::uuid) AS match_stats,
           (SELECT count(*) FROM players WHERE team_id = %(t)s::uuid) AS players,
           (SELECT count(*) FROM team_managers WHERE team_id = %(t)s::uuid) AS team_managers,
           (SELECT count(*) FROM transfers WHERE from_team_id = %(t)s::uuid OR to_team_id = %(t)s::uuid) AS transfers,
           (SELECT count(*) FROM nhl_match_stats WHERE team_id = %(t)s::uuid) AS nhl_match_stats,
           (SELECT count(*) FROM nhl_match_advanced_stats WHERE team_id = %(t)s::uuid) AS nhl_match_advanced_stats
"""

TWINS_AFTER_SQL = """
    SELECT m.id::text AS a, o.id::text AS b
      FROM matches m
      JOIN matches o ON o.home_team_id = m.home_team_id AND o.away_team_id = m.away_team_id AND o.id <> m.id
                    AND o.match_date BETWEEN m.match_date - %(window)s::interval AND m.match_date + %(window)s::interval
     WHERE m.id = ANY(%(ids)s::uuid[])
"""

TEAM_RECORD_SQL = """
    UPDATE teams
       SET metadata = jsonb_set(jsonb_set(COALESCE(metadata, '{}'::jsonb), '{aliases}', %(aliases)s::jsonb),
                                '{merged_from}', %(merged_from)s::jsonb),
           external_ids = COALESCE(external_ids, '{}'::jsonb) || %(external_patch)s::jsonb,
           updated_at = NOW()
     WHERE id = %(id)s::uuid AND sport = 'soccer' AND name = %(name)s
       AND COALESCE(metadata->'aliases', '[]'::jsonb) = %(observed_aliases)s::jsonb
       AND COALESCE(metadata->'merged_from', '[]'::jsonb) = %(observed_merged_from)s::jsonb
       AND COALESCE(external_ids->>'espn', '') = %(observed_espn)s
"""

TEAM_DELETE_SQL = """
    DELETE FROM teams t
     WHERE t.id = %(id)s::uuid AND t.sport = 'soccer' AND t.name = %(name)s AND t.normalized_name = %(normalized_name)s
       AND NOT EXISTS (SELECT 1 FROM matches m WHERE m.home_team_id = t.id OR m.away_team_id = t.id)
       AND NOT EXISTS (SELECT 1 FROM match_stats x WHERE x.team_id = t.id)
       AND NOT EXISTS (SELECT 1 FROM players x WHERE x.team_id = t.id)
       AND NOT EXISTS (SELECT 1 FROM team_managers x WHERE x.team_id = t.id)
       AND NOT EXISTS (SELECT 1 FROM transfers x WHERE x.from_team_id = t.id OR x.to_team_id = t.id)
       AND NOT EXISTS (SELECT 1 FROM nhl_match_stats x WHERE x.team_id = t.id)
       AND NOT EXISTS (SELECT 1 FROM nhl_match_advanced_stats x WHERE x.team_id = t.id)
"""


def _expect(cur, sql: str, params: Any, n: int, what: str) -> None:
    cur.execute(sql, params)
    if cur.rowcount != n:
        raise GuardError(f"{what}: {cur.rowcount} row(s) changed, the plan expected {n}")


def _check_teams(rows: list[dict], plan: PairPlan) -> None:
    by_id = {r["id"]: r for r in rows}
    new, old = by_id.get(plan.pair.new_team_id), by_id.get(plan.pair.old_team_id)
    for label, live, seen in (("new", new, plan.new_team), ("old", old, plan.old_team)):
        if live is None:
            raise GuardError(f"{label} team {seen.get('id')} is gone")
        for k in ("name", "normalized_name", "sport", "external_ids", "metadata"):
            if (live.get(k) or None) != (seen.get(k) or None):
                raise GuardError(f"{label} team {seen.get('id')} changed since the plan ({k})")


def apply_pair(conn, plan: PairPlan) -> Counter:
    """Write one planned pair in one transaction; GuardError / psycopg2.Error
    roll it back. Returns what it wrote."""
    pair = plan.pair
    done: Counter = Counter()
    window = f"{int(TWIN_WINDOW.total_seconds() // 3600)} hours"
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SET LOCAL lock_timeout = %s", (LOCK_TIMEOUT,))
            cur.execute(LOCK_TEAMS_SQL, ([pair.new_team_id, pair.old_team_id],))
            _check_teams([dict(r) for r in cur.fetchall()], plan)
            cur.execute(LOCK_NEW_TEAM_MATCHES_SQL, {"t": pair.new_team_id})
            live = sorted(r["id"] for r in cur.fetchall())
            if live != sorted(plan.new_match_ids):
                gained = sorted(set(live) - set(plan.new_match_ids))
                lost = sorted(set(plan.new_match_ids) - set(live))
                raise GuardError(f"new team's matches changed since the plan (gained {gained[:3]}, lost {lost[:3]})")
            touched = sorted({d.match_id for d in plan.drops} | {d.keep_id for d in plan.drops})
            if touched:
                cur.execute(LOCK_ROWS_SQL, (touched,))
            for d in plan.drops:
                for sql, ids, label in (
                    (MOVE_ODDS_SQL, d.odds_ids, "odds"),
                    (MOVE_SNAPSHOTS_SQL, d.snapshot_ids, "odds_snapshots"),
                    (MOVE_MATCH_STATS_SQL, d.match_stats_ids, "match_stats"),
                ):
                    if ids:
                        _expect(
                            cur,
                            sql,
                            {"keep": d.keep_id, "drop": d.match_id, "ids": ids, "cutoff": d.cutoff},
                            len(ids),
                            f"move {label} {d.match_id} -> {d.keep_id}",
                        )
                        done[f"moved_{label}"] += len(ids)
                _expect(cur, DROP_TWIN_SQL, {"id": d.match_id, **d.observed}, 1, f"drop twin {d.match_id}")
                done["twins_dropped"] += 1
            for e in plan.enrich:
                o = e.observed
                params = {
                    "id": e.match_id,
                    "status": e.sets.get("status", o["status"]),
                    "home_score": e.sets.get("home_score", o["home_score"]),
                    "away_score": e.sets.get("away_score", o["away_score"]),
                    "half_time_home": e.sets.get("half_time_home", o["half_time_home"]),
                    "half_time_away": e.sets.get("half_time_away", o["half_time_away"]),
                    "external_patch": json.dumps(e.external_patch),
                    "metadata_patch": json.dumps(e.metadata_patch),
                    **{
                        f"o_{k}": o[k]
                        for k in ("status", "home_score", "away_score", "half_time_home", "half_time_away", "espn")
                    },
                }
                _expect(cur, ENRICH_SQL, params, 1, f"enrich kept row {e.match_id}")
                done["kept_rows_enriched"] += 1
            for r in plan.repoints:
                o = r["observed"]
                params = {
                    "id": r["match_id"],
                    "old": pair.old_team_id,
                    "new": pair.new_team_id,
                    "home_team_id": o["home_team_id"],
                    "away_team_id": o["away_team_id"],
                    "match_date": o["match_date"],
                }
                _expect(cur, REPOINT_SQL[r["side"]], params, 1, f"repoint {r['side']} of {r['match_id']}")
                done["matches_repointed"] += 1
            cur.execute(NEW_TEAM_REFS_SQL, {"t": pair.new_team_id})
            refs = {k: int(v) for k, v in dict(cur.fetchone()).items() if v}
            if refs:
                raise GuardError(f"new team still referenced after the re-point: {refs}")
            ids = [r["match_id"] for r in plan.repoints]
            if ids:
                cur.execute(TWINS_AFTER_SQL, {"ids": ids, "window": window})
                found = {tuple(sorted((r["a"], r["b"]))) for r in cur.fetchall()}
                forks = {(a, b) for f in plan.forks for a in f["ids"] for b in f["ids"] if a < b}
                left = sorted(found - forks)
                if left:
                    raise GuardError(f"{len(left)} same-pair twin(s) within 36h remain after the re-point: {left[:3]}")
            t = plan.team_patch
            _expect(
                cur,
                TEAM_RECORD_SQL,
                {
                    "id": pair.old_team_id,
                    "name": pair.old_name,
                    "aliases": json.dumps(t["aliases"]),
                    "merged_from": json.dumps(t["merged_from"]),
                    "external_patch": json.dumps(t["external_patch"]),
                    "observed_aliases": json.dumps(t["observed_aliases"]),
                    "observed_merged_from": json.dumps(t["observed_merged_from"]),
                    "observed_espn": t["observed_espn"],
                },
                1,
                f"record alias on {pair.old_name}",
            )
            _expect(
                cur,
                TEAM_DELETE_SQL,
                {"id": pair.new_team_id, "name": pair.new_name, "normalized_name": plan.new_team["normalized_name"]},
                1,
                f"delete team {pair.new_name}",
            )
            done["teams_deleted"] += 1
        conn.commit()
        return done
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _stamp() -> str:
    # Microseconds: a dry run and an apply in the same second must not collide on the write-once names.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def default_out(kind: str) -> Path:
    base = Path(os.environ.get("BACKUP_LOCAL_DIR", "/app/backups"))
    return base / (f"merge_soccer_teams_{_stamp()}.csv" if kind == "csv" else f"merge_soccer_teams_{_stamp()}.jsonl.gz")


def log_plan(plans: list[PairPlan]) -> None:
    for p in plans:
        tag = f"[{p.index}/{len(plans)}] {p.pair.new_name} -> {p.pair.old_name}"
        if p.action == "refuse":
            logger.error("%s: REFUSED — %s", tag, p.reason)
        elif p.action == "done":
            logger.info("%s: done (%s)", tag, p.reason)
        else:
            logger.info(
                "%s: repoint %d, drop %d twin(s), enrich %d, leave %d fork(s)",
                tag,
                len(p.repoints),
                len(p.drops),
                len(p.enrich),
                len(p.forks),
            )


def adopted_with_open_recs(plans: list[PairPlan]) -> list[str]:
    return [e.match_id for p in plans if p.action == "merge" for e in p.enrich if "status" in e.sets and e.open_recs]


def run(
    conn, apply_changes: bool, pairs: list[MapRow], out_csv: Path, archive: Path, resolver_reason: Optional[str]
) -> int:
    if apply_changes and resolver_reason:
        logger.error("REFUSED: %s — ship the alias resolver first; nothing read or written", resolver_reason)
        return 2
    now = datetime.now(timezone.utc)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        teams, refs, matches, deps = load_state(cur, pairs)
    conn.rollback()  # close the read snapshot
    plans = plan_all(pairs, teams, refs, matches, deps, now)
    write_csv(out_csv, csv_rows(plans))
    log_plan(plans)
    planned = summarize(plans)
    logger.info("Planned: %s", dict(sorted(planned.items())))
    logger.info("Audit CSV: %s", out_csv)
    # One fixed-format line for scripts/ops/merge_soccer_teams.sh to parse.
    logger.info(
        "PLAN pairs_merge=%d pairs_done=%d pairs_refuse=%d",
        planned["pairs_merge"],
        planned["pairs_done"],
        planned["pairs_refuse"],
    )
    for mid in adopted_with_open_recs(plans):
        logger.info("adopts a result with open recs, grade after: grade_completed_matches.py --match-id %s", mid)
    if resolver_reason:
        logger.error("DRY-RUN ONLY: --apply would refuse: %s", resolver_reason)
        return 3
    refused = planned["pairs_refuse"]
    if not apply_changes:
        logger.info("DRY-RUN: nothing written; re-run with --apply to merge %d pair(s)", planned["pairs_merge"])
        return 1 if refused else 0
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        n = write_archive(cur, archive, plans)
    conn.rollback()
    logger.info("Archived %d row(s) to %s", n, archive)
    written: Counter = Counter()
    failed = []
    for p in plans:
        if p.action != "merge":
            continue
        try:
            written.update(apply_pair(conn, p))
            written["pairs_merged"] += 1
        except (GuardError, psycopg2.Error) as e:
            failed.append(p)
            logger.error(
                "[%d] %s -> %s: ROLLED BACK — %s",
                p.index,
                p.pair.new_name,
                p.pair.old_name,
                str(e).strip().splitlines()[0],
            )
    logger.info("Written: %s", dict(sorted(written.items())))
    logger.info("WRITTEN pairs_merged=%d pairs_failed=%d", written["pairs_merged"], len(failed))
    if failed or refused:
        logger.error(
            "%d pair(s) rolled back, %d refused — re-run after inspecting them (merged pairs plan as done)",
            len(failed),
            refused,
        )
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="Write (default: dry run).")
    p.add_argument("--only", default=None, help="Comma-separated new_team_ids from the map to merge (default: all).")
    p.add_argument("--map", type=Path, default=MAP_PATH, help=argparse.SUPPRESS)
    p.add_argument("--out-csv", type=Path, default=None, help="Audit CSV (default $BACKUP_LOCAL_DIR or /app/backups).")
    p.add_argument("--archive", type=Path, default=None, help="gzip JSONL archive written before --apply writes.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    try:
        pairs = select_pairs(load_map(args.map), args.only)
    except MapError as e:
        logger.error("REFUSED: %s", e)
        return 2
    conn = psycopg2.connect(args.database_url)
    try:
        return run(
            conn,
            args.apply,
            pairs,
            args.out_csv or default_out("csv"),
            args.archive or default_out("archive"),
            resolver_refusal(),
        )
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
