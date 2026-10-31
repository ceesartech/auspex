"""Merge soccer teams that ESPN minted under a second spelling into the
established team row.

THE BUG THIS REPAIRS
--------------------
fetch_upcoming used to resolve an ESPN competitor by display name only and
inserted a new team when nothing matched. "Manchester City", "Tottenham
Hotspur", "BK Häcken" and ~170 other clubs missed, so since 2026-05-28 their
fixtures have landed on a history-less twin of the football-data row ("Man
City", "Tottenham", "Hacken"). The name-keyed Dixon-Coles and Poisson members
know none of the twins, the eligibility gate counts a club's games per
team_id, and where both sources wrote the same summer games the corpus holds
the same match twice (prod 2026-10-31, read only: 1,953 matches on the 171
twin teams). Investigation: audit §7, 2026-10-31 "team identity".

RUN ORDER (2026-10-31; deploy the team-identity code first, then
scripts/ops/soccer_repair_runbook.sh runs each step through its own wrapper;
this tool is step 3, through scripts/ops/merge_soccer_teams.sh, and step 11):
  1. scripts/ops/repoint_team_matches.sh --pass 1
  2. scripts/ops/migrate_027_outcome_trigger.sh
  3. scripts/ops/merge_soccer_teams.sh
  4. scripts/ops/repair_football_data_kickoffs.sh
  5. scripts/ops/repoint_team_matches.sh --pass 2
  6. scripts/ops/repair_stranded_recs_soccer.sh
  7. scripts/ops/cleanup_soccer_orphans.sh
  8. scripts/ops/backfill_soccer_regulation_scores.sh (+ the re-grade it prints)
  9. scripts/ops/backfill_season_labels.sh
 10. compute_features.py --backfill-finished soccer
 11. merge_soccer_teams.py --check-after <plan> --odds-since <MERGED_AT> --after-later-steps
Step 1 moves Santos Laguna's and Chivas' Liga MX rows off the wrong rows and
stamps the seeded ids; promote_raw stays blocked until step 4 has run.

THE MAP
-------
scripts/data/soccer_team_merge_map.csv, reviewed by hand: one row per split
club (new_team_id -> old_team_id), with the evidence and the ESPN team id
the new spelling carries. The tool merges ONLY pairs listed there, and
--only can narrow a run to some of them (an id not in the map is refused).
The map must agree with the ESPN seed (scripts/data/soccer_team_espn_seed.csv):
a seeded ESPN id names the kept row, and no seeded row is merged away.

WHAT ONE PAIR DOES (one transaction per pair; a failing pair rolls back
alone and is reported, the others go ahead)
-----------------------------------------------------------------------
1. Twin fixtures first. Re-pointing the new team's matches onto the old
   team puts some of them next to a row for the same fixture: same home and
   away team within 36 h (TWIN_WINDOW). Every such cluster keeps ONE row:
     * the row holding recommendations (recs on the match, recs on one of
       its predictions, or betting_history) — two holders refuse the pair;
     * else a finished row: one with ESPN identity first (an ESPN
       competition id, or a result off the ESPN scoreboard — the row
       promote_raw writes onto), then one with an ESPN id, then the one with
       more predictions, then more odds rows. Finished rows that disagree on
       the score refuse the pair.
   A kept row that is not finished while a twin is (the rec holder is the
   stale scheduled row) ADOPTS the twin's result — status, score, half-time.
   Its kickoff: from an ESPN twin it adopts the twin's kickoff too (only
   when it holds no open rec; with open recs the kickoffs must already
   agree, else the pair is refused). A football-data twin's stored kickoff
   is the UK wall clock read as UTC (an hour late in summer), so it is
   never copied: the kept row must sit where promote_raw will look for that
   fixture (within 3 h of football-data's corrected kickoff, on its UK
   date) — exactly on it with open recs — else the pair is refused.
   The kept row also takes a dropped twin's half-time pair when it has none,
   and its ESPN competition id when it has none, but only from a row with
   the kept row's (final) kickoff or from a FINISHED row whose kickoff it
   then adopts (no open recs): an ESPN id beside a different kickoff would
   let fetch_upcoming realign the row, and on a football-data row it would
   stop repair_football_data_kickoffs from correcting it. A kept
   football-data row never adopts a kickoff outside the window promote_raw
   searches for its own fixture: the pair is refused (promote_raw would
   insert that fixture again beside it).
   Each dropped twin hands the kept row, before it is deleted:
     odds            the freshest pre-game price per current-price key
                     (book, market, selection, line, opening, live) the kept
                     row does not already hold; in-play rows never move;
     odds_snapshots  captures before kickoff the kept row lacks;
     predictions     per (model, version, type) the latest one written (and
                     last updated) before kickoff, when the kept row has no
                     prediction under that key (the unique key) and has
                     kicked off: the move is an UPDATE, so
                     trg_predictions_updated_at stamps updated_at = NOW()
                     (the archive keeps the original), and on a fixture still
                     to be played generate_recommendations, which takes the
                     newest ensemble row by updated_at, would price recs off
                     an older model version until the next precompute tick;
     match_stats     a team's stats row when the kept row has none for it.
   "Before kickoff" is before the EARLIEST kickoff in the cluster:
   football-data summer kickoffs sit an hour late, so the later time could
   let in-play prices through. Everything else on the dropped twin (later
   predictions, later prices, features_cache, weather) goes with it, archived
   first. A prediction a rec points at is never dropped: the plan refuses
   that pair, and migration 025 makes the DELETE fail if one appears anyway.
   A dropped football-data twin whose corrected kickoff promote_raw would
   not find the kept row near refuses its cluster (promote_raw would insert
   it again beside the kept row); one whose corrected kickoff merely differs
   is flagged in the audit CSV and counted.
   A cluster whose rows already share both team ids is an ESPN kickoff
   fork older than the merge (a stale 'scheduled' row next to the played
   one): the same rules fold it, but one they would refuse is left alone
   and reported instead of blocking the pair (repair_stranded_recs owns it).
2. Re-point matches.home_team_id / away_team_id from the new team to the
   old one, each UPDATE guarded on the row's observed teams and kickoff.
3. Assert nothing references the new team any more (every foreign key to
   teams, read from pg_constraint) and that no same-pair twin within 36 h
   touches a re-pointed row, except the forks left alone.
4. The ESPN team id moves with the merge: the twin's stamped
   external_ids.espn (sweeps stamp it once the identity code is live) is
   removed from the twin, then stamped onto the kept team through
   team_identity.stamp_espn_id (its guard: the kept row holds no id and no
   other row holds this one). The plan refuses a pair whose kept row already
   holds a different id, whose twin's stamp disagrees with the map, whose id
   a third row holds, or whose id the seed pins to another row.
5. Record the merge on the kept team: the new spelling joins
   metadata.aliases (a JSON array of strings, team_identity's contract; a
   malformed value refuses the pair) and metadata.merged_from gets
   {team_id, name, espn, merged_at}.
6. Delete the new team row (guarded: zero references, same name). matches
   -> teams is ON DELETE CASCADE, so the delete runs only after step 3.
7. Before COMMIT, the deployed resolvers must resolve the merged spelling to
   the kept team in this transaction, or the pair rolls back:
   fetch_upcoming.find_team(new name, soccer, the new team's league) without
   and with the ESPN id, and fetch_live_odds.find_match_id for one
   re-pointed fixture under the merged spelling. Deleting "manchester city"
   frees that name: if the resolvers missed it, the next 15-minute fetch
   would insert the team again (the Montreal lesson, audit §7 2026-10-26).
   The odds feed spells clubs its own way ('Brighton and Hove Albion',
   'Paris Saint Germain'), so every spelling of the club in
   scripts/data/soccer_team_odds_spellings.csv (reviewed, from prod's
   fetch_live_odds logs) is probed on the same fixture before the pair's
   first write, while it still names the twin, and each one that found it
   then must find it again before COMMIT.
   The dry run predicts the same outcome from the database (no other row
   holds the name exactly, no other row claims the alias, no third row holds
   the ESPN id, the seed agrees) and refuses the pair when it would fail.

SAFETY
------
Every foreign key to teams and matches is read from pg_constraint first; a
table the tool does not know refuses the whole run (it could cascade
unarchived), and a dropped twin holding a row the archive does not hold
(written after it) rolls its pair back instead of cascading it away. The
whole run is planned up front (pairs in map order, each
planned on the state the earlier ones leave), and the write-once audit CSV
of every row the run touches and a write-once plan file land BEFORE the
first write, dry run included. --apply also writes a write-once gzip JSONL
archive of every row it will delete or change (dropped twins with all their
dependents, kept rows it enriches, both team rows) before the first write.
--expect-plan <dry-run plan file> makes --apply refuse (nothing written)
unless its own plan is the one the dry run showed. Each pair locks its two
team rows, and its plan is re-checked against what it locks: a pair whose
new team gained or lost a match since the plan, or any guarded statement
that hits a row that moved, rolls that pair back and is reported.
Re-runnable: a merged pair plans as 'done'.

--check-after <plan file> (read-only) checks the after-state the plan
implies: no team row or match left on a merged id, every pair recorded, no
dropped twin left, every re-point landed, no finished same-pair twin on the
kept teams, every merged spelling resolving to its kept team (the deployed
resolvers again), each ESPN id on its kept team only, no team re-minted
under a merged spelling, no unsettled rec on a row that adopted a result
with open recs. Re-run it after the next fetch_live_odds run with
--odds-since <when the merge committed>: a resolver miss shows up as a
re-minted team, an odds miss as a merged club's fixture the feed was quoting
in the hours before the merge and that run did not quote (odds_fixtures_stopped;
odds_control_stopped counts the same for every other club, the base rate).
The runbook runs that odds check last (step 11), after
repair_football_data_kickoffs and cleanup_soccer_orphans have deleted some
re-pointed rows on purpose: --after-later-steps reports repoints_missing
instead of failing on it; every other count must still be 0.

Usage (inside the api container; dry run is the default):
    python /app/scripts/merge_soccer_teams.py --plan-out /app/backups/merge.plan.json
    python /app/scripts/merge_soccer_teams.py --apply --expect-plan /app/backups/merge.plan.json
    python /app/scripts/merge_soccer_teams.py --check-after /app/backups/merge.plan.json
    python /app/scripts/merge_soccer_teams.py --check-after /app/backups/merge.plan.json --odds-since <UTC ISO>
    python /app/scripts/merge_soccer_teams.py --check-after <plan> --odds-since <UTC ISO> --after-later-steps
    python /app/scripts/merge_soccer_teams.py --only <new_team_id>[,<new_team_id>...] --apply
Afterwards: grade_completed_matches.py --match-id <id> for every kept row
the plan lists under GRADE_AFTER (moved predictions on finished rows, and
results adopted with open recs), then compute_features.py
--backfill-finished soccer (every re-pointed finished match carries form
computed on the thin twin's history).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gzip
import hashlib
import importlib
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("merge_soccer_teams")

SCRIPTS_DIR = Path(__file__).resolve().parent
MAP_PATH = SCRIPTS_DIR / "data" / "soccer_team_merge_map.csv"
MAP_FIELDS = ("new_team_id", "old_team_id", "new_name", "old_name", "espn_team_id", "espn_league", "method", "evidence")
# How the odds feed spells merged clubs (prod's fetch_live_odds logs, reviewed):
# the in-transaction odds check probes each of them, not only ESPN's spelling.
ODDS_SPELLINGS_PATH = SCRIPTS_DIR / "data" / "soccer_team_odds_spellings.csv"
ODDS_SPELLING_FIELDS = ("new_team_id", "new_name", "odds_name", "evidence")
SPORT = "soccer"
TWIN_WINDOW = timedelta(hours=36)
# promote_raw.EXISTING_MATCH_WINDOW / existing_match_window (the football-data
# branch): a football-data fixture is written onto a row for the same team
# ids within 3 h of its corrected kickoff, on football-data's UK date.
PROMOTE_WINDOW = timedelta(hours=3)
FOOTBALL_DATA_TZ = ZoneInfo("Europe/London")
ALIASES_KEY = "aliases"
MERGED_FROM_KEY = "merged_from"
LOCK_TIMEOUT = "10s"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
FINISHED = "finished"
RESOLVABLE_STATUSES = ("scheduled", "finished")  # what fetch_live_odds.find_match_id looks at

# Every foreign key the tool knows, by (referenced table, table, column). A
# key pg_constraint reports that is not here refuses the run: it could
# cascade rows nobody archived. Only keys the database has are checked.
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
# Rows hanging off a match that a dropped twin takes with it (archived first).
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
# Rows that make a match a rec holder: such a twin is kept, never deleted.
MATCH_HOLDERS = ("betting_recommendations", "betting_history")


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
    predictions: list[dict] = field(default_factory=list)


@dataclass
class Drop:
    match_id: str
    keep_id: str
    observed: dict  # the row as planned: teams, kickoff, status, scores
    odds_ids: list[str]
    snapshot_ids: list[str]
    match_stats_ids: list[str]
    cutoff: datetime  # earliest kickoff in the cluster: only rows from before it move
    fork: bool = False  # folds a pre-merge ESPN fork rather than a merge-made twin
    prediction_ids: list[str] = field(default_factory=list)
    flags: dict = field(default_factory=dict)  # football-data kickoff facts (see football_data_flags)


@dataclass
class Enrich:
    match_id: str
    observed: dict
    sets: dict  # status / home_score / away_score / half_time_home / half_time_away / match_date
    external_patch: dict
    metadata_patch: dict
    reason: str
    open_recs: int = 0
    takes_predictions: bool = False


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
    team_patch: dict = field(default_factory=dict)  # aliases / merged_from / ESPN id move
    odds_names: list[str] = field(default_factory=list)  # the odds feed's spellings of the club (not in the digest)


@dataclass
class Ctx:
    """The name-resolution facts the dry run predicts the resolvers from:
    every soccer team's exact name and ESPN id, the alias index, the seed.
    plan_all advances a private copy pair by pair."""

    alias_form: Callable[[str], str]
    names: dict[str, set[str]] = field(default_factory=dict)  # normalized_name -> team ids
    espn: dict[str, set[str]] = field(default_factory=dict)  # ESPN team id -> team ids
    aliases: dict[str, set[str]] = field(default_factory=dict)  # alias_form(alias) -> team ids
    seed: dict[str, Any] = field(default_factory=dict)  # ESPN team id -> team_identity.SeedEntry

    def copy(self) -> "Ctx":
        def dup(d: dict[str, set[str]]) -> dict[str, set[str]]:
            return {k: set(v) for k, v in d.items()}

        return Ctx(self.alias_form, dup(self.names), dup(self.espn), dup(self.aliases), dict(self.seed))


def normalize_name(name: str) -> str:
    """teams.normalized_name as fetch_upcoming.normalize_team builds it."""
    return " ".join((name or "").strip().lower().split())


# ---------------------------------------------------------------------------
# The deployed resolvers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolvers:
    """The resolution code the pipeline runs (scripts/ is bind-mounted, so
    this is what the next sweep uses): team_identity's alias contract, the
    guarded ESPN stamp and the seed, fetch_upcoming.find_team for soccer, and
    fetch_live_odds.find_match_id."""

    alias_form: Callable[[str], str]
    load_alias_index: Callable[[Any, str], dict]
    espn_seed: Callable[[], dict]
    stamp_espn_id: Callable[..., bool]
    find_team: Callable[[Any, str, Optional[str], Optional[str]], Optional[str]]  # cur, name, league_id, espn_id
    find_match_id: Callable[[Any, str, str, datetime], Optional[str]]  # cur, home, away, kickoff


RESOLVER_API = {
    "team_identity": ("alias_form", "load_alias_index", "resolve_alias", "espn_seed", "stamp_espn_id"),
    "fetch_upcoming": ("find_team", "SPORT_CONFIGS"),
    "fetch_live_odds": ("find_match_id",),
}


def load_resolvers(scripts_dir: Path = SCRIPTS_DIR) -> tuple[Optional[Resolvers], Optional[str]]:
    """(Resolvers, None), or (None, why not): an import error or a missing
    function is a refusal, never a crash."""
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    mods = {}
    for name, attrs in RESOLVER_API.items():
        try:
            mods[name] = importlib.import_module(name)
        except Exception as e:  # the refusal names it
            return None, f"cannot import scripts/{name}.py: {type(e).__name__}: {e}"
        missing = [a for a in attrs if not hasattr(mods[name], a)]
        if missing:
            return None, f"scripts/{name}.py lacks {', '.join(missing)} (the alias/ESPN resolver is not deployed)"
    cfg = mods["fetch_upcoming"].SPORT_CONFIGS.get(SPORT)
    if cfg is None:
        return None, "fetch_upcoming.SPORT_CONFIGS has no soccer config"
    fu, flo, ti = mods["fetch_upcoming"], mods["fetch_live_odds"], mods["team_identity"]
    return (
        Resolvers(
            alias_form=ti.alias_form,
            load_alias_index=ti.load_alias_index,
            espn_seed=ti.espn_seed,
            stamp_espn_id=ti.stamp_espn_id,
            find_team=lambda cur, name, league_id, espn_id: fu.find_team(cur, cfg, name, league_id, espn_id),
            find_match_id=lambda cur, home, away, when: flo.find_match_id(cur, SPORT, home, away, when),
        ),
        None,
    )


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
    dup_espn = [k for k, n in Counter(r.espn_team_id for r in rows if r.espn_team_id).items() if n > 1]
    if dup_espn:
        raise MapError(f"{path}: one ESPN team id on two pairs: {dup_espn}")
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


def load_odds_spellings(path: Path, rows: list[MapRow]) -> dict[str, list[str]]:
    """{new_team_id: [the odds feed's spellings of that club]} from the
    reviewed list ('#' lines are comments). Every row must name a pair of the
    map by id and by its new_name, carry evidence, and appear once. Raises
    MapError otherwise: an odds spelling nobody can place is not a probe."""
    with path.open(newline="") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    if tuple(reader.fieldnames or ()) != ODDS_SPELLING_FIELDS:
        raise MapError(f"{path}: header {reader.fieldnames} != {list(ODDS_SPELLING_FIELDS)}")
    by_id = {r.new_team_id: r for r in rows}
    out: dict[str, list[str]] = {}
    for r in reader:
        nid = (r["new_team_id"] or "").strip()
        pair = by_id.get(nid)
        if pair is None:
            raise MapError(f"{path}: {nid!r} ({r['new_name']}) is not a pair of the merge map")
        if r["new_name"] != pair.new_name:
            raise MapError(f"{path}: {nid} is {pair.new_name!r} in the merge map, not {r['new_name']!r}")
        name = r["odds_name"] or ""
        if not name.strip() or not (r["evidence"] or "").strip():
            raise MapError(f"{path}: {pair.new_name} needs an odds_name and its evidence")
        if name in out.get(nid, []):
            raise MapError(f"{path}: {name!r} listed twice for {pair.new_name}")
        out.setdefault(nid, []).append(name)
    return out


def seed_conflict(new_name: str, old_name: str, espn: str, seed: dict[str, Any]) -> Optional[str]:
    """Why merging `new_name` into `old_name` with ESPN team id `espn` ('' if
    none) contradicts the ESPN seed (team_identity.load_espn_seed: espn_id ->
    entry with .espn_id/.team_name/.normalized_name), else None. Pure.

    fetch_upcoming refuses a side whose stamped ESPN id names a row other
    than the one the seed pins, so the seed and the merge must agree: a
    seeded id goes onto the row the seed names, a seeded row is never merged
    away, and a seeded row never takes another club's id."""
    by_name = {e.normalized_name: e for e in seed.values()}
    entry = seed.get(espn) if espn else None
    if entry is not None and entry.normalized_name != normalize_name(old_name):
        return f"the ESPN seed pins ESPN team {espn} to {entry.team_name!r}, not the kept team {old_name!r}"
    gone = by_name.get(normalize_name(new_name))
    if gone is not None:
        return f"{new_name!r} is the seeded row of ESPN team {gone.espn_id}: a seeded row is never merged away"
    kept = by_name.get(normalize_name(old_name))
    if kept is not None and espn and kept.espn_id != espn:
        return f"the seed pins {old_name!r} to ESPN team {kept.espn_id}, this pair brings {espn}"
    return None


def map_seed_conflicts(pairs: list[MapRow], seed: dict[str, Any]) -> dict[str, str]:
    """{new_team_id: why} for every map row the seed contradicts. Pure."""
    out = {}
    for p in pairs:
        why = seed_conflict(p.new_name, p.old_name, p.espn_team_id, seed)
        if why:
            out[p.new_team_id] = why
    return out


# ---------------------------------------------------------------------------
# Foreign keys: refuse a table the tool does not know
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
    """The foreign keys the database has, as far as the tool knows them."""

    team_refs: tuple[tuple[str, str], ...]  # (table, column) -> teams.id
    match_children: tuple[str, ...]  # tables whose match_id -> matches.id cascades with a dropped twin


def read_schema(rows: Iterable[dict]) -> tuple[Optional[Schema], list[str]]:
    """(Schema, []) from FOREIGN_KEYS_SQL rows, or (None, the unknown keys).
    Pure."""
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


def _ref_counts(schema: Schema, team: str, with_matches: bool) -> list[str]:
    """count(*) subqueries, one per foreign key to teams. Table and column
    names come from read_schema, i.e. from the TEAM_REFERENCES whitelist."""
    out = []
    by_table: dict[str, list[str]] = defaultdict(list)
    for tbl, col in schema.team_refs:
        by_table[tbl].append(col)
    for tbl, cols in by_table.items():
        if tbl == "matches" and not with_matches:
            continue
        cond = " OR ".join(f"x.{c} = {team}" for c in cols)
        out.append(f'(SELECT count(*) FROM {tbl} x WHERE {cond}) AS "{tbl}"')
    return out


def team_refs_sql(schema: Schema) -> str:
    """Plan time: references to each new team outside matches."""
    cols = ",\n           ".join(_ref_counts(schema, "t.id", with_matches=False)) or "0 AS none"
    return f"SELECT t.id::text AS id,\n           {cols}\n      FROM teams t WHERE t.id = ANY(%s::uuid[])"


def new_team_refs_sql(schema: Schema) -> str:
    """Apply time: every reference left to the new team (matches included)."""
    cols = ",\n           ".join(_ref_counts(schema, "%(t)s::uuid", with_matches=True))
    return f"SELECT {cols}"


def team_delete_sql(schema: Schema) -> str:
    """The guarded team delete: same name, and no row references it under any
    foreign key (matches -> teams cascades, so this runs after the re-point
    and the reference check)."""
    by_table: dict[str, list[str]] = defaultdict(list)
    for tbl, col in schema.team_refs:
        by_table[tbl].append(col)
    guards = "\n".join(
        f"       AND NOT EXISTS (SELECT 1 FROM {tbl} x WHERE {' OR '.join(f'x.{c} = t.id' for c in cols)})"
        for tbl, cols in by_table.items()
    )
    return (
        "DELETE FROM teams t\n"
        "     WHERE t.id = %(id)s::uuid AND t.sport = 'soccer' AND t.name = %(name)s\n"
        "       AND t.normalized_name = %(normalized_name)s\n" + guards
    )


# ---------------------------------------------------------------------------
# football-data kickoffs
# ---------------------------------------------------------------------------


def has_espn_identity(m: dict) -> bool:
    """A row the ESPN path owns (promote_raw.has_espn_identity): it carries an
    ESPN competition id, or its result came off the ESPN scoreboard."""
    return bool(_espn(m)) or (m.get("metadata") or {}).get("result_source") == "espn_scoreboard"


def is_football_data_row(m: dict) -> bool:
    """A row promote_raw wrote: finished, without ESPN identity (football-data's
    own rows are always finished; an unfinished row without identity is a
    stale pre-migration-023 ESPN fixture)."""
    return _finished(m) and not has_espn_identity(m)


def _wall_clock(legacy: datetime) -> datetime:
    """The naive UK wall clock promote_raw stored as UTC before 2026-10-31."""
    return legacy.astimezone(timezone.utc).replace(tzinfo=None)


def corrected_kickoff(legacy: datetime) -> datetime:
    """The UTC kickoff promote_raw now writes for a football-data row it stored
    at `legacy` (football-data's UK local time read as UTC, an hour late in
    summer). A row stored at midnight is read as a timed 00:00 kickoff: the
    2026 country files all carry a Time, and Bragantino/Gremio/Club America
    rows at 00:00 sit beside ESPN rows at 23:00Z the day before. Pure."""
    return _wall_clock(legacy).replace(tzinfo=FOOTBALL_DATA_TZ).astimezone(timezone.utc)


def promote_window(legacy: datetime) -> tuple[datetime, datetime]:
    """The kickoffs promote_raw (football-data branch, existing_match_window)
    searches for an existing row of a timed football-data fixture stored at
    `legacy`: within PROMOTE_WINDOW of its corrected kickoff and on its UK
    date, both bounds inclusive. Were a midnight row date-only, promote_raw
    would search its whole UK date, which holds this window: checking this one
    stays conservative. Pure."""
    wall = _wall_clock(legacy)
    day = wall.date()
    day_start = datetime.combine(day, time(), tzinfo=FOOTBALL_DATA_TZ).astimezone(timezone.utc)
    next_day = datetime.combine(day + timedelta(days=1), time(), tzinfo=FOOTBALL_DATA_TZ).astimezone(timezone.utc)
    day_end = next_day - timedelta(microseconds=1)
    kickoff = corrected_kickoff(legacy)
    return max(kickoff - PROMOTE_WINDOW, day_start), min(kickoff + PROMOTE_WINDOW, day_end)


def promote_finds(kickoff: datetime, legacy: datetime) -> bool:
    """Whether promote_raw, re-promoting the football-data row stored at
    `legacy`, would find a row of the same team ids at `kickoff` (and write
    onto it) rather than insert the fixture again. Pure."""
    lo, hi = promote_window(legacy)
    return lo <= kickoff <= hi


def football_data_flags(drop: dict, keep_kickoff: datetime) -> dict:
    """Audit facts about a dropped football-data twin: its corrected kickoff,
    whether that differs from the kept row's (final) kickoff, and whether
    promote_raw would find the kept row for it. {} for any other row. Pure."""
    if not is_football_data_row(drop):
        return {}
    corrected = corrected_kickoff(drop["match_date"])
    return {
        "fd_corrected_kickoff": corrected.isoformat(),
        "kept_kickoff": keep_kickoff.isoformat(),
        "kickoff_differs": corrected != keep_kickoff,
        "promote_finds_kept": promote_finds(keep_kickoff, drop["match_date"]),
    }


# ---------------------------------------------------------------------------
# Planning (pure: rows in, plan out)
# ---------------------------------------------------------------------------


def _espn(m: dict) -> str:
    return str((m.get("external_ids") or {}).get("espn") or "")


def _team_espn(t: Optional[dict]) -> str:
    return str(((t or {}).get("external_ids") or {}).get("espn") or "")


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
    return (
        _finished(m),
        has_espn_identity(m),
        bool(_espn(m)),
        d.n_preds,
        len(d.odds) + len(d.snapshots),
        m["id"],
    )


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
    'espn_from', 'kickoff', 'notes'}: 'kickoff' is the kept row's new
    kickoff, or None when it stays. Every row promote_raw re-promotes must
    stay where it searches: a dropped football-data twin's kept row, and a
    kept football-data row itself. Pure."""
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
    open_recs = deps.get(keep["id"], Deps()).open_recs
    kickoff: Optional[datetime] = None
    notes: list[str] = []
    adopt_from = None
    if not _finished(keep) and finished:
        adopt_from = max(finished, key=lambda m: _rank(m, deps))
        if has_espn_identity(adopt_from):
            if keep["match_date"] != adopt_from["match_date"]:
                if open_recs:
                    return {
                        "refuse": f"the rec holder {keep['id']} is {keep.get('status')} with open recs and its "
                        f"kickoff {keep['match_date']} differs from the finished twin's {adopt_from['match_date']}: "
                        f"settle it with repair_stranded_recs.py after the merge, not here"
                    }
                kickoff = adopt_from["match_date"]
        else:
            corrected = corrected_kickoff(adopt_from["match_date"])
            if open_recs and keep["match_date"] != corrected:
                return {
                    "refuse": f"the rec holder {keep['id']} is {keep.get('status')} with open recs and its kickoff "
                    f"{keep['match_date']} is not football-data's corrected kickoff {corrected} (stored "
                    f"{adopt_from['match_date']}): settle it with repair_stranded_recs.py after the merge, not here"
                }
            if not promote_finds(keep["match_date"], adopt_from["match_date"]):
                return {
                    "refuse": f"the rec holder {keep['id']} at {keep['match_date']} is not near football-data's "
                    f"corrected kickoff {corrected} (stored {adopt_from['match_date']}): a postponed fixture? Its "
                    f"kickoff cannot be taken from football-data; settle it with repair_stranded_recs.py"
                }
    ht_from = None
    if (keep.get("half_time_home") is None or keep.get("half_time_away") is None) and finished:
        for m in sorted(finished, key=lambda m: _rank(m, deps), reverse=True):
            if m["id"] != keep["id"] and m.get("half_time_home") is not None and m.get("half_time_away") is not None:
                ht_from = m
                break
    espn_from = None
    if not _espn(keep):
        cand = next((m for m in ranked if _espn(m)), None)
        if cand is not None:
            final = kickoff or keep["match_date"]
            if cand["match_date"] == final:
                espn_from = cand
            elif kickoff is None and not open_recs and _finished(cand):
                if is_football_data_row(keep) and not promote_finds(cand["match_date"], keep["match_date"]):
                    return {
                        "refuse": f"the kept football-data row {keep['id']} at {keep['match_date']} would take the "
                        f"ESPN kickoff {cand['match_date']} of {cand['id']}, outside the window promote_raw searches "
                        f"for that fixture (corrected {corrected_kickoff(keep['match_date'])}): promote_raw would "
                        f"insert it again beside the kept row"
                    }
                espn_from, kickoff = cand, cand["match_date"]
            else:
                notes.append(
                    f"ESPN id {_espn(cand)} of {cand['id']} not copied: its kickoff {cand['match_date']} is not the "
                    f"kept row's {final}"
                    + (" and the kept row holds open recs" if open_recs else "")
                    + ("" if _finished(cand) else " and the ESPN row is not finished")
                )
    final_kickoff = kickoff or keep["match_date"]
    drops = [m for m in cluster if m["id"] != keep["id"]]
    for d in drops:
        if is_football_data_row(d) and not promote_finds(final_kickoff, d["match_date"]):
            return {
                "refuse": f"promote_raw would insert football-data's {d['id']} (stored {d['match_date']}, corrected "
                f"{corrected_kickoff(d['match_date'])}) again: the kept row {keep['id']} at {final_kickoff} is "
                f"outside its search window"
            }
    return {
        "keep": keep,
        "drops": drops,
        "adopt_from": adopt_from,
        "ht_from": ht_from,
        "espn_from": espn_from,
        "kickoff": kickoff,
        "notes": notes,
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


def _pred_key(p: dict) -> tuple:
    return (p["model_name"], p.get("model_version"), p["prediction_type"])


def _written(p: dict) -> datetime:
    """When a prediction's content was last written (an upsert after kickoff
    rewrites it): COALESCE(updated_at, created_at), as repair_fixture_orphans."""
    return p.get("updated_at") or p["created_at"]


def plan_moves(
    drop: Deps, keep: Deps, cutoff: datetime, keep_teams: set[str], move_predictions: bool = True
) -> tuple[list, list, list, list]:
    """Rows of a dropped twin that move to the kept row: (odds, snapshots,
    match_stats, predictions). Mutates `keep` so a later twin of the same
    cluster sees them as already there. Pure apart from that.

    `move_predictions` is False for a kept row whose kickoff is still ahead:
    moving a prediction is an UPDATE, trg_predictions_updated_at stamps it
    updated_at = NOW(), and generate_recommendations takes the newest
    ensemble row per type by updated_at, so an older model version would
    price recs until the next precompute tick. Such a row's predictions are
    regenerated anyway; the twin's are archived and dropped."""
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
    taken = {_pred_key(p) for p in keep.predictions}
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for p in drop.predictions if move_predictions else ():
        if _written(p) < cutoff:
            by_key[_pred_key(p)].append(p)
    # per key the latest write; on a tie the smallest id (repair_fixture_orphans' DISTINCT ON order)
    latest = {k: min(ps, key=lambda p: (-_written(p).timestamp(), p["id"])) for k, ps in by_key.items()}
    preds = sorted((p for k, p in latest.items() if k not in taken), key=lambda p: p["id"])
    keep.odds = keep.odds + odds
    keep.snapshots = keep.snapshots + snaps
    keep.match_stats = keep.match_stats + stats
    keep.predictions = keep.predictions + preds
    return odds, snaps, stats, preds


def aliases_of(team: Optional[dict]) -> Optional[list[str]]:
    """teams.metadata.aliases as a list of strings; None when the value is not
    an array of strings (team_identity then ignores it whole)."""
    value = ((team or {}).get("metadata") or {}).get(ALIASES_KEY)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(a, str) for a in value):
        return None
    return list(value)


def final_espn(pair: MapRow, new: Optional[dict]) -> str:
    """The ESPN team id the kept team holds after the merge: the twin's stamp,
    else the map's ('' when neither)."""
    return _team_espn(new) or pair.espn_team_id


def assess_teams(pair: MapRow, new: Optional[dict], old: Optional[dict], refs: dict, ctx: Ctx) -> tuple[str, str]:
    """('merge' | 'done' | 'refuse', reason) from the two team rows. Pure."""
    alias = ctx.alias_form(pair.new_name)
    if old is None:
        return "refuse", "kept team row not found"
    if old.get("sport") != SPORT or old.get("name") != pair.old_name:
        return "refuse", f"kept team is {old.get('sport')!r} {old.get('name')!r}, the map says soccer {pair.old_name!r}"
    old_aliases = aliases_of(old)
    if old_aliases is None:
        return "refuse", (
            f"kept team's metadata.aliases is {(old.get('metadata') or {}).get(ALIASES_KEY)!r}, not an array of "
            f"names: team_identity would ignore it, so no alias could resolve — repair it first"
        )
    old_espn = _team_espn(old)
    meta = old.get("metadata") or {}
    if new is None:
        merged = {str(e.get("team_id")) for e in meta.get(MERGED_FROM_KEY) or [] if isinstance(e, dict)}
        if pair.new_team_id in merged and alias in {ctx.alias_form(a) for a in old_aliases}:
            return "done", "already merged"
        return "refuse", "new team row is gone but the kept team does not record the merge"
    if new.get("sport") != SPORT or new.get("name") != pair.new_name:
        return "refuse", f"new team is {new.get('sport')!r} {new.get('name')!r}, the map says soccer {pair.new_name!r}"
    new_espn = _team_espn(new)
    if pair.espn_team_id and new_espn and new_espn != pair.espn_team_id:
        return "refuse", f"new team carries ESPN team {new_espn}, the map says {pair.espn_team_id}"
    espn = final_espn(pair, new)
    if espn and old_espn and old_espn != espn:
        return "refuse", f"kept team already carries ESPN team {old_espn}, this pair brings {espn}"
    busy = {k: v for k, v in (refs or {}).items() if v}
    if busy:
        return "refuse", f"the new team is referenced outside matches: {busy}"
    return "merge", ""


def resolution_refusal(pair: MapRow, new: dict, old: dict, ctx: Ctx) -> Optional[str]:
    """Why the deployed resolvers would NOT resolve the merged spelling to the
    kept team once this pair is merged, predicted from the database, else
    None. Pure. fetch_upcoming.find_team tries the ESPN id, then the seed,
    then the exact name, then aliases (team_identity.resolve_alias:
    alias_form, an alias two teams claim resolves to none); fetch_live_odds
    and promote_raw use the exact name, then aliases."""
    norm = normalize_name(pair.new_name)
    exact = ctx.names.get(norm, set()) - {pair.new_team_id}
    if exact:
        return (
            f"team(s) {sorted(exact)} already hold the name {norm!r}: after the merge the resolvers would pick "
            f"that row by exact name, not {pair.old_name!r}"
        )
    claim = ctx.aliases.get(ctx.alias_form(pair.new_name), set()) - {pair.old_team_id}
    if claim:
        return (
            f"team(s) {sorted(claim)} already claim the alias {ctx.alias_form(pair.new_name)!r}: a second claim "
            f"resolves to no team at all"
        )
    espn = final_espn(pair, new) or _team_espn(old)  # the id find_team is checked with in the transaction
    if espn:
        third = ctx.espn.get(espn, set()) - {pair.new_team_id, pair.old_team_id}
        if third:
            return f"ESPN team {espn} is also stamped on team(s) {sorted(third)}: one ESPN team must be one row"
    why = seed_conflict(pair.new_name, pair.old_name, espn, ctx.seed)
    if why:
        return why
    return None


def plan_pair(
    index: int,
    pair: MapRow,
    teams: dict[str, dict],
    refs: dict[str, dict],
    state: dict[str, dict],
    deps: dict[str, Deps],
    now: datetime,
    ctx: Ctx,
) -> PairPlan:
    """Plan one pair against the current `state` (earlier pairs applied).
    Pure, except that a merging pair records its planned moves in `deps`."""
    new, old = teams.get(pair.new_team_id), teams.get(pair.old_team_id)
    action, reason = assess_teams(pair, new, old, refs.get(pair.new_team_id, {}), ctx)
    plan = PairPlan(index=index, pair=pair, action=action, reason=reason)
    if action != "merge":
        return plan
    assert new is not None and old is not None
    why = resolution_refusal(pair, new, old, ctx)
    if why:
        return _refuse(plan, why)
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
        final_kickoff = verdict["kickoff"] or keep["match_date"]
        keep_teams = {oid if t == nid else t for t in (keep["home_team_id"], keep["away_team_id"])}
        keep_deps = scratch.setdefault(keep["id"], dataclasses.replace(deps.get(keep["id"], Deps())))
        took_predictions = False
        started = final_kickoff <= now  # predictions never move onto a fixture still to be played (plan_moves)
        for d in verdict["drops"]:
            odds, snaps, stats, preds = plan_moves(deps.get(d["id"], Deps()), keep_deps, cutoff, keep_teams, started)
            took_predictions = took_predictions or bool(preds)
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
                    prediction_ids=[p["id"] for p in preds],
                    flags=football_data_flags(d, final_kickoff),
                )
            )
            dropped.add(d["id"])
        enrich = _enrichment(keep, verdict, keep_deps, took_predictions)
        if enrich:
            plan.enrich.append(enrich)
    moved_kickoff = {e.match_id: e.sets["match_date"] for e in plan.enrich if "match_date" in e.sets}
    for m in mine:
        if m["id"] in dropped:
            continue
        side = "home" if m["home_team_id"] == nid else "away"
        observed = _observed(m)
        if m["id"] in moved_kickoff:  # the enrich runs first in the transaction
            observed["match_date"] = moved_kickoff[m["id"]]
        plan.repoints.append({"match_id": m["id"], "side": side, "observed": observed})
    plan.team_patch = _team_patch(pair, new, old, now, ctx)
    deps.update(scratch)
    return plan


def _refuse(plan: PairPlan, reason: str) -> PairPlan:
    plan.action, plan.reason = "refuse", reason
    plan.repoints, plan.drops, plan.enrich, plan.forks = [], [], [], []
    return plan


def _enrichment(keep: dict, verdict: dict, keep_deps: Deps, took_predictions: bool) -> Optional[Enrich]:
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
    if verdict.get("kickoff") is not None:
        sets["match_date"] = verdict["kickoff"]
        metadata["kickoff_adopted_from"] = (verdict["espn_from"] or src or {}).get("id")
        why.append(f"kickoff {keep['match_date']} -> {verdict['kickoff']}")
    ht = verdict["ht_from"]
    if ht is not None:
        sets.update(half_time_home=ht["half_time_home"], half_time_away=ht["half_time_away"])
        why.append(f"half-time {ht['half_time_home']}-{ht['half_time_away']} from {ht['id']}")
    espn = verdict["espn_from"]
    if espn is not None:
        external["espn"] = _espn(espn)
        why.append(f"ESPN id {external['espn']} from {espn['id']}")
    why.extend(verdict.get("notes") or [])
    if not (sets or external or took_predictions):
        return None
    if took_predictions:
        why.append("takes predictions from its twin(s)")
    return Enrich(
        match_id=keep["id"],
        observed=_observed(keep),
        sets=sets,
        external_patch=external,
        metadata_patch=metadata,
        reason="; ".join(why),
        open_recs=keep_deps.open_recs if src is not None else 0,
        takes_predictions=took_predictions,
    )


def _team_patch(pair: MapRow, new: dict, old: dict, now: datetime, ctx: Ctx) -> dict:
    meta = old.get("metadata") or {}
    observed = aliases_of(old) or []
    aliases = list(observed)
    if ctx.alias_form(pair.new_name) not in {ctx.alias_form(a) for a in aliases}:
        aliases.append(normalize_name(pair.new_name))
    brought = final_espn(pair, new)
    merged_from = list(meta.get(MERGED_FROM_KEY) or [])
    merged_from.append(
        {"team_id": pair.new_team_id, "name": pair.new_name, "espn": brought or None, "merged_at": now.isoformat()}
    )
    old_espn = _team_espn(old)
    return {
        "observed_aliases": observed,
        "observed_merged_from": list(meta.get(MERGED_FROM_KEY) or []),
        "observed_espn": old_espn,
        "new_team_espn": _team_espn(new),  # removed from the twin before the kept team is stamped
        "aliases": sorted(aliases),
        "merged_from": merged_from,
        "espn": brought or old_espn,  # what the kept team holds afterwards ('' = none)
        "stamp": bool(brought) and brought != old_espn,
    }


def apply_to_state(plan: PairPlan, teams: dict[str, dict], state: dict[str, dict], ctx: Ctx) -> None:
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
    t = plan.team_patch
    old = teams[plan.pair.old_team_id]
    old["metadata"] = {**(old.get("metadata") or {}), ALIASES_KEY: t["aliases"], MERGED_FROM_KEY: t["merged_from"]}
    if t["espn"]:
        old["external_ids"] = {**(old.get("external_ids") or {}), "espn": t["espn"]}
    new = teams.pop(plan.pair.new_team_id, None) or {}
    ctx.names.get(normalize_name(new.get("normalized_name") or plan.pair.new_name), set()).discard(
        plan.pair.new_team_id
    )
    if t["new_team_espn"]:
        ctx.espn.get(t["new_team_espn"], set()).discard(plan.pair.new_team_id)
    if t["espn"]:
        ctx.espn.setdefault(t["espn"], set()).add(plan.pair.old_team_id)
    for a in t["aliases"]:
        ctx.aliases.setdefault(ctx.alias_form(a), set()).add(plan.pair.old_team_id)


def build_ctx(team_rows: Iterable[dict], alias_index: dict[str, set[str]], seed: dict, alias_form: Callable) -> Ctx:
    """The Ctx from every soccer team row (id, normalized_name, espn or
    external_ids), team_identity's alias index and the seed."""
    ctx = Ctx(alias_form=alias_form, seed=dict(seed))
    for t in team_rows:
        tid = str(t["id"])
        ctx.names.setdefault(t["normalized_name"], set()).add(tid)
        espn = str(t.get("espn") or _team_espn(t))
        if espn:
            ctx.espn.setdefault(espn, set()).add(tid)
    ctx.aliases = {k: set(v) for k, v in alias_index.items()}
    return ctx


def plan_all(
    pairs: list[MapRow],
    teams: dict[str, dict],
    refs: dict[str, dict],
    matches: dict[str, dict],
    deps: dict[str, Deps],
    now: datetime,
    ctx: Ctx,
    odds_names: Optional[dict[str, list[str]]] = None,
) -> list[PairPlan]:
    """Plan every pair in map order on a private copy of the state.
    `odds_names` (load_odds_spellings) rides along on each merging plan for
    the in-transaction odds check."""
    teams = {k: dict(v) for k, v in teams.items()}
    state = {k: dict(v) for k, v in matches.items()}
    deps = {k: dataclasses.replace(v) for k, v in deps.items()}  # plan_moves re-binds the lists, never edits them
    ctx = ctx.copy()
    plans = []
    for i, pair in enumerate(pairs, 1):
        plan = plan_pair(i, pair, teams, refs, state, deps, now, ctx)
        if plan.action == "merge":
            plan.odds_names = list((odds_names or {}).get(pair.new_team_id, ()))
        apply_to_state(plan, teams, state, ctx)
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


def grade_after(plans: list[PairPlan]) -> list[str]:
    """Kept rows grade_completed_matches.py --match-id must grade after the
    run: a result adopted with open recs (the trigger settles them from the
    prediction, without P&L), or predictions moved onto a row that is
    finished (they arrive ungraded; no trigger fires on a finished row)."""
    out = []
    for p in plans:
        if p.action != "merge":
            continue
        for e in p.enrich:
            finished_after = e.sets.get("status", e.observed["status"]) == FINISHED
            if ("status" in e.sets and e.open_recs) or (e.takes_predictions and finished_after):
                out.append(e.match_id)
    return sorted(set(out))


def adopted_with_open_recs(plans: list[PairPlan]) -> list[str]:
    return [e.match_id for p in plans if p.action == "merge" for e in p.enrich if "status" in e.sets and e.open_recs]


def summarize(plans: list[PairPlan]) -> Counter:
    c: Counter = Counter()
    for p in plans:
        c[f"pairs_{p.action}"] += 1
        if p.action != "merge":
            continue
        c["matches_repointed"] += len(p.repoints)
        c["twins_dropped"] += len(p.drops)
        c["twins_dropped_finished"] += sum(1 for d in p.drops if d.observed["status"] == FINISHED)
        c["kept_rows_enriched"] += sum(1 for e in p.enrich if e.sets or e.external_patch or e.metadata_patch)
        c["kept_rows_taking_predictions"] += sum(1 for e in p.enrich if e.takes_predictions)
        c["results_adopted"] += sum(1 for e in p.enrich if "status" in e.sets)
        c["kickoffs_adopted"] += sum(1 for e in p.enrich if "match_date" in e.sets)
        c["odds_moved"] += sum(len(d.odds_ids) for d in p.drops)
        c["snapshots_moved"] += sum(len(d.snapshot_ids) for d in p.drops)
        c["match_stats_moved"] += sum(len(d.match_stats_ids) for d in p.drops)
        c["predictions_moved"] += sum(len(d.prediction_ids) for d in p.drops)
        c["fd_twins_dropped"] += sum(1 for d in p.drops if d.flags)
        c["fd_kickoff_differs"] += sum(1 for d in p.drops if d.flags.get("kickoff_differs"))
        c["forks_folded"] += len({d.keep_id for d in p.drops if d.fork})
        c["forks_left_alone"] += len(p.forks)
        c["espn_moved_from_twin"] += 1 if p.team_patch["new_team_espn"] else 0
        c["espn_stamped"] += 1 if p.team_patch["stamp"] else 0
        c["teams_deleted"] += 1
    c["grade_after"] = len(grade_after(plans))
    return c


# ---------------------------------------------------------------------------
# Audit CSV, plan file, archive (write-once, before any write)
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
    "flags",
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
            add(
                "drop_twin",
                "matches",
                d.match_id,
                keep_id=d.keep_id,
                before=_j(d.observed),
                reason=why,
                flags=_j(d.flags) if d.flags else "",
            )
            for table, ids in (
                ("odds", d.odds_ids),
                ("odds_snapshots", d.snapshot_ids),
                ("match_stats", d.match_stats_ids),
                ("predictions", d.prediction_ids),
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
            after=_j({"aliases": t["aliases"], "espn": t["espn"]}),
            reason=(
                f"ESPN id {t['new_team_espn']} moves off the new team"
                if t["new_team_espn"]
                else "no ESPN id on the twin"
            ),
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


def plan_document(plans: list[PairPlan], mode: str) -> dict:
    """The plan as data: what --expect-plan compares and --check-after checks.
    'digest' covers the structural decisions (which pairs merge, which rows
    are dropped onto which, re-pointed, enriched, which ESPN id lands where),
    not the price rows that move with them."""
    pairs = []
    for p in plans:
        entry: dict[str, Any] = {
            "index": p.index,
            "new_team_id": p.pair.new_team_id,
            "old_team_id": p.pair.old_team_id,
            "new_name": p.pair.new_name,
            "old_name": p.pair.old_name,
            "action": p.action,
            "reason": p.reason,
        }
        if p.action == "merge":
            entry.update(
                league_id=str(p.new_team.get("league_id") or "") or None,
                espn=p.team_patch["espn"],
                drops=sorted([d.match_id, d.keep_id] for d in p.drops),
                repoints=sorted([r["match_id"], r["side"]] for r in p.repoints),
                enrich=sorted([e.match_id, sorted(e.sets), sorted(e.external_patch)] for e in p.enrich),
                forks=sorted(sorted(f["ids"]) for f in p.forks),
            )
        pairs.append(entry)
    structural = [
        {
            k: v
            for k, v in e.items()
            if k in ("new_team_id", "old_team_id", "action", "espn", "drops", "repoints", "enrich")
        }
        for e in pairs
    ]
    return {
        "tool": "merge_soccer_teams",
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "digest": hashlib.sha256(json.dumps(structural, sort_keys=True).encode()).hexdigest(),
        "pairs": pairs,
        "grade_after": grade_after(plans),
        "adopted_with_open_recs": sorted(adopted_with_open_recs(plans)),
    }


def write_plan(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as fh:  # write-once, like the CSV
        json.dump(doc, fh, indent=1, sort_keys=True)


def plan_difference(expected: dict, doc: dict) -> Optional[str]:
    """None when `doc` is the plan `expected` (a dry run's plan file) showed,
    else the first difference. Pure."""
    if expected.get("tool") != "merge_soccer_teams" or "digest" not in expected:
        return "the expected plan file is not a merge_soccer_teams plan"
    if expected["digest"] == doc["digest"]:
        return None
    want = {e["new_team_id"]: e for e in expected.get("pairs", [])}
    for e in doc["pairs"]:
        w = want.pop(e["new_team_id"], None)
        if w is None:
            return f"pair {e['new_name']} is not in the expected plan"
        for k in ("action", "espn", "drops", "repoints", "enrich"):
            if w.get(k) != e.get(k):
                return f"pair {e['new_name']}: {k} differs from the expected plan ({_delta(w.get(k), e.get(k))})"
    if want:
        return f"the expected plan has pair(s) this run does not: {sorted(want)[:3]}"
    return "the plan digest differs from the expected plan"


def _delta(want: Any, got: Any) -> str:
    """'+[added] -[removed]' for two lists of entries, else 'want -> got'."""
    if isinstance(want, list) and isinstance(got, list):
        w, g = {json.dumps(x) for x in want}, {json.dumps(x) for x in got}
        added, removed = sorted(g - w), sorted(w - g)
        return f"{len(added)} added {added[:3]}, {len(removed)} removed {removed[:3]}"
    return f"{want!r} -> {got!r}"


Archived = dict[str, dict[str, set[str]]]  # dropped twin id -> child table -> archived row ids


def archived_ids(archived: Archived) -> dict[str, set[str]]:
    """child table -> every row id the archive holds for it, across all
    dropped twins. A row archived under one twin can sit on another one by
    the time that one is dropped: pair i moves it from its twin onto kept row
    K, then pair j drops K. It is in the archive all the same. Pure."""
    out: dict[str, set[str]] = defaultdict(set)
    for tables in archived.values():
        for table, ids in tables.items():
            out[table] |= ids
    return dict(out)


def write_archive(cur, path: Path, plans: list[PairPlan], children: tuple[str, ...]) -> tuple[int, Archived]:
    """gzip JSONL of every row the run deletes or changes, written before the
    first write. Returns (record count, the child rows archived per dropped
    twin): apply_pair refuses to delete a twin holding a row not in it.
    `children` are the tables whose rows cascade with a dropped twin
    (Schema.match_children)."""
    merges = [p for p in plans if p.action == "merge"]
    drop_ids = [d.match_id for p in merges for d in p.drops]
    keep_of = {d.match_id: d.keep_id for p in merges for d in p.drops}
    enrich_ids = [e.match_id for p in merges for e in p.enrich]
    team_ids = [t for p in merges for t in (p.pair.new_team_id, p.pair.old_team_id)]
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    archived: Archived = {mid: {} for mid in drop_ids}
    with gzip.open(path, "xt") as fh:

        def emit(table: str, role: str, rows: Iterable[dict], **extra: Any) -> None:
            nonlocal n
            for r in rows:
                rec = {"table": table, "role": role, **extra, "row": {k: _jsonable_row(v) for k, v in r.items()}}
                if table == "matches" and role == "dropped_twin":
                    rec["keep_id"] = keep_of.get(str(r["id"]))
                if role == "dropped_twin_child":
                    archived.setdefault(str(r["match_id"]), {}).setdefault(table, set()).add(str(r["id"]))
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
            for table in children:
                cur.execute(f"SELECT * FROM {table} WHERE match_id = ANY(%s::uuid[])", (chunk,))
                emit(table, "dropped_twin_child", cur.fetchall())
    return n, archived


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
    SELECT id::text AS id, name, normalized_name, sport, league_id::text AS league_id, external_ids, metadata
      FROM teams WHERE id = ANY(%s::uuid[])
"""

ALL_SOCCER_TEAMS_SQL = """
    SELECT id::text AS id, normalized_name, external_ids->>'espn' AS espn FROM teams WHERE sport = 'soccer'
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

PREDICTIONS_SQL = """
    SELECT id::text AS id, match_id::text AS match_id, model_name, model_version, prediction_type,
           created_at, updated_at
      FROM predictions WHERE match_id = ANY(%s::uuid[])
"""


@dataclass
class State:
    teams: dict[str, dict]
    refs: dict[str, dict]
    matches: dict[str, dict]
    deps: dict[str, Deps]
    ctx: Ctx


def load_schema(cur) -> tuple[Optional[Schema], list[str]]:
    cur.execute(FOREIGN_KEYS_SQL)
    return read_schema(dict(r) for r in cur.fetchall())


def load_state(cur, pairs: list[MapRow], schema: Schema, resolvers: Resolvers) -> State:
    """Everything the plan needs, read-only."""
    new_ids = [p.new_team_id for p in pairs]
    old_ids = sorted({p.old_team_id for p in pairs})
    cur.execute(TEAMS_SQL, (new_ids + old_ids,))
    teams = {r["id"]: dict(r) for r in cur.fetchall()}
    cur.execute(team_refs_sql(schema), (new_ids,))
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
        for sql, attr in (
            (ODDS_SQL, "odds"),
            (SNAPSHOTS_SQL, "snapshots"),
            (MATCH_STATS_SQL, "match_stats"),
            (PREDICTIONS_SQL, "predictions"),
        ):
            cur.execute(sql, (chunk,))
            for r in cur.fetchall():
                getattr(deps[r["match_id"]], attr).append(dict(r))
    cur.execute(ALL_SOCCER_TEAMS_SQL)
    team_rows = [dict(r) for r in cur.fetchall()]
    ctx = build_ctx(team_rows, resolvers.load_alias_index(cur, SPORT), resolvers.espn_seed(), resolvers.alias_form)
    return State(teams, refs, matches, deps, ctx)


# ---------------------------------------------------------------------------
# Writing one pair (one transaction)
# ---------------------------------------------------------------------------

LOCK_TEAMS_SQL = """
    SELECT id::text AS id, name, normalized_name, sport, league_id::text AS league_id, external_ids, metadata
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

# Strictly pre-kickoff content (an upsert after kickoff rewrites a row, so
# updated_at counts), and only where the kept row has no prediction under the
# unique key (match_id, model_name, model_version, prediction_type).
MOVE_PREDICTIONS_SQL = """
    UPDATE predictions p SET match_id = %(keep)s::uuid
     WHERE p.id = ANY(%(ids)s::uuid[])
       AND p.match_id = %(drop)s::uuid
       AND COALESCE(p.updated_at, p.created_at) < %(cutoff)s
       AND NOT EXISTS (
             SELECT 1 FROM predictions t
              WHERE t.match_id = %(keep)s::uuid
                AND t.model_name = p.model_name
                AND t.model_version IS NOT DISTINCT FROM p.model_version
                AND t.prediction_type = p.prediction_type)
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
           match_date = %(match_date)s,
           external_ids = COALESCE(external_ids, '{}'::jsonb) || %(external_patch)s::jsonb,
           metadata = COALESCE(metadata, '{}'::jsonb) || %(metadata_patch)s::jsonb,
           updated_at = NOW()
     WHERE id = %(id)s::uuid
       AND status = %(o_status)s
       AND match_date = %(o_match_date)s
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

TWINS_AFTER_SQL = """
    SELECT m.id::text AS a, o.id::text AS b
      FROM matches m
      JOIN matches o ON o.home_team_id = m.home_team_id AND o.away_team_id = m.away_team_id AND o.id <> m.id
                    AND o.match_date BETWEEN m.match_date - %(window)s::interval AND m.match_date + %(window)s::interval
     WHERE m.id = ANY(%(ids)s::uuid[])
"""

# The twin's stamp comes off first: team_identity.STAMP_ESPN_SQL refuses an id
# another row of the sport still holds.
CLEAR_TWIN_ESPN_SQL = """
    UPDATE teams SET external_ids = external_ids - 'espn', updated_at = NOW()
     WHERE id = %(id)s::uuid AND sport = 'soccer' AND external_ids->>'espn' = %(espn)s
"""

TEAM_RECORD_SQL = """
    UPDATE teams
       SET metadata = jsonb_set(jsonb_set(COALESCE(metadata, '{}'::jsonb), '{aliases}', %(aliases)s::jsonb),
                                '{merged_from}', %(merged_from)s::jsonb),
           updated_at = NOW()
     WHERE id = %(id)s::uuid AND sport = 'soccer' AND name = %(name)s
       AND COALESCE(metadata->'aliases', '[]'::jsonb) = %(observed_aliases)s::jsonb
       AND COALESCE(metadata->'merged_from', '[]'::jsonb) = %(observed_merged_from)s::jsonb
       AND COALESCE(external_ids->>'espn', '') = %(observed_espn)s
"""


def twin_children_sql(schema: Schema) -> str:
    """Every row hanging off one dropped twin (%(m)s), per child table. The
    table names come from read_schema, i.e. from the MATCH_CHILDREN whitelist."""
    return "\nUNION ALL\n".join(
        f"SELECT '{t}' AS tbl, id::text AS id FROM {t} WHERE match_id = %(m)s::uuid" for t in schema.match_children
    )


REPOINTED_FIXTURE_SQL = """
    SELECT m.match_date, m.status, ht.name AS home_name, at.name AS away_name
      FROM matches m JOIN teams ht ON ht.id = m.home_team_id JOIN teams at ON at.id = m.away_team_id
     WHERE m.id = %s::uuid
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


def odds_probe(plan: PairPlan) -> Optional[dict]:
    """The re-pointed fixture the odds resolver is checked on: the latest
    re-point fetch_live_odds.find_match_id can see (scheduled or finished
    after the run), None when the pair re-points none."""
    final_status = {e.match_id: e.sets.get("status", e.observed["status"]) for e in plan.enrich}
    cands = [
        r for r in plan.repoints if final_status.get(r["match_id"], r["observed"]["status"]) in RESOLVABLE_STATUSES
    ]
    if not cands:
        return None
    return max(cands, key=lambda r: (r["observed"]["match_date"], r["match_id"]))


def probe_fixture(cur, plan: PairPlan) -> Optional[tuple[dict, dict]]:
    """(the odds_probe re-point, its fixture as it reads now: kickoff, team
    names), or None when the pair re-points no fixture the odds resolver sees."""
    probe = odds_probe(plan)
    if probe is None:
        return None
    cur.execute(REPOINTED_FIXTURE_SQL, (probe["match_id"],))
    row = cur.fetchone()
    if not row:
        raise GuardError(f"re-pointed fixture {probe['match_id']} is gone")
    return probe, dict(row)


def odds_event(probe: dict, row: dict, spelling: str) -> tuple[str, str, datetime]:
    """(home, away, kickoff) of an odds event for the probe fixture that names
    the merged club `spelling` and the opponent as our row does."""
    home = spelling if probe["side"] == "home" else row["home_name"]
    away = spelling if probe["side"] == "away" else row["away_name"]
    return home, away, row["match_date"]


def odds_before(cur, plan: PairPlan, resolvers: Resolvers) -> dict[str, Optional[str]]:
    """Before the pair's first write: the fixture fetch_live_odds.find_match_id
    finds for each of the odds feed's spellings of the club (plan.odds_names)
    on the probe fixture, which still names the twin. verify_resolution holds
    the merge to it."""
    if not plan.odds_names:
        return {}
    found = probe_fixture(cur, plan)
    if found is None:
        return {}
    probe, row = found
    out: dict[str, Optional[str]] = {}
    for spelling in plan.odds_names:
        got = resolvers.find_match_id(cur, *odds_event(probe, row, spelling))
        out[spelling] = str(got) if got else None
    return out


def verify_resolution(cur, plan: PairPlan, resolvers: Resolvers, before: dict[str, Optional[str]]) -> Counter:
    """Inside the pair's transaction, after the team delete: the deployed
    resolvers must resolve the merged spelling to the kept team, and every
    odds-feed spelling that found the probe fixture before the first write
    (`before`, from odds_before) must still find it, else GuardError (the
    pair rolls back). Returns the checks run."""
    pair, t = plan.pair, plan.team_patch
    league = plan.new_team.get("league_id")
    done: Counter = Counter()
    checks = 0
    got = resolvers.find_team(cur, pair.new_name, league, None)
    checks += 1
    if str(got or "") != pair.old_team_id:
        raise GuardError(
            f"fetch_upcoming.find_team({pair.new_name!r}) resolves to {got!r}, not the kept team {pair.old_team_id}: "
            f"the next sweep would mint the team again"
        )
    if t["espn"]:
        got = resolvers.find_team(cur, pair.new_name, league, t["espn"])
        checks += 1
        if str(got or "") != pair.old_team_id:
            raise GuardError(
                f"fetch_upcoming.find_team({pair.new_name!r}, ESPN {t['espn']}) resolves to {got!r}, not the kept "
                f"team {pair.old_team_id}"
            )
    found = probe_fixture(cur, plan)
    if found is not None:
        probe, row = found
        home, away, when = odds_event(probe, row, pair.new_name)
        got = resolvers.find_match_id(cur, home, away, when)
        checks += 1
        if str(got or "") != probe["match_id"]:
            raise GuardError(
                f"fetch_live_odds.find_match_id({home!r}, {away!r}, {when}) finds {got!r}, not the "
                f"re-pointed fixture {probe['match_id']}: odds for the merged club would stop matching"
            )
        keep_of = {d.match_id: d.keep_id for d in plan.drops}
        for spelling in plan.odds_names:
            prior = before.get(spelling)
            if keep_of.get(prior or "", prior) != probe["match_id"]:
                # Nothing to keep: it missed before the merge too. Loud, and counted on the WRITTEN line, which
                # scripts/ops/merge_soccer_teams.sh fails on: the in-transaction odds check shrank for this club.
                done["odds_spellings_unmatched_before"] += 1
                logger.warning(
                    "%s -> %s: the odds feed's spelling %r did not find the probe fixture %s before the merge "
                    "(found %r) — not checked after it",
                    pair.new_name,
                    pair.old_name,
                    spelling,
                    probe["match_id"],
                    prior,
                )
                continue
            home, away, when = odds_event(probe, row, spelling)
            got = resolvers.find_match_id(cur, home, away, when)
            done["odds_spelling_checks"] += 1
            if str(got or "") != probe["match_id"]:
                raise GuardError(
                    f"fetch_live_odds.find_match_id({home!r}, {away!r}, {when}) — the odds feed's spelling — found "
                    f"the fixture {probe['match_id']} before the merge and finds {got!r} after it: odds for the "
                    f"merged club would stop matching"
                )
    done["resolver_checks"] += checks
    return done


def apply_pair(conn, plan: PairPlan, resolvers: Resolvers, schema: Schema, archived: Archived) -> Counter:
    """Write one planned pair in one transaction; GuardError / psycopg2.Error
    roll it back. Returns what it wrote. `archived` is write_archive's record
    of the child rows it archived per dropped twin."""
    pair = plan.pair
    t = plan.team_patch
    done: Counter = Counter()
    window = f"{int(TWIN_WINDOW.total_seconds() // 3600)} hours"
    held: Optional[dict[str, set[str]]] = None  # archived_ids(archived), built on the first dropped twin
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
            before = odds_before(cur, plan, resolvers)
            for d in plan.drops:
                for sql, ids, label in (
                    (MOVE_ODDS_SQL, d.odds_ids, "odds"),
                    (MOVE_SNAPSHOTS_SQL, d.snapshot_ids, "odds_snapshots"),
                    (MOVE_MATCH_STATS_SQL, d.match_stats_ids, "match_stats"),
                    (MOVE_PREDICTIONS_SQL, d.prediction_ids, "predictions"),
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
                if schema.match_children:
                    # The twin row is locked FOR UPDATE, so no child row can be added before COMMIT; one added
                    # between the archive and the lock would cascade away unarchived. Any row the archive holds
                    # passes, under whichever twin it was archived (an earlier pair may have moved it here).
                    cur.execute(twin_children_sql(schema), {"m": d.match_id})
                    kept = held if held is not None else archived_ids(archived)
                    held = kept
                    extra = sorted(
                        f"{r['tbl']}:{r['id']}" for r in cur.fetchall() if str(r["id"]) not in kept.get(r["tbl"], ())
                    )
                    if extra:
                        raise GuardError(
                            f"twin {d.match_id} holds {len(extra)} row(s) written after the archive, e.g. {extra[:3]}: "
                            f"deleting it would lose them unarchived"
                        )
                _expect(cur, DROP_TWIN_SQL, {"id": d.match_id, **d.observed}, 1, f"drop twin {d.match_id}")
                done["twins_dropped"] += 1
            for e in plan.enrich:
                if not (e.sets or e.external_patch or e.metadata_patch):
                    continue  # only took predictions
                o = e.observed
                params = {
                    "id": e.match_id,
                    "status": e.sets.get("status", o["status"]),
                    "home_score": e.sets.get("home_score", o["home_score"]),
                    "away_score": e.sets.get("away_score", o["away_score"]),
                    "half_time_home": e.sets.get("half_time_home", o["half_time_home"]),
                    "half_time_away": e.sets.get("half_time_away", o["half_time_away"]),
                    "match_date": e.sets.get("match_date", o["match_date"]),
                    "external_patch": json.dumps(e.external_patch),
                    "metadata_patch": json.dumps(e.metadata_patch, default=_jsonable),
                    **{
                        f"o_{k}": o[k]
                        for k in (
                            "status",
                            "match_date",
                            "home_score",
                            "away_score",
                            "half_time_home",
                            "half_time_away",
                            "espn",
                        )
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
            cur.execute(new_team_refs_sql(schema), {"t": pair.new_team_id})
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
            if t["new_team_espn"]:
                _expect(
                    cur,
                    CLEAR_TWIN_ESPN_SQL,
                    {"id": pair.new_team_id, "espn": t["new_team_espn"]},
                    1,
                    f"take ESPN id {t['new_team_espn']} off {pair.new_name}",
                )
            _expect(
                cur,
                TEAM_RECORD_SQL,
                {
                    "id": pair.old_team_id,
                    "name": pair.old_name,
                    "aliases": json.dumps(t["aliases"]),
                    "merged_from": json.dumps(t["merged_from"]),
                    "observed_aliases": json.dumps(t["observed_aliases"]),
                    "observed_merged_from": json.dumps(t["observed_merged_from"]),
                    "observed_espn": t["observed_espn"],
                },
                1,
                f"record alias on {pair.old_name}",
            )
            if t["stamp"]:
                if not resolvers.stamp_espn_id(cur, pair.old_team_id, t["espn"], f"soccer {pair.old_name!r} (merge)"):
                    raise GuardError(f"the guarded stamp refused ESPN id {t['espn']} on {pair.old_name} (see above)")
                done["espn_stamped"] += 1
            _expect(
                cur,
                team_delete_sql(schema),
                {"id": pair.new_team_id, "name": pair.new_name, "normalized_name": plan.new_team["normalized_name"]},
                1,
                f"delete team {pair.new_name}",
            )
            done["teams_deleted"] += 1
            done.update(verify_resolution(cur, plan, resolvers, before))
        conn.commit()
        return done
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# After-state (read-only)
# ---------------------------------------------------------------------------

FINISHED_TWINS_SQL = """
    SELECT a.id::text AS a, b.id::text AS b
      FROM matches a
      JOIN matches b ON b.home_team_id = a.home_team_id AND b.away_team_id = a.away_team_id AND b.id > a.id
                    AND b.match_date BETWEEN a.match_date - %(window)s::interval AND a.match_date + %(window)s::interval
     WHERE a.status = 'finished' AND b.status = 'finished'
       AND (a.home_team_id = ANY(%(teams)s::uuid[]) OR a.away_team_id = ANY(%(teams)s::uuid[]))
"""

UNSETTLED_RECS_SQL = """
    SELECT count(*) AS n FROM betting_recommendations r
     WHERE r.match_id = ANY(%s::uuid[]) AND (r.status IN ('pending', 'placed') OR r.profit_loss IS NULL)
"""


# The first fetch_live_odds run after `since` (the merge) that quoted soccer:
# the latest stamp on an upcoming soccer fixture's odds. Odds are reached
# through matches (indexed match_id), never by scanning odds.
ODDS_LAST_RUN_SQL = """
    SELECT max(o.last_seen_at) AS last_run
      FROM matches m
      JOIN teams ht ON ht.id = m.home_team_id
      JOIN odds o ON o.match_id = m.id
     WHERE ht.sport = 'soccer'
       AND m.match_date > %(since)s AND m.match_date <= %(since)s::timestamptz + interval '14 days'
       AND o.last_seen_at >= %(since)s
"""

# Fixtures still to be played at that run whose odds the feed was quoting in
# the hours before `since` and that got no quote since: the run did not tie
# the feed's event to them any more. `merged` splits the merged clubs'
# fixtures from the rest (the base rate of a book pulling a market).
ODDS_STOPPED_SQL = """
    SELECT m.id::text AS id, m.match_date, ht.name AS home, at.name AS away,
           (m.home_team_id = ANY(%(teams)s::uuid[]) OR m.away_team_id = ANY(%(teams)s::uuid[])) AS merged
      FROM matches m
      JOIN teams ht ON ht.id = m.home_team_id
      JOIN teams at ON at.id = m.away_team_id
     WHERE ht.sport = 'soccer' AND m.status = 'scheduled'
       AND m.match_date > %(last_run)s AND m.match_date <= %(since)s::timestamptz + interval '14 days'
       AND EXISTS (SELECT 1 FROM odds o WHERE o.match_id = m.id
                     AND o.last_seen_at >= %(since)s::timestamptz - %(lookback)s::interval
                     AND o.last_seen_at < %(since)s)
       AND NOT EXISTS (SELECT 1 FROM odds o WHERE o.match_id = m.id AND o.last_seen_at >= %(since)s)
     ORDER BY m.match_date, m.id
"""
ODDS_LOOKBACK = timedelta(hours=6)  # fetch_live_odds runs every 90 min: the last four runs before the merge

# Counts check_after reports that are not "must be 0".
AFTER_INFO = ("pairs_recorded", "pairs_expected", "odds_control_stopped")
# Also reported only, with --after-later-steps: the runbook's odds check (step 11) runs after
# repair_football_data_kickoffs and cleanup_soccer_orphans, which delete some re-pointed rows on
# purpose (football-data copies, stale orphans). Every other count still holds after them.
AFTER_LATER_STEPS_INFO = ("repoints_missing",)


def check_odds(cur, old_ids: list[str], since: datetime) -> dict:
    """{odds_not_yet_observed, odds_fixtures_stopped, odds_control_stopped}
    for the first fetch_live_odds run after `since`. Logs every stopped
    fixture of a merged club."""
    cur.execute(ODDS_LAST_RUN_SQL, {"since": since})
    row = cur.fetchone()
    last_run = row["last_run"] if row else None
    if last_run is None:
        logger.error(
            "ODDS: no fetch_live_odds run has quoted a soccer fixture since %s — re-run this check later", since
        )
        return {"odds_not_yet_observed": 1, "odds_fixtures_stopped": 0, "odds_control_stopped": 0}
    cur.execute(
        ODDS_STOPPED_SQL,
        {
            "teams": old_ids,
            "since": since,
            "last_run": last_run,
            "lookback": f"{int(ODDS_LOOKBACK.total_seconds())} seconds",
        },
    )
    rows = [dict(r) for r in cur.fetchall()]
    stopped = [r for r in rows if r["merged"]]
    for r in stopped:
        logger.error(
            "ODDS STOPPED: %s %s v %s @ %s was quoted before %s and not by the run at %s",
            r["id"],
            r["home"],
            r["away"],
            r["match_date"],
            since,
            last_run,
        )
    logger.info(
        "ODDS: last run %s; %d merged-club fixture(s) stopped, %d other(s)",
        last_run,
        len(stopped),
        len(rows) - len(stopped),
    )
    return {
        "odds_not_yet_observed": 0,
        "odds_fixtures_stopped": len(stopped),
        "odds_control_stopped": len(rows) - len(stopped),
    }


def check_after(conn, doc: dict, resolvers: Resolvers, odds_since: Optional[datetime] = None) -> dict:
    """The after-state a plan file implies, read-only: {check: count}. Every
    count must be 0 except AFTER_INFO (pairs_recorded must equal
    pairs_expected). With `odds_since` (when the merge committed), it also
    checks that the first fetch_live_odds run after it still quoted every
    merged club's fixture the feed was quoting before (check_odds)."""
    pairs = [p for p in doc["pairs"] if p["action"] in ("merge", "done")]
    new_ids = [p["new_team_id"] for p in pairs]
    old_ids = sorted({p["old_team_id"] for p in pairs})
    merges = [p for p in pairs if p["action"] == "merge"]
    drop_ids = [d for p in merges for d, _ in p.get("drops", [])]
    forks = {tuple(sorted((a, b))) for p in merges for f in p.get("forks", []) for a in f for b in f if a < b}
    out: dict[str, int] = {"pairs_expected": len(pairs)}
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SELECT count(*) AS n FROM teams WHERE id = ANY(%s::uuid[])", (new_ids,))
        out["merged_team_rows_left"] = int(cur.fetchone()["n"])
        cur.execute(
            "SELECT count(*) AS n FROM matches "
            "WHERE home_team_id = ANY(%(t)s::uuid[]) OR away_team_id = ANY(%(t)s::uuid[])",
            {"t": new_ids},
        )
        out["matches_on_merged_ids"] = int(cur.fetchone()["n"])
        cur.execute("SELECT id::text AS id, metadata, external_ids FROM teams WHERE id = ANY(%s::uuid[])", (old_ids,))
        kept = {r["id"]: dict(r) for r in cur.fetchall()}
        recorded = 0
        for p in pairs:
            t = kept.get(p["old_team_id"]) or {}
            meta = t.get("metadata") or {}
            merged = {str(e.get("team_id")) for e in meta.get(MERGED_FROM_KEY) or [] if isinstance(e, dict)}
            forms = {resolvers.alias_form(a) for a in (aliases_of(t) or [])}
            recorded += int(p["new_team_id"] in merged and resolvers.alias_form(p["new_name"]) in forms)
        out["pairs_recorded"] = recorded
        cur.execute("SELECT count(*) AS n FROM matches WHERE id = ANY(%s::uuid[])", (drop_ids,))
        out["dropped_twins_left"] = int(cur.fetchone()["n"])
        # A row one pair re-points can become a twin a later pair drops: gone on purpose.
        gone = set(drop_ids)
        repointed = [(m, s, p["old_team_id"]) for p in merges for m, s in p.get("repoints", []) if m not in gone]
        cur.execute(
            "SELECT id::text AS id, home_team_id::text AS home, away_team_id::text AS away FROM matches "
            "WHERE id = ANY(%s::uuid[])",
            ([m for m, _, _ in repointed],),
        )
        rows = {r["id"]: r for r in cur.fetchall()}
        out["repoints_missing"] = sum(1 for m, _, _ in repointed if m not in rows)
        out["repoints_not_landed"] = sum(1 for m, s, old in repointed if m in rows and rows[m][s] != old)
        cur.execute(
            FINISHED_TWINS_SQL, {"teams": old_ids, "window": f"{int(TWIN_WINDOW.total_seconds() // 3600)} hours"}
        )
        out["finished_twins"] = sum(1 for r in cur.fetchall() if tuple(sorted((r["a"], r["b"]))) not in forks)
        cur.execute(ALL_SOCCER_TEAMS_SQL)
        teams = [dict(r) for r in cur.fetchall()]
        spellings = {resolvers.alias_form(p["new_name"]): p["old_team_id"] for p in pairs}
        out["teams_reminted"] = sum(
            1 for r in teams if spellings.get(resolvers.alias_form(r["normalized_name"])) not in (None, r["id"])
        )
        holders: dict[str, set[str]] = defaultdict(set)
        for r in teams:
            if r.get("espn"):
                holders[str(r["espn"])].add(r["id"])
        misses = misplaced = 0
        for p in pairs:
            espn = p.get("espn") or str(
                ((kept.get(p["old_team_id"]) or {}).get("external_ids") or {}).get("espn") or ""
            )
            league = p.get("league_id")
            misses += int(str(resolvers.find_team(cur, p["new_name"], league, None) or "") != p["old_team_id"])
            if espn:
                misses += int(str(resolvers.find_team(cur, p["new_name"], league, espn) or "") != p["old_team_id"])
                misplaced += int(holders.get(espn, set()) != {p["old_team_id"]})
        out["resolver_misses"] = misses
        out["espn_misplaced"] = misplaced
        adopted = doc.get("adopted_with_open_recs") or []
        if adopted:
            cur.execute(UNSETTLED_RECS_SQL, (adopted,))
            out["unsettled_recs_on_adopted"] = int(cur.fetchone()["n"])
        else:
            out["unsettled_recs_on_adopted"] = 0
        if odds_since is not None:
            out.update(check_odds(cur, old_ids, odds_since))
    conn.rollback()
    return out


def after_state_ok(out: dict, after_later_steps: bool = False) -> bool:
    info = AFTER_INFO + (AFTER_LATER_STEPS_INFO if after_later_steps else ())
    return out.get("pairs_recorded") == out.get("pairs_expected") and all(
        v == 0 for k, v in out.items() if k not in info
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _stamp() -> str:
    # Microseconds: a dry run and an apply in the same second must not collide on the write-once names.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def default_out(kind: str) -> Path:
    base = Path(os.environ.get("BACKUP_LOCAL_DIR", "/app/backups"))
    ext = {"csv": "csv", "archive": "jsonl.gz", "plan": "plan.json"}[kind]
    return base / f"merge_soccer_teams_{_stamp()}.{ext}"


def log_plan(plans: list[PairPlan]) -> None:
    for p in plans:
        tag = f"[{p.index}/{len(plans)}] {p.pair.new_name} -> {p.pair.old_name}"
        if p.action == "refuse":
            logger.error("%s: REFUSED — %s", tag, p.reason)
        elif p.action == "done":
            logger.info("%s: done (%s)", tag, p.reason)
        else:
            logger.info(
                "%s: repoint %d, drop %d twin(s), enrich %d, leave %d fork(s)%s",
                tag,
                len(p.repoints),
                len(p.drops),
                len(p.enrich),
                len(p.forks),
                (
                    f", ESPN id {p.team_patch['espn']}"
                    + (" moves off the twin" if p.team_patch["new_team_espn"] else "")
                    if p.team_patch["espn"]
                    else ""
                ),
            )
            for d in p.drops:
                if d.flags.get("kickoff_differs"):
                    logger.warning(
                        "%s: dropped football-data twin %s corrects to %s, the kept row %s sits at %s "
                        "(promote_raw finds it: %s)",
                        tag,
                        d.match_id,
                        d.flags["fd_corrected_kickoff"],
                        d.keep_id,
                        d.flags["kept_kickoff"],
                        d.flags["promote_finds_kept"],
                    )


def run(
    conn,
    apply_changes: bool,
    pairs: list[MapRow],
    out_csv: Path,
    archive: Path,
    resolvers: Optional[Resolvers],
    resolver_reason: Optional[str] = None,
    plan_out: Optional[Path] = None,
    expect_plan: Optional[dict] = None,
    odds_names: Optional[dict[str, list[str]]] = None,
) -> int:
    if resolvers is None or resolver_reason:
        logger.error("REFUSED: %s — deploy the team-identity code first; nothing read or written", resolver_reason)
        return 2
    seed = resolvers.espn_seed()
    clash = map_seed_conflicts(pairs, seed)
    if clash:
        for nid, why in clash.items():
            logger.error("REFUSED: map row %s contradicts the ESPN seed: %s", nid, why)
        logger.error("REFUSED: fix the merge map or the seed; nothing read or written")
        return 2
    now = datetime.now(timezone.utc)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        schema, unknown = load_schema(cur)
        if schema is None:
            conn.rollback()
            logger.error(
                "REFUSED: foreign keys this tool does not know (a delete could cascade into them unarchived): %s — "
                "add them to TEAM_REFERENCES / MATCH_CHILDREN first; nothing written",
                "; ".join(unknown),
            )
            return 2
        state = load_state(cur, pairs, schema, resolvers)
    conn.rollback()  # close the read snapshot
    plans = plan_all(pairs, state.teams, state.refs, state.matches, state.deps, now, state.ctx, odds_names)
    write_csv(out_csv, csv_rows(plans))
    doc = plan_document(plans, "apply" if apply_changes else "dry-run")
    plan_path = plan_out or out_csv.with_suffix(".plan.json")
    write_plan(plan_path, doc)
    log_plan(plans)
    planned = summarize(plans)
    logger.info("Planned: %s", dict(sorted(planned.items())))
    logger.info("Audit CSV: %s", out_csv)
    logger.info("PLAN_FILE %s", plan_path)
    # One fixed-format line for scripts/ops/merge_soccer_teams.sh to parse.
    logger.info(
        "PLAN pairs_merge=%d pairs_done=%d pairs_refuse=%d kickoffs_adopted=%d digest=%s",
        planned["pairs_merge"],
        planned["pairs_done"],
        planned["pairs_refuse"],
        planned["kickoffs_adopted"],
        doc["digest"],
    )
    for mid in doc["grade_after"]:
        logger.info("GRADE_AFTER %s (grade_completed_matches.py --match-id %s)", mid, mid)
    refused = planned["pairs_refuse"]
    if expect_plan is not None:
        diff = plan_difference(expect_plan, doc)
        if diff:
            logger.error("REFUSED: this plan is not the one the dry run showed — %s; nothing written", diff)
            return 2
        logger.info("Plan matches the expected plan (digest %s)", doc["digest"])
    if not apply_changes:
        logger.info("DRY-RUN: nothing written; re-run with --apply to merge %d pair(s)", planned["pairs_merge"])
        return 1 if refused else 0
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        n, archived = write_archive(cur, archive, plans, schema.match_children)
    conn.rollback()
    logger.info("Archived %d row(s) to %s", n, archive)
    written: Counter = Counter()
    failed = []
    for p in plans:
        if p.action != "merge":
            continue
        try:
            written.update(apply_pair(conn, p, resolvers, schema, archived))
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
    logger.info(
        "WRITTEN pairs_merged=%d pairs_failed=%d odds_spelling_checks=%d odds_spellings_unmatched_before=%d",
        written["pairs_merged"],
        len(failed),
        written["odds_spelling_checks"],
        written["odds_spellings_unmatched_before"],
    )
    if failed or refused:
        logger.error(
            "%d pair(s) rolled back, %d refused — re-run after inspecting them (merged pairs plan as done)",
            len(failed),
            refused,
        )
        return 1
    return 0


def run_check_after(
    conn,
    plan_path: Path,
    resolvers: Optional[Resolvers],
    resolver_reason: Optional[str],
    odds_since: Optional[datetime] = None,
    after_later_steps: bool = False,
) -> int:
    if resolvers is None or resolver_reason:
        logger.error("REFUSED: %s", resolver_reason)
        return 2
    doc = json.loads(plan_path.read_text())
    if doc.get("tool") != "merge_soccer_teams":
        logger.error("REFUSED: %s is not a merge_soccer_teams plan file", plan_path)
        return 2
    out = check_after(conn, doc, resolvers, odds_since)
    ok = after_state_ok(out, after_later_steps)
    logger.info("AFTER %s", " ".join(f"{k}={v}" for k, v in out.items()))
    if after_later_steps and out.get("repoints_missing"):
        logger.info(
            "repoints_missing=%d reported, not failed (--after-later-steps): later repairs deleted those rows",
            out["repoints_missing"],
        )
    if ok:
        logger.info("AFTER-STATE OK: %d/%d pairs merged and resolving", out["pairs_recorded"], out["pairs_expected"])
        return 0
    logger.error(
        "AFTER-STATE FAILED: every count above must be 0%s and pairs_recorded=pairs_expected",
        " (but repoints_missing)" if after_later_steps else "",
    )
    return 1


def _utc_arg(s: str) -> datetime:
    """An ISO timestamp; naive means UTC."""
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="Write (default: dry run).")
    p.add_argument("--only", default=None, help="Comma-separated new_team_ids from the map to merge (default: all).")
    p.add_argument("--map", type=Path, default=MAP_PATH, help=argparse.SUPPRESS)
    p.add_argument("--out-csv", type=Path, default=None, help="Audit CSV (default $BACKUP_LOCAL_DIR or /app/backups).")
    p.add_argument("--plan-out", type=Path, default=None, help="Plan file (default: the audit CSV's .plan.json).")
    p.add_argument("--archive", type=Path, default=None, help="gzip JSONL archive written before --apply writes.")
    p.add_argument("--expect-plan", type=Path, default=None, help="--apply only if the plan equals this dry run's.")
    p.add_argument("--check-after", type=Path, default=None, help="Read-only: check the after-state of a plan file.")
    p.add_argument(
        "--odds-since",
        type=_utc_arg,
        default=None,
        help="With --check-after: when the merge committed (ISO, UTC); also check the next fetch_live_odds run.",
    )
    p.add_argument(
        "--after-later-steps",
        action="store_true",
        help="With --check-after: the run order's later repairs have run since the merge (they delete some "
        "re-pointed rows on purpose): repoints_missing is reported, not failed.",
    )
    p.add_argument("--odds-spellings", type=Path, default=ODDS_SPELLINGS_PATH, help=argparse.SUPPRESS)
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    resolvers, reason = load_resolvers()
    if (args.odds_since or args.after_later_steps) and not args.check_after:
        logger.error("REFUSED: --odds-since and --after-later-steps only go with --check-after")
        return 2
    if args.check_after:
        conn = psycopg2.connect(args.database_url)
        try:
            return run_check_after(conn, args.check_after, resolvers, reason, args.odds_since, args.after_later_steps)
        finally:
            conn.close()
    try:
        full_map = load_map(args.map)
        pairs = select_pairs(full_map, args.only)
        odds_names = load_odds_spellings(args.odds_spellings, full_map)
    except (MapError, OSError) as e:
        logger.error("REFUSED: %s", e)
        return 2
    if args.expect_plan and not args.apply:
        logger.error("REFUSED: --expect-plan only goes with --apply")
        return 2
    expect = json.loads(args.expect_plan.read_text()) if args.expect_plan else None
    out_csv = args.out_csv or default_out("csv")
    conn = psycopg2.connect(args.database_url)
    try:
        return run(
            conn,
            args.apply,
            pairs,
            out_csv,
            args.archive or default_out("archive"),
            resolvers,
            reason,
            plan_out=args.plan_out,
            expect_plan=expect,
            odds_names=odds_names,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
