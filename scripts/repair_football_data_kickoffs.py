"""Move the football-data kickoffs promote_raw stored 1 h late back to UTC,
and merge the duplicate matches that left behind into their ESPN rows.

Until 2026-10-31 promote_raw.parse_match_date stored football-data's Time,
which is UK local time, as if it were UTC, so every kickoff on a British
Summer Time date sits 1 h late (63,768 of the 101,158 timed promoted rows on
2026-10-31; the GMT ones are right). Where the ESPN results path had also
written the fixture at its real kickoff, the exact-triple upsert missed and
the fixture exists twice: 228 finished matches, 68 of them under the same
team ids (in the other 160 ESPN created its own team row; they become
same-id pairs once those teams are merged). promote_raw now parses UTC,
writes onto an existing row for the pair near the kickoff instead of beside
it (find_existing_match), and refuses to run while a legacy row is left
(check_legacy_kickoffs). This tool moves them.

A matches row is tied to its raw_football_data row the way promote_raw tied
them and backfill_season_labels ties them: league code, normalised team
names, and the kickoff promote_raw wrote (the UK wall clock,
promote_raw.parse_wall_clock, read in the database TimeZone) or, once moved,
the one it writes now.

--dedupe (run first). A football-data row is a copy when the row
promote_raw would now write its fixture onto is another row: an ESPN row
for the same home and away team ids within 3 h of the corrected kickoff, on
football-data's date (promote_raw.existing_match_window / choose_existing).
On prod that row sits 1 h (+-10 min) earlier for 67 of the 68 same-id pairs;
the 68th ESPN row kicked off 76 min after football-data's legacy time, and
two fixtures also keep a stale 'scheduled' row 60-70 min earlier, so a
strict 1 h +-10 min rule would miss three. The copy must be finished with
the ESPN row's final score. It is merged INTO the ESPN row: its half-time
score when the ESPN row has none, the pair is complete and neither half
exceeds the ESPN final score (promote_raw.raw_result's check; it needs no
ESPN plays); its odds where the ESPN row has no price under the same key
(uq_odds_current_price_key); its match_stats where the ESPN row has none for
that team. Then the copy row, the ESPN row's before-state and every row that
references the copy, moved rows included (every FK to matches, read from
pg_constraint so a new table is never missed), go to a write-once gzip JSONL
archive, and the copy is deleted, guarded: never while a prediction,
recommendation or betting_history row (or any FK that does not cascade)
references it. Refused and reported: a copy whose nearby row has no ESPN
identity (on prod: stale 'scheduled' fixture rows from before migration 023,
repair_fixture_orphans' business), is not finished, has another score, or is
ambiguous.

--shift. Every remaining football-data row on a BST date moves back 1 h, to
the kickoff promote_raw now writes. Refused and reported: a row with ESPN
identity, a row whose score is no longer its raw row's, and a row whose new
kickoff lands within 3 h of another row for the same team pair (a UNIQUE
(home_team_id, away_team_id, match_date) collision included; --dedupe
first). A shifted row that lands within 10 min of an ESPN row for a
DIFFERENT team pair in the same league, with the same score, is a probable
alias twin (one fixture under two team rows): shifted, and listed in
alias_twin_id for the team merge.

--check (read-only). Runs promote_raw.check_legacy_kickoffs and counts the
football-data rows that still have another FINISHED row for the pair where
promote_raw looks; exits 1 unless both are clear. Also counts, for
information, football-data rows beside an unfinished row (stale fixtures)
and probable alias twins.

Both writing modes plan first and write every planned row, refusals
included, to a write-once audit CSV, dry run included. --expect-planned and
--expect-refused pin the plan an operator reviewed: a plan that differs
writes nothing and exits 1. --apply writes the planned rows in bounded
batches, one transaction each, every UPDATE and DELETE guarded on the values
the plan observed; refused rows are left alone (exit 1 unless they were
expected). A batch whose guard misses a row rolls back and stops the run;
earlier batches stay committed and a re-run plans them as done.

Usage (inside the api container; scripts/ops/repair_football_data_kickoffs.sh
runs the whole sequence with a verified backup first):
    python /app/scripts/repair_football_data_kickoffs.py --dedupe            # dry run
    python /app/scripts/repair_football_data_kickoffs.py --dedupe --apply --expect-planned 68 --expect-refused 26
    python /app/scripts/repair_football_data_kickoffs.py --shift             # dry run
    python /app/scripts/repair_football_data_kickoffs.py --shift --apply --expect-planned 63718 --expect-refused 26
    python /app/scripts/repair_football_data_kickoffs.py --check
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import promote_raw as pr  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("repair_football_data_kickoffs")

# A shifted row may not land this close to another row for the same pair:
# promote_raw would then have to pick between them (find_existing_match).
NEAR_WINDOW = pr.EXISTING_MATCH_WINDOW
# A different team pair in the same league with the same score, kicking off this close to a
# football-data row (or 1 h before it, at the legacy kickoff): a probable alias twin.
ALIAS_TOLERANCE = timedelta(minutes=10)
LEGACY_OFFSET = timedelta(hours=1)
# Tables whose rows keep a copy from being deleted, on top of every FK to matches that does not cascade.
GUARD_TABLES = frozenset({"predictions", "betting_recommendations", "betting_history"})
DEFAULT_BATCH_SIZE = {"dedupe": 100, "shift": 5000}

RAW_SQL = (
    'SELECT league, season, "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG" '
    "FROM raw_football_data"
)
LEAGUES_SQL = (
    "SELECT id::text AS id, external_ids->>'football_data' AS code FROM leagues WHERE external_ids ? 'football_data'"
)
TEAMS_SQL = "SELECT id::text AS id, name, normalized_name FROM teams WHERE sport = 'soccer'"
MATCHES_SQL = """
    SELECT m.id::text AS id, m.league_id::text AS league_id,
           m.home_team_id::text AS home_id, m.away_team_id::text AS away_id,
           m.match_date, m.status, m.home_score, m.away_score, m.half_time_home, m.half_time_away,
           m.external_ids->>'espn' AS espn_id, m.metadata->>'result_source' AS result_source
      FROM matches m
      JOIN leagues l ON l.id = m.league_id
     WHERE l.sport = 'soccer'
"""
FK_SQL = """
    SELECT n.nspname AS schema_name, r.relname AS table_name, a.attname AS column_name,
           c.confdeltype AS on_delete, cardinality(c.conkey) AS n_columns
      FROM pg_constraint c
      JOIN pg_class r ON r.oid = c.conrelid
      JOIN pg_namespace n ON n.oid = r.relnamespace
      JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1]
     WHERE c.contype = 'f' AND c.confrelid = 'matches'::regclass
     ORDER BY r.relname, a.attname
"""
COUNT_REFS_SQL = "SELECT {col}::text AS match_id, count(*) AS n FROM {tbl} WHERE {col} = ANY(%s::uuid[]) GROUP BY 1"

# What a row looked like when planned; every dedupe write is guarded on it.
OBSERVED = (
    "match_date",
    "status",
    "home_score",
    "away_score",
    "half_time_home",
    "half_time_away",
    "espn_id",
    "result_source",
)
LOCK_SQL = """
    SELECT m.id::text AS id, m.match_date, m.status, m.home_score, m.away_score, m.half_time_home,
           m.half_time_away, m.external_ids->>'espn' AS espn_id, m.metadata->>'result_source' AS result_source
      FROM matches m
     WHERE m.id = ANY(%s::uuid[])
     ORDER BY m.id
       FOR UPDATE
"""
ARCHIVE_MATCHES_SQL = (
    "SELECT m.id::text AS match_id, row_to_json(m) AS data FROM matches m WHERE m.id = ANY(%s::uuid[])"
)
ARCHIVE_REFS_SQL = "SELECT {col}::text AS match_id, row_to_json(x) AS data FROM {tbl} x WHERE {col} = ANY(%s::uuid[])"

# The ESPN row has no row under the moving row's unique key (odds:
# uq_odds_current_price_key; match_stats: UNIQUE(match_id, team_id)).
ODDS_FREE_ON_TWIN = """NOT EXISTS (
        SELECT 1 FROM odds x
         WHERE x.match_id = v.twin_id::uuid AND x.bookmaker = o.bookmaker AND x.market_type = o.market_type
           AND x.selection = o.selection AND COALESCE(x.line, -1) = COALESCE(o.line, -1)
           AND COALESCE(x.is_opening, false) = COALESCE(o.is_opening, false)
           AND COALESCE(x.is_live, false) = COALESCE(o.is_live, false))"""
STATS_FREE_ON_TWIN = """NOT EXISTS (
        SELECT 1 FROM match_stats x WHERE x.match_id = v.twin_id::uuid AND x.team_id = s.team_id)"""

MOVE_COUNTS_SQL = f"""
    SELECT v.copy_id,
           (SELECT count(*) FROM odds o WHERE o.match_id = v.copy_id::uuid) AS copy_odds,
           (SELECT count(*) FROM odds o WHERE o.match_id = v.copy_id::uuid AND {ODDS_FREE_ON_TWIN}) AS odds_to_move,
           (SELECT count(*) FROM match_stats s WHERE s.match_id = v.copy_id::uuid) AS copy_match_stats,
           (SELECT count(*) FROM match_stats s WHERE s.match_id = v.copy_id::uuid AND {STATS_FREE_ON_TWIN})
               AS stats_to_move
      FROM (VALUES %s) AS v(copy_id, twin_id)
"""
MOVE_ODDS_SQL = f"""
    UPDATE odds o SET match_id = v.twin_id::uuid
      FROM (VALUES %s) AS v(copy_id, twin_id)
     WHERE o.match_id = v.copy_id::uuid AND {ODDS_FREE_ON_TWIN}
    RETURNING o.id::text
"""
MOVE_STATS_SQL = f"""
    UPDATE match_stats s SET match_id = v.twin_id::uuid
      FROM (VALUES %s) AS v(copy_id, twin_id)
     WHERE s.match_id = v.copy_id::uuid AND {STATS_FREE_ON_TWIN}
    RETURNING s.id::text
"""
COPY_HT_SQL = """
    UPDATE matches
       SET half_time_home = %(ht_home)s, half_time_away = %(ht_away)s
     WHERE id = %(id)s::uuid
       AND status = 'finished' AND home_score = %(home)s AND away_score = %(away)s
       AND half_time_home IS NULL AND half_time_away IS NULL
"""
# {guards}: one NOT EXISTS per guard table (delete_sql).
DELETE_COPY_SQL = """
    DELETE FROM matches m
     USING (VALUES %s) AS v(id, match_date, status, home_score, away_score)
     WHERE m.id = v.id
       AND m.match_date = v.match_date AND m.status = v.status
       AND m.home_score = v.home_score AND m.away_score = v.away_score
       AND m.external_ids->>'espn' IS NULL
       AND m.metadata->>'result_source' IS DISTINCT FROM 'espn_scoreboard'
       {guards}
    RETURNING m.id::text
"""
DELETE_TEMPLATE = "(%s::uuid, %s::timestamptz, %s::text, %s::int, %s::int)"

_NEAR = f"interval '{int(NEAR_WINDOW.total_seconds())} seconds'"
SHIFT_SQL = f"""
    UPDATE matches AS m
       SET match_date = v.new_kickoff
      FROM (VALUES %s) AS v(id, old_kickoff, new_kickoff, status, home_score, away_score)
     WHERE m.id = v.id
       AND m.match_date = v.old_kickoff
       AND m.status = v.status
       AND m.home_score IS NOT DISTINCT FROM v.home_score
       AND m.away_score IS NOT DISTINCT FROM v.away_score
       AND m.external_ids->>'espn' IS NULL
       AND m.metadata->>'result_source' IS DISTINCT FROM 'espn_scoreboard'
       AND NOT EXISTS (
           SELECT 1 FROM matches x
            WHERE x.home_team_id = m.home_team_id AND x.away_team_id = m.away_team_id AND x.id <> m.id
              AND x.match_date BETWEEN v.new_kickoff - {_NEAR} AND v.new_kickoff + {_NEAR})
    RETURNING m.id::text
"""
SHIFT_TEMPLATE = "(%s::uuid, %s::timestamptz, %s::timestamptz, %s::text, %s::int, %s::int)"

DEDUPE_FIELDS = (
    "copy_id",
    "espn_row_id",
    "other_row_ids",
    "league",
    "home",
    "away",
    "copy_kickoff",
    "espn_kickoff",
    "offset_min",
    "bst",
    "home_score",
    "away_score",
    "copy_ht",
    "espn_ht",
    "ht_action",
    "copy_odds",
    "odds_to_move",
    "copy_match_stats",
    "stats_to_move",
    "copy_refs",
    "action",
    "reason",
    "batch",
)
SHIFT_FIELDS = (
    "match_id",
    "league",
    "home",
    "away",
    "raw_date",
    "raw_time",
    "old_kickoff",
    "new_kickoff",
    "status",
    "home_score",
    "away_score",
    "action",
    "reason",
    "near_row_id",
    "alias_twin_id",
    "batch",
)


class PlanDrift(RuntimeError):
    """A batch's guard missed a row the plan observed: rolled back, run stopped."""


@dataclass
class FkTable:
    schema_name: str
    table_name: str
    column_name: str
    on_delete: str  # pg_constraint.confdeltype: c cascade, n set null, d set default, a no action, r restrict

    @property
    def ident(self) -> str:
        return f"{quote_ident(self.schema_name)}.{quote_ident(self.table_name)}"

    @property
    def column(self) -> str:
        return quote_ident(self.column_name)


def quote_ident(name: str) -> str:
    """A catalog name as a quoted SQL identifier (the standard quoting: no
    connection needed, so every statement can be built and tested offline)."""
    return '"' + name.replace('"', '""') + '"'


@dataclass
class Snapshot:
    """Everything the plans read, from one read snapshot. Each match dict
    gains 'kickoff' (aware UTC) and 'espn' (promote_raw.has_espn_identity)."""

    db_tz: tzinfo
    raw_rows: list[dict]
    league_ids: dict[str, str]  # football-data code -> leagues.id
    team_ids: dict[str, str]  # normalized_name -> teams.id
    team_names: dict[str, str]  # teams.id -> name
    matches: list[dict]
    by_key: dict[tuple, dict] = field(init=False)
    by_pair: dict[tuple, list[dict]] = field(init=False)

    def __post_init__(self) -> None:
        self.by_key, self.by_pair = {}, defaultdict(list)
        for m in self.matches:
            m["kickoff"] = m["match_date"].astimezone(timezone.utc)
            m["espn"] = pr.has_espn_identity(m.get("espn_id"), m.get("result_source"))
            self.by_key[(m["home_id"], m["away_id"], m["kickoff"])] = m
            self.by_pair[(m["home_id"], m["away_id"])].append(m)


@dataclass
class Link:
    """A matches row and the raw_football_data row(s) promote_raw wrote it from."""

    match: dict
    raw: dict
    legacy: datetime  # where promote_raw wrote it until 2026-10-31 (aware UTC)
    corrected: datetime  # where promote_raw writes it now (aware UTC)
    fixture_day: date  # football-data's Date (UK)
    raw_scores: set[tuple[int, int]]

    @property
    def at_legacy(self) -> bool:
        return self.match["kickoff"] == self.legacy

    @property
    def bst(self) -> bool:
        return self.legacy != self.corrected


def _iso(dt: Optional[datetime]) -> str:
    return dt.astimezone(timezone.utc).isoformat() if dt else ""


def observed(row: dict) -> tuple:
    return tuple(row.get(k) for k in OBSERVED)


def load_snapshot(cur) -> Snapshot:
    cur.execute("SHOW TimeZone")
    db_tz = ZoneInfo(cur.fetchone()["TimeZone"])  # loud if Postgres reports a zone Python cannot load
    cur.execute(RAW_SQL)
    raw_rows = [dict(r) for r in cur.fetchall()]
    cur.execute(LEAGUES_SQL)
    league_ids = {r["code"]: r["id"] for r in cur.fetchall()}
    cur.execute(TEAMS_SQL)
    teams = cur.fetchall()
    cur.execute(MATCHES_SQL)
    matches = [dict(r) for r in cur.fetchall()]
    return Snapshot(
        db_tz=db_tz,
        raw_rows=raw_rows,
        league_ids=league_ids,
        team_ids={t["normalized_name"]: t["id"] for t in teams},
        team_names={t["id"]: t["name"] for t in teams},
        matches=matches,
    )


def link_rows(snap: Snapshot, counts: Counter) -> dict[str, Link]:
    """{match_id: Link} for every timed raw row with a result whose matches
    row sits at its legacy kickoff or, once moved, at its corrected one. At
    the corrected kickoff only a row with the raw row's score counts: once a
    copy is merged away, the row there is the ESPN row it went into, or a
    stale fixture row, not football-data's. Pure."""
    links: dict[str, Link] = {}
    for r in snap.raw_rows:
        result = pr.raw_result(r)
        if result is None:
            counts["raw_no_result"] += 1  # never promoted
            continue
        league_id = snap.league_ids.get(r.get("league"))
        home_id = snap.team_ids.get(pr.normalize_team_name(r.get("HomeTeam") or ""))
        away_id = snap.team_ids.get(pr.normalize_team_name(r.get("AwayTeam") or ""))
        wall = pr.parse_wall_clock(r.get("Date") or "", r.get("Time"))
        if not (league_id and home_id and away_id and wall):
            counts["raw_unresolved"] += 1
            continue
        corrected = pr.parse_match_date(r.get("Date") or "", r.get("Time"))
        if corrected is None or corrected.tzinfo is None:
            counts["raw_date_only"] += 1  # its kickoff never changed
            continue
        legacy = wall.replace(tzinfo=snap.db_tz).astimezone(timezone.utc)
        score = (result.home_score, result.away_score)
        m = snap.by_key.get((home_id, away_id, legacy))
        if m is None and legacy != corrected:
            m = snap.by_key.get((home_id, away_id, corrected))
            if m is not None and (m["home_score"], m["away_score"]) != score:
                counts["raw_corrected_slot_other_row"] += 1
                continue
        if m is None:
            counts["raw_no_matches_row"] += 1
            continue
        if m["league_id"] != league_id:
            counts["raw_league_mismatch"] += 1
            continue
        if m["id"] in links:
            links[m["id"]].raw_scores.add(score)
            counts["raw_duplicate_rows"] += 1
            continue
        links[m["id"]] = Link(m, r, legacy, corrected, wall.date(), {score})
    counts["linked_rows"] = len(links)
    return links


def window_rows(snap: Snapshot, link: Link) -> list[dict]:
    """The other rows for this pair where promote_raw now looks for the
    fixture: promote_raw.existing_match_window around the corrected kickoff. Pure."""
    m = link.match
    lo, hi = pr.existing_match_window(link.corrected, link.fixture_day)
    return [x for x in snap.by_pair[(m["home_id"], m["away_id"])] if x is not m and lo <= x["kickoff"] <= hi]


def _base_entry(snap: Snapshot, link: Link) -> dict:
    m = link.match
    return {
        "league": link.raw.get("league"),
        "home": snap.team_names.get(m["home_id"], m["home_id"]),
        "away": snap.team_names.get(m["away_id"], m["away_id"]),
        "home_score": m["home_score"],
        "away_score": m["away_score"],
    }


# --- dedupe -------------------------------------------------------------------------------------------------------


def dedupe_candidates(snap: Snapshot, links: dict[str, Link], counts: Counter) -> list[tuple[Link, list[dict]]]:
    """(link, nearby rows) for each linked football-data row without ESPN
    identity that has another row for the pair in window_rows. Pure."""
    out = []
    for link in links.values():
        if link.match["espn"]:
            counts["linked_row_is_espn"] += 1  # the upsert landed on the ESPN row: not a copy
            continue
        nearby = window_rows(snap, link)
        if nearby:
            out.append((link, nearby))
    return out


def ht_plan(copy: dict, twin: dict) -> tuple[str, Optional[tuple[int, int]]]:
    """(ht_action, the pair to write onto the ESPN row or None). Pure."""
    copy_ht = (copy["half_time_home"], copy["half_time_away"])
    twin_ht = (twin["half_time_home"], twin["half_time_away"])
    if copy_ht == (None, None):
        return "none_on_copy", None
    if twin_ht != (None, None):
        return ("same" if twin_ht == copy_ht else "conflict_kept_espn"), None
    if None in copy_ht:
        return "incomplete_on_copy", None
    if copy_ht[0] > twin["home_score"] or copy_ht[1] > twin["away_score"]:
        return "exceeds_espn_final", None
    return "copy", copy_ht  # type: ignore[return-value]


def plan_dedupe(
    snap: Snapshot,
    candidates: list[tuple[Link, list[dict]]],
    refs: dict[str, dict[str, int]],
    guard_tables: Iterable[str],
    counts: Counter,
) -> list[dict]:
    """One entry per candidate copy: action 'merge' (into the row
    promote_raw.choose_existing picks among the nearby rows) or 'refuse'.
    `refs` is {copy_id: {table: rows referencing it}}. Pure."""
    guards = set(guard_tables)
    chosen = [pr.choose_existing(nearby) for _link, nearby in candidates]
    twin_uses = Counter(row["id"] for row, _why in chosen if row is not None)
    entries = []
    for (link, nearby), (twin, ambiguity) in zip(candidates, chosen):
        copy = link.match
        shown = [twin] if twin is not None else nearby
        copy_refs = {t: n for t, n in sorted(refs.get(copy["id"], {}).items()) if n}
        entry = {
            **_base_entry(snap, link),
            "copy_id": copy["id"],
            "espn_row_id": ";".join(x["id"] for x in shown),
            "other_row_ids": ";".join(x["id"] for x in nearby if x is not twin) if twin is not None else "",
            "copy_kickoff": _iso(copy["kickoff"]),
            "espn_kickoff": ";".join(_iso(x["kickoff"]) for x in shown),
            "offset_min": ";".join(str(int((copy["kickoff"] - x["kickoff"]).total_seconds() // 60)) for x in shown),
            "bst": link.bst,
            "copy_ht": f"{copy['half_time_home']}-{copy['half_time_away']}",
            "espn_ht": ";".join(f"{x['half_time_home']}-{x['half_time_away']}" for x in shown),
            "copy_refs": json.dumps(copy_refs, sort_keys=True),
            "copy": copy,
            "twin": twin,
            "action": "refuse",
            "reason": "",
            "ht_action": "",
            "ht_write": None,
        }
        blocking = sorted(t for t in copy_refs if t in guards)
        if twin is None:
            entry["reason"] = f"ambiguous: {ambiguity}"
        elif not twin["espn"]:
            entry["reason"] = f"the row for this pair nearby is a '{twin['status']}' row without ESPN identity" + (
                " (a stale fixture row: repair_fixture_orphans' business)"
                if twin["status"] != "finished"
                else " (two football-data rows? inspect)"
            )
        elif twin_uses[twin["id"]] > 1:
            entry["reason"] = f"the ESPN row is the nearby row of {twin_uses[twin['id']]} football-data rows"
        elif copy["status"] != "finished" or twin["status"] != "finished":
            entry["reason"] = f"not both finished (copy {copy['status']}, ESPN row {twin['status']})"
        elif None in (copy["home_score"], copy["away_score"]) or (copy["home_score"], copy["away_score"]) != (
            twin["home_score"],
            twin["away_score"],
        ):
            entry["reason"] = (
                f"scores differ (copy {copy['home_score']}-{copy['away_score']}, "
                f"ESPN row {twin['home_score']}-{twin['away_score']})"
            )
        elif blocking:
            entry["reason"] = "copy referenced by " + ", ".join(f"{t} ({copy_refs[t]})" for t in blocking)
        else:
            entry["action"] = "merge"
            entry["ht_action"], entry["ht_write"] = ht_plan(copy, twin)
        counts[f"dedupe_{entry['action']}"] += 1
        entries.append(entry)
    return entries


def load_fk_tables(cur) -> list[FkTable]:
    cur.execute(FK_SQL)
    rows = cur.fetchall()
    multi = [r for r in rows if r["n_columns"] != 1]
    if multi:
        raise RuntimeError(f"multi-column foreign keys to matches are not handled: {multi}")
    return [FkTable(r["schema_name"], r["table_name"], r["column_name"], r["on_delete"]) for r in rows]


def guard_table_names(fk_tables: list[FkTable]) -> set[str]:
    """Tables whose rows keep a copy alive: GUARD_TABLES, and any FK whose
    delete would not take the row with the copy."""
    return {t.table_name for t in fk_tables if t.table_name in GUARD_TABLES or t.on_delete != "c"}


def load_refs(cur, fk_tables: list[FkTable], match_ids: list[str]) -> dict[str, dict[str, int]]:
    refs: dict[str, dict[str, int]] = defaultdict(dict)
    if not match_ids:
        return refs
    for t in fk_tables:
        cur.execute(
            COUNT_REFS_SQL.format(col=t.column, tbl=t.ident),
            (match_ids,),
        )
        for r in cur.fetchall():
            refs[r["match_id"]][t.table_name] = refs[r["match_id"]].get(t.table_name, 0) + int(r["n"])
    return refs


def add_move_counts(cur, entries: list[dict]) -> None:
    merges = [e for e in entries if e["action"] == "merge"]
    if not merges:
        return
    rows = execute_values(
        cur, MOVE_COUNTS_SQL, [(e["copy_id"], e["twin"]["id"]) for e in merges], page_size=len(merges), fetch=True
    )
    by_copy = {r["copy_id"]: r for r in rows}
    for e in merges:
        for k in ("copy_odds", "odds_to_move", "copy_match_stats", "stats_to_move"):
            e[k] = int(by_copy[e["copy_id"]][k])


def delete_sql(guard_tables: list[FkTable]) -> str:
    """DELETE_COPY_SQL with one NOT EXISTS per guard table."""
    guards = "\n       ".join(
        f"AND NOT EXISTS (SELECT 1 FROM {t.ident} g WHERE g.{t.column} = m.id)" for t in guard_tables
    )
    return DELETE_COPY_SQL.format(guards=guards)


def write_archive(path: Path, records: list[dict]) -> None:
    """gzip JSONL, write-once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "xt", encoding="utf-8") as fh:  # 'x' refuses an existing file
        for rec in records:
            fh.write(json.dumps(rec, default=str, sort_keys=True) + "\n")


def apply_dedupe_batch(conn, batch: list[dict], batch_no: int, fk_tables: list[FkTable], archive_path: Path) -> Counter:
    """Merge one batch in one transaction: lock and re-check every row,
    archive, then write. Raises PlanDrift (rolled back) when a guard misses."""
    done: Counter = Counter()
    copy_ids = [e["copy_id"] for e in batch]
    twin_ids = [e["twin"]["id"] for e in batch]
    pairs = [(e["copy_id"], e["twin"]["id"]) for e in batch]
    guard_names = guard_table_names(fk_tables)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(LOCK_SQL, (copy_ids + twin_ids,))
            now = {r["id"]: r for r in cur.fetchall()}
            for e in batch:
                for role in ("copy", "twin"):
                    planned = e[role]
                    if planned["id"] not in now or observed(now[planned["id"]]) != observed(planned):
                        raise PlanDrift(f"batch {batch_no}: {role} {planned['id']} changed since the plan")

            records = []
            cur.execute(ARCHIVE_MATCHES_SQL, (copy_ids,))
            records += [{"table": "matches", "role": "copy", **r} for r in cur.fetchall()]
            cur.execute(ARCHIVE_MATCHES_SQL, (twin_ids,))
            records += [{"table": "matches", "role": "espn_row_before", **r} for r in cur.fetchall()]
            for t in fk_tables:
                cur.execute(
                    ARCHIVE_REFS_SQL.format(col=t.column, tbl=t.ident),
                    (copy_ids,),
                )
                records += [{"table": t.table_name, "role": "references_copy", **r} for r in cur.fetchall()]
            for rec in records:
                rec["batch"] = batch_no
            write_archive(archive_path, records)
            done["archived_records"] += len(records)

            for e in batch:
                if e["ht_write"] is None:
                    continue
                twin = e["twin"]
                cur.execute(
                    COPY_HT_SQL,
                    {
                        "id": twin["id"],
                        "ht_home": e["ht_write"][0],
                        "ht_away": e["ht_write"][1],
                        "home": twin["home_score"],
                        "away": twin["away_score"],
                    },
                )
                if cur.rowcount != 1:
                    raise PlanDrift(f"batch {batch_no}: half-time on ESPN row {twin['id']} changed since the plan")
                done["ht_copied"] += 1
            done["odds_moved"] += len(execute_values(cur, MOVE_ODDS_SQL, pairs, page_size=len(pairs), fetch=True))
            done["match_stats_moved"] += len(
                execute_values(cur, MOVE_STATS_SQL, pairs, page_size=len(pairs), fetch=True)
            )
            deleted = execute_values(
                cur,
                delete_sql([t for t in fk_tables if t.table_name in guard_names]),
                [
                    (e["copy_id"], e["copy"]["match_date"], e["copy"]["status"], e["home_score"], e["away_score"])
                    for e in batch
                ],
                template=DELETE_TEMPLATE,
                page_size=len(batch),
                fetch=True,
            )
            if len(deleted) != len(batch):
                missing = sorted(set(copy_ids) - {r["id"] for r in deleted})
                raise PlanDrift(f"batch {batch_no}: {len(missing)} copy row(s) not deleted by the guard: {missing[:5]}")
            done["copies_deleted"] += len(deleted)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return done


# --- shift --------------------------------------------------------------------------------------------------------


class _LeagueKickoffs:
    """Rows by league, sorted by kickoff, for the alias-twin lookup."""

    def __init__(self, matches: list[dict]):
        by_league: dict[str, list[dict]] = defaultdict(list)
        for m in matches:
            by_league[m["league_id"]].append(m)
        self._rows = {lg: sorted(rows, key=lambda m: m["kickoff"]) for lg, rows in by_league.items()}
        self._keys = {lg: [m["kickoff"] for m in rows] for lg, rows in self._rows.items()}

    def near(self, league_id: str, when: datetime, tolerance: timedelta) -> list[dict]:
        keys, rows = self._keys.get(league_id, []), self._rows.get(league_id, [])
        return rows[bisect.bisect_left(keys, when - tolerance) : bisect.bisect_right(keys, when + tolerance)]


def alias_twins(index: _LeagueKickoffs, row: dict, when: datetime) -> list[dict]:
    """ESPN rows in `row`'s league within ALIAS_TOLERANCE of `when`, same
    final score, another team pair: one fixture under two team rows. Pure."""
    if row["home_score"] is None:
        return []
    return [
        m
        for m in index.near(row["league_id"], when, ALIAS_TOLERANCE)
        if m["espn"]
        and (m["home_id"], m["away_id"]) != (row["home_id"], row["away_id"])
        and (m["home_score"], m["away_score"]) == (row["home_score"], row["away_score"])
    ]


def plan_shift(snap: Snapshot, links: dict[str, Link], counts: Counter) -> list[dict]:
    """One entry per linked BST row still at its legacy kickoff: action
    'shift' or 'refuse'. Pure."""
    index = _LeagueKickoffs(snap.matches)
    entries = []
    for link in links.values():
        m = link.match
        if not link.bst:
            counts["gmt_kickoff_right"] += 1
            continue
        if not link.at_legacy:
            # a moved football-data row, or the ESPN row a copy was merged into
            counts["corrected_slot_espn_row" if m["espn"] else "already_shifted"] += 1
            continue
        target = link.corrected
        near = [
            x
            for x in snap.by_pair[(m["home_id"], m["away_id"])]
            if x is not m and abs(x["kickoff"] - target) <= NEAR_WINDOW
        ]
        entry = {
            **_base_entry(snap, link),
            "match_id": m["id"],
            "raw_date": link.raw.get("Date"),
            "raw_time": link.raw.get("Time"),
            "old_kickoff": _iso(m["kickoff"]),
            "new_kickoff": _iso(target),
            "status": m["status"],
            "near_row_id": ";".join(x["id"] for x in near),
            "alias_twin_id": "",
            "old": m["kickoff"],
            "new": target,
            "action": "refuse",
            "reason": "",
        }
        if m["espn"]:
            entry["reason"] = "carries ESPN identity: its kickoff is ESPN's"
        elif (m["home_score"], m["away_score"]) not in link.raw_scores:
            entry["reason"] = (
                f"score {m['home_score']}-{m['away_score']} is not its raw row's {sorted(link.raw_scores)}: "
                "edited since"
            )
        elif near:
            exact = any(x["kickoff"] == target for x in near)
            entry["reason"] = (
                f"{len(near)} row(s) for this team pair within {NEAR_WINDOW} of the new kickoff"
                + (" (one on it: a UNIQUE collision)" if exact else "")
                + f": {', '.join(x['status'] for x in near)} — --dedupe first, or inspect"
            )
        else:
            entry["action"] = "shift"
            twins = alias_twins(index, m, target)
            if twins:
                entry["alias_twin_id"] = ";".join(t["id"] for t in twins)
                counts["shift_alias_twins"] += 1
        counts[f"shift_{entry['action']}"] += 1
        entries.append(entry)
    return entries


def apply_shift_batch(conn, batch: list[dict], batch_no: int) -> int:
    """Shift one batch in one transaction; raises PlanDrift (rolled back) when
    a guard misses."""
    try:
        with conn.cursor() as cur:
            moved = execute_values(
                cur,
                SHIFT_SQL,
                [(e["match_id"], e["old"], e["new"], e["status"], e["home_score"], e["away_score"]) for e in batch],
                template=SHIFT_TEMPLATE,
                page_size=len(batch),
                fetch=True,
            )
            if len(moved) != len(batch):
                missing = sorted({e["match_id"] for e in batch} - {r[0] for r in moved})
                raise PlanDrift(
                    f"batch {batch_no}: {len(missing)} of {len(batch)} row(s) changed since the plan or gained a "
                    f"same-pair neighbour: {missing[:5]}"
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(moved)


# --- check --------------------------------------------------------------------------------------------------------


def football_data_neighbours(snap: Snapshot, links: dict[str, Link]) -> tuple[list, list]:
    """([(football-data row, FINISHED row)], [(football-data row, unfinished
    row)]) for each linked row without ESPN identity and each other row for
    its pair in window_rows. The first list is the duplicates; the second is
    stale fixture rows beside a result. Pure."""
    finished, unfinished = [], []
    for link in links.values():
        if link.match["espn"]:
            continue
        for x in window_rows(snap, link):
            (finished if x["status"] == "finished" else unfinished).append((link.match, x))
    return finished, unfinished


def probable_alias_twins(snap: Snapshot, links: dict[str, Link]) -> list[tuple[dict, dict]]:
    """(football-data row, ESPN row) under another team pair, same league and
    score, kicking off with it or 1 h before it (+-10 min). Pure."""
    index = _LeagueKickoffs(snap.matches)
    out = []
    for link in links.values():
        m = link.match
        if m["espn"]:
            continue
        for when in (m["kickoff"], m["kickoff"] - LEGACY_OFFSET):
            out += [(m, t) for t in alias_twins(index, m, when)]
    return out


def run_check(conn) -> int:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        snap = load_snapshot(cur)
        legacy_error = None
        try:
            pr.check_legacy_kickoffs(cur, snap.raw_rows, snap.league_ids, snap.team_ids)
        except pr.LegacyKickoffsPresent as e:
            legacy_error = str(e)
    conn.rollback()
    counts: Counter = Counter()
    links = link_rows(snap, counts)
    duplicates, stale = football_data_neighbours(snap, links)
    aliases = probable_alias_twins(snap, links)
    if legacy_error:
        logger.error("check_legacy_kickoffs FAILS: %s", legacy_error)
    for m, x in duplicates[:10]:
        logger.error(
            "football-data row %s (%s) has a finished same-pair row %s at %s",
            m["id"],
            _iso(m["kickoff"]),
            x["id"],
            _iso(x["kickoff"]),
        )
    logger.info(
        "CHECK legacy_kickoffs=%s football_data_duplicates=%d | info: beside_unfinished_rows=%d "
        "probable_alias_twins=%d linked=%d",
        "FAIL" if legacy_error else "PASS",
        len(duplicates),
        len(stale),
        len(aliases),
        len(links),
    )
    return 1 if legacy_error or duplicates else 0


# --- run ----------------------------------------------------------------------------------------------------------


def default_path(kind: str, suffix: str, base: Optional[str] = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base = (base or os.environ.get("BACKUP_LOCAL_DIR", "/app/backups")).rstrip("/")
    return f"{base}/football_data_{kind}_{stamp}{suffix}"


def write_csv(path: str, fields: tuple[str, ...], entries: list[dict]) -> None:
    out = Path(path)
    if out.exists():
        raise FileExistsError(f"{out} exists — the audit CSV is write-once")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(entries)


def assign_batches(entries: list[dict], action: str, size: int) -> list[list[dict]]:
    todo = [e for e in entries if e["action"] == action]
    batches = [todo[i : i + size] for i in range(0, len(todo), size)]
    for n, batch in enumerate(batches, 1):
        for e in batch:
            e["batch"] = n
    return batches


def plan_mismatch(planned: int, refused: int, expect_planned: Optional[int], expect_refused: Optional[int]) -> str:
    """Why the plan is not the one the operator pinned, or ''."""
    wrong = []
    if expect_planned is not None and planned != expect_planned:
        wrong.append(f"{planned} planned, expected {expect_planned}")
    if expect_refused is not None and refused != expect_refused:
        wrong.append(f"{refused} refused, expected {expect_refused}")
    return "; ".join(wrong)


def _log_plan(
    mode: str, planned: list[dict], refused: list[dict], batches: list, counts: Counter, out_csv: str, pinned: bool
) -> None:
    """The plan's summary; refusals at ERROR unless `pinned` (the operator expected exactly these many)."""
    log_refusal = logger.warning if pinned else logger.error
    by_league = dict(sorted(Counter(e["league"] for e in planned).items()))
    if mode == "dedupe":
        logger.info(
            "Planned %d merge(s) in %d batch(es) %s: %d BST copies, offsets (copy - ESPN, min) %s, half-time to copy "
            "%d, odds to move %d of %d, match_stats to move %d of %d; audit CSV %s",
            len(planned),
            len(batches),
            by_league,
            sum(1 for e in planned if e["bst"]),
            dict(sorted(Counter(e["offset_min"] for e in planned).items())),
            sum(1 for e in planned if e["ht_write"]),
            sum(e.get("odds_to_move", 0) for e in planned),
            sum(e.get("copy_odds", 0) for e in planned),
            sum(e.get("stats_to_move", 0) for e in planned),
            sum(e.get("copy_match_stats", 0) for e in planned),
            out_csv,
        )
        for e in planned:
            if e["ht_action"] == "conflict_kept_espn":
                logger.error(
                    "%s %s v %s: half-time differs (copy %s, ESPN row %s); keeping the ESPN row's",
                    e["league"],
                    e["home"],
                    e["away"],
                    e["copy_ht"],
                    e["espn_ht"],
                )
    else:
        logger.info(
            "Planned %d shift(s) in %d batch(es) %s; %d land on a probable alias twin (alias_twin_id); audit CSV %s",
            len(planned),
            len(batches),
            by_league,
            counts["shift_alias_twins"],
            out_csv,
        )
    reasons = Counter(e["reason"].split(" (")[0].split(":")[0] for e in refused)
    for e in refused[:20]:
        log_refusal(
            "refused %s %s v %s %s: %s",
            e["league"],
            e["home"],
            e["away"],
            e.get("copy_id") or e.get("match_id"),
            e["reason"],
        )
    if refused:
        log_refusal("%d row(s) refused and left alone (all in the audit CSV): %s", len(refused), dict(reasons))


def run(
    conn,
    mode: str,
    apply_changes: bool,
    out_csv: Optional[str] = None,
    archive_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    expect_planned: Optional[int] = None,
    expect_refused: Optional[int] = None,
) -> Counter:
    counts: Counter = Counter()
    fk_tables: list[FkTable] = []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        snap = load_snapshot(cur)
        links = link_rows(snap, counts)
        if mode == "dedupe":
            fk_tables = load_fk_tables(cur)
            candidates = dedupe_candidates(snap, links, counts)
            refs = load_refs(cur, fk_tables, [link.match["id"] for link, _nearby in candidates])
            entries = plan_dedupe(snap, candidates, refs, guard_table_names(fk_tables), counts)
            add_move_counts(cur, entries)
        else:
            entries = plan_shift(snap, links, counts)
    conn.rollback()  # close the read snapshot before writing
    logger.info(
        "Loaded %d raw rows, %d soccer matches (db TimeZone %s); linked %d football-data rows",
        len(snap.raw_rows),
        len(snap.matches),
        snap.db_tz,
        len(links),
    )

    action = "merge" if mode == "dedupe" else "shift"
    batches = assign_batches(entries, action, batch_size or DEFAULT_BATCH_SIZE[mode])
    planned = [e for e in entries if e["action"] == action]
    refused = [e for e in entries if e["action"] == "refuse"]
    out_csv = out_csv or default_path(mode, ".csv", archive_dir)
    write_csv(out_csv, DEDUPE_FIELDS if mode == "dedupe" else SHIFT_FIELDS, entries)
    counts["planned"], counts["refused"] = len(planned), len(refused)
    _log_plan(mode, planned, refused, batches, counts, out_csv, pinned=expect_refused == len(refused))

    mismatch = plan_mismatch(len(planned), len(refused), expect_planned, expect_refused)
    if mismatch:
        counts["plan_mismatch"] = 1
        logger.error("The plan is not the one pinned (%s): nothing written; review the audit CSV", mismatch)
        return counts
    counts["refused_unexpected"] = 0 if expect_refused is not None else len(refused)
    if not apply_changes:
        logger.info("DRY-RUN: re-run with --apply to write %d row(s)", len(planned))
        return counts
    if mode == "dedupe":
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        base = (archive_dir or os.environ.get("BACKUP_LOCAL_DIR", "/app/backups")).rstrip("/")
        for n, batch in enumerate(batches, 1):
            archive = Path(f"{base}/football_data_dedupe_{stamp}_b{n:03d}.jsonl.gz")
            done = apply_dedupe_batch(conn, batch, n, fk_tables, archive)
            counts.update(done)
            logger.info("Batch %d/%d committed: %s; archive %s", n, len(batches), dict(done), archive)
    else:
        for n, batch in enumerate(batches, 1):
            counts["shifted"] += apply_shift_batch(conn, batch, n)
            logger.info("Batch %d/%d committed: %d shifted so far", n, len(batches), counts["shifted"])
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dedupe", action="store_const", dest="mode", const="dedupe", help="Merge copies.")
    mode.add_argument("--shift", action="store_const", dest="mode", const="shift", help="Move BST kickoffs.")
    mode.add_argument("--check", action="store_const", dest="mode", const="check", help="Read-only after-state.")
    p.add_argument("--apply", action="store_true", help="Write (default: dry run).")
    p.add_argument("--expect-planned", type=int, default=None, help="Write nothing unless exactly this many planned.")
    p.add_argument("--expect-refused", type=int, default=None, help="Write nothing unless exactly this many refused.")
    p.add_argument("--out-csv", default=None, help="Audit CSV (default $BACKUP_LOCAL_DIR or /app/backups).")
    p.add_argument("--archive-dir", default=None, help="Archive/CSV dir (default $BACKUP_LOCAL_DIR or /app/backups).")
    p.add_argument("--batch-size", type=int, default=None, help="Rows per transaction (dedupe 100, shift 5000).")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = p.parse_args(argv)
    if args.mode == "check" and (args.apply or args.expect_planned is not None or args.expect_refused is not None):
        p.error("--check is read-only and plans nothing")
    if args.batch_size is not None and args.batch_size < 1:
        p.error("--batch-size must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    with psycopg2.connect(args.database_url) as conn:
        if args.mode == "check":
            return run_check(conn)
        try:
            counts = run(
                conn,
                args.mode,
                args.apply,
                args.out_csv,
                args.archive_dir,
                args.batch_size,
                args.expect_planned,
                args.expect_refused,
            )
        except PlanDrift as e:
            logger.error("Stopped, batch rolled back (earlier batches are committed; a re-run re-plans): %s", e)
            return 1
    logger.info("Done: %s", dict(sorted(counts.items())))
    return 1 if counts["plan_mismatch"] or counts["refused_unexpected"] else 0


if __name__ == "__main__":
    sys.exit(main())
