"""Move reviewed soccer match rows from one team row to another.

NOT YET APPROVED — the apply needs explicit operator approval and a verified
backup first (scripts/ops/repoint_team_matches.sh takes it).

Why: ESPN calls two clubs by one display name, and the old exact-name lookup
put both on one row. Santos Laguna (mex.1, ESPN 225) arrives as 'Santos', so
from 2026-07-19 its Liga MX fixtures landed on 'Santos', Santos FC's row
(Brasileirao history since 2012): 20 rows. ESPN's 'Guadalajara' (Chivas, ESPN
219) exact-matched the 'Guadalajara' row, which holds CD Guadalajara's Segunda
Division history (84 matches, 2011-13): 19 rows; Chivas' own history is on
'Guadalajara Chivas'. Each club's rolling form mixes the other's results, and
once fetch_upcoming resolves 225 / 219 through the seed (scripts/data/
soccer_team_espn_seed.csv) every ESPN-identified row still on the wrong team
raises ForkRefused. Run this right after that code deploys.

The map (scripts/data/repoint_team_matches.csv, reviewed: one row per match
side, each with the ESPN event that proves it) names the match, the side, the
team it is on, the team it moves to, and what the review saw (league, UTC
kickoff to the minute, the opponent). Matches keep their ids, so predictions,
odds, recommendations and features_cache stay attached; match_stats rows of
the moved side move with it. The seeded ESPN team ids of every team the map
names are stamped too (guarded like team_identity.stamp_espn_id).

A team the review named may since have been merged into another row by
scripts/merge_soccer_teams.py (the ESPN spelling 'América' folded into
'Club America'): a side or an opponent counts as that team when the row it
sits on is the named team, or a team whose metadata.aliases
(team_identity.alias_form) or metadata.merged_from names it. So a side the
first pass moved still plans as 'done' after the merge renamed its opponent,
and a genuine drift (another opponent) is still refused.

Refused (written to the CSV, counted, never written):
  * a row that is not what the review saw (moved, re-dated, another opponent);
  * a target team that does not exist, or a side that would play itself;
  * a move that would collide on UNIQUE(home_team_id, away_team_id,
    match_date), or leave a same-pair twin within 36 h — unless the twin moves
    with it from the same pair (a stale row and its finished copy, already
    twins before the move);
  * a match_stats row of the target team already on that match;
  * every side of a match when one side is refused (no half-moved match).
A refused team stamp (another row holds the id, or the row holds another id)
stops the whole apply.

Safety: dry run by default; the write-once audit CSV lands before any write
(dry run included); --apply plans again inside one transaction after locking
the rows (SELECT ... FOR UPDATE), and every UPDATE is guarded on the observed
values with a rowcount check — any mismatch rolls the whole thing back.
--expect-repoint / --expect-refused / --expect-done pin the reviewed plan: a
different plan writes nothing. Re-runnable: a moved side plans as 'done'. One
fixed-format line 'PLAN repoint=N done=N refuse=N not_found=N stamp_...=N' is
logged for scripts/ops/repoint_team_matches.sh.

TWO PASSES (scripts/ops/soccer_repair_runbook.sh runs both, each through
scripts/ops/repoint_team_matches.sh --pass N with its own backup and pins):
  pass 1 (runbook step 1, right after the deploy): 33 sides move, 6 refused
    (a football-data copy sits on the target pair 1 h later);
  pass 2 (runbook step 5, after merge_soccer_teams and
    repair_football_data_kickoffs alias-merged those copies): the 6 move,
    32 read 'done' (13 of them through a merged opponent's alias), 1 is
    refused as 'match row not found' (cb4e21fc: the team merge folded that
    stale fork into its played twin 45064ae5, itself 'done').

Usage (inside the api container; the wrapper is the supported way to apply):
    python /app/scripts/repoint_team_matches.py                      # dry run
    python /app/scripts/repoint_team_matches.py --apply --expect-repoint 33 --expect-done 0 --expect-refused 6
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import team_identity  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("repoint_team_matches")

MAP_PATH = Path(__file__).resolve().parent / "data" / "repoint_team_matches.csv"
MAP_FIELDS = ("match_id", "side", "from_team", "to_team", "league", "kickoff_utc", "opponent", "espn_event", "evidence")
TWIN_WINDOW = timedelta(hours=36)
KICKOFF_FORMAT = "%Y-%m-%d %H:%M"


@dataclass(frozen=True)
class MapRow:
    match_id: str
    side: str  # 'home' | 'away'
    from_team: str
    to_team: str
    league: str
    kickoff_utc: str  # 'YYYY-MM-DD HH:MM', UTC
    opponent: str
    espn_event: str
    evidence: str


def _norm(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def load_map(path: Path | str = MAP_PATH) -> list[MapRow]:
    """The reviewed map. Raises ValueError on a malformed one: a missing
    column, a side that is not home/away, a bad kickoff, a team moved onto
    itself, or one match side listed twice."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [f for f in MAP_FIELDS if f not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        rows: list[MapRow] = []
        seen: set[tuple[str, str]] = set()
        for line, raw in enumerate(reader, start=2):
            row = MapRow(*(str(raw.get(f) or "").strip() for f in MAP_FIELDS))
            if row.side not in ("home", "away"):
                raise ValueError(f"{path}:{line}: side must be home or away, not {row.side!r}")
            datetime.strptime(row.kickoff_utc, KICKOFF_FORMAT)  # ValueError on a bad kickoff
            if not row.match_id or not row.from_team or not row.to_team or _norm(row.from_team) == _norm(row.to_team):
                raise ValueError(f"{path}:{line}: match_id, from_team and a different to_team are required")
            if (row.match_id, row.side) in seen:
                raise ValueError(f"{path}:{line}: {row.match_id} {row.side} is listed twice")
            seen.add((row.match_id, row.side))
            rows.append(row)
    return rows


SELECT_MATCHES_SQL = """
    SELECT m.id::text AS id, l.sport, l.name AS league, m.league_id::text AS league_id,
           m.home_team_id::text AS home_id, m.away_team_id::text AS away_id,
           th.name AS home, ta.name AS away, m.match_date, m.status, m.external_ids->>'espn' AS espn
      FROM matches m
      JOIN leagues l ON l.id = m.league_id
      JOIN teams th ON th.id = m.home_team_id
      JOIN teams ta ON ta.id = m.away_team_id
     WHERE m.id = ANY(%s::uuid[])
"""
LOCK_MATCHES_SQL = "SELECT id FROM matches WHERE id = ANY(%s::uuid[]) ORDER BY id FOR UPDATE"
# The named teams, plus every team carrying a merge record: a named team may
# have been merged into one of those (resolve_team).
SELECT_TEAMS_SQL = """
    SELECT id::text AS id, name, normalized_name, external_ids->>'espn' AS espn, metadata
      FROM teams
     WHERE sport = 'soccer'
       AND (normalized_name = ANY(%s) OR metadata ? 'aliases' OR metadata ? 'merged_from')
"""
ESPN_HOLDERS_SQL = """
    SELECT id::text AS id, external_ids->>'espn' AS espn FROM teams
     WHERE sport = 'soccer' AND external_ids->>'espn' = ANY(%s)
"""
NEIGHBOURS_SQL = """
    SELECT m.id::text AS id, m.home_team_id::text AS home_id, m.away_team_id::text AS away_id,
           m.match_date, m.status, m.external_ids->>'espn' AS espn
      FROM matches m
     WHERE (m.home_team_id = ANY(%(teams)s::uuid[]) OR m.away_team_id = ANY(%(teams)s::uuid[]))
       AND m.match_date BETWEEN %(lo)s AND %(hi)s
"""
MATCH_STATS_SQL = (
    "SELECT match_id::text AS match_id, team_id::text AS team_id FROM match_stats WHERE match_id = ANY(%s::uuid[])"
)
MONEY_SQL = """
    SELECT m.id::text AS id,
           (SELECT count(*) FROM predictions p WHERE p.match_id = m.id) AS predictions,
           (SELECT count(*) FROM betting_recommendations r WHERE r.match_id = m.id) AS recs
      FROM matches m WHERE m.id = ANY(%s::uuid[])
"""
# Guarded on every observed value of the row; rowcount 0 means it moved.
UPDATE_MATCH_SQL = """
    UPDATE matches
       SET home_team_id = %(new_home)s::uuid, away_team_id = %(new_away)s::uuid, updated_at = NOW()
     WHERE id = %(id)s::uuid
       AND home_team_id = %(old_home)s::uuid AND away_team_id = %(old_away)s::uuid
       AND league_id = %(league_id)s::uuid AND match_date = %(match_date)s
"""
UPDATE_STATS_SQL = """
    UPDATE match_stats SET team_id = %(to)s::uuid
     WHERE match_id = %(match_id)s::uuid AND team_id = %(from)s::uuid
"""
AFFECTED_MATCHES_SQL = """
    SELECT id::text AS id FROM matches
     WHERE (home_team_id = ANY(%(teams)s::uuid[]) OR away_team_id = ANY(%(teams)s::uuid[]))
       AND match_date >= %(since)s AND status IN ('scheduled', 'finished')
     ORDER BY match_date
"""

CSV_FIELDS = (
    "kind",
    "match_id",
    "side",
    "from_team",
    "to_team",
    "league",
    "kickoff_utc",
    "opponent",
    "espn_event",
    "action",
    "reason",
    "status",
    "espn_competition",
    "home_before",
    "away_before",
    "home_after",
    "away_after",
    "match_stats_moved",
    "predictions",
    "recs",
    "team_id",
    "espn_team_id",
)


def _kickoff(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(KICKOFF_FORMAT)


def merged_names(team: Optional[dict]) -> set[str]:
    """alias_form of every name a team row answers to besides its own: its
    metadata.aliases (an array of strings, team_identity's contract; any
    other value is ignored, as team_identity ignores it) and the names in its
    metadata.merged_from records (merge_soccer_teams writes both). Pure."""
    meta = (team or {}).get("metadata") or {}
    out: set[str] = set()
    aliases = meta.get("aliases")
    if isinstance(aliases, list) and all(isinstance(a, str) for a in aliases):
        out |= {team_identity.alias_form(a) for a in aliases}
    merged = meta.get("merged_from")
    if isinstance(merged, list):
        out |= {team_identity.alias_form(e["name"]) for e in merged if isinstance(e, dict) and e.get("name")}
    out.discard("")
    return out


def resolve_team(teams: dict[str, dict], name: str) -> Optional[dict]:
    """The team row a reviewed name means now: the row with that normalized
    name, else the ONE row that took it over in a team merge (merged_names);
    None when none does or two claim it. Pure."""
    exact = teams.get(_norm(name))
    if exact is not None:
        return exact
    form = team_identity.alias_form(name)
    claims = [t for t in teams.values() if form in merged_names(t)]
    return claims[0] if len(claims) == 1 else None


def answers_to(team_id: str, team_name: str, wanted: str, by_id: dict[str, dict]) -> bool:
    """True when the team on a side (its id and name) is the team the review
    named `wanted`: the same name, or a team that took `wanted` over in a
    merge (its row is in `by_id`, the teams read_state loaded). Pure."""
    if _norm(team_name) == _norm(wanted):
        return True
    return team_identity.alias_form(wanted) in merged_names(by_id.get(team_id))


@dataclass
class Plan:
    entries: list[dict]  # one per map row
    stamps: list[dict]  # one per seeded team the map names
    updates: dict[str, dict]  # match_id -> UPDATE_MATCH_SQL params
    stats_moves: list[dict]  # UPDATE_STATS_SQL params, each moving exactly one row
    counts: Counter


def plan_repoints(  # noqa: C901 — one rule per refusal, in order
    rows: list[MapRow],
    matches: dict[str, dict],
    teams: dict[str, dict],
    neighbours: list[dict],
    stats: list[dict],
    seed: dict[str, team_identity.SeedEntry],
    espn_holders: dict[str, list[str]],
    money: Optional[dict[str, dict]] = None,
) -> Plan:
    """The plan for every map row. Pure: `matches` by id (SELECT_MATCHES_SQL),
    `teams` by normalized name (SELECT_TEAMS_SQL: the named teams and every
    team with a merge record), `neighbours` every match of the teams involved
    near the kickoffs (NEIGHBOURS_SQL), `stats` the match_stats rows of the
    mapped matches, `espn_holders` {espn id: [team ids holding it]}. A name
    the review used counts through a team merge (resolve_team, answers_to)."""
    money = money or {}
    by_id = {t["id"]: t for t in teams.values()}
    by_match: dict[str, list[MapRow]] = defaultdict(list)
    for row in rows:
        by_match[row.match_id].append(row)
    verdict: dict[tuple[str, str], tuple[str, str]] = {}
    post: dict[str, tuple[str, str]] = {}

    for match_id, sides in by_match.items():
        m = matches.get(match_id)
        if m is None:
            for row in sides:
                verdict[(match_id, row.side)] = ("refuse", "match row not found")
            continue
        new_home, new_away = m["home_id"], m["away_id"]
        moved_to = {row.side: row.to_team for row in sides}
        for row in sides:
            key = (match_id, row.side)
            on_side = m["home"] if row.side == "home" else m["away"]
            on_side_id = m["home_id"] if row.side == "home" else m["away_id"]
            other = m["away"] if row.side == "home" else m["home"]
            other_id = m["away_id"] if row.side == "home" else m["home_id"]
            other_side = "away" if row.side == "home" else "home"
            opp_ok = answers_to(other_id, other, row.opponent, by_id) or answers_to(
                other_id, other, moved_to.get(other_side, row.opponent), by_id
            )
            target = resolve_team(teams, row.to_team)
            if m.get("sport") != "soccer" or m["league"] != row.league or _kickoff(m["match_date"]) != row.kickoff_utc:
                verdict[key] = (
                    "refuse",
                    f"not the reviewed fixture: {m.get('sport')} {m['league']!r} {_kickoff(m['match_date'])} "
                    f"!= soccer {row.league!r} {row.kickoff_utc}",
                )
            elif target is None:
                verdict[key] = ("refuse", f"target team {row.to_team!r} does not exist")
            elif on_side_id == target["id"] and opp_ok:
                verdict[key] = ("done", "already on the target team")
            elif not answers_to(on_side_id, on_side, row.from_team, by_id) or not opp_ok:
                verdict[key] = (
                    "refuse",
                    f"drifted since the review: {row.side} is {on_side!r} v {other!r} "
                    f"(expected {row.from_team!r} v {row.opponent!r})",
                )
            else:
                verdict[key] = ("repoint", "")
                if row.side == "home":
                    new_home = target["id"]
                else:
                    new_away = target["id"]
        if new_home == new_away:
            for row in sides:
                verdict[(match_id, row.side)] = ("refuse", "both sides would be one team")
        post[match_id] = (new_home, new_away)

    def refuse_match(match_id: str, reason: str) -> bool:
        changed = False
        for row in by_match[match_id]:
            if verdict[(match_id, row.side)][0] == "repoint":
                verdict[(match_id, row.side)] = ("refuse", reason)
                changed = True
        return changed

    for match_id, sides in by_match.items():  # no half-moved match
        if any(verdict[(match_id, r.side)][0] == "refuse" for r in sides):
            refuse_match(match_id, "another side of this match is refused")

    # match_stats: the target team must not already have a row on the match.
    stats_by_match: dict[str, set[str]] = defaultdict(set)
    for s in stats:
        stats_by_match[s["match_id"]].add(s["team_id"])
    for match_id, sides in by_match.items():
        m = matches.get(match_id)
        for row in sides:
            if m and verdict[(match_id, row.side)][0] == "repoint":
                target = resolve_team(teams, row.to_team)["id"]  # type: ignore[index]  # planned: it resolved
                if target in stats_by_match.get(match_id, set()):
                    refuse_match(match_id, f"match_stats already has a row for {row.to_team!r} on this match")

    # Twins on the post-move pair, to a fixed point: a twin that moves with the
    # row from the same pair is allowed only while it is not refused itself.
    near = {n["id"]: n for n in neighbours}
    for match_id, m in matches.items():
        near.setdefault(match_id, {**m, "id": match_id})
    while True:
        changed = False
        moving = {mid for mid in by_match if any(verdict[(mid, r.side)][0] == "repoint" for r in by_match[mid])}
        for match_id in sorted(moving):
            m = matches[match_id]
            pair = post[match_id]
            for t in near.values():
                if t["id"] == match_id:
                    continue
                t_pair = post[t["id"]] if t["id"] in moving else (t["home_id"], t["away_id"])
                gap = abs(t["match_date"] - m["match_date"])
                if t_pair != pair or gap >= TWIN_WINDOW:
                    continue
                same_origin = t["id"] in moving and (t["home_id"], t["away_id"]) == (m["home_id"], m["away_id"])
                if gap == timedelta(0):
                    reason = f"would collide on UNIQUE(home, away, match_date) with {t['id']}"
                elif same_origin:
                    continue  # already twins on the old pair; they move together
                else:
                    reason = (
                        f"would leave a same-pair twin within 36 h: {t['id']} at {_kickoff(t['match_date'])} "
                        f"({t.get('status')}, ESPN {t.get('espn') or '-'})"
                    )
                changed |= refuse_match(match_id, reason)
                break
        for match_id in sorted(moving):  # a twin that moves with a refused row stays too
            m = matches[match_id]
            for other in by_match:  # every mapped match, moving or not
                if other == match_id or other not in matches:
                    continue
                o = matches[other]
                if (o["home_id"], o["away_id"]) == (m["home_id"], m["away_id"]) and abs(
                    o["match_date"] - m["match_date"]
                ) < TWIN_WINDOW:
                    if not any(verdict[(other, r.side)][0] == "repoint" for r in by_match[other]):
                        changed |= refuse_match(match_id, f"its twin {other} on the same pair is refused")
        if not changed:
            break

    entries: list[dict] = []
    updates: dict[str, dict] = {}
    stats_moves: list[dict] = []
    for row in rows:
        action, reason = verdict[(row.match_id, row.side)]
        m = matches.get(row.match_id) or {}
        moves = action == "repoint"
        stats_moved = 0
        if moves:
            target = resolve_team(teams, row.to_team)["id"]  # type: ignore[index]  # planned: it resolved
            source = m["home_id"] if row.side == "home" else m["away_id"]
            stats_moved = 1 if source in stats_by_match.get(row.match_id, set()) else 0
            if stats_moved:
                stats_moves.append({"match_id": row.match_id, "from": source, "to": target})
            updates.setdefault(
                row.match_id,
                {
                    "id": row.match_id,
                    "old_home": m["home_id"],
                    "old_away": m["away_id"],
                    "league_id": m["league_id"],
                    "match_date": m["match_date"],
                },
            )
        entries.append(
            {
                "kind": "match",
                **{f: getattr(row, f) for f in MAP_FIELDS if f != "evidence"},
                "action": action,
                "reason": reason,
                "status": m.get("status"),
                "espn_competition": m.get("espn"),
                "home_before": m.get("home"),
                "away_before": m.get("away"),
                "home_after": target_name(teams, row.to_team) if moves and row.side == "home" else m.get("home"),
                "away_after": target_name(teams, row.to_team) if moves and row.side == "away" else m.get("away"),
                "match_stats_moved": stats_moved,
                "predictions": (money.get(row.match_id) or {}).get("predictions"),
                "recs": (money.get(row.match_id) or {}).get("recs"),
            }
        )
    # A both-sides match carries the moves of both entries in one UPDATE.
    for match_id, params in updates.items():
        moved_sides = {r.side for r in by_match[match_id] if verdict[(match_id, r.side)][0] == "repoint"}
        params["new_home"] = post[match_id][0] if "home" in moved_sides else params["old_home"]
        params["new_away"] = post[match_id][1] if "away" in moved_sides else params["old_away"]

    stamps = []
    named = sorted({_norm(r.from_team) for r in rows} | {_norm(r.to_team) for r in rows})
    for espn_id, entry in sorted(seed.items()):
        if entry.normalized_name not in named:
            continue
        team = teams.get(entry.normalized_name)
        holders = espn_holders.get(espn_id, [])
        if team is None:
            action, reason = "refuse", f"seeded team {entry.team_name!r} does not exist"
        elif team.get("espn") == espn_id and holders == [team["id"]]:
            action, reason = "done", "already stamped"
        elif team.get("espn"):
            action, reason = "refuse", f"{entry.team_name!r} carries ESPN id {team['espn']}"
        elif holders:
            action, reason = "refuse", f"ESPN id {espn_id} is on team(s) {', '.join(holders)}"
        else:
            action, reason = "stamp", ""
        stamps.append(
            {
                "kind": "team",
                "to_team": entry.team_name,
                "team_id": (team or {}).get("id"),
                "espn_team_id": espn_id,
                "action": action,
                "reason": reason,
            }
        )
    counts = Counter(e["action"] for e in entries)
    counts["not_found"] = sum(1 for e in entries if e["action"] == "refuse" and e["reason"] == "match row not found")
    counts.update(f"stamp_{s['action']}" for s in stamps)
    return Plan(entries, stamps, updates, stats_moves, counts)


def target_name(teams: dict[str, dict], name: str) -> str:
    """The name of the row a reviewed target name resolves to (the name itself
    when it resolves to none). Pure."""
    t = resolve_team(teams, name)
    return t["name"] if t else name


def plan_line(counts: Counter) -> str:
    """The fixed-format line scripts/ops/repoint_team_matches.sh parses."""
    keys = ("repoint", "done", "refuse", "not_found", "stamp_stamp", "stamp_done", "stamp_refuse")
    return "PLAN " + " ".join(f"{k}={counts.get(k, 0)}" for k in keys)


def read_state(cur, rows: list[MapRow], seed: dict) -> tuple:
    """Everything plan_repoints needs, read through `cur`."""
    ids = sorted({r.match_id for r in rows})
    cur.execute(SELECT_MATCHES_SQL, (ids,))
    matches = {r["id"]: dict(r) for r in cur.fetchall()}
    names = sorted(
        {_norm(r.from_team) for r in rows} | {_norm(r.to_team) for r in rows} | {_norm(r.opponent) for r in rows}
    )
    cur.execute(SELECT_TEAMS_SQL, (names,))
    teams = {r["normalized_name"]: dict(r) for r in cur.fetchall()}
    team_ids = sorted(
        {t["id"] for t in teams.values()} | {m[k] for m in matches.values() for k in ("home_id", "away_id")}
    )
    neighbours: list[dict] = []
    if matches:
        dates = [m["match_date"] for m in matches.values()]
        cur.execute(NEIGHBOURS_SQL, {"teams": team_ids, "lo": min(dates) - TWIN_WINDOW, "hi": max(dates) + TWIN_WINDOW})
        neighbours = [dict(r) for r in cur.fetchall()]
    cur.execute(MATCH_STATS_SQL, (ids,))
    stats = [dict(r) for r in cur.fetchall()]
    cur.execute(ESPN_HOLDERS_SQL, (sorted(seed),))
    holders: dict[str, list[str]] = defaultdict(list)
    for r in cur.fetchall():
        holders[r["espn"]].append(r["id"])
    cur.execute(MONEY_SQL, (ids,))
    money = {r["id"]: dict(r) for r in cur.fetchall()}
    return matches, teams, neighbours, stats, dict(holders), money


def default_out_csv() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return f"{os.environ.get('BACKUP_LOCAL_DIR', '/app/backups')}/repoint_team_matches_{stamp}.csv"


def write_csv(path: str, plan: Plan) -> None:
    out = Path(path)
    if out.exists():
        raise FileExistsError(f"{out} exists — the audit CSV is write-once")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(plan.entries + plan.stamps)


def pins_ok(counts: Counter, expect: dict[str, Optional[int]]) -> bool:
    """True when every given pin equals the plan's count (logged otherwise)."""
    ok = True
    for action, want in expect.items():
        if want is not None and counts[action] != want:
            logger.error("Pinned %d %s, planned %d — writing nothing", want, action, counts[action])
            ok = False
    return ok


def apply_plan(cur, plan: Plan) -> int:
    """Write the plan through `cur` (caller owns the transaction). Raises
    RuntimeError on any guard miss; returns the number of rows written."""
    written = 0
    for match_id, params in sorted(plan.updates.items()):
        cur.execute(UPDATE_MATCH_SQL, params)
        if cur.rowcount != 1:
            raise RuntimeError(f"match {match_id} changed between plan and write (rowcount {cur.rowcount})")
        written += 1
    for move in plan.stats_moves:
        cur.execute(UPDATE_STATS_SQL, move)
        if cur.rowcount != 1:
            raise RuntimeError(f"match_stats of {move['match_id']} changed between plan and write ({cur.rowcount})")
        written += 1
    for s in plan.stamps:
        if s["action"] != "stamp":
            continue
        cur.execute(team_identity.STAMP_ESPN_SQL, {"id": s["team_id"], "espn": s["espn_team_id"]})
        if cur.rowcount != 1:
            raise RuntimeError(f"ESPN id {s['espn_team_id']} could not be stamped on {s['to_team']!r}")
        written += 1
    return written


def features_command(cur, plan: Plan, teams: dict[str, dict]) -> Optional[str]:
    """The compute_features command for every match of the teams involved
    from the earliest moved kickoff on (their rolling form changed). Printed,
    never run."""
    moved = [e for e in plan.entries if e["action"] in ("repoint", "done")]
    if not moved:
        return None
    resolved = (resolve_team(teams, e[k]) for e in moved for k in ("from_team", "to_team"))
    team_ids = sorted({t["id"] for t in resolved if t is not None})
    since = min(datetime.strptime(e["kickoff_utc"], KICKOFF_FORMAT).replace(tzinfo=timezone.utc) for e in moved)
    cur.execute(AFFECTED_MATCHES_SQL, {"teams": team_ids, "since": since})
    ids = [r["id"] for r in cur.fetchall()]
    if not ids:
        return None
    return "docker compose exec -T api python /app/scripts/compute_features.py --match-ids " + ",".join(ids)


def run(conn, rows: list[MapRow], seed: dict, apply_changes: bool, out_csv: str, expect: dict) -> Counter:
    from psycopg2.extras import RealDictCursor

    ids = sorted({r.match_id for r in rows})
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if apply_changes:
            cur.execute(LOCK_MATCHES_SQL, (ids,))
        matches, teams, neighbours, stats, holders, money = read_state(cur, rows, seed)
        plan = plan_repoints(rows, matches, teams, neighbours, stats, seed, holders, money)
        write_csv(out_csv, plan)  # before any write, dry run included
        for e in plan.entries + plan.stamps:
            log = logger.error if e["action"] == "refuse" else logger.info
            log(
                "%-7s %s %s %r -> %r %s: %s",
                e["action"],
                e.get("match_id") or "team",
                e.get("side") or "",
                e.get("from_team") or "",
                e["to_team"],
                e.get("kickoff_utc") or e.get("espn_team_id"),
                e["reason"],
            )
        counts = plan.counts
        logger.info("Planned %s; audit CSV %s", dict(sorted(counts.items())), out_csv)
        logger.info("%s", plan_line(counts))
        command = features_command(cur, plan, teams)
        if not pins_ok(counts, expect):
            conn.rollback()
            counts["pin_mismatch"] += 1
            return counts
        if counts["stamp_refuse"]:
            logger.error("A seeded ESPN id cannot be stamped — writing nothing; repair the team rows first")
            conn.rollback()
            return counts
        if not apply_changes:
            conn.rollback()
            logger.info("DRY-RUN: re-run with --apply to move %d side(s)", counts["repoint"])
            if command:
                logger.info("After the apply, refresh features (not run here):\n%s", command)
            return counts
        try:
            counts["written"] = apply_plan(cur, plan)
        except Exception:
            conn.rollback()
            logger.exception("Rolled back — nothing written")
            raise
    conn.commit()
    logger.info("Moved %d match row(s); %d write(s) in all", len(plan.updates), counts["written"])
    if command:
        logger.info("Refresh features for the teams involved (not run here):\n%s", command)
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="Write (default: dry run).")
    p.add_argument("--map", default=str(MAP_PATH), help="The reviewed map CSV.")
    p.add_argument("--seed", default=None, help="The ESPN seed CSV (default: team_identity.SEED_PATH).")
    p.add_argument("--out-csv", default=None, help="Audit CSV (default $BACKUP_LOCAL_DIR or /app/backups).")
    p.add_argument("--expect-repoint", type=int, help="Pin: sides to move.")
    p.add_argument("--expect-refused", type=int, help="Pin: sides refused.")
    p.add_argument("--expect-done", type=int, help="Pin: sides already moved.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    try:
        rows = load_map(args.map)
        seed = team_identity.load_espn_seed(args.seed)
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    import psycopg2

    expect = {"repoint": args.expect_repoint, "refuse": args.expect_refused, "done": args.expect_done}
    with psycopg2.connect(args.database_url) as conn:
        counts = run(conn, rows, seed, args.apply, args.out_csv or default_out_csv(), expect)
    logger.info("Done: %s", dict(sorted(counts.items())))
    pinned_refusals = expect["refuse"] is not None and counts["refuse"] == expect["refuse"]
    if counts["pin_mismatch"] or counts["stamp_refuse"] or (counts["refuse"] and not pinned_refusals):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
