"""Record ESPN's result on the soccer rows that were never resulted, then grade them, pinned.

THE ROWS
--------
repair_fixture_orphans.py (scripts/ops/cleanup_soccer_orphans.sh) leaves
alone, and reports as `never_resulted`, the stale 'scheduled' soccer rows
(kickoff more than 96 h ago) that have no finished twin and fit no other
class: real matches whose result never reached the row, mostly from before
the results path shipped (2026-07-13). The runbook's replica after its steps
1-11 (prod export 2026-11-01 02:42Z) holds 42: Chile 18, J1 14, Copa
Libertadores 4, and one each in Belgium, Colombia, Greece, La Liga, Liga MX
and the Champions League; 21 of them carry recs, all already 'void' with
their P&L set, and ~4,500 ungraded predictions. A blanket results sweep
(fetch_upcoming.py --results --days-back 160) would also re-write every
other completed event of those leagues and days, healing, realigning and
stamping rows nobody reviewed, and grade_completed_matches --days 160 would
grade every finished match of every sport in the window. This touches the
listed rows only, and scripts/ops/backfill_soccer_never_resulted.sh (runbook
step 12) lists only the rows reviewed for healing
(scripts/data/soccer_never_resulted_reviewed.csv: the rehearsal's 38, plus
Guadalajara v Atlante of 2026-11-01, reviewed ahead because it joins the
class if the runbook starts late): a row that joins the class later and is
not on that list is listed by the wrapper and left alone, for its own
review.

WHAT IT DOES
------------
For each listed row (--match-ids-file, e.g. the never_resulted ids of a
repair_fixture_orphans dry run) that is still a stale 'scheduled' soccer row,
the scoreboard of its league (fetch_upcoming's soccer slugs) is read for its
UTC kickoff day and the days either side, and each competition is matched to
a listed row only: by the row's ESPN id, else by both sides resolving
(fetch_upcoming.find_team, which never writes) to the row's two teams with
ESPN's kickoff within 72 h of the row's (fetch_upcoming's healing window).
A row is HEALED only when exactly one competition matches it and ESPN says
it is completed, with a final score its scoring plays agree with. Anything
else is REFUSED, with the reason, and left as it is: not found, ambiguous,
not completed, ESPN postponed / cancelled / abandoned (left for review: a
blanket sweep would record that status), no final score, plays that
disagree, a failed ESPN request, no ESPN slug for the league, or a row that
is no longer stale and scheduled. A listed row already finished counts as
DONE.

The heal is fetch_upcoming._record_result on that competition — the
pipeline's own results path, unchanged: team resolution (which may stamp a
side's ESPN team id), the row's kickoff moved to ESPN's, the final score,
half-time, the 90-minute score of a tie decided after 90 minutes, the ESPN
id stamped. It is checked right after, inside a savepoint: the listed row
must now be finished with ESPN's final score, no matches or teams row
inserted, no other matches row touched and nothing counted as an identity
refusal; otherwise the savepoint is rolled back and the row is refused
('the results path did not write it as planned'). The status flip fires
the outcome trigger (migration 027: pre-kickoff predictions graded, open
pre-kickoff recs marked); then grade_completed_matches.grade_match grades
the rest of the row, exactly as grade_matches.py does (every market it
knows, recs settled with P&L, recs written after kickoff voided).

All of it is ONE transaction. The dry run does all of it and rolls back, so
its PLAN line is the apply's:
  PLAN candidates= heal= done= refuse= predictions_graded= predictions_regraded=
       recs_changed= teams_stamped= espn_failed=
--apply needs --expect-plan with every key (the reviewed dry run's counts)
and commits only when the run's own plan equals it; otherwise it rolls back
and exits 1.

ESPN once: --espn-cache PATH. When PATH does not exist the scoreboards are
fetched and written there (write-once JSON); when it exists they are read
from it and ESPN is not called, and a day it does not hold refuses the run.
The wrapper's apply reuses its dry run's cache, so the apply writes exactly
the results that dry run planned. A day whose request failed is recorded as
failed: its rows are refused, and espn_failed counts it.

Records, all write-once, all written before the commit (dry run included):
  --out-csv            one row per listed row: action, reason, the ESPN
                       competition, its final, the row's status / score /
                       kickoff after, and what was graded on it;
  <out-csv>.changes.csv  every prediction and rec that changed, old and new
                       values (grade_matches.CSV_FIELDS): the reversal
                       record for the grading;
  <out-csv>.archive.jsonl.gz  the healed rows' matches, predictions and
                       recommendations rows as they were before.

Usage (inside the api container):
    python /app/scripts/backfill_never_resulted.py --match-ids-file /app/backups/nr_ids.txt \\
        --espn-cache /app/backups/nr_espn.json --out-csv /app/backups/nr_dry.csv
    python /app/scripts/backfill_never_resulted.py --match-ids-file /app/backups/nr_ids.txt \\
        --espn-cache /app/backups/nr_espn.json --apply --expect-plan candidates=42,heal=40,... \\
        --out-csv /app/backups/nr_apply.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_upcoming as fu  # noqa: E402
import grade_matches as gm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_never_resulted")

SOCCER = fu.SPORT_CONFIGS["soccer"]
# repair_fixture_orphans' --stale-after-hours default: past the results sweep's 3-day look-back.
STALE_HOURS = 96
MATCH_WINDOW = timedelta(hours=fu.IDENTITY_WINDOW_HOURS)

HEAL, DONE, REFUSE = "heal", "done", "refuse"
PLAN_KEYS = (
    "candidates",
    "heal",
    "done",
    "refuse",
    "predictions_graded",
    "predictions_regraded",
    "recs_changed",
    "teams_stamped",
    "espn_failed",
)

ROWS_SQL = """
    SELECT m.id::text AS id, m.league_id::text AS league_id, l.name AS league_name, l.country, l.sport,
           m.match_date, m.home_team_id::text AS home_id, m.away_team_id::text AS away_id,
           th.name AS home, ta.name AS away, m.status, m.home_score, m.away_score,
           m.external_ids->>'espn' AS espn_id
      FROM matches m
      JOIN leagues l ON l.id = m.league_id
      JOIN teams th ON th.id = m.home_team_id
      JOIN teams ta ON ta.id = m.away_team_id
     WHERE m.id = ANY(%s::uuid[])
"""
# The planned rows, locked and re-read before the first write: a row that moved since the plan is refused.
LOCK_SQL = """
    SELECT id::text AS id, status, match_date, home_team_id::text AS home_id, away_team_id::text AS away_id
      FROM matches WHERE id = ANY(%s::uuid[]) ORDER BY id FOR UPDATE
"""
ARCHIVE_SQL = {
    "matches": "SELECT * FROM matches WHERE id = ANY(%s::uuid[]) ORDER BY id",
    "predictions": "SELECT * FROM predictions WHERE match_id = ANY(%s::uuid[]) ORDER BY id",
    "betting_recommendations": "SELECT * FROM betting_recommendations WHERE match_id = ANY(%s::uuid[]) ORDER BY id",
}
TEAMS_SQL = "SELECT id::text AS id, external_ids::text AS external_ids FROM teams WHERE id = ANY(%s::uuid[])"
ROW_AFTER_SQL = """
    SELECT status, home_score, away_score, match_date, external_ids->>'espn' AS espn_id
      FROM matches WHERE id = %s::uuid
"""
CREATED_SQL = (
    "SELECT (SELECT count(*) FROM matches WHERE created_at = now()) AS matches, "
    "(SELECT count(*) FROM teams WHERE created_at = now()) AS teams"
)
TOUCHED_SQL = "SELECT id::text AS id FROM matches WHERE updated_at = now()"
ESPN_OWNER_SQL = """
    SELECT m.id::text AS id FROM matches m JOIN leagues l ON l.id = m.league_id
     WHERE l.sport = 'soccer' AND m.external_ids->>'espn' = %s AND m.id <> %s::uuid
     ORDER BY m.id LIMIT 1
"""

CSV_FIELDS = (
    "match_id",
    "league",
    "kickoff",
    "home",
    "away",
    "status_before",
    "action",
    "reason",
    "espn_id",
    "espn_kickoff",
    "espn_final",
    "espn_detail",
    "status_after",
    "score_after",
    "kickoff_after",
    "predictions_graded",
    "recs_changed",
)


class Refused(RuntimeError):
    """The run cannot go ahead as asked; nothing is written."""


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def league_slugs() -> dict[tuple[str, str], str]:
    """(league name, country) -> ESPN slug, from fetch_upcoming's soccer config."""
    return {(name, country): slug for slug, (_code, name, country) in SOCCER.leagues.items()}


def scoreboard_days(kickoffs) -> list[date]:
    """Each kickoff's UTC day and the days either side (ESPN files a late-UTC kickoff under the US date), sorted."""
    days = {_utc(k).date() for k in kickoffs}
    return sorted({d + timedelta(days=o) for d in days for o in (-1, 0, 1)})


class EspnCache:
    """Scoreboards by (slug, day). Replay mode (the file exists): read only, a missing day raises Refused. Fetch mode:
    fu.fetch_day, each result recorded (a failed request as None), saved write-once by save()."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.replay = self.path.exists()
        self.days: dict[str, Optional[list]] = {}
        if self.replay:
            try:
                self.days = json.loads(self.path.read_text())["days"]
            except (OSError, ValueError, KeyError) as e:
                raise Refused(f"cannot read the ESPN cache {self.path}: {e}") from None

    @staticmethod
    def key(slug: str, day: date) -> str:
        return f"{slug}|{day:%Y%m%d}"

    def events(self, slug: str, day: date) -> Optional[list]:
        """The day's events, or None when its request failed."""
        k = self.key(slug, day)
        if self.replay:
            if k not in self.days:
                raise Refused(f"the ESPN cache {self.path} does not hold {k} — run a fresh dry run (a new cache)")
            return self.days[k]
        if k not in self.days:
            failed = fu.FETCH_STATS["failed"]
            got = fu.fetch_day(SOCCER, slug, day)
            self.days[k] = None if fu.FETCH_STATS["failed"] > failed else got
        return self.days[k]

    def save(self) -> None:
        if self.replay:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fh = self.path.open("x")
        except FileExistsError:
            raise Refused(f"{self.path} appeared while fetching — the ESPN cache is write-once") from None
        with fh:
            json.dump({"fetched_at": datetime.now(timezone.utc).isoformat(), "days": self.days}, fh)


def entry_for(row: dict, slug: str = "", action: str = REFUSE, reason: str = "") -> dict:
    return {
        "match_id": row["id"],
        "league": slug or f"{row['league_name']} ({row['country']})",
        "kickoff": _utc(row["match_date"]).isoformat(),
        "home": row["home"],
        "away": row["away"],
        "status_before": row["status"],
        "action": action,
        "reason": reason,
        "espn_id": row.get("espn_id") or "",
    }


def screen(row: dict, as_of: datetime) -> Optional[tuple[str, str]]:
    """(action, reason) for a listed row that is not a candidate for ESPN, else None. Pure."""
    if row["sport"] != "soccer":
        return REFUSE, f"not a soccer row ({row['sport']})"
    if row["status"] == "finished" and row["home_score"] is not None and row["away_score"] is not None:
        return DONE, "already finished"
    if row["status"] != "scheduled":
        return REFUSE, f"status '{row['status']}', not a stale scheduled row"
    if _utc(row["match_date"]) > as_of - timedelta(hours=STALE_HOURS):
        return REFUSE, f"kickoff within {STALE_HOURS} h of {as_of.isoformat()}: the results sweep may still heal it"
    return None


def assess(row: dict, comp: dict, parts: tuple) -> tuple[str, str, Optional[tuple[int, int]], str]:
    """(action, reason, final score, ESPN detail) for a row and the one competition matched to it. Pure."""
    home, away, _hn, _an, match_dt, state, _venue, _espn = parts
    status_type = comp.get("status", {}).get("type", {})
    detail = (status_type.get("detail") or status_type.get("shortDetail") or "").upper()
    if abs(_utc(match_dt) - _utc(row["match_date"])) > MATCH_WINDOW:
        return REFUSE, f"ESPN's kickoff {match_dt.isoformat()} is more than {MATCH_WINDOW} from the row's", None, detail
    terminal = fu.espn_terminal_status(comp)
    if terminal is not None:
        return REFUSE, f"ESPN status {terminal[1]} (left for review)", None, detail
    if state != "post" or not status_type.get("completed", False):
        return REFUSE, f"ESPN: not a completed competition ({state}, {status_type.get('name')})", None, detail
    final = fu._final_scores(SOCCER, home, away)
    if final is None:
        return REFUSE, "ESPN: no final score", None, detail
    tally = fu.scoring_play_tally(comp, home, away)
    if tally is not None and tally != (final[0], final[1], 0):
        return (
            REFUSE,
            f"ESPN header {final[0]}-{final[1]} but its scoring plays add up to {tally[0]}-{tally[1]} "
            f"(+{tally[2]} unattributed)",
            None,
            detail,
        )
    return HEAL, "", final, detail


def match_competitions(cur, rows: list[dict], slug: str, events: list[dict]) -> dict[str, dict[str, tuple]]:
    """{row id: {ESPN competition id: (comp, context, parts)}} for the listed rows of one league: by the row's ESPN id,
    else both sides resolving to the row's teams with ESPN's kickoff within MATCH_WINDOW. Only listed rows can
    match, so no other row is ever touched."""
    by_espn = {r["espn_id"]: r for r in rows if r.get("espn_id")}
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_pair[(r["home_id"], r["away_id"])].append(r)
    league_id = rows[0]["league_id"]
    found: dict[str, dict[str, tuple]] = defaultdict(dict)
    for event in events:
        for comp, context in fu._iter_event_competitions(event, SOCCER):
            parts = fu._competition_parts(comp, SOCCER)
            if parts is None:
                continue
            home, away, home_name, away_name, match_dt, _state, _venue, espn_id = parts
            key = espn_id or f"{home_name}|{away_name}|{match_dt.isoformat()}"
            if espn_id and espn_id in by_espn:
                found[by_espn[espn_id]["id"]][key] = (comp, context, parts)
                continue
            home_id = fu.find_team(cur, SOCCER, home_name, league_id, espn_id=fu._competitor_espn_id(home, False))
            away_id = fu.find_team(cur, SOCCER, away_name, league_id, espn_id=fu._competitor_espn_id(away, False))
            if not (home_id and away_id):
                continue
            for r in by_pair.get((str(home_id), str(away_id)), []):
                if abs(_utc(match_dt) - _utc(r["match_date"])) <= MATCH_WINDOW:
                    found[r["id"]][key] = (comp, context, parts)
    return found


def fork_reasons(rows: list[dict], found: dict[str, dict[str, tuple]]) -> dict[str, str]:
    """{row id: why it is refused} for listed rows that share one ESPN competition: a fixture forked before
    2026-09-04 (a kickoff change made a second row). The row that carries the competition's ESPN id, else the one
    at ESPN's exact kickoff, owns it and may be healed; the others are refused as its forks (repair_fixture_orphans
    merges them once their twin is finished). With no single owner, every one of them is refused as ambiguous. Pure."""
    by_comp: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for key in found.get(r["id"], {}):
            by_comp[key].append(r)
    out: dict[str, str] = {}
    for key, sharing in by_comp.items():
        if len(sharing) < 2:
            continue
        espn_id, kickoff = found[sharing[0]["id"]][key][2][7], _utc(found[sharing[0]["id"]][key][2][4])
        owners = [r for r in sharing if espn_id and r.get("espn_id") == espn_id]
        owners = owners or [r for r in sharing if _utc(r["match_date"]) == kickoff]
        for r in sharing:
            if len(owners) == 1 and r is not owners[0]:
                out[r["id"]] = (
                    f"a fork of row {owners[0]['id']}, which owns ESPN competition {key}: merge it after that row "
                    "is finished (scripts/ops/cleanup_soccer_orphans.sh)"
                )
            elif len(owners) != 1:
                out[r["id"]] = f"ambiguous: ESPN competition {key} matches {len(sharing)} listed rows"
    return out


def espn_owner(cur, espn_id: Optional[str], row_id: str) -> Optional[str]:
    """Another soccer row already carrying this ESPN competition id, or None: the results path would write onto that
    row (fetch_upcoming resolves by ESPN id first), never onto this one."""
    if not espn_id:
        return None
    cur.execute(ESPN_OWNER_SQL, (espn_id, row_id))
    r = cur.fetchone()
    return r["id"] if r else None


def candidates_by_league(rows: list[dict], as_of: datetime) -> dict[tuple[str, str], list[dict]]:
    """The listed rows screen() lets through, by (league name, country). Pure."""
    by_league: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if screen(r, as_of) is None:
            by_league[(r["league_name"], r["country"])].append(r)
    return dict(by_league)


def prefetch(rows: list[dict], as_of: datetime, cache: EspnCache) -> None:
    """Every scoreboard plan() will read, fetched (or checked present in a replayed cache) before any transaction."""
    slugs = league_slugs()
    for (league_name, country), group in sorted(candidates_by_league(rows, as_of).items()):
        slug = slugs.get((league_name, country))
        if slug is not None:
            for day in scoreboard_days(r["match_date"] for r in group):
                cache.events(slug, day)


def plan(cur, rows: list[dict], as_of: datetime, cache: EspnCache, counts: Counter) -> list[dict]:
    """One entry per listed row: heal (with its competition), done or refuse."""
    slugs = league_slugs()
    entries: dict[str, dict] = {}
    for r in rows:
        screened = screen(r, as_of)
        if screened is not None:
            entries[r["id"]] = entry_for(r, action=screened[0], reason=screened[1])
    for (league_name, country), group in sorted(candidates_by_league(rows, as_of).items()):
        slug = slugs.get((league_name, country))
        if slug is None:
            for r in group:
                entries[r["id"]] = entry_for(r, reason=f"no ESPN slug for {league_name} ({country})")
            continue
        events, failed = [], set()
        for day in scoreboard_days(r["match_date"] for r in group):
            got = cache.events(slug, day)
            if got is None:
                failed.add(day)
                counts["espn_failed"] += 1
            else:
                events.extend(got)
        pair_index = fu.build_pair_index(SOCCER, events)
        found = match_competitions(cur, group, slug, events)
        forks = fork_reasons(group, found)
        for r in group:
            e = entry_for(r, slug)
            entries[r["id"]] = e
            hit = found.get(r["id"], {})
            near_failed = {d for d in scoreboard_days([r["match_date"]]) if d in failed}
            if near_failed:
                e["reason"] = f"ESPN request failed for {slug} {', '.join(str(d) for d in sorted(near_failed))}"
                continue
            if not hit:
                e["reason"] = f"not found on ESPN's {slug} scoreboard"
                continue
            if r["id"] in forks:
                e["reason"] = forks[r["id"]]
                continue
            if len(hit) > 1:
                e["reason"] = f"ambiguous: {len(hit)} ESPN competitions match ({', '.join(sorted(hit))})"
                continue
            ((comp, context, parts),) = hit.values()
            owner = espn_owner(cur, parts[7], r["id"])
            if owner:
                e["reason"] = (
                    f"ESPN competition {parts[7]} is already on row {owner}: this row is a fork of it "
                    "(scripts/ops/cleanup_soccer_orphans.sh merges it once that row is finished)"
                )
                continue
            action, reason, final, detail = assess(r, comp, parts)
            e.update(
                action=action,
                reason=reason,
                espn_id=parts[7] or "",
                espn_kickoff=_utc(parts[4]).isoformat(),
                espn_final=f"{final[0]}-{final[1]}" if final else "",
                espn_detail=detail,
            )
            if action == HEAL:
                e["_comp"], e["_context"] = comp, {**context, "pair_index": pair_index, "league_slug": slug}
                e["_final"], e["_league_id"] = final, r["league_id"]
    return [entries[r["id"]] for r in rows]


def heal_one(cur, e: dict, healed: set[str]) -> str:
    """Write one planned heal inside a savepoint and check it; '' when it stands, else why it was rolled back."""
    mid = e["match_id"]
    cur.execute("SAVEPOINT heal_one")
    stats_before = dict(fu.IDENTITY_STATS)
    try:
        written = fu._record_result(cur, SOCCER, e["_league_id"], e["_comp"], e["_context"])
    except psycopg2.Error as exc:
        cur.execute("ROLLBACK TO SAVEPOINT heal_one")
        return f"the results path raised {type(exc).__name__}: {str(exc).strip()[:200]}"
    cur.execute(ROW_AFTER_SQL, (mid,))
    row = cur.fetchone()
    cur.execute(CREATED_SQL)
    created = cur.fetchone()
    cur.execute(TOUCHED_SQL)
    touched = {r["id"] for r in cur.fetchall()} - healed - {mid}
    problems = []
    if not written:
        problems.append("it wrote nothing")
    if row is None or row["status"] != "finished" or (row["home_score"], row["away_score"]) != tuple(e["_final"]):
        problems.append(
            f"the row is {row['status'] if row else 'gone'} {row and row['home_score']}-"
            f"{row and row['away_score']}, not finished {e['espn_final']}"
        )
    if created["matches"] or created["teams"]:
        problems.append(f"it inserted {created['matches']} matches / {created['teams']} teams row(s)")
    if touched:
        problems.append(f"it touched other matches rows {sorted(touched)[:3]}")
    if dict(fu.IDENTITY_STATS) != stats_before:
        delta = {k: v - stats_before.get(k, 0) for k, v in fu.IDENTITY_STATS.items() if v != stats_before.get(k, 0)}
        problems.append(f"identity refusals {delta}")
    if problems:
        cur.execute("ROLLBACK TO SAVEPOINT heal_one")
        fu.IDENTITY_STATS.update(stats_before)
        return "the results path did not write it as planned: " + "; ".join(problems)
    gm.grade_listed(cur, gm.finished_matches(cur, [mid]))
    cur.execute("RELEASE SAVEPOINT heal_one")
    e.update(
        status_after=row["status"],
        score_after=f"{row['home_score']}-{row['away_score']}",
        kickoff_after=_utc(row["match_date"]).isoformat(),
    )
    return ""


def _jsonable(v):
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return str(v) if not isinstance(v, (int, float, bool, str, type(None), dict, list)) else v


def write_archive(cur, path: str, ids: list[str]) -> int:
    """The healed rows and their predictions and recs as they are now, gzip JSONL, write-once."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        fh = gzip.open(out, "xt")
    except FileExistsError:
        raise Refused(f"{out} exists — the archive is write-once") from None
    with fh:
        for table, sql in ARCHIVE_SQL.items():
            cur.execute(sql, (ids,))
            for r in cur.fetchall():
                fh.write(json.dumps({"table": table, **{k: _jsonable(v) for k, v in r.items()}}) + "\n")
                n += 1
    return n


def write_csv(path: str, entries: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = out.open("x", newline="")
    except FileExistsError:
        raise Refused(f"{out} exists — the audit CSV is write-once") from None
    with fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(entries)


def plan_line(counts: Counter) -> str:
    return "PLAN " + " ".join(f"{k}={counts[k]}" for k in PLAN_KEYS)


def run(
    conn,
    ids: list[str],
    as_of: datetime,
    cache_path: str,
    apply_changes: bool,
    expected: Optional[dict[str, int]],
    out_csv: str,
) -> Counter:
    counts: Counter = Counter({k: 0 for k in PLAN_KEYS})
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(ROWS_SQL, (ids,))
        rows = {r["id"]: dict(r) for r in cur.fetchall()}
    conn.rollback()  # no transaction stays open while ESPN answers
    missing = [i for i in ids if i not in rows]
    if missing:
        raise Refused(f"{len(missing)} listed id(s) are not matches rows: {', '.join(missing[:5])}")
    ordered = [rows[i] for i in ids]
    counts["candidates"] = len(ordered)
    cache = EspnCache(cache_path)
    logger.info(
        "%d listed row(s); ESPN %s %s", len(ordered), "replayed from" if cache.replay else "fetched into", cache_path
    )
    prefetch(ordered, as_of, cache)
    cache.save()

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        entries = plan(cur, ordered, as_of, cache, counts)
        heal_ids = [e["match_id"] for e in entries if e["action"] == HEAL]
        changes: list[dict] = []
        if heal_ids:
            cur.execute(LOCK_SQL, (heal_ids,))
            locked = {r["id"]: r for r in cur.fetchall()}
            for e in entries:
                if e["action"] != HEAL:
                    continue
                now_row, was = locked.get(e["match_id"]), rows[e["match_id"]]
                if now_row is None or (
                    now_row["status"],
                    now_row["match_date"],
                    now_row["home_id"],
                    now_row["away_id"],
                ) != (
                    was["status"],
                    was["match_date"],
                    was["home_id"],
                    was["away_id"],
                ):
                    e["action"], e["reason"] = REFUSE, "the row changed since it was read"
            heal_ids = [e["match_id"] for e in entries if e["action"] == HEAL]
        if heal_ids:
            team_ids = sorted({rows[i][k] for i in heal_ids for k in ("home_id", "away_id")})
            cur.execute(TEAMS_SQL, (team_ids,))
            teams_before = {r["id"]: r["external_ids"] for r in cur.fetchall()}
            write_archive(cur, f"{out_csv}.archive.jsonl.gz", heal_ids)
            before = gm.snapshot(cur, heal_ids)
            healed: set[str] = set()
            for e in entries:
                if e["action"] != HEAL:
                    continue
                why = heal_one(cur, e, healed)
                if why:
                    e["action"], e["reason"] = REFUSE, why
                else:
                    healed.add(e["match_id"])
            changes = gm.diff(before, gm.snapshot(cur, heal_ids))
            cur.execute(TEAMS_SQL, (team_ids,))
            counts["teams_stamped"] = sum(1 for r in cur.fetchall() if r["external_ids"] != teams_before.get(r["id"]))
        per_match: Counter = Counter()
        for c in changes:
            per_match[(c["match_id"], c["kind"])] += 1
            counts[
                (
                    "predictions_graded"
                    if c["kind"] == "graded"
                    else "predictions_regraded" if c["kind"] == "regraded" else "recs_changed"
                )
            ] += 1
            if c["kind"] == "rec":
                counts[f"rec:{c['old_status']}->{c['new_status']}"] += 1
        for e in entries:
            counts[e["action"]] += 1
            e["predictions_graded"] = per_match[(e["match_id"], "graded")]
            e["recs_changed"] = per_match[(e["match_id"], "rec")]
    write_csv(out_csv, entries)
    gm.write_csv(f"{out_csv}.changes.csv", changes)
    for e in entries:
        (logger.info if e["action"] != REFUSE else logger.warning)(
            "%s %s %s v %s (%s): %s%s%s",
            e["league"],
            e["kickoff"][:16],
            e["home"],
            e["away"],
            e["match_id"],
            e["action"],
            f" {e.get('espn_final')} (ESPN {e.get('espn_id')}, {e.get('espn_detail')})" if e["action"] == HEAL else "",
            f" — {e['reason']}" if e["reason"] else "",
        )
    reasons = Counter(e["reason"].split(":")[0].split(" (")[0] for e in entries if e["action"] == REFUSE)
    for reason, n in sorted(reasons.items()):
        logger.info("refused %d: %s", n, reason)
    for k in sorted(k for k in counts if k.startswith("rec:")):
        logger.info("recs %s: %d", k[4:], counts[k])
    logger.info("%s (audit CSV %s)", plan_line(counts), out_csv)
    if not apply_changes:
        conn.rollback()
        logger.info("DRY-RUN: rolled back, nothing written")
        return counts
    drift = {k: (expected[k], counts[k]) for k in PLAN_KEYS if expected is not None and expected[k] != counts[k]}
    if expected is None or drift:
        conn.rollback()
        raise Refused(f"the plan drifted from --expect-plan (expected, now): {drift} — rolled back, nothing written")
    conn.commit()
    logger.info("Committed: %s", plan_line(counts)[5:])
    return counts


def parse_expect_plan(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in (text or "").split(","):
        key, sep, val = part.strip().partition("=")
        if not sep or key not in PLAN_KEYS or not val.isdigit() or key in out:
            raise Refused(f"--expect-plan: bad entry {part!r} (keys: {', '.join(PLAN_KEYS)})")
        out[key] = int(val)
    missing = [k for k in PLAN_KEYS if k not in out]
    if missing:
        raise Refused(f"--expect-plan must pin every key; missing {', '.join(missing)}")
    return out


def parse_as_of(text: Optional[str]) -> datetime:
    if not text:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise Refused(f"--as-of {text!r} is not an ISO timestamp") from None
    if dt.tzinfo is None:
        raise Refused(f"--as-of {text!r} needs a time zone (…Z)")
    return dt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--match-ids-file", required=True, help="One match id per line (the never_resulted rows).")
    p.add_argument("--espn-cache", required=True, help="ESPN scoreboards: fetched into it, or replayed from it.")
    p.add_argument("--as-of", default=None, help="The instant staleness is measured from (default: now).")
    p.add_argument("--apply", action="store_true", help="Commit (default: dry run, rolled back).")
    p.add_argument("--expect-plan", default=None, help="With --apply: every PLAN key, from the reviewed dry run.")
    p.add_argument("--out-csv", required=True, help="Write-once audit CSV (and its .changes.csv / .archive).")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    try:
        if args.apply and not args.expect_plan:
            raise Refused("--apply needs --expect-plan (every PLAN key, from the reviewed dry run)")
        expected = parse_expect_plan(args.expect_plan) if args.expect_plan else None
        as_of = parse_as_of(args.as_of)
        ids = gm.read_ids(args.match_ids_file)
        with psycopg2.connect(args.database_url) as conn:
            run(conn, ids, as_of, args.espn_cache, args.apply, expected, args.out_csv)
    except (Refused, gm.Refused) as e:
        logger.error("REFUSED: %s", e)
        return 1
    if fu.FETCH_STATS["failed"] and not fu.FETCH_STATS["ok"]:
        logger.error("Every ESPN request failed (%d) — an outage; nothing healed", fu.FETCH_STATS["failed"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
