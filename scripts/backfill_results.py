"""Backfill ESPN results for named days, in one transaction, and settle what they resolve.

WHY THIS EXISTS
---------------
Results ingestion shipped 2026-07-13 and sweeps 3 days back
(fetch_upcoming.py --results --days-back 3), so a bout fought before
2026-07-10 was never resulted from ESPN: prod 2026-11-01 held 50 UFC rows of
June 2026 still 'scheduled', 20 of them carrying a pending recommendation.
They are not fixture duplicates — no finished twin exists, so
repair_stranded_recs.py finds nothing to settle them from — and voiding them
would destroy recs that ESPN's final results settle honestly (the rule
repair_stranded_recs enforces as RECOVERABLE_BY_BACKFILL). ESPN still serves
those cards with every winner. A bare `fetch_upcoming.py --results
--days-back 150` would write them too, but it has no dry run, sweeps every
card since, and leaves the recs to a grader that only walks 14 days back.

WHAT THIS DOES
--------------
For each --date (an ESPN scoreboard day) the league's events are fetched
first (deduplicated by event id). Then ONE transaction runs exactly what the
15-min results sweep runs, fetch_upcoming.record_results (_record_result per
competition, then cancel_dropped_bouts for a fight card), and settles what it
resolved with the regular grader's own functions:
  * every row the run turned 'finished' goes through
    grade_completed_matches.grade_match: predictions graded, open recs settled
    on the result, a rec written at or after kickoff voided;
  * the open recs on the rows this run cancelled — those rows only, read by
    id — are voided as the grader voids them (void_open_rec, 'void: match
    cancelled'), when the kickoff is CANCELLED_VOID_HOURS old (a cancelled
    duplicate, metadata.orphan_of, is never voided).
The transaction runs at REPEATABLE READ: its reads see only its own writes, so
a 15-min sweep or grader committing meanwhile cannot leak into the plan (a row
both touch fails the run with a serialization error, rolled back). The sport's
matches and recs are read before and after, inside the transaction, and every
difference is classified into the PLAN line:

  results          competitions the results path wrote (an unchanged rewrite
                   counts: a re-run after success still writes them)
  flipped          'scheduled' -> 'finished'
  inserted         new rows (a bout ESPN resulted that we never had)
  cancelled        'scheduled' -> 'cancelled' (dropped from a card, re-drawn)
  stamped          a finished row whose ESPN id / metadata only changed
  rescored         a finished row whose score changed (never expected)
  other            any other row change (never expected)
  recs_settled     'pending' -> won / lost with a profit_loss
  recs_voided      'pending' -> void
  recs_other       any other rec change (never expected)
  recs_left        pending recs of the sport still on a row that is not
                   finished, kickoff over 24 h old (the stranded set after the run)
  fetch_failed     scoreboard days ESPN did not serve (the plan is partial)
  identity_refused writes the results path refused or team problems it met
                   (fetch_upcoming.IDENTITY_STATS)

Every other sport's matches and recs are fingerprinted before and after
(OUTSIDE_SQL): a run that changed one of them raises and rolls back, since
neither the plan, the pin nor the audit files would show it.

Every changed row goes to a write-once audit CSV (before and after values) and,
with --apply, the full before/after rows to a write-once gzip JSONL archive
(the reversal record), both BEFORE the commit. A dry run (the default) rolls
back. --apply commits only when the plan equals --expect-plan, every key; any
difference rolls back and exits 1. --expect-recs <csv> (columns rec_id, status,
profit_loss, actual_result; others ignored) pins each rec's outcome the same
way, inside the transaction: the recs the run changed must be exactly the
file's, each with the file's status, profit_loss and actual_result. A re-run
after success plans 0 for every key but results and recs_left.

Sports: mma only — the one whose backfill was reviewed (2026-11-01). Another
sport needs its own review of what a wide results sweep does to its rows
before it is added to ALLOWED_SPORTS.

Usage (inside the api container; scripts/ops/repair_stranded_recs_mma.sh pins it):
    python /app/scripts/backfill_results.py --sport mma --dates 2026-06-06,2026-06-14
    python /app/scripts/backfill_results.py --sport mma --dates ... --apply --expect-plan results=N,... \
        --expect-recs /app/scripts/data/<reviewed recs>.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_upcoming as fu  # noqa: E402
import grade_completed_matches as gcm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_results")

ALLOWED_SPORTS = ("mma",)

# The live sweep covers today back through 3 days; a day that recent is its
# business, and backfilling it would race the pipeline.
MIN_DAYS_AGO = 4
MAX_DATES = 31

PLAN_KEYS = (
    "results",
    "flipped",
    "inserted",
    "cancelled",
    "stamped",
    "rescored",
    "other",
    "recs_settled",
    "recs_voided",
    "recs_other",
    "recs_left",
    "fetch_failed",
    "identity_refused",
)
# Keys counting a write this run makes (a re-run after success plans 0 of each).
WRITE_KEYS = (
    "flipped",
    "inserted",
    "cancelled",
    "stamped",
    "rescored",
    "other",
    "recs_settled",
    "recs_voided",
    "recs_other",
)

MATCHES_SQL = """
    SELECT m.id::text AS id, m.status, m.match_date, m.home_team_id::text AS home_team_id,
           m.away_team_id::text AS away_team_id, m.home_score, m.away_score,
           COALESCE(m.external_ids, '{}'::jsonb) AS external_ids, COALESCE(m.metadata, '{}'::jsonb) AS metadata,
           to_jsonb(m) AS row_json
      FROM matches m
      JOIN leagues l ON l.id = m.league_id
     WHERE l.sport = %s
"""

RECS_SQL = """
    SELECT br.id::text AS id, br.match_id::text AS match_id, br.status, br.actual_result, br.profit_loss,
           br.settled_at, to_jsonb(br) AS row_json
      FROM betting_recommendations br
      JOIN matches m ON m.id = br.match_id
      JOIN leagues l ON l.id = m.league_id
     WHERE l.sport = %s
"""

# The stranded set: what repair_stranded_recs reads, plus a cancelled row's
# recs the grader has not voided yet.
RECS_LEFT_SQL = """
    SELECT count(*) AS n
      FROM betting_recommendations br
      JOIN matches m ON m.id = br.match_id
      JOIN leagues l ON l.id = m.league_id
     WHERE l.sport = %s
       AND br.status IN ('pending', 'placed')
       AND m.status <> 'finished'
       AND m.match_date < NOW() - interval '24 hours'
"""

# A fingerprint of every OTHER sport's matches and recs (count:md5 of the
# fields this tool could write). The run only ever means to write --sport's
# rows; the plan and the audit files see nothing else.
OUTSIDE_SQL = """
    SELECT (SELECT count(*)::text || ':' || COALESCE(md5(string_agg(md5(ROW(
                       m.id, m.status, m.match_date, m.home_team_id, m.away_team_id, m.home_score, m.away_score,
                       m.external_ids, m.metadata)::text), '' ORDER BY m.id)), '')
              FROM matches m
              LEFT JOIN leagues l ON l.id = m.league_id
             WHERE l.sport IS DISTINCT FROM %s) AS matches,
           (SELECT count(*)::text || ':' || COALESCE(md5(string_agg(md5(ROW(
                       br.id, br.match_id, br.status, br.actual_result, br.profit_loss, br.settled_at)::text), ''
                       ORDER BY br.id)), '')
              FROM betting_recommendations br
              JOIN matches m ON m.id = br.match_id
              LEFT JOIN leagues l ON l.id = m.league_id
             WHERE l.sport IS DISTINCT FROM %s) AS recs
"""

MATCH_FIELDS = (
    "status",
    "match_date",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "external_ids",
    "metadata",
)
REC_FIELDS = ("match_id", "status", "actual_result", "profit_loss", "settled_at")


class UnsafeRequest(RuntimeError):
    """Arguments that would make the backfill unsafe or meaningless."""


def snapshot(cur, sport: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """({match id: row}, {rec id: row}) for every match and rec of the sport."""
    cur.execute(MATCHES_SQL, (sport,))
    matches = {r["id"]: dict(r) for r in cur.fetchall()}
    cur.execute(RECS_SQL, (sport,))
    recs = {r["id"]: dict(r) for r in cur.fetchall()}
    return matches, recs


def classify_match(old: Optional[dict], new: Optional[dict]) -> Optional[str]:
    """The PLAN key of one match's change, or None when it did not change. Pure."""
    if old is None:
        return "inserted" if new is not None else None
    if new is None:
        return "other"  # a deleted row: this tool never deletes
    changed = {f for f in MATCH_FIELDS if old.get(f) != new.get(f)}
    if not changed:
        return None
    if old["status"] == "scheduled" and new["status"] == "finished":
        return "flipped"
    if old["status"] == "scheduled" and new["status"] == "cancelled":
        return "cancelled"
    if old["status"] == new["status"] == "finished":
        if changed & {"home_score", "away_score"}:
            return "rescored"
        if changed <= {"external_ids", "metadata"}:
            return "stamped"
    return "other"


def classify_rec(old: Optional[dict], new: Optional[dict]) -> Optional[str]:
    """The PLAN key of one rec's change, or None when it did not change. Pure."""
    if old is not None and new is not None and all(old.get(f) == new.get(f) for f in REC_FIELDS):
        return None
    if old is None or new is None:
        return "recs_other"  # a rec appeared or vanished: this tool writes neither
    if old["status"] == "pending" and new["status"] in ("won", "lost") and new.get("profit_loss") is not None:
        return "recs_settled"
    if old["status"] == "pending" and new["status"] == "void":
        return "recs_voided"
    return "recs_other"


def diff(before: tuple[dict, dict], after: tuple[dict, dict]) -> tuple[Counter, list[dict]]:
    """(counts by PLAN key, one audit entry per changed row). Pure."""
    counts: Counter = Counter()
    entries: list[dict] = []
    (m0, r0), (m1, r1) = before, after
    for mid in sorted(set(m0) | set(m1)):
        key = classify_match(m0.get(mid), m1.get(mid))
        if key:
            counts[key] += 1
            entries.append({"kind": "match", "id": mid, "change": key, "before": m0.get(mid), "after": m1.get(mid)})
    for rid in sorted(set(r0) | set(r1)):
        key = classify_rec(r0.get(rid), r1.get(rid))
        if key:
            counts[key] += 1
            entries.append({"kind": "rec", "id": rid, "change": key, "before": r0.get(rid), "after": r1.get(rid)})
    return counts, entries


def plan_line(totals: dict[str, int]) -> str:
    return "PLAN " + " ".join(f"{k}={totals.get(k, 0)}" for k in PLAN_KEYS)


def parse_expect_plan(text: str) -> dict[str, int]:
    """'results=58,flipped=44,...' -> {key: n}; UnsafeRequest on anything else."""
    out: dict[str, int] = {}
    for part in (p.strip() for p in text.split(",")):
        key, sep, value = part.partition("=")
        if not sep or key not in PLAN_KEYS or not value.isdigit() or key in out:
            raise UnsafeRequest(f"--expect-plan {text!r}: want KEY=N pairs with KEY one of {', '.join(PLAN_KEYS)}")
        out[key] = int(value)
    return out


EXPECT_REC_COLUMNS = ("rec_id", "status", "profit_loss", "actual_result")


def parse_expect_recs(path: Path) -> dict[str, dict]:
    """The --expect-recs file -> {rec id: {status, profit_loss (Decimal), actual_result}};
    UnsafeRequest when it cannot be read, lacks a column, repeats a rec or holds no rec."""
    try:
        with Path(path).open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        raise UnsafeRequest(f"--expect-recs {path}: {exc}") from exc
    out: dict[str, dict] = {}
    for i, r in enumerate(rows, start=2):
        if any(r.get(c) in (None, "") for c in EXPECT_REC_COLUMNS[:3]):
            raise UnsafeRequest(f"--expect-recs {path} line {i}: want columns {', '.join(EXPECT_REC_COLUMNS)}")
        if r["rec_id"] in out:
            raise UnsafeRequest(f"--expect-recs {path} line {i}: rec {r['rec_id']} is listed twice")
        try:
            pl = Decimal(r["profit_loss"])
        except ArithmeticError as exc:
            raise UnsafeRequest(f"--expect-recs {path} line {i}: profit_loss {r['profit_loss']!r}") from exc
        out[r["rec_id"]] = {"status": r["status"], "profit_loss": pl, "actual_result": r.get("actual_result") or None}
    if not out:
        raise UnsafeRequest(f"--expect-recs {path}: no rec")
    return out


def rec_outcome_mismatch(entries: list[dict], expect: dict[str, dict]) -> list[str]:
    """The recs whose change differs from --expect-recs: one the file lists that
    the run did not change or changed to another outcome, or one the run changed
    that the file does not list. Pure."""
    got = {e["id"]: e["after"] or {} for e in entries if e["kind"] == "rec"}
    bad = []
    for rid in sorted(set(expect) | set(got)):
        want, after = expect.get(rid), got.get(rid)
        if want is None:
            bad.append(f"{rid} changed but not expected")
        elif after is None:
            bad.append(f"{rid} not changed")
        elif (
            after.get("status") != want["status"]
            or after.get("profit_loss") is None
            or Decimal(str(after["profit_loss"])) != want["profit_loss"]
            or after.get("actual_result") != want["actual_result"]
        ):
            bad.append(
                f"{rid} {after.get('status')}/{after.get('profit_loss')}/{after.get('actual_result')!r}, expected "
                f"{want['status']}/{want['profit_loss']}/{want['actual_result']!r}"
            )
    return bad


def parse_dates(text: str, today: date) -> list[date]:
    """The --dates list, refused when empty, malformed, repeated, too long, or
    inside the live sweep's window (or the future)."""
    out: list[date] = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        try:
            d = date.fromisoformat(part)
        except ValueError as exc:
            raise UnsafeRequest(f"--dates: {part!r} is not YYYY-MM-DD") from exc
        if d in out:
            raise UnsafeRequest(f"--dates: {d} is listed twice")
        if d > today - timedelta(days=MIN_DAYS_AGO):
            raise UnsafeRequest(
                f"--dates: {d} is within {MIN_DAYS_AGO} days of today ({today}) — the 15-min results sweep "
                "covers it; backfilling it would race the pipeline"
            )
        out.append(d)
    if not out:
        raise UnsafeRequest("--dates is empty")
    if len(out) > MAX_DATES:
        raise UnsafeRequest(f"--dates: {len(out)} days, more than {MAX_DATES} in one transaction")
    return sorted(out)


def fetch_events(cfg, dates: list[date]) -> dict[str, list[dict]]:
    """{league slug: events of those days}, each ESPN event once."""
    out: dict[str, list[dict]] = {}
    for slug in cfg.leagues:
        seen: set[str] = set()
        events: list[dict] = []
        for d in dates:
            for ev in fu.fetch_day(cfg, slug, d):
                key = str(ev.get("id"))
                if ev.get("id") is not None and key in seen:
                    continue
                seen.add(key)
                events.append(ev)
        out[slug] = events
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _brief(row: Optional[dict], fields: tuple[str, ...]) -> str:
    return "" if row is None else json.dumps({f: _jsonable(row.get(f)) for f in fields}, sort_keys=True)


def write_csv(path: Path, entries: list[dict]) -> Path:
    """Write-once ("x"): the before values are the reversal record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "id", "change", "before", "after"])
        for e in entries:
            fields = MATCH_FIELDS if e["kind"] == "match" else REC_FIELDS
            w.writerow([e["kind"], e["id"], e["change"], _brief(e["before"], fields), _brief(e["after"], fields)])
    logger.info("Wrote %d changed row(s) to %s", len(entries), path)
    return path


def write_archive(path: Path, entries: list[dict]) -> Path:
    """Write-once gzip JSONL of every changed row's full before/after row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "xt", encoding="utf-8") as fh:
        for e in entries:
            fh.write(
                json.dumps(
                    {
                        "kind": e["kind"],
                        "id": e["id"],
                        "change": e["change"],
                        "before": _jsonable((e["before"] or {}).get("row_json")),
                        "after": _jsonable((e["after"] or {}).get("row_json")),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    logger.info("Archived %d changed row(s) to %s", len(entries), path)
    return path


def default_paths(sport: str, apply: bool) -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(os.environ.get("BACKUP_LOCAL_DIR", "/app/backups"))
    mode = "apply" if apply else "dryrun"
    return base / f"backfill_results_{sport}_{mode}_{stamp}.csv", base / f"backfill_results_{sport}_{stamp}.jsonl.gz"


def settle(cur, before_matches: dict[str, dict], mid_matches: dict[str, dict]) -> dict:
    """Grade the rows this run turned 'finished' and void the open recs on the
    rows it cancelled, with the grader's own functions. Returns grade stats."""
    stats: dict = {k: 0 for k in gcm.STAT_KEYS}
    graded = settled = 0
    finished = [
        mid
        for mid, row in mid_matches.items()
        if row["status"] == "finished" and (before_matches.get(mid) or {}).get("status") != "finished"
    ]
    for mid in sorted(finished):
        for match in gcm.list_finished_matches(cur, 0, mid):
            per = gcm.grade_match(cur, match, stats)
            graded += per["predictions_graded"]
            settled += per["recs_settled"]
    cancelled = {
        mid
        for mid, row in mid_matches.items()
        if row["status"] == "cancelled" and (before_matches.get(mid) or {}).get("status") != "cancelled"
    }
    voided = 0
    # Only the rows this run cancelled, by id: the grader's own read spans every
    # sport's cancelled rows, which are the 15-min grader's business.
    open_recs = gcm.list_recs_on_cancelled_matches(cur, match_ids=sorted(cancelled)) if cancelled else []
    for rec in open_recs:
        if rec.get("orphan_of"):
            logger.error(
                "Not voiding rec %s (match %s): the row this run cancelled is a duplicate of match %s, which was "
                "played — repair_stranded_recs settles it from the twin",
                rec["rec_id"],
                rec["match_id"],
                rec["orphan_of"],
            )
            continue
        voided += gcm.void_open_rec(cur, rec, "cancelled", gcm.CANCELLED_VOID_RESULT)
    logger.info(
        "Graded %d prediction(s) and settled %d rec(s) on %d newly finished row(s); voided %d rec(s) on %d newly "
        "cancelled row(s); post-kickoff voids %d",
        graded,
        settled,
        len(finished),
        voided,
        len(cancelled),
        stats.get("post_kickoff_rec_voids", 0),
    )
    return stats


def identity_refused() -> int:
    s = fu.IDENTITY_STATS
    return sum(s[k] for k in ("forks_refused", "same_team", "score_mismatch", "status_refused")) + (
        s["team_namesakes"] + s["team_conflicts"]
    )


def run(
    conn,
    sport: str,
    dates: list[date],
    apply_changes: bool = False,
    expect_plan: Optional[dict[str, int]] = None,
    out_csv: Optional[Path] = None,
    archive: Optional[Path] = None,
    events_by_slug: Optional[dict[str, list[dict]]] = None,
    expect_recs: Optional[dict[str, dict]] = None,
) -> dict:
    """Fetch, write and settle in one transaction; commit only on --apply with
    a matching plan (and, given `expect_recs`, matching rec outcomes).
    `events_by_slug` replaces the ESPN fetch (tests)."""
    if sport not in ALLOWED_SPORTS:
        raise UnsafeRequest(f"--sport {sport!r}: allowed {', '.join(ALLOWED_SPORTS)} (see the module docstring)")
    cfg = fu.SPORT_CONFIGS[sport]
    failed_before = fu.FETCH_STATS["failed"]
    if events_by_slug is None:
        events_by_slug = fetch_events(cfg, dates)
    fetch_failed = fu.FETCH_STATS["failed"] - failed_before
    refused_before = identity_refused()
    default_csv, default_archive = default_paths(sport, apply_changes)
    out_csv = out_csv or default_csv
    archive = archive or default_archive
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # First statement of the transaction (a connection mid-transaction fails here, loudly).
            cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cur.execute(OUTSIDE_SQL, (sport, sport))
            outside_before = dict(cur.fetchone())
            before = snapshot(cur, sport)
            written = 0
            for slug, events in events_by_slug.items():
                code, name, country = cfg.leagues[slug]
                league_id = fu.ensure_league(cur, cfg, code, name, country)
                if not league_id:
                    raise RuntimeError(f"ensure_league returned no id for {sport} {slug}")
                n = fu.record_results(cur, cfg, league_id, events, slug)
                logger.info("%s %s: %d event(s), %d result(s) written", sport, slug, len(events), n)
                written += n
            mid = snapshot(cur, sport)
            settle(cur, before[0], mid[0])
            after = snapshot(cur, sport)
            cur.execute(OUTSIDE_SQL, (sport, sport))
            outside_after = dict(cur.fetchone())
            if outside_after != outside_before:
                raise RuntimeError(
                    f"the run changed rows outside --sport {sport} (matches {outside_before['matches']} -> "
                    f"{outside_after['matches']}, recs {outside_before['recs']} -> {outside_after['recs']}); "
                    "neither the plan nor the audit files cover them"
                )
            counts, entries = diff(before, after)
            cur.execute(RECS_LEFT_SQL, (sport,))
            counts["recs_left"] = int(cur.fetchone()["n"])
            counts["results"] = written
            counts["fetch_failed"] = fetch_failed
            counts["identity_refused"] = identity_refused() - refused_before
            totals = {k: int(counts.get(k, 0)) for k in PLAN_KEYS}
            for e in entries:
                logger.info("%s %s %s", e["change"], e["kind"], e["id"])
            write_csv(out_csv, entries)
            logger.info("%s", plan_line(totals))
            mismatch = "; ".join(
                f"{totals[k]} {k}, expected {n}" for k, n in sorted((expect_plan or {}).items()) if totals[k] != n
            )
            if mismatch:
                logger.error("The plan is not the one pinned (%s): nothing written; review %s", mismatch, out_csv)
            bad_recs = rec_outcome_mismatch(entries, expect_recs) if expect_recs is not None else []
            if bad_recs:
                logger.error(
                    "%d rec outcome(s) differ from --expect-recs: nothing written; review %s:\n  %s",
                    len(bad_recs),
                    out_csv,
                    "\n  ".join(bad_recs),
                )
                mismatch = "; ".join(filter(None, [mismatch, f"{len(bad_recs)} rec outcome(s) not as expected"]))
            elif expect_recs is not None:
                logger.info("Rec outcomes as expected: %d rec(s), pinned", len(expect_recs))
            if apply_changes and not mismatch:
                write_archive(archive, entries)
                conn.commit()
                logger.info("APPLIED: committed %d changed row(s)", len(entries))
            else:
                conn.rollback()
                if not mismatch:
                    logger.info("DRY-RUN: rolled back. Re-run with --apply --expect-plan <the counts above>.")
    except Exception:
        conn.rollback()
        logger.error("backfill failed; the transaction was rolled back — nothing written", exc_info=True)
        raise
    return {"plan": totals, "plan_mismatch": bool(mismatch), "csv_path": str(out_csv), "entries": entries}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sport", required=True, choices=ALLOWED_SPORTS)
    p.add_argument("--dates", required=True, help="Comma-separated ESPN scoreboard days, YYYY-MM-DD.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--out-csv", type=Path, default=None, help="Write-once audit CSV (default /app/backups/...).")
    p.add_argument("--archive", type=Path, default=None, help="Write-once gzip JSONL reversal record (--apply).")
    p.add_argument(
        "--expect-plan",
        default=None,
        help=f"Commit nothing unless the plan has exactly these counts (KEY=N,... with KEY one of "
        f"{', '.join(PLAN_KEYS)}); --apply needs every key.",
    )
    p.add_argument(
        "--expect-recs",
        type=Path,
        default=None,
        help="Commit nothing unless the recs the run changes are exactly this CSV's (columns rec_id, status, "
        "profit_loss, actual_result), each with its outcome.",
    )
    p.add_argument("--apply", action="store_true", help="Commit (default: dry run, rolled back).")
    args = p.parse_args(argv)
    try:
        args.dates = parse_dates(args.dates, datetime.now(timezone.utc).date())
        args.expect_plan = parse_expect_plan(args.expect_plan) if args.expect_plan is not None else None
        args.expect_recs = parse_expect_recs(args.expect_recs) if args.expect_recs is not None else None
    except UnsafeRequest as exc:
        p.error(str(exc))
    if args.apply:
        missing = [k for k in PLAN_KEYS if k not in (args.expect_plan or {})]
        if missing:
            p.error(f"--apply needs --expect-plan with every key; missing: {', '.join(missing)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set (pass --database-url or export DATABASE_URL)")
        return 2
    conn = psycopg2.connect(args.database_url)
    try:
        result = run(
            conn,
            args.sport,
            args.dates,
            apply_changes=args.apply,
            expect_plan=args.expect_plan,
            out_csv=args.out_csv,
            archive=args.archive,
            expect_recs=args.expect_recs,
        )
    except UnsafeRequest as exc:
        logger.error("%s", exc)
        return 2
    finally:
        conn.close()
    return 1 if result["plan_mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
