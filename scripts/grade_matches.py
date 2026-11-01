"""Grade exactly the listed finished matches, pinned and in one transaction.

grade_completed_matches.py --match-id grades one match and commits;
--days N grades every finished match in the window, every sport, whatever
else is ungraded there. A repair that finished known rows needs neither:
this runs grade_completed_matches.grade_match — the pipeline grader's own
per-match path (ungraded predictions, open recs, recs written after kickoff
voided, the 90-minute re-grade) — over exactly the listed matches, all in
ONE transaction, and reports what changed by diffing those matches'
predictions and recommendations before and after.

  * Dry run unless --apply. The dry run grades for real inside the
    transaction and rolls it back, so its PLAN line is the apply's.
  * Every listed id must be a finished match, or nothing runs (exit 1).
  * --apply needs --expect-plan with every PLAN key (the reviewed dry run's
    counts): any drift rolls the whole run back before the commit (exit 1).
  * Every change (old and new values) goes to a write-once CSV before the
    commit, dry run included. That CSV is the reversal record: predictions
    were ungraded (or graded as its old_* columns say), recs as their old_*
    columns say.

PLAN keys: matches, predictions_graded (an ungraded row now carries a
verdict or an actual outcome), predictions_regraded (a graded row
rewritten: the 90-minute re-grade), recs_changed (status, actual_result or
profit_loss changed). The per-status rec breakdown is logged beside it.

Usage (inside the api container):
    python /app/scripts/grade_matches.py --match-ids-file /app/backups/ids.txt --out-csv /app/backups/g_dry.csv
    python /app/scripts/grade_matches.py --match-ids-file /app/backups/ids.txt --apply \\
        --expect-plan matches=26,predictions_graded=210,predictions_regraded=0,recs_changed=0 \\
        --out-csv /app/backups/g_apply.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grade_completed_matches as gcm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("grade_matches")

PLAN_KEYS = ("matches", "predictions_graded", "predictions_regraded", "recs_changed")

PREDICTIONS_SQL = """
    SELECT p.id::text AS id, p.match_id::text AS match_id, p.prediction_type, p.model_name,
           p.is_correct, p.actual_outcome
      FROM predictions p
     WHERE p.match_id = ANY(%s::uuid[])
     ORDER BY p.id
"""
RECS_SQL = """
    SELECT r.id::text AS id, r.match_id::text AS match_id, r.bet_type, r.selection,
           r.status, r.actual_result, r.profit_loss::text AS profit_loss
      FROM betting_recommendations r
     WHERE r.match_id = ANY(%s::uuid[])
     ORDER BY r.id
"""
PRED_FIELDS = ("is_correct", "actual_outcome")
REC_FIELDS = ("status", "actual_result", "profit_loss")
CSV_FIELDS = (
    "table",
    "id",
    "match_id",
    "kind",
    "detail",
    "old_is_correct",
    "new_is_correct",
    "old_actual_outcome",
    "new_actual_outcome",
    "old_status",
    "new_status",
    "old_actual_result",
    "new_actual_result",
    "old_profit_loss",
    "new_profit_loss",
)


class Refused(RuntimeError):
    """The run cannot go ahead as asked; nothing is written."""


def read_ids(path: str) -> list[str]:
    """One match id per line (blank lines ignored), each a UUID, no repeats, at least one."""
    ids: list[str] = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        try:
            ids.append(str(uuid.UUID(s)))
        except ValueError:
            raise Refused(f"{path}:{n}: {s!r} is not a match id") from None
    if not ids:
        raise Refused(f"{path} lists no match id")
    if len(set(ids)) != len(ids):
        raise Refused(f"{path} lists a match id more than once")
    return ids


def parse_expect_plan(text: str) -> dict[str, int]:
    """'matches=26,predictions_graded=210,...' -> dict; every PLAN key, nothing else."""
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


def snapshot(cur, ids: list[str]) -> dict[tuple[str, str], dict]:
    """{(table, id): row} for every prediction and recommendation on the matches."""
    out: dict[tuple[str, str], dict] = {}
    cur.execute(PREDICTIONS_SQL, (ids,))
    for r in cur.fetchall():
        out[("predictions", r["id"])] = dict(r)
    cur.execute(RECS_SQL, (ids,))
    for r in cur.fetchall():
        out[("betting_recommendations", r["id"])] = dict(r)
    return out


def diff(before: dict, after: dict) -> list[dict]:
    """One CSV row per prediction or recommendation whose graded fields changed. A row that appeared or vanished
    is an error: grading never inserts or deletes either. Pure."""
    if set(before) != set(after):
        gone, appeared = sorted(set(before) - set(after)), sorted(set(after) - set(before))
        raise Refused(f"rows appeared or vanished while grading (gone {gone[:3]}, new {appeared[:3]})")
    changes = []
    for key in sorted(before):
        table, rid = key
        old, new = before[key], after[key]
        fields = PRED_FIELDS if table == "predictions" else REC_FIELDS
        if all(old[f] == new[f] for f in fields):
            continue
        row = {"table": table, "id": rid, "match_id": old["match_id"]}
        for f in fields:
            row[f"old_{f}"], row[f"new_{f}"] = old[f], new[f]
        if table == "predictions":
            graded_before = old["is_correct"] is not None or old["actual_outcome"] is not None
            row["kind"] = "regraded" if graded_before else "graded"
            row["detail"] = f"{old['model_name']} {old['prediction_type']}"
        else:
            row["kind"] = "rec"
            row["detail"] = f"{old['bet_type']} {old['selection']}"
        changes.append(row)
    return changes


def plan_counts(n_matches: int, changes: list[dict]) -> Counter:
    counts: Counter = Counter({k: 0 for k in PLAN_KEYS})
    counts["matches"] = n_matches
    for c in changes:
        if c["kind"] == "graded":
            counts["predictions_graded"] += 1
        elif c["kind"] == "regraded":
            counts["predictions_regraded"] += 1
        else:
            counts["recs_changed"] += 1
            counts[f"rec:{c['old_status']}->{c['new_status']}"] += 1
    return counts


def plan_line(counts: Counter) -> str:
    return "PLAN " + " ".join(f"{k}={counts[k]}" for k in PLAN_KEYS)


def finished_matches(cur, ids: list[str]) -> list[dict]:
    """grade_completed_matches' own row for each id, in the given order; refuses when one is not finished."""
    rows, missing = [], []
    for mid in ids:
        found = gcm.list_finished_matches(cur, 0, mid)
        if not found:
            missing.append(mid)
        else:
            rows.append(found[0])
    if missing:
        raise Refused(f"{len(missing)} listed match(es) are not finished (or do not exist): {', '.join(missing[:5])}")
    return rows


def grade_listed(cur, matches: list[dict], stats: Optional[dict] = None) -> None:
    """grade_completed_matches.grade_match over each match (in the caller's transaction)."""
    for m in matches:
        gcm.grade_match(cur, m, stats)


def write_csv(path: str, changes: list[dict]) -> None:
    """Write-once (mode 'x'): a re-run can never overwrite an earlier run's record."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = out.open("x", newline="")
    except FileExistsError:
        raise Refused(f"{out} exists — the audit CSV is write-once") from None
    with fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(changes)


def run(conn, ids: list[str], apply_changes: bool, expected: Optional[dict[str, int]], out_csv: str) -> Counter:
    stats: dict = {}
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        matches = finished_matches(cur, ids)
        before = snapshot(cur, ids)
        grade_listed(cur, matches, stats)
        changes = diff(before, snapshot(cur, ids))
    counts = plan_counts(len(matches), changes)
    write_csv(out_csv, changes)
    for k in sorted(k for k in counts if k.startswith("rec:")):
        logger.info("recs %s: %d", k[4:], counts[k])
    if stats:
        logger.info("grader diagnostics: %s", dict(sorted(stats.items())))
    logger.info("%s (audit CSV %s)", plan_line(counts), out_csv)
    if not apply_changes:
        conn.rollback()
        logger.info(
            "DRY-RUN: rolled back; re-run with --apply --expect-plan %s", plan_line(counts)[5:].replace(" ", ",")
        )
        return counts
    drift = {k: (expected[k], counts[k]) for k in PLAN_KEYS if expected is not None and expected[k] != counts[k]}
    if drift:
        conn.rollback()
        raise Refused(f"the plan drifted from --expect-plan (expected, now): {drift} — rolled back, nothing written")
    conn.commit()
    logger.info("Committed: %s", plan_line(counts)[5:])
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--match-ids-file", required=True, help="One finished match id per line.")
    p.add_argument("--apply", action="store_true", help="Commit (default: dry run, rolled back).")
    p.add_argument("--expect-plan", default=None, help="With --apply: every PLAN key, from the reviewed dry run.")
    p.add_argument("--out-csv", required=True, help="Write-once audit CSV of every change.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    try:
        if args.apply and not args.expect_plan:
            raise Refused("--apply needs --expect-plan (the reviewed dry run's PLAN counts)")
        expected = parse_expect_plan(args.expect_plan) if args.expect_plan else None
        ids = read_ids(args.match_ids_file)
        with psycopg2.connect(args.database_url) as conn:
            run(conn, ids, args.apply, expected, args.out_csv)
    except Refused as e:
        logger.error("REFUSED: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
