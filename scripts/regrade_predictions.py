"""Re-grade ALREADY-graded predictions with the current grading code.

grade_completed_matches only grades rows whose is_correct is still NULL,
so a fix to grading_outcomes never reaches a row the old code already
graded. This tool re-runs grade_completed_matches.compute_prediction_grade
— the exact verdict path the pipeline grader runs — over graded rows and
rewrites only those whose actual_outcome or is_correct changes.

Written for the 2026-10-31 NHL grading fixes: dixon_coles_nhl
double_chance had been graded on the final score (never a draw in the
NHL, so every '12' pick was credited: 286/286), and correct_score 'other'
by string equality (never credited: 0/281). Planned locally from a
read-only export of those 572 rows (matches 2026-06-05..10-31): 67
double_chance rows True -> False (regulation ties, 100% -> 76.6%) and 146
correct_score 'other' rows False -> True (0% -> 51.0%); nothing else moves.

Guarantees:
  * Dry run unless --apply; refuses to run without --model-name and/or
    --prediction-type.
  * Every planned change goes to a write-once CSV (old + new values)
    BEFORE any UPDATE, dry run included. That CSV is the rollback.
  * Never touches betting_recommendations: recs settle from their own
    (bet_type, selection) via grade_rec_selection, never from a
    prediction row.
  * Rows written after kickoff are skipped, as in the grader.
  * Rewrites a verdict into another verdict, never un-grades: a row the
    current code cannot grade (phantom line, regulation_winner or
    half-time score missing, ...) is left as it is and counted, and so is
    a graded row the current code would NULL without a push (a
    correct_score 'other' pick whose probabilities lack its scoreline
    keys, ...): skipped_would_ungrade, each one logged.
  * Each UPDATE is guarded on the values it was planned from; a row that
    changed since the plan is left alone, logged as an error, and its
    prediction_id written to a second write-once file, <csv>.stale.txt
    (one id per line, empty when every planned row was written). A CSV
    rollback must exclude those ids: their old values are not this run's
    to restore.

The before/after accuracy summary is per ROW, not deduped per match:
weekly retrains can write several rows per (match, prediction_type), so
the match count is printed alongside.

Usage (inside the api container):
    python /app/scripts/regrade_predictions.py --model-name dixon_coles_nhl \\
        --prediction-type double_chance,correct_score --days 365          # dry run
    python /app/scripts/regrade_predictions.py --model-name dixon_coles_nhl \\
        --prediction-type double_chance,correct_score --days 365 --apply
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grade_completed_matches as gcm  # noqa: E402
from grading_outcomes import is_push  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("regrade_predictions")

# Graded = carries a verdict or a push (actual_outcome set, is_correct
# NULL). Rows with neither are grade_completed_matches' job, not this one.
GRADED_PREDICTIONS_SQL = """
    SELECT p.id::text AS prediction_id,
           p.match_id::text AS match_id,
           p.prediction_type, p.model_name, p.predicted_outcome, p.probabilities,
           p.actual_outcome, p.is_correct,
           (p.created_at > m.match_date) AS after_kickoff,
           m.home_score, m.away_score, m.metadata, m.half_time_home, m.half_time_away
    FROM predictions p
    JOIN matches m ON m.id = p.match_id
    WHERE m.status = 'finished'
      AND m.match_date >= NOW() - (%(days)s || ' days')::interval
      AND (p.actual_outcome IS NOT NULL OR p.is_correct IS NOT NULL)
      AND (%(model_name)s::text IS NULL OR p.model_name = %(model_name)s::text)
      AND (%(types)s::text[] IS NULL OR p.prediction_type = ANY(%(types)s::text[]))
    ORDER BY m.match_date, p.match_id, p.prediction_type
"""

# Guarded on the plan's old values: a row the grader or anything else
# rewrote in between is not overwritten with a stale plan.
GUARDED_UPDATE_SQL = """
    UPDATE predictions
    SET actual_outcome = %(new_actual_outcome)s,
        is_correct = %(new_is_correct)s,
        updated_at = NOW()
    WHERE id = %(prediction_id)s
      AND actual_outcome IS NOT DISTINCT FROM %(old_actual_outcome)s
      AND is_correct IS NOT DISTINCT FROM %(old_is_correct)s
"""

CSV_FIELDS = (
    "prediction_id",
    "match_id",
    "model_name",
    "prediction_type",
    "predicted_outcome",
    "old_actual_outcome",
    "old_is_correct",
    "new_actual_outcome",
    "new_is_correct",
)

MATCH_FIELDS = ("match_id", "home_score", "away_score", "metadata", "half_time_home", "half_time_away")


def parse_types(values: Optional[list[str]]) -> Optional[list[str]]:
    """--prediction-type is repeatable and comma-separated. None (no flag,
    or only blanks) means every type."""
    types = sorted({t.strip() for v in values or [] for t in v.split(",") if t.strip()})
    return types or None


def match_of(row: dict) -> dict:
    """The matches-row view compute_prediction_grade reads."""
    return {k: row[k] for k in MATCH_FIELDS}


def matches_needing_context(rows: list[dict]) -> set[str]:
    """Matches with a line-dependent (NBA/NFL spread/total) row — the only
    verdicts that read features_cache or the odds probes, so the only
    matches worth a grading-context round trip."""
    return {r["match_id"] for r in rows if gcm.is_line_dependent(r["prediction_type"], r["model_name"])}


def plan_regrade(rows: list[dict], contexts: dict[str, dict]) -> tuple[list[dict], Counter]:
    """Re-grade every row through compute_prediction_grade. Returns the
    rows whose (actual_outcome, is_correct) changes, as CSV-shaped dicts,
    plus counts. Rows the current code skips are left out entirely, as
    are verdicts it would turn into NULL without a push
    (skipped_would_ungrade): "the current code cannot say" is not a
    reason to erase what an earlier grade said.

    `contexts` maps match_id -> fetch_match_grading_context dict and must
    cover every match in matches_needing_context(rows): a line-dependent
    row graded without its odds probes would skip the phantom-line guard,
    so a missing context raises instead."""
    changes: list[dict] = []
    counts: Counter = Counter()
    for row in rows:
        ctx = contexts.get(row["match_id"])
        if ctx is None:
            if gcm.is_line_dependent(row["prediction_type"], row["model_name"]):
                raise ValueError(f"no grading context for line-dependent prediction {row['prediction_id']}")
            ctx = {}
        skip, actual, correct = gcm.compute_prediction_grade(row, match_of(row), ctx)
        if skip is not None:
            counts[f"skipped_{skip}"] += 1
            continue
        if (actual, correct) == (row["actual_outcome"], row["is_correct"]):
            counts["unchanged"] += 1
            continue
        if correct is None and row["is_correct"] is not None and not is_push(actual):
            counts["skipped_would_ungrade"] += 1
            logger.warning(
                "Not un-grading prediction %s (%s/%s, pick %r): the current code grades (%r, NULL), "
                "the row holds (%r, %r) — left as it is",
                row["prediction_id"],
                row["model_name"],
                row["prediction_type"],
                row["predicted_outcome"],
                actual,
                row["actual_outcome"],
                row["is_correct"],
            )
            continue
        counts["changed"] += 1
        changes.append(
            {
                "prediction_id": row["prediction_id"],
                "match_id": row["match_id"],
                "model_name": row["model_name"],
                "prediction_type": row["prediction_type"],
                "predicted_outcome": row["predicted_outcome"],
                "old_actual_outcome": row["actual_outcome"],
                "old_is_correct": row["is_correct"],
                "new_actual_outcome": actual,
                "new_is_correct": correct,
            }
        )
    return changes, counts


def summarize(rows: list[dict], changes: list[dict]) -> dict[tuple[str, str], dict]:
    """Per (model_name, prediction_type): rows, matches, and graded/correct
    counts before and after the planned changes (skipped rows count the
    same on both sides)."""
    after = {c["prediction_id"]: c["new_is_correct"] for c in changes}
    out: dict[tuple[str, str], dict] = defaultdict(Counter)
    matches: dict[tuple[str, str], set] = defaultdict(set)
    for row in rows:
        key = (row["model_name"], row["prediction_type"])
        s = out[key]
        matches[key].add(row["match_id"])
        old = row["is_correct"]
        new = after.get(row["prediction_id"], old)
        s["rows"] += 1
        s["before_graded"] += old is not None
        s["before_correct"] += old is True
        s["after_graded"] += new is not None
        s["after_correct"] += new is True
        if row["prediction_id"] in after:
            s["changed"] += 1
            s["to_true"] += new is True and old is not True
            s["to_false"] += new is False and old is not False
    for key, s in out.items():
        s["matches"] = len(matches[key])
    return dict(out)


def _acc(correct: int, graded: int) -> str:
    return f"{correct}/{graded} ({100.0 * correct / graded:.1f}%)" if graded else "0/0 (-)"


def format_summary(summary: dict[tuple[str, str], dict]) -> list[str]:
    lines = [f"{'model_name/prediction_type':<44}{'rows':>7}{'matches':>9}  {'before':<20}{'after':<20}changed (→T/→F)"]
    for (model_name, ptype), s in sorted(summary.items()):
        lines.append(
            f"{model_name + '/' + ptype:<44}{s['rows']:>7}{s['matches']:>9}  "
            f"{_acc(s['before_correct'], s['before_graded']):<20}"
            f"{_acc(s['after_correct'], s['after_graded']):<20}"
            f"{s['changed']} (+{s['to_true']}/-{s['to_false']})"
        )
    return lines


def default_out_csv() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base = os.environ.get("BACKUP_LOCAL_DIR", "/app/backups")
    return f"{base}/regrade_{stamp}.csv"


def stale_ids_path(out_csv: str) -> str:
    """Where --apply lists the prediction_ids whose guarded UPDATE matched
    no row: the audit CSV's path plus '.stale.txt'."""
    return f"{out_csv}.stale.txt"


def _open_write_once(path: str, what: str):
    """Mode 'x' refuses an existing file atomically, so a re-run can never
    overwrite the rollback record of an earlier one."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        return out.open("x", newline="")
    except FileExistsError:
        raise FileExistsError(f"{out} exists — the regrade {what} is write-once") from None


def write_csv(path: str, entries: list[dict]) -> None:
    """Write-once (see _open_write_once)."""
    fh = _open_write_once(path, "CSV")
    with fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(entries)


def run(
    conn,
    days: int,
    model_name: Optional[str],
    types: Optional[list[str]],
    apply_changes: bool,
    out_csv: Optional[str] = None,
) -> Counter:
    if model_name is None and types is None:
        logger.warning("No --model-name or --prediction-type: re-grading EVERY graded prediction in %d days", days)
    if not apply_changes:
        conn.set_session(readonly=True)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(GRADED_PREDICTIONS_SQL, {"days": str(days), "model_name": model_name, "types": types})
        rows = [dict(r) for r in cur.fetchall()]
        contexts = {mid: gcm.fetch_match_grading_context(cur, mid) for mid in sorted(matches_needing_context(rows))}
    conn.rollback()  # close the read snapshot before writing

    changes, counts = plan_regrade(rows, contexts)
    counts["scanned"] = len(rows)
    out_csv = out_csv or default_out_csv()
    write_csv(out_csv, changes)
    logger.info("Scanned %d graded row(s), %d change(s) planned; audit CSV %s", len(rows), len(changes), out_csv)
    for line in format_summary(summarize(rows, changes)):
        print(line)
    if counts["skipped_ungradable"]:
        logger.warning(
            "%d graded row(s) the current code cannot grade were left as they are", counts["skipped_ungradable"]
        )
    if counts["skipped_would_ungrade"]:
        logger.warning(
            "%d graded row(s) the current code would un-grade were left as they are (ids logged above)",
            counts["skipped_would_ungrade"],
        )

    if not apply_changes:
        logger.info("DRY-RUN: re-run with --apply to rewrite %d row(s)", len(changes))
        return counts
    # Checked before the first UPDATE, written after the last one and
    # before the commit: a run whose stale list cannot be recorded commits
    # nothing, and a run that dies mid-way leaves no (misleadingly empty)
    # list behind.
    stale_path = stale_ids_path(out_csv)
    if Path(stale_path).exists():
        raise FileExistsError(f"{stale_path} exists — the regrade stale-id list is write-once")
    stale: list[str] = []
    with conn.cursor() as cur:
        for c in changes:
            cur.execute(GUARDED_UPDATE_SQL, c)
            if cur.rowcount == 1:
                counts["updated"] += 1
            else:
                stale.append(c["prediction_id"])
    with _open_write_once(stale_path, "stale-id list") as stale_fh:
        stale_fh.writelines(f"{pid}\n" for pid in stale)
    counts["changed_since_plan"] = len(stale)
    conn.commit()
    if stale:
        logger.error(
            "%d row(s) changed between plan and write — left as they were; exclude them from any rollback "
            "of %s (listed in %s): %s",
            len(stale),
            out_csv,
            stale_path,
            ", ".join(stale),
        )
    logger.info("Rewrote %d prediction row(s); betting_recommendations untouched", counts["updated"])
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default=None, help="Only this predictions.model_name (default: all).")
    p.add_argument(
        "--prediction-type",
        action="append",
        default=None,
        help="Only these prediction types; repeatable and/or comma-separated (default: all).",
    )
    p.add_argument("--days", type=int, default=365, help="Finished matches this recent (default 365).")
    p.add_argument("--apply", action="store_true", help="Write (default: dry run).")
    p.add_argument(
        "--out-csv",
        default=None,
        help="Write-once audit CSV (default $BACKUP_LOCAL_DIR or /app/backups, regrade_<utc>.csv).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    if args.days <= 0:
        logger.error("--days must be positive (got %d)", args.days)
        return 2
    types = parse_types(args.prediction_type)
    if args.model_name is None and types is None:
        # A grading fix is market-specific; an unscoped pass re-reads every
        # graded row in the window (58k for 30 days on 2026-10-31).
        logger.error("Refusing an unscoped re-grade: pass --model-name and/or --prediction-type")
        return 2
    with psycopg2.connect(args.database_url) as conn:
        counts = run(conn, args.days, args.model_name, types, args.apply, args.out_csv)
    logger.info("Done: %s", dict(sorted(counts.items())))
    return 1 if counts["changed_since_plan"] else 0


if __name__ == "__main__":
    sys.exit(main())
