"""Restore recommendations that a fixture-orphan delete took with it.

WHAT HAPPENED
-------------
repair_stranded_recs.py settles a stranded rec and re-points its match_id
onto the finished twin, but its prediction_id still names a prediction on
the orphan row. repair_fixture_orphans.py then deleted the orphan, and
betting_recommendations.prediction_id is ON DELETE CASCADE: the prediction
went, and the rec with it. Prod 2026-10-31 00:49Z, tennis: 45 of the 50
settled recs were deleted that way (the 5 survivors had their prediction
moved; the 30 voided recs kept their orphan). The orphan tool now refuses
to delete such a row; this puts the 45 back.

WHAT THIS DOES
--------------
Inputs, all written before the damage:
  --audit-csv  repair_stranded_recs' write-once CSV (rec, orphan, twin,
               planned status / outcome / P&L);
  --dump       a pg_dump (custom format) taken BEFORE that repair — the
               full rec rows, still pending;
  --archive    repair_fixture_orphans' gzip JSONL — the deleted predictions.

For every settle row of the CSV whose rec no longer exists, the dump row is
re-inserted exactly as it was, plus the settlement the repair made:
status / actual_result / profit_loss from the CSV, match_id = the twin,
settled_at = updated_at = NOW(). prediction_id cannot name the deleted
prediction (gone) nor a re-inserted copy (the twin already holds that
(model, version, type) — predictions_match_model_version_type_key), so it
names the twin's own prediction of the same model, version and type: the
same model's pre-kickoff output for the same fixture. The original
prediction id goes to the audit CSV; its row stays in the archive.

A row is refused (counted, never inserted) unless: the dump row is
'pending' on the CSV's orphan; the archived prediction is that rec's, on
that orphan; the twin is finished with the CSV's score; grading the rec's
own selection on that score (grading_outcomes, never re-implemented)
reproduces the CSV's status and P&L; and the twin holds exactly one
prediction of the same key. The insert is all-or-nothing and ON CONFLICT
(id) DO NOTHING; any row that does not insert rolls everything back. Every
planned row goes to a write-once CSV first, dry run included.

Usage (inside the api container; dry run is the default):
    python /app/scripts/restore_cascaded_recs.py \\
        --audit-csv /app/backups/repaired_rec_ids_tennis_2026-10-31T004906Z.csv \\
        --dump /app/backups/betting_system-2026-10-31T004703Z.dump \\
        --archive /app/backups/fixture_orphans_tennis_20261031T004915Z.jsonl.gz
    ... --apply
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grading_outcomes import grade_rec_selection, rec_status_and_pl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("restore_cascaded_recs")

TABLE = "betting_recommendations"
PL_TOLERANCE = Decimal("0.0001")  # the CSV stores P&L to 4-6 dp

CSV_FIELDS = (
    "rec_id",
    "orphan_match_id",
    "twin_id",
    "original_prediction_id",
    "new_prediction_id",
    "bet_type",
    "selection",
    "status",
    "outcome",
    "profit_loss",
)

TWIN_SQL = "SELECT id::text AS id, status, home_score, away_score FROM matches WHERE id = %s"

TWIN_PREDICTION_SQL = """
    SELECT id::text AS id FROM predictions
     WHERE match_id = %s AND model_name = %s AND model_version = %s AND prediction_type = %s
"""

# Columns the restore sets; every other column comes from the dump row as-is.
OVERRIDES = {
    "prediction_id": "%(prediction_id)s::uuid",
    "match_id": "%(match_id)s::uuid",
    "status": "%(status)s",
    "actual_result": "%(outcome)s",
    "profit_loss": "%(profit_loss)s::numeric",
    "settled_at": "NOW()",
    "updated_at": "NOW()",
}


def copy_block(dump_text: str, table: str, wanted_ids: set[str]) -> tuple[list[str], list[str]]:
    """(columns, data lines) of `table`'s COPY section in pg_restore's SQL
    output, keeping only rows whose id is wanted. Postgres parses the lines
    (COPY FROM STDIN), so no COPY-escaping is re-implemented here."""
    lines = dump_text.splitlines()
    header = f"COPY public.{table} ("
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(header))
    except StopIteration:
        raise RuntimeError(f"no COPY section for {table} in the dump") from None
    cols = lines[start][len(header) :].split(")", 1)[0].split(", ")
    id_at = cols.index("id")
    kept = []
    for line in lines[start + 1 :]:
        if line == "\\.":
            return cols, kept
        if line.split("\t")[id_at] in wanted_ids:
            kept.append(line)
    raise RuntimeError(f"COPY section for {table} is truncated (no terminator)")


def load_dump_recs(cur, dump: Path, wanted_ids: set[str]) -> dict[str, dict]:
    out = subprocess.run(
        ["pg_restore", "--data-only", "--table", TABLE, "--file", "-", str(dump)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    cols, kept = copy_block(out, TABLE, wanted_ids)
    cur.execute(f"CREATE TEMP TABLE restore_recs (LIKE {TABLE})")
    cur.copy_expert(f"COPY restore_recs ({', '.join(cols)}) FROM STDIN", io.StringIO("".join(f"{k}\n" for k in kept)))
    cur.execute("SELECT * FROM restore_recs")
    return {str(r["id"]): dict(r) for r in cur.fetchall()}


def load_archived_predictions(archive: Path, wanted_ids: set[str]) -> dict[str, dict]:
    out = {}
    with gzip.open(archive, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            if d["table"] == "predictions" and d["row"]["id"] in wanted_ids:
                out[d["row"]["id"]] = d["row"]
    return out


def check_row(audit: dict, rec: Optional[dict], pred: Optional[dict], twin: Optional[dict]) -> Optional[str]:
    """None when the restore is safe, else the refusal reason. Pure."""
    if rec is None:
        return "not_in_dump"
    if rec["status"] != "pending" or str(rec["match_id"]) != audit["match_id"]:
        return "dump_row_not_the_repaired_one"
    if pred is None or pred["match_id"] != audit["match_id"]:
        return "prediction_not_archived"
    if twin is None or twin["status"] != "finished":
        return "twin_not_finished"
    if (str(twin["home_score"]), str(twin["away_score"])) != (audit["twin_home_score"], audit["twin_away_score"]):
        return "twin_score_changed"
    outcome = grade_rec_selection(rec["bet_type"], rec["selection"], twin["home_score"], twin["away_score"])
    if outcome is None or outcome != audit["outcome"]:
        return "regrade_disagrees"
    status, pl = rec_status_and_pl(outcome, rec["recommended_stake"], rec["odds_at_recommendation"])
    if status != audit["status"] or abs(pl - Decimal(audit["profit_loss"])) > PL_TOLERANCE:
        return "regrade_disagrees"
    return None


def insert_sql(columns: list[str]) -> str:
    select = ", ".join(OVERRIDES.get(c, c) for c in columns)
    return (
        f"INSERT INTO {TABLE} ({', '.join(columns)}) "
        f"SELECT {select} FROM restore_recs WHERE id = %(rec_id)s::uuid "
        "ON CONFLICT (id) DO NOTHING"
    )


def default_out_csv() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return f"{os.environ.get('BACKUP_LOCAL_DIR', '/app/backups')}/restored_recs_{stamp}.csv"


def write_csv(path: str, entries: list[dict]) -> None:
    out = Path(path)
    if out.exists():
        raise FileExistsError(f"{out} exists — the audit CSV is write-once")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(entries)


def run(conn, audit_csv: Path, dump: Path, archive: Path, apply_changes: bool, out_csv: Optional[str]) -> Counter:
    counts: Counter = Counter()
    audit = {r["rec_id"]: r for r in csv.DictReader(audit_csv.open()) if r["action"] == "settle"}
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"SELECT id::text AS id FROM {TABLE} WHERE id = ANY(%s::uuid[])", (list(audit),))
        alive = {r["id"] for r in cur.fetchall()}
        lost = {k: v for k, v in audit.items() if k not in alive}
        counts["settled_in_csv"], counts["still_present"], counts["missing"] = len(audit), len(alive), len(lost)
        if not lost:
            logger.info("nothing to restore: every settled rec in the CSV still exists")
            return counts

        recs = load_dump_recs(cur, dump, set(lost))
        preds = load_archived_predictions(archive, {str(r["prediction_id"]) for r in recs.values()})
        entries = []
        for rec_id, a in sorted(lost.items()):
            rec = recs.get(rec_id)
            pred = preds.get(str(rec["prediction_id"])) if rec else None
            cur.execute(TWIN_SQL, (a["new_match_id"],))
            twin = cur.fetchone()
            reason = check_row(a, rec, pred, twin)
            if reason is None:
                cur.execute(
                    TWIN_PREDICTION_SQL,
                    (a["new_match_id"], pred["model_name"], pred["model_version"], pred["prediction_type"]),
                )
                twin_preds = cur.fetchall()
                if len(twin_preds) != 1:
                    reason = "twin_has_no_same_key_prediction"
            if reason is not None:
                counts[f"refused_{reason}"] += 1
                logger.error("REFUSED rec %s: %s", rec_id, reason)
                continue
            entries.append(
                {
                    "rec_id": rec_id,
                    "orphan_match_id": a["match_id"],
                    "twin_id": a["new_match_id"],
                    "original_prediction_id": str(rec["prediction_id"]),
                    "new_prediction_id": twin_preds[0]["id"],
                    "bet_type": rec["bet_type"],
                    "selection": rec["selection"],
                    "status": a["status"],
                    "outcome": a["outcome"],
                    "profit_loss": a["profit_loss"],
                }
            )
        counts["planned"] = len(entries)
        counts["planned_won"] = sum(1 for e in entries if e["status"] == "won")
        counts["planned_lost"] = sum(1 for e in entries if e["status"] == "lost")
        net = sum((Decimal(e["profit_loss"]) for e in entries), Decimal("0"))
        out_csv = out_csv or default_out_csv()
        write_csv(out_csv, entries)
        logger.info("planned %d restore(s), net P&L %s; audit CSV %s", len(entries), round(net, 2), out_csv)

        if not apply_changes:
            logger.info("DRY-RUN: re-run with --apply to insert %d row(s)", len(entries))
            conn.rollback()
            return counts
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position",
            (TABLE,),
        )
        sql = insert_sql([r["column_name"] for r in cur.fetchall()])
        try:
            for e in entries:
                cur.execute(
                    sql,
                    {
                        "rec_id": e["rec_id"],
                        "prediction_id": e["new_prediction_id"],
                        "match_id": e["twin_id"],
                        "status": e["status"],
                        "outcome": e["outcome"],
                        "profit_loss": e["profit_loss"],
                    },
                )
                if cur.rowcount != 1:
                    raise RuntimeError(f"rec {e['rec_id']} did not insert (rowcount={cur.rowcount}) — rolled back")
                counts["inserted"] += 1
            conn.commit()
        except Exception:
            conn.rollback()
            counts["inserted"] = 0
            raise
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit-csv", type=Path, required=True, help="repair_stranded_recs audit CSV")
    p.add_argument("--dump", type=Path, required=True, help="pg_dump (custom format) from BEFORE that repair")
    p.add_argument("--archive", type=Path, required=True, help="repair_fixture_orphans archive (.jsonl.gz)")
    p.add_argument("--apply", action="store_true", help="insert (default: dry run)")
    p.add_argument("--out-csv", default=None, help="audit CSV (default /app/backups/restored_recs_<utc>.csv)")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    for path in (args.audit_csv, args.dump, args.archive):
        if not path.is_file():
            logger.error("%s does not exist", path)
            return 2
    with psycopg2.connect(args.database_url) as conn:
        counts = run(conn, args.audit_csv, args.dump, args.archive, args.apply, args.out_csv)
    logger.info("Done: %s", dict(sorted(counts.items())))
    refused = sum(v for k, v in counts.items() if k.startswith("refused_"))
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
