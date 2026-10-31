"""Void settled recommendations that were written at or after kickoff.

THE BUG THIS CORRECTS
---------------------
A bet placed after kickoff is hindsight: it must be void, never won or lost.
Until 2026-10-30 neither settlement path checked when a rec was written.
repair_stranded_recs.py settled recs stranded on orphan fixture rows from
their finished twin, and the orphan carried a STALE kickoff: ESPN had moved
the fixture earlier, the fetcher forked a new row at the new time, and rec
generation kept writing recs on the orphan because it still looked upcoming.
Prod, verified read-only 2026-10-30: the 2026-09-04 17:36Z repair run
settled 21 soccer recs written 46 minutes to 24 hours after the real kickoff
(8W/12L/1V, net -228.25) and 2 MMA recs written after their bout had ended
(1W/1L, net -8.80). ESPN's scoreboard confirms every twin's kickoff and
score. grade_completed_matches.py and repair_stranded_recs.py now void such
a rec; this tool corrects the ones already settled.

WHAT THIS DOES
--------------
Candidates: recs of --sport whose created_at >= their match's match_date and
that are SETTLED with money on them: status won/lost, or void with a
profit_loss other than 0. A post-kickoff rec that is already void with
profit 0 (a push) is already right; it is counted and left alone.

Each candidate gets status 'void', profit_loss 0, settled_at NOW() and
actual_result 'void: written after kickoff (was <status> <profit_loss>)'.
The settlement paths write the same prefix, so one LIKE finds every such
row. match_id is NOT changed: a repaired rec already sits on its twin.

Every UPDATE is guarded on the values the plan observed (match_id, status,
profit_loss, actual_result) and re-checks created_at >= the match's kickoff.
If a row has moved, the whole transaction rolls back and nothing is written.
All rows go in ONE transaction.

The audit CSV is write-once (exclusive create) and lands BEFORE any UPDATE,
dry run included. It holds each row's old status / actual_result /
profit_loss / settled_at and is the only reversal record.

Usage (inside the api container; dry run is the default):
    python /app/scripts/void_post_kickoff_recs.py --sport all
    python /app/scripts/void_post_kickoff_recs.py --sport soccer --sport mma --apply
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grade_completed_matches import POST_KICKOFF_VOID_RESULT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("void_post_kickoff_recs")

# Sports whose recs live in betting_recommendations (horse racing has its own
# race_recommendations table). A typo must fail loudly, not void nothing.
ALLOWED_SPORTS = ("soccer", "tennis", "mma", "nba", "nfl", "nhl")
SPORT_ALL = "all"

# The known set is 22 rows. Anything far larger means the candidate query is
# wrong; inspect it before rewriting settled money.
DEFAULT_MAX_ROWS = 100

ACTION_VOID = "void"
ACTION_ALREADY_VOID = "already_void"

# betting_recommendations.actual_result is VARCHAR(100).
ACTUAL_RESULT_MAX_LEN = 100

SELECT_SQL = """
    SELECT br.id::text AS rec_id,
           l.sport,
           br.match_id::text AS match_id,
           m.status AS match_status,
           m.match_date,
           br.created_at,
           br.bet_type,
           br.selection,
           br.recommended_stake,
           br.odds_at_recommendation,
           br.status,
           br.actual_result,
           br.profit_loss,
           br.settled_at
    FROM betting_recommendations br
    JOIN matches m ON m.id = br.match_id
    JOIN leagues l ON l.id = m.league_id
    WHERE br.created_at >= m.match_date
      AND br.status IN ('won', 'lost', 'void')
      AND l.sport = ANY(%s)
    ORDER BY l.sport, br.created_at, br.id
"""

# Guarded on everything the plan observed; rowcount 0 means the row moved.
UPDATE_SQL = """
    UPDATE betting_recommendations
    SET status = 'void',
        profit_loss = 0,
        actual_result = %(new_result)s,
        settled_at = NOW(),
        updated_at = NOW()
    WHERE id = %(rec_id)s::uuid
      AND match_id = %(match_id)s::uuid
      AND status = %(old_status)s
      AND profit_loss IS NOT DISTINCT FROM %(old_pl)s
      AND actual_result IS NOT DISTINCT FROM %(old_result)s
      AND created_at >= (SELECT m.match_date FROM matches m WHERE m.id = %(match_id)s::uuid)
"""

CSV_FIELDS = (
    "rec_id",
    "sport",
    "match_id",
    "match_status",
    "match_date",
    "created_at",
    "minutes_after_kickoff",
    "bet_type",
    "selection",
    "stake",
    "odds",
    "action",
    "old_status",
    "old_actual_result",
    "old_profit_loss",
    "old_settled_at",
    "new_status",
    "new_profit_loss",
    "new_actual_result",
)


class UnsafeRequest(RuntimeError):
    """Arguments that would make the correction unsafe or meaningless."""


class MaxRowsExceeded(RuntimeError):
    """More rows to void than --max-rows and --yes was not passed."""


# ── Pure planning ─────────────────────────────────────────────────────


def resolve_sports(sports: Optional[Iterable[str]]) -> list[str]:
    """Expand 'all' and refuse anything unknown. Order follows ALLOWED_SPORTS."""
    requested = list(sports or [])
    if not requested:
        raise UnsafeRequest("No sport selected: pass --sport <sport> (repeatable) or --sport all.")
    unknown = sorted({s for s in requested if s != SPORT_ALL and s not in ALLOWED_SPORTS})
    if unknown:
        raise UnsafeRequest(f"Unknown sport(s) {unknown}. Allowed: {', '.join(ALLOWED_SPORTS)} or {SPORT_ALL}.")
    if SPORT_ALL in requested:
        return list(ALLOWED_SPORTS)
    return [s for s in ALLOWED_SPORTS if s in requested]


def _fmt_pl(value: Any) -> str:
    return "null" if value is None else f"{Decimal(str(value)):.2f}"


def new_actual_result(old_status: str, old_pl: Any) -> str:
    """'void: written after kickoff (was won 31.41)'. Raises rather than let
    Postgres truncate or reject a label longer than the column."""
    text = f"{POST_KICKOFF_VOID_RESULT} (was {old_status} {_fmt_pl(old_pl)})"
    if len(text) > ACTUAL_RESULT_MAX_LEN:
        raise ValueError(f"actual_result {text!r} exceeds {ACTUAL_RESULT_MAX_LEN} characters")
    return text


def classify(row: dict) -> str:
    """ACTION_VOID for a post-kickoff rec carrying a won/lost verdict or
    money; ACTION_ALREADY_VOID for one already void at profit 0. The
    created_at check repeats the WHERE clause on purpose: a row that reaches
    here without it is a query bug, and voiding a pre-kickoff bet would be
    as wrong as settling a post-kickoff one."""
    created_at, kickoff = row.get("created_at"), row.get("match_date")
    if created_at is None or kickoff is None or created_at < kickoff:
        raise ValueError(
            f"rec {row.get('rec_id')} is not provably post-kickoff (created_at={created_at!r}, "
            f"match_date={kickoff!r}); refusing to plan it"
        )
    status = row.get("status")
    if status in ("won", "lost"):
        return ACTION_VOID
    if status == "void":
        pl = row.get("profit_loss")
        if pl is not None and Decimal(str(pl)) == 0:
            return ACTION_ALREADY_VOID
        return ACTION_VOID
    raise ValueError(f"rec {row.get('rec_id')} has status {status!r}, which is not settled")


def plan(rows: Iterable[dict]) -> list[dict]:
    """One CSV-ready entry per post-kickoff settled rec. Pure."""
    entries = []
    for row in rows:
        action = classify(row)
        minutes = round((row["created_at"] - row["match_date"]).total_seconds() / 60.0, 1)
        voiding = action == ACTION_VOID
        entries.append(
            {
                "rec_id": row["rec_id"],
                "sport": row["sport"],
                "match_id": row["match_id"],
                "match_status": row.get("match_status"),
                "match_date": row["match_date"],
                "created_at": row["created_at"],
                "minutes_after_kickoff": minutes,
                "bet_type": row.get("bet_type"),
                "selection": row.get("selection"),
                "stake": row.get("recommended_stake"),
                "odds": row.get("odds_at_recommendation"),
                "action": action,
                "old_status": row["status"],
                "old_actual_result": row.get("actual_result"),
                "old_profit_loss": row.get("profit_loss"),
                "old_settled_at": row.get("settled_at"),
                "new_status": "void",
                "new_profit_loss": Decimal("0") if voiding else row.get("profit_loss"),
                "new_actual_result": (
                    new_actual_result(row["status"], row.get("profit_loss")) if voiding else row.get("actual_result")
                ),
            }
        )
    return entries


def update_params(entry: dict) -> dict:
    return {
        "rec_id": entry["rec_id"],
        "match_id": entry["match_id"],
        "old_status": entry["old_status"],
        "old_pl": entry["old_profit_loss"],
        "old_result": entry["old_actual_result"],
        "new_result": entry["new_actual_result"],
    }


def summarize(entries: list[dict]) -> dict[str, dict[str, Any]]:
    """Per-sport counts and the P&L this correction removes from the book."""
    per: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "void_won": 0,
            "void_lost": 0,
            "void_void": 0,
            "already_void": 0,
            "stake": Decimal("0"),
            "won_pl": Decimal("0"),
            "lost_pl": Decimal("0"),
            "pl_removed": Decimal("0"),
        }
    )
    for e in entries:
        c = per[e["sport"]]
        if e["action"] == ACTION_ALREADY_VOID:
            c["already_void"] += 1
            continue
        pl = Decimal(str(e["old_profit_loss"] or 0))
        c[f"void_{e['old_status']}"] += 1
        c["stake"] += Decimal(str(e["stake"] or 0))
        c["pl_removed"] += pl
        if e["old_status"] == "won":
            c["won_pl"] += pl
        elif e["old_status"] == "lost":
            c["lost_pl"] += pl
    return dict(per)


def log_summary(per_sport: dict[str, dict[str, Any]], applied: bool) -> None:
    header = (
        f"{'sport':<8}{'won':>6}{'lost':>6}{'void':>6}{'ok_void':>9}"
        f"{'stake':>12}{'won_pl':>12}{'lost_pl':>12}{'pl_removed':>12}"
    )
    row_fmt = "%-8s%6d%6d%6d%9d%12s%12s%12s%12s"
    logger.info(
        "%s summary (won/lost/void = rows %s; ok_void = already void at 0.00, untouched):",
        "APPLIED" if applied else "DRY-RUN",
        "voided" if applied else "to void",
    )
    logger.info(header)
    totals = {"void_won": 0, "void_lost": 0, "void_void": 0, "already_void": 0}
    money = {"stake": Decimal("0"), "won_pl": Decimal("0"), "lost_pl": Decimal("0"), "pl_removed": Decimal("0")}
    for sport, c in sorted(per_sport.items()):
        logger.info(
            row_fmt,
            sport,
            c["void_won"],
            c["void_lost"],
            c["void_void"],
            c["already_void"],
            _fmt_pl(c["stake"]),
            _fmt_pl(c["won_pl"]),
            _fmt_pl(c["lost_pl"]),
            _fmt_pl(c["pl_removed"]),
        )
        for k in totals:
            totals[k] += c[k]
        for k in money:
            money[k] += c[k]
    logger.info(
        row_fmt,
        "TOTAL",
        totals["void_won"],
        totals["void_lost"],
        totals["void_void"],
        totals["already_void"],
        _fmt_pl(money["stake"]),
        _fmt_pl(money["won_pl"]),
        _fmt_pl(money["lost_pl"]),
        _fmt_pl(money["pl_removed"]),
    )
    logger.info(
        "P&L %s from the book: %s (won %s, lost %s) — the book's net P&L moves by %s",
        "removed" if applied else "to remove",
        _fmt_pl(money["pl_removed"]),
        _fmt_pl(money["won_pl"]),
        _fmt_pl(money["lost_pl"]),
        _fmt_pl(-money["pl_removed"]),
    )


# ── CSV ───────────────────────────────────────────────────────────────


def default_out_csv(sports: list[str]) -> str:
    """Timestamped and scoped so two runs never collide; absolute, in the
    volume backup_postgres.py writes to."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    scope = SPORT_ALL if list(sports) == list(ALLOWED_SPORTS) else "-".join(sports)
    base = os.environ.get("BACKUP_LOCAL_DIR", "/app/backups")
    return f"{base}/void_post_kickoff_recs_{scope}_{stamp}.csv"


def write_csv(path: str, entries: list[dict]) -> str:
    """Write every planned row BEFORE any UPDATE. Opened "x" (exclusive
    create): this file is the only reversal record, so an existing path is
    an error, never an overwrite."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for e in entries:
            writer.writerow({k: ("" if e.get(k) is None else str(e.get(k))) for k in CSV_FIELDS})
    logger.info("Wrote %d planned row(s) to %s", len(entries), out)
    return str(out)


# ── Orchestration ─────────────────────────────────────────────────────


def run(
    conn,
    sports: Iterable[str],
    apply_changes: bool = False,
    out_csv: Optional[str] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    yes: bool = False,
) -> dict:
    scope = resolve_sports(sports)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(SELECT_SQL, (scope,))
        rows = [dict(r) for r in cur.fetchall()]
    conn.rollback()  # close the read snapshot before writing

    entries = plan(rows)
    to_void = [e for e in entries if e["action"] == ACTION_VOID]
    if len(to_void) > max_rows and not yes:
        raise MaxRowsExceeded(
            f"{len(to_void)} rec(s) to void (> --max-rows {max_rows}). The known set is 22 — inspect the "
            f"candidate query before rewriting settled money. Re-run with --yes to proceed anyway."
        )

    for e in entries:
        log = logger.warning if e["action"] == ACTION_VOID else logger.info
        log(
            "%s rec %s (%s, match %s): %s %s %s, written %s = %.1f min after kickoff %s -> %s",
            e["action"],
            e["rec_id"],
            e["sport"],
            e["match_id"],
            e["bet_type"],
            e["selection"],
            e["old_status"],
            e["created_at"],
            e["minutes_after_kickoff"],
            e["match_date"],
            e["new_actual_result"],
        )

    csv_path = write_csv(out_csv or default_out_csv(scope), entries)
    per_sport = summarize(entries)

    written = 0
    if not to_void:
        logger.info("Nothing to void for %s", ", ".join(scope))
    elif not apply_changes:
        logger.info("DRY-RUN: %d rec(s) would be voided. Re-run with --apply.", len(to_void))
    else:
        try:
            with conn.cursor() as cur:
                for e in to_void:
                    cur.execute(UPDATE_SQL, update_params(e))
                    if cur.rowcount != 1:
                        raise RuntimeError(
                            f"rec {e['rec_id']} changed between plan and write (rowcount {cur.rowcount}) — "
                            f"rolled back, nothing written"
                        )
                    written += 1
            conn.commit()
        except Exception:
            conn.rollback()
            logger.error("Apply failed after %d row(s); transaction rolled back", written, exc_info=True)
            raise
        logger.info("Voided %d rec(s) in one transaction", written)

    log_summary(per_sport, applied=bool(apply_changes and to_void))
    return {
        "sports": scope,
        "planned": len(entries),
        "to_void": len(to_void),
        "already_void": len(entries) - len(to_void),
        "written": written,
        "per_sport": per_sport,
        "csv_path": csv_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sport",
        action="append",
        dest="sports",
        required=True,
        help=f"Sport to correct; repeatable. One of {', '.join(ALLOWED_SPORTS)}, or {SPORT_ALL}.",
    )
    p.add_argument("--apply", action="store_true", help="Write (default: dry run).")
    p.add_argument(
        "--out-csv",
        default=None,
        help="Audit CSV (default $BACKUP_LOCAL_DIR or /app/backups, timestamped). Write-once.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"Abort if more than this many recs would be voided (default {DEFAULT_MAX_ROWS}).",
    )
    p.add_argument("--yes", action="store_true", help="Proceed past the --max-rows guard.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set (pass --database-url or export DATABASE_URL)")
        return 2
    try:
        resolve_sports(args.sports)
    except UnsafeRequest as exc:
        logger.error("%s", exc)
        return 2
    conn = psycopg2.connect(args.database_url)
    try:
        result = run(
            conn,
            sports=args.sports,
            apply_changes=args.apply,
            out_csv=args.out_csv,
            max_rows=args.max_rows,
            yes=args.yes,
        )
    except (UnsafeRequest, MaxRowsExceeded) as exc:
        logger.error("%s", exc)
        return 2
    finally:
        conn.close()
    logger.info(
        "Done: planned=%d to_void=%d already_void=%d written=%d csv=%s",
        result["planned"],
        result["to_void"],
        result["already_void"],
        result["written"],
        result["csv_path"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
