"""Settle recommendations stranded on duplicate ("orphan") fixture rows.

THE BUG THIS REPAIRS
--------------------
`matches` is keyed UNIQUE (home_team_id, away_team_id, match_date) and
`scripts/fetch_upcoming.py` upserts on that exact triple, storing no ESPN
event id. ESPN slides scheduled kickoff times (tennis every ~15 min,
soccer/MMA by minutes to hours), so ONE real fixture becomes SEVERAL
`matches` rows. Predictions, odds and recommendations land on an early
row; the result lands on a later row. `scripts/grade_completed_matches.py`
only walks status='finished' rows, so recommendations sitting on the
abandoned row stay status='pending' forever.

WHAT THIS DOES
--------------
For every stale pending rec (rec.status='pending', match.status='scheduled',
match_date older than --stale-after-hours) it looks for the fixture's
finished TWIN: a `matches` row with the SAME home_team_id AND the SAME
away_team_id (identical orientation — a swapped pair is a different
fixture and is never matched), status='finished', both scores present,
kickoff within --twin-window-hours. The nearest such twin supplies the
final score, and the rec is graded by the SAME functions the real
settlement loop uses — `grading_outcomes.grade_rec_selection` and
`grading_outcomes.rec_status_and_pl`. Grading is never re-implemented here.

If more than one twin sits in the window with DIFFERENT scores, or two
twins tie on |Δ kickoff|, the row is AMBIGUOUS: it is skipped, counted and
logged. We never guess. That matters more than the row count: settling the
tennis backlog turns ~1,681 dead rows into the first honest measurement of
the tennis recommendation stream (only n=21 have ever settled), and a
guessed settlement would silently poison that readout.

A settled rec is REPOINTED onto its twin (`match_id = twin_id`) in the
same UPDATE. Without that the rec keeps pointing at a status='scheduled'
row and stays invisible to the only ROI readout in the system — the recs
P&L query in `services/api/src/routes/accuracy.py` joins `matches` and
filters `m.status = 'finished'`. Settling without repointing produces a
summary table that no product surface agrees with. The orphan id stays in
the CSV's `match_id` column, so the move is reversible.

Because N orphan rows of one real fixture each carry their own duplicate
rec, the summary also prints the COLLAPSE FACTOR (distinct twins, and how
many settled rows are duplicates of an already-covered
(twin, bet_type, selection)). Read that before quoting an n.

WHY nba/nhl ARE NOT IN THE DEFAULT --sport LIST
-----------------------------------------------
Do NOT "helpfully" add them. The stranded nba/nhl rows (18 as of
2026-09-04) have NO twin at all: they are June-2026 Finals fixtures whose
RESULTS were simply never ingested, because results ingestion only shipped
2026-07-13 and runs with --days-back 3. Those matches are RECOVERABLE by a
results backfill — running this tool with --void-no-twin over nba/nhl would
destroy recs that a backfill can settle honestly. Fix the data first; the
repair tool is not the remedy for missing results. That rule is ENFORCED in
code (see RECOVERABLE_BY_BACKFILL), not left to whoever reads this docstring.

Every affected row is written to --out-csv BEFORE any UPDATE runs (in
dry-run too), mirroring backups/voided_rec_ids_2026-07-13.csv and
backups/gate_voided_ids_2026-07-13.csv. The CSV is the ONLY reversal record
for the rewritten rows (a repaired row is indistinguishable from a natively
settled one inside `betting_recommendations`), so it is write-once: the
default name carries a full UTC timestamp and the sport scope, and an
explicit --out-csv that already exists raises FileExistsError rather than
silently destroying the previous run's audit trail. Each row carries the
twin's scores, so every settlement is re-gradable offline without the DB.

Usage (inside the api container):
    python /app/scripts/repair_stranded_recs.py                    # dry-run
    python /app/scripts/repair_stranded_recs.py --apply
    python /app/scripts/repair_stranded_recs.py --sport tennis --apply
    python /app/scripts/repair_stranded_recs.py --void-no-twin --apply

Each run writes its OWN timestamped CSV, so the dry-run / apply / verify
sequence above never destroys an earlier audit file. Acceptance after an
apply is the product, not this tool's own summary:
    GET /accuracy/summary?sport=tennis&days=365   -> settled must exceed 21
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Make the shared grading-outcomes helper importable (same idiom as
# scripts/grade_completed_matches.py).
sys.path.insert(0, os.path.dirname(__file__))

from grading_outcomes import grade_rec_selection, rec_status_and_pl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("repair_stranded_recs")


# Deliberately excludes nba/nhl — see the module docstring.
DEFAULT_SPORTS = ("soccer", "tennis", "mma")

# Sports this tool understands at all. A typo (`--sport tenis`) must fail
# loudly, not silently repair nothing.
ALLOWED_SPORTS = ("soccer", "tennis", "mma", "nba", "nhl")

# Sports whose stranded recs have NO twin because their RESULTS were never
# ingested, not because the fixture row was duplicated. Voiding these
# destroys rows a results backfill can settle honestly — refused in code.
RECOVERABLE_BY_BACKFILL = frozenset({"nba", "nhl"})

# Bounds on the interval knobs. Both feed SQL interval arithmetic:
# a NEGATIVE --stale-after-hours flips `NOW() - interval '-24 hours'` into
# NOW() + 24h and pulls FUTURE, in-play fixtures into the candidate set;
# an unbounded --twin-window-hours reaches a rematch or a next-season
# meeting of the same ordered pair and settles from the wrong fixture.
MIN_STALE_AFTER_HOURS = 1
MIN_TWIN_WINDOW_HOURS = 1
MAX_TWIN_WINDOW_HOURS = 168

# Real duplicate rows sit minutes-to-hours apart (tennis slides ~15 min).
# Anything further is settled, but LOUDLY, so a far twin is visible in the
# run output and not only in the CSV.
FAR_TWIN_WARN_HOURS = 6.0

VOID_ACTUAL_RESULT = "void: match never resulted (fixture-identity duplicate)"

ACTION_SETTLE = "settle"
ACTION_VOID = "void"

REASON_OK = "ok"
REASON_NO_TWIN = "no_twin"
REASON_AMBIGUOUS_SCORES = "ambiguous_scores"
REASON_AMBIGUOUS_TIE = "ambiguous_delta_tie"

CSV_COLUMNS = [
    "rec_id",
    "match_id",  # the orphan row the rec was generated on (reversal target)
    "twin_id",
    "new_match_id",  # where the rec now points after the repair
    "sport",
    "bet_type",
    "selection",
    "odds",
    "stake",
    "outcome",
    "status",
    "profit_loss",
    "action",
    "twin_delta_hours",
    "twin_home_score",  # grading inputs, so every row is re-gradable offline
    "twin_away_score",
    "twin_match_date",
    "twin_count",  # finished twins seen in the window for this orphan
    "duplicate_of_group",  # 1 = another rec already covers (twin, bet, selection)
]


class MaxRowsExceeded(RuntimeError):
    """Candidate set is larger than --max-rows and --yes was not passed."""


class UnsafeRequest(RuntimeError):
    """Arguments that would make the repair unsafe or meaningless."""


# ── DB I/O ────────────────────────────────────────────────────────────


def list_candidates(cur, sport: str, stale_after_hours: int) -> list[dict]:
    """Stale pending recs for one sport, with the orphan match's identity
    columns attached. `matches` carries no sport column — sport lives on
    `leagues`, so the join is load-bearing."""
    cur.execute(
        """
        SELECT br.id::text AS rec_id,
               br.match_id::text AS match_id,
               br.bet_type,
               br.selection,
               br.recommended_stake,
               br.odds_at_recommendation,
               m.home_team_id::text AS home_team_id,
               m.away_team_id::text AS away_team_id,
               m.match_date,
               l.sport
        FROM betting_recommendations br
        JOIN matches m ON m.id = br.match_id
        JOIN leagues l ON l.id = m.league_id
        WHERE br.status = 'pending'
          AND m.status = 'scheduled'
          AND m.match_date < NOW() - (%s || ' hours')::interval
          AND l.sport = %s
        ORDER BY m.match_date ASC, br.id ASC
        """,
        (str(stale_after_hours), sport),
    )
    return [dict(r) for r in cur.fetchall()]


def find_twins(
    cur,
    match_id: str,
    home_team_id: str,
    away_team_id: str,
    match_date: datetime,
    twin_window_hours: int,
) -> list[dict]:
    """Finished twins of one orphan fixture: SAME home_team_id AND SAME
    away_team_id (orientation is part of fixture identity — a swapped pair
    is a different fixture), both scores present, kickoff within the
    window, and not the orphan row itself."""
    cur.execute(
        """
        SELECT t.id::text AS twin_id,
               t.home_team_id::text AS home_team_id,
               t.away_team_id::text AS away_team_id,
               t.home_score,
               t.away_score,
               t.match_date
        FROM matches t
        WHERE t.home_team_id = %s
          AND t.away_team_id = %s
          AND t.id <> %s
          AND t.status = 'finished'
          AND t.home_score IS NOT NULL
          AND t.away_score IS NOT NULL
          AND t.match_date >= %s::timestamptz - (%s || ' hours')::interval
          AND t.match_date <= %s::timestamptz + (%s || ' hours')::interval
        ORDER BY t.match_date ASC
        """,
        (
            home_team_id,
            away_team_id,
            match_id,
            match_date,
            str(twin_window_hours),
            match_date,
            str(twin_window_hours),
        ),
    )
    rows = [dict(r) for r in cur.fetchall()]
    return verify_orientation(rows, match_id, home_team_id, away_team_id)


def settle_rec(
    cur,
    rec_id: str,
    new_status: str,
    actual_result: str,
    profit_loss: Decimal,
    new_match_id: str,
) -> int:
    """Apply one settlement and REPOINT the rec onto the twin. Returns the
    UPDATE's rowcount — the caller MUST check it.

    Two load-bearing details:

    * The `AND status = 'pending'` guard is MANDATORY: a concurrent
      grade_completed_matches run (or the migration-011 trigger) may have
      settled the row between our read and this write, and clobbering a
      real settlement with a twin-derived one would be silent corruption.
      A 0 rowcount is that guard firing, not success.
    * `match_id = %s` moves the rec off the abandoned scheduled row onto
      the finished twin. accuracy.py's recs P&L query filters
      `m.status = 'finished'`, so without this the repaired rows settle
      into a table nothing reads.
    """
    cur.execute(
        """
        UPDATE betting_recommendations
        SET status = %s,
            actual_result = %s,
            profit_loss = %s,
            match_id = %s,
            settled_at = NOW(),
            updated_at = NOW()
        WHERE id = %s
          AND status = 'pending'
        """,
        (new_status, actual_result, profit_loss, new_match_id, rec_id),
    )
    return cur.rowcount


def void_rec(cur, rec_id: str) -> int:
    """Void a rec whose fixture never resulted at all. Same mandatory
    status guard as settle_rec; returns the rowcount for the same reason.
    No repoint: there is no twin to point at."""
    cur.execute(
        """
        UPDATE betting_recommendations
        SET status = 'void',
            actual_result = %s,
            profit_loss = 0,
            settled_at = NOW(),
            updated_at = NOW()
        WHERE id = %s
          AND status = 'pending'
        """,
        (VOID_ACTUAL_RESULT, rec_id),
    )
    return cur.rowcount


# ── Pure twin selection ───────────────────────────────────────────────


def verify_orientation(twins: Iterable[dict], match_id: str, home_team_id: str, away_team_id: str) -> list[dict]:
    """Re-check orientation in Python rather than trusting the WHERE
    clause. Duplicate-row orientation was identical in 100% of 1,638
    observed tennis twin pairs, but a swapped pair would settle every rec
    backwards, so the assumption is verified per row and violations are
    LOUD."""
    kept = []
    for t in twins:
        if t.get("home_team_id") != home_team_id or t.get("away_team_id") != away_team_id:
            logger.error(
                "Orientation mismatch: orphan match %s (%s vs %s) got twin %s (%s vs %s) — dropping it",
                match_id,
                home_team_id,
                away_team_id,
                t.get("twin_id"),
                t.get("home_team_id"),
                t.get("away_team_id"),
            )
            continue
        if t.get("twin_id") == match_id:
            logger.error("Twin query returned the orphan row itself (%s) — dropping it", match_id)
            continue
        kept.append(t)
    return kept


def _delta_seconds(twin: dict, match_date: datetime) -> float:
    return abs((twin["match_date"] - match_date).total_seconds())


def select_twin(twins: list[dict], match_date: datetime) -> tuple[Optional[dict], str]:
    """(twin, reason). Returns (None, reason) whenever the fixture's real
    result is not unambiguous:

      * no finished twin in the window            -> no_twin
      * twins disagree on the score               -> ambiguous_scores
      * two twins tie on |Δ kickoff|              -> ambiguous_delta_tie

    Only a single, unanimous, strictly-nearest twin settles a row.
    """
    if not twins:
        return None, REASON_NO_TWIN

    scores = {(t["home_score"], t["away_score"]) for t in twins}
    if len(scores) > 1:
        return None, REASON_AMBIGUOUS_SCORES

    ranked = sorted(twins, key=lambda t: (_delta_seconds(t, match_date), str(t["twin_id"])))
    if len(ranked) > 1 and _delta_seconds(ranked[0], match_date) == _delta_seconds(ranked[1], match_date):
        return None, REASON_AMBIGUOUS_TIE

    return ranked[0], REASON_OK


# ── Planning ──────────────────────────────────────────────────────────


def _new_counts() -> dict[str, Any]:
    return {
        "candidates": 0,
        "settled_won": 0,
        "settled_lost": 0,
        "settled_push_void": 0,
        "ambiguous_skipped": 0,
        "ungradable_skipped": 0,
        "no_twin": 0,
        "no_twin_voided": 0,
        # Rows planned but NOT written because the rec was no longer
        # pending at apply time (concurrent settlement).
        "skipped_not_pending": 0,
        # Collapse factor: distinct real fixtures behind the settled rows,
        # and how many settled rows duplicate an already-covered
        # (twin, bet_type, selection). n is inflated by the latter.
        "distinct_twins": 0,
        "duplicate_settles": 0,
        "net_pl": Decimal("0"),
        "total_stake": Decimal("0"),
    }


def plan_sport(
    cur,
    sport: str,
    twin_window_hours: int,
    stale_after_hours: int,
    void_no_twin: bool,
) -> tuple[list[dict], dict[str, Any]]:
    """Read-only pass: decide what would happen to every candidate rec of
    one sport. Returns (affected_entries, counts)."""
    counts = _new_counts()
    entries: list[dict] = []
    twin_cache: dict[str, list[dict]] = {}
    settled_twins: set[str] = set()
    seen_groups: set[tuple[str, str, str]] = set()

    candidates = list_candidates(cur, sport, stale_after_hours)
    counts["candidates"] = len(candidates)

    for rec in candidates:
        match_id = rec["match_id"]
        if match_id not in twin_cache:
            twin_cache[match_id] = find_twins(
                cur,
                match_id,
                rec["home_team_id"],
                rec["away_team_id"],
                rec["match_date"],
                twin_window_hours,
            )
        twin, reason = select_twin(twin_cache[match_id], rec["match_date"])

        if twin is None:
            if reason == REASON_NO_TWIN:
                counts["no_twin"] += 1
                if not void_no_twin:
                    continue
                counts["no_twin_voided"] += 1
                entries.append(
                    {
                        "rec_id": rec["rec_id"],
                        "match_id": match_id,
                        "twin_id": "",
                        "new_match_id": match_id,
                        "sport": sport,
                        "bet_type": rec["bet_type"],
                        "selection": rec["selection"],
                        "odds": rec["odds_at_recommendation"],
                        "stake": rec["recommended_stake"],
                        "outcome": "",
                        "status": "void",
                        "profit_loss": Decimal("0"),
                        "action": ACTION_VOID,
                        "twin_delta_hours": "",
                        "twin_home_score": "",
                        "twin_away_score": "",
                        "twin_match_date": "",
                        "twin_count": 0,
                        "duplicate_of_group": 0,
                    }
                )
                continue
            counts["ambiguous_skipped"] += 1
            logger.warning(
                "AMBIGUOUS (%s): rec %s on match %s (%s) has %d finished twins in the window — skipping",
                reason,
                rec["rec_id"],
                match_id,
                sport,
                len(twin_cache[match_id]),
            )
            continue

        outcome = grade_rec_selection(
            rec["bet_type"],
            rec["selection"],
            twin["home_score"],
            twin["away_score"],
        )
        if outcome is None:
            counts["ungradable_skipped"] += 1
            logger.warning(
                "UNGRADABLE: rec %s bet_type=%r selection=%r (%s) — leaving pending",
                rec["rec_id"],
                rec["bet_type"],
                rec["selection"],
                sport,
            )
            continue

        status, profit_loss = rec_status_and_pl(
            outcome,
            rec["recommended_stake"],
            rec["odds_at_recommendation"],
        )
        if status == "won":
            counts["settled_won"] += 1
        elif status == "lost":
            counts["settled_lost"] += 1
        else:
            counts["settled_push_void"] += 1
        counts["net_pl"] += profit_loss
        counts["total_stake"] += Decimal(str(rec["recommended_stake"] or 0))

        delta_hours = round(_delta_seconds(twin, rec["match_date"]) / 3600.0, 3)
        if delta_hours > FAR_TWIN_WARN_HOURS:
            logger.warning(
                "FAR TWIN (%.2fh, expected minutes-to-hours): rec %s on match %s (%s) settled from twin %s — "
                "verify this is the same fixture and not a rematch",
                delta_hours,
                rec["rec_id"],
                match_id,
                sport,
                twin["twin_id"],
            )

        # Fan-in: several orphan rows of ONE real fixture each carry their
        # own duplicate rec. Every one of them settles correctly, but the
        # aggregate counts one real bet N times — flag the repeats so the
        # operator can subtract them before quoting an n.
        group = (str(twin["twin_id"]), str(rec["bet_type"]), str(rec["selection"]))
        duplicate = group in seen_groups
        seen_groups.add(group)
        settled_twins.add(str(twin["twin_id"]))
        if duplicate:
            counts["duplicate_settles"] += 1

        entries.append(
            {
                "rec_id": rec["rec_id"],
                "match_id": match_id,
                "twin_id": twin["twin_id"],
                "new_match_id": twin["twin_id"],
                "sport": sport,
                "bet_type": rec["bet_type"],
                "selection": rec["selection"],
                "odds": rec["odds_at_recommendation"],
                "stake": rec["recommended_stake"],
                "outcome": outcome,
                "status": status,
                "profit_loss": profit_loss,
                "action": ACTION_SETTLE,
                "twin_delta_hours": delta_hours,
                "twin_home_score": twin["home_score"],
                "twin_away_score": twin["away_score"],
                "twin_match_date": twin["match_date"],
                "twin_count": len(twin_cache[match_id]),
                "duplicate_of_group": int(duplicate),
            }
        )

    counts["distinct_twins"] = len(settled_twins)
    settle_rows = sum(1 for e in entries if e["action"] == ACTION_SETTLE)
    if settle_rows:
        logger.info(
            "%s collapse factor: %d settle row(s) cover %d distinct fixture(s); "
            "%d row(s) duplicate an already-covered (twin, bet_type, selection)",
            sport,
            settle_rows,
            counts["distinct_twins"],
            counts["duplicate_settles"],
        )

    return entries, counts


# ── CSV ───────────────────────────────────────────────────────────────


def write_csv(path: str, entries: list[dict]) -> str:
    """Write every affected row BEFORE any UPDATE runs — dry-run included.
    Without this file an operator cannot reverse a bad repair.

    Opened "x" (exclusive create) on purpose. A repaired row is
    indistinguishable from a natively settled one inside
    `betting_recommendations`, so this file is the ONLY reversal record;
    the natural follow-up action — a verification dry-run after the apply —
    would otherwise rewrite it as a header-only file and destroy it.
    FileExistsError is the correct, loud outcome."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for e in entries:
            writer.writerow({k: ("" if e.get(k) is None else str(e.get(k))) for k in CSV_COLUMNS})
    logger.info("Wrote %d affected row(s) to %s", len(entries), out)
    return str(out)


# ── Apply ─────────────────────────────────────────────────────────────


def apply_entries(conn, entries: list[dict]) -> tuple[int, list[dict]]:
    """Execute one sport's UPDATEs inside a single transaction. Any error
    rolls the whole sport back and re-raises — a half-settled sport is
    worse than an unsettled one.

    Returns (applied, skipped). A 0 rowcount means the mandatory
    `AND status = 'pending'` guard fired: something settled the row
    between the plan snapshot and this write. Counting that as applied
    would put a settlement this tool never made into the reversal CSV and
    into the P&L summary — exactly the silent-failure class the guard
    exists to catch."""
    applied = 0
    skipped: list[dict] = []
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for e in entries:
                if e["action"] == ACTION_VOID:
                    rowcount = void_rec(cur, e["rec_id"])
                else:
                    rowcount = settle_rec(
                        cur,
                        e["rec_id"],
                        e["status"],
                        e["outcome"],
                        e["profit_loss"],
                        e["twin_id"],
                    )
                if rowcount == 1:
                    applied += 1
                    continue
                logger.error(
                    "rec %s was no longer pending at apply time (rowcount=%s) — NOT repaired "
                    "(planned status=%s, twin=%s). Removing it from the summary and the audit CSV.",
                    e["rec_id"],
                    rowcount,
                    e["status"],
                    e["twin_id"] or "-",
                )
                skipped.append(e)
        conn.commit()
    except Exception:
        conn.rollback()
        logger.error("Apply failed after %d row(s); transaction rolled back", applied, exc_info=True)
        raise
    return applied, skipped


def _discount_skipped(counts: dict[str, Any], skipped: list[dict]) -> None:
    """Back out rows the DB refused from the plan-phase counters, so the
    printed P&L describes what this tool actually wrote."""
    for e in skipped:
        counts["skipped_not_pending"] += 1
        if e["action"] == ACTION_VOID:
            counts["no_twin_voided"] -= 1
            continue
        if e["status"] == "won":
            counts["settled_won"] -= 1
        elif e["status"] == "lost":
            counts["settled_lost"] -= 1
        else:
            counts["settled_push_void"] -= 1
        counts["net_pl"] -= Decimal(str(e["profit_loss"] or 0))
        counts["total_stake"] -= Decimal(str(e["stake"] or 0))


# ── Orchestration ─────────────────────────────────────────────────────


def _fmt_money(value: Decimal) -> str:
    return f"{Decimal(value):.2f}"


def print_summary(per_sport: dict[str, dict], applied: bool) -> None:
    header = (
        f"{'sport':<10}{'cand':>7}{'won':>7}{'lost':>7}{'void':>7}"
        f"{'ambig':>8}{'ungrad':>8}{'no_twin':>9}{'voided':>8}{'skipped':>9}"
        f"{'fixtures':>10}{'dups':>7}{'stake':>12}{'net_pl':>12}"
    )
    row_fmt = "%-10s%7d%7d%7d%7d%8d%8d%9d%8d%9d%10d%7d%12s%12s"
    logger.info("%s run summary:", "APPLIED" if applied else "DRY-RUN")
    logger.info(header)
    logger.info("-" * len(header))
    totals = _new_counts()
    for sport, c in per_sport.items():
        logger.info(
            row_fmt,
            sport,
            c["candidates"],
            c["settled_won"],
            c["settled_lost"],
            c["settled_push_void"],
            c["ambiguous_skipped"],
            c["ungradable_skipped"],
            c["no_twin"],
            c["no_twin_voided"],
            c["skipped_not_pending"],
            c["distinct_twins"],
            c["duplicate_settles"],
            _fmt_money(c["total_stake"]),
            _fmt_money(c["net_pl"]),
        )
        for k, v in c.items():
            totals[k] += v
    logger.info("-" * len(header))
    logger.info(
        row_fmt,
        "TOTAL",
        totals["candidates"],
        totals["settled_won"],
        totals["settled_lost"],
        totals["settled_push_void"],
        totals["ambiguous_skipped"],
        totals["ungradable_skipped"],
        totals["no_twin"],
        totals["no_twin_voided"],
        totals["skipped_not_pending"],
        totals["distinct_twins"],
        totals["duplicate_settles"],
        _fmt_money(totals["total_stake"]),
        _fmt_money(totals["net_pl"]),
    )
    if totals["duplicate_settles"]:
        logger.warning(
            "%d settled row(s) are duplicate recs of a fixture already covered — the honest bet count is "
            "%d, not %d. Subtract them before quoting an ROI n.",
            totals["duplicate_settles"],
            totals["settled_won"] + totals["settled_lost"] + totals["settled_push_void"] - totals["duplicate_settles"],
            totals["settled_won"] + totals["settled_lost"] + totals["settled_push_void"],
        )
    if totals["skipped_not_pending"]:
        logger.error(
            "%d planned row(s) were NOT written (no longer pending at apply time) — see the errors above and "
            "the .not_applied.csv companion file.",
            totals["skipped_not_pending"],
        )


def run(
    conn,
    sports: Iterable[str],
    twin_window_hours: int = 72,
    stale_after_hours: int = 24,
    void_no_twin: bool = False,
    max_rows: int = 3000,
    yes: bool = False,
    apply_changes: bool = False,
    out_csv: Optional[str] = None,
) -> dict:
    """Plan every sport read-only, write the CSV, then (only with
    apply_changes) execute one transaction per sport."""
    sports = list(sports)
    validate_request(sports, twin_window_hours, stale_after_hours, void_no_twin)
    out_csv = out_csv or default_out_csv(sports)

    per_sport: dict[str, dict] = {}
    entries_by_sport: dict[str, list[dict]] = {}
    total_candidates = 0

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        for sport in sports:
            entries, counts = plan_sport(cur, sport, twin_window_hours, stale_after_hours, void_no_twin)
            entries_by_sport[sport] = entries
            per_sport[sport] = counts
            total_candidates += counts["candidates"]
            if total_candidates > max_rows and not yes:
                raise MaxRowsExceeded(
                    f"Candidate set is {total_candidates} rows (> --max-rows {max_rows}). "
                    f"That is far more than the known backlog — inspect the candidate query before "
                    f"repairing. Re-run with --yes to proceed anyway."
                )
    # Close the read snapshot before the write transactions start.
    conn.rollback()

    all_entries = [e for sport in sports for e in entries_by_sport[sport]]

    # CSV goes out BEFORE any UPDATE, dry-run included.
    csv_path = write_csv(out_csv, all_entries)

    applied = 0
    skipped_entries: list[dict] = []
    if apply_changes:
        for sport in sports:
            entries = entries_by_sport[sport]
            if not entries:
                continue
            n_applied, skipped = apply_entries(conn, entries)
            applied += n_applied
            if skipped:
                _discount_skipped(per_sport[sport], skipped)
                skipped_entries.extend(skipped)
            logger.info(
                "Applied %d of %d planned row(s) for %s (%d skipped: no longer pending)",
                n_applied,
                len(entries),
                sport,
                len(skipped),
            )
    else:
        logger.info("DRY-RUN: %d row(s) would be written. Re-run with --apply.", len(all_entries))

    # The audit CSV must describe what LANDED. Rows the DB refused get
    # their own companion file rather than a silent edit of the original.
    not_applied_csv = None
    if skipped_entries:
        not_applied_csv = write_csv(str(Path(csv_path).with_suffix(".not_applied.csv")), skipped_entries)

    print_summary(per_sport, applied=apply_changes)

    return {
        "per_sport": per_sport,
        "total_candidates": total_candidates,
        "affected": len(all_entries),
        "applied": applied,
        "skipped_not_pending": len(skipped_entries),
        "csv_path": csv_path,
        "not_applied_csv_path": not_applied_csv,
    }


def validate_request(
    sports: list[str],
    twin_window_hours: int,
    stale_after_hours: int,
    void_no_twin: bool,
) -> None:
    """Refuse arguments that would make the repair unsafe. Enforced here,
    in the tested entry point, rather than in argparse defaults — the
    nba/nhl rule in particular was documentation-only, so
    `--sport nhl --void-no-twin --apply` was accepted."""
    unknown = [s for s in sports if s not in ALLOWED_SPORTS]
    if unknown:
        raise UnsafeRequest(f"Unknown sport(s) {unknown}. Allowed: {', '.join(ALLOWED_SPORTS)}.")
    if not sports:
        raise UnsafeRequest("No sports selected.")
    if stale_after_hours < MIN_STALE_AFTER_HOURS:
        raise UnsafeRequest(
            f"--stale-after-hours must be >= {MIN_STALE_AFTER_HOURS} (got {stale_after_hours}). "
            f"A value <= 0 pulls future / in-play fixtures into the candidate set."
        )
    if not MIN_TWIN_WINDOW_HOURS <= twin_window_hours <= MAX_TWIN_WINDOW_HOURS:
        raise UnsafeRequest(
            f"--twin-window-hours must be between {MIN_TWIN_WINDOW_HOURS} and {MAX_TWIN_WINDOW_HOURS} "
            f"(got {twin_window_hours}). A wide window reaches a rematch or a next-season fixture between "
            f"the same ordered pair and settles from the wrong score."
        )
    blocked = sorted(set(sports) & RECOVERABLE_BY_BACKFILL)
    if void_no_twin and blocked:
        raise UnsafeRequest(
            f"--void-no-twin is refused for {', '.join(blocked)}: those stranded recs have no twin because "
            f"their RESULTS were never ingested, not because the fixture row was duplicated. They are "
            f"recoverable — run the results backfill first "
            f"(scripts/fetch_upcoming.py --results --days-back N). Voiding them destroys settleable rows."
        )


def default_out_csv(sports: Optional[Iterable[str]] = None) -> str:
    """Timestamped + scoped so two runs on the same day cannot collide.
    The old date-only name meant a verification dry-run after an apply
    silently overwrote the only reversal record with an empty file."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    scope = "-".join(sorted(sports)) if sports else "all"
    return f"backups/repaired_rec_ids_{scope}_{stamp}.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sport",
        action="append",
        dest="sports",
        default=None,
        help=f"Sport to repair; repeatable. Default: {', '.join(DEFAULT_SPORTS)}. "
        f"Allowed: {', '.join(ALLOWED_SPORTS)}. nba/nhl are excluded from the default on purpose — "
        "their results are recoverable by a backfill, and --void-no-twin is refused for them.",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument(
        "--twin-window-hours",
        type=int,
        default=72,
        help=f"Max |Δ kickoff| to a twin (default 72, allowed {MIN_TWIN_WINDOW_HOURS}-{MAX_TWIN_WINDOW_HOURS}).",
    )
    p.add_argument(
        "--stale-after-hours",
        type=int,
        default=24,
        help=f"A pending rec counts as stranded this long after kickoff (default 24, minimum "
        f"{MIN_STALE_AFTER_HOURS}).",
    )
    p.add_argument(
        "--void-no-twin",
        action="store_true",
        help="Also void recs whose fixture has NO finished twin. OFF by default; refused for nba/nhl.",
    )
    p.add_argument("--max-rows", type=int, default=3000, help="Abort if the candidate set exceeds this (default 3000).")
    p.add_argument("--yes", action="store_true", help="Proceed past the --max-rows guard.")
    p.add_argument(
        "--out-csv",
        default=None,
        help="CSV of affected rows (default backups/repaired_rec_ids_<sports>_<utc timestamp>.csv). "
        "Write-once: an existing path is an error, never an overwrite.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan only (DEFAULT).")
    mode.add_argument("--apply", action="store_true", help="Execute the UPDATEs.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set (pass --database-url or export DATABASE_URL)")
        return 2

    sports = args.sports or list(DEFAULT_SPORTS)

    # Refuse unsafe arguments BEFORE opening a connection — run() re-checks,
    # but a bad invocation should not even touch the database.
    try:
        validate_request(sports, args.twin_window_hours, args.stale_after_hours, args.void_no_twin)
    except UnsafeRequest as exc:
        logger.error("%s", exc)
        return 2

    conn = psycopg2.connect(args.database_url)
    try:
        result = run(
            conn,
            sports=sports,
            twin_window_hours=args.twin_window_hours,
            stale_after_hours=args.stale_after_hours,
            void_no_twin=args.void_no_twin,
            max_rows=args.max_rows,
            yes=args.yes,
            apply_changes=args.apply,
            out_csv=args.out_csv,
        )
    except (MaxRowsExceeded, UnsafeRequest) as exc:
        logger.error("%s", exc)
        return 2
    finally:
        conn.close()

    if result["total_candidates"] == 0:
        logger.info("No stranded pending recs found for %s", ", ".join(sports))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
