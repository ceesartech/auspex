"""Retire the four soccer rows football-data wrote as finished matches
that never finished.

promote_raw wrote every football-data row with status 'finished' and copied
HTHG/HTAG into the half-time columns. Four rows have no full-time score
(FTHG/FTAG blank): two abandoned matches whose HTHG/HTAG hold the awarded
0-3 (HTR 'A') rather than a half-time score, and two fixtures that were
never played. The half-time Dixon-Coles trainer and its A/B gate (status
'finished' + half_time_home set) read the two 0-3 rows as real halves.
promote_raw no longer promotes a row without a full-time score, and its
conflict branch leaves a 'cancelled' row alone, so the fix survives a
re-run of the loader.

Per row (evidence read from ESPN's public API on 2026-10-31):
  * status -> 'cancelled': the matches CHECK allows only scheduled / live /
    finished / postponed / cancelled, and an abandoned or awarded match has
    no played score, which is what 'cancelled' means to every reader;
  * the bogus half-time pair -> NULL (the two 0-3 rows);
  * external_ids.espn -> the ESPN competition id where ESPN lists the
    fixture (stored as a string, as fetch_upcoming stores it);
  * metadata.result_note -> what happened and the source.

None of the four has predictions or recommendations; their odds,
match_stats and match_weather rows stay. Status leaves 'finished' rather
than entering it, so the match-outcome trigger does nothing.

Every UPDATE is guarded on the row's observed values (status 'finished',
no full-time score, the half-time pair as seen, no other ESPN id): a re-run
is a no-op and a row that drifted is refused, never overwritten. The four
rows are written in one transaction or not at all. The audit CSV is
write-once and lands before any write, dry run included.

Usage (inside the api container):
    python /app/scripts/fix_unplayed_soccer_rows.py            # dry run
    python /app/scripts/fix_unplayed_soccer_rows.py --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("fix_unplayed_soccer_rows")


@dataclass(frozen=True)
class RowFix:
    match_id: str
    league: str  # football-data code on leagues.external_ids
    home: str
    away: str
    match_day: str  # UTC date of matches.match_date
    observed_ht: Optional[tuple[int, int]]  # the half-time pair the row holds before the fix
    espn_id: Optional[str]  # ESPN competition id, when ESPN lists the fixture
    note: str


FIXES: tuple[RowFix, ...] = (
    RowFix(
        match_id="d7d8a9b7-39a6-4340-bb53-09982171fa8a",
        league="G1",
        home="Panathinaikos",
        away="Olympiakos",
        match_day="2019-03-17",
        observed_ht=(0, 3),
        espn_id="524152",
        note=(
            "Abandoned, not finished: ESPN gre.1 event 524152 is STATUS_ABANDONED at 90'+1' with Olympiakos "
            "1-0 up (goal 53', half-time 0-0). football-data (G1 1819) left FTHG/FTAG blank and put the "
            "awarded 0-3 in HTHG/HTAG (HTR 'A'); neither is a played score. "
            "Set by scripts/fix_unplayed_soccer_rows.py, 2026-10-31."
        ),
    ),
    RowFix(
        match_id="ff74d919-d543-4729-a438-5f59e826b7b9",
        league="F2",
        home="Bastia",
        away="Red Star",
        match_day="2025-12-05",
        observed_ht=(0, 3),
        espn_id="747597",
        note=(
            "Abandoned, not finished: ESPN fra.2 event 747597 stops at 58' (period 2) at 0-0 with no "
            "second-half events and flags Red Star the winner on that 0-0, although it labels the event "
            "STATUS_FULL_TIME. football-data (F2 2526) left FTHG/FTAG blank and put the awarded 0-3 in "
            "HTHG/HTAG (HTR 'A'); neither is a played score. "
            "Set by scripts/fix_unplayed_soccer_rows.py, 2026-10-31."
        ),
    ),
    RowFix(
        match_id="681794ae-f1e5-4e10-a365-8e71f86ef4a1",
        league="G1",
        home="Niki Volos",
        away="OFI",
        match_day="2015-05-10",
        observed_ht=None,
        espn_id=None,
        note=(
            "Not played: ESPN gre.1 lists eight fixtures on 2015-05-10 and none between these clubs. "
            "football-data (G1 1415) carries the fixture with no score at all. "
            "Set by scripts/fix_unplayed_soccer_rows.py, 2026-10-31."
        ),
    ),
    RowFix(
        match_id="83ce848f-64ad-4147-ab32-4a770a5eed15",
        league="BR1",
        home="Chapecoense-SC",
        away="Atletico-MG",
        match_day="2016-12-11",
        observed_ht=None,
        espn_id="445952",
        note=(
            "Cancelled, not played: ESPN bra.1 event 445952 is STATUS_CANCELED. football-data (BR1 2016) "
            "carries the fixture with no score at all. "
            "Set by scripts/fix_unplayed_soccer_rows.py, 2026-10-31."
        ),
    ),
)

SELECT_SQL = """
    SELECT m.id::text AS id,
           l.sport,
           l.external_ids->>'football_data' AS league,
           ht.name AS home,
           at.name AS away,
           (m.match_date AT TIME ZONE 'UTC')::date::text AS match_day,
           m.status, m.home_score, m.away_score, m.half_time_home, m.half_time_away,
           m.external_ids, m.metadata
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams ht ON ht.id = m.home_team_id
    JOIN teams at ON at.id = m.away_team_id
    WHERE m.id = ANY(%s::uuid[])
"""

# Guarded on everything the plan observed; rowcount 0 means the row moved.
UPDATE_SQL = """
    UPDATE matches
    SET status = 'cancelled',
        half_time_home = NULL,
        half_time_away = NULL,
        external_ids = COALESCE(external_ids, '{}'::jsonb) || %(external)s::jsonb,
        metadata = COALESCE(metadata, '{}'::jsonb) || %(metadata)s::jsonb,
        updated_at = NOW()
    WHERE id = %(id)s::uuid
      AND status = 'finished'
      AND home_score IS NULL
      AND away_score IS NULL
      AND half_time_home IS NOT DISTINCT FROM %(ht_home)s
      AND half_time_away IS NOT DISTINCT FROM %(ht_away)s
      AND COALESCE(external_ids->>'espn', '') IN ('', %(espn)s)
"""

CSV_FIELDS = (
    "match_id",
    "league",
    "match_day",
    "home",
    "away",
    "action",
    "reason",
    "status_before",
    "home_score_before",
    "away_score_before",
    "half_time_home_before",
    "half_time_away_before",
    "external_ids_before",
    "metadata_before",
    "status_after",
    "espn_id",
    "result_note",
)


def _espn_of(row: dict) -> Optional[str]:
    return (row.get("external_ids") or {}).get("espn")


def assess(fix: RowFix, row: Optional[dict]) -> tuple[str, str]:
    """('fix' | 'done' | 'refuse', reason) for one planned fix. Pure."""
    if row is None:
        return "refuse", "row not found"
    identity = (row.get("sport"), row.get("league"), row.get("home"), row.get("away"), row.get("match_day"))
    expected = ("soccer", fix.league, fix.home, fix.away, fix.match_day)
    if identity != expected:
        return "refuse", f"not the expected fixture: {identity} != {expected}"
    ht = (row.get("half_time_home"), row.get("half_time_away"))
    no_ft = row.get("home_score") is None and row.get("away_score") is None
    espn = _espn_of(row)
    if (
        row.get("status") == "cancelled"
        and no_ft
        and ht == (None, None)
        and espn == fix.espn_id
        and (row.get("metadata") or {}).get("result_note") == fix.note
    ):
        return "done", "already fixed"
    observed = fix.observed_ht if fix.observed_ht is not None else (None, None)
    if row.get("status") == "finished" and no_ft and ht == observed and espn in (None, fix.espn_id):
        return "fix", ""
    return "refuse", (
        f"drifted since 2026-10-31: status={row.get('status')!r} "
        f"ft={row.get('home_score')!r}-{row.get('away_score')!r} ht={ht[0]!r}-{ht[1]!r} "
        f"espn={espn!r} (expected finished, no ft, ht={observed[0]!r}-{observed[1]!r}, espn None or {fix.espn_id!r})"
    )


def update_params(fix: RowFix) -> dict:
    observed = fix.observed_ht if fix.observed_ht is not None else (None, None)
    return {
        "id": fix.match_id,
        "external": json.dumps({"espn": fix.espn_id} if fix.espn_id else {}),
        "metadata": json.dumps({"result_note": fix.note}),
        "ht_home": observed[0],
        "ht_away": observed[1],
        "espn": fix.espn_id or "",
    }


def plan(rows_by_id: dict[str, dict]) -> list[dict]:
    """One CSV-ready entry per fix, with its assessed action. Pure."""
    entries = []
    for fix in FIXES:
        row = rows_by_id.get(fix.match_id)
        action, reason = assess(fix, row)
        row = row or {}
        entries.append(
            {
                "match_id": fix.match_id,
                "league": fix.league,
                "match_day": fix.match_day,
                "home": fix.home,
                "away": fix.away,
                "action": action,
                "reason": reason,
                "status_before": row.get("status"),
                "home_score_before": row.get("home_score"),
                "away_score_before": row.get("away_score"),
                "half_time_home_before": row.get("half_time_home"),
                "half_time_away_before": row.get("half_time_away"),
                "external_ids_before": json.dumps(row.get("external_ids")),
                "metadata_before": json.dumps(row.get("metadata")),
                "status_after": "cancelled",
                "espn_id": fix.espn_id or "",
                "result_note": fix.note,
            }
        )
    return entries


def default_out_csv() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base = os.environ.get("BACKUP_LOCAL_DIR", "/app/backups")
    return f"{base}/fix_unplayed_soccer_rows_{stamp}.csv"


def write_csv(path: str, entries: list[dict]) -> None:
    out = Path(path)
    if out.exists():
        raise FileExistsError(f"{out} exists — the audit CSV is write-once")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(entries)


def run(conn, apply_changes: bool, out_csv: Optional[str] = None) -> Counter:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(SELECT_SQL, ([f.match_id for f in FIXES],))
        rows_by_id = {r["id"]: dict(r) for r in cur.fetchall()}
    conn.rollback()  # close the read snapshot before writing

    entries = plan(rows_by_id)
    counts = Counter(e["action"] for e in entries)
    out_csv = out_csv or default_out_csv()
    write_csv(out_csv, entries)
    for e in entries:
        log = logger.error if e["action"] == "refuse" else logger.info
        log(
            "%s %s %s v %s %s: %s %s",
            e["league"],
            e["match_day"],
            e["home"],
            e["away"],
            e["match_id"],
            e["action"],
            e["reason"],
        )
    logger.info("Planned %s; audit CSV %s", dict(counts), out_csv)

    if counts["refuse"]:
        logger.error("%d row(s) refused — nothing written; inspect them before re-running", counts["refuse"])
        return counts
    to_fix = [f for f, e in zip(FIXES, entries) if e["action"] == "fix"]
    if not apply_changes:
        logger.info("DRY-RUN: re-run with --apply to fix %d row(s)", len(to_fix))
        return counts
    with conn.cursor() as cur:
        for fix in to_fix:
            cur.execute(UPDATE_SQL, update_params(fix))
            if cur.rowcount != 1:
                conn.rollback()
                raise RuntimeError(
                    f"{fix.match_id} ({fix.home} v {fix.away}) changed between plan and write "
                    f"(rowcount {cur.rowcount}) — rolled back, nothing written"
                )
            counts["written"] += 1
    conn.commit()
    logger.info("Fixed %d row(s)", counts["written"])
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="Write (default: dry run).")
    p.add_argument(
        "--out-csv", default=None, help="Audit CSV (default $BACKUP_LOCAL_DIR or /app/backups, timestamped)."
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    with psycopg2.connect(args.database_url) as conn:
        counts = run(conn, args.apply, args.out_csv)
    logger.info("Done: %s", dict(sorted(counts.items())))
    return 1 if counts["refuse"] else 0


if __name__ == "__main__":
    sys.exit(main())
