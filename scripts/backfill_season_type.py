"""Stamp matches.metadata->>'season_type' on EXISTING rows that lack it.

Why this script exists
----------------------
Every ESPN scoreboard event carries event.season.type (1 = preseason,
2 = regular season, 3 = post-season). scripts/fetch_upcoming.py now
stamps it onto matches.metadata->>'season_type' as
'preseason' | 'regular' | 'postseason', and every downstream reader
(training frames, rolling-form windows, the model monitor) excludes
preseason via utils.training_data.preseason_exclusion_sql.

That predicate deliberately treats a MISSING marker as *unknown* and
keeps the row, because a COALESCE(..., 'regular') would silently
re-admit the 294 known historical preseason games (147 NFL + 147 NBA)
as if they had been verified. This script is the other half of that
decision: it removes the unknowns by stamping historical rows from
per-sport date rules that were checked against prod.

Date rules (each one is a claim about the calendar, not a guess)
----------------------------------------------------------------
NFL
  * August  -> 'preseason'. The NFL regular season has never started
    before September. Verified on prod: 49 August games in each of
    2023, 2024 and 2026 (147 total), none of which has a single odds
    row. Their scoring is structurally different: mean total 37.37
    (sd 11.44, n=147) vs 44.85 (sd 14.22, n=857) for the rest.
  * September / October / November / December -> 'regular'. The NFL
    regular season runs Week 1 (early September) through Week 18
    (early January); no post-season game is ever played before
    January, and no preseason game after August.
  * January / February -> NOTHING. January mixes Weeks 17-18 with the
    Wild Card round and the boundary moves year to year; February is
    the Super Bowl plus the Pro Bowl, which is neither. Reported, not
    stamped.

NBA
  * October, strictly BEFORE that season's opener -> 'preseason'.
    Verified on prod: 73 games on 2023-10-05..21 (opener 2023-10-24)
    and 74 games on 2024-10-04..19 (opener 2024-10-22), 147 total.
    Openers are a documented table below rather than a derived guess;
    an October with no opener in the table is reported, not stamped.
  * Opener .. March 31 -> 'regular'. November through March is
    unambiguously regular season for every modern NBA calendar.
  * April onward -> NOTHING. April mixes the last regular-season week
    with the play-in tournament and the first round; May/June are the
    play-offs. Reported, not stamped.

NHL
  * NOTHING is stamped, ever. scripts/load_nhl_historical.py already
    drops gameType != 2/3 at discovery and writes
    metadata.game_type = 'regular'/'playoff' on its 6,551 rows, and
    preseason_exclusion_sql honours that legacy marker directly. There
    is no historical NHL preseason in the corpus to find. (New ESPN
    NHL preseason ingestion from ~2026-09-20 arrives already carrying
    season_type, so it needs no backfill either.) The sport is
    supported here only so a run reports its counts.

Safety
------
Dry-run by DEFAULT; --apply writes. The CSV of affected match ids is
written BEFORE any UPDATE (dry-run included) into $BACKUP_LOCAL_DIR
(default /app/backups, the same artifact directory backup_postgres.py
uses), in the style of backups/voided_rec_ids_2026-07-13.csv, and is
opened "x" so a
verification dry-run after an apply cannot overwrite the only record
of what changed. Rows that already carry a season_type are never
touched, so the script is idempotent.

Usage:
    python /app/scripts/backfill_season_type.py                    # dry run, all sports
    python /app/scripts/backfill_season_type.py --sport nfl --sport nba
    python /app/scripts/backfill_season_type.py --sport nfl --apply
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_season_type")


ALLOWED_SPORTS = ("nfl", "nba", "nhl")
DEFAULT_SPORTS = ("nfl", "nba", "nhl")

PRESEASON = "preseason"
REGULAR = "regular"
POSTSEASON = "postseason"

# NBA regular-season opening dates. October games BEFORE the opener are
# preseason; games on or after it are regular season. Keyed by the
# calendar year the season opened in. Extend this table when a new
# season is ingested — an October with no entry is reported, never
# guessed at.
NBA_REGULAR_SEASON_OPENERS = {
    2021: dt.date(2021, 10, 19),
    2022: dt.date(2022, 10, 18),
    2023: dt.date(2023, 10, 24),
    2024: dt.date(2024, 10, 22),
    2025: dt.date(2025, 10, 21),
}

CSV_COLUMNS = (
    "match_id",
    "sport",
    "match_date",
    "season",
    "home_team",
    "away_team",
    "existing_season_type",
    "existing_game_type",
    "new_season_type",
    "rule",
)


# ── Classification (pure — this is what the tests pin) ───────────────


def classify(sport: str, match_date: dt.date) -> tuple[Optional[str], str]:
    """Return (season_type, rule) for one match date.

    season_type is None when the date rule cannot justify a value; the
    caller reports those rather than stamping them. `rule` is a short
    human-readable tag that lands in the CSV so an operator can see
    WHY each row was classified the way it was.
    """
    if sport == "nfl":
        return _classify_nfl(match_date)
    if sport == "nba":
        return _classify_nba(match_date)
    if sport == "nhl":
        # See the module docstring: the loader already filtered
        # preseason out and stamped metadata.game_type, which
        # preseason_exclusion_sql honours. Nothing to find.
        return None, "nhl:loader-filtered"
    raise ValueError(f"unsupported sport for season-type backfill: {sport!r}")


def _classify_nfl(d: dt.date) -> tuple[Optional[str], str]:
    if d.month == 8:
        return PRESEASON, "nfl:august"
    if d.month in (9, 10, 11, 12):
        return REGULAR, "nfl:sep-dec"
    # January (Weeks 17-18 vs Wild Card) and February (Super Bowl,
    # Pro Bowl) are genuinely ambiguous from the date alone.
    return None, "nfl:jan-feb-ambiguous"


def _classify_nba(d: dt.date) -> tuple[Optional[str], str]:
    if d.month == 10:
        opener = NBA_REGULAR_SEASON_OPENERS.get(d.year)
        if opener is None:
            return None, "nba:october-no-opener-in-table"
        if d < opener:
            return PRESEASON, "nba:before-opener"
        return REGULAR, "nba:on-or-after-opener"
    if d.month in (11, 12, 1, 2, 3):
        return REGULAR, "nba:nov-mar"
    # April (last regular week + play-in + round 1), May, June
    # (play-offs), and the July-September dead zone (summer league,
    # which should not be in the corpus at all).
    return None, "nba:apr-sep-ambiguous"


# ── DB I/O ───────────────────────────────────────────────────────────


def fetch_marker_counts(cur, sport: str) -> dict:
    """Current distribution of the two markers for one sport — the
    before/after picture the run prints."""
    cur.execute(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE m.metadata->>'season_type' = 'preseason')  AS preseason,
            COUNT(*) FILTER (WHERE m.metadata->>'season_type' = 'regular')    AS regular,
            COUNT(*) FILTER (WHERE m.metadata->>'season_type' = 'postseason') AS postseason,
            COUNT(*) FILTER (WHERE m.metadata->>'season_type' IS NULL
                               AND m.metadata->>'game_type' IS NOT NULL)      AS legacy_game_type_only,
            COUNT(*) FILTER (WHERE m.metadata->>'season_type' IS NULL
                               AND m.metadata->>'game_type' IS NULL)          AS unmarked
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = %s
        """,
        (sport,),
    )
    return dict(cur.fetchone() or {})


def fetch_unmarked(cur, sport: str) -> list[dict]:
    """Every row for the sport that carries NO season_type. Rows with
    the legacy NHL game_type marker are excluded: the predicate
    already honours them and overwriting a loader-verified marker with
    a date guess would be a downgrade."""
    cur.execute(
        """
        SELECT m.id::text        AS match_id,
               m.match_date,
               m.season,
               ht.name           AS home_team,
               at.name           AS away_team,
               m.metadata->>'season_type' AS existing_season_type,
               m.metadata->>'game_type'   AS existing_game_type
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        LEFT JOIN teams ht ON ht.id = m.home_team_id
        LEFT JOIN teams at ON at.id = m.away_team_id
        WHERE l.sport = %s
          AND m.metadata->>'season_type' IS NULL
          AND m.metadata->>'game_type' IS NULL
        ORDER BY m.match_date ASC
        """,
        (sport,),
    )
    return [dict(r) for r in cur.fetchall()]


def stamp_season_type(cur, match_id: str, season_type: str) -> None:
    """Merge the marker into metadata without clobbering the rest of
    the object. COALESCE covers rows whose metadata is NULL."""
    cur.execute(
        """
        UPDATE matches
        SET metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb,
            updated_at = NOW()
        WHERE id = %s
        """,
        (Json({"season_type": season_type}), match_id),
    )


# ── Planning ─────────────────────────────────────────────────────────


def plan_sport(cur, sport: str) -> tuple[list[dict], dict]:
    """Classify every unmarked row for one sport. Returns
    (entries to stamp, counts) where counts includes the rows the date
    rules refused to classify — those are reported, never guessed."""
    rows = fetch_unmarked(cur, sport)
    entries: list[dict] = []
    counts = {
        "unmarked": len(rows),
        "to_preseason": 0,
        "to_regular": 0,
        "to_postseason": 0,
        "unclassified": 0,
    }
    unclassified_rules: dict[str, int] = {}
    for r in rows:
        md = r["match_date"]
        d = md.date() if isinstance(md, dt.datetime) else md
        season_type, rule = classify(sport, d)
        if season_type is None:
            counts["unclassified"] += 1
            unclassified_rules[rule] = unclassified_rules.get(rule, 0) + 1
            continue
        counts[f"to_{season_type}"] += 1
        entries.append(
            {
                "match_id": r["match_id"],
                "sport": sport,
                "match_date": r["match_date"],
                "season": r.get("season"),
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
                "existing_season_type": r.get("existing_season_type"),
                "existing_game_type": r.get("existing_game_type"),
                "new_season_type": season_type,
                "rule": rule,
            }
        )
    counts["unclassified_by_rule"] = unclassified_rules
    return entries, counts


# ── CSV ──────────────────────────────────────────────────────────────


def default_out_csv(sports: Optional[Iterable[str]] = None) -> str:
    """ABSOLUTE, in the artifact directory the rest of the repo uses.

    A relative "backups/..." would resolve against the process CWD, and the
    documented invocation runs inside the api container whose WORKDIR is
    /app/services/api/src — so the only record of what changed would land in
    the bind-mounted source tree (host services/api/src/backups/, which the
    non-root appuser may not even be able to create) instead of the
    /app/backups volume that backup_postgres.py writes to.
    """
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    scope = "-".join(sorted(sports)) if sports else "all"
    base = os.environ.get("BACKUP_LOCAL_DIR", "/app/backups")
    return f"{base}/season_type_backfill_{scope}_{stamp}.csv"


def write_csv(path: str, entries: list[dict]) -> str:
    """Write every affected row BEFORE any UPDATE runs — dry-run
    included. Opened "x" (exclusive create): this file is the only
    record of which rows were stamped, and the natural follow-up
    action (a verification dry-run after the apply) would otherwise
    rewrite it as a header-only file. FileExistsError is the correct,
    loud outcome."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for e in entries:
            writer.writerow({k: ("" if e.get(k) is None else str(e.get(k))) for k in CSV_COLUMNS})
    logger.info("Wrote %d affected row(s) to %s", len(entries), out)
    return str(out)


# ── Orchestration ────────────────────────────────────────────────────


def _log_counts(sport: str, label: str, counts: dict) -> None:
    logger.info(
        "%s %s: total=%s preseason=%s regular=%s postseason=%s legacy_game_type_only=%s unmarked=%s",
        sport,
        label,
        counts.get("total"),
        counts.get("preseason"),
        counts.get("regular"),
        counts.get("postseason"),
        counts.get("legacy_game_type_only"),
        counts.get("unmarked"),
    )


def run(database_url: str, sports: list[str], apply: bool, out_csv: str) -> dict:
    summary: dict = {"sports": {}, "stamped": 0, "csv": out_csv, "applied": apply}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            all_entries: list[dict] = []
            plans: dict[str, dict] = {}
            for sport in sports:
                before = fetch_marker_counts(cur, sport)
                _log_counts(sport, "BEFORE", before)
                entries, counts = plan_sport(cur, sport)
                logger.info(
                    "%s plan: unmarked=%d -> preseason=%d regular=%d postseason=%d "
                    "(unclassified=%d, left alone: %s)",
                    sport,
                    counts["unmarked"],
                    counts["to_preseason"],
                    counts["to_regular"],
                    counts["to_postseason"],
                    counts["unclassified"],
                    counts["unclassified_by_rule"] or "none",
                )
                plans[sport] = {"before": before, "counts": counts}
                all_entries.extend(entries)

            # CSV first — dry-run included. No UPDATE may run without a
            # written record of what it will touch.
            write_csv(out_csv, all_entries)

            if not apply:
                logger.info(
                    "DRY RUN: %d row(s) would be stamped. Re-run with --apply to write.",
                    len(all_entries),
                )
                for sport in sports:
                    summary["sports"][sport] = {
                        "before": plans[sport]["before"],
                        "after": plans[sport]["before"],
                        "counts": plans[sport]["counts"],
                    }
                summary["stamped"] = 0
                return summary

            for e in all_entries:
                stamp_season_type(cur, e["match_id"], e["new_season_type"])
            conn.commit()
            summary["stamped"] = len(all_entries)
            logger.info("Applied: stamped %d row(s)", len(all_entries))

            for sport in sports:
                after = fetch_marker_counts(cur, sport)
                _log_counts(sport, "AFTER ", after)
                summary["sports"][sport] = {
                    "before": plans[sport]["before"],
                    "after": after,
                    "counts": plans[sport]["counts"],
                }
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sport",
        action="append",
        dest="sports",
        default=None,
        help=f"Sport to stamp; repeatable. Default: {', '.join(DEFAULT_SPORTS)}. "
        f"Allowed: {', '.join(ALLOWED_SPORTS)}.",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument(
        "--out-csv",
        default=None,
        help="CSV of affected rows (default $BACKUP_LOCAL_DIR or /app/backups, "
        "season_type_backfill_<sports>_<utc timestamp>.csv). "
        "Write-once: an existing path is an error, never an overwrite.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan only (DEFAULT).")
    mode.add_argument("--apply", action="store_true", help="Write the season_type markers.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set (pass --database-url or export DATABASE_URL)")
        return 2

    sports = args.sports or list(DEFAULT_SPORTS)
    unknown = [s for s in sports if s not in ALLOWED_SPORTS]
    if unknown:
        logger.error("Unsupported --sport value(s): %s (allowed: %s)", ", ".join(unknown), ", ".join(ALLOWED_SPORTS))
        return 2

    out_csv = args.out_csv or default_out_csv(sports)
    run(args.database_url, sports, apply=args.apply, out_csv=out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
