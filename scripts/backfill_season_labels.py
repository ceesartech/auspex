"""Correct matches.season on the soccer rows promote_raw mislabelled.

promote_raw.season_label read every 4-digit football-data season as a
'YYZZ' code, so a calendar-year season of the /new/ country files
(Argentina, Brazil, China, Finland, Ireland, Japan, MLS, Norway, Sweden)
was stored as '2020-2116' for 2016, '2020-2112' for 2012 and so on (32,794
rows on 2026-10-31) and as '2020-2021' for 2021 (2,746 rows that look right
and are not), while the cross-year files' 'YYYY/YYYY' was stored verbatim
(26,890 rows) beside the 'YYYY-YYYY' every other loader writes. promote_raw now writes '2016' and
'2016-2017'; its conflict branch never touches season, so re-running it
does not repair the rows already written. This does.

Nothing live reads the soccer label (checked 2026-10-31): the soccer
training query (training_data.DEFAULT_TRAINING_QUERY), compute_features,
precompute_predictions, generate_recommendations, the Dixon-Coles trainers
and monitor_models never select it, and features_cache holds only the
'baseline' set. It is displayed by the match API, grouped on by the
ab_soccer_feature_screen harness and the dormant feature-engineering
orchestrator, and turned into fbref URLs by scrape_fbref_venues (fbref
spells these seasons '2016' and '2016-2017').

Each matches row is tied to its raw_football_data row the way promote_raw
tied them — league code, normalised team names, and the kickoff as
promote_raw wrote it until 2026-10-31 (the UK wall clock,
promote_raw.parse_wall_clock, read in the database's TimeZone), falling back
to the corrected UTC kickoff (promote_raw.parse_match_date) — and relabelled
only when its season is still the label the old code produced from that
raw season. A row the ESPN path created, or one edited since, is left
alone and counted. Raw duplicates that disagree on the new label leave
their row alone too. Every planned change goes to a write-once CSV first,
dry run included; the UPDATE is guarded on the old value and runs as one
transaction.

Usage (inside the api container):
    python /app/scripts/backfill_season_labels.py            # dry run
    python /app/scripts/backfill_season_labels.py --apply
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import promote_raw as pr  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_season_labels")

RAW_SQL = 'SELECT league, season, "Date", "Time", "HomeTeam", "AwayTeam" FROM raw_football_data'
LEAGUES_SQL = (
    "SELECT id::text AS id, external_ids->>'football_data' AS code FROM leagues WHERE external_ids ? 'football_data'"
)
TEAMS_SQL = "SELECT id::text AS id, normalized_name FROM teams WHERE sport = 'soccer'"
MATCHES_SQL = """
    SELECT m.id::text AS id, m.league_id::text AS league_id,
           m.home_team_id::text AS home_id, m.away_team_id::text AS away_id,
           m.match_date, m.season
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    WHERE l.sport = 'soccer'
"""

# Guarded on the old value; RETURNING counts what actually changed.
UPDATE_SQL = """
    UPDATE matches AS m
    SET season = v.new_season
    FROM (VALUES %s) AS v(id, old_season, new_season)
    WHERE m.id = v.id::uuid
      AND m.season = v.old_season
    RETURNING m.id
"""

CANONICAL_SEASON = re.compile(r"\d{4}|\d{4}-\d{4}")
CSV_FIELDS = ("match_id", "league", "match_date", "home", "away", "raw_season", "old_season", "new_season")


def legacy_season_label(code: str) -> str:
    """promote_raw.season_label as it was until 2026-10-31 — what a row
    written from `code` holds now. Frozen here; do not 'fix' it."""
    if not code or len(code) != 4 or not code.isdigit():
        return code or ""
    start = int(code[:2])
    end = int(code[2:])
    start_year = (1900 if start >= 50 else 2000) + start
    end_year = (1900 if end >= 50 else 2000) + end
    if end_year < start_year:
        end_year += 100
    return f"{start_year}-{end_year}"


def kickoff_key(match_dt: datetime, db_tz: tzinfo) -> datetime:
    """A kickoff as an aware UTC datetime. promote_raw inserts naive
    datetimes, which Postgres reads in its session TimeZone."""
    if match_dt.tzinfo is None:
        match_dt = match_dt.replace(tzinfo=db_tz)
    return match_dt.astimezone(timezone.utc)


def plan(
    raw_rows: Iterable[dict],
    league_ids: dict[str, str],
    team_ids: dict[str, str],
    matches_by_key: dict[tuple, dict],
    db_tz: tzinfo,
    counts: Counter,
) -> list[dict]:
    """The relabels to make, one per matches row. Pure."""
    planned: dict[str, dict] = {}
    conflicts: set[str] = set()
    for r in raw_rows:
        code, raw_season = r.get("league"), r.get("season") or ""
        old, new = legacy_season_label(raw_season), pr.season_label(raw_season, code)
        if old == new:
            counts["label_unchanged"] += 1
            continue
        if not CANONICAL_SEASON.fullmatch(new):
            counts["new_label_not_canonical"] += 1
            continue
        league_id = league_ids.get(code)
        home_id = team_ids.get(pr.normalize_team_name(r.get("HomeTeam") or ""))
        away_id = team_ids.get(pr.normalize_team_name(r.get("AwayTeam") or ""))
        wall_clock = pr.parse_wall_clock(r.get("Date") or "", r.get("Time"))
        if not (league_id and home_id and away_id and wall_clock):
            counts["raw_row_unresolved"] += 1
            continue
        # The legacy kickoff first (where promote_raw wrote the row), then the
        # corrected one, in case the operator's UK-time shift has run already.
        m = matches_by_key.get((home_id, away_id, kickoff_key(wall_clock, db_tz)))
        if m is None:
            corrected = pr.parse_match_date(r.get("Date") or "", r.get("Time"))
            m = matches_by_key.get((home_id, away_id, kickoff_key(corrected, db_tz))) if corrected else None
        if m is None:
            counts["no_matches_row"] += 1
            continue
        if m["league_id"] != league_id:
            counts["league_mismatch"] += 1
            continue
        if m["season"] == new:
            counts["already_correct"] += 1
            continue
        if m["season"] != old:
            counts["season_not_from_this_raw_row"] += 1  # ESPN-created, or edited since
            continue
        entry = {
            "match_id": m["id"],
            "league": code,
            "match_date": m["match_date"].isoformat() if hasattr(m["match_date"], "isoformat") else m["match_date"],
            "home": r.get("HomeTeam"),
            "away": r.get("AwayTeam"),
            "raw_season": raw_season,
            "old_season": old,
            "new_season": new,
        }
        prior = planned.get(m["id"])
        if prior is not None and (prior["old_season"], prior["new_season"]) != (old, new):
            conflicts.add(m["id"])
        planned.setdefault(m["id"], entry)
    for match_id in conflicts:
        del planned[match_id]
    counts["raw_duplicates_disagree"] = len(conflicts)
    return list(planned.values())


def default_out_csv() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base = os.environ.get("BACKUP_LOCAL_DIR", "/app/backups")
    return f"{base}/season_labels_{stamp}.csv"


def write_csv(path: str, entries: list[dict]) -> None:
    out = Path(path)
    if out.exists():
        raise FileExistsError(f"{out} exists — the audit CSV is write-once")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(entries)


def load_inputs(cur) -> tuple[list[dict], dict[str, str], dict[str, str], dict[tuple, dict], tzinfo]:
    cur.execute("SHOW TimeZone")
    db_tz = ZoneInfo(cur.fetchone()["TimeZone"])  # loud if Postgres reports a zone Python cannot load
    cur.execute(RAW_SQL)
    raw_rows = [dict(r) for r in cur.fetchall()]
    cur.execute(LEAGUES_SQL)
    league_ids = {r["code"]: r["id"] for r in cur.fetchall()}
    cur.execute(TEAMS_SQL)
    team_ids = {r["normalized_name"]: r["id"] for r in cur.fetchall()}
    cur.execute(MATCHES_SQL)
    matches_by_key = {
        (r["home_id"], r["away_id"], r["match_date"].astimezone(timezone.utc)): dict(r) for r in cur.fetchall()
    }
    return raw_rows, league_ids, team_ids, matches_by_key, db_tz


def run(conn, apply_changes: bool, out_csv: Optional[str] = None) -> Counter:
    counts: Counter = Counter()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        raw_rows, league_ids, team_ids, matches_by_key, db_tz = load_inputs(cur)
    conn.rollback()  # close the read snapshot before writing
    logger.info(
        "Loaded %d raw rows, %d leagues, %d soccer teams, %d soccer matches (db TimeZone %s)",
        len(raw_rows),
        len(league_ids),
        len(team_ids),
        len(matches_by_key),
        db_tz,
    )

    entries = plan(raw_rows, league_ids, team_ids, matches_by_key, db_tz, counts)
    out_csv = out_csv or default_out_csv()
    write_csv(out_csv, entries)
    by_league = Counter(e["league"] for e in entries)
    logger.info("Planned %d relabel(s) %s; audit CSV %s", len(entries), dict(sorted(by_league.items())), out_csv)
    if counts["raw_duplicates_disagree"]:
        logger.error(
            "%d row(s) left alone: their raw_football_data duplicates disagree on the season",
            counts["raw_duplicates_disagree"],
        )
    if counts["new_label_not_canonical"]:
        logger.error("%d raw row(s) with an unrecognised season left alone", counts["new_label_not_canonical"])
    counts["planned"] = len(entries)

    if not apply_changes:
        logger.info("DRY-RUN: re-run with --apply to relabel %d row(s)", len(entries))
        return counts
    with conn.cursor() as cur:
        updated = execute_values(
            cur,
            UPDATE_SQL,
            [(e["match_id"], e["old_season"], e["new_season"]) for e in entries],
            page_size=1000,
            fetch=True,
        )
    conn.commit()
    counts["relabelled"] = len(updated)
    counts["changed_since_plan"] = len(entries) - len(updated)
    if counts["changed_since_plan"]:
        logger.error("%d row(s) changed between plan and write — left as they were", counts["changed_since_plan"])
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
    return 1 if counts["changed_since_plan"] else 0


if __name__ == "__main__":
    sys.exit(main())
