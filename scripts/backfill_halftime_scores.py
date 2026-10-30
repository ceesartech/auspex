"""Backfill metadata.halftime_{home,away}_score on finished soccer matches.

grading_outcomes grades the four half-time markets (match_result_ht,
btts_ht, over_under_ht, ht_ft_double_result) from those two keys, which
the ESPN results path stamps from 2026-10-30 on (fetch_upcoming.
halftime_scores). Earlier matches never had them, so ~30k half-time
predictions sat is_correct NULL on every grading run.

Only the (league, day) scoreboards holding a finished match that still
lacks the keys are fetched, plus the day before (ESPN files a late-UTC
kickoff under the previous US date). Each completed competition is
matched to an EXISTING finished row — by ESPN id, else the exact (home
team, away team, kickoff) triple the results path wrote — and written
only when that row's final score equals ESPN's. Never inserts a match,
team or league: teams resolve through fetch_upcoming.find_team.

The write is additive (two metadata keys), and the match-outcome trigger
fires only on a status flip, so nothing re-grades or re-settles here —
grade_completed_matches picks the rows up afterwards. Every planned row
goes to a write-once CSV first, dry run included.

Usage (inside the api container):
    python /app/scripts/backfill_halftime_scores.py --days-back 160            # dry run
    python /app/scripts/backfill_halftime_scores.py --days-back 160 --apply
"""

from __future__ import annotations

import argparse
import csv
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_halftime_scores")

SOCCER = fu.SPORT_CONFIGS["soccer"]

PENDING_DAYS_SQL = """
    SELECT l.name AS league_name, l.country, (m.match_date AT TIME ZONE 'UTC')::date AS day
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    WHERE l.sport = 'soccer'
      AND m.status = 'finished'
      AND m.match_date >= NOW() - (%s || ' days')::interval
      AND NOT (COALESCE(m.metadata, '{}'::jsonb) ? 'halftime_home_score')
    GROUP BY 1, 2, 3
"""

LEAGUE_ID_SQL = "SELECT id::text AS id FROM leagues WHERE name = %s AND country = %s AND sport = 'soccer'"

ROW_BY_ESPN_SQL = """
    SELECT m.id::text AS id, m.home_score, m.away_score, m.metadata
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    WHERE l.sport = 'soccer' AND m.status = 'finished' AND m.external_ids->>'espn' = %s
"""

ROW_BY_TRIPLE_SQL = """
    SELECT id::text AS id, home_score, away_score, metadata
    FROM matches
    WHERE home_team_id = %s AND away_team_id = %s AND match_date = %s AND status = 'finished'
"""

UPDATE_SQL = """
    UPDATE matches
    SET metadata = COALESCE(metadata, '{}'::jsonb)
                   || jsonb_build_object('halftime_home_score', %s::int, 'halftime_away_score', %s::int)
    WHERE id = %s
      AND status = 'finished'
      AND NOT (COALESCE(metadata, '{}'::jsonb) ? 'halftime_home_score')
"""

CSV_FIELDS = ("match_id", "league", "match_date", "home", "away", "espn_id", "halftime_home", "halftime_away")


def league_slugs() -> dict[tuple[str, str], str]:
    """(league name, country) -> ESPN slug, from fetch_upcoming's config."""
    return {(name, country): slug for slug, (_code, name, country) in SOCCER.leagues.items()}


def pending_days(cur, days_back: int) -> dict[tuple[str, str], set[date]]:
    cur.execute(PENDING_DAYS_SQL, (str(days_back),))
    out: dict[tuple[str, str], set[date]] = defaultdict(set)
    for r in cur.fetchall():
        out[(r["league_name"], r["country"])].add(r["day"])
    return out


def scoreboard_days(days: set[date]) -> list[date]:
    """Each pending day plus the day before, oldest first."""
    return sorted({d for day in days for d in (day, day - timedelta(days=1))})


def find_row(cur, league_id: Optional[str], espn_id: Optional[str], home_name, away_name, match_dt):
    """(row, how) for the existing finished row, or (None, reason)."""
    if espn_id:
        cur.execute(ROW_BY_ESPN_SQL, (str(espn_id),))
        rows = cur.fetchall()
        if len(rows) > 1:
            return None, "ambiguous"
        if rows:
            return rows[0], "espn"
    home_id = fu.find_team(cur, SOCCER, home_name, league_id)
    away_id = fu.find_team(cur, SOCCER, away_name, league_id)
    if not (home_id and away_id):
        return None, "unmatched"
    cur.execute(ROW_BY_TRIPLE_SQL, (home_id, away_id, match_dt))
    rows = cur.fetchall()
    if len(rows) != 1:
        return None, "unmatched" if not rows else "ambiguous"
    return rows[0], "triple"


def plan_competition(cur, league_id: Optional[str], comp: dict, counts: Counter) -> Optional[dict]:
    """One planned write for a completed competition, or None (counted)."""
    parts = fu._competition_parts(comp, SOCCER)
    if parts is None:
        counts["unusable"] += 1
        return None
    home, away, home_name, away_name, match_dt, state, _venue, espn_id = parts
    if state != "post" or not comp.get("status", {}).get("type", {}).get("completed", False):
        counts["not_completed"] += 1
        return None
    final = fu._final_scores(SOCCER, home, away)
    if final is None:
        counts["no_final_score"] += 1
        return None
    ht = fu.halftime_scores(comp, home, away, final)
    if ht is None:
        counts["plays_do_not_add_up"] += 1
        return None
    row, how = find_row(cur, league_id, espn_id, home_name, away_name, match_dt)
    if row is None:
        counts[how] += 1
        return None
    if (row["home_score"], row["away_score"]) != final:
        counts["score_mismatch"] += 1
        return None
    if "halftime_home_score" in (row["metadata"] or {}):
        counts["already_set"] += 1
        return None
    counts[f"matched_by_{how}"] += 1
    return {
        "match_id": row["id"],
        "match_date": match_dt.isoformat() if hasattr(match_dt, "isoformat") else str(match_dt),
        "home": home_name,
        "away": away_name,
        "espn_id": espn_id or "",
        "halftime_home": ht[0],
        "halftime_away": ht[1],
    }


def default_out_csv() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base = os.environ.get("BACKUP_LOCAL_DIR", "/app/backups")
    return f"{base}/halftime_backfill_{stamp}.csv"


def write_csv(path: str, entries: list[dict]) -> None:
    out = Path(path)
    if out.exists():
        raise FileExistsError(f"{out} exists — the audit CSV is write-once")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(entries)


def run(conn, days_back: int, apply_changes: bool, out_csv: Optional[str] = None) -> Counter:
    counts: Counter = Counter()
    slugs = league_slugs()
    planned: dict[str, dict] = {}
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        for (league_name, country), days in sorted(pending_days(cur, days_back).items()):
            slug = slugs.get((league_name, country))
            if slug is None:
                logger.warning("No ESPN slug for %s (%s) — %d day(s) skipped", league_name, country, len(days))
                counts["league_without_slug"] += 1
                continue
            cur.execute(LEAGUE_ID_SQL, (league_name, country))
            league_row = cur.fetchone()
            league_id = league_row["id"] if league_row else None
            for day in scoreboard_days(days):
                for event in fu.fetch_day(SOCCER, slug, day):
                    for comp, _ctx in fu._iter_event_competitions(event, SOCCER):
                        entry = plan_competition(cur, league_id, comp, counts)
                        if entry is not None:
                            # The same match shows on two fetched days; one write.
                            planned.setdefault(entry["match_id"], {**entry, "league": slug})
            logger.info("%s: %d match(es) planned so far", slug, len(planned))
    conn.rollback()  # close the read snapshot before writing

    entries = list(planned.values())
    out_csv = out_csv or default_out_csv()
    write_csv(out_csv, entries)
    logger.info("Planned %d half-time stamp(s); audit CSV %s", len(entries), out_csv)

    if fu.FETCH_STATS["failed"] and not fu.FETCH_STATS["ok"]:
        raise RuntimeError(f"Every ESPN request failed ({fu.FETCH_STATS['failed']}) — nothing written")
    if fu.FETCH_STATS["failed"]:
        # Partial: those days are simply not planned; a re-run picks them up.
        logger.error(
            "%d of %d ESPN scoreboard request(s) failed", fu.FETCH_STATS["failed"], sum(fu.FETCH_STATS.values())
        )
    counts["espn_requests_failed"] = fu.FETCH_STATS["failed"]

    if apply_changes and entries:
        with conn.cursor() as cur:
            for e in entries:
                cur.execute(UPDATE_SQL, (e["halftime_home"], e["halftime_away"], e["match_id"]))
                counts["written"] += cur.rowcount
        conn.commit()
    elif not apply_changes:
        logger.info("DRY-RUN: re-run with --apply to write %d row(s)", len(entries))
    counts["planned"] = len(entries)
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days-back", type=int, default=160, help="Finished matches this recent (default 160).")
    p.add_argument("--apply", action="store_true", help="Write the keys (default: dry run).")
    p.add_argument("--out-csv", default=None, help="Audit CSV (default /app/backups/halftime_backfill_<utc>.csv).")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    with psycopg2.connect(args.database_url) as conn:
        counts = run(conn, args.days_back, args.apply, args.out_csv)
    logger.info("Done: %s", dict(sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
