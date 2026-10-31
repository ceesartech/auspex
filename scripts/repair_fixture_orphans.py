"""Merge orphan fixture rows into their finished twin, then delete them.

THE BUG THIS CLEANS UP
----------------------
Before 2026-09-04 fixture identity was keyed on the exact kickoff time, so
every ESPN time change forked a tennis/MMA fixture into another `matches`
row; until 2026-10-30 a re-drawn 1v1 draw slot did the same (see
fetch_upcoming._release_redrawn_slot). Prod 2026-10-30: 19,486 stale
'scheduled' tennis rows holding 89k odds rows, 147k CLV snapshots and 20.6k
predictions, while only 864 of the 3,214 real finished matches they belong
to carried any odds of their own. `repair_stranded_recs.py` already settled
and re-pointed the RECOMMENDATIONS; this tool does the rest.

WHAT THIS DOES
--------------
Candidates: rows of --sport with status 'scheduled' (or 'cancelled' by the
re-draw fix) whose kickoff is older than --stale-after-hours (default 96 h,
past the results path's 3-day look-back so a row the pipeline could still
heal is never touched).

Each candidate's twin is chosen by repair_stranded_recs.find_twins +
select_twin — the SAME rule the rec repair used: identical orientation,
status finished, both scores present, nearest kickoff within
--twin-window-hours, ambiguous (different scores / tie) -> skipped.

  twin found  -> MOVE onto the twin, strictly pre-game only:
                 odds            one row per current-price key (book,
                                 market, selection, line, opening, live):
                                 the freshest price with timestamp < the
                                 twin's kickoff, and only for keys the twin
                                 does not already hold (its own rows win);
                 odds_snapshots  every capture before the twin's kickoff
                                 not already on the twin (same book,
                                 market, selection, line, captured_at);
                 predictions     per (model, version, type) the latest one
                                 written before the twin's kickoff, unless
                                 the twin already has that key.
               then DELETE the orphan (cascade drops what was not moved:
               post-kickoff prices, superseded duplicates, features_cache).
  no twin     -> DELETE (a pairing that never played: a re-drawn slot, a
               walkover); its prices have no result to train on.
  ambiguous   -> left alone, counted.

An orphan that still holds ANY betting_recommendations or betting_history
row is never deleted (run repair_stranded_recs.py first; it re-points
settled recs onto the twin) — it is counted as `kept_has_recs`. Nor is one
holding a PREDICTION a rec references: re-pointing moves the rec's match_id
but not its prediction_id. Until migration 025 betting_recommendations.
prediction_id was ON DELETE CASCADE, so deleting that orphan deleted the rec
with it — the 2026-10-31 tennis run lost 45 settled recs exactly that way
(restored by restore_cascaded_recs.py); the twin usually already holds the
same (model, version, type), so the rec's prediction is never the one moved.
Since 025 (NO ACTION) the DELETE would fail instead; the guard keeps the
batch from aborting.

Every row the run moves or deletes is written to --archive (gzip JSONL,
write-once) BEFORE the first write, so the run is reversible offline.
Batches of --batch-size twins commit separately, keeping locks short while
the 15-min pipeline runs.

Usage (inside the api container; dry-run is the default):
    python /app/scripts/repair_fixture_orphans.py --sport tennis
    python /app/scripts/repair_fixture_orphans.py --sport tennis --apply
Afterwards: compute_features_<sport>.py --all-finished --force, and
grade_completed_matches.py --days <span> so moved predictions are graded.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repair_stranded_recs as rsr  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("repair_fixture_orphans")

ALLOWED_SPORTS = ("tennis", "mma")
DEFAULT_STALE_AFTER_HOURS = 96
MIN_STALE_AFTER_HOURS = 96  # the results path looks back 3 days; never race it

LIST_CANDIDATES_SQL = """
    SELECT m.id::text AS id, m.home_team_id::text AS home_team_id, m.away_team_id::text AS away_team_id,
           m.match_date, m.status,
           EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.match_id = m.id)
             OR EXISTS (SELECT 1 FROM betting_history h WHERE h.match_id = m.id)
             OR EXISTS (SELECT 1 FROM betting_recommendations r
                          JOIN predictions p ON p.id = r.prediction_id
                         WHERE p.match_id = m.id) AS has_recs
      FROM matches m
      JOIN leagues l ON l.id = m.league_id
     WHERE l.sport = %s
       AND m.status IN ('scheduled', 'cancelled')
       AND m.match_date < NOW() - (%s || ' hours')::interval
     ORDER BY m.match_date
"""

_PRICE_KEY = (
    "bookmaker, market_type, selection, COALESCE(line, -1), COALESCE(is_opening, false), COALESCE(is_live, false)"
)

MOVE_ODDS_SQL = f"""
    WITH cand AS (
        SELECT DISTINCT ON ({_PRICE_KEY}) id
          FROM odds
         WHERE match_id = ANY(%(orphans)s::uuid[])
           AND "timestamp" < %(kickoff)s
         ORDER BY {_PRICE_KEY}, "timestamp" DESC, created_at DESC, id
    )
    UPDATE odds o
       SET match_id = %(twin)s::uuid
      FROM cand
     WHERE o.id = cand.id
       AND NOT EXISTS (
             SELECT 1 FROM odds t
              WHERE t.match_id = %(twin)s::uuid
                AND t.bookmaker = o.bookmaker
                AND t.market_type = o.market_type
                AND t.selection = o.selection
                AND COALESCE(t.line, -1) = COALESCE(o.line, -1)
                AND COALESCE(t.is_opening, false) = COALESCE(o.is_opening, false)
                AND COALESCE(t.is_live, false) = COALESCE(o.is_live, false))
"""

MOVE_SNAPSHOTS_SQL = """
    WITH cand AS (
        SELECT DISTINCT ON (bookmaker, market_type, selection, COALESCE(line, -1), captured_at) id
          FROM odds_snapshots
         WHERE match_id = ANY(%(orphans)s::uuid[])
           AND captured_at < %(kickoff)s
         ORDER BY bookmaker, market_type, selection, COALESCE(line, -1), captured_at, id
    )
    UPDATE odds_snapshots s
       SET match_id = %(twin)s::uuid
      FROM cand
     WHERE s.id = cand.id
       AND NOT EXISTS (
             SELECT 1 FROM odds_snapshots t
              WHERE t.match_id = %(twin)s::uuid
                AND t.bookmaker = s.bookmaker
                AND t.market_type = s.market_type
                AND t.selection = s.selection
                AND COALESCE(t.line, -1) = COALESCE(s.line, -1)
                AND t.captured_at = s.captured_at)
"""

MOVE_PREDICTIONS_SQL = """
    WITH cand AS (
        SELECT DISTINCT ON (model_name, model_version, prediction_type) id
          FROM predictions
         WHERE match_id = ANY(%(orphans)s::uuid[])
           AND COALESCE(updated_at, created_at) < %(kickoff)s
         ORDER BY model_name, model_version, prediction_type, COALESCE(updated_at, created_at) DESC, id
    )
    UPDATE predictions p
       SET match_id = %(twin)s::uuid
      FROM cand
     WHERE p.id = cand.id
       AND NOT EXISTS (
             SELECT 1 FROM predictions t
              WHERE t.match_id = %(twin)s::uuid
                AND t.model_name = p.model_name
                AND t.model_version = p.model_version
                AND t.prediction_type = p.prediction_type)
"""

DELETE_ORPHANS_SQL = """
    DELETE FROM matches m
     WHERE m.id = ANY(%(orphans)s::uuid[])
       AND m.status IN ('scheduled', 'cancelled')
       AND NOT EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.match_id = m.id)
       AND NOT EXISTS (SELECT 1 FROM betting_history h WHERE h.match_id = m.id)
       -- a rec whose prediction lives here: CASCADE deleted it before 025, NO ACTION fails
       AND NOT EXISTS (SELECT 1 FROM betting_recommendations r
                         JOIN predictions p ON p.id = r.prediction_id
                        WHERE p.match_id = m.id)
"""

# What the archive captures per orphan: the row itself and everything a
# move or the cascade can touch that carries data worth restoring.
ARCHIVE_TABLES = ("odds", "odds_snapshots", "predictions")


class UnsafeRequest(RuntimeError):
    """Arguments that would make the repair unsafe."""


def plan(candidates: list[dict], twins_for) -> dict:
    """Group candidates by outcome. `twins_for(candidate)` returns the
    already-orientation-checked twin list (find_twins). Pure apart from it."""
    groups: dict[str, list[str]] = defaultdict(list)  # twin_id -> orphan ids
    kickoff: dict[str, datetime] = {}
    no_twin: list[str] = []
    counts: Counter = Counter()
    for c in candidates:
        if c["has_recs"]:
            counts["kept_has_recs"] += 1
            continue
        twin, reason = rsr.select_twin(twins_for(c), c["match_date"])
        if twin is None and reason == rsr.REASON_NO_TWIN:
            no_twin.append(c["id"])
            counts["no_twin"] += 1
        elif twin is None:
            counts[f"skipped_{reason}"] += 1
        else:
            groups[twin["twin_id"]].append(c["id"])
            kickoff[twin["twin_id"]] = twin["match_date"]
            counts["merge"] += 1
    counts["twins"] = len(groups)
    return {"groups": dict(groups), "kickoff": kickoff, "no_twin": no_twin, "counts": counts}


def _jsonable(v):
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return str(v)
    return v


def write_archive(cur, path: Path, orphan_ids: list[str], twin_of: dict[str, str]) -> int:
    if path.exists():
        raise FileExistsError(f"{path} exists — the archive is write-once")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt") as fh:
        for i in range(0, len(orphan_ids), 2000):
            chunk = orphan_ids[i : i + 2000]
            cur.execute("SELECT * FROM matches WHERE id = ANY(%s::uuid[])", (chunk,))
            for r in cur.fetchall():
                row = {k: _jsonable(v) for k, v in r.items()}
                fh.write(json.dumps({"table": "matches", "twin_id": twin_of.get(str(r["id"])), "row": row}) + "\n")
                n += 1
            for table in ARCHIVE_TABLES:
                cur.execute(f"SELECT * FROM {table} WHERE match_id = ANY(%s::uuid[])", (chunk,))
                for r in cur.fetchall():
                    row = {k: _jsonable(v) for k, v in r.items()}
                    fh.write(json.dumps({"table": table, "row": row}) + "\n")
                    n += 1
    return n


def _count_refused(moved: Counter, planned: int, deleted: int) -> None:
    """The delete guard refusing a planned orphan means a rec or a history
    row appeared on it after the plan: never silent."""
    if deleted < planned:
        moved["delete_refused_has_recs"] += planned - deleted
        logger.warning(
            "delete guard kept %d planned orphan(s): a rec or history row references them", planned - deleted
        )


def apply_plan(conn, p: dict, batch_size: int) -> Counter:
    moved: Counter = Counter()
    twins = list(p["groups"].items())
    for i in range(0, len(twins), batch_size):
        with conn.cursor() as cur:
            for twin_id, orphans in twins[i : i + batch_size]:
                params = {"orphans": orphans, "twin": twin_id, "kickoff": p["kickoff"][twin_id]}
                for label, sql in (
                    ("odds", MOVE_ODDS_SQL),
                    ("odds_snapshots", MOVE_SNAPSHOTS_SQL),
                    ("predictions", MOVE_PREDICTIONS_SQL),
                ):
                    cur.execute(sql, params)
                    moved[f"moved_{label}"] += cur.rowcount
                cur.execute(DELETE_ORPHANS_SQL, {"orphans": orphans})
                moved["deleted_merged_orphans"] += cur.rowcount
                _count_refused(moved, len(orphans), cur.rowcount)
        conn.commit()
        logger.info("batch %d-%d committed (%s)", i, i + len(twins[i : i + batch_size]), dict(moved))
    no_twin = p["no_twin"]
    for i in range(0, len(no_twin), batch_size * 10):
        with conn.cursor() as cur:
            chunk = no_twin[i : i + batch_size * 10]
            cur.execute(DELETE_ORPHANS_SQL, {"orphans": chunk})
            moved["deleted_no_twin_orphans"] += cur.rowcount
            _count_refused(moved, len(chunk), cur.rowcount)
        conn.commit()
    return moved


def default_archive(sport: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/app/backups") / f"fixture_orphans_{sport}_{stamp}.jsonl.gz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sport", required=True, choices=ALLOWED_SPORTS)
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--stale-after-hours", type=int, default=DEFAULT_STALE_AFTER_HOURS)
    ap.add_argument("--twin-window-hours", type=int, default=72)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--archive", type=Path, default=None, help="gzip JSONL reversal record (write-once)")
    ap.add_argument("--apply", action="store_true", help="execute (default: dry-run)")
    args = ap.parse_args(argv)
    if args.stale_after_hours < MIN_STALE_AFTER_HOURS:
        raise UnsafeRequest(f"--stale-after-hours must be >= {MIN_STALE_AFTER_HOURS} (results path look-back)")
    if not rsr.MIN_TWIN_WINDOW_HOURS <= args.twin_window_hours <= rsr.MAX_TWIN_WINDOW_HOURS:
        raise UnsafeRequest(
            f"--twin-window-hours must be in [{rsr.MIN_TWIN_WINDOW_HOURS}, {rsr.MAX_TWIN_WINDOW_HOURS}]"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        raise UnsafeRequest("DATABASE_URL not set")
    conn = psycopg2.connect(args.database_url, cursor_factory=RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(LIST_CANDIDATES_SQL, (args.sport, str(args.stale_after_hours)))
            candidates = [dict(r) for r in cur.fetchall()]

            def twins_for(c: dict) -> list[dict]:
                return rsr.find_twins(
                    cur, c["id"], c["home_team_id"], c["away_team_id"], c["match_date"], args.twin_window_hours
                )

            p = plan(candidates, twins_for)
        counts = p["counts"]
        logger.info("candidates=%d plan=%s", len(candidates), dict(counts))
        if not args.apply:
            logger.info("dry-run: nothing written (re-run with --apply)")
            return 0
        twin_of = {o: t for t, orphans in p["groups"].items() for o in orphans}
        targets = list(twin_of) + p["no_twin"]
        archive = args.archive or default_archive(args.sport)
        with conn.cursor() as cur:
            n = write_archive(cur, archive, targets, twin_of)
        conn.rollback()  # end the read transaction before the batched writes
        logger.info("archived %d rows to %s", n, archive)
        moved = apply_plan(conn, p, args.batch_size)
        logger.info("DONE %s", dict(moved))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
