"""Stamp the 90-minute score on finished soccer ties decided after 90 minutes.

Soccer full-time markets and recs settle on 90 minutes plus stoppage time
(grading_outcomes.settlement_scores). From 2026-10-31 the ESPN results path
stores a tie decided in extra time or on penalties with its final,
post-extra-time score in home_score / away_score and its 90-minute score in
metadata.regulation_home_score / regulation_away_score plus went_extra=true
(fetch_upcoming.regulation_scores). The rows it wrote before carry only
metadata.result_detail 'AET' / 'FT-PENS' and grade on the post-extra-time
score: 15 rows on 2026-10-31 (5 AET, 10 FT-PENS; 2026 World Cup and Copa
Libertadores). Five change: Belgium v Senegal 3-2 AET was 2-2 after 90
minutes, Argentina v Cape Verde 3-2 was 1-1, Norway v England 1-2 was 1-1,
Argentina v Switzerland 3-1 was 1-1 and Spain v Argentina 1-0 was 0-0. The
ten shootouts keep their score: the four World Cup ones were level after 90
minutes and after 120, and the six Libertadores second legs went straight to
penalties.

For each such finished row still lacking the keys, its (league, day)
scoreboard is fetched, plus the day before (ESPN files a late-UTC kickoff
under the previous US date), and each completed competition is matched to a
CANDIDATE row only — by ESPN id, else the exact (home team, away team,
kickoff) triple, teams resolved through fetch_upcoming.find_team, which
never writes. The keys are planned only when:
  * ESPN's final score equals the row's;
  * ESPN still says the match was decided after 90 minutes;
  * ESPN's header score agrees with its scoring plays, and the plays
    reproduce the final score (so the 90-minute split is complete).
Anything else is refused and counted, never guessed; a refusal exits 1 and
writes nothing unless --skip-refused.

Every candidate goes to a write-once CSV first (planned and refused alike,
with the reason), dry run included. --apply writes the stamps in one
transaction, each UPDATE guarded on what the plan read (finished, the final
score, result_detail, no regulation keys); a row that changed rolls the
whole run back. Merging metadata on a finished row fires no trigger, so
nothing re-grades here. Afterwards re-grade the full-time markets
(grading_outcomes.SOCCER_FT_TYPES; the half-time-only markets do not
change) of exactly the stamped matches, dry run first, then the same with
--apply (scripts/ops/backfill_soccer_regulation_scores.sh writes the ids from
its apply CSV and runs the dry run):

    python /app/scripts/regrade_predictions.py --match-ids-file <stamped ids> \\
        --prediction-type <grading_outcomes.SOCCER_FT_TYPES, comma-separated>

That re-grade also rewrites the DB trigger's 'over' / 'under' over_under
actual_outcome on those matches into the grader's 'over_0.5' form without
changing a verdict; its summary's "changed (→T/→F)" column shows the real
moves. A rec settled with money on such a match is re-settled by
grade_completed_matches when the match is in its window; all 28 recs on the
15 matches of 2026-10-31 were already void (constant-prior incident), and a
'void: ...' rec is never re-settled.

Usage (inside the api container):
    python /app/scripts/backfill_soccer_regulation_scores.py            # dry run
    python /app/scripts/backfill_soccer_regulation_scores.py --apply
"""

from __future__ import annotations

import argparse
import csv
import json
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
from grading_outcomes import REGULATION_AWAY_KEY, REGULATION_HOME_KEY, SOCCER_PAST_90_DETAIL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_soccer_regulation_scores")

SOCCER = fu.SPORT_CONFIGS["soccer"]

# Broad on purpose ('PEN' also matches 'SUSPENDED'); candidates() keeps only
# what grading_outcomes.SOCCER_PAST_90_DETAIL reads as decided after 90.
CANDIDATES_SQL = """
    SELECT m.id::text AS id, l.name AS league_name, l.country,
           m.match_date, m.home_team_id::text AS home_id, m.away_team_id::text AS away_id,
           th.name AS home, ta.name AS away, m.home_score, m.away_score,
           m.external_ids->>'espn' AS espn_id,
           m.metadata->>'result_detail' AS result_detail,
           m.metadata->>'regulation_home_score' AS reg_home,
           m.metadata->>'regulation_away_score' AS reg_away
    FROM matches m
    JOIN leagues l ON l.id = m.league_id
    JOIN teams th ON th.id = m.home_team_id
    JOIN teams ta ON ta.id = m.away_team_id
    WHERE l.sport = 'soccer'
      AND m.status = 'finished'
      AND m.match_date >= NOW() - (%s || ' days')::interval
      AND m.metadata->>'result_detail' ~* '(AET|PEN)'
    ORDER BY m.match_date
"""

LEAGUE_ID_SQL = "SELECT id::text AS id FROM leagues WHERE name = %s AND country = %s AND sport = 'soccer'"

# Guarded on everything the plan read; rowcount 0 means the row moved.
UPDATE_SQL = """
    UPDATE matches
    SET metadata = COALESCE(metadata, '{}'::jsonb) || %(patch)s::jsonb,
        updated_at = NOW()
    WHERE id = %(id)s::uuid
      AND status = 'finished'
      AND home_score = %(home_score)s
      AND away_score = %(away_score)s
      AND metadata->>'result_detail' = %(result_detail)s
      AND NOT (metadata ? 'regulation_home_score')
      AND NOT (metadata ? 'regulation_away_score')
"""

CSV_FIELDS = (
    "match_id",
    "league",
    "match_date",
    "home",
    "away",
    "espn_id",
    "result_detail",
    "home_score",
    "away_score",
    "regulation_home",
    "regulation_away",
    "action",
    "reason",
)

STAMP, ALREADY_SET, REFUSE = "stamp", "already_set", "refuse"


def candidates(cur, days_back: int) -> list[dict]:
    """Finished soccer rows whose result_detail says decided after 90."""
    cur.execute(CANDIDATES_SQL, (str(days_back),))
    return [dict(r) for r in cur.fetchall() if SOCCER_PAST_90_DETAIL.search(r.get("result_detail") or "")]


def league_slugs() -> dict[tuple[str, str], str]:
    """(league name, country) -> ESPN slug, from fetch_upcoming's config."""
    return {(name, country): slug for slug, (_code, name, country) in SOCCER.leagues.items()}


def scoreboard_days(match_dates) -> list[date]:
    """Each candidate's UTC day plus the day before, oldest first."""
    days = {_utc(d).date() for d in match_dates}
    return sorted({d for day in days for d in (day, day - timedelta(days=1))})


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def entry_for(cand: dict, league: str, action: str, reason: str = "", regulation=None, espn_id=None) -> dict:
    """One CSV row for a candidate."""
    return {
        "match_id": cand["id"],
        "league": league,
        "match_date": _utc(cand["match_date"]).isoformat(),
        "home": cand["home"],
        "away": cand["away"],
        "espn_id": espn_id or cand.get("espn_id") or "",
        "result_detail": cand.get("result_detail"),
        "home_score": cand["home_score"],
        "away_score": cand["away_score"],
        "regulation_home": regulation[0] if regulation else "",
        "regulation_away": regulation[1] if regulation else "",
        "action": action,
        "reason": reason,
    }


def assess(cand: dict, comp: dict, parts: tuple) -> tuple[str, str, Optional[tuple[int, int]]]:
    """(action, reason, 90-minute score) for a candidate and the ESPN
    competition matched to it. Pure."""
    home, away = parts[0], parts[1]
    status_type = comp.get("status", {}).get("type", {})
    if parts[5] != "post" or not status_type.get("completed", False):
        return REFUSE, "ESPN: not a completed competition", None
    final = fu._final_scores(SOCCER, home, away)
    if final is None:
        return REFUSE, "ESPN: no final score", None
    if final != (cand["home_score"], cand["away_score"]):
        return REFUSE, f"ESPN final {final[0]}-{final[1]} differs from the row's", None
    detail = (status_type.get("detail") or status_type.get("shortDetail") or "").upper()
    if not fu._soccer_went_past_90(status_type, detail):
        return REFUSE, f"ESPN no longer says decided after 90 minutes ({status_type.get('name')}, {detail!r})", None
    tally = fu.scoring_play_tally(comp, home, away)
    if tally is not None and tally != (final[0], final[1], 0):
        return REFUSE, f"ESPN scoring plays add up to {tally[0]}-{tally[1]} (+{tally[2]} unattributed)", None
    regulation = fu.regulation_scores(comp, home, away, final)
    if regulation is None:
        return REFUSE, "ESPN scoring plays do not give the 90-minute score", None
    if cand.get("reg_home") is not None or cand.get("reg_away") is not None:
        stored = (cand.get("reg_home"), cand.get("reg_away"))
        if stored == (str(regulation[0]), str(regulation[1])):
            return ALREADY_SET, "", regulation
        return REFUSE, f"row already holds a different 90-minute score {stored[0]}-{stored[1]}", regulation
    return STAMP, "", regulation


def match_candidate(cur, league_id, parts: tuple, by_espn: dict, by_triple: dict) -> Optional[dict]:
    """The candidate this competition is, or None. Only candidates can
    match, so no other row is ever touched. A row without an ESPN id is
    matched by its teams, each resolved as the sweeps resolve it: its ESPN
    team id first (fetch_upcoming.find_team), since ESPN's 'Santos' is both
    Santos FC and Santos Laguna by name."""
    home, away, home_name, away_name, match_dt, _state, _venue, espn_id = parts
    if espn_id and espn_id in by_espn:
        return by_espn[espn_id]
    if not by_triple:
        return None
    home_id = fu.find_team(cur, SOCCER, home_name, league_id, espn_id=fu._competitor_espn_id(home, False))
    away_id = fu.find_team(cur, SOCCER, away_name, league_id, espn_id=fu._competitor_espn_id(away, False))
    if not (home_id and away_id):
        return None
    return by_triple.get((str(home_id), str(away_id), _utc(match_dt)))


def plan(cur, cands: list[dict], counts: Counter) -> list[dict]:
    """One CSV entry per candidate: stamp, already_set or refuse."""
    slugs = league_slugs()
    entries: dict[str, dict] = {}
    by_league: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in cands:
        by_league[(c["league_name"], c["country"])].append(c)
    for (league_name, country), group in sorted(by_league.items()):
        slug = slugs.get((league_name, country))
        if slug is None:
            for c in group:
                entries[c["id"]] = entry_for(c, league_name, REFUSE, "no ESPN slug for this league")
            continue
        cur.execute(LEAGUE_ID_SQL, (league_name, country))
        row = cur.fetchone()
        league_id = row["id"] if row else None
        by_espn = {c["espn_id"]: c for c in group if c.get("espn_id")}
        by_triple = {(c["home_id"], c["away_id"], _utc(c["match_date"])): c for c in group if not c.get("espn_id")}
        for day in scoreboard_days(c["match_date"] for c in group):
            for event in fu.fetch_day(SOCCER, slug, day):
                for comp, _ctx in fu._iter_event_competitions(event, SOCCER):
                    parts = fu._competition_parts(comp, SOCCER)
                    if parts is None:
                        continue
                    cand = match_candidate(cur, league_id, parts, by_espn, by_triple)
                    if cand is None or cand["id"] in entries:  # the same match on two fetched days
                        continue
                    action, reason, regulation = assess(cand, comp, parts)
                    entries[cand["id"]] = entry_for(cand, slug, action, reason, regulation, parts[7])
        for c in group:
            entries.setdefault(c["id"], entry_for(c, slug, REFUSE, "not found on ESPN's scoreboard"))
    ordered = [entries[c["id"]] for c in cands]
    for e in ordered:
        counts[e["action"]] += 1
        log = logger.error if e["action"] == REFUSE else logger.info
        log(
            "%s %s %s v %s %s-%s (%s, %s): %s %s%s",
            e["league"],
            e["match_date"][:10],
            e["home"],
            e["away"],
            e["home_score"],
            e["away_score"],
            e["result_detail"],
            e["match_id"],
            e["action"],
            f"90 min {e['regulation_home']}-{e['regulation_away']} " if e["regulation_home"] != "" else "",
            e["reason"],
        )
    return ordered


def update_params(entry: dict) -> dict:
    return {
        "id": entry["match_id"],
        "patch": json.dumps(
            {
                REGULATION_HOME_KEY: int(entry["regulation_home"]),
                REGULATION_AWAY_KEY: int(entry["regulation_away"]),
                "went_extra": True,
            }
        ),
        "home_score": entry["home_score"],
        "away_score": entry["away_score"],
        "result_detail": entry["result_detail"],
    }


def default_out_csv() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base = os.environ.get("BACKUP_LOCAL_DIR", "/app/backups")
    return f"{base}/soccer_regulation_scores_{stamp}.csv"


def write_csv(path: str, entries: list[dict]) -> None:
    """Write-once: mode 'x' refuses an existing file, so a re-run can never
    overwrite an earlier run's audit record."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = out.open("x", newline="")
    except FileExistsError:
        raise FileExistsError(f"{out} exists — the audit CSV is write-once") from None
    with fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(entries)


def run(conn, days_back: int, apply_changes: bool, out_csv: Optional[str] = None, skip_refused: bool = False):
    counts: Counter = Counter()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cands = candidates(cur, days_back)
        counts["candidates"] = len(cands)
        entries = plan(cur, cands, counts)
    conn.rollback()  # close the read snapshot before writing

    out_csv = out_csv or default_out_csv()
    write_csv(out_csv, entries)
    counts["espn_requests_failed"] = fu.FETCH_STATS["failed"]
    logger.info(
        "%d candidate(s): %d to stamp, %d already set, %d refused; audit CSV %s",
        len(cands),
        counts[STAMP],
        counts[ALREADY_SET],
        counts[REFUSE],
        out_csv,
    )
    # One fixed-format line for scripts/ops/backfill_soccer_regulation_scores.sh.
    logger.info(
        "PLAN candidates=%d stamp=%d already_set=%d refuse=%d",
        len(cands),
        counts[STAMP],
        counts[ALREADY_SET],
        counts[REFUSE],
    )
    if fu.FETCH_STATS["failed"] and not fu.FETCH_STATS["ok"]:
        raise RuntimeError(f"Every ESPN request failed ({fu.FETCH_STATS['failed']}) — nothing written")
    if fu.FETCH_STATS["failed"]:
        logger.error("%d ESPN scoreboard request(s) failed; their matches are refused", fu.FETCH_STATS["failed"])

    to_stamp = [e for e in entries if e["action"] == STAMP]
    if counts[REFUSE] and not skip_refused:
        logger.error("%d candidate(s) refused — nothing written (inspect them, or pass --skip-refused)", counts[REFUSE])
        return counts
    if not apply_changes:
        logger.info("DRY-RUN: re-run with --apply to stamp %d row(s)", len(to_stamp))
        return counts
    with conn.cursor() as cur:
        for e in to_stamp:
            cur.execute(UPDATE_SQL, update_params(e))
            if cur.rowcount != 1:
                conn.rollback()
                raise RuntimeError(
                    f"{e['match_id']} ({e['home']} v {e['away']}) changed between plan and write "
                    f"(rowcount {cur.rowcount}) — rolled back, nothing written"
                )
            counts["written"] += 1
    conn.commit()
    logger.info("Stamped the 90-minute score on %d row(s)", counts["written"])
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days-back", type=int, default=365, help="Finished matches this recent (default 365).")
    p.add_argument("--apply", action="store_true", help="Write (default: dry run).")
    p.add_argument(
        "--skip-refused",
        action="store_true",
        help="Stamp the planned rows even when others were refused (default: a refusal writes nothing).",
    )
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
    if args.days_back <= 0:
        logger.error("--days-back must be positive (got %d)", args.days_back)
        return 2
    with psycopg2.connect(args.database_url) as conn:
        counts = run(conn, args.days_back, args.apply, args.out_csv, args.skip_refused)
    logger.info("Done: %s", dict(sorted(counts.items())))
    return 1 if counts[REFUSE] else 0


if __name__ == "__main__":
    sys.exit(main())
