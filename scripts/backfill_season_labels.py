"""Relabel matches.season on soccer rows to the one season scheme.

Two writers labelled soccer seasons wrongly until 2026-10-31:

  * promote_raw.season_label read every 4-digit football-data season as a
    'YYZZ' code, so a calendar-year season of the /new/ country files
    (Argentina, Brazil, China, Finland, Ireland, Japan, MLS, Norway, Sweden)
    was stored as '2020-2116' for 2016 and as '2020-2021' for 2021 (looks
    right, is not), and the cross-year files' 'YYYY/YYYY' was stored verbatim
    beside the 'YYYY-YYYY' every other loader writes;
  * fetch_upcoming labelled EVERY league Aug->May (soccer_seasons.
    cross_year_label), so the ESPN-created rows of a calendar-year league
    (MLS, Brasileirao, Chile, Colombia, the World Cup ...) carry '2025-2026'
    before July and '2026-2027' after it.

Both now label by scripts/soccer_seasons.py. Their conflict branches never
touch season, so re-running them does not repair the rows already written.
This does, per row:

  * a row tied to a raw_football_data row (league code, normalised team
    names, and the kickoff as promote_raw wrote it until 2026-10-31 — the UK
    wall clock read in the database's TimeZone — or the corrected UTC
    kickoff): the label is football-data's season
    (promote_raw.season_label); football-data names it, even in a league's
    irregular eras;
  * any other row: soccer_seasons.season_for_kickoff for its league and
    kickoff; a kickoff in an irregular era has no date rule and is left
    alone (counted as left_no_date_rule).

A row is relabelled only when its current label is one a known old writer
produced for it: the old promote_raw label of the raw season it is tied to
(kind 'promote_raw'), or the Aug->May label of its kickoff (kind 'espn').
Anything else — a row edited since, another loader's label, raw duplicates
that disagree, a raw tie in another league — is left alone and counted.
Every planned change goes to a write-once CSV first, dry run included.

--apply re-plans and writes in ONE transaction, every UPDATE guarded on the
old value; a row that changed since the plan rolls the whole run back, so a
failed apply changes nothing. --expect-plan pins the apply to the counts a
reviewed dry run printed (its PLAN line): any drift refuses before a write.
--verify <apply CSV> checks, read-only, that every row of that CSV now holds
its new label.

Nothing live reads the soccer label (checked 2026-10-31): the soccer
training query (training_data.DEFAULT_TRAINING_QUERY), compute_features,
precompute_predictions, generate_recommendations, the Dixon-Coles trainers
and monitor_models never select it, and features_cache holds only the
'baseline' set. It is displayed by the match API, grouped on by the
ab_soccer_feature_screen harness and the dormant feature-engineering
orchestrator, and turned into fbref URLs by scrape_fbref_venues (fbref
spells these seasons '2016' and '2016-2017').

Usage (inside the api container):
    python /app/scripts/backfill_season_labels.py                     # dry run
    python /app/scripts/backfill_season_labels.py --apply --expect-plan relabel_promote_raw=N,relabel_espn=M
    python /app/scripts/backfill_season_labels.py --verify /app/backups/season_labels_apply_<utc>.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import promote_raw as pr  # noqa: E402
import soccer_seasons  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_season_labels")

RAW_SQL = 'SELECT league, season, "Date", "Time", "HomeTeam", "AwayTeam" FROM raw_football_data'
LEAGUES_SQL = (
    "SELECT id::text AS id, external_ids->>'football_data' AS code FROM leagues "
    "WHERE sport = 'soccer' AND external_ids ? 'football_data'"
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
VERIFY_SQL = "SELECT id::text AS id, season FROM matches WHERE id = ANY(%s::uuid[])"

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
_YEAR_PAIR = re.compile(r"(\d{4})[-/](\d{4})")
CSV_FIELDS = ("match_id", "league", "match_date", "home", "away", "kind", "raw_season", "old_season", "new_season")

KIND_PROMOTE_RAW = "promote_raw"  # the old promote_raw label of the raw season the row is tied to
KIND_ESPN = "espn"  # the Aug->May label fetch_upcoming gave every league
KINDS = (KIND_PROMOTE_RAW, KIND_ESPN)

# Counters a re-plan after a clean apply must reproduce exactly (nothing but
# the planned rows changed).
LEFT_ALONE_COUNTERS = (
    "left_unknown_writer",
    "left_no_date_rule",
    "left_raw_disagree",
    "left_league_mismatch",
    "left_no_league_code",
)
PLAN_COUNTERS = (
    "relabel",
    "relabel_promote_raw",
    "relabel_espn",
    "relabel_bad_shape",
    "already_correct",
    *LEFT_ALONE_COUNTERS,
    "raw_label_not_canonical",
    "raw_row_unresolved",
    "raw_no_matches_row",
)


class PlanDrift(RuntimeError):
    """--expect-plan does not match the plan computed at apply time."""


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


def is_bad_shape(label: Optional[str]) -> bool:
    """A label only the old promote_raw produced: 'YYYY/YYYY', or a
    'YYYY-YYYY' that does not span one year ('2020-2116'). The ops wrapper's
    after-state counts these in SQL (the same test)."""
    m = _YEAR_PAIR.fullmatch(label or "")
    return bool(m) and ("/" in (label or "") or int(m.group(2)) - int(m.group(1)) != 1)


def kickoff_key(match_dt: datetime, db_tz: tzinfo) -> datetime:
    """A kickoff as an aware UTC datetime. promote_raw inserts naive
    datetimes, which Postgres reads in its session TimeZone."""
    if match_dt.tzinfo is None:
        match_dt = match_dt.replace(tzinfo=db_tz)
    return match_dt.astimezone(timezone.utc)


def _iso(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def tie_raw_rows(
    raw_rows: Iterable[dict],
    known_codes: set[str],
    team_ids: dict[str, str],
    matches_by_key: dict[tuple, dict],
    db_tz: tzinfo,
    counts: Counter,
) -> dict[str, list[tuple[str, dict]]]:
    """match id -> [(league code, raw row), ...] for every raw row that ties
    to a soccer matches row, at the legacy kickoff, the corrected one, or
    both (a football-data row 1 h late beside the ESPN row of the same
    fixture). Pure."""
    ties: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for r in raw_rows:
        code = r.get("league")
        home_id = team_ids.get(pr.normalize_team_name(r.get("HomeTeam") or ""))
        away_id = team_ids.get(pr.normalize_team_name(r.get("AwayTeam") or ""))
        wall_clock = pr.parse_wall_clock(r.get("Date") or "", r.get("Time"))
        if not (code in known_codes and home_id and away_id and wall_clock):
            counts["raw_row_unresolved"] += 1
            continue
        corrected = pr.parse_match_date(r.get("Date") or "", r.get("Time"))
        keys = {kickoff_key(wall_clock, db_tz)} | ({kickoff_key(corrected, db_tz)} if corrected else set())
        found = [matches_by_key[(home_id, away_id, k)] for k in sorted(keys) if (home_id, away_id, k) in matches_by_key]
        if not found:
            counts["raw_no_matches_row"] += 1
            continue
        for m in found:
            ties[m["id"]].append((code, r))
    return ties


def plan(
    raw_rows: Iterable[dict],
    league_codes: dict[str, str],
    team_ids: dict[str, str],
    matches: list[dict],
    db_tz: tzinfo,
    counts: Counter,
    team_names: Optional[dict[str, str]] = None,
) -> list[dict]:
    """The relabels to make, one per matches row. Pure.

    `league_codes` maps leagues.id -> football-data code; `matches` holds
    every soccer row (id, league_id, home_id, away_id, match_date, season)."""
    matches_by_key = {(m["home_id"], m["away_id"], kickoff_key(m["match_date"], db_tz)): m for m in matches}
    ties = tie_raw_rows(raw_rows, set(league_codes.values()), team_ids, matches_by_key, db_tz, counts)
    names = team_names or {}
    planned: list[dict] = []
    for m in matches:
        code = league_codes.get(m["league_id"])
        if code is None:
            counts["left_no_league_code"] += 1
            continue
        kickoff = kickoff_key(m["match_date"], db_tz)
        raw_season = ""
        tied = ties.get(m["id"], [])
        if tied:
            if any(raw_code != code for raw_code, _r in tied):
                counts["left_league_mismatch"] += 1
                continue
            expected = {pr.season_label(r.get("season") or "", code) for _c, r in tied}
            if len(expected) > 1:
                counts["left_raw_disagree"] += 1
                continue
            new = expected.pop()
            if not CANONICAL_SEASON.fullmatch(new):
                counts["raw_label_not_canonical"] += 1
                continue
            raw_season = tied[0][1].get("season") or ""
            known_old = {legacy_season_label(r.get("season") or ""): KIND_PROMOTE_RAW for _c, r in tied}
        else:
            date_rule = soccer_seasons.season_for_kickoff(code, kickoff)
            if date_rule is None:
                counts["left_no_date_rule"] += 1
                continue
            new = date_rule
            known_old = {}
        # The Aug->May label the ESPN path wrote, whoever wrote the row since.
        known_old.setdefault(soccer_seasons.cross_year_label(kickoff), KIND_ESPN)
        if m["season"] == new:
            counts["already_correct"] += 1
            continue
        kind = known_old.get(m["season"])
        if kind is None:
            counts["left_unknown_writer"] += 1
            continue
        counts["relabel"] += 1
        counts[f"relabel_{kind}"] += 1
        counts["relabel_bad_shape"] += is_bad_shape(m["season"])
        planned.append(
            {
                "match_id": m["id"],
                "league": code,
                "match_date": _iso(m["match_date"]),
                "home": names.get(m["home_id"], m["home_id"]),
                "away": names.get(m["away_id"], m["away_id"]),
                "kind": kind,
                "raw_season": raw_season,
                "old_season": m["season"],
                "new_season": new,
            }
        )
    return planned


def plan_line(counts: Counter) -> str:
    """The one line a wrapper parses: 'PLAN key=n ...'."""
    return "PLAN " + " ".join(f"{k}={counts[k]}" for k in PLAN_COUNTERS)


def plan_leagues_line(entries: list[dict]) -> str:
    """'PLAN_LEAGUES promote_raw=AR1,AT1 espn=MLS,WC': which leagues each kind
    touches, so a wrapper can hold them to the reviewed set."""
    by_kind = {kind: sorted({e["league"] for e in entries if e["kind"] == kind}) for kind in KINDS}
    return "PLAN_LEAGUES " + " ".join(f"{kind}={','.join(codes) or '-'}" for kind, codes in by_kind.items())


def parse_expect_plan(spec: str) -> dict[str, int]:
    """'relabel_promote_raw=N,relabel_espn=M' -> {kind counter: n}."""
    out: dict[str, int] = {}
    for part in (spec or "").split(","):
        key, sep, value = part.strip().partition("=")
        if not sep or key not in PLAN_COUNTERS or not value.isdigit():
            raise ValueError(f"--expect-plan: bad item {part!r} (want key=N, key one of {PLAN_COUNTERS})")
        out[key] = int(value)
    return out


def default_out_csv(kind: str = "dryrun") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base = os.environ.get("BACKUP_LOCAL_DIR", "/app/backups")
    return f"{base}/season_labels_{kind}_{stamp}.csv"


def write_csv(path: str, entries: list[dict]) -> None:
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


def load_inputs(cur) -> tuple[list[dict], dict[str, str], dict[str, str], list[dict], tzinfo]:
    cur.execute("SHOW TimeZone")
    db_tz = ZoneInfo(cur.fetchone()["TimeZone"])  # loud if Postgres reports a zone Python cannot load
    cur.execute(RAW_SQL)
    raw_rows = [dict(r) for r in cur.fetchall()]
    cur.execute(LEAGUES_SQL)
    league_codes = {r["id"]: r["code"] for r in cur.fetchall()}
    cur.execute(TEAMS_SQL)
    team_ids = {r["normalized_name"]: r["id"] for r in cur.fetchall()}
    cur.execute(MATCHES_SQL)
    matches = [dict(r) for r in cur.fetchall()]
    return raw_rows, league_codes, team_ids, matches, db_tz


def run(conn, apply_changes: bool, out_csv: Optional[str] = None, expect: Optional[dict[str, int]] = None) -> Counter:
    counts: Counter = Counter()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        raw_rows, league_codes, team_ids, matches, db_tz = load_inputs(cur)
    conn.rollback()  # close the read snapshot before writing
    logger.info(
        "Loaded %d raw rows, %d soccer leagues, %d soccer teams, %d soccer matches (db TimeZone %s)",
        len(raw_rows),
        len(league_codes),
        len(team_ids),
        len(matches),
        db_tz,
    )
    names = {tid: name for name, tid in team_ids.items()}
    entries = plan(raw_rows, league_codes, team_ids, matches, db_tz, counts, names)
    out_csv = out_csv or default_out_csv("apply" if apply_changes else "dryrun")
    write_csv(out_csv, entries)
    by_league = Counter((e["league"], e["kind"]) for e in entries)
    logger.info("Planned %d relabel(s); audit CSV %s", len(entries), out_csv)
    for (league, kind), n in sorted(by_league.items()):
        logger.info("  %-5s %-11s %d", league, kind, n)
    print(plan_line(counts), flush=True)
    print(plan_leagues_line(entries), flush=True)
    if counts["left_raw_disagree"]:
        logger.error("%d row(s) left alone: their raw_football_data duplicates disagree", counts["left_raw_disagree"])
    if counts["left_league_mismatch"]:
        logger.error("%d row(s) left alone: tied to a raw row of another league", counts["left_league_mismatch"])
    if counts["raw_label_not_canonical"]:
        logger.error("%d row(s) tied to a raw season that is not a label", counts["raw_label_not_canonical"])

    if not apply_changes:
        logger.info("DRY-RUN: re-run with --apply to relabel %d row(s)", len(entries))
        return counts
    if expect is not None:
        got = {k: counts[k] for k in expect}
        if got != expect:
            raise PlanDrift(f"plan {got} differs from --expect-plan {expect} — nothing written; review a fresh dry run")
    try:
        with conn.cursor() as cur:
            updated = execute_values(
                cur,
                UPDATE_SQL,
                [(e["match_id"], e["old_season"], e["new_season"]) for e in entries],
                page_size=1000,
                fetch=True,
            )
        if len(updated) != len(entries):
            raise RuntimeError(
                f"{len(entries) - len(updated)} of {len(entries)} row(s) changed between plan and write — "
                "rolled back, nothing written"
            )
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    counts["relabelled"] = len(updated)
    logger.info("Relabelled %d row(s)", counts["relabelled"])
    return counts


def verify(conn, csv_path: str) -> Counter:
    """Read-only: does every row of an apply CSV now hold its new label?"""
    rows = list(csv.DictReader(open(csv_path, newline="")))
    counts: Counter = Counter(rows=len(rows))
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(VERIFY_SQL, ([r["match_id"] for r in rows],))
        now = {r["id"]: r["season"] for r in cur.fetchall()}
    conn.rollback()
    for r in rows:
        season = now.get(r["match_id"])
        if season is None:
            counts["missing"] += 1
        elif season == r["new_season"]:
            counts["now_new"] += 1
        elif season == r["old_season"]:
            counts["still_old"] += 1
        else:
            counts["other"] += 1
    print(
        "VERIFY " + " ".join(f"{k}={counts[k]}" for k in ("rows", "now_new", "still_old", "other", "missing")),
        flush=True,
    )
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="Write (default: dry run).")
    p.add_argument(
        "--expect-plan",
        default=None,
        help="With --apply: the dry run's counts (e.g. relabel_promote_raw=N,relabel_espn=M); any drift refuses.",
    )
    p.add_argument("--verify", default=None, metavar="CSV", help="Read-only check of an apply CSV; writes nothing.")
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
    try:
        expect = parse_expect_plan(args.expect_plan) if args.expect_plan else None
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    if expect is not None and not args.apply:
        logger.error("--expect-plan pins an --apply; it means nothing on a dry run")
        return 2
    with psycopg2.connect(args.database_url) as conn:
        if args.verify:
            counts = verify(conn, args.verify)
            return 0 if counts["now_new"] == counts["rows"] else 1
        try:
            counts = run(conn, args.apply, args.out_csv, expect)
        except PlanDrift as exc:
            logger.error("%s", exc)
            return 1
    logger.info("Done: %s", dict(sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
