"""Backfill match_stats.expected_goals(_against) from Understat (audit §3
rank 5 — the open soccer model lever; expected_goals is 100% NULL today).

Understat redesigned in 2025: league pages are JS stubs and the data comes
from  GET https://understat.com/getLeagueData/{league}/{season}  (JSON with
'dates': one row per match — h/a titles, goals, xG h/a, datetime). Covered
leagues in our corpus: EPL, La Liga, Serie A, Ligue 1, Bundesliga (~17.9k
matches, 76.5%). The Championship (our largest league) is NOT on Understat.

Matching discipline (the deleted scraper's sins, inverted):
  - exact normalized-name + date matching with an explicit alias map —
    no ILIKE '%name%' fuzzing;
  - the final score must AGREE between Understat and our row, else the
    match is skipped loudly (never attach xG to the wrong fixture);
  - TWO match_stats rows per match with REAL team ids (the old scraper
    wrote the match UUID into team_id);
  - unmatched titles are counted and printed so the alias map can be
    extended deliberately.

Usage (inside the api container):
    python /app/scripts/backfill_understat_xg.py --seasons 2016-2025
    python /app/scripts/backfill_understat_xg.py --league EPL --seasons 2024
    python /app/scripts/backfill_understat_xg.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("backfill_understat_xg")

BASE_URL = "https://understat.com/getLeagueData"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0",
    "X-Requested-With": "XMLHttpRequest",
    "Accept-Encoding": "gzip",
}
TIMEOUT_S = 30
SLEEP_BETWEEN_FETCHES_S = 1.0

# Understat league slug -> our leagues.name
LEAGUES = {
    "EPL": "Premier League",
    "La_liga": "La Liga",
    "Serie_A": "Serie A",
    "Ligue_1": "Ligue 1",
    "Bundesliga": "Bundesliga",
}

# Understat title -> our normalized team name, for names that normalize
# differently. Extend deliberately when the unmatched report shows a new
# alias — never loosen the matcher itself.
ALIASES = {
    # EPL
    "wolverhampton wanderers": "wolves",
    "manchester united": "man united",
    "manchester city": "man city",
    "newcastle united": "newcastle",
    "west bromwich albion": "west brom",
    "nottingham forest": "nott m forest",
    # Bundesliga (ours are football-data.co.uk short forms)
    "borussia dortmund": "dortmund",
    "borussia m gladbach": "m gladbach",
    "rasenballsport leipzig": "leipzig",
    "mainz 05": "mainz",
    "eintracht frankfurt": "ein frankfurt",
    "bayer leverkusen": "leverkusen",
    "cologne": "koln",
    "fortuna duesseldorf": "fortuna dusseldorf",
    "hertha berlin": "hertha",
    "arminia bielefeld": "bielefeld",
    "greuther fuerth": "greuther furth",
    "hamburger": "hamburg",
    "nuernberg": "nurnberg",
    # La Liga
    "atletico madrid": "ath madrid",
    "athletic club": "ath bilbao",
    "real sociedad": "sociedad",
    "real betis": "betis",
    "espanyol": "espanol",
    "rayo vallecano": "vallecano",
    "deportivo la coruna": "la coruna",
    "sporting gijon": "sp gijon",
    "real valladolid": "valladolid",
    "real oviedo": "oviedo",
    "celta vigo": "celta",
    # Serie A
    "hellas verona": "verona",
    "parma calcio 1913": "parma",
    # Ligue 1
    "paris saint germain": "paris sg",
    "saint etienne": "st etienne",
}


def norm(name: str) -> str:
    """Accent-strip, lowercase, drop punctuation + filler tokens."""
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r"\b(fc|cf|afc|ac|as|ssc|rc|sc|sv|vfb|vfl|tsg|rb|1899|1)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)


def fetch_season(session: requests.Session, slug: str, season: int) -> list[dict]:
    resp = session.get(f"{BASE_URL}/{slug}/{season}", headers=HEADERS, timeout=TIMEOUT_S)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("dates") or []
    out = []
    for r in rows:
        if not r.get("isResult"):
            continue
        xg = r.get("xG") or {}
        goals = r.get("goals") or {}
        if xg.get("h") is None or xg.get("a") is None:
            continue
        try:
            out.append(
                {
                    "home": r["h"]["title"],
                    "away": r["a"]["title"],
                    "date": datetime.fromisoformat(r["datetime"]).date(),
                    "home_goals": int(goals["h"]),
                    "away_goals": int(goals["a"]),
                    "home_xg": float(xg["h"]),
                    "away_xg": float(xg["a"]),
                }
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.error("%s/%s: malformed row (%s) — skipping", slug, season, e)
    return out


def load_our_matches(cur, league_name: str) -> dict:
    """(date, norm_home, norm_away) -> match row, with ±1-day duplicates so
    timezone drift between Understat kickoff dates and ours still matches."""
    cur.execute(
        """
        SELECT m.id::text AS match_id, m.match_date::date AS d,
               m.home_score, m.away_score,
               ht.id::text AS home_team_id, ht.name AS home_name,
               at.id::text AS away_team_id, at.name AS away_name
        FROM matches m
        JOIN leagues l ON l.id = m.league_id AND l.sport = 'soccer'
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        WHERE l.name = %s AND m.status = 'finished'
        """,
        (league_name,),
    )
    idx: dict = {}
    for r in cur.fetchall():
        key_base = (norm(r["home_name"]), norm(r["away_name"]))
        for delta in (0, -1, 1):
            idx.setdefault((r["d"] + timedelta(days=delta),) + key_base, r)
    return idx


def upsert_xg(cur, match_id: str, team_id: str, xg: float, xga: float) -> None:
    cur.execute(
        """
        INSERT INTO match_stats (match_id, team_id, expected_goals, expected_goals_against)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (match_id, team_id) DO UPDATE
        SET expected_goals = EXCLUDED.expected_goals,
            expected_goals_against = EXCLUDED.expected_goals_against
        """,
        (match_id, team_id, xg, xga),
    )


def run(database_url: str, slugs: list[str], seasons: list[int], dry_run: bool) -> dict:
    totals = Counter()
    unmatched: Counter = Counter()
    session = requests.Session()
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for slug in slugs:
                idx = load_our_matches(cur, LEAGUES[slug])
                logger.info("%s: %d finished matches in our corpus", slug, len(idx) // 3)
                for season in seasons:
                    try:
                        rows = fetch_season(session, slug, season)
                    except (requests.RequestException, ValueError) as e:
                        logger.error("%s/%s: fetch failed: %s", slug, season, e)
                        totals["fetch_failures"] += 1
                        continue
                    totals["fetched"] += len(rows)
                    for r in rows:
                        ours = idx.get((r["date"], norm(r["home"]), norm(r["away"])))
                        if ours is None:
                            unmatched[f'{slug}: {r["home"]} vs {r["away"]}'] += 1
                            totals["unmatched"] += 1
                            continue
                        if (ours["home_score"], ours["away_score"]) != (r["home_goals"], r["away_goals"]):
                            logger.error(
                                "%s %s: score mismatch (understat %d-%d vs ours %s-%s) — skipping",
                                r["date"],
                                r["home"],
                                r["home_goals"],
                                r["away_goals"],
                                ours["home_score"],
                                ours["away_score"],
                            )
                            totals["score_mismatch"] += 1
                            continue
                        if not dry_run:
                            upsert_xg(cur, ours["match_id"], ours["home_team_id"], r["home_xg"], r["away_xg"])
                            upsert_xg(cur, ours["match_id"], ours["away_team_id"], r["away_xg"], r["home_xg"])
                        totals["matched"] += 1
                    time.sleep(SLEEP_BETWEEN_FETCHES_S)
            if not dry_run:
                conn.commit()

    logger.info(
        "Done: %(fetched)d fetched, %(matched)d matched, %(unmatched)d unmatched, "
        "%(score_mismatch)d score-mismatched, %(fetch_failures)d fetch failures",
        {k: totals.get(k, 0) for k in ("fetched", "matched", "unmatched", "score_mismatch", "fetch_failures")},
    )
    for title, n in unmatched.most_common(20):
        logger.warning("unmatched: %s (x%d)", title, n)
    return dict(totals)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--league", choices=sorted(LEAGUES), help="Single Understat league slug (default: all five).")
    p.add_argument(
        "--seasons",
        default="2016-2025",
        help="Season start-years: '2016-2025' or '2024' or '2022,2023' (default 2016-2025).",
    )
    p.add_argument("--dry-run", action="store_true", help="Fetch + match, write nothing.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def parse_seasons(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    slugs = [args.league] if args.league else sorted(LEAGUES)
    totals = run(args.database_url, slugs, parse_seasons(args.seasons), args.dry_run)
    if totals.get("matched", 0) == 0 and totals.get("fetched", 0) > 0:
        logger.error("Fetched matches but matched NONE — name normalization or league mapping is broken")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
