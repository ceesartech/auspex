"""Ingest UK + Irish horse racing data from theracingapi.com.

Plan-aware endpoint dispatch:
  Basic ($30/mo)   → /racecards/basic (TODAY only, no date param,
                     no odds). /results is Standard+.
  Standard         → /racecards/standard (TODAY only, no date param,
                     bookmaker odds array per runner) +
                     /results (date-rangeable via start_date/end_date,
                     with sp/sp_dec and finish position).
  Pro              → /racecards/pro (date-rangeable, deepest payload)
                     + /results + per-horse stable/quotes data.

The script tries pro → standard → basic in order and falls back on
401/403/422 ("Plan required" or rejected params). On Basic, the
date range to --upcoming N is silently capped to today (one call);
ditto on Standard since /racecards/standard also returns today only.
--results is the multi-day workhorse (Standard+).

Auth: HTTP Basic with THE_RACING_API_USER + THE_RACING_API_PASS.

Lazy entity creation: horses / jockeys / trainers get one DB row
per unique normalized_name. Same horse appearing in many races
resolves to one horses.id, letting rolling-form features be cheap
JOINs at training time.

Usage:
    # Daily upcoming-racecards ingest (DAG runs this)
    python scripts/load_racing_api.py --upcoming 7

    # Historical backfill (Standard+ plan only)
    python scripts/load_racing_api.py --results --start 2024-01-01 --end 2024-12-31

    # Single-day diagnostic
    python scripts/load_racing_api.py --results --start 2024-09-08 --end 2024-09-08 --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional

import psycopg2
import requests
from psycopg2.extras import Json, RealDictCursor
from requests.auth import HTTPBasicAuth

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("load_racing_api")

BASE_URL = "https://api.theracingapi.com/v1"
REQUEST_DELAY_SEC = 0.20
DEFAULT_TIMEOUT_SEC = 30

# Furlong → meters (1 furlong = 201.168 m). Used to normalise the
# Racing API's distance representation (always furlongs as integers
# in the `distance_f` field).
METERS_PER_FURLONG = 201.168


# ── Auth helper ────────────────────────────────────────────────────


def _auth() -> Optional[HTTPBasicAuth]:
    """Return HTTPBasicAuth(user, pass) from env, or None if missing."""
    user = os.environ.get("THE_RACING_API_USER")
    pwd = os.environ.get("THE_RACING_API_PASS")
    if not (user and pwd):
        return None
    return HTTPBasicAuth(user, pwd)


# ── Normalization helpers ──────────────────────────────────────────


def normalize_name(name: str) -> str:
    """Same loose normalization shape as teams.normalized_name —
    lowercase, collapsed whitespace, strip country-suffix parentheticals.
    'Big Brown (USA)' and 'BIG BROWN' both collapse to 'big brown'."""
    if not name:
        return ""
    # Drop country-code suffixes like (USA), (GB), (IRE)
    cleaned = re.sub(r"\s*\([A-Z]{2,4}\)\s*$", "", name.strip())
    return " ".join(cleaned.lower().split())


def parse_country_suffix(name: str) -> Optional[str]:
    """Pull the (XXX) country code off a horse name if present."""
    m = re.search(r"\(([A-Z]{2,4})\)\s*$", (name or "").strip())
    return m.group(1) if m else None


def furlongs_to_meters(furlongs: Optional[float]) -> Optional[int]:
    """Convert distance from furlongs (Racing API native) to meters
    (our SI-unit storage). Rounds to nearest meter."""
    if furlongs is None:
        return None
    try:
        return int(round(float(furlongs) * METERS_PER_FURLONG))
    except (TypeError, ValueError):
        return None


def parse_offdt(raw: Optional[str]) -> Optional[datetime]:
    """Racing API uses ISO 8601 with optional 'Z'. Returns tz-aware UTC."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ── HTTP fetchers ──────────────────────────────────────────────────


# Plan-aware fetcher caches the highest-tier endpoint that works on
# the current credentials. Saves a probe call on every request once
# the tier is known.
_CACHED_RACECARDS_PATH: Optional[str] = None


def _try_racecards_endpoint(path: str, day: date, auth: HTTPBasicAuth, send_date: bool) -> tuple[int, list[dict]]:
    """Returns (status_code, racecards). status_code lets the caller
    decide whether to fall back to a lower-tier endpoint."""
    url = f"{BASE_URL}{path}"
    params: dict = {}
    if send_date:
        params["date"] = day.isoformat()
    try:
        r = requests.get(url, params=params, auth=auth, timeout=DEFAULT_TIMEOUT_SEC)
    except requests.RequestException as e:
        logger.warning("Racing API %s failed for %s: %s", path, day, e)
        return 0, []
    if r.status_code >= 400:
        return r.status_code, []
    return r.status_code, r.json().get("racecards", []) or []


# Only /racecards/pro accepts a `date` query param. /racecards/standard
# and /racecards/basic both reject it with 422 and return today's card
# regardless. Keep this in sync with _ONLY_TODAY_ENDPOINTS in run().
_PATHS_THAT_ACCEPT_DATE = frozenset({"/racecards/pro"})
_ONLY_TODAY_ENDPOINTS = frozenset({"/racecards/standard", "/racecards/basic"})


def fetch_racecards(day: date, auth: HTTPBasicAuth) -> list[dict]:
    """Pull racecards. Tries /racecards/pro → /standard → /basic and
    caches the first tier that succeeds. Standard + Basic return
    today's racecards regardless of the day param, so we don't send
    one to them (would 422). Only /pro is date-rangeable."""
    global _CACHED_RACECARDS_PATH
    if _CACHED_RACECARDS_PATH:
        send_date = _CACHED_RACECARDS_PATH in _PATHS_THAT_ACCEPT_DATE
        _, cards = _try_racecards_endpoint(_CACHED_RACECARDS_PATH, day, auth, send_date)
        return cards

    for path in ("/racecards/pro", "/racecards/standard", "/racecards/basic"):
        send_date = path in _PATHS_THAT_ACCEPT_DATE
        status, cards = _try_racecards_endpoint(path, day, auth, send_date)
        if status == 200:
            _CACHED_RACECARDS_PATH = path
            logger.info("Racing API tier detected: %s", path)
            return cards
        if status not in (401, 402, 403, 422):
            # Network error / unexpected — log and keep probing.
            logger.warning("Racing API %s returned %s", path, status)
    logger.error("Racing API: none of pro/standard/basic worked. " "Check THE_RACING_API_USER + THE_RACING_API_PASS.")
    return []


def fetch_results(day: date, auth: HTTPBasicAuth) -> list[dict]:
    """Pull finished-race results for one day. Requires Standard+
    plan — Basic returns 403 'Standard Plan required'. The caller
    logs the plan limitation and falls back to forward-accumulation."""
    url = f"{BASE_URL}/results"
    params = {"start_date": day.isoformat(), "end_date": day.isoformat()}
    try:
        r = requests.get(url, params=params, auth=auth, timeout=DEFAULT_TIMEOUT_SEC)
    except requests.RequestException as e:
        logger.warning("Racing API results failed for %s: %s", day, e)
        return []
    if r.status_code >= 400:
        body_snippet = r.text[:200] if r.text else ""
        logger.warning("Racing API results %s → %d: %s", day, r.status_code, body_snippet)
        return []
    return r.json().get("results", []) or []


# ── DB I/O ─────────────────────────────────────────────────────────


def ensure_league(cur, course: str, country: str) -> Optional[str]:
    """One league row per track. sport='horse_racing' so the
    matches-based sports don't accidentally pick these up.
    Returns the league_id or None on failure."""
    cur.execute(
        """
        INSERT INTO leagues (name, country, sport, external_ids)
        VALUES (%s, %s, 'horse_racing', %s)
        ON CONFLICT (name, country, sport) DO UPDATE
            SET external_ids = leagues.external_ids || EXCLUDED.external_ids
        RETURNING id
        """,
        (course, country, Json({"racing_api": course})),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def ensure_horse(cur, name: str) -> Optional[str]:
    """Insert OR lookup a horse by normalized name. Country suffix
    is parsed and stored separately so 'Big Brown (USA)' and 'Big
    Brown' resolve to the same row."""
    if not name:
        return None
    norm = normalize_name(name)
    if not norm:
        return None
    country = parse_country_suffix(name)
    cur.execute(
        """
        INSERT INTO horses (name, normalized_name, country)
        VALUES (%s, %s, %s)
        ON CONFLICT (normalized_name) DO UPDATE
            SET name = EXCLUDED.name,
                country = COALESCE(horses.country, EXCLUDED.country)
        RETURNING id
        """,
        (name, norm, country),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def ensure_jockey(cur, name: str) -> Optional[str]:
    if not name:
        return None
    norm = normalize_name(name)
    if not norm:
        return None
    cur.execute(
        """
        INSERT INTO jockeys (name, normalized_name)
        VALUES (%s, %s)
        ON CONFLICT (normalized_name) DO UPDATE
            SET name = EXCLUDED.name
        RETURNING id
        """,
        (name, norm),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def ensure_trainer(cur, name: str) -> Optional[str]:
    if not name:
        return None
    norm = normalize_name(name)
    if not norm:
        return None
    cur.execute(
        """
        INSERT INTO trainers (name, normalized_name)
        VALUES (%s, %s)
        ON CONFLICT (normalized_name) DO UPDATE
            SET name = EXCLUDED.name
        RETURNING id
        """,
        (name, norm),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def upsert_race(cur, *, league_id: str, payload: dict, status: str) -> Optional[str]:
    """Insert OR update a race. Returns race_id."""
    off_dt = parse_offdt(payload.get("off_dt") or payload.get("off"))
    if not off_dt:
        return None
    cur.execute(
        """
        INSERT INTO races
            (league_id, external_ids, race_date, track_name, race_number,
             distance_meters, surface, track_condition, race_class,
             purse_currency, purse_amount, field_size, status, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        -- Dedup on the STABLE racing-API race id, not
        -- (track_name, race_date, race_number): the /racecards/standard
        -- feed leaves race_number NULL, and NULLs are DISTINCT in a
        -- unique index, so the old conflict target never fired and the
        -- every-15-min ingest duplicated every race (migration 020 fixed
        -- the historical mess + added idx_races_racing_api_id). Inference
        -- here matches that partial expression index.
        ON CONFLICT ((external_ids->>'racing_api_race_id'))
            WHERE (external_ids->>'racing_api_race_id') IS NOT NULL
        DO UPDATE
            SET status = EXCLUDED.status,
                surface = COALESCE(EXCLUDED.surface, races.surface),
                track_condition = COALESCE(EXCLUDED.track_condition, races.track_condition),
                race_class = COALESCE(EXCLUDED.race_class, races.race_class),
                field_size = COALESCE(EXCLUDED.field_size, races.field_size),
                purse_amount = COALESCE(EXCLUDED.purse_amount, races.purse_amount),
                external_ids = races.external_ids || EXCLUDED.external_ids,
                metadata = races.metadata || EXCLUDED.metadata,
                updated_at = NOW()
        RETURNING id::text
        """,
        (
            league_id,
            Json({"racing_api_race_id": payload.get("race_id")} if payload.get("race_id") else {}),
            off_dt,
            payload.get("course") or payload.get("track"),
            _safe_int(payload.get("race_number") or payload.get("race_no")),
            furlongs_to_meters(payload.get("distance_f")),
            payload.get("surface"),
            payload.get("going") or payload.get("track_condition"),
            payload.get("race_class") or payload.get("pattern") or payload.get("class"),
            "GBP" if (payload.get("course") and _looks_uk(payload.get("course"))) else "USD",
            _parse_prize(payload.get("prize")),
            _safe_int(payload.get("field_size")),
            status,
            Json(
                {
                    "type": payload.get("type"),
                    "race_name": payload.get("race_name"),
                    "distance_text": payload.get("distance"),
                }
            ),
        ),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def upsert_entrant(
    cur,
    *,
    race_id: str,
    horse_id: str,
    jockey_id: Optional[str],
    trainer_id: Optional[str],
    runner: dict,
    finished: bool,
) -> None:
    """Insert OR update one race_entrant. Updates result fields when
    `finished=True` (the results endpoint pass)."""
    cur.execute(
        """
        INSERT INTO race_entrants
            (race_id, horse_id, jockey_id, trainer_id, program_number,
             post_position, weight_carried_lbs, morning_line_odds,
             starting_price, finish_position, scratched, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        -- Dedup on the STABLE (race_id, horse_id) identity, not
        -- (race_id, program_number): program_number is `_safe_int(...)
        -- or 0` below, so a runner the feed omits a number for lands
        -- under 0, and the SAME horse can appear under 0 in one pass
        -- and its real number in another → duplicate entrants (migration
        -- 021 fixed the existing dupes + swapped the unique key). The
        -- program_number SET prefers a real (non-zero) number when a
        -- later pass supplies one.
        ON CONFLICT (race_id, horse_id) DO UPDATE
            SET program_number = COALESCE(
                    NULLIF(EXCLUDED.program_number, 0), race_entrants.program_number
                ),
                jockey_id = COALESCE(EXCLUDED.jockey_id, race_entrants.jockey_id),
                trainer_id = COALESCE(EXCLUDED.trainer_id, race_entrants.trainer_id),
                post_position = COALESCE(EXCLUDED.post_position, race_entrants.post_position),
                weight_carried_lbs = COALESCE(EXCLUDED.weight_carried_lbs, race_entrants.weight_carried_lbs),
                morning_line_odds = COALESCE(EXCLUDED.morning_line_odds, race_entrants.morning_line_odds),
                starting_price = COALESCE(EXCLUDED.starting_price, race_entrants.starting_price),
                finish_position = COALESCE(EXCLUDED.finish_position, race_entrants.finish_position),
                scratched = EXCLUDED.scratched OR race_entrants.scratched,
                metadata = race_entrants.metadata || EXCLUDED.metadata,
                updated_at = NOW()
        """,
        (
            race_id,
            horse_id,
            jockey_id,
            trainer_id,
            _safe_int(runner.get("number") or runner.get("program_number")) or 0,
            _safe_int(runner.get("draw") or runner.get("post_position")),
            _safe_float(runner.get("weight_lbs") or runner.get("weight")),
            _safe_float(_strip_odds_prefix(runner.get("odds"))) if not finished else None,
            _safe_float(runner.get("sp_dec") or runner.get("starting_price")) if finished else None,
            _safe_int(runner.get("position")) if finished else None,
            (runner.get("status") or "").lower() in ("ns", "wd", "scratched"),
            Json(
                {
                    "form": runner.get("form"),
                    "age": runner.get("age"),
                    "sex": runner.get("sex"),
                    "or_rating": runner.get("or") or runner.get("official_rating"),
                    # Full bookmaker odds array from /racecards/standard
                    # (or /pro) — kept so generate_recommendations can do
                    # best-of-N pricing for value-bet selection. Lossy if
                    # we only kept the first decimal in morning_line_odds
                    # because that defeats the whole point of having a
                    # market-consensus baseline — the prediction would
                    # equal the only bookmaker's implied prob and EV
                    # against it is zero by construction. Format mirrors
                    # the API response: a list of {bookmaker, fractional,
                    # decimal, ew_places, ew_denom, updated} dicts.
                    "bookmaker_odds": _normalize_bookmaker_odds(runner.get("odds")),
                }
            ),
        ),
    )


# ── Misc parsers ──────────────────────────────────────────────────


def _safe_int(v) -> Optional[int]:
    if v in (None, ""):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _safe_float(v) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_UK_COURSES = frozenset(
    {
        "ascot",
        "aintree",
        "ayr",
        "bangor",
        "bath",
        "beverley",
        "brighton",
        "carlisle",
        "cartmel",
        "catterick",
        "chelmsford",
        "cheltenham",
        "chester",
        "doncaster",
        "epsom",
        "exeter",
        "fakenham",
        "ffos las",
        "fontwell",
        "goodwood",
        "hamilton",
        "haydock",
        "hereford",
        "hexham",
        "huntingdon",
        "kelso",
        "kempton",
        "leicester",
        "lingfield",
        "ludlow",
        "market rasen",
        "musselburgh",
        "newbury",
        "newcastle",
        "newmarket",
        "newton abbot",
        "nottingham",
        "perth",
        "plumpton",
        "pontefract",
        "redcar",
        "ripon",
        "salisbury",
        "sandown",
        "sedgefield",
        "southwell",
        "stratford",
        "taunton",
        "thirsk",
        "towcester",
        "uttoxeter",
        "warwick",
        "wetherby",
        "wincanton",
        "windsor",
        "wolverhampton",
        "worcester",
        "yarmouth",
        "york",
        "curragh",
        "leopardstown",
        "fairyhouse",
        "navan",
        "punchestown",
        "galway",
        "gowran park",
        "naas",
        "tipperary",
        "limerick",
        "cork",
        "ballinrobe",
        "bellewstown",
        "clonmel",
        "down royal",
        "downpatrick",
        "dundalk",
        "kilbeggan",
        "killarney",
        "laytown",
        "listowel",
        "roscommon",
        "sligo",
        "thurles",
        "tramore",
        "wexford",
    }
)


def _looks_uk(course: str) -> bool:
    """Heuristic: is this a UK or Irish course? Used for purse currency
    (£ vs $). Free-text source so we conservatively guess GBP for any
    course in the UK + Ireland list."""
    return (course or "").strip().lower() in _UK_COURSES


_PRIZE_RE = re.compile(r"([\d,]+(?:\.\d+)?)")


def _parse_prize(raw: Optional[str]) -> Optional[float]:
    """'£1,000,000' / '$500K' / '€250,000' → 1000000.0 etc.
    Currency symbol is dropped; the caller already stored the
    currency code separately. K suffix expanded to thousands."""
    if not raw:
        return None
    s = str(raw).strip().lower()
    multiplier = 1.0
    if s.endswith("k"):
        multiplier = 1_000.0
        s = s[:-1]
    elif s.endswith("m"):
        multiplier = 1_000_000.0
        s = s[:-1]
    m = _PRIZE_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "")) * multiplier
    except ValueError:
        return None


def _strip_odds_prefix(odds_payload) -> Optional[float]:
    """The Racing API's racecards endpoint puts odds inside an array
    of {bookmaker, decimal} dicts. Take the first decimal as the
    morning-line proxy."""
    if not odds_payload:
        return None
    if isinstance(odds_payload, list) and odds_payload:
        first = odds_payload[0]
        if isinstance(first, dict):
            return first.get("decimal") or first.get("price")
    if isinstance(odds_payload, (int, float, str)):
        return odds_payload
    return None


def _normalize_bookmaker_odds(odds_payload) -> list[dict]:
    """Normalise the racecards `odds` array into a clean list of
    {bookmaker, decimal} dicts. Drops entries without a usable
    decimal price; keeps fractional + ew_places when present for
    audit. Returns [] for empty / malformed input so downstream code
    can iterate unconditionally.

    The list is what generate_recommendations needs to do best-of-N
    pricing — for a given horse, the longest decimal across
    bookmakers becomes the bettor's best available price and the
    basis for EV vs the devigged consensus."""
    if not isinstance(odds_payload, list):
        return []
    out: list[dict] = []
    for item in odds_payload:
        if not isinstance(item, dict):
            continue
        decimal = _safe_float(item.get("decimal") or item.get("price"))
        if decimal is None or decimal <= 1.0:
            # decimal odds <= 1.0 are nonsensical (implied prob >=
            # 100%); typically a stub from a bookmaker that hasn't
            # priced yet. Drop them.
            continue
        out.append(
            {
                "bookmaker": item.get("bookmaker"),
                "decimal": decimal,
                "fractional": item.get("fractional"),
                "ew_places": item.get("ew_places"),
                "ew_denom": item.get("ew_denom"),
                "updated": item.get("updated"),
            }
        )
    return out


# ── Orchestration ──────────────────────────────────────────────────


def iter_dates(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=1)


def ingest_day(cur, day: date, auth: HTTPBasicAuth, kind: str) -> dict:
    """Pull racecards (kind='upcoming') or results (kind='results')
    for one day, upserting races + entrants. Returns per-call counts."""
    fetcher = fetch_racecards if kind == "upcoming" else fetch_results
    payloads = fetcher(day, auth)
    counts = {"races": 0, "entrants": 0, "skipped_races": 0, "skipped_entrants": 0}
    finished = kind == "results"
    for race in payloads:
        course = race.get("course") or race.get("track")
        if not course:
            counts["skipped_races"] += 1
            continue
        country = "United Kingdom" if _looks_uk(course) else "Ireland"
        league_id = ensure_league(cur, course, country)
        if not league_id:
            counts["skipped_races"] += 1
            continue
        race_id = upsert_race(cur, league_id=league_id, payload=race, status="finished" if finished else "scheduled")
        if not race_id:
            counts["skipped_races"] += 1
            continue
        counts["races"] += 1
        for runner in race.get("runners", []) or []:
            horse_id = ensure_horse(cur, runner.get("horse"))
            if not horse_id:
                counts["skipped_entrants"] += 1
                continue
            jockey_id = ensure_jockey(cur, runner.get("jockey"))
            trainer_id = ensure_trainer(cur, runner.get("trainer"))
            upsert_entrant(
                cur,
                race_id=race_id,
                horse_id=horse_id,
                jockey_id=jockey_id,
                trainer_id=trainer_id,
                runner=runner,
                finished=finished,
            )
            counts["entrants"] += 1
    return counts


def run(database_url: str, kind: str, start: date, end: date) -> dict:
    auth = _auth()
    if not auth:
        logger.error("THE_RACING_API_USER / THE_RACING_API_PASS not set")
        return {"error": "missing_credentials"}

    totals = {"days_processed": 0, "races": 0, "entrants": 0, "skipped_races": 0, "skipped_entrants": 0}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for day in iter_dates(start, end):
                counts = ingest_day(cur, day, auth, kind)
                totals["days_processed"] += 1
                for k in ("races", "entrants", "skipped_races", "skipped_entrants"):
                    totals[k] += counts.get(k, 0)
                conn.commit()
                time.sleep(REQUEST_DELAY_SEC)
                # Basic + Standard /racecards/{tier} return today's data
                # regardless of the day param. After the first call
                # the rest of the date range is wasted work — break
                # out and log the cap. Pro is date-rangeable so it
                # keeps iterating.
                if kind == "upcoming" and _CACHED_RACECARDS_PATH in _ONLY_TODAY_ENDPOINTS:
                    logger.info(
                        "%s returns today only — stopping after pass 1 "
                        "(date range %s..%s ignored)",
                        _CACHED_RACECARDS_PATH,
                        start,
                        end,
                    )
                    break
                if totals["days_processed"] % 30 == 0:
                    logger.info(
                        "Progress: %d days, %d races, %d entrants",
                        totals["days_processed"],
                        totals["races"],
                        totals["entrants"],
                    )
    logger.info("Done. %s", totals)
    return totals


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--upcoming",
        type=int,
        metavar="DAYS",
        help="Fetch racecards (forecast) for the next N days starting today.",
    )
    group.add_argument(
        "--results",
        action="store_true",
        help="Fetch historical results between --start and --end.",
    )
    p.add_argument("--start", help="Inclusive start date (YYYY-MM-DD), used with --results.")
    p.add_argument("--end", help="Inclusive end date (YYYY-MM-DD), used with --results.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2

    if args.upcoming is not None:
        today = date.today()
        end = today + timedelta(days=args.upcoming - 1)
        run(args.database_url, "upcoming", today, end)
    else:
        if not (args.start and args.end):
            logger.error("--start and --end are required with --results")
            return 2
        try:
            start = date.fromisoformat(args.start)
            end = date.fromisoformat(args.end)
        except ValueError as e:
            logger.error("Bad date: %s", e)
            return 2
        if end < start:
            logger.error("--end must be >= --start")
            return 2
        run(args.database_url, "results", start, end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
