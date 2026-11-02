"""Ingest UK + Irish horse racing data from theracingapi.com.

Plan-aware endpoint dispatch:
  Basic ($30/mo)   → /racecards/basic (TODAY only, no date param,
                     no odds). /results is Standard+.
  Standard         → /racecards/standard (TODAY only, no date param,
                     bookmaker odds array per runner) +
                     /results (date-rangeable via start_date/end_date,
                     with sp/sp_dec and finish position; PAGINATED,
                     50 per page — see fetch_results) +
                     /results/{race_id} (one race's result).
  Pro              → /racecards/pro (date-rangeable, deepest payload)
                     + /results + per-horse stable/quotes data.

The script tries pro → standard → basic in order and falls back on
401/403/422 ("Plan required" or rejected params). On Basic, the
date range to --upcoming N is silently capped to today (one call);
ditto on Standard since /racecards/standard also returns today only.
--results is the multi-day workhorse (Standard+).

--drain is what the auspex_pipeline DAG runs while horse racing is
dormant (services/data-ingestion/dags/sport_switches.py): one
/results/{race_id} call per race that still carries a pending rec, has
been off DRAIN_GRACE_MINUTES and is at most DRAIN_DAYS old. Nothing
pending in that window → no API call at all. A result that does not
account for an entrant a pending rec rides on (a name the feeds spell
differently) is not ingested and fails the run (drain_field_problem).
Recs on races older than the window are "stranded"; the drain logs them
and never touches them again (scripts/close_out_stranded_race_recs.py,
operator-run, settles or voids them).

A result also marks the entrants its `non_runners` string names as
scratched, so grade_completed_races voids a rec on a withdrawn horse
instead of settling it lost (before 2026-11-01 nothing ever set
race_entrants.scratched: 0 of 36,800 entrants in 60 days).

Auth: HTTP Basic with THE_RACING_API_USER + THE_RACING_API_PASS.

After the subscription ends (or with the credentials unset), every mode
fails loudly where there is work it cannot do and stays quiet where there
is none: --drain makes no call and exits 0 while no race in its window
carries a pending rec, and exits 1 when one does (missing credentials, or a
401/402/403 from /results/{race_id}, is an error, never "no result");
--upcoming and --results exit 1 on missing credentials, on racecards that no
tier serves, and on a /results page refused with 401/402/403 (other failed
/results pages are logged at ERROR and the next run re-reads the day).

Lazy entity creation: horses / jockeys / trainers get one DB row
per unique normalized_name. Same horse appearing in many races
resolves to one horses.id, letting rolling-form features be cheap
JOINs at training time.

Usage:
    # Daily upcoming-racecards ingest (DAG runs this while horse racing is enabled)
    python scripts/load_racing_api.py --upcoming 7

    # Historical backfill (Standard+ plan only)
    python scripts/load_racing_api.py --results --start 2024-01-01 --end 2024-12-31

    # Single-day diagnostic
    python scripts/load_racing_api.py --results --start 2024-09-08 --end 2024-09-08 --verbose

    # Dormant mode: results for the races that still carry a pending rec
    python scripts/load_racing_api.py --drain
    python scripts/load_racing_api.py --drain --dry-run   # fetch + report, write nothing
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
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

# /results is paginated: `limit` (50 — the default, and what one page
# serves), `skip`, and the day's `total` in the body. Until 2026-11-01
# fetch_results read the first page only, so on a busy day every race
# past the 50th was never ingested: prod 2026-11-01 had seven recs pending
# for five months on page-2 races (June 4: 56 results, June 6: 69; Oct 26
# had 71). MAX_RESULTS_PAGES only bounds a misbehaving `total`.
RESULTS_PAGE_LIMIT = 50
MAX_RESULTS_PAGES = 20

# Drain mode (horse racing dormant). A race is drained while it carries a
# pending/placed rec, has been off DRAIN_GRACE_MINUTES (results post
# minutes after the off) and is at most DRAIN_DAYS old — the late-posted
# Deauville result of 2026-08-23 missed a two-day window. Older races are
# stranded: the drain never asks about them again, and
# close_out_stranded_race_recs.py reads this same constant, so the two never
# act on the same race.
DRAIN_DAYS = 3
DRAIN_GRACE_MINUTES = 10

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


class RacingApiError(RuntimeError):
    """A Racing API answer that is neither a result nor 'no result for this race', or one that refuses the
    credentials or the plan."""


# The statuses a plan or credential refusal comes back with (a lapsed subscription included).
_REFUSED_STATUSES = frozenset({401, 402, 403})


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

    answers = []
    for path in ("/racecards/pro", "/racecards/standard", "/racecards/basic"):
        send_date = path in _PATHS_THAT_ACCEPT_DATE
        status, cards = _try_racecards_endpoint(path, day, auth, send_date)
        if status == 200:
            _CACHED_RACECARDS_PATH = path
            logger.info("Racing API tier detected: %s", path)
            return cards
        answers.append(f"{path} -> {status or 'no answer'}")
        if status not in (401, 402, 403, 422):
            # Network error / unexpected — log and keep probing.
            logger.warning("Racing API %s returned %s", path, status)
    # Not an empty card (that is a 200 with no racecards): no tier answered at all. Before 2026-11-02 this returned
    # [] and the task stayed green with no racecards — what a lapsed subscription would have looked like.
    raise RacingApiError(
        f"Racing API: no racecards tier answered 200 for {day} ({'; '.join(answers)}). Check THE_RACING_API_USER / "
        "THE_RACING_API_PASS and that the subscription is still active."
    )


def fetch_results(day: date, auth: HTTPBasicAuth) -> list[dict]:
    """Pull every finished-race result for one day, page by page
    (RESULTS_PAGE_LIMIT per page until the body's `total` is read).
    Requires Standard+ plan — Basic returns 403 'Standard Plan required'.
    A failed page is logged at ERROR with how far the read got and ends
    it with what arrived; the next run re-reads the day."""
    url = f"{BASE_URL}/results"
    out: list[dict] = []
    total = None
    for page in range(1, MAX_RESULTS_PAGES + 1):
        params = {
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
            "limit": RESULTS_PAGE_LIMIT,
            "skip": len(out),
        }
        try:
            r = requests.get(url, params=params, auth=auth, timeout=DEFAULT_TIMEOUT_SEC)
        except requests.RequestException as e:
            logger.error("Racing API results %s page %d failed after %d result(s): %s", day, page, len(out), e)
            return out
        if r.status_code in _REFUSED_STATUSES:
            # The credentials or the plan were refused: no later page or day can succeed, and returning what
            # arrived would leave the task green.
            raise RacingApiError(
                f"Racing API results {day} page {page} -> HTTP {r.status_code}: {(r.text or '')[:200]} — the "
                "credentials or the plan were refused (check THE_RACING_API_USER / THE_RACING_API_PASS and that "
                "the subscription is still active)"
            )
        if r.status_code >= 400:
            body_snippet = r.text[:200] if r.text else ""
            logger.error(
                "Racing API results %s page %d → %d after %d result(s): %s",
                day,
                page,
                r.status_code,
                len(out),
                body_snippet,
            )
            return out
        body = r.json()
        rows = body.get("results", []) or []
        out.extend(rows)
        total = _safe_int(body.get("total"))
        if not rows or total is None or len(out) >= total:
            return out
        time.sleep(REQUEST_DELAY_SEC)
    logger.error(
        "Racing API results %s: stopped after %d pages with %d of %s result(s)", day, MAX_RESULTS_PAGES, len(out), total
    )
    return out


def _is_no_result(r, api_race_id: str) -> bool:
    """Is this the API's per-race 404 — {"detail": "Result not found - race
    <id>"}, the body of all 25 prod probes on 2026-11-01 — and not its
    generic route 404 {"detail": "Not Found"} (a wrong path, base URL or
    version), which would otherwise read every race as 'no result yet'
    and, three days on, have the close-out void them all as abandoned."""
    if r.status_code != 404:
        return False
    try:
        body = r.json()
    except ValueError:
        return False
    detail = body.get("detail") if isinstance(body, dict) else None
    return isinstance(detail, str) and detail.startswith("Result not found") and api_race_id in detail


def fetch_race_result(api_race_id: str, auth: HTTPBasicAuth) -> Optional[dict]:
    """One race's result by its Racing API id (/results/{race_id}).

    Returns the payload, or None on the API's 404 "Result not found - race
    <id>" (_is_no_result): the race has not run, its result is not posted
    yet, or it never will be — an abandoned meeting is dropped from the
    feed without a status (prod 2026-11-01: Newbury 24 Oct, Aintree 25 Oct,
    ...). Anything else — another status, another 404 body, a non-JSON
    body, a payload for a different race — raises RacingApiError, so an
    auth, plan, routing or network failure is never read as 'no result'."""
    url = f"{BASE_URL}/results/{api_race_id}"
    try:
        r = requests.get(url, auth=auth, timeout=DEFAULT_TIMEOUT_SEC)
    except requests.RequestException as e:
        raise RacingApiError(f"/results/{api_race_id}: {e}") from e
    snippet = (r.text or "")[:200]
    if _is_no_result(r, api_race_id):
        return None
    if r.status_code != 200:
        raise RacingApiError(f"/results/{api_race_id} -> HTTP {r.status_code}: {snippet}")
    try:
        payload = r.json()
    except ValueError as e:
        raise RacingApiError(f"/results/{api_race_id}: body is not JSON: {snippet}") from e
    if not isinstance(payload, dict) or payload.get("race_id") != api_race_id:
        got = payload.get("race_id") if isinstance(payload, dict) else type(payload).__name__
        raise RacingApiError(f"/results/{api_race_id}: the payload is for race {got!r}")
    return payload


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
            -- Status is FORWARD-ONLY: never downgrade a terminal state
            -- back to 'scheduled'. The upcoming-cards ingest runs every
            -- 15 min and marks today's races 'scheduled'; without this
            -- guard it would revert a race the results ingest already
            -- set to 'finished' (now that both passes upsert the SAME
            -- row post-dedup), so finished races flapped back to
            -- scheduled — invisible in Recent results + un-settling
            -- their graded recs. Keep terminal states sticky.
            SET status = CASE
                    WHEN races.status IN ('finished', 'cancelled', 'abandoned')
                        THEN races.status
                    ELSE EXCLUDED.status
                END,
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


def mark_non_runners(cur, race_id: str, names: set[str]) -> int:
    """Set scratched on this race's entrants whose horse a result lists
    as a non-runner (names normalized, see parse_non_runners). Sticky,
    like upsert_entrant's own `scratched OR`. Returns the rows changed."""
    if not names:
        return 0
    cur.execute(
        """
        UPDATE race_entrants e
        SET scratched = TRUE,
            updated_at = NOW()
        FROM horses h
        WHERE h.id = e.horse_id
          AND e.race_id = %s
          AND h.normalized_name = ANY(%s)
          AND NOT e.scratched
        """,
        (race_id, sorted(names)),
    )
    return cur.rowcount


# ── Misc parsers ──────────────────────────────────────────────────


_TRAILING_PARENS_RE = re.compile(r"\s*\([^()]*\)\s*$")


def parse_non_runners(text: Optional[str]) -> set[str]:
    """A result's `non_runners` string -> normalized horse names.

    The feed lists them as ONE comma-separated string, each name followed
    by an optional parenthesised reason that can itself hold commas:
    'Hall Lane (self certificate, lame)' (prod rac_11958661). Split on
    commas outside parentheses, drop every trailing (...) group (reason,
    country code), normalize like every other horse name."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text or "":
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    names: set[str] = set()
    for part in parts:
        name = part.strip()
        while True:
            stripped = _TRAILING_PARENS_RE.sub("", name)
            if stripped == name:
                break
            name = stripped
        norm = normalize_name(name)
        if norm:
            names.add(norm)
    return names


def result_names(payload: dict) -> tuple[dict[str, dict], set[str]]:
    """One result's runners keyed by normalized horse name, and the
    normalized names of its non-runners."""
    runners: dict[str, dict] = {}
    for ru in payload.get("runners") or []:
        name = normalize_name(ru.get("horse") or "")
        if name:
            runners[name] = ru
    return runners, parse_non_runners(payload.get("non_runners"))


def unaccounted_entrants(field: list[dict], payload: dict) -> list[str]:
    """Our entrants ({horse: normalized name, scratched}) that are not
    scratched and that the result names neither as a runner nor as a
    non-runner — usually a name the two feeds spell differently: 'Jo
    Arengeot Bey Aa' on the card, 'Jo Arengeot Bey (FR)' in the result
    (prod, Ayr 31 Oct 13:18). Ingesting the result leaves such an entrant
    running without a position — a rec on it is graded lost — next to a
    new row for the result's spelling, which also inflates the field the
    each-way place terms count."""
    runners, non_runners = result_names(payload)
    named = runners.keys() | non_runners
    return sorted({f["horse"] for f in field if not f["scratched"] and f["horse"] not in named})


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


_COUNT_KEYS = ("races", "entrants", "skipped_races", "skipped_entrants", "non_runners")


def league_key(race: dict) -> Optional[tuple[str, str]]:
    """(name, country) of the leagues row a payload's track upserts into, or
    None when the payload names no track."""
    course = race.get("course") or race.get("track")
    if not course:
        return None
    return course, "United Kingdom" if _looks_uk(course) else "Ireland"


def ingest_race(cur, race: dict, finished: bool) -> tuple[Optional[str], dict]:
    """Upsert one racecard (finished=False) or result (finished=True)
    payload: its track's league, the race, every runner. A result also
    marks the entrants its `non_runners` names as scratched. Returns
    (race_id, or None when the payload was skipped; per-race counts)."""
    counts = dict.fromkeys(_COUNT_KEYS, 0)
    league = league_key(race)
    if not league:
        counts["skipped_races"] += 1
        return None, counts
    league_id = ensure_league(cur, *league)
    if not league_id:
        counts["skipped_races"] += 1
        return None, counts
    race_id = upsert_race(cur, league_id=league_id, payload=race, status="finished" if finished else "scheduled")
    if not race_id:
        counts["skipped_races"] += 1
        return None, counts
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
    if finished:
        counts["non_runners"] += mark_non_runners(cur, race_id, parse_non_runners(race.get("non_runners")))
    return race_id, counts


def ingest_day(cur, day: date, auth: HTTPBasicAuth, kind: str) -> dict:
    """Pull racecards (kind='upcoming') or results (kind='results')
    for one day, upserting races + entrants. Returns per-call counts."""
    fetcher = fetch_racecards if kind == "upcoming" else fetch_results
    payloads = fetcher(day, auth)
    counts = dict.fromkeys(_COUNT_KEYS, 0)
    for race in payloads:
        _, race_counts = ingest_race(cur, race, finished=kind == "results")
        for k in _COUNT_KEYS:
            counts[k] += race_counts[k]
    return counts


def run(database_url: str, kind: str, start: date, end: date) -> dict:
    auth = _auth()
    if not auth:
        logger.error("THE_RACING_API_USER / THE_RACING_API_PASS not set")
        return {"error": "missing_credentials"}

    totals = {"days_processed": 0, **dict.fromkeys(_COUNT_KEYS, 0)}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for day in iter_dates(start, end):
                counts = ingest_day(cur, day, auth, kind)
                totals["days_processed"] += 1
                for k in _COUNT_KEYS:
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
                        "%s returns today only — stopping after pass 1 " "(date range %s..%s ignored)",
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


# Pending/placed recs on races that are not finished/cancelled/abandoned:
# the window the drain asks about, and the backlog either side of it.
_OPEN_REC_RACE = """
    FROM race_recommendations rec
    JOIN races r ON r.id = rec.race_id
    WHERE rec.status IN ('pending', 'placed')
      AND r.status NOT IN ('finished', 'cancelled', 'abandoned')
"""
DRAIN_RACES_SQL = f"""
    SELECT r.id::text AS race_id,
           r.external_ids->>'racing_api_race_id' AS api_race_id,
           r.race_date,
           r.track_name,
           count(*) AS pending_recs
    {_OPEN_REC_RACE}
      AND r.race_date <= NOW() - make_interval(mins => %(grace_minutes)s)
      AND r.race_date >= NOW() - make_interval(days => %(drain_days)s)
    GROUP BY r.id
    ORDER BY r.race_date, r.id
"""
DRAIN_BACKLOG_SQL = f"""
    SELECT count(*) FILTER (WHERE r.race_date > NOW() - make_interval(mins => %(grace_minutes)s)) AS upcoming_recs,
           count(*) FILTER (WHERE r.race_date < NOW() - make_interval(days => %(drain_days)s)) AS stranded_recs
    {_OPEN_REC_RACE}
"""
# One race's field as the drain checks it against a result: each entrant's
# normalized horse name, scratched, and its pending recs (place ones apart).
DRAIN_FIELD_SQL = """
    SELECT h.normalized_name AS horse,
           e.scratched,
           count(rec.id) AS pending_recs,
           count(rec.id) FILTER (WHERE rec.bet_type = 'place') AS pending_place_recs
    FROM race_entrants e
    JOIN horses h ON h.id = e.horse_id
    LEFT JOIN race_recommendations rec
           ON rec.race_id = e.race_id AND rec.entrant_id = e.id AND rec.status IN ('pending', 'placed')
    WHERE e.race_id = %s
    GROUP BY e.id, h.normalized_name, e.scratched
"""


def drain_field_problem(field: list[dict], payload: dict) -> Optional[str]:
    """Why ingesting this result could settle a pending rec wrong, or None.

    It could when one of our entrants is unaccounted for
    (unaccounted_entrants) and either carries a pending rec itself (graded
    lost with no position, whatever it did) or shares the race with a
    pending place rec (the grader's place terms count it as a runner, plus
    the new row for the result's spelling). An unaccounted entrant without
    either settles every rec right — the winner is read from positions —
    and is only logged. Prod over the 60 days to 2026-11-01: 12 finished
    races had a card entrant left without a position next to a result-only
    runner, 1 of them carried recs, none on that entrant or a place rec."""
    unaccounted = unaccounted_entrants(field, payload)
    if not unaccounted:
        return None
    runners, _ = result_names(payload)
    ours = {f["horse"] for f in field}
    extra = ", ".join(sorted(set(runners) - ours)) or "none"
    with_recs = sorted(f["horse"] for f in field if f["horse"] in unaccounted and f["pending_recs"])
    if with_recs:
        return (
            f"our entrant(s) {', '.join(with_recs)} carry a pending rec but the result names them neither as a "
            f"runner nor as a non-runner (result runners not on our card: {extra})"
        )
    if any(f["pending_place_recs"] for f in field):
        return (
            f"our entrant(s) {', '.join(unaccounted)} are in neither the result's runners nor its non-runners, and "
            f"a pending place rec's terms count the field (result runners not on our card: {extra})"
        )
    return None


def run_drain(database_url: str, dry_run: bool = False) -> dict:
    """Dormant-mode results: for each race DRAIN_RACES_SQL lists, ask
    /results/{race_id} and, when the result is there, ingest it exactly as
    the results pass does (the grader then settles its recs). One commit
    per race. No race in the window → no credentials needed, no API call.
    A result that does not account for an entrant a pending rec rides on
    (drain_field_problem) is not ingested. Returns counters; `errors` > 0
    means a race could not be drained (missing credentials or race id, an
    API failure, such a result, an ingest that raised or did not land on
    the race row) and main() exits 1 for it."""
    totals = {
        "races": 0,
        "resolved": 0,
        "not_posted": 0,
        "errors": 0,
        "api_calls": 0,
        "upcoming_recs": 0,
        "stranded_recs": 0,
    }
    params = {"grace_minutes": DRAIN_GRACE_MINUTES, "drain_days": DRAIN_DAYS}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(DRAIN_RACES_SQL, params)
            races = list(cur.fetchall())
            cur.execute(DRAIN_BACKLOG_SQL, params)
            backlog = cur.fetchone() or {}
            conn.rollback()  # close the read snapshot; each race commits on its own
            totals["races"] = len(races)
            totals["upcoming_recs"] = int(backlog.get("upcoming_recs") or 0)
            totals["stranded_recs"] = int(backlog.get("stranded_recs") or 0)
            auth = _auth() if races else None
            if races and not auth:
                logger.error(
                    "THE_RACING_API_USER / THE_RACING_API_PASS not set — %d race(s) with pending recs cannot be "
                    "drained",
                    len(races),
                )
                totals["errors"] = len(races)
                races = []
            for race in races:
                off = race["race_date"].astimezone(timezone.utc)
                label = f"{race['track_name']} {off:%Y-%m-%d %H:%M}Z ({race['race_id']})"
                if not race["api_race_id"]:
                    logger.error("drain: %s has no racing_api_race_id — its result cannot be asked for", label)
                    totals["errors"] += 1
                    continue
                if totals["api_calls"]:
                    time.sleep(REQUEST_DELAY_SEC)
                totals["api_calls"] += 1
                try:
                    payload = fetch_race_result(race["api_race_id"], auth)
                except RacingApiError as e:
                    logger.error("drain: %s: %s", label, e)
                    totals["errors"] += 1
                    continue
                if payload is None:
                    totals["not_posted"] += 1
                    logger.info("drain: %s — no result yet (%d pending rec(s))", label, race["pending_recs"])
                    continue
                cur.execute(DRAIN_FIELD_SQL, (race["race_id"],))
                field = [dict(r) for r in cur.fetchall()]
                conn.rollback()
                problem = drain_field_problem(field, payload)
                if problem:
                    logger.error(
                        "drain: %s — %s. The result is NOT ingested: the drain asks again every tick, and once the "
                        "race is stranded close_out_stranded_race_recs refuses it the same way (unmatched) — "
                        "review the race's entrants.",
                        label,
                        problem,
                    )
                    totals["errors"] += 1
                    continue
                unaccounted = unaccounted_entrants(field, payload)
                if unaccounted:
                    logger.warning(
                        "drain: %s — our entrant(s) %s are not in the result; no pending rec rides on them or on "
                        "the place terms, so it is ingested",
                        label,
                        ", ".join(unaccounted),
                    )
                if dry_run:
                    totals["resolved"] += 1
                    logger.info("drain: %s — result posted; DRY RUN, not ingested", label)
                    continue
                try:
                    race_id, counts = ingest_race(cur, payload, finished=True)
                except Exception:
                    # One bad payload (a constraint, an overflow) must not block the races behind it:
                    # roll it back, count it — the run still exits 1 — and go on.
                    conn.rollback()
                    logger.error("drain: %s — ingesting its result failed; rolled back", label, exc_info=True)
                    totals["errors"] += 1
                    continue
                if race_id != race["race_id"]:
                    conn.rollback()
                    logger.error("drain: %s — the result upserted race %s, not this row; rolled back", label, race_id)
                    totals["errors"] += 1
                    continue
                conn.commit()
                totals["resolved"] += 1
                logger.info(
                    "drain: %s — result ingested (%d runner(s), %d non-runner(s) scratched)",
                    label,
                    counts["entrants"],
                    counts["non_runners"],
                )
    logger.info(
        "Drain%s: %d race(s) with pending recs in the %d-day window — %d resolved, %d not posted yet, %d error(s); "
        "%d API call(s). %d pending rec(s) on races not yet off.",
        " (dry run)" if dry_run else "",
        totals["races"],
        DRAIN_DAYS,
        totals["resolved"],
        totals["not_posted"],
        totals["errors"],
        totals["api_calls"],
        totals["upcoming_recs"],
    )
    if totals["stranded_recs"]:
        logger.warning(
            "%d pending race rec(s) sit on races more than %d days old with no result: the drain no longer asks "
            "about them. scripts/ops/close_out_stranded_race_recs.sh settles or voids them for races before its "
            "reviewed RACED_BEFORE; it names any on later races, which need a new reviewed pin there.",
            totals["stranded_recs"],
            DRAIN_DAYS,
        )
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
    group.add_argument(
        "--drain",
        action="store_true",
        help=f"Dormant mode: fetch the result of each race with a pending rec that is off and at most "
        f"{DRAIN_DAYS} days old (one call per race; none when nothing is pending).",
    )
    p.add_argument("--start", help="Inclusive start date (YYYY-MM-DD), used with --results.")
    p.add_argument("--end", help="Inclusive end date (YYYY-MM-DD), used with --results.")
    p.add_argument("--dry-run", action="store_true", help="With --drain: fetch and report, write nothing.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.dry_run and not args.drain:
        p.error("--dry-run applies to --drain only")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2

    if args.drain:
        totals = run_drain(args.database_url, dry_run=args.dry_run)
        return 1 if totals["errors"] else 0
    if args.upcoming is not None:
        kind = "upcoming"
        start = date.today()
        end = start + timedelta(days=args.upcoming - 1)
    else:
        kind = "results"
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
    try:
        totals = run(args.database_url, kind, start, end)
    except RacingApiError as e:
        logger.error("%s — exit 1 (the days before this one are committed)", e)
        return 1
    # Missing credentials: run() wrote nothing. Before 2026-11-02 this still exited 0.
    return 1 if totals.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
