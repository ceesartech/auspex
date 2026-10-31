"""Merge orphan fixture rows into their finished twin, then delete them.

THE BUG THIS CLEANS UP
----------------------
Before 2026-09-04 fixture identity was keyed on the exact kickoff time, so
every ESPN time change forked a tennis/MMA/soccer fixture into another
`matches` row; until 2026-10-30 a re-drawn 1v1 draw slot did the same (see
fetch_upcoming._release_redrawn_slot). Prod 2026-10-30: 19,486 stale
'scheduled' tennis rows holding 89k odds rows, 147k CLV snapshots and 20.6k
predictions, while only 864 of the 3,214 real finished matches they belong
to carried any odds of their own. Soccer 2026-10-31: 449 stale rows, 231 of
them with a finished twin. `repair_stranded_recs.py` settles and re-points
the RECOMMENDATIONS; this tool does the rest.

WHAT THIS DOES
--------------
Candidates: rows of --sport with status 'scheduled' (tennis/MMA also
'cancelled', set by the re-draw fix) whose kickoff is older than
--stale-after-hours (default 96 h, past the results path's 3-day look-back
so a row the pipeline could still heal is never touched).

Each candidate's twin is chosen by repair_stranded_recs.find_twins +
select_twin — the SAME rule the rec repair used: identical orientation,
status finished, both scores present, nearest kickoff within
--twin-window-hours, ambiguous (different scores / tie) -> skipped.

  twin found  -> MOVE onto the twin, strictly pre-game only. The cutoff is
                 LEAST(orphan kickoff, twin kickoff): either row can carry
                 the wrong time (ESPN moved the fixture; football-data's
                 summer kickoffs sit 1 h late), so neither alone proves a
                 price was taken before the match started.
                 odds            one row per current-price key (book,
                                 market, selection, line, opening, live):
                                 the freshest price before the cutoff, and
                                 only for keys the twin does not already
                                 hold (its own rows win);
                 odds_snapshots  every capture before the cutoff not
                                 already on the twin (same book, market,
                                 selection, line, captured_at);
                 predictions     per (model, version, type) the latest one
                                 written before the cutoff, unless the twin
                                 already has that key or a rec references
                                 that prediction (a rec's prediction stays
                                 where the rec put it).
               then DELETE the orphan (cascade drops what was not moved:
               post-kickoff prices, superseded duplicates, features_cache).
  no twin     -> tennis/MMA: DELETE (a pairing that never played: a re-drawn
               slot, a walkover); its prices have no result to train on.
               soccer: LEFT ALONE and reported by class (see below).
  ambiguous   -> left alone, counted.

An orphan that still holds ANY betting_recommendations or betting_history
row is never deleted (run repair_stranded_recs.py first; it re-points
settled recs onto the twin). Nor is one holding a PREDICTION a rec
references: re-pointing moves the rec's match_id but not its prediction_id.
Until migration 025 betting_recommendations.prediction_id was ON DELETE
CASCADE, so deleting that orphan deleted the rec with it — the 2026-10-31
tennis run lost 45 settled recs exactly that way (restored by
restore_cascaded_recs.py). Since 025/026 (both rec FKs NO ACTION) such a
DELETE fails instead of cascading; the guard keeps the batch from aborting.
  tennis/MMA: a pinned orphan is untouched, counted `kept_has_recs`.
  soccer:     a pinned orphan WITH a twin is merged like any other, then
              marked status 'cancelled' with metadata.orphan_of = <twin id>
              instead of deleted, so it leaves the stale set while its recs
              stay exactly as they are. Refused while a PENDING rec sits on
              it: repair_stranded_recs only reads 'scheduled' rows, so that
              rec could never settle — run it first (--apply refuses).

WHY SOCCER NEVER DELETES A ROW WITHOUT A TWIN
---------------------------------------------
A tennis orphan without a twin is a pairing that never played. A soccer
one usually DID play: the row was never resulted (ESPN has the score — a
results backfill can settle it and its recs), or the real match sits under
another identity (a football-data row with other team ids, a renamed club),
or it was postponed and replayed outside the twin window. Deleting any of
those destroys a real fixture. They are left alone and reported:
  flipped         a finished row with home/away swapped inside the window;
  other_identity  a finished row in the same league within 2 h sharing a
                  team id, or whose two clubs match by name (football-data
                  writes 'Orgryte' for ESPN's 'Örgryte IS'): a club never
                  plays twice in two hours;
  replayed_later  the same pairing finished later, beyond the window and
                  within 120 days (postponed and replayed);
  never_resulted  none of the above: a results backfill can settle it.

PLACEHOLDERS (--delete-placeholders)
------------------------------------
Knockout-bracket slots ESPN publishes before the teams are known ('Round of
32 1 Winner', 'Group A 2nd Place' — fetch_upcoming._is_placeholder_name) and
rows of a team against itself (the slots fuzzy-matched each other into one
team row) are not fixtures. Prod 2026-10-31: 65 such World Cup rows.
fetch_upcoming refuses them at the door now; with --delete-placeholders the
stale ones are deleted (soccer only needs the flag — for tennis/MMA they are
no-twin rows, deleted anyway), never when a rec references them. Such a row
is never merged into a twin.

SAFETY
------
Dry run is the default. Every planned row goes to the write-once audit CSV
(--out-csv, default $BACKUP_LOCAL_DIR or /app/backups) BEFORE any write, dry
run included; with --apply every row the run moves, cancels or deletes is
also written to --archive (gzip JSONL, write-once) before the first write,
so the run is reversible offline. Each orphan is locked and re-checked
against the values the plan observed (status, kickoff, teams, recs) and the
twin against its kickoff before anything moves; every DELETE / UPDATE of a
matches row is guarded on those values too. A row that changed is left as
it is, logged, and the run exits 1. Batches of --batch-size twins commit
separately, keeping locks short while the 15-min pipeline runs.
--expect-plan pins an apply to the counts its dry run printed (the PLAN
line): any difference refuses the run before anything is written.

Usage (inside the api container; dry-run is the default):
    python /app/scripts/repair_fixture_orphans.py --sport tennis
    python /app/scripts/repair_fixture_orphans.py --sport tennis --apply
    python /app/scripts/repair_fixture_orphans.py --sport soccer --delete-placeholders
Soccer: scripts/ops/cleanup_soccer_orphans.sh wraps the whole sequence.
Afterwards: compute_features_<sport>.py --all-finished --force, and
grade_completed_matches.py --days <span> so moved predictions are graded.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repair_stranded_recs as rsr  # noqa: E402
from fetch_upcoming import SPORT_CONFIGS, _is_placeholder_name, _strip_club_tokens, normalize_team  # noqa: E402
from team_identity import strip_accents  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("repair_fixture_orphans")

ALLOWED_SPORTS = ("tennis", "mma", "soccer")
DEFAULT_STALE_AFTER_HOURS = 96
MIN_STALE_AFTER_HOURS = 96  # the results path looks back 3 days; never race it

# 'cancelled' is a tennis/MMA re-drawn slot (fetch_upcoming._release_redrawn_slot).
# A soccer 'cancelled' row was set on purpose — an abandoned match, or a
# pinned orphan this tool already retired — so soccer reads 'scheduled' only.
CANDIDATE_STATUSES = {
    "tennis": ["scheduled", "cancelled"],
    "mma": ["scheduled", "cancelled"],
    "soccer": ["scheduled"],
}

# Soccer no-twin classification (report only — nothing is written).
OTHER_IDENTITY_HOURS = 2  # football-data's summer kickoffs sit 1 h late
REPLAY_LOOKAHEAD_DAYS = 120
SAME_CLUB_RATIO = 0.85  # fetch_upcoming.find_team's fuzzy threshold

ACTION_MERGE_DELETE = "merge_delete"
ACTION_MERGE_CANCEL = "merge_cancel"  # soccer: pinned by recs, merged then marked cancelled
ACTION_DELETE_NO_TWIN = "delete_no_twin"  # tennis/MMA only
ACTION_DELETE_PLACEHOLDER = "delete_placeholder"
ACTION_LEAVE = "leave"
WRITE_ACTIONS = (ACTION_MERGE_DELETE, ACTION_MERGE_CANCEL, ACTION_DELETE_NO_TWIN, ACTION_DELETE_PLACEHOLDER)

# Why a row is left alone (the `reason` of ACTION_LEAVE).
LEFT_HAS_RECS = "has_recs"  # tennis/MMA: pinned by recs
LEFT_PENDING_RECS = "pending_recs"  # soccer: has a twin, but a pending rec sits on it
LEFT_PLACEHOLDER = "placeholder"  # soccer without --delete-placeholders
LEFT_PLACEHOLDER_HAS_RECS = "placeholder_has_recs"
NO_TWIN_FLIPPED = "flipped"
NO_TWIN_OTHER_IDENTITY = "other_identity"
NO_TWIN_REPLAYED_LATER = "replayed_later"
NO_TWIN_NEVER_RESULTED = "never_resulted"

_HAS_RECS = """
    EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.match_id = m.id)
      OR EXISTS (SELECT 1 FROM betting_history h WHERE h.match_id = m.id)
      OR EXISTS (SELECT 1 FROM betting_recommendations r
                   JOIN predictions p ON p.id = r.prediction_id
                  WHERE p.match_id = m.id)"""
_PENDING_RECS = "(SELECT count(*) FROM betting_recommendations r WHERE r.match_id = m.id AND r.status = 'pending')"

LIST_CANDIDATES_SQL = f"""
    SELECT m.id::text AS id, m.league_id::text AS league_id,
           m.home_team_id::text AS home_team_id, m.away_team_id::text AS away_team_id,
           ht.name AS home, at.name AS away,
           m.match_date, m.status,
           {_HAS_RECS} AS has_recs,
           {_PENDING_RECS} AS pending_recs
      FROM matches m
      JOIN leagues l ON l.id = m.league_id
      JOIN teams ht ON ht.id = m.home_team_id
      JOIN teams at ON at.id = m.away_team_id
     WHERE l.sport = %s
       AND m.status = ANY(%s)
       AND m.match_date < NOW() - (%s || ' hours')::interval
     ORDER BY m.match_date
"""

# Finished rows around one soccer no-twin candidate: the same league inside
# the twin window, and the same two clubs (either orientation) up to the
# replay look-ahead. Read only; feeds classify_no_twin.
NO_TWIN_CONTEXT_SQL = """
    SELECT t.id::text AS id, t.league_id::text AS league_id,
           t.home_team_id::text AS home_team_id, t.away_team_id::text AS away_team_id,
           ht.name AS home, at.name AS away, t.match_date
      FROM matches t
      JOIN teams ht ON ht.id = t.home_team_id
      JOIN teams at ON at.id = t.away_team_id
     WHERE t.id <> %(id)s::uuid
       AND t.status = 'finished'
       AND t.home_score IS NOT NULL
       AND t.away_score IS NOT NULL
       AND t.match_date >= %(kickoff)s::timestamptz - (%(window)s || ' hours')::interval
       AND ((t.league_id = %(league_id)s::uuid
             AND t.match_date <= %(kickoff)s::timestamptz + (%(window)s || ' hours')::interval)
            OR (t.home_team_id IN (%(home)s::uuid, %(away)s::uuid)
                AND t.away_team_id IN (%(home)s::uuid, %(away)s::uuid)
                AND t.match_date <= %(kickoff)s::timestamptz + (%(lookahead)s || ' days')::interval))
"""

# Per-orphan pre-game cutoff, LEAST(orphan kickoff, twin kickoff), computed
# in the plan and passed as parallel arrays.
_CUTOFFS = "SELECT * FROM unnest(%(orphans)s::uuid[], %(cutoffs)s::timestamptz[]) AS c(match_id, cutoff)"

_PRICE_KEY = (
    "o.bookmaker, o.market_type, o.selection, COALESCE(o.line, -1), "
    "COALESCE(o.is_opening, false), COALESCE(o.is_live, false)"
)

MOVE_ODDS_SQL = f"""
    WITH cut AS ({_CUTOFFS}),
    cand AS (
        SELECT DISTINCT ON ({_PRICE_KEY}) o.id
          FROM odds o
          JOIN cut ON cut.match_id = o.match_id
         WHERE o."timestamp" < cut.cutoff
         ORDER BY {_PRICE_KEY}, o."timestamp" DESC, o.created_at DESC, o.id
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

MOVE_SNAPSHOTS_SQL = f"""
    WITH cut AS ({_CUTOFFS}),
    cand AS (
        SELECT DISTINCT ON (s.bookmaker, s.market_type, s.selection, COALESCE(s.line, -1), s.captured_at) s.id
          FROM odds_snapshots s
          JOIN cut ON cut.match_id = s.match_id
         WHERE s.captured_at < cut.cutoff
         ORDER BY s.bookmaker, s.market_type, s.selection, COALESCE(s.line, -1), s.captured_at, s.id
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

MOVE_PREDICTIONS_SQL = f"""
    WITH cut AS ({_CUTOFFS}),
    cand AS (
        SELECT DISTINCT ON (p.model_name, p.model_version, p.prediction_type) p.id
          FROM predictions p
          JOIN cut ON cut.match_id = p.match_id
         WHERE COALESCE(p.updated_at, p.created_at) < cut.cutoff
         ORDER BY p.model_name, p.model_version, p.prediction_type, COALESCE(p.updated_at, p.created_at) DESC, p.id
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
       -- a rec's prediction stays where the rec put it
       AND NOT EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.prediction_id = p.id)
"""

# Lock the planned orphans (and, separately, the twin) before anything moves;
# the plan's observed values are re-checked against these rows.
LOCK_ORPHANS_SQL = """
    SELECT m.id::text AS id, m.status, m.match_date,
           m.home_team_id::text AS home_team_id, m.away_team_id::text AS away_team_id
      FROM matches m
     WHERE m.id = ANY(%s::uuid[])
     ORDER BY m.id
       FOR UPDATE
"""

PINS_SQL = f"""
    SELECT m.id::text AS id, {_HAS_RECS} AS has_recs, {_PENDING_RECS} AS pending_recs
      FROM matches m
     WHERE m.id = ANY(%s::uuid[])
"""

LOCK_TWIN_SQL = """
    SELECT status, match_date FROM matches WHERE id = %s::uuid FOR SHARE
"""

# Guarded on every value the plan observed (parallel arrays), on the status
# set this tool may delete, and on nothing referencing the row by rec.
DELETE_ORPHANS_SQL = """
    DELETE FROM matches m
     USING unnest(%(ids)s::uuid[], %(statuses)s::text[], %(kickoffs)s::timestamptz[],
                  %(homes)s::uuid[], %(aways)s::uuid[]) AS obs(id, status, match_date, home_team_id, away_team_id)
     WHERE m.id = obs.id
       AND m.status IN ('scheduled', 'cancelled')
       AND m.status = obs.status
       AND m.match_date = obs.match_date
       AND m.home_team_id = obs.home_team_id
       AND m.away_team_id = obs.away_team_id
       AND NOT EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.match_id = m.id)
       AND NOT EXISTS (SELECT 1 FROM betting_history h WHERE h.match_id = m.id)
       -- a rec whose prediction lives here: CASCADE deleted it before 025, NO ACTION fails
       AND NOT EXISTS (SELECT 1 FROM betting_recommendations r
                         JOIN predictions p ON p.id = r.prediction_id
                        WHERE p.match_id = m.id)
"""

# Soccer: a merged orphan that recs still reference is retired, not deleted.
# Guarded on the observed values; never twice (orphan_of), never under a
# pending rec (repair_stranded_recs only reads 'scheduled' rows).
CANCEL_PINNED_SQL = """
    UPDATE matches m
       SET status = 'cancelled',
           metadata = COALESCE(m.metadata, '{}'::jsonb)
                      || jsonb_build_object('orphan_of', %(twin)s::text, 'orphan_note', %(note)s::text),
           updated_at = NOW()
     WHERE m.id = %(id)s::uuid
       AND m.status = 'scheduled'
       AND m.match_date = %(match_date)s
       AND m.home_team_id = %(home)s::uuid
       AND m.away_team_id = %(away)s::uuid
       AND NOT (COALESCE(m.metadata, '{}'::jsonb) ? 'orphan_of')
       AND NOT EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.match_id = m.id AND r.status = 'pending')
"""

# What the archive captures per orphan: the row itself and everything a
# move or the cascade can touch that carries data worth restoring
# (features_cache is derived and recomputed).
ARCHIVE_TABLES = ("odds", "odds_snapshots", "predictions", "match_stats", "match_weather")

CSV_FIELDS = (
    "match_id",
    "sport",
    "action",
    "reason",
    "twin_id",
    "twin_match_date",
    "twin_delta_hours",
    "cutoff",
    "status",
    "match_date",
    "home_team_id",
    "away_team_id",
    "home",
    "away",
    "has_recs",
    "pending_recs",
)


class UnsafeRequest(RuntimeError):
    """Arguments or state that would make the repair unsafe."""


@dataclass(frozen=True)
class Policy:
    """What a sport's orphans may have done to them."""

    delete_no_twin: bool  # tennis/MMA: a pairing that never played. Soccer: never.
    cancel_pinned: bool  # soccer: merge a pinned orphan, then mark it cancelled
    delete_placeholders: bool


def policy_for(sport: str, delete_placeholders: bool) -> Policy:
    if sport == "soccer":
        return Policy(delete_no_twin=False, cancel_pinned=True, delete_placeholders=delete_placeholders)
    # tennis/MMA: a placeholder has no twin and was always deleted with them
    return Policy(delete_no_twin=True, cancel_pinned=False, delete_placeholders=True)


def is_placeholder(c: dict) -> bool:
    """An ESPN draw-slot placeholder side, or a team against itself."""
    return (
        _is_placeholder_name(c.get("home") or "")
        or _is_placeholder_name(c.get("away") or "")
        or c["home_team_id"] == c["away_team_id"]
    )


_SOCCER = SPORT_CONFIGS["soccer"]


def _club_key(name: str) -> str:
    return _strip_club_tokens(strip_accents(normalize_team(name or "")), _SOCCER)


def same_club(a: str, b: str) -> bool:
    """Two spellings of one club: equal after folding accents and club
    tokens, one's words a subset of the other's ('vasco' / 'vasco da gama',
    'orgryte' / 'orgryte is'), or find_team's fuzzy threshold."""
    ka, kb = _club_key(a), _club_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    ta, tb = set(ka.split()), set(kb.split())
    if ta <= tb or tb <= ta:
        return True
    return SequenceMatcher(None, ka, kb).ratio() >= SAME_CLUB_RATIO


def classify_no_twin(c: dict, rows: list[dict], twin_window_hours: int) -> str:
    """Why a soccer orphan has no twin, from the finished rows around it
    (NO_TWIN_CONTEXT_SQL). Pure. Report only: every class is left alone."""
    kick, home, away = c["match_date"], c["home_team_id"], c["away_team_id"]
    window = timedelta(hours=twin_window_hours)
    near = timedelta(hours=OTHER_IDENTITY_HOURS)
    lookahead = timedelta(days=REPLAY_LOOKAHEAD_DAYS)
    for t in rows:
        if (t["home_team_id"], t["away_team_id"]) == (away, home) and abs(t["match_date"] - kick) <= window:
            return NO_TWIN_FLIPPED
    for t in rows:
        if t.get("league_id") != c.get("league_id") or abs(t["match_date"] - kick) > near:
            continue
        if (t["home_team_id"], t["away_team_id"]) == (home, away):
            continue
        shares_a_team = t["home_team_id"] == home or t["away_team_id"] == away
        if shares_a_team or (same_club(t["home"], c["home"]) and same_club(t["away"], c["away"])):
            return NO_TWIN_OTHER_IDENTITY
    for t in rows:
        if (t["home_team_id"], t["away_team_id"]) == (home, away) and window < t["match_date"] - kick <= lookahead:
            return NO_TWIN_REPLAYED_LATER
    return NO_TWIN_NEVER_RESULTED


def _observed(row: dict) -> tuple:
    """The values every write is guarded on, in one comparable tuple."""
    return (
        row["status"],
        row["match_date"],
        str(row["home_team_id"]),
        str(row["away_team_id"]),
        bool(row["has_recs"]),
        int(row.get("pending_recs") or 0),
    )


def plan(
    candidates: list[dict],
    twins_for: Callable[[dict], list[dict]],
    policy: Optional[Policy] = None,
    context_for: Optional[Callable[[dict], list[dict]]] = None,
    twin_window_hours: int = 72,
) -> dict:
    """Decide every candidate's action. `twins_for(candidate)` returns the
    already-orientation-checked twin list (find_twins); `context_for` the
    finished rows around a soccer no-twin row (classification only). Pure
    apart from them."""
    policy = policy or policy_for("tennis", False)
    groups: dict[str, list[str]] = defaultdict(list)  # twin_id -> orphans to merge, then delete
    cancel: dict[str, list[str]] = defaultdict(list)  # twin_id -> orphans to merge, then mark cancelled
    kickoff: dict[str, datetime] = {}
    cutoff: dict[str, datetime] = {}
    observed: dict[str, tuple] = {}
    no_twin: list[str] = []
    placeholders: list[str] = []
    entries: list[dict] = []
    counts: Counter = Counter()

    for c in candidates:
        e = {
            "match_id": c["id"],
            "action": ACTION_LEAVE,
            "reason": "",
            "status": c["status"],
            "match_date": c["match_date"],
            "home_team_id": c["home_team_id"],
            "away_team_id": c["away_team_id"],
            "home": c.get("home"),
            "away": c.get("away"),
            "has_recs": bool(c["has_recs"]),
            "pending_recs": int(c.get("pending_recs") or 0),
        }
        entries.append(e)
        observed[c["id"]] = _observed(c)

        if is_placeholder(c):
            if c["has_recs"]:
                e["reason"] = LEFT_PLACEHOLDER_HAS_RECS
            elif policy.delete_placeholders:
                e["action"] = ACTION_DELETE_PLACEHOLDER
                placeholders.append(c["id"])
            else:
                e["reason"] = LEFT_PLACEHOLDER
            continue
        if c["has_recs"] and not policy.cancel_pinned:
            e["reason"] = LEFT_HAS_RECS
            continue

        twin, reason = rsr.select_twin(twins_for(c), c["match_date"])
        if twin is None and reason == rsr.REASON_NO_TWIN:
            if policy.delete_no_twin and not c["has_recs"]:
                e["action"] = ACTION_DELETE_NO_TWIN
                no_twin.append(c["id"])
            elif context_for is not None:
                e["reason"] = classify_no_twin(c, context_for(c), twin_window_hours)
            else:
                e["reason"] = "no_twin"
            continue
        if twin is None:
            e["reason"] = reason
            continue

        twin_id = twin["twin_id"]
        e["twin_id"] = twin_id
        e["twin_match_date"] = twin["match_date"]
        e["twin_delta_hours"] = round(abs((twin["match_date"] - c["match_date"]).total_seconds()) / 3600.0, 3)
        e["cutoff"] = min(c["match_date"], twin["match_date"])
        if c["has_recs"] and e["pending_recs"]:
            e["reason"] = LEFT_PENDING_RECS
            continue
        kickoff[twin_id] = twin["match_date"]
        cutoff[c["id"]] = e["cutoff"]
        if c["has_recs"]:
            e["action"] = ACTION_MERGE_CANCEL
            cancel[twin_id].append(c["id"])
        else:
            e["action"] = ACTION_MERGE_DELETE
            groups[twin_id].append(c["id"])

    for e in entries:
        counts[e["action"] if e["action"] != ACTION_LEAVE else f"left_{e['reason']}"] += 1
        if e["action"] == ACTION_LEAVE and e["has_recs"]:
            counts["left_pinned"] += 1
    counts["twins"] = len(set(groups) | set(cancel))
    return {
        "entries": entries,
        "groups": dict(groups),
        "cancel": dict(cancel),
        "kickoff": kickoff,
        "cutoff": cutoff,
        "observed": observed,
        "no_twin": no_twin,
        "placeholders": placeholders,
        "counts": counts,
    }


def plan_line(counts: Counter) -> str:
    """The machine-readable plan summary the ops wrapper parses and pins.
    blocked_pending_recs > 0 means --apply refuses (repair_stranded_recs first)."""
    writes = " ".join(f"{a}={counts.get(a, 0)}" for a in WRITE_ACTIONS)
    left = sum(v for k, v in counts.items() if k.startswith("left_") and k != "left_pinned")
    return f"PLAN {writes} left={left} blocked_pending_recs={counts.get(f'left_{LEFT_PENDING_RECS}', 0)}"


def parse_expect_plan(text: str) -> dict[str, int]:
    """'merge_delete=203,merge_cancel=28' -> {...}; every key must be a write action."""
    expected: dict[str, int] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        key, sep, value = part.partition("=")
        if not sep or key not in WRITE_ACTIONS or not value.isdigit():
            raise UnsafeRequest(f"--expect-plan: bad entry {part!r} (want <action>=<n>, action in {WRITE_ACTIONS})")
        expected[key] = int(value)
    if not expected:
        raise UnsafeRequest("--expect-plan is empty")
    return expected


def check_expected(counts: Counter, expected: dict[str, int]) -> None:
    diff = {k: (v, counts.get(k, 0)) for k, v in expected.items() if counts.get(k, 0) != v}
    if diff:
        raise UnsafeRequest(
            "plan differs from --expect-plan (expected, planned): "
            + ", ".join(f"{k} {v[0]} != {v[1]}" for k, v in sorted(diff.items()))
            + " — the data moved since the dry run; re-run the dry run and review"
        )


def _jsonable(v):
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return str(v)
    return v


def _csv_value(v) -> str:
    """Timestamps in UTC whatever the session time zone, so the CSV reads
    the same as the plan."""
    if v is None:
        return ""
    if isinstance(v, datetime) and v.tzinfo is not None:
        return v.astimezone(timezone.utc).isoformat()
    return str(_jsonable(v))


def write_csv(path: Path, sport: str, entries: list[dict]) -> Path:
    """Every planned row, BEFORE any write (dry run included). Opened "x":
    write-once, an existing path is an error, never an overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for e in entries:
            row = {**e, "sport": sport}
            writer.writerow({k: _csv_value(row.get(k)) for k in CSV_FIELDS})
    logger.info("wrote %d planned row(s) to %s", len(entries), path)
    return path


def write_archive(cur, path: Path, orphan_ids: list[str], twin_of: dict[str, str], action_of: dict[str, str]) -> int:
    if path.exists():
        raise FileExistsError(f"{path} exists — the archive is write-once")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "xt") as fh:
        for i in range(0, len(orphan_ids), 2000):
            chunk = orphan_ids[i : i + 2000]
            cur.execute("SELECT * FROM matches WHERE id = ANY(%s::uuid[])", (chunk,))
            for r in cur.fetchall():
                row = {k: _jsonable(v) for k, v in r.items()}
                mid = str(r["id"])
                line = {"table": "matches", "twin_id": twin_of.get(mid), "action": action_of.get(mid), "row": row}
                fh.write(json.dumps(line) + "\n")
                n += 1
            for table in ARCHIVE_TABLES:
                cur.execute(f"SELECT * FROM {table} WHERE match_id = ANY(%s::uuid[])", (chunk,))
                for r in cur.fetchall():
                    row = {k: _jsonable(v) for k, v in r.items()}
                    fh.write(json.dumps({"table": table, "row": row}) + "\n")
                    n += 1
    return n


def _count_refused(done: Counter, planned: int, deleted: int) -> None:
    """The delete guard refusing a planned orphan means a rec or a history
    row appeared on it after the lock: never silent."""
    if deleted < planned:
        done["delete_refused_has_recs"] += planned - deleted
        logger.error("delete guard kept %d planned orphan(s): a rec or history row references them", planned - deleted)


def _lock_unchanged(cur, ids: list[str], observed: dict[str, tuple], done: Counter) -> list[str]:
    """Lock the planned orphans and keep only those still exactly as the plan
    saw them; any other is left as it is, loudly."""
    cur.execute(LOCK_ORPHANS_SQL, (ids,))
    rows = {r["id"]: dict(r) for r in cur.fetchall()}
    cur.execute(PINS_SQL, (ids,))
    for r in cur.fetchall():
        if r["id"] in rows:
            rows[r["id"]].update(has_recs=r["has_recs"], pending_recs=r["pending_recs"])
    live = []
    for i in ids:
        now = _observed(rows[i]) if i in rows else None
        if now != observed[i]:
            done["refused_changed_since_plan"] += 1
            logger.error("orphan %s changed since the plan (%s -> %s) — left as it is", i, observed[i], now)
            continue
        live.append(i)
    return live


def _guard_params(ids: list[str], observed: dict[str, tuple]) -> dict:
    return {
        "ids": ids,
        "statuses": [observed[i][0] for i in ids],
        "kickoffs": [observed[i][1] for i in ids],
        "homes": [observed[i][2] for i in ids],
        "aways": [observed[i][3] for i in ids],
    }


def _orphan_note() -> str:
    return (
        "fixture-identity fork of the orphan_of row; kept, not deleted, because recommendations reference it. "
        f"Set by scripts/repair_fixture_orphans.py, {datetime.now(timezone.utc):%Y-%m-%d}."
    )


def _merge_group(cur, p: dict, twin_id: str, done: Counter) -> None:
    to_delete_planned = p["groups"].get(twin_id, [])
    to_cancel_planned = p["cancel"].get(twin_id, [])
    planned = to_delete_planned + to_cancel_planned
    cur.execute(LOCK_TWIN_SQL, (twin_id,))
    twin = cur.fetchone()
    if twin is None or twin["status"] != "finished" or twin["match_date"] != p["kickoff"][twin_id]:
        done["refused_changed_since_plan"] += len(planned)
        logger.error(
            "twin %s changed since the plan (%s) — its %d orphan(s) left as they are", twin_id, twin, len(planned)
        )
        return
    live = _lock_unchanged(cur, planned, p["observed"], done)
    if not live:
        return
    params = {"orphans": live, "cutoffs": [p["cutoff"][o] for o in live], "twin": twin_id}
    for label, sql in (
        ("odds", MOVE_ODDS_SQL),
        ("odds_snapshots", MOVE_SNAPSHOTS_SQL),
        ("predictions", MOVE_PREDICTIONS_SQL),
    ):
        cur.execute(sql, params)
        done[f"moved_{label}"] += cur.rowcount
    delete_set, cancel_set = set(to_delete_planned), set(to_cancel_planned)
    to_delete = [o for o in live if o in delete_set]
    if to_delete:
        cur.execute(DELETE_ORPHANS_SQL, _guard_params(to_delete, p["observed"]))
        done["deleted_merged_orphans"] += cur.rowcount
        _count_refused(done, len(to_delete), cur.rowcount)
    note = _orphan_note()
    for o in (o for o in live if o in cancel_set):
        _status, match_date, home, away = p["observed"][o][:4]
        cur.execute(
            CANCEL_PINNED_SQL,
            {"id": o, "twin": twin_id, "note": note, "match_date": match_date, "home": home, "away": away},
        )
        if cur.rowcount == 1:
            done["cancelled_pinned_orphans"] += 1
        else:
            done["cancel_refused"] += 1
            logger.error("cancel guard refused pinned orphan %s (twin %s): it changed under the lock", o, twin_id)


def apply_plan(conn, p: dict, batch_size: int) -> Counter:
    done: Counter = Counter()
    twins = list(dict.fromkeys(list(p["groups"]) + list(p.get("cancel", {}))))
    for i in range(0, len(twins), batch_size):
        with conn.cursor() as cur:
            for twin_id in twins[i : i + batch_size]:
                _merge_group(cur, p, twin_id, done)
        conn.commit()
        logger.info("batch %d-%d committed (%s)", i, i + len(twins[i : i + batch_size]), dict(done))
    for label, ids in (("no_twin", p["no_twin"]), ("placeholder", p.get("placeholders", []))):
        for i in range(0, len(ids), batch_size * 10):
            with conn.cursor() as cur:
                live = _lock_unchanged(cur, ids[i : i + batch_size * 10], p["observed"], done)
                if live:
                    cur.execute(DELETE_ORPHANS_SQL, _guard_params(live, p["observed"]))
                    done[f"deleted_{label}_orphans"] += cur.rowcount
                    _count_refused(done, len(live), cur.rowcount)
            conn.commit()
    return done


def refused(done: Counter) -> int:
    return done["refused_changed_since_plan"] + done["delete_refused_has_recs"] + done["cancel_refused"]


def _backup_dir() -> Path:
    return Path(os.environ.get("BACKUP_LOCAL_DIR", "/app/backups"))


def default_paths(sport: str, apply: bool) -> tuple[Path, Path]:
    """(audit CSV, archive), one UTC stamp, in the volume backup_postgres.py
    writes to; the mode is in the CSV name so a dry run and the apply that
    follows it within the same second never collide."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = _backup_dir()
    mode = "apply" if apply else "dryrun"
    return base / f"fixture_orphans_{sport}_{mode}_{stamp}.csv", base / f"fixture_orphans_{sport}_{stamp}.jsonl.gz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sport", required=True, choices=ALLOWED_SPORTS)
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--stale-after-hours", type=int, default=DEFAULT_STALE_AFTER_HOURS)
    ap.add_argument("--twin-window-hours", type=int, default=72)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument(
        "--delete-placeholders",
        action="store_true",
        help="soccer: also delete stale draw-slot placeholder rows and team-vs-itself rows no rec references",
    )
    ap.add_argument("--out-csv", type=Path, default=None, help="write-once audit CSV of the plan (dry run included)")
    ap.add_argument("--archive", type=Path, default=None, help="gzip JSONL reversal record (write-once)")
    ap.add_argument(
        "--expect-plan",
        default=None,
        help="refuse unless the plan's write counts equal these, e.g. merge_delete=203,merge_cancel=28",
    )
    ap.add_argument("--apply", action="store_true", help="execute (default: dry-run)")
    args = ap.parse_args(argv)
    if args.stale_after_hours < MIN_STALE_AFTER_HOURS:
        raise UnsafeRequest(f"--stale-after-hours must be >= {MIN_STALE_AFTER_HOURS} (results path look-back)")
    if not rsr.MIN_TWIN_WINDOW_HOURS <= args.twin_window_hours <= rsr.MAX_TWIN_WINDOW_HOURS:
        raise UnsafeRequest(
            f"--twin-window-hours must be in [{rsr.MIN_TWIN_WINDOW_HOURS}, {rsr.MAX_TWIN_WINDOW_HOURS}]"
        )
    if args.batch_size < 1:
        raise UnsafeRequest("--batch-size must be >= 1")
    args.expected = parse_expect_plan(args.expect_plan) if args.expect_plan else None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        raise UnsafeRequest("DATABASE_URL not set")
    policy = policy_for(args.sport, args.delete_placeholders)
    conn = psycopg2.connect(args.database_url, cursor_factory=RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(LIST_CANDIDATES_SQL, (args.sport, CANDIDATE_STATUSES[args.sport], str(args.stale_after_hours)))
            candidates = [dict(r) for r in cur.fetchall()]

            def twins_for(c: dict) -> list[dict]:
                return rsr.find_twins(
                    cur, c["id"], c["home_team_id"], c["away_team_id"], c["match_date"], args.twin_window_hours
                )

            def context_for(c: dict) -> list[dict]:
                cur.execute(
                    NO_TWIN_CONTEXT_SQL,
                    {
                        "id": c["id"],
                        "league_id": c["league_id"],
                        "home": c["home_team_id"],
                        "away": c["away_team_id"],
                        "kickoff": c["match_date"],
                        "window": str(args.twin_window_hours),
                        "lookahead": str(REPLAY_LOOKAHEAD_DAYS),
                    },
                )
                return [dict(r) for r in cur.fetchall()]

            p = plan(
                candidates,
                twins_for,
                policy,
                context_for=None if policy.delete_no_twin else context_for,
                twin_window_hours=args.twin_window_hours,
            )
        conn.rollback()  # end the read transaction
        counts = p["counts"]
        logger.info("candidates=%d plan=%s", len(candidates), dict(sorted(counts.items())))
        logger.info(plan_line(counts))
        pending = counts.get(f"left_{LEFT_PENDING_RECS}", 0)
        if pending:
            logger.error(
                "%d orphan(s) with a twin still hold PENDING recs — run repair_stranded_recs.py --sport %s first "
                "(it settles them onto the twin); --apply refuses until then",
                pending,
                args.sport,
            )
        if args.expected is not None:
            check_expected(counts, args.expected)
        if args.apply and pending:
            raise UnsafeRequest(f"{pending} orphan(s) with a twin hold pending recs: run repair_stranded_recs.py first")
        default_csv, default_archive = default_paths(args.sport, args.apply)
        write_csv(args.out_csv or default_csv, args.sport, p["entries"])
        if not args.apply:
            logger.info("dry-run: nothing written (re-run with --apply)")
            return 0
        targets = [e for e in p["entries"] if e["action"] in WRITE_ACTIONS]
        if not targets:
            logger.info("DONE nothing to write")
            return 0
        twin_of = {e["match_id"]: e["twin_id"] for e in targets if e.get("twin_id")}
        action_of = {e["match_id"]: e["action"] for e in targets}
        archive = args.archive or default_archive
        with conn.cursor() as cur:
            n = write_archive(cur, archive, [e["match_id"] for e in targets], twin_of, action_of)
        conn.rollback()  # end the read transaction before the batched writes
        logger.info("archived %d rows to %s", n, archive)
        done = apply_plan(conn, p, args.batch_size)
        logger.info("DONE %s", dict(sorted(done.items())))
        if refused(done):
            logger.error(
                "%d planned row(s) were NOT written (changed since the plan or refused by a guard); "
                "every other batch is committed — inspect, then re-run the dry run",
                refused(done),
            )
            return 1
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
