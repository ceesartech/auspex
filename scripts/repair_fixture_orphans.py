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
               That holds only once the days ESPN resulted are in: no UFC
               bout before 2026-07-10 was ever resulted (results ingestion
               shipped 07-13 with a 3-day look-back), and on 2026-11-01 44
               fought June bouts sat here without a twin. Backfill such days
               first (scripts/backfill_results.py;
               scripts/ops/repair_stranded_recs_mma.sh runs it before this).
               Likewise a bout stored under a fighter's second spelling, or
               the other way round, has no twin HERE (same team ids, same
               order) though it was fought: 9 such UFC bouts on 2026-11-01.
               Merge the fighter rows first (scripts/merge_mma_fighters.py
               through scripts/ops/merge_mma_fighters.sh, which folds them);
               the MMA run order is repair_stranded_recs_mma.sh (A),
               merge_mma_fighters.sh, repair_stranded_recs_mma.sh
               --retire-stale-rows (this tool).
               soccer: LEFT ALONE and reported by class (see below).
  ambiguous   -> left alone, counted.
  soccer twin that is a football-data copy -> LEFT ALONE, whether or not
               recs pin the orphan: fd_copy_awaiting_repair. The twin has
               no ESPN identity (no external_ids.espn, result not from the
               ESPN scoreboard) and kicks off exactly 1 h after the orphan:
               promote_raw stored football-data's UK local time as UTC until
               2026-10-31, so the orphan holds the real kickoff.
               repair_football_data_kickoffs folds such a copy INTO the
               stale row at its corrected kickoff (it runs before this tool,
               see RUN ORDER); merging the other way round would delete or
               cancel the row on the right slot and leave the training row
               1 h late for good. A row still held here means the fold
               refused it: nothing is written, the class says so. A twin
               WITH ESPN identity 1 h later is an ESPN kickoff move and
               merges as usual (the kickoff repair never moves an ESPN row).

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
  never_resulted  none of the above: a results backfill can settle it
                  (scripts/backfill_never_resulted.py, runbook step 12).

POSTPONED (--postponed-list, soccer)
------------------------------------
A replayed_later row whose postponement ESPN confirms (the replay is the
same ESPN event, moved out of the original round) is not a fork: it is the
original date of a postponed fixture, and recs priced for that date should
be voided as books void them. --postponed-list names a reviewed CSV
(scripts/data/soccer_postponed_stale_rows.csv: match_id, kickoff,
replay_match_id, replay_kickoff, replay_espn_event, evidence, ...). Each
listed row must still be a stale 'scheduled' candidate at the listed
kickoff, with no twin, and its replay the listed finished row for the same
pairing at the listed kickoff, beyond the twin window; it is then set to
status 'postponed' with metadata.postponed_kickoff (its kickoff) and
postponed_at (now), as fetch_upcoming._record_status_change records ESPN's
STATUS_POSTPONED, plus replayed_as / replay_espn_event / postponed_evidence.
Nothing moves and no rec is written: grade_completed_matches voids the open
recs of a match still 'postponed' 72 h after its kickoff. A listed row that
no longer matches is reported (list_mismatch on the PLAN line): the dry run
exits 1 and --apply refuses before any write. A listed row already marked
by an earlier run counts as done.

RUN ORDER (soccer; scripts/ops/soccer_repair_runbook.sh runs it, each step
through its own wrapper)
------------------
  1. scripts/ops/repoint_team_matches.sh --pass 1
  2. scripts/ops/migrate_027_outcome_trigger.sh
  3. scripts/ops/merge_soccer_teams.sh
  4. scripts/ops/repair_football_data_kickoffs.sh
  5. scripts/ops/repoint_team_matches.sh --pass 2
  6. scripts/ops/repair_stranded_recs_soccer.sh
  7. scripts/ops/cleanup_soccer_orphans.sh
  8. scripts/ops/backfill_soccer_regulation_scores.sh (+ the re-grade it prints)
  9. scripts/ops/backfill_season_labels.sh
 10. compute_features.py --backfill-finished soccer
 11. merge_soccer_teams.py --check-after <plan> --odds-since <MERGED_AT> --after-later-steps
Step 3 turns forks under an ESPN-spelling team row into same-id rows and
folds the twin clusters that creates; step 4 folds the 1 h-late
football-data copies into the stale rows at their corrected kickoff, then
shifts the rest; step 5 moves the last Liga MX rows (run after this tool, it
would find two of them merged away); step 6 settles and re-points the
pending recs on orphans that have a twin (--apply here refuses until it
has); step 7 is this tool (scripts/ops/cleanup_soccer_orphans.sh).

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
line): any difference refuses the run before anything is written. --as-of
fixes the instant staleness is measured from (default now), so a dry run,
its apply, a re-plan and the wrapper's state queries share one cutoff.

Usage (inside the api container; dry-run is the default):
    python /app/scripts/repair_fixture_orphans.py --sport tennis
    python /app/scripts/repair_fixture_orphans.py --sport tennis --apply
    python /app/scripts/repair_fixture_orphans.py --sport soccer --delete-placeholders \
        --postponed-list /app/scripts/data/soccer_postponed_stale_rows.csv
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
ACTION_MARK_POSTPONED = "mark_postponed"  # soccer: on the reviewed, ESPN-verified postponed list
ACTION_LEAVE = "leave"
WRITE_ACTIONS = (
    ACTION_MERGE_DELETE,
    ACTION_MERGE_CANCEL,
    ACTION_DELETE_NO_TWIN,
    ACTION_DELETE_PLACEHOLDER,
    ACTION_MARK_POSTPONED,
)

# Why a row is left alone (the `reason` of ACTION_LEAVE).
LEFT_HAS_RECS = "has_recs"  # tennis/MMA: pinned by recs
LEFT_PENDING_RECS = "pending_recs"  # soccer: has a twin, but a pending rec sits on it
LEFT_PLACEHOLDER = "placeholder"  # soccer without --delete-placeholders
LEFT_PLACEHOLDER_HAS_RECS = "placeholder_has_recs"
NO_TWIN_FLIPPED = "flipped"
NO_TWIN_OTHER_IDENTITY = "other_identity"
NO_TWIN_REPLAYED_LATER = "replayed_later"
NO_TWIN_NEVER_RESULTED = "never_resulted"
# soccer: the twin is a football-data copy 1 h late; repair_football_data_kickoffs folds it
HELD_FD_COPY = "fd_copy_awaiting_repair"
LEFT_POSTPONED_MISMATCH = "postponed_list_mismatch"

# promote_raw stored football-data's UK local Time as UTC until 2026-10-31:
# a summer football-data row sits exactly this much after the real kickoff.
FD_LEGACY_OFFSET = timedelta(hours=1)

# The reviewed postponed list (--postponed-list): every column must be present.
POSTPONED_LIST_FIELDS = ("match_id", "kickoff", "replay_match_id", "replay_kickoff", "replay_espn_event", "evidence")

_HAS_RECS = """
    EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.match_id = m.id)
      OR EXISTS (SELECT 1 FROM betting_history h WHERE h.match_id = m.id)
      OR EXISTS (SELECT 1 FROM betting_recommendations r
                   JOIN predictions p ON p.id = r.prediction_id
                  WHERE p.match_id = m.id)"""
# Open recs, the set grade_completed_matches lists ('pending' and 'placed'): CANCEL_PINNED_SQL never retires a row
# holding one, and the plan blocks on the same set, so the two never disagree.
_PENDING_RECS = (
    "(SELECT count(*) FROM betting_recommendations r WHERE r.match_id = m.id AND r.status IN ('pending', 'placed'))"
)

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
       AND m.match_date < %s::timestamptz - (%s || ' hours')::interval
     ORDER BY m.match_date
"""

# A twin's ESPN identity (promote_raw.has_espn_identity on the football-data
# branch): an ESPN competition id (fetch_upcoming, since migration 023) or a
# result off the ESPN scoreboard. A twin without either is a football-data row.
TWIN_IDENTITY_SQL = """
    SELECT external_ids->>'espn' AS espn_id, metadata->>'result_source' AS result_source
      FROM matches
     WHERE id = %s::uuid
"""

# Where the listed postponed rows that are not candidates stand (done, or a mismatch).
LISTED_STATE_SQL = """
    SELECT m.id::text AS id, m.status, m.match_date,
           m.metadata->>'postponed_kickoff' AS postponed_kickoff, m.metadata->>'replayed_as' AS replayed_as
      FROM matches m
     WHERE m.id = ANY(%s::uuid[])
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
           -- a rec's prediction stays where the rec put it. Filtered HERE, before
           -- DISTINCT ON picks: a pick that is rec-referenced (on a merge_cancel
           -- sibling) would otherwise move nothing for its key, and the older
           -- movable one on a merge_delete sibling would go down with its orphan.
           AND NOT EXISTS (SELECT 1 FROM betting_recommendations r WHERE r.prediction_id = p.id)
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
    SELECT status, match_date, external_ids->>'espn' AS espn_id, metadata->>'result_source' AS result_source
      FROM matches
     WHERE id = %s::uuid
       FOR SHARE
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
# Guarded on the observed values; never twice (orphan_of), never under an
# open ('pending' or 'placed') rec: grade_completed_matches refuses to void
# an open rec on a cancelled orphan_of row (the match was played, on the
# twin), so such a rec would only be settled by hand.
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
       AND NOT EXISTS (SELECT 1 FROM betting_recommendations r
                        WHERE r.match_id = m.id AND r.status IN ('pending', 'placed'))
"""

# Soccer: a listed, ESPN-verified postponed stale row. Guarded on the
# observed values, on never having been marked (postponed_kickoff is
# stamped once, as fetch_upcoming does), and on its replay still being the
# finished row for the same pairing at the listed kickoff.
MARK_POSTPONED_SQL = """
    UPDATE matches m
       SET status = 'postponed',
           metadata = COALESCE(m.metadata, '{}'::jsonb)
                      || jsonb_build_object('postponed_kickoff', %(postponed_kickoff)s::text,
                                            'postponed_at', %(postponed_at)s::text,
                                            'status_source', 'repair_fixture_orphans',
                                            'replayed_as', %(replay)s::text,
                                            'replay_espn_event', %(espn_event)s::text,
                                            'postponed_evidence', %(evidence)s::text),
           updated_at = NOW()
     WHERE m.id = %(id)s::uuid
       AND m.status = 'scheduled'
       AND m.match_date = %(match_date)s
       AND m.home_team_id = %(home)s::uuid
       AND m.away_team_id = %(away)s::uuid
       AND NOT (COALESCE(m.metadata, '{}'::jsonb) ? 'postponed_kickoff')
       AND EXISTS (SELECT 1 FROM matches t
                    WHERE t.id = %(replay)s::uuid
                      AND t.status = 'finished'
                      AND t.home_team_id = m.home_team_id
                      AND t.away_team_id = m.away_team_id
                      AND t.match_date = %(replay_kickoff)s)
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
    hold_fd_copies: bool = False  # soccer: a 1 h-late football-data twin waits for its fold


def policy_for(sport: str, delete_placeholders: bool) -> Policy:
    if sport == "soccer":
        return Policy(
            delete_no_twin=False, cancel_pinned=True, delete_placeholders=delete_placeholders, hold_fd_copies=True
        )
    # tennis/MMA: a placeholder has no twin and was always deleted with them
    return Policy(delete_no_twin=True, cancel_pinned=False, delete_placeholders=True)


def has_espn_identity(espn_id: Optional[str], result_source: Optional[str]) -> bool:
    """A row the ESPN path owns (promote_raw.has_espn_identity): an ESPN
    competition id, or a result written off the ESPN scoreboard."""
    return bool(espn_id) or result_source == "espn_scoreboard"


def is_fd_copy_twin(orphan_kickoff: datetime, twin_kickoff: datetime, twin_is_espn: bool) -> bool:
    """The twin is a football-data copy at the legacy UK-local kickoff:
    no ESPN identity and exactly FD_LEGACY_OFFSET after the orphan."""
    return not twin_is_espn and twin_kickoff - orphan_kickoff == FD_LEGACY_OFFSET


@dataclass(frozen=True)
class Postponement:
    """One row of the reviewed, ESPN-verified postponed list."""

    match_id: str
    kickoff: datetime
    replay_match_id: str
    replay_kickoff: datetime
    replay_espn_event: str
    evidence: str


def _aware(value: str, what: str) -> datetime:
    try:
        ts = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise UnsafeRequest(f"{what}: {value!r} is not an ISO-8601 timestamp") from exc
    if ts.tzinfo is None:
        raise UnsafeRequest(f"{what}: {value!r} has no time zone")
    return ts


def load_postponed_list(path: Path) -> dict[str, Postponement]:
    """The reviewed postponed list, validated: every POSTPONED_LIST_FIELDS
    column present and filled, zoned timestamps, one row per match, the
    replay after the original kickoff. Anything else refuses the run."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [f for f in POSTPONED_LIST_FIELDS if f not in (reader.fieldnames or [])]
        if missing:
            raise UnsafeRequest(f"{path}: missing column(s) {missing}")
        rows = list(reader)
    out: dict[str, Postponement] = {}
    for n, r in enumerate(rows, start=2):
        empty = [f for f in POSTPONED_LIST_FIELDS if not (r.get(f) or "").strip()]
        if empty:
            raise UnsafeRequest(f"{path}:{n}: empty {empty} — only ESPN-verified rows, with their evidence, go here")
        p = Postponement(
            match_id=r["match_id"].strip(),
            kickoff=_aware(r["kickoff"], f"{path}:{n} kickoff"),
            replay_match_id=r["replay_match_id"].strip(),
            replay_kickoff=_aware(r["replay_kickoff"], f"{path}:{n} replay_kickoff"),
            replay_espn_event=r["replay_espn_event"].strip(),
            evidence=r["evidence"].strip(),
        )
        if p.match_id in out:
            raise UnsafeRequest(f"{path}:{n}: {p.match_id} is listed twice")
        if p.replay_kickoff <= p.kickoff:
            raise UnsafeRequest(f"{path}:{n}: the replay kicks off before the postponed fixture")
        out[p.match_id] = p
    return out


def postponed_problem(
    c: dict, listed: Postponement, twin: Optional[dict], twin_reason: str, rows: list[dict], twin_window_hours: int
) -> Optional[str]:
    """Why a listed row may NOT be marked postponed (None: it may). Pure."""
    if is_placeholder(c):
        return "it is a placeholder row"
    if c["match_date"] != listed.kickoff:
        return f"its kickoff is {c['match_date']}, the list says {listed.kickoff}"
    if twin is not None:
        return f"it has a finished twin {twin['twin_id']} inside the window: a fork, not a postponement"
    if twin_reason != rsr.REASON_NO_TWIN:
        return f"its twins are ambiguous ({twin_reason})"
    replay = next((t for t in rows if t["id"] == listed.replay_match_id), None)
    if replay is None:
        return (
            f"replay {listed.replay_match_id} is not a finished row of this pairing within {REPLAY_LOOKAHEAD_DAYS} days"
        )
    if (replay["home_team_id"], replay["away_team_id"]) != (c["home_team_id"], c["away_team_id"]):
        return f"replay {listed.replay_match_id} is another pairing"
    if replay["match_date"] != listed.replay_kickoff:
        return f"replay kicks off {replay['match_date']}, the list says {listed.replay_kickoff}"
    if replay["match_date"] - c["match_date"] <= timedelta(hours=twin_window_hours):
        return "the replay is inside the twin window"
    return None


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
    twin_is_espn: Optional[Callable[[str], bool]] = None,
    postponed: Optional[dict[str, Postponement]] = None,
) -> dict:
    """Decide every candidate's action. `twins_for(candidate)` returns the
    already-orientation-checked twin list (find_twins); `context_for` the
    finished rows around a soccer no-twin row (classification, and the
    replay of a listed postponed row); `twin_is_espn(twin_id)` the twin's
    ESPN identity (asked only for a twin exactly 1 h late; unknown counts as
    none, which holds the row). `postponed`: the listed rows among the
    candidates. Pure apart from the callables."""
    policy = policy or policy_for("tennis", False)
    postponed = postponed or {}
    groups: dict[str, list[str]] = defaultdict(list)  # twin_id -> orphans to merge, then delete
    cancel: dict[str, list[str]] = defaultdict(list)  # twin_id -> orphans to merge, then mark cancelled
    kickoff: dict[str, datetime] = {}
    cutoff: dict[str, datetime] = {}
    observed: dict[str, tuple] = {}
    no_twin: list[str] = []
    placeholders: list[str] = []
    to_postpone: list[str] = []
    mismatches: list[str] = []
    twin_espn: dict[str, bool] = {}  # the identity a 1 h twin's decision rested on, re-checked under the lock
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

        listed = postponed.get(c["id"])
        if listed is not None:
            twin, reason = rsr.select_twin(twins_for(c), c["match_date"])
            rows = context_for(c) if context_for is not None else []
            problem = postponed_problem(c, listed, twin, reason, rows, twin_window_hours)
            if problem:
                e["reason"] = LEFT_POSTPONED_MISMATCH
                mismatches.append(f"{c['id']} ({c.get('home')} v {c.get('away')}): {problem}")
            else:
                e["action"] = ACTION_MARK_POSTPONED
                e["reason"] = NO_TWIN_REPLAYED_LATER
                e["twin_id"] = listed.replay_match_id
                e["twin_match_date"] = listed.replay_kickoff
                to_postpone.append(c["id"])
            continue
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
        if policy.hold_fd_copies and twin["match_date"] - c["match_date"] == FD_LEGACY_OFFSET:
            espn = twin_is_espn(twin_id) if twin_is_espn is not None else False
            if is_fd_copy_twin(c["match_date"], twin["match_date"], espn):
                e["reason"] = HELD_FD_COPY  # pinned or not: repair_football_data_kickoffs folds it
                continue
            twin_espn[twin_id] = espn
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
        "postponed": to_postpone,
        "postponed_meta": {i: postponed[i] for i in to_postpone},
        "mismatches": mismatches,
        "twin_espn": twin_espn,
        "counts": counts,
    }


def plan_line(counts: Counter) -> str:
    """The machine-readable plan summary the ops wrapper parses and pins.
    blocked_pending_recs > 0 means --apply refuses (repair_stranded_recs first);
    list_mismatch > 0 that the reviewed postponed list no longer matches
    (the dry run exits 1, --apply refuses)."""
    writes = " ".join(f"{a}={counts.get(a, 0)}" for a in WRITE_ACTIONS)
    left = sum(v for k, v in counts.items() if k.startswith("left_") and k != "left_pinned")
    return (
        f"PLAN {writes} left={left} blocked_pending_recs={counts.get(f'left_{LEFT_PENDING_RECS}', 0)} "
        f"list_mismatch={counts.get('list_mismatch', 0)}"
    )


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
    espn_seen = p.get("twin_espn", {}).get(twin_id)
    if (
        twin is None
        or twin["status"] != "finished"
        or twin["match_date"] != p["kickoff"][twin_id]
        or (espn_seen is not None and has_espn_identity(twin["espn_id"], twin["result_source"]) != espn_seen)
    ):
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
    ids = p.get("postponed", [])
    for i in range(0, len(ids), batch_size):
        with conn.cursor() as cur:
            for o in _lock_unchanged(cur, ids[i : i + batch_size], p["observed"], done):
                _mark_postponed(cur, o, p["observed"][o], p["postponed_meta"][o], done)
        conn.commit()
    return done


def _mark_postponed(cur, match_id: str, observed: tuple, listed: Postponement, done: Counter) -> None:
    _status, match_date, home, away = observed[:4]
    cur.execute(
        MARK_POSTPONED_SQL,
        {
            "id": match_id,
            "match_date": match_date,
            "home": home,
            "away": away,
            "postponed_kickoff": match_date.astimezone(timezone.utc).isoformat(),
            "postponed_at": datetime.now(timezone.utc).isoformat(),
            "replay": listed.replay_match_id,
            "replay_kickoff": listed.replay_kickoff,
            "espn_event": listed.replay_espn_event,
            "evidence": listed.evidence,
        },
    )
    if cur.rowcount == 1:
        done["marked_postponed"] += 1
    else:
        done["postpone_refused"] += 1
        logger.error(
            "postpone guard refused %s (replay %s): the row or its replay changed under the lock",
            match_id,
            listed.replay_match_id,
        )


def refused(done: Counter) -> int:
    return (
        done["refused_changed_since_plan"]
        + done["delete_refused_has_recs"]
        + done["cancel_refused"]
        + done["postpone_refused"]
    )


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
    ap.add_argument(
        "--postponed-list",
        type=Path,
        default=None,
        help="soccer: reviewed CSV of ESPN-verified postponed stale rows to mark 'postponed' (see POSTPONED)",
    )
    ap.add_argument(
        "--as-of",
        default=None,
        help="measure staleness from this instant (ISO-8601 with a zone, not in the future; default now)",
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
    if args.postponed_list is not None and args.sport != "soccer":
        raise UnsafeRequest("--postponed-list is for soccer")
    args.as_of = _aware(args.as_of, "--as-of") if args.as_of else datetime.now(timezone.utc)
    if args.as_of > datetime.now(timezone.utc):
        # a later instant would reach rows younger than the results path's look-back
        raise UnsafeRequest(f"--as-of {args.as_of.isoformat()} is in the future")
    args.expected = parse_expect_plan(args.expect_plan) if args.expect_plan else None
    return args


def listed_outside_candidates(cur, listed: dict[str, Postponement], candidate_ids: set[str]) -> tuple[int, list[str]]:
    """(already marked, mismatches) for the listed rows the candidate query
    did not return: done when an earlier run marked them for the listed
    kickoff and replay, a mismatch otherwise."""
    outside = sorted(i for i in listed if i not in candidate_ids)
    if not outside:
        return 0, []
    cur.execute(LISTED_STATE_SQL, (outside,))
    state = {r["id"]: r for r in cur.fetchall()}
    done, problems = 0, []
    for i in outside:
        r, want = state.get(i), listed[i]
        marked = None
        if r is not None and r["postponed_kickoff"]:
            try:
                marked = _aware(r["postponed_kickoff"], "postponed_kickoff")
            except UnsafeRequest:
                marked = None
        if (
            r is not None
            and r["status"] == "postponed"
            and marked == want.kickoff
            and r["replayed_as"] == want.replay_match_id
        ):
            done += 1
        elif r is None:
            problems.append(f"{i}: no such match")
        else:
            problems.append(
                f"{i}: not a stale 'scheduled' candidate (status {r['status']}, kickoff {r['match_date']}) "
                "and not marked postponed for the listed kickoff and replay"
            )
    return done, problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        raise UnsafeRequest("DATABASE_URL not set")
    policy = policy_for(args.sport, args.delete_placeholders)
    listed = load_postponed_list(args.postponed_list) if args.postponed_list else {}
    conn = psycopg2.connect(args.database_url, cursor_factory=RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(
                LIST_CANDIDATES_SQL,
                (args.sport, CANDIDATE_STATUSES[args.sport], args.as_of, str(args.stale_after_hours)),
            )
            candidates = [dict(r) for r in cur.fetchall()]
            candidate_ids = {c["id"] for c in candidates}
            already_postponed, outside_problems = listed_outside_candidates(cur, listed, candidate_ids)

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

            def twin_is_espn(twin_id: str) -> bool:
                cur.execute(TWIN_IDENTITY_SQL, (twin_id,))
                r = cur.fetchone()
                return r is not None and has_espn_identity(r["espn_id"], r["result_source"])

            p = plan(
                candidates,
                twins_for,
                policy,
                context_for=None if policy.delete_no_twin else context_for,
                twin_window_hours=args.twin_window_hours,
                twin_is_espn=twin_is_espn,
                postponed={i: v for i, v in listed.items() if i in candidate_ids},
            )
        conn.rollback()  # end the read transaction
        counts = p["counts"]
        mismatches = p["mismatches"] + outside_problems
        counts["list_mismatch"] = len(mismatches)
        logger.info(
            "as_of=%s candidates=%d plan=%s", args.as_of.isoformat(), len(candidates), dict(sorted(counts.items()))
        )
        if listed:
            logger.info(
                "postponed list: %d row(s), %d to mark, %d already marked, %d mismatch(es)",
                len(listed),
                len(p["postponed"]),
                already_postponed,
                len(mismatches),
            )
        for m in mismatches:
            logger.error("postponed list does not match the data — %s", m)
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
        if args.apply and mismatches:
            raise UnsafeRequest(f"{len(mismatches)} postponed-list row(s) no longer match: review the list first")
        default_csv, default_archive = default_paths(args.sport, args.apply)
        write_csv(args.out_csv or default_csv, args.sport, p["entries"])
        if not args.apply:
            if mismatches:
                logger.error("dry-run: %d postponed-list row(s) no longer match — exit 1", len(mismatches))
                return 1
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
