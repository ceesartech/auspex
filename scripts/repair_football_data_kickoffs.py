"""Move the football-data kickoffs promote_raw stored 1 h late back to UTC,
and merge the duplicate matches that left behind into the row that stays.

Until 2026-10-31 promote_raw.parse_match_date stored football-data's Time,
which is UK local time, as if it were UTC, so every kickoff on a British
Summer Time date sits 1 h late (63,768 of the 101,158 timed promoted rows on
2026-10-31; the GMT ones are right). Where the ESPN path had also written
the fixture at its real kickoff, the exact-triple upsert missed and the
fixture exists twice: 228 finished matches, 68 of them under the same team
ids (in the other 160 ESPN created its own team row), plus 26 football-data
results beside a stale 'scheduled' ESPN fixture row from before migration
023. promote_raw now parses UTC, writes onto an existing row for the pair
near the kickoff instead of beside it (find_existing_matches), and refuses
to run while any row is left at a legacy kickoff (check_legacy_kickoffs).
This tool merges and moves them; until it has run, promote_raw stays blocked.

RUN ORDER (scripts/ops/soccer_repair_runbook.sh runs it, each step through
its own wrapper with its own verified backup and checks; 2026-10-31):
  1. scripts/ops/repoint_team_matches.sh --pass 1, right after the
     team-identity deploy: it moves the Liga MX rows ESPN filed under Santos
     (the Brazilian club's row) and Guadalajara (the Spanish club's row)
     onto Santos Laguna and Guadalajara Chivas, and refuses a row while
     football-data's copy sits on the target pair (6 sides on 2026-10-31:
     4 ESPN rows and 2 stale rows moving with them);
  2. scripts/ops/migrate_027_outcome_trigger.sh;
  3. scripts/ops/merge_soccer_teams.sh: folds the 160 alias pairs where
     ESPN minted its own team row (keeps the ESPN row, deletes about 171
     football-data rows; promote_raw then writes those fixtures onto the
     surviving ESPN row, never beside it);
  4. this tool (scripts/ops/repair_football_data_kickoffs.sh): --dedupe
     alias-merges the 4 copies step 1 refused for (below), folds and
     merges the rest, --shift moves every other BST row;
  5. scripts/ops/repoint_team_matches.sh --pass 2 (never a bare --apply):
     its refused sides have no twin left and move (6); until then
     promote_raw refuses those fixtures (find_alias_slot_rows), loudly;
  6. scripts/ops/repair_stranded_recs_soccer.sh;
  7. scripts/ops/cleanup_soccer_orphans.sh: the 26 stale rows this tool
     folds are finished by then and leave its set;
  8. scripts/ops/backfill_soccer_regulation_scores.sh (+ its re-grade);
  9. scripts/ops/backfill_season_labels.sh;
 10. compute_features.py --backfill-finished soccer;
 11. merge_soccer_teams.py --check-after ... --odds-since ... once a
     fetch_live_odds run has finished after step 3.
If step 1 moves every row (football-data twins allowed), --dedupe merges
those 4 copies like the 68 and step 5 has nothing to do. If the team merge
has NOT run, its 160 pairs are left for it: --shift refuses each copy whose
new kickoff is the exact kickoff of the same fixture under other team ids
(see below), so this tool never creates a pair the merge would trip over,
and the wrapper aborts on the refusals before writing anything.

The reviewed re-point map (scripts/data/repoint_team_matches.csv, step 1's:
one row per match side, each proven by its ESPN event; --repoint-map) says
where a row filed under the wrong club is about to be: a mapped row whose
every mapped side is still on its from_team counts as a row of its
to_team pair as well (pending_repoints). Without the map nothing does, and
the 4 copies stay refused (alias slot).

A matches row is tied to its raw_football_data row the way promote_raw tied
them and backfill_season_labels ties them: league code, normalised team
names, and the kickoff promote_raw wrote (the UK wall clock,
promote_raw.parse_wall_clock, read in the database TimeZone) or, once moved,
the one it writes now.

--dedupe (run first). A football-data row is a copy when the row
promote_raw would now write its fixture onto is another row for the same
home and away team ids within 3 h of the corrected kickoff, on
football-data's date (promote_raw.existing_match_window / choose_existing).
The rows the reviewed re-point map is about to move onto the pair count
(pending_repoints). Kinds, all one guarded transaction per batch:
  merge: that row is an ESPN row, finished with the copy's final score. On
    prod it sits 1 h (+-10 min) earlier for 67 of the 68 same-id pairs; the
    68th ESPN row kicked off 76 min after football-data's legacy time, and
    two fixtures also keep a stale 'scheduled' row 60-70 min earlier, so a
    strict 1 h +-10 min rule would miss three. The copy's half-time goes onto
    the ESPN row when it has none, the pair is complete and neither half
    exceeds the ESPN final score (promote_raw.raw_result's check; it needs no
    ESPN plays).
  alias_merge: the same, where that ESPN row is still filed under the wrong
    club and the map moves it onto the copy's pair (the 4 Liga MX copies on
    2026-10-31; repoint_team_matches refused to move their rows while the
    copy sat on the target pair, and --shift refused the copy beside them).
    The row keeps its team ids (step 5 moves it); the copy's match_stats
    move only for a club in that row (STATS_FREE_ON_TWIN), the other is
    archived with the copy and promote_raw writes it again after step 5
    (a separate approval: run promote_raw once the second pass is done).
    Every write is guarded on the row's team ids too: a row re-pointed
    between plan and write stops the batch. Never a fold across team ids.
  fold: that row is a stale 'scheduled' fixture row without ESPN identity
    (pre-migration 023) at EXACTLY the copy's corrected kickoff, with no
    score, no pending/placed recommendation (on the row or on one of its
    predictions) and no prediction written after its kickoff. The copy's
    final score (and its half-time, when complete and within the final)
    goes onto it and it becomes 'finished', guarded on the values observed:
    the outcome trigger then grades its pre-kickoff predictions, identically
    under migration 011 and 027 because there is no post-kickoff row for 027
    to skip, and settles no recommendation because none is open. The audit
    CSV lists, per fold, the row's recommendations by status, those
    without a profit_loss (fold_unsettled_recs: grade_completed_matches.py
    --match-id would settle them by the final score; the PLAN line sums
    them; --apply refuses folds holding any unless --expect-plan pins
    fold_unsettled_recs=N to that sum, and the wrapper refuses unless 0)
    and its ungraded predictions; the run logs what the trigger graded.
Then, for all: the copy's odds move where the kept row has no price under
the same key (uq_odds_current_price_key), its match_stats where the kept
row has none for that team and that team plays in it; the copy row, the
kept row's before-state (for a fold also its predictions and
recommendations), and every row that references the copy, moved rows
included (every FK to matches, read from pg_constraint so a new table is
never missed), go to a write-once gzip JSONL archive before the first
write; and the copy is deleted, guarded:
never while a prediction, recommendation or betting_history row (or any FK
that does not cascade) references it. Refused and reported: a copy that is
not finished or has another score, a kept row that is ambiguous or wanted by
two copies, a stale row off the corrected kickoff or with open recs or
post-kickoff predictions (repair_stranded_recs / repair_fixture_orphans'
business), and two football-data rows.

--shift. Every remaining football-data row on a BST date moves back 1 h, to
the kickoff promote_raw now writes. Refused and reported: a row with ESPN
identity, a row whose score is no longer its raw row's, a row whose new
kickoff lands within 3 h of another row for the same team pair (a UNIQUE
(home_team_id, away_team_id, match_date) collision included), and an
'alias slot': a row whose new kickoff is the exact kickoff of the same
fixture under other team ids, i.e. a row in the same league at that kickoff
that shares a team with it (no club plays twice at once), or that has ESPN
identity, the same final score and, on both sides, a name that could spell
the same club (same_club_name; 'Inter Miami v Columbus Crew' 2-2 beside
'DC United v Nashville SC' 2-2 at 23:30Z is not one). Two football-data rows
sharing a team that would both land on one kickoff are refused the same
way. Shifting beside an alias pair would leave two rows of one match on one
slot, and whatever later unifies the team ids would collide on UNIQUE. A
copy --dedupe still plans to merge, alias-merge or fold is not shifted
(counted as left_to_dedupe; the PLAN line's dedupe_pending counts every
copy --dedupe still plans), and --shift --apply refuses to write anything
while dedupe_pending is not 0. A row the re-point map is about to move onto
the pair counts as a same-pair row here too. A shifted row that lands
within 10 min of an ESPN row for a DIFFERENT team pair in the same league,
with the same score, and is not refused as above, is a probable alias twin:
shifted, and listed in alias_twin_id for whoever unifies the teams.

--check (read-only). Runs promote_raw.check_legacy_kickoffs (every row at a
legacy kickoff, whether its corrected slot is free or not) and counts the
football-data rows that still have another FINISHED row for the pair where
promote_raw looks (a row the re-point map is about to move onto it
included); exits 1 unless both are clear. Also counts, for information,
football-data rows beside an unfinished row (stale fixtures), probable
alias twins, and the rows the re-point map has still to move
(repoint_pending).

Both writing modes plan first and write every planned row, refusals
included, to a write-once audit CSV, dry run included, and log one line
'PLAN mode=<mode> <action>=<n> ...' for the wrapper. --expect-plan pins the
plan an operator reviewed (dedupe: merge=,alias_merge=,fold=,refuse= and,
optionally, fold_unsettled_recs=; shift: shift=,refuse=): a plan that
differs writes nothing and exits 1. --apply needs --expect-plan naming every
one of its mode's keys (an old pin that leaves out alias_merge cannot let
alias merges through unpinned).
--apply writes the planned rows in bounded batches, one transaction each,
every UPDATE and DELETE guarded on the values the plan observed; refused
rows are left alone (exit 1 unless the refusals were pinned). A batch whose
guard misses a row rolls back and stops the run; earlier batches stay
committed and a re-run plans them as done.

Usage (inside the api container; scripts/ops/repair_football_data_kickoffs.sh
runs the whole sequence with a verified backup first):
    python /app/scripts/repair_football_data_kickoffs.py --dedupe            # dry run
    python /app/scripts/repair_football_data_kickoffs.py --dedupe --apply \\
        --expect-plan merge=68,alias_merge=4,fold=26,refuse=0,fold_unsettled_recs=0
    python /app/scripts/repair_football_data_kickoffs.py --shift             # dry run
    python /app/scripts/repair_football_data_kickoffs.py --shift --apply --expect-plan shift=63533,refuse=0
    python /app/scripts/repair_football_data_kickoffs.py --check
"""

from __future__ import annotations

import argparse
import bisect
import csv
import difflib
import gzip
import json
import logging
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import promote_raw as pr  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("repair_football_data_kickoffs")

# A shifted row may not land this close to another row for the same pair:
# promote_raw would then have to pick between them (find_existing_matches).
NEAR_WINDOW = pr.EXISTING_MATCH_WINDOW
# A different team pair in the same league with the same score, kicking off this close to a
# football-data row (or 1 h before it, at the legacy kickoff): a probable alias twin.
ALIAS_TOLERANCE = timedelta(minutes=10)
LEGACY_OFFSET = timedelta(hours=1)
# Words two spellings of one club may share without that saying anything (alias_slot_rows).
GENERIC_CLUB_WORDS = frozenset(
    "fc cf sc ac afc cd ca club de del la el fk if ik bk sk ff rj sp mg pr sv us as ss cfc united city sporting "
    "real athletic atletico deportivo football futbol calcio the and san santa sao central new".split()
)
# Tables whose rows keep a copy from being deleted, on top of every FK to matches that does not cascade.
GUARD_TABLES = frozenset({"predictions", "betting_recommendations", "betting_history"})
DEFAULT_BATCH_SIZE = {"dedupe": 100, "shift": 5000}
# The actions each mode plans; --expect-plan pins any of them (refuse included).
PLAN_KEYS = {"dedupe": ("merge", "alias_merge", "fold", "refuse"), "shift": ("shift", "refuse")}
# Optional --expect-plan keys beyond PLAN_KEYS: the open-rec sum of the folds (dedupe only).
EXTRA_PIN_KEYS = {"dedupe": ("fold_unsettled_recs",), "shift": ()}
# The dedupe actions that delete a copy into the row it keeps.
DEDUPE_WRITES = ("merge", "alias_merge", "fold")
# scripts/repoint_team_matches.py's reviewed map (one row per match side, each proven by its ESPN event): the
# rows ESPN filed under the wrong club, and the club each side moves to. --dedupe reads it to see where such a
# row is about to be (pending_repoints); absent, nothing is read from it.
REPOINT_MAP_PATH = Path(__file__).resolve().parent / "data" / "repoint_team_matches.csv"
REPOINT_MAP_FIELDS = ("match_id", "side", "from_team", "to_team")

RAW_SQL = (
    'SELECT league, season, "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG" '
    "FROM raw_football_data"
)
LEAGUES_SQL = (
    "SELECT id::text AS id, external_ids->>'football_data' AS code FROM leagues WHERE external_ids ? 'football_data'"
)
TEAMS_SQL = "SELECT id::text AS id, name, normalized_name FROM teams WHERE sport = 'soccer'"
MATCHES_SQL = """
    SELECT m.id::text AS id, m.league_id::text AS league_id,
           m.home_team_id::text AS home_id, m.away_team_id::text AS away_id,
           m.match_date, m.status, m.home_score, m.away_score, m.half_time_home, m.half_time_away,
           m.external_ids->>'espn' AS espn_id, m.metadata->>'result_source' AS result_source
      FROM matches m
      JOIN leagues l ON l.id = m.league_id
     WHERE l.sport = 'soccer'
"""
FK_SQL = """
    SELECT n.nspname AS schema_name, r.relname AS table_name, a.attname AS column_name,
           c.confdeltype AS on_delete, cardinality(c.conkey) AS n_columns
      FROM pg_constraint c
      JOIN pg_class r ON r.oid = c.conrelid
      JOIN pg_namespace n ON n.oid = r.relnamespace
      JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1]
     WHERE c.contype = 'f' AND c.confrelid = 'matches'::regclass
     ORDER BY r.relname, a.attname
"""
COUNT_REFS_SQL = "SELECT {col}::text AS match_id, count(*) AS n FROM {tbl} WHERE {col} = ANY(%s::uuid[]) GROUP BY 1"

# A stale row's recommendations: on the row itself, or on one of its predictions
# (the outcome trigger settles through the prediction).
_RECS_OF = "(br.match_id = m.id OR br.prediction_id IN (SELECT p.id FROM predictions p WHERE p.match_id = m.id))"
# What a fold depends on, per stale row: it folds only with open_recs = 0 and
# post_kickoff_predictions = 0 (planned, and re-checked under the batch lock).
FOLD_FACTS_SQL = f"""
    SELECT m.id::text AS id,
           (SELECT count(*) FROM betting_recommendations br
             WHERE br.status IN ('pending', 'placed') AND {_RECS_OF}) AS open_recs,
           (SELECT count(*) FROM predictions p WHERE p.match_id = m.id AND p.created_at > m.match_date)
               AS post_kickoff_predictions,
           (SELECT count(*) FROM predictions p WHERE p.match_id = m.id) AS predictions,
           (SELECT count(*) FROM predictions p WHERE p.match_id = m.id AND p.is_correct IS NULL)
               AS ungraded_predictions,
           (SELECT COALESCE(json_object_agg(x.status, x.n), '{{}}'::json)
              FROM (SELECT br.status, count(*) AS n FROM betting_recommendations br WHERE {_RECS_OF}
                     GROUP BY br.status) x) AS recs_by_status,
           (SELECT count(*) FROM betting_recommendations br WHERE br.profit_loss IS NULL AND {_RECS_OF})
               AS unsettled_recs
      FROM matches m
     WHERE m.id = ANY(%s::uuid[])
"""
# Graded predictions per type on the fold rows, before and after the flip.
GRADED_SQL = """
    SELECT prediction_type, count(*) FILTER (WHERE is_correct IS NOT NULL) AS graded
      FROM predictions
     WHERE match_id = ANY(%s::uuid[])
     GROUP BY prediction_type
"""

# What a row looked like when planned; every dedupe write is guarded on it (the team ids too: a row
# repoint_team_matches moved after the plan is no longer the row the plan saw).
OBSERVED = (
    "home_id",
    "away_id",
    "match_date",
    "status",
    "home_score",
    "away_score",
    "half_time_home",
    "half_time_away",
    "espn_id",
    "result_source",
)
LOCK_SQL = """
    SELECT m.id::text AS id, m.home_team_id::text AS home_id, m.away_team_id::text AS away_id,
           m.match_date, m.status, m.home_score, m.away_score, m.half_time_home,
           m.half_time_away, m.external_ids->>'espn' AS espn_id, m.metadata->>'result_source' AS result_source
      FROM matches m
     WHERE m.id = ANY(%s::uuid[])
     ORDER BY m.id
       FOR UPDATE
"""
ARCHIVE_MATCHES_SQL = (
    "SELECT m.id::text AS match_id, row_to_json(m) AS data FROM matches m WHERE m.id = ANY(%s::uuid[])"
)
ARCHIVE_REFS_SQL = "SELECT {col}::text AS match_id, row_to_json(x) AS data FROM {tbl} x WHERE {col} = ANY(%s::uuid[])"
# A fold row's predictions and recommendations before the outcome trigger grades them.
ARCHIVE_FOLD_PREDICTIONS_SQL = (
    "SELECT p.match_id::text AS match_id, row_to_json(p) AS data FROM predictions p WHERE p.match_id = ANY(%s::uuid[])"
)
ARCHIVE_FOLD_RECS_SQL = f"""
    SELECT m.id::text AS match_id, row_to_json(br) AS data
      FROM matches m JOIN betting_recommendations br ON {_RECS_OF}
     WHERE m.id = ANY(%s::uuid[])
"""

# The ESPN row has no row under the moving row's unique key (odds:
# uq_odds_current_price_key; match_stats: UNIQUE(match_id, team_id)).
ODDS_FREE_ON_TWIN = """NOT EXISTS (
        SELECT 1 FROM odds x
         WHERE x.match_id = v.twin_id::uuid AND x.bookmaker = o.bookmaker AND x.market_type = o.market_type
           AND x.selection = o.selection AND COALESCE(x.line, -1) = COALESCE(o.line, -1)
           AND COALESCE(x.is_opening, false) = COALESCE(o.is_opening, false)
           AND COALESCE(x.is_live, false) = COALESCE(o.is_live, false))"""
# ... and plays in it: an alias merge's kept row is still filed under the wrong club, so the copy's row for the
# club it is about to be re-pointed to stays with the copy (archived; promote_raw writes it again once the row
# is re-pointed) instead of landing on a match that club is not in, where repoint_team_matches would refuse it.
STATS_FREE_ON_TWIN = """NOT EXISTS (
        SELECT 1 FROM match_stats x WHERE x.match_id = v.twin_id::uuid AND x.team_id = s.team_id)
       AND EXISTS (
        SELECT 1 FROM matches k
         WHERE k.id = v.twin_id::uuid AND s.team_id IN (k.home_team_id, k.away_team_id))"""

MOVE_COUNTS_SQL = f"""
    SELECT v.copy_id,
           (SELECT count(*) FROM odds o WHERE o.match_id = v.copy_id::uuid) AS copy_odds,
           (SELECT count(*) FROM odds o WHERE o.match_id = v.copy_id::uuid AND {ODDS_FREE_ON_TWIN}) AS odds_to_move,
           (SELECT count(*) FROM match_stats s WHERE s.match_id = v.copy_id::uuid) AS copy_match_stats,
           (SELECT count(*) FROM match_stats s WHERE s.match_id = v.copy_id::uuid AND {STATS_FREE_ON_TWIN})
               AS stats_to_move
      FROM (VALUES %s) AS v(copy_id, twin_id)
"""
MOVE_ODDS_SQL = f"""
    UPDATE odds o SET match_id = v.twin_id::uuid
      FROM (VALUES %s) AS v(copy_id, twin_id)
     WHERE o.match_id = v.copy_id::uuid AND {ODDS_FREE_ON_TWIN}
    RETURNING o.id::text
"""
MOVE_STATS_SQL = f"""
    UPDATE match_stats s SET match_id = v.twin_id::uuid
      FROM (VALUES %s) AS v(copy_id, twin_id)
     WHERE s.match_id = v.copy_id::uuid AND {STATS_FREE_ON_TWIN}
    RETURNING s.id::text
"""
COPY_HT_SQL = """
    UPDATE matches
       SET half_time_home = %(ht_home)s, half_time_away = %(ht_away)s
     WHERE id = %(id)s::uuid
       AND status = 'finished' AND home_score = %(home)s AND away_score = %(away)s
       AND half_time_home IS NULL AND half_time_away IS NULL
"""
# A stale 'scheduled' fixture row takes the copy's result: guarded on every
# value observed and, in the same statement, on no open recommendation and no
# post-kickoff prediction, so the outcome trigger this fires grades only
# pre-kickoff predictions and settles nothing (the same under migrations 011
# and 027).
FOLD_SQL = f"""
    UPDATE matches m
       SET status = 'finished', home_score = %(home)s, away_score = %(away)s,
           half_time_home = %(ht_home)s, half_time_away = %(ht_away)s,
           metadata = COALESCE(m.metadata, '{{}}'::jsonb)
                      || jsonb_build_object('result_source', 'football_data', 'result_adopted_from', %(copy_id)s::text),
           updated_at = NOW()
     WHERE m.id = %(id)s::uuid
       AND m.status = 'scheduled' AND m.match_date = %(kickoff)s
       AND m.home_score IS NULL AND m.away_score IS NULL
       AND m.half_time_home IS NULL AND m.half_time_away IS NULL
       AND m.external_ids->>'espn' IS NULL
       AND m.metadata->>'result_source' IS DISTINCT FROM 'espn_scoreboard'
       AND NOT EXISTS (SELECT 1 FROM betting_recommendations br WHERE br.status IN ('pending', 'placed') AND {_RECS_OF})
       AND NOT EXISTS (SELECT 1 FROM predictions p WHERE p.match_id = m.id AND p.created_at > m.match_date)
    RETURNING m.id::text
"""
# {guards}: one NOT EXISTS per guard table (delete_sql).
DELETE_COPY_SQL = """
    DELETE FROM matches m
     USING (VALUES %s) AS v(id, match_date, status, home_score, away_score)
     WHERE m.id = v.id
       AND m.match_date = v.match_date AND m.status = v.status
       AND m.home_score = v.home_score AND m.away_score = v.away_score
       AND m.external_ids->>'espn' IS NULL
       AND m.metadata->>'result_source' IS DISTINCT FROM 'espn_scoreboard'
       {guards}
    RETURNING m.id::text
"""
DELETE_TEMPLATE = "(%s::uuid, %s::timestamptz, %s::text, %s::int, %s::int)"

_NEAR = f"interval '{int(NEAR_WINDOW.total_seconds())} seconds'"
SHIFT_SQL = f"""
    UPDATE matches AS m
       SET match_date = v.new_kickoff
      FROM (VALUES %s) AS v(id, old_kickoff, new_kickoff, status, home_score, away_score)
     WHERE m.id = v.id
       AND m.match_date = v.old_kickoff
       AND m.status = v.status
       AND m.home_score IS NOT DISTINCT FROM v.home_score
       AND m.away_score IS NOT DISTINCT FROM v.away_score
       AND m.external_ids->>'espn' IS NULL
       AND m.metadata->>'result_source' IS DISTINCT FROM 'espn_scoreboard'
       AND NOT EXISTS (
           SELECT 1 FROM matches x
            WHERE x.home_team_id = m.home_team_id AND x.away_team_id = m.away_team_id AND x.id <> m.id
              AND x.match_date BETWEEN v.new_kickoff - {_NEAR} AND v.new_kickoff + {_NEAR})
    RETURNING m.id::text
"""
SHIFT_TEMPLATE = "(%s::uuid, %s::timestamptz, %s::timestamptz, %s::text, %s::int, %s::int)"

DEDUPE_FIELDS = (
    "copy_id",
    "espn_row_id",
    "other_row_ids",
    "league",
    "home",
    "away",
    "copy_kickoff",
    "espn_kickoff",
    "offset_min",
    "bst",
    "home_score",
    "away_score",
    "copy_ht",
    "espn_ht",
    "ht_action",
    "copy_odds",
    "odds_to_move",
    "copy_match_stats",
    "stats_to_move",
    "copy_refs",
    "fold_open_recs",
    "fold_post_kickoff_predictions",
    "fold_predictions",
    "fold_ungraded_predictions",
    "fold_recs_by_status",
    "fold_unsettled_recs",
    "kept_row_repoint",
    "action",
    "reason",
    "batch",
)
SHIFT_FIELDS = (
    "match_id",
    "league",
    "home",
    "away",
    "raw_date",
    "raw_time",
    "old_kickoff",
    "new_kickoff",
    "status",
    "home_score",
    "away_score",
    "action",
    "reason",
    "near_row_id",
    "alias_slot_row_id",
    "alias_twin_id",
    "batch",
)


class PlanDrift(RuntimeError):
    """A batch's guard missed a row the plan observed: rolled back, run stopped."""


@dataclass
class FkTable:
    schema_name: str
    table_name: str
    column_name: str
    on_delete: str  # pg_constraint.confdeltype: c cascade, n set null, d set default, a no action, r restrict

    @property
    def ident(self) -> str:
        return f"{quote_ident(self.schema_name)}.{quote_ident(self.table_name)}"

    @property
    def column(self) -> str:
        return quote_ident(self.column_name)


def quote_ident(name: str) -> str:
    """A catalog name as a quoted SQL identifier (the standard quoting: no
    connection needed, so every statement can be built and tested offline)."""
    return '"' + name.replace('"', '""') + '"'


@dataclass
class Snapshot:
    """Everything the plans read, from one read snapshot. Each match dict
    gains 'kickoff' (aware UTC) and 'espn' (promote_raw.has_espn_identity).
    `pending` (set_pending) is {match_id: (home_id, away_id) once
    re-pointed} for the rows the reviewed re-point map is about to move;
    pair_rows() lists them under that pair too."""

    db_tz: tzinfo
    raw_rows: list[dict]
    league_ids: dict[str, str]  # football-data code -> leagues.id
    team_ids: dict[str, str]  # normalized_name -> teams.id
    team_names: dict[str, str]  # teams.id -> name
    matches: list[dict]
    by_key: dict[tuple, dict] = field(init=False)
    by_pair: dict[tuple, list[dict]] = field(init=False)
    pending: dict[str, tuple[str, str]] = field(init=False)
    by_post_pair: dict[tuple, list[dict]] = field(init=False)

    def __post_init__(self) -> None:
        self.by_key, self.by_pair = {}, defaultdict(list)
        for m in self.matches:
            m["kickoff"] = m["match_date"].astimezone(timezone.utc)
            m["espn"] = pr.has_espn_identity(m.get("espn_id"), m.get("result_source"))
            self.by_key[(m["home_id"], m["away_id"], m["kickoff"])] = m
            self.by_pair[(m["home_id"], m["away_id"])].append(m)
        self.set_pending({})

    def set_pending(self, pending: dict[str, tuple[str, str]]) -> None:
        self.pending, self.by_post_pair = dict(pending), defaultdict(list)
        for m in self.matches:
            if m["id"] in self.pending:
                self.by_post_pair[self.pending[m["id"]]].append(m)

    def pair_rows(self, pair: tuple) -> list[dict]:
        """The rows for a team pair: on it now, or about to be (pending)."""
        return self.by_pair.get(pair, []) + self.by_post_pair.get(pair, [])


@dataclass
class Link:
    """A matches row and the raw_football_data row(s) promote_raw wrote it from."""

    match: dict
    raw: dict
    legacy: datetime  # where promote_raw wrote it until 2026-10-31 (aware UTC)
    corrected: datetime  # where promote_raw writes it now (aware UTC)
    fixture_day: date  # football-data's Date (UK)
    raw_scores: set[tuple[int, int]]

    @property
    def at_legacy(self) -> bool:
        return self.match["kickoff"] == self.legacy

    @property
    def bst(self) -> bool:
        return self.legacy != self.corrected


def _iso(dt: Optional[datetime]) -> str:
    return dt.astimezone(timezone.utc).isoformat() if dt else ""


def observed(row: dict) -> tuple:
    return tuple(row.get(k) for k in OBSERVED)


def load_snapshot(cur) -> Snapshot:
    cur.execute("SHOW TimeZone")
    db_tz = ZoneInfo(cur.fetchone()["TimeZone"])  # loud if Postgres reports a zone Python cannot load
    cur.execute(RAW_SQL)
    raw_rows = [dict(r) for r in cur.fetchall()]
    cur.execute(LEAGUES_SQL)
    league_ids = {r["code"]: r["id"] for r in cur.fetchall()}
    cur.execute(TEAMS_SQL)
    teams = cur.fetchall()
    cur.execute(MATCHES_SQL)
    matches = [dict(r) for r in cur.fetchall()]
    return Snapshot(
        db_tz=db_tz,
        raw_rows=raw_rows,
        league_ids=league_ids,
        team_ids={t["normalized_name"]: t["id"] for t in teams},
        team_names={t["id"]: t["name"] for t in teams},
        matches=matches,
    )


def link_rows(snap: Snapshot, counts: Counter) -> dict[str, Link]:
    """{match_id: Link} for every timed raw row with a result whose matches
    row sits at its legacy kickoff or, once moved, at its corrected one. At
    the corrected kickoff only a row with the raw row's score counts: once a
    copy is merged away, the row there is the ESPN row it went into, or a
    stale fixture row, not football-data's. Pure."""
    links: dict[str, Link] = {}
    for r in snap.raw_rows:
        result = pr.raw_result(r)
        if result is None:
            counts["raw_no_result"] += 1  # never promoted
            continue
        league_id = snap.league_ids.get(r.get("league") or "")
        home_id = snap.team_ids.get(pr.normalize_team_name(r.get("HomeTeam") or ""))
        away_id = snap.team_ids.get(pr.normalize_team_name(r.get("AwayTeam") or ""))
        wall = pr.parse_wall_clock(r.get("Date") or "", r.get("Time"))
        if not (league_id and home_id and away_id and wall):
            counts["raw_unresolved"] += 1
            continue
        corrected = pr.parse_match_date(r.get("Date") or "", r.get("Time"))
        if corrected is None or corrected.tzinfo is None:
            counts["raw_date_only"] += 1  # its kickoff never changed
            continue
        legacy = wall.replace(tzinfo=snap.db_tz).astimezone(timezone.utc)
        score = (result.home_score, result.away_score)
        m = snap.by_key.get((home_id, away_id, legacy))
        if m is None and legacy != corrected:
            m = snap.by_key.get((home_id, away_id, corrected))
            if m is not None and (m["home_score"], m["away_score"]) != score:
                counts["raw_corrected_slot_other_row"] += 1
                continue
        if m is None:
            counts["raw_no_matches_row"] += 1
            continue
        if m["league_id"] != league_id:
            counts["raw_league_mismatch"] += 1
            continue
        if m["id"] in links:
            links[m["id"]].raw_scores.add(score)
            counts["raw_duplicate_rows"] += 1
            continue
        links[m["id"]] = Link(m, r, legacy, corrected, wall.date(), {score})
    counts["linked_rows"] = len(links)
    return links


def load_repoint_map(path: Path | str) -> dict[str, dict[str, tuple[str, str]]]:
    """scripts/repoint_team_matches.py's reviewed map as {match_id: {side:
    (from_team, to_team)}}, names normalised (promote_raw.normalize_team_name).
    Raises ValueError on a malformed map: a missing column, a side that is not
    home/away, an empty name, a side listed twice."""
    path = Path(path)
    out: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [f for f in REPOINT_MAP_FIELDS if f not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        for line, raw in enumerate(reader, start=2):
            match_id, side, frm, to = (str(raw.get(f) or "").strip() for f in REPOINT_MAP_FIELDS)
            if side not in ("home", "away") or not (match_id and frm and to):
                raise ValueError(f"{path}:{line}: want match_id, side home/away, from_team and to_team")
            if side in out[match_id]:
                raise ValueError(f"{path}:{line}: {match_id} {side} is listed twice")
            out[match_id][side] = (pr.normalize_team_name(frm), pr.normalize_team_name(to))
    return dict(out)


def pending_repoints(
    snap: Snapshot, repoint_map: dict[str, dict[str, tuple[str, str]]], counts: Counter
) -> dict[str, tuple[str, str]]:
    """{match_id: (home_id, away_id) once re-pointed} for each mapped row
    whose every mapped side is still on its from_team: where
    repoint_team_matches will put it. A row it already moved is 'done'; a
    row half moved, on another team, or naming a team that does not exist is
    counted and left out (nothing is planned from it). Pure."""
    by_id = {m["id"]: m for m in snap.matches}
    pending: dict[str, tuple[str, str]] = {}
    for match_id, sides in sorted(repoint_map.items()):
        m = by_id.get(match_id)
        if m is None:
            counts["repoint_row_absent"] += 1
            continue
        post = {"home": m["home_id"], "away": m["away_id"]}
        states = set()
        for side, (frm, to) in sides.items():
            to_id = snap.team_ids.get(to)
            if to_id is None:
                states.add("unresolved")
            elif m[f"{side}_id"] == to_id:
                states.add("done")
            elif pr.normalize_team_name(snap.team_names.get(m[f"{side}_id"], "")) == frm:
                states.add("pending")
                post[side] = to_id
            else:
                states.add("drifted")
        if states == {"done"}:
            counts["repoint_done"] += 1
        elif states == {"pending"} and post["home"] != post["away"]:
            pending[match_id] = (post["home"], post["away"])
        else:
            counts["repoint_unclear"] += 1
            logger.warning("re-point map row %s is %s: nothing planned from it", match_id, "/".join(sorted(states)))
    counts["repoint_pending"] = len(pending)
    return pending


def window_rows(snap: Snapshot, link: Link) -> list[dict]:
    """The other rows for this pair where promote_raw now looks for the
    fixture: promote_raw.existing_match_window around the corrected kickoff,
    a row the reviewed re-point map is about to move onto the pair included
    (Snapshot.pair_rows). Pure."""
    m = link.match
    lo, hi = pr.existing_match_window(link.corrected, link.fixture_day)
    return [x for x in snap.pair_rows((m["home_id"], m["away_id"])) if x is not m and lo <= x["kickoff"] <= hi]


def _on_other_pair(copy: dict, row: dict) -> bool:
    return (row["home_id"], row["away_id"]) != (copy["home_id"], copy["away_id"])


def _base_entry(snap: Snapshot, link: Link) -> dict:
    m = link.match
    return {
        "league": link.raw.get("league"),
        "home": snap.team_names.get(m["home_id"], m["home_id"]),
        "away": snap.team_names.get(m["away_id"], m["away_id"]),
        "home_score": m["home_score"],
        "away_score": m["away_score"],
    }


# --- dedupe -------------------------------------------------------------------------------------------------------


def dedupe_candidates(snap: Snapshot, links: dict[str, Link], counts: Counter) -> list[tuple[Link, list[dict]]]:
    """(link, nearby rows) for each linked football-data row without ESPN
    identity that has another row for the pair in window_rows. Pure."""
    out = []
    for link in links.values():
        if link.match["espn"]:
            counts["linked_row_is_espn"] += 1  # the upsert landed on the ESPN row: not a copy
            continue
        nearby = window_rows(snap, link)
        if nearby:
            out.append((link, nearby))
    return out


def ht_plan(copy: dict, twin: dict) -> tuple[str, Optional[tuple[int, int]]]:
    """(ht_action, the pair to write onto the ESPN row or None). Pure."""
    copy_ht = (copy["half_time_home"], copy["half_time_away"])
    twin_ht = (twin["half_time_home"], twin["half_time_away"])
    if copy_ht == (None, None):
        return "none_on_copy", None
    if twin_ht != (None, None):
        return ("same" if twin_ht == copy_ht else "conflict_kept_espn"), None
    if None in copy_ht:
        return "incomplete_on_copy", None
    if copy_ht[0] > twin["home_score"] or copy_ht[1] > twin["away_score"]:
        return "exceeds_espn_final", None
    return "copy", copy_ht  # type: ignore[return-value]


def fold_refusal(link: Link, row: dict, facts: Optional[dict]) -> str:
    """Why a copy may not be folded into `row`, the unfinished row without
    ESPN identity promote_raw would pick for it, or '' when it may (the
    copy's own checks are plan_dedupe's). Pure."""
    if row["status"] != "scheduled":
        return (
            f"the row for this pair nearby is a '{row['status']}' row without ESPN identity, not a stale "
            "'scheduled' fixture row: inspect"
        )
    if row["kickoff"] != link.corrected:
        return (
            f"the stale 'scheduled' row for this pair sits at {_iso(row['kickoff'])}, not on the corrected kickoff "
            f"{_iso(link.corrected)} (a stale fixture row: repair_fixture_orphans' business)"
        )
    if any(row.get(k) is not None for k in ("home_score", "away_score", "half_time_home", "half_time_away")):
        return "the stale 'scheduled' row carries a score: inspect"
    if facts is None:
        return "no fold facts were read for the stale row"
    if facts["open_recs"]:
        return (
            f"the stale row holds {facts['open_recs']} pending/placed recommendation(s) (on it or its predictions): "
            "settle them first (repair_stranded_recs)"
        )
    if facts["post_kickoff_predictions"]:
        return (
            f"the stale row has {facts['post_kickoff_predictions']} prediction(s) written after its kickoff: the "
            "outcome trigger would grade them under migration 011 but not under 027 — inspect"
        )
    return ""


def plan_dedupe(
    snap: Snapshot,
    candidates: list[tuple[Link, list[dict]]],
    refs: dict[str, dict[str, int]],
    guard_tables: Iterable[str],
    counts: Counter,
    fold_facts: Optional[dict[str, dict]] = None,
) -> list[dict]:
    """One entry per candidate copy: action 'merge' (into the ESPN row
    promote_raw.choose_existing picks among the nearby rows), 'alias_merge'
    (the same, into an ESPN row still filed under the wrong club that the
    reviewed re-point map is about to move onto the copy's pair: Snapshot.
    pending), 'fold' (into the stale 'scheduled' row it picks, see
    fold_refusal) or 'refuse'.
    `refs` is {copy_id: {table: rows referencing it}}, `fold_facts`
    {row_id: FOLD_FACTS_SQL row} for the unfinished rows without ESPN
    identity. Pure."""
    guards = set(guard_tables)
    facts_by_row = fold_facts or {}
    chosen = [pr.choose_existing(nearby) for _link, nearby in candidates]
    twin_uses = Counter(row["id"] for row, _why in chosen if row is not None)
    entries = []
    for (link, nearby), (twin, ambiguity) in zip(candidates, chosen):
        copy = link.match
        shown = [twin] if twin is not None else nearby
        copy_refs = {t: n for t, n in sorted(refs.get(copy["id"], {}).items()) if n}
        facts = facts_by_row.get(twin["id"]) if twin is not None and not twin["espn"] else None
        entry = {
            **_base_entry(snap, link),
            "copy_id": copy["id"],
            "espn_row_id": ";".join(x["id"] for x in shown),
            "other_row_ids": ";".join(x["id"] for x in nearby if x is not twin) if twin is not None else "",
            "copy_kickoff": _iso(copy["kickoff"]),
            "espn_kickoff": ";".join(_iso(x["kickoff"]) for x in shown),
            "offset_min": ";".join(str(int((copy["kickoff"] - x["kickoff"]).total_seconds() // 60)) for x in shown),
            "bst": link.bst,
            "copy_ht": f"{copy['half_time_home']}-{copy['half_time_away']}",
            "espn_ht": ";".join(f"{x['half_time_home']}-{x['half_time_away']}" for x in shown),
            "copy_refs": json.dumps(copy_refs, sort_keys=True),
            "copy": copy,
            "twin": twin,
            "action": "refuse",
            "reason": "",
            "ht_action": "",
            "ht_write": None,
        }
        if facts is not None:
            entry.update(
                fold_open_recs=facts["open_recs"],
                fold_post_kickoff_predictions=facts["post_kickoff_predictions"],
                fold_predictions=facts["predictions"],
                fold_ungraded_predictions=facts["ungraded_predictions"],
                fold_recs_by_status=json.dumps(facts["recs_by_status"] or {}, sort_keys=True),
                fold_unsettled_recs=facts.get("unsettled_recs", 0),
            )
        if twin is not None and _on_other_pair(copy, twin):
            entry["kept_row_repoint"] = "; ".join(
                f"{side}: {snap.team_names.get(now, now)} -> {snap.team_names.get(then, then)}"
                for side, now, then in zip(
                    ("home", "away"), (twin["home_id"], twin["away_id"]), snap.pending.get(twin["id"], ("", ""))
                )
                if now != then
            )
        blocking = sorted(t for t in copy_refs if t in guards)
        copy_score = (copy["home_score"], copy["away_score"])
        if twin is None:
            entry["reason"] = f"ambiguous: {ambiguity}"
        elif twin_uses[twin["id"]] > 1:
            entry["reason"] = f"the kept row is the nearby row of {twin_uses[twin['id']]} football-data rows"
        elif copy["status"] != "finished" or None in copy_score:
            entry["reason"] = f"the copy is not a finished result ({copy['status']} {copy_score[0]}-{copy_score[1]})"
        elif blocking:
            entry["reason"] = "copy referenced by " + ", ".join(f"{t} ({copy_refs[t]})" for t in blocking)
        elif _on_other_pair(copy, twin) and not twin["espn"]:
            entry["reason"] = (
                f"the row for this fixture is a '{twin['status']}' row without ESPN identity filed under other team "
                f"ids ({entry['kept_row_repoint']}, per the reviewed re-point map): re-point it first "
                "(scripts/repoint_team_matches.py), then re-run"
            )
        elif twin["espn"]:
            if twin["status"] != "finished":
                entry["reason"] = f"not both finished (copy {copy['status']}, ESPN row {twin['status']})"
            elif copy_score != (twin["home_score"], twin["away_score"]):
                entry["reason"] = (
                    f"scores differ (copy {copy_score[0]}-{copy_score[1]}, "
                    f"ESPN row {twin['home_score']}-{twin['away_score']})"
                )
            else:
                entry["action"] = "alias_merge" if _on_other_pair(copy, twin) else "merge"
                entry["ht_action"], entry["ht_write"] = ht_plan(copy, twin)
        elif twin["status"] == "finished":
            entry["reason"] = (
                "the row for this pair nearby is a finished row without ESPN identity (two football-data rows? inspect)"
            )
        elif copy_score not in link.raw_scores:
            # the fold's result comes from the copy alone: it must still be football-data's
            entry["reason"] = (
                f"the copy's score {copy_score[0]}-{copy_score[1]} is not its raw row's {sorted(link.raw_scores)}: "
                "edited since, not folded"
            )
        else:
            why = fold_refusal(link, twin, facts)
            if why:
                entry["reason"] = why
            else:
                entry["action"] = "fold"
                # the stale row has no score: the copy's half-time goes on whole and within its own final
                entry["ht_action"], entry["ht_write"] = ht_plan(
                    copy, {**twin, "home_score": copy["home_score"], "away_score": copy["away_score"]}
                )
        counts[f"dedupe_{entry['action']}"] += 1
        entries.append(entry)
    return entries


def load_fold_facts(cur, row_ids: list[str]) -> dict[str, dict]:
    """FOLD_FACTS_SQL for each row id."""
    if not row_ids:
        return {}
    cur.execute(FOLD_FACTS_SQL, (row_ids,))
    return {r["id"]: dict(r) for r in cur.fetchall()}


def fold_fact_rows(candidates: list[tuple[Link, list[dict]]]) -> list[str]:
    """The unfinished rows without ESPN identity promote_raw would pick for a
    candidate copy: the rows a fold needs facts for. Pure."""
    rows = set()
    for _link, nearby in candidates:
        row, _why = pr.choose_existing(nearby)
        if row is not None and not row["espn"] and row["status"] != "finished":
            rows.add(row["id"])
    return sorted(rows)


def load_fk_tables(cur) -> list[FkTable]:
    cur.execute(FK_SQL)
    rows = cur.fetchall()
    multi = [r for r in rows if r["n_columns"] != 1]
    if multi:
        raise RuntimeError(f"multi-column foreign keys to matches are not handled: {multi}")
    return [FkTable(r["schema_name"], r["table_name"], r["column_name"], r["on_delete"]) for r in rows]


def guard_table_names(fk_tables: list[FkTable]) -> set[str]:
    """Tables whose rows keep a copy alive: GUARD_TABLES, and any FK whose
    delete would not take the row with the copy."""
    return {t.table_name for t in fk_tables if t.table_name in GUARD_TABLES or t.on_delete != "c"}


def load_refs(cur, fk_tables: list[FkTable], match_ids: list[str]) -> dict[str, dict[str, int]]:
    refs: dict[str, dict[str, int]] = defaultdict(dict)
    if not match_ids:
        return refs
    for t in fk_tables:
        cur.execute(
            COUNT_REFS_SQL.format(col=t.column, tbl=t.ident),
            (match_ids,),
        )
        for r in cur.fetchall():
            refs[r["match_id"]][t.table_name] = refs[r["match_id"]].get(t.table_name, 0) + int(r["n"])
    return refs


def add_move_counts(cur, entries: list[dict]) -> None:
    merges = [e for e in entries if e["action"] in DEDUPE_WRITES]
    if not merges:
        return
    rows = execute_values(
        cur, MOVE_COUNTS_SQL, [(e["copy_id"], e["twin"]["id"]) for e in merges], page_size=len(merges), fetch=True
    )
    by_copy = {r["copy_id"]: r for r in rows}
    for e in merges:
        for k in ("copy_odds", "odds_to_move", "copy_match_stats", "stats_to_move"):
            e[k] = int(by_copy[e["copy_id"]][k])


def delete_sql(guard_tables: list[FkTable]) -> str:
    """DELETE_COPY_SQL with one NOT EXISTS per guard table."""
    guards = "\n       ".join(
        f"AND NOT EXISTS (SELECT 1 FROM {t.ident} g WHERE g.{t.column} = m.id)" for t in guard_tables
    )
    return DELETE_COPY_SQL.format(guards=guards)


def write_archive(path: Path, records: list[dict]) -> None:
    """gzip JSONL, write-once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "xt", encoding="utf-8") as fh:  # 'x' refuses an existing file
        for rec in records:
            fh.write(json.dumps(rec, default=str, sort_keys=True) + "\n")


def _graded(cur, row_ids: list[str]) -> Counter:
    """Graded predictions per type on these rows."""
    if not row_ids:
        return Counter()
    cur.execute(GRADED_SQL, (row_ids,))
    return Counter({r["prediction_type"]: int(r["graded"]) for r in cur.fetchall()})


def apply_dedupe_batch(conn, batch: list[dict], batch_no: int, fk_tables: list[FkTable], archive_path: Path) -> Counter:
    """Merge and fold one batch in one transaction: lock and re-check every
    row (a fold row's open recs and post-kickoff predictions too), archive,
    then write. Raises PlanDrift (rolled back) when a guard misses."""
    done: Counter = Counter()
    copy_ids = [e["copy_id"] for e in batch]
    merge_ids = [e["twin"]["id"] for e in batch if e["action"] in ("merge", "alias_merge")]
    folds = [e for e in batch if e["action"] == "fold"]
    fold_ids = [e["twin"]["id"] for e in folds]
    pairs = [(e["copy_id"], e["twin"]["id"]) for e in batch]
    guard_names = guard_table_names(fk_tables)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(LOCK_SQL, (copy_ids + merge_ids + fold_ids,))
            now = {r["id"]: r for r in cur.fetchall()}
            for e in batch:
                for role in ("copy", "twin"):
                    planned = e[role]
                    if planned["id"] not in now or observed(now[planned["id"]]) != observed(planned):
                        raise PlanDrift(f"batch {batch_no}: {role} {planned['id']} changed since the plan")
            # The lock blocks new rows referencing the fold rows (their FK takes KEY SHARE on them); a
            # rec on one of their predictions is caught by FOLD_SQL itself.
            for row_id, facts in load_fold_facts(cur, fold_ids).items():
                if facts["open_recs"] or facts["post_kickoff_predictions"]:
                    raise PlanDrift(
                        f"batch {batch_no}: fold row {row_id} gained {facts['open_recs']} open rec(s) / "
                        f"{facts['post_kickoff_predictions']} post-kickoff prediction(s) since the plan"
                    )

            records = []
            cur.execute(ARCHIVE_MATCHES_SQL, (copy_ids,))
            records += [{"table": "matches", "role": "copy", **r} for r in cur.fetchall()]
            if merge_ids:
                cur.execute(ARCHIVE_MATCHES_SQL, (merge_ids,))
                records += [{"table": "matches", "role": "espn_row_before", **r} for r in cur.fetchall()]
            if fold_ids:
                cur.execute(ARCHIVE_MATCHES_SQL, (fold_ids,))
                records += [{"table": "matches", "role": "fold_row_before", **r} for r in cur.fetchall()]
                cur.execute(ARCHIVE_FOLD_PREDICTIONS_SQL, (fold_ids,))
                records += [{"table": "predictions", "role": "fold_row_before", **r} for r in cur.fetchall()]
                cur.execute(ARCHIVE_FOLD_RECS_SQL, (fold_ids,))
                records += [
                    {"table": "betting_recommendations", "role": "fold_row_before", **r} for r in cur.fetchall()
                ]
            for t in fk_tables:
                cur.execute(
                    ARCHIVE_REFS_SQL.format(col=t.column, tbl=t.ident),
                    (copy_ids,),
                )
                records += [{"table": t.table_name, "role": "references_copy", **r} for r in cur.fetchall()]
            for rec in records:
                rec["batch"] = batch_no
            write_archive(archive_path, records)
            done["archived_records"] += len(records)

            for e in batch:
                if e["action"] not in ("merge", "alias_merge") or e["ht_write"] is None:
                    continue
                twin = e["twin"]
                cur.execute(
                    COPY_HT_SQL,
                    {
                        "id": twin["id"],
                        "ht_home": e["ht_write"][0],
                        "ht_away": e["ht_write"][1],
                        "home": twin["home_score"],
                        "away": twin["away_score"],
                    },
                )
                if cur.rowcount != 1:
                    raise PlanDrift(f"batch {batch_no}: half-time on ESPN row {twin['id']} changed since the plan")
                done["ht_copied"] += 1
            done["odds_moved"] += len(execute_values(cur, MOVE_ODDS_SQL, pairs, page_size=len(pairs), fetch=True))
            done["match_stats_moved"] += len(
                execute_values(cur, MOVE_STATS_SQL, pairs, page_size=len(pairs), fetch=True)
            )

            graded_before = _graded(cur, fold_ids)
            for e in folds:
                ht = e["ht_write"] or (None, None)
                cur.execute(
                    FOLD_SQL,
                    {
                        "id": e["twin"]["id"],
                        "kickoff": e["twin"]["match_date"],
                        "home": e["home_score"],
                        "away": e["away_score"],
                        "ht_home": ht[0],
                        "ht_away": ht[1],
                        "copy_id": e["copy_id"],
                    },
                )
                if cur.rowcount != 1:
                    raise PlanDrift(
                        f"batch {batch_no}: fold row {e['twin']['id']} changed since the plan (or gained an open "
                        "rec / a post-kickoff prediction)"
                    )
                done["folded"] += 1
                done["fold_ht_copied"] += int(e["ht_write"] is not None)
            graded = _graded(cur, fold_ids) - graded_before
            for ptype, n in graded.items():
                done[f"trigger_graded:{ptype}"] += n
            done["trigger_graded_predictions"] += sum(graded.values())

            deleted = execute_values(
                cur,
                delete_sql([t for t in fk_tables if t.table_name in guard_names]),
                [
                    (e["copy_id"], e["copy"]["match_date"], e["copy"]["status"], e["home_score"], e["away_score"])
                    for e in batch
                ],
                template=DELETE_TEMPLATE,
                page_size=len(batch),
                fetch=True,
            )
            if len(deleted) != len(batch):
                missing = sorted(set(copy_ids) - {r["id"] for r in deleted})
                raise PlanDrift(f"batch {batch_no}: {len(missing)} copy row(s) not deleted by the guard: {missing[:5]}")
            done["copies_deleted"] += len(deleted)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return done


# --- shift --------------------------------------------------------------------------------------------------------


class _LeagueKickoffs:
    """Rows by league, sorted by kickoff, for the alias-twin lookup."""

    def __init__(self, matches: list[dict]):
        by_league: dict[str, list[dict]] = defaultdict(list)
        for m in matches:
            by_league[m["league_id"]].append(m)
        self._rows = {lg: sorted(rows, key=lambda m: m["kickoff"]) for lg, rows in by_league.items()}
        self._keys = {lg: [m["kickoff"] for m in rows] for lg, rows in self._rows.items()}

    def near(self, league_id: str, when: datetime, tolerance: timedelta) -> list[dict]:
        keys, rows = self._keys.get(league_id, []), self._rows.get(league_id, [])
        return rows[bisect.bisect_left(keys, when - tolerance) : bisect.bisect_right(keys, when + tolerance)]


def alias_twins(index: _LeagueKickoffs, row: dict, when: datetime) -> list[dict]:
    """ESPN rows in `row`'s league within ALIAS_TOLERANCE of `when`, same
    final score, another team pair: one fixture under two team rows. Pure."""
    if row["home_score"] is None:
        return []
    return [
        m
        for m in index.near(row["league_id"], when, ALIAS_TOLERANCE)
        if m["espn"]
        and (m["home_id"], m["away_id"]) != (row["home_id"], row["away_id"])
        and (m["home_score"], m["away_score"]) == (row["home_score"], row["away_score"])
    ]


def _fold_name(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower())


def same_club_name(a: str, b: str) -> bool:
    """Whether two team names could spell one club: one inside the other
    once accents, case and punctuation go ('Hacken' / 'BK Häcken'), a
    distinctive word one starts the other's with ('Inter Miami' / 'Inter
    Miami CF', 'Man City' / 'Manchester City'), or a close spelling. Loose
    on purpose: alias_slot_rows needs it on BOTH sides of a fixture, at the
    same kickoff, in the same league, with the same score. Pure."""
    fa, fb = _fold_name(a), _fold_name(b)
    ca, cb = fa.replace(" ", ""), fb.replace(" ", "")
    if not ca or not cb:
        return False
    if ca in cb or cb in ca:
        return True
    words_a = {w for w in fa.split() if len(w) >= 3 and w not in GENERIC_CLUB_WORDS}
    words_b = {w for w in fb.split() if len(w) >= 3 and w not in GENERIC_CLUB_WORDS}
    if any(x.startswith(y) or y.startswith(x) for x in words_a for y in words_b):  # 'Man' / 'Manchester'
        return True
    return difflib.SequenceMatcher(None, ca, cb).ratio() >= 0.75


def alias_slot_rows(
    index: _LeagueKickoffs, row: dict, when: datetime, planned_away: frozenset[str], names: dict[str, str]
) -> list[dict]:
    """Rows in `row`'s league at exactly `when` under another team pair that
    are the same fixture: one shares a team with it (no club plays twice at
    once), or one with ESPN identity, the same final score and, on both
    sides, a name that could spell the same club (same_club_name; `names`
    is teams.id -> name). Shifting `row` onto `when` would leave one match on
    one slot twice, under two team ids, and the repair that later unifies
    those ids (the team merge, or a re-point of a row filed under the wrong
    club) would collide on UNIQUE(home_team_id, away_team_id, match_date);
    --shift leaves such a row for it. A row --dedupe still plans to merge or
    fold away (`planned_away`) does not count: it is gone before the shift
    writes. Pure."""
    teams = {row["home_id"], row["away_id"]}
    score = (row["home_score"], row["away_score"])

    def spells(a: str, b: str) -> bool:
        return same_club_name(names.get(a, a), names.get(b, b))

    return [
        m
        for m in index.near(row["league_id"], when, timedelta(0))
        if m is not row
        and m["id"] not in planned_away
        and (m["home_id"], m["away_id"]) != (row["home_id"], row["away_id"])
        and (
            {m["home_id"], m["away_id"]} & teams
            or (
                m["espn"]
                and (m["home_score"], m["away_score"]) == score
                and spells(m["home_id"], row["home_id"])
                and spells(m["away_id"], row["away_id"])
            )
        )
    ]


def plan_shift(
    snap: Snapshot, links: dict[str, Link], counts: Counter, planned_away: frozenset[str] = frozenset()
) -> list[dict]:
    """One entry per linked BST row still at its legacy kickoff: action
    'shift' or 'refuse'. A copy in `planned_away` (--dedupe still plans to
    merge or fold it) gets no entry: it is counted as left_to_dedupe. Pure."""
    index = _LeagueKickoffs(snap.matches)
    entries = []
    for link in links.values():
        m = link.match
        if not link.bst:
            counts["gmt_kickoff_right"] += 1
            continue
        if not link.at_legacy:
            # a moved football-data row, or the row a copy was merged or folded into
            counts["corrected_slot_espn_row" if m["espn"] else "already_shifted"] += 1
            continue
        if m["id"] in planned_away:
            counts["left_to_dedupe"] += 1
            continue
        target = link.corrected
        near = [
            x
            for x in snap.pair_rows((m["home_id"], m["away_id"]))
            if x is not m and abs(x["kickoff"] - target) <= NEAR_WINDOW
        ]
        slot = alias_slot_rows(index, m, target, planned_away, snap.team_names)
        entry = {
            **_base_entry(snap, link),
            "match_id": m["id"],
            "raw_date": link.raw.get("Date"),
            "raw_time": link.raw.get("Time"),
            "old_kickoff": _iso(m["kickoff"]),
            "new_kickoff": _iso(target),
            "status": m["status"],
            "near_row_id": ";".join(x["id"] for x in near),
            "alias_slot_row_id": ";".join(x["id"] for x in slot),
            "alias_twin_id": "",
            "old": m["kickoff"],
            "new": target,
            "action": "refuse",
            "reason": "",
        }
        if m["espn"]:
            entry["reason"] = "carries ESPN identity: its kickoff is ESPN's"
        elif (m["home_score"], m["away_score"]) not in link.raw_scores:
            entry["reason"] = (
                f"score {m['home_score']}-{m['away_score']} is not its raw row's {sorted(link.raw_scores)}: "
                "edited since"
            )
        elif near:
            exact = any(x["kickoff"] == target for x in near)
            entry["reason"] = (
                f"{len(near)} row(s) for this team pair within {NEAR_WINDOW} of the new kickoff"
                + (" (one on it: a UNIQUE collision)" if exact else "")
                + f": {', '.join(x['status'] for x in near)} — --dedupe first, or inspect"
            )
        elif slot:
            names = ", ".join(
                f"{x['id']} ({snap.team_names.get(x['home_id'], x['home_id'])} v "
                f"{snap.team_names.get(x['away_id'], x['away_id'])}, {x['status']} {x['home_score']}-{x['away_score']})"
                for x in slot
            )
            entry["reason"] = (
                f"alias slot: the new kickoff is the exact kickoff of the same fixture under other team ids ({names}) "
                "— unify the teams first (scripts/ops/merge_soccer_teams.sh, or re-point a row filed under the wrong "
                "club), then re-run"
            )
        else:
            entry["action"] = "shift"
            entry["alias_twin_id"] = ";".join(t["id"] for t in alias_twins(index, m, target))
        entries.append(entry)
    # Two shifts onto one kickoff in one league that share a team: the same fixture twice in the raw table under
    # two spellings of the other club. Neither moves.
    by_slot: dict[tuple, list[dict]] = defaultdict(list)
    for e in entries:
        if e["action"] == "shift":
            by_slot[(links[e["match_id"]].match["league_id"], e["new"])].append(e)
    for group in by_slot.values():
        for e in group:
            row = links[e["match_id"]].match
            mates = [
                o
                for o in group
                if o is not e
                and {links[o["match_id"]].match["home_id"], links[o["match_id"]].match["away_id"]}
                & {row["home_id"], row["away_id"]}
            ]
            if mates:
                e["action"] = "refuse"
                e["alias_slot_row_id"] = ";".join(o["match_id"] for o in mates)
                e["reason"] = (
                    "alias slot: another football-data row sharing a team would shift onto the same kickoff "
                    f"({e['alias_slot_row_id']}) — the same fixture twice in raw_football_data? inspect"
                )
    for e in entries:
        counts[f"shift_{e['action']}"] += 1
        if e["action"] == "shift" and e["alias_twin_id"]:
            counts["shift_alias_twins"] += 1
    return entries


def apply_shift_batch(conn, batch: list[dict], batch_no: int) -> int:
    """Shift one batch in one transaction; raises PlanDrift (rolled back) when
    a guard misses."""
    try:
        with conn.cursor() as cur:
            moved = execute_values(
                cur,
                SHIFT_SQL,
                [(e["match_id"], e["old"], e["new"], e["status"], e["home_score"], e["away_score"]) for e in batch],
                template=SHIFT_TEMPLATE,
                page_size=len(batch),
                fetch=True,
            )
            if len(moved) != len(batch):
                missing = sorted({e["match_id"] for e in batch} - {r[0] for r in moved})
                raise PlanDrift(
                    f"batch {batch_no}: {len(missing)} of {len(batch)} row(s) changed since the plan or gained a "
                    f"same-pair neighbour: {missing[:5]}"
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(moved)


# --- check --------------------------------------------------------------------------------------------------------


def football_data_neighbours(snap: Snapshot, links: dict[str, Link]) -> tuple[list, list]:
    """([(football-data row, FINISHED row)], [(football-data row, unfinished
    row)]) for each linked row without ESPN identity and each other row for
    its pair in window_rows. The first list is the duplicates; the second is
    stale fixture rows beside a result. Pure."""
    finished: list[tuple[dict, dict]] = []
    unfinished: list[tuple[dict, dict]] = []
    for link in links.values():
        if link.match["espn"]:
            continue
        for x in window_rows(snap, link):
            (finished if x["status"] == "finished" else unfinished).append((link.match, x))
    return finished, unfinished


def probable_alias_twins(snap: Snapshot, links: dict[str, Link]) -> list[tuple[dict, dict]]:
    """(football-data row, ESPN row) under another team pair, same league and
    score, kicking off with it or 1 h before it (+-10 min). Pure."""
    index = _LeagueKickoffs(snap.matches)
    out = []
    for link in links.values():
        m = link.match
        if m["espn"]:
            continue
        for when in (m["kickoff"], m["kickoff"] - LEGACY_OFFSET):
            out += [(m, t) for t in alias_twins(index, m, when)]
    return out


def run_check(conn, repoint_map: Optional[dict] = None) -> int:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        snap = load_snapshot(cur)
        legacy_error = None
        try:
            pr.check_legacy_kickoffs(cur, snap.raw_rows, snap.league_ids, snap.team_ids)
        except pr.LegacyKickoffsPresent as e:
            legacy_error = str(e)
    conn.rollback()
    counts: Counter = Counter()
    snap.set_pending(pending_repoints(snap, repoint_map or {}, counts))
    links = link_rows(snap, counts)
    duplicates, stale = football_data_neighbours(snap, links)
    aliases = probable_alias_twins(snap, links)
    if legacy_error:
        logger.error("check_legacy_kickoffs FAILS: %s", legacy_error)
    for m, x in duplicates[:10]:
        logger.error(
            "football-data row %s (%s) has a finished same-pair row %s at %s",
            m["id"],
            _iso(m["kickoff"]),
            x["id"],
            _iso(x["kickoff"]),
        )
    logger.info(
        "CHECK legacy_kickoffs=%s football_data_duplicates=%d | info: beside_unfinished_rows=%d "
        "probable_alias_twins=%d repoint_pending=%d linked=%d",
        "FAIL" if legacy_error else "PASS",
        len(duplicates),
        len(stale),
        len(aliases),
        len(snap.pending),
        len(links),
    )
    return 1 if legacy_error or duplicates else 0


# --- run ----------------------------------------------------------------------------------------------------------


def default_path(kind: str, suffix: str, base: Optional[str] = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base = (base or os.environ.get("BACKUP_LOCAL_DIR") or "/app/backups").rstrip("/")
    return f"{base}/football_data_{kind}_{stamp}{suffix}"


def write_csv(path: str, fields: tuple[str, ...], entries: list[dict]) -> None:
    out = Path(path)
    if out.exists():
        raise FileExistsError(f"{out} exists — the audit CSV is write-once")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(entries)


def assign_batches(entries: list[dict], actions: tuple[str, ...], size: int) -> list[list[dict]]:
    todo = [e for e in entries if e["action"] in actions]
    batches = [todo[i : i + size] for i in range(0, len(todo), size)]
    for n, batch in enumerate(batches, 1):
        for e in batch:
            e["batch"] = n
    return batches


def plan_counts(mode: str, entries: list[dict]) -> dict[str, int]:
    """{action: rows} over PLAN_KEYS[mode], zeros included. Pure."""
    by_action = Counter(e["action"] for e in entries)
    return {k: by_action.get(k, 0) for k in PLAN_KEYS[mode]}


def plan_mismatch(counts: dict[str, int], expect: Optional[dict[str, int]]) -> str:
    """Why the plan is not the one the operator pinned, or ''. Pure."""
    wrong = [
        f"{counts.get(k, 0)} {k}, expected {n}" for k, n in sorted((expect or {}).items()) if counts.get(k, 0) != n
    ]
    return "; ".join(wrong)


def parse_expect_plan(text: str, mode: str) -> dict[str, int]:
    """'merge=68,fold=26,refuse=0' -> {'merge': 68, 'fold': 26, 'refuse': 0}; ValueError on anything else."""
    out: dict[str, int] = {}
    allowed = PLAN_KEYS[mode] + EXTRA_PIN_KEYS[mode]
    for part in (p.strip() for p in text.split(",")):
        key, sep, value = part.partition("=")
        if not sep or key not in allowed or not value.isdigit() or key in out:
            raise ValueError(f"--expect-plan {text!r}: want KEY=N pairs with KEY one of {', '.join(allowed)}")
        out[key] = int(value)
    return out


def missing_pins(expect: Optional[dict[str, int]], mode: str) -> list[str]:
    """The PLAN_KEYS of `mode` an --apply's --expect-plan leaves out. Pure."""
    return [k for k in PLAN_KEYS[mode] if k not in (expect or {})]


def _log_plan(
    mode: str,
    planned: list[dict],
    refused: list[dict],
    batches: list,
    counts: Counter,
    out_csv: str,
    pinned: bool,
) -> None:
    """The plan's summary; refusals at ERROR unless `pinned` (the operator expected exactly these many)."""
    log_refusal = logger.warning if pinned else logger.error
    by_league = dict(sorted(Counter(e["league"] for e in planned).items()))
    if mode == "dedupe":
        merges = [e for e in planned if e["action"] in ("merge", "alias_merge")]
        aliases = [e for e in planned if e["action"] == "alias_merge"]
        folds = [e for e in planned if e["action"] == "fold"]
        logger.info(
            "Planned %d merge(s) (%d of them alias merges) and %d fold(s) in %d batch(es) %s: %d BST copies, offsets "
            "(copy - kept row, min) %s, half-time to copy %d, odds to move %d of %d, match_stats to move %d of %d; "
            "audit CSV %s",
            len(merges),
            len(aliases),
            len(folds),
            len(batches),
            by_league,
            sum(1 for e in planned if e["bst"]),
            dict(sorted(Counter(e["offset_min"] for e in planned).items())),
            sum(1 for e in planned if e["ht_write"]),
            sum(e.get("odds_to_move", 0) for e in planned),
            sum(e.get("copy_odds", 0) for e in planned),
            sum(e.get("stats_to_move", 0) for e in planned),
            sum(e.get("copy_match_stats", 0) for e in planned),
            out_csv,
        )
        for e in aliases:
            logger.info(
                "Alias merge: %s %s v %s copy %s into ESPN row %s, still filed under the wrong club (%s): "
                "scripts/repoint_team_matches.py moves it once the copy is gone",
                e["league"],
                e["home"],
                e["away"],
                e["copy_id"],
                e["twin"]["id"],
                e["kept_row_repoint"],
            )
        if folds:
            recs: Counter = Counter()
            for e in folds:
                recs.update(json.loads(e.get("fold_recs_by_status") or "{}"))
            logger.info(
                "Folds: %d stale 'scheduled' rows take the copy's result and become 'finished' %s; on them %d "
                "predictions (%d ungraded, 0 written after kickoff: the outcome trigger grades the pre-kickoff ones "
                "it knows) and recommendations by status %s (none open: the trigger settles none; %d without a "
                "profit_loss, which grade_completed_matches.py --match-id would settle by the final score)",
                len(folds),
                dict(sorted(Counter(e["league"] for e in folds).items())),
                sum(e.get("fold_predictions", 0) for e in folds),
                sum(e.get("fold_ungraded_predictions", 0) for e in folds),
                dict(sorted(recs.items())),
                sum(e.get("fold_unsettled_recs", 0) for e in folds),
            )
        for e in merges:
            if e["ht_action"] == "conflict_kept_espn":
                logger.error(
                    "%s %s v %s: half-time differs (copy %s, ESPN row %s); keeping the ESPN row's",
                    e["league"],
                    e["home"],
                    e["away"],
                    e["copy_ht"],
                    e["espn_ht"],
                )
    else:
        logger.info(
            "Planned %d shift(s) in %d batch(es) %s; %d land on a probable alias twin (alias_twin_id); %d copies "
            "--dedupe still merges or folds (not shifted; --shift --apply refuses until it has run); audit CSV %s",
            len(planned),
            len(batches),
            by_league,
            counts["shift_alias_twins"],
            counts["dedupe_pending"],
            out_csv,
        )
    reasons = Counter(e["reason"].split(" (")[0].split(":")[0] for e in refused)
    for e in refused[:20]:
        log_refusal(
            "refused %s %s v %s %s: %s",
            e["league"],
            e["home"],
            e["away"],
            e.get("copy_id") or e.get("match_id"),
            e["reason"],
        )
    if refused:
        log_refusal("%d row(s) refused and left alone (all in the audit CSV): %s", len(refused), dict(reasons))


def plan_dedupe_from_db(cur, snap: Snapshot, links: dict[str, Link], counts: Counter) -> tuple[list[dict], list]:
    """(dedupe entries, FK tables to matches): plan_dedupe over what the
    database says references each copy and each fold row."""
    fk_tables = load_fk_tables(cur)
    candidates = dedupe_candidates(snap, links, counts)
    refs = load_refs(cur, fk_tables, [link.match["id"] for link, _nearby in candidates])
    facts = load_fold_facts(cur, fold_fact_rows(candidates))
    return plan_dedupe(snap, candidates, refs, guard_table_names(fk_tables), counts, facts), fk_tables


def run(
    conn,
    mode: str,
    apply_changes: bool,
    out_csv: Optional[str] = None,
    archive_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    expect_plan: Optional[dict[str, int]] = None,
    repoint_map: Optional[dict[str, dict[str, tuple[str, str]]]] = None,
) -> Counter:
    counts: Counter = Counter()
    fk_tables: list[FkTable] = []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        snap = load_snapshot(cur)
        snap.set_pending(pending_repoints(snap, repoint_map or {}, counts))
        links = link_rows(snap, counts)
        if mode == "dedupe":
            entries, fk_tables = plan_dedupe_from_db(cur, snap, links, counts)
            add_move_counts(cur, entries)
        else:
            # what --dedupe would still merge or fold, so the shift plan is the one after it
            pending, _fks = plan_dedupe_from_db(cur, snap, links, Counter())
            planned_away = frozenset(e["copy_id"] for e in pending if e["action"] in DEDUPE_WRITES)
            counts["dedupe_pending"] = len(planned_away)
            entries = plan_shift(snap, links, counts, planned_away)
    conn.rollback()  # close the read snapshot before writing
    logger.info(
        "Loaded %d raw rows, %d soccer matches (db TimeZone %s); linked %d football-data rows; %d row(s) the "
        "reviewed re-point map is about to move",
        len(snap.raw_rows),
        len(snap.matches),
        snap.db_tz,
        len(links),
        len(snap.pending),
    )

    batches = assign_batches(
        entries, DEDUPE_WRITES if mode == "dedupe" else ("shift",), batch_size or DEFAULT_BATCH_SIZE[mode]
    )
    planned = [e for e in entries if e["action"] != "refuse"]
    refused = [e for e in entries if e["action"] == "refuse"]
    out_csv = out_csv or default_path(mode, ".csv", archive_dir)
    write_csv(out_csv, DEDUPE_FIELDS if mode == "dedupe" else SHIFT_FIELDS, entries)
    plan = plan_counts(mode, entries)
    counts["planned"], counts["refused"] = len(planned), len(refused)
    _log_plan(
        mode, planned, refused, batches, counts, out_csv, pinned=(expect_plan or {}).get("refuse") == len(refused)
    )
    unsettled = sum(e.get("fold_unsettled_recs", 0) for e in planned if e["action"] == "fold")
    if mode == "shift":
        extra = f" dedupe_pending={counts['dedupe_pending']}"
    else:
        extra = f" fold_unsettled_recs={unsettled}"
    logger.info("PLAN mode=%s %s%s", mode, " ".join(f"{k}={v}" for k, v in plan.items()), extra)

    mismatch = plan_mismatch({**plan, "fold_unsettled_recs": unsettled} if mode == "dedupe" else plan, expect_plan)
    if mismatch:
        counts["plan_mismatch"] = 1
        logger.error("The plan is not the one pinned (%s): nothing written; review the audit CSV", mismatch)
        return counts
    counts["refused_unexpected"] = 0 if "refuse" in (expect_plan or {}) else len(refused)
    if not apply_changes:
        logger.info("DRY-RUN: re-run with --apply to write %d row(s)", len(planned))
        return counts
    if mode == "dedupe" and unsettled and "fold_unsettled_recs" not in (expect_plan or {}):
        counts["unsettled_unpinned"] = 1
        logger.error(
            "Refusing to fold: the fold rows hold %d rec(s) without a profit_loss (fold_unsettled_recs in the audit "
            "CSV), which grade_completed_matches.py --match-id would settle by the final score — review them, then "
            "pin fold_unsettled_recs=%d in --expect-plan to fold anyway (nothing written)",
            unsettled,
            unsettled,
        )
        return counts
    if mode == "shift" and counts["dedupe_pending"]:
        counts["shift_blocked"] = 1
        logger.error(
            "Refusing to shift: --dedupe still plans to merge or fold %d copies — run it first (nothing written)",
            counts["dedupe_pending"],
        )
        return counts
    if mode == "dedupe":
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        base = (archive_dir or os.environ.get("BACKUP_LOCAL_DIR") or "/app/backups").rstrip("/")
        for n, batch in enumerate(batches, 1):
            archive = Path(f"{base}/football_data_dedupe_{stamp}_b{n:03d}.jsonl.gz")
            done = apply_dedupe_batch(conn, batch, n, fk_tables, archive)
            counts.update(done)
            logger.info("Batch %d/%d committed: %s; archive %s", n, len(batches), dict(sorted(done.items())), archive)
        graded = {k.split(":", 1)[1]: v for k, v in counts.items() if k.startswith("trigger_graded:")}
        if counts["folded"]:
            logger.info(
                "Folded %d stale rows; the outcome trigger graded %d of their predictions %s (other types stay "
                "ungraded until grade_completed_matches.py --match-id runs on them)",
                counts["folded"],
                counts["trigger_graded_predictions"],
                dict(sorted(graded.items())),
            )
        alias_kept = sorted(e["twin"]["id"] for b in batches for e in b if e["action"] == "alias_merge")
        if alias_kept:
            logger.info(
                "NEXT: %d kept row(s) are still filed under the wrong club (%s): re-run "
                "scripts/repoint_team_matches.py (dry run first; the sides it refused for a same-pair twin now plan "
                "as 'repoint'); until then "
                "promote_raw refuses their fixtures (alias slot)",
                len(alias_kept),
                ", ".join(alias_kept),
            )
    else:
        for n, batch in enumerate(batches, 1):
            counts["shifted"] += apply_shift_batch(conn, batch, n)
            logger.info("Batch %d/%d committed: %d shifted so far", n, len(batches), counts["shifted"])
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dedupe", action="store_const", dest="mode", const="dedupe", help="Merge copies.")
    mode.add_argument("--shift", action="store_const", dest="mode", const="shift", help="Move BST kickoffs.")
    mode.add_argument("--check", action="store_const", dest="mode", const="check", help="Read-only after-state.")
    p.add_argument("--apply", action="store_true", help="Write (default: dry run).")
    p.add_argument(
        "--expect-plan",
        default=None,
        help="Write nothing unless the plan has exactly these counts: dedupe merge=N,alias_merge=N,fold=N,refuse=N "
        "[,fold_unsettled_recs=N]; shift shift=N,refuse=N. A dry run takes any subset; --apply needs every key.",
    )
    p.add_argument("--out-csv", default=None, help="Audit CSV (default $BACKUP_LOCAL_DIR or /app/backups).")
    p.add_argument("--archive-dir", default=None, help="Archive/CSV dir (default $BACKUP_LOCAL_DIR or /app/backups).")
    p.add_argument("--batch-size", type=int, default=None, help="Rows per transaction (dedupe 100, shift 5000).")
    p.add_argument(
        "--repoint-map",
        default=str(REPOINT_MAP_PATH),
        help="scripts/repoint_team_matches.py's reviewed map (default: scripts/data/repoint_team_matches.csv; "
        "absent, no row is taken as about to move).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = p.parse_args(argv)
    if args.mode == "check" and (args.apply or args.expect_plan is not None):
        p.error("--check is read-only and plans nothing")
    if args.expect_plan is not None:
        try:
            args.expect_plan = parse_expect_plan(args.expect_plan, args.mode)
        except ValueError as e:
            p.error(str(e))
    if args.apply and missing_pins(args.expect_plan, args.mode):
        p.error(
            f"--apply needs --expect-plan naming every {args.mode} key ({', '.join(PLAN_KEYS[args.mode])}); "
            f"missing: {', '.join(missing_pins(args.expect_plan, args.mode))}"
        )
    if args.batch_size is not None and args.batch_size < 1:
        p.error("--batch-size must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    if Path(args.repoint_map).is_file():
        repoint_map = load_repoint_map(args.repoint_map)  # loud on a malformed map
        logger.info("Re-point map %s: %d match row(s)", args.repoint_map, len(repoint_map))
    else:
        repoint_map = {}
        logger.warning(
            "No re-point map at %s: a copy whose ESPN row is filed under the wrong club stays refused", args.repoint_map
        )
    with psycopg2.connect(args.database_url) as conn:
        if args.mode == "check":
            return run_check(conn, repoint_map)
        try:
            counts = run(
                conn,
                args.mode,
                args.apply,
                args.out_csv,
                args.archive_dir,
                args.batch_size,
                args.expect_plan,
                repoint_map,
            )
        except PlanDrift as e:
            logger.error("Stopped, batch rolled back (earlier batches are committed; a re-run re-plans): %s", e)
            return 1
    logger.info("Done: %s", dict(sorted(counts.items())))
    failed = ("plan_mismatch", "refused_unexpected", "shift_blocked", "unsettled_unpinned")
    return 1 if any(counts[k] for k in failed) else 0


if __name__ == "__main__":
    sys.exit(main())
