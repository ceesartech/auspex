"""Grade predictions + settle recommendations for finished matches.

The closing-the-loop step that Phase 5 introduces. For every match
with status='finished' in the lookback window we:

  1. Compute actual_outcome per (prediction_type, model_name) via
     scripts/grading_outcomes.py — handles soccer + NHL markets.
  2. UPDATE predictions: actual_outcome, is_correct, profit_loss for
     every row whose is_correct is still NULL.
  3. UPDATE betting_recommendations: status (won/lost/void),
     actual_result, profit_loss, settled_at for every row whose status
     is still 'pending' or 'placed'. A rec written at or after the
     match's kickoff is VOIDED (profit 0, actual_result 'void: written
     after kickoff'), never graded: a bet placed after kickoff is
     hindsight. scripts/void_post_kickoff_recs.py corrects the ones
     settled before this guard existed.

Soccer full-time markets and soccer recs settle on 90 minutes: a cup tie
decided in extra time or on penalties carries its 90-minute score in
metadata (grading_outcomes.settlement_scores). The DB trigger grades
match_result / over_under / btts on the post-extra-time home_score /
away_score at the status flip, so on a match carrying that score this
script also re-grades the already-graded full-time predictions whose
verdict changes (guarded on the old values, never un-grading); the trigger's
rec settlements carry no P&L and are re-settled here anyway.

A 90-minute score can arrive after the match was graded (a later results
sweep in the 3-day window, or scripts/backfill_soccer_regulation_scores.py):
recs already settled with P&L on the final score are then re-settled when
their verdict changes, guarded, each logged at ERROR.

Postponements (fetch_upcoming records ESPN's STATUS_POSTPONED): a rec on a
match still 'postponed' POSTPONED_VOID_HOURS after its kickoff, or on one
played more than that after its original kickoff when the rec predates the
postponement, is VOIDED (profit 0, actual_result 'void: match postponed
>72h'), as books void them. Cancellations likewise: a rec on a match
'cancelled' with a kickoff CANCELLED_VOID_HOURS old is VOIDED ('void: match
cancelled'), except on a cancelled duplicate of a played fixture
(metadata.orphan_of), which is logged at ERROR for repair_stranded_recs.
Both voids are guarded on the rec's and the match's status.

A bout with no winner (an MMA draw or no contest: fetch_upcoming records it
finished with NULL scores and metadata.no_winner, see
grading_outcomes.NO_WINNER_KEY) is graded by convention: its predictions are
neither right nor wrong (actual_outcome 'draw' / 'no_contest', is_correct
NULL, logged once when written; such a row is not read back afterwards), and
its open recs are VOIDED (profit 0, actual_result 'void: draw' / 'void: no
contest'), as most books void a two-way fight moneyline. A recorded no-winner
kind this script does not know is logged at ERROR and nothing on that match is
graded or settled.

Phantom-line guard: NBA/NFL spread + total grade against the closing
line stored in features_cache, but compute_features_{nba,nfl}.py write
NEUTRAL_DEFAULTS (spread 0.0, total 45.0) for matches no book ever
priced. Grading against those fabricated lines produced 230 graded NFL
preseason TOTAL verdicts and 230 SPREAD verdicts in Aug-2026 against a
45.0 line that never existed (real preseason totals averaged 37.4).
Such predictions are now left ungraded (is_correct stays NULL) unless
the match has at least one non-live odds row in that market.

Idempotent: rows already graded (is_correct IS NOT NULL) are left
alone, so re-running on the same window is a cheap no-op except for
matches that were freshly graded or whose predictions/recs landed
after the previous run. That also means a grading FIX never reaches a
row graded under the old code: scripts/regrade_predictions.py re-runs
compute_prediction_grade over already-graded rows for that.

The existing DB trigger calculate_match_outcome (migration 003)
covers soccer match_result / over_under / btts at the moment
status flips to 'finished'. This script is the comprehensive
follow-up: it grades NHL, covers markets the trigger misses, and
re-grades any predictions that were written AFTER the status flip
(the trigger won't fire again on those).

Usage (inside the api container):
    python /app/scripts/grade_completed_matches.py
    python /app/scripts/grade_completed_matches.py --days 30
    python /app/scripts/grade_completed_matches.py --match-id <uuid>  # ad-hoc
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Make the shared grading-outcomes helper importable.
sys.path.insert(0, os.path.dirname(__file__))

from grading_outcomes import (  # noqa: E402
    NO_WINNER_KEY,
    NO_WINNER_KINDS,
    NO_WINNER_VOID_RESULTS,
    REC_HALF_LOST,
    REC_HALF_WON,
    REC_LOST,
    REC_PUSH,
    REC_WON,
    REGULATION_HOME_KEY,
    SOCCER_FT_TYPES,
    actual_outcome,
    grade_prediction,
    grade_rec_selection,
    halftime_scores,
    is_nba_spread,
    is_nba_total,
    is_nfl_spread,
    is_nfl_total,
    is_push,
    no_winner_result,
    rec_status_and_pl,
    regulation_scores,
    soccer_went_past_90,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("grade_completed_matches")

# actual_result of a rec voided because it was written at or after kickoff.
# Shared with repair_stranded_recs.py and void_post_kickoff_recs.py, so every
# such row is findable by this one prefix.
POST_KICKOFF_VOID_RESULT = "void: written after kickoff"

# A postponed match not played within this many hours of its kickoff voids
# the recs priced for it (books void them too). fetch_upcoming stamps the
# original kickoff and the time it recorded the postponement under these
# metadata keys, so a match played after a longer delay is still recognised.
POSTPONED_VOID_HOURS = 72
POSTPONED_VOID_RESULT = "void: match postponed >72h"
POSTPONED_KICKOFF_KEY = "postponed_kickoff"
POSTPONED_AT_KEY = "postponed_at"

# A cancelled match (an ESPN cancel or abandonment, an operator's cancel, a
# re-drawn tennis/MMA slot) voids its open recs once its kickoff is this old,
# the same shape as the postponement rule. A cancelled row that is a
# DUPLICATE of a played fixture (repair_fixture_orphans' metadata.orphan_of)
# is not voided: its recs settle from the twin (repair_stranded_recs).
CANCELLED_VOID_HOURS = 72
CANCELLED_VOID_RESULT = "void: match cancelled"
ORPHAN_OF_KEY = "orphan_of"

# Every void above (post-kickoff, postponed, cancelled, never-resulted, the
# constant-prior incident) writes an actual_result with this prefix: no
# score decides such a rec, so a later 90-minute score never re-settles it.
SCORE_INDEPENDENT_VOID_PREFIX = "void:"
REC_OUTCOMES = frozenset({REC_WON, REC_LOST, REC_PUSH, REC_HALF_WON, REC_HALF_LOST})

# A prediction on a bout with no winner is recorded once (actual_outcome set,
# is_correct NULL for good) and never read back: without this, the 14-day
# listing below would re-visit the match, and re-log it, every 15 minutes.
# m is the prediction's match.
NO_WINNER_RECORDED_SQL = f"(m.metadata->>'{NO_WINNER_KEY}' IS NOT NULL AND p.actual_outcome IS NOT NULL)"


# ── DB I/O ────────────────────────────────────────────────────────────


def list_finished_matches(cur, days: int, match_id: Optional[str]) -> list[dict]:
    """Finished matches needing grading. If --match-id is given we
    short-circuit to that one row (operator escape hatch for ad-hoc
    re-grading). Otherwise: matches that finished in the last `days`
    days AND have at least one ungraded prediction OR unsettled rec, or
    soccer matches that carry a 90-minute score (their trigger-graded
    full-time markets and recs settled on the final score need re-grading;
    a no-op once done, and such ties are a handful a month — NHL games carry
    a key of the same name, so the clause is soccer-only, as grade_match
    treats a league-less match). The EXISTS subqueries keep this from churning over
    already-graded matches on every run. `sport` and `match_date` feed the
    90-minute and postponement rules in grade_match."""
    if match_id is not None:
        cur.execute(
            """
            SELECT m.id::text AS match_id, m.match_date, l.sport,
                   m.home_score, m.away_score, m.status, m.metadata, m.half_time_home, m.half_time_away
            FROM matches m
            LEFT JOIN leagues l ON l.id = m.league_id
            WHERE m.id = %s AND m.status = 'finished'
            """,
            (match_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    cur.execute(
        f"""
        SELECT m.id::text AS match_id, m.match_date, l.sport,
               m.home_score, m.away_score, m.status, m.metadata, m.half_time_home, m.half_time_away
        FROM matches m
        LEFT JOIN leagues l ON l.id = m.league_id
        WHERE m.status = 'finished'
          AND m.match_date >= NOW() - (%s || ' days')::interval
          AND (
              EXISTS (
                  SELECT 1 FROM predictions p
                  WHERE p.match_id = m.id AND p.is_correct IS NULL
                    AND NOT {NO_WINNER_RECORDED_SQL}
              )
              OR EXISTS (
                  SELECT 1 FROM betting_recommendations br
                  WHERE br.match_id = m.id
                    AND (
                          br.status IN ('pending', 'placed')
                       -- trigger-settled but P&L never computed (the
                       -- migration-011 trigger sets status only) —
                       -- must reach grade_match for the money fill
                       OR (br.status IN ('won', 'lost', 'void')
                           AND br.profit_loss IS NULL)
                    )
              )
              -- a soccer match decided after 90 minutes: the trigger graded
              -- its full-time markets on the post-extra-time score. Soccer
              -- only: load_nhl_historical --attach-only writes the same key
              -- on every finished NHL game (96 in a 14-day window on
              -- 2026-10-31), which would be walked every tick for nothing.
              OR (COALESCE(l.sport, 'soccer') = 'soccer' AND m.metadata ? %s)
          )
        ORDER BY m.match_date ASC
        """,
        (str(days), REGULATION_HOME_KEY),
    )
    return [dict(r) for r in cur.fetchall()]


def list_ungraded_predictions(cur, match_id: str) -> list[dict]:
    """Every prediction row for the match whose is_correct is still
    NULL. We grade and update each in place.

    `after_kickoff` flags rows written after the match started: hindsight,
    not a prediction. 467 such rows (written 2026-06-02/03 for 2023-25 NFL,
    NBA and tennis matches) sat ungraded only because they were older than
    the --days window; a wide run would have folded them into accuracy.

    `probabilities` is needed to grade a correct_score 'other' pick, which
    covers whatever that prediction did not list.

    A prediction already recorded on a bout with no winner (actual_outcome
    set, is_correct NULL for good: NO_WINNER_RECORDED_SQL) is not read back."""
    cur.execute(
        f"""
        SELECT p.id::text AS prediction_id,
               p.prediction_type, p.model_name, p.predicted_outcome,
               p.probabilities,
               (p.created_at > m.match_date) AS after_kickoff
        FROM predictions p
        JOIN matches m ON m.id = p.match_id
        WHERE p.match_id = %s AND p.is_correct IS NULL
          AND NOT {NO_WINNER_RECORDED_SQL}
        """,
        (match_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def is_line_dependent(prediction_type: str, model_name: str) -> bool:
    """True when actual_outcome() will read the grading line out of
    features_cache (NBA + NFL spread/total) — the only predictions the
    phantom-line guard applies to.

    NHL is excluded on purpose: its puck line (-1.5) and total (5.5)
    are constants in grading_outcomes, not book lines, so an NHL
    spread/total is gradable from the final score with no odds at all.
    """
    if prediction_type == "spread":
        return is_nba_spread(model_name) or is_nfl_spread(model_name)
    if prediction_type == "total":
        return is_nba_total(model_name) or is_nfl_total(model_name)
    return False


# Emitted once per process if the grading-context row comes back without
# the odds-presence probe columns (only possible against a stubbed
# cursor — the real query always returns booleans). Loud, not silent.
_PROBE_WARNED = False


def fetch_match_grading_context(cur, match_id: str) -> dict:
    """Everything grade_match needs about one match, in one round trip:

      * `features` — most-recent features_cache JSON regardless of
        feature_set. Used by NBA + NFL spread/total grading to read the
        closing line the model conditioned on (closing_spread_home /
        closing_total_line). None if no features were ever cached, in
        which case those predictions stay ungraded until a row appears.
        Returns the freshest row by computed_at; for matches with
        multiple feature_sets the latest pipeline write wins
        (nba_baseline for NBA, nfl_baseline for NFL, nhl_baseline for
        NHL, baseline for soccer). The dispatch only reads the shared
        closing-line keys, so soccer/NHL features pass through
        harmlessly.

      * `has_spread_odds` / `has_total_odds` — whether a REAL LINE for
        that market exists, i.e. whether a book ever priced it.

        The probes match the exact row the feature builder reads, not
        merely "some row in this market": compute_features_{nba,nfl}
        .fetch_closing_odds populates closing_spread_home only from a
        `selection='home'` spread row with a non-NULL `line`, and
        closing_total_line only from a `selection='over'` total row with
        a non-NULL `line`. A one-sided or NULL-line market would pass a
        coarser EXISTS and still be graded against the fabricated
        default.

    The odds probes exist because compute_features_{nba,nfl}.py fill
    NEUTRAL_DEFAULTS (closing_spread_home = 0.0, closing_total_line =
    45.0) when a match has no odds at all, and this script used to read
    those straight back out of features_cache as "the closing line the
    model conditioned on". That is how 230 NFL preseason TOTAL and 230
    SPREAD predictions got graded against a 45.0 line that never
    existed (actual preseason total averaged 37.4). A match with no
    non-live odds in the market has no closing line, full stop.

    One query so the mocked cursors in the existing test suite keep
    seeing a single features_cache read. The scalar sub-SELECT (rather
    than FROM features_cache) guarantees exactly one row back even when
    the match has no cached features.
    """
    cur.execute(
        """
        SELECT
            (
                SELECT features
                FROM features_cache
                WHERE match_id = %s
                ORDER BY computed_at DESC
                LIMIT 1
            ) AS features,
            EXISTS (
                SELECT 1 FROM odds o
                WHERE o.match_id = %s AND o.market_type = 'spread'
                  AND o.selection = 'home' AND o.line IS NOT NULL AND NOT o.is_live
            ) AS has_spread_odds,
            EXISTS (
                SELECT 1 FROM odds o
                WHERE o.match_id = %s AND o.market_type = 'total'
                  AND o.selection = 'over' AND o.line IS NOT NULL AND NOT o.is_live
            ) AS has_total_odds
        """,
        (match_id, match_id, match_id),
    )
    row = cur.fetchone() or {}
    ctx = {
        "features": row.get("features") or None,
        "has_spread_odds": row.get("has_spread_odds"),
        "has_total_odds": row.get("has_total_odds"),
    }
    if "has_spread_odds" not in row:
        global _PROBE_WARNED
        if not _PROBE_WARNED:
            _PROBE_WARNED = True
            logger.warning(
                "grading-context row lacks the odds-presence probe columns "
                "(cursor=%s); line-dependent predictions will be graded "
                "WITHOUT the phantom-line guard",
                type(cur).__name__,
            )
    return ctx


def fetch_match_features(cur, match_id: str) -> Optional[dict]:
    """Back-compat shim: just the features half of
    fetch_match_grading_context. Kept because the DAG-era call sites
    and tests import it by name."""
    return fetch_match_grading_context(cur, match_id)["features"]


def update_prediction(cur, prediction_id: str, actual: Optional[str], correct: Optional[bool]) -> None:
    """Persist one prediction's grade. actual_outcome is set even on
    pushes so the recommendation settler can detect the push from the
    prediction row alone (without re-running the grading math)."""
    cur.execute(
        """
        UPDATE predictions
        SET actual_outcome = %s,
            is_correct = %s,
            updated_at = NOW()
        WHERE id = %s
        """,
        (actual, correct, prediction_id),
    )


def list_graded_full_time_predictions(cur, match_id: str) -> list[dict]:
    """The match's GRADED soccer full-time predictions (is_correct set), with
    the verdict they hold. Only read for a match carrying a 90-minute score:
    the DB trigger graded match_result / over_under / btts on the
    post-extra-time score when the match flipped to 'finished'. Pushes
    (is_correct NULL) are list_ungraded_predictions' rows already."""
    cur.execute(
        """
        SELECT p.id::text AS prediction_id,
               p.prediction_type, p.model_name, p.predicted_outcome,
               p.probabilities, p.actual_outcome, p.is_correct,
               (p.created_at > m.match_date) AS after_kickoff
        FROM predictions p
        JOIN matches m ON m.id = p.match_id
        WHERE p.match_id = %s
          AND p.is_correct IS NOT NULL
          AND p.prediction_type = ANY(%s)
        """,
        (match_id, sorted(SOCCER_FT_TYPES)),
    )
    return [dict(r) for r in cur.fetchall()]


def regrade_prediction(cur, pred: dict, actual: Optional[str], correct: Optional[bool]) -> bool:
    """Rewrite one graded prediction's verdict, guarded on the verdict it
    was read with: a row anything else rewrote in between is left alone.
    True when the row was written."""
    cur.execute(
        """
        UPDATE predictions
        SET actual_outcome = %s,
            is_correct = %s,
            updated_at = NOW()
        WHERE id = %s
          AND actual_outcome IS NOT DISTINCT FROM %s
          AND is_correct IS NOT DISTINCT FROM %s
        """,
        (actual, correct, pred["prediction_id"], pred["actual_outcome"], pred["is_correct"]),
    )
    return cur.rowcount == 1


def regrade_on_regulation(cur, match: dict, ctx: dict) -> int:
    """Re-grade a 90-minute-scored soccer match's already-graded full-time
    predictions under the current code, writing only verdicts that change
    and never un-grading one (a NULL verdict is written only for a push),
    the same rule scripts/regrade_predictions.py applies. Returns the rows
    rewritten. A no-op for every other match."""
    if (match.get("sport") or "soccer") != "soccer" or regulation_scores(match.get("metadata")) is None:
        return 0
    written = 0
    for pred in list_graded_full_time_predictions(cur, match["match_id"]):
        skip, actual, correct = compute_prediction_grade(pred, match, ctx)
        if skip is not None or (actual, correct) == (pred["actual_outcome"], pred["is_correct"]):
            continue
        if correct is None and not is_push(actual):
            continue
        if regrade_prediction(cur, pred, actual, correct):
            written += 1
            logger.warning(
                "Re-graded prediction %s (match %s, %s/%s, pick %r) on the 90-minute score: (%r, %r) -> (%r, %r)",
                pred["prediction_id"],
                match["match_id"],
                pred["model_name"],
                pred["prediction_type"],
                pred["predicted_outcome"],
                pred["actual_outcome"],
                pred["is_correct"],
                actual,
                correct,
            )
        else:
            logger.error(
                "Prediction %s (match %s) changed while it was being re-graded — left as it is",
                pred["prediction_id"],
                match["match_id"],
            )
    return written


def list_settled_recs_for_match(cur, match_id: str) -> list[dict]:
    """The match's recs settled WITH money (profit_loss set), with what
    grade_rec_selection needs. Only read for a soccer match carrying a
    90-minute score: a rec settled on the final score before the score
    arrived (a later results sweep stamped it, or the backfill did) holds
    the wrong verdict and P&L, and list_open_recs_for_match never reads it
    again."""
    cur.execute(
        """
        SELECT br.id::text AS rec_id,
               br.bet_type, br.selection,
               br.recommended_stake, br.odds_at_recommendation,
               br.status, br.actual_result, br.profit_loss,
               p.model_name AS p_model_name,
               l.sport AS sport
        FROM betting_recommendations br
        LEFT JOIN predictions p ON p.id = br.prediction_id
        LEFT JOIN matches m ON m.id = br.match_id
        LEFT JOIN leagues l ON l.id = m.league_id
        WHERE br.match_id = %s
          AND br.status IN ('won', 'lost', 'void')
          AND br.profit_loss IS NOT NULL
        """,
        (match_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def resettle_recommendation(cur, rec: dict, new_status: str, actual_result: str, profit_loss: Decimal) -> bool:
    """Rewrite one settled rec, guarded on the status, actual_result and
    profit_loss it was read with. True when the row was written."""
    cur.execute(
        """
        UPDATE betting_recommendations
        SET status = %s,
            actual_result = %s,
            profit_loss = %s,
            settled_at = NOW(),
            updated_at = NOW()
        WHERE id = %s
          AND status = %s
          AND actual_result IS NOT DISTINCT FROM %s
          AND profit_loss IS NOT DISTINCT FROM %s
        """,
        (
            new_status,
            actual_result,
            profit_loss,
            rec["rec_id"],
            rec["status"],
            rec["actual_result"],
            rec["profit_loss"],
        ),
    )
    return cur.rowcount == 1


def _cents(value) -> Decimal:
    """A P&L as profit_loss stores it (DECIMAL(10,2), ties away from zero)."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def resettle_recs_on_regulation(cur, match: dict) -> int:
    """Re-settle a 90-minute-scored soccer match's recs that were settled,
    with money, on the final score: only those whose verdict changes, each
    guarded on what it was read with, each logged at ERROR (a settled bet's
    P&L moves). A rec voided for a reason no score decides ('void: ...') is
    never touched, and one the 90-minute score cannot grade keeps its
    settlement (logged). Returns the recs rewritten; a no-op for every other
    match."""
    meta = match.get("metadata") or {}
    if (match.get("sport") or "soccer") != "soccer" or regulation_scores(meta) is None:
        return 0
    written = 0
    for rec in list_settled_recs_for_match(cur, match["match_id"]):
        if str(rec.get("actual_result") or "").startswith(SCORE_INDEPENDENT_VOID_PREFIX):
            continue
        outcome = grade_rec_selection(
            rec["bet_type"],
            rec["selection"],
            match["home_score"],
            match["away_score"],
            sport=rec["sport"],
            model_name=rec["p_model_name"],
            metadata=meta,
        )
        if outcome is None:
            logger.warning(
                "Rec %s (match %s, %s %s) is settled %s (%s) but the 90-minute score cannot grade it — left as it is",
                rec["rec_id"],
                match["match_id"],
                rec["bet_type"],
                rec["selection"],
                rec["status"],
                rec["actual_result"],
            )
            continue
        new_status, pl = rec_status_and_pl(outcome, rec["recommended_stake"], rec["odds_at_recommendation"])
        same_verdict = rec["actual_result"] not in REC_OUTCOMES or rec["actual_result"] == outcome
        if new_status == rec["status"] and same_verdict and _cents(pl) == _cents(rec["profit_loss"]):
            continue
        if resettle_recommendation(cur, rec, new_status, outcome, pl):
            written += 1
            logger.error(
                "Re-settled rec %s (match %s, %s %s @ %s, stake %s) on the 90-minute score %s: (%s, %s, %s) -> "
                "(%s, %s, %s). It had settled on the final, post-extra-time %s-%s.",
                rec["rec_id"],
                match["match_id"],
                rec["bet_type"],
                rec["selection"],
                rec["odds_at_recommendation"],
                rec["recommended_stake"],
                regulation_scores(meta),
                rec["status"],
                rec["actual_result"],
                rec["profit_loss"],
                new_status,
                outcome,
                _cents(pl),
                match.get("home_score"),
                match.get("away_score"),
            )
        else:
            logger.error(
                "Rec %s (match %s) changed while it was being re-settled on the 90-minute score — left as it is",
                rec["rec_id"],
                match["match_id"],
            )
    return written


def _as_utc(value) -> Optional[datetime]:
    """A timestamp from metadata (ISO string) or the DB (datetime), tz-aware
    (naive read as UTC), or None when it cannot be read."""
    if isinstance(value, datetime):
        ts = value
    else:
        try:
            ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def postponed_past_void_limit(match: dict, rec: dict) -> bool:
    """True for a rec written before ESPN postponed its match (fetch_upcoming
    stamps metadata.postponed_kickoff / postponed_at) when the match was then
    played more than POSTPONED_VOID_HOURS after that original kickoff: the
    rec priced a fixture that did not take place when it was priced for."""
    meta = match.get("metadata") or {}
    if POSTPONED_KICKOFF_KEY not in meta:
        return False
    original, recorded = _as_utc(meta.get(POSTPONED_KICKOFF_KEY)), _as_utc(meta.get(POSTPONED_AT_KEY))
    kickoff, created = _as_utc(match.get("match_date")), _as_utc(rec.get("created_at"))
    if original is None or recorded is None or kickoff is None or created is None:
        logger.warning(
            "Rec %s (match %s): cannot apply the postponement rule (postponed_kickoff=%r, postponed_at=%r, "
            "kickoff=%r, created_at=%r) — settled as usual",
            rec.get("rec_id"),
            match.get("match_id"),
            meta.get(POSTPONED_KICKOFF_KEY),
            meta.get(POSTPONED_AT_KEY),
            match.get("match_date"),
            rec.get("created_at"),
        )
        return False
    return kickoff - original > timedelta(hours=POSTPONED_VOID_HOURS) and created < recorded


def list_recs_on_long_postponed_matches(cur) -> list[dict]:
    """Open recs on matches still 'postponed' POSTPONED_VOID_HOURS after their
    kickoff. A rescheduled match flips back to 'scheduled' (fetch_upcoming's
    fixtures path) and is judged by postponed_past_void_limit when it
    finishes instead."""
    cur.execute(
        """
        SELECT br.id::text AS rec_id, br.match_id::text AS match_id,
               br.bet_type, br.selection, br.status, br.created_at,
               m.match_date AS kickoff
        FROM betting_recommendations br
        JOIN matches m ON m.id = br.match_id
        WHERE br.status IN ('pending', 'placed')
          AND m.status = 'postponed'
          AND m.match_date < NOW() - (%s || ' hours')::interval
        ORDER BY m.match_date, br.id
        """,
        (str(POSTPONED_VOID_HOURS),),
    )
    return [dict(r) for r in cur.fetchall()]


# Guarded on the rec's status and on its match's status as read: a rec
# settled meanwhile, or a match rescheduled meanwhile, is left alone.
VOID_OPEN_REC_SQL = """
    UPDATE betting_recommendations br
    SET status = 'void',
        actual_result = %s,
        profit_loss = 0,
        settled_at = NOW(),
        updated_at = NOW()
    FROM matches m
    WHERE br.id = %s
      AND br.status = %s
      AND m.id = br.match_id
      AND m.status = %s
    RETURNING br.id
"""


def void_open_rec(cur, rec: dict, match_status: str, actual_result: str) -> bool:
    """Void (profit 0) one open rec whose match is `match_status`, guarded
    (VOID_OPEN_REC_SQL). True when written; a guard that fires is logged."""
    cur.execute(VOID_OPEN_REC_SQL, (actual_result, rec["rec_id"], rec["status"], match_status))
    if cur.fetchone() is not None:
        return True
    logger.error(
        "Rec %s (match %s) or its match changed while voiding it (%r) — left as it is",
        rec["rec_id"],
        rec["match_id"],
        actual_result,
    )
    return False


def void_recs_on_long_postponed_matches(cur) -> int:
    """Void (profit 0) every open rec list_recs_on_long_postponed_matches
    finds. Returns how many were written."""
    voided = 0
    for rec in list_recs_on_long_postponed_matches(cur):
        logger.warning(
            "Voiding rec %s (match %s, %s %s, status %s): the match is still postponed %dh after its kickoff %s",
            rec["rec_id"],
            rec["match_id"],
            rec.get("bet_type"),
            rec.get("selection"),
            rec.get("status"),
            POSTPONED_VOID_HOURS,
            rec.get("kickoff"),
        )
        voided += void_open_rec(cur, rec, "postponed", POSTPONED_VOID_RESULT)
    return voided


def list_recs_on_cancelled_matches(cur, match_ids: Optional[list[str]] = None) -> list[dict]:
    """Open recs on matches 'cancelled' with a kickoff CANCELLED_VOID_HOURS
    old. `orphan_of` names the played twin of a cancelled duplicate row.
    `match_ids` limits the read to those matches (scripts/backfill_results.py:
    only the rows its own run cancelled); None reads every sport's."""
    only = "" if match_ids is None else "AND m.id = ANY(%s::uuid[])"
    params: tuple = (str(CANCELLED_VOID_HOURS),) if match_ids is None else (str(CANCELLED_VOID_HOURS), match_ids)
    cur.execute(
        f"""
        SELECT br.id::text AS rec_id, br.match_id::text AS match_id,
               br.bet_type, br.selection, br.status, br.created_at,
               m.match_date AS kickoff,
               m.metadata->>'orphan_of' AS orphan_of
        FROM betting_recommendations br
        JOIN matches m ON m.id = br.match_id
        WHERE br.status IN ('pending', 'placed')
          AND m.status = 'cancelled'
          AND m.match_date < NOW() - (%s || ' hours')::interval
          {only}
        ORDER BY m.match_date, br.id
        """,
        params,
    )
    return [dict(r) for r in cur.fetchall()]


def void_recs_on_cancelled_matches(cur) -> tuple[int, int]:
    """Void (profit 0) the open recs on cancelled matches. Returns (voided,
    refused): a rec on a cancelled DUPLICATE of a played fixture is refused
    and logged at ERROR, never voided — the match was played, on the twin.
    The ERROR names the remedy: repair_stranded_recs settles (or voids, when
    written after kickoff) a 'pending' rec on such a row from its twin and
    re-points it there; it does not touch 'placed', so a 'placed' one is set
    back to 'pending' first. repair_fixture_orphans never retires a row that
    holds an open rec (CANCEL_PINNED_SQL), so this needs a rec written or
    re-opened on the row afterwards."""
    voided = refused = 0
    for rec in list_recs_on_cancelled_matches(cur):
        if rec.get("orphan_of"):
            refused += 1
            reopen = (
                ""
                if rec.get("status") == "pending"
                else f"UPDATE betting_recommendations SET status = 'pending' WHERE id = '{rec['rec_id']}' "
                f"AND status = '{rec.get('status')}' (it only reads 'pending' recs), then "
            )
            logger.error(
                "Not voiding rec %s (match %s, %s %s, status %s): the cancelled row is a duplicate of match %s, "
                "which was played — settle it from the twin: %sdocker compose exec -T api python "
                "/app/scripts/repair_stranded_recs.py --sport <sport> (dry run, then --apply; it reads pending recs "
                "on cancelled rows carrying orphan_of and re-points them onto the twin)",
                rec["rec_id"],
                rec["match_id"],
                rec.get("bet_type"),
                rec.get("selection"),
                rec.get("status"),
                rec["orphan_of"],
                reopen,
            )
            continue
        logger.warning(
            "Voiding rec %s (match %s, %s %s, status %s): the match was cancelled (kickoff %s)",
            rec["rec_id"],
            rec["match_id"],
            rec.get("bet_type"),
            rec.get("selection"),
            rec.get("status"),
            rec.get("kickoff"),
        )
        voided += void_open_rec(cur, rec, "cancelled", CANCELLED_VOID_RESULT)
    return voided, refused


def list_open_recs_for_match(cur, match_id: str) -> list[dict]:
    """Recommendations needing settlement math. Two shapes:
    (1) still in flight (pending/placed) — need status + P&L + settled_at;
    (2) already flipped won/lost/void by the migration-011 match-outcome
        DB TRIGGER, which sets status/actual_result but computes NO
        profit_loss — without this branch those rows stay P&L-less
        forever and every ROI rollup silently understates. The settle
        loop recomputes the same status and fills the money columns.

    `sport` (leagues.sport) and `p_model_name` tell grade_rec_selection
    an NHL rec from a soccer one: NHL double_chance is the regulation
    3-way and must not settle on the final score. LEFT JOINs, so a rec
    never drops out of settlement over a missing league or prediction.

    `after_kickoff` flags a rec written at or after the match's kickoff
    (created_at >= match_date): hindsight, not a bet. grade_match voids
    those instead of grading them, in either shape — the trigger settles
    from the prediction, so a post-kickoff rec can arrive already 'won'.
    The 2026-09-04 stranded-rec repair settled 21 soccer + 2 MMA recs
    written 46 minutes to 24 hours after the real kickoff."""
    cur.execute(
        """
        SELECT br.id::text AS rec_id,
               br.prediction_id::text AS prediction_id,
               br.bet_type, br.selection,
               br.recommended_stake, br.odds_at_recommendation, br.status,
               br.created_at,
               m.match_date AS kickoff,
               (br.created_at >= m.match_date) AS after_kickoff,
               p.actual_outcome AS p_actual_outcome,
               p.is_correct AS p_is_correct,
               p.model_name AS p_model_name,
               l.sport AS sport
        FROM betting_recommendations br
        LEFT JOIN predictions p ON p.id = br.prediction_id
        LEFT JOIN matches m ON m.id = br.match_id
        LEFT JOIN leagues l ON l.id = m.league_id
        WHERE br.match_id = %s
          AND (
                br.status IN ('pending', 'placed')
             OR (br.status IN ('won', 'lost', 'void') AND br.profit_loss IS NULL)
          )
        """,
        (match_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def settle_recommendation(
    cur,
    rec_id: str,
    new_status: str,
    actual_result: Optional[str],
    profit_loss: Decimal,
) -> None:
    """Apply the won/lost/void settlement. settled_at goes to NOW()."""
    cur.execute(
        """
        UPDATE betting_recommendations
        SET status = %s,
            actual_result = %s,
            profit_loss = %s,
            settled_at = NOW(),
            updated_at = NOW()
        WHERE id = %s
        """,
        (new_status, actual_result, profit_loss, rec_id),
    )


# ── Math: profit_loss given outcome + stake + odds ───────────────────


def profit_loss_for(
    is_correct: Optional[bool],
    push: bool,
    stake: Decimal,
    odds: Decimal,
) -> tuple[str, Decimal]:
    """Compute (new_status, profit_loss) from a settled outcome.

    Win:  status='won',  P/L = stake × (odds - 1)
    Loss: status='lost', P/L = -stake
    Push: status='void', P/L = 0
    """
    if push:
        return "void", Decimal("0")
    if is_correct is True:
        return "won", (stake * (odds - Decimal("1"))).quantize(Decimal("0.01"))
    return "lost", (-stake).quantize(Decimal("0.01"))


# ── Orchestration ─────────────────────────────────────────────────────

# compute_prediction_grade's skip reasons. The first two are also the
# `stats` keys grade_match accumulates ('<reason>_skips').
SKIP_AFTER_KICKOFF = "after_kickoff"
SKIP_PHANTOM_LINE = "phantom_line"
SKIP_UNGRADABLE = "ungradable"


def compute_prediction_grade(pred: dict, match: dict, ctx: dict) -> tuple[Optional[str], Optional[str], Optional[bool]]:
    """(skip_reason, actual_outcome, is_correct) for one prediction under
    the CURRENT grading code. skip_reason is None when the row grades
    (is_correct may still be None — a push); otherwise one of the SKIP_*
    reasons and the row must be left as it is.

    `pred` carries prediction_type, model_name, predicted_outcome and,
    when available, probabilities + after_kickoff. `match` is a matches
    row (scores, metadata, half_time_*). `ctx` is
    fetch_match_grading_context's dict. Shared by grade_match (ungraded
    rows) and scripts/regrade_predictions.py (already-graded rows), so the
    two can never disagree about a verdict."""
    # Written after kickoff: never graded (see list_ungraded_predictions).
    if pred.get("after_kickoff"):
        return SKIP_AFTER_KICKOFF, None, None

    # A bout with no winner (an MMA draw / no contest): no pick is right or
    # wrong. The row records what happened, with a NULL verdict, as a push
    # does (grading_outcomes.NO_WINNER_KEY); a kind this code does not know
    # is left alone (grade_match logs it at ERROR).
    no_winner = no_winner_result(match.get("metadata"))
    if no_winner is not None:
        if no_winner not in NO_WINNER_KINDS:
            return SKIP_UNGRADABLE, None, None
        return None, no_winner, None

    # PHANTOM-LINE GUARD. A line-dependent prediction is only
    # gradable if a real book priced that market: otherwise the
    # closing_spread_home / closing_total_line sitting in
    # features_cache is a NEUTRAL_DEFAULT the feature builder
    # invented, and grading against it manufactures a verdict out
    # of nothing. Leave is_correct NULL — neither correct nor
    # incorrect — and move on. (A None probe means the cursor
    # didn't return the column at all; fetch_match_grading_context
    # has already warned about that, and we grade as before.)
    if is_line_dependent(pred["prediction_type"], pred["model_name"]):
        market_has_odds = {"spread": ctx.get("has_spread_odds"), "total": ctx.get("has_total_odds")}
        if market_has_odds.get(pred["prediction_type"]) is False:
            return SKIP_PHANTOM_LINE, None, None

    actual = actual_outcome(
        prediction_type=pred["prediction_type"],
        model_name=pred["model_name"],
        predicted_outcome=pred["predicted_outcome"],
        home_score=match["home_score"],
        away_score=match["away_score"],
        metadata=match["metadata"],
        features=ctx.get("features"),
        halftime=halftime_scores(match),
    )
    if actual is None:
        return SKIP_UNGRADABLE, None, None
    correct = grade_prediction(
        pred["predicted_outcome"],
        actual,
        prediction_type=pred["prediction_type"],
        probabilities=pred.get("probabilities"),
    )
    return None, actual, correct


def grade_match(cur, match: dict, stats: Optional[dict] = None) -> dict:
    """Grade every ungraded prediction + settle every open
    recommendation for one match. Returns counts for the run summary.

    `stats`, when given, accumulates cross-match diagnostics that don't
    belong in the per-match counts — 'phantom_line_skips',
    'after_kickoff_skips', 'post_kickoff_rec_voids' (recs voided
    because they were written at or after kickoff; not counted in
    recs_settled), 'postponed_rec_voids' (recs priced before a postponement
    longer than POSTPONED_VOID_HOURS; not counted either),
    'regulation_regrades' (graded predictions rewritten on the 90-minute
    score), 'regulation_resettles' (money-settled recs re-settled on it),
    'no_winner_predictions' (predictions recorded on a bout with no winner,
    is_correct NULL; not counted in predictions_graded) and
    'no_winner_rec_voids' (recs voided on such a bout; not counted in
    recs_settled).
    run() passes one so it can log a single summary line;
    callers that don't care omit it.
    """
    counts = {"predictions_graded": 0, "predictions_skipped": 0, "recs_settled": 0}
    stats = stats if stats is not None else {}
    meta = match.get("metadata") or {}
    no_winner = no_winner_result(meta)
    if no_winner is not None and no_winner not in NO_WINNER_KINDS:
        logger.error(
            "Match %s is recorded with no winner of an unknown kind (metadata.%s=%r, expected one of %s): none of "
            "its predictions is graded and none of its recs settled — fix the marker, then re-grade it with "
            "--match-id",
            match["match_id"],
            NO_WINNER_KEY,
            no_winner,
            ", ".join(NO_WINNER_KINDS),
        )
    no_winner_recorded = 0
    is_soccer = (match.get("sport") or "soccer") == "soccer"
    # Decided after 90 minutes but without the 90-minute score (the plays
    # could not give it): its full-time markets grade on the final score.
    final_score_fallback = is_soccer and regulation_scores(meta) is None and soccer_went_past_90(meta)
    full_time_graded = 0

    # Features carry the closing line for NBA/NFL spread/total grading,
    # and the odds probes say whether that line was ever real. Fetched
    # once per match instead of per-prediction.
    ctx = fetch_match_grading_context(cur, match["match_id"])

    # First, so the rows graded below are not read back: the trigger's
    # post-extra-time verdicts on a 90-minute-scored match.
    regraded = regrade_on_regulation(cur, match, ctx)
    if regraded:
        stats["regulation_regrades"] = stats.get("regulation_regrades", 0) + regraded
    # The money-settled recs on it, before the open ones below are settled
    # (those are settled with the 90-minute score already).
    resettled = resettle_recs_on_regulation(cur, match)
    if resettled:
        stats["regulation_resettles"] = stats.get("regulation_resettles", 0) + resettled

    for pred in list_ungraded_predictions(cur, match["match_id"]):
        skip, actual, correct = compute_prediction_grade(pred, match, ctx)
        if skip is not None:
            counts["predictions_skipped"] += 1
            if skip in (SKIP_AFTER_KICKOFF, SKIP_PHANTOM_LINE):
                stats[f"{skip}_skips"] = stats.get(f"{skip}_skips", 0) + 1
            continue
        update_prediction(cur, pred["prediction_id"], actual, correct)
        if no_winner is not None:
            # Recorded, not graded: is_correct stays NULL (compute_prediction_grade).
            no_winner_recorded += 1
            continue
        counts["predictions_graded"] += 1
        # A push stays is_correct NULL and is re-read every tick; only a
        # verdict counts, so the warning below fires once per match.
        full_time_graded += correct is not None and pred["prediction_type"] in SOCCER_FT_TYPES
    if no_winner_recorded:
        stats["no_winner_predictions"] = stats.get("no_winner_predictions", 0) + no_winner_recorded
        logger.warning(
            "Match %s ended with no winner (%s; %s): %d prediction(s) recorded with actual_outcome %r and left "
            "ungraded (is_correct NULL) — a two-way pick has no winner to be right or wrong about",
            match["match_id"],
            no_winner,
            meta.get("no_winner_detail") or "no detail",
            no_winner_recorded,
            no_winner,
        )

    # Recs join to predictions for the actual_outcome — the prediction
    # update we just did is visible in the same txn via the join.
    # Settlement grades each rec's OWN (bet_type, selection) against the
    # final score — NOT the prediction's argmax verdict, which is wrong for
    # any non-argmax selection and for every lined market (audit §7
    # 2026-08-06). Ungradable selections stay open rather than guess.
    for rec in list_open_recs_for_match(cur, match["match_id"]):
        # Written at or after kickoff: void before any grading. The score
        # is irrelevant to a bet that should never have existed.
        if rec.get("after_kickoff"):
            logger.warning(
                "Voiding rec %s (match %s, %s %s, status %s): written %s, at or after kickoff %s",
                rec["rec_id"],
                match["match_id"],
                rec.get("bet_type"),
                rec.get("selection"),
                rec.get("status"),
                rec.get("created_at"),
                rec.get("kickoff"),
            )
            settle_recommendation(cur, rec["rec_id"], "void", POST_KICKOFF_VOID_RESULT, Decimal("0"))
            stats["post_kickoff_rec_voids"] = stats.get("post_kickoff_rec_voids", 0) + 1
            continue
        # Priced before a postponement longer than POSTPONED_VOID_HOURS:
        # void, as books void it, whatever the rescheduled match produced.
        if postponed_past_void_limit(match, rec):
            logger.warning(
                "Voiding rec %s (match %s, %s %s, status %s): written %s, before the match was postponed from %s "
                "to %s (more than %dh)",
                rec["rec_id"],
                match["match_id"],
                rec.get("bet_type"),
                rec.get("selection"),
                rec.get("status"),
                rec.get("created_at"),
                meta.get(POSTPONED_KICKOFF_KEY),
                match.get("match_date"),
                POSTPONED_VOID_HOURS,
            )
            settle_recommendation(cur, rec["rec_id"], "void", POSTPONED_VOID_RESULT, Decimal("0"))
            stats["postponed_rec_voids"] = stats.get("postponed_rec_voids", 0) + 1
            continue
        # A bout with no winner: the two-way moneyline is void, stake returned
        # (grading_outcomes.NO_WINNER_KEY: the convention, and the books that
        # differ). Only an open rec: one that is already won/lost/void without
        # a profit_loss was settled by something else and needs a person.
        if no_winner is not None:
            if no_winner not in NO_WINNER_KINDS:
                continue  # logged above, once per match
            if rec.get("status") not in ("pending", "placed"):
                logger.error(
                    "Rec %s (match %s, %s %s) is %r without a profit_loss on a bout that ended with no winner (%s) "
                    "— left as it is: nothing should have settled it; check what did",
                    rec["rec_id"],
                    match["match_id"],
                    rec.get("bet_type"),
                    rec.get("selection"),
                    rec.get("status"),
                    no_winner,
                )
                continue
            logger.warning(
                "Voiding rec %s (match %s, %s %s, status %s): the bout ended with no winner (%s) — a two-way "
                "moneyline is void, stake returned",
                rec["rec_id"],
                match["match_id"],
                rec.get("bet_type"),
                rec.get("selection"),
                rec.get("status"),
                no_winner,
            )
            settle_recommendation(cur, rec["rec_id"], "void", NO_WINNER_VOID_RESULTS[no_winner], Decimal("0"))
            stats["no_winner_rec_voids"] = stats.get("no_winner_rec_voids", 0) + 1
            continue
        if match["home_score"] is None or match["away_score"] is None:
            continue
        outcome = grade_rec_selection(
            rec["bet_type"],
            rec["selection"],
            match["home_score"],
            match["away_score"],
            sport=rec["sport"],
            model_name=rec["p_model_name"],
            metadata=meta,
        )
        if outcome is None:
            continue
        new_status, pl = rec_status_and_pl(outcome, rec["recommended_stake"], rec["odds_at_recommendation"])
        settle_recommendation(cur, rec["rec_id"], new_status, outcome, pl)
        counts["recs_settled"] += 1

    if final_score_fallback and (full_time_graded or counts["recs_settled"]):
        # Once per match that graded something, not every tick.
        logger.warning(
            "Match %s was decided after 90 minutes (%s) but carries no 90-minute score: %d full-time "
            "prediction(s) and %d rec(s) graded on the final, post-extra-time %s-%s. Stamp it with "
            "scripts/backfill_soccer_regulation_scores.py, then re-grade.",
            match["match_id"],
            meta.get("result_detail"),
            full_time_graded,
            counts["recs_settled"],
            match.get("home_score"),
            match.get("away_score"),
        )
    return counts


STAT_KEYS = (
    "phantom_line_skips",
    "after_kickoff_skips",
    "post_kickoff_rec_voids",
    "postponed_rec_voids",
    "cancelled_rec_voids",
    "cancelled_orphan_recs",
    "regulation_regrades",
    "regulation_resettles",
    "no_winner_predictions",
    "no_winner_rec_voids",
)


def run(database_url: str, days: int, match_id: Optional[str]) -> dict:
    summary = {
        "matches_processed": 0,
        "predictions_graded": 0,
        "predictions_skipped": 0,
        "recs_settled": 0,
        **{k: 0 for k in STAT_KEYS},
    }
    stats: dict = {k: 0 for k in STAT_KEYS}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if match_id is None:
                # Not tied to a finished match, so not in grade_match.
                stats["postponed_rec_voids"] += void_recs_on_long_postponed_matches(cur)
                voided, refused = void_recs_on_cancelled_matches(cur)
                stats["cancelled_rec_voids"] += voided
                stats["cancelled_orphan_recs"] += refused
            matches = list_finished_matches(cur, days, match_id)
            if not matches:
                logger.info("No finished matches needing grading in the last %d days", days)
            else:
                logger.info("Grading %d finished matches", len(matches))
            for m in matches:
                per = grade_match(cur, m, stats)
                summary["matches_processed"] += 1
                for k in ("predictions_graded", "predictions_skipped", "recs_settled"):
                    summary[k] += per[k]
            conn.commit()
    for k in STAT_KEYS:
        summary[k] = stats[k]
    if stats["postponed_rec_voids"]:
        logger.warning(
            "Voided %d recommendation(s) on matches postponed more than %dh (%r)",
            stats["postponed_rec_voids"],
            POSTPONED_VOID_HOURS,
            POSTPONED_VOID_RESULT,
        )
    if stats["cancelled_rec_voids"]:
        logger.warning(
            "Voided %d recommendation(s) on matches cancelled more than %dh ago (%r)",
            stats["cancelled_rec_voids"],
            CANCELLED_VOID_HOURS,
            CANCELLED_VOID_RESULT,
        )
    if stats["cancelled_orphan_recs"]:
        logger.error(
            "Left %d open recommendation(s) on cancelled duplicates of played fixtures (metadata.%s): settle them "
            "from the twin with scripts/repair_stranded_recs.py --sport <sport> (it reads pending recs on those "
            "rows; set a 'placed' one back to 'pending' first — the per-rec ERROR lines above give the commands)",
            stats["cancelled_orphan_recs"],
            ORPHAN_OF_KEY,
        )
    if stats["regulation_regrades"]:
        logger.warning(
            "Re-graded %d full-time prediction(s) on the 90-minute score of matches decided after 90 minutes",
            stats["regulation_regrades"],
        )
    if stats["regulation_resettles"]:
        logger.error(
            "Re-settled %d recommendation(s) that had settled on the post-extra-time score (ids above)",
            stats["regulation_resettles"],
        )
    if stats["no_winner_rec_voids"]:
        logger.warning(
            "Voided %d recommendation(s) on bouts that ended with no winner (a draw or a no contest; %s)",
            stats["no_winner_rec_voids"],
            " / ".join(repr(v) for v in NO_WINNER_VOID_RESULTS.values()),
        )
    if stats["no_winner_predictions"]:
        logger.warning(
            "Recorded %d prediction(s) on bouts that ended with no winner, left ungraded (is_correct NULL)",
            stats["no_winner_predictions"],
        )
    if stats["post_kickoff_rec_voids"]:
        # ERROR, not WARNING: rec generation must never write a rec for a
        # match that has started, so each of these is an upstream defect
        # (or a kickoff ESPN moved earlier) worth chasing.
        logger.error(
            "Voided %d recommendation(s) written at or after kickoff instead of settling them (%r)",
            stats["post_kickoff_rec_voids"],
            POST_KICKOFF_VOID_RESULT,
        )
    if stats["after_kickoff_skips"]:
        logger.warning(
            "Left %d prediction(s) ungraded because they were written after kickoff",
            stats["after_kickoff_skips"],
        )
    if stats["phantom_line_skips"]:
        # One summary line, not one per prediction. These rows stay
        # is_correct NULL on purpose: the market was never priced, so
        # the line in features_cache is a NEUTRAL_DEFAULT, not a
        # closing line.
        logger.warning(
            "Phantom-line guard: left %d spread/total prediction(s) ungraded "
            "because the match has no non-live odds in that market",
            stats["phantom_line_skips"],
        )
    logger.info(
        "Done. matches=%d, predictions_graded=%d (skipped=%d), recs_settled=%d",
        summary["matches_processed"],
        summary["predictions_graded"],
        summary["predictions_skipped"],
        summary["recs_settled"],
    )
    return summary


def warn_stuck_post_kickoff_recs(database_url: str, days: int) -> int:
    """Canary, run after every grading run. Migration 027 leaves a rec
    written at or after kickoff 'pending' for grade_match to void, but
    grade_match only visits matches whose kickoff is inside the --days
    window: a match marked finished after it aged out of that window (a
    results backlog, a repaired fixture) would hold such a rec open forever,
    counted in the dashboard's active_recs, and void_post_kickoff_recs.py
    only selects settled recs. Logs ERROR with the count, the first 10 rec
    ids and the command that voids them; returns the count (0: logs nothing).
    Joins on the rec's own match_id with the `>=` comparison, as
    list_open_recs_for_match does."""
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT br.id::text AS rec_id,
                       COUNT(*) OVER () AS n,
                       CEIL(EXTRACT(EPOCH FROM NOW() - MIN(m.match_date) OVER ()) / 86400)::int + 1 AS days_needed
                FROM betting_recommendations br
                JOIN matches m ON m.id = br.match_id
                WHERE br.status IN ('pending', 'placed')
                  AND m.status = 'finished'
                  AND br.created_at >= m.match_date
                  AND m.match_date < NOW() - (%s || ' days')::interval
                ORDER BY m.match_date, br.id
                LIMIT 10
                """,
                (str(days),),
            )
            rows = cur.fetchall()
    if not rows:
        return 0
    logger.error(
        "%d pending/placed rec(s) written at or after kickoff sit on finished matches older than the --days %d "
        "window, so no scheduled run will void them (first %d: %s). Void them with: "
        "python /app/scripts/grade_completed_matches.py --days %d",
        rows[0]["n"],
        days,
        len(rows),
        ", ".join(r["rec_id"] for r in rows),
        rows[0]["days_needed"],
    )
    return int(rows[0]["n"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--days",
        type=int,
        default=14,
        help="Lookback window for finished matches (default 14).",
    )
    p.add_argument(
        "--match-id",
        default=None,
        help="Grade ONE specific match by UUID (operator escape hatch).",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set")
        return 2
    run(args.database_url, args.days, args.match_id)
    warn_stuck_post_kickoff_recs(args.database_url, args.days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
