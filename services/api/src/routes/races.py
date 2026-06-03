"""Horse-racing race-card endpoints.

Parallel to scripts/precompute_predictions_horse_racing.py +
scripts/generate_recommendations_horse_racing.py — both read/write
the races / race_entrants / race_predictions / race_recommendations
tables introduced in migration 013.

Two endpoints:
  GET /api/v1/races                    — list (race-card index)
  GET /api/v1/races/{race_id}          — detail (per-entrant view)

The list endpoint returns a lightweight summary suitable for a
race-card grid; the detail endpoint joins in every entrant with the
consensus prediction confidence + best bookmaker odds + the active
value-bet recommendation when one fired.

Horse racing has its OWN schema for multi-runner shapes (matches +
predictions are 2-team), so we intentionally don't reuse the
matches / predictions / recommendations routers — duplicating a
race_id::text into match_id would either break the team-sport
contracts or force every consumer to handle two shapes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from auth.dependencies import require_auth
from database import get_db
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("")
async def list_races(
    status_filter: Optional[str] = Query(
        None, alias="status", description="Filter by status: scheduled / live / finished."
    ),
    days_ahead: int = Query(2, ge=1, le=7, description="Lookahead window in days."),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> List[Dict[str, Any]]:
    """Race-card index. Default: every scheduled race in the next
    `days_ahead` days. Pass status='finished' to see graded races
    (lookback equals `days_ahead`); status='live' for races that
    have flipped to live status.

    Each row carries the race-level metadata + a count of how many
    value-bet recommendations fired for the race so the UI can badge
    "3 picks" without a second roundtrip."""

    if status_filter and status_filter not in ("scheduled", "live", "finished"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status: {status_filter!r}",
        )

    # 'finished' uses a lookback; 'scheduled' / 'live' use a lookahead.
    # Default (no filter) → scheduled future races.
    if status_filter == "finished":
        time_clause = (
            "r.race_date BETWEEN NOW() - make_interval(days => :window) AND NOW()"
        )
    else:
        time_clause = (
            "r.race_date BETWEEN NOW() AND NOW() + make_interval(days => :window)"
        )

    status_clause = "r.status = :status_filter" if status_filter else "r.status = 'scheduled'"

    query = text(
        f"""
        SELECT
            r.id::text                  AS race_id,
            r.race_date,
            r.track_name,
            r.race_number,
            r.distance_meters,
            r.surface,
            r.track_condition,
            r.race_class,
            r.purse_currency,
            r.purse_amount,
            r.field_size,
            r.status,
            (SELECT COUNT(*) FROM race_entrants e
              WHERE e.race_id = r.id AND NOT e.scratched) AS runners,
            (SELECT COUNT(*) FROM race_recommendations rec
              WHERE rec.race_id = r.id AND rec.bet_type = 'win'
                AND rec.status IN ('pending', 'placed', 'won'))
                                        AS recommendation_count
        FROM races r
        WHERE {status_clause}
          AND {time_clause}
        ORDER BY r.race_date ASC
        LIMIT :limit
        """
    )

    rows = db.execute(
        query,
        {
            "status_filter": status_filter,
            "window": days_ahead,
            "limit": limit,
        },
    ).fetchall()

    return [
        {
            "race_id": row.race_id,
            "race_date": row.race_date.isoformat(),
            "track_name": row.track_name,
            "race_number": row.race_number,
            "distance_meters": row.distance_meters,
            "surface": row.surface,
            "track_condition": row.track_condition,
            "race_class": row.race_class,
            "purse_currency": row.purse_currency,
            "purse_amount": float(row.purse_amount) if row.purse_amount is not None else None,
            # field_size comes from the racecard payload; for /results-
            # loaded rows it's NULL. Derive from race_entrants when
            # missing so the UI always has a number to show.
            "field_size": int(row.field_size) if row.field_size is not None else int(row.runners or 0),
            "runners": int(row.runners or 0),
            "status": row.status,
            "recommendation_count": int(row.recommendation_count or 0),
        }
        for row in rows
    ]


@router.get("/{race_id}")
async def get_race_detail(
    race_id: str,
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """Race + entrants + per-entrant prediction + recommendation
    (when one fired) for a single race. The shape:

      {
        race: {...race-level metadata...},
        entrants: [
          {
            entrant_id, program_number, horse_name, jockey_name,
            trainer_name, weight_carried_lbs, morning_line_odds,
            starting_price, finish_position, scratched,
            consensus_prob,        # NULL if no prediction stored
            consensus_field_probs, # full {entrant_id → prob} dict
            recommendation: {
              odds, bookmaker, expected_value, recommended_stake,
              confidence_rating, status, profit_loss
            } | None
          },
          ...
        ]
      }
    """

    race_row = db.execute(
        text(
            """
            SELECT
                r.id::text          AS race_id,
                r.race_date,
                r.track_name,
                r.race_number,
                r.distance_meters,
                r.surface,
                r.track_condition,
                r.race_class,
                r.purse_currency,
                r.purse_amount,
                r.field_size,
                r.status
            FROM races r
            WHERE r.id::text = :race_id
            """
        ),
        {"race_id": race_id},
    ).fetchone()
    if not race_row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Race {race_id} not found",
        )

    entrant_rows = db.execute(
        text(
            """
            SELECT
                e.id::text             AS entrant_id,
                e.program_number,
                e.post_position,
                e.weight_carried_lbs,
                e.morning_line_odds,
                e.starting_price,
                e.finish_position,
                e.disqualified,
                e.scratched,
                h.name                 AS horse_name,
                j.name                 AS jockey_name,
                t.name                 AS trainer_name,
                rp.id::text            AS prediction_id,
                rp.confidence          AS consensus_prob,
                rp.probabilities       AS consensus_field_probs,
                rp.actual_outcome,
                rp.is_correct,
                rec.id::text           AS recommendation_id,
                rec.odds_at_recommendation,
                rec.bookmaker,
                rec.expected_value,
                rec.recommended_stake,
                rec.confidence_rating,
                rec.status             AS rec_status,
                rec.profit_loss
            FROM race_entrants e
            JOIN horses h ON h.id = e.horse_id
            LEFT JOIN jockeys j ON j.id = e.jockey_id
            LEFT JOIN trainers t ON t.id = e.trainer_id
            LEFT JOIN race_predictions rp
                ON rp.entrant_id = e.id
               AND rp.model_name = 'market_consensus_v1'
               AND rp.prediction_type = 'win'
            LEFT JOIN race_recommendations rec
                ON rec.entrant_id = e.id
               AND rec.bet_type = 'win'
            WHERE e.race_id = :race_id
            ORDER BY e.program_number ASC
            """
        ),
        {"race_id": race_id},
    ).fetchall()

    entrants = []
    for row in entrant_rows:
        recommendation: Optional[Dict[str, Any]] = None
        if row.recommendation_id is not None:
            recommendation = {
                "recommendation_id": row.recommendation_id,
                "odds": float(row.odds_at_recommendation)
                if row.odds_at_recommendation is not None
                else None,
                "bookmaker": row.bookmaker,
                "expected_value": float(row.expected_value)
                if row.expected_value is not None
                else None,
                "recommended_stake": float(row.recommended_stake)
                if row.recommended_stake is not None
                else None,
                "confidence_rating": row.confidence_rating,
                "status": row.rec_status,
                "profit_loss": float(row.profit_loss)
                if row.profit_loss is not None
                else None,
            }
        entrants.append(
            {
                "entrant_id": row.entrant_id,
                "program_number": row.program_number,
                "post_position": row.post_position,
                "weight_carried_lbs": float(row.weight_carried_lbs)
                if row.weight_carried_lbs is not None
                else None,
                "morning_line_odds": float(row.morning_line_odds)
                if row.morning_line_odds is not None
                else None,
                "starting_price": float(row.starting_price)
                if row.starting_price is not None
                else None,
                "finish_position": row.finish_position,
                "disqualified": row.disqualified,
                "scratched": row.scratched,
                "horse_name": row.horse_name,
                "jockey_name": row.jockey_name,
                "trainer_name": row.trainer_name,
                "consensus_prob": float(row.consensus_prob)
                if row.consensus_prob is not None
                else None,
                "consensus_field_probs": row.consensus_field_probs,
                "actual_outcome": float(row.actual_outcome)
                if row.actual_outcome is not None
                else None,
                "is_correct": row.is_correct,
                "recommendation": recommendation,
            }
        )

    return {
        "race": {
            "race_id": race_row.race_id,
            "race_date": race_row.race_date.isoformat(),
            "track_name": race_row.track_name,
            "race_number": race_row.race_number,
            "distance_meters": race_row.distance_meters,
            "surface": race_row.surface,
            "track_condition": race_row.track_condition,
            "race_class": race_row.race_class,
            "purse_currency": race_row.purse_currency,
            "purse_amount": float(race_row.purse_amount)
            if race_row.purse_amount is not None
            else None,
            "field_size": int(race_row.field_size)
            if race_row.field_size is not None
            else len(entrants),
            "status": race_row.status,
        },
        "entrants": entrants,
    }
