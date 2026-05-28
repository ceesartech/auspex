"""User endpoints - preferences, betting history, dashboard"""

import json
import logging
from typing import Any, Dict, Optional

from auth.dependencies import require_auth
from auth.jwt_handler import create_access_token, verify_password
from database import get_db
from fastapi import APIRouter, Depends, HTTPException, Query, status
from models.requests import BettingHistoryRecord, LoginRequest, UserPreferencesUpdate
from sqlalchemy import text
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Authenticate by username-or-email + password.

    Returns the same 401 for "unknown user" and "wrong password" so the
    endpoint doesn't leak which usernames exist.
    """

    row = db.execute(
        text(
            """
            SELECT id, username, email, password_hash, role, is_active
            FROM users
            WHERE LOWER(username) = LOWER(:identifier)
               OR LOWER(email) = LOWER(:identifier)
            LIMIT 1
            """
        ),
        {"identifier": request.username},
    ).fetchone()

    invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
    )

    if row is None or not row.is_active:
        # Still call verify_password against a dummy hash so the response
        # time doesn't fingerprint "user exists vs. user missing".
        verify_password(request.password, "$2b$12$" + "x" * 53)
        raise invalid

    if not verify_password(request.password, row.password_hash):
        raise invalid

    db.execute(
        text("UPDATE users SET last_login = NOW() WHERE id = :id"),
        {"id": row.id},
    )
    db.commit()

    token = create_access_token(
        data={
            "user_id": str(row.id),
            "username": row.username,
            "role": row.role,
        }
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "username": row.username,
            "email": row.email,
            "role": row.role,
        },
    }


@router.get("/preferences")
async def get_preferences(
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """Get all user preferences"""

    query = text(
        """
        SELECT preference_key, preference_value, description
        FROM user_preferences
        ORDER BY preference_key
    """
    )

    results = db.execute(query).fetchall()

    preferences = {}
    for row in results:
        preferences[row.preference_key] = {
            "value": row.preference_value,
            "description": row.description,
        }

    return preferences


@router.put("/preferences")
async def update_preferences(
    updates: UserPreferencesUpdate,
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """Update user preferences"""

    # Use SQL-standard CAST(); ":value::jsonb" confuses SQLAlchemy's named-
    # parameter parser (the trailing "::" looks like part of the bind name).
    upsert_query = text(
        """
        INSERT INTO user_preferences (preference_key, preference_value, description)
        VALUES (:key, CAST(:value AS jsonb), :description)
        ON CONFLICT (preference_key) DO UPDATE
        SET preference_value = EXCLUDED.preference_value,
            updated_at = NOW()
    """
    )

    update_data = updates.model_dump(exclude_none=True)

    descriptions = {
        "bankroll": "Total bankroll amount",
        "max_stake_per_bet": "Maximum stake per individual bet",
        "risk_tolerance": "Risk tolerance level (LOW/MEDIUM/HIGH)",
        "confidence_threshold": "Minimum confidence for recommendations",
        "kelly_fraction": "Kelly criterion fraction multiplier",
        "favorite_sports": "Preferred sports for recommendations",
        "notification_enabled": "Enable/disable notifications",
    }

    for key, value in update_data.items():
        # Store the raw value as JSON. The GET endpoint already wraps the
        # JSONB column under a "value" key in its response, so wrapping
        # again here would produce {"value": {"value": X}} on read.
        db.execute(
            upsert_query,
            {
                "key": key,
                "value": json.dumps(value),
                "description": descriptions.get(key, ""),
            },
        )

    db.commit()

    return {"status": "updated", "updated_fields": list(update_data.keys())}


@router.get("/betting-history")
async def get_betting_history(
    status_filter: Optional[str] = Query(None, alias="status"),
    bookmaker: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """Get user betting history with P&L"""

    query = text(
        """
        SELECT bh.id, bh.bookmaker, bh.bet_type, bh.selection,
               bh.odds, bh.stake, bh.potential_return, bh.actual_return,
               bh.status, bh.notes, bh.placed_at, bh.settled_at,
               m.match_date,
               ht.name as home_team, at.name as away_team,
               l.name as league_name
        FROM betting_history bh
        LEFT JOIN matches m ON bh.match_id = m.id
        LEFT JOIN teams ht ON m.home_team_id = ht.id
        LEFT JOIN teams at ON m.away_team_id = at.id
        LEFT JOIN leagues l ON m.league_id = l.id
        WHERE (:status_filter IS NULL OR bh.status = :status_filter)
        AND (:bookmaker IS NULL OR bh.bookmaker = :bookmaker)
        ORDER BY bh.placed_at DESC
        LIMIT :limit OFFSET :offset
    """
    )

    results = db.execute(
        query,
        {
            "status_filter": status_filter,
            "bookmaker": bookmaker,
            "limit": limit,
            "offset": offset,
        },
    ).fetchall()

    # Get total count
    count_query = text(
        """
        SELECT COUNT(*) as total FROM betting_history
        WHERE (:status_filter IS NULL OR status = :status_filter)
        AND (:bookmaker IS NULL OR bookmaker = :bookmaker)
    """
    )
    count_result = db.execute(count_query, {"status_filter": status_filter, "bookmaker": bookmaker}).fetchone()
    total = count_result.total if count_result is not None else 0

    bets = []
    for row in results:
        bets.append(
            {
                "id": str(row.id),
                "bookmaker": row.bookmaker,
                "bet_type": row.bet_type,
                "selection": row.selection,
                "odds": float(row.odds),
                "stake": float(row.stake),
                "potential_return": float(row.potential_return) if row.potential_return else None,
                "actual_return": float(row.actual_return) if row.actual_return else None,
                "status": row.status,
                "notes": row.notes,
                "placed_at": row.placed_at.isoformat() if row.placed_at else None,
                "settled_at": row.settled_at.isoformat() if row.settled_at else None,
                "match": (
                    {
                        "home_team": row.home_team,
                        "away_team": row.away_team,
                        "match_date": row.match_date.isoformat() if row.match_date else None,
                        "league": row.league_name,
                    }
                    if row.home_team
                    else None
                ),
            }
        )

    return {"bets": bets, "total": total, "limit": limit, "offset": offset}


@router.post("/betting-history")
async def record_bet(
    bet: BettingHistoryRecord,
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """Record a placed bet"""

    query = text(
        """
        INSERT INTO betting_history
        (recommendation_id, match_id, bookmaker, bet_type, selection,
         odds, stake, potential_return, notes)
        VALUES (:recommendation_id, :match_id, :bookmaker, :bet_type,
                :selection, :odds, :stake, :potential_return, :notes)
        RETURNING id
    """
    )

    result = db.execute(
        query,
        {
            "recommendation_id": bet.recommendation_id,
            "match_id": bet.match_id,
            "bookmaker": bet.bookmaker,
            "bet_type": bet.bet_type,
            "selection": bet.selection,
            "odds": bet.odds,
            "stake": bet.stake,
            "potential_return": round(bet.odds * bet.stake, 2),
            "notes": bet.notes,
        },
    ).fetchone()

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record bet",
        )

    db.commit()

    return {"id": str(result.id), "status": "recorded"}


@router.get("/dashboard")
async def get_dashboard(
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """Get user dashboard summary statistics"""

    stats_query = text(
        """
        SELECT
            COUNT(*) as total_bets,
            COUNT(*) FILTER (WHERE status = 'pending') as active_bets,
            COALESCE(SUM(stake), 0) as total_staked,
            COALESCE(SUM(actual_return), 0) as total_returns,
            COALESCE(SUM(actual_return) - SUM(stake), 0) as profit_loss,
            CASE WHEN SUM(stake) > 0
                THEN ROUND(
                    ((COALESCE(SUM(actual_return), 0) - SUM(stake)) / SUM(stake) * 100)::numeric,
                    2
                )
                ELSE 0 END as roi_pct,
            CASE WHEN COUNT(*) FILTER (WHERE status IN ('won', 'lost')) > 0
                THEN ROUND((COUNT(*) FILTER (WHERE status = 'won')::numeric /
                     COUNT(*) FILTER (WHERE status IN ('won', 'lost')) * 100)::numeric, 2)
                ELSE 0 END as win_rate
        FROM betting_history
    """
    )

    stats = db.execute(stats_query).fetchone()
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load dashboard statistics",
        )

    # Count active recommendations
    rec_query = text(
        """
        SELECT COUNT(*) as count FROM betting_recommendations
        WHERE status IN ('pending', 'placed')
    """
    )
    active_recs_result = db.execute(rec_query).fetchone()
    active_recs = active_recs_result.count if active_recs_result is not None else 0

    # Count upcoming matches
    upcoming_query = text(
        """
        SELECT COUNT(*) as count FROM matches
        WHERE status = 'scheduled' AND match_date > NOW()
    """
    )
    upcoming_result = db.execute(upcoming_query).fetchone()
    upcoming = upcoming_result.count if upcoming_result is not None else 0

    return {
        "total_bets": stats.total_bets,
        "active_bets": stats.active_bets,
        "total_staked": float(stats.total_staked),
        "total_returns": float(stats.total_returns),
        "profit_loss": float(stats.profit_loss),
        "roi_pct": float(stats.roi_pct),
        "win_rate": float(stats.win_rate),
        "active_recommendations": active_recs,
        "upcoming_matches": upcoming,
    }
