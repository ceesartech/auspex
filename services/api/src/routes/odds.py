"""Odds endpoints"""

import logging
from typing import Any, Dict, List, Optional

from auth.dependencies import require_auth
from database import get_db
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/live")
async def get_live_odds(
    sport: Optional[str] = Query(None),
    bookmaker: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> List[Dict[str, Any]]:
    """Get live odds for in-play matches"""

    query = text("""
        SELECT o.id, o.match_id, o.bookmaker, o.market_type, o.selection,
               o.odds_decimal, o.implied_probability, o.timestamp,
               m.match_date,
               ht.name as home_team, at.name as away_team,
               l.name as league_name
        FROM odds o
        JOIN matches m ON o.match_id = m.id
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        JOIN leagues l ON m.league_id = l.id
        WHERE o.is_live = true
        AND m.status = 'live'
        AND (:sport IS NULL OR l.sport = :sport)
        AND (:bookmaker IS NULL OR o.bookmaker = :bookmaker)
        ORDER BY o.timestamp DESC
        LIMIT :limit
    """)

    results = db.execute(query, {"sport": sport, "bookmaker": bookmaker, "limit": limit}).fetchall()

    return [
        {
            "odds_id": str(row.id),
            "match_id": str(row.match_id),
            "home_team": row.home_team,
            "away_team": row.away_team,
            "league_name": row.league_name,
            "bookmaker": row.bookmaker,
            "market_type": row.market_type,
            "selection": row.selection,
            "odds_decimal": float(row.odds_decimal),
            "implied_probability": float(row.implied_probability) if row.implied_probability else None,
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
        }
        for row in results
    ]


@router.get("/match/{match_id}")
async def get_match_odds(
    match_id: str,
    market_type: Optional[str] = Query(None),
    bookmaker: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> List[Dict[str, Any]]:
    """Get all odds for a specific match"""

    query = text("""
        SELECT o.id, o.bookmaker, o.market_type, o.selection,
               o.odds_decimal, o.odds_american, o.line,
               o.implied_probability, o.timestamp,
               o.is_opening, o.is_live
        FROM odds o
        WHERE o.match_id = :match_id
        AND (:market_type IS NULL OR o.market_type = :market_type)
        AND (:bookmaker IS NULL OR o.bookmaker = :bookmaker)
        ORDER BY o.market_type, o.bookmaker, o.timestamp DESC
    """)

    results = db.execute(
        query,
        {"match_id": match_id, "market_type": market_type, "bookmaker": bookmaker},
    ).fetchall()

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No odds found for this match",
        )

    return [
        {
            "odds_id": str(row.id),
            "bookmaker": row.bookmaker,
            "market_type": row.market_type,
            "selection": row.selection,
            "odds_decimal": float(row.odds_decimal),
            "odds_american": row.odds_american,
            "line": float(row.line) if row.line else None,
            "implied_probability": float(row.implied_probability) if row.implied_probability else None,
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "is_opening": row.is_opening,
            "is_live": row.is_live,
        }
        for row in results
    ]


@router.get("/movements/{match_id}")
async def get_odds_movements(
    match_id: str,
    market_type: str = Query("1x2"),
    bookmaker: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """Get odds movement history for a match"""

    query = text("""
        SELECT o.selection, o.odds_decimal, o.implied_probability,
               o.timestamp, o.bookmaker, o.is_opening
        FROM odds o
        WHERE o.match_id = :match_id
        AND o.market_type = :market_type
        AND (:bookmaker IS NULL OR o.bookmaker = :bookmaker)
        ORDER BY o.selection, o.timestamp ASC
    """)

    results = db.execute(
        query,
        {"match_id": match_id, "market_type": market_type, "bookmaker": bookmaker},
    ).fetchall()

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No odds movements found",
        )

    # Group by selection
    movements = {}
    for row in results:
        selection = row.selection
        if selection not in movements:
            movements[selection] = []

        movements[selection].append(
            {
                "odds_decimal": float(row.odds_decimal),
                "implied_probability": float(row.implied_probability) if row.implied_probability else None,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "bookmaker": row.bookmaker,
                "is_opening": row.is_opening,
            }
        )

    # Calculate movement summary
    summary = {}
    for selection, history in movements.items():
        if len(history) >= 2:
            opening = history[0]["odds_decimal"]
            current = history[-1]["odds_decimal"]
            summary[selection] = {
                "opening_odds": opening,
                "current_odds": current,
                "change": round(current - opening, 4),
                "change_pct": round((current - opening) / opening * 100, 2),
                "direction": "shortening" if current < opening else "drifting" if current > opening else "stable",
                "data_points": len(history),
            }

    return {
        "match_id": match_id,
        "market_type": market_type,
        "movements": movements,
        "summary": summary,
    }
