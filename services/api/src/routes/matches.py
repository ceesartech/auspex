"""Match endpoints"""

import logging
from typing import Any, Dict, List, Optional

from auth.dependencies import require_auth
from database import get_db
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/upcoming")
async def get_upcoming_matches(
    sport: Optional[str] = Query(None),
    league: Optional[str] = Query(None),
    days_ahead: int = Query(7, ge=1, le=30),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> List[Dict[str, Any]]:
    """Get upcoming matches with latest odds from the view"""

    query = text(
        """
        SELECT match_id, match_date, league_name, league_country,
               home_team, away_team, venue,
               home_odds, draw_odds, away_odds, bookmaker
        FROM vw_upcoming_matches_with_odds
        WHERE match_date <= NOW() + make_interval(days => :days_ahead)
        AND (:league IS NULL OR league_name = :league)
        AND (:sport IS NULL OR league_country IS NOT NULL)
        ORDER BY match_date ASC
        LIMIT :limit
    """
    )

    results = db.execute(
        query,
        {"days_ahead": days_ahead, "league": league, "sport": sport, "limit": limit},
    ).fetchall()

    matches = []
    for row in results:
        matches.append(
            {
                "match_id": str(row.match_id),
                "match_date": row.match_date.isoformat(),
                "league_name": row.league_name,
                "league_country": row.league_country,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "venue": row.venue,
                "odds": {
                    "home": float(row.home_odds) if row.home_odds else None,
                    "draw": float(row.draw_odds) if row.draw_odds else None,
                    "away": float(row.away_odds) if row.away_odds else None,
                    "bookmaker": row.bookmaker,
                },
            }
        )

    return matches


@router.get("/{match_id}")
async def get_match_detail(
    match_id: str,
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """Get detailed match information"""

    query = text(
        """
        SELECT m.id, m.match_date, m.status, m.home_score, m.away_score,
               m.half_time_home, m.half_time_away, m.season, m.round,
               m.venue, m.attendance,
               l.name as league_name, l.country as league_country, l.sport,
               ht.name as home_team, ht.id as home_team_id,
               at.name as away_team, at.id as away_team_id,
               r.name as referee_name
        FROM matches m
        JOIN leagues l ON m.league_id = l.id
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        LEFT JOIN referees r ON m.referee_id = r.id
        WHERE m.id = :match_id
    """
    )

    result = db.execute(query, {"match_id": match_id}).fetchone()

    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Match not found")

    return {
        "match_id": str(result.id),
        "match_date": result.match_date.isoformat(),
        "status": result.status,
        "home_score": result.home_score,
        "away_score": result.away_score,
        "half_time_home": result.half_time_home,
        "half_time_away": result.half_time_away,
        "season": result.season,
        "round": result.round,
        "venue": result.venue,
        "attendance": result.attendance,
        "league": {
            "name": result.league_name,
            "country": result.league_country,
            "sport": result.sport,
        },
        "home_team": {"id": str(result.home_team_id), "name": result.home_team},
        "away_team": {"id": str(result.away_team_id), "name": result.away_team},
        "referee": result.referee_name,
    }


@router.get("/{match_id}/stats")
async def get_match_stats(
    match_id: str,
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """Get match statistics and team form"""

    # Get match stats
    stats_query = text(
        """
        SELECT ms.*, t.name as team_name
        FROM match_stats ms
        JOIN teams t ON ms.team_id = t.id
        WHERE ms.match_id = :match_id
    """
    )

    stats_results = db.execute(stats_query, {"match_id": match_id}).fetchall()

    # Get team IDs for form lookup
    match_query = text(
        """
        SELECT home_team_id, away_team_id FROM matches WHERE id = :match_id
    """
    )
    match_result = db.execute(match_query, {"match_id": match_id}).fetchone()

    if not match_result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Match not found")

    # Get team form using DB function
    home_form = _get_team_form(db, str(match_result.home_team_id))
    away_form = _get_team_form(db, str(match_result.away_team_id))

    # Get H2H using DB function
    h2h = _get_h2h_stats(db, str(match_result.home_team_id), str(match_result.away_team_id))

    stats = {}
    for row in stats_results:
        stats[row.team_name] = {
            "possession": float(row.possession) if row.possession else None,
            "shots": row.shots,
            "shots_on_target": row.shots_on_target,
            "expected_goals": float(row.expected_goals) if row.expected_goals else None,
            "passes": row.passes,
            "pass_accuracy": float(row.pass_accuracy) if row.pass_accuracy else None,
            "corners": row.corners,
            "fouls": row.fouls,
            "yellow_cards": row.yellow_cards,
            "red_cards": row.red_cards,
            "tackles": row.tackles,
            "interceptions": row.interceptions,
        }

    return {
        "match_id": match_id,
        "match_stats": stats,
        "home_form": home_form,
        "away_form": away_form,
        "head_to_head": h2h,
    }


@router.get("/{match_id}/odds")
async def get_match_odds(
    match_id: str,
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> List[Dict[str, Any]]:
    """Get all odds for a match"""

    query = text(
        """
        SELECT id, bookmaker, market_type, selection, odds_decimal,
               odds_american, line, implied_probability,
               timestamp, is_opening, is_live
        FROM odds
        WHERE match_id = :match_id
        ORDER BY timestamp DESC
    """
    )

    results = db.execute(query, {"match_id": match_id}).fetchall()

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


def _get_team_form(db: Session, team_id: str, num_matches: int = 5) -> List[Dict]:
    """Call get_team_form() database function"""
    query = text(
        """
        SELECT match_id, match_date, opponent_id, is_home,
               goals_for, goals_against, result, points
        FROM get_team_form(:team_id, :num_matches)
    """
    )

    results = db.execute(query, {"team_id": team_id, "num_matches": num_matches}).fetchall()

    return [
        {
            "match_id": str(row.match_id),
            "match_date": row.match_date.isoformat(),
            "is_home": row.is_home,
            "goals_for": row.goals_for,
            "goals_against": row.goals_against,
            "result": row.result,
            "points": row.points,
        }
        for row in results
    ]


def _get_h2h_stats(db: Session, team1_id: str, team2_id: str, num_matches: int = 10) -> List[Dict]:
    """Call get_h2h_stats() database function"""
    query = text(
        """
        SELECT match_id, match_date, home_team_id, away_team_id,
               home_score, away_score, league_name
        FROM get_h2h_stats(:team1_id, :team2_id, :num_matches)
    """
    )

    results = db.execute(
        query,
        {"team1_id": team1_id, "team2_id": team2_id, "num_matches": num_matches},
    ).fetchall()

    return [
        {
            "match_id": str(row.match_id),
            "match_date": row.match_date.isoformat(),
            "home_team_id": str(row.home_team_id),
            "away_team_id": str(row.away_team_id),
            "home_score": row.home_score,
            "away_score": row.away_score,
            "league_name": row.league_name,
        }
        for row in results
    ]
