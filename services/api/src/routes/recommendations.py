"""Betting recommendation endpoints"""

import logging
from typing import List, Optional

from auth.dependencies import require_auth
from database import get_db
from fastapi import APIRouter, Depends, HTTPException, Query, status
from models.requests import AccumulatorRequest
from models.responses import AccumulatorResponse, RecommendationResponse
from services.recommendation_service import RecommendationService
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=List[RecommendationResponse])
async def get_recommendations(
    confidence_level: Optional[str] = Query(None, pattern="^(LOW|MEDIUM|HIGH|VERY_HIGH)$"),
    market_type: Optional[str] = Query(None),
    min_odds: Optional[float] = Query(None, ge=1.01),
    max_odds: Optional[float] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
):
    """Get active betting recommendations"""

    recommendation_service = RecommendationService(db)

    recommendations = recommendation_service.get_active_recommendations(
        confidence_level=confidence_level,
        market_type=market_type,
        min_odds=min_odds,
        max_odds=max_odds,
        limit=limit,
        user_id=user["user_id"],
    )

    return recommendations


@router.get("/high-value", response_model=List[RecommendationResponse])
async def get_high_value_bets(
    min_ev: float = Query(0.05, ge=0.01),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
):
    """Get highest expected value recommendations"""

    recommendation_service = RecommendationService(db)

    recommendations = recommendation_service.get_high_value_recommendations(
        min_ev=min_ev, limit=limit, user_id=user["user_id"]
    )

    return recommendations


@router.post("/accumulator", response_model=AccumulatorResponse)
async def build_accumulator(
    request: AccumulatorRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
):
    """Build optimized accumulator bet"""

    recommendation_service = RecommendationService(db)

    try:
        accumulator = recommendation_service.build_accumulator(
            min_legs=request.min_legs,
            max_legs=request.max_legs,
            min_total_odds=request.min_total_odds,
            max_total_odds=request.max_total_odds,
            confidence_threshold=request.confidence_threshold,
            user_id=user["user_id"],
        )

        return accumulator

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
