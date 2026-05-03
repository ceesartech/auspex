"""Prediction endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from database import get_db
from auth.dependencies import require_auth
from models.requests import PredictionRequest, BulkPredictionRequest
from models.responses import PredictionResponse
from services.prediction_service import PredictionService
from services.cache_service import CacheService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=PredictionResponse)
async def get_prediction(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
):
    """Get prediction for a single match"""

    logger.info(f"Prediction request for match {request.match_id}")

    cache_service = CacheService()
    cached = cache_service.get_prediction(request.match_id)
    if cached:
        logger.info(f"Returning cached prediction for match {request.match_id}")
        return cached

    prediction_service = PredictionService(db)

    try:
        prediction = prediction_service.predict_match(
            match_id=request.match_id,
            include_explanation=request.include_explanation,
            include_alternate_models=request.include_alternate_models,
        )

        cache_service.set_prediction(request.match_id, prediction.model_dump())

        return prediction

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate prediction",
        )


@router.post("/bulk", response_model=List[PredictionResponse])
async def get_bulk_predictions(
    request: BulkPredictionRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
):
    """Get predictions for multiple matches"""

    logger.info(f"Bulk prediction request for {len(request.match_ids)} matches")

    prediction_service = PredictionService(db)
    cache_service = CacheService()

    predictions = []
    for match_id in request.match_ids:
        cached = cache_service.get_prediction(match_id)
        if cached:
            predictions.append(cached)
        else:
            try:
                prediction = prediction_service.predict_match(
                    match_id=match_id,
                    include_explanation=request.include_explanation,
                )
                cache_service.set_prediction(match_id, prediction.model_dump())
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to predict match {match_id}: {e}")
                continue

    return predictions


@router.get("/upcoming", response_model=List[PredictionResponse])
async def get_upcoming_predictions(
    sport: Optional[str] = Query(None),
    league: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
):
    """Get predictions for upcoming matches"""

    prediction_service = PredictionService(db)

    predictions = prediction_service.get_upcoming_predictions(
        sport=sport, league=league, limit=limit
    )

    return predictions


@router.get("/live", response_model=List[PredictionResponse])
async def get_live_predictions(
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
):
    """Get predictions for live matches"""

    prediction_service = PredictionService(db)
    predictions = prediction_service.get_live_predictions()

    return predictions
