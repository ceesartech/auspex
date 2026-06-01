"""Prediction endpoints"""

import logging
from typing import List, Optional

from auth.dependencies import require_auth
from database import get_db
from fastapi import APIRouter, Depends, HTTPException, Query, status
from models.requests import BulkPredictionRequest, PredictionRequest
from models.responses import PredictionResponse
from services.cache_service import CacheService
from services.prediction_service import PredictionService, get_model_version
from sqlalchemy.orm import Session

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
    model_version = get_model_version()

    cached = cache_service.get_prediction(request.match_id, model_version=model_version)
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

        cache_service.set_prediction(
            request.match_id,
            prediction.model_dump(),
            model_version=model_version,
        )

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
    model_version = get_model_version()

    predictions = []
    for match_id in request.match_ids:
        cached = cache_service.get_prediction(match_id, model_version=model_version)
        if cached:
            predictions.append(cached)
        else:
            try:
                prediction = prediction_service.predict_match(
                    match_id=match_id,
                    include_explanation=request.include_explanation,
                )
                cache_service.set_prediction(
                    match_id,
                    prediction.model_dump(),
                    model_version=model_version,
                )
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to predict match {match_id}: {e}")
                continue

    return predictions


@router.get("/upcoming", response_model=List[PredictionResponse])
async def get_upcoming_predictions(
    sport: Optional[str] = Query(None, description="Filter by sport ('soccer' or 'nhl')."),
    league: Optional[str] = Query(None, description="Filter by league name (exact match)."),
    market: Optional[str] = Query(
        None,
        description=(
            "Filter by market — e.g. 'moneyline', 'regulation', 'puck_line', "
            "'total' for NHL, 'match_result' for soccer. Omit to get the "
            "default headline market per sport (soccer→match_result, "
            "nhl→moneyline)."
        ),
    ),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
):
    """Get predictions for upcoming matches.

    With market omitted, returns one prediction per match (the headline
    market for each sport). With market set, returns predictions for
    that market only — useful for sport-specific dashboards.
    """

    prediction_service = PredictionService(db)

    predictions = prediction_service.get_upcoming_predictions(
        sport=sport,
        league=league,
        market=market,
        limit=limit,
    )

    return predictions


@router.get("/match/{match_id}", response_model=List[PredictionResponse])
async def get_match_predictions(
    match_id: str,
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
):
    """Return every market's prediction for a single match.

    Soccer matches have one row (match_result); NHL matches have up to
    four (moneyline, regulation, puck_line, total). The frontend uses
    this to render a multi-market card per game without having to make
    one /upcoming call per market.
    """

    prediction_service = PredictionService(db)

    try:
        predictions = prediction_service.get_match_predictions(match_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

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
