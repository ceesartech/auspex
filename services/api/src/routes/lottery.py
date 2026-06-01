"""Lottery endpoints - draws, analysis, combination recommendations.

Analysis + generation logic lives in services/lottery_analysis.py (pure engine)
behind services/lottery_service.py. IMPORTANT: lottery draws are random; these
endpoints never forecast a draw — see the engine's DISCLAIMER, surfaced on the
recommendations response.
"""

import logging
from typing import List, Optional

from auth.dependencies import require_auth
from database import get_db
from fastapi import APIRouter, Depends, HTTPException, Query, status
from models.responses import LotteryAnalysisResponse, LotteryDrawResponse, LotteryRecommendationsResponse
from services.lottery_service import LotteryService
from sqlalchemy import text
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/draws", response_model=List[LotteryDrawResponse])
async def get_lottery_draws(
    game: Optional[str] = Query(None, pattern="^(powerball|mega_millions)$"),
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> List[LotteryDrawResponse]:
    """Get recent lottery draw results"""

    query = text(
        """
        SELECT id, game, draw_date, draw_number, numbers, bonus_number,
               multiplier, jackpot_amount
        FROM lottery_draws
        WHERE (:game IS NULL OR game = :game)
        ORDER BY draw_date DESC
        LIMIT :limit
    """
    )

    results = db.execute(query, {"game": game, "limit": limit}).fetchall()

    return [
        LotteryDrawResponse(
            draw_id=str(row.id),
            game=row.game,
            draw_date=row.draw_date,
            numbers=row.numbers,
            bonus_number=row.bonus_number,
            multiplier=row.multiplier,
            jackpot_amount=float(row.jackpot_amount) if row.jackpot_amount else None,
        )
        for row in results
    ]


@router.get("/analysis", response_model=LotteryAnalysisResponse)
async def get_lottery_analysis(
    game: str = Query(..., pattern="^(powerball|mega_millions)$"),
    num_draws: int = Query(100, ge=10, le=1000),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> LotteryAnalysisResponse:
    """Hot / cold / overdue numbers plus the historical combination profile."""

    result = LotteryService(db).analyze(game, num_draws=num_draws)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No draw data found for {game}",
        )
    return LotteryAnalysisResponse(**result)


@router.post("/recommendations", response_model=LotteryRecommendationsResponse)
async def get_lottery_recommendations(
    game: str = Query(..., pattern="^(powerball|mega_millions)$"),
    strategy: str = Query("blend", pattern="^(blend|statistical|ev|hot|due|random)$"),
    num_sets: int = Query(5, ge=1, le=20),
    persist: bool = Query(False, description="Store the generated lines for backtesting."),
    seed: Optional[int] = Query(None, description="Seed for reproducible generation."),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> LotteryRecommendationsResponse:
    """Generate `num_sets` scored combinations for a strategy.

    Strategies: blend (default — all dimensions), statistical (match the
    historical winning profile), ev (avoid commonly-picked numbers to maximize
    expected payout), hot (frequent numbers), due (overdue numbers), random.
    `score` ranks lines internally; it is NOT a probability of winning.
    """

    result = LotteryService(db).recommend(
        game,
        strategy=strategy,
        num_sets=num_sets,
        persist=persist,
        user_id=user.get("user_id"),
        seed=seed,
    )
    return LotteryRecommendationsResponse(**result)
