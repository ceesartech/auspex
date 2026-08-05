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
from models.responses import (
    LotteryAnalysisResponse,
    LotteryDrawResponse,
    LotteryEVResponse,
    LotteryRecommendationsResponse,
    LotteryTrackedLine,
)
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


@router.get("/lines", response_model=List[LotteryTrackedLine])
async def get_lottery_lines(
    game: Optional[str] = Query(None, pattern="^(powerball|mega_millions)$"),
    limit: int = Query(48, ge=1, le=500),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> List[LotteryTrackedLine]:
    """The tracked backtest ledger: lines the daily pipeline generated for
    upcoming draws, and their settlement once the draw's numbers land.
    Newest target draws first."""

    query = text(
        """
        SELECT id, game, strategy, numbers, bonus_number, score,
               target_draw_date, created_at,
               matched_main, matched_bonus, prize_tier, settled_at
        FROM lottery_predictions
        WHERE (:game IS NULL OR game = :game)
        ORDER BY target_draw_date DESC NULLS LAST, game, strategy, created_at ASC
        LIMIT :limit
    """
    )

    results = db.execute(query, {"game": game, "limit": limit}).fetchall()

    return [
        LotteryTrackedLine(
            line_id=str(row.id),
            game=row.game,
            strategy=row.strategy,
            numbers=row.numbers,
            bonus_number=row.bonus_number,
            score=float(row.score) if row.score is not None else None,
            target_draw_date=row.target_draw_date,
            created_at=row.created_at,
            matched_main=row.matched_main,
            matched_bonus=row.matched_bonus,
            prize_tier=row.prize_tier,
            settled_at=row.settled_at,
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


@router.get("/ev", response_model=LotteryEVResponse)
async def get_lottery_ev(
    game: str = Query(..., pattern="^(powerball|mega_millions)$"),
    jackpot: Optional[float] = Query(
        None,
        gt=0,
        description="Advertised jackpot in dollars. Omit to use the live next-draw estimate.",
    ),
    state_tax: float = Query(0.0, ge=0.0, lt=0.6, description="State marginal tax rate, e.g. 0.0685."),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> LotteryEVResponse:
    """Per-ticket EV verdict for the next draw — the honest lottery product.

    Computes expected value from the advertised jackpot (live-fetched with its
    actual cash value when `jackpot` is omitted), federal + state taxes, exact
    combinatorial tier odds, and expected co-winner sharing via a Poisson model
    of ticket sales. The answer is almost always "don't play"; the point is to
    quantify by how much, and which game is least bad tonight.
    """
    result = LotteryService(db).ev(game, jackpot=jackpot, state_tax=state_tax)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Live jackpot estimate unavailable — pass the advertised jackpot " "explicitly, e.g. ?jackpot=500000000"
            ),
        )
    return LotteryEVResponse(**result)


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
