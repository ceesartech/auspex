"""Lottery endpoints - draws, analysis, recommendations"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from auth.dependencies import require_auth
from database import get_db
from fastapi import APIRouter, Depends, HTTPException, Query, status
from models.responses import LotteryAnalysisResponse, LotteryDrawResponse
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

    query = text("""
        SELECT id, game, draw_date, draw_number, numbers, bonus_number,
               multiplier, jackpot_amount
        FROM lottery_draws
        WHERE (:game IS NULL OR game = :game)
        ORDER BY draw_date DESC
        LIMIT :limit
    """)

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
    """Get lottery number frequency analysis"""

    query = text("""
        SELECT numbers, bonus_number, draw_date
        FROM lottery_draws
        WHERE game = :game
        ORDER BY draw_date DESC
        LIMIT :num_draws
    """)

    results = db.execute(query, {"game": game, "num_draws": num_draws}).fetchall()

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No draw data found for {game}",
        )

    # Frequency analysis
    number_counts: Counter[int] = Counter()
    bonus_counts: Counter[int] = Counter()

    for row in results:
        for num in row.numbers:
            number_counts[num] += 1
        bonus_counts[row.bonus_number] += 1

    total_draws = len(results)

    # Determine number range based on game
    if game == "powerball":
        main_range = range(1, 70)
    else:  # mega_millions
        main_range = range(1, 71)

    # Hot numbers (most frequent)
    hot = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    hot_numbers = [
        {
            "number": num,
            "frequency": count,
            "percentage": round(count / total_draws * 100, 2),
        }
        for num, count in hot
    ]

    # Cold numbers (least frequent)
    cold = sorted(number_counts.items(), key=lambda x: x[1])[:10]
    cold_numbers = [
        {
            "number": num,
            "frequency": count,
            "percentage": round(count / total_draws * 100, 2),
        }
        for num, count in cold
    ]

    # Overdue numbers (not drawn recently)
    last_seen = {}
    for i, row in enumerate(results):
        for num in row.numbers:
            if num not in last_seen:
                last_seen[num] = i  # draws ago

    all_numbers = set(main_range)
    never_drawn = all_numbers - set(number_counts.keys())

    overdue = []
    for num in never_drawn:
        overdue.append({"number": num, "draws_since_last": total_draws, "frequency": 0})

    for num, last in sorted(last_seen.items(), key=lambda x: x[1], reverse=True)[:10]:
        if num not in [o["number"] for o in overdue]:
            overdue.append(
                {
                    "number": num,
                    "draws_since_last": last,
                    "frequency": number_counts.get(num, 0),
                }
            )

    overdue = sorted(overdue, key=lambda x: x["draws_since_last"], reverse=True)[:10]

    # Full frequency distribution
    frequency_distribution = {str(num): count for num, count in sorted(number_counts.items())}

    return LotteryAnalysisResponse(
        game=game,
        total_draws_analyzed=total_draws,
        hot_numbers=hot_numbers,
        cold_numbers=cold_numbers,
        overdue_numbers=overdue,
        frequency_distribution=frequency_distribution,
    )


@router.post("/recommendations")
async def get_lottery_recommendations(
    game: str = Query(..., pattern="^(powerball|mega_millions)$"),
    strategy: str = Query("balanced", pattern="^(hot|cold|balanced|random)$"),
    num_sets: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """Generate lottery number recommendations based on analysis"""

    import random

    # Get frequency data
    query = text("""
        SELECT numbers, bonus_number
        FROM lottery_draws
        WHERE game = :game
        ORDER BY draw_date DESC
        LIMIT 200
    """)

    results = db.execute(query, {"game": game}).fetchall()

    number_counts: Counter[int] = Counter()
    bonus_counts: Counter[int] = Counter()

    for row in results:
        for num in row.numbers:
            number_counts[num] += 1
        bonus_counts[row.bonus_number] += 1

    # Number ranges
    if game == "powerball":
        main_max = 69
        bonus_max = 26
        pick_count = 5
    else:
        main_max = 70
        bonus_max = 25
        pick_count = 5

    all_main = list(range(1, main_max + 1))
    list(range(1, bonus_max + 1))

    recommendations = []

    for _ in range(num_sets):
        if strategy == "hot":
            # Pick from most frequent numbers
            hot = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
            pool = [n for n, _ in hot[:20]]
            numbers = sorted(random.sample(pool, min(pick_count, len(pool))))
            bonus_hot = sorted(bonus_counts.items(), key=lambda x: x[1], reverse=True)
            bonus = bonus_hot[0][0] if bonus_hot else random.randint(1, bonus_max)

        elif strategy == "cold":
            # Pick from least frequent numbers
            cold = sorted(number_counts.items(), key=lambda x: x[1])
            pool = [n for n, _ in cold[:20]]
            missing = [n for n in all_main if n not in number_counts]
            pool.extend(missing[:5])
            numbers = sorted(random.sample(pool, min(pick_count, len(pool))))
            bonus_cold = sorted(bonus_counts.items(), key=lambda x: x[1])
            bonus = bonus_cold[0][0] if bonus_cold else random.randint(1, bonus_max)

        elif strategy == "balanced":
            # Mix of hot and cold
            hot = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
            cold = sorted(number_counts.items(), key=lambda x: x[1])
            hot_pool = [n for n, _ in hot[:15]]
            cold_pool = [n for n, _ in cold[:15]]
            hot_picks = random.sample(hot_pool, min(3, len(hot_pool)))
            remaining = [n for n in cold_pool if n not in hot_picks]
            cold_picks = random.sample(remaining, min(2, len(remaining)))
            numbers = sorted(hot_picks + cold_picks)
            bonus = random.randint(1, bonus_max)

        else:  # random
            numbers = sorted(random.sample(all_main, pick_count))
            bonus = random.randint(1, bonus_max)

        recommendations.append(
            {
                "numbers": numbers,
                "bonus_number": bonus,
                "strategy": strategy,
            }
        )

    return {
        "game": game,
        "strategy": strategy,
        "total_draws_analyzed": len(results),
        "recommendations": recommendations,
        "disclaimer": "Lottery numbers are random. Past frequency does not predict future draws. Play responsibly.",
    }
