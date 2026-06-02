"""Accuracy + performance endpoints.

Backs the frontend's performance widget and the model-monitoring
dashboards. Reads from the graded predictions + settled
betting_recommendations populated by scripts/grade_completed_matches.py
(Phase 5 grading), so accuracy reflects real outcomes — not
pre-publication validation metrics.

The summary endpoint is intentionally a single GET so the frontend
can fetch + cache it once per page load. Sport / market / lookback
filters compose to give a per-sport-per-market view; omitting the
filters returns the all-up summary.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from auth.dependencies import require_auth
from database import get_db
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Response models ─────────────────────────────────────────────────


class MarketAccuracy(BaseModel):
    """One row of the per-market accuracy breakdown."""

    # `model_name` collides with pydantic v2's protected `model_`
    # namespace; the field is part of the API contract so suppress
    # the warning instead of renaming.
    model_config = ConfigDict(protected_namespaces=())

    sport: str  # leagues.sport — 'soccer' | 'nhl'
    prediction_type: str  # predictions.prediction_type
    model_name: str  # predictions.model_name
    total: int  # all graded + ungraded predictions in window
    graded: int  # subset where is_correct IS NOT NULL
    correct: int  # subset where is_correct = TRUE
    accuracy: float  # correct / graded; 0.0 when graded == 0


class RecsPnL(BaseModel):
    """Settled betting_recommendations P/L summary for the window."""

    settled: int  # recs with status in ('won','lost','void')
    won: int
    lost: int
    void: int
    total_staked: float  # sum of recommended_stake on settled recs
    total_profit_loss: float  # sum of profit_loss
    roi_pct: float  # 100 × total_profit_loss / total_staked


class AccuracySummary(BaseModel):
    """The endpoint payload. `by_market` is the per-(sport, market)
    breakdown; the top-level numbers are the simple aggregates the
    frontend renders at the top of the widget."""

    days: int  # lookback window
    sport_filter: Optional[str]
    market_filter: Optional[str]
    total: int
    graded: int
    correct: int
    accuracy: float
    by_market: List[MarketAccuracy]
    recs: RecsPnL


# ── Route ───────────────────────────────────────────────────────────


@router.get("/summary", response_model=AccuracySummary)
async def accuracy_summary(
    sport: Optional[str] = Query(None, description="Filter by leagues.sport (e.g. 'soccer', 'nhl')."),
    market: Optional[str] = Query(None, description="Filter by predictions.prediction_type."),
    days: int = Query(30, ge=1, le=365, description="Lookback window in days (default 30)."),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
):
    """Aggregated accuracy + recommendations P/L over the lookback window."""

    # Predictions side. Group by (sport, prediction_type, model_name)
    # so the per-market table self-documents what each row covers.
    rows = db.execute(
        text(
            """
            SELECT
                l.sport AS sport,
                p.prediction_type AS prediction_type,
                p.model_name AS model_name,
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE p.is_correct IS NOT NULL) AS graded,
                COUNT(*) FILTER (WHERE p.is_correct = TRUE) AS correct
            FROM predictions p
            JOIN matches m ON m.id = p.match_id
            JOIN leagues l ON l.id = m.league_id
            WHERE m.status = 'finished'
              AND m.match_date >= NOW() - (:days || ' days')::interval
              AND (:sport IS NULL OR l.sport = :sport)
              AND (:market IS NULL OR p.prediction_type = :market)
            GROUP BY l.sport, p.prediction_type, p.model_name
            ORDER BY l.sport, p.prediction_type, p.model_name
            """
        ),
        {"days": days, "sport": sport, "market": market},
    ).fetchall()

    by_market: List[MarketAccuracy] = []
    total_all = graded_all = correct_all = 0
    for r in rows:
        total = int(r.total)
        graded = int(r.graded)
        correct = int(r.correct)
        by_market.append(
            MarketAccuracy(
                sport=r.sport,
                prediction_type=r.prediction_type,
                model_name=r.model_name,
                total=total,
                graded=graded,
                correct=correct,
                accuracy=(correct / graded) if graded > 0 else 0.0,
            )
        )
        total_all += total
        graded_all += graded
        correct_all += correct

    # Recommendations P/L. Same sport/market scoping so a sport-only
    # view doesn't include the other sport's settled bets.
    rec_row: Dict[str, Any] = (
        db.execute(
            text(
                """
            SELECT
                COUNT(*) FILTER (WHERE br.status IN ('won','lost','void')) AS settled,
                COUNT(*) FILTER (WHERE br.status = 'won')  AS won,
                COUNT(*) FILTER (WHERE br.status = 'lost') AS lost,
                COUNT(*) FILTER (WHERE br.status = 'void') AS void,
                COALESCE(SUM(br.recommended_stake)
                         FILTER (WHERE br.status IN ('won','lost','void')), 0) AS total_staked,
                COALESCE(SUM(br.profit_loss)
                         FILTER (WHERE br.status IN ('won','lost','void')), 0) AS total_profit_loss
            FROM betting_recommendations br
            JOIN matches m ON m.id = br.match_id
            JOIN leagues l ON l.id = m.league_id
            JOIN predictions p ON p.id = br.prediction_id
            WHERE m.status = 'finished'
              AND m.match_date >= NOW() - (:days || ' days')::interval
              AND (:sport IS NULL OR l.sport = :sport)
              AND (:market IS NULL OR p.prediction_type = :market)
            """
            ),
            {"days": days, "sport": sport, "market": market},
        )
        .fetchone()
        ._asdict()
    )

    staked = float(rec_row.get("total_staked") or 0)
    pnl = float(rec_row.get("total_profit_loss") or 0)
    recs_summary = RecsPnL(
        settled=int(rec_row.get("settled") or 0),
        won=int(rec_row.get("won") or 0),
        lost=int(rec_row.get("lost") or 0),
        void=int(rec_row.get("void") or 0),
        total_staked=round(staked, 2),
        total_profit_loss=round(pnl, 2),
        roi_pct=round((100.0 * pnl / staked), 2) if staked > 0 else 0.0,
    )

    return AccuracySummary(
        days=days,
        sport_filter=sport,
        market_filter=market,
        total=total_all,
        graded=graded_all,
        correct=correct_all,
        accuracy=(correct_all / graded_all) if graded_all > 0 else 0.0,
        by_market=by_market,
        recs=recs_summary,
    )
