"""Model performance and management endpoints"""

import logging
from typing import Any, Dict, List, Optional

from auth.dependencies import require_auth
from database import get_db
from fastapi import APIRouter, Depends, Query
from models.responses import ModelPerformanceResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/performance", response_model=List[ModelPerformanceResponse])
async def get_model_performance(
    model_name: Optional[str] = Query(None),
    sport: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> List[ModelPerformanceResponse]:
    """Get model performance metrics"""

    query = text(
        """
        SELECT mpl.model_name, mpl.model_version, mpl.sport,
               mpl.evaluation_date, mpl.metrics, mpl.sample_size,
               mpl.date_range_start, mpl.date_range_end, mpl.notes,
               l.name as league_name
        FROM model_performance_logs mpl
        LEFT JOIN leagues l ON mpl.league_id = l.id
        WHERE (:model_name IS NULL OR mpl.model_name = :model_name)
        AND (:sport IS NULL OR mpl.sport = :sport)
        ORDER BY mpl.evaluation_date DESC
        LIMIT :limit
    """
    )

    results = db.execute(query, {"model_name": model_name, "sport": sport, "limit": limit}).fetchall()

    performances = []
    for row in results:
        metrics = row.metrics or {}
        # metrics is free-form JSONB. The calibration monitor (scripts/
        # monitor_models.py, since 2026-06) writes non-scalar entries
        # (reliability `buckets` list, `prediction_type` string) that
        # violate by_sport's Dict[str, Dict[str, float]] contract and
        # 500'd this endpoint on the first monitor row. Pass through
        # numeric entries only; bool is an int subclass, exclude it.
        numeric_metrics = {
            k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        performances.append(
            ModelPerformanceResponse(
                model_name=row.model_name,
                model_version=row.model_version,
                evaluation_date=row.evaluation_date,
                total_predictions=row.sample_size or 0,
                accuracy=numeric_metrics.get("accuracy", 0),
                log_loss=numeric_metrics.get("log_loss", 0),
                brier_score=numeric_metrics.get("brier_score", 0),
                roi=numeric_metrics.get("roi", 0),
                sharpe_ratio=numeric_metrics.get("sharpe_ratio"),
                by_sport={row.sport: numeric_metrics} if row.sport else None,
            )
        )

    return performances


@router.get("/active")
async def get_active_models(
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> List[Dict[str, Any]]:
    """Get currently active models with latest performance"""

    query = text(
        """
        WITH latest_performance AS (
            SELECT model_name, model_version, metrics, evaluation_date, sample_size,
                   ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY evaluation_date DESC) as rn
            FROM model_performance_logs
        )
        SELECT model_name, model_version, metrics, evaluation_date, sample_size
        FROM latest_performance
        WHERE rn = 1
        ORDER BY model_name
    """
    )

    results = db.execute(query).fetchall()

    models = []
    for row in results:
        metrics = row.metrics or {}
        models.append(
            {
                "model_name": row.model_name,
                "model_version": row.model_version,
                "last_evaluated": row.evaluation_date.isoformat(),
                "sample_size": row.sample_size,
                "accuracy": metrics.get("accuracy", 0),
                "log_loss": metrics.get("log_loss", 0),
                "roi": metrics.get("roi", 0),
                "status": "active",
            }
        )

    return models


@router.get("/comparison")
async def compare_models(
    model_names: str = Query(..., description="Comma-separated model names"),
    sport: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> Dict[str, Any]:
    """Compare performance across models"""

    names = [n.strip() for n in model_names.split(",")]

    query = text(
        """
        WITH latest AS (
            SELECT model_name, model_version, metrics, evaluation_date, sample_size,
                   ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY evaluation_date DESC) as rn
            FROM model_performance_logs
            WHERE model_name = ANY(:names)
            AND (:sport IS NULL OR sport = :sport)
        )
        SELECT model_name, model_version, metrics, evaluation_date, sample_size
        FROM latest
        WHERE rn = 1
    """
    )

    results = db.execute(query, {"names": names, "sport": sport}).fetchall()

    comparison = {}
    for row in results:
        metrics = row.metrics or {}
        comparison[row.model_name] = {
            "model_version": row.model_version,
            "evaluation_date": row.evaluation_date.isoformat(),
            "sample_size": row.sample_size,
            "metrics": {
                "accuracy": metrics.get("accuracy", 0),
                "log_loss": metrics.get("log_loss", 0),
                "brier_score": metrics.get("brier_score", 0),
                "roi": metrics.get("roi", 0),
                "rps": metrics.get("rps", 0),
            },
        }

    # Find best model per metric
    best = {}
    if comparison:
        for metric in ["accuracy", "roi"]:
            best_model = max(
                comparison.items(),
                key=lambda x: x[1]["metrics"].get(metric, 0),
            )
            best[metric] = best_model[0]

        for metric in ["log_loss", "brier_score"]:
            best_model = min(
                comparison.items(),
                key=lambda x: x[1]["metrics"].get(metric, float("inf")),
            )
            best[metric] = best_model[0]

    return {"models": comparison, "best_by_metric": best}


@router.get("/retraining-history")
async def get_retraining_history(
    model_name: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: dict = Depends(require_auth),
) -> List[Dict[str, Any]]:
    """Get model retraining history"""

    query = text(
        """
        SELECT model_name, old_version, new_version, trigger_reason,
               training_data_start, training_data_end, training_samples,
               validation_metrics, test_metrics, training_duration_seconds,
               status, error_message, created_at, completed_at
        FROM model_retraining_logs
        WHERE (:model_name IS NULL OR model_name = :model_name)
        ORDER BY created_at DESC
        LIMIT :limit
    """
    )

    results = db.execute(query, {"model_name": model_name, "limit": limit}).fetchall()

    return [
        {
            "model_name": row.model_name,
            "old_version": row.old_version,
            "new_version": row.new_version,
            "trigger_reason": row.trigger_reason,
            "training_data_start": row.training_data_start.isoformat() if row.training_data_start else None,
            "training_data_end": row.training_data_end.isoformat() if row.training_data_end else None,
            "training_samples": row.training_samples,
            "validation_metrics": row.validation_metrics,
            "test_metrics": row.test_metrics,
            "training_duration_seconds": row.training_duration_seconds,
            "status": row.status,
            "error_message": row.error_message,
            "created_at": row.created_at.isoformat(),
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        }
        for row in results
    ]
