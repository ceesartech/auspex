"""All evaluation metrics for model assessment."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.model_config import PredictionTask

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    task: PredictionTask = PredictionTask.MATCH_OUTCOME,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    metrics: Dict[str, float] = {}

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Multiclass or binary
    is_binary = task in (PredictionTask.OVER_UNDER, PredictionTask.BTTS)
    n_classes = 2 if is_binary else len(np.unique(y_true))

    # Log loss
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
    except Exception:
        pass

    # Precision, recall, F1
    avg = "binary" if is_binary else "weighted"
    try:
        metrics["precision"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    except Exception:
        pass

    # Brier score
    try:
        if is_binary:
            if y_pred_proba.ndim == 1:
                metrics["brier_score"] = float(brier_score_loss(y_true, y_pred_proba))
            else:
                metrics["brier_score"] = float(brier_score_loss(y_true, y_pred_proba[:, 1]))
        else:
            brier_scores = []
            for c in range(n_classes):
                y_binary = (y_true == c).astype(int)
                if y_pred_proba.ndim > 1 and c < y_pred_proba.shape[1]:
                    brier_scores.append(brier_score_loss(y_binary, y_pred_proba[:, c]))
            if brier_scores:
                metrics["brier_score"] = float(np.mean(brier_scores))
    except Exception:
        pass

    # ROC AUC
    try:
        if is_binary:
            p = y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba
            metrics["roc_auc"] = float(roc_auc_score(y_true, p))
        elif n_classes > 2 and y_pred_proba.ndim > 1:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="weighted")
            )
    except Exception:
        pass

    # Ranked Probability Score (RPS) for ordered outcomes
    try:
        if not is_binary and y_pred_proba.ndim > 1:
            metrics["rps"] = float(ranked_probability_score(y_true, y_pred_proba))
    except Exception:
        pass

    return metrics


def ranked_probability_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Ranked Probability Score — penalizes predictions further from the truth.

    Better than log loss for ordered outcomes (home/draw/away).
    """
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]
    rps_sum = 0.0

    for i in range(n_samples):
        cum_pred = np.cumsum(y_pred_proba[i])
        cum_true = np.cumsum(np.eye(n_classes)[int(y_true[i])])
        rps_sum += np.sum((cum_pred - cum_true) ** 2) / (n_classes - 1)

    return rps_sum / n_samples


def compute_roi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    odds: np.ndarray,
    stake: float = 1.0,
    min_confidence: float = 0.0,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute Return on Investment from predictions.

    Args:
        y_true: True outcomes.
        y_pred: Predicted outcomes.
        odds: Decimal odds for the predicted outcome.
        stake: Stake per bet.
        min_confidence: Minimum prediction confidence to place bet.
        y_pred_proba: Prediction probabilities for confidence filtering.
    """
    total_staked = 0.0
    total_return = 0.0
    bets_placed = 0
    bets_won = 0

    for i in range(len(y_true)):
        # Confidence filter
        if y_pred_proba is not None and min_confidence > 0:
            conf = np.max(y_pred_proba[i]) if y_pred_proba.ndim > 1 else y_pred_proba[i]
            if conf < min_confidence:
                continue

        total_staked += stake
        bets_placed += 1

        if y_pred[i] == y_true[i]:
            total_return += stake * odds[i]
            bets_won += 1

    profit = total_return - total_staked
    roi = profit / total_staked if total_staked > 0 else 0.0

    return {
        "total_staked": total_staked,
        "total_return": total_return,
        "profit": profit,
        "roi": roi,
        "bets_placed": bets_placed,
        "bets_won": bets_won,
        "win_rate": bets_won / bets_placed if bets_placed > 0 else 0.0,
    }
