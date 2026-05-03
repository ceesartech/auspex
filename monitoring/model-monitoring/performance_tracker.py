"""Model performance monitoring and tracking."""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
model_accuracy_gauge = Gauge('model_accuracy_7d', 'Model accuracy over last 7 days', ['model_version'])
model_roi_gauge = Gauge('model_roi_7d', 'Model ROI over last 7 days', ['model_version'])
model_calibration_gauge = Gauge('model_calibration_error', 'Model calibration error', ['model_version'])
prediction_confidence_hist = Histogram(
    'prediction_confidence',
    'Prediction confidence distribution',
    buckets=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
)
predictions_counter = Counter('predictions_total', 'Total predictions', ['predicted_outcome', 'model_version'])
predictions_correct_counter = Counter('predictions_correct', 'Correct predictions', ['confidence_level', 'model_version'])

OUTCOME_MAP = {'home_win': 0, 'draw': 1, 'away_win': 2}
CONFIDENCE_BANDS = [(0.5, 0.6, 'low'), (0.6, 0.7, 'medium'), (0.7, 0.8, 'high'), (0.8, 1.0, 'very_high')]


@dataclass
class PerformanceMetrics:
    accuracy: float
    log_loss_score: float
    brier_score: float
    roi: float
    total_predictions: int
    correct_predictions: int
    avg_confidence: float
    calibration_error: float
    model_version: str
    period_days: int


class ModelPerformanceMonitor:
    """Track and expose model performance metrics."""

    def __init__(self, db_connection, redis_client):
        self.db = db_connection
        self.redis = redis_client

    def evaluate_recent_predictions(self, lookback_days: int = 7) -> Optional[PerformanceMetrics]:
        """Evaluate model performance over the given lookback window."""
        query = """
            SELECT
                p.prediction_id,
                p.predicted_outcome,
                p.probability_home,
                p.probability_draw,
                p.probability_away,
                p.confidence,
                p.model_version,
                m.actual_outcome
            FROM predictions p
            JOIN matches m ON p.match_id = m.match_id
            WHERE m.status = 'finished'
              AND m.match_date >= CURRENT_DATE - INTERVAL '%s days'
              AND p.prediction_timestamp >= CURRENT_DATE - INTERVAL '%s days'
        """
        cursor = self.db.cursor()
        cursor.execute(query, (lookback_days, lookback_days))
        rows = cursor.fetchall()

        if not rows:
            logger.warning("No completed predictions found for evaluation (lookback=%d days)", lookback_days)
            return None

        y_true, y_pred, y_pred_proba, confidences = [], [], [], []
        model_version = rows[0][6] if rows else 'unknown'

        for row in rows:
            pred_outcome = row[1]
            actual_outcome = row[7]
            y_true.append(OUTCOME_MAP[actual_outcome])
            y_pred.append(OUTCOME_MAP[pred_outcome])
            y_pred_proba.append([float(row[2]), float(row[3]), float(row[4])])
            confidences.append(float(row[5]))

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        y_pred_proba_arr = np.array(y_pred_proba)

        accuracy = accuracy_score(y_true_arr, y_pred_arr)
        logloss = log_loss(y_true_arr, y_pred_proba_arr)
        brier = float(np.mean([
            brier_score_loss((y_true_arr == i).astype(int), y_pred_proba_arr[:, i])
            for i in range(3)
        ]))
        roi = self._calculate_roi(rows)
        calibration_error = self._calculate_calibration_error(y_true_arr, y_pred_proba_arr)

        metrics = PerformanceMetrics(
            accuracy=accuracy,
            log_loss_score=logloss,
            brier_score=brier,
            roi=roi,
            total_predictions=len(rows),
            correct_predictions=int(np.sum(y_true_arr == y_pred_arr)),
            avg_confidence=float(np.mean(confidences)),
            calibration_error=calibration_error,
            model_version=model_version,
            period_days=lookback_days,
        )

        self._update_prometheus(metrics)
        self._store_metrics(metrics)

        logger.info(
            "Performance evaluation: accuracy=%.4f roi=%.4f predictions=%d (period=%dd)",
            accuracy, roi, len(rows), lookback_days,
        )
        return metrics

    def check_for_performance_issues(self) -> List[str]:
        """Return a list of human-readable performance degradation messages."""
        issues: List[str] = []

        current = self.evaluate_recent_predictions(lookback_days=7)
        baseline = self.evaluate_recent_predictions(lookback_days=30)

        if not current or not baseline:
            return issues

        if current.accuracy < baseline.accuracy - 0.05:
            issues.append(
                f"Accuracy dropped from {baseline.accuracy:.4f} to {current.accuracy:.4f}"
            )
        if current.roi < baseline.roi - 0.03:
            issues.append(
                f"ROI dropped from {baseline.roi:.4f} to {current.roi:.4f}"
            )
        if current.calibration_error > 0.1:
            issues.append(f"High calibration error: {current.calibration_error:.4f}")

        return issues

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calculate_roi(self, rows: list) -> float:
        """Calculate ROI from associated betting recommendations."""
        query = """
            SELECT recommended_stake, actual_profit
            FROM betting_recommendations
            WHERE prediction_id = %s AND actual_profit IS NOT NULL
        """
        cursor = self.db.cursor()
        total_staked = 0.0
        total_profit = 0.0

        for row in rows:
            cursor.execute(query, (row[0],))
            for bet in cursor.fetchall():
                stake = float(bet[0])
                profit = float(bet[1])
                total_staked += stake
                total_profit += profit

        if total_staked == 0:
            return 0.0
        return total_profit / total_staked

    def _calculate_calibration_error(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10
    ) -> float:
        """Expected Calibration Error (ECE)."""
        pred_probs = np.max(y_pred_proba, axis=1)
        pred_class = np.argmax(y_pred_proba, axis=1)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(pred_probs, bins) - 1

        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            n = np.sum(mask)
            if n > 0:
                bin_acc = np.mean(y_true[mask] == pred_class[mask])
                bin_conf = np.mean(pred_probs[mask])
                ece += (n / len(y_true)) * abs(bin_acc - bin_conf)
        return float(ece)

    def _update_prometheus(self, metrics: PerformanceMetrics) -> None:
        model_accuracy_gauge.labels(model_version=metrics.model_version).set(metrics.accuracy)
        model_roi_gauge.labels(model_version=metrics.model_version).set(metrics.roi)
        model_calibration_gauge.labels(model_version=metrics.model_version).set(metrics.calibration_error)

    def _store_metrics(self, metrics: PerformanceMetrics) -> None:
        query = """
            INSERT INTO model_performance_logs
            (evaluation_date, model_version, model_type, evaluation_period_days,
             total_predictions, correct_predictions, accuracy, log_loss,
             brier_score_avg, roi, avg_confidence, calibration_error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """
        cursor = self.db.cursor()
        cursor.execute(query, (
            datetime.utcnow(),
            metrics.model_version,
            'ensemble',
            metrics.period_days,
            metrics.total_predictions,
            metrics.correct_predictions,
            metrics.accuracy,
            metrics.log_loss_score,
            metrics.brier_score,
            metrics.roi,
            metrics.avg_confidence,
            metrics.calibration_error,
        ))
        self.db.commit()
