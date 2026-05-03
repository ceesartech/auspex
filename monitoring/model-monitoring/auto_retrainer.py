"""Automated model retraining decision engine and trigger."""

import logging
import subprocess
from datetime import datetime
from typing import Any, Dict, Tuple

from .drift_detector import DriftDetector
from .performance_tracker import ModelPerformanceMonitor

logger = logging.getLogger(__name__)

MAX_DAYS_WITHOUT_RETRAINING = 30
DRIFT_SCORE_THRESHOLD = 0.3


class AutomatedRetrainer:
    """Decide when to retrain and trigger the Kubernetes retraining job."""

    def __init__(
        self,
        db_connection,
        redis_client,
        performance_monitor: ModelPerformanceMonitor,
        drift_detector: DriftDetector,
        namespace: str = 'betting-system',
        cronjob_name: str = 'model-retraining',
    ):
        self.db = db_connection
        self.redis = redis_client
        self.performance_monitor = performance_monitor
        self.drift_detector = drift_detector
        self.namespace = namespace
        self.cronjob_name = cronjob_name

    def should_retrain(self) -> Tuple[bool, str]:
        """Return (needs_retraining, reason)."""
        # 1. Performance degradation
        issues = self.performance_monitor.check_for_performance_issues()
        if issues:
            return True, f"Performance degradation: {'; '.join(issues)}"

        # 2. Data drift
        drift = self.drift_detector.detect_drift()
        if drift and drift['drift_score'] > DRIFT_SCORE_THRESHOLD:
            return True, (
                f"Data drift detected (score={drift['drift_score']:.4f}, "
                f"features={len(drift['features_with_drift'])})"
            )

        # 3. Time since last successful retraining
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT retraining_date
            FROM model_retraining_logs
            WHERE deployed = true
            ORDER BY retraining_date DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        if row is None:
            return True, "No previous retraining record found"

        days_since = (datetime.utcnow() - row[0]).days
        if days_since > MAX_DAYS_WITHOUT_RETRAINING:
            return True, f"Model not retrained in {days_since} days (threshold={MAX_DAYS_WITHOUT_RETRAINING})"

        return False, "No retraining needed"

    def trigger_retraining(self, reason: str) -> Dict[str, Any]:
        """Log the retraining trigger and launch a Kubernetes Job."""
        logger.info("Triggering automated retraining. Reason: %s", reason)

        cursor = self.db.cursor()
        cursor.execute(
            """
            INSERT INTO model_retraining_logs (retraining_date, trigger_reason, deployed)
            VALUES (%s, %s, false)
            RETURNING retraining_id
            """,
            (datetime.utcnow(), reason),
        )
        retraining_id = cursor.fetchone()[0]
        self.db.commit()

        job_name = f"{self.cronjob_name}-{retraining_id}"
        try:
            result = subprocess.run(
                [
                    'kubectl', 'create', 'job', job_name,
                    f'--from=cronjob/{self.cronjob_name}',
                    '-n', self.namespace,
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            logger.info("Retraining job created: %s", result.stdout.strip())
            return {
                'retraining_id': retraining_id,
                'job_name': job_name,
                'status': 'triggered',
                'reason': reason,
            }
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to create retraining job: %s", exc.stderr)
            raise RuntimeError(f"kubectl failed: {exc.stderr}") from exc

    def run(self) -> Dict[str, Any]:
        """Convenience entry-point: check and trigger if needed."""
        needs_retrain, reason = self.should_retrain()
        if needs_retrain:
            return self.trigger_retraining(reason)
        logger.info("Retraining check passed: %s", reason)
        return {'status': 'skipped', 'reason': reason}
