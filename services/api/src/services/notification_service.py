"""Notification service - alerts for high-value bets, model drift"""

import logging
from typing import Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for managing system notifications and alerts"""

    def __init__(self, db: Session):
        self.db = db

    def create_notification(
        self,
        notification_type: str,
        title: str,
        message: str,
        source: str = "api",
        metadata: Optional[Dict] = None,
    ) -> str:
        """Create a system notification"""

        query = text("""
            INSERT INTO system_notifications (type, title, message, source, metadata)
            VALUES (:type, :title, :message, :source, CAST(:metadata AS jsonb))
            RETURNING id
        """)

        import json

        result = self.db.execute(
            query,
            {
                "type": notification_type,
                "title": title,
                "message": message,
                "source": source,
                "metadata": json.dumps(metadata or {}),
            },
        ).fetchone()

        if result is None:
            raise RuntimeError("Failed to create notification")

        self.db.commit()
        return str(result.id)

    def get_unread_notifications(self, limit: int = 50) -> List[Dict]:
        """Get unread notifications"""

        query = text("""
            SELECT id, type, title, message, source, metadata, created_at
            FROM system_notifications
            WHERE is_read = false
            ORDER BY created_at DESC
            LIMIT :limit
        """)

        results = self.db.execute(query, {"limit": limit}).fetchall()

        return [
            {
                "id": str(row.id),
                "type": row.type,
                "title": row.title,
                "message": row.message,
                "source": row.source,
                "metadata": row.metadata or {},
                "created_at": row.created_at.isoformat(),
            }
            for row in results
        ]

    def mark_as_read(self, notification_id: str):
        """Mark a notification as read"""

        query = text("""
            UPDATE system_notifications SET is_read = true
            WHERE id = :id
        """)
        self.db.execute(query, {"id": notification_id})
        self.db.commit()

    def check_high_value_bets(self, min_ev: float = 0.10) -> List[Dict]:
        """Check for high-value bet opportunities and create alerts"""

        query = text("""
            SELECT br.id, br.bet_type, br.selection, br.odds_at_recommendation,
                   br.expected_value, br.bookmaker,
                   m.match_date, ht.name as home_team, at.name as away_team,
                   l.name as league_name
            FROM betting_recommendations br
            JOIN matches m ON br.match_id = m.id
            JOIN leagues l ON m.league_id = l.id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE br.status = 'pending'
            AND m.match_date > NOW()
            AND br.expected_value >= :min_ev
            ORDER BY br.expected_value DESC
        """)

        results = self.db.execute(query, {"min_ev": min_ev}).fetchall()

        alerts = []
        for row in results:
            alert = {
                "recommendation_id": str(row.id),
                "match": f"{row.home_team} vs {row.away_team}",
                "league": row.league_name,
                "bet_type": row.bet_type,
                "selection": row.selection,
                "odds": float(row.odds_at_recommendation),
                "expected_value": float(row.expected_value),
                "bookmaker": row.bookmaker,
                "match_date": row.match_date.isoformat(),
            }
            alerts.append(alert)

            self.create_notification(
                notification_type="recommendation",
                title=f"High Value: {row.home_team} vs {row.away_team}",
                message=(
                    f"{row.bet_type} - {row.selection} @ {row.odds_at_recommendation} "
                    f"on {row.bookmaker}. EV: {float(row.expected_value):.2%}"
                ),
                source="recommendation_engine",
                metadata=alert,
            )

        return alerts

    def check_model_drift(self, threshold: float = 0.05) -> List[Dict]:
        """Check for model performance drift and alert if degradation detected"""

        query = text("""
            WITH recent_performance AS (
                SELECT model_name, model_version, metrics,
                       evaluation_date,
                       ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY evaluation_date DESC) as rn
                FROM model_performance_logs
                ORDER BY evaluation_date DESC
            )
            SELECT curr.model_name, curr.model_version,
                   curr.metrics as current_metrics,
                   prev.metrics as previous_metrics,
                   curr.evaluation_date
            FROM recent_performance curr
            JOIN recent_performance prev
                ON curr.model_name = prev.model_name AND curr.rn = 1 AND prev.rn = 2
        """)

        results = self.db.execute(query).fetchall()

        drift_alerts = []
        for row in results:
            current = row.current_metrics or {}
            previous = row.previous_metrics or {}

            curr_accuracy = current.get("accuracy", 0)
            prev_accuracy = previous.get("accuracy", 0)

            if prev_accuracy > 0 and (prev_accuracy - curr_accuracy) > threshold:
                drift_info = {
                    "model_name": row.model_name,
                    "model_version": row.model_version,
                    "current_accuracy": curr_accuracy,
                    "previous_accuracy": prev_accuracy,
                    "drift": prev_accuracy - curr_accuracy,
                    "evaluation_date": row.evaluation_date.isoformat(),
                }
                drift_alerts.append(drift_info)

                self.create_notification(
                    notification_type="warning",
                    title=f"Model Drift: {row.model_name}",
                    message=(
                        f"Accuracy dropped from {prev_accuracy:.4f} to {curr_accuracy:.4f} "
                        f"(drift: {prev_accuracy - curr_accuracy:.4f})"
                    ),
                    source="drift_detector",
                    metadata=drift_info,
                )

        return drift_alerts
