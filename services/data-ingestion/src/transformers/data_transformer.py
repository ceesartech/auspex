"""Data transformation utilities for normalizing scraped data."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger


class DataTransformer:
    """Transforms raw scraped data into the canonical database schema."""

    @staticmethod
    def normalize_team_name(name: str) -> str:
        """Normalize team name for consistent matching."""
        replacements = {
            "FC": "",
            "AFC": "",
            "SC": "",
            "CF": "",
            "United": "Utd",
        }
        normalized = name.strip()
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return " ".join(normalized.split()).strip()

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert decimal odds to American format."""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        return int(-100 / (decimal_odds - 1))

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal format."""
        if american_odds > 0:
            return round(1 + (american_odds / 100), 2)
        return round(1 + (100 / abs(american_odds)), 2)

    @staticmethod
    def transform_match(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw match data into canonical format."""
        return {
            "external_id": str(raw.get("id", raw.get("external_id", ""))),
            "home_team": raw.get("home_team", raw.get("homeTeam", "")),
            "away_team": raw.get("away_team", raw.get("awayTeam", "")),
            "match_date": raw.get("date", raw.get("match_date")),
            "home_score": raw.get("home_score", raw.get("homeScore")),
            "away_score": raw.get("away_score", raw.get("awayScore")),
            "status": raw.get("status", "scheduled"),
            "venue": raw.get("venue", raw.get("stadium")),
            "referee": raw.get("referee"),
            "attendance": raw.get("attendance"),
        }

    @staticmethod
    def transform_odds(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw odds data into canonical format."""
        odds_decimal = raw.get("odds", raw.get("odds_decimal", 0))
        return {
            "match_external_id": str(raw.get("match_id", raw.get("event_id", ""))),
            "bookmaker": raw.get("bookmaker", "").lower(),
            "market_type": raw.get("market", raw.get("market_type", "")),
            "outcome": raw.get("outcome", raw.get("selection", "")),
            "odds_decimal": float(odds_decimal),
            "odds_american": DataTransformer.decimal_to_american(float(odds_decimal)) if odds_decimal else None,
            "timestamp": raw.get("timestamp", datetime.utcnow()),
        }

    @staticmethod
    def transform_batch(
        records: List[Dict[str, Any]],
        transform_fn: Any,
    ) -> List[Dict[str, Any]]:
        """Apply a transformation function to a batch of records."""
        transformed = []
        for record in records:
            try:
                transformed.append(transform_fn(record))
            except Exception as e:
                logger.warning(f"Transform failed for record: {e}")
        return transformed
