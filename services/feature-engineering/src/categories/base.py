"""Base class and shared context for feature computers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..core.config import FeatureConfig
from ..core.database import DatabaseManager
from ..core.registry import FeatureMetadata

logger = logging.getLogger(__name__)


@dataclass
class MatchContext:
    """All contextual information needed to compute features for a match."""

    match_id: UUID
    home_team_id: UUID
    away_team_id: UUID
    league_id: UUID
    season: str
    match_date: datetime
    venue: Optional[str] = None
    referee: Optional[str] = None
    home_team_name: Optional[str] = None
    away_team_name: Optional[str] = None
    league_name: Optional[str] = None
    country: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFeatureComputer(ABC):
    """Abstract base class for all feature category computers."""

    category: str = ""  # Override in subclasses

    def __init__(self, config: FeatureConfig, db_manager: DatabaseManager):
        self.config = config
        self.db = db_manager
        self._feature_defs: List[FeatureMetadata] = []
        self._register_features()

    @abstractmethod
    def _register_features(self) -> None:
        """Register all features this computer produces.

        Subclasses must populate self._feature_defs.
        """
        pass

    @abstractmethod
    def compute(
        self, ctx: MatchContext, **kwargs
    ) -> Dict[str, Optional[float]]:
        """Compute all features for a given match context.

        Returns dict of {feature_name: value}.
        None values indicate the feature could not be computed.
        """
        pass

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names this computer produces."""
        return [f.name for f in self._feature_defs]

    def get_feature_metadata(self) -> List[FeatureMetadata]:
        """Get metadata for all features."""
        return list(self._feature_defs)

    @property
    def feature_count(self) -> int:
        """Number of features this computer produces."""
        return len(self._feature_defs)

    # ---------- Shared Helpers ----------

    def _safe_get(
        self, row: Dict, key: str, default: float = 0.0
    ) -> float:
        """Safely get a numeric value from a dict row."""
        val = row.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _compute_win_draw_loss(
        self, matches: List[Dict], team_id: UUID
    ) -> Dict[str, int]:
        """Compute W/D/L counts for a team from match records."""
        wins = draws = losses = 0
        for m in matches:
            home_id = m.get("home_team_id")
            home_score = m.get("home_score", 0) or 0
            away_score = m.get("away_score", 0) or 0

            if str(home_id) == str(team_id):
                if home_score > away_score:
                    wins += 1
                elif home_score == away_score:
                    draws += 1
                else:
                    losses += 1
            else:
                if away_score > home_score:
                    wins += 1
                elif away_score == home_score:
                    draws += 1
                else:
                    losses += 1

        return {"wins": wins, "draws": draws, "losses": losses}

    def _compute_points(self, wdl: Dict[str, int]) -> int:
        """Compute points from W/D/L dict."""
        return wdl["wins"] * 3 + wdl["draws"]

    def _goals_for_against(
        self, matches: List[Dict], team_id: UUID
    ) -> List[Dict[str, int]]:
        """Extract goals for/against per match for a team."""
        results = []
        for m in matches:
            home_id = m.get("home_team_id")
            home_score = m.get("home_score", 0) or 0
            away_score = m.get("away_score", 0) or 0

            if str(home_id) == str(team_id):
                results.append({"gf": home_score, "ga": away_score})
            else:
                results.append({"gf": away_score, "ga": home_score})
        return results

    def _empty_features(self) -> Dict[str, Optional[float]]:
        """Return dict with all features set to None."""
        return {name: None for name in self.get_feature_names()}
