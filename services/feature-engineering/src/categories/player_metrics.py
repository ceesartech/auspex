"""Player metrics features — 32 features from squad data.

Covers squad value, availability, fatigue, and key player form.
"""

import logging
from typing import Dict, List, Optional

from ..core.registry import FeatureMetadata
from ..utils.math_helpers import safe_divide
from ..utils.sql_queries import (
    PLAYER_AVAILABILITY,
    PLAYER_RECENT_STATS,
    TEAM_SQUAD,
)
from .base import BaseFeatureComputer, MatchContext

logger = logging.getLogger(__name__)


class PlayerMetricsComputer(BaseFeatureComputer):
    """Compute player-related features for a match."""

    category = "player"

    def _register_features(self) -> None:
        defs = []
        for side in ["home", "away"]:
            defs.extend(
                [
                    # Squad value
                    FeatureMetadata(
                        name=f"player__{side}__squad_value_total",
                        category=self.category,
                        description=f"{side} team total squad market value",
                        min_value=0,
                    ),
                    FeatureMetadata(
                        name=f"player__{side}__squad_value_avg",
                        category=self.category,
                        description=f"{side} team average player market value",
                        min_value=0,
                    ),
                    FeatureMetadata(
                        name=f"player__{side}__squad_size",
                        category=self.category,
                        description=f"{side} team active squad size",
                        min_value=0,
                    ),
                    FeatureMetadata(
                        name=f"player__{side}__avg_age",
                        category=self.category,
                        description=f"{side} team average player age",
                        min_value=15,
                        max_value=45,
                    ),
                    # Availability
                    FeatureMetadata(
                        name=f"player__{side}__injured_count",
                        category=self.category,
                        description=f"{side} team number of injured players",
                        min_value=0,
                    ),
                    FeatureMetadata(
                        name=f"player__{side}__suspended_count",
                        category=self.category,
                        description=f"{side} team number of suspended players",
                        min_value=0,
                    ),
                    FeatureMetadata(
                        name=f"player__{side}__doubtful_count",
                        category=self.category,
                        description=f"{side} team number of doubtful players",
                        min_value=0,
                    ),
                    FeatureMetadata(
                        name=f"player__{side}__unavailable_pct",
                        category=self.category,
                        description=f"{side} team percentage of squad unavailable",
                        min_value=0,
                        max_value=1,
                    ),
                    FeatureMetadata(
                        name=f"player__{side}__injured_value_pct",
                        category=self.category,
                        description=f"{side} team pct of squad value unavailable",
                        min_value=0,
                        max_value=1,
                    ),
                    # Top scorer form
                    FeatureMetadata(
                        name=f"player__{side}__top_scorer_goals_last5",
                        category=self.category,
                        description=f"{side} team top scorer goals in last 5",
                        min_value=0,
                    ),
                    FeatureMetadata(
                        name=f"player__{side}__top_scorer_avg_rating",
                        category=self.category,
                        description=f"{side} team top scorer avg rating last 5",
                    ),
                    FeatureMetadata(
                        name=f"player__{side}__top_scorer_minutes_pct",
                        category=self.category,
                        description=f"{side} team top scorer minutes played pct last 5",
                        min_value=0,
                        max_value=1,
                    ),
                    # Team key player form (avg of top 3)
                    FeatureMetadata(
                        name=f"player__{side}__key_players_avg_rating",
                        category=self.category,
                        description=f"{side} team avg rating of top 3 players",
                    ),
                    FeatureMetadata(
                        name=f"player__{side}__key_players_avg_xg",
                        category=self.category,
                        description=f"{side} team avg xG of top 3 players",
                        min_value=0,
                    ),
                    FeatureMetadata(
                        name=f"player__{side}__key_players_avg_xa",
                        category=self.category,
                        description=f"{side} team avg xA of top 3 players",
                        min_value=0,
                    ),
                    FeatureMetadata(
                        name=f"player__{side}__key_players_available",
                        category=self.category,
                        description=f"{side} team number of top 3 players available",
                        min_value=0,
                        max_value=3,
                    ),
                ]
            )
        self._feature_defs = defs

    def compute(self, ctx: MatchContext, **kwargs) -> Dict[str, Optional[float]]:
        features = self._empty_features()

        try:
            self._compute_side(features, "home", ctx.home_team_id, ctx)
            self._compute_side(features, "away", ctx.away_team_id, ctx)
        except Exception as e:
            logger.error(f"Failed to compute player metrics: {e}")

        return features

    def _compute_side(
        self,
        features: Dict[str, Optional[float]],
        side: str,
        team_id,
        ctx: MatchContext,
    ) -> None:
        squad = self._fetch_squad(team_id)
        if not squad:
            return

        availability = self._fetch_availability(team_id)
        unavailable_ids = {str(a["player_id"]) for a in availability}

        # Squad value & demographics
        values = [self._safe_get(p, "market_value") for p in squad if p.get("market_value") is not None]
        total_value = sum(values) if values else 0
        features[f"player__{side}__squad_value_total"] = total_value
        features[f"player__{side}__squad_value_avg"] = safe_divide(total_value, len(values) if values else 1)
        features[f"player__{side}__squad_size"] = float(len(squad))

        # Average age
        ages = []
        for p in squad:
            dob = p.get("date_of_birth")
            if dob:
                age = (ctx.match_date.date() - dob).days / 365.25
                ages.append(age)
        features[f"player__{side}__avg_age"] = safe_divide(sum(ages), len(ages)) if ages else None

        # Availability counts
        injured = sum(1 for a in availability if a.get("status") == "injured")
        suspended = sum(1 for a in availability if a.get("status") == "suspended")
        doubtful = sum(1 for a in availability if a.get("status") == "doubtful")
        total_unavail = injured + suspended + doubtful

        features[f"player__{side}__injured_count"] = float(injured)
        features[f"player__{side}__suspended_count"] = float(suspended)
        features[f"player__{side}__doubtful_count"] = float(doubtful)
        features[f"player__{side}__unavailable_pct"] = safe_divide(total_unavail, len(squad))

        # Value of unavailable players
        unavail_value = sum(
            self._safe_get(p, "market_value")
            for p in squad
            if str(p.get("player_id")) in unavailable_ids and p.get("market_value") is not None
        )
        features[f"player__{side}__injured_value_pct"] = (
            safe_divide(unavail_value, total_value) if total_value > 0 else 0.0
        )

        # Key player form — fetch recent stats for top players by value
        sorted_squad = sorted(
            squad,
            key=lambda p: self._safe_get(p, "market_value"),
            reverse=True,
        )
        top_players = sorted_squad[:5]  # Top 5 by value
        player_stats = {}
        for p in top_players:
            pid = p["player_id"]
            stats = self._fetch_player_recent(pid, ctx.match_date)
            if stats:
                player_stats[str(pid)] = stats

        # Top scorer (highest goals in recent matches)
        top_scorer_stats = self._find_top_scorer(player_stats)
        if top_scorer_stats:
            features[f"player__{side}__top_scorer_goals_last5"] = float(
                sum(s.get("goals", 0) or 0 for s in top_scorer_stats)
            )
            ratings = [self._safe_get(s, "rating") for s in top_scorer_stats if s.get("rating")]
            features[f"player__{side}__top_scorer_avg_rating"] = (
                safe_divide(sum(ratings), len(ratings)) if ratings else None
            )
            mins = [self._safe_get(s, "minutes_played") for s in top_scorer_stats]
            features[f"player__{side}__top_scorer_minutes_pct"] = safe_divide(sum(mins), len(mins) * 90)

        # Key players (top 3 by value) averages
        key_ids = [str(p["player_id"]) for p in sorted_squad[:3]]
        key_ratings = []
        key_xg = []
        key_xa = []
        key_available = 0

        for kid in key_ids:
            if kid not in unavailable_ids:
                key_available += 1
            stats = player_stats.get(kid, [])
            for s in stats:
                if s.get("rating"):
                    key_ratings.append(self._safe_get(s, "rating"))
                key_xg.append(self._safe_get(s, "xg"))
                key_xa.append(self._safe_get(s, "xa"))

        features[f"player__{side}__key_players_avg_rating"] = (
            safe_divide(sum(key_ratings), len(key_ratings)) if key_ratings else None
        )
        features[f"player__{side}__key_players_avg_xg"] = safe_divide(sum(key_xg), len(key_xg)) if key_xg else None
        features[f"player__{side}__key_players_avg_xa"] = safe_divide(sum(key_xa), len(key_xa)) if key_xa else None
        features[f"player__{side}__key_players_available"] = float(key_available)

    def _fetch_squad(self, team_id) -> List[Dict]:
        return self.db.execute_query(TEAM_SQUAD, (str(team_id),), fetch=True)

    def _fetch_availability(self, team_id) -> List[Dict]:
        return self.db.execute_query(PLAYER_AVAILABILITY, (str(team_id),), fetch=True)

    def _fetch_player_recent(self, player_id, before_date) -> List[Dict]:
        return self.db.execute_query(
            PLAYER_RECENT_STATS,
            (str(player_id), before_date, 5),
            fetch=True,
        )

    def _find_top_scorer(self, player_stats: Dict[str, List[Dict]]) -> Optional[List[Dict]]:
        """Find the player with most goals in recent matches."""
        best_pid = None
        best_goals = -1
        for pid, stats in player_stats.items():
            total = sum(s.get("goals", 0) or 0 for s in stats)
            if total > best_goals:
                best_goals = total
                best_pid = pid
        return player_stats.get(best_pid) if best_pid else None
