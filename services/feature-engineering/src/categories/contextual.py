"""Contextual features — 42 features covering venue, rest days, referee,
match importance, league stats, and manager/derby indicators.
"""

import logging
from typing import Dict, Optional

from ..core.registry import FeatureMetadata
from ..utils.math_helpers import safe_divide
from ..utils.sql_queries import (
    LEAGUE_SEASON_STATS,
    REFEREE_STATS,
    TEAM_LAST_MATCH_DATE,
)
from .base import BaseFeatureComputer, MatchContext

logger = logging.getLogger(__name__)


class ContextualFeatureComputer(BaseFeatureComputer):
    """Compute contextual features for a match."""

    category = "ctx"

    def _register_features(self) -> None:
        defs = []

        # Venue
        defs.extend([
            FeatureMetadata(
                name="ctx__venue__is_neutral",
                category=self.category,
                description="1 if neutral venue",
                dtype="bool", min_value=0, max_value=1,
            ),
        ])

        # Rest days for each side
        for side in ["home", "away"]:
            defs.extend([
                FeatureMetadata(
                    name=f"ctx__{side}__rest_days",
                    category=self.category,
                    description=f"{side} team days since last match",
                    min_value=0,
                ),
                FeatureMetadata(
                    name=f"ctx__{side}__is_short_rest",
                    category=self.category,
                    description=f"1 if {side} team has <4 days rest",
                    dtype="bool", min_value=0, max_value=1,
                ),
                FeatureMetadata(
                    name=f"ctx__{side}__is_long_rest",
                    category=self.category,
                    description=f"1 if {side} team has >10 days rest",
                    dtype="bool", min_value=0, max_value=1,
                ),
            ])

        # Rest difference
        defs.append(FeatureMetadata(
            name="ctx__rest_diff",
            category=self.category,
            description="Rest days difference (home - away)",
        ))

        # Referee features
        defs.extend([
            FeatureMetadata(
                name="ctx__referee__avg_yellows",
                category=self.category,
                description="Referee average yellow cards per match",
                min_value=0,
            ),
            FeatureMetadata(
                name="ctx__referee__avg_reds",
                category=self.category,
                description="Referee average red cards per match",
                min_value=0,
            ),
            FeatureMetadata(
                name="ctx__referee__avg_fouls",
                category=self.category,
                description="Referee average fouls per match",
                min_value=0,
            ),
            FeatureMetadata(
                name="ctx__referee__avg_goals",
                category=self.category,
                description="Referee average goals per match",
                min_value=0,
            ),
            FeatureMetadata(
                name="ctx__referee__matches_refereed",
                category=self.category,
                description="Total matches refereed by this referee",
                min_value=0,
            ),
            FeatureMetadata(
                name="ctx__referee__is_strict",
                category=self.category,
                description="1 if referee avg yellows > 4",
                dtype="bool", min_value=0, max_value=1,
            ),
        ])

        # League season stats
        defs.extend([
            FeatureMetadata(
                name="ctx__league__avg_goals",
                category=self.category,
                description="League average goals per match this season",
                min_value=0,
            ),
            FeatureMetadata(
                name="ctx__league__avg_home_goals",
                category=self.category,
                description="League average home goals per match",
                min_value=0,
            ),
            FeatureMetadata(
                name="ctx__league__avg_away_goals",
                category=self.category,
                description="League average away goals per match",
                min_value=0,
            ),
            FeatureMetadata(
                name="ctx__league__goals_std",
                category=self.category,
                description="League goals standard deviation",
                min_value=0,
            ),
            FeatureMetadata(
                name="ctx__league__home_win_rate",
                category=self.category,
                description="League home win rate this season",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="ctx__league__draw_rate",
                category=self.category,
                description="League draw rate this season",
                min_value=0, max_value=1,
            ),
        ])

        # Standings-based importance
        for side in ["home", "away"]:
            defs.extend([
                FeatureMetadata(
                    name=f"ctx__{side}__league_position",
                    category=self.category,
                    description=f"{side} team league position",
                    min_value=1,
                ),
                FeatureMetadata(
                    name=f"ctx__{side}__points",
                    category=self.category,
                    description=f"{side} team total points",
                    min_value=0,
                ),
                FeatureMetadata(
                    name=f"ctx__{side}__goal_difference",
                    category=self.category,
                    description=f"{side} team goal difference",
                ),
                FeatureMetadata(
                    name=f"ctx__{side}__ppg",
                    category=self.category,
                    description=f"{side} team points per game in league",
                    min_value=0, max_value=3,
                ),
                FeatureMetadata(
                    name=f"ctx__{side}__is_top4",
                    category=self.category,
                    description=f"1 if {side} team in top 4",
                    dtype="bool", min_value=0, max_value=1,
                ),
                FeatureMetadata(
                    name=f"ctx__{side}__is_bottom3",
                    category=self.category,
                    description=f"1 if {side} team in bottom 3",
                    dtype="bool", min_value=0, max_value=1,
                ),
            ])

        # Position difference and importance
        defs.extend([
            FeatureMetadata(
                name="ctx__position_diff",
                category=self.category,
                description="League position difference (away - home, positive = home better)",
            ),
            FeatureMetadata(
                name="ctx__points_diff",
                category=self.category,
                description="Points difference (home - away)",
            ),
            FeatureMetadata(
                name="ctx__is_top_clash",
                category=self.category,
                description="1 if both teams in top 6",
                dtype="bool", min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="ctx__is_relegation_battle",
                category=self.category,
                description="1 if both teams in bottom 5",
                dtype="bool", min_value=0, max_value=1,
            ),
        ])

        self._feature_defs = defs

    def compute(
        self, ctx: MatchContext, **kwargs
    ) -> Dict[str, Optional[float]]:
        features = self._empty_features()

        # Venue
        features["ctx__venue__is_neutral"] = (
            1.0 if ctx.metadata.get("neutral_venue") else 0.0
        )

        # Rest days
        self._compute_rest_days(features, ctx)

        # Referee
        self._compute_referee(features, ctx)

        # League stats
        self._compute_league_stats(features, ctx)

        # Standings
        self._compute_standings(features, ctx)

        return features

    def _compute_rest_days(
        self, features: Dict[str, Optional[float]], ctx: MatchContext
    ) -> None:
        home_rest = self._get_rest_days(ctx.home_team_id, ctx.match_date)
        away_rest = self._get_rest_days(ctx.away_team_id, ctx.match_date)

        if home_rest is not None:
            features["ctx__home__rest_days"] = home_rest
            features["ctx__home__is_short_rest"] = 1.0 if home_rest < 4 else 0.0
            features["ctx__home__is_long_rest"] = 1.0 if home_rest > 10 else 0.0

        if away_rest is not None:
            features["ctx__away__rest_days"] = away_rest
            features["ctx__away__is_short_rest"] = 1.0 if away_rest < 4 else 0.0
            features["ctx__away__is_long_rest"] = 1.0 if away_rest > 10 else 0.0

        if home_rest is not None and away_rest is not None:
            features["ctx__rest_diff"] = home_rest - away_rest

    def _get_rest_days(self, team_id, match_date) -> Optional[float]:
        try:
            result = self.db.execute_query(
                TEAM_LAST_MATCH_DATE,
                (str(team_id), str(team_id), match_date),
                fetch=True,
            )
            if result and result[0].get("last_match_date"):
                delta = match_date - result[0]["last_match_date"]
                return float(delta.days)
        except Exception as e:
            logger.error(f"Failed to get rest days: {e}")
        return None

    def _compute_referee(
        self, features: Dict[str, Optional[float]], ctx: MatchContext
    ) -> None:
        if not ctx.referee:
            return

        try:
            result = self.db.execute_query(
                REFEREE_STATS, (ctx.referee,), fetch=True
            )
            if not result:
                return

            r = result[0]
            matches_ref = self._safe_get(r, "matches_refereed")
            avg_yellows = self._safe_get(r, "avg_yellows")
            avg_reds = self._safe_get(r, "avg_reds")
            avg_fouls = self._safe_get(r, "avg_fouls")
            avg_goals = self._safe_get(r, "avg_goals")

            features["ctx__referee__matches_refereed"] = matches_ref
            features["ctx__referee__avg_yellows"] = avg_yellows
            features["ctx__referee__avg_reds"] = avg_reds
            features["ctx__referee__avg_fouls"] = avg_fouls
            features["ctx__referee__avg_goals"] = avg_goals
            features["ctx__referee__is_strict"] = (
                1.0 if avg_yellows > 4 else 0.0
            )
        except Exception as e:
            logger.error(f"Failed to compute referee features: {e}")

    def _compute_league_stats(
        self, features: Dict[str, Optional[float]], ctx: MatchContext
    ) -> None:
        try:
            result = self.db.execute_query(
                LEAGUE_SEASON_STATS,
                (str(ctx.league_id), ctx.season),
                fetch=True,
            )
            if not result:
                return

            r = result[0]
            features["ctx__league__avg_goals"] = self._safe_get(r, "avg_goals")
            features["ctx__league__avg_home_goals"] = self._safe_get(r, "avg_home_goals")
            features["ctx__league__avg_away_goals"] = self._safe_get(r, "avg_away_goals")
            features["ctx__league__goals_std"] = self._safe_get(r, "std_goals")
            features["ctx__league__home_win_rate"] = self._safe_get(r, "home_win_rate")
            features["ctx__league__draw_rate"] = self._safe_get(r, "draw_rate")
        except Exception as e:
            logger.error(f"Failed to compute league stats: {e}")

    def _compute_standings(
        self, features: Dict[str, Optional[float]], ctx: MatchContext
    ) -> None:
        try:
            standings = self.db.call_function(
                "get_league_standings",
                (str(ctx.league_id), ctx.season),
            )
        except Exception as e:
            logger.error(f"Failed to fetch standings: {e}")
            return

        if not standings:
            return

        total_teams = len(standings)
        home_pos = away_pos = None
        home_data = away_data = None

        for i, row in enumerate(standings, 1):
            tid = str(row.get("team_id"))
            if tid == str(ctx.home_team_id):
                home_pos = i
                home_data = row
            elif tid == str(ctx.away_team_id):
                away_pos = i
                away_data = row

        for side, pos, data in [
            ("home", home_pos, home_data),
            ("away", away_pos, away_data),
        ]:
            if pos is None or data is None:
                continue
            played = self._safe_get(data, "played", 1)
            points = self._safe_get(data, "points")
            gd = self._safe_get(data, "goal_difference")

            features[f"ctx__{side}__league_position"] = float(pos)
            features[f"ctx__{side}__points"] = points
            features[f"ctx__{side}__goal_difference"] = gd
            features[f"ctx__{side}__ppg"] = safe_divide(points, played)
            features[f"ctx__{side}__is_top4"] = 1.0 if pos <= 4 else 0.0
            features[f"ctx__{side}__is_bottom3"] = (
                1.0 if pos > total_teams - 3 else 0.0
            )

        if home_pos is not None and away_pos is not None:
            features["ctx__position_diff"] = float(away_pos - home_pos)
        if home_data is not None and away_data is not None:
            features["ctx__points_diff"] = (
                self._safe_get(home_data, "points")
                - self._safe_get(away_data, "points")
            )

        if home_pos is not None and away_pos is not None:
            features["ctx__is_top_clash"] = (
                1.0 if home_pos <= 6 and away_pos <= 6 else 0.0
            )
            features["ctx__is_relegation_battle"] = (
                1.0
                if home_pos > total_teams - 5 and away_pos > total_teams - 5
                else 0.0
            )
