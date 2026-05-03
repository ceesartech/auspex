"""Team performance features — 62 features from recent match data.

Fetches last 20 matches per team via SQL, then computes windowed aggregations
in Python for windows [3, 5, 10, 20].
"""

import logging
from typing import Dict, List, Optional
from uuid import UUID

import numpy as np

from ..core.registry import FeatureMetadata
from ..utils.math_helpers import (
    coefficient_of_variation,
    exponential_weighted_avg,
    linear_trend,
    safe_divide,
)
from ..utils.sql_queries import TEAM_RECENT_MATCHES
from .base import BaseFeatureComputer, MatchContext

logger = logging.getLogger(__name__)

# Metrics computed per window
_WINDOW_METRICS = [
    "wins", "draws", "losses", "points_per_game",
    "goals_scored", "goals_conceded", "goal_diff",
    "clean_sheets", "btts",
    "avg_xg", "avg_xg_against",
    "avg_shots", "avg_shots_on_target",
    "avg_possession", "avg_corners",
]

_TREND_METRICS = ["points_trend", "goals_trend", "xg_trend"]
_OVERALL_METRICS = [
    "scoring_consistency", "defensive_consistency",
]


class TeamPerformanceComputer(BaseFeatureComputer):
    """Compute team form and performance features."""

    category = "team"

    def _register_features(self) -> None:
        defs = []
        windows = self.config.form_windows  # [3, 5, 10, 20]

        # Per-window metrics for both home and away team
        for side in ["home", "away"]:
            for w in windows:
                for metric in _WINDOW_METRICS:
                    defs.append(FeatureMetadata(
                        name=f"team__{side}__form__{metric}__last{w}",
                        category=self.category,
                        description=f"{side} team {metric} over last {w} matches",
                    ))

            # Trends (last 5 only to keep count manageable)
            for metric in _TREND_METRICS:
                defs.append(FeatureMetadata(
                    name=f"team__{side}__{metric}",
                    category=self.category,
                    description=f"{side} team {metric} trend slope",
                ))

            # Consistency metrics
            for metric in _OVERALL_METRICS:
                defs.append(FeatureMetadata(
                    name=f"team__{side}__{metric}",
                    category=self.category,
                    description=f"{side} team {metric}",
                    min_value=0,
                ))

        self._feature_defs = defs

    def compute(
        self, ctx: MatchContext, **kwargs
    ) -> Dict[str, Optional[float]]:
        features = self._empty_features()

        try:
            home_matches = self._fetch_recent_matches(
                ctx.home_team_id, ctx.match_date
            )
            away_matches = self._fetch_recent_matches(
                ctx.away_team_id, ctx.match_date
            )
        except Exception as e:
            logger.error(f"Failed to fetch team matches: {e}")
            return features

        self._compute_side_features(
            features, "home", home_matches, ctx.home_team_id
        )
        self._compute_side_features(
            features, "away", away_matches, ctx.away_team_id
        )

        return features

    def _fetch_recent_matches(self, team_id: UUID, before_date) -> List[Dict]:
        max_window = max(self.config.form_windows)
        return self.db.execute_query(
            TEAM_RECENT_MATCHES,
            (str(team_id), str(team_id), before_date, max_window),
            fetch=True,
        )

    def _compute_side_features(
        self,
        features: Dict[str, Optional[float]],
        side: str,
        matches: List[Dict],
        team_id: UUID,
    ) -> None:
        if not matches:
            return

        # Pre-compute per-match stats
        match_stats = self._extract_match_stats(matches, team_id)

        # Windowed aggregations
        for w in self.config.form_windows:
            window_data = match_stats[:w]
            if not window_data:
                continue
            self._compute_window_features(features, side, w, window_data)

        # Trends (using last 10 matches)
        trend_data = match_stats[:10]
        if len(trend_data) >= 3:
            points_seq = [d["points"] for d in trend_data]
            goals_seq = [d["gf"] for d in trend_data]
            xg_seq = [d["xg"] for d in trend_data]

            features[f"team__{side}__points_trend"] = linear_trend(
                list(reversed(points_seq))
            )
            features[f"team__{side}__goals_trend"] = linear_trend(
                list(reversed(goals_seq))
            )
            features[f"team__{side}__xg_trend"] = linear_trend(
                list(reversed(xg_seq))
            )

        # Consistency (CV over last 10)
        if len(match_stats) >= 3:
            gf_vals = [d["gf"] for d in match_stats[:10]]
            ga_vals = [d["ga"] for d in match_stats[:10]]
            features[f"team__{side}__scoring_consistency"] = coefficient_of_variation(gf_vals)
            features[f"team__{side}__defensive_consistency"] = coefficient_of_variation(ga_vals)

    def _extract_match_stats(
        self, matches: List[Dict], team_id: UUID
    ) -> List[Dict]:
        """Convert raw match rows into per-match stat dicts for a team."""
        stats = []
        for m in matches:
            is_home = str(m.get("home_team_id")) == str(team_id)

            if is_home:
                gf = (m.get("home_score") or 0)
                ga = (m.get("away_score") or 0)
                xg = self._safe_get(m, "xg_home")
                xg_against = self._safe_get(m, "xg_away")
                shots = self._safe_get(m, "shots_home")
                sot = self._safe_get(m, "shots_on_target_home")
                poss = self._safe_get(m, "possession_home", 50.0)
                corners = self._safe_get(m, "corners_home")
            else:
                gf = (m.get("away_score") or 0)
                ga = (m.get("home_score") or 0)
                xg = self._safe_get(m, "xg_away")
                xg_against = self._safe_get(m, "xg_home")
                shots = self._safe_get(m, "shots_away")
                sot = self._safe_get(m, "shots_on_target_away")
                poss = self._safe_get(m, "possession_away", 50.0)
                corners = self._safe_get(m, "corners_away")

            if gf > ga:
                result = "W"
                points = 3
            elif gf == ga:
                result = "D"
                points = 1
            else:
                result = "L"
                points = 0

            stats.append({
                "gf": float(gf),
                "ga": float(ga),
                "xg": xg,
                "xg_against": xg_against,
                "shots": shots,
                "sot": sot,
                "possession": poss,
                "corners": corners,
                "result": result,
                "points": float(points),
                "clean_sheet": 1.0 if ga == 0 else 0.0,
                "btts": 1.0 if gf > 0 and ga > 0 else 0.0,
            })
        return stats

    def _compute_window_features(
        self,
        features: Dict[str, Optional[float]],
        side: str,
        window: int,
        data: List[Dict],
    ) -> None:
        n = len(data)
        prefix = f"team__{side}__form"

        wins = sum(1 for d in data if d["result"] == "W")
        draws = sum(1 for d in data if d["result"] == "D")
        losses = sum(1 for d in data if d["result"] == "L")

        features[f"{prefix}__wins__last{window}"] = float(wins)
        features[f"{prefix}__draws__last{window}"] = float(draws)
        features[f"{prefix}__losses__last{window}"] = float(losses)
        features[f"{prefix}__points_per_game__last{window}"] = safe_divide(
            sum(d["points"] for d in data), n
        )

        features[f"{prefix}__goals_scored__last{window}"] = sum(
            d["gf"] for d in data
        )
        features[f"{prefix}__goals_conceded__last{window}"] = sum(
            d["ga"] for d in data
        )
        features[f"{prefix}__goal_diff__last{window}"] = sum(
            d["gf"] - d["ga"] for d in data
        )

        features[f"{prefix}__clean_sheets__last{window}"] = sum(
            d["clean_sheet"] for d in data
        )
        features[f"{prefix}__btts__last{window}"] = sum(
            d["btts"] for d in data
        )

        features[f"{prefix}__avg_xg__last{window}"] = safe_divide(
            sum(d["xg"] for d in data), n
        )
        features[f"{prefix}__avg_xg_against__last{window}"] = safe_divide(
            sum(d["xg_against"] for d in data), n
        )
        features[f"{prefix}__avg_shots__last{window}"] = safe_divide(
            sum(d["shots"] for d in data), n
        )
        features[f"{prefix}__avg_shots_on_target__last{window}"] = safe_divide(
            sum(d["sot"] for d in data), n
        )
        features[f"{prefix}__avg_possession__last{window}"] = safe_divide(
            sum(d["possession"] for d in data), n
        )
        features[f"{prefix}__avg_corners__last{window}"] = safe_divide(
            sum(d["corners"] for d in data), n
        )
