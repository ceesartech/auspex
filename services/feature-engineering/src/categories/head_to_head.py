"""Head-to-head features — 22 features from historical matchups.

Fetches H2H match history and computes win rates, goal patterns,
recent form between the two teams.
"""

import logging
from typing import Dict, List, Optional

from ..core.registry import FeatureMetadata
from ..utils.math_helpers import exponential_weighted_avg, safe_divide
from ..utils.sql_queries import H2H_MATCHES
from .base import BaseFeatureComputer, MatchContext

logger = logging.getLogger(__name__)


class HeadToHeadComputer(BaseFeatureComputer):
    """Compute head-to-head features between two teams."""

    category = "h2h"

    def _register_features(self) -> None:
        self._feature_defs = [
            FeatureMetadata(
                name="h2h__total_matches",
                category=self.category,
                description="Total H2H matches found",
                min_value=0,
            ),
            FeatureMetadata(
                name="h2h__home_wins",
                category=self.category,
                description="H2H wins for the home team",
                min_value=0,
            ),
            FeatureMetadata(
                name="h2h__away_wins",
                category=self.category,
                description="H2H wins for the away team",
                min_value=0,
            ),
            FeatureMetadata(
                name="h2h__draws",
                category=self.category,
                description="H2H draws",
                min_value=0,
            ),
            FeatureMetadata(
                name="h2h__home_win_rate",
                category=self.category,
                description="Home team H2H win rate",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="h2h__away_win_rate",
                category=self.category,
                description="Away team H2H win rate",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="h2h__draw_rate",
                category=self.category,
                description="H2H draw rate",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="h2h__avg_total_goals",
                category=self.category,
                description="Average total goals in H2H matches",
                min_value=0,
            ),
            FeatureMetadata(
                name="h2h__avg_home_goals",
                category=self.category,
                description="Average goals by home team in H2H",
                min_value=0,
            ),
            FeatureMetadata(
                name="h2h__avg_away_goals",
                category=self.category,
                description="Average goals by away team in H2H",
                min_value=0,
            ),
            FeatureMetadata(
                name="h2h__btts_rate",
                category=self.category,
                description="Both teams to score rate in H2H",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="h2h__over25_rate",
                category=self.category,
                description="Over 2.5 goals rate in H2H",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="h2h__clean_sheet_home_rate",
                category=self.category,
                description="Home team clean sheet rate in H2H",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="h2h__clean_sheet_away_rate",
                category=self.category,
                description="Away team clean sheet rate in H2H",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="h2h__home_scored_rate",
                category=self.category,
                description="Rate home team scored in H2H",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="h2h__away_scored_rate",
                category=self.category,
                description="Rate away team scored in H2H",
                min_value=0, max_value=1,
            ),
            # Recent H2H (last 3)
            FeatureMetadata(
                name="h2h__recent3__home_wins",
                category=self.category,
                description="Home team wins in last 3 H2H",
                min_value=0, max_value=3,
            ),
            FeatureMetadata(
                name="h2h__recent3__away_wins",
                category=self.category,
                description="Away team wins in last 3 H2H",
                min_value=0, max_value=3,
            ),
            FeatureMetadata(
                name="h2h__recent3__avg_goals",
                category=self.category,
                description="Average goals in last 3 H2H",
                min_value=0,
            ),
            # Weighted recent performance
            FeatureMetadata(
                name="h2h__ewa_home_goals",
                category=self.category,
                description="Exponential weighted avg of home goals in H2H",
                min_value=0,
            ),
            FeatureMetadata(
                name="h2h__ewa_away_goals",
                category=self.category,
                description="Exponential weighted avg of away goals in H2H",
                min_value=0,
            ),
            FeatureMetadata(
                name="h2h__days_since_last",
                category=self.category,
                description="Days since last H2H meeting",
                min_value=0,
            ),
        ]

    def compute(
        self, ctx: MatchContext, **kwargs
    ) -> Dict[str, Optional[float]]:
        features = self._empty_features()

        try:
            matches = self._fetch_h2h(ctx)
        except Exception as e:
            logger.error(f"Failed to fetch H2H matches: {e}")
            return features

        if not matches:
            features["h2h__total_matches"] = 0.0
            return features

        n = len(matches)
        features["h2h__total_matches"] = float(n)

        # Compute results relative to the current home/away
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals_list = []
        away_goals_list = []

        for m in matches:
            h_goals, a_goals = self._normalize_goals(m, ctx)
            home_goals_list.append(h_goals)
            away_goals_list.append(a_goals)

            if h_goals > a_goals:
                home_wins += 1
            elif h_goals < a_goals:
                away_wins += 1
            else:
                draws += 1

        features["h2h__home_wins"] = float(home_wins)
        features["h2h__away_wins"] = float(away_wins)
        features["h2h__draws"] = float(draws)
        features["h2h__home_win_rate"] = safe_divide(home_wins, n)
        features["h2h__away_win_rate"] = safe_divide(away_wins, n)
        features["h2h__draw_rate"] = safe_divide(draws, n)

        total_goals = [h + a for h, a in zip(home_goals_list, away_goals_list)]
        features["h2h__avg_total_goals"] = safe_divide(sum(total_goals), n)
        features["h2h__avg_home_goals"] = safe_divide(sum(home_goals_list), n)
        features["h2h__avg_away_goals"] = safe_divide(sum(away_goals_list), n)

        btts = sum(1 for h, a in zip(home_goals_list, away_goals_list) if h > 0 and a > 0)
        over25 = sum(1 for t in total_goals if t > 2.5)
        cs_home = sum(1 for a in away_goals_list if a == 0)
        cs_away = sum(1 for h in home_goals_list if h == 0)
        home_scored = sum(1 for h in home_goals_list if h > 0)
        away_scored = sum(1 for a in away_goals_list if a > 0)

        features["h2h__btts_rate"] = safe_divide(btts, n)
        features["h2h__over25_rate"] = safe_divide(over25, n)
        features["h2h__clean_sheet_home_rate"] = safe_divide(cs_home, n)
        features["h2h__clean_sheet_away_rate"] = safe_divide(cs_away, n)
        features["h2h__home_scored_rate"] = safe_divide(home_scored, n)
        features["h2h__away_scored_rate"] = safe_divide(away_scored, n)

        # Recent 3
        recent3 = min(3, n)
        r3_home = home_goals_list[:recent3]
        r3_away = away_goals_list[:recent3]
        r3_hw = sum(1 for h, a in zip(r3_home, r3_away) if h > a)
        r3_aw = sum(1 for h, a in zip(r3_home, r3_away) if a > h)

        features["h2h__recent3__home_wins"] = float(r3_hw)
        features["h2h__recent3__away_wins"] = float(r3_aw)
        features["h2h__recent3__avg_goals"] = safe_divide(
            sum(h + a for h, a in zip(r3_home, r3_away)), recent3
        )

        # EWA (oldest first for EWA)
        features["h2h__ewa_home_goals"] = exponential_weighted_avg(
            list(reversed(home_goals_list))
        )
        features["h2h__ewa_away_goals"] = exponential_weighted_avg(
            list(reversed(away_goals_list))
        )

        # Days since last meeting
        last_date = matches[0].get("match_date")
        if last_date:
            delta = (ctx.match_date - last_date).days
            features["h2h__days_since_last"] = float(max(0, delta))

        return features

    def _fetch_h2h(self, ctx: MatchContext) -> List[Dict]:
        return self.db.execute_query(
            H2H_MATCHES,
            (
                str(ctx.home_team_id), str(ctx.away_team_id),
                str(ctx.away_team_id), str(ctx.home_team_id),
                ctx.match_date,
                self.config.h2h_max_matches,
            ),
            fetch=True,
        )

    def _normalize_goals(self, match: Dict, ctx: MatchContext):
        """Normalize goals so home_goals always refers to ctx.home_team_id."""
        match_home_id = str(match.get("home_team_id"))
        h_score = match.get("home_score", 0) or 0
        a_score = match.get("away_score", 0) or 0

        if match_home_id == str(ctx.home_team_id):
            return float(h_score), float(a_score)
        else:
            return float(a_score), float(h_score)
