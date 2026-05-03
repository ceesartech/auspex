"""Market odds features — 26 features from bookmaker odds data.

Covers current/opening odds, movements, implied probabilities,
market efficiency, and overround.
"""

import logging
from typing import Dict, List, Optional

from ..core.registry import FeatureMetadata
from ..utils.math_helpers import (
    decimal_odds_to_implied_probability,
    remove_overround,
    safe_divide,
)
from ..utils.sql_queries import LATEST_ODDS, MATCH_ODDS, OPENING_ODDS
from .base import BaseFeatureComputer, MatchContext

logger = logging.getLogger(__name__)


class MarketOddsComputer(BaseFeatureComputer):
    """Compute features from bookmaker odds data."""

    category = "odds"

    def _register_features(self) -> None:
        self._feature_defs = [
            # Current implied probabilities (best available)
            FeatureMetadata(
                name="odds__implied_prob__home",
                category=self.category,
                description="Implied probability of home win (fair)",
                min_value=0,
                max_value=1,
            ),
            FeatureMetadata(
                name="odds__implied_prob__draw",
                category=self.category,
                description="Implied probability of draw (fair)",
                min_value=0,
                max_value=1,
            ),
            FeatureMetadata(
                name="odds__implied_prob__away",
                category=self.category,
                description="Implied probability of away win (fair)",
                min_value=0,
                max_value=1,
            ),
            # Raw current odds
            FeatureMetadata(
                name="odds__current__home",
                category=self.category,
                description="Current best home win odds",
                min_value=1,
            ),
            FeatureMetadata(
                name="odds__current__draw",
                category=self.category,
                description="Current best draw odds",
                min_value=1,
            ),
            FeatureMetadata(
                name="odds__current__away",
                category=self.category,
                description="Current best away win odds",
                min_value=1,
            ),
            # Opening odds
            FeatureMetadata(
                name="odds__opening__home",
                category=self.category,
                description="Opening home win odds",
                min_value=1,
            ),
            FeatureMetadata(
                name="odds__opening__draw",
                category=self.category,
                description="Opening draw odds",
                min_value=1,
            ),
            FeatureMetadata(
                name="odds__opening__away",
                category=self.category,
                description="Opening away win odds",
                min_value=1,
            ),
            # Odds movement (current - opening)
            FeatureMetadata(
                name="odds__movement__home",
                category=self.category,
                description="Home odds movement (current - opening)",
            ),
            FeatureMetadata(
                name="odds__movement__draw",
                category=self.category,
                description="Draw odds movement (current - opening)",
            ),
            FeatureMetadata(
                name="odds__movement__away",
                category=self.category,
                description="Away odds movement (current - opening)",
            ),
            # Movement direction (1=shortened, -1=drifted, 0=stable)
            FeatureMetadata(
                name="odds__drift__home",
                category=self.category,
                description="Home odds drift direction",
                min_value=-1,
                max_value=1,
            ),
            FeatureMetadata(
                name="odds__drift__away",
                category=self.category,
                description="Away odds drift direction",
                min_value=-1,
                max_value=1,
            ),
            # Market overround
            FeatureMetadata(
                name="odds__overround",
                category=self.category,
                description="Market overround (sum of implied probs - 1)",
                min_value=0,
            ),
            # Favourite indicator
            FeatureMetadata(
                name="odds__favourite",
                category=self.category,
                description="1=home fav, 0=draw fav, -1=away fav",
                min_value=-1,
                max_value=1,
            ),
            FeatureMetadata(
                name="odds__favourite_prob",
                category=self.category,
                description="Implied probability of the favourite",
                min_value=0,
                max_value=1,
            ),
            # Odds ratio
            FeatureMetadata(
                name="odds__home_away_ratio",
                category=self.category,
                description="Home odds / Away odds ratio",
                min_value=0,
            ),
            # Market consensus (std of implied probs across bookmakers)
            FeatureMetadata(
                name="odds__consensus__home_std",
                category=self.category,
                description="Std dev of home win prob across bookmakers",
                min_value=0,
            ),
            FeatureMetadata(
                name="odds__consensus__draw_std",
                category=self.category,
                description="Std dev of draw prob across bookmakers",
                min_value=0,
            ),
            FeatureMetadata(
                name="odds__consensus__away_std",
                category=self.category,
                description="Std dev of away win prob across bookmakers",
                min_value=0,
            ),
            # Over/under odds
            FeatureMetadata(
                name="odds__over25__odds",
                category=self.category,
                description="Over 2.5 goals odds",
                min_value=1,
            ),
            FeatureMetadata(
                name="odds__under25__odds",
                category=self.category,
                description="Under 2.5 goals odds",
                min_value=1,
            ),
            FeatureMetadata(
                name="odds__over25__implied_prob",
                category=self.category,
                description="Implied probability of over 2.5 goals",
                min_value=0,
                max_value=1,
            ),
            # BTTS odds
            FeatureMetadata(
                name="odds__btts_yes__odds",
                category=self.category,
                description="Both teams to score Yes odds",
                min_value=1,
            ),
            FeatureMetadata(
                name="odds__btts_yes__implied_prob",
                category=self.category,
                description="Implied probability of BTTS Yes",
                min_value=0,
                max_value=1,
            ),
        ]

    def compute(self, ctx: MatchContext, **kwargs) -> Dict[str, Optional[float]]:
        features = self._empty_features()

        try:
            current = self._fetch_latest_odds(ctx.match_id)
            opening = self._fetch_opening_odds(ctx.match_id)
            all_odds = self._fetch_all_odds(ctx.match_id)
        except Exception as e:
            logger.error(f"Failed to fetch odds: {e}")
            return features

        self._compute_current_odds(features, current)
        self._compute_opening_odds(features, opening)
        self._compute_movements(features)
        self._compute_consensus(features, all_odds)
        self._compute_secondary_markets(features, current)

        return features

    def _fetch_latest_odds(self, match_id) -> List[Dict]:
        return self.db.execute_query(LATEST_ODDS, (str(match_id),), fetch=True)

    def _fetch_opening_odds(self, match_id) -> List[Dict]:
        return self.db.execute_query(OPENING_ODDS, (str(match_id),), fetch=True)

    def _fetch_all_odds(self, match_id) -> List[Dict]:
        return self.db.execute_query(MATCH_ODDS, (str(match_id),), fetch=True)

    def _compute_current_odds(self, features: Dict[str, Optional[float]], odds: List[Dict]) -> None:
        # Find best 1X2 odds across bookmakers
        home_odds = []
        draw_odds = []
        away_odds = []

        for o in odds:
            mt = o.get("market_type", "")
            sel = o.get("selection", "")
            val = self._safe_get(o, "odds_decimal")
            if val <= 1:
                continue

            if mt == "1x2":
                if sel in ("home", "1"):
                    home_odds.append(val)
                elif sel in ("draw", "X"):
                    draw_odds.append(val)
                elif sel in ("away", "2"):
                    away_odds.append(val)

        if home_odds:
            best_home = max(home_odds)  # Best odds = highest
            features["odds__current__home"] = best_home
        if draw_odds:
            best_draw = max(draw_odds)
            features["odds__current__draw"] = best_draw
        if away_odds:
            best_away = max(away_odds)
            features["odds__current__away"] = best_away

        # Implied probabilities (using average across bookmakers, then remove overround)
        if home_odds and draw_odds and away_odds:
            avg_h = sum(home_odds) / len(home_odds)
            avg_d = sum(draw_odds) / len(draw_odds)
            avg_a = sum(away_odds) / len(away_odds)

            raw_probs = [
                decimal_odds_to_implied_probability(avg_h),
                decimal_odds_to_implied_probability(avg_d),
                decimal_odds_to_implied_probability(avg_a),
            ]
            overround = sum(raw_probs) - 1.0
            features["odds__overround"] = max(0, overround)

            fair = remove_overround(raw_probs)
            features["odds__implied_prob__home"] = fair[0]
            features["odds__implied_prob__draw"] = fair[1]
            features["odds__implied_prob__away"] = fair[2]

            # Favourite
            max_prob = max(fair)
            if fair[0] == max_prob:
                features["odds__favourite"] = 1.0
            elif fair[2] == max_prob:
                features["odds__favourite"] = -1.0
            else:
                features["odds__favourite"] = 0.0
            features["odds__favourite_prob"] = max_prob

            # Ratio
            features["odds__home_away_ratio"] = safe_divide(avg_h, avg_a)

    def _compute_opening_odds(self, features: Dict[str, Optional[float]], odds: List[Dict]) -> None:
        for o in odds:
            mt = o.get("market_type", "")
            sel = o.get("selection", "")
            val = self._safe_get(o, "odds_decimal")
            if mt != "1x2" or val <= 1:
                continue

            if sel in ("home", "1"):
                features["odds__opening__home"] = val
            elif sel in ("draw", "X"):
                features["odds__opening__draw"] = val
            elif sel in ("away", "2"):
                features["odds__opening__away"] = val

    def _compute_movements(self, features: Dict[str, Optional[float]]) -> None:
        for sel in ["home", "draw", "away"]:
            current = features.get(f"odds__current__{sel}")
            opening = features.get(f"odds__opening__{sel}")
            if current is not None and opening is not None:
                movement = current - opening
                features[f"odds__movement__{sel}"] = movement

                if sel in ("home", "away"):
                    threshold = 0.05
                    if movement < -threshold:
                        features[f"odds__drift__{sel}"] = 1.0  # Shortened (money coming in)
                    elif movement > threshold:
                        features[f"odds__drift__{sel}"] = -1.0  # Drifted
                    else:
                        features[f"odds__drift__{sel}"] = 0.0

    def _compute_consensus(self, features: Dict[str, Optional[float]], all_odds: List[Dict]) -> None:
        import numpy as np

        bookmaker_probs: Dict[str, List[float]] = {
            "home": [],
            "draw": [],
            "away": [],
        }

        for o in all_odds:
            mt = o.get("market_type", "")
            sel = o.get("selection", "")
            val = self._safe_get(o, "odds_decimal")
            if mt != "1x2" or val <= 1:
                continue

            prob = decimal_odds_to_implied_probability(val)
            if sel in ("home", "1"):
                bookmaker_probs["home"].append(prob)
            elif sel in ("draw", "X"):
                bookmaker_probs["draw"].append(prob)
            elif sel in ("away", "2"):
                bookmaker_probs["away"].append(prob)

        for sel in ["home", "draw", "away"]:
            vals = bookmaker_probs[sel]
            if len(vals) >= 2:
                features[f"odds__consensus__{sel}_std"] = float(np.std(vals))

    def _compute_secondary_markets(self, features: Dict[str, Optional[float]], odds: List[Dict]) -> None:
        for o in odds:
            mt = o.get("market_type", "")
            sel = o.get("selection", "")
            val = self._safe_get(o, "odds_decimal")
            if val <= 1:
                continue

            if mt == "over_under_25":
                if sel == "over":
                    features["odds__over25__odds"] = val
                    features["odds__over25__implied_prob"] = decimal_odds_to_implied_probability(val)
                elif sel == "under":
                    features["odds__under25__odds"] = val
            elif mt == "btts":
                if sel == "yes":
                    features["odds__btts_yes__odds"] = val
                    features["odds__btts_yes__implied_prob"] = decimal_odds_to_implied_probability(val)
