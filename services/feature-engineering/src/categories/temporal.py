"""Temporal features — pure Python, no database queries.

15 features capturing time-based patterns: day-of-week, month, season phase,
time-of-day, and cyclical encodings.
"""

import math
from datetime import datetime
from typing import Dict, Optional

from ..core.registry import FeatureMetadata
from .base import BaseFeatureComputer, MatchContext


class TemporalFeatureComputer(BaseFeatureComputer):
    """Compute temporal features from match date/time."""

    category = "temporal"

    def _register_features(self) -> None:
        defs = [
            # Day of week (0=Mon, 6=Sun)
            FeatureMetadata(
                name="temporal__dow__raw",
                category=self.category,
                description="Day of week (0=Monday, 6=Sunday)",
                min_value=0, max_value=6,
            ),
            FeatureMetadata(
                name="temporal__dow__sin",
                category=self.category,
                description="Day of week sine encoding",
                min_value=-1, max_value=1,
            ),
            FeatureMetadata(
                name="temporal__dow__cos",
                category=self.category,
                description="Day of week cosine encoding",
                min_value=-1, max_value=1,
            ),
            FeatureMetadata(
                name="temporal__dow__is_weekend",
                category=self.category,
                description="1 if Saturday or Sunday",
                dtype="bool", min_value=0, max_value=1,
            ),
            # Month
            FeatureMetadata(
                name="temporal__month__raw",
                category=self.category,
                description="Month of year (1-12)",
                min_value=1, max_value=12,
            ),
            FeatureMetadata(
                name="temporal__month__sin",
                category=self.category,
                description="Month sine encoding",
                min_value=-1, max_value=1,
            ),
            FeatureMetadata(
                name="temporal__month__cos",
                category=self.category,
                description="Month cosine encoding",
                min_value=-1, max_value=1,
            ),
            # Hour
            FeatureMetadata(
                name="temporal__hour__raw",
                category=self.category,
                description="Hour of day (0-23)",
                min_value=0, max_value=23,
            ),
            FeatureMetadata(
                name="temporal__hour__sin",
                category=self.category,
                description="Hour sine encoding",
                min_value=-1, max_value=1,
            ),
            FeatureMetadata(
                name="temporal__hour__cos",
                category=self.category,
                description="Hour cosine encoding",
                min_value=-1, max_value=1,
            ),
            # Season phase
            FeatureMetadata(
                name="temporal__season__phase",
                category=self.category,
                description="Season phase: 0=early, 0.5=mid, 1=late",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="temporal__season__is_early",
                category=self.category,
                description="1 if in first third of season",
                dtype="bool", min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="temporal__season__is_late",
                category=self.category,
                description="1 if in last third of season",
                dtype="bool", min_value=0, max_value=1,
            ),
            # Day of year
            FeatureMetadata(
                name="temporal__doy__sin",
                category=self.category,
                description="Day of year sine encoding",
                min_value=-1, max_value=1,
            ),
            FeatureMetadata(
                name="temporal__doy__cos",
                category=self.category,
                description="Day of year cosine encoding",
                min_value=-1, max_value=1,
            ),
        ]
        self._feature_defs = defs

    def compute(
        self, ctx: MatchContext, **kwargs
    ) -> Dict[str, Optional[float]]:
        """Compute all temporal features from match date."""
        dt = ctx.match_date
        features: Dict[str, Optional[float]] = {}

        # Day of week
        dow = dt.weekday()  # 0=Monday
        features["temporal__dow__raw"] = float(dow)
        angle = 2.0 * math.pi * dow / 7.0
        features["temporal__dow__sin"] = math.sin(angle)
        features["temporal__dow__cos"] = math.cos(angle)
        features["temporal__dow__is_weekend"] = 1.0 if dow >= 5 else 0.0

        # Month
        month = dt.month
        features["temporal__month__raw"] = float(month)
        angle = 2.0 * math.pi * (month - 1) / 12.0
        features["temporal__month__sin"] = math.sin(angle)
        features["temporal__month__cos"] = math.cos(angle)

        # Hour
        hour = dt.hour
        features["temporal__hour__raw"] = float(hour)
        angle = 2.0 * math.pi * hour / 24.0
        features["temporal__hour__sin"] = math.sin(angle)
        features["temporal__hour__cos"] = math.cos(angle)

        # Season phase — European season Aug-May mapped to 0..1
        season_phase = self._compute_season_phase(dt)
        features["temporal__season__phase"] = season_phase
        features["temporal__season__is_early"] = 1.0 if season_phase < 0.33 else 0.0
        features["temporal__season__is_late"] = 1.0 if season_phase > 0.67 else 0.0

        # Day of year
        doy = dt.timetuple().tm_yday
        angle = 2.0 * math.pi * (doy - 1) / 365.0
        features["temporal__doy__sin"] = math.sin(angle)
        features["temporal__doy__cos"] = math.cos(angle)

        return features

    def _compute_season_phase(self, dt: datetime) -> float:
        """Map date to season phase 0..1 for European football.

        Aug=0.0, Dec=0.4, May=1.0 (approx).
        """
        month = dt.month
        # European season: Aug(8) -> May(5) = 10 months
        if month >= 8:
            # Aug=0, Sep=1, ..., Dec=4
            months_in = month - 8
        elif month <= 5:
            # Jan=5, Feb=6, ..., May=9
            months_in = month + 4
        else:
            # Jun/Jul = off-season, treat as end
            months_in = 9

        return months_in / 9.0
