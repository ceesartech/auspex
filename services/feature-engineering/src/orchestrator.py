"""Feature orchestrator — two-phase computation with parallel execution and caching.

Phase 1: 6 independent category computers run in parallel via ThreadPoolExecutor.
Phase 2: Derived feature computer runs with all phase-1 features as input.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from redis import Redis

from .categories.base import MatchContext
from .categories.contextual import ContextualFeatureComputer
from .categories.derived import DerivedFeatureComputer
from .categories.head_to_head import HeadToHeadComputer
from .categories.market_odds import MarketOddsComputer
from .categories.player_metrics import PlayerMetricsComputer
from .categories.team_performance import TeamPerformanceComputer
from .categories.temporal import TemporalFeatureComputer
from .core.cache import FeatureCacheManager
from .core.config import FeatureConfig
from .core.database import DatabaseManager
from .core.registry import global_registry
from .utils.sql_queries import MATCH_DETAILS
from .validator import FeatureValidator

logger = logging.getLogger(__name__)


class RealTimeFeatureComputer:
    """Orchestrates feature computation for a match.

    Two-phase approach:
    1. Categories 1-6 compute in parallel (team, h2h, player, ctx, odds, temporal)
    2. Derived features compute using all phase-1 outputs
    """

    def __init__(
        self,
        config: FeatureConfig,
        db_manager: DatabaseManager,
        redis_client: Redis,
    ):
        self.config = config
        self.db = db_manager
        self.redis = redis_client

        # Initialize cache
        self.cache = FeatureCacheManager(config, db_manager, redis_client)

        # Initialize phase-1 computers
        self.phase1_computers = {
            "team": TeamPerformanceComputer(config, db_manager),
            "h2h": HeadToHeadComputer(config, db_manager),
            "player": PlayerMetricsComputer(config, db_manager),
            "ctx": ContextualFeatureComputer(config, db_manager),
            "odds": MarketOddsComputer(config, db_manager),
            "temporal": TemporalFeatureComputer(config, db_manager),
        }

        # Phase-2 computer
        self.derived_computer = DerivedFeatureComputer(config, db_manager)

        # Registry
        self.registry = global_registry
        self._register_all_features()

        # Validator
        self.validator = FeatureValidator(config, self.registry)

    def _register_all_features(self) -> None:
        """Register all features from all computers."""
        for computer in self.phase1_computers.values():
            self.registry.register_many(computer.get_feature_metadata())
        self.registry.register_many(self.derived_computer.get_feature_metadata())

    def compute_features(
        self,
        match_id: str,
        use_cache: bool = True,
        validate: bool = True,
    ) -> Dict[str, Optional[float]]:
        """Compute all features for a match.

        Args:
            match_id: UUID of the match.
            use_cache: Whether to check/update cache.
            validate: Whether to validate output.

        Returns:
            Dict of {feature_name: value} with 250+ features.
        """
        start_time = time.time()

        # Check cache
        if use_cache:
            cached = self.cache.get(match_id)
            if cached:
                logger.info(f"Match {match_id}: returning {len(cached)} cached features")
                return cached

        # Build match context
        ctx = self._build_context(match_id)
        if ctx is None:
            logger.error(f"Match {match_id}: could not build context")
            return {}

        # Phase 1: Parallel computation of independent categories
        phase1_features = self._run_phase1(ctx)

        # Phase 2: Derived features using phase-1 outputs
        phase2_features = self._run_phase2(ctx, phase1_features)

        # Merge all features
        all_features = {**phase1_features, **phase2_features}

        # Validate
        if validate:
            result = self.validator.validate(all_features)
            if not result.is_valid:
                logger.warning(f"Match {match_id}: validation issues — {result.summary}")

        # Cache result
        if use_cache:
            self.cache.set(match_id, all_features)

        duration = time.time() - start_time
        logger.info(f"Match {match_id}: computed {len(all_features)} features " f"in {duration:.2f}s")

        return all_features

    def compute_batch(self, match_ids: List[str], use_cache: bool = True) -> Dict[str, Dict[str, Optional[float]]]:
        """Compute features for multiple matches.

        Returns dict of {match_id: features}.
        """
        results = {}
        for mid in match_ids:
            try:
                results[mid] = self.compute_features(mid, use_cache=use_cache)
            except Exception as e:
                logger.error(f"Failed to compute features for {mid}: {e}")
                results[mid] = {}
        return results

    def get_feature_names(self) -> List[str]:
        """Get all feature names in order."""
        return self.registry.get_all_names()

    def get_feature_count(self) -> int:
        """Get total number of features."""
        return self.registry.count

    def get_category_counts(self) -> Dict[str, int]:
        """Get feature counts per category."""
        return self.registry.category_counts()

    def _build_context(self, match_id: str) -> Optional[MatchContext]:
        """Build MatchContext from database."""
        try:
            result = self.db.execute_query(MATCH_DETAILS, (match_id,), fetch=True)
            if not result:
                return None

            row = result[0]
            return MatchContext(
                match_id=row["match_id"],
                home_team_id=row["home_team_id"],
                away_team_id=row["away_team_id"],
                league_id=row["league_id"],
                season=row["season"],
                match_date=row["match_date"],
                venue=row.get("venue"),
                referee=row.get("referee"),
                home_team_name=row.get("home_team_name"),
                away_team_name=row.get("away_team_name"),
                league_name=row.get("league_name"),
                country=row.get("country"),
            )
        except Exception as e:
            logger.error(f"Failed to build context for {match_id}: {e}")
            return None

    def _run_phase1(self, ctx: MatchContext) -> Dict[str, Optional[float]]:
        """Run all phase-1 computers in parallel."""
        all_features: Dict[str, Optional[float]] = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            for name, computer in self.phase1_computers.items():
                future = executor.submit(self._safe_compute, name, computer, ctx)
                futures[future] = name

            for future in as_completed(futures):
                category_name = futures[future]
                try:
                    result = future.result()
                    all_features.update(result)
                    logger.debug(f"Phase 1 — {category_name}: {len(result)} features")
                except Exception as e:
                    logger.error(f"Phase 1 — {category_name} failed: {e}")

        return all_features

    def _run_phase2(self, ctx: MatchContext, prior_features: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """Run phase-2 derived feature computation."""
        try:
            return self.derived_computer.compute(ctx, prior_features=prior_features)
        except Exception as e:
            logger.error(f"Phase 2 — derived failed: {e}")
            return self.derived_computer._empty_features()

    def _safe_compute(self, name: str, computer, ctx: MatchContext) -> Dict[str, Optional[float]]:
        """Safely compute features for a category, returning empty on error."""
        try:
            return computer.compute(ctx)
        except Exception as e:
            logger.error(f"Category {name} computation failed: {e}")
            return computer._empty_features()
