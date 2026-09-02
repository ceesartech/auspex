"""Tests for prediction endpoints"""

import sys
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture(autouse=True)
def mock_settings():
    with patch("config.settings") as mock:
        mock.JWT_SECRET = "test-secret-key-for-testing-only"
        mock.JWT_ALGORITHM = "HS256"
        mock.JWT_EXPIRATION_HOURS = 24
        mock.USER_DOB = "1994-05-09"
        mock.DATABASE_URL = "postgresql://test:test@localhost:5432/test_db"
        mock.REDIS_URL = "redis://localhost:6379/15"
        mock.MODEL_PATH = "/tmp/models"
        mock.REDIS_CACHE_TTL = 3600
        yield mock


class TestPredictionService:
    """Test PredictionService business logic"""

    def test_predict_match_not_found(self, mock_db, mock_settings):
        from services.prediction_service import PredictionService

        mock_db.execute.return_value.fetchone.return_value = None

        service = PredictionService(mock_db)

        with pytest.raises(ValueError, match="not found"):
            service.predict_match(match_id=str(uuid.uuid4()))

    def test_predict_match_with_stored_prediction(self, mock_db, mock_row, mock_settings):
        from services.prediction_service import PredictionService

        match_id = str(uuid.uuid4())

        # Mock match data query. Phase 4a adds leagues.sport to the
        # SELECT so predict_match can route NHL matches to the right
        # task ensemble; soccer rows default to sport="soccer".
        match_result = mock_row(
            id=uuid.UUID(match_id),
            league_name="Premier League",
            sport="soccer",
            home_team="Arsenal",
            away_team="Chelsea",
            match_date=datetime(2025, 3, 15, 15, 0),
            venue="Emirates Stadium",
        )

        # Mock features query
        features_result = mock_row(features={"team__home__form__wins__last5": 3})

        # Mock stored prediction
        prediction_result = mock_row(
            predicted_outcome="home",
            confidence=0.72,
            probabilities={"home": 0.55, "draw": 0.25, "away": 0.20},
        )

        call_count = [0]

        def side_effect(query, params=None):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:  # Match data
                result.fetchone.return_value = match_result
            elif call_count[0] == 2:  # Features
                result.fetchone.return_value = features_result
            elif call_count[0] == 3:  # Stored prediction
                result.fetchone.return_value = prediction_result
            else:
                result.fetchone.return_value = None
            return result

        mock_db.execute.side_effect = side_effect

        service = PredictionService(mock_db)
        response = service.predict_match(match_id=match_id)

        assert response.predicted_outcome == "home"
        assert response.confidence == 0.72
        assert response.match_info.home_team == "Arsenal"

    def test_get_upcoming_predictions(self, mock_db, mock_row, mock_settings):
        from services.prediction_service import PredictionService

        rows = [
            mock_row(
                id=uuid.uuid4(),
                match_id=uuid.uuid4(),
                predicted_outcome="home",
                confidence=0.72,
                probabilities={"home": 0.55, "draw": 0.25, "away": 0.20},
                model_name="ensemble",
                model_version="v1.0",
                prediction_type="match_result",
                features_used={},
                feature_importance={},
                match_date=datetime(2025, 3, 15, 15, 0),
                venue="Emirates Stadium",
                league_name="Premier League",
                home_team="Arsenal",
                away_team="Chelsea",
            ),
        ]

        mock_db.execute.return_value.fetchall.return_value = rows

        service = PredictionService(mock_db)
        predictions = service.get_upcoming_predictions(limit=10)

        assert len(predictions) == 1
        assert predictions[0].predicted_outcome == "home"
        assert predictions[0].match_info.home_team == "Arsenal"
        # Soccer match_result row should be labeled 'match_result' via
        # the (ensemble_name, prediction_type) → market mapping.
        assert predictions[0].market == "match_result"

    def test_get_upcoming_predictions_nhl_default_market_is_moneyline(self, mock_db, mock_row, mock_settings):
        """NHL without an explicit market should default to moneyline so
        one row per match — without this filter, 4 markets × N matches
        would saturate the limit."""
        from services.prediction_service import PredictionService

        mock_db.execute.return_value.fetchall.return_value = []
        service = PredictionService(mock_db)

        service.get_upcoming_predictions(sport="nhl", limit=10)

        # Last call's params is the SQL execute — assert it pinned
        # prediction_type to 'moneyline' even though caller didn't pass it.
        call_args = mock_db.execute.call_args
        assert call_args is not None
        assert call_args[0][1]["prediction_type"] == "moneyline"

    def test_get_upcoming_predictions_explicit_market_overrides_default(self, mock_db, mock_row, mock_settings):
        """Explicit market='puck_line' should map to prediction_type='spread'
        via the TASKS registry."""
        from services.prediction_service import PredictionService

        mock_db.execute.return_value.fetchall.return_value = []
        service = PredictionService(mock_db)

        service.get_upcoming_predictions(sport="nhl", market="puck_line", limit=10)

        call_args = mock_db.execute.call_args
        assert call_args is not None
        assert call_args[0][1]["prediction_type"] == "spread"

    def test_get_upcoming_predictions_soccer_default_market_is_match_result(self, mock_db, mock_row, mock_settings):
        """Soccer stores ~19 prediction_type rows per match (match_result
        plus the Dixon-Coles-derived markets), so the default MUST pin
        the headline 'match_result' or the row LIMIT is eaten by ~3
        matches and whole leagues vanish from the list."""
        from services.prediction_service import PredictionService

        mock_db.execute.return_value.fetchall.return_value = []
        service = PredictionService(mock_db)

        service.get_upcoming_predictions(sport="soccer", limit=10)

        call_args = mock_db.execute.call_args
        assert call_args is not None
        params = call_args[0][1]
        assert params["prediction_type"] == "match_result"
        assert params["headline_pairs"] is None

    def test_get_upcoming_predictions_all_sports_default_restricts_to_headline_set(
        self, mock_db, mock_row, mock_settings
    ):
        """No sport + no market → restrict to each sport's headline
        (sport, prediction_type) PAIR via the list bind, scalar left None.

        Pairs, not bare prediction_types: NHL 'regulation' persists as
        prediction_type='match_result' (soccer's headline), so a bare
        `prediction_type = ANY(['match_result','moneyline'])` would return
        two rows per NHL game (moneyline + regulation) and the frontend
        would render two cards per match."""
        from services.prediction_service import HEADLINE_PAIRS, TASKS, PredictionService

        mock_db.execute.return_value.fetchall.return_value = []
        service = PredictionService(mock_db)

        service.get_upcoming_predictions(limit=10)

        call_args = mock_db.execute.call_args
        assert call_args is not None
        params = call_args[0][1]
        assert params["prediction_type"] is None
        assert params["headline_pairs"] == [
            "mma:moneyline",
            "nba:moneyline",
            "nfl:moneyline",
            "nhl:moneyline",
            "soccer:match_result",
            "tennis:moneyline",
        ]
        assert params["headline_pairs"] == HEADLINE_PAIRS
        # Regression guard: NHL regulation shares soccer's headline
        # prediction_type but must NOT be in the cross-sport default.
        assert TASKS["nhl:regulation"].prediction_type == "match_result"
        assert "nhl:match_result" not in params["headline_pairs"]
        # The SQL must pair sport with prediction_type and use the list bind.
        sql = str(call_args[0][0])
        assert "(l.sport || ':' || p.prediction_type) = ANY(:headline_pairs)" in sql

    def test_get_upcoming_predictions_tennis_default_market_is_moneyline(self, mock_db, mock_row, mock_settings):
        from services.prediction_service import PredictionService

        mock_db.execute.return_value.fetchall.return_value = []
        service = PredictionService(mock_db)

        service.get_upcoming_predictions(sport="tennis", limit=10)

        call_args = mock_db.execute.call_args
        assert call_args is not None
        params = call_args[0][1]
        assert params["prediction_type"] == "moneyline"
        assert params["headline_pairs"] is None

    def test_get_upcoming_predictions_unregistered_sport_has_no_filter(self, mock_db, mock_row, mock_settings):
        """Sports with no TaskSpec (horse_racing) keep today's behaviour:
        no prediction_type filter at all."""
        from services.prediction_service import PredictionService

        mock_db.execute.return_value.fetchall.return_value = []
        service = PredictionService(mock_db)

        service.get_upcoming_predictions(sport="horse_racing", limit=10)

        params = mock_db.execute.call_args[0][1]
        assert params["prediction_type"] is None
        assert params["headline_pairs"] is None

    def test_get_upcoming_predictions_explicit_soccer_submarket_passes_through(self, mock_db, mock_row, mock_settings):
        """Soccer derived markets have no TaskSpec; the raw string must
        reach the SQL unchanged and disable the headline default."""
        from services.prediction_service import PredictionService

        mock_db.execute.return_value.fetchall.return_value = []
        service = PredictionService(mock_db)

        service.get_upcoming_predictions(sport="soccer", market="over_under", limit=10)

        params = mock_db.execute.call_args[0][1]
        assert params["prediction_type"] == "over_under"
        assert params["headline_pairs"] is None

    def test_get_match_predictions_returns_all_markets(self, mock_db, mock_row, mock_settings):
        """NHL match returns all 4 markets ordered by prediction_type."""
        from services.prediction_service import PredictionService

        match_uuid = uuid.uuid4()

        existence_row = mock_row(exists=True)
        prediction_rows = [
            mock_row(
                id=uuid.uuid4(),
                match_id=match_uuid,
                predicted_outcome="home",
                confidence=0.58,
                probabilities={"home": 0.58, "away": 0.42},
                model_name="ensemble_nhl_ml",
                model_version="v1.0",
                prediction_type="moneyline",
                match_date=datetime(2025, 3, 15, 19, 0),
                venue="Bell Centre",
                league_name="NHL",
                sport="nhl",
                home_team="Canadiens",
                away_team="Rangers",
            ),
            mock_row(
                id=uuid.uuid4(),
                match_id=match_uuid,
                predicted_outcome="over",
                confidence=0.54,
                probabilities={"over": 0.54, "under": 0.46},
                model_name="ensemble_nhl_tot",
                model_version="v1.0",
                prediction_type="total",
                match_date=datetime(2025, 3, 15, 19, 0),
                venue="Bell Centre",
                league_name="NHL",
                sport="nhl",
                home_team="Canadiens",
                away_team="Rangers",
            ),
        ]

        call_count = [0]

        def side_effect(query, params=None):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:  # existence check
                result.fetchone.return_value = existence_row
            else:  # prediction rows
                result.fetchall.return_value = prediction_rows
            return result

        mock_db.execute.side_effect = side_effect

        service = PredictionService(mock_db)
        predictions = service.get_match_predictions(str(match_uuid))

        assert len(predictions) == 2
        assert predictions[0].market == "moneyline"
        assert predictions[1].market == "total"

    @staticmethod
    def _match_rows(mock_row, match_uuid, sport, league_name, model_name, prediction_types):
        """Rows in the SQL's prediction_type ASC order, as the DB returns them."""
        return [
            mock_row(
                id=uuid.uuid4(),
                match_id=match_uuid,
                predicted_outcome="home",
                confidence=0.5,
                probabilities={"home": 0.5},
                model_name=model_name,
                model_version="v1.0",
                prediction_type=pt,
                match_date=datetime(2025, 3, 15, 19, 0),
                venue="Somewhere",
                league_name=league_name,
                sport=sport,
                home_team="A",
                away_team="B",
            )
            for pt in sorted(prediction_types)
        ]

    def _run_match_predictions(self, mock_db, mock_row, rows):
        from services.prediction_service import PredictionService

        existence_row = mock_row(exists=True)
        call_count = [0]

        def side_effect(query, params=None):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.fetchone.return_value = existence_row
            else:
                result.fetchall.return_value = rows
            return result

        mock_db.execute.side_effect = side_effect
        return PredictionService(mock_db).get_match_predictions(str(rows[0].match_id))

    def test_get_match_predictions_soccer_headline_first(self, mock_db, mock_row, mock_settings):
        """Soccer's 19 markets sort alphabetically with asian_handicap
        (51 outcomes) first — the detail-page chart consumes index 0,
        so match_result must be promoted to the front and the rest keep
        prediction_type order."""
        rows = self._match_rows(
            mock_row,
            uuid.uuid4(),
            "soccer",
            "Premier League",
            "ensemble",
            ["asian_handicap", "btts", "correct_score", "match_result", "over_under", "winning_margin"],
        )
        predictions = self._run_match_predictions(mock_db, mock_row, rows)

        assert [p.market for p in predictions] == [
            "match_result",
            "asian_handicap",
            "btts",
            "correct_score",
            "over_under",
            "winning_margin",
        ]

    def test_get_match_predictions_nhl_moneyline_before_spread(self, mock_db, mock_row, mock_settings):
        """NHL: 'moneyline' sorts after 'match_result' (regulation) in
        prediction_type order; headline promotion puts moneyline first
        and leaves regulation/spread/total in their SQL order."""
        rows = [
            mock_row(
                id=uuid.uuid4(),
                match_id=uuid.uuid4(),
                predicted_outcome="home",
                confidence=0.5,
                probabilities={"home": 0.5},
                model_name=model_name,
                model_version="v1.0",
                prediction_type=pt,
                match_date=datetime(2025, 3, 15, 19, 0),
                venue="Bell Centre",
                league_name="NHL",
                sport="nhl",
                home_team="Canadiens",
                away_team="Rangers",
            )
            for pt, model_name in [
                ("match_result", "ensemble_nhl_reg"),
                ("moneyline", "ensemble_nhl_ml"),
                ("spread", "ensemble_nhl_pl"),
                ("total", "ensemble_nhl_tot"),
            ]
        ]
        for r in rows:
            r.match_id = rows[0].match_id
        predictions = self._run_match_predictions(mock_db, mock_row, rows)

        assert [p.market for p in predictions] == ["moneyline", "regulation", "puck_line", "total"]

    def test_get_upcoming_predictions_keeps_only_latest_row_per_market(self, mock_db, mock_settings):
        """Weekly retrains mint a new model_version and the precompute
        inserts a fresh row per version (the unique key includes
        model_version), so long-lead fixtures carry 2-3 rows per market.
        The list must collapse to the NEWEST row per (match, market) or
        the pane renders duplicate cards (prod 2026-09-02: 28 MMA, 12
        tennis and 2 NFL matches carried duplicates)."""
        from services.prediction_service import PredictionService

        mock_db.execute.return_value.fetchall.return_value = []
        PredictionService(mock_db).get_upcoming_predictions(sport="tennis", limit=10)

        sql = str(mock_db.execute.call_args[0][0])
        assert "DISTINCT ON (p.match_id, p.prediction_type)" in sql
        assert "ORDER BY p.match_id, p.prediction_type, p.created_at DESC" in sql
        # Outer ordering is still by kickoff so the pane reads chronologically.
        assert "ORDER BY match_date ASC" in sql

    def test_get_match_predictions_keeps_only_latest_row_per_market(self, mock_db, mock_row, mock_settings):
        """Same retrain duplicates on the detail page: exactly one card per
        market, the newest one."""
        row = mock_row(
            id=uuid.uuid4(),
            match_id=uuid.uuid4(),
            predicted_outcome="home",
            confidence=0.5,
            probabilities={"home": 0.5, "draw": 0.3, "away": 0.2},
            model_name="ensemble",
            model_version="v1.0",
            prediction_type="match_result",
            match_date=datetime(2025, 3, 15, 15, 0),
            venue="Emirates Stadium",
            league_name="Premier League",
            sport="soccer",
            home_team="Arsenal",
            away_team="Chelsea",
        )
        self._run_match_predictions(mock_db, mock_row, [row])

        sql = str(mock_db.execute.call_args_list[-1][0][0])
        assert "DISTINCT ON (p.prediction_type)" in sql
        assert "ORDER BY p.prediction_type, p.created_at DESC" in sql

    def test_get_match_predictions_match_not_found(self, mock_db, mock_settings):
        from services.prediction_service import PredictionService

        mock_db.execute.return_value.fetchone.return_value = None
        service = PredictionService(mock_db)

        with pytest.raises(ValueError, match="not found"):
            service.get_match_predictions(str(uuid.uuid4()))

    def test_maybe_enqueue_alert_below_threshold_is_noop(self, mock_db, mock_settings):
        """Low-confidence picks must NOT enqueue — otherwise the digest
        fills with junk and the user loses trust in the alerts."""
        from services.prediction_service import TASKS, PredictionService

        service = PredictionService(mock_db)
        task = TASKS["nhl:moneyline"]  # threshold = 0.60

        with patch("services.prediction_service._MODEL_VERSIONS", {"nhl:moneyline": "v1.0"}):
            # 0.55 is below the 0.60 moneyline gate.
            prediction = {
                "predicted_label": "home",
                "confidence": 0.55,
                "probabilities": {"home": 0.55, "away": 0.45},
            }
            match_data = {
                "match_id": "abc123",
                "league_name": "NHL",
                "home_team": "Canadiens",
                "away_team": "Rangers",
                "match_date": datetime(2025, 3, 15, 19, 0),
            }
            with patch("services.cache_service.CacheService") as mock_cache_cls:
                mock_cache_cls.return_value.redis = MagicMock()
                service._maybe_enqueue_alert(prediction, task, match_data)
            # The dedup SETNX must not have been touched (we never got
            # past the threshold check).
            mock_cache_cls.return_value.redis.set.assert_not_called()

    def test_maybe_enqueue_alert_dedup_blocks_repeat(self, mock_db, mock_settings):
        """Second call for the same (date, match, market, model_version)
        must short-circuit — opening the same UI match twice should NOT
        produce two digest entries."""
        from services.prediction_service import TASKS, PredictionService

        service = PredictionService(mock_db)
        task = TASKS["nhl:moneyline"]

        prediction = {
            "predicted_label": "home",
            "confidence": 0.82,
            "probabilities": {"home": 0.82, "away": 0.18},
        }
        match_data = {
            "match_id": "abc123",
            "league_name": "NHL",
            "home_team": "Canadiens",
            "away_team": "Rangers",
            "match_date": datetime(2025, 3, 15, 19, 0),
        }

        with (
            patch("services.prediction_service._MODEL_VERSIONS", {"nhl:moneyline": "v1.0"}),
            patch("services.cache_service.CacheService") as mock_cache_cls,
            patch.dict(
                "sys.modules",
                {"telegram_notify": MagicMock(Alert=MagicMock, enqueue_alerts=MagicMock(return_value=1))},
            ),
        ):
            mock_redis = MagicMock()
            # First call wins (returns truthy), second loses.
            mock_redis.set.side_effect = [True, False]
            mock_cache_cls.return_value.redis = mock_redis

            service._maybe_enqueue_alert(prediction, task, match_data)
            service._maybe_enqueue_alert(prediction, task, match_data)

            # SETNX called twice (one per call), but enqueue_alerts only
            # once — second call short-circuits at dedup.
            assert mock_redis.set.call_count == 2
            telegram_mock = sys.modules["telegram_notify"]
            assert telegram_mock.enqueue_alerts.call_count == 1

    def test_get_live_predictions_empty(self, mock_db, mock_settings):
        from services.prediction_service import PredictionService

        mock_db.execute.return_value.fetchall.return_value = []

        service = PredictionService(mock_db)
        predictions = service.get_live_predictions()

        assert predictions == []

    def test_store_prediction(self, mock_db, mock_settings):
        from services.prediction_service import PredictionService

        service = PredictionService(mock_db)

        prediction = {
            "predicted_label": "home",
            "confidence": 0.72,
            "probabilities": {"home": 0.55, "draw": 0.25, "away": 0.20},
        }

        service._store_prediction(str(uuid.uuid4()), prediction)

        mock_db.execute.assert_called()
        mock_db.commit.assert_called()


class TestCacheService:
    """Test CacheService"""

    def test_get_prediction_cache_miss(self, mock_settings):
        with patch("services.cache_service.Redis") as mock_redis_cls:
            mock_redis = MagicMock()
            mock_redis.get.return_value = None
            mock_redis_cls.from_url.return_value = mock_redis

            from services.cache_service import CacheService

            service = CacheService()
            service.redis = mock_redis

            result = service.get_prediction("match123")
            assert result is None

    def test_get_prediction_cache_hit(self, mock_settings):
        import json

        with patch("services.cache_service.Redis") as mock_redis_cls:
            mock_redis = MagicMock()
            cached_data = {"predicted_outcome": "home", "confidence": 0.72}
            mock_redis.get.return_value = json.dumps(cached_data)
            mock_redis_cls.from_url.return_value = mock_redis

            from services.cache_service import CacheService

            service = CacheService()
            service.redis = mock_redis

            result = service.get_prediction("match123")
            assert result["predicted_outcome"] == "home"

    def test_set_prediction(self, mock_settings):
        with patch("services.cache_service.Redis") as mock_redis_cls:
            mock_redis = MagicMock()
            mock_redis_cls.from_url.return_value = mock_redis

            from services.cache_service import CacheService

            service = CacheService()
            service.redis = mock_redis

            service.set_prediction("match123", {"outcome": "home"})
            mock_redis.setex.assert_called_once()

    def test_invalidate_prediction(self, mock_settings):
        with patch("services.cache_service.Redis") as mock_redis_cls:
            mock_redis = MagicMock()
            # Cache keys are now scoped by model version + feature hash, so
            # invalidation has to SCAN-and-delete across all variants for a
            # given match_id.
            mock_redis.scan_iter.return_value = iter(
                [
                    "prediction:match123:v0:nofeat",
                    "prediction:match123:ensemble_v1.0+12345:abc123",
                ]
            )
            mock_redis_cls.from_url.return_value = mock_redis

            from services.cache_service import CacheService

            service = CacheService()
            service.redis = mock_redis

            service.invalidate_prediction("match123")
            mock_redis.scan_iter.assert_called_once()
            assert mock_redis.delete.call_count == 2

    def test_cache_graceful_redis_failure(self, mock_settings):
        with patch("services.cache_service.Redis") as mock_redis_cls:
            mock_redis = MagicMock()
            mock_redis.get.side_effect = Exception("Connection refused")
            mock_redis_cls.from_url.return_value = mock_redis

            from services.cache_service import CacheService

            service = CacheService()
            service.redis = mock_redis

            result = service.get_prediction("match123")
            assert result is None  # Graceful degradation
