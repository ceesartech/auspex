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

    def test_hidden_pairs_are_exactly_the_derived_markets_that_lost_to_the_constant(self, mock_settings):
        """The 2026-11-01 gates. NHL (scripts/ab_nhl_regulation_ties.py): the
        served dixon_coles_nhl artifact lost to a per-season constant beyond
        the noise on five markets; its puck line, clean sheet and win to nil
        did not. Soccer (scripts/ab_soccer_derived_vs_constant.py): of the 14
        full-time derived types only odd_even lost to the per-league constant
        in both O/U-anchor regimes. btts lost in the live regime only and
        is hidden by operator decision; soccer's same-named over_under / correct_score /
        double_chance beat or tie their constant and stay visible."""
        from services.prediction_service import HIDDEN_PREDICTION_PAIRS

        assert HIDDEN_PREDICTION_PAIRS == [
            "dixon_coles_nhl:correct_score",
            "dixon_coles_nhl:double_chance",
            "dixon_coles_nhl:over_under",
            "dixon_coles_nhl:regulation_1x2",
            "dixon_coles_nhl:total_goals",
            "ensemble:btts",
            "ensemble:odd_even",
        ]
        soccer_full_time_derived = (
            "over_under",
            "correct_score",
            "double_chance",
            "draw_no_bet",
            "asian_handicap",
            "team_total",
            "clean_sheet",
            "win_to_nil",
            "winning_margin",
            "total_goals",
            "result_btts",
            "result_over_under",
        )
        soccer_other = ("match_result", "over_under_ht", "match_result_ht", "btts_ht", "ht_ft_double_result")
        for visible in (
            "dixon_coles_nhl:puck_line",
            "dixon_coles_nhl:clean_sheet",
            "dixon_coles_nhl:win_to_nil",
            "ensemble_nhl_reg:match_result",
            *(f"ensemble:{t}" for t in soccer_full_time_derived + soccer_other),
        ):
            assert visible not in HIDDEN_PREDICTION_PAIRS

    def test_frontend_soccer_market_filter_mirrors_the_hidden_soccer_types(self, mock_settings):
        """services/frontend/src/lib/markets.ts keeps its own copy of the hidden
        soccer types (the frontend cannot import Python). It must equal the
        'ensemble:' entries here, and the soccer market filter must offer none
        of them: an offered hidden market would always come back empty."""
        import re
        from pathlib import Path

        from services.prediction_service import HIDDEN_PREDICTION_PAIRS, HIDDEN_SOCCER_DERIVED_TYPES

        markets_ts = Path(__file__).resolve().parents[3] / "frontend" / "src" / "lib" / "markets.ts"
        source = markets_ts.read_text()
        block = re.search(r"export const HIDDEN_SOCCER_MARKETS[^=]*=\s*\[([^\]]*)\]", source)
        assert block, f"HIDDEN_SOCCER_MARKETS not found in {markets_ts}"
        frontend_hidden = re.findall(r"'([^']+)'", block.group(1))
        api_hidden = [p.split(":", 1)[1] for p in HIDDEN_PREDICTION_PAIRS if p.startswith("ensemble:")]
        assert frontend_hidden == api_hidden == list(HIDDEN_SOCCER_DERIVED_TYPES)
        assert "!HIDDEN_SOCCER_MARKETS.includes(" in source  # the filter that keeps them out of the dropdown

    def test_every_display_query_withholds_the_hidden_pairs(self, mock_db, mock_row, mock_settings):
        """upcoming (default, an explicit market=regulation_1x2 and soccer's
        market=odd_even), the match detail page and the live list all filter
        in SQL — before DISTINCT ON and LIMIT, so a hidden row can neither
        show nor shadow another."""
        from services.prediction_service import HIDDEN_PREDICTION_PAIRS, PredictionService

        predicate = "NOT ((p.model_name || ':' || p.prediction_type) = ANY(:hidden_pairs))"
        service = PredictionService(mock_db)

        mock_db.execute.return_value.fetchall.return_value = []
        for kwargs in ({}, {"sport": "nhl", "market": "regulation_1x2"}, {"sport": "soccer", "market": "odd_even"}):
            service.get_upcoming_predictions(**kwargs)
            sql, params = str(mock_db.execute.call_args[0][0]), mock_db.execute.call_args[0][1]
            assert predicate in sql and sql.index(predicate) < sql.index("ORDER BY p.match_id")
            assert params["hidden_pairs"] == HIDDEN_PREDICTION_PAIRS
            if kwargs.get("market") == "odd_even":
                assert params["prediction_type"] == "odd_even" and "ensemble:odd_even" in params["hidden_pairs"]

        service.get_live_predictions()
        sql, params = str(mock_db.execute.call_args[0][0]), mock_db.execute.call_args[0][1]
        assert predicate in sql and params["hidden_pairs"] == HIDDEN_PREDICTION_PAIRS

        row = mock_row(
            id=uuid.uuid4(),
            match_id=uuid.uuid4(),
            predicted_outcome="home",
            confidence=0.6,
            probabilities={"home": 0.6, "away": 0.4},
            model_name="ensemble_nhl_ml",
            model_version="v1.0",
            prediction_type="moneyline",
            match_date=datetime(2026, 11, 3, 0, 0),
            venue=None,
            league_name="NHL",
            sport="nhl",
            home_team="Boston Bruins",
            away_team="Montreal Canadiens",
        )
        self._run_match_predictions(mock_db, mock_row, [row])
        sql, params = str(mock_db.execute.call_args_list[-1][0][0]), mock_db.execute.call_args_list[-1][0][1]
        assert predicate in sql and sql.index(predicate) < sql.index("ORDER BY p.prediction_type")
        assert params == {"match_id": params["match_id"], "hidden_pairs": HIDDEN_PREDICTION_PAIRS}

    def test_stored_prediction_fallback_returns_the_headline_market_only(self, mock_db, mock_row, mock_settings):
        """predict_match's DB fallback (no model produced the headline) used
        to return the newest row of ANY market and type; precompute writes a
        match's markets in one transaction (one created_at), so an NHL match
        could answer for its moneyline with a dixon_coles_nhl correct_score
        row. It now asks for the headline task's prediction_type (or the
        pinned market's) and never returns a hidden pair."""
        from services.prediction_service import HIDDEN_PREDICTION_PAIRS, PredictionService

        predicate = "NOT ((p.model_name || ':' || p.prediction_type) = ANY(:hidden_pairs))"
        stored = mock_row(predicted_outcome="home", confidence=0.58, probabilities={"home": 0.58, "away": 0.42})

        def fallback_call(sport, **kwargs):
            match = mock_row(
                id=uuid.uuid4(),
                league_name="NHL",
                sport=sport,
                home_team="Boston Bruins",
                away_team="Montreal Canadiens",
                match_date=datetime(2026, 11, 3, 0, 0),
                venue=None,
            )
            calls = []

            def side_effect(query, params=None):
                calls.append((str(query), params))
                result = MagicMock()
                result.fetchone.return_value = {1: match, 2: mock_row(features={})}.get(len(calls), stored)
                return result

            mock_db.execute.side_effect = side_effect
            # No model loaded: every task is skipped and the fallback answers.
            response = PredictionService(mock_db, models={}).predict_match(match_id=str(match.id), **kwargs)
            assert response.predicted_outcome == "home" and len(calls) == 3
            return calls[-1]

        for sport, kwargs, prediction_type in (
            ("nhl", {}, "moneyline"),
            ("nhl", {"market": "total"}, "total"),
            ("soccer", {}, "match_result"),
            ("lottery", {}, None),  # no TaskSpec: any market, still never a hidden pair
        ):
            sql, params = fallback_call(sport, **kwargs)
            assert "(:prediction_type IS NULL OR p.prediction_type = :prediction_type)" in sql, sport
            assert predicate in sql and sql.index(predicate) < sql.index("ORDER BY p.created_at DESC LIMIT 1")
            assert params["prediction_type"] == prediction_type, (sport, kwargs)
            assert params["hidden_pairs"] == HIDDEN_PREDICTION_PAIRS

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
