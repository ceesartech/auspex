"""Unit tests for the /api/v1/accuracy/summary endpoint."""

import sys
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


def _mock_row(**kwargs):
    """MockRow stand-in matching the conftest factory."""
    obj = MagicMock()
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


def _rec_row(**kwargs):
    """SQLAlchemy Row stub for the recs P/L query — supports ._asdict()."""
    m = MagicMock()
    m._asdict.return_value = dict(kwargs)
    return m


class TestAccuracySummaryRoute:
    @pytest.mark.asyncio
    async def test_returns_per_market_breakdown(self, mock_db, mock_settings):
        from routes.accuracy import accuracy_summary

        predictions_rows = [
            _mock_row(
                sport="soccer",
                prediction_type="match_result",
                model_name="ensemble",
                total=120,
                graded=100,
                correct=58,
            ),
            _mock_row(
                sport="nhl",
                prediction_type="moneyline",
                model_name="ensemble_nhl_ml",
                total=40,
                graded=35,
                correct=22,
            ),
        ]
        rec_row = _rec_row(
            settled=15,
            won=8,
            lost=6,
            void=1,
            total_staked=300.0,
            total_profit_loss=42.5,
        )

        call_count = [0]

        def side_effect(query, params=None):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:  # predictions aggregate
                result.fetchall.return_value = predictions_rows
            else:  # recs P/L
                result.fetchone.return_value = rec_row
            return result

        mock_db.execute.side_effect = side_effect

        resp = await accuracy_summary(sport=None, market=None, days=30, db=mock_db, user={"user_id": "owner"})

        # Top-level aggregate is the sum across markets.
        assert resp.total == 160
        assert resp.graded == 135
        assert resp.correct == 80
        # 80 / 135 ≈ 0.5926
        assert resp.accuracy == pytest.approx(80 / 135, rel=1e-4)

        # Per-market rows preserved + accuracy computed per row.
        assert len(resp.by_market) == 2
        assert resp.by_market[0].sport == "soccer"
        assert resp.by_market[0].accuracy == pytest.approx(58 / 100)
        assert resp.by_market[1].sport == "nhl"
        assert resp.by_market[1].accuracy == pytest.approx(22 / 35)

        # Recs P/L surfaces + ROI computed.
        assert resp.recs.settled == 15
        assert resp.recs.total_staked == 300.0
        assert resp.recs.total_profit_loss == 42.5
        # 100 × 42.5 / 300 = 14.17%
        assert resp.recs.roi_pct == pytest.approx(14.17, abs=0.01)

    @pytest.mark.asyncio
    async def test_empty_window_returns_zero_accuracy_not_div_by_zero(self, mock_db, mock_settings):
        """Brand-new deploy with no graded picks → endpoint must return
        accuracy=0.0 and roi_pct=0.0, NOT crash on division by zero."""
        from routes.accuracy import accuracy_summary

        def side_effect(query, params=None):
            result = MagicMock()
            result.fetchall.return_value = []
            result.fetchone.return_value = _rec_row(
                settled=0,
                won=0,
                lost=0,
                void=0,
                total_staked=0,
                total_profit_loss=0,
            )
            return result

        mock_db.execute.side_effect = side_effect

        resp = await accuracy_summary(sport=None, market=None, days=30, db=mock_db, user={"user_id": "owner"})

        assert resp.total == 0
        assert resp.graded == 0
        assert resp.accuracy == 0.0  # not NaN, not crash
        assert resp.recs.roi_pct == 0.0
        assert resp.by_market == []

    @pytest.mark.asyncio
    async def test_sport_filter_passed_to_sql(self, mock_db, mock_settings):
        """Sport filter must reach the SQL params or the route would
        silently return ALL sports despite the URL param."""
        from routes.accuracy import accuracy_summary

        def side_effect(query, params=None):
            result = MagicMock()
            result.fetchall.return_value = []
            result.fetchone.return_value = _rec_row(
                settled=0, won=0, lost=0, void=0, total_staked=0, total_profit_loss=0
            )
            return result

        mock_db.execute.side_effect = side_effect

        await accuracy_summary(sport="nhl", market=None, days=14, db=mock_db, user={"user_id": "owner"})

        # Both SQL invocations should have received sport="nhl".
        for call in mock_db.execute.call_args_list:
            params = call.args[1]
            assert params["sport"] == "nhl"
            assert params["days"] == 14

    @pytest.mark.asyncio
    async def test_market_filter_passed_to_sql(self, mock_db, mock_settings):
        from routes.accuracy import accuracy_summary

        def side_effect(query, params=None):
            result = MagicMock()
            result.fetchall.return_value = []
            result.fetchone.return_value = _rec_row(
                settled=0, won=0, lost=0, void=0, total_staked=0, total_profit_loss=0
            )
            return result

        mock_db.execute.side_effect = side_effect

        await accuracy_summary(
            sport=None,
            market="moneyline",
            days=30,
            db=mock_db,
            user={"user_id": "owner"},
        )

        for call in mock_db.execute.call_args_list:
            assert call.args[1]["market"] == "moneyline"
