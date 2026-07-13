"""Unit tests for /api/v1/models/performance.

Regression coverage for the July-2026 500: model_performance_logs.metrics
is free-form JSONB, and the calibration monitor writes non-scalar entries
(reliability `buckets` list, `prediction_type` string). Passing the raw
dict into ModelPerformanceResponse.by_sport (Dict[str, Dict[str, float]])
raised pydantic ValidationError and 500'd the endpoint on the first
monitor-written row.
"""

import sys
from datetime import date
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
    obj = MagicMock()
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


class TestGetModelPerformance:
    @pytest.mark.asyncio
    async def test_calibration_monitor_rows_do_not_500(self):
        # A metrics blob exactly like scripts/monitor_models.py persists:
        # floats + a string + a list. The endpoint must serve it, keeping
        # only numeric entries in by_sport.
        from routes.models import get_model_performance

        rows = [
            _mock_row(
                model_name="market_consensus_v1",
                model_version="monitor-rolling",
                sport="horse_racing",
                evaluation_date=date(2026, 7, 1),
                sample_size=10898,
                metrics={
                    "n": 10898,
                    "accuracy": 0.104,
                    "brier_score": 0.0806,
                    "log_loss": 0.278,
                    "ece": 0.006,
                    "mce": 0.178,
                    "prediction_type": "win",
                    "buckets": [{"lo": 0.0, "hi": 0.1, "n": 9000}],
                    "well_calibrated": True,
                },
                date_range_start=date(2026, 6, 1),
                date_range_end=date(2026, 7, 1),
                notes="calibration monitor",
                league_name=None,
            ),
        ]
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = rows

        out = await get_model_performance(model_name=None, sport=None, limit=20, db=mock_db, user={"user_id": "u"})

        assert len(out) == 1
        perf = out[0]
        assert perf.accuracy == pytest.approx(0.104)
        assert perf.brier_score == pytest.approx(0.0806)
        assert perf.roi == 0  # absent in monitor rows -> default
        hr = perf.by_sport["horse_racing"]
        assert "buckets" not in hr and "prediction_type" not in hr
        # bool is an int subclass — must be excluded, not coerced to 1.0.
        assert "well_calibrated" not in hr
        assert hr["ece"] == pytest.approx(0.006)

    @pytest.mark.asyncio
    async def test_empty_metrics_row(self):
        from routes.models import get_model_performance

        rows = [
            _mock_row(
                model_name="ensemble",
                model_version="1.0.0",
                sport=None,
                evaluation_date=date(2026, 7, 1),
                sample_size=None,
                metrics=None,
                date_range_start=None,
                date_range_end=None,
                notes=None,
                league_name=None,
            )
        ]
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = rows

        out = await get_model_performance(model_name=None, sport=None, limit=20, db=mock_db, user={"user_id": "u"})
        assert len(out) == 1
        assert out[0].total_predictions == 0
        assert out[0].by_sport is None
