"""Tests for the horse-racing route. Exercises the two endpoints
end-to-end (with mocked DB rows) so the response-shape invariants
the frontend depends on are locked in CI."""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture(autouse=True)
def mock_settings():
    with patch("config.settings") as mock:
        mock.JWT_SECRET = "test-secret-key-for-testing-only"
        mock.JWT_ALGORITHM = "HS256"
        mock.JWT_EXPIRATION_HOURS = 24
        mock.DATABASE_URL = "postgresql://test:test@localhost:5432/test_db"
        mock.REDIS_URL = "redis://localhost:6379/15"
        yield mock


# ── list_races ──────────────────────────────────────────────────────


class TestListRaces:
    @pytest.mark.asyncio
    async def test_default_filter_is_scheduled(self, mock_db, mock_row):
        # No status arg → server filters to scheduled future races.
        # Lock the param-passing so a future "smart default" doesn't
        # accidentally leak finished races into the upcoming page.
        from routes.races import list_races

        mock_db.execute.return_value.fetchall.return_value = []
        await list_races(
            status_filter=None,
            days_ahead=2,
            limit=100,
            db=mock_db,
            user={"user_id": "owner"},
        )
        args, kwargs = mock_db.execute.call_args
        assert "r.status = 'scheduled'" in str(args[0])

    @pytest.mark.asyncio
    async def test_finished_filter_uses_lookback(self, mock_db, mock_row):
        # 'finished' must use a lookback (race_date in the past). If
        # someone flips the time clause to lookahead, the index would
        # silently show 0 finished races.
        from routes.races import list_races

        mock_db.execute.return_value.fetchall.return_value = []
        await list_races(
            status_filter="finished",
            days_ahead=7,
            limit=100,
            db=mock_db,
            user={"user_id": "owner"},
        )
        args, kwargs = mock_db.execute.call_args
        assert "NOW() - make_interval" in str(args[0])

    @pytest.mark.asyncio
    async def test_invalid_status_returns_400(self, mock_db):
        from routes.races import list_races

        with pytest.raises(HTTPException) as ei:
            await list_races(
                status_filter="abandoned",  # not in the allowed set
                days_ahead=2,
                limit=100,
                db=mock_db,
                user={"user_id": "owner"},
            )
        assert ei.value.status_code == 400

    @pytest.mark.asyncio
    async def test_response_shape_matches_frontend_types(self, mock_db, mock_row):
        # The frontend's RaceSummary depends on exactly this shape.
        # Refactor the SELECT and this test catches a desync.
        from routes.races import list_races

        rid = uuid.uuid4()
        mock_db.execute.return_value.fetchall.return_value = [
            mock_row(
                race_id=str(rid),
                race_date=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
                track_name="Newton Abbot",
                race_number=3,
                distance_meters=2414,
                surface="turf",
                track_condition="Good",
                race_class="Class 4",
                purse_currency="GBP",
                purse_amount=5446.0,
                field_size=9,
                runners=9,
                status="scheduled",
                recommendation_count=2,
            )
        ]
        out = await list_races(
            status_filter=None,
            days_ahead=2,
            limit=100,
            db=mock_db,
            user={"user_id": "owner"},
        )
        assert len(out) == 1
        race = out[0]
        # Field set — every key the React side expects, no extras.
        assert set(race.keys()) == {
            "race_id",
            "race_date",
            "track_name",
            "race_number",
            "distance_meters",
            "surface",
            "track_condition",
            "race_class",
            "purse_currency",
            "purse_amount",
            "field_size",
            "runners",
            "status",
            "recommendation_count",
        }
        # ISO 8601 for date — TS Date constructor consumes this directly.
        assert race["race_date"] == "2026-06-04T15:00:00+00:00"
        assert race["recommendation_count"] == 2

    @pytest.mark.asyncio
    async def test_field_size_falls_back_to_runner_count(self, mock_db, mock_row):
        # /results-loaded rows have field_size=NULL because the
        # historical endpoint doesn't expose it. Derive from the
        # race_entrants count so the UI always has a number to show.
        from routes.races import list_races

        mock_db.execute.return_value.fetchall.return_value = [
            mock_row(
                race_id=str(uuid.uuid4()),
                race_date=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
                track_name="Curragh",
                race_number=None,
                distance_meters=None,
                surface=None,
                track_condition=None,
                race_class=None,
                purse_currency=None,
                purse_amount=None,
                field_size=None,  # not present on /results rows
                runners=12,
                status="finished",
                recommendation_count=0,
            )
        ]
        out = await list_races(
            status_filter="finished",
            days_ahead=7,
            limit=100,
            db=mock_db,
            user={"user_id": "owner"},
        )
        assert out[0]["field_size"] == 12


# ── get_race_detail ─────────────────────────────────────────────────


class TestGetRaceDetail:
    @pytest.mark.asyncio
    async def test_returns_404_when_race_not_found(self, mock_db):
        from routes.races import get_race_detail

        mock_db.execute.return_value.fetchone.return_value = None
        with pytest.raises(HTTPException) as ei:
            await get_race_detail(
                race_id="00000000-0000-0000-0000-000000000000",
                db=mock_db,
                user={"user_id": "owner"},
            )
        assert ei.value.status_code == 404

    @pytest.mark.asyncio
    async def test_shapes_response_for_frontend(self, mock_db, mock_row):
        from routes.races import get_race_detail

        rid = uuid.uuid4()
        race_row = mock_row(
            race_id=str(rid),
            race_date=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            track_name="Newton Abbot",
            race_number=3,
            distance_meters=2414,
            surface="turf",
            track_condition="Good",
            race_class="Class 4",
            purse_currency="GBP",
            purse_amount=5446.0,
            field_size=9,
            status="scheduled",
        )
        entrant_row = mock_row(
            entrant_id=str(uuid.uuid4()),
            program_number=4,
            post_position=4,
            weight_carried_lbs=159.0,
            morning_line_odds=3.75,
            starting_price=None,
            finish_position=None,
            disqualified=False,
            scratched=False,
            horse_name="Jena d'Oudairies",
            jockey_name="J. Snowden",
            trainer_name="J. Snowden",
            prediction_id=str(uuid.uuid4()),
            consensus_prob=0.308,
            consensus_field_probs={"e1": 0.308, "e2": 0.21},
            actual_outcome=None,
            is_correct=None,
            recommendation_id=str(uuid.uuid4()),
            odds_at_recommendation=3.75,
            bookmaker="Betfair Exchange",
            expected_value=0.154,
            recommended_stake=13.99,
            confidence_rating="high",
            rec_status="pending",
            profit_loss=None,
        )
        # First execute → race header; second → entrants. Wire up the
        # two-call sequence with side_effect.
        race_fetchone = MagicMock()
        race_fetchone.fetchone.return_value = race_row
        race_fetchone.fetchall.return_value = []
        entrants_fetchall = MagicMock()
        entrants_fetchall.fetchall.return_value = [entrant_row]
        entrants_fetchall.fetchone.return_value = None
        mock_db.execute.side_effect = [race_fetchone, entrants_fetchall]

        out = await get_race_detail(
            race_id=str(rid),
            db=mock_db,
            user={"user_id": "owner"},
        )
        # Top-level shape — race + entrants.
        assert set(out.keys()) == {"race", "entrants"}
        assert out["race"]["race_id"] == str(rid)
        assert out["race"]["track_name"] == "Newton Abbot"

        # Entrant shape — every key the EntrantRow component reads.
        assert len(out["entrants"]) == 1
        e = out["entrants"][0]
        assert e["horse_name"] == "Jena d'Oudairies"
        assert e["consensus_prob"] == 0.308
        # consensus_field_probs round-trips as a dict so the rec card
        # can show the field-level breakdown.
        assert e["consensus_field_probs"] == {"e1": 0.308, "e2": 0.21}
        # Recommendation surfaces as a nested dict (not flattened).
        assert e["recommendation"]["bookmaker"] == "Betfair Exchange"
        assert e["recommendation"]["expected_value"] == 0.154
        # Floats coerced from Decimal — the JSON serialiser doesn't
        # handle Decimal; if this drops back to Decimal the frontend
        # breaks at `(ev * 100).toFixed(0)`.
        assert isinstance(e["recommendation"]["recommended_stake"], float)

    @pytest.mark.asyncio
    async def test_entrant_without_recommendation_renders_null(self, mock_db, mock_row):
        # LEFT JOIN means rec fields are NULL when no value bet
        # fired. The route must shape that as `recommendation: None`
        # (a top-level null, not a dict full of nulls) so the
        # frontend's `isPicked` boolean check works.
        from routes.races import get_race_detail

        race_row = mock_row(
            race_id=str(uuid.uuid4()),
            race_date=datetime(2026, 6, 4, 15, 0, tzinfo=timezone.utc),
            track_name="Curragh",
            race_number=1,
            distance_meters=1200,
            surface="turf",
            track_condition="Good",
            race_class=None,
            purse_currency="EUR",
            purse_amount=None,
            field_size=8,
            status="scheduled",
        )
        entrant_row = mock_row(
            entrant_id=str(uuid.uuid4()),
            program_number=1,
            post_position=1,
            weight_carried_lbs=None,
            morning_line_odds=5.0,
            starting_price=None,
            finish_position=None,
            disqualified=False,
            scratched=False,
            horse_name="No Value Horse",
            jockey_name=None,
            trainer_name=None,
            prediction_id=None,
            consensus_prob=None,
            consensus_field_probs=None,
            actual_outcome=None,
            is_correct=None,
            recommendation_id=None,
            odds_at_recommendation=None,
            bookmaker=None,
            expected_value=None,
            recommended_stake=None,
            confidence_rating=None,
            rec_status=None,
            profit_loss=None,
        )
        race_fetchone = MagicMock()
        race_fetchone.fetchone.return_value = race_row
        entrants_fetchall = MagicMock()
        entrants_fetchall.fetchall.return_value = [entrant_row]
        mock_db.execute.side_effect = [race_fetchone, entrants_fetchall]

        out = await get_race_detail(
            race_id=str(uuid.uuid4()),
            db=mock_db,
            user={"user_id": "owner"},
        )
        assert out["entrants"][0]["recommendation"] is None
