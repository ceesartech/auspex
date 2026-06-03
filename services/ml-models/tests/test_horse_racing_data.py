"""Unit tests for horse-racing training-data utilities.

The race-grouped data shape is THE load-bearing invariant for the
LambdaMART pipeline — if the group array desyncs from the
DataFrame ordering by even one row, the model will silently learn
that horse A and horse B from two unrelated races are in the same
field, and the pairwise gradient becomes garbage. These tests pin
the contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the horse_racing_data module importable without modifying
# sys.path on the package's package level.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from utils import horse_racing_data as hrd  # noqa: E402

# ── _extract_entrant_features: JSONB → per-entrant features ─────────


class TestExtractEntrantFeatures:
    def test_returns_empty_for_none(self):
        # No features_cache row for the race — caller falls back to
        # raw odds / per-row columns; never crash on None.
        assert hrd._extract_entrant_features(None, "e1") == {}

    def test_parses_json_string_blob(self):
        # The loader keeps JSONB as dicts on most psycopg2 versions
        # but ad-hoc CSV inputs can land as strings — handle both.
        blob = json.dumps(
            {
                "race_level": {"field_size": 10.0},
                "entrants": [{"entrant_id": "e1", "features": {"speed": 80.0}}],
            }
        )
        out = hrd._extract_entrant_features(blob, "e1")
        assert out["race_field_size"] == 10.0
        assert out["speed"] == 80.0

    def test_prefixes_race_level_keys(self):
        # Race-level keys live under `race_*` so they never collide
        # with per-entrant keys of the same name (e.g. field_size).
        blob = {
            "race_level": {"field_size": 8.0, "going_severity": 1.0},
            "entrants": [{"entrant_id": "e1", "features": {}}],
        }
        out = hrd._extract_entrant_features(blob, "e1")
        assert out["race_field_size"] == 8.0
        assert out["race_going_severity"] == 1.0
        assert "field_size" not in out

    def test_picks_the_right_entrant(self):
        blob = {
            "race_level": {},
            "entrants": [
                {"entrant_id": "e1", "features": {"speed": 70.0}},
                {"entrant_id": "e2", "features": {"speed": 85.0}},
            ],
        }
        assert hrd._extract_entrant_features(blob, "e2")["speed"] == 85.0

    def test_handles_missing_entrant(self):
        # Stored features cover only a subset of the field (e.g.
        # the precomputer didn't see this entrant). Return race
        # level only; trainer falls back to raw odds for the
        # rest.
        blob = {
            "race_level": {"field_size": 6.0},
            "entrants": [{"entrant_id": "other", "features": {"speed": 60.0}}],
        }
        out = hrd._extract_entrant_features(blob, "e1")
        assert out == {"race_field_size": 6.0}

    def test_handles_malformed_blob(self):
        # A bad type / non-dict shouldn't blow up — the loader
        # iterates 50k rows, one bad blob can't break the whole run.
        assert hrd._extract_entrant_features("not json", "e1") == {}
        assert hrd._extract_entrant_features(42, "e1") == {}


# ── _consensus_implied_from_odds ───────────────────────────────────


class TestConsensusImpliedFromOdds:
    def test_prefers_ml(self):
        # Same precedence as the predictor's _consensus_decimal.
        # Drift here would mean the model trains on a different
        # signal than the predictor scores with — the worst kind
        # of silent skew.
        assert hrd._consensus_implied_from_odds(2.0, 5.0) == pytest.approx(0.5)

    def test_falls_back_to_sp_when_ml_missing(self):
        assert hrd._consensus_implied_from_odds(None, 4.0) == pytest.approx(0.25)

    def test_returns_none_when_both_missing(self):
        assert hrd._consensus_implied_from_odds(None, None) is None

    def test_returns_none_when_both_stub(self):
        # Stub values (<=1.0) are non-sensical decimal odds; treat
        # the same as missing.
        assert hrd._consensus_implied_from_odds(1.0, 0.5) is None


# ── prepare_training_frame: end-to-end flatten + target ────────────


def _raw_row(*, race_id, race_date, entrant_id, finish, ml=None, sp=None, features_blob=None):
    return {
        "race_id": race_id,
        "race_date": race_date,
        "track_name": "Curragh",
        "race_number": None,
        "entrant_id": entrant_id,
        "program_number": int(entrant_id[-1]),
        "morning_line_odds": ml,
        "starting_price": sp,
        "finish_position": finish,
        "scratched": False,
        "disqualified": False,
        "consensus_prob": None,
        "features_blob": features_blob,
    }


class TestPrepareTrainingFrame:
    def _frame(self):
        raw = pd.DataFrame(
            [
                _raw_row(
                    race_id="r1",
                    race_date="2026-05-01 14:00:00+00:00",
                    entrant_id="e1",
                    finish=1,
                    ml=2.0,
                    features_blob={
                        "race_level": {"field_size": 4.0},
                        "entrants": [{"entrant_id": "e1", "features": {"speed": 85.0}}],
                    },
                ),
                _raw_row(
                    race_id="r1",
                    race_date="2026-05-01 14:00:00+00:00",
                    entrant_id="e2",
                    finish=2,
                    ml=4.0,
                    features_blob={
                        "race_level": {"field_size": 4.0},
                        "entrants": [{"entrant_id": "e2", "features": {"speed": 70.0}}],
                    },
                ),
                _raw_row(
                    race_id="r2",
                    race_date="2026-05-02 14:00:00+00:00",
                    entrant_id="e3",
                    finish=1,
                    sp=3.0,
                    features_blob=None,
                ),
            ]
        )
        return hrd.prepare_training_frame(raw)

    def test_adds_target_column(self):
        out = self._frame()
        assert list(out["target"]) == [1, 0, 1]

    def test_adds_consensus_implied_prob(self):
        out = self._frame()
        assert out["consensus_implied_prob"].iloc[0] == pytest.approx(0.5)
        assert out["consensus_implied_prob"].iloc[1] == pytest.approx(0.25)
        # Third row uses SP fallback.
        assert out["consensus_implied_prob"].iloc[2] == pytest.approx(1 / 3)

    def test_flattens_features_blob(self):
        out = self._frame()
        assert out.loc[0, "speed"] == 85.0
        assert out.loc[0, "race_field_size"] == 4.0

    def test_handles_empty_input(self):
        assert hrd.prepare_training_frame(pd.DataFrame()).empty


# ── get_feature_columns: identifiers excluded ──────────────────────


class TestGetFeatureColumns:
    def test_excludes_identifier_and_target_columns(self):
        frame = pd.DataFrame(
            {
                "race_id": ["r1"],
                "entrant_id": ["e1"],
                "target": [1],
                "consensus_prob": [0.5],
                "speed": [80.0],
                "morning_line_odds": [2.0],
                "starting_price": [3.0],
                "finish_position": [1],
                "consensus_implied_prob": [0.5],
            }
        )
        cols = hrd.get_feature_columns(frame)
        # consensus_implied_prob is KEPT (the model should see the
        # market signal); raw odds are dropped (it's the same info
        # at a different scale).
        assert "consensus_implied_prob" in cols
        assert "speed" in cols
        assert "race_id" not in cols
        assert "target" not in cols
        assert "consensus_prob" not in cols
        assert "morning_line_odds" not in cols
        assert "starting_price" not in cols

    def test_empty_frame(self):
        assert hrd.get_feature_columns(pd.DataFrame()) == []


# ── split_by_date: walk-forward boundary ───────────────────────────


class TestSplitByDate:
    def _frame(self):
        return pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r2", "r2"],
                "race_date": pd.to_datetime(
                    [
                        "2026-04-20 14:00:00+00:00",
                        "2026-04-20 14:00:00+00:00",
                        "2026-05-05 14:00:00+00:00",
                        "2026-05-05 14:00:00+00:00",
                    ],
                    utc=True,
                ),
                "target": [1, 0, 0, 1],
            }
        )

    def test_splits_on_date(self):
        train, test = hrd.split_by_date(self._frame(), "2026-05-01")
        assert list(train["race_id"]) == ["r1", "r1"]
        assert list(test["race_id"]) == ["r2", "r2"]

    def test_preserves_race_group_integrity(self):
        # Two entrants of the same race share race_date by design;
        # the split must keep them together or LambdaMART silently
        # treats them as two different races.
        train, test = hrd.split_by_date(self._frame(), "2026-05-01")
        # Each side has all entrants of its races (2 rows each).
        assert len(train) % 2 == 0
        assert len(test) % 2 == 0


# ── group_array: aligned with frame ordering ───────────────────────


class TestGroupArray:
    def test_groups_contiguous_race_ids(self):
        frame = pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r1", "r2", "r2", "r3"],
                "target": [1, 0, 0, 0, 1, 1],
            }
        )
        groups = hrd.group_array(frame)
        assert list(groups) == [3, 2, 1]
        # Sum equals row count — load-bearing invariant for LGBMRanker.
        assert int(groups.sum()) == len(frame)

    def test_empty_frame_returns_empty_array(self):
        assert len(hrd.group_array(pd.DataFrame())) == 0

    def test_missing_race_id_column_returns_empty_array(self):
        assert len(hrd.group_array(pd.DataFrame({"target": [1, 0]}))) == 0


# ── validate_training_frame: guards against silent corruption ──────


class TestValidateTrainingFrame:
    def _frame(self, races=120):
        rows = []
        for i in range(races):
            for j, pos in enumerate([1, 2, 3, 4, 5]):
                rows.append(
                    {
                        "race_id": f"r{i}",
                        "race_date": pd.Timestamp("2026-04-01", tz="UTC") + pd.Timedelta(days=i),
                        "entrant_id": f"r{i}_e{j}",
                        "target": 1 if pos == 1 else 0,
                        "feat_a": j * 1.0,
                        "feat_b": j * 2.0,
                        "feat_c": j * 0.5,
                        "feat_d": j * 1.5,
                        "feat_e": j * 0.25,
                    }
                )
        return pd.DataFrame(rows)

    def test_returns_quality_summary(self):
        q = hrd.validate_training_frame(self._frame())
        assert q.races == 120
        assert q.rows == 600
        assert q.win_rate == pytest.approx(0.2)
        assert q.feature_count >= 5

    def test_rejects_empty_frame(self):
        with pytest.raises(ValueError, match="empty"):
            hrd.validate_training_frame(pd.DataFrame())

    def test_rejects_below_min_races(self):
        with pytest.raises(ValueError, match="races"):
            hrd.validate_training_frame(self._frame(races=10))

    def test_rejects_race_with_no_winner(self):
        # Critical invariant: every race in the corpus must have a
        # finish_position=1 row, else LambdaMART has no positive
        # label to rank against. EXISTS in the SQL catches this on
        # the SELECT path; this catches CSV-import callers.
        frame = self._frame(races=120)
        frame.loc[frame["race_id"] == "r0", "target"] = 0
        with pytest.raises(ValueError, match="no winner"):
            hrd.validate_training_frame(frame)
