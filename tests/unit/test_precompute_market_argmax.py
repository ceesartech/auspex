"""store_market_predictions argmax hygiene (scripts/precompute_predictions.py).

The correct_score market carries an aggregated 'other' tail bucket that is
usually larger than any single scoreline. Left in the argmax it became
predicted_outcome on essentially every row — permanently ungradable against
the grader's '<h>-<a>' labels (0/1124 live correct_score grades ever
matched). 'other' and '*_push' keys must both stay in the stored
probabilities JSONB (consumers need them) but never win the argmax."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


precompute = _load("precompute_predictions", "precompute_predictions.py")


class FakeCursor:
    def __init__(self):
        self.rows = []

    def execute(self, _sql, params):
        self.rows.append(params)


def test_other_bucket_never_wins_argmax():
    cur = FakeCursor()
    markets = {
        # 'other' (the tail sum) deliberately beats every concrete score.
        "correct_score": {"1-0": 0.11, "1-1": 0.10, "2-1": 0.08, "other": 0.71},
    }
    n = precompute.store_market_predictions(cur, "match-1", markets, "v1")
    assert n == 1
    row = cur.rows[0]
    assert row["predicted"] == "1-0"
    assert row["confidence"] == 0.11
    # 'other' stays in the stored JSONB for consumers.
    assert json.loads(row["probs"])["other"] == 0.71


def test_push_keys_still_excluded_and_kept_in_jsonb():
    cur = FakeCursor()
    markets = {
        "asian_handicap": {"home_-0.5": 0.30, "away_-0.5": 0.25, "line_-0.5_push": 0.45},
    }
    precompute.store_market_predictions(cur, "match-1", markets, "v1")
    row = cur.rows[0]
    assert row["predicted"] == "home_-0.5"
    assert "line_-0.5_push" in json.loads(row["probs"])


def test_all_excluded_keys_skips_market():
    cur = FakeCursor()
    markets = {"correct_score": {"other": 1.0}}
    n = precompute.store_market_predictions(cur, "match-1", markets, "v1")
    assert n == 0
    assert cur.rows == []
