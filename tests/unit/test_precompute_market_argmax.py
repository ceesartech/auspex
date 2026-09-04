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


class TestEnsembleFrameCarriesTeamIdentity:
    """The ensemble's scoreline members look up strengths by NAME.

    PoissonMatchPredictor.predict_proba falls back to np.array([""] * len(X))
    when the frame has no home_team column, and the Dixon-Coles member does the
    same, so an ensemble frame built purely from features_cache made BOTH of
    them emit their unknown-team constant on every live match — while the
    weight optimiser had handed them ~9-10% of the blend on the strength of
    their team-AWARE validation scores. A tenth of every served 1x2 probability
    was therefore a constant regardless of how many teams the artifact knew.

    This is the same failure shape as the July-2026 constant-prior incident, so
    it is pinned at the source rather than left to a reviewer to notice again.
    """

    def _serve_source(self) -> str:
        return (Path(__file__).resolve().parents[2] / "scripts" / "precompute_predictions.py").read_text()

    def test_identity_columns_are_injected_before_predict_proba(self):
        src = self._serve_source()
        idx_home = src.find('filled["home_team"]')
        idx_away = src.find('filled["away_team"]')
        idx_league = src.find('filled["league_id"]')
        idx_predict = src.find("ensemble.predict_proba(pd.DataFrame([filled]))")
        assert idx_home != -1, "home_team is never injected into the ensemble frame"
        assert idx_away != -1, "away_team is never injected into the ensemble frame"
        assert idx_league != -1, "league_id is never injected into the ensemble frame"
        assert idx_predict != -1, "the ensemble predict_proba call moved — update this test"
        for name, idx in (("home_team", idx_home), ("away_team", idx_away), ("league_id", idx_league)):
            assert idx < idx_predict, f"{name} must be set BEFORE ensemble.predict_proba"

    def test_identity_is_injected_after_the_median_fill(self):
        """The median fill replaces any non-numeric value with a feature median,
        so injecting identity before it would silently turn the team names into
        numbers and restore the constant-prior behaviour."""
        src = self._serve_source()
        idx_fill = src.find("feature_medians.get(k)")
        idx_home = src.find('filled["home_team"]')
        assert idx_fill != -1, "the median fill moved — update this test"
        assert idx_fill < idx_home, "identity columns must be set AFTER the median fill"

    def test_league_column_name_matches_what_the_model_looks_for(self):
        """poisson_models.LEAGUE_COLUMN_CANDIDATES governs the lookup; if the
        serve path injects a key that is not in that tuple the per-league
        baselines silently fall back to the global one."""
        candidates = (
            Path(__file__).resolve().parents[2] / "services" / "ml-models" / "src" / "predictors" / "poisson_models.py"
        ).read_text()
        assert 'LEAGUE_COLUMN_CANDIDATES = ("league_id"' in candidates, "league column candidates changed"
        assert 'filled["league_id"]' in self._serve_source()
