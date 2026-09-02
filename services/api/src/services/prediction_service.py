"""Prediction service"""

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from models.responses import MatchInfo, PredictionResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Make the shared telegram_notify module importable. It lives under
# /app/scripts/ alongside the precompute scripts because the original
# consumers were CLI scripts; the API now also enqueues alerts (so
# UI-triggered predictions show up in the digest), and re-using one
# Alert dataclass + queue helper avoids two implementations drifting.
_SCRIPTS_DIR = "/app/scripts"
if os.path.isdir(_SCRIPTS_DIR) and _SCRIPTS_DIR not in sys.path:
    sys.path.append(_SCRIPTS_DIR)

# Per-market notify thresholds for the API alert hook. Kept in sync
# with scripts/precompute_predictions_nhl.py's MARKET_NOTIFY_THRESHOLDS
# — when those move, move these too. Soccer is single-market and uses
# the same default the precompute soccer script uses.
_NHL_NOTIFY_THRESHOLDS: Dict[str, float] = {
    "moneyline": 0.60,
    "regulation": 0.55,
    "puck_line": 0.58,
    "total": 0.58,
}
_NBA_NOTIFY_THRESHOLDS: Dict[str, float] = {
    "moneyline": 0.60,
    "spread": 0.58,
    "total": 0.58,
}
_NFL_NOTIFY_THRESHOLDS: Dict[str, float] = {
    "moneyline": 0.60,
    "spread": 0.58,
    "total": 0.58,
}
# Tennis ML threshold matches the tour modal favorite at -200 (67%).
# Single market in v1 — total/set-betting come later.
_TENNIS_NOTIFY_THRESHOLDS: Dict[str, float] = {
    "moneyline": 0.65,
}
# MMA threshold matches the UFC modal favorite at -160 (62%). Single
# market in v1 — method-of-victory / round-group come later.
_MMA_NOTIFY_THRESHOLDS: Dict[str, float] = {
    "moneyline": 0.62,
}
_SOCCER_NOTIFY_THRESHOLD = 0.65

# Friendly market labels keyed by (sport, market) so the few overloaded
# names disambiguate cleanly:
#   NHL "total"  = Over/Under 5.5 goals
#   NBA "total"  = Over/Under ~225 points
#   NHL "spread" = Puck Line ±1.5
#   NBA "spread" = variable closing line (line-as-feature)
# Lookup helper falls back to market-only for legacy callers and
# finally to the raw market string.
_MARKET_DISPLAY_LABELS: Dict[tuple, str] = {
    ("soccer", "match_result"): "1X2",
    ("nhl", "moneyline"): "Moneyline",
    ("nhl", "regulation"): "Regulation (60 min)",
    ("nhl", "puck_line"): "Puck Line",
    ("nhl", "total"): "Total Goals O/U 5.5",
    ("nba", "moneyline"): "Moneyline",
    ("nba", "spread"): "Spread",
    ("nba", "total"): "Total Points",
    ("nfl", "moneyline"): "Moneyline",
    ("nfl", "spread"): "Spread",
    ("nfl", "total"): "Total Points",
    ("tennis", "moneyline"): "Match Winner",
    ("mma", "moneyline"): "Fight Winner",
}


def _display_label(sport: str, market: str) -> str:
    return _MARKET_DISPLAY_LABELS.get((sport, market), market)


# ── Multi-sport task registry ─────────────────────────────────────────
# Each entry maps a (sport, market) pair to the registry directory, label
# set, DB prediction_type, base-model list, and features_cache key needed
# to compute and persist that task's predictions. Adding a new sport or
# market means adding a TaskSpec entry — the orchestration code is
# task-agnostic. Soccer's match_result is the legacy task and uses the
# unsuffixed "ensemble" registry name for backwards compatibility with
# the pre-NHL deployment.


@dataclass(frozen=True)
class TaskSpec:
    """Configuration for one prediction task (sport + market combo).

    Naming convention, kept parallel across sports so it's obvious at a
    glance what each artifact is for:

      ensemble_<sport>_<market>      e.g. ensemble_soccer_match_result,
                                          ensemble_nhl_ml
      <base_model>_<sport>_<market>  e.g. xgboost_soccer_match_result,
                                          dixon_coles_soccer_match_result,
                                          hockey_poisson_nhl_ml
    """

    sport: str  # leagues.sport value: 'soccer' or 'nhl'
    market: str  # human-readable market label
    ensemble_name: str  # registry directory name following the convention above
    labels: List[str]  # output label order — index matches predict_proba columns
    prediction_type: str  # value persisted to predictions.prediction_type (CHECK constraint)
    base_models: List[str]  # registry directories of base models to load alongside the ensemble
    feature_set: str  # features_cache.feature_set filter for compute / lookup
    # Override for `predictions.model_name` column (and the SQL filter
    # in generate_recommendations.py). Defaults to ensemble_name, but
    # soccer pins to "ensemble" so the DB column stays stable —
    # recommendations + analytics queries reference that value and
    # renaming would require a multi-row migration. Decouples the
    # on-disk artifact identifier from the persistent DB identifier.
    db_model_name: Optional[str] = None

    @property
    def predictions_model_name(self) -> str:
        """The string written to predictions.model_name and matched by
        downstream consumers (recommendations SQL, frontend label,
        monitoring queries). Falls back to ensemble_name when not
        explicitly pinned."""
        return self.db_model_name or self.ensemble_name


# Markets the soccer ensemble's Dixon-Coles model DERIVES from a single
# match_result prediction (no separate training needed). The 1x2 model
# emits home/draw/away probabilities; the Dixon-Coles scoreline matrix
# then back-fills:
#
#   1x2 variants:      match_result, double_chance, draw_no_bet,
#                      home_away_draw
#   over/under totals: over_under_0.5 through over_under_4.5
#   both teams score:  btts (yes/no)
#   asian handicaps:   ah_-1.5, ah_-0.5, ah_+0.5, ah_+1.5
#   exact score:       top-5 most likely scorelines
#
# So `SOCCER_BUNDLE` in train_all_models.py trains EXACTLY ONE ensemble
# (soccer:match_result); the other ~15 markets are derived in
# scripts/precompute_predictions.py via derive_from_lambdas(). NHL is
# structurally different — its 4 markets (moneyline/regulation/
# puck_line/total) are direct classifiers with different label sets,
# so each gets its own SportBundle + ensemble.
DERIVED_SOCCER_MARKETS: List[str] = [
    "match_result",
    "double_chance",
    "draw_no_bet",
    "over_under_0.5",
    "over_under_1.5",
    "over_under_2.5",
    "over_under_3.5",
    "over_under_4.5",
    "btts",
    "ah_-1.5",
    "ah_-0.5",
    "ah_+0.5",
    "ah_+1.5",
    "correct_score",
]


# Task registry. The key format is "{sport}:{market}" so call sites can
# look up a task by composite key. Insertion order is stable so iteration
# at load time is deterministic.
TASKS: Dict[str, TaskSpec] = {
    # Soccer trains ONE ensemble for match_result (1X2). Everything in
    # DERIVED_SOCCER_MARKETS above gets reconstructed at predict time
    # from this ensemble + Dixon-Coles, no per-market training needed.
    "soccer:match_result": TaskSpec(
        sport="soccer",
        market="match_result",
        ensemble_name="ensemble_soccer_match_result",
        labels=["home", "draw", "away"],
        prediction_type="match_result",
        base_models=[
            "xgboost_soccer_match_result",
            "lightgbm_soccer_match_result",
            "neural_network_soccer_match_result",
            "poisson_soccer_match_result",
            "dixon_coles_soccer_match_result",
        ],
        # DB column stays "ensemble" (existing rows + recommendations
        # SQL reference it). The on-disk artifact name and the DB
        # identifier are intentionally decoupled.
        db_model_name="ensemble",
        feature_set="baseline",
    ),
    "nhl:moneyline": TaskSpec(
        sport="nhl",
        market="moneyline",
        ensemble_name="ensemble_nhl_ml",
        labels=["home", "away"],
        prediction_type="moneyline",
        base_models=["xgboost_nhl_ml", "lightgbm_nhl_ml", "neural_network_nhl_ml", "hockey_poisson_nhl_ml"],
        feature_set="nhl_baseline",
    ),
    "nhl:regulation": TaskSpec(
        sport="nhl",
        market="regulation",
        ensemble_name="ensemble_nhl_reg",
        # 3-class: home reg win / regulation tie / away reg win
        labels=["home", "tie", "away"],
        # 'match_result' is the closest existing CHECK-allowed value; a
        # follow-up migration could add 'regulation_winner' explicitly.
        prediction_type="match_result",
        base_models=["xgboost_nhl_reg", "lightgbm_nhl_reg", "neural_network_nhl_reg", "hockey_poisson_nhl_reg"],
        feature_set="nhl_baseline",
    ),
    "nhl:puck_line": TaskSpec(
        sport="nhl",
        market="puck_line",
        ensemble_name="ensemble_nhl_pl",
        labels=["cover", "no_cover"],
        prediction_type="spread",
        base_models=["xgboost_nhl_pl", "lightgbm_nhl_pl", "neural_network_nhl_pl", "hockey_poisson_nhl_pl"],
        feature_set="nhl_baseline",
    ),
    "nhl:total": TaskSpec(
        sport="nhl",
        market="total",
        ensemble_name="ensemble_nhl_tot",
        labels=["over", "under"],
        prediction_type="total",
        base_models=["xgboost_nhl_tot", "lightgbm_nhl_tot", "neural_network_nhl_tot", "hockey_poisson_nhl_tot"],
        feature_set="nhl_baseline",
    ),
    # NBA: line-as-feature design. Spread + total models consume the
    # actual closing line as an input column so one trained model
    # handles every line the book offers per game. No Poisson —
    # basketball scoring is too continuous / high-variance for a
    # discrete prior. See services/ml-models/src/training/
    # train_all_models.py for the matching bundle definitions.
    "nba:moneyline": TaskSpec(
        sport="nba",
        market="moneyline",
        ensemble_name="ensemble_nba_ml",
        labels=["home", "away"],
        prediction_type="moneyline",
        base_models=["xgboost_nba_ml", "lightgbm_nba_ml", "neural_network_nba_ml"],
        feature_set="nba_baseline",
    ),
    "nba:spread": TaskSpec(
        sport="nba",
        market="spread",
        ensemble_name="ensemble_nba_sp",
        # 'home' = home covers closing line, 'away' = away covers
        # (i.e. home did not cover). The CONDITIONAL probability —
        # given the closing line, what's the chance home covers?
        labels=["home", "away"],
        prediction_type="spread",
        base_models=["xgboost_nba_sp", "lightgbm_nba_sp", "neural_network_nba_sp"],
        feature_set="nba_baseline",
    ),
    "nba:total": TaskSpec(
        sport="nba",
        market="total",
        ensemble_name="ensemble_nba_tot",
        labels=["over", "under"],
        prediction_type="total",
        base_models=["xgboost_nba_tot", "lightgbm_nba_tot", "neural_network_nba_tot"],
        feature_set="nba_baseline",
    ),
    # NFL: same shape as NBA — three markets (moneyline, spread, total)
    # with line-as-feature for spread + total. No Poisson / Dixon-Coles
    # entries — NFL scoring is too matchup-specific for a discrete
    # prior to help. Bundles in
    # services/ml-models/src/training/train_all_models.py.
    "nfl:moneyline": TaskSpec(
        sport="nfl",
        market="moneyline",
        ensemble_name="ensemble_nfl_ml",
        labels=["home", "away"],
        prediction_type="moneyline",
        base_models=["xgboost_nfl_ml", "lightgbm_nfl_ml", "neural_network_nfl_ml"],
        feature_set="nfl_baseline",
    ),
    "nfl:spread": TaskSpec(
        sport="nfl",
        market="spread",
        ensemble_name="ensemble_nfl_sp",
        labels=["home", "away"],
        prediction_type="spread",
        base_models=["xgboost_nfl_sp", "lightgbm_nfl_sp", "neural_network_nfl_sp"],
        feature_set="nfl_baseline",
    ),
    "nfl:total": TaskSpec(
        sport="nfl",
        market="total",
        ensemble_name="ensemble_nfl_tot",
        labels=["over", "under"],
        prediction_type="total",
        base_models=["xgboost_nfl_tot", "lightgbm_nfl_tot", "neural_network_nfl_tot"],
        feature_set="nfl_baseline",
    ),
    # Tennis: first 1v1 sport. labels=["home", "away"] map to
    # player1/player2 — the positional convention from
    # fetch_upcoming.process_event when is_individual=True. Single
    # market in v1 (moneyline); total games + set-betting are v2
    # once linescores are parsed.
    "tennis:moneyline": TaskSpec(
        sport="tennis",
        market="moneyline",
        ensemble_name="ensemble_tennis_ml",
        labels=["home", "away"],
        prediction_type="moneyline",
        base_models=["xgboost_tennis_ml", "lightgbm_tennis_ml", "neural_network_tennis_ml"],
        feature_set="tennis_baseline",
    ),
    # MMA: second 1v1 sport. labels=["home", "away"] map to
    # fighter1/fighter2 — positional convention from process_event
    # when is_individual=True (MMA payloads lack homeAway).
    "mma:moneyline": TaskSpec(
        sport="mma",
        market="moneyline",
        ensemble_name="ensemble_mma_ml",
        labels=["home", "away"],
        prediction_type="moneyline",
        base_models=["xgboost_mma_ml", "lightgbm_mma_ml", "neural_network_mma_ml"],
        feature_set="mma_baseline",
    ),
}


def tasks_for_sport(sport: str) -> List[TaskSpec]:
    """All registered tasks for a given sport, in registry order."""
    return [spec for spec in TASKS.values() if spec.sport == sport]


def headline_task(sport: str) -> Optional[TaskSpec]:
    """The HEADLINE task for a sport — the market shown when a caller
    doesn't ask for one (soccer→match_result, every other sport→
    moneyline). Defined as the FIRST TaskSpec registered for the sport,
    so registry order is load-bearing: keep the headline entry first
    when adding markets. Returns None for sports with no TaskSpec
    (e.g. horse_racing), which callers treat as "no market filter".
    """
    for spec in TASKS.values():
        if spec.sport == sport:
            return spec
    return None


# "<sport>:<prediction_type>" pairs for every sport's HEADLINE task —
# today ['mma:moneyline', 'nba:moneyline', 'nfl:moneyline',
# 'nhl:moneyline', 'soccer:match_result', 'tennis:moneyline']. Used by
# the cross-sport upcoming list to keep ONE row per match without
# knowing the sport up front. Pairs (not bare prediction_types) because
# a non-headline market can reuse a headline prediction_type for a
# different sport: NHL 'regulation' persists as 'match_result', which
# a bare-type filter would let through as a second NHL row per game.
# Sorted so the SQL bind is deterministic (tests + logs).
HEADLINE_PAIRS: List[str] = sorted(
    f"{spec.sport}:{spec.prediction_type}" for spec in TASKS.values() if headline_task(spec.sport) is spec
)


# Process-wide model registry. Populated once during app startup
# (see services/api/src/main.py lifespan) and shared across requests.
# Loading joblib pickles on every request adds ~10-50ms; this avoids that.
# Keys are the TaskSpec composite key ("soccer:match_result",
# "nhl:moneyline", etc.). Soccer is also mirrored under the bare
# ensemble_name ("ensemble_soccer_match_result") so call sites that
# look up by ensemble_name resolve.
_MODELS: Dict[str, Any] = {}
_MODEL_VERSIONS: Dict[str, str] = {}
_MODEL_VERSION: str = "ensemble_v1.0"  # Legacy global; soccer version


def get_loaded_models() -> Dict[str, Any]:
    """Return the process-wide loaded model registry."""
    return _MODELS


def get_model_version(task_key: Optional[str] = None) -> str:
    """Version string for a task's ensemble. Defaults to soccer match_result
    for backwards compatibility with cache-key callers."""
    if task_key and task_key in _MODEL_VERSIONS:
        return _MODEL_VERSIONS[task_key]
    return _MODEL_VERSION


def _predict_one(model, features: Dict[str, Any], labels: List[str]) -> Dict[str, Any]:
    """Adapter from the (features dict) call shape used by the route + DAG
    to EnsemblePredictor.predict_proba(DataFrame).

    `labels` defines the output label order — must match the model's
    predict_proba column order (which in turn matches the LabelEncoder
    that was fit at training time). 2-class tasks (NHL moneyline/PL/total)
    pass 2 labels; 3-class tasks (soccer match_result, NHL regulation)
    pass 3.
    """
    import pandas as pd

    # Mirror every key into its feature__-prefixed form. The training
    # queries SELECT raw columns (odds_home_ml, etc.) alongside the
    # features_cache JSONB which gets flattened to feature__* names via
    # utils.training_data._flatten_features. So each trained model's
    # feature_names list contains BOTH versions. The features dict we
    # get here comes only from the JSONB blob (unprefixed). Mirroring
    # both forms lets the model's X[self.feature_names] lookup find
    # every column it expects. Existing feature__X entries are left
    # alone.
    X_dict: Dict[str, Any] = {}
    for k, v in features.items():
        X_dict[k] = v
        prefixed = k if k.startswith("feature__") else f"feature__{k}"
        if prefixed not in X_dict:
            X_dict[prefixed] = v
    X = pd.DataFrame([X_dict])

    proba = model.predict_proba(X)[0]
    if len(proba) != len(labels):
        raise ValueError(
            f"predict_proba returned {len(proba)} classes but {len(labels)} labels were provided. " f"Labels: {labels}"
        )
    idx = int(proba.argmax())
    return {
        "predicted_label": labels[idx],
        "confidence": float(proba[idx]),
        "probabilities": {labels[i]: float(proba[i]) for i in range(len(labels))},
    }


def _latest_model_bin(root: Path) -> Optional[Path]:
    """Return the most recent `*/model.bin` under `root` (by mtime), or None."""
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*/model.bin"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _klass_registry():
    """Build the {base_model_name → (predictor class, ModelConfig)} map.

    Pulled into a helper so we can call it once per process and reuse
    across multiple ensemble loads. Each NHL ensemble references its
    own per-task config (e.g. XGBOOST_NHL_MONEYLINE) so the LabelEncoder
    and num_class settings match what was used at training time.
    """
    import sys

    # The ml-models predictor classes live under `predictors.*` (renamed
    # from the old `models.*` to avoid collision with api/src/models/,
    # which is the pydantic schema package). PYTHONPATH from
    # Dockerfile.api includes /app/services/ml-models/src so these
    # resolve directly.
    ml_src = "/app/services/ml-models/src"
    if ml_src not in sys.path:
        sys.path.insert(0, ml_src)

    from predictors.lightgbm_model import LightGBMMatchPredictor
    from predictors.model_config import (
        DIXON_COLES_CONFIG,
        HOCKEY_POISSON_NHL_MONEYLINE,
        HOCKEY_POISSON_NHL_PUCK_LINE,
        HOCKEY_POISSON_NHL_REGULATION,
        HOCKEY_POISSON_NHL_TOTAL,
        LIGHTGBM_MATCH_OUTCOME,
        LIGHTGBM_MMA_MONEYLINE,
        LIGHTGBM_NBA_MONEYLINE,
        LIGHTGBM_NBA_SPREAD,
        LIGHTGBM_NBA_TOTAL,
        LIGHTGBM_NFL_MONEYLINE,
        LIGHTGBM_NFL_SPREAD,
        LIGHTGBM_NFL_TOTAL,
        LIGHTGBM_NHL_MONEYLINE,
        LIGHTGBM_NHL_PUCK_LINE,
        LIGHTGBM_NHL_REGULATION,
        LIGHTGBM_NHL_TOTAL,
        LIGHTGBM_TENNIS_MONEYLINE,
        NEURAL_NETWORK_CONFIG,
        NEURAL_NETWORK_MMA_MONEYLINE,
        NEURAL_NETWORK_NBA_MONEYLINE,
        NEURAL_NETWORK_NBA_SPREAD,
        NEURAL_NETWORK_NBA_TOTAL,
        NEURAL_NETWORK_NFL_MONEYLINE,
        NEURAL_NETWORK_NFL_SPREAD,
        NEURAL_NETWORK_NFL_TOTAL,
        NEURAL_NETWORK_NHL_MONEYLINE,
        NEURAL_NETWORK_NHL_PUCK_LINE,
        NEURAL_NETWORK_NHL_REGULATION,
        NEURAL_NETWORK_NHL_TOTAL,
        NEURAL_NETWORK_TENNIS_MONEYLINE,
        POISSON_CONFIG,
        XGBOOST_MATCH_OUTCOME,
        XGBOOST_MMA_MONEYLINE,
        XGBOOST_NBA_MONEYLINE,
        XGBOOST_NBA_SPREAD,
        XGBOOST_NBA_TOTAL,
        XGBOOST_NFL_MONEYLINE,
        XGBOOST_NFL_SPREAD,
        XGBOOST_NFL_TOTAL,
        XGBOOST_NHL_MONEYLINE,
        XGBOOST_NHL_PUCK_LINE,
        XGBOOST_NHL_REGULATION,
        XGBOOST_NHL_TOTAL,
        XGBOOST_TENNIS_MONEYLINE,
    )
    from predictors.neural_network import NeuralNetworkMatchPredictor
    from predictors.poisson_models import DixonColesPredictor, HockeyPoissonPredictor, PoissonMatchPredictor
    from predictors.xgboost_model import XGBoostMatchPredictor

    return {
        # Soccer match_result base models. Same *_<sport>_<market>
        # naming convention as NHL.
        "xgboost_soccer_match_result": (XGBoostMatchPredictor, XGBOOST_MATCH_OUTCOME),
        "lightgbm_soccer_match_result": (LightGBMMatchPredictor, LIGHTGBM_MATCH_OUTCOME),
        "neural_network_soccer_match_result": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_CONFIG),
        "poisson_soccer_match_result": (PoissonMatchPredictor, POISSON_CONFIG),
        "dixon_coles_soccer_match_result": (DixonColesPredictor, DIXON_COLES_CONFIG),
        # NHL moneyline
        "xgboost_nhl_ml": (XGBoostMatchPredictor, XGBOOST_NHL_MONEYLINE),
        "lightgbm_nhl_ml": (LightGBMMatchPredictor, LIGHTGBM_NHL_MONEYLINE),
        "neural_network_nhl_ml": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NHL_MONEYLINE),
        "hockey_poisson_nhl_ml": (HockeyPoissonPredictor, HOCKEY_POISSON_NHL_MONEYLINE),
        # NHL regulation
        "xgboost_nhl_reg": (XGBoostMatchPredictor, XGBOOST_NHL_REGULATION),
        "lightgbm_nhl_reg": (LightGBMMatchPredictor, LIGHTGBM_NHL_REGULATION),
        "neural_network_nhl_reg": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NHL_REGULATION),
        "hockey_poisson_nhl_reg": (HockeyPoissonPredictor, HOCKEY_POISSON_NHL_REGULATION),
        # NHL puck line
        "xgboost_nhl_pl": (XGBoostMatchPredictor, XGBOOST_NHL_PUCK_LINE),
        "lightgbm_nhl_pl": (LightGBMMatchPredictor, LIGHTGBM_NHL_PUCK_LINE),
        "neural_network_nhl_pl": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NHL_PUCK_LINE),
        "hockey_poisson_nhl_pl": (HockeyPoissonPredictor, HOCKEY_POISSON_NHL_PUCK_LINE),
        # NHL total
        "xgboost_nhl_tot": (XGBoostMatchPredictor, XGBOOST_NHL_TOTAL),
        "lightgbm_nhl_tot": (LightGBMMatchPredictor, LIGHTGBM_NHL_TOTAL),
        "neural_network_nhl_tot": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NHL_TOTAL),
        "hockey_poisson_nhl_tot": (HockeyPoissonPredictor, HOCKEY_POISSON_NHL_TOTAL),
        # NBA moneyline
        "xgboost_nba_ml": (XGBoostMatchPredictor, XGBOOST_NBA_MONEYLINE),
        "lightgbm_nba_ml": (LightGBMMatchPredictor, LIGHTGBM_NBA_MONEYLINE),
        "neural_network_nba_ml": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NBA_MONEYLINE),
        # NBA spread (line-as-feature)
        "xgboost_nba_sp": (XGBoostMatchPredictor, XGBOOST_NBA_SPREAD),
        "lightgbm_nba_sp": (LightGBMMatchPredictor, LIGHTGBM_NBA_SPREAD),
        "neural_network_nba_sp": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NBA_SPREAD),
        # NBA total (line-as-feature)
        "xgboost_nba_tot": (XGBoostMatchPredictor, XGBOOST_NBA_TOTAL),
        "lightgbm_nba_tot": (LightGBMMatchPredictor, LIGHTGBM_NBA_TOTAL),
        "neural_network_nba_tot": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NBA_TOTAL),
        # NFL moneyline
        "xgboost_nfl_ml": (XGBoostMatchPredictor, XGBOOST_NFL_MONEYLINE),
        "lightgbm_nfl_ml": (LightGBMMatchPredictor, LIGHTGBM_NFL_MONEYLINE),
        "neural_network_nfl_ml": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NFL_MONEYLINE),
        # NFL spread (line-as-feature)
        "xgboost_nfl_sp": (XGBoostMatchPredictor, XGBOOST_NFL_SPREAD),
        "lightgbm_nfl_sp": (LightGBMMatchPredictor, LIGHTGBM_NFL_SPREAD),
        "neural_network_nfl_sp": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NFL_SPREAD),
        # NFL total (line-as-feature)
        "xgboost_nfl_tot": (XGBoostMatchPredictor, XGBOOST_NFL_TOTAL),
        "lightgbm_nfl_tot": (LightGBMMatchPredictor, LIGHTGBM_NFL_TOTAL),
        "neural_network_nfl_tot": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_NFL_TOTAL),
        # Tennis moneyline (first 1v1 sport)
        "xgboost_tennis_ml": (XGBoostMatchPredictor, XGBOOST_TENNIS_MONEYLINE),
        "lightgbm_tennis_ml": (LightGBMMatchPredictor, LIGHTGBM_TENNIS_MONEYLINE),
        "neural_network_tennis_ml": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_TENNIS_MONEYLINE),
        # MMA moneyline (second 1v1 sport)
        "xgboost_mma_ml": (XGBoostMatchPredictor, XGBOOST_MMA_MONEYLINE),
        "lightgbm_mma_ml": (LightGBMMatchPredictor, LIGHTGBM_MMA_MONEYLINE),
        "neural_network_mma_ml": (NeuralNetworkMatchPredictor, NEURAL_NETWORK_MMA_MONEYLINE),
    }


def _build_ensemble_for_task(
    model_path: Path,
    task: TaskSpec,
    klass_for: Dict[str, tuple],
) -> Tuple[Optional[Any], Optional[Path]]:
    """Reconstitute the EnsemblePredictor + its base models for one task.

    The training pipeline writes:
      {model_path}/{task.ensemble_name}/{version}/model.bin (JSON: weights + model_names)
      {model_path}/{base_model_name}/{version}/model.bin (per base model)

    EnsemblePredictor.save() only persists weights + base-model names —
    not the base models themselves — so reloading requires us to walk
    the registry, instantiate each referenced base model class, load
    its artifact, and re-attach via add_model() before applying weights.

    Returns (ensemble, meta_path) or (None, None) if the task can't be
    loaded (missing artifacts, etc.) — caller logs the warning and
    continues.
    """
    import json
    import sys

    ml_src = "/app/services/ml-models/src"
    if ml_src not in sys.path:
        sys.path.insert(0, ml_src)

    from predictors.ensemble import EnsemblePredictor
    from predictors.model_config import (
        ENSEMBLE_CONFIG,
        ENSEMBLE_MMA_MONEYLINE,
        ENSEMBLE_NBA_MONEYLINE,
        ENSEMBLE_NBA_SPREAD,
        ENSEMBLE_NBA_TOTAL,
        ENSEMBLE_NFL_MONEYLINE,
        ENSEMBLE_NFL_SPREAD,
        ENSEMBLE_NFL_TOTAL,
        ENSEMBLE_NHL_MONEYLINE,
        ENSEMBLE_NHL_PUCK_LINE,
        ENSEMBLE_NHL_REGULATION,
        ENSEMBLE_NHL_TOTAL,
        ENSEMBLE_TENNIS_MONEYLINE,
    )

    # Pick the task-specific ensemble config so the EnsemblePredictor's
    # internal LabelEncoder and prediction_task match what was used at
    # training time. Falls back to ENSEMBLE_CONFIG (soccer's) for any
    # unmapped key.
    ensemble_cfg_for = {
        "ensemble_soccer_match_result": ENSEMBLE_CONFIG,
        "ensemble_nhl_ml": ENSEMBLE_NHL_MONEYLINE,
        "ensemble_nhl_reg": ENSEMBLE_NHL_REGULATION,
        "ensemble_nhl_pl": ENSEMBLE_NHL_PUCK_LINE,
        "ensemble_nhl_tot": ENSEMBLE_NHL_TOTAL,
        "ensemble_nba_ml": ENSEMBLE_NBA_MONEYLINE,
        "ensemble_nba_sp": ENSEMBLE_NBA_SPREAD,
        "ensemble_nba_tot": ENSEMBLE_NBA_TOTAL,
        "ensemble_nfl_ml": ENSEMBLE_NFL_MONEYLINE,
        "ensemble_nfl_sp": ENSEMBLE_NFL_SPREAD,
        "ensemble_nfl_tot": ENSEMBLE_NFL_TOTAL,
        "ensemble_tennis_ml": ENSEMBLE_TENNIS_MONEYLINE,
        "ensemble_mma_ml": ENSEMBLE_MMA_MONEYLINE,
    }
    ensemble_cfg = ensemble_cfg_for.get(task.ensemble_name, ENSEMBLE_CONFIG)

    ensemble_meta = _latest_model_bin(model_path / task.ensemble_name)
    if ensemble_meta is None:
        return None, None

    with open(ensemble_meta) as f:
        meta = json.load(f)

    ensemble = EnsemblePredictor(ensemble_cfg)
    loaded_names: List[str] = []
    for name in meta.get("model_names", []):
        if name not in klass_for:
            logger.warning("Task %s ensemble references unknown base model %r; skipping", task.ensemble_name, name)
            continue
        artifact = _latest_model_bin(model_path / name)
        if artifact is None:
            logger.warning(
                "Task %s ensemble references %r but no artifact under %s/%s/",
                task.ensemble_name,
                name,
                model_path,
                name,
            )
            continue
        cls, cfg = klass_for[name]
        try:
            model = cls(cfg)
            model.load(str(artifact))
            ensemble.add_model(name, model)
            loaded_names.append(name)
        except Exception as e:
            logger.error("Failed to load base model %r from %s: %s", name, artifact, e)

    if not loaded_names:
        logger.error("Could not reconstitute any base models for task %s", task.ensemble_name)
        return None, None

    # ensemble.load() restores the blend weights from JSON metadata.
    ensemble.load(str(ensemble_meta))
    return ensemble, ensemble_meta


def load_models_into_process(model_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load ML models into the process-wide registry. Idempotent.

    Called once from FastAPI's lifespan startup. Walks every TaskSpec in
    the TASKS registry and tries to load the corresponding ensemble +
    base models from disk. Tasks without on-disk artifacts (e.g. NHL
    ensembles on a fresh deployment where Phase 3 hasn't run yet) get
    a single warning and are skipped — the rest still load. Returns
    the loaded model registry keyed by task composite key.
    """
    global _MODEL_VERSION

    if _MODELS:
        return _MODELS

    model_path = model_dir or (Path(settings.MODEL_PATH) / "production")
    try:
        klass_for = _klass_registry()
    except Exception as e:
        logger.error("Could not import predictor classes: %s", e, exc_info=True)
        return _MODELS

    for task_key, task in TASKS.items():
        try:
            ensemble, meta_path = _build_ensemble_for_task(model_path, task, klass_for)
        except Exception as e:
            logger.error("Failed to load task %s: %s", task_key, e, exc_info=True)
            continue
        if ensemble is None:
            logger.warning(
                "Task %s: ensemble not found under %s/%s/*/model.bin; skipping",
                task_key,
                model_path,
                task.ensemble_name,
            )
            continue
        _MODELS[task_key] = ensemble
        mtime = int(meta_path.stat().st_mtime)
        version_label = meta_path.parent.name
        _MODEL_VERSIONS[task_key] = f"{task.ensemble_name}_{version_label}+{mtime}"
        logger.info(
            "Loaded task %s (version=%s, base_models=%s)",
            task_key,
            _MODEL_VERSIONS[task_key],
            list(ensemble.models.keys()),
        )

    # Mirror soccer's loaded model under the bare explicit name so
    # call sites that look up by ensemble_name (instead of composite
    # task key) still resolve. _MODEL_VERSION stays a global for
    # callers that want a default cache-key version string.
    soccer_key = "soccer:match_result"
    if soccer_key in _MODELS:
        _MODELS["ensemble_soccer_match_result"] = _MODELS[soccer_key]
        _MODEL_VERSION = _MODEL_VERSIONS[soccer_key]

    return _MODELS


class PredictionService:
    """Service for generating predictions"""

    def __init__(self, db: Optional[Session] = None, models: Optional[Dict[str, Any]] = None):
        self.db = db
        # Default to the process-wide registry. Tests can pass their own.
        self.models: Dict[str, Any] = models if models is not None else _MODELS

    def _require_db(self) -> Session:
        if self.db is None:
            raise RuntimeError("PredictionService requires a database session for this operation")
        return self.db

    def load_models(self) -> None:
        """Compatibility shim. Real loading happens in load_models_into_process()."""
        load_models_into_process()
        self.models = _MODELS

    def predict_match(
        self,
        match_id: str,
        include_explanation: bool = True,
        include_alternate_models: bool = False,
        market: Optional[str] = None,
    ) -> PredictionResponse:
        """Generate prediction(s) for a match. Routes based on the match's
        sport: soccer matches predict 1 task (match_result); NHL matches
        predict 4 tasks (moneyline, regulation, puck_line, total) and
        return the moneyline as the headline response (since it's the
        most-marketable single prediction).

        `market` lets the caller pin to a specific task instead of the
        headline default — useful for the API when a user explicitly
        wants e.g. NHL totals.
        """

        match_data = self._get_match_data(match_id)
        if not match_data:
            raise ValueError(f"Match {match_id} not found")

        sport = match_data.get("sport") or "soccer"
        features = self._get_match_features(match_id, feature_set=None)

        # Pick which task(s) to run for this match.
        applicable = tasks_for_sport(sport)
        if not applicable:
            # Unknown sport — fall through to stored-prediction lookup.
            prediction = self._get_stored_prediction(match_id)
            if not prediction:
                raise ValueError(f"No tasks registered for sport={sport!r}")
            return self._response_from_prediction(match_data, prediction, model_version=_MODEL_VERSION)

        # Determine the headline task (the one whose response we return).
        headline_task: Optional[TaskSpec] = None
        if market:
            headline_task = next((t for t in applicable if t.market == market), None)
            if headline_task is None:
                raise ValueError(
                    f"No task for sport={sport!r} market={market!r}. "
                    f"Available markets: {[t.market for t in applicable]}"
                )
        else:
            # Default headline: match_result for soccer, moneyline for NHL.
            headline_task = applicable[0]

        # Run + persist every applicable task; remember the headline
        # prediction for the response object.
        headline_prediction: Optional[Dict[str, Any]] = None
        for task in applicable:
            task_key = f"{task.sport}:{task.market}"
            model = self.models.get(task_key)
            if model is None:
                logger.warning("Skipping task %s: model not loaded", task_key)
                continue
            try:
                prediction = _predict_one(model, features, task.labels)
            except Exception as e:
                logger.error("predict failed for task %s match %s: %s", task_key, match_id, e)
                continue
            self._store_prediction(match_id, prediction, task)
            # Enqueue Telegram alert for high-confidence picks. This is
            # the same Redis queue the precompute scripts use; the
            # send_pipeline_digest DAG task drains and sends them in
            # one combined message. Without this hook, UI-triggered
            # predictions (predict_match path) would store to DB but
            # never reach Telegram — that's the regression the
            # consolidation change introduced.
            self._maybe_enqueue_alert(prediction, task, match_data)
            if task is headline_task:
                headline_prediction = prediction

        if headline_prediction is None:
            # Final fallback: stored prediction from DB so the request
            # still returns something usable while we wait for models
            # to populate.
            headline_prediction = self._get_stored_prediction(match_id)
            if not headline_prediction:
                raise ValueError(f"No prediction available for match {match_id}")

        headline_version = _MODEL_VERSIONS.get(
            f"{headline_task.sport}:{headline_task.market}",
            _MODEL_VERSION,
        )
        response = self._response_from_prediction(
            match_data,
            headline_prediction,
            model_version=headline_version,
            include_explanation=include_explanation,
        )
        return response

    def _response_from_prediction(
        self,
        match_data: Dict,
        prediction: Dict,
        model_version: str,
        include_explanation: bool = True,
    ) -> PredictionResponse:
        """Build a PredictionResponse from a prediction dict. Filters
        out the 'sport' key from match_data before passing to MatchInfo
        since MatchInfo doesn't accept it."""
        match_info_kwargs = {k: v for k, v in match_data.items() if k != "sport"}
        return PredictionResponse(
            match_info=MatchInfo(**match_info_kwargs),
            predicted_outcome=prediction["predicted_label"],
            probabilities=prediction["probabilities"],
            confidence=prediction["confidence"],
            model_version=model_version,
            timestamp=datetime.utcnow(),
            explanation=prediction.get("explanation") if include_explanation else None,
        )

    def get_upcoming_predictions(
        self,
        sport: Optional[str] = None,
        league: Optional[str] = None,
        market: Optional[str] = None,
        limit: int = 20,
    ) -> List[PredictionResponse]:
        """Get predictions for upcoming matches — ONE ROW PER MATCH by
        default.

        Every sport stores several prediction_type rows per match
        (soccer: 19 — match_result plus ~18 Dixon-Coles-derived markets
        like over_under / asian_handicap; NHL: 4; NFL/NBA: 3). The SQL
        LIMIT applies to ROWS, so an unfiltered list would be saturated
        by a handful of matches (prod: limit=50 rendered ~3 soccer
        matches and whole leagues never appeared). With `market` omitted
        we therefore filter to each sport's HEADLINE market, derived
        from the TASKS registry via headline_task():

          * sport given  → that sport's headline prediction_type
                           (soccer→match_result, nhl/nba/nfl/tennis/
                           mma→moneyline). Sports with no TaskSpec keep
                           no filter.
          * sport omitted → (sport, prediction_type) restricted to each
                           sport's headline pair (HEADLINE_PAIRS), bound
                           as a list of "sport:type" strings and matched
                           with `(l.sport || ':' || p.prediction_type)
                           = ANY(:headline_pairs)`. Pairs, not bare
                           types, so NHL 'regulation' (persisted as
                           'match_result') doesn't leak in as a second
                           NHL row per game.

        Set `market` explicitly to override: friendly names map through
        TASKS (e.g. market='puck_line' for NHL → prediction_type
        'spread'); unknown strings pass straight through as a raw
        prediction_type filter (e.g. market='over_under' for soccer).
        Results are ordered by match_date ASC.
        """

        # Map the friendly market name through the TASKS registry to the
        # DB prediction_type value. Unknown markets pass through to the
        # SQL filter unchanged so callers can target prediction_type
        # values that don't have a TaskSpec yet (forward-compat).
        prediction_type: Optional[str] = None
        headline_pairs: Optional[List[str]] = None
        if market is not None:
            matched_task = next(
                (t for t in TASKS.values() if t.sport == (sport or t.sport) and t.market == market),
                None,
            )
            prediction_type = matched_task.prediction_type if matched_task else market
        elif sport is not None:
            # Default the single-sport list to its headline market so
            # one row per match. Sports without a TaskSpec (e.g.
            # horse_racing) get no filter.
            headline = headline_task(sport)
            prediction_type = headline.prediction_type if headline else None
        else:
            # Cross-sport list: keep only rows whose (sport,
            # prediction_type) is that sport's headline pair. Sports
            # with no TaskSpec (e.g. horse_racing) have no pair and are
            # therefore excluded from the default cross-sport list —
            # same as before (they have no 'match_result'/'moneyline'
            # rows either).
            headline_pairs = list(HEADLINE_PAIRS)

        query = text(
            """
            SELECT p.id, p.match_id, p.predicted_outcome, p.confidence,
                   p.probabilities, p.model_name, p.model_version,
                   p.prediction_type,
                   p.features_used, p.feature_importance,
                   m.match_date, m.venue,
                   l.name as league_name,
                   ht.name as home_team, at.name as away_team
            FROM predictions p
            JOIN matches m ON p.match_id = m.id
            JOIN leagues l ON m.league_id = l.id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.status = 'scheduled' AND m.match_date > NOW()
            AND (:sport IS NULL OR l.sport = :sport)
            AND (:league IS NULL OR l.name = :league)
            AND (:prediction_type IS NULL OR p.prediction_type = :prediction_type)
            AND (:headline_pairs IS NULL OR (l.sport || ':' || p.prediction_type) = ANY(:headline_pairs))
            ORDER BY m.match_date ASC
            LIMIT :limit
        """
        )

        db = self._require_db()
        results = db.execute(
            query,
            {
                "sport": sport,
                "league": league,
                "prediction_type": prediction_type,
                # psycopg2 adapts a Python list to a Postgres ARRAY, so
                # `= ANY(:headline_pairs)` works without an expanding
                # bindparam; None short-circuits the clause.
                "headline_pairs": headline_pairs,
                "limit": limit,
            },
        ).fetchall()

        # Same (ensemble_name, prediction_type) → market mapping used by
        # get_match_predictions; keeps both endpoints labeling rows
        # consistently for the frontend.
        ensemble_to_market = {(t.ensemble_name, t.prediction_type): t.market for t in TASKS.values()}

        predictions = []
        for row in results:
            market_label = ensemble_to_market.get(
                (row.model_name, row.prediction_type),
                row.prediction_type,
            )
            predictions.append(
                PredictionResponse(
                    match_info=MatchInfo(
                        match_id=str(row.match_id),
                        league_name=row.league_name,
                        home_team=row.home_team,
                        away_team=row.away_team,
                        match_date=row.match_date,
                        venue=row.venue,
                    ),
                    predicted_outcome=row.predicted_outcome,
                    probabilities=row.probabilities or {},
                    confidence=float(row.confidence),
                    model_version=row.model_version,
                    timestamp=datetime.utcnow(),
                    market=market_label,
                )
            )

        return predictions

    def get_match_predictions(self, match_id: str) -> List[PredictionResponse]:
        """Return every stored prediction for a single match — one row
        per (model, prediction_type). Soccer matches return match_result
        plus ~18 Dixon-Coles-derived markets (asian_handicap, over_under,
        btts, correct_score, ...); NHL matches return up to four
        (moneyline, regulation, puck_line, total).

        Ordering is stable and the frontend relies on it: the sport's
        HEADLINE market (headline_task(): soccer→match_result, others→
        moneyline) is ALWAYS index 0, followed by the remaining markets
        in prediction_type order. Without this, a soccer match's first
        row was asian_handicap (51 outcome keys) and the detail-page
        chart rendered that instead of the 1X2 probabilities.

        Raises ValueError if the match itself doesn't exist. An empty
        list (vs. ValueError) means the match exists but predictions
        haven't been precomputed yet — that's the "freshly fetched but
        not yet scored" window and the caller can render a pending UI.
        """

        # Verify the match exists before returning an empty list — lets
        # the caller distinguish "no predictions yet" from "bad ID".
        db = self._require_db()
        exists = db.execute(
            text("SELECT 1 FROM matches WHERE id = :match_id"),
            {"match_id": match_id},
        ).fetchone()
        if not exists:
            raise ValueError(f"Match {match_id} not found")

        query = text(
            """
            SELECT p.id, p.match_id, p.predicted_outcome, p.confidence,
                   p.probabilities, p.model_name, p.model_version,
                   p.prediction_type,
                   m.match_date, m.venue,
                   l.name as league_name, l.sport as sport,
                   ht.name as home_team, at.name as away_team
            FROM predictions p
            JOIN matches m ON p.match_id = m.id
            JOIN leagues l ON m.league_id = l.id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE p.match_id = :match_id
            ORDER BY p.prediction_type ASC
        """
        )

        results = list(db.execute(query, {"match_id": match_id}).fetchall())

        # Headline market first, everything else keeps the SQL
        # prediction_type order (Python's sort is stable). Sport comes
        # from the leagues join so detection doesn't depend on
        # model_name conventions (soccer pins model_name='ensemble').
        def _is_headline(row: Any) -> bool:
            headline = headline_task(row.sport)
            return headline is not None and row.prediction_type == headline.prediction_type

        results.sort(key=lambda row: 0 if _is_headline(row) else 1)

        # Map (ensemble_name, prediction_type) → friendly market name.
        # Keying on ensemble_name alone would work today but tomorrow
        # two ensembles could share a name across sports; keying on
        # prediction_type alone is ambiguous (soccer match_result vs
        # NHL regulation both serialize as 'match_result'). The pair
        # is unique per TaskSpec.
        ensemble_to_market = {(t.ensemble_name, t.prediction_type): t.market for t in TASKS.values()}

        predictions = []
        for row in results:
            market_label = ensemble_to_market.get(
                (row.model_name, row.prediction_type),
                row.prediction_type,
            )
            predictions.append(
                PredictionResponse(
                    match_info=MatchInfo(
                        match_id=str(row.match_id),
                        league_name=row.league_name,
                        home_team=row.home_team,
                        away_team=row.away_team,
                        match_date=row.match_date,
                        venue=row.venue,
                    ),
                    predicted_outcome=row.predicted_outcome,
                    probabilities=dict(row.probabilities or {}),
                    confidence=float(row.confidence),
                    model_version=row.model_version,
                    timestamp=datetime.utcnow(),
                    market=market_label,
                )
            )

        return predictions

    def get_live_predictions(self) -> List[PredictionResponse]:
        """Get predictions for live matches"""

        query = text(
            """
            SELECT p.id, p.match_id, p.predicted_outcome, p.confidence,
                   p.probabilities, p.model_version,
                   m.match_date, m.venue,
                   l.name as league_name,
                   ht.name as home_team, at.name as away_team
            FROM predictions p
            JOIN matches m ON p.match_id = m.id
            JOIN leagues l ON m.league_id = l.id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.status = 'live'
            ORDER BY m.match_date ASC
        """
        )

        db = self._require_db()
        results = db.execute(query).fetchall()

        predictions = []
        for row in results:
            predictions.append(
                PredictionResponse(
                    match_info=MatchInfo(
                        match_id=str(row.match_id),
                        league_name=row.league_name,
                        home_team=row.home_team,
                        away_team=row.away_team,
                        match_date=row.match_date,
                        venue=row.venue,
                    ),
                    predicted_outcome=row.predicted_outcome,
                    probabilities=row.probabilities or {},
                    confidence=float(row.confidence),
                    model_version=row.model_version,
                    timestamp=datetime.utcnow(),
                )
            )

        return predictions

    def _get_match_data(self, match_id: str) -> Optional[Dict]:
        """Get match information from database. Includes sport so
        predict_match can route to the right task ensemble."""

        query = text(
            """
            SELECT m.id, l.name as league_name, l.sport as sport,
                   ht.name as home_team, at.name as away_team,
                   m.match_date, m.venue
            FROM matches m
            JOIN leagues l ON m.league_id = l.id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE m.id = :match_id
        """
        )

        db = self._require_db()
        result = db.execute(query, {"match_id": match_id}).fetchone()

        if result:
            return {
                "match_id": str(result.id),
                "league_name": result.league_name,
                "sport": result.sport,
                "home_team": result.home_team,
                "away_team": result.away_team,
                "match_date": result.match_date,
                "venue": result.venue,
            }
        return None

    def _get_match_features(
        self,
        match_id: str,
        feature_set: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get computed features for a match from cache. Without
        feature_set, returns the most-recent fresh row regardless of
        which feature set wrote it — fine for soccer where there's
        only one set ('baseline') but ambiguous for NHL once both
        soccer and NHL feature sets coexist. Passing feature_set pins
        the lookup."""

        if feature_set is None:
            query = text(
                """
                SELECT features FROM features_cache
                WHERE match_id = :match_id
                AND expires_at > NOW()
                ORDER BY computed_at DESC LIMIT 1
            """
            )
            params = {"match_id": match_id}
        else:
            query = text(
                """
                SELECT features FROM features_cache
                WHERE match_id = :match_id
                AND feature_set = :feature_set
                AND expires_at > NOW()
                ORDER BY computed_at DESC LIMIT 1
            """
            )
            params = {"match_id": match_id, "feature_set": feature_set}

        db = self._require_db()
        result = db.execute(query, params).fetchone()
        if result:
            return result.features
        return {}

    def _get_stored_prediction(self, match_id: str) -> Optional[Dict]:
        """Get existing prediction from database"""

        query = text(
            """
            SELECT predicted_outcome, confidence, probabilities
            FROM predictions
            WHERE match_id = :match_id
            ORDER BY created_at DESC LIMIT 1
        """
        )

        db = self._require_db()
        result = db.execute(query, {"match_id": match_id}).fetchone()
        if result:
            return {
                "predicted_label": result.predicted_outcome,
                "confidence": float(result.confidence),
                "probabilities": result.probabilities or {},
            }
        return None

    def _store_prediction(
        self,
        match_id: str,
        prediction: Dict,
        task: Optional[TaskSpec] = None,
    ):
        """Store prediction in database. With `task`, uses the task's
        ensemble_name + prediction_type so soccer and NHL rows coexist
        without colliding on the (match_id, model_name, model_version,
        prediction_type) unique constraint. Without `task`, falls back
        to the legacy soccer match_result values for backwards compat."""

        query = text(
            """
            INSERT INTO predictions
            (match_id, model_name, model_version, prediction_type,
             predicted_outcome, confidence, probabilities)
            VALUES (:match_id, :model_name, :model_version, :prediction_type,
                    :predicted_outcome, :confidence, CAST(:probabilities AS jsonb))
            ON CONFLICT (match_id, model_name, model_version, prediction_type)
            DO UPDATE SET
                predicted_outcome = EXCLUDED.predicted_outcome,
                confidence = EXCLUDED.confidence,
                probabilities = EXCLUDED.probabilities,
                updated_at = NOW()
        """
        )

        if task is not None:
            task_key = f"{task.sport}:{task.market}"
            # Use predictions_model_name (not ensemble_name) so soccer
            # keeps writing "ensemble" to the DB column despite the
            # on-disk ensemble dir being "ensemble_soccer_match_result".
            # Decouples the artifact identifier from the persistent DB
            # identifier — recommendations SQL + analytics don't need
            # a parallel rename.
            model_name = task.predictions_model_name
            model_version = _MODEL_VERSIONS.get(task_key, "v1.0")
            prediction_type = task.prediction_type
        else:
            model_name = "ensemble"
            model_version = "v1.0"
            prediction_type = "match_result"

        try:
            import json

            db = self._require_db()
            db.execute(
                query,
                {
                    "match_id": match_id,
                    "model_name": model_name,
                    "model_version": model_version,
                    "prediction_type": prediction_type,
                    "predicted_outcome": prediction["predicted_label"],
                    "confidence": prediction["confidence"],
                    "probabilities": json.dumps(prediction["probabilities"]),
                },
            )
            db.commit()
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
            if self.db is not None:
                self.db.rollback()

    def _maybe_enqueue_alert(
        self,
        prediction: Dict[str, Any],
        task: TaskSpec,
        match_data: Dict[str, Any],
    ) -> None:
        """Queue a high-confidence prediction for the next Telegram
        digest. No-op (with a warning, not an exception) on any failure:
        a broken alert path must never break the predict_match response.

        Dedup: a Redis SETNX-with-TTL guards against duplicates across
        the day. Without it, every API call that re-predicts the same
        (match, market) would re-enqueue, producing a digest full of
        duplicates. The dedup key includes model_version so a fresh
        deploy with re-tuned weights gets a fresh chance to alert.
        """
        try:
            confidence = float(prediction.get("confidence", 0.0))
            if task.sport == "nhl":
                threshold = _NHL_NOTIFY_THRESHOLDS.get(task.market, 0.70)
            elif task.sport == "nba":
                threshold = _NBA_NOTIFY_THRESHOLDS.get(task.market, 0.65)
            elif task.sport == "nfl":
                threshold = _NFL_NOTIFY_THRESHOLDS.get(task.market, 0.65)
            elif task.sport == "tennis":
                threshold = _TENNIS_NOTIFY_THRESHOLDS.get(task.market, 0.65)
            elif task.sport == "mma":
                threshold = _MMA_NOTIFY_THRESHOLDS.get(task.market, 0.62)
            else:
                threshold = _SOCCER_NOTIFY_THRESHOLD
            if confidence < threshold:
                return

            from telegram_notify import Alert, enqueue_alerts  # type: ignore
        except Exception as e:
            # Helper not importable (test env / cold container / missing
            # /app/scripts mount). Don't surface as an error since it
            # doesn't affect the actual prediction; just log once.
            logger.debug("telegram_notify unavailable — alert hook skipped: %s", e)
            return

        # Dedup via Redis SETNX. Falls back to "always enqueue" if Redis
        # is unreachable so we don't silently drop alerts when dedup
        # state is unavailable.
        try:
            from services.cache_service import CacheService

            cache = CacheService()
            redis = getattr(cache, "redis", None)
            model_version = _MODEL_VERSIONS.get(f"{task.sport}:{task.market}", "v1.0")
            dedup_key = (
                f"auspex:alert_dedup:"
                f"{datetime.utcnow().date().isoformat()}:"
                f"{match_data['match_id']}:{task.sport}:{task.market}:{model_version}"
            )
            if redis is not None:
                # nx=True only sets if absent; True if we won the race
                # to enqueue, False if a previous call already did.
                if not redis.set(dedup_key, "1", nx=True, ex=86400):
                    return
        except Exception as e:
            logger.debug("Alert dedup check skipped (%s) — enqueueing anyway", e)

        alert = Alert(
            sport=task.sport,
            league_name=match_data.get("league_name", ""),
            home_team=match_data.get("home_team", ""),
            away_team=match_data.get("away_team", ""),
            match_date=match_data["match_date"],
            market_label=_display_label(task.sport, task.market),
            predicted_outcome=prediction["predicted_label"],
            confidence=confidence,
            probabilities={k: float(v) for k, v in (prediction.get("probabilities") or {}).items()},
        )
        try:
            enqueue_alerts([alert])
            logger.info(
                "Enqueued %s:%s alert for match %s (confidence=%.2f)",
                task.sport,
                task.market,
                match_data["match_id"],
                confidence,
            )
        except Exception as e:
            logger.warning("Failed to enqueue alert for %s:%s: %s", task.sport, task.market, e)
