"""Model configuration system."""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List


class ModelType(Enum):
    """Model types."""

    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NEURAL_NETWORK = "neural_network"
    POISSON = "poisson"
    DIXON_COLES = "dixon_coles"
    ENSEMBLE = "ensemble"


class PredictionTask(Enum):
    """Prediction tasks."""

    MATCH_OUTCOME = "match_outcome"  # 3-class: home/draw/away (soccer)
    OVER_UNDER = "over_under"  # Binary: over/under 2.5 (soccer)
    BTTS = "btts"  # Binary: both teams score (soccer)
    CORRECT_SCORE = "correct_score"  # Multi-class (soccer)
    ASIAN_HANDICAP = "asian_handicap"  # Multi-class (soccer)
    # NHL tasks. All modeled as 2-class softmax (num_class=2, NOT binary
    # objective) so they pass 2D probability arrays through the ensemble
    # — the ensemble's optimizer indexes proba[:, y] which requires 2D.
    NHL_MONEYLINE = "nhl_moneyline"  # 2-class: home/away game winner incl. OT/SO
    NHL_REGULATION = "nhl_regulation"  # 3-class: home reg / tie / away reg
    NHL_PUCK_LINE = "nhl_puck_line"  # 2-class: home covers -1.5 / not
    NHL_TOTAL = "nhl_total"  # 2-class: over 5.5 / under 5.5
    # NBA tasks. Spread + total are LINE-AS-FEATURE — one trained
    # model handles every line the book offers (no fixed-line filter).
    # All three are 2-class softmax for ensemble compatibility.
    NBA_MONEYLINE = "nba_moneyline"  # 2-class: home/away winner incl. OT
    NBA_SPREAD = "nba_spread"  # 2-class: home covers closing line / not
    NBA_TOTAL = "nba_total"  # 2-class: over closing line / under
    # NFL tasks. Same line-as-feature design as NBA — spread + total
    # consume the actual closing line as a numeric input. NFL ties are
    # excluded from training so moneyline stays cleanly 2-class.
    NFL_MONEYLINE = "nfl_moneyline"  # 2-class: home/away (non-tie games)
    NFL_SPREAD = "nfl_spread"  # 2-class: home covers closing line / not
    NFL_TOTAL = "nfl_total"  # 2-class: over closing line / under
    # Tennis tasks. First 1v1 sport — no draw possible (matches always
    # produce a winner), so moneyline is cleanly 2-class without a
    # tie-exclusion filter. Total games + set-betting are v2 (need
    # linescore parsing).
    TENNIS_MONEYLINE = "tennis_moneyline"  # 2-class: player1 / player2
    # MMA tasks. 1v1 fight winner. Draws (~1% of MMA decisions) are
    # dropped at ingest so the training corpus is cleanly 2-class.
    # Method-of-victory / round-group are v2 (separate models).
    MMA_MONEYLINE = "mma_moneyline"  # 2-class: fighter1 / fighter2


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    model_type: ModelType
    prediction_task: PredictionTask
    version: str
    hyperparameters: Dict[str, Any]
    features: List[str]
    target_column: str
    loss_function: str
    metrics: List[str]
    training_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "prediction_task": self.prediction_task.value,
            "version": self.version,
            "hyperparameters": self.hyperparameters,
            "features": self.features,
            "target_column": self.target_column,
            "loss_function": self.loss_function,
            "metrics": self.metrics,
            "training_config": self.training_config,
        }


# ============= PREDEFINED MODEL CONFIGURATIONS =============

XGBOOST_MATCH_OUTCOME = ModelConfig(
    name="xgboost_match_outcome",
    model_type=ModelType.XGBOOST,
    prediction_task=PredictionTask.MATCH_OUTCOME,
    version="1.0.0",
    hyperparameters={
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
    },
    features=[],
    target_column="match_outcome",
    loss_function="multi:softprob",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={
        "early_stopping_rounds": 50,
        "eval_metric": "mlogloss",
        "verbose": 100,
    },
)

LIGHTGBM_OVER_UNDER = ModelConfig(
    name="lightgbm_over_under",
    model_type=ModelType.LIGHTGBM,
    prediction_task=PredictionTask.OVER_UNDER,
    version="1.0.0",
    hyperparameters={
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    features=[],
    target_column="over_under_2_5",
    loss_function="binary",
    metrics=["accuracy", "log_loss", "roc_auc"],
    training_config={"early_stopping_rounds": 50, "verbose": 100},
)

LIGHTGBM_MATCH_OUTCOME = ModelConfig(
    name="lightgbm_match_outcome",
    model_type=ModelType.LIGHTGBM,
    prediction_task=PredictionTask.MATCH_OUTCOME,
    version="1.0.0",
    hyperparameters={
        "objective": "multiclass",
        "num_class": 3,
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    features=[],
    target_column="match_outcome",
    loss_function="multiclass",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "verbose": 100},
)

NEURAL_NETWORK_CONFIG = ModelConfig(
    name="neural_network_match_outcome",
    model_type=ModelType.NEURAL_NETWORK,
    prediction_task=PredictionTask.MATCH_OUTCOME,
    version="1.0.0",
    hyperparameters={
        "hidden_layers": [256, 128, 64, 32],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100,
        "optimizer": "adam",
        "activation": "relu",
        "output_activation": "softmax",
        "batch_norm": True,
    },
    features=[],
    target_column="match_outcome",
    loss_function="categorical_crossentropy",
    metrics=["accuracy", "log_loss"],
    training_config={
        "early_stopping_patience": 15,
        "reduce_lr_patience": 5,
        "verbose": 1,
    },
)

POISSON_CONFIG = ModelConfig(
    name="poisson_goals",
    model_type=ModelType.POISSON,
    prediction_task=PredictionTask.MATCH_OUTCOME,
    version="1.0.0",
    hyperparameters={
        "max_goals": 6,
        "home_advantage": 0.25,
        "regularization": 0.001,
        "max_iterations": 1000,
        "convergence_threshold": 1e-6,
    },
    features=[],
    target_column="match_outcome",
    loss_function="poisson_nll",
    metrics=["accuracy", "log_loss", "ranked_probability_score"],
    training_config={},
)

# The SOCCER FULL-TIME Dixon-Coles config, and the only one the 2026-09 refit
# was measured on. Anything else that wants a Dixon-Coles takes the pinned copy
# below (dixon_coles_config_pinned_to_legacy_fit) rather than this object.
#
# NOTE: there is deliberately no "home_advantage" key here, so
# PoissonMatchPredictor.__init__ falls back to its hard-coded 0.25. With
# per_league_baselines on, the fitted home/away baselines make that a
# fallback-only value for soccer. It stays LIVE in the pinned copy — which is
# how a SOCCER constant still reaches the NHL Dixon-Coles against
# league_avg_goals ~3.10. That is the deferred NHL defect, tracked separately.
DIXON_COLES_CONFIG = ModelConfig(
    name="dixon_coles",
    model_type=ModelType.DIXON_COLES,
    prediction_task=PredictionTask.MATCH_OUTCOME,
    version="1.0.0",
    hyperparameters={
        # 6 truncated real scoreline mass (market_derivation.MAX_GOALS_DERIVE
        # has been 10 for a while; this only ever bit the model's own 1x2).
        "max_goals": 10,
        "rho_init": -0.13,
        # Per-day exponential recency weight inside the MLE (730-day
        # half-life). NOT the closed GBM recency/horizon lever — see
        # PoissonMatchPredictor._time_weights.
        "time_decay": 0.00095,
        # Shrinkage prior toward strength 1.0, in effective matches. The
        # old 0.001 was inert against weighted counts of order 30-500;
        # ~20 makes a thin-history team fall back to its LEAGUE baseline
        # and is what gets the fit converging (~91 iterations vs never).
        "regularization": 20.0,
        # Fit home/away baselines per league instead of forcing one
        # global level across 41 competitions whose goals-per-match run
        # 2.222 (Primera Division AR) to 3.094 (Eredivisie).
        "per_league_baselines": True,
        # Effective matches of shrinkage toward the global weighted means,
        # so thin (Copa Libertadores n=16) and brand-new leagues stay safe.
        "league_shrinkage": 200.0,
        "max_iterations": 500,
        "convergence_threshold": 1e-5,
    },
    features=[],
    target_column="match_outcome",
    loss_function="dixon_coles_nll",
    metrics=["accuracy", "log_loss", "ranked_probability_score"],
    training_config={},
)

# ── Dixon-Coles fits that are NOT the validated soccer full-time one ──────────
#
# DIXON_COLES_CONFIG is SHARED. Three scripts build a Dixon-Coles straight from
# it and write the artifact themselves:
#     scripts/train_hockey_dixon_coles.py       -> dixon_coles_nhl
#     scripts/train_halftime_dixon_coles.py     -> dixon_coles_ht_soccer
#     scripts/train_second_half_dixon_coles.py  -> dixon_coles_h2_soccer
# None of the three goes through training/train_all_models.py, so neither the
# 1x2 promote gate nor the derived-market guard ever sees them — whatever they
# fit is what serves.
#
# The four refit knobs above (max_goals 10, a LIVE time_decay, regularization as
# a 20-effective-match prior, per_league_baselines) were measured on exactly one
# object: the soccer FULL-TIME over-2.5 walk-forward. Nothing about hockey goals
# or halftime goals was measured anywhere, and the tuned values do not obviously
# transfer — a 20-effective-match prior is a materially stronger shrink against
# halftime's ~0.6 goals/match than against full-time's ~2.65. So those three
# trainers are pinned back to the pre-refit values until each is separately
# measured and gated.
#
# What this deliberately does NOT do: fix the deferred NHL defect. That artifact
# carries max_goals=6 and home_advantage=0.25 against league_avg_goals ~3.10,
# truncating ~3.4% of per-side mass and feeding 8 live NHL markets. It is real,
# it is tracked separately, and raising max_goals here would ship an unmeasured
# NHL refit under cover of a soccer change.
#
# What it CANNOT pin: the scale-gauge correction in PoissonMatchPredictor.train
# (normalise attack only). That is a correctness fix to a shared fitting routine,
# not a tuning knob, so every Poisson-family model picks it up on its next
# retrain — see that method's docstring for the full blast radius.
LEGACY_DIXON_COLES_HYPERPARAMETERS: Dict[str, Any] = {
    "max_goals": 6,
    "time_decay": 0.0,
    "regularization": 0.001,
    "per_league_baselines": False,
}


def dixon_coles_config_pinned_to_legacy_fit(name: str = "dixon_coles") -> ModelConfig:
    """DIXON_COLES_CONFIG with the 2026-09 soccer refit knobs pinned OFF.

    Use this from any trainer whose artifact was not measured by
    scripts/ab_soccer_dixon_coles_refit.py. Passing a distinct ``name``
    keeps the artifact's embedded config self-describing.
    """
    return replace(
        DIXON_COLES_CONFIG,
        name=name,
        hyperparameters={**DIXON_COLES_CONFIG.hyperparameters, **LEGACY_DIXON_COLES_HYPERPARAMETERS},
    )


ENSEMBLE_CONFIG = ModelConfig(
    name="ensemble_match_outcome",
    model_type=ModelType.ENSEMBLE,
    prediction_task=PredictionTask.MATCH_OUTCOME,
    version="1.0.0",
    hyperparameters={
        "combination_method": "weighted_average",
        "optimize_weights": True,
        "weight_optimization_metric": "log_loss",
        "min_weight": 0.05,
    },
    features=[],
    target_column="match_outcome",
    loss_function="ensemble",
    metrics=["accuracy", "log_loss", "brier_score", "roi"],
    training_config={},
)

MODEL_CONFIGS = {
    "match_outcome_xgboost": XGBOOST_MATCH_OUTCOME,
    "match_outcome_lightgbm": LIGHTGBM_MATCH_OUTCOME,
    "over_under_lightgbm": LIGHTGBM_OVER_UNDER,
    "match_outcome_nn": NEURAL_NETWORK_CONFIG,
    "poisson_goals": POISSON_CONFIG,
    "dixon_coles": DIXON_COLES_CONFIG,
    "ensemble_match_outcome": ENSEMBLE_CONFIG,
}


# ============= NHL MODEL CONFIGURATIONS =============
#
# Hyperparameters intentionally close to the soccer baselines so the
# first NHL training run inherits proven defaults; tune per-task once
# we have holdout numbers. Two structural differences from soccer:
#   * num_class=2 (not 3) for moneyline / puck-line / totals.
#   * objective="multi:softprob" / "multiclass" (NOT "binary") so the
#     model emits 2D (N, 2) probability arrays — the EnsemblePredictor
#     optimizer indexes proba[:, y], which requires 2D input.

XGBOOST_NHL_MONEYLINE = ModelConfig(
    name="xgboost_nhl_moneyline",
    model_type=ModelType.XGBOOST,
    prediction_task=PredictionTask.NHL_MONEYLINE,
    version="1.0.0",
    hyperparameters={
        "objective": "multi:softprob",
        "num_class": 2,
        "max_depth": 6,  # shallower than soccer — less feature interaction
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
    },
    features=[],
    target_column="nhl_moneyline",
    loss_function="multi:softprob",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={
        "early_stopping_rounds": 50,
        "eval_metric": "mlogloss",
        "verbose": 100,
    },
)

LIGHTGBM_NHL_MONEYLINE = ModelConfig(
    name="lightgbm_nhl_moneyline",
    model_type=ModelType.LIGHTGBM,
    prediction_task=PredictionTask.NHL_MONEYLINE,
    version="1.0.0",
    hyperparameters={
        "objective": "multiclass",
        "num_class": 2,
        "boosting_type": "gbdt",
        "num_leaves": 48,  # smaller than soccer's 64
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 25,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    features=[],
    target_column="nhl_moneyline",
    loss_function="multiclass",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "verbose": 100},
)

NEURAL_NETWORK_NHL_MONEYLINE = ModelConfig(
    name="neural_network_nhl_moneyline",
    model_type=ModelType.NEURAL_NETWORK,
    prediction_task=PredictionTask.NHL_MONEYLINE,
    version="1.0.0",
    hyperparameters={
        "hidden_layers": [128, 64, 32],  # smaller than soccer — fewer features
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100,
        "optimizer": "adam",
        "activation": "relu",
        "output_activation": "softmax",
        "batch_norm": True,
    },
    features=[],
    target_column="nhl_moneyline",
    loss_function="categorical_crossentropy",
    metrics=["accuracy", "log_loss"],
    training_config={
        "early_stopping_patience": 15,
        "reduce_lr_patience": 5,
        "verbose": 1,
    },
)

ENSEMBLE_NHL_MONEYLINE = ModelConfig(
    name="ensemble_nhl_moneyline",
    model_type=ModelType.ENSEMBLE,
    prediction_task=PredictionTask.NHL_MONEYLINE,
    version="1.0.0",
    hyperparameters={
        "combination_method": "weighted_average",
        "optimize_weights": True,
        "weight_optimization_metric": "log_loss",
        "min_weight": 0.05,
    },
    features=[],
    target_column="nhl_moneyline",
    loss_function="ensemble",
    metrics=["accuracy", "log_loss", "brier_score", "roi"],
    training_config={},
)

NHL_MONEYLINE_CONFIGS = {
    "xgboost_nhl_ml": XGBOOST_NHL_MONEYLINE,
    "lightgbm_nhl_ml": LIGHTGBM_NHL_MONEYLINE,
    "neural_network_nhl_ml": NEURAL_NETWORK_NHL_MONEYLINE,
    "ensemble_nhl_ml": ENSEMBLE_NHL_MONEYLINE,
}


# ── NHL REGULATION 3-WAY (60-minute outcome) ─────────────────────────
#
# Three classes: 0=home reg win, 1=regulation tie (game went to OT/SO),
# 2=away reg win. Target column derived from matches.metadata->>'regulation_winner'
# (string 'home' | 'tie' | 'away'), mapped through a CASE in the query so the
# frame already has the integer class label. NHL regulation distribution from
# recent seasons:
#   * home reg win  ~42%
#   * tie (OT/SO)   ~22%
#   * away reg win  ~36%
# Class imbalance is mild — softmax handles it without explicit reweighting.
#
# Hyperparameters mirror the moneyline configs except num_class=3 and (for NN)
# the slightly wider hidden layers — 3-class softmax can usually exploit a
# bit more capacity than 2-class.

XGBOOST_NHL_REGULATION = ModelConfig(
    name="xgboost_nhl_regulation",
    model_type=ModelType.XGBOOST,
    prediction_task=PredictionTask.NHL_REGULATION,
    version="1.0.0",
    hyperparameters={
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
    },
    features=[],
    target_column="nhl_regulation",
    loss_function="multi:softprob",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={
        "early_stopping_rounds": 50,
        "eval_metric": "mlogloss",
        "verbose": 100,
    },
)

LIGHTGBM_NHL_REGULATION = ModelConfig(
    name="lightgbm_nhl_regulation",
    model_type=ModelType.LIGHTGBM,
    prediction_task=PredictionTask.NHL_REGULATION,
    version="1.0.0",
    hyperparameters={
        "objective": "multiclass",
        "num_class": 3,
        "boosting_type": "gbdt",
        "num_leaves": 48,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 25,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    features=[],
    target_column="nhl_regulation",
    loss_function="multiclass",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "verbose": 100},
)

NEURAL_NETWORK_NHL_REGULATION = ModelConfig(
    name="neural_network_nhl_regulation",
    model_type=ModelType.NEURAL_NETWORK,
    prediction_task=PredictionTask.NHL_REGULATION,
    version="1.0.0",
    hyperparameters={
        "hidden_layers": [192, 96, 48],  # slightly wider than moneyline — 3-class room
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100,
        "optimizer": "adam",
        "activation": "relu",
        "output_activation": "softmax",
        "batch_norm": True,
    },
    features=[],
    target_column="nhl_regulation",
    loss_function="categorical_crossentropy",
    metrics=["accuracy", "log_loss"],
    training_config={
        "early_stopping_patience": 15,
        "reduce_lr_patience": 5,
        "verbose": 1,
    },
)

ENSEMBLE_NHL_REGULATION = ModelConfig(
    name="ensemble_nhl_regulation",
    model_type=ModelType.ENSEMBLE,
    prediction_task=PredictionTask.NHL_REGULATION,
    version="1.0.0",
    hyperparameters={
        "combination_method": "weighted_average",
        "optimize_weights": True,
        "weight_optimization_metric": "log_loss",
        "min_weight": 0.05,
    },
    features=[],
    target_column="nhl_regulation",
    loss_function="ensemble",
    metrics=["accuracy", "log_loss", "brier_score", "roi"],
    training_config={},
)

NHL_REGULATION_CONFIGS = {
    "xgboost_nhl_reg": XGBOOST_NHL_REGULATION,
    "lightgbm_nhl_reg": LIGHTGBM_NHL_REGULATION,
    "neural_network_nhl_reg": NEURAL_NETWORK_NHL_REGULATION,
    "ensemble_nhl_reg": ENSEMBLE_NHL_REGULATION,
}


# ── NHL PUCK LINE (home covers -1.5) ─────────────────────────────────
#
# 2-class binary classifier framed as 2-class softmax (same reasoning
# as NHL_MONEYLINE — ensemble needs 2D proba). Target derived from
# final score margin INCLUDING OT/SO goals (puck-line bets settle on
# the full game's final score, not regulation alone):
#   * 0 = home covers -1.5 (home_score - away_score >= 2)
#   * 1 = home does NOT cover (home wins by 1 OR any away win)
# Class distribution is roughly 50/50 because NHL margins cluster
# tightly around 1-2 goals. Hyperparameters cloned from moneyline.

XGBOOST_NHL_PUCK_LINE = ModelConfig(
    name="xgboost_nhl_puck_line",
    model_type=ModelType.XGBOOST,
    prediction_task=PredictionTask.NHL_PUCK_LINE,
    version="1.0.0",
    hyperparameters={
        "objective": "multi:softprob",
        "num_class": 2,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
    },
    features=[],
    target_column="nhl_puck_line",
    loss_function="multi:softprob",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "eval_metric": "mlogloss", "verbose": 100},
)

LIGHTGBM_NHL_PUCK_LINE = ModelConfig(
    name="lightgbm_nhl_puck_line",
    model_type=ModelType.LIGHTGBM,
    prediction_task=PredictionTask.NHL_PUCK_LINE,
    version="1.0.0",
    hyperparameters={
        "objective": "multiclass",
        "num_class": 2,
        "boosting_type": "gbdt",
        "num_leaves": 48,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 25,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    features=[],
    target_column="nhl_puck_line",
    loss_function="multiclass",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "verbose": 100},
)

NEURAL_NETWORK_NHL_PUCK_LINE = ModelConfig(
    name="neural_network_nhl_puck_line",
    model_type=ModelType.NEURAL_NETWORK,
    prediction_task=PredictionTask.NHL_PUCK_LINE,
    version="1.0.0",
    hyperparameters={
        "hidden_layers": [128, 64, 32],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100,
        "optimizer": "adam",
        "activation": "relu",
        "output_activation": "softmax",
        "batch_norm": True,
    },
    features=[],
    target_column="nhl_puck_line",
    loss_function="categorical_crossentropy",
    metrics=["accuracy", "log_loss"],
    training_config={"early_stopping_patience": 15, "reduce_lr_patience": 5, "verbose": 1},
)

ENSEMBLE_NHL_PUCK_LINE = ModelConfig(
    name="ensemble_nhl_puck_line",
    model_type=ModelType.ENSEMBLE,
    prediction_task=PredictionTask.NHL_PUCK_LINE,
    version="1.0.0",
    hyperparameters={
        "combination_method": "weighted_average",
        "optimize_weights": True,
        "weight_optimization_metric": "log_loss",
        "min_weight": 0.05,
    },
    features=[],
    target_column="nhl_puck_line",
    loss_function="ensemble",
    metrics=["accuracy", "log_loss", "brier_score", "roi"],
    training_config={},
)

NHL_PUCK_LINE_CONFIGS = {
    "xgboost_nhl_pl": XGBOOST_NHL_PUCK_LINE,
    "lightgbm_nhl_pl": LIGHTGBM_NHL_PUCK_LINE,
    "neural_network_nhl_pl": NEURAL_NETWORK_NHL_PUCK_LINE,
    "ensemble_nhl_pl": ENSEMBLE_NHL_PUCK_LINE,
}


# ── NHL TOTAL (over 5.5) ─────────────────────────────────────────────
#
# 2-class binary classifier for the canonical 5.5-goal NHL total. As
# with puck line, the bet settles on the full game total INCLUDING
# OT/SO goals (NHL convention: shootout winner is credited with +1
# goal toward the total). Target:
#   * 0 = over  5.5  (home_score + away_score >= 6)
#   * 1 = under 5.5  (home_score + away_score <= 5)
# League average NHL total is ~6.0, so the line is moderately tilted
# toward over but the class split is close enough to balanced that we
# don't need explicit class weighting.

XGBOOST_NHL_TOTAL = ModelConfig(
    name="xgboost_nhl_total",
    model_type=ModelType.XGBOOST,
    prediction_task=PredictionTask.NHL_TOTAL,
    version="1.0.0",
    hyperparameters={
        "objective": "multi:softprob",
        "num_class": 2,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
    },
    features=[],
    target_column="nhl_total",
    loss_function="multi:softprob",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "eval_metric": "mlogloss", "verbose": 100},
)

LIGHTGBM_NHL_TOTAL = ModelConfig(
    name="lightgbm_nhl_total",
    model_type=ModelType.LIGHTGBM,
    prediction_task=PredictionTask.NHL_TOTAL,
    version="1.0.0",
    hyperparameters={
        "objective": "multiclass",
        "num_class": 2,
        "boosting_type": "gbdt",
        "num_leaves": 48,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 25,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    features=[],
    target_column="nhl_total",
    loss_function="multiclass",
    metrics=["accuracy", "log_loss", "brier_score"],
    training_config={"early_stopping_rounds": 50, "verbose": 100},
)

NEURAL_NETWORK_NHL_TOTAL = ModelConfig(
    name="neural_network_nhl_total",
    model_type=ModelType.NEURAL_NETWORK,
    prediction_task=PredictionTask.NHL_TOTAL,
    version="1.0.0",
    hyperparameters={
        "hidden_layers": [128, 64, 32],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 100,
        "optimizer": "adam",
        "activation": "relu",
        "output_activation": "softmax",
        "batch_norm": True,
    },
    features=[],
    target_column="nhl_total",
    loss_function="categorical_crossentropy",
    metrics=["accuracy", "log_loss"],
    training_config={"early_stopping_patience": 15, "reduce_lr_patience": 5, "verbose": 1},
)

ENSEMBLE_NHL_TOTAL = ModelConfig(
    name="ensemble_nhl_total",
    model_type=ModelType.ENSEMBLE,
    prediction_task=PredictionTask.NHL_TOTAL,
    version="1.0.0",
    hyperparameters={
        "combination_method": "weighted_average",
        "optimize_weights": True,
        "weight_optimization_metric": "log_loss",
        "min_weight": 0.05,
    },
    features=[],
    target_column="nhl_total",
    loss_function="ensemble",
    metrics=["accuracy", "log_loss", "brier_score", "roi"],
    training_config={},
)

NHL_TOTAL_CONFIGS = {
    "xgboost_nhl_tot": XGBOOST_NHL_TOTAL,
    "lightgbm_nhl_tot": LIGHTGBM_NHL_TOTAL,
    "neural_network_nhl_tot": NEURAL_NETWORK_NHL_TOTAL,
    "ensemble_nhl_tot": ENSEMBLE_NHL_TOTAL,
}


# ── Hockey-Poisson configs (Phase 3d) ────────────────────────────────
#
# One HockeyPoissonPredictor fits per task; each emits a task-specific
# probability array derived from the same joint goal distribution.
# Hyperparameters are shared across all four task variants because the
# underlying MLE is the same — only the predict_proba derivation differs.
#
# Hockey home_advantage 0.15 (NHL home win rate ~55%) vs soccer's 0.25
# (soccer home win rate ~45-50%). max_goals=10 covers NHL's long tail.

_HOCKEY_POISSON_HYPERPARAMS = {
    "max_goals": 10,
    "home_advantage": 0.15,
    "regularization": 0.001,
    "max_iterations": 1000,
    "convergence_threshold": 1e-6,
}


def _hockey_poisson_config(name: str, task: PredictionTask, target: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        model_type=ModelType.POISSON,
        prediction_task=task,
        version="1.0.0",
        hyperparameters=dict(_HOCKEY_POISSON_HYPERPARAMS),
        features=[],
        target_column=target,
        loss_function="poisson_nll",
        metrics=["accuracy", "log_loss", "brier_score"],
        training_config={},
    )


HOCKEY_POISSON_NHL_MONEYLINE = _hockey_poisson_config(
    "hockey_poisson_nhl_ml", PredictionTask.NHL_MONEYLINE, "nhl_moneyline"
)
HOCKEY_POISSON_NHL_REGULATION = _hockey_poisson_config(
    "hockey_poisson_nhl_reg", PredictionTask.NHL_REGULATION, "nhl_regulation"
)
HOCKEY_POISSON_NHL_PUCK_LINE = _hockey_poisson_config(
    "hockey_poisson_nhl_pl", PredictionTask.NHL_PUCK_LINE, "nhl_puck_line"
)
HOCKEY_POISSON_NHL_TOTAL = _hockey_poisson_config("hockey_poisson_nhl_tot", PredictionTask.NHL_TOTAL, "nhl_total")


# ─────────────────────── NBA MODEL CONFIGS ───────────────────────────
#
# Three markets (moneyline, spread, total), three base models per
# market (XGBoost, LightGBM, Neural Network) + 1 ensemble = 12
# configs total. No Poisson/Dixon-Coles — basketball scoring is too
# continuous and high-variance for a discrete-distribution prior to
# add signal over GBDT.
#
# Hyperparameters mirror the NHL tuning: shallower trees than soccer
# (less feature interaction), 400 estimators, hist-method XGBoost,
# 2-class softmax for ensemble compatibility. We may re-tune per
# market after watching real out-of-sample performance — for now a
# unified set keeps the matrix small.


def _nba_xgb_config(name: str, task: PredictionTask, target: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        model_type=ModelType.XGBOOST,
        prediction_task=task,
        version="1.0.0",
        hyperparameters={
            "objective": "multi:softprob",
            "num_class": 2,
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "random_state": 42,
        },
        features=[],
        target_column=target,
        loss_function="multi:softprob",
        metrics=["accuracy", "log_loss", "brier_score"],
        training_config={"early_stopping_rounds": 50, "eval_metric": "mlogloss", "verbose": 100},
    )


def _nba_lgb_config(name: str, task: PredictionTask, target: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        model_type=ModelType.LIGHTGBM,
        prediction_task=task,
        version="1.0.0",
        hyperparameters={
            "objective": "multiclass",
            "num_class": 2,
            "boosting_type": "gbdt",
            "num_leaves": 48,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_samples": 25,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "random_state": 42,
        },
        features=[],
        target_column=target,
        loss_function="multiclass",
        metrics=["accuracy", "log_loss", "brier_score"],
        training_config={"early_stopping_rounds": 50, "verbose": 100},
    )


def _nba_nn_config(name: str, task: PredictionTask, target: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        model_type=ModelType.NEURAL_NETWORK,
        prediction_task=task,
        version="1.0.0",
        hyperparameters={
            "hidden_layers": [128, 64, 32],
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 100,
            "optimizer": "adam",
            "activation": "relu",
            "output_activation": "softmax",
            "batch_norm": True,
        },
        features=[],
        target_column=target,
        loss_function="categorical_crossentropy",
        metrics=["accuracy", "log_loss"],
        training_config={"early_stopping_patience": 15, "reduce_lr_patience": 5, "verbose": 1},
    )


def _nba_ensemble_config(name: str, task: PredictionTask, target: str) -> ModelConfig:
    return ModelConfig(
        name=name,
        model_type=ModelType.ENSEMBLE,
        prediction_task=task,
        version="1.0.0",
        hyperparameters={
            "combination_method": "weighted_average",
            "optimize_weights": True,
            "weight_optimization_metric": "log_loss",
            "min_weight": 0.05,
        },
        features=[],
        target_column=target,
        loss_function="ensemble",
        metrics=["accuracy", "log_loss", "brier_score", "roi"],
        training_config={},
    )


# Moneyline
XGBOOST_NBA_MONEYLINE = _nba_xgb_config("xgboost_nba_ml", PredictionTask.NBA_MONEYLINE, "nba_moneyline")
LIGHTGBM_NBA_MONEYLINE = _nba_lgb_config("lightgbm_nba_ml", PredictionTask.NBA_MONEYLINE, "nba_moneyline")
NEURAL_NETWORK_NBA_MONEYLINE = _nba_nn_config("neural_network_nba_ml", PredictionTask.NBA_MONEYLINE, "nba_moneyline")
ENSEMBLE_NBA_MONEYLINE = _nba_ensemble_config("ensemble_nba_ml", PredictionTask.NBA_MONEYLINE, "nba_moneyline")

# Spread (line-as-feature)
XGBOOST_NBA_SPREAD = _nba_xgb_config("xgboost_nba_sp", PredictionTask.NBA_SPREAD, "nba_spread")
LIGHTGBM_NBA_SPREAD = _nba_lgb_config("lightgbm_nba_sp", PredictionTask.NBA_SPREAD, "nba_spread")
NEURAL_NETWORK_NBA_SPREAD = _nba_nn_config("neural_network_nba_sp", PredictionTask.NBA_SPREAD, "nba_spread")
ENSEMBLE_NBA_SPREAD = _nba_ensemble_config("ensemble_nba_sp", PredictionTask.NBA_SPREAD, "nba_spread")

# Total (line-as-feature)
XGBOOST_NBA_TOTAL = _nba_xgb_config("xgboost_nba_tot", PredictionTask.NBA_TOTAL, "nba_total")
LIGHTGBM_NBA_TOTAL = _nba_lgb_config("lightgbm_nba_tot", PredictionTask.NBA_TOTAL, "nba_total")
NEURAL_NETWORK_NBA_TOTAL = _nba_nn_config("neural_network_nba_tot", PredictionTask.NBA_TOTAL, "nba_total")
ENSEMBLE_NBA_TOTAL = _nba_ensemble_config("ensemble_nba_tot", PredictionTask.NBA_TOTAL, "nba_total")


# ─────────────────────── NFL MODEL CONFIGS ───────────────────────────
#
# Three markets (moneyline, spread, total), three base models per
# market (XGBoost, LightGBM, Neural Network) + 1 ensemble = 12
# configs total. Same shape as NBA — no Poisson/Dixon-Coles (NFL
# scoring is too high-variance + matchup-specific for a discrete
# prior to add signal over GBDT + NN).
#
# NFL has FEWER training rows than NBA (~855 across 3 seasons vs
# NBA's 4125), so we use SHALLOWER trees + smaller learning rate
# than NBA to avoid overfitting. n_estimators stays at 400 but
# max_depth drops to 5 (NBA uses 6) and num_leaves to 32 (NBA's
# 48) to keep model capacity proportional to sample size.

# NFL uses the same factory functions as NBA — _nba_xgb_config etc.
# are sport-agnostic at the moment; we just override max_depth /
# num_leaves per-bundle on top. Define small NFL-specific factory
# wrappers so the hyperparameter delta is explicit rather than
# scattered.


def _nfl_xgb_config(name: str, task: PredictionTask, target: str) -> ModelConfig:
    cfg = _nba_xgb_config(name, task, target)
    # Shallower trees on smaller corpus to reduce overfit risk.
    cfg.hyperparameters["max_depth"] = 5
    cfg.hyperparameters["min_child_weight"] = 8  # was 5 for NBA
    return cfg


def _nfl_lgb_config(name: str, task: PredictionTask, target: str) -> ModelConfig:
    cfg = _nba_lgb_config(name, task, target)
    cfg.hyperparameters["num_leaves"] = 32  # was 48 for NBA
    cfg.hyperparameters["min_child_samples"] = 40  # was 25 for NBA
    return cfg


def _nfl_nn_config(name: str, task: PredictionTask, target: str) -> ModelConfig:
    cfg = _nba_nn_config(name, task, target)
    # Smaller layers — fewer params for fewer rows.
    cfg.hyperparameters["hidden_layers"] = [64, 32, 16]
    cfg.hyperparameters["dropout_rate"] = 0.4  # was 0.3 for NBA
    return cfg


# Moneyline
XGBOOST_NFL_MONEYLINE = _nfl_xgb_config("xgboost_nfl_ml", PredictionTask.NFL_MONEYLINE, "nfl_moneyline")
LIGHTGBM_NFL_MONEYLINE = _nfl_lgb_config("lightgbm_nfl_ml", PredictionTask.NFL_MONEYLINE, "nfl_moneyline")
NEURAL_NETWORK_NFL_MONEYLINE = _nfl_nn_config("neural_network_nfl_ml", PredictionTask.NFL_MONEYLINE, "nfl_moneyline")
ENSEMBLE_NFL_MONEYLINE = _nba_ensemble_config("ensemble_nfl_ml", PredictionTask.NFL_MONEYLINE, "nfl_moneyline")

# Spread (line-as-feature)
XGBOOST_NFL_SPREAD = _nfl_xgb_config("xgboost_nfl_sp", PredictionTask.NFL_SPREAD, "nfl_spread")
LIGHTGBM_NFL_SPREAD = _nfl_lgb_config("lightgbm_nfl_sp", PredictionTask.NFL_SPREAD, "nfl_spread")
NEURAL_NETWORK_NFL_SPREAD = _nfl_nn_config("neural_network_nfl_sp", PredictionTask.NFL_SPREAD, "nfl_spread")
ENSEMBLE_NFL_SPREAD = _nba_ensemble_config("ensemble_nfl_sp", PredictionTask.NFL_SPREAD, "nfl_spread")

# Total (line-as-feature)
XGBOOST_NFL_TOTAL = _nfl_xgb_config("xgboost_nfl_tot", PredictionTask.NFL_TOTAL, "nfl_total")
LIGHTGBM_NFL_TOTAL = _nfl_lgb_config("lightgbm_nfl_tot", PredictionTask.NFL_TOTAL, "nfl_total")
NEURAL_NETWORK_NFL_TOTAL = _nfl_nn_config("neural_network_nfl_tot", PredictionTask.NFL_TOTAL, "nfl_total")
ENSEMBLE_NFL_TOTAL = _nba_ensemble_config("ensemble_nfl_tot", PredictionTask.NFL_TOTAL, "nfl_total")


# ─────────────────────── TENNIS MODEL CONFIGS ────────────────────────
#
# One market (moneyline), three base models (XGBoost, LightGBM, NN)
# + ensemble = 4 configs total. Tennis training corpus is ~12-15k
# matches across 3 seasons (ATP + WTA combined) — comparable to NBA
# and much larger than NFL, so we reuse the NBA hyperparameter shape
# without the NFL-style shallowing.
#
# No Poisson / Dixon-Coles — tennis scoring is set-level (binary
# game outcomes) with no Poisson assumption that helps.

XGBOOST_TENNIS_MONEYLINE = _nba_xgb_config("xgboost_tennis_ml", PredictionTask.TENNIS_MONEYLINE, "tennis_moneyline")
LIGHTGBM_TENNIS_MONEYLINE = _nba_lgb_config("lightgbm_tennis_ml", PredictionTask.TENNIS_MONEYLINE, "tennis_moneyline")
NEURAL_NETWORK_TENNIS_MONEYLINE = _nba_nn_config(
    "neural_network_tennis_ml", PredictionTask.TENNIS_MONEYLINE, "tennis_moneyline"
)
ENSEMBLE_TENNIS_MONEYLINE = _nba_ensemble_config(
    "ensemble_tennis_ml", PredictionTask.TENNIS_MONEYLINE, "tennis_moneyline"
)


# ─────────────────────── MMA MODEL CONFIGS ───────────────────────────
#
# Same shape as tennis (single moneyline market, XGB+LGB+NN+Ensemble).
# UFC corpus is ~1500 fights/3 yrs — between NFL (855) and NBA (4125),
# so use NBA hyperparameters with no NFL-style shallowing.
#
# NOTE: tennis NN trains to all-zeros for some reason (see
# tennis-pipeline-state memory). MMA may hit the same issue given the
# smaller corpus; if so, production training should run with
# --skip-models neural_network. Locked here for v1; revisit when we
# debug the NN failure.

XGBOOST_MMA_MONEYLINE = _nba_xgb_config("xgboost_mma_ml", PredictionTask.MMA_MONEYLINE, "mma_moneyline")
LIGHTGBM_MMA_MONEYLINE = _nba_lgb_config("lightgbm_mma_ml", PredictionTask.MMA_MONEYLINE, "mma_moneyline")
NEURAL_NETWORK_MMA_MONEYLINE = _nba_nn_config("neural_network_mma_ml", PredictionTask.MMA_MONEYLINE, "mma_moneyline")
ENSEMBLE_MMA_MONEYLINE = _nba_ensemble_config("ensemble_mma_ml", PredictionTask.MMA_MONEYLINE, "mma_moneyline")
