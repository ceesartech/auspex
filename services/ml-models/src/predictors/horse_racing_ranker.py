"""LightGBM ranker (LambdaMART) adapter for horse racing win prediction.

Why a ranker over a binary classifier (rationale also in the
training script docstring):

  * The problem is intrinsically learning-to-rank, not independent
    binary classification. P(horse A wins) is conditional on the
    field — exactly the relationship pairwise/listwise ranking
    losses (LambdaRank, LambdaMART) capture.
  * The metric we care about is "did the model rank the actual
    winner near the top" (NDCG@1, MRR). Binary cross-entropy
    optimises per-row calibration; LambdaRank optimises the right
    objective directly.
  * The consensus baseline overconfidence in the 40-50% and 50+
    buckets (see `horse-racing-baseline` memory) is the kind of
    systematic bias a ranker can fix because it learns relative
    strength in field context, not absolute likelihood.

Score → probability calibration

LightGBM Ranker outputs unnormalised scores (raw real numbers, no
calibration). To convert to probabilities that sum to 1.0 across a
race we apply a softmax over the field at inference time. The
temperature is tunable; a temperature of 1.0 (vanilla softmax)
typically over-spikes — the favourite gets too much mass — because
the ranker's score gap doesn't directly correspond to log-odds.
We pick the temperature by minimising race-level cross-entropy
on the validation set during fit (a small isotonic calibration
would do the same job and might be cleaner; deferred to v2).

Predict-time contract:

  predict_scores(X)       → np.ndarray of raw scores
  predict_probabilities(X, groups) → list of np.ndarray, one per race,
                          each summing to 1.0 across the race's entrants
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Hyperparameters bundle ──────────────────────────────────────────


@dataclass
class HorseRacingRankerConfig:
    """LightGBM Ranker hyperparameters + training knobs.

    Defaults reflect a tuning pass on the 5,770-race UK/IRE corpus
    (commit a547bef): top-1 acc 22.9% vs 20.7% baseline (+2.2pts)
    with best_iteration=78 and a sensible softmax temperature of
    0.35. The v1 defaults (n_estimators=500, learning_rate=0.05,
    num_leaves=31, reg_lambda=0.1, early_stopping=30) early-stopped
    at iteration 1 with a degenerate temperature of 0.05 — see
    `horse-racing-ml-ranker-v1` memory for the comparison.

    For corpora >>10x larger, expect to need more boosting headroom
    (higher n_estimators) + stronger regularisation. The current
    defaults are a tested sweet spot for ~3 months of UK+IRE data.
    """

    n_estimators: int = 2000
    learning_rate: float = 0.01
    max_depth: int = -1
    num_leaves: int = 63
    min_child_samples: int = 50
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    label_gain: List[int] = field(default_factory=lambda: [0, 1])
    # LambdaMART truncation depth — the NDCG / LambdaRank gradient is
    # computed over the top-`max_position` results. For horse racing
    # we only care about top-1 (the winner) so this is the strongest
    # signal we have; v2 could expose place / show by lifting to 3.
    eval_at_n: int = 1
    early_stopping_rounds: int = 100
    random_state: int = 42
    verbose: int = -1


# ── Predictor ───────────────────────────────────────────────────────


class HorseRacingRanker:
    """LightGBM Ranker wrapper.

    Doesn't subclass BaseModel because the data shape (race-grouped)
    and prediction shape (per-race-softmax) don't fit the team-sport
    interface. Standalone interface:
      fit(X_train, y_train, groups_train, X_val, y_val, groups_val)
      predict_scores(X)              → raw scores per row
      predict_probabilities(X, groups) → list of arrays, race-softmaxed
    """

    def __init__(self, config: Optional[HorseRacingRankerConfig] = None):
        self.config = config or HorseRacingRankerConfig()
        self.model = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        # Picked on the validation set by minimising race-level cross
        # entropy. 1.0 = vanilla softmax; <1.0 sharpens (favourite
        # gets more mass); >1.0 flattens. Stored so predict-time
        # gets the same calibration as training-time evaluation.
        self.temperature: float = 1.0
        self.training_history: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}
        self.validation_metrics: Dict[str, float] = {}

    # ── Fit ─────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        groups_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train the ranker. Validation set is optional but strongly
        recommended — it drives both early stopping AND the
        score→probability temperature tuning."""
        import lightgbm as lgb

        if int(np.sum(groups_train)) != len(X_train):
            raise ValueError(f"groups_train sum ({int(np.sum(groups_train))}) != X_train rows ({len(X_train)})")

        # Defensive median fill — LightGBM handles NaN natively but
        # downstream calibration code wants finite inputs.
        self.feature_names = list(X_train.columns)
        median = X_train.median()
        X_train_filled = X_train.fillna(median)

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            subsample_freq=self.config.subsample_freq,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            label_gain=self.config.label_gain,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
        )

        callbacks = [lgb.log_evaluation(period=0)]
        eval_set: List[Any] = []
        eval_group: List[Any] = []
        eval_names: List[str] = []

        if X_val is not None and y_val is not None and groups_val is not None:
            if int(np.sum(groups_val)) != len(X_val):
                raise ValueError(f"groups_val sum ({int(np.sum(groups_val))}) != X_val rows ({len(X_val)})")
            X_val_filled = X_val[self.feature_names].fillna(median)
            eval_set = [(X_val_filled, y_val)]
            eval_group = [groups_val]
            eval_names = ["val"]
            callbacks.append(lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds))

        logger.info(
            "Training LGBMRanker on %d rows / %d races, %d features",
            len(X_train_filled),
            len(groups_train),
            len(self.feature_names),
        )

        self.model.fit(
            X_train_filled,
            y_train,
            group=groups_train,
            eval_set=eval_set if eval_set else None,
            eval_group=eval_group if eval_group else None,
            eval_names=eval_names if eval_names else None,
            eval_at=[self.config.eval_at_n],
            callbacks=callbacks,
        )

        self.is_fitted = True

        # Feature importance.
        importance = self.model.booster_.feature_importance(importance_type="gain")
        self.feature_importance = dict(
            sorted(
                zip(self.feature_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # Temperature calibration + validation metrics (only when val
        # available — without val we ship with temperature=1.0 and
        # accept the spike).
        if X_val is not None and y_val is not None and groups_val is not None:
            X_val_filled = X_val[self.feature_names].fillna(median)
            self.temperature = self._fit_temperature(X_val_filled, y_val, groups_val)
            self.validation_metrics = self._evaluate(X_val_filled, y_val, groups_val)

        return {
            "feature_importance": self.feature_importance,
            "validation_metrics": self.validation_metrics,
            "temperature": self.temperature,
            "best_iteration": getattr(self.model, "best_iteration_", None),
        }

    # ── Score / probability ────────────────────────────────────────

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Raw LambdaMART scores. Higher = more likely winner; not
        calibrated to a probability scale on its own."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")
        X_filled = X[self.feature_names].fillna(X[self.feature_names].median())
        return np.asarray(self.model.predict(X_filled), dtype=np.float64)

    def predict_probabilities(
        self, X: pd.DataFrame, groups: np.ndarray, temperature: Optional[float] = None
    ) -> List[np.ndarray]:
        """Per-race-softmax probabilities. Returns a list of arrays,
        one per race, each summing to 1.0 across the race's entrants.
        Temperature defaults to the value tuned during fit."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        if int(np.sum(groups)) != len(X):
            raise ValueError(f"groups sum ({int(np.sum(groups))}) != X rows ({len(X)})")
        t = temperature if temperature is not None else self.temperature
        if t <= 0:
            raise ValueError(f"Temperature must be positive, got {t}")
        scores = self.predict_scores(X)
        out: List[np.ndarray] = []
        cursor = 0
        for size in groups:
            race_scores = scores[cursor : cursor + size]
            cursor += int(size)
            out.append(_softmax(race_scores, temperature=t))
        return out

    # ── Temperature calibration ────────────────────────────────────

    def _fit_temperature(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        groups_val: np.ndarray,
    ) -> float:
        """Grid-search the temperature that minimises mean negative-
        log-likelihood of the winner across the validation set.
        Coarse grid → narrow grid pattern; enough to find the right
        order of magnitude without overfitting the val set."""
        scores = self.predict_scores(X_val)
        winners = self._winner_index_per_race(y_val, groups_val)
        coarse = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        best_t = 1.0
        best_nll = float("inf")
        for t in coarse:
            nll = _race_nll(scores, groups_val, winners, t)
            if nll < best_nll:
                best_nll = nll
                best_t = t
        # Narrow grid around the coarse winner.
        narrow = np.linspace(best_t * 0.5, best_t * 2.0, 16)
        for t in narrow:
            if t <= 0:
                continue
            nll = _race_nll(scores, groups_val, winners, float(t))
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)
        logger.info("Tuned softmax temperature: %.3f (val NLL=%.4f)", best_t, best_nll)
        return best_t

    # ── Validation metrics ─────────────────────────────────────────

    def _evaluate(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        groups_val: np.ndarray,
    ) -> Dict[str, float]:
        """Top-1 accuracy + mean reciprocal rank + per-race log-loss.
        These three together cover what we care about: did the model
        pick the winner, how high did it rank the winner on average,
        and how calibrated are the probabilities."""
        scores = self.predict_scores(X_val)
        winners = self._winner_index_per_race(y_val, groups_val)
        top1_hits = 0
        mrr_sum = 0.0
        nll_sum = 0.0
        n_races = 0
        cursor = 0
        for size, winner_local_idx in zip(groups_val, winners):
            race_scores = scores[cursor : cursor + size]
            cursor += int(size)
            if winner_local_idx is None:
                continue
            n_races += 1
            # Higher score → better rank.
            order = np.argsort(-race_scores)
            rank = int(np.where(order == winner_local_idx)[0][0]) + 1
            mrr_sum += 1.0 / rank
            if rank == 1:
                top1_hits += 1
            probs = _softmax(race_scores, self.temperature)
            nll_sum += -np.log(max(probs[winner_local_idx], 1e-12))
        if n_races == 0:
            return {"top1_accuracy": 0.0, "mrr": 0.0, "nll": 0.0, "races": 0}
        return {
            "top1_accuracy": top1_hits / n_races,
            "mrr": mrr_sum / n_races,
            "nll": nll_sum / n_races,
            "races": n_races,
        }

    @staticmethod
    def _winner_index_per_race(y: np.ndarray, groups: np.ndarray) -> List[Optional[int]]:
        """Local winner index per race (relative to the group start),
        or None for races without a recorded winner. The
        load_training_frame query filters those out but we still
        guard defensively for ad-hoc CSV inputs."""
        out: List[Optional[int]] = []
        cursor = 0
        for size in groups:
            block = y[cursor : cursor + size]
            cursor += int(size)
            winners = np.where(block == 1)[0]
            out.append(int(winners[0]) if len(winners) else None)
        return out


# ── Pure helpers (exposed for tests + reuse) ───────────────────────


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically-stable softmax with explicit temperature. Falls
    back to uniform on an empty input rather than crash."""
    if len(scores) == 0:
        return np.array([], dtype=np.float64)
    scaled = np.asarray(scores, dtype=np.float64) / temperature
    scaled -= scaled.max()
    exp = np.exp(scaled)
    s = exp.sum()
    if s == 0:
        return np.full(len(scores), 1.0 / len(scores))
    return exp / s


def _race_nll(
    scores: np.ndarray,
    groups: np.ndarray,
    winners: List[Optional[int]],
    temperature: float,
) -> float:
    """Mean per-race negative-log-likelihood of the winner. Used by
    the temperature tuner."""
    total = 0.0
    n = 0
    cursor = 0
    for size, winner in zip(groups, winners):
        race_scores = scores[cursor : cursor + size]
        cursor += int(size)
        if winner is None:
            continue
        probs = _softmax(race_scores, temperature)
        total += -np.log(max(probs[winner], 1e-12))
        n += 1
    return total / n if n else float("inf")
