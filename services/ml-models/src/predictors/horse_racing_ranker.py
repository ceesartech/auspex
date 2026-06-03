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

    n_estimators is sized at 400 not 2000 because the best iteration
    on the small-corpus tuning was 78 and ~150 was where the val NDCG
    plateaued; 400 leaves 5x headroom for the larger corpus while
    avoiding 26x over-provisioning. early_stopping_rounds=50 closes
    the loop fast when the model converges. n_jobs is explicit (not
    -1) so we don't over-thread on shared hosts.

    For corpora >>10x larger again, expect to need more boosting
    headroom (raise n_estimators) + stronger regularisation. The
    current defaults are a tested sweet spot for the 13k-race
    UK+IRE corpus.
    """

    n_estimators: int = 400
    learning_rate: float = 0.01
    max_depth: int = -1
    num_leaves: int = 63
    min_child_samples: int = 50
    subsample: float = 0.8
    subsample_freq: int = 5
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    label_gain: List[int] = field(default_factory=lambda: [0, 1])
    # LambdaMART truncation depth — the NDCG / LambdaRank gradient is
    # computed over the top-`max_position` results. For horse racing
    # we only care about top-1 (the winner) so this is the strongest
    # signal we have; v2 could expose place / show by lifting to 3.
    eval_at_n: int = 1
    early_stopping_rounds: int = 50
    n_jobs: int = 4
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
        # Picked on the validation set by minimising race-level Brier.
        # 1.0 = vanilla softmax; <1.0 sharpens (favourite gets more
        # mass); >1.0 flattens. Stored so predict-time gets the same
        # calibration as training-time evaluation.
        self.temperature: float = 1.0
        # Isotonic calibrator (post-softmax). Two parallel arrays
        # representing a monotonic piecewise-linear mapping from
        # softmax_prob → calibrated_prob, fit on the val set against
        # actual win/lose outcomes. Predict-time interpolates via
        # numpy.interp, no sklearn dependency at inference. None when
        # the model was fit without a val set (precompute-time
        # fallback) — the predictor then ships uncalibrated softmax.
        self.calibrator_x: Optional[np.ndarray] = None
        self.calibrator_y: Optional[np.ndarray] = None
        self.training_history: Dict[str, Any] = {}
        self.feature_importance: Dict[str, float] = {}
        self.validation_metrics: Dict[str, float] = {}

    # ── Persistence ─────────────────────────────────────────────────

    @classmethod
    def load(cls, model_dir) -> "HorseRacingRanker":
        """Reconstruct a fitted ranker from artefacts written by the
        training script (model.bin + feature_names.json + metadata.json).

        Critical for production: training on the full corpus
        (155k+ rows × 32 features) blows the api container's memory
        budget at every cron tick. The training script runs as a
        one-shot (weekly or on demand), saves to /app/models/, and
        the precompute task in the DAG just loads + scores.

        Skips reconstructing the LGBMRanker sklearn wrapper —
        loaded ranker uses the lightgbm Booster directly. predict()
        works the same way on both, so predict_scores /
        predict_probabilities stay unchanged on the read path.
        """
        import json
        from pathlib import Path

        import lightgbm as lgb

        model_dir = Path(model_dir)
        model_path = model_dir / "model.bin"
        names_path = model_dir / "feature_names.json"
        meta_path = model_dir / "metadata.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model not found at {model_path}")
        if not names_path.exists():
            raise FileNotFoundError(f"feature_names.json not found at {names_path}")

        instance = cls.__new__(cls)
        instance.config = HorseRacingRankerConfig()
        instance.model = lgb.Booster(model_file=str(model_path))
        with open(names_path) as f:
            instance.feature_names = list(json.load(f))
        instance.temperature = 1.0
        instance.calibrator_x = None
        instance.calibrator_y = None
        instance.training_history = {}
        instance.feature_importance = {}
        instance.validation_metrics = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            # Walk through the training metadata for the tuned softmax
            # temperature. Default 1.0 (uniform-like) is intentionally
            # bad to force operators to retrain via the standalone
            # script rather than ship without calibration.
            instance.temperature = float(meta.get("fit_result", {}).get("temperature", 1.0))
            instance.validation_metrics = meta.get("test_metrics", {})
        # Isotonic calibrator is a separate artefact (calibrator.json)
        # so loading it stays optional — backwards-compatible with
        # model directories saved before the calibrator was added.
        cal_path = model_dir / "calibrator.json"
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            instance.calibrator_x = np.asarray(cal["x"], dtype=np.float64)
            instance.calibrator_y = np.asarray(cal["y"], dtype=np.float64)
        instance.is_fitted = True
        return instance

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
            n_jobs=self.config.n_jobs,
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

        # Temperature calibration + isotonic post-cal + validation
        # metrics (only when val available — without val we ship with
        # temperature=1.0 and no calibrator).
        if X_val is not None and y_val is not None and groups_val is not None:
            X_val_filled = X_val[self.feature_names].fillna(median)
            self.temperature = self._fit_temperature(X_val_filled, y_val, groups_val)
            # Fit isotonic AFTER temperature is set so the calibrator
            # maps post-softmax probs (at the chosen temperature) to
            # actual win rates. Order is load-bearing — fitting before
            # the temperature is picked would calibrate against a
            # different probability scale than predict-time uses.
            self._fit_isotonic_calibrator(X_val_filled, y_val, groups_val)
            self.validation_metrics = self._evaluate(X_val_filled, y_val, groups_val)

        return {
            "feature_importance": self.feature_importance,
            "validation_metrics": self.validation_metrics,
            "temperature": self.temperature,
            "best_iteration": getattr(self.model, "best_iteration_", None),
            "has_calibrator": self.calibrator_x is not None,
        }

    def _fit_isotonic_calibrator(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        groups_val: np.ndarray,
    ) -> None:
        """Fit an isotonic calibrator on (softmax_prob, actual_outcome)
        across all val rows. The calibrator is a monotonic piecewise-
        linear mapping from raw softmax prob to calibrated win-rate
        prob; predict-time interpolates via numpy.interp.

        Why this matters: the temperature tuner finds the softmax T
        that minimises race-level Brier, but the per-bucket calibration
        is still imperfect — e.g. on the 13k-race corpus, ranker
        softmax(temp=0.06) puts 10%-prob horses where actual win rate
        is ~3%. Isotonic post-cal fixes that bucket-by-bucket without
        touching the rank order.

        Stores only the knot points (X / Y arrays) so the loaded model
        doesn't need sklearn at inference — numpy.interp suffices.
        """
        from sklearn.isotonic import IsotonicRegression

        scores = self.predict_scores(X_val)
        # Generate per-row softmax probs at the tuned temperature.
        probs_pre = np.empty_like(scores, dtype=np.float64)
        cursor = 0
        for size in groups_val:
            race_probs = _softmax(scores[cursor : cursor + int(size)], self.temperature)
            probs_pre[cursor : cursor + int(size)] = race_probs
            cursor += int(size)

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(probs_pre, y_val.astype(np.float64))
        # Store the piecewise-linear knot points. sklearn exposes them
        # as IsotonicRegression's X_thresholds_ / y_thresholds_ after fit.
        self.calibrator_x = np.asarray(iso.X_thresholds_, dtype=np.float64)
        self.calibrator_y = np.asarray(iso.y_thresholds_, dtype=np.float64)
        logger.info(
            "Fit isotonic calibrator: %d knots, X∈[%.3f,%.3f], Y∈[%.3f,%.3f]",
            len(self.calibrator_x),
            float(self.calibrator_x.min()),
            float(self.calibrator_x.max()),
            float(self.calibrator_y.min()),
            float(self.calibrator_y.max()),
        )

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
        """Per-race calibrated probabilities. Returns a list of arrays,
        one per race, each summing to 1.0 across the race's entrants.

        Pipeline:
          1. softmax(scores / temperature) → per-race raw probs
          2. (optional) apply isotonic calibrator to each raw prob
          3. renormalise per race so the row sums back to 1.0

        Without the calibrator the result is just the temperature-
        scaled softmax. With the calibrator the per-bucket mapping
        corrects the systematic over/under-confidence patterns picked
        up at training time (e.g. ranker softmax(0.06) putting 10%
        prob on horses whose actual win rate is ~3%)."""
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
            race_probs = _softmax(race_scores, temperature=t)
            if self.calibrator_x is not None and self.calibrator_y is not None:
                # Apply isotonic via piecewise-linear interpolation.
                # numpy.interp clips X out-of-range to the endpoint Y,
                # matching IsotonicRegression(out_of_bounds='clip').
                race_probs = np.interp(race_probs, self.calibrator_x, self.calibrator_y)
                # Renormalise so the race's probs still sum to 1.0
                # — calibration can break that invariant since it
                # remaps each entrant independently.
                s = race_probs.sum()
                if s > 0:
                    race_probs = race_probs / s
                else:
                    # Pathological edge case: calibrator mapped every
                    # entrant to 0. Fall back to uniform so downstream
                    # code never sees a NaN-or-divide-by-zero.
                    race_probs = np.full(len(race_probs), 1.0 / len(race_probs))
            out.append(race_probs)
        return out

    # ── Temperature calibration ────────────────────────────────────

    def _fit_temperature(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        groups_val: np.ndarray,
    ) -> float:
        """Grid-search the temperature that minimises mean per-entrant
        Brier score across the validation set.

        Why Brier and not NLL:
          * NLL rewards CONFIDENCE on the right answer — it picks a
            temperature that makes the favourite get as much mass as
            possible. Empirically that produces temperatures so small
            (~0.06) that the probability distribution has nothing to do
            with the real win frequency: a 10-1 longshot gets 13% mass
            instead of its actual ~3-5% rate, the recs engine sees a
            huge EV gap, and fires 520 false-positive picks across 672
            races (verified empirically).
          * Brier measures CALIBRATION — it rewards probabilities that
            match the actual win rate per bucket. That's the right
            objective for downstream EV math because EV = prob × odds
            and we want prob to be a real probability.

        Coarse grid → narrow grid pattern; enough to find the right
        order of magnitude without overfitting the val set."""
        scores = self.predict_scores(X_val)
        winners = self._winner_index_per_race(y_val, groups_val)
        coarse = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        best_t = 1.0
        best_brier = float("inf")
        for t in coarse:
            brier = _race_brier(scores, groups_val, winners, t)
            if brier < best_brier:
                best_brier = brier
                best_t = t
        # Narrow grid around the coarse winner.
        narrow = np.linspace(best_t * 0.5, best_t * 2.0, 16)
        for t in narrow:
            if t <= 0:
                continue
            brier = _race_brier(scores, groups_val, winners, float(t))
            if brier < best_brier:
                best_brier = brier
                best_t = float(t)
        logger.info("Tuned softmax temperature: %.3f (val Brier=%.4f)", best_t, best_brier)
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
    """Mean per-race negative-log-likelihood of the winner. Used as
    a validation metric (see _evaluate); the temperature tuner
    minimises Brier instead — see _race_brier."""
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


def _race_brier(
    scores: np.ndarray,
    groups: np.ndarray,
    winners: List[Optional[int]],
    temperature: float,
) -> float:
    """Mean per-entrant Brier score = (predicted_prob - actual)^2
    averaged over every entrant of every race. The temperature
    tuner uses this rather than NLL because the downstream
    consumer (recs engine + EV math) needs calibrated probabilities
    — see _fit_temperature docstring for the empirical context."""
    total = 0.0
    n = 0
    cursor = 0
    for size, winner in zip(groups, winners):
        race_scores = scores[cursor : cursor + size]
        cursor += int(size)
        if winner is None:
            continue
        probs = _softmax(race_scores, temperature)
        # actuals: 1.0 at winner, 0.0 elsewhere.
        actual = np.zeros(int(size), dtype=np.float64)
        actual[winner] = 1.0
        total += float(np.mean((probs - actual) ** 2))
        n += 1
    return total / n if n else float("inf")
