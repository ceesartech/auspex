"""Probability calibration for model outputs.

Fits a calibration map on a held-out validation set and applies it to
predicted probabilities so served confidences match reality (if the model
says 60% for many events, ~60% should occur). Three methods:

  temperature  one scalar T for the whole probability vector, p_c ∝ p_c^(1/T).
               Smooth, order-preserving, cannot invent steps or extremes —
               the method the ensemble self-gate uses since 2026-10-25.
  isotonic     per-class step function. Kept for diagnostics only: on a
               ~500-row val set it produced a 34-knot step map for NBA total
               that served 22 distinct probabilities and an overconfident
               P(over)≈0.13-0.25 cluster live (audit §7, 2026-10-25).
  platt        per-class sigmoid on the logit.

The fitted maps are stored as plain arrays (isotonic step-function knots)
rather than sklearn objects, so the calibrator round-trips through JSON
(to_dict / from_dict) and applies at serve time with a bare np.interp —
no sklearn dependency or pickle-version coupling on the serving path.
"""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Probability calibration (temperature, isotonic or Platt/sigmoid)."""

    METHODS = ("temperature", "isotonic", "platt")

    def __init__(self, method: str = "isotonic"):
        """
        Args:
            method: 'temperature', 'isotonic' or 'platt' (sigmoid).
        """
        if method not in self.METHODS:
            raise ValueError(f"unknown calibration method {method!r}; expected one of {self.METHODS}")
        self.method = method
        # class index -> serializable params:
        #   isotonic: {"x": [...knot x...], "y": [...knot y...]}
        #   platt:    {"a": slope, "b": intercept}
        #   temperature: one entry under key -1: {"t": T, "n_classes": k}
        self.params: Dict[int, Dict[str, Any]] = {}
        self.is_fitted = False

    @staticmethod
    def _class_proba(y_proba: np.ndarray, c: int) -> np.ndarray:
        if y_proba.ndim > 1:
            return y_proba[:, c]
        return y_proba if c == 1 else 1 - y_proba

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> None:
        """Fit calibration on validation predictions.

        Args:
            y_proba: predicted probabilities, shape (n_samples, n_classes).
            y_true: integer class labels, shape (n_samples,).
        """
        from sklearn.isotonic import IsotonicRegression

        y_true = np.asarray(y_true)
        n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2
        self.params = {}

        if self.method == "temperature":
            t = fit_temperature(_as_matrix(y_proba), y_true)
            self.params = {-1: {"t": float(t), "n_classes": int(n_classes)}}
            self.is_fitted = True
            logger.info("Fitted temperature calibrator: T=%.4f over %d classes", t, n_classes)
            return

        for c in range(n_classes):
            p = self._class_proba(y_proba, c)
            y_binary = (y_true == c).astype(int)

            if self.method == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p, y_binary)
                # Store the step-function knots; np.interp reproduces
                # iso.predict() (with clip) at serve without sklearn.
                self.params[c] = {
                    "x": np.asarray(iso.X_thresholds_, dtype=float).tolist(),
                    "y": np.asarray(iso.y_thresholds_, dtype=float).tolist(),
                }
            else:
                from sklearn.linear_model import LogisticRegression

                eps = 1e-10
                pc = np.clip(p, eps, 1 - eps)
                logits = np.log(pc / (1 - pc))
                lr = LogisticRegression()
                lr.fit(logits.reshape(-1, 1), y_binary)
                self.params[c] = {
                    "a": float(lr.coef_.ravel()[0]),
                    "b": float(lr.intercept_.ravel()[0]),
                }

        self.is_fitted = True
        logger.info("Fitted %s calibrator for %d classes", self.method, n_classes)

    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply calibration; returns probabilities normalized to sum to 1."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")

        if self.method == "temperature":
            return apply_temperature(_as_matrix(y_proba), float(self.params[-1]["t"]))

        n_classes = len(self.params)
        n_samples = y_proba.shape[0] if y_proba.ndim > 1 else len(y_proba)
        calibrated = np.zeros((n_samples, n_classes))

        for c in range(n_classes):
            p = self._class_proba(y_proba, c)
            prm = self.params[c]
            if self.method == "isotonic":
                x = np.asarray(prm["x"], dtype=float)
                y = np.asarray(prm["y"], dtype=float)
                # np.interp clips to [x[0], x[-1]] like IsotonicRegression.
                calibrated[:, c] = np.interp(p, x, y)
            else:
                eps = 1e-10
                pc = np.clip(p, eps, 1 - eps)
                logits = np.log(pc / (1 - pc))
                z = prm["a"] * logits + prm["b"]
                calibrated[:, c] = 1.0 / (1.0 + np.exp(-z))

        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return calibrated / row_sums

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form (rides inside the ensemble metadata)."""
        return {
            "method": self.method,
            # JSON object keys must be strings.
            "params": {str(c): prm for c, prm in self.params.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProbabilityCalibrator":
        obj = cls(method=d.get("method", "isotonic"))
        obj.params = {int(c): prm for c, prm in d.get("params", {}).items()}
        obj.is_fitted = bool(obj.params)
        return obj

    def calibration_error(self, y_proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
        """Expected (ECE) + Maximum (MCE) Calibration Error on (y_proba, y_true)."""
        y_true = np.asarray(y_true)
        n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2
        ece_total = 0.0
        mce_total = 0.0

        for c in range(n_classes):
            p = self._class_proba(y_proba, c)
            y_binary = (y_true == c).astype(int)
            bin_edges = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            mce = 0.0
            n_total = len(p)

            for i in range(n_bins):
                mask = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
                if mask.sum() == 0:
                    continue
                gap = abs(y_binary[mask].mean() - p[mask].mean())
                ece += gap * mask.sum() / n_total
                mce = max(mce, gap)

            ece_total += ece
            mce_total = max(mce_total, mce)

        return {"ece": ece_total / n_classes, "mce": mce_total}


def _as_matrix(y_proba: np.ndarray) -> np.ndarray:
    """(n,) positive-class vector -> (n, 2) [1-p, p]; (n, k) passes through."""
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim == 1:
        return np.column_stack([1.0 - y_proba, y_proba])
    return y_proba


def apply_temperature(y_proba: np.ndarray, t: float) -> np.ndarray:
    """p_c ∝ p_c^(1/T), renormalised. T > 1 softens, T < 1 sharpens, T = 1 is identity."""
    logp = np.log(np.clip(np.asarray(y_proba, dtype=float), 1e-12, 1.0)) / float(t)
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(y_proba: np.ndarray, y_true: np.ndarray, lo: float = 0.25, hi: float = 4.0) -> float:
    """The T in [lo, hi] minimising multiclass log-loss of apply_temperature(y_proba, T).

    Golden-section search on a 1-D, smooth objective (log-loss in T is unimodal
    for this family), so no optimiser dependency. The bounds keep a degenerate
    val set from producing an absurd map; a T at a bound is logged by the
    caller's gate metrics, never silently trusted."""
    y = np.asarray(y_true).astype(int)
    idx = np.arange(len(y))

    def nll(t: float) -> float:
        p = apply_temperature(y_proba, t)
        return float(-np.mean(np.log(np.clip(p[idx, y], 1e-12, 1.0))))

    g = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = np.log(lo), np.log(hi)  # search in log T: symmetric for sharpen/soften
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(np.exp(c)), nll(np.exp(d))
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(np.exp(d))
        if b - a < 1e-6:
            break
    return float(np.exp(0.5 * (a + b)))


def brier_rows(y_proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Per-row multiclass Brier (sum of squares vs one-hot) — the paired unit."""
    y_true = np.asarray(y_true)
    onehot = np.zeros_like(y_proba, dtype=float)
    onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return np.sum((y_proba - onehot) ** 2, axis=1)


def brier_multiclass(y_proba: np.ndarray, y_true: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error of the one-hot vs proba.
    Lower is better. Used to compare raw vs calibrated on the held-out test."""
    y_true = np.asarray(y_true)
    onehot = np.zeros_like(y_proba)
    onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


__all__: List[str] = [
    "ProbabilityCalibrator",
    "apply_temperature",
    "brier_multiclass",
    "brier_rows",
    "fit_temperature",
]
