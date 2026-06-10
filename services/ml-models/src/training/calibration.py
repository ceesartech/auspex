"""Probability calibration for model outputs.

Fits a per-class isotonic (or Platt) map on a held-out validation set and
applies it to predicted probabilities so served confidences match reality
(if the model says 60% for many events, ~60% should occur).

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
    """Per-class probability calibration (isotonic or Platt/sigmoid)."""

    def __init__(self, method: str = "isotonic"):
        """
        Args:
            method: 'isotonic' or 'platt' (sigmoid).
        """
        self.method = method
        # class index -> serializable params:
        #   isotonic: {"x": [...knot x...], "y": [...knot y...]}
        #   platt:    {"a": slope, "b": intercept}
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


def brier_multiclass(y_proba: np.ndarray, y_true: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error of the one-hot vs proba.
    Lower is better. Used to compare raw vs calibrated on the held-out test."""
    y_true = np.asarray(y_true)
    onehot = np.zeros_like(y_proba)
    onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


__all__: List[str] = ["ProbabilityCalibrator", "brier_multiclass"]
