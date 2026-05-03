"""Probability calibration for model outputs."""

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Calibrate predicted probabilities using Platt scaling or isotonic regression.

    Ensures predicted probabilities are well-calibrated (e.g., if a model says
    60% for many events, ~60% of those should actually occur).
    """

    def __init__(self, method: str = "isotonic"):
        """
        Args:
            method: 'isotonic' or 'platt' (sigmoid).
        """
        self.method = method
        self.calibrators: Dict[int, Any] = {}
        self.is_fitted = False

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> None:
        """Fit calibration model.

        Args:
            y_proba: Predicted probabilities, shape (n_samples, n_classes).
            y_true: True labels, shape (n_samples,) as integers.
        """
        n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2

        for c in range(n_classes):
            if y_proba.ndim > 1:
                p = y_proba[:, c]
            else:
                p = y_proba if c == 1 else 1 - y_proba

            y_binary = (y_true == c).astype(int)

            if self.method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(p, y_binary)
            else:
                # Platt scaling (logistic regression on logits)
                from sklearn.linear_model import LogisticRegression

                logits = np.log(np.clip(p, 1e-10, 1 - 1e-10) / (1 - np.clip(p, 1e-10, 1 - 1e-10)))
                cal = LogisticRegression()
                cal.fit(logits.reshape(-1, 1), y_binary)

            self.calibrators[c] = cal

        self.is_fitted = True
        logger.info(f"Fitted {self.method} calibrator for {n_classes} classes")

    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply calibration to predicted probabilities.

        Returns calibrated probabilities that sum to 1.
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")

        n_classes = len(self.calibrators)
        n_samples = y_proba.shape[0]
        calibrated = np.zeros((n_samples, n_classes))

        for c in range(n_classes):
            if y_proba.ndim > 1:
                p = y_proba[:, c]
            else:
                p = y_proba if c == 1 else 1 - y_proba

            cal = self.calibrators[c]

            if self.method == "isotonic":
                calibrated[:, c] = cal.predict(p)
            else:
                logits = np.log(np.clip(p, 1e-10, 1 - 1e-10) / (1 - np.clip(p, 1e-10, 1 - 1e-10)))
                calibrated[:, c] = cal.predict_proba(logits.reshape(-1, 1))[:, 1]

        # Normalize rows to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        calibrated = calibrated / row_sums

        return calibrated

    def calibration_error(
        self, y_proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10
    ) -> Dict[str, float]:
        """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
        n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2

        ece_total = 0.0
        mce_total = 0.0

        for c in range(n_classes):
            if y_proba.ndim > 1:
                p = y_proba[:, c]
            else:
                p = y_proba if c == 1 else 1 - y_proba

            y_binary = (y_true == c).astype(int)

            bin_edges = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            mce = 0.0
            n_total = len(p)

            for i in range(n_bins):
                mask = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
                if mask.sum() == 0:
                    continue

                bin_acc = y_binary[mask].mean()
                bin_conf = p[mask].mean()
                bin_size = mask.sum()

                gap = abs(bin_acc - bin_conf)
                ece += gap * bin_size / n_total
                mce = max(mce, gap)

            ece_total += ece
            mce_total = max(mce_total, mce)

        return {
            "ece": ece_total / n_classes,
            "mce": mce_total,
        }
