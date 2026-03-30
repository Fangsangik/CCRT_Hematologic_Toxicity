import logging
from typing import Tuple

import numpy as np
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)


class FindThreshold:

    @staticmethod
    def execute(y_true: np.ndarray, y_prob: np.ndarray, min_sensitivity: float = 0.85) -> Tuple[float, float, float]:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        valid_mask = tpr >= min_sensitivity
        if not valid_mask.any():
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            logger.warning(f"민감도 ≥ {min_sensitivity} 달성 불가, Youden's J fallback")
            return float(thresholds[best_idx]), float(tpr[best_idx]), float(1 - fpr[best_idx])

        valid_idx = np.where(valid_mask)[0]
        best_idx = valid_idx[np.argmin(fpr[valid_idx])]

        thresholds = float(thresholds[best_idx])
        sensitivity = float(tpr[best_idx])
        specificity = float(1 - fpr[best_idx])

        logger.info(f"임상 임계값 : {thresholds: .3f} (Sens={sensitivity:.3f}, Spec={specificity:.3f})")
        return thresholds, sensitivity, specificity
