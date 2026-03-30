import logging
from typing import Dict

import numpy as np
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

class EvaluateScreening :

    @staticmethod
    def compute_screening_metrics(y_true : np.ndarray, y_prob : np.ndarray, threshold : float) -> Dict:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels= [0, 1]).ravel()

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)

        return {
            "threshold": threshold,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
            "n_screened_positive": int(tp + fp),
            "n_total": len(y_true),
        }

    @staticmethod
    def compute_dca(y_true : np.ndarray, y_prob : np.ndarray, thresholds : np.ndarray = None) -> Dict:
        if thresholds is None:
            thresholds = np.arange(0.01, 0.99, 0.01)

        prevalence = y_true.mean()
        net_benefits = []
        treat_all = []

        for t in thresholds :
            y_pred = (y_prob >= t).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            n = len(y_true)

            nb = (tp / n) - (fp / n) * (t / (1 - t)) if t < 1 else 0
            ta = prevalence - (1 - prevalence) * (t / (1 - t)) if t < 1 else 0

            net_benefits.append(nb)
            treat_all.append(ta)

        return {
            "thresholds": thresholds,
            "net_benefit_model": np.array(net_benefits),
            "net_benefit_treat_all": np.array(treat_all),
        }