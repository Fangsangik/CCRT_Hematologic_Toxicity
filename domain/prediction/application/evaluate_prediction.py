import logging
from typing import Tuple, Union, Dict

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, brier_score_loss, precision_recall_curve, \
    average_precision_score, roc_auc_score, f1_score

logger = logging.getLogger(__name__)


class EvaluatePrediction:
    """예측 모델 평가 application service입니다."""

    @staticmethod
    def find_clinical_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                                min_sensitivity: float = 0.85, min_specificity: float = 0.30) -> Tuple[
        float, float, float]:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_prob)
        specificity_arr = 1 - false_positive_rate

        valid_mask = (true_positive_rate >= min_sensitivity) & (specificity_arr >= min_specificity)

        if not valid_mask.any():
            sens_only = true_positive_rate >= min_sensitivity
            if sens_only.any():
                idx_arr = np.where(sens_only)[0]  # where : 조건을 만족하는 요소의 인덱스를 반환
                best_idx = idx_arr[np.argmin(false_positive_rate[sens_only])]  # argmin : 배열에서 가장 작은 값의 인덱스를 반환
                if (1 - false_positive_rate[best_idx]) < min_specificity:
                    j_scores = true_positive_rate - false_positive_rate  # Youden's J statistic : 민감도 - (1 - 특이도) = 민감도 - 거짓 양성률
                    best_idx = np.argmax(j_scores)  # argmax : 배열에서 가장 큰 값의 인덱스를 반환
                return float(thresholds[best_idx]), float(true_positive_rate[best_idx]), float(
                    1 - false_positive_rate[best_idx])
            j_scores = true_positive_rate - false_positive_rate
            best_idx = np.argmax(j_scores)
            return float(thresholds[best_idx]), float(true_positive_rate[best_idx]), float(
                1 - false_positive_rate[best_idx])

        valid_indices = np.where(valid_mask)[0]
        best_idx = valid_indices[np.argmin(false_positive_rate[valid_indices])]
        return float(thresholds[best_idx]), float(true_positive_rate[best_idx]), float(
            1 - false_positive_rate[best_idx])

    def compute_all_metrics(self, y_true: np.ndarray, y_prob: np.ndarray,
                            threshold: Union[float, str] = "auto", min_sensitivity: float = 0.85) -> Dict:
        """모든 평가 지표를 계산합니다."""
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            fallback_threshold = 0.5
            if isinstance(threshold, str):
                threshold = fallback_threshold
            else:
                threshold = float(threshold)

            y_pred = (y_prob >= threshold).astype(int)
            return {
                "auroc": float("nan"), "auprc": float("nan"),
                "accuracy": accuracy_score(y_true, y_pred),
                "brier_score": brier_score_loss(y_true, y_prob),
                "n_samples": len(y_true), "n_positive": int(y_true.sum()),
            }

        false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(y_true, y_prob)
        auroc = auc(false_positive_rate, true_positive_rate)

        # Youden's J statistic
        j_scores = true_positive_rate - false_positive_rate
        optimal_idx = np.argmax(j_scores)
        youden_threshold = float(roc_thresholds[optimal_idx])

        # clinical threshold
        clinical_threshold, clinical_sensitivity, clinical_specificity = self.find_clinical_threshold(
            y_true, y_prob, min_sensitivity=min_sensitivity
        )

        if threshold == "auto":
            used_threshold = youden_threshold
        elif threshold == "clinical":
            used_threshold = clinical_threshold
        else:
            used_threshold = float(threshold)

        y_pred = (y_prob >= used_threshold).astype(int)
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_true, y_pred,
                                                                                       labels=[0, 1]).ravel()

        auprc = average_precision_score(y_true, y_prob)
        return {
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": accuracy_score(y_true, y_pred),
            "sensitivity": true_positive / max(true_positive + false_negative, 1),
            "specificity": true_negative / max(true_negative + false_positive, 1),
            "ppv": true_positive / max(true_positive + false_positive, 1),
            "npv": true_negative / max(true_negative + false_negative, 1),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "tp": int(true_positive), "fp": int(false_positive), "tn": int(true_negative), "fn": int(false_negative),
            "optimal_threshold": youden_threshold,
            "clinical_threshold": clinical_threshold,
            "clinical_sensitivity": clinical_sensitivity,
            "clinical_specificity": clinical_specificity,
            "brier_score": brier_score_loss(y_true, y_prob),
            "threshold_used": used_threshold,
            "n_samples": len(y_true),
            "n_positive": int(y_true.sum()),
            "prevalence": float(y_true.mean()),
        }

    @staticmethod
    def compute_calibration(y_true : np.ndarray, y_prob : np.ndarray, n_bins : int = 10) -> Dict:
        """보조성능을 평가"""
        fraction_positive, mean_prediction = calibration_curve(y_true, y_prob, n_bins = n_bins, strategy= "uniform")
        bin_counts = np.histogram(y_prob, bins = n_bins, range= (0, 1))[0]
        ece = float(np.sum(np.abs(fraction_positive - mean_prediction) * bin_counts[:len(fraction_positive)] / len(y_true)))
        return {
            "fraction_of_positives": fraction_positive,
            "mean_predicted_value": mean_prediction,
            "bin_counts": bin_counts,
            "ece": ece,
            "brier_score": brier_score_loss(y_true, y_prob)
        }

    @staticmethod
    def bootstrap_ci(
            y_true : np.ndarray, y_prob : np.ndarray, metric_function = roc_auc_score, n_bootstraps : int = 1000, ci_level : float = 0.95, random_state : int = 42) -> Tuple[float, float, float]:
        """보정 성능을 평가"""
        if len(np.unique(y_true)) < 2:
            return (float("nan"), float("nan"), float("nan"))
        rng = np.random.RandomState(random_state)
        scores = []
        for _ in range(n_bootstraps):
            idx = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            score = metric_function(y_true[idx], y_prob[idx])
            scores.append(score)
        if not scores:
            return (float("nan"), float("nan"), float("nan"))
        alpha = (1 - ci_level) / 2
        return metric_function(y_true, y_prob), np.percentile(scores, 100 * alpha), np.percentile(scores, 100 * (1 - alpha))

    @staticmethod
    def compute_incremental_value(baseline_model_metrics : Dict, enhanced_metrics : Dict) -> Dict[str, float]:
        keys = ["auroc", "auprc", "sensitivity", "specificity", "f1_score"]
        incremental = {}
        for key in keys :
            base = baseline_model_metrics.get(key, 0)
            enhanced = enhanced_metrics.get(key, 0)
            incremental[f"{key}_baseline"] = base
            incremental[f"{key}_enhanced"] = enhanced
            incremental[f"{key}_increment"] = enhanced - base
        return incremental

    