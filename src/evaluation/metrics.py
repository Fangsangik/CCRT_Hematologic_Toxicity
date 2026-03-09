"""
metrics.py - 모델 평가 지표 모듈

이진 분류 모델의 성능을 다각도로 평가합니다.
의료 AI 모델에 특화된 지표(민감도, 특이도, NPV, PPV)를 포함합니다.

혈액독성 예측에서는 고위험군을 놓치지 않는 것(높은 민감도)이
특히 중요합니다.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
) -> Dict:
    """모든 평가 지표를 한 번에 계산합니다.

    의료 AI에 필요한 다양한 지표를 포함합니다:
        - 판별 지표: AUC-ROC, AUC-PR
        - 분류 지표: 정확도, 민감도, 특이도, PPV, NPV
        - 보정 지표: Brier Score
        - 최적 임계값: Youden's J statistic 기반

    Args:
        y_true: 실제 레이블 (0 또는 1)
        y_prob: 예측 확률 (0~1)
        threshold: 분류 임계값
        model_name: 결과 출력용 모델 이름

    Returns:
        모든 평가 지표를 담은 딕셔너리
    """
    y_pred = (y_prob >= threshold).astype(int)

    # ----- 단일 클래스 데이터 처리 -----
    # 소규모 데이터나 템플릿 데이터에서는 한 클래스만 존재할 수 있음
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logger.warning(
            f"[{model_name}] 단일 클래스만 존재 (class={unique_classes[0]}). "
            "대부분의 지표를 계산할 수 없습니다. 실제 데이터에서는 양성/음성이 모두 필요합니다."
        )
        return {
            "model_name": model_name,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "accuracy": accuracy_score(y_true, y_pred),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "ppv": float("nan"),
            "npv": float("nan"),
            "f1": float("nan"),
            "brier_score": brier_score_loss(y_true, y_prob),
            "optimal_threshold": threshold,
            "n_samples": len(y_true),
            "n_positive": int(y_true.sum()),
            "prevalence": float(y_true.mean()),
        }

    # ----- 혼동 행렬 기반 지표 -----
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / max(tp + fn, 1)   # 민감도 (=Recall): 실제 양성 중 올바르게 예측한 비율
    specificity = tn / max(tn + fp, 1)   # 특이도: 실제 음성 중 올바르게 예측한 비율
    ppv = tp / max(tp + fp, 1)           # 양성예측도 (=Precision): 양성 예측 중 실제 양성 비율
    npv = tn / max(tn + fn, 1)           # 음성예측도: 음성 예측 중 실제 음성 비율

    # ----- ROC 곡선 지표 -----
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    auroc = auc(fpr, tpr)

    # ----- PR 곡선 지표 -----
    precision_arr, recall_arr, pr_thresholds = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    # ----- 최적 임계값 (Youden's J statistic) -----
    # J = Sensitivity + Specificity - 1 을 최대화하는 임계값
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]

    # ----- 보정 지표 -----
    brier = brier_score_loss(y_true, y_prob)

    metrics = {
        # 판별 지표
        "auroc": auroc,
        "auprc": auprc,

        # 분류 지표 (주어진 threshold 기준)
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,   # = recall
        "specificity": specificity,
        "ppv": ppv,                    # = precision
        "npv": npv,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),

        # 혼동 행렬 원소
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),

        # 최적 임계값 정보
        "optimal_threshold": float(optimal_threshold),
        "optimal_sensitivity": float(optimal_sensitivity),
        "optimal_specificity": float(optimal_specificity),

        # 보정 지표
        "brier_score": brier,

        # ROC/PR 곡선 데이터 (시각화용)
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "pr_curve": {
            "precision": precision_arr,
            "recall": recall_arr,
            "thresholds": pr_thresholds,
        },

        # 메타데이터
        "threshold_used": threshold,
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
    }

    # 결과 로깅
    logger.info(
        f"[{model_name}] AUROC={auroc:.4f}, AUPRC={auprc:.4f}, "
        f"Sens={sensitivity:.4f}, Spec={specificity:.4f}, "
        f"Optimal Threshold={optimal_threshold:.3f}"
    )

    return metrics


def compare_models(
    results: Dict[str, Dict],
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """여러 모델의 성능을 비교합니다.

    Baseline-only vs Baseline+CBC 모델 비교 또는
    LSTM vs XGBoost vs LightGBM 비교에 사용됩니다.

    Args:
        results: 모델별 평가 결과 딕셔너리
            {"모델명": compute_all_metrics 결과, ...}
        metric_keys: 비교할 지표 키 목록
            (None이면 주요 지표 자동 선택)

    Returns:
        모델별 주요 지표 비교 딕셔너리
    """
    if metric_keys is None:
        metric_keys = [
            "auroc",
            "auprc",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "f1_score",
            "brier_score",
        ]

    comparison = {}
    for model_name, metrics in results.items():
        comparison[model_name] = {
            key: metrics.get(key, None) for key in metric_keys
        }

    # 비교 테이블 로깅
    header = f"{'모델':<25}" + "".join(f"{k:<15}" for k in metric_keys)
    logger.info(f"\n{'='*120}\n모델 성능 비교\n{'='*120}")
    logger.info(header)
    logger.info("-" * 120)

    for model_name, model_metrics in comparison.items():
        row = f"{model_name:<25}"
        row += "".join(
            f"{v:<15.4f}" if v is not None else f"{'N/A':<15}"
            for v in model_metrics.values()
        )
        logger.info(row)

    return comparison


def compute_incremental_value(
    baseline_metrics: Dict,
    enhanced_metrics: Dict,
) -> Dict[str, float]:
    """CBC 시계열 추가의 incremental value를 계산합니다.

    연구 계획서의 핵심 비교 실험:
        Baseline-only 모델 vs Baseline + CBC 시계열 모델

    Args:
        baseline_metrics: Baseline-only 모델 결과
        enhanced_metrics: Baseline + CBC 시계열 모델 결과

    Returns:
        각 지표별 개선량 딕셔너리
    """
    comparison_keys = ["auroc", "auprc", "sensitivity", "specificity", "f1_score"]

    incremental = {}
    for key in comparison_keys:
        base_val = baseline_metrics.get(key, 0)
        enhanced_val = enhanced_metrics.get(key, 0)

        incremental[f"{key}_baseline"] = base_val
        incremental[f"{key}_enhanced"] = enhanced_val
        incremental[f"{key}_delta"] = enhanced_val - base_val
        incremental[f"{key}_relative_improvement"] = (
            (enhanced_val - base_val) / max(base_val, 1e-6) * 100
        )

    # DeLong test를 통한 AUC 유의미성 검정 (간소화)
    logger.info(
        f"\n=== Incremental Value (CBC 시계열 추가 효과) ===\n"
        f"AUROC: {incremental['auroc_baseline']:.4f} → "
        f"{incremental['auroc_enhanced']:.4f} "
        f"(Δ={incremental['auroc_delta']:+.4f}, "
        f"{incremental['auroc_relative_improvement']:+.1f}%)\n"
        f"Sensitivity: {incremental['sensitivity_baseline']:.4f} → "
        f"{incremental['sensitivity_enhanced']:.4f} "
        f"(Δ={incremental['sensitivity_delta']:+.4f})"
    )

    return incremental


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """모델 보정(calibration) 성능을 평가합니다.

    잘 보정된 모델은 "50% 확률로 예측한 환자 중 실제 50%가 양성"이어야 합니다.
    임상 의사결정에 확률값을 직접 사용하려면 좋은 보정이 필수입니다.

    Args:
        y_true: 실제 레이블
        y_prob: 예측 확률
        n_bins: 보정 곡선의 bin 수

    Returns:
        보정 곡선 데이터 딕셔너리
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # Expected Calibration Error (ECE)
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    ece = np.sum(
        np.abs(fraction_of_positives - mean_predicted_value)
        * (bin_counts[: len(fraction_of_positives)] / len(y_true))
    )

    return {
        "fraction_of_positives": fraction_of_positives,
        "mean_predicted_value": mean_predicted_value,
        "ece": ece,
        "brier_score": brier_score_loss(y_true, y_prob),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn=roc_auc_score,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap으로 신뢰구간을 계산합니다.

    의료 연구에서 성능 지표의 불확실성을 보고하기 위해 필요합니다.

    Args:
        y_true: 실제 레이블
        y_prob: 예측 확률
        metric_fn: 계산할 지표 함수
        n_bootstraps: 부트스트랩 횟수
        ci_level: 신뢰 수준 (기본 95%)
        random_state: 재현성 시드

    Returns:
        (지표값, 하한, 상한) 튜플
    """
    # 단일 클래스 데이터 처리
    if len(np.unique(y_true)) < 2:
        logger.warning("Bootstrap CI: 단일 클래스 데이터로 AUC를 계산할 수 없습니다.")
        return (float("nan"), float("nan"), float("nan"))

    rng = np.random.RandomState(random_state)
    scores = []

    for _ in range(n_bootstraps):
        # 복원 추출
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            # 단일 클래스만 포함된 경우 건너뜀
            continue
        score = metric_fn(y_true[indices], y_prob[indices])
        scores.append(score)

    if len(scores) == 0:
        logger.warning("Bootstrap CI: 유효한 부트스트랩 샘플이 없습니다.")
        return (float("nan"), float("nan"), float("nan"))

    scores = np.array(scores)
    alpha = (1 - ci_level) / 2

    point_estimate = metric_fn(y_true, y_prob)
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    logger.info(
        f"Bootstrap CI ({ci_level*100:.0f}%): "
        f"{point_estimate:.4f} [{lower:.4f} - {upper:.4f}]"
    )

    return point_estimate, lower, upper
