"""
visualization.py - 결과 시각화 모듈

모델 성능, 학습 과정, 특성 중요도 등을 시각화합니다.
의료 연구 논문에 적합한 고품질 그래프를 생성합니다.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

logger = logging.getLogger(__name__)

# ----- 논문용 그래프 스타일 설정 -----
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def plot_roc_curves(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "ROC Curves - Hematologic Toxicity Prediction",
) -> plt.Figure:
    """여러 모델의 ROC 곡선을 한 그래프에 비교합니다.

    Args:
        results: 모델별 평가 결과
            {"모델명": {"roc_curve": {"fpr": ..., "tpr": ...}, "auroc": ...}}
        save_path: 저장 경로 (None이면 저장하지 않음)
        title: 그래프 제목

    Returns:
        matplotlib Figure 객체
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 색상 팔레트 (모델별 구분)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#607D8B"]

    for idx, (model_name, metrics) in enumerate(results.items()):
        roc_data = metrics.get("roc_curve", {})
        fpr = roc_data.get("fpr", [])
        tpr = roc_data.get("tpr", [])
        auroc = metrics.get("auroc", 0)

        color = colors[idx % len(colors)]
        ax.plot(
            fpr, tpr,
            color=color,
            label=f"{model_name} (AUC = {auroc:.3f})",
            linewidth=2,
        )

    # 대각선 (랜덤 분류기)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="Random")

    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        logger.info(f"ROC 곡선 저장: {save_path}")

    return fig


def plot_pr_curves(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curves",
) -> plt.Figure:
    """여러 모델의 Precision-Recall 곡선을 비교합니다.

    클래스 불균형이 심한 경우 ROC보다 PR 곡선이 더 정보적입니다.

    Args:
        results: 모델별 평가 결과
        save_path: 저장 경로
        title: 그래프 제목

    Returns:
        matplotlib Figure 객체
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for idx, (model_name, metrics) in enumerate(results.items()):
        pr_data = metrics.get("pr_curve", {})
        precision = pr_data.get("precision", [])
        recall = pr_data.get("recall", [])
        auprc = metrics.get("auprc", 0)

        color = colors[idx % len(colors)]
        ax.plot(
            recall, precision,
            color=color,
            label=f"{model_name} (AUPRC = {auprc:.3f})",
            linewidth=2,
        )

    # 기준선: 유병률
    prevalence = list(results.values())[0].get("prevalence", 0.25)
    ax.axhline(y=prevalence, color="gray", linestyle="--", alpha=0.5, label=f"Prevalence ({prevalence:.2f})")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        logger.info(f"PR 곡선 저장: {save_path}")

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "LSTM Training History",
) -> plt.Figure:
    """LSTM 학습 히스토리를 시각화합니다.

    손실 곡선과 AUC 곡선을 동시에 표시하여
    과적합 여부를 확인할 수 있습니다.

    Args:
        history: 학습 히스토리 딕셔너리
            {"train_loss": [...], "val_loss": [...], "train_auc": [...], "val_auc": [...]}
        save_path: 저장 경로
        title: 그래프 제목

    Returns:
        matplotlib Figure 객체
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # ----- 손실 곡선 -----
    ax1 = axes[0]
    if "train_loss" in history:
        ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (BCE)")
    ax1.set_title("Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ----- AUC 곡선 -----
    ax2 = axes[1]
    if "train_auc" in history:
        ax2.plot(epochs, history["train_auc"], "b-", label="Train AUC")
    if "val_auc" in history:
        ax2.plot(epochs, history["val_auc"], "r-", label="Val AUC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("AUC Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"학습 히스토리 저장: {save_path}")

    return fig


def plot_feature_importance(
    importances: Dict[str, float],
    top_n: int = 20,
    save_path: Optional[str] = None,
    title: str = "Feature Importance",
) -> plt.Figure:
    """특성 중요도를 수평 막대 그래프로 시각화합니다.

    AMC 관련 특성이 상위에 위치하는지 확인하기 위해 사용됩니다.
    (핵심 가설 검증)

    Args:
        importances: 특성명-중요도 딕셔너리
        top_n: 표시할 상위 특성 수
        save_path: 저장 경로
        title: 그래프 제목

    Returns:
        matplotlib Figure 객체
    """
    # 상위 N개 추출
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:top_n]
    features, values = zip(*reversed(top_items))  # 역순 (위에서 아래로)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))

    # AMC 관련 특성은 빨간색으로 강조
    colors = [
        "#FF5722" if "AMC" in f or "amc" in f else "#2196F3"
        for f in features
    ]

    bars = ax.barh(range(len(features)), values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    # 범례 추가
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FF5722", alpha=0.8, label="AMC-related"),
        Patch(facecolor="#2196F3", alpha=0.8, label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"특성 중요도 저장: {save_path}")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    labels: Optional[List[str]] = None,
) -> plt.Figure:
    """혼동 행렬을 히트맵으로 시각화합니다.

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        save_path: 저장 경로
        title: 그래프 제목
        labels: 클래스 라벨 (기본: ["No HT (Grade <3)", "HT (Grade 3+)"])

    Returns:
        matplotlib Figure 객체
    """
    from sklearn.metrics import confusion_matrix as cm_func

    if labels is None:
        labels = ["No HT\n(Grade <3)", "HT\n(Grade 3+)"]

    cm = cm_func(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))

    # 히트맵 그리기
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Actual",
        xlabel="Predicted",
        title=title,
    )

    # 각 셀에 숫자 표시
    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=18, fontweight="bold",
            )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"혼동 행렬 저장: {save_path}")

    return fig


def plot_cbc_timeseries(
    df,
    patient_ids: Optional[List[str]] = None,
    feature: str = "AMC",
    target_col: str = "grade3_neutropenia",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """CBC 시계열 변화를 양성/음성 그룹별로 시각화합니다.

    핵심 가설 검증: AMC 감소 패턴이 양성 그룹에서 더 뚜렷한지 확인합니다.

    Args:
        df: CBC 시계열이 포함된 DataFrame
        patient_ids: 개별 표시할 환자 ID (None이면 그룹 평균만)
        feature: 시각화할 CBC 변수 (기본: "AMC")
        target_col: 타겟 컬럼명
        save_path: 저장 경로

    Returns:
        matplotlib Figure 객체
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    timepoints = [0, 1, 2]

    # 양성/음성 그룹으로 분리
    for label, color, group_name in [
        (1, "#FF5722", "Grade 3+ HT (Positive)"),
        (0, "#2196F3", "No HT (Negative)"),
    ]:
        group = df[df[target_col] == label]
        values = np.array([
            group[f"{feature}_week{t}"].values for t in timepoints
        ])  # (n_timepoints, n_patients)

        # 그룹 평균 및 표준편차
        mean_vals = values.mean(axis=1)
        std_vals = values.std(axis=1)

        ax.plot(
            timepoints, mean_vals,
            color=color, linewidth=2.5,
            marker="o", markersize=8,
            label=f"{group_name} (n={len(group)})",
        )

        # 신뢰구간 (평균 ± 1 SD)
        ax.fill_between(
            timepoints,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=color, alpha=0.15,
        )

    ax.set_xlabel("Treatment Week")
    ax.set_ylabel(f"{feature} (10³/μL)")
    ax.set_title(f"{feature} Time-Series by Hematologic Toxicity Status")
    ax.set_xticks(timepoints)
    ax.set_xticklabels(["Week 0\n(Baseline)", "Week 1", "Week 2"])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"CBC 시계열 그래프 저장: {save_path}")

    return fig


def plot_model_comparison_bar(
    comparison: Dict[str, Dict[str, float]],
    metrics_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Model Performance Comparison",
) -> plt.Figure:
    """모델별 성능을 그룹 막대 그래프로 비교합니다.

    Args:
        comparison: 모델별 지표 딕셔너리
        metrics_to_plot: 표시할 지표 (None이면 주요 지표)
        save_path: 저장 경로
        title: 그래프 제목

    Returns:
        matplotlib Figure 객체
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["auroc", "auprc", "sensitivity", "specificity", "f1_score"]

    model_names = list(comparison.keys())
    n_models = len(model_names)
    n_metrics = len(metrics_to_plot)

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 2), 6))

    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for i, model_name in enumerate(model_names):
        values = [
            comparison[model_name].get(m, 0) or 0
            for m in metrics_to_plot
        ]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values,
            width, label=model_name,
            color=colors[i % len(colors)], alpha=0.85,
        )

        # 값 표시
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot])
    ax.set_ylim([0, 1.1])
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info(f"모델 비교 그래프 저장: {save_path}")

    return fig
