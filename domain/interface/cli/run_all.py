"""전체 파이프라인 실행 + 시각화"""
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / RUN_TIMESTAMP
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"train_{RUN_TIMESTAMP}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    from shared.infrastructure.repository.csv_repository import CSVRepository
    from prediction.domain import FeatureService
    from prediction.application.use_cases.train_prediction import TrainPrediction
    from prediction.application.use_cases.evaluate_prediction import EvaluatePrediction
    from prediction.application.use_cases.run_shap import RunSHAP
    from screening.application.use_cases.find_threshold import FindThreshold
    from screening.application.use_cases.evaluate_screening import EvaluateScreening
    from shared.infrastructure.ml.xgboost_model import XGBoostModel
    from shared.infrastructure.ml.lightgbm_model import LightGBMModel
    from shared.infrastructure.ml.logistic_model import LogisticModel

    # ── 1. 데이터 준비 ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("1. 데이터 준비")
    logger.info("=" * 60)

    csv_repo = CSVRepository()
    data_path = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
    if not data_path.exists():
        logger.error(f"데이터 파일이 없습니다: {data_path}")
        logger.error("먼저 전처리를 실행하세요: python interfaces/cli/preprocess.py --data <파일경로>")
        return

    df = csv_repo.load(str(data_path))
    df = csv_repo.handle_missing(df)

    feature_service = FeatureService()
    df, feature_names = feature_service.extract_all(df, mode="baseline_cbc")

    # 범주형 인코딩
    cat_cols = [c for c in ["sex", "stage", "t_stage", "n_stage", "ecog_ps", "chemo_regimen"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        encoded = [c for c in df.columns if any(c.startswith(f"{cat}_") for cat in cat_cols)]
        feature_names = [f for f in feature_names if f not in cat_cols] + encoded

    train_df, val_df, test_df = csv_repo.split(df)
    target = "grade3_neutropenia"
    avail = [f for f in feature_names if f in train_df.columns]

    x_train = train_df[avail].values.astype(np.float32)
    y_train = train_df[target].values
    x_val = val_df[avail].values.astype(np.float32)
    y_val = val_df[target].values
    x_test = test_df[avail].values.astype(np.float32)
    y_test = test_df[target].values

    logger.info(f"Features: {len(avail)}개, Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    logger.info(f"양성률 - Train: {y_train.mean():.1%}, Test: {y_test.mean():.1%}")

    # ── 2. 모델 학습 ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("2. 모델 학습 (XGBoost, LightGBM, LogisticRegression)")
    logger.info("=" * 60)

    models = {
        "XGBoost": XGBoostModel,
        "LightGBM": LightGBMModel,
        "LogisticRegression": LogisticModel,
    }

    trainer = TrainPrediction(n_folds=5, seed=42)
    evaluator = EvaluatePrediction()
    results = {}

    for name, model_cls in models.items():
        logger.info(f"\n--- {name} ---")
        res = trainer.train_final(model_cls, x_train, y_train, x_test, y_test, x_val, y_val)
        metrics = evaluator.compute_all_metrics(y_test, res["test_proba"])
        point, lower, upper = evaluator.bootstrap_ci(y_test, res["test_proba"])
        metrics["auroc_ci_lower"] = lower
        metrics["auroc_ci_upper"] = upper
        res["metrics"] = metrics
        results[name] = res

    # ── 3. 모델 비교 테이블 ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("3. 모델 성능 비교")
    logger.info("=" * 60)

    comparison = []
    for name, res in results.items():
        m = res["metrics"]
        comparison.append({
            "Model": name,
            "AUROC": f"{m['auroc']:.4f}",
            "AUROC 95% CI": f"[{m.get('auroc_ci_lower', 0):.3f}-{m.get('auroc_ci_upper', 0):.3f}]",
            "AUPRC": f"{m['auprc']:.4f}",
            "Sensitivity": f"{m['sensitivity']:.4f}",
            "Specificity": f"{m['specificity']:.4f}",
            "PPV": f"{m['ppv']:.4f}",
            "NPV": f"{m['npv']:.4f}",
            "Brier": f"{m['brier_score']:.4f}",
            "CV AUC": f"{res['cv_results']['mean_auc']:.4f}±{res['cv_results']['std_auc']:.4f}",
        })
    comp_df = pd.DataFrame(comparison)
    print("\n" + comp_df.to_string(index=False))
    comp_df.to_csv(FIGURE_DIR / "model_comparison.csv", index=False)

    colors = {"XGBoost": "#2196F3", "LightGBM": "#4CAF50", "LogisticRegression": "#FF9800"}

    # ── 4. CV AUC Boxplot ─────────────────────────────────────
    logger.info("4. CV AUC Boxplot 생성")

    cv_data = []
    for name, res in results.items():
        for auc_val in res["cv_results"]["fold_aucs"]:
            cv_data.append({"Model": name, "AUC": auc_val})
    cv_df = pd.DataFrame(cv_data)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=cv_df, x="Model", y="AUC", hue="Model", palette=colors, ax=ax, legend=False)
    sns.stripplot(data=cv_df, x="Model", y="AUC", color="black", size=6, ax=ax)
    for name, res in results.items():
        i = list(results.keys()).index(name)
        m_auc = res["cv_results"]["mean_auc"]
        s_auc = res["cv_results"]["std_auc"]
        ax.text(i, m_auc + 0.01, f"{m_auc:.3f}±{s_auc:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("5-Fold Cross Validation AUC", fontsize=14)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cv_boxplot.png", dpi=150)
    plt.close()

    # ── 5. ROC Curves ──────────────────────────────────────────
    logger.info("5. ROC Curves 생성")
    from sklearn.metrics import roc_curve, auc as sk_auc

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["test_proba"])
        auroc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} (AUC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Grade 3+ Neutropenia Prediction", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curves.png", dpi=150)
    plt.close()

    # ── 5. PR Curves ──────────────────────────────────────────
    logger.info("6. Precision-Recall Curves 생성")
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["test_proba"])
        ap = average_precision_score(y_test, res["test_proba"])
        ax.plot(rec, prec, color=colors[name], lw=2, label=f"{name} (AP={ap:.3f})")
    ax.axhline(y=y_test.mean(), color="gray", ls="--", alpha=0.5, label=f"Prevalence ({y_test.mean():.2f})")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "pr_curves.png", dpi=150)
    plt.close()

    # ── 6. Confusion Matrices ──────────────────────────────────
    logger.info("7. Confusion Matrices 생성")
    from sklearn.metrics import confusion_matrix as cm_func

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (name, res) in enumerate(results.items()):
        m = res["metrics"]
        thr = m["optimal_threshold"]
        y_pred = (res["test_proba"] >= thr).astype(int)
        cm = cm_func(y_test, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        axes[idx].set_title(f"{name}\n(threshold={thr:.3f})", fontsize=12)
        axes[idx].set_ylabel("Actual")
        axes[idx].set_xlabel("Predicted")
    plt.suptitle("Confusion Matrices (Youden Threshold)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 7. Calibration Curves ──────────────────────────────────
    logger.info("8. Calibration Curves 생성")

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    for name, res in results.items():
        cal = evaluator.compute_calibration(y_test, res["test_proba"], n_bins=5)
        ax.plot(cal["mean_predicted_value"], cal["fraction_of_positives"],
                "o-", color=colors[name], lw=2, label=f"{name} (ECE={cal['ece']:.3f})")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "calibration_curves.png", dpi=150)
    plt.close()

    # ── 8. Feature Importance (XGBoost) ────────────────────────
    logger.info("9. Feature Importance 생성")

    xgb_model = results["XGBoost"]["model"]
    imp = xgb_model.get_feature_importance(avail)
    top_20 = list(imp.items())[:20]
    names_top = [x[0] for x in top_20][::-1]
    vals_top = [x[1] for x in top_20][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.barh(names_top, vals_top, color="#2196F3", alpha=0.8)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("XGBoost Feature Importance (Top 20)", fontsize=14)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_importance_xgboost.png", dpi=150)
    plt.close()

    # ── 9. SHAP Analysis (XGBoost) ────────────────────────────
    logger.info("10. SHAP Analysis 생성")

    shap_analyzer = RunSHAP(xgb_model.model, x_test, avail)
    shap_analyzer.plot_summary(save_path=str(FIGURE_DIR / "shap_summary.png"))
    shap_analyzer.plot_summary(save_path=str(FIGURE_DIR / "shap_bar.png"), plot_type="bar")
    top_shap = shap_analyzer.get_top_features(10)
    logger.info("SHAP Top 10 Features:")
    for fname, val in top_shap:
        logger.info(f"  {fname}: {val:.4f}")

    # ── 10. Clinical Threshold (Screening) ─────────────────────
    logger.info("11. 임상 임계값 분석")

    xgb_proba = results["XGBoost"]["test_proba"]
    thr, sens, spec = FindThreshold.execute(y_test, xgb_proba, min_sensitivity=0.85)

    screener = EvaluateScreening()
    screen_metrics = screener.compute_screening_metrics(y_test, xgb_proba, thr)
    print(f"\n임상 임계값 (Sensitivity ≥ 0.85):")
    print(f"  Threshold: {thr:.3f}")
    print(f"  Sensitivity: {screen_metrics['sensitivity']:.3f}")
    print(f"  Specificity: {screen_metrics['specificity']:.3f}")
    print(f"  PPV: {screen_metrics['ppv']:.3f}")
    print(f"  NPV: {screen_metrics['npv']:.3f}")
    print(f"  스크리닝 양성: {screen_metrics['n_screened_positive']}/{screen_metrics['n_total']}")

    # ── 11. DCA (Decision Curve Analysis) ──────────────────────
    logger.info("12. Decision Curve Analysis 생성")

    dca = screener.compute_dca(y_test, xgb_proba)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(dca["thresholds"], dca["net_benefit_model"], color="#2196F3", lw=2, label="XGBoost Model")
    ax.plot(dca["thresholds"], dca["net_benefit_treat_all"], color="gray", lw=1.5, ls="--", label="Treat All")
    ax.axhline(y=0, color="black", lw=1, ls=":", label="Treat None")
    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title("Decision Curve Analysis", fontsize=14)
    ax.set_xlim(0, 0.8)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "dca_curve.png", dpi=150)
    plt.close()

    # ── 12. Model Comparison Bar Chart ─────────────────────────
    logger.info("13. 모델 비교 Bar Chart 생성")

    metric_keys = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv"]
    x_pos = np.arange(len(metric_keys))
    width = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i, (name, res) in enumerate(results.items()):
        vals = [res["metrics"].get(k, 0) for k in metric_keys]
        ax.bar(x_pos + i * width, vals, width, label=name, color=list(colors.values())[i], alpha=0.85)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([k.upper() for k in metric_keys], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for i, (name, res) in enumerate(results.items()):
        vals = [res["metrics"].get(k, 0) for k in metric_keys]
        for j, v in enumerate(vals):
            ax.text(x_pos[j] + i * width, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison_bar.png", dpi=150)
    plt.close()

    # ── 13. Feature Set Ablation Study (A/B/C) ─────────────────
    logger.info("=" * 60)
    logger.info("14. Feature Set Ablation Study (AMC→ANC lead-lag hypothesis)")
    logger.info("=" * 60)

    from prediction.application.use_cases.compare_feature_sets import CompareFeatureSets

    # 원본 df에서 train/val/test 인덱스 복원
    train_idx = train_df.index.values
    val_idx = val_df.index.values
    test_idx = test_df.index.values

    comparator = CompareFeatureSets(n_folds=5, seed=42)
    ablation_results = comparator.execute(
        df=df,
        model_class=XGBoostModel,
        target=target,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    ablation_df = comparator.to_dataframe(ablation_results)
    print("\n" + ablation_df.to_string(index=False))
    ablation_df.to_csv(FIGURE_DIR / "feature_set_comparison.csv", index=False)

    # Incremental value logging
    for step_key, step_label in [("A_to_B", "A→B (ANC 추가)"), ("B_to_C", "B→C (ALC+PLT 추가)")]:
        if step_key in ablation_results:
            inc = ablation_results[step_key]
            logger.info(f"  {step_label}:")
            for k in ["auroc", "sensitivity"]:
                delta = inc.get(f"{k}_delta", 0)
                logger.info(f"    {k}: {inc.get(f'{k}_baseline', 0):.4f} → {inc.get(f'{k}_enhanced', 0):.4f} (Δ={delta:+.4f})")

    # Ablation comparison bar chart
    logger.info("  Feature Set Comparison Chart 생성")
    set_names = []
    set_metrics = {k: [] for k in ["auroc", "auprc", "sensitivity", "npv"]}
    for sn in ["A", "B", "C"]:
        if sn not in ablation_results:
            continue
        r = ablation_results[sn]
        set_names.append(f"{sn}\n({r['label']})")
        for k in set_metrics:
            set_metrics[k].append(r["metrics"].get(k, 0))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x_pos = np.arange(len(set_names))
    width = 0.2
    metric_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, (metric_key, color) in enumerate(zip(set_metrics.keys(), metric_colors)):
        vals = set_metrics[metric_key]
        bars = ax.bar(x_pos + i * width, vals, width, label=metric_key.upper(), color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(set_names, fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Feature Set Ablation: AMC→ANC Lead-Lag Hypothesis", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_set_comparison.png", dpi=150)
    plt.close()

    # ── 완료 ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("전체 파이프라인 완료!")
    logger.info(f"생성된 그래프: {FIGURE_DIR}")
    figs = list(FIGURE_DIR.glob("*.png"))
    for f in sorted(figs):
        logger.info(f"  {f.name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
