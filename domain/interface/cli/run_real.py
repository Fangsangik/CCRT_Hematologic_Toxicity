"""실제 데이터 파이프라인 - pseudo_dataset.xlsx로 전체 평가"""
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

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    from prediction.domain import FeatureService
    from prediction.application import TrainPrediction, EvaluatePrediction, RunSHAP
    from screening.application import FindThreshold, EvaluateScreening
    from shared.infrastructure.ml.xgboost_model import XGBoostModel
    from shared.infrastructure.ml.lightgbm_model import LightGBMModel
    from shared.infrastructure.ml.logistic_model import LogisticModel
    from shared.infrastructure.repository.csv_repository import CSVRepository
    from shared.infrastructure.repository.excel_repository import ExcelRepository

    # ── 1. 데이터 로드 & 전처리 ────────────────────────────────
    logger.info("=" * 60)
    logger.info("1. 데이터 로드 (pseudo_dataset.xlsx)")
    logger.info("=" * 60)

    excel_repo = ExcelRepository()
    import sys as _sys
    if len(_sys.argv) > 1:
        _data_path = _sys.argv[1]
    else:
        _data_path = str(PROJECT_ROOT / "data" / "pseudo_dataset.xlsx")
    df = excel_repo.load(_data_path)

    print(f"\n데이터 요약:")
    print(f"  환자 수: {len(df)}")
    print(f"  양성(Grade3+): {df['grade3_neutropenia'].sum()} ({df['grade3_neutropenia'].mean():.1%})")
    print(f"  결측치: {df.isnull().sum().sum()}개")
    print(f"  결측 컬럼: {dict(df.isnull().sum()[df.isnull().sum() > 0])}")

    # 결측치 처리
    csv_repo = CSVRepository()
    df = csv_repo.handle_missing(df)

    # ── 2. Feature 추출 ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("2. Feature 추출 (baseline_cbc mode)")
    logger.info("=" * 60)

    feature_service = FeatureService()
    df, feature_names = feature_service.extract_all(df, mode="baseline_cbc")

    # 범주형 인코딩
    cat_cols = [c for c in ["sex", "stage", "t_stage", "n_stage", "ecog_ps", "chemo_regimen"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        encoded = [c for c in df.columns if any(c.startswith(f"{cat}_") for cat in cat_cols)]
        feature_names = [f for f in feature_names if f not in cat_cols] + encoded

    target = "grade3_neutropenia"
    avail = [f for f in feature_names if f in df.columns]
    print(f"\n사용 Feature: {len(avail)}개")

    # ── 3. 데이터 분할 ──────────────────────────────────────
    train_df, val_df, test_df = csv_repo.split(df, target_col=target)

    x_train = train_df[avail].values.astype(np.float32)
    y_train = train_df[target].values
    x_val = val_df[avail].values.astype(np.float32)
    y_val = val_df[target].values
    x_test = test_df[avail].values.astype(np.float32)
    y_test = test_df[target].values

    # 전체 데이터 (CV용)
    x_all = df[avail].values.astype(np.float32)
    y_all = df[target].values

    print(f"  Train: {len(y_train)} (양성 {y_train.mean():.1%})")
    print(f"  Val: {len(y_val)} (양성 {y_val.mean():.1%})")
    print(f"  Test: {len(y_test)} (양성 {y_test.mean():.1%})")

    # ── 4. 5-Fold CV + 최종 모델 학습 ────────────────────────
    logger.info("=" * 60)
    logger.info("3. 5-Fold Cross Validation + 모델 학습")
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
        logger.info(f"\n{'='*40}")
        logger.info(f"  {name}")
        logger.info(f"{'='*40}")

        # 5-fold CV on full data
        cv_results = trainer.cross_validate(
            model_cls, x_all, y_all, feature_names=avail,
        )

        # Final model: train on train+val, test on test
        x_trainval = np.vstack([x_train, x_val])
        y_trainval = np.concatenate([y_train, y_val])

        model = model_cls()
        model.fit(x_trainval, y_trainval, x_val, y_val)
        test_proba = model.predict_proba(x_test)

        from sklearn.metrics import roc_auc_score
        test_auc = roc_auc_score(y_test, test_proba)

        # 전체 평가 지표
        metrics = evaluator.compute_all_metrics(y_test, test_proba)
        point, lower, upper = evaluator.bootstrap_ci(y_test, test_proba)
        metrics["auroc_ci_lower"] = lower
        metrics["auroc_ci_upper"] = upper

        results[name] = {
            "model": model,
            "cv_results": cv_results,
            "test_proba": test_proba,
            "test_auc": test_auc,
            "metrics": metrics,
        }

        print(f"\n  {name} 결과:")
        print(f"    5-Fold CV AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        print(f"    Fold별 AUC: {[f'{a:.4f}' for a in cv_results['fold_aucs']]}")
        print(f"    Test AUC: {test_auc:.4f} [{lower:.3f}-{upper:.3f}]")
        print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"    Specificity: {metrics['specificity']:.4f}")
        print(f"    PPV: {metrics['ppv']:.4f}, NPV: {metrics['npv']:.4f}")
        print(f"    Brier Score: {metrics['brier_score']:.4f}")

    # ── 5. 모델 비교 테이블 ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("4. 모델 성능 비교")
    logger.info("=" * 60)

    rows = []
    for name, res in results.items():
        m = res["metrics"]
        cv = res["cv_results"]
        rows.append({
            "Model": name,
            "CV AUC (mean±std)": f"{cv['mean_auc']:.4f}±{cv['std_auc']:.4f}",
            "Test AUROC": f"{m['auroc']:.4f}",
            "95% CI": f"[{m.get('auroc_ci_lower',0):.3f}-{m.get('auroc_ci_upper',0):.3f}]",
            "AUPRC": f"{m['auprc']:.4f}",
            "Sensitivity": f"{m['sensitivity']:.4f}",
            "Specificity": f"{m['specificity']:.4f}",
            "PPV": f"{m['ppv']:.4f}",
            "NPV": f"{m['npv']:.4f}",
            "F1": f"{m['f1_score']:.4f}",
            "Brier": f"{m['brier_score']:.4f}",
        })
    comp_df = pd.DataFrame(rows)
    print("\n" + comp_df.to_string(index=False))
    comp_df.to_csv(RESULTS_DIR / "model_comparison_real.csv", index=False)

    # ── 6. 시각화 ────────────────────────────────────────────
    from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve, average_precision_score
    from sklearn.metrics import confusion_matrix as cm_func

    colors = {"XGBoost": "#2196F3", "LightGBM": "#4CAF50", "LogisticRegression": "#FF9800"}

    # 6-1. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["test_proba"])
        ax.plot(fpr, tpr, color=colors[name], lw=2,
                label=f"{name} (AUC={res['metrics']['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - pseudo_dataset.xlsx", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "real_roc_curves.png", dpi=150)
    plt.close()

    # 6-2. PR Curves
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["test_proba"])
        ap = average_precision_score(y_test, res["test_proba"])
        ax.plot(rec, prec, color=colors[name], lw=2, label=f"{name} (AP={ap:.3f})")
    ax.axhline(y=y_test.mean(), color="gray", ls="--", alpha=0.5, label=f"Prevalence ({y_test.mean():.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "real_pr_curves.png", dpi=150)
    plt.close()

    # 6-3. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (name, res) in enumerate(results.items()):
        thr = res["metrics"]["optimal_threshold"]
        y_pred = (res["test_proba"] >= thr).astype(int)
        cm = cm_func(y_test, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        axes[idx].set_title(f"{name}\n(thr={thr:.3f})", fontsize=12)
        axes[idx].set_ylabel("Actual")
        axes[idx].set_xlabel("Predicted")
    plt.suptitle("Confusion Matrices (Youden Threshold)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "real_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6-4. CV AUC Box Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    cv_data = []
    for name, res in results.items():
        for auc_val in res["cv_results"]["fold_aucs"]:
            cv_data.append({"Model": name, "AUC": auc_val})
    cv_df = pd.DataFrame(cv_data)
    sns.boxplot(data=cv_df, x="Model", y="AUC", palette=colors, ax=ax)
    sns.stripplot(data=cv_df, x="Model", y="AUC", color="black", size=6, ax=ax)
    ax.set_title("5-Fold Cross Validation AUC Distribution", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    for i, (name, res) in enumerate(results.items()):
        cv = res["cv_results"]
        ax.text(i, cv["mean_auc"] + 0.03, f"{cv['mean_auc']:.3f}±{cv['std_auc']:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "real_cv_boxplot.png", dpi=150)
    plt.close()

    # 6-5. Calibration Curves
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
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
    plt.savefig(FIGURE_DIR / "real_calibration.png", dpi=150)
    plt.close()

    # 6-6. Feature Importance (XGBoost)
    xgb_model = results["XGBoost"]["model"]
    imp = xgb_model.get_feature_importance(avail)
    top_20 = list(imp.items())[:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([x[0] for x in top_20][::-1], [x[1] for x in top_20][::-1], color="#2196F3", alpha=0.8)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("XGBoost Feature Importance (Top 20)", fontsize=14)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "real_feature_importance.png", dpi=150)
    plt.close()

    # 6-7. SHAP
    shap_analyzer = RunSHAP(xgb_model.model, x_test, avail)
    shap_analyzer.plot_summary(save_path=str(FIGURE_DIR / "real_shap_summary.png"))
    shap_analyzer.plot_summary(save_path=str(FIGURE_DIR / "real_shap_bar.png"), plot_type="bar")
    top_shap = shap_analyzer.get_top_features(10)

    print("\nSHAP Top 10 Features:")
    for fname, val in top_shap:
        print(f"  {fname}: {val:.4f}")

    # 6-8. DCA
    xgb_proba = results["XGBoost"]["test_proba"]
    screener = EvaluateScreening()
    dca = screener.compute_dca(y_test, xgb_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dca["thresholds"], dca["net_benefit_model"], color="#2196F3", lw=2, label="XGBoost")
    ax.plot(dca["thresholds"], dca["net_benefit_treat_all"], color="gray", lw=1.5, ls="--", label="Treat All")
    ax.axhline(y=0, color="black", lw=1, ls=":", label="Treat None")
    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title("Decision Curve Analysis", fontsize=14)
    ax.set_xlim(0, 0.8)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "real_dca.png", dpi=150)
    plt.close()

    # 6-9. Model Comparison Bar Chart
    metric_keys = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv"]
    x_pos = np.arange(len(metric_keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, res) in enumerate(results.items()):
        vals = [res["metrics"].get(k, 0) for k in metric_keys]
        ax.bar(x_pos + i * width, vals, width, label=name, color=list(colors.values())[i], alpha=0.85)
        for j, v in enumerate(vals):
            ax.text(x_pos[j] + i * width, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([k.upper() for k in metric_keys], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "real_model_comparison.png", dpi=150)
    plt.close()

    # ── 7. 임상 임계값 ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("5. 임상 임계값 분석 (Sensitivity ≥ 0.85)")
    logger.info("=" * 60)

    for name, res in results.items():
        thr, sens, spec = FindThreshold.execute(y_test, res["test_proba"], min_sensitivity=0.85)
        screen = screener.compute_screening_metrics(y_test, res["test_proba"], thr)
        print(f"\n  {name}:")
        print(f"    Threshold: {thr:.3f}")
        print(f"    Sensitivity: {sens:.3f}, Specificity: {spec:.3f}")
        print(f"    PPV: {screen['ppv']:.3f}, NPV: {screen['npv']:.3f}")
        print(f"    스크리닝 양성: {screen['n_screened_positive']}/{screen['n_total']}")

    # ── 완료 ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("전체 파이프라인 완료!")
    figs = sorted(FIGURE_DIR.glob("real_*.png"))
    logger.info(f"생성된 그래프 ({len(figs)}개):")
    for f in figs:
        logger.info(f"  {f.name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
