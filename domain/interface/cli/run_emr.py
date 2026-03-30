"""Raw EMR Excel → 전처리 → 5-Fold CV 파이프라인

pseudo_EMR_raw.xlsx 형식:
  - 시트1 '임상정보': 환자번호, 나이, 성별, BMI, ECOG_PS, Stage, T, N, Creatinine, Albumin ...
  - 시트2 'CBC검사결과': long format (환자번호, 검사주차, 검사명, 결과값, 단위 ...)
  - 시트3 '치료정보': 환자번호, 항암요법, 용량, 주기 ...
"""
import logging
import re
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "pseudo_EMR_week0to6.xlsx"

# ─── 검사명 → 표준 CBC 항목 매핑 ───────────────────────────────
TEST_NAME_MAP = {
    # WBC
    "WBC": "WBC", "WBC count": "WBC", "WBC(자동)": "WBC",
    "White Blood Cell": "WBC", "Leukocyte": "WBC",
    "백혈구": "WBC", "총백혈구수": "WBC",
    # ANC
    "ANC": "ANC", "Neut#": "ANC", "Seg.Neutrophil": "ANC",
    "Neutrophil(abs)": "ANC", "호중구": "ANC",
    "호중구(절대값)": "ANC", "절대호중구수": "ANC",
    # ALC
    "ALC": "ALC", "Lymphocyte": "ALC", "Lymphocyte(abs)": "ALC",
    "Lymph#": "ALC", "림프구": "ALC",
    "절대림프구수": "ALC",
    # AMC
    "AMC": "AMC", "Monocyte": "AMC", "Monocyte(abs)": "AMC",
    "Mono#": "AMC", "단핵구": "AMC",
    "단핵구수": "AMC", "절대단핵구수": "AMC",
    # PLT
    "PLT": "PLT", "PLT count": "PLT", "Plt(자동)": "PLT",
    "Platelet": "PLT", "Thrombocyte": "PLT",
    "혈소판": "PLT", "혈소판수": "PLT",
    # Hb
    "Hb": "Hb", "HGB": "Hb", "Hgb": "Hb",
    "Hemoglobin": "Hb", "Hemoglobin(g/dL)": "Hb",
    "헤모글로빈": "Hb", "혈색소": "Hb",
}

# ─── 검사주차 → 숫자 매핑 ──────────────────────────────────────
def parse_week(raw: str) -> int | None:
    """다양한 주차 표기를 숫자로 변환."""
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    # "week -1" → 제외
    if "-1" in s:
        return None
    # 숫자만 있는 경우: "0"~"7"
    if re.match(r"^\d$", s):
        return int(s)
    # w0, w1, w2
    m = re.match(r"w(\d)", s)
    if m:
        return int(m.group(1))
    # week 0, week 1, week 2, week 3
    m = re.search(r"week\s*(\d)", s)
    if m:
        return int(m.group(1))
    # 0주차, 1주차, 2주차
    m = re.match(r"(\d)주차?", s)
    if m:
        return int(m.group(1))
    # 주0, 주1, 주2
    m = re.match(r"주(\d)", s)
    if m:
        return int(m.group(1))
    # ccrt 0주, ccrt 1주, ccrt 2주
    m = re.search(r"ccrt\s*(\d)", s)
    if m:
        return int(m.group(1))
    return None


def parse_result_value(raw) -> float | None:
    """결과값을 float로 변환. 측정불가, <0.1, >10.0 등 처리."""
    if pd.isna(raw):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if s in ("측정불가", "검체부족", "검체오류", ""):
        return None
    # "<0.1" → 0.05 (하한의 절반)
    m = re.match(r"<\s*([\d.]+)", s)
    if m:
        return float(m.group(1)) / 2.0
    # ">10.0" → 값 그대로 사용
    m = re.match(r">\s*([\d.]+)", s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except ValueError:
        return None


def normalize_unit(value: float, test: str, unit: str) -> float:
    """단위를 ×10⁹/L (또는 g/dL for Hb)로 정규화.

    EMR에서 같은 검사도 단위가 다르게 기록됨:
      - cells/uL, /μL  → ÷1000 (WBC/ANC/ALC/AMC), ÷1000 (PLT)
      - K/uL, 10³/μL, 10^3/uL, ×10⁹/L → 그대로
      - 만/μL → ×10 (1만/μL = 10 ×10⁹/L)
    단위 정보가 없거나 부정확할 때는 값 크기로 휴리스틱 판단.
    """
    if pd.isna(value):
        return value

    # 단위 문자열 정규화
    u = str(unit).strip().lower() if pd.notna(unit) else ""

    # 단위 기반 변환
    is_absolute = any(k in u for k in ["cells/u", "/μl", "/ul", "cells/μ"])
    is_kilo = any(k in u for k in ["k/u", "10³", "10^3", "10^9", "×10", "g/d", "g%", "gm/d"])
    is_man = "만" in u  # 만/μL

    if test == "Hb":
        # Hb는 g/dL 기준. mg/dL이면 ÷1000
        if "mg" in u:
            return value / 1000.0
        # 값이 비정상적으로 크면 단위 오류
        if value > 30:
            return value / 10.0
        return value

    if test == "PLT":
        # PLT ×10⁹/L: 정상 150-400
        if is_absolute and value > 1000:
            return value / 1000.0
        if is_man:
            return value * 10.0
        # 휴리스틱: 값이 1000 이상이면 cells/uL로 간주
        if value > 1000:
            return value / 1000.0
        return value

    # WBC, ANC, ALC, AMC: ×10⁹/L 기준 (정상 범위 ~0.1-15)
    if is_absolute:
        return value / 1000.0
    if is_man:
        return value * 10.0
    # 휴리스틱: 값이 50 이상이면 cells/uL로 간주
    if value > 50:
        return value / 1000.0
    return value


def clean_sex(raw) -> int | None:
    """성별 → 0=F, 1=M."""
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    if s in ("male", "m", "남", "남성", "1"):
        return 1
    if s in ("female", "f", "여", "여성", "0"):
        return 0
    return None


def clean_ecog(raw) -> int | None:
    """ECOG PS → 0~4 정수."""
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    m = re.search(r"(\d)", s)
    if m:
        return int(m.group(1))
    return None


def clean_stage(raw) -> int | None:
    """Stage → 1~4 정수."""
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    stage_map = {
        "i": 1, "1": 1,
        "ii": 2, "2": 2,
        "iii": 3, "3": 3, "iiia": 3, "iiib": 3,
        "iv": 4, "4": 4,
    }
    # "stage3", "4기" 등에서 숫자 추출
    cleaned = re.sub(r"(stage|기|a|b)", "", s).strip()
    if cleaned in stage_map:
        return stage_map[cleaned]
    # 로마 숫자 직접 매핑
    for k, v in stage_map.items():
        if k in s:
            return v
    return None


def clean_t_stage(raw) -> int | None:
    """T stage → 1~4 정수."""
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    m = re.search(r"(\d)", s)
    if m:
        return int(m.group(1))
    return None


def clean_n_stage(raw) -> int | None:
    """N stage → 0~3 정수."""
    if pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    m = re.search(r"(\d)", s)
    if m:
        return int(m.group(1))
    return None


def safe_float(raw) -> float | None:
    """안전한 float 변환."""
    if pd.isna(raw):
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


# ─── 메인 전처리 ──────────────────────────────────────────────
def preprocess_emr(path: Path) -> pd.DataFrame:
    """3개 시트를 읽어 wide format DataFrame으로 변환."""
    xls = pd.ExcelFile(path)

    # ── 1) 임상정보 ──
    logger.info("임상정보 시트 처리...")
    clin = pd.read_excel(xls, sheet_name="임상정보")
    df_clin = pd.DataFrame()
    df_clin["patient_id"] = clin["환자번호"]
    df_clin["age"] = clin["나이"].apply(safe_float)
    df_clin["sex"] = clin["성별"].apply(clean_sex)
    df_clin["bmi"] = clin["BMI"].apply(safe_float)
    df_clin["ecog_ps"] = clin["ECOG_PS"].apply(clean_ecog)
    df_clin["stage"] = clin["Stage"].apply(clean_stage)
    df_clin["t_stage"] = clin["T"].apply(clean_t_stage)
    df_clin["n_stage"] = clin["N"].apply(clean_n_stage)
    df_clin["creatinine"] = clin["Creatinine"].apply(safe_float)
    df_clin["albumin"] = clin["Albumin"].apply(safe_float)

    # Grade3+ 라벨 (임상정보 시트에 있으면 직접 사용)
    if "Grade3+발생" in clin.columns:
        df_clin["grade3_neutropenia"] = clin["Grade3+발생"].fillna(0).astype(int)
        logger.info(f"  Grade3+발생 라벨 사용: 양성 {df_clin['grade3_neutropenia'].sum()}/{len(df_clin)} "
                    f"({df_clin['grade3_neutropenia'].mean():.1%})")

    logger.info(f"  임상정보: {len(df_clin)}명")
    logger.info(f"  결측: {df_clin.isnull().sum()[df_clin.isnull().sum() > 0].to_dict()}")

    # ── 2) CBC 검사결과 → wide format ──
    logger.info("CBC검사결과 시트 처리...")
    cbc = pd.read_excel(xls, sheet_name="CBC검사결과")

    cbc["week"] = cbc["검사주차"].apply(parse_week)
    cbc["test"] = cbc["검사명"].map(TEST_NAME_MAP)
    cbc["raw_value"] = cbc["결과값"].apply(parse_result_value)

    # 주차/검사명 매핑 실패 로그
    unmapped_tests = cbc[cbc["test"].isna()]["검사명"].unique()
    if len(unmapped_tests) > 0:
        logger.warning(f"  매핑 실패 검사명: {unmapped_tests}")
    unmapped_weeks = cbc[cbc["week"].isna()]["검사주차"].unique()
    if len(unmapped_weeks) > 0:
        logger.info(f"  제외된 주차: {unmapped_weeks}")

    # 유효한 데이터만 필터 (week 0~3, 표준 검사명)
    cbc_valid = cbc.dropna(subset=["week", "test", "raw_value"]).copy()
    cbc_valid = cbc_valid[cbc_valid["week"].isin([0, 1, 2, 3, 4, 5, 6, 7])]

    # 단위 정규화: ×10⁹/L (또는 g/dL for Hb)로 통일
    cbc_valid["value"] = cbc_valid.apply(
        lambda r: normalize_unit(r["raw_value"], r["test"], r.get("단위")), axis=1
    )
    logger.info(f"  유효 CBC 레코드: {len(cbc_valid)}/{len(cbc)}")

    # 단위 정규화 검증
    for t in ["WBC", "ANC", "ALC", "AMC", "PLT", "Hb"]:
        vals = cbc_valid[cbc_valid["test"] == t]["value"]
        if len(vals) > 0:
            logger.info(f"  {t}: n={len(vals)}, mean={vals.mean():.2f}, "
                        f"min={vals.min():.2f}, max={vals.max():.2f}")

    # 환자×주차×검사별 중복 → 중앙값 사용
    cbc_agg = cbc_valid.groupby(["환자번호", "week", "test"])["value"].median().reset_index()

    # pivot: 환자번호 × (test_weekN)
    cbc_agg["col_name"] = cbc_agg["test"] + "_week" + cbc_agg["week"].astype(int).astype(str)
    df_cbc = cbc_agg.pivot_table(index="환자번호", columns="col_name", values="value").reset_index()
    df_cbc.rename(columns={"환자번호": "patient_id"}, inplace=True)

    logger.info(f"  CBC wide: {df_cbc.shape}")

    # ── 3) 치료정보 ──
    logger.info("치료정보 시트 처리...")
    tx = pd.read_excel(xls, sheet_name="치료정보")
    df_tx = pd.DataFrame()
    df_tx["patient_id"] = tx["환자번호"]
    df_tx["chemo_regimen"] = tx["항암요법"].str.lower().str.strip()
    df_tx["chemo_dose"] = tx["용량(mg/m²)"].apply(safe_float)
    df_tx["chemo_cycles"] = tx["실제투여횟수"].apply(safe_float)

    logger.info(f"  치료정보: {len(df_tx)}명, 항암요법: {df_tx['chemo_regimen'].unique()}")

    # ── 4) 병합 ──
    logger.info("데이터 병합...")
    df = df_clin.merge(df_tx, on="patient_id", how="left")
    df = df.merge(df_cbc, on="patient_id", how="left")

    # ── 5) 라벨: Week 3~6 ANC nadir < 1.0 ×10⁹/L → Grade 3+ Neutropenia ──
    logger.info("라벨 생성: Week 3~6 ANC nadir < 1.0 ×10⁹/L (CTCAE v5.0 Grade 3+)...")
    anc_late_cols = [c for c in df.columns if c.startswith("ANC_week")
                     and int(c.split("week")[1]) >= 3]
    if anc_late_cols:
        anc_nadir = df[anc_late_cols].min(axis=1)
        df["grade3_neutropenia"] = (anc_nadir < 1.0).astype(int)
        logger.info(f"  사용 컬럼: {anc_late_cols}")
        logger.info(f"  ANC nadir (w3-6) 분포: mean={anc_nadir.mean():.2f}, "
                    f"min={anc_nadir.min():.2f}, <1.0: {(anc_nadir < 1.0).sum()}")
    else:
        # fallback: 임상정보 시트의 Grade3+발생
        if "grade3_neutropenia" not in df.columns:
            logger.warning("  Week 3~6 ANC 데이터 없음, Grade3+발생 컬럼도 없음 → 라벨 0 처리")
            df["grade3_neutropenia"] = 0

    pos_rate = df["grade3_neutropenia"].mean()
    logger.info(f"  양성률: {df['grade3_neutropenia'].sum()}/{len(df)} ({pos_rate:.1%})")

    # week 3~7 컬럼 제거 (모델 입력에는 week 0~2만 사용, 나머지는 data leakage)
    leak_cols = [c for c in df.columns if any(f"week{w}" in c for w in range(3, 8))]
    if leak_cols:
        df.drop(columns=leak_cols, inplace=True)
        logger.info(f"  week3~7 컬럼 제거 ({len(leak_cols)}개): {leak_cols[:5]}...")

    # ── 6) 결측치 처리 ──
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    logger.info(f"최종 데이터: {df.shape}")
    return df


def main():
    from prediction.domain import FeatureService
    from prediction.application.use_cases.train_prediction import TrainPrediction
    from prediction.application.use_cases.evaluate_prediction import EvaluatePrediction
    from prediction.application.use_cases.run_shap import RunSHAP
    from screening.application.use_cases.find_threshold import FindThreshold
    from screening.application.use_cases.evaluate_screening import EvaluateScreening
    from shared.infrastructure.ml.xgboost_model import XGBoostModel
    from shared.infrastructure.ml.lightgbm_model import LightGBMModel
    from shared.infrastructure.ml.logistic_model import LogisticModel

    # ── 1. EMR 전처리 ──
    logger.info("=" * 60)
    logger.info("1. EMR 데이터 전처리")
    logger.info("=" * 60)
    df = preprocess_emr(DATA_PATH)

    # ── 2. Feature 추출 ──
    logger.info("=" * 60)
    logger.info("2. Feature 추출")
    logger.info("=" * 60)
    feature_service = FeatureService()
    df, feature_names = feature_service.extract_all(df, mode="baseline_cbc")

    # 범주형 인코딩
    cat_cols = [c for c in ["sex", "stage", "t_stage", "n_stage", "ecog_ps", "chemo_regimen"]
                if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        encoded = [c for c in df.columns if any(c.startswith(f"{cat}_") for cat in cat_cols)]
        feature_names = [f for f in feature_names if f not in cat_cols] + encoded

    target = "grade3_neutropenia"
    avail = [f for f in feature_names if f in df.columns]
    logger.info(f"Feature 수: {len(avail)}")

    # ── 3. Train/Val/Test 분할 ──
    from sklearn.model_selection import train_test_split
    train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df[target], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.19, stratify=train_val_df[target], random_state=42)

    x_train = train_df[avail].values.astype(np.float32)
    y_train = train_df[target].values
    x_val = val_df[avail].values.astype(np.float32)
    y_val = val_df[target].values
    x_test = test_df[avail].values.astype(np.float32)
    y_test = test_df[target].values

    logger.info(f"Train: {len(y_train)} (양성 {y_train.mean():.1%}), "
                f"Val: {len(y_val)} (양성 {y_val.mean():.1%}), "
                f"Test: {len(y_test)} (양성 {y_test.mean():.1%})")

    # ── 4. 5-Fold CV + 모델 학습 ──
    logger.info("=" * 60)
    logger.info("3. 5-Fold Cross Validation + 모델 학습")
    logger.info("=" * 60)

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    models = {
        "XGBoost": XGBoostModel,
        "LightGBM": LightGBMModel,
        "LogisticRegression": LogisticModel,
    }

    trainer = TrainPrediction(n_folds=5, seed=42)
    evaluator = EvaluatePrediction()
    results = {}

    x_all = df[avail].values.astype(np.float32)
    y_all = df[target].values

    for name, model_cls in models.items():
        logger.info(f"\n{'=' * 40}")
        logger.info(f"  {name}")
        logger.info(f"{'=' * 40}")

        # 5-fold CV on ALL data
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(x_all, y_all), 1):
            m = model_cls()
            m.fit(x_all[tr_idx], y_all[tr_idx], x_all[va_idx], y_all[va_idx])
            proba = m.predict_proba(x_all[va_idx])
            auc = roc_auc_score(y_all[va_idx], proba)
            fold_aucs.append(auc)
            logger.info(f"  Fold {fold_i}/5: AUC = {auc:.4f}")

        cv_mean = np.mean(fold_aucs)
        cv_std = np.std(fold_aucs)
        logger.info(f"  CV 결과: AUC = {cv_mean:.4f} ± {cv_std:.4f}")

        # Final model on train+val → test
        res = trainer.train_final(model_cls, x_train, y_train, x_test, y_test, x_val, y_val)
        metrics = evaluator.compute_all_metrics(y_test, res["test_proba"])
        point, lower, upper = evaluator.bootstrap_ci(y_test, res["test_proba"])
        metrics["auroc_ci_lower"] = lower
        metrics["auroc_ci_upper"] = upper

        results[name] = {
            **res,
            "metrics": metrics,
            "cv_fold_aucs": fold_aucs,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }

    # ── 5. 결과 출력 ──
    logger.info("=" * 60)
    logger.info("4. 모델 성능 비교")
    logger.info("=" * 60)

    comparison = []
    for name, res in results.items():
        m = res["metrics"]
        comparison.append({
            "Model": name,
            "CV AUC (mean±std)": f"{res['cv_mean']:.4f}±{res['cv_std']:.4f}",
            "Test AUROC": f"{m['auroc']:.4f}",
            "95% CI": f"[{m.get('auroc_ci_lower', 0):.3f}-{m.get('auroc_ci_upper', 0):.3f}]",
            "AUPRC": f"{m['auprc']:.4f}",
            "Sensitivity": f"{m['sensitivity']:.4f}",
            "Specificity": f"{m['specificity']:.4f}",
            "PPV": f"{m['ppv']:.4f}",
            "NPV": f"{m['npv']:.4f}",
            "F1": f"{2 * m['sensitivity'] * m['ppv'] / max(m['sensitivity'] + m['ppv'], 1e-8):.4f}",
            "Brier": f"{m['brier_score']:.4f}",
        })
    comp_df = pd.DataFrame(comparison)
    print("\n" + comp_df.to_string(index=False))
    comp_df.to_csv(FIGURE_DIR / "emr_model_comparison.csv", index=False)

    # ── 6. 시각화 ──
    colors = {"XGBoost": "#2196F3", "LightGBM": "#4CAF50", "LogisticRegression": "#FF9800"}
    prefix = "emr_"

    # 6-1. CV Boxplot
    cv_data = []
    for name, res in results.items():
        for auc in res["cv_fold_aucs"]:
            cv_data.append({"Model": name, "AUC": auc})
    cv_df = pd.DataFrame(cv_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=cv_df, x="Model", y="AUC", hue="Model", palette=colors, ax=ax, legend=False)
    sns.stripplot(data=cv_df, x="Model", y="AUC", color="black", size=6, ax=ax)
    for name, res in results.items():
        i = list(results.keys()).index(name)
        ax.text(i, res["cv_mean"] + 0.01, f"{res['cv_mean']:.3f}±{res['cv_std']:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("5-Fold Cross Validation AUC (EMR Data)", fontsize=14)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{prefix}cv_boxplot.png", dpi=150)
    plt.close()

    # 6-2. ROC Curves
    from sklearn.metrics import roc_curve, auc as sk_auc
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["test_proba"])
        auroc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} (AUC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (EMR Data)", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{prefix}roc_curves.png", dpi=150)
    plt.close()

    # 6-3. PR Curves
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["test_proba"])
        ap = average_precision_score(y_test, res["test_proba"])
        ax.plot(rec, prec, color=colors[name], lw=2, label=f"{name} (AP={ap:.3f})")
    ax.axhline(y=y_test.mean(), color="gray", ls="--", alpha=0.5, label=f"Prevalence ({y_test.mean():.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves (EMR Data)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{prefix}pr_curves.png", dpi=150)
    plt.close()

    # 6-4. Confusion Matrices
    from sklearn.metrics import confusion_matrix as cm_func
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (name, res) in enumerate(results.items()):
        m = res["metrics"]
        thr = m["optimal_threshold"]
        y_pred = (res["test_proba"] >= thr).astype(int)
        cm = cm_func(y_test, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        axes[idx].set_title(f"{name}\n(thr={thr:.3f})", fontsize=12)
        axes[idx].set_ylabel("Actual")
        axes[idx].set_xlabel("Predicted")
    plt.suptitle("Confusion Matrices (EMR Data)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{prefix}confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6-5. Calibration
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    for name, res in results.items():
        cal = evaluator.compute_calibration(y_test, res["test_proba"], n_bins=5)
        ax.plot(cal["mean_predicted_value"], cal["fraction_of_positives"],
                "o-", color=colors[name], lw=2, label=f"{name} (ECE={cal['ece']:.3f})")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curves (EMR Data)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{prefix}calibration.png", dpi=150)
    plt.close()

    # 6-6. Feature Importance (XGBoost)
    xgb_model = results["XGBoost"]["model"]
    imp = xgb_model.get_feature_importance(avail)
    top_20 = list(imp.items())[:20]
    names_top = [x[0] for x in top_20][::-1]
    vals_top = [x[1] for x in top_20][::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(names_top, vals_top, color="#2196F3", alpha=0.8)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("XGBoost Feature Importance - Top 20 (EMR Data)", fontsize=14)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{prefix}feature_importance.png", dpi=150)
    plt.close()

    # 6-7. SHAP
    shap_analyzer = RunSHAP(xgb_model.model, x_test, avail)
    shap_analyzer.plot_summary(save_path=str(FIGURE_DIR / f"{prefix}shap_summary.png"))
    shap_analyzer.plot_summary(save_path=str(FIGURE_DIR / f"{prefix}shap_bar.png"), plot_type="bar")
    top_shap = shap_analyzer.get_top_features(10)
    logger.info("SHAP Top 10 Features:")
    for fname, val in top_shap:
        logger.info(f"  {fname}: {val:.4f}")

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
    ax.set_title("Decision Curve Analysis (EMR Data)", fontsize=14)
    ax.set_xlim(0, 0.8)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{prefix}dca.png", dpi=150)
    plt.close()

    # 6-9. Model Comparison Bar
    metric_keys = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv"]
    x_pos = np.arange(len(metric_keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, res) in enumerate(results.items()):
        vals = [res["metrics"].get(k, 0) for k in metric_keys]
        ax.bar(x_pos + i * width, vals, width, label=name, color=list(colors.values())[i], alpha=0.85)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([k.upper() for k in metric_keys], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison (EMR Data)", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for i, (name, res) in enumerate(results.items()):
        vals = [res["metrics"].get(k, 0) for k in metric_keys]
        for j, v in enumerate(vals):
            ax.text(x_pos[j] + i * width, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{prefix}model_comparison.png", dpi=150)
    plt.close()

    # ── 7. 임상 임계값 분석 ──
    logger.info("=" * 60)
    logger.info("5. 임상 임계값 분석 (Sensitivity ≥ 0.85)")
    logger.info("=" * 60)
    for name, res in results.items():
        thr, sens, spec = FindThreshold.execute(y_test, res["test_proba"], min_sensitivity=0.85)
        sm = screener.compute_screening_metrics(y_test, res["test_proba"], thr)
        logger.info(f"  {name}: Thr={thr:.3f}, Sens={sens:.3f}, Spec={spec:.3f}, "
                    f"PPV={sm['ppv']:.3f}, NPV={sm['npv']:.3f}, "
                    f"양성={sm['n_screened_positive']}/{sm['n_total']}")

    # ── 완료 ──
    logger.info("=" * 60)
    logger.info("전체 파이프라인 완료!")
    figs = sorted(FIGURE_DIR.glob(f"{prefix}*.png"))
    logger.info(f"생성된 그래프 ({len(figs)}개):")
    for f in figs:
        logger.info(f"  {f.name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
