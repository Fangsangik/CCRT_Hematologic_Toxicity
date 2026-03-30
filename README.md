# CCRT Module

CCRT (Concurrent Chemoradiation Therapy) 중 **CTCAE v6.0 Grade 3+ 호중구감소증**을 조기 예측하는 ML 파이프라인.

0~2주차 CBC 시계열 + 임상 정보로 3~6주차 Grade 3+ Neutropenia 발생을 예측한다.

---

## 실행 방법

```bash
python domain/interface/cli/run_pipeline.py --data /path/to/file.xlsx
```

### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--data` | 입력 Excel 파일 경로 (필수) | - |
| `--output` | 출력 디렉토리 | `./outputs` |
| `--n-folds` | CV fold 수 | `5` |
| `--seed` | 랜덤 시드 | `42` |
| `--no-figures` | 그래프 생성 건너뛰기 | `False` |

### 예시

```bash
# 기본 실행
python domain/interface/cli/run_pipeline.py --data data.xlsx

# 10-Fold CV + 결과를 results/ 에 저장
python domain/interface/cli/run_pipeline.py --data data.xlsx --n-folds 10 --output ./results

# 수치만 빠르게 확인
python domain/interface/cli/run_pipeline.py --data data.xlsx --no-figures
```

### 입력 데이터 형식

Excel 파일에 다음 시트 중 하나 이상 필요:

**방법 1: Feature_Matrix 시트 (권장)**
```
patient_id, label, WBC_w0, ANC_w0, ..., age, sex, bmi, gcsf, ...
```

**방법 2: Raw EMR 시트**
```
임상정보:      환자ID, 성별, 나이, BMI, ECOG_PS, Albumin, Cr, ...
CBC검사결과:   환자ID, 주차, 항목명, 결과값, 단위, ...
치료정보:      환자ID, RT총선량, ...
Label_참고:    환자ID, Label, ...
```

Feature_Matrix가 있으면 우선 사용, 없으면 Raw EMR에서 자동 전처리.

### 출력

```
outputs/
├── results/
│   └── model_comparison.csv
└── figures/
    ├── fig1_roc_curves.png
    ├── fig2_pr_curves.png
    ├── fig3_cv_roc_bands.png
    ├── fig4_cv_boxplot.png
    ├── fig5_confusion_matrices.png
    ├── fig6_calibration.png
    ├── fig7_feature_importance.png
    ├── fig8_shap_summary.png
    ├── fig8b_shap_bar.png
    ├── fig9_dca.png
    ├── fig10_model_comparison.png
    └── fig11_gcsf_subgroup.png
```

---

## 파이프라인 흐름

```
Excel 입력
  │
  ├─ 1. 데이터 로드
  │     Feature_Matrix 시트 우선, 없으면 Raw EMR 자동 전처리
  │     (한글 매핑, 단위 통일, CBC long→wide 피벗)
  │
  ├─ 2. Feature 추출
  │     ├─ Baseline: age, sex, bmi, albumin, creatinine, ECOG, G-CSF, ...
  │     └─ CBC 파생 (0~2주차):
  │          delta, total_delta, pct_change, slope, cv, nadir, mean
  │
  ├─ 3. K-Fold Stratified CV (OOF)
  │     ├─ XGBoost
  │     ├─ LightGBM
  │     └─ Logistic Regression
  │
  ├─ 4. 평가
  │     ├─ AUROC, AUPRC, Sensitivity, Specificity, PPV, NPV, F1
  │     ├─ Bootstrap 95% CI
  │     ├─ Calibration (ECE, Brier)
  │     ├─ DCA (Decision Curve Analysis)
  │     ├─ SHAP Feature Importance
  │     └─ G-CSF Subgroup Analysis
  │
  └─ 5. 그래프 생성 (12개)
```

---

## 검증 전략

현재 `run_pipeline.py`는 **K-Fold Stratified CV + OOF** 방식을 사용.

다른 검증 전략 비교 → [`docs/validation_options.txt`](docs/validation_options.txt)

| Option | 방법 | 적합한 상황 |
|--------|------|------------|
| A | 5-Fold CV (OOF) | 단일 기관, n < 1000 |
| B | 80/20 + CV on Train | n > 1000 |
| C | Nested CV | 하이퍼파라미터 튜닝 엄밀성 |
| D | Repeated K-Fold | Robustness 확인 |
| E | Stratified Group K-Fold | 다기관 연구 |
| F | LOOCV | n < 50 극소 데이터 |
| G | Bootstrap .632+ | 통계적 엄밀성 최우선 |
| H | Temporal Split | 시간 기반, 외부검증 대체 |

---

## CTCAE v6.0 호중구감소증 등급

| Grade | ANC (× 10⁹/L) |
|-------|---------------|
| 0     | >= 1.5        |
| 1     | 1.0 ~ 1.5    |
| 2     | 0.5 ~ 1.0    |
| 3     | 0.1 ~ 0.5    |
| 4     | < 0.1         |

Label 기준: **Grade 3+ (ANC < 0.5)**

---

## 프로젝트 구조

```
CCRT module/
├── domain/
│   ├── interface/cli/
│   │   ├── run_pipeline.py      ← 메인 실행 모듈
│   │   ├── run_all.py
│   │   ├── run_real.py
│   │   ├── run_emr.py
│   │   ├── run_mimic.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── preprocess.py
│   │
│   ├── prediction/
│   │   ├── application/
│   │   │   ├── evaluate_prediction.py
│   │   │   ├── train_prediction.py
│   │   │   └── shap.py
│   │   ├── domain/
│   │   │   ├── feature_service.py
│   │   │   └── label_service.py
│   │   └── dto/
│   │
│   ├── screening/
│   │   └── application/use_case/
│   │       ├── evaluate_screening.py
│   │       ├── find_threshold.py
│   │       └── train_screening.py
│   │
│   └── utils/
│       ├── application/ml/
│       │   ├── xgboost_model.py
│       │   ├── lightgbm_model.py
│       │   └── logisticModel.py
│       ├── domain/
│       ├── repository/
│       │   ├── csv_repository.py
│       │   ├── excel_repository.py
│       │   ├── emr_repository.py
│       │   └── model_repository.py
│       └── label.py
│
├── outputs/
│   ├── figures/
│   └── results/
│
├── docs/
│   ├── README.md
│   └── validation_options.txt
│
└── TroubleShooting.md
```

---

## 의존성

```
Python 3.11+
pandas, numpy, scikit-learn
xgboost, lightgbm
shap, matplotlib, seaborn
pyyaml
```

---

## 참고

- [CTCAE v6.0 (NCI)](https://dctd.cancer.gov/research/ctep-trials/for-sites/adverse-events/ctcae-v6.pdf)
- [TRIPOD Statement](https://www.tripod-statement.org/)
- Troubleshooting 기록: [`TroubleShooting.md`](TroubleShooting.md)
