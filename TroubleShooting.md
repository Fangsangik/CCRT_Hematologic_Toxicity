# CCRT Module — Troubleshooting Log

CCRT (Concurrent Chemoradiation Therapy) 중 Grade 3+ 호중구감소증 예측 모듈의 코드 리뷰 및 수정 기록.

---

## 1. CTCAE v5.0 → v6.0 마이그레이션

### 변경 사항

CTCAE v6.0 (2025년 발표, 2026.01.01 NCI 시행)에서 호중구감소증 등급 기준이 전면 변경됨.

| Grade | v5.0 (이전) | v6.0 (변경 후) |
|-------|------------|--------------|
| 0     | >= 2.0     | >= 1.5       |
| 1     | 1.5 ~ 2.0  | 1.0 ~ 1.5   |
| 2     | 1.0 ~ 1.5  | 0.5 ~ 1.0   |
| 3     | 0.5 ~ 1.0  | 0.1 ~ 0.5   |
| 4     | < 0.5      | < 0.1        |

단위: ANC × 10^9/L

### 수정 파일

- `domain/utils/repository/emr_repository.py` — `CTCAE_NEUTROPENIA` dict + `calculate_ctcae_grade()`
- `domain/screening/domain/anc_value.py` — `ctcae_grade()` 임계값 + docstring
- `domain/utils/label.py` — docstring

### 참고

- v6.0에서 기존 v5.0 Grade 1 (LLN ~ 1.5)은 등급 체계에서 제거됨
- Grade 4가 < 0.5 → < 0.1로 대폭 축소 (Duffy-null phenotype 고려)
- `_ANC_GRADE3_THRESHOLD = 0.5` (label_service.py)

---

## 2. 코드 리뷰 — 버그 수정 (28건)

### CRITICAL (7건)

| # | 이슈 | 수정 | 파일 |
|---|------|------|------|
| C1 | `from turtle import pd` | `import pandas as pd` | 6개 파일 (shap.py, feature_service.py, label_service.py, csv_repository.py, emr_repository.py, excel_repository.py) |
| C2 | `import np` | `import numpy as np` | screening_input.py |
| C3 | `y_train,sum()` (tuple 생성) | `y_train.sum()` | xgboost_model.py |
| C4 | `f1_score`, `fn`, `clinical_sens`, `clinical_spec` 미정의 | import 추가 + 변수명 수정 | evaluate_prediction.py |
| C5 | `confusion_matrix(label=...)` | `labels=` (복수형) | evaluate_prediction.py |
| C6 | `get_top_features` — shap_values 있으면 None 반환 | 들여쓰기 수정 | shap.py |
| C7 | `features_names = List[str] = []` | `features_names: List[str] = []` | feature_service.py |

### HIGH (8건)

| # | 이슈 | 수정 | 파일 |
|---|------|------|------|
| H1 | 메서드명 `spilt` | `split` | csv_repository.py |
| H2 | 로깅 `len(train_val[0])` | `len(train), len(val)` | csv_repository.py |
| H3 | `Path(filepath).mkdir()` 파일경로에 디렉토리 생성 | `Path(filepath).parent.mkdir()` | shap.py |
| H4 | `pd.Dataframe` (소문자 f) | `pd.DataFrame` | shap.py, feature_service.py |
| H5 | `baseline_only` 모드에서 `extract_cbc_features` 호출 | `extract_baseline_features` | feature_service.py |
| H6 | `pct_change` 계산이 `total_delta` 컬럼 덮어씀 | `col = f"{feature}_pct_change"` 추가 | feature_service.py |
| H7 | `compute_calibration` 반환 키 불일치 | `fraction_of_positives`, `mean_predicted_value`, `brier_score` | evaluate_prediction.py |
| H8 | ECE 계산 결과가 배열 | `float(np.sum(...))` | evaluate_prediction.py |

### MEDIUM (8건)

| # | 이슈 | 수정 | 파일 |
|---|------|------|------|
| M1 | `CBCRecord` import 누락 | import 추가 | label_service.py |
| M2 | `_LABEL_WEEKS`, `_ANC_GRADE3_THRESHOLD` 미정의 | 상수 정의 추가 | label_service.py |
| M3 | `yaml` import 누락 | `import yaml` 추가 | excel_repository.py |
| M4 | `mapping = Dict[str, str] = {}` | `mapping: Dict[str, str] = {}` | excel_repository.py |
| M5 | `convert_long_to_wide` — value_cols 전달 시 None 반환 | 들여쓰기 수정 | emr_repository.py |
| M6 | DCA 반환 키 불일치 | `net_benefit_model`, `net_benefit_treat_all` | evaluate_screening.py |
| M7 | 하드코딩 파일 경로 | `sys.argv[1]` 또는 기본값 | run_real.py |
| M8 | 하드코딩 파일 경로 | `sys.argv[1]` 또는 기본값 | run_emr.py |

### LOW (2건)

| # | 이슈 | 수정 | 파일 |
|---|------|------|------|
| L1 | 클래스명 PascalCase 불일치 | `Run_SHAP`→`RunSHAP`, `Train_Prediction`→`TrainPrediction`, `Evaluate_Screening`→`EvaluateScreening`, `Find_Threshold`→`FindThreshold`, `train_screening`→`TrainScreening`, `Prediciton_input`→`PredictionInput` | 6개 파일 |
| L2 | 파일명 오타 `prediciton` | `train_prediction.py`, `prediction_input.py` | 2개 파일 리네이밍 |

---

## 3. 시각화 Troubleshooting

### Fig 2 (PR Curves) — 계단식 곡선

**원인**: Test set 80명으로 threshold 포인트가 적어 계단형 그래프 발생.

**해결**: 5-Fold CV의 fold별 PR curve를 보간(interpolation) 후 평균 ± 1SD band로 출력.

```python
# recall 축으로 보간
prec_interp = np.interp(mean_recall, rec[::-1], prec[::-1])
```

### Fig 6 (Calibration) — 지그재그 불안정

**원인**: uniform bins 사용 시 고확률 구간에 환자가 적어 fraction이 불안정.

**해결**: `strategy='quantile'` (각 bin에 동일 환자 수)로 변경. 5 bins 유지.

```python
calibration_curve(y_true, y_prob, n_bins=5, strategy='quantile')
```

### Fig 9 (DCA) — y축 스케일 문제

**원인**: Treat All 곡선이 높은 threshold에서 -30 이하로 급락하면서 모델 곡선이 보이지 않음.

**해결**: y축 범위를 `(-0.05, 0.40)`으로 제한. DCA에서 관심 영역은 Net Benefit > 0 구간.

```python
ax.set_ylim(-0.05, 0.40)
```

### Fig 11 (G-CSF Subgroup) — 한글 깨짐

**원인**: matplotlib 기본 폰트에 한글 글리프 없음.

**해결**: 영문으로 변경 (`G-CSF Unexposed` / `G-CSF Exposed`).

### 전체 커브 부드러움 개선

**원인**: holdout test set (n=80)으로 그리면 포인트가 적음.

**해결**: 5-Fold OOF (Out-of-Fold) 예측값 (n=400)으로 모든 커브 재생성.

```python
# OOF: 각 fold에서 validation set 예측값을 합쳐서 전체 400명 예측값 확보
oof_pred = np.zeros(len(y_all))
for tr_idx, val_idx in skf.split(x_all, y_all):
    model.fit(x_all[tr_idx], y_all[tr_idx])
    oof_pred[val_idx] = model.predict_proba(x_all[val_idx])
```

---

## 4. 데이터 관련 이슈

### Pseudo 데이터 AUC 0.99 문제

**원인**: Feature와 label 간 관계가 결정론적으로 생성됨 (ANC 낮으면 100% Grade 3+).

**해결**: 노이즈 추가 — 경계값 환자(boundary), 일시적 저하(transient_dip), G-CSF 반등(gcsf_rebound) 등의 label_type으로 현실적 불확실성 부여.

### Pseudo 데이터 양성 0명 문제

**원인**: CTCAE v6.0 Grade 3+ 기준 ANC < 0.5인데, pseudo 데이터 ANC 최솟값이 0.61.

**해결**: Feature_Matrix가 포함된 데이터셋 사용 (feature-label 관계가 반영된 데이터).

### G-CSF 교란변수

**발견**: G-CSF 미사용 코호트 AUC 0.959 vs G-CSF 사용 코호트 AUC 0.995.

**대응 전략**:
1. Primary analysis: G-CSF 미사용 코호트
2. Sensitivity analysis: 전체 + G-CSF 공변량 보정
3. Subgroup analysis: G-CSF 사용/미사용 하위군 비교

### AMC 결측률 > 30%

**현상**: AMC (Absolute Monocyte Count) 전 주차가 결측률 30% 초과로 자동 제거됨.

**원인**: 검사 오더에 AMC가 포함되지 않는 기관이 많음.

**영향**: AMC→ANC lead-lag 가설 검증 불가. AMC 파생 특징 (`AMC_decline_rate` 등) 모두 제거됨.

---

## 5. 최종 생성 그래프 (12개)

| Fig | 파일 | 내용 |
|-----|------|------|
| 1 | fig1_roc_curves.png | Mean ROC Curves ± 1SD (5-Fold CV) |
| 2 | fig2_pr_curves.png | Mean PR Curves ± 1SD (5-Fold CV) |
| 3 | fig3_cv_roc_bands.png | Fold별 ROC with confidence band |
| 4 | fig4_cv_boxplot.png | 5-Fold CV AUC Boxplot |
| 5 | fig5_confusion_matrices.png | Confusion Matrices (OOF, Youden threshold) |
| 6 | fig6_calibration.png | Calibration Curves (quantile bins) |
| 7 | fig7_feature_importance.png | XGBoost Feature Importance Top 20 |
| 8 | fig8_shap_summary.png | SHAP Summary (dot plot) |
| 8b | fig8b_shap_bar.png | SHAP Bar Plot |
| 9 | fig9_dca.png | Decision Curve Analysis |
| 10 | fig10_model_comparison.png | Model Performance Comparison Bar |
| 11 | fig11_gcsf_subgroup.png | G-CSF Subgroup ROC |