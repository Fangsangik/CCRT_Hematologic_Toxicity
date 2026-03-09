# 폐암 CCRT 혈액독성 예측 모델

## AI-Based Prediction of Hematologic Toxicity in CCRT for Lung Cancer

아주대학교 방사선종양학교실

---

## 연구 개요

폐암 동시항암방사선치료(CCRT) 환자에서 **치료 중 CBC 시계열 변화**를 활용하여 **Grade 3 이상 혈액독성(Hematologic Toxicity)** 발생을 조기 예측하는 Machine Learning 모델입니다.

### 핵심 가설

> 치료 초반(Week 1-2) **AMC(절대단핵구수) 감소 패턴**이 이후 Grade 3+ neutropenia 발생을 예측할 수 있다.

Monocyte는 Neutrophil과 동일한 Granulocyte-Monocyte Progenitor(GMP)에서 분화되지만, postmitotic cell로 더 빨리 분화되고 반감기가 짧아 **평균 3.81일 먼저 감소**합니다 (Ouyang et al., 2018).

### 기존 연구와의 차별점

| 항목 | 기존 연구 | 본 연구 |
|------|----------|---------|
| **입력** | Radiomics + Dosiomics (baseline) | Baseline 임상 + **CBC 시계열** |
| **예측 시점** | 치료 전 | **치료 중 (Week 2)** |
| **암종** | 다양한 암종 | **폐암** |
| **방법론** | XGBoost (baseline) | **LSTM (시계열)** |

---

## 프로젝트 구조

```
ccrt_hematologic_toxicity/
├── config.py                          # 전역 설정 (하이퍼파라미터, 경로)
├── main.py                            # 메인 실행 파일
├── requirements.txt                   # 의존성 패키지
├── README.md                          # 프로젝트 설명서 (본 문서)
│
├── src/
│   ├── data/
│   │   ├── data_loader.py             # 데이터 로드, 결측치 처리, 분할
│   │   ├── feature_engineer.py        # CBC 시계열 파생 변수 생성
│   │   └── dataset.py                 # PyTorch Dataset/DataLoader
│   │
│   ├── models/
│   │   ├── lstm_model.py              # LSTM + Attention 모델 (주요 모델)
│   │   ├── baseline_models.py         # XGBoost, LightGBM, LogReg (비교 모델)
│   │   └── trainer.py                 # 학습 루프, Early Stopping, 교차 검증
│   │
│   ├── evaluation/
│   │   ├── metrics.py                 # AUROC, AUPRC, 민감도, 특이도, Bootstrap CI
│   │   └── visualization.py           # ROC 곡선, 학습 곡선, 특성 중요도 등
│   │
│   └── utils/
│       └── helpers.py                 # 로깅, 시드 고정, 결과 저장
│
├── data/
│   ├── raw/                           # 원본 데이터
│   └── processed/                     # 전처리된 데이터
│
├── outputs/
│   ├── models/                        # 학습된 모델 저장
│   ├── figures/                       # 시각화 결과
│   └── logs/                          # 학습 로그, 실험 결과 JSON
│
└── notebooks/                         # 탐색적 분석 노트북
```

---

## 설치 방법

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. PyTorch 설치 (GPU 사용 시)

```bash
# Apple Silicon (M1/M2/M3/M4)
pip install torch torchvision

# NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 사용 방법

### 데모 모드 (합성 데이터)

실제 데이터 없이 전체 파이프라인을 테스트할 수 있습니다.

```bash
python main.py --mode demo
```

### 실제 데이터로 학습

```bash
# 전체 모델 학습
python main.py --mode train --data data/raw/patients.csv

# 특정 모델만 학습
python main.py --mode train --data data/raw/patients.csv --models lstm xgboost

# 커스텀 설정 사용
python main.py --mode train --data data/raw/patients.csv --config my_config.yaml
```

### 데이터 형식

입력 CSV 파일은 다음 컬럼을 포함해야 합니다:

| 분류 | 변수 | 설명 |
|------|------|------|
| **CBC 시계열** | `WBC_week0`, `WBC_week1`, `WBC_week2` | 백혈구 수 (주차별) |
| | `ANC_week0`, `ANC_week1`, `ANC_week2` | 절대호중구수 |
| | `ALC_week0`, `ALC_week1`, `ALC_week2` | 절대림프구수 |
| | `AMC_week0`, `AMC_week1`, `AMC_week2` | 절대단핵구수 **(핵심)** |
| | `PLT_week0`, `PLT_week1`, `PLT_week2` | 혈소판 수 |
| | `Hb_week0`, `Hb_week1`, `Hb_week2` | 헤모글로빈 |
| **Baseline 임상** | `age`, `sex`, `bmi` | 기본 인구통계 |
| | `ecog_ps`, `stage`, `t_stage`, `n_stage` | 질병 특성 |
| | `creatinine`, `albumin` | 혈액 검사 |
| **치료 변수** | `rt_total_dose`, `rt_fraction_dose` | 방사선 치료 |
| | `chemo_regimen`, `chemo_dose`, `chemo_cycles` | 항암 치료 |
| **타겟** | `grade3_neutropenia` | Grade 3+ 호중구감소증 (0/1) |

---

## 모델 아키텍처

### LSTM + Temporal Attention (주요 모델)

```
CBC 시계열 (Week 0→1→2)          Baseline 임상+치료 변수
    │                                    │
    ▼                                    ▼
┌─────────┐                      ┌──────────────┐
│  LSTM   │                      │  FC Encoder  │
│ (2-layer)│                      │  (Linear→ReLU│
└────┬────┘                      │  →BN→Dropout)│
     │                           └──────┬───────┘
     ▼                                  │
┌──────────┐                            │
│ Temporal │                            │
│ Attention│                            │
└────┬─────┘                            │
     │                                  │
     └──────────┬───────────────────────┘
                │ Concatenate
                ▼
        ┌──────────────┐
        │  Classifier  │
        │  (FC→ReLU→FC)│
        └──────┬───────┘
               │
               ▼
         P(Grade 3+ HT)
```

### 비교 모델

- **XGBoost**: Gradient Boosting 기반, 특성 중요도 해석 가능
- **LightGBM**: 빠른 학습 속도, 리프 단위 분할 전략
- **Logistic Regression**: 기본 비교 모델, 회귀 계수 해석 가능

---

## 실험 설계

연구계획서에 따라 두 가지 실험 모드를 비교합니다:

### 실험 1: Baseline-only
- **입력**: 치료 전 임상 변수 + 치료 변수
- **목적**: 기존 방법론의 성능 기준선 확립

### 실험 2: Baseline + CBC 시계열
- **입력**: Baseline + Week 0/1/2 CBC 시계열 + 파생 변수
- **목적**: CBC 시계열 추가의 incremental value 평가

### 생성되는 파생 변수

CBC 시계열에서 자동으로 생성되는 파생 변수:

- **변화량 (delta)**: Week 간 절대 변화량
- **변화율 (%)**: 상대적 변화율
- **기울기 (slope)**: 전체 기간의 선형 추세
- **변동 계수 (CV)**: 시계열 변동성
- **AMC/ANC 비율**: Monocyte-Neutrophil 관계 (핵심 가설)
- **Nadir**: 관측 기간 내 최소값

---

## 평가 지표

의료 AI에 적합한 다각적 평가를 수행합니다:

| 지표 | 설명 | 임상 의미 |
|------|------|----------|
| **AUROC** | ROC 곡선 아래 면적 | 전체적 판별 성능 |
| **AUPRC** | PR 곡선 아래 면적 | 불균형 데이터에서의 성능 |
| **민감도 (Sensitivity)** | 실제 양성 탐지율 | 고위험군을 놓치지 않는 능력 |
| **특이도 (Specificity)** | 실제 음성 배제율 | 불필요한 개입 최소화 |
| **PPV** | 양성예측도 | 양성 예측의 신뢰성 |
| **NPV** | 음성예측도 | 음성 예측의 신뢰성 |
| **Brier Score** | 보정 지표 | 확률 예측의 정확성 |
| **Bootstrap 95% CI** | 신뢰구간 | 결과의 통계적 불확실성 |

---

## 출력 결과

학습 완료 후 `outputs/` 디렉토리에 생성되는 파일:

```
outputs/
├── models/
│   ├── lstm_best.pt                    # LSTM 최적 체크포인트
│   ├── xgboost_baseline_cbc.pkl        # XGBoost 모델
│   ├── lightgbm_baseline_cbc.pkl       # LightGBM 모델
│   └── logistic_regression_*.pkl       # Logistic Regression 모델
│
├── figures/
│   ├── cbc_timeseries_AMC.png          # AMC 시계열 (양성/음성 비교)
│   ├── roc_comparison.png              # ROC 곡선 비교
│   ├── model_comparison.png            # 모델 성능 막대 그래프
│   ├── lstm_training_history.png       # LSTM 학습 곡선
│   └── feature_importance_*.png        # 특성 중요도
│
└── logs/
    ├── experiment_results.json         # 전체 실험 결과
    └── config.yaml                     # 사용된 설정
```

---

## 설정 커스터마이징

`config.py`에서 모든 하이퍼파라미터를 변경할 수 있습니다:

```python
from config import Config

# 기본 설정 로드
cfg = Config()

# LSTM 하이퍼파라미터 변경
cfg.lstm.hidden_size = 128
cfg.lstm.num_layers = 3
cfg.lstm.learning_rate = 5e-4

# 데이터 설정 변경
cfg.data.test_size = 0.3
cfg.data.missing_strategy = "knn"

# 설정 저장 (YAML)
cfg.save("my_config.yaml")

# 저장된 설정 로드
cfg = Config.load("my_config.yaml")
```

---

## 임상적 활용

예측 결과에 따른 개입 전략:

| 예측 결과 | 위험도 | 개입 전략 |
|-----------|--------|----------|
| 고위험 (>0.7) | 높음 | 선제적 G-CSF 투여, 항암 용량 감량, 외래 f/u 간격 단축 |
| 중등도 위험 (0.3~0.7) | 중간 | CBC 집중 모니터링, G-CSF 대기 |
| 저위험 (<0.3) | 낮음 | 일반적 관리 |

---

## 참고문헌

1. Costa GJ, et al. Ann Transl Med. 2018;6(Suppl 1):S96.
2. Jiang L, et al. Support Care Cancer. 2013;21:785-791.
3. Deek MP, et al. Am J Clin Oncol. 2018;41(4):362-366.
4. van Rossum PSN, et al. Front Oncol. 2023;13:1278723.
5. Deek MP, et al. Int J Radiat Oncol Biol Phys. 2016;94(1):147-154.
6. Conibear J, et al. Br J Cancer. 2020;123:10-17.
7. Ouyang W, et al. J Cancer Res Ther. 2018;14:S565-S570.
8. Shimanuki M, et al. Oncotarget. 2018;9:18970-18984.
9. Zheng B, et al. Support Care Cancer. 2020;28:1289-1294.

---

## 라이선스

본 프로젝트는 아주대학교 방사선종양학교실 연구 목적으로 개발되었습니다.
