"""
generate_emr_data.py - 리얼한 EMR 합성 데이터 생성기

실제 EMR(전자의무기록)에서 추출된 것과 유사한 형태의 합성 데이터를 생성합니다.
JSON 형식으로 출력하며, CSV 변환 후 전처리 파이프라인을 테스트할 수 있습니다.

============================================================
CTCAE v5.0 혈액독성 등급 기준 (본 연구에서 사용하는 공식)
============================================================

1. Neutropenia (호중구감소증) - ANC 기준 (10³/μL)
   ┌──────────┬────────────────────────────────┐
   │ Grade 1  │ 1.5 ≤ ANC < 2.0 (LLN)         │
   │ Grade 2  │ 1.0 ≤ ANC < 1.5               │
   │ Grade 3  │ 0.5 ≤ ANC < 1.0               │
   │ Grade 4  │       ANC < 0.5               │
   └──────────┴────────────────────────────────┘

2. Anemia (빈혈) - Hemoglobin 기준 (g/dL)
   ┌──────────┬────────────────────────────────┐
   │ Grade 1  │ 10.0 ≤ Hb < 12.0 (LLN)        │
   │ Grade 2  │  8.0 ≤ Hb < 10.0              │
   │ Grade 3  │  6.5 ≤ Hb <  8.0 (수혈 필요)  │
   │ Grade 4  │        Hb <  6.5 (생명 위협)   │
   └──────────┴────────────────────────────────┘

3. Thrombocytopenia (혈소판감소증) - PLT 기준 (10³/μL)
   ┌──────────┬────────────────────────────────┐
   │ Grade 1  │  75 ≤ PLT < 150 (LLN)          │
   │ Grade 2  │  50 ≤ PLT <  75                │
   │ Grade 3  │  25 ≤ PLT <  50                │
   │ Grade 4  │       PLT <  25                │
   └──────────┴────────────────────────────────┘

4. Leukopenia (백혈구감소증) - WBC 기준 (10³/μL)
   ┌──────────┬────────────────────────────────┐
   │ Grade 1  │ 3.0 ≤ WBC < 4.0 (LLN)         │
   │ Grade 2  │ 2.0 ≤ WBC < 3.0               │
   │ Grade 3  │ 1.0 ≤ WBC < 2.0               │
   │ Grade 4  │       WBC < 1.0               │
   └──────────┴────────────────────────────────┘

5. Lymphopenia (림프구감소증) - ALC 기준 (10³/μL)
   ┌──────────┬────────────────────────────────┐
   │ Grade 1  │ 0.8 ≤ ALC < 1.0 (LLN)         │
   │ Grade 2  │ 0.5 ≤ ALC < 0.8               │
   │ Grade 3  │ 0.2 ≤ ALC < 0.5               │
   │ Grade 4  │       ALC < 0.2               │
   └──────────┴────────────────────────────────┘

참고: Grade 0 = 정상 (LLN 이상)
     Grade 3+ = 심각한 독성 → 용량 감량/치료 중단 고려

핵심 가설:
    AMC(절대단핵구수)는 ANC와 동일한 GMP에서 분화되지만
    반감기가 짧아 평균 3.81일 먼저 감소 (Ouyang et al., 2018)
    → Week 1-2 AMC 감소 패턴으로 Grade 3+ neutropenia 조기 예측

사용법:
    python generate_emr_data.py                    # 기본 100명 생성
    python generate_emr_data.py --n_patients 200   # 200명 생성
    python generate_emr_data.py --test              # 생성 + 파이프라인 테스트
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# 임상적으로 현실적인 파라미터 범위
# ============================================================

# 정상 CBC 참고치 범위
NORMAL_CBC_RANGES = {
    "WBC":  {"mean": 7.0,  "std": 2.0,  "min": 4.0,  "max": 11.0},   # 10³/μL
    "ANC":  {"mean": 4.5,  "std": 1.5,  "min": 2.0,  "max": 7.5},    # 10³/μL
    "ALC":  {"mean": 1.8,  "std": 0.5,  "min": 1.0,  "max": 3.5},    # 10³/μL
    "AMC":  {"mean": 0.5,  "std": 0.15, "min": 0.2,  "max": 1.0},    # 10³/μL
    "PLT":  {"mean": 250,  "std": 60,   "min": 150,  "max": 400},     # 10³/μL
    "Hb":   {"mean": 13.5, "std": 1.5,  "min": 12.0, "max": 17.0},   # g/dL
}

# 항암 레지멘별 혈액독성 심각도 가중치
# EP(Etoposide+Cisplatin)이 가장 심함, weekly_paclitaxel이 가장 약함
REGIMEN_TOXICITY = {
    "EP":                {"neutropenia_risk": 0.40, "decay_rate": 0.35},
    "TP":                {"neutropenia_risk": 0.30, "decay_rate": 0.28},
    "GP":                {"neutropenia_risk": 0.25, "decay_rate": 0.25},
    "weekly_paclitaxel": {"neutropenia_risk": 0.15, "decay_rate": 0.18},
    "DP":                {"neutropenia_risk": 0.28, "decay_rate": 0.26},
}

# 폐암 CCRT 환자 인구통계 분포
DEMOGRAPHICS = {
    "age":     {"mean": 64, "std": 8, "min": 40, "max": 82},
    "bmi":     {"mean": 23.5, "std": 3.0, "min": 16.0, "max": 35.0},
    "stages":  ["IIIA", "IIIB", "IIIC"],
    "stage_weights": [0.45, 0.40, 0.15],
    "t_stages": ["T1", "T2", "T3", "T4"],
    "t_weights": [0.10, 0.30, 0.35, 0.25],
    "n_stages": ["N0", "N1", "N2", "N3"],
    "n_weights": [0.05, 0.15, 0.50, 0.30],
    "ecog_ps":  [0, 1, 2],
    "ecog_weights": [0.25, 0.55, 0.20],
    "sex_ratio": 0.70,  # 남성 비율 (폐암은 남성이 더 많음)
}


def generate_patients(n_patients: int, seed: int = 42) -> list:
    """환자 기본정보를 생성합니다.

    실제 폐암 CCRT 코호트와 유사한 인구통계 분포를 따릅니다.

    Args:
        n_patients: 생성할 환자 수
        seed: 랜덤 시드

    Returns:
        환자 정보 딕셔너리 리스트
    """
    rng = np.random.RandomState(seed)
    patients = []

    # 치료 시작일 범위: 2023-01 ~ 2024-06 (18개월)
    start_base = datetime(2023, 1, 1)
    date_range_days = 540  # 18개월

    for i in range(n_patients):
        pid = f"PT{i+1:04d}"
        sex = "M" if rng.random() < DEMOGRAPHICS["sex_ratio"] else "F"

        # 나이, BMI
        age = int(np.clip(rng.normal(DEMOGRAPHICS["age"]["mean"], DEMOGRAPHICS["age"]["std"]),
                          DEMOGRAPHICS["age"]["min"], DEMOGRAPHICS["age"]["max"]))
        bmi = round(np.clip(rng.normal(DEMOGRAPHICS["bmi"]["mean"], DEMOGRAPHICS["bmi"]["std"]),
                            DEMOGRAPHICS["bmi"]["min"], DEMOGRAPHICS["bmi"]["max"]), 1)

        # 병기
        stage = rng.choice(DEMOGRAPHICS["stages"], p=DEMOGRAPHICS["stage_weights"])
        t_stage = rng.choice(DEMOGRAPHICS["t_stages"], p=DEMOGRAPHICS["t_weights"])
        n_stage = rng.choice(DEMOGRAPHICS["n_stages"], p=DEMOGRAPHICS["n_weights"])
        ecog = int(rng.choice(DEMOGRAPHICS["ecog_ps"], p=DEMOGRAPHICS["ecog_weights"]))

        # 혈액검사 (baseline)
        creatinine = round(np.clip(rng.normal(0.9, 0.25), 0.5, 2.0), 2)
        albumin = round(np.clip(rng.normal(3.8, 0.4), 2.5, 5.0), 1)

        # 치료 정보
        rt_start = start_base + timedelta(days=int(rng.uniform(0, date_range_days)))
        regimen = rng.choice(list(REGIMEN_TOXICITY.keys()),
                             p=[0.35, 0.25, 0.15, 0.15, 0.10])

        # 방사선량: 대부분 60-66 Gy
        rt_total = float(rng.choice([60.0, 63.0, 66.0], p=[0.40, 0.35, 0.25]))
        rt_frac = round(rt_total / rng.choice([30, 33], p=[0.6, 0.4]), 1)

        # 항암 용량 (mg/m²) - 레지멘별 표준 용량에 변동 추가
        base_doses = {"EP": 100, "TP": 175, "GP": 1000, "weekly_paclitaxel": 50, "DP": 75}
        chemo_dose = int(base_doses[regimen] * rng.uniform(0.85, 1.0))
        chemo_cycles = int(rng.choice([2, 3, 4], p=[0.50, 0.35, 0.15]))

        patient = {
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "ecog_ps": ecog,
            "stage": stage,
            "t_stage": t_stage,
            "n_stage": n_stage,
            "creatinine": creatinine,
            "albumin": albumin,
            "rt_start_date": rt_start.strftime("%Y-%m-%d"),
            "rt_total_dose": rt_total,
            "rt_fraction_dose": rt_frac,
            "chemo_regimen": regimen,
            "chemo_dose": chemo_dose,
            "chemo_cycles": chemo_cycles,
        }
        patients.append(patient)

    return patients


# ============================================================
# 궤적 유형별 CBC 감소율 계산 헬퍼 함수들
# ============================================================

def _calc_classic_toxic_drop(marker, week_idx, decay_rate, pt, rng, sensitivity):
    """전형적 독성 패턴: AMC 선행 감소 → ANC Nadir → 회복/재감소"""
    s = sensitivity  # 개인 감수성 스케일링

    if marker == "AMC":
        drops = {
            0: rng.uniform(0.0, 0.15),
            1: rng.uniform(0.20, 0.50) * s,
            2: rng.uniform(0.40, 0.70) * s,
            3: rng.uniform(0.50, 0.80) * s,
            4: rng.uniform(0.35, 0.70) * s,
            5: rng.uniform(0.25, 0.60) * s,
        }
        if week_idx <= 5:
            return min(drops.get(week_idx, 0), 0.95)
        else:
            if pt["chemo_cycles"] >= 2 and rng.random() < 0.5:
                return min(rng.uniform(0.45, 0.80) * s, 0.95)
            return min(rng.uniform(0.15, 0.50) * s, 0.95)

    elif marker == "ANC":
        drops = {
            0: rng.uniform(0.0, 0.12),
            1: rng.uniform(0.10, 0.30) * s,
            2: rng.uniform(0.40, 0.75) * s,
            3: rng.uniform(0.55, 0.85) * s,
            4: rng.uniform(0.40, 0.75) * s,
            5: rng.uniform(0.30, 0.60) * s,
        }
        if week_idx <= 5:
            return min(drops.get(week_idx, 0), 0.95)
        else:
            if pt["chemo_cycles"] >= 2 and rng.random() < 0.5:
                return min(rng.uniform(0.50, 0.85) * s, 0.95)
            return min(rng.uniform(0.15, 0.45) * s, 0.95)

    elif marker == "WBC":
        if week_idx <= 3:
            return min(week_idx * decay_rate * rng.uniform(0.7, 1.3) * s, 0.90)
        elif week_idx <= 5:
            peak = 3 * decay_rate * rng.uniform(0.7, 1.3) * s
            recovery = (week_idx - 3) * 0.10 * rng.uniform(0.5, 1.5)
            return max(peak - recovery, 0.08)
        else:
            if pt["chemo_cycles"] >= 2 and rng.random() < 0.5:
                return min(week_idx * decay_rate * rng.uniform(0.5, 1.0) * s, 0.90)
            return rng.uniform(0.08, 0.35)

    elif marker == "ALC":
        if week_idx == 0:
            return rng.uniform(0.0, 0.20)
        elif week_idx <= 3:
            return min(rng.uniform(0.3, 0.75) * min(week_idx / 2, 1.0) * s, 0.90)
        else:
            return min(rng.uniform(0.50, 0.88) * s, 0.95)

    elif marker == "PLT":
        if week_idx <= 2:
            return min(week_idx * decay_rate * 0.4 * rng.uniform(0.5, 1.1) * s, 0.50)
        elif week_idx <= 5:
            return min(week_idx * decay_rate * 0.5 * rng.uniform(0.6, 1.1) * s, 0.75)
        else:
            peak = 5 * decay_rate * 0.5 * rng.uniform(0.6, 1.1) * s
            recovery = (week_idx - 5) * 0.07 * rng.uniform(0.4, 1.5)
            return max(peak - recovery, 0.08)

    elif marker == "Hb":
        return min(week_idx * 0.022 * rng.uniform(0.4, 1.6) * s, 0.45)

    return 0.0


def _calc_gradual_toxic_drop(marker, week_idx, decay_rate, pt, rng, sensitivity):
    """서서히 진행하는 독성: 초기(Week 0-2) 신호 약하고 Week 3-5에서 본격 하락"""
    s = sensitivity

    if marker == "AMC":
        drops = {
            0: rng.uniform(0.0, 0.08),
            1: rng.uniform(0.05, 0.25) * s,  # 초기 감소 약함
            2: rng.uniform(0.15, 0.40) * s,
            3: rng.uniform(0.40, 0.70) * s,  # 본격 감소 시작
            4: rng.uniform(0.50, 0.80) * s,
            5: rng.uniform(0.45, 0.75) * s,
        }
        return min(drops.get(week_idx, rng.uniform(0.30, 0.65) * s), 0.95)

    elif marker == "ANC":
        drops = {
            0: rng.uniform(0.0, 0.08),
            1: rng.uniform(0.03, 0.18) * s,  # 거의 변화 없음
            2: rng.uniform(0.10, 0.35) * s,  # 약간 감소
            3: rng.uniform(0.35, 0.65) * s,  # 본격 감소
            4: rng.uniform(0.55, 0.85) * s,  # Nadir (지연)
            5: rng.uniform(0.50, 0.80) * s,
        }
        return min(drops.get(week_idx, rng.uniform(0.35, 0.70) * s), 0.95)

    elif marker == "WBC":
        if week_idx <= 2:
            return min(week_idx * decay_rate * 0.4 * rng.uniform(0.5, 1.2) * s, 0.40)
        else:
            return min(week_idx * decay_rate * rng.uniform(0.7, 1.2) * s, 0.90)

    elif marker == "ALC":
        return min(rng.uniform(0.05, 0.15) * week_idx * s, 0.90)

    elif marker == "PLT":
        if week_idx <= 2:
            return min(week_idx * decay_rate * 0.3 * rng.uniform(0.4, 0.9) * s, 0.30)
        else:
            return min(week_idx * decay_rate * 0.5 * rng.uniform(0.6, 1.1) * s, 0.75)

    elif marker == "Hb":
        return min(week_idx * 0.020 * rng.uniform(0.4, 1.5) * s, 0.40)

    return 0.0


def _calc_late_onset_drop(marker, week_idx, decay_rate, pt, rng, sensitivity):
    """Late onset: Week 0-2 거의 정상 → Week 3+ 갑자기 급감"""
    s = sensitivity

    if week_idx <= 2:
        # 초기에는 비독성 환자와 거의 구분 불가
        if marker == "ALC":
            return min(week_idx * rng.uniform(0.05, 0.18), 0.35)
        elif marker == "AMC":
            return min(week_idx * rng.uniform(0.02, 0.12), 0.25)
        else:
            return min(week_idx * decay_rate * 0.25 * rng.uniform(0.3, 0.8), 0.20)
    else:
        # Week 3+: 갑자기 급감
        if marker == "ANC":
            return min(rng.uniform(0.50, 0.90) * s, 0.95)
        elif marker == "AMC":
            return min(rng.uniform(0.45, 0.85) * s, 0.95)
        elif marker == "WBC":
            return min(rng.uniform(0.40, 0.80) * s, 0.90)
        elif marker == "ALC":
            return min(rng.uniform(0.55, 0.90) * s, 0.95)
        elif marker == "PLT":
            return min(rng.uniform(0.30, 0.70) * s, 0.85)
        elif marker == "Hb":
            return min(rng.uniform(0.10, 0.35) * s, 0.45)

    return 0.0


def _calc_false_alarm_drop(marker, week_idx, decay_rate, rng, sensitivity):
    """False alarm: 초기 급감 → 회복 (결과적으로 비독성)"""
    s = sensitivity

    if marker == "AMC":
        drops = {
            0: rng.uniform(0.0, 0.10),
            1: rng.uniform(0.25, 0.50) * s,  # 독성 환자처럼 급감
            2: rng.uniform(0.30, 0.55) * s,  # 더 떨어짐
            3: rng.uniform(0.20, 0.45) * s,  # 회복 시작
            4: rng.uniform(0.10, 0.30) * s,  # 회복 진행
            5: rng.uniform(0.05, 0.20) * s,
        }
        return min(drops.get(week_idx, rng.uniform(0.03, 0.15)), 0.70)

    elif marker == "ANC":
        drops = {
            0: rng.uniform(0.0, 0.10),
            1: rng.uniform(0.15, 0.40) * s,  # 상당히 감소
            2: rng.uniform(0.25, 0.55) * s,  # 더 감소 (Grade 2 수준)
            3: rng.uniform(0.15, 0.40) * s,  # 회복 시작
            4: rng.uniform(0.08, 0.25) * s,
            5: rng.uniform(0.03, 0.15) * s,
        }
        return min(drops.get(week_idx, rng.uniform(0.02, 0.10)), 0.65)

    elif marker == "WBC":
        if week_idx <= 2:
            return min(week_idx * decay_rate * rng.uniform(0.6, 1.2) * s, 0.55)
        else:
            # 회복
            peak = 2 * decay_rate * rng.uniform(0.6, 1.2) * s
            recovery = (week_idx - 2) * 0.10 * rng.uniform(0.8, 1.5)
            return max(peak - recovery, 0.03)

    elif marker == "ALC":
        return min(week_idx * rng.uniform(0.06, 0.16), 0.55)

    elif marker == "PLT":
        return min(week_idx * decay_rate * 0.3 * rng.uniform(0.4, 0.9), 0.30)

    elif marker == "Hb":
        return min(week_idx * 0.015 * rng.uniform(0.5, 1.3), 0.15)

    return 0.0


def _calc_moderate_nontoxic_drop(marker, week_idx, decay_rate, rng, sensitivity):
    """중등도 비독성: 꽤 감소하지만 Grade 3 미만 유지"""
    s = sensitivity

    if marker == "ANC":
        # Grade 2 수준까지 감소 가능 (ANC 1.0-1.5)
        drops = {
            0: rng.uniform(0.0, 0.08),
            1: rng.uniform(0.08, 0.25) * s,
            2: rng.uniform(0.20, 0.45) * s,
            3: rng.uniform(0.30, 0.55) * s,  # 최대 감소
            4: rng.uniform(0.25, 0.45) * s,
            5: rng.uniform(0.15, 0.35) * s,
        }
        return min(drops.get(week_idx, rng.uniform(0.10, 0.30) * s), 0.60)

    elif marker == "AMC":
        drops = {
            0: rng.uniform(0.0, 0.10),
            1: rng.uniform(0.10, 0.30) * s,
            2: rng.uniform(0.20, 0.45) * s,
            3: rng.uniform(0.30, 0.55) * s,
            4: rng.uniform(0.20, 0.45) * s,
            5: rng.uniform(0.12, 0.30) * s,
        }
        return min(drops.get(week_idx, rng.uniform(0.08, 0.25) * s), 0.65)

    elif marker == "WBC":
        return min(week_idx * decay_rate * 0.6 * rng.uniform(0.5, 1.1) * s, 0.55)

    elif marker == "ALC":
        return min(week_idx * rng.uniform(0.08, 0.18) * s, 0.65)

    elif marker == "PLT":
        return min(week_idx * decay_rate * 0.4 * rng.uniform(0.4, 0.9) * s, 0.40)

    elif marker == "Hb":
        return min(week_idx * 0.018 * rng.uniform(0.5, 1.4) * s, 0.20)

    return 0.0


def _calc_stable_nontoxic_drop(marker, week_idx, decay_rate, rng):
    """안정적 비독성: 소폭 감소, 정상 범위 유지"""
    mild_decay = decay_rate * 0.3

    if marker == "ALC":
        return min(week_idx * rng.uniform(0.04, 0.14), 0.55)
    elif marker == "AMC":
        return min(week_idx * rng.uniform(0.02, 0.08), 0.35)
    elif marker == "PLT":
        return min(week_idx * mild_decay * rng.uniform(0.2, 0.5), 0.20)
    elif marker == "Hb":
        return min(week_idx * 0.012 * rng.uniform(0.4, 1.2), 0.12)
    else:
        return min(week_idx * mild_decay * rng.uniform(0.3, 0.6), 0.25)


def generate_cbc_timeseries(patients: list, seed: int = 42) -> list:
    """각 환자의 CBC 시계열 검사 결과를 생성합니다 (현실적 노이즈 포함).

    현실적 패턴 반영:
        1. False Positive (~15%): 초기 CBC 급감했으나 Week 3+ 에서 회복
        2. False Negative (~10%): 초기 정상이었으나 Week 3+ 에서 갑자기 독성
        3. 독성/비독성 그룹 간 Week 0-2 패턴 겹침 (overlapping distributions)
        4. 측정 노이즈 8-12% (실제 CBC 변동 + 검사 오차)
        5. Baseline부터 경계선/비정상인 환자 (~20%)
        6. 환자별 개인 감수성 (individual sensitivity) 연속 스펙트럼

    AMC 선행 감소 가설은 유지하되, 효과 크기에 개인차 추가.

    Args:
        patients: generate_patients()의 출력
        seed: 랜덤 시드

    Returns:
        CBC 검사 결과 딕셔너리 리스트
    """
    rng = np.random.RandomState(seed + 1)
    all_cbc = []

    # 환자별 독성 발생 여부 기록 (디버깅/검증용)
    toxicity_labels = {}

    for pt in patients:
        pid = pt["patient_id"]
        rt_start = datetime.strptime(pt["rt_start_date"], "%Y-%m-%d")
        regimen = pt["chemo_regimen"]
        tox_info = REGIMEN_TOXICITY[regimen]

        # ============================================================
        # 환자별 독성 위험도 계산
        # ============================================================
        base_risk = tox_info["neutropenia_risk"]

        risk_modifiers = 0.0
        if pt["age"] >= 70:
            risk_modifiers += 0.10
        if pt["ecog_ps"] >= 2:
            risk_modifiers += 0.08
        if pt["albumin"] < 3.5:
            risk_modifiers += 0.07
        if pt["bmi"] < 18.5:
            risk_modifiers += 0.05
        if pt["sex"] == "F":
            risk_modifiers += 0.03
        if pt["chemo_cycles"] >= 3:
            risk_modifiers += 0.05

        total_risk = min(base_risk + risk_modifiers, 0.75)
        will_develop_toxicity = rng.random() < total_risk
        toxicity_labels[pid] = will_develop_toxicity

        # ============================================================
        # 환자별 궤적 유형 결정 (현실적 다양성)
        # ============================================================
        # 독성 환자 중 일부는 초기에 신호가 약함 (late-onset)
        # 비독성 환자 중 일부는 초기에 급감했다 회복 (false alarm)
        trajectory_roll = rng.random()

        if will_develop_toxicity:
            if trajectory_roll < 0.10:
                # Late-onset (10%): Week 0-2 거의 정상, Week 3+ 갑자기 독성
                trajectory_type = "late_onset"
            elif trajectory_roll < 0.25:
                # Gradual (15%): 서서히 감소, 초기 신호 약함
                trajectory_type = "gradual_toxic"
            else:
                # Classic (75%): 전형적 골수억제 패턴
                trajectory_type = "classic_toxic"
        else:
            if trajectory_roll < 0.15:
                # False alarm (15%): 초기 급감 후 회복
                trajectory_type = "false_alarm"
            elif trajectory_roll < 0.30:
                # Moderate decline (15%): 꽤 감소하지만 Grade 3 미만
                trajectory_type = "moderate_nontoxic"
            else:
                # Stable (70%): 안정적, 약간의 감소
                trajectory_type = "stable_nontoxic"

        # 환자별 개인 감수성 (0.5~1.5배 스케일링) — 같은 유형이라도 개인차
        individual_sensitivity = rng.uniform(0.6, 1.4)

        # ============================================================
        # Baseline CBC 생성 (일부 환자는 이미 경계선/비정상)
        # ============================================================
        baseline = {}
        # 약 20% 환자는 baseline부터 정상 하한 근처 또는 약간 비정상
        has_abnormal_baseline = rng.random() < 0.20

        for marker, ranges in NORMAL_CBC_RANGES.items():
            if marker == "Hb" and pt["sex"] == "F":
                mean_val = ranges["mean"] - 1.5
            else:
                mean_val = ranges["mean"]

            if has_abnormal_baseline:
                # 정상 하한 근처에서 시작 (기존 질환, 이전 치료 등)
                val = rng.normal(mean_val - ranges["std"] * 0.8, ranges["std"] * 0.8)
            else:
                val = rng.normal(mean_val, ranges["std"])

            baseline[marker] = np.clip(val, ranges["min"] * 0.8, ranges["max"])

        # ============================================================
        # 주차별 CBC 변화 시뮬레이션 (Week 0~7)
        # ============================================================
        exam_schedule = [
            (-5, 0),     # Week 0: Baseline (치료 시작 전)
            (3, 10),     # Week 1: 3~10일
            (11, 17),    # Week 2: 11~17일
        ]

        if rng.random() < 0.85:
            exam_schedule.append((19, 24))  # Week 3
        if rng.random() < 0.80:
            exam_schedule.append((26, 31))  # Week 4
        if rng.random() < 0.75:
            exam_schedule.append((33, 38))  # Week 5
        if rng.random() < 0.70:
            exam_schedule.append((40, 45))  # Week 6
        if rng.random() < 0.65:
            exam_schedule.append((47, 52))  # Week 7

        decay_rate = tox_info["decay_rate"]

        for week_idx, (day_min, day_max) in enumerate(exam_schedule):
            exam_day = int(rng.uniform(day_min, day_max))
            exam_date = rt_start + timedelta(days=exam_day)
            week_fraction = max(week_idx, 0)

            cbc_values = {}
            for marker in ["WBC", "ANC", "ALC", "AMC", "PLT", "Hb"]:
                base_val = baseline[marker]

                # ============================================================
                # 궤적 유형별 CBC 변화 패턴
                # ============================================================
                if trajectory_type == "classic_toxic":
                    # 전형적 독성: 초기 감소 → Nadir → 회복/재감소
                    drop = _calc_classic_toxic_drop(
                        marker, week_idx, decay_rate, pt, rng, individual_sensitivity
                    )
                    val = base_val * max(1 - drop, 0.05)
                    # Grade 3+ ANC 보장 (Nadir 주변)
                    if marker == "ANC" and week_idx in (2, 3) and rng.random() < 0.60:
                        val = min(val, rng.uniform(0.15, 0.95))
                    if marker == "ANC" and week_idx in (6, 7) and pt["chemo_cycles"] >= 2 and rng.random() < 0.35:
                        val = min(val, rng.uniform(0.3, 1.2))

                elif trajectory_type == "gradual_toxic":
                    # 서서히 독성: Week 0-2 감소 약하고, Week 3-5에서 본격 하락
                    drop = _calc_gradual_toxic_drop(
                        marker, week_idx, decay_rate, pt, rng, individual_sensitivity
                    )
                    val = base_val * max(1 - drop, 0.05)
                    if marker == "ANC" and week_idx in (3, 4, 5) and rng.random() < 0.55:
                        val = min(val, rng.uniform(0.2, 0.95))

                elif trajectory_type == "late_onset":
                    # Late onset: Week 0-2 거의 정상, Week 3+ 갑자기 독성
                    drop = _calc_late_onset_drop(
                        marker, week_idx, decay_rate, pt, rng, individual_sensitivity
                    )
                    val = base_val * max(1 - drop, 0.05)
                    if marker == "ANC" and week_idx >= 3 and rng.random() < 0.65:
                        val = min(val, rng.uniform(0.15, 0.90))

                elif trajectory_type == "false_alarm":
                    # 초기 급감 → 회복 (비독성)
                    drop = _calc_false_alarm_drop(
                        marker, week_idx, decay_rate, rng, individual_sensitivity
                    )
                    val = base_val * max(1 - drop, 0.20)

                elif trajectory_type == "moderate_nontoxic":
                    # 꽤 감소하지만 Grade 3 미만 유지
                    drop = _calc_moderate_nontoxic_drop(
                        marker, week_idx, decay_rate, rng, individual_sensitivity
                    )
                    val = base_val * max(1 - drop, 0.30)

                else:  # stable_nontoxic
                    # 안정적, 소폭 감소
                    drop = _calc_stable_nontoxic_drop(
                        marker, week_idx, decay_rate, rng
                    )
                    val = base_val * max(1 - drop, 0.60)

                # ============================================================
                # 현실적 측정 노이즈 (8-12% 수준)
                # ============================================================
                # 생물학적 변동 + 검사 오차 + 채혈 시점 차이
                noise_pct = rng.uniform(0.08, 0.12)
                noise = rng.normal(0, base_val * noise_pct)
                val = max(val + noise, 0.01)

                # 가끔 이상치 (검체 오류, 용혈 등) — 2% 확률
                if rng.random() < 0.02:
                    val *= rng.uniform(0.7, 1.4)

                # 단위에 맞게 반올림
                if marker == "PLT":
                    cbc_values[marker] = round(val, 0)
                elif marker == "Hb":
                    cbc_values[marker] = round(val, 1)
                else:
                    cbc_values[marker] = round(val, 2)

            cbc_record = {
                "patient_id": pid,
                "exam_date": exam_date.strftime("%Y-%m-%d"),
                **cbc_values,
            }
            all_cbc.append(cbc_record)

    # 독성 발생 통계
    n_toxic = sum(toxicity_labels.values())
    logger.info(f"독성 발생률: {n_toxic}/{len(patients)} ({n_toxic/len(patients)*100:.1f}%)")

    return all_cbc, toxicity_labels


def save_as_json(patients: list, cbc_records: list, output_dir: str):
    """EMR 데이터를 JSON 파일로 저장합니다.

    실제 EMR 시스템에서 JSON API로 데이터를 추출하는 시나리오를 시뮬레이션합니다.

    Args:
        patients: 환자 정보 리스트
        cbc_records: CBC 검사 기록 리스트
        output_dir: 출력 디렉토리
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 환자 기본정보 JSON
    patients_json = {
        "metadata": {
            "source": "아주대학교병원 EMR",
            "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "폐암 CCRT 환자 기본정보",
            "n_patients": len(patients),
        },
        "patients": patients,
    }
    patients_path = output_dir / "emr_patients.json"
    with open(patients_path, "w", encoding="utf-8") as f:
        json.dump(patients_json, f, ensure_ascii=False, indent=2)

    # 2) CBC 검사결과 JSON
    cbc_json = {
        "metadata": {
            "source": "아주대학교병원 진단검사의학과",
            "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "CBC 검사 결과 (Long 형식, 날짜 기반)",
            "n_records": len(cbc_records),
            "columns": {
                "WBC": "백혈구 수 (10³/μL)",
                "ANC": "절대호중구수 (10³/μL)",
                "ALC": "절대림프구수 (10³/μL)",
                "AMC": "절대단핵구수 (10³/μL)",
                "PLT": "혈소판 수 (10³/μL)",
                "Hb": "헤모글로빈 (g/dL)",
            },
        },
        "results": cbc_records,
    }
    cbc_path = output_dir / "emr_cbc_results.json"
    with open(cbc_path, "w", encoding="utf-8") as f:
        json.dump(cbc_json, f, ensure_ascii=False, indent=2)

    # 3) CTCAE v5.0 기준표 JSON (참조용)
    ctcae_ref = {
        "title": "CTCAE v5.0 Hematologic Toxicity Grading Criteria",
        "reference": "https://ctep.cancer.gov/protocoldevelopment/electronic_applications/ctc.htm",
        "note": "Grade 0 = 정상 (LLN 이상), Grade 3+ = 심각한 독성",
        "criteria": {
            "neutropenia": {
                "marker": "ANC (10³/μL)",
                "grade_1": "1.5 ≤ ANC < 2.0",
                "grade_2": "1.0 ≤ ANC < 1.5",
                "grade_3": "0.5 ≤ ANC < 1.0",
                "grade_4": "ANC < 0.5",
            },
            "anemia": {
                "marker": "Hemoglobin (g/dL)",
                "grade_1": "10.0 ≤ Hb < 12.0",
                "grade_2": "8.0 ≤ Hb < 10.0",
                "grade_3": "6.5 ≤ Hb < 8.0",
                "grade_4": "Hb < 6.5",
            },
            "thrombocytopenia": {
                "marker": "Platelet (10³/μL)",
                "grade_1": "75 ≤ PLT < 150",
                "grade_2": "50 ≤ PLT < 75",
                "grade_3": "25 ≤ PLT < 50",
                "grade_4": "PLT < 25",
            },
            "leukopenia": {
                "marker": "WBC (10³/μL)",
                "grade_1": "3.0 ≤ WBC < 4.0",
                "grade_2": "2.0 ≤ WBC < 3.0",
                "grade_3": "1.0 ≤ WBC < 2.0",
                "grade_4": "WBC < 1.0",
            },
            "lymphopenia": {
                "marker": "ALC (10³/μL)",
                "grade_1": "0.8 ≤ ALC < 1.0",
                "grade_2": "0.5 ≤ ALC < 0.8",
                "grade_3": "0.2 ≤ ALC < 0.5",
                "grade_4": "ALC < 0.2",
            },
        },
    }
    ctcae_path = output_dir / "ctcae_v5_criteria.json"
    with open(ctcae_path, "w", encoding="utf-8") as f:
        json.dump(ctcae_ref, f, ensure_ascii=False, indent=2)

    logger.info(f"JSON 파일 저장 완료:")
    logger.info(f"  환자 정보: {patients_path}")
    logger.info(f"  CBC 결과: {cbc_path}")
    logger.info(f"  CTCAE 기준: {ctcae_path}")

    return patients_path, cbc_path


def json_to_csv(json_dir: str) -> tuple:
    """JSON 파일을 CSV로 변환합니다.

    EMR에서 JSON으로 추출된 데이터를 전처리 파이프라인에 입력할 수 있는
    CSV 형식으로 변환합니다.

    Args:
        json_dir: JSON 파일이 있는 디렉토리

    Returns:
        (patients_csv_path, cbc_csv_path) 튜플
    """
    json_dir = Path(json_dir)

    # 환자 정보 변환
    with open(json_dir / "emr_patients.json", "r", encoding="utf-8") as f:
        patients_data = json.load(f)
    patients_df = pd.DataFrame(patients_data["patients"])
    patients_csv = json_dir / "emr_patients.csv"
    patients_df.to_csv(patients_csv, index=False, encoding="utf-8-sig")

    # CBC 결과 변환
    with open(json_dir / "emr_cbc_results.json", "r", encoding="utf-8") as f:
        cbc_data = json.load(f)
    cbc_df = pd.DataFrame(cbc_data["results"])
    cbc_csv = json_dir / "emr_cbc_results.csv"
    cbc_df.to_csv(cbc_csv, index=False, encoding="utf-8-sig")

    logger.info(f"CSV 변환 완료:")
    logger.info(f"  환자: {patients_csv} ({len(patients_df)}명)")
    logger.info(f"  CBC:  {cbc_csv} ({len(cbc_df)}건)")

    return str(patients_csv), str(cbc_csv)


def run_pipeline_test(patients_csv: str, cbc_csv: str):
    """전처리 + 학습 파이프라인을 실행합니다.

    Args:
        patients_csv: 환자 정보 CSV 경로
        cbc_csv: CBC 결과 CSV 경로
    """
    # 프로젝트 모듈 임포트
    sys.path.insert(0, str(Path(__file__).parent))
    from config import Config
    from src.data.preprocessing import EMRPreprocessor
    from src.utils.helpers import setup_logging

    config = Config()
    setup_logging(log_dir=str(config.paths.log_dir))

    # ============================================================
    # 1단계: EMR 전처리
    # ============================================================
    logger.info("=" * 60)
    logger.info("EMR 전처리 시작")
    logger.info("=" * 60)

    preprocessor = EMRPreprocessor(config)
    patients_df = pd.read_csv(patients_csv)
    cbc_df = pd.read_csv(cbc_csv)

    processed_df = preprocessor.run_full_pipeline(patients_df, cbc_df)

    # 전처리 결과 저장
    processed_path = config.paths.processed_data_dir / "emr_processed.csv"
    processed_df.to_csv(processed_path, index=False, encoding="utf-8-sig")
    logger.info(f"전처리 완료: {processed_path} ({len(processed_df)}명, {len(processed_df.columns)}변수)")

    # ============================================================
    # 2단계: 학습 파이프라인
    # ============================================================
    logger.info("=" * 60)
    logger.info("학습 파이프라인 시작")
    logger.info("=" * 60)

    from main import run_pipeline
    results = run_pipeline(config, data_path=str(processed_path))

    return results


def print_data_summary(patients: list, cbc_records: list, toxicity_labels: dict):
    """생성된 데이터의 요약 통계를 출력합니다."""
    logger.info("=" * 60)
    logger.info("생성 데이터 요약")
    logger.info("=" * 60)

    # 환자 통계
    ages = [p["age"] for p in patients]
    n_male = sum(1 for p in patients if p["sex"] == "M")
    regimens = {}
    for p in patients:
        r = p["chemo_regimen"]
        regimens[r] = regimens.get(r, 0) + 1

    logger.info(f"총 환자: {len(patients)}명")
    logger.info(f"성별: 남 {n_male}명 ({n_male/len(patients)*100:.0f}%), "
                f"여 {len(patients)-n_male}명 ({(len(patients)-n_male)/len(patients)*100:.0f}%)")
    logger.info(f"나이: {np.mean(ages):.1f} ± {np.std(ages):.1f}세 "
                f"(범위: {min(ages)}-{max(ages)})")

    logger.info(f"\n레지멘 분포:")
    for reg, cnt in sorted(regimens.items(), key=lambda x: -x[1]):
        logger.info(f"  {reg:20s}: {cnt}명 ({cnt/len(patients)*100:.1f}%)")

    # 독성 통계
    n_toxic = sum(toxicity_labels.values())
    logger.info(f"\n예상 Grade 3+ neutropenia: {n_toxic}명 ({n_toxic/len(patients)*100:.1f}%)")

    # CBC 검사 통계
    records_per_patient = {}
    for r in cbc_records:
        pid = r["patient_id"]
        records_per_patient[pid] = records_per_patient.get(pid, 0) + 1

    counts = list(records_per_patient.values())
    logger.info(f"\nCBC 검사: 총 {len(cbc_records)}건")
    logger.info(f"환자당 검사: {np.mean(counts):.1f} ± {np.std(counts):.1f}건 "
                f"(범위: {min(counts)}-{max(counts)})")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="리얼한 EMR 합성 데이터 생성기 (CCRT 혈액독성 연구용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python generate_emr_data.py                       # 기본 100명 생성
  python generate_emr_data.py --n_patients 200      # 200명 생성
  python generate_emr_data.py --test                 # 생성 + 파이프라인 테스트
  python generate_emr_data.py --output data/emr/     # 출력 경로 지정
        """,
    )

    parser.add_argument("--n_patients", type=int, default=100, help="생성할 환자 수 (기본: 100)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (기본: 42)")
    parser.add_argument("--output", type=str, default="data/raw/emr_synthetic",
                        help="출력 디렉토리 (기본: data/raw/emr_synthetic)")
    parser.add_argument("--test", action="store_true", help="생성 후 전처리+학습 파이프라인 테스트 실행")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"EMR 합성 데이터 생성 ({args.n_patients}명)")
    logger.info("=" * 60)

    # 1) 환자 정보 생성
    patients = generate_patients(args.n_patients, seed=args.seed)

    # 2) CBC 시계열 생성
    cbc_records, toxicity_labels = generate_cbc_timeseries(patients, seed=args.seed)

    # 3) 데이터 요약 출력
    print_data_summary(patients, cbc_records, toxicity_labels)

    # 4) JSON 저장
    save_as_json(patients, cbc_records, args.output)

    # 5) CSV 변환
    patients_csv, cbc_csv = json_to_csv(args.output)

    # 6) 파이프라인 테스트 (옵션)
    if args.test:
        logger.info("\n" + "=" * 60)
        logger.info("파이프라인 테스트 실행")
        logger.info("=" * 60)
        run_pipeline_test(patients_csv, cbc_csv)


if __name__ == "__main__":
    main()
