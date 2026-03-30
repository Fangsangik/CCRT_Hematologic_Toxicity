"""특징 추출 domain service"""
import pandas as pd
from typing import List, Tuple

import numpy as np

CBC_FEATURES: List[str] = ["WBC", "ANC", "ALC", "AMC", "PLT", "Hb"]
CBC_INPUT_TIMEPOINTS: List[int] = [0, 1, 2]

BASELINE_CLINICAL: List[str] = [
    "age",
    "sex",
    "bmi",
    "ecog_ps",
    "stage",
    "t_stage",
    "n_stage",
    "creatinine",
    "albumin",
]

TREATMENT_FEATURES : List[str] = [
    "rt_total_dose",
    "chemo_regimen"
    ]

class FeatureService:
    """
    환자 데이터프레임에서 모델 입력 특징을 추출하는 domain service.

    CBCTemporalExtractor의 로직을 도메인 레이어로 이식한 구현체다.
    """

    def extract_baseline_features(self, df : pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        임상 및 치료 특징ㅇ르 추출했다
        Args:
            df: 전처리된 환자 데이터프레임

        Returns:
            (특징 데이터프레임, 특징 이름 목록) 튜플.
            데이터프레임에 없는 컬럼은 조용히 생략한다.
        """
        all_baseline = BASELINE_CLINICAL + TREATMENT_FEATURES
        available = [col for col in all_baseline
                     if col in df.columns]
        return df[available].copy(), available

    def extract_cbc_features(self, df:pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        CBC 시계열 파생 특징을 추출한다.

        각 CBC 항목(WBC, ANC, ALC, AMC, PLT, Hb)에 대해 다음 특징을 생성한다:
            1. delta: 주차 간 변화량
            2. total_delta: 0주차 → 2주차 총 변화량
            3. pct_change: 기저치 대비 변화율 (%)
            4. slope: 선형 추세 기울기
            5. cv: 변동계수 (표준편차 / 평균)
            6. nadir: 최솟값
            7. mean: 평균값

        Args:
            df: 전처리된 환자 데이터프레임
                (컬럼명 형식: {CBC항목}_week{주차}, 예: ANC_week0)

        Returns:
            (파생 특징이 추가된 데이터프레임, 파생 특징 이름 목록) 튜플
        """

        df = df.copy()
        features_names: List[str] = []
        timepoints = CBC_INPUT_TIMEPOINTS

        t = np.array(timepoints, dtype = float)
        t_mean = t.mean()
        t_var = float(((t - t_mean) ** 2).sum())

        for feature in CBC_FEATURES :
            cols = [f"{feature}_week{tp}" for tp in timepoints]
            if not all(c in df.columns for c in cols) :
                continue
            values = df[cols].values

            # delta
            for i in range(1, len(timepoints)) :
                col = f"{feature}_delta_w{timepoints[i - 1]}w{timepoints[i]}"
                df[col] = values[:, i] - values[:, i - 1]
                features_names.append(col)

            # total_delta
            col = f"{feature}_total_delta"
            df[col] = values[:, -1] - values[:, 0]
            features_names.append(col)

            # pct_change : 0주차 대비 2주차 변화율
            col = f"{feature}_pct_change"
            baseline = np.clip(values[:, 0], a_min = 1e-6, a_max = None)
            df[col] = ((values[:, -1] - baseline) / baseline) * 100
            features_names.append(col)

            # slope
            col = f"{feature}_slope"
            row_means = values.mean(axis = 1, keepdims = True)
            df[col] =(
                np.sum((t- t_mean) * (values - row_means), axis = 1) / t_var
            )
            features_names.append(col)

            # cv : 변동계수 (표준편차 / 평균, 평균이 0에 가까울 경우 0으로 처리)
            col = f"{feature}_cv"
            means = values.mean(axis = 1)
            stds = values.std(axis = 1)
            df[col] = np.where(means > 1e-6, stds / means, 0.0)
            features_names.append(col)

            # nadir : 최솟값
            col = f"{feature}_nadir"
            df[col] = values.min(axis = 1)
            features_names.append(col)

            # mean : 평균값
            col = f"{feature}_mean"
            df[col] = values.mean(axis = 1)
            features_names.append(col)

        return df, features_names

    def extract_all(self, df : pd.DataFrame, mode : str = "baseline_cbc") -> Tuple[pd.DataFrame, List[str]]:
        """
        기저 특징과 CBC 파생 특징을 모드에 따라 결합하여 반환한다.

        Args:
            df: 전처리된 환자 데이터프레임
            mode: 특징 추출 모드.
                - "baseline_only": 기저 임상/치료 특징만 반환
                - "cbc_only": CBC 파생 특징만 반환
                - "baseline_cbc" (기본값): 두 특징 집합 모두 반환

        Returns:
            (결합된 특징 데이터프레임, 특징 이름 목록) 튜플

        Raises:
            ValueError: 알 수 없는 mode 값이 전달된 경우
        """
        if mode == "baseline_only" :
            _, names_base = self.extract_baseline_features(df)
            return df, names_base

        elif mode == "cbc_only" :
            df_cbc, names_cbc = self.extract_cbc_features(df)
            return df_cbc, names_cbc

        elif mode == "baseline_cbc" :
            _, names_base = self.extract_baseline_features(df)
            df_cbc, names_cbc = self.extract_cbc_features(df)
            all_names = names_base + names_cbc
            return df_cbc, all_names

        else :
            raise ValueError(f"알 수 없는 mode 값: {mode}")


