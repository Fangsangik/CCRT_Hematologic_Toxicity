"""
feature_engineer.py - 특성 공학 모듈

CBC 시계열 데이터에서 의미 있는 파생 변수를 추출하고,
범주형 변수를 인코딩하며, 스케일링을 수행합니다.

핵심 가설에 따라 AMC 감소 패턴 관련 특성을 중점적으로 생성합니다.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """CBC 시계열 및 임상 변수에서 특성을 추출·변환하는 클래스입니다.

    주요 기능:
        1. CBC 시계열 파생 변수 생성 (변화율, 기울기, 비율 등)
        2. 범주형 변수 인코딩
        3. 수치형 변수 스케일링
        4. Baseline-only / Baseline+CBC 모드 지원

    사용 예시:
        fe = FeatureEngineer(config)
        df = fe.create_cbc_temporal_features(df)
        df = fe.encode_categorical(df)
        X_scaled = fe.scale_features(X_train, fit=True)
    """

    def __init__(self, config):
        """FeatureEngineer를 초기화합니다.

        Args:
            config: Config 인스턴스
        """
        self.config = config
        self.data_config = config.data

        # 스케일러와 인코더를 저장 (학습 세트에 fit 후 테스트에 transform 적용)
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoder: Optional[OneHotEncoder] = None

        # 생성된 특성 이름을 추적
        self.temporal_feature_names: List[str] = []
        self.encoded_feature_names: List[str] = []

    # ============================================================
    # CBC 시계열 파생 변수 생성
    # ============================================================
    def create_cbc_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """CBC 시계열 데이터에서 파생 변수를 생성합니다.

        생성되는 변수 유형:
            1. 변화량 (delta): week간 절대 변화량
            2. 변화율 (rate): week간 상대 변화율
            3. 기울기 (slope): 전체 기간의 선형 기울기
            4. 변동 계수 (cv): 시계열 변동성
            5. AMC/ANC 비율: Monocyte-Neutrophil 관계 (핵심 가설)
            6. 최소값 (nadir): 관측 기간 내 최저값

        Args:
            df: CBC 시계열 컬럼이 포함된 DataFrame
                (예: WBC_week0, WBC_week1, WBC_week2, ...)

        Returns:
            파생 변수가 추가된 DataFrame
        """
        df = df.copy()
        self.temporal_feature_names = []
        timepoints = self.data_config.cbc_input_timepoints  # [0, 1, 2] 조기 예측용

        for feature in self.data_config.cbc_features:
            # 해당 feature의 시계열 컬럼명 생성
            cols = [f"{feature}_week{t}" for t in timepoints]

            # 존재하지 않는 컬럼은 건너뜀
            if not all(c in df.columns for c in cols):
                logger.warning(f"시계열 컬럼 누락, 건너뜀: {feature}")
                continue

            values = df[cols].values  # (n_patients, n_timepoints)

            # ----- 1) Week 간 변화량 (delta) -----
            for i in range(1, len(timepoints)):
                col_name = f"{feature}_delta_w{timepoints[i-1]}w{timepoints[i]}"
                df[col_name] = values[:, i] - values[:, i - 1]
                self.temporal_feature_names.append(col_name)

            # ----- 2) 전체 변화량 (Week 0 → Week 2) -----
            col_name = f"{feature}_total_delta"
            df[col_name] = values[:, -1] - values[:, 0]
            self.temporal_feature_names.append(col_name)

            # ----- 3) 변화율 (%) -----
            # baseline 값이 0인 경우를 방지하기 위해 epsilon 추가
            baseline = np.clip(values[:, 0], a_min=1e-6, a_max=None)
            col_name = f"{feature}_pct_change"
            df[col_name] = ((values[:, -1] - values[:, 0]) / baseline) * 100
            self.temporal_feature_names.append(col_name)

            # ----- 4) 선형 기울기 (시계열 추세) -----
            # 시간축 [0, 1, 2]에 대한 단순 선형 회귀 기울기
            col_name = f"{feature}_slope"
            t = np.array(timepoints, dtype=float)
            t_mean = t.mean()
            t_var = ((t - t_mean) ** 2).sum()
            df[col_name] = np.sum(
                (t - t_mean) * (values - values.mean(axis=1, keepdims=True)),
                axis=1,
            ) / t_var
            self.temporal_feature_names.append(col_name)

            # ----- 5) 변동 계수 (Coefficient of Variation) -----
            col_name = f"{feature}_cv"
            means = values.mean(axis=1)
            stds = values.std(axis=1)
            df[col_name] = np.where(means > 1e-6, stds / means, 0)
            self.temporal_feature_names.append(col_name)

            # ----- 6) Nadir (관측 기간 내 최소값) -----
            col_name = f"{feature}_nadir"
            df[col_name] = values.min(axis=1)
            self.temporal_feature_names.append(col_name)

            # ----- 7) 평균값 -----
            col_name = f"{feature}_mean"
            df[col_name] = values.mean(axis=1)
            self.temporal_feature_names.append(col_name)

        # ----- 핵심 가설 관련 파생 변수: AMC/ANC 비율 -----
        # Monocyte가 Neutrophil보다 먼저 감소하므로
        # AMC/ANC 비율의 변화가 예측에 중요할 수 있음
        for t in timepoints:
            amc_col = f"AMC_week{t}"
            anc_col = f"ANC_week{t}"
            if amc_col in df.columns and anc_col in df.columns:
                ratio_col = f"AMC_ANC_ratio_week{t}"
                df[ratio_col] = df[amc_col] / df[anc_col].clip(lower=1e-6)
                self.temporal_feature_names.append(ratio_col)

        # AMC/ANC 비율의 변화율
        if "AMC_ANC_ratio_week0" in df.columns and "AMC_ANC_ratio_week2" in df.columns:
            df["AMC_ANC_ratio_change"] = (
                df["AMC_ANC_ratio_week2"] - df["AMC_ANC_ratio_week0"]
            )
            self.temporal_feature_names.append("AMC_ANC_ratio_change")

        logger.info(f"시계열 파생 변수 {len(self.temporal_feature_names)}개 생성 완료")
        return df

    # ============================================================
    # 범주형 변수 인코딩
    # ============================================================
    def encode_categorical(
        self,
        df: pd.DataFrame,
        method: str = "onehot",
        fit: bool = True,
    ) -> pd.DataFrame:
        """범주형 변수를 수치형으로 인코딩합니다.

        Args:
            df: 입력 DataFrame
            method: 인코딩 방법
                - "onehot": One-Hot Encoding (기본)
                - "label": Label Encoding
                - "ordinal": 순서형 인코딩 (ecog_ps, stage 등)
            fit: True면 인코더를 새로 학습, False면 기존 인코더로 변환만 수행

        Returns:
            인코딩된 DataFrame
        """
        df = df.copy()
        cat_cols = [
            c for c in self.data_config.categorical_features
            if c in df.columns
        ]

        if not cat_cols:
            logger.info("인코딩할 범주형 변수가 없습니다.")
            return df

        if method == "onehot":
            if fit:
                self.onehot_encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                    drop="first",  # 다중공선성 방지
                )
                encoded = self.onehot_encoder.fit_transform(df[cat_cols])
            else:
                encoded = self.onehot_encoder.transform(df[cat_cols])

            # 인코딩된 컬럼명 생성
            self.encoded_feature_names = list(
                self.onehot_encoder.get_feature_names_out(cat_cols)
            )
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoded_feature_names,
                index=df.index,
            )

            # 원본 범주형 컬럼 제거 후 인코딩 결과 추가
            df = df.drop(columns=cat_cols)
            df = pd.concat([df, encoded_df], axis=1)

        elif method == "label":
            for col in cat_cols:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    # 학습 시 보지 못한 카테고리는 -1로 처리
                    known = set(self.encoders[col].classes_)
                    df[col] = df[col].apply(
                        lambda x: (
                            self.encoders[col].transform([str(x)])[0]
                            if str(x) in known
                            else -1
                        )
                    )

        logger.info(f"범주형 인코딩 완료: {len(cat_cols)}개 변수 ({method})")
        return df

    # ============================================================
    # 수치형 변수 스케일링
    # ============================================================
    def scale_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "standard",
        fit: bool = True,
    ) -> pd.DataFrame:
        """수치형 변수를 스케일링합니다.

        LSTM은 입력 스케일에 민감하므로 반드시 스케일링이 필요합니다.

        Args:
            df: 입력 DataFrame
            columns: 스케일링 대상 컬럼 (None이면 수치형 전체)
            method: 스케일링 방법
                - "standard": StandardScaler (평균=0, 표준편차=1)
                - "minmax": MinMaxScaler (0~1 범위)
            fit: True면 스케일러를 새로 학습, False면 기존 스케일러로 변환만 수행

        Returns:
            스케일링된 DataFrame
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        scaler_cls = StandardScaler if method == "standard" else MinMaxScaler

        if fit:
            self.scalers[method] = scaler_cls()
            df[columns] = self.scalers[method].fit_transform(df[columns])
        else:
            df[columns] = self.scalers[method].transform(df[columns])

        logger.info(f"스케일링 완료: {len(columns)}개 변수 ({method})")
        return df

    # ============================================================
    # 특성 선택 (실험 모드별)
    # ============================================================
    def get_feature_columns(
        self,
        df: pd.DataFrame,
        mode: str = "baseline_cbc",
    ) -> Dict[str, List[str]]:
        """실험 모드에 따라 사용할 특성 컬럼을 반환합니다.

        연구계획서의 비교 실험을 지원합니다:
            - baseline_only: baseline 임상 + 치료 변수만
            - baseline_cbc: baseline + CBC 시계열 + 파생 변수

        Args:
            df: 특성이 포함된 DataFrame
            mode: 실험 모드 ("baseline_only" 또는 "baseline_cbc")

        Returns:
            특성 그룹별 컬럼 목록 딕셔너리
                {
                    "baseline": [...],
                    "cbc_raw": [...],
                    "cbc_derived": [...],
                    "all": [...],
                }
        """
        # Baseline 특성 (인코딩 후의 컬럼 포함)
        baseline_cols = []
        for col in self.data_config.baseline_clinical_features:
            if col in df.columns:
                baseline_cols.append(col)
        for col in self.data_config.treatment_features:
            if col in df.columns:
                baseline_cols.append(col)
        # One-Hot 인코딩된 컬럼 추가
        for col in self.encoded_feature_names:
            if col in df.columns:
                baseline_cols.append(col)

        # CBC 원본 시계열 컬럼 (모델 입력용 = Week 0~2만)
        cbc_raw_cols = []
        for feature in self.data_config.cbc_features:
            for t in self.data_config.cbc_input_timepoints:
                col = f"{feature}_week{t}"
                if col in df.columns:
                    cbc_raw_cols.append(col)

        # CBC 파생 변수 컬럼
        cbc_derived_cols = [
            c for c in self.temporal_feature_names if c in df.columns
        ]

        # 모드별 특성 조합
        if mode == "baseline_only":
            all_cols = baseline_cols
        elif mode == "baseline_cbc":
            all_cols = baseline_cols + cbc_raw_cols + cbc_derived_cols
        else:
            raise ValueError(f"알 수 없는 실험 모드: {mode}")

        # 중복 제거 및 순서 유지
        seen = set()
        unique_cols = []
        for c in all_cols:
            if c not in seen:
                seen.add(c)
                unique_cols.append(c)

        result = {
            "baseline": baseline_cols,
            "cbc_raw": cbc_raw_cols,
            "cbc_derived": cbc_derived_cols,
            "all": unique_cols,
        }

        logger.info(
            f"특성 선택 ({mode}): "
            f"baseline={len(baseline_cols)}, "
            f"cbc_raw={len(cbc_raw_cols)}, "
            f"cbc_derived={len(cbc_derived_cols)}, "
            f"전체={len(unique_cols)}"
        )

        return result

    # ============================================================
    # LSTM용 시계열 텐서 준비
    # ============================================================
    def prepare_lstm_sequences(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """CBC 시계열 데이터를 LSTM 입력 형태로 변환합니다.

        LSTM 입력 형태: (n_samples, seq_length, n_features)
            - seq_length: 시계열 길이 (Week 0, 1, 2 → 3)
            - n_features: CBC 변수 수 (WBC, ANC, ALC, AMC, PLT, Hb → 6)

        Args:
            df: CBC 시계열 컬럼이 포함된 DataFrame

        Returns:
            3D numpy 배열 (n_samples, seq_length, n_features)
        """
        timepoints = self.data_config.cbc_timepoints  # [0, 1, 2, 3, 4, 5, 6, 7] 전체 시계열
        features = self.data_config.cbc_features

        n_samples = len(df)
        seq_length = len(timepoints)
        n_features = len(features)

        # 3D 배열 초기화
        sequences = np.zeros((n_samples, seq_length, n_features))

        for t_idx, t in enumerate(timepoints):
            for f_idx, feature in enumerate(features):
                col = f"{feature}_week{t}"
                if col in df.columns:
                    sequences[:, t_idx, f_idx] = df[col].values
                else:
                    logger.warning(f"LSTM 시퀀스 컬럼 누락: {col}")

        logger.info(
            f"LSTM 시퀀스 생성: shape={sequences.shape} "
            f"(환자={n_samples}, 시점={seq_length}, 변수={n_features})"
        )

        return sequences
