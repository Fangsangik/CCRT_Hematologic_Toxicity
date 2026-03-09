"""
data_loader.py - 데이터 로딩 및 전처리 모듈

EMR에서 추출된 폐암 CCRT 환자 데이터를 로드하고,
결측치 처리, 이상치 제거, 데이터 분할을 수행합니다.

지원 형식: CSV, Excel, Parquet
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 로깅 설정
logger = logging.getLogger(__name__)


class DataLoader:
    """CCRT 혈액독성 연구 데이터를 로드하고 전처리하는 클래스입니다.

    지원 기능:
        - 다양한 파일 형식 로드 (CSV, Excel, Parquet)
        - 결측치 탐색 및 처리
        - 이상치 탐지 (IQR 방식)
        - 학습/검증/테스트 세트 분할

    사용 예시:
        loader = DataLoader(config)
        df = loader.load_data("patients.csv")
        df = loader.handle_missing_values(df)
        train_df, val_df, test_df = loader.split_data(df)
    """

    def __init__(self, config):
        """DataLoader를 초기화합니다.

        Args:
            config: Config 인스턴스 (config.py 참조)
        """
        self.config = config
        self.data_config = config.data
        self.path_config = config.paths

    # ============================================================
    # 데이터 로드
    # ============================================================
    def load_data(
        self,
        filename: str,
        data_dir: Optional[Path] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """파일에서 데이터를 로드합니다.

        Args:
            filename: 파일 이름 (예: "patients.csv")
            data_dir: 데이터 디렉토리 (None이면 config의 raw_data_dir 사용)
            **kwargs: pandas read 함수에 전달할 추가 인자

        Returns:
            로드된 DataFrame

        Raises:
            FileNotFoundError: 파일을 찾을 수 없는 경우
            ValueError: 지원하지 않는 파일 형식인 경우
        """
        if data_dir is None:
            data_dir = self.path_config.raw_data_dir

        filepath = Path(data_dir) / filename

        if not filepath.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

        # 파일 확장자에 따라 적절한 로더 선택
        suffix = filepath.suffix.lower()
        loaders = {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel,
            ".xls": pd.read_excel,
            ".parquet": pd.read_parquet,
        }

        if suffix not in loaders:
            raise ValueError(
                f"지원하지 않는 파일 형식: {suffix}. "
                f"지원 형식: {list(loaders.keys())}"
            )

        df = loaders[suffix](filepath, **kwargs)
        logger.info(f"데이터 로드 완료: {filepath} ({len(df)}행 x {len(df.columns)}열)")

        return df

    def load_multiple_sources(
        self,
        filenames: List[str],
        merge_on: str = "patient_id",
        how: str = "inner",
    ) -> pd.DataFrame:
        """여러 데이터 소스를 병합하여 로드합니다.

        EMR 데이터가 여러 테이블로 분산된 경우 사용합니다.
        예: 환자 기본정보 + CBC 검사결과 + 치료정보

        Args:
            filenames: 병합할 파일 이름 목록
            merge_on: 병합 키 컬럼
            how: 병합 방식 ("inner", "left", "outer")

        Returns:
            병합된 DataFrame
        """
        dfs = [self.load_data(f) for f in filenames]

        # 첫 번째 DataFrame을 기준으로 순차 병합
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on=merge_on, how=how)
            logger.info(f"병합 후 행 수: {len(merged)}")

        return merged

    # ============================================================
    # 결측치 처리
    # ============================================================
    def explore_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 현황을 탐색하여 요약 테이블을 반환합니다.

        Args:
            df: 입력 DataFrame

        Returns:
            결측치 요약 DataFrame (컬럼별 결측 수, 비율)
        """
        missing_count = df.isnull().sum()
        missing_rate = df.isnull().mean()

        summary = pd.DataFrame({
            "결측_수": missing_count,
            "결측_비율": missing_rate,
            "데이터_타입": df.dtypes,
        })

        # 결측 비율 내림차순 정렬
        summary = summary.sort_values("결측_비율", ascending=False)

        # 결측이 있는 컬럼만 표시
        summary_with_missing = summary[summary["결측_수"] > 0]
        if len(summary_with_missing) > 0:
            logger.info(f"결측치가 있는 컬럼 수: {len(summary_with_missing)}")
        else:
            logger.info("결측치가 없습니다.")

        return summary

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Optional[str] = None,
        max_missing_rate: Optional[float] = None,
    ) -> pd.DataFrame:
        """결측치를 처리합니다.

        1단계: 결측률이 높은 변수 제거
        2단계: 선택한 전략에 따라 결측치 대체

        Args:
            df: 입력 DataFrame
            strategy: 결측치 대체 전략 (None이면 config 값 사용)
                - "median": 중앙값 대체
                - "mean": 평균값 대체
                - "knn": KNN 기반 대체
                - "mice": 다중 대체 (MICE)
            max_missing_rate: 최대 허용 결측률 (None이면 config 값 사용)

        Returns:
            결측치가 처리된 DataFrame
        """
        df = df.copy()
        strategy = strategy or self.data_config.missing_strategy
        max_missing_rate = max_missing_rate or self.data_config.max_missing_rate

        # 1단계: 결측률이 높은 변수 제거
        missing_rates = df.isnull().mean()
        cols_to_drop = missing_rates[missing_rates > max_missing_rate].index.tolist()
        if cols_to_drop:
            logger.warning(
                f"결측률 {max_missing_rate*100:.0f}% 초과 변수 제거: {cols_to_drop}"
            )
            df = df.drop(columns=cols_to_drop)

        # 2단계: 수치형 변수 결측치 대체
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if strategy == "median":
            # 중앙값 대체 (이상치에 강건)
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        elif strategy == "mean":
            # 평균값 대체
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        elif strategy == "knn":
            # KNN 기반 대체 (유사 환자의 값 참조)
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        elif strategy == "mice":
            # MICE (Multiple Imputation by Chained Equations)
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            imputer = IterativeImputer(random_state=self.data_config.random_state)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        # 3단계: 범주형 변수 결측치는 최빈값으로 대체
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)

        logger.info(f"결측치 처리 완료 (전략: {strategy})")
        return df

    # ============================================================
    # 이상치 처리
    # ============================================================
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        factor: float = 1.5,
    ) -> pd.DataFrame:
        """이상치를 탐지합니다.

        Args:
            df: 입력 DataFrame
            columns: 이상치 탐지 대상 컬럼 (None이면 수치형 전체)
            method: 탐지 방법 ("iqr" 또는 "zscore")
            factor: IQR 배수 또는 Z-score 임계값

        Returns:
            이상치 여부를 나타내는 boolean DataFrame
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)

        for col in columns:
            if method == "iqr":
                # IQR 방식: Q1 - factor*IQR 미만 또는 Q3 + factor*IQR 초과
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - factor * iqr
                upper = q3 + factor * iqr
                outlier_mask[col] = (df[col] < lower) | (df[col] > upper)

            elif method == "zscore":
                # Z-score 방식: |z| > factor인 경우
                z_scores = (df[col] - df[col].mean()) / df[col].std()
                outlier_mask[col] = z_scores.abs() > factor

        n_outliers = outlier_mask.sum().sum()
        logger.info(f"탐지된 이상치 수: {n_outliers} (방법: {method})")

        return outlier_mask

    def clip_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ) -> pd.DataFrame:
        """이상치를 지정된 백분위수로 클리핑(윈저화)합니다.

        의료 데이터에서는 극단값 제거보다 클리핑이 더 안전합니다.
        실제 관측된 극단 수치일 수 있기 때문입니다.

        Args:
            df: 입력 DataFrame
            columns: 클리핑 대상 컬럼
            lower_percentile: 하한 백분위수
            upper_percentile: 상한 백분위수

        Returns:
            클리핑된 DataFrame
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            lower = df[col].quantile(lower_percentile)
            upper = df[col].quantile(upper_percentile)
            df[col] = df[col].clip(lower, upper)

        return df

    # ============================================================
    # 데이터 분할
    # ============================================================
    def split_data(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터를 학습/검증/테스트 세트로 분할합니다.

        혈액독성 데이터는 클래스 불균형이 심할 수 있으므로
        층화 샘플링을 기본으로 사용합니다.

        Args:
            df: 입력 DataFrame
            target_col: 타겟 컬럼명 (None이면 config의 primary_target 사용)

        Returns:
            (train_df, val_df, test_df) 튜플
        """
        target_col = target_col or self.data_config.primary_target

        # 타겟 변수가 존재하는지 확인
        stratify_col = df[target_col] if (
            self.data_config.stratify and target_col in df.columns
        ) else None

        # 1차 분할: 학습+검증 vs 테스트
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.data_config.test_size,
            random_state=self.data_config.random_state,
            stratify=stratify_col,
        )

        # 2차 분할: 학습 vs 검증
        stratify_col_tv = train_val_df[target_col] if (
            self.data_config.stratify and target_col in train_val_df.columns
        ) else None

        # val_size는 전체 대비 비율이므로 학습+검증 내에서의 비율로 변환
        val_ratio = self.data_config.val_size / (1 - self.data_config.test_size)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=self.data_config.random_state,
            stratify=stratify_col_tv,
        )

        logger.info(
            f"데이터 분할 완료: "
            f"학습={len(train_df)}, 검증={len(val_df)}, 테스트={len(test_df)}"
        )

        # 클래스 분포 확인
        if target_col in df.columns:
            for name, subset in [("학습", train_df), ("검증", val_df), ("테스트", test_df)]:
                pos_rate = subset[target_col].mean()
                logger.info(f"  {name} 양성 비율: {pos_rate:.3f}")

        return train_df, val_df, test_df

    # ============================================================
    # 합성 데이터 생성 (개발/테스트용)
    # ============================================================
    @staticmethod
    def generate_synthetic_data(
        n_patients: int = 200,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """개발 및 테스트를 위한 합성 데이터를 생성합니다.

        실제 EMR 데이터를 사용하기 전에 파이프라인을 검증하기 위해
        임상적으로 그럴듯한 합성 데이터를 생성합니다.

        Args:
            n_patients: 생성할 환자 수
            random_state: 재현성을 위한 시드

        Returns:
            합성 환자 데이터 DataFrame
        """
        rng = np.random.RandomState(random_state)

        data = {
            "patient_id": [f"PT{i:04d}" for i in range(n_patients)],

            # ----- Baseline 임상 변수 -----
            "age": rng.normal(65, 10, n_patients).clip(30, 90).astype(int),
            "sex": rng.choice(["M", "F"], n_patients, p=[0.65, 0.35]),
            "bmi": rng.normal(23, 3.5, n_patients).clip(15, 40).round(1),
            "ecog_ps": rng.choice([0, 1, 2], n_patients, p=[0.3, 0.5, 0.2]),
            "stage": rng.choice(
                ["IIIA", "IIIB", "IIIC"], n_patients, p=[0.4, 0.4, 0.2]
            ),
            "t_stage": rng.choice(
                ["T1", "T2", "T3", "T4"], n_patients, p=[0.1, 0.3, 0.35, 0.25]
            ),
            "n_stage": rng.choice(
                ["N0", "N1", "N2", "N3"], n_patients, p=[0.1, 0.2, 0.5, 0.2]
            ),
            "creatinine": rng.normal(0.9, 0.3, n_patients).clip(0.4, 2.5).round(2),
            "albumin": rng.normal(3.8, 0.5, n_patients).clip(2.0, 5.0).round(1),

            # ----- 치료 변수 -----
            "rt_total_dose": rng.choice([60.0, 63.0, 66.0], n_patients),
            "rt_fraction_dose": rng.choice([2.0, 2.1], n_patients),
            "chemo_regimen": rng.choice(
                ["EP", "TP", "GP", "weekly_paclitaxel"],
                n_patients,
                p=[0.35, 0.3, 0.2, 0.15],
            ),
            "chemo_dose": rng.normal(100, 15, n_patients).clip(60, 130).round(0),
            "chemo_cycles": rng.choice([2, 3, 4], n_patients, p=[0.3, 0.5, 0.2]),
        }

        # ----- CBC 시계열 데이터 (Week 0, 1, 2) -----
        for week in [0, 1, 2]:
            # 주차별 감소 추세 반영 (방사선+항암 효과)
            decay_factor = 1.0 - week * 0.15

            data[f"WBC_week{week}"] = (
                rng.normal(7.0, 2.0, n_patients) * decay_factor
            ).clip(0.5, 20.0).round(2)

            data[f"ANC_week{week}"] = (
                rng.normal(4.5, 1.5, n_patients) * decay_factor
            ).clip(0.2, 15.0).round(2)

            data[f"ALC_week{week}"] = (
                rng.normal(1.8, 0.6, n_patients) * (1.0 - week * 0.2)
            ).clip(0.1, 5.0).round(2)

            # AMC: 핵심 변수 - neutrophil보다 더 빠르게 감소
            data[f"AMC_week{week}"] = (
                rng.normal(0.5, 0.2, n_patients) * (1.0 - week * 0.25)
            ).clip(0.01, 2.0).round(3)

            data[f"PLT_week{week}"] = (
                rng.normal(250, 70, n_patients) * (1.0 - week * 0.08)
            ).clip(20, 500).round(0)

            data[f"Hb_week{week}"] = (
                rng.normal(12.5, 1.5, n_patients) * (1.0 - week * 0.05)
            ).clip(5.0, 18.0).round(1)

        df = pd.DataFrame(data)

        # ----- 타겟 변수 생성 (CTCAE v5.0 기준) -----
        # Grade 3+ neutropenia: ANC < 1.0 at any point during treatment
        # AMC 감소가 빠른 환자일수록 neutropenia 발생 확률 높음 (핵심 가설)
        amc_decline = (
            df["AMC_week0"] - df["AMC_week2"]
        ) / df["AMC_week0"].clip(lower=0.01)

        # 위험 점수 산출 (AMC 감소율 + age + baseline ANC)
        risk_score = (
            amc_decline * 2.0
            + (df["age"] - 50) / 40.0
            - (df["ANC_week0"] - 3.0) / 3.0
            + rng.normal(0, 0.3, n_patients)  # 잡음 추가
        )

        # 약 25% 양성률로 이진 분류 타겟 생성
        threshold = np.percentile(risk_score, 75)
        df["grade3_neutropenia"] = (risk_score >= threshold).astype(int)

        # Grade 3+ anemia: Hb < 8.0
        df["grade3_anemia"] = (
            rng.random(n_patients) < 0.10  # 약 10% 발생률
        ).astype(int)

        # Grade 3+ thrombocytopenia: PLT < 50
        df["grade3_thrombocytopenia"] = (
            rng.random(n_patients) < 0.08  # 약 8% 발생률
        ).astype(int)

        # 일부 결측치 추가 (현실 반영)
        for col in ["albumin", "bmi", "creatinine"]:
            mask = rng.random(n_patients) < 0.05  # 5% 결측
            df.loc[mask, col] = np.nan

        logger.info(
            f"합성 데이터 생성 완료: {n_patients}명, "
            f"neutropenia 양성률={df['grade3_neutropenia'].mean():.2%}"
        )

        return df
