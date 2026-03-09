"""
preprocessing.py - EMR 원본 데이터 전처리 모듈

실제 EMR에서 추출된 날짜 기반 CBC 검사 데이터를 처리합니다.
주요 기능:
    1. 검사 날짜 → 치료 Week 변환
    2. Long 형식 → Wide 형식 변환 (환자 1행 구조)
    3. CTCAE v5.0 기준 혈액독성 Grade 자동 계산
    4. 불규칙 검사 간격 보간

EMR 데이터 흐름:
    원본 (Long, 날짜) → Week 매핑 → Wide 변환 → Grade 계산 → 학습 데이터
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# CTCAE v5.0 혈액독성 등급 기준
# ============================================================
# 참고: https://ctep.cancer.gov/protocoldevelopment/electronic_applications/ctc.htm

CTCAE_CRITERIA = {
    "neutropenia": {
        # ANC 기준 (10^3/uL = K/uL)
        "column": "ANC",
        "grades": [
            (4, 0.0, 0.5),      # Grade 4: ANC < 500/mm³
            (3, 0.5, 1.0),      # Grade 3: 500 - 1000/mm³
            (2, 1.0, 1.5),      # Grade 2: 1000 - 1500/mm³
            (1, 1.5, 2.0),      # Grade 1: 1500 - LLN
        ],
    },
    "anemia": {
        # Hemoglobin 기준 (g/dL)
        "column": "Hb",
        "grades": [
            (4, 0.0, 6.5),      # Grade 4: 생명 위협적
            (3, 6.5, 8.0),      # Grade 3: 수혈 필요
            (2, 8.0, 10.0),     # Grade 2: 의학적 유의미
            (1, 10.0, 12.0),    # Grade 1: LLN - 10.0 (여성 기준 약간 다를 수 있음)
        ],
    },
    "thrombocytopenia": {
        # Platelet 기준 (10^3/uL = K/uL)
        "column": "PLT",
        "grades": [
            (4, 0.0, 25.0),     # Grade 4: < 25,000/mm³
            (3, 25.0, 50.0),    # Grade 3: 25,000 - 50,000/mm³
            (2, 50.0, 75.0),    # Grade 2: 50,000 - 75,000/mm³
            (1, 75.0, 150.0),   # Grade 1: 75,000 - LLN
        ],
    },
    "leukopenia": {
        # WBC 기준 (10^3/uL)
        "column": "WBC",
        "grades": [
            (4, 0.0, 1.0),      # Grade 4: < 1,000/mm³
            (3, 1.0, 2.0),      # Grade 3: 1,000 - 2,000/mm³
            (2, 2.0, 3.0),      # Grade 2: 2,000 - 3,000/mm³
            (1, 3.0, 4.0),      # Grade 1: 3,000 - LLN
        ],
    },
    "lymphopenia": {
        # ALC 기준 (10^3/uL)
        "column": "ALC",
        "grades": [
            (4, 0.0, 0.2),      # Grade 4: < 200/mm³
            (3, 0.2, 0.5),      # Grade 3: 200 - 500/mm³
            (2, 0.5, 0.8),      # Grade 2: 500 - 800/mm³
            (1, 0.8, 1.0),      # Grade 1: 800 - LLN
        ],
    },
}


class EMRPreprocessor:
    """EMR 원본 데이터를 학습 가능한 형태로 변환하는 클래스입니다.

    EMR에서 추출된 날짜 기반 CBC 데이터를 처리하여
    주차(Week) 기반 Wide 형식으로 변환하고,
    CTCAE Grade를 자동 계산합니다.

    사용 예시:
        preprocessor = EMRPreprocessor(config)

        # 1) 환자 정보 + CBC 검사 데이터 로드
        patients_df = pd.read_csv("patients.csv")
        cbc_df = pd.read_csv("cbc_results.csv")

        # 2) 날짜 → Week 변환 및 Wide 형식 변환
        wide_df = preprocessor.convert_long_to_wide(cbc_df, patients_df)

        # 3) CTCAE Grade 계산
        wide_df = preprocessor.calculate_ctcae_grades(wide_df)

        # 4) 학습 데이터 완성
        final_df = preprocessor.merge_all(patients_df, wide_df)
    """

    def __init__(self, config):
        """EMRPreprocessor를 초기화합니다.

        Args:
            config: Config 인스턴스
        """
        self.config = config
        self.data_config = config.data

        # Week 할당 기준 (일 단위)
        # Week 0: 치료 시작 전 7일 ~ 치료 시작일
        # Week 1~7: 각각 7일 간격 (±3일 허용)
        self.week_boundaries = {
            0: (-7, 0),     # 치료 시작 전 7일 ~ 당일
            1: (1, 10),     # 치료 시작 후 1~10일
            2: (11, 17),    # 치료 시작 후 11~17일
            3: (18, 24),    # 치료 시작 후 18~24일
            4: (25, 31),    # 치료 시작 후 25~31일
            5: (32, 38),    # 치료 시작 후 32~38일
            6: (39, 45),    # 치료 시작 후 39~45일
            7: (46, 52),    # 치료 시작 후 46~52일
        }

    # ============================================================
    # 날짜 → Week 변환
    # ============================================================
    def assign_treatment_week(
        self,
        cbc_df: pd.DataFrame,
        rt_start_col: str = "rt_start_date",
        exam_date_col: str = "exam_date",
        patient_id_col: str = "patient_id",
    ) -> pd.DataFrame:
        """CBC 검사 날짜를 치료 Week으로 변환합니다.

        방사선 치료 시작일을 기준으로 각 CBC 검사가
        몇 주차에 해당하는지 계산합니다.

        Args:
            cbc_df: CBC 검사 데이터 (Long 형식)
                필수 컬럼: patient_id, exam_date, WBC, ANC, ALC, AMC, PLT, Hb
            rt_start_col: 방사선 치료 시작일 컬럼명
            exam_date_col: CBC 검사 날짜 컬럼명
            patient_id_col: 환자 ID 컬럼명

        Returns:
            treatment_week 컬럼이 추가된 DataFrame
        """
        df = cbc_df.copy()

        # 날짜 컬럼 파싱
        df[exam_date_col] = pd.to_datetime(df[exam_date_col])
        df[rt_start_col] = pd.to_datetime(df[rt_start_col])

        # 치료 시작일로부터의 일수 차이 계산
        df["days_from_rt_start"] = (
            df[exam_date_col] - df[rt_start_col]
        ).dt.days

        # 일수를 Week으로 매핑
        df["treatment_week"] = df["days_from_rt_start"].apply(
            self._map_day_to_week
        )

        # Week 할당이 안 된 검사 건수 확인
        unmapped = df["treatment_week"].isna().sum()
        if unmapped > 0:
            logger.warning(
                f"Week 할당 불가 검사: {unmapped}건 "
                f"(전체 {len(df)}건 중 {unmapped/len(df)*100:.1f}%)"
            )

        # Week이 할당된 데이터만 유지
        df = df.dropna(subset=["treatment_week"])
        df["treatment_week"] = df["treatment_week"].astype(int)

        logger.info(
            f"Week 할당 완료: "
            + ", ".join(
                f"Week {w}={len(df[df['treatment_week']==w])}건"
                for w in sorted(df["treatment_week"].unique())
            )
        )

        return df

    def _map_day_to_week(self, day: int) -> Optional[int]:
        """치료 시작일 기준 일수를 Week으로 매핑합니다.

        Args:
            day: 치료 시작일로부터의 일수

        Returns:
            Week 번호 (매핑 불가 시 None)
        """
        for week, (start, end) in self.week_boundaries.items():
            if start <= day <= end:
                return week
        return None

    # ============================================================
    # Long → Wide 형식 변환
    # ============================================================
    def convert_long_to_wide(
        self,
        cbc_df: pd.DataFrame,
        patient_id_col: str = "patient_id",
        week_col: str = "treatment_week",
        agg_method: str = "closest_to_target",
    ) -> pd.DataFrame:
        """Long 형식 CBC 데이터를 Wide 형식으로 변환합니다.

        같은 Week에 여러 검사가 있는 경우 처리 전략:
            - "closest_to_target": 목표 시점에 가장 가까운 검사 사용
            - "mean": 평균값 사용
            - "worst": 가장 나쁜(낮은) 값 사용

        Args:
            cbc_df: Week이 할당된 CBC 데이터 (Long 형식)
            patient_id_col: 환자 ID 컬럼명
            week_col: Week 컬럼명
            agg_method: 같은 Week 내 복수 검사 처리 방법

        Returns:
            Wide 형식 DataFrame (환자 1행, WBC_week0, WBC_week1, ... 형태)
        """
        cbc_features = self.data_config.cbc_features  # [WBC, ANC, ALC, AMC, PLT, Hb]
        timepoints = self.data_config.cbc_timepoints  # [0, 1, 2]

        # 사용 가능한 CBC 컬럼만 필터링
        available_features = [f for f in cbc_features if f in cbc_df.columns]

        if agg_method == "mean":
            # 같은 환자-Week의 평균값 사용
            grouped = cbc_df.groupby([patient_id_col, week_col])[available_features].mean()
        elif agg_method == "worst":
            # 가장 낮은(나쁜) 값 사용
            grouped = cbc_df.groupby([patient_id_col, week_col])[available_features].min()
        elif agg_method == "closest_to_target":
            # 목표 시점에 가장 가까운 검사 선택
            # (예: Week 1 목표일=Day 7, 실제 검사일이 Day 5와 Day 8이면 Day 8 선택)
            target_days = {0: 0, 1: 7, 2: 14, 3: 21, 4: 28, 5: 35, 6: 42, 7: 49}
            grouped = self._select_closest_exam(
                cbc_df, patient_id_col, week_col, available_features, target_days
            )
        else:
            raise ValueError(f"알 수 없는 집계 방법: {agg_method}")

        # Wide 형식으로 피벗
        wide_data = {}
        patient_ids = cbc_df[patient_id_col].unique()

        for pid in patient_ids:
            row = {patient_id_col: pid}
            for week in timepoints:
                for feature in available_features:
                    col_name = f"{feature}_week{week}"
                    try:
                        if isinstance(grouped, pd.DataFrame) and (pid, week) in grouped.index:
                            row[col_name] = grouped.loc[(pid, week), feature]
                        else:
                            row[col_name] = np.nan  # 해당 Week 검사가 없는 경우
                    except (KeyError, TypeError):
                        row[col_name] = np.nan
            wide_data[pid] = row

        wide_df = pd.DataFrame(wide_data.values())

        # 결측 현황 보고
        for week in timepoints:
            week_cols = [f"{f}_week{week}" for f in available_features]
            available_cols = [c for c in week_cols if c in wide_df.columns]
            missing_rate = wide_df[available_cols].isnull().any(axis=1).mean()
            logger.info(f"Week {week} CBC 결측 환자: {missing_rate*100:.1f}%")

        logger.info(
            f"Wide 변환 완료: {len(wide_df)}명, "
            f"{len(wide_df.columns)-1}개 컬럼"
        )

        return wide_df

    def _select_closest_exam(
        self,
        cbc_df: pd.DataFrame,
        patient_id_col: str,
        week_col: str,
        features: List[str],
        target_days: Dict[int, int],
    ) -> pd.DataFrame:
        """목표 시점에 가장 가까운 검사를 선택합니다.

        Args:
            cbc_df: CBC 데이터
            patient_id_col: 환자 ID 컬럼
            week_col: Week 컬럼
            features: CBC 변수 목록
            target_days: Week별 목표 일수

        Returns:
            환자-Week별 대표 CBC 값 DataFrame
        """
        df = cbc_df.copy()

        # 각 검사의 목표일과의 거리 계산
        df["target_day"] = df[week_col].map(target_days)
        df["distance_to_target"] = abs(
            df["days_from_rt_start"] - df["target_day"]
        )

        # 환자-Week별로 가장 가까운 검사 선택
        idx = df.groupby([patient_id_col, week_col])["distance_to_target"].idxmin()
        selected = df.loc[idx].set_index([patient_id_col, week_col])

        return selected[features]

    # ============================================================
    # CTCAE Grade 자동 계산
    # ============================================================
    def calculate_ctcae_grades(
        self,
        df: pd.DataFrame,
        toxicity_types: Optional[List[str]] = None,
        observation_weeks: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """CBC 수치에서 CTCAE v5.0 Grade를 자동 계산합니다.

        전체 치료 기간(Week 0~마지막) 중 최악의 Grade를 산출하여
        이진 타겟 변수(Grade 3 이상 여부)를 생성합니다.

        Args:
            df: Wide 형식 CBC 데이터
            toxicity_types: 계산할 독성 유형 (None이면 전체)
                ["neutropenia", "anemia", "thrombocytopenia", "leukopenia", "lymphopenia"]
            observation_weeks: Grade 관찰 기간 (None이면 전체 Week)

        Returns:
            Grade 컬럼이 추가된 DataFrame
        """
        df = df.copy()

        if toxicity_types is None:
            toxicity_types = list(CTCAE_CRITERIA.keys())

        if observation_weeks is None:
            # 데이터에 존재하는 모든 Week 사용
            observation_weeks = self.data_config.cbc_timepoints

        for tox_type in toxicity_types:
            criteria = CTCAE_CRITERIA.get(tox_type)
            if criteria is None:
                logger.warning(f"알 수 없는 독성 유형: {tox_type}")
                continue

            cbc_col = criteria["column"]
            grade_thresholds = criteria["grades"]  # [(grade, lower, upper), ...]

            # 각 Week의 Grade 계산
            week_grades = []
            for week in observation_weeks:
                value_col = f"{cbc_col}_week{week}"
                if value_col not in df.columns:
                    continue

                grade_col = f"{tox_type}_grade_week{week}"
                df[grade_col] = df[value_col].apply(
                    lambda v: self._assign_grade(v, grade_thresholds)
                )
                week_grades.append(grade_col)

            # 전체 관찰 기간 중 최고(최악) Grade
            if week_grades:
                df[f"max_grade_{tox_type}"] = df[week_grades].max(axis=1)

                # 이진 타겟: Grade 3 이상 여부
                df[f"grade3_{tox_type}"] = (
                    df[f"max_grade_{tox_type}"] >= 3
                ).astype(int)

                # 통계 로깅
                grade_dist = df[f"max_grade_{tox_type}"].value_counts().sort_index()
                grade3_rate = df[f"grade3_{tox_type}"].mean()
                logger.info(
                    f"[{tox_type}] Grade 분포: {grade_dist.to_dict()}, "
                    f"Grade 3+ 발생률: {grade3_rate*100:.1f}%"
                )

        return df

    @staticmethod
    def _assign_grade(value: float, thresholds: List[Tuple]) -> int:
        """CBC 수치에 CTCAE Grade를 할당합니다.

        Args:
            value: CBC 수치
            thresholds: [(grade, lower, upper), ...] 내림차순 정렬

        Returns:
            CTCAE Grade (0-4)
        """
        if pd.isna(value):
            return 0  # 결측은 Grade 0으로 처리

        for grade, lower, upper in thresholds:
            if lower <= value < upper:
                return grade

        return 0  # 정상 범위이면 Grade 0

    # ============================================================
    # 불규칙 검사 간격 보간
    # ============================================================
    def interpolate_missing_weeks(
        self,
        df: pd.DataFrame,
        method: str = "linear",
    ) -> pd.DataFrame:
        """누락된 Week의 CBC 값을 보간합니다.

        모든 환자가 매주 검사를 받지는 않으므로,
        누락된 시점의 값을 인접 시점에서 보간합니다.

        Args:
            df: Wide 형식 CBC 데이터
            method: 보간 방법
                - "linear": 선형 보간
                - "ffill": 이전 값 사용 (Forward Fill)
                - "nearest": 가장 가까운 값

        Returns:
            보간된 DataFrame
        """
        df = df.copy()
        timepoints = self.data_config.cbc_timepoints

        for feature in self.data_config.cbc_features:
            cols = [f"{feature}_week{t}" for t in timepoints]
            available_cols = [c for c in cols if c in df.columns]

            if len(available_cols) < 2:
                continue

            # 각 환자별로 시계열 보간
            for idx in df.index:
                values = df.loc[idx, available_cols].values.astype(float)

                if np.all(np.isnan(values)):
                    continue  # 전부 결측이면 건너뜀

                if np.any(np.isnan(values)):
                    # pandas Series로 변환 후 보간
                    series = pd.Series(values, index=range(len(values)))

                    if method == "linear":
                        interpolated = series.interpolate(method="linear")
                    elif method == "ffill":
                        interpolated = series.ffill().bfill()
                    elif method == "nearest":
                        interpolated = series.interpolate(method="nearest")
                    else:
                        interpolated = series.interpolate(method="linear")

                    df.loc[idx, available_cols] = interpolated.values

        # 보간 후 결측 현황
        total_missing = 0
        for feature in self.data_config.cbc_features:
            for t in timepoints:
                col = f"{feature}_week{t}"
                if col in df.columns:
                    total_missing += df[col].isna().sum()

        logger.info(f"보간 완료 (방법: {method}), 잔여 결측: {total_missing}건")
        return df

    # ============================================================
    # 전체 전처리 파이프라인
    # ============================================================
    def run_full_pipeline(
        self,
        patients_df: pd.DataFrame,
        cbc_df: pd.DataFrame,
        patient_id_col: str = "patient_id",
        rt_start_col: str = "rt_start_date",
        exam_date_col: str = "exam_date",
    ) -> pd.DataFrame:
        """EMR 원본 데이터를 학습용 데이터로 변환하는 전체 파이프라인입니다.

        단계:
            1. 검사 날짜 → Week 할당
            2. Long → Wide 형식 변환
            3. 결측 Week 보간
            4. CTCAE Grade 계산
            5. 환자 정보 병합

        Args:
            patients_df: 환자 기본정보 DataFrame
                필수 컬럼: patient_id, age, sex, ..., rt_start_date
            cbc_df: CBC 검사 결과 DataFrame (Long 형식)
                필수 컬럼: patient_id, exam_date, WBC, ANC, ALC, AMC, PLT, Hb
            patient_id_col: 환자 ID 컬럼명
            rt_start_col: 방사선 치료 시작일 컬럼명
            exam_date_col: CBC 검사일 컬럼명

        Returns:
            학습 준비가 완료된 DataFrame
        """
        logger.info("=" * 60)
        logger.info("EMR 전처리 파이프라인 시작")
        logger.info("=" * 60)

        # 1단계: RT 시작일 정보를 CBC 데이터에 병합
        logger.info("1단계: 치료 시작일 매핑")
        cbc_with_rt = cbc_df.merge(
            patients_df[[patient_id_col, rt_start_col]],
            on=patient_id_col,
            how="left",
        )

        # 2단계: 검사 날짜 → Week 할당
        logger.info("2단계: 검사 날짜 → Week 변환")
        cbc_with_week = self.assign_treatment_week(
            cbc_with_rt,
            rt_start_col=rt_start_col,
            exam_date_col=exam_date_col,
            patient_id_col=patient_id_col,
        )

        # 3단계: Long → Wide 변환
        logger.info("3단계: Long → Wide 형식 변환")
        wide_df = self.convert_long_to_wide(
            cbc_with_week,
            patient_id_col=patient_id_col,
        )

        # 4단계: 결측 Week 보간
        logger.info("4단계: 결측 Week 보간")
        wide_df = self.interpolate_missing_weeks(wide_df)

        # 5단계: CTCAE Grade 계산
        logger.info("5단계: CTCAE Grade 계산")
        wide_df = self.calculate_ctcae_grades(wide_df)

        # 6단계: 환자 정보 병합
        logger.info("6단계: 환자 정보 병합")
        # patients_df에서 rt_start_date 등 날짜 컬럼 제외하고 병합
        patient_cols = [
            c for c in patients_df.columns
            if c != rt_start_col and c != exam_date_col
        ]
        final_df = wide_df.merge(
            patients_df[patient_cols],
            on=patient_id_col,
            how="inner",
        )

        logger.info(
            f"전처리 완료: {len(final_df)}명, {len(final_df.columns)}개 변수"
        )
        logger.info("=" * 60)

        return final_df

    # ============================================================
    # EMR 추출용 데이터 템플릿 생성
    # ============================================================
    @staticmethod
    def generate_data_template(
        output_dir: str = ".",
        n_example_rows: int = 3,
    ):
        """EMR 데이터 추출을 위한 CSV 템플릿을 생성합니다.

        IRB 데이터 요청 시 제공할 수 있는 형식 가이드입니다.

        Args:
            output_dir: 템플릿 저장 디렉토리
            n_example_rows: 예시 행 수
        """
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ----- 1) 환자 기본정보 템플릿 -----
        patients_template = pd.DataFrame({
            "patient_id": ["PT0001", "PT0002", "PT0003"][:n_example_rows],
            "age": [65, 58, 72][:n_example_rows],
            "sex": ["M", "F", "M"][:n_example_rows],
            "bmi": [23.5, 21.2, 25.8][:n_example_rows],
            "ecog_ps": [1, 0, 1][:n_example_rows],
            "stage": ["IIIA", "IIIB", "IIIA"][:n_example_rows],
            "t_stage": ["T2", "T3", "T4"][:n_example_rows],
            "n_stage": ["N2", "N2", "N1"][:n_example_rows],
            "creatinine": [0.9, 0.7, 1.1][:n_example_rows],
            "albumin": [3.8, 4.2, 3.5][:n_example_rows],
            "rt_start_date": ["2024-01-15", "2024-02-01", "2024-03-10"][:n_example_rows],
            "rt_total_dose": [60.0, 66.0, 63.0][:n_example_rows],
            "rt_fraction_dose": [2.0, 2.0, 2.1][:n_example_rows],
            "chemo_regimen": ["EP", "TP", "weekly_paclitaxel"][:n_example_rows],
            "chemo_dose": [100, 85, 60][:n_example_rows],
            "chemo_cycles": [2, 3, 4][:n_example_rows],
        })

        patients_path = output_dir / "template_patients.csv"
        patients_template.to_csv(patients_path, index=False, encoding="utf-8-sig")

        # ----- 2) CBC 검사 결과 템플릿 (Long 형식) -----
        cbc_rows = []
        example_patients = ["PT0001", "PT0002", "PT0003"][:n_example_rows]
        example_rt_starts = ["2024-01-15", "2024-02-01", "2024-03-10"][:n_example_rows]

        for pid, rt_start in zip(example_patients, example_rt_starts):
            rt_date = pd.Timestamp(rt_start)
            # 각 환자에 대해 치료 전, 1주, 2주 검사
            for day_offset, desc in [(-3, "치료 전"), (6, "1주차"), (13, "2주차")]:
                exam_date = rt_date + pd.Timedelta(days=day_offset)
                cbc_rows.append({
                    "patient_id": pid,
                    "exam_date": exam_date.strftime("%Y-%m-%d"),
                    "WBC": round(np.random.uniform(3.0, 10.0), 2),
                    "ANC": round(np.random.uniform(1.5, 7.0), 2),
                    "ALC": round(np.random.uniform(0.5, 3.0), 2),
                    "AMC": round(np.random.uniform(0.2, 1.0), 3),
                    "PLT": round(np.random.uniform(100, 350), 0),
                    "Hb": round(np.random.uniform(10.0, 16.0), 1),
                })

        cbc_template = pd.DataFrame(cbc_rows)
        cbc_path = output_dir / "template_cbc_results.csv"
        cbc_template.to_csv(cbc_path, index=False, encoding="utf-8-sig")

        logger.info(f"템플릿 생성 완료:")
        logger.info(f"  환자 정보: {patients_path}")
        logger.info(f"  CBC 검사: {cbc_path}")

        # ----- 3) 데이터 사전 (Data Dictionary) -----
        data_dict = pd.DataFrame([
            # 환자 정보
            {"파일": "patients", "변수명": "patient_id", "설명": "환자 고유 번호", "타입": "문자열", "예시": "PT0001", "필수": "Y"},
            {"파일": "patients", "변수명": "age", "설명": "진단 시 나이", "타입": "정수", "예시": "65", "필수": "Y"},
            {"파일": "patients", "변수명": "sex", "설명": "성별", "타입": "M/F", "예시": "M", "필수": "Y"},
            {"파일": "patients", "변수명": "bmi", "설명": "체질량지수 (kg/m²)", "타입": "실수", "예시": "23.5", "필수": "N"},
            {"파일": "patients", "변수명": "ecog_ps", "설명": "ECOG Performance Status", "타입": "정수 (0-4)", "예시": "1", "필수": "Y"},
            {"파일": "patients", "변수명": "stage", "설명": "AJCC 병기", "타입": "문자열", "예시": "IIIA", "필수": "Y"},
            {"파일": "patients", "변수명": "t_stage", "설명": "T 병기", "타입": "문자열", "예시": "T2", "필수": "Y"},
            {"파일": "patients", "변수명": "n_stage", "설명": "N 병기", "타입": "문자열", "예시": "N2", "필수": "Y"},
            {"파일": "patients", "변수명": "creatinine", "설명": "혈청 크레아티닌 (mg/dL)", "타입": "실수", "예시": "0.9", "필수": "N"},
            {"파일": "patients", "변수명": "albumin", "설명": "혈청 알부민 (g/dL)", "타입": "실수", "예시": "3.8", "필수": "N"},
            {"파일": "patients", "변수명": "rt_start_date", "설명": "방사선 치료 시작일", "타입": "YYYY-MM-DD", "예시": "2024-01-15", "필수": "Y"},
            {"파일": "patients", "변수명": "rt_total_dose", "설명": "방사선 총 선량 (Gy)", "타입": "실수", "예시": "60.0", "필수": "Y"},
            {"파일": "patients", "변수명": "rt_fraction_dose", "설명": "분할 선량 (Gy)", "타입": "실수", "예시": "2.0", "필수": "N"},
            {"파일": "patients", "변수명": "chemo_regimen", "설명": "항암 레지멘", "타입": "문자열", "예시": "EP, TP, GP 등", "필수": "Y"},
            {"파일": "patients", "변수명": "chemo_dose", "설명": "항암 용량 (mg/m²)", "타입": "실수", "예시": "100", "필수": "N"},
            {"파일": "patients", "변수명": "chemo_cycles", "설명": "항암 주기 수", "타입": "정수", "예시": "2", "필수": "N"},
            # CBC 검사
            {"파일": "cbc_results", "변수명": "patient_id", "설명": "환자 고유 번호", "타입": "문자열", "예시": "PT0001", "필수": "Y"},
            {"파일": "cbc_results", "변수명": "exam_date", "설명": "CBC 검사 날짜", "타입": "YYYY-MM-DD", "예시": "2024-01-12", "필수": "Y"},
            {"파일": "cbc_results", "변수명": "WBC", "설명": "백혈구 수 (10³/μL)", "타입": "실수", "예시": "7.5", "필수": "Y"},
            {"파일": "cbc_results", "변수명": "ANC", "설명": "절대호중구수 (10³/μL)", "타입": "실수", "예시": "4.2", "필수": "Y"},
            {"파일": "cbc_results", "변수명": "ALC", "설명": "절대림프구수 (10³/μL)", "타입": "실수", "예시": "1.8", "필수": "Y"},
            {"파일": "cbc_results", "변수명": "AMC", "설명": "절대단핵구수 (10³/μL)", "타입": "실수", "예시": "0.5", "필수": "Y"},
            {"파일": "cbc_results", "변수명": "PLT", "설명": "혈소판 수 (10³/μL)", "타입": "실수", "예시": "250", "필수": "Y"},
            {"파일": "cbc_results", "변수명": "Hb", "설명": "헤모글로빈 (g/dL)", "타입": "실수", "예시": "13.5", "필수": "Y"},
        ])

        dict_path = output_dir / "data_dictionary.csv"
        data_dict.to_csv(dict_path, index=False, encoding="utf-8-sig")

        logger.info(f"  데이터 사전: {dict_path}")

        return patients_path, cbc_path, dict_path
