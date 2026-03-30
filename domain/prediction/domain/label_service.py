"""label service"""
import logging
import pandas as pd
from typing import Optional, List

from domain.utils.domain.cbc_record import CBCRecord
from domain.utils.label import Label

_LABEL_WEEKS = [3, 4, 5, 6]
_ANC_GRADE3_THRESHOLD = 0.5

logger = logging.getLogger(__name__)

class LabelService :
    """라벨 생성 도메인 서비스입니다."""

    @staticmethod
    def generate_label(cbc_records : List[CBCRecord], emr_grade : Optional[int] = None) -> Label :
        """CBC 기록으로부터 라벨을 생성하는 메서드입니다."""
        return Label.resolve(cbc_records, emr_grade=emr_grade)

    @staticmethod
    def ensure_label(df : pd.DataFrame) -> pd.DataFrame :
        """
        Dataframe에 grade3_neutropenia 컬럼 보장

        이미 존재하면 그대로 사용하고,
        없으면 ANC 컬럼들에서 ANC < 1.0 기준으로 Grade 3+ 여부를 판정한다.

        Args:
            df: ANC_week{n} 컬럼을 포함하는 DataFrame

        Returns:
            grade3_neutropenia 컬럼이 보장된 DataFrame
        """
        df = df.copy()

        if "grade3_neutropenia" in df.columns :
            logger.info("grade3_neutropenia 컬럼 존재 -> EMR 값 사용")
            return df

        # ANC_week{n} 컬럼에서 Grade 3+ 판정
        anc_cols = [
            f"ANC_week{w}" for w in _LABEL_WEEKS
            if f"ANC_week{w}" in df.columns
        ]

        if not anc_cols :
            raise ValueError("grade3_neutropenia 컬럼이 없고, ANC_week{n} 컬럼도 존재하지 않습니다.")

        df["grade3_neutropenia"] = (df[anc_cols].lt(_ANC_GRADE3_THRESHOLD).any(axis=1).astype(int))
        n_positive = df["grade3_neutropenia"].sum()
        logger.info(
            f"grade3_neutropenia 컬럼 없음 -> ANC < {_ANC_GRADE3_THRESHOLD} 기준 판정 "
            f"({n_positive}/{len(df)}명 Grade 3+)"
        )
        return df