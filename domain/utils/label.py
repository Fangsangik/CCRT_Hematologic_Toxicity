from dataclasses import dataclass
from typing import Optional

from domain.screening.domain.anc_value import ANCValue


@dataclass
class Label :
    """
    치료 중 혈액학적 독성 라벨

    Attributes:
        grade3_neutropenia: CTCAE v6.0 기준 Grade 3 이상 호중구감소증 발생 여부
        source: 라벨 출처 ("emr" = EMR 명시, "cell_count" = ANC 값 기반 판정)
    """

    grade3_neutropenia : bool
    source : str = "cell_count"

    @classmethod
    def from_emr(cls, emr_grade : int) -> Optional["Label"]:
        """
        EMR에 명시된 grade 값으로 라벨을 생성한다.

        Args:
            emr_grade: EMR에 기록된 grade 값 (0~4). None이면 기록 없음.

        Returns:
            Label 인스턴스. EMR에 grade가 없으면(None) None 반환.
        """
        if emr_grade is None :
            return None
        return cls(grade3_neutropenia=emr_grade >= 3, source="emr")

    @classmethod
    def from_cbc_records(cls, records : list) -> "Label":
        """
        CBC 기록의 ANC 값으로 라벨을 생성한다.

        3~6주차 중 ANC < 1.0 x 10³/uL이면 Grade 3+ 판정.

        Args:
            records: 주차별 CBC 기록 목록

        :returns
            생성된 Label 인스턴스
        """
        grade3_neutropenia = any(
            ANCValue(record.anc).is_grade3_plus()
            for record in records
            if record.week in [3, 4, 5, 6]
        )
        return cls(grade3_neutropenia=grade3_neutropenia, source="cell_count")

    @classmethod
    def resolve(cls, records : list, emr_grade : Optional[int] = None) -> "Label":
        """
        EMR grade 우선, 없으면 cell count fallback으로 라벨을 결정한다.

        Args:
            records: 주차별 CBC 기록 목록
            emr_grade: EMR에 명시된 grade (None이면 미기록)

        :returns
            생성된 Label 인스턴스
        """
        emr_label = cls.from_emr(emr_grade)
        if emr_label is not None :
            return emr_label
        return cls.from_cbc_records(records)
