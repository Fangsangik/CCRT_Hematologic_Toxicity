"""절대 호중구 ANC 값 객체."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ANCValue :
    """
    ANC를 나타내는 값 객체

    CTCAE v6.0 기준으로 등급을 평가

    Attributes :
            value : ANC 측정값 (10³/uL)
    """
    value : float

    def is_grade3_plus(self) -> bool:
        """
        Grade 3 이상 호중구감소증 여부를 반환한다.
        Returns :
            CTCAE v6.0 기준 Grade 3 이상이면 True
        """
        return self.ctcae_grade() >= 3

    def ctcae_grade(self) -> int:
        """
        CTCAE v6.0 기준 호중구감소증 등급을 반환한다.
        """

        if self.value < 0.1 :
            return 4
        elif self.value < 0.5 :
            return 3
        elif self.value < 1.0 :
            return 2
        elif self.value < 1.5 :
            return 1
        else :
            return 0