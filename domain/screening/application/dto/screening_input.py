"""ScreeningInput DTO."""
from dataclasses import dataclass

import numpy as np


@dataclass
class ScreeningInput:
    """
    스크리닝 모델 입력 데이터
    y_true: 실제 레이블 (0 또는 1)
    y_prob: 모델이 예측한 양성 클래스의 확률
    min_sensitivity: 모델이 만족해야 하는 최소 민감도 (기본값: 0.85)
    """
    y_true: np.ndarray
    y_prob: np.ndarray
    min_sensitivity: float = 0.85