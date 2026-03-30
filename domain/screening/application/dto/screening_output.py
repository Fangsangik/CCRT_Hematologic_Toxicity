"""Screening Output DTO"""
from dataclasses import dataclass


@dataclass
class ScreeningOutput :
    """스크리닝 모델 평가 결과."""
    threshold: float  # Screening시에만 적용
    sensitivity: float
    specificity: float