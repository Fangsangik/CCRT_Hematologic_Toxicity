"""
lstm_model.py - LSTM 기반 혈액독성 예측 모델

CBC 시계열 패턴을 학습하는 LSTM과 Baseline 특성을 결합하는
멀티모달 아키텍처를 구현합니다.

핵심 가설: 치료 초반(Week 1-2) AMC 감소 패턴이
이후 Grade 3+ neutropenia 발생을 예측할 수 있다.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTMPredictor(nn.Module):
    """CBC 시계열 + Baseline 특성을 결합한 LSTM 예측 모델입니다.

    아키텍처:
        1) LSTM 인코더: CBC 시계열 (Week 0→1→2) 패턴 학습
        2) FC 인코더: Baseline 임상+치료 특성 처리
        3) 결합 레이어: 두 표현을 concat 후 최종 예측

    입력:
        - cbc_seq: (batch, seq_length=3, n_cbc_features=6)
        - baseline: (batch, n_baseline_features)

    출력:
        - logits: (batch, 1) - 이진 분류 로짓

    사용 예시:
        model = LSTMPredictor(config.lstm)
        output = model(cbc_seq, baseline)
    """

    def __init__(self, lstm_config):
        """모델을 초기화합니다.

        Args:
            lstm_config: LSTMConfig 인스턴스
        """
        super().__init__()
        self.config = lstm_config

        # ----- LSTM 인코더 -----
        # CBC 시계열의 시간적 패턴을 학습합니다
        # 입력: (batch, seq_length, input_size)
        # 출력: (batch, seq_length, hidden_size * num_directions)
        self.lstm = nn.LSTM(
            input_size=lstm_config.input_size,    # CBC 변수 수 (6)
            hidden_size=lstm_config.hidden_size,   # hidden 차원 (64)
            num_layers=lstm_config.num_layers,     # 레이어 수 (2)
            batch_first=True,                      # (batch, seq, feature) 순서
            dropout=lstm_config.dropout if lstm_config.num_layers > 1 else 0,
            bidirectional=lstm_config.bidirectional,  # False: 시퀀스가 3 time steps로 짧아 양방향 이득 제한적
        )

        # 양방향이면 hidden_size * 2, 단방향이면 hidden_size * 1
        lstm_output_size = lstm_config.hidden_size * (
            2 if lstm_config.bidirectional else 1
        )

        # LSTM 출력에 대한 Attention 메커니즘 (시점별 중요도 학습)
        self.attention = TemporalAttention(lstm_output_size)

        # ----- Baseline FC 인코더 -----
        # 임상+치료 특성을 압축 표현으로 변환합니다
        # LayerNorm 사용: BatchNorm과 달리 batch_size=1에서도 안정적으로 동작
        self.baseline_encoder = nn.Sequential(
            nn.Linear(lstm_config.baseline_input_size, lstm_config.fc_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(lstm_config.fc_hidden_size),
            nn.Dropout(lstm_config.dropout),
        )

        # ----- 결합 및 분류 레이어 -----
        # LSTM 표현 + Baseline 표현 → 최종 예측
        combined_size = lstm_output_size + lstm_config.fc_hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(combined_size, combined_size // 2),
            nn.ReLU(),
            nn.LayerNorm(combined_size // 2),
            nn.Dropout(lstm_config.dropout),
            nn.Linear(combined_size // 2, lstm_config.num_classes),
        )

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """가중치를 Xavier 초기화합니다."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        cbc_seq: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        """순전파를 수행합니다.

        Args:
            cbc_seq: CBC 시계열 (batch, seq_length, n_cbc_features)
            baseline: Baseline 특성 (batch, n_baseline_features)

        Returns:
            logits: 예측 로짓 (batch, 1)
        """
        # 1) LSTM으로 시계열 인코딩
        # lstm_out: (batch, seq_length, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(cbc_seq)

        # 2) Attention으로 시점별 가중 합산
        # attended: (batch, hidden_size)
        attended = self.attention(lstm_out)

        # 3) Baseline 특성 인코딩
        # baseline_repr: (batch, fc_hidden_size)
        baseline_repr = self.baseline_encoder(baseline)

        # 4) 두 표현 결합
        combined = torch.cat([attended, baseline_repr], dim=1)

        # 5) 최종 분류
        logits = self.classifier(combined)

        return logits

    def predict_proba(
        self,
        cbc_seq: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        """예측 확률을 반환합니다.

        Args:
            cbc_seq: CBC 시계열 텐서
            baseline: Baseline 특성 텐서

        Returns:
            확률값 (batch, 1), 0~1 범위
        """
        logits = self.forward(cbc_seq, baseline)
        return torch.sigmoid(logits)


class TemporalAttention(nn.Module):
    """시간 축에 대한 Attention 메커니즘입니다.

    각 시점(Week 0, 1, 2)에 다른 가중치를 부여하여
    예측에 중요한 시점에 집중할 수 있도록 합니다.

    예: AMC가 급격히 감소한 Week 2 시점에 높은 attention 부여
    """

    def __init__(self, hidden_size: int):
        """Attention 레이어를 초기화합니다.

        Args:
            hidden_size: LSTM hidden state 차원
        """
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """시점별 가중치를 계산하고 가중 합산합니다.

        Args:
            lstm_output: LSTM 출력 (batch, seq_length, hidden_size)

        Returns:
            가중 합산된 표현 (batch, hidden_size)
        """
        # 각 시점의 attention score 계산
        # scores: (batch, seq_length, 1)
        scores = self.attention_weights(lstm_output)

        # Softmax로 정규화하여 가중치 생성
        # weights: (batch, seq_length, 1)
        weights = torch.softmax(scores, dim=1)

        # 가중 합산
        # attended: (batch, hidden_size)
        attended = (lstm_output * weights).sum(dim=1)

        return attended


class BaselineLSTMPredictor(nn.Module):
    """Baseline-only 비교 모델 (LSTM 없이 FC만 사용)입니다.

    CBC 시계열 없이 baseline 임상+치료 변수만으로 예측하는 모델입니다.
    LSTM 모델과의 incremental value 비교를 위해 사용됩니다.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.3):
        """모델을 초기화합니다.

        Args:
            input_size: 입력 특성 수
            hidden_size: 은닉 레이어 차원
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """순전파를 수행합니다.

        Args:
            features: 입력 특성 (batch, n_features)

        Returns:
            logits: 예측 로짓 (batch, 1)
        """
        return self.network(features)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """예측 확률을 반환합니다."""
        return torch.sigmoid(self.forward(features))
