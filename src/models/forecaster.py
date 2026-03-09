"""
forecaster.py - CBC 시계열 예측 모델

Week 0-2 CBC 데이터를 입력받아 Week 3-7 CBC 값을 예측하는
LSTM 기반 Forecaster입니다.

V1 (CBCForecaster): 단순 LSTM 인코더 + FC 디코더
V2 (CBCForecasterV2): Seq2Seq + Attention + Autoregressive + Baseline + Delta

용도:
    조기 예측 시나리오에서 미래 CBC 값을 예측하여,
    분류 모델에 전체 시퀀스를 제공하는 파이프라인에 사용됩니다.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class CBCForecaster(nn.Module):
    """Week 0-2 CBC → Week 3-7 CBC 예측 모델입니다.

    아키텍처:
        1) LSTM 인코더: Week 0-2 시계열 패턴 인코딩
        2) FC 디코더: 인코딩된 표현에서 Week 3-7 CBC 값 예측

    입력:
        - cbc_input: (batch, 3, n_cbc_features) — Week 0-2 CBC
    출력:
        - cbc_predicted: (batch, 5, n_cbc_features) — Week 3-7 예측 CBC
    """

    def __init__(
        self,
        n_cbc_features: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_forecast_weeks: int = 5,
    ):
        super().__init__()
        self.n_cbc_features = n_cbc_features
        self.n_forecast_weeks = n_forecast_weeks
        self.hidden_size = hidden_size

        # LSTM 인코더: Week 0-2 시계열 패턴 학습
        self.encoder = nn.LSTM(
            input_size=n_cbc_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # FC 디코더: hidden state → Week 3-7 CBC 값 예측
        output_size = n_forecast_weeks * n_cbc_features  # 5 * 6 = 30
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

        self._init_weights()

    def _init_weights(self):
        """가중치를 Xavier 초기화합니다."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, cbc_input: torch.Tensor) -> torch.Tensor:
        """Week 0-2 CBC에서 Week 3-7 CBC를 예측합니다.

        Args:
            cbc_input: (batch, 3, n_cbc_features) — Week 0-2 CBC

        Returns:
            (batch, 5, n_cbc_features) — Week 3-7 예측 CBC
        """
        # LSTM 인코딩
        _, (h_n, _) = self.encoder(cbc_input)
        # 마지막 레이어의 hidden state 사용
        hidden = h_n[-1]  # (batch, hidden_size)

        # FC 디코딩 → (batch, 5 * 6)
        output = self.decoder(hidden)

        # reshape → (batch, 5, 6)
        output = output.view(-1, self.n_forecast_weeks, self.n_cbc_features)

        return output


class ForecastDataset(Dataset):
    """CBC 시계열 예측 학습용 Dataset입니다.

    입력: Week 0-2 CBC (scaled)
    타겟: Week 3-7 CBC (scaled)
    """

    def __init__(
        self,
        cbc_sequences: np.ndarray,
        input_weeks: int = 3,
    ):
        """
        Args:
            cbc_sequences: 전체 CBC 시퀀스 (n_samples, 8, 6) — scaled
            input_weeks: 입력으로 사용할 주차 수 (기본 3 = Week 0-2)
        """
        self.cbc_input = torch.FloatTensor(cbc_sequences[:, :input_weeks, :])
        self.cbc_target = torch.FloatTensor(cbc_sequences[:, input_weeks:, :])

    def __len__(self) -> int:
        return len(self.cbc_input)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input": self.cbc_input[idx],   # (3, 6)
            "target": self.cbc_target[idx],  # (5, 6)
        }


class ForecasterTrainer:
    """CBCForecaster의 학습 루프를 관리합니다."""

    def __init__(
        self,
        model: CBCForecaster,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
    ):
        self.device = self._get_device(device)
        self.model = model.to(self.device)

        self.optimizer = Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=7,
        )
        self.criterion = nn.MSELoss()

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
        }

    def _get_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 200,
        patience: int = 20,
    ) -> Dict[str, List[float]]:
        """Forecaster를 학습합니다.

        Args:
            train_loader: 학습 DataLoader
            val_loader: 검증 DataLoader
            num_epochs: 최대 에폭 수
            patience: Early stopping patience

        Returns:
            학습 히스토리 딕셔너리
        """
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # ----- 학습 -----
            self.model.train()
            train_losses = []
            for batch in train_loader:
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            # ----- 검증 -----
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["input"].to(self.device)
                    targets = batch["target"].to(self.device)
                    predictions = self.model(inputs)
                    loss = self.criterion(predictions, targets)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            self.scheduler.step(val_loss)

            if epoch % 20 == 0:
                logger.info(
                    f"Forecaster Epoch {epoch:3d}/{num_epochs} | "
                    f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Forecaster Early stopping at epoch {epoch}")
                    break

        # 최적 가중치 복원
        if best_state is not None:
            self.model.load_state_dict(best_state)

        logger.info(f"Forecaster 학습 완료 (best val MSE: {best_val_loss:.6f})")
        return self.history

    def predict(self, cbc_input: np.ndarray) -> np.ndarray:
        """Week 0-2 CBC에서 Week 3-7 CBC를 예측합니다.

        Args:
            cbc_input: (n_samples, 3, 6) — scaled Week 0-2 CBC

        Returns:
            (n_samples, 5, 6) — 예측된 Week 3-7 CBC (scaled)
        """
        self.model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(cbc_input).to(self.device)
            predictions = self.model(inputs)
            return predictions.cpu().numpy()


# ============================================================
# V2: 개선된 Forecaster
# ============================================================


class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) Attention.

    디코더의 현재 hidden state와 인코더의 모든 출력을 비교하여
    관련성 높은 시점에 가중치를 부여합니다.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: (batch, hidden_size)
            encoder_outputs: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size)
            weights: (batch, seq_len, 1)
        """
        # (batch, 1, hidden)
        decoder_hidden = decoder_hidden.unsqueeze(1)

        # (batch, seq_len, 1)
        scores = self.v(torch.tanh(
            self.W1(decoder_hidden) + self.W2(encoder_outputs)
        ))

        weights = torch.softmax(scores, dim=1)
        context = (weights * encoder_outputs).sum(dim=1)  # (batch, hidden)

        return context, weights


class CBCForecasterV2(nn.Module):
    """개선된 CBC Forecaster — Seq2Seq + Attention + Autoregressive + Baseline + Delta.

    개선 사항:
        1) Seq2Seq + Attention: 인코더 전체 출력 활용 (정보 병목 해소)
        2) Autoregressive 디코더: Week별 순차 예측 (시간 의존성 반영)
        3) Baseline 조건부: 환자 특성(나이, 항암제 등) 반영 (개인화)
        4) Delta 예측: 절대값 대신 변화량 예측 (학습 안정성)
        5) Teacher Forcing: 학습 시 실제값 피드백 (수렴 가속)

    입력:
        - cbc_input: (batch, 3, n_cbc_features) — Week 0-2 CBC
        - baseline: (batch, n_baseline_features) — 환자 Baseline 특성
    출력:
        - cbc_predicted: (batch, 5, n_cbc_features) — Week 3-7 예측 CBC
    """

    def __init__(
        self,
        n_cbc_features: int = 6,
        n_baseline_features: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_forecast_weeks: int = 5,
    ):
        super().__init__()
        self.n_cbc_features = n_cbc_features
        self.n_forecast_weeks = n_forecast_weeks
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # --- Encoder LSTM ---
        self.encoder = nn.LSTM(
            input_size=n_cbc_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # --- Baseline 인코더 ---
        baseline_dim = hidden_size // 2  # 32: 파라미터 절약
        self.baseline_encoder = nn.Sequential(
            nn.Linear(n_baseline_features, baseline_dim),
            nn.ReLU(),
            nn.LayerNorm(baseline_dim),
        )

        # --- Bahdanau Attention ---
        self.attention = BahdanauAttention(hidden_size)

        # --- Decoder LSTM (1 layer, 경량화) ---
        # Input: prev_cbc(6) + context(hidden) + baseline(baseline_dim)
        decoder_input_size = n_cbc_features + hidden_size + baseline_dim
        self.decoder = nn.LSTMCell(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
        )

        # --- Output projection: hidden → delta_cbc ---
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_cbc_features),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        cbc_input: torch.Tensor,
        baseline: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            cbc_input: (batch, 3, 6) — Week 0-2 CBC
            baseline: (batch, n_baseline) — 환자 Baseline 특성
            target: (batch, 5, 6) — 학습 시 실제 Week 3-7 (Teacher Forcing용)
            teacher_forcing_ratio: Teacher Forcing 비율 (0=자가회귀만, 1=항상 실제값)

        Returns:
            (batch, 5, 6) — Week 3-7 예측 CBC
        """
        # 1. Encode input sequence
        encoder_outputs, (h_n, c_n) = self.encoder(cbc_input)
        # encoder_outputs: (batch, 3, hidden_size)

        # 2. Encode baseline
        baseline_repr = self.baseline_encoder(baseline)  # (batch, baseline_dim)

        # 3. Decoder 초기 상태: 인코더 마지막 레이어의 hidden/cell
        decoder_h = h_n[-1]  # (batch, hidden_size)
        decoder_c = c_n[-1]  # (batch, hidden_size)

        # 4. Autoregressive decoding
        last_cbc = cbc_input[:, -1, :]  # Week 2 값 (batch, 6)
        predictions = []

        for t in range(self.n_forecast_weeks):
            # Attention: 디코더 hidden으로 인코더 출력에 attend
            context, _ = self.attention(decoder_h, encoder_outputs)
            # context: (batch, hidden_size)

            # Decoder input: [이전 CBC, context, baseline]
            decoder_input = torch.cat([last_cbc, context, baseline_repr], dim=1)

            # LSTMCell step
            decoder_h, decoder_c = self.decoder(decoder_input, (decoder_h, decoder_c))

            # Delta 예측: 이전 시점 대비 변화량
            delta = self.output_proj(decoder_h)  # (batch, 6)
            predicted_cbc = last_cbc + delta

            predictions.append(predicted_cbc)

            # 다음 step 입력 결정
            if target is not None and random.random() < teacher_forcing_ratio:
                last_cbc = target[:, t, :]  # Teacher Forcing: 실제값
            else:
                last_cbc = predicted_cbc  # Autoregressive: 예측값

        return torch.stack(predictions, dim=1)  # (batch, 5, 6)


class ForecastDatasetV2(Dataset):
    """Baseline 포함 Forecast Dataset.

    V2 Forecaster는 환자 Baseline 특성도 입력으로 받습니다.
    """

    def __init__(
        self,
        cbc_sequences: np.ndarray,
        baseline_features: np.ndarray,
        input_weeks: int = 3,
    ):
        """
        Args:
            cbc_sequences: (n_samples, 8, 6) — scaled CBC 시퀀스
            baseline_features: (n_samples, n_baseline) — scaled Baseline 특성
            input_weeks: 입력 주차 수 (기본 3)
        """
        self.cbc_input = torch.FloatTensor(cbc_sequences[:, :input_weeks, :])
        self.cbc_target = torch.FloatTensor(cbc_sequences[:, input_weeks:, :])
        self.baseline = torch.FloatTensor(baseline_features)

    def __len__(self) -> int:
        return len(self.cbc_input)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input": self.cbc_input[idx],      # (3, 6)
            "target": self.cbc_target[idx],     # (5, 6)
            "baseline": self.baseline[idx],     # (n_baseline,)
        }


class WeightedMSELoss(nn.Module):
    """Feature별 가중치를 적용한 MSE Loss.

    CBC 변수 순서: [WBC, ANC, ALC, AMC, PLT, Hb]
    Neutropenia 예측에 핵심인 ANC, WBC에 높은 가중치를 부여합니다.
    """

    def __init__(self, feature_weights: Optional[List[float]] = None):
        super().__init__()
        if feature_weights is None:
            # 기본: ANC(3x), WBC(2x), AMC(1.5x), 나머지(1x)
            feature_weights = [2.0, 3.0, 1.0, 1.5, 1.0, 1.0]
        self.feature_weights = torch.FloatTensor(feature_weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch, seq_len, n_features)
            target: (batch, seq_len, n_features)
        """
        weights = self.feature_weights.to(pred.device)
        squared_error = (pred - target) ** 2
        weighted_error = squared_error * weights  # broadcasting
        return weighted_error.mean()


class ForecasterTrainerV2:
    """CBCForecasterV2의 학습/예측을 관리합니다.

    개선 사항:
        - Feature-weighted MSE: ANC, WBC 예측 오차에 높은 가중치
        - Scheduled Sampling: Teacher Forcing ratio 점진 감소
    """

    def __init__(
        self,
        model: CBCForecasterV2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        feature_weights: Optional[List[float]] = None,
        device: str = "auto",
    ):
        self.device = self._get_device(device)
        self.model = model.to(self.device)

        self.optimizer = Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10,
        )
        self.criterion = WeightedMSELoss(feature_weights).to(self.device)

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
        }

    def _get_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 300,
        patience: int = 30,
    ) -> Dict[str, List[float]]:
        """Forecaster V2를 학습합니다.

        Scheduled Sampling: teacher_forcing_ratio를 1.0 → 0.0으로 감소
        """
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Scheduled Sampling: 선형 감소
            tf_ratio = max(0.0, 1.0 - epoch / num_epochs)

            # ----- 학습 -----
            self.model.train()
            train_losses = []
            for batch in train_loader:
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
                baseline = batch["baseline"].to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(
                    inputs, baseline,
                    target=targets,
                    teacher_forcing_ratio=tf_ratio,
                )
                loss = self.criterion(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            # ----- 검증 (Teacher Forcing 없음) -----
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["input"].to(self.device)
                    targets = batch["target"].to(self.device)
                    baseline = batch["baseline"].to(self.device)

                    predictions = self.model(
                        inputs, baseline,
                        target=None,
                        teacher_forcing_ratio=0.0,
                    )
                    loss = self.criterion(predictions, targets)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            self.scheduler.step(val_loss)

            if epoch % 20 == 0:
                logger.info(
                    f"ForecasterV2 Epoch {epoch:3d}/{num_epochs} | "
                    f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | "
                    f"TF ratio: {tf_ratio:.2f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"ForecasterV2 Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        logger.info(f"ForecasterV2 학습 완료 (best val MSE: {best_val_loss:.6f})")
        return self.history

    def predict(self, cbc_input: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        """Week 0-2 CBC + Baseline에서 Week 3-7 CBC를 예측합니다.

        Args:
            cbc_input: (n_samples, 3, 6) — scaled Week 0-2 CBC
            baseline: (n_samples, n_baseline) — scaled Baseline 특성

        Returns:
            (n_samples, 5, 6) — 예측된 Week 3-7 CBC (scaled)
        """
        self.model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(cbc_input).to(self.device)
            base = torch.FloatTensor(baseline).to(self.device)
            predictions = self.model(
                inputs, base, target=None, teacher_forcing_ratio=0.0,
            )
            return predictions.cpu().numpy()
