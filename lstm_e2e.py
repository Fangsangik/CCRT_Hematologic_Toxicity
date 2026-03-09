"""
lstm_e2e.py - End-to-End Forecast → Classify 파이프라인

Forecaster와 Classifier를 하나의 계산 그래프로 묶어,
Classification loss(BCE)가 Forecaster까지 역전파됩니다.

핵심 차이점:
    - 기존 lstm_forecast.py: Forecaster(MSE) 따로 → Classifier(BCE) 따로
    - 이 파일: BCE + alpha*MSE가 하나의 그래프에서 역전파
      → Forecaster가 "분류에 유용한 예측"을 학습

학습 전략:
    Phase 1: Forecaster warm-up (MSE만, 50 에폭) — 합리적 초기값 확보
    Phase 2: End-to-End fine-tune (BCE + alpha*MSE) — 분류 성능 최적화

사용법:
    python lstm_e2e.py --n_patients 200 --seed 42 --ext_patients 150 --ext_seed 9999
"""

import argparse
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader as TorchDataLoader, Dataset

from config import Config
from generate_emr_data import generate_patients, generate_cbc_timeseries
from src.data.data_loader import DataLoader as EMRDataLoader
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessing import EMRPreprocessor
from src.evaluation.metrics import bootstrap_ci, compute_all_metrics
from src.models.baseline_models import create_model
from src.models.forecaster import (
    CBCForecasterV2, ForecastDatasetV2, ForecasterTrainerV2,
)
from src.models.lstm_model import LSTMPredictor
from src.utils.helpers import set_seed

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# End-to-End 모델
# ============================================================
class E2EModel(nn.Module):
    """Forecaster V2 + Classifier를 하나로 묶은 End-to-End 모델입니다.

    순전파 흐름:
        Week 0-2 CBC + Baseline → Forecaster V2 → 예측 Week 3-7
        [실제 Week 0-2] + [예측 Week 3-7] → Classifier → 독성 예측

    역전파:
        BCE loss + alpha * MSE loss → Classifier → Forecaster
        Forecaster가 "분류에 유용한 CBC 예측"을 학습합니다.

    V2 개선점:
        - Seq2Seq + Attention: 인코더 전체 출력 활용
        - Autoregressive: Week별 순차 예측
        - Baseline 조건부: 환자 특성 반영
        - Delta 예측: 변화량 학습
        - Teacher Forcing: 학습 안정화
    """

    def __init__(self, forecaster: CBCForecasterV2, classifier: LSTMPredictor):
        super().__init__()
        self.forecaster = forecaster
        self.classifier = classifier

    def forward(
        self,
        cbc_input: torch.Tensor,
        baseline: torch.Tensor,
        cbc_target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """End-to-End 순전파입니다.

        Args:
            cbc_input: (batch, 3, 6) — Week 0-2 CBC (scaled)
            baseline: (batch, n_baseline) — Baseline 특성
            cbc_target: (batch, 5, 6) — 학습 시 실제 Week 3-7 (Teacher Forcing용)
            teacher_forcing_ratio: Teacher Forcing 비율

        Returns:
            logits: (batch, 1) — 분류 로짓
            predicted_cbc: (batch, 5, 6) — 예측된 Week 3-7 CBC
        """
        # 1) Forecaster V2: Week 0-2 + Baseline → Week 3-7 예측
        predicted_cbc = self.forecaster(
            cbc_input, baseline,
            target=cbc_target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )  # (batch, 5, 6)

        # 2) 실제 Week 0-2 + 예측 Week 3-7 결합
        full_seq = torch.cat([cbc_input, predicted_cbc], dim=1)  # (batch, 8, 6)

        # 3) Classifier: 전체 시퀀스 → 독성 예측
        logits = self.classifier(full_seq, baseline)  # (batch, 1)

        return logits, predicted_cbc


# ============================================================
# E2E Dataset
# ============================================================
class E2EDataset(Dataset):
    """End-to-End 학습용 Dataset입니다.

    Week 0-2 입력, Week 3-7 타겟 (MSE 감독), baseline, 분류 타겟을 제공합니다.
    """

    def __init__(self, cbc_sequences, baseline_features, targets):
        """
        Args:
            cbc_sequences: (n, 8, 6) — 전체 scaled CBC 시퀀스
            baseline_features: (n, n_baseline) — Baseline 특성
            targets: (n,) — 분류 타겟
        """
        self.cbc_input = torch.FloatTensor(cbc_sequences[:, :3, :])   # Week 0-2
        self.cbc_target = torch.FloatTensor(cbc_sequences[:, 3:, :])  # Week 3-7
        self.baseline = torch.FloatTensor(baseline_features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "cbc_input": self.cbc_input[idx],    # (3, 6)
            "cbc_target": self.cbc_target[idx],  # (5, 6)
            "baseline": self.baseline[idx],
            "target": self.targets[idx],
        }


# ============================================================
# E2E Trainer
# ============================================================
class E2ETrainer:
    """End-to-End 모델의 학습을 관리합니다.

    Combined Loss = BCE(분류) + alpha * MSE(예측)
    alpha를 통해 Forecaster의 MSE 정규화 강도를 조절합니다.
    """

    def __init__(
        self,
        model: E2EModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        alpha: float = 0.3,
        device: str = "auto",
    ):
        self.device = self._get_device(device)
        self.model = model.to(self.device)
        self.alpha = alpha

        self.optimizer = Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=7,
        )

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        self.best_val_loss = float("inf")
        self.best_state = None
        self.patience_counter = 0

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_bce": [], "train_mse": [],
            "train_auc": [], "val_auc": [],
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
        train_loader: TorchDataLoader,
        val_loader: TorchDataLoader,
        num_epochs: int = 100,
        patience: int = 15,
    ) -> Dict[str, List[float]]:
        """End-to-End 학습을 수행합니다."""
        logger.info(
            f"E2E 학습 시작: {num_epochs} 에폭, alpha={self.alpha}, 장치={self.device}"
        )

        for epoch in range(1, num_epochs + 1):
            start = time.time()

            # Scheduled Sampling: teacher forcing ratio 감소
            tf_ratio = max(0.0, 1.0 - epoch / num_epochs)

            # 학습
            train_metrics = self._train_epoch(train_loader, tf_ratio=tf_ratio)

            # 검증 (teacher forcing 없음)
            val_metrics = self._validate_epoch(val_loader)

            # 스케줄러
            self.scheduler.step(val_metrics["loss"])

            # 히스토리
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_bce"].append(train_metrics["bce"])
            self.history["train_mse"].append(train_metrics["mse"])
            self.history["train_auc"].append(train_metrics["auc"])
            self.history["val_auc"].append(val_metrics["auc"])

            elapsed = time.time() - start

            if epoch % 10 == 0 or epoch == num_epochs:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                    f"BCE: {train_metrics['bce']:.4f} MSE: {train_metrics['mse']:.4f} | "
                    f"AUC: {train_metrics['auc']:.4f}/{val_metrics['auc']:.4f} | "
                    f"LR: {lr:.2e} | {elapsed:.1f}s"
                )

            # Early stopping
            if val_metrics["loss"] < self.best_val_loss - 1e-4:
                self.best_val_loss = val_metrics["loss"]
                self.best_state = deepcopy(self.model.state_dict())
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(best val loss: {self.best_val_loss:.4f})"
                    )
                    break

        # 최적 가중치 복원
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return self.history

    def _train_epoch(self, loader, tf_ratio: float = 0.3) -> Dict[str, float]:
        self.model.train()
        total_loss = total_bce = total_mse = 0.0
        all_targets, all_probs = [], []
        n = 0

        for batch in loader:
            cbc_input = batch["cbc_input"].to(self.device)
            cbc_target = batch["cbc_target"].to(self.device)
            baseline = batch["baseline"].to(self.device)
            targets = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            logits, predicted_cbc = self.model(
                cbc_input, baseline,
                cbc_target=cbc_target,
                teacher_forcing_ratio=tf_ratio,
            )
            logits = logits.squeeze(-1)

            bce = self.bce_loss(logits, targets)
            mse = self.mse_loss(predicted_cbc, cbc_target)
            loss = bce + self.alpha * mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            bs = len(targets)
            total_loss += loss.item() * bs
            total_bce += bce.item() * bs
            total_mse += mse.item() * bs
            n += bs

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())

        try:
            auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auc = 0.0

        return {
            "loss": total_loss / n,
            "bce": total_bce / n,
            "mse": total_mse / n,
            "auc": auc,
        }

    @torch.no_grad()
    def _validate_epoch(self, loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_targets, all_probs = [], []
        n = 0

        for batch in loader:
            cbc_input = batch["cbc_input"].to(self.device)
            cbc_target = batch["cbc_target"].to(self.device)
            baseline = batch["baseline"].to(self.device)
            targets = batch["target"].to(self.device)

            logits, predicted_cbc = self.model(cbc_input, baseline)
            logits = logits.squeeze(-1)

            bce = self.bce_loss(logits, targets)
            mse = self.mse_loss(predicted_cbc, cbc_target)
            loss = bce + self.alpha * mse

            bs = len(targets)
            total_loss += loss.item() * bs
            n += bs

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())

        try:
            auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auc = 0.0

        return {"loss": total_loss / n, "auc": auc}

    @torch.no_grad()
    def predict(self, loader) -> Tuple[np.ndarray, np.ndarray]:
        """예측 확률을 반환합니다."""
        self.model.eval()
        all_probs, all_targets = [], []

        for batch in loader:
            cbc_input = batch["cbc_input"].to(self.device)
            baseline = batch["baseline"].to(self.device)

            logits, _ = self.model(cbc_input, baseline)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.extend(probs)

            if "target" in batch:
                all_targets.extend(batch["target"].numpy())

        return np.array(all_probs), np.array(all_targets)


# ============================================================
# 데이터 준비 (다른 파일과 동일)
# ============================================================
def prepare_data(config, n_patients=200, seed=42):
    """데이터 생성 → 전처리 → 분할 → 특성 공학을 수행합니다."""
    set_seed(seed)

    patients = generate_patients(n_patients, seed=seed)
    cbc_results, _ = generate_cbc_timeseries(patients, seed=seed)
    patients_df = pd.DataFrame(patients)
    cbc_df = pd.DataFrame(cbc_results)
    preprocessor = EMRPreprocessor(config)
    df = preprocessor.run_full_pipeline(patients_df, cbc_df)

    data_loader = EMRDataLoader(config)
    processed_path = config.paths.processed_data_dir / "emr_processed.csv"
    df.to_csv(processed_path, index=False)
    df = data_loader.load_data(str(processed_path))
    df = data_loader.handle_missing_values(df)
    train_df, val_df, test_df = data_loader.split_data(df)

    fe = FeatureEngineer(config)
    for split_df in [train_df, val_df, test_df]:
        fe.create_cbc_temporal_features(split_df)
    train_df = fe.create_cbc_temporal_features(train_df)
    val_df = fe.create_cbc_temporal_features(val_df)
    test_df = fe.create_cbc_temporal_features(test_df)

    train_df = fe.encode_categorical(train_df, fit=True)
    val_df = fe.encode_categorical(val_df, fit=False)
    test_df = fe.encode_categorical(test_df, fit=False)

    feature_cols = fe.get_feature_columns(train_df, mode="baseline_cbc")
    all_feature_cols = feature_cols["all"]

    target = config.data.primary_target
    y_train = train_df[target].values
    y_val = val_df[target].values
    y_test = test_df[target].values

    baseline_cols = [c for c in feature_cols["baseline"] if c in train_df.columns]
    X_base_train = train_df[baseline_cols].values.astype(np.float32)
    X_base_val = val_df[baseline_cols].values.astype(np.float32)
    X_base_test = test_df[baseline_cols].values.astype(np.float32)

    cbc_train = fe.prepare_lstm_sequences(train_df)
    cbc_val = fe.prepare_lstm_sequences(val_df)
    cbc_test = fe.prepare_lstm_sequences(test_df)

    scaler_cbc = StandardScaler()
    n_s, seq_len, n_f = cbc_train.shape
    cbc_train = scaler_cbc.fit_transform(
        cbc_train.reshape(-1, n_f)
    ).reshape(n_s, seq_len, n_f)
    cbc_val = scaler_cbc.transform(
        cbc_val.reshape(-1, n_f)
    ).reshape(len(val_df), seq_len, n_f)
    cbc_test = scaler_cbc.transform(
        cbc_test.reshape(-1, n_f)
    ).reshape(len(test_df), seq_len, n_f)

    scaler_base = StandardScaler()
    X_base_train = scaler_base.fit_transform(X_base_train)
    X_base_val = scaler_base.transform(X_base_val)
    X_base_test = scaler_base.transform(X_base_test)

    return (
        train_df, val_df, test_df, fe,
        y_train, y_val, y_test,
        cbc_train, cbc_val, cbc_test,
        X_base_train, X_base_val, X_base_test,
        scaler_cbc, scaler_base, baseline_cols, all_feature_cols,
    )


def train_tree_models(X_train, y_train, config):
    """Tree 모델을 학습합니다."""
    models = {}
    for model_name in ["xgboost", "lightgbm", "logistic_regression"]:
        model = create_model(model_name, config)
        model.fit(X_train, y_train)
        models[model_name] = model
        logger.info(f"  {model_name} 학습 완료")
    return models


def train_e2e(
    cbc_train, cbc_val,
    X_base_train, X_base_val,
    y_train, y_val,
    config,
    alpha=0.3,
):
    """End-to-End Forecast → Classify 학습을 수행합니다.

    Phase 1: Forecaster warm-up (MSE만, 50 에폭)
    Phase 2: End-to-End fine-tune (BCE + alpha*MSE)
    """
    n_cbc_features = cbc_train.shape[2]  # 6
    n_baseline_features = X_base_train.shape[1]
    lstm_config = config.lstm
    lstm_config.input_size = 6  # 마스크 채널 불필요
    lstm_config.baseline_input_size = n_baseline_features

    # ============================================================
    # Phase 1: Forecaster V2 Warm-up (MSE만)
    # ============================================================
    logger.info("=" * 60)
    logger.info("Phase 1: Forecaster V2 Warm-up (MSE only, 50 에폭)")
    logger.info("=" * 60)

    forecaster = CBCForecasterV2(
        n_cbc_features=n_cbc_features,
        n_baseline_features=n_baseline_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        n_forecast_weeks=5,
    )

    n_params = sum(p.numel() for p in forecaster.parameters())
    logger.info(f"ForecasterV2 파라미터 수: {n_params:,}")

    # Forecaster V2 단독 학습 (Baseline 포함)
    f_train_ds = ForecastDatasetV2(cbc_train, X_base_train, input_weeks=3)
    f_val_ds = ForecastDatasetV2(cbc_val, X_base_val, input_weeks=3)
    f_train_loader = TorchDataLoader(f_train_ds, batch_size=32, shuffle=True)
    f_val_loader = TorchDataLoader(f_val_ds, batch_size=32, shuffle=False)

    f_trainer = ForecasterTrainerV2(forecaster, lr=1e-3, weight_decay=1e-4)
    f_trainer.train(f_train_loader, f_val_loader, num_epochs=50, patience=15)

    # Warm-up 결과 확인
    pred_val = f_trainer.predict(cbc_val[:, :3, :], X_base_val)
    warmup_mse = np.mean((pred_val - cbc_val[:, 3:, :]) ** 2)
    logger.info(f"ForecasterV2 warm-up 완료 (Val MSE: {warmup_mse:.4f})")

    # ============================================================
    # Phase 2: End-to-End Fine-tune (BCE + alpha*MSE)
    # ============================================================
    logger.info("=" * 60)
    logger.info(f"Phase 2: End-to-End 학습 (BCE + {alpha}*MSE)")
    logger.info("=" * 60)

    classifier = LSTMPredictor(lstm_config)
    e2e_model = E2EModel(forecaster, classifier)

    # E2E Dataset
    train_ds = E2EDataset(cbc_train, X_base_train, y_train)
    val_ds = E2EDataset(cbc_val, X_base_val, y_val)
    train_loader = TorchDataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = TorchDataLoader(val_ds, batch_size=32, shuffle=False)

    e2e_trainer = E2ETrainer(
        e2e_model, lr=5e-4, weight_decay=1e-4, alpha=alpha,
    )
    e2e_trainer.train(train_loader, val_loader, num_epochs=150, patience=20)

    return e2e_trainer


def evaluate_external(
    tree_models, e2e_trainer,
    fe, scaler_cbc, scaler_base,
    baseline_cols, all_feature_cols,
    config,
    ext_patients=150, ext_seed=9999,
):
    """외부 검증 데이터에서 모든 모델의 성능을 평가합니다."""
    logger.info("=" * 60)
    logger.info("External Validation 데이터 생성")
    logger.info("=" * 60)

    ext_patients_list = generate_patients(ext_patients, seed=ext_seed)
    ext_cbc_list, _ = generate_cbc_timeseries(ext_patients_list, seed=ext_seed)
    ext_patients_df = pd.DataFrame(ext_patients_list)
    ext_cbc_df = pd.DataFrame(ext_cbc_list)
    preprocessor = EMRPreprocessor(config)
    ext_df = preprocessor.run_full_pipeline(ext_patients_df, ext_cbc_df)

    ext_path = config.paths.processed_data_dir / "e2e_external.csv"
    ext_df.to_csv(ext_path, index=False)

    ext_df = fe.create_cbc_temporal_features(ext_df)
    ext_df = fe.encode_categorical(ext_df, fit=False)

    target = config.data.primary_target
    y_ext = ext_df[target].values

    logger.info(f"External 데이터: {len(ext_df)}명")
    logger.info(f"  양성: {y_ext.sum():.0f} ({y_ext.mean()*100:.1f}%)")

    # ----- Tree 모델 평가 -----
    for col in all_feature_cols:
        if col not in ext_df.columns:
            ext_df[col] = 0
    X_ext = ext_df[all_feature_cols].values.astype(np.float32)
    X_ext = np.nan_to_num(X_ext, nan=0.0)

    results = {}

    logger.info("")
    logger.info("=" * 60)
    logger.info("External Validation 결과")
    logger.info("=" * 60)

    for model_name, model in tree_models.items():
        ext_proba = model.predict_proba(X_ext)
        metrics = compute_all_metrics(y_ext, ext_proba, model_name=f"{model_name}_ext")
        _, ci_lower, ci_upper = bootstrap_ci(y_ext, ext_proba)
        metrics["auroc_ci"] = (ci_lower, ci_upper)
        results[model_name] = metrics

        logger.info(
            f"  {model_name:25s} | AUROC={metrics['auroc']:.4f} "
            f"({ci_lower:.4f}-{ci_upper:.4f}) | "
            f"Sens={metrics['sensitivity']:.4f} | Spec={metrics['specificity']:.4f}"
        )

    # ----- LSTM E2E 평가 (Week 0-2만 사용) -----
    cbc_ext = fe.prepare_lstm_sequences(ext_df)
    n_s, seq_len, n_f = cbc_ext.shape
    cbc_ext_scaled = scaler_cbc.transform(
        cbc_ext.reshape(-1, n_f)
    ).reshape(n_s, seq_len, n_f)

    for col in baseline_cols:
        if col not in ext_df.columns:
            ext_df[col] = 0
    X_base_ext = ext_df[baseline_cols].values.astype(np.float32)
    X_base_ext = np.nan_to_num(X_base_ext, nan=0.0)
    X_base_ext = scaler_base.transform(X_base_ext)

    # E2E 모델: Week 0-2 입력 → Forecaster → Classifier
    ext_ds = E2EDataset(cbc_ext_scaled, X_base_ext, y_ext)
    ext_loader = TorchDataLoader(ext_ds, batch_size=32, shuffle=False)

    ext_proba, _ = e2e_trainer.predict(ext_loader)
    metrics = compute_all_metrics(y_ext, ext_proba, model_name="lstm_e2e_ext")
    _, ci_lower, ci_upper = bootstrap_ci(y_ext, ext_proba)
    metrics["auroc_ci"] = (ci_lower, ci_upper)
    results["lstm_e2e"] = metrics

    logger.info(
        f"  {'lstm_e2e':25s} | AUROC={metrics['auroc']:.4f} "
        f"({ci_lower:.4f}-{ci_upper:.4f}) | "
        f"Sens={metrics['sensitivity']:.4f} | Spec={metrics['specificity']:.4f}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="LSTM E2E Forecast+Classify")
    parser.add_argument("--n_patients", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ext_patients", type=int, default=150)
    parser.add_argument("--ext_seed", type=int, default=9999)
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="MSE 정규화 가중치 (기본: 0.3)")
    args = parser.parse_args()

    config = Config()
    config.paths.ensure_dirs()

    logger.info("=" * 60)
    logger.info("LSTM End-to-End (Forecast → Classify)")
    logger.info(f"  alpha={args.alpha} (BCE + alpha*MSE)")
    logger.info("=" * 60)

    # 1) 데이터 준비
    (
        train_df, val_df, test_df, fe,
        y_train, y_val, y_test,
        cbc_train, cbc_val, cbc_test,
        X_base_train, X_base_val, X_base_test,
        scaler_cbc, scaler_base, baseline_cols, all_feature_cols,
    ) = prepare_data(config, args.n_patients, args.seed)

    # 2) Tree 모델 학습
    logger.info("")
    logger.info("Tree 모델 학습")
    X_train_flat = train_df[all_feature_cols].values.astype(np.float32)
    X_train_flat = np.nan_to_num(X_train_flat, nan=0.0)
    tree_models = train_tree_models(X_train_flat, y_train, config)

    # 3) E2E 학습
    e2e_trainer = train_e2e(
        cbc_train, cbc_val,
        X_base_train, X_base_val,
        y_train, y_val,
        config,
        alpha=args.alpha,
    )

    # 4) 내부 테스트
    logger.info("")
    logger.info("내부 테스트 평가")
    test_ds = E2EDataset(cbc_test, X_base_test, y_test)
    test_loader = TorchDataLoader(test_ds, batch_size=32, shuffle=False)
    test_proba, _ = e2e_trainer.predict(test_loader)
    test_metrics = compute_all_metrics(y_test, test_proba, model_name="e2e_internal")
    logger.info(
        f"  E2E Internal AUROC={test_metrics['auroc']:.4f} | "
        f"Sens={test_metrics['sensitivity']:.4f} | Spec={test_metrics['specificity']:.4f}"
    )

    # 5) 외부 검증
    results = evaluate_external(
        tree_models, e2e_trainer,
        fe, scaler_cbc, scaler_base,
        baseline_cols, all_feature_cols,
        config,
        ext_patients=args.ext_patients, ext_seed=args.ext_seed,
    )

    # 6) 최종 요약
    logger.info("")
    logger.info("=" * 60)
    logger.info("최종 결과 요약 (External Validation)")
    logger.info("=" * 60)
    logger.info(f"{'모델':25s} | {'AUROC':>7s} | {'Sens':>6s} | {'Spec':>6s}")
    logger.info("-" * 55)
    for name, m in results.items():
        logger.info(
            f"{name:25s} | {m['auroc']:7.4f} | "
            f"{m['sensitivity']:6.4f} | {m['specificity']:6.4f}"
        )

    logger.info("")
    logger.info("End-to-End 학습 완료!")


if __name__ == "__main__":
    main()
