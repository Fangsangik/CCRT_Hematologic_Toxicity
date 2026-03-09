"""
trainer.py - 모델 학습 및 평가 통합 모듈

LSTM과 전통적 ML 모델의 학습 루프, early stopping,
교차 검증을 통합된 인터페이스로 제공합니다.
"""

import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EarlyStopping:
    """검증 손실 기반 Early Stopping을 구현합니다.

    검증 손실이 patience 에폭 동안 개선되지 않으면 학습을 중단합니다.
    최적 모델 가중치를 자동 저장합니다.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        """Early Stopping을 초기화합니다.

        Args:
            patience: 개선 없이 허용할 에폭 수
            min_delta: 개선으로 인정할 최소 변화량
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None
        self.should_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """매 에폭 후 호출하여 조기 종료 여부를 판단합니다.

        Args:
            val_loss: 현재 에폭의 검증 손실
            model: 현재 모델

        Returns:
            True면 학습 중단
        """
        if val_loss < self.best_loss - self.min_delta:
            # 개선됨: 카운터 리셋, 최적 모델 저장
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = deepcopy(model.state_dict())
        else:
            # 개선 안됨: 카운터 증가
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping 발동 (patience={self.patience}, "
                    f"best_loss={self.best_loss:.4f})"
                )

        return self.should_stop


class LSTMTrainer:
    """LSTM 모델의 학습 루프를 관리하는 클래스입니다.

    기능:
        - 학습/검증 루프
        - Early stopping
        - 학습률 스케줄러
        - 모델 체크포인트 저장/로드
        - 학습 히스토리 기록

    사용 예시:
        trainer = LSTMTrainer(model, config)
        history = trainer.train(train_loader, val_loader)
        trainer.save_checkpoint("model.pt")
    """

    def __init__(self, model: nn.Module, config, device: Optional[str] = None,
                 pos_weight: Optional[float] = None):
        """LSTMTrainer를 초기화합니다.

        Args:
            model: 학습할 PyTorch 모델
            config: Config 인스턴스
            device: 학습 장치 ("cpu", "cuda", "mps", None=자동감지)
            pos_weight: 양성 클래스 가중치 (None이면 동일 가중치)
        """
        self.config = config
        self.lstm_config = config.lstm

        # 장치 설정
        self.device = self._get_device(device or config.train.device)
        self.model = model.to(self.device)

        # 옵티마이저: Adam (적응적 학습률)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lstm_config.learning_rate,
            weight_decay=self.lstm_config.weight_decay,
        )

        # 학습률 스케줄러: 검증 손실 정체 시 학습률 감소
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,      # 학습률을 절반으로 감소
            patience=7,       # 7 에폭 동안 개선 없으면
        )

        # 손실 함수: Binary Cross Entropy with Logits
        # pos_weight로 클래스 불균형 보정 (양성 케이스 놓침 방지)
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], dtype=torch.float32, device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            logger.info(f"Weighted BCE 적용: pos_weight={pos_weight:.2f}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.lstm_config.early_stopping_patience
        )

        # 학습 히스토리 기록
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_auc": [],
            "val_auc": [],
            "lr": [],
        }

    def _get_device(self, device_str: str) -> torch.device:
        """학습 장치를 자동 감지합니다.

        Args:
            device_str: 장치 문자열 ("auto", "cpu", "cuda", "mps")

        Returns:
            torch.device 인스턴스
        """
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")  # Apple Silicon GPU
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_str)

        logger.info(f"학습 장치: {device}")
        return device

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """모델을 학습합니다.

        Args:
            train_loader: 학습 DataLoader
            val_loader: 검증 DataLoader
            num_epochs: 에폭 수 (None이면 config 값 사용)

        Returns:
            학습 히스토리 딕셔너리
        """
        num_epochs = num_epochs or self.lstm_config.num_epochs
        logger.info(f"학습 시작: {num_epochs} 에폭, 장치={self.device}")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # ----- 학습 단계 -----
            train_loss, train_auc = self._train_epoch(train_loader)

            # ----- 검증 단계 -----
            val_loss, val_auc = self._validate_epoch(val_loader)

            # 학습률 스케줄러 업데이트
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            # 히스토리 기록
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_auc"].append(train_auc)
            self.history["val_auc"].append(val_auc)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - start_time

            # 10 에폭마다 또는 마지막 에폭에 로그 출력
            if epoch % 10 == 0 or epoch == num_epochs:
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
                    f"LR: {current_lr:.2e} | {elapsed:.1f}s"
                )

            # Early stopping 확인
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping at epoch {epoch}")
                # 최적 모델 가중치 복원
                self.model.load_state_dict(self.early_stopping.best_model_state)
                break

        return self.history

    def _train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """한 에폭의 학습을 수행합니다.

        Args:
            dataloader: 학습 DataLoader

        Returns:
            (평균 손실, AUC) 튜플
        """
        self.model.train()
        total_loss = 0.0
        all_targets = []
        all_probs = []

        for batch in dataloader:
            # 데이터를 장치로 이동
            cbc_seq = batch["cbc_seq"].to(self.device)
            baseline = batch["baseline"].to(self.device)
            targets = batch["target"].to(self.device)

            # 순전파
            self.optimizer.zero_grad()
            logits = self.model(cbc_seq, baseline).squeeze(-1)

            # 손실 계산
            loss = self.criterion(logits, targets)

            # 역전파
            loss.backward()

            # 그래디언트 클리핑 (LSTM의 기울기 폭발 방지)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * len(targets)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)

        # AUC 계산 (클래스가 하나뿐인 경우 예외 처리)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auc = 0.0

        return avg_loss, auc

    @torch.no_grad()
    def _validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """한 에폭의 검증을 수행합니다.

        Args:
            dataloader: 검증 DataLoader

        Returns:
            (평균 손실, AUC) 튜플
        """
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_probs = []

        for batch in dataloader:
            cbc_seq = batch["cbc_seq"].to(self.device)
            baseline = batch["baseline"].to(self.device)
            targets = batch["target"].to(self.device)

            logits = self.model(cbc_seq, baseline).squeeze(-1)
            loss = self.criterion(logits, targets)

            total_loss += loss.item() * len(targets)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auc = 0.0

        return avg_loss, auc

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """테스트 데이터에 대한 예측을 수행합니다.

        Args:
            dataloader: 테스트 DataLoader

        Returns:
            (예측 확률, 실제 레이블) 튜플
        """
        self.model.eval()
        all_probs = []
        all_targets = []

        for batch in dataloader:
            cbc_seq = batch["cbc_seq"].to(self.device)
            baseline = batch["baseline"].to(self.device)

            logits = self.model(cbc_seq, baseline).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)

            if "target" in batch:
                all_targets.extend(batch["target"].numpy())

        return np.array(all_probs), np.array(all_targets) if all_targets else None

    def save_checkpoint(self, filepath: str):
        """모델 체크포인트를 저장합니다.

        모델 가중치, 옵티마이저 상태, 히스토리를 함께 저장합니다.

        Args:
            filepath: 저장할 파일 경로
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.lstm_config,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"체크포인트 저장: {filepath}")

    def load_checkpoint(self, filepath: str):
        """저장된 체크포인트를 로드합니다.

        Args:
            filepath: 로드할 파일 경로
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {})
        logger.info(f"체크포인트 로드: {filepath}")

    def reset_optimizer(self, lr: float = None):
        """Fine-tune를 위해 optimizer, scheduler, early stopping을 리셋합니다.

        Pre-trained 모델의 가중치는 유지하면서,
        optimizer 상태와 학습률을 초기화하여 새로운 학습 단계를 시작합니다.

        Args:
            lr: 새로운 학습률 (None이면 config 기본값 사용)
        """
        lr = lr or self.lstm_config.learning_rate
        self.optimizer = Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.lstm_config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5,
        )
        self.early_stopping = EarlyStopping(
            patience=self.lstm_config.early_stopping_patience
        )
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_auc": [], "val_auc": [], "lr": [],
        }
        logger.info(f"Optimizer 리셋 완료 (lr={lr})")


# ============================================================
# 교차 검증 실행기
# ============================================================
class CrossValidator:
    """K-Fold 교차 검증을 실행합니다.

    전통적 ML 모델(XGBoost, LightGBM, LogReg)에 대한
    교차 검증을 수행하고 결과를 집계합니다.

    사용 예시:
        cv = CrossValidator(config)
        results = cv.run(model, X, y, feature_names)
    """

    def __init__(self, config):
        """CrossValidator를 초기화합니다.

        Args:
            config: Config 인스턴스
        """
        self.config = config
        self.n_folds = config.train.n_folds

    def run(
        self,
        model_factory,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """교차 검증을 실행합니다.

        Args:
            model_factory: 모델 생성 함수 (매 fold마다 새 모델 생성)
            X: 입력 특성
            y: 타겟 레이블
            feature_names: 특성 이름 목록

        Returns:
            교차 검증 결과 딕셔너리:
                - fold_results: 각 fold의 AUC
                - mean_auc: 평균 AUC
                - std_auc: AUC 표준편차
                - feature_importances: 평균 특성 중요도 (해당하는 경우)
        """
        from sklearn.metrics import roc_auc_score

        kfold = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.config.train.seed,
        )

        fold_aucs = []
        fold_importances = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 매 fold마다 새 모델 생성
            model = model_factory()
            model.fit(X_train, y_train, X_val, y_val)

            # 검증 세트 예측
            val_proba = model.predict_proba(X_val)
            auc = roc_auc_score(y_val, val_proba)
            fold_aucs.append(auc)

            # 특성 중요도 수집 (해당하는 경우)
            if hasattr(model, "get_feature_importance"):
                importances = model.get_feature_importance(feature_names)
                fold_importances.append(importances)

            logger.info(f"  Fold {fold}/{self.n_folds} - AUC: {auc:.4f}")

        # 결과 집계
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)

        result = {
            "fold_results": fold_aucs,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
        }

        # 특성 중요도 평균 계산
        if fold_importances:
            avg_importance = {}
            all_features = fold_importances[0].keys()
            for feat in all_features:
                vals = [fi.get(feat, 0) for fi in fold_importances]
                avg_importance[feat] = np.mean(vals)
            result["feature_importances"] = dict(
                sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            )

        logger.info(
            f"교차 검증 완료: AUC = {mean_auc:.4f} (+/- {std_auc:.4f})"
        )

        return result
