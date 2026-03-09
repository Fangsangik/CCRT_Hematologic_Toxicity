"""
dataset.py - PyTorch Dataset 클래스 모듈

LSTM 모델 학습을 위한 커스텀 Dataset과 DataLoader 생성 유틸리티를 제공합니다.
CBC 시계열 + Baseline 특성을 함께 처리하는 멀티모달 데이터셋을 지원합니다.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class CCRTDataset(Dataset):
    """CCRT 혈액독성 예측을 위한 PyTorch Dataset입니다.

    LSTM 모델에 필요한 두 가지 입력을 관리합니다:
        1. CBC 시계열: (seq_length, n_cbc_features) 텐서
        2. Baseline 특성: (n_baseline_features,) 텐서

    Args:
        cbc_sequences: CBC 시계열 데이터 (n_samples, seq_length, n_features)
        baseline_features: Baseline 임상+치료 특성 (n_samples, n_features)
        targets: 타겟 레이블 (n_samples,), None이면 추론 모드
        patient_ids: 환자 ID 목록 (추적용, 선택)
    """

    def __init__(
        self,
        cbc_sequences: np.ndarray,
        baseline_features: np.ndarray,
        targets: Optional[np.ndarray] = None,
        patient_ids: Optional[List[str]] = None,
        mask_future_prob: float = 0.0,
        mask_start_idx: int = 3,
    ):
        """
        Args:
            cbc_sequences: CBC 시계열 (n_samples, seq_length, n_features)
            baseline_features: Baseline 특성 (n_samples, n_features)
            targets: 타겟 레이블 (n_samples,)
            patient_ids: 환자 ID 목록
            mask_future_prob: 학습 시 Week 3-7을 0으로 마스킹할 확률 (0.0~1.0)
                조기 예측 능력을 학습하기 위한 augmentation.
                0.5 = 50% 확률로 Week 3-7 마스킹 → Week 0-2만으로 예측 학습
            mask_start_idx: 마스킹 시작 인덱스 (기본 3 = Week 3부터)
        """
        # numpy → torch tensor 변환
        self.cbc_sequences = torch.FloatTensor(cbc_sequences)
        self.baseline_features = torch.FloatTensor(baseline_features)

        # 타겟이 있으면 학습 모드, 없으면 추론 모드
        self.targets = (
            torch.FloatTensor(targets) if targets is not None else None
        )

        self.patient_ids = patient_ids
        self.is_inference = targets is None

        # Masking augmentation 설정
        self.mask_future_prob = mask_future_prob
        self.mask_start_idx = mask_start_idx

        # 데이터 크기 검증
        assert len(self.cbc_sequences) == len(self.baseline_features), (
            f"CBC 시퀀스({len(self.cbc_sequences)})와 "
            f"Baseline 특성({len(self.baseline_features)}) 수가 다릅니다."
        )

    def __len__(self) -> int:
        """데이터셋 크기를 반환합니다."""
        return len(self.cbc_sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """인덱스에 해당하는 데이터 샘플을 반환합니다.

        학습 모드에서 mask_future_prob > 0이면, 확률적으로
        Week 3-7 데이터를 0으로 마스킹하여 조기 예측 능력을 학습합니다.

        Returns:
            딕셔너리:
                - "cbc_seq": CBC 시계열 텐서 (seq_length, n_features)
                - "baseline": Baseline 특성 텐서 (n_features,)
                - "target": 타겟 레이블 (학습 모드인 경우)
                - "patient_id": 환자 ID (있는 경우)
        """
        cbc_seq = self.cbc_sequences[idx].clone()
        seq_length = cbc_seq.shape[0]

        # 마스크 표시 채널: 1=실제 데이터, 0=마스킹됨
        mask_indicator = torch.ones(seq_length, 1)

        # Masking augmentation: 학습 시 확률적으로 Week 3-7을 마스킹
        if self.mask_future_prob > 0 and not self.is_inference:
            if torch.rand(1).item() < self.mask_future_prob:
                cbc_seq[self.mask_start_idx:, :] = 0.0
                mask_indicator[self.mask_start_idx:, :] = 0.0

        # CBC 데이터 + 마스크 표시 채널 결합 → (seq_length, n_features + 1)
        cbc_seq = torch.cat([cbc_seq, mask_indicator], dim=1)

        sample = {
            "cbc_seq": cbc_seq,
            "baseline": self.baseline_features[idx],
        }

        if not self.is_inference:
            sample["target"] = self.targets[idx]

        if self.patient_ids is not None:
            sample["patient_id"] = self.patient_ids[idx]

        return sample


class BaselineOnlyDataset(Dataset):
    """Baseline-only 모드용 간단한 Dataset입니다.

    CBC 시계열 없이 baseline 특성만 사용하는 모델 비교 실험에 사용됩니다.

    Args:
        features: 입력 특성 (n_samples, n_features)
        targets: 타겟 레이블 (n_samples,), None이면 추론 모드
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
    ):
        self.features = torch.FloatTensor(features)
        self.targets = (
            torch.FloatTensor(targets) if targets is not None else None
        )
        self.is_inference = targets is None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {"features": self.features[idx]}
        if not self.is_inference:
            sample["target"] = self.targets[idx]
        return sample


# ============================================================
# DataLoader 생성 유틸리티
# ============================================================
def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    handle_imbalance: str = "none",
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """학습/검증/테스트 DataLoader를 생성합니다.

    클래스 불균형 처리를 위한 WeightedRandomSampler를 지원합니다.

    Args:
        train_dataset: 학습 Dataset
        val_dataset: 검증 Dataset
        test_dataset: 테스트 Dataset (선택)
        batch_size: 배치 크기
        handle_imbalance: 불균형 처리 방법
            - "none": 처리 안 함
            - "oversampling": WeightedRandomSampler로 오버샘플링
        num_workers: 데이터 로딩 워커 수

    Returns:
        DataLoader 딕셔너리 {"train": ..., "val": ..., "test": ...}
    """
    # 학습 데이터 샘플러 설정 (클래스 불균형 대응)
    train_sampler = None
    shuffle_train = True

    if handle_imbalance == "oversampling" and hasattr(train_dataset, "targets"):
        targets = train_dataset.targets.numpy()
        # 각 클래스의 가중치 계산 (소수 클래스에 높은 가중치)
        class_counts = np.bincount(targets.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[targets.astype(int)]

        train_sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle_train = False  # sampler 사용 시 shuffle 비활성화

        logger.info(
            f"WeightedRandomSampler 적용: "
            f"클래스 비율={class_counts}, 가중치={class_weights.round(3)}"
        )

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=num_workers,
            drop_last=False,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    if test_dataset is not None:
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    logger.info(
        f"DataLoader 생성 완료: "
        f"batch_size={batch_size}, "
        f"train={len(train_dataset)}, val={len(val_dataset)}"
        + (f", test={len(test_dataset)}" if test_dataset else "")
    )

    return dataloaders
