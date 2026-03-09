"""
helpers.py - 유틸리티 함수 모듈

프로젝트 전반에서 사용되는 공통 함수들을 제공합니다.
로깅 설정, 시드 고정, 결과 저장 등을 담당합니다.
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """프로젝트 로깅을 설정합니다.

    콘솔과 파일에 동시 출력하며, 타임스탬프가 포함된 포맷을 사용합니다.

    Args:
        log_dir: 로그 파일 저장 디렉토리 (None이면 콘솔만)
        level: 로깅 레벨
        log_filename: 로그 파일명 (None이면 타임스탬프 자동 생성)

    Returns:
        루트 Logger 인스턴스
    """
    # 로그 포맷 설정
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 기존 핸들러 제거 (중복 출력 방지)
    root_logger.handlers.clear()

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 파일 핸들러 (선택)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"train_{timestamp}.log"

        file_handler = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def set_seed(seed: int = 42):
    """모든 난수 생성기의 시드를 고정하여 재현성을 보장합니다.

    Python, NumPy, PyTorch(CPU/GPU)의 시드를 동시에 고정합니다.

    Args:
        seed: 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 결정론적 연산 설정 (재현성 ↑, 성능 약간 ↓)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logging.getLogger(__name__).info(f"시드 고정: {seed}")


def save_results(
    results: Dict[str, Any],
    filepath: str,
    indent: int = 2,
):
    """실험 결과를 JSON 파일로 저장합니다.

    numpy 배열 등 JSON 직렬화가 불가능한 객체를 자동 변환합니다.

    Args:
        results: 저장할 결과 딕셔너리
        filepath: 저장 경로
        indent: JSON 들여쓰기
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # numpy/torch 객체를 Python 기본 타입으로 변환
    serializable = _make_serializable(results)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=indent, ensure_ascii=False)

    logging.getLogger(__name__).info(f"결과 저장: {filepath}")


def _make_serializable(obj: Any) -> Any:
    """JSON 직렬화가 불가능한 객체를 변환합니다.

    Args:
        obj: 변환할 객체

    Returns:
        JSON 직렬화 가능한 객체
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, Path):
        return str(obj)
    return obj


def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """클래스 가중치를 계산합니다.

    불균형한 클래스 분포에서 소수 클래스에 높은 가중치를 부여합니다.

    Args:
        y: 타겟 레이블 배열

    Returns:
        {클래스: 가중치} 딕셔너리
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)

    weights = {}
    for cls, count in zip(classes, counts):
        weights[int(cls)] = total / (n_classes * count)

    logging.getLogger(__name__).info(f"클래스 가중치: {weights}")
    return weights


def print_data_summary(df, target_col: str = "grade3_neutropenia"):
    """데이터셋 요약 정보를 출력합니다.

    Args:
        df: 요약할 DataFrame
        target_col: 타겟 컬럼명
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("데이터셋 요약")
    logger.info("=" * 60)
    logger.info(f"총 환자 수: {len(df)}")
    logger.info(f"변수 수: {len(df.columns)}")
    logger.info(f"결측치: {df.isnull().sum().sum()}")

    if target_col in df.columns:
        pos = df[target_col].sum()
        neg = len(df) - pos
        logger.info(f"양성 (Grade 3+): {pos} ({pos/len(df)*100:.1f}%)")
        logger.info(f"음성 (Grade <3): {neg} ({neg/len(df)*100:.1f}%)")

    logger.info("=" * 60)
