import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class ModelRepository:
    """학습된 모델을 pickle로 저장/로드하는 저장소입니다."""

    def save(self, model: Any, filepath: str) -> None:
        """모델을 pickle로 저장합니다."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"모델 저장 완료: {path}")

    def load(self, filepath: str) -> Any:
        """pickle에서 모델을 로드합니다."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"모델 로드 완료: {path}")
        return model
