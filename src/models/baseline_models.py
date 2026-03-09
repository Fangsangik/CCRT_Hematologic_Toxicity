"""
baseline_models.py - 비교 모델 모듈

LSTM 모델의 성능을 비교하기 위한 전통적 ML 모델들을 제공합니다.
XGBoost, LightGBM, Logistic Regression을 통일된 인터페이스로 래핑합니다.

시계열 데이터를 flatten하여 tabular 형태로 입력받습니다.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """모든 비교 모델의 공통 인터페이스입니다.

    학습, 예측, 저장/로드를 통일된 API로 제공합니다.
    """

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """모델을 학습합니다."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """예측 확률을 반환합니다."""
        pass

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """이진 예측을 반환합니다.

        Args:
            X: 입력 특성
            threshold: 분류 임계값 (기본 0.5)

        Returns:
            이진 예측 배열
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def save(self, filepath: Union[str, Path]):
        """학습된 모델을 파일로 저장합니다.

        Args:
            filepath: 저장할 파일 경로
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"모델 저장: {filepath}")

    def load(self, filepath: Union[str, Path]):
        """저장된 모델을 로드합니다.

        Args:
            filepath: 로드할 파일 경로
        """
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        logger.info(f"모델 로드: {filepath}")


# ============================================================
# XGBoost 모델
# ============================================================
class XGBoostModel(BaseModel):
    """XGBoost 기반 혈액독성 예측 모델입니다.

    Gradient Boosting 계열 중 가장 널리 사용되는 모델로,
    baseline 비교 모델의 핵심입니다.

    특징:
        - Feature importance 기반 해석 가능
        - 결측치를 내부적으로 처리 가능
        - Early stopping으로 과적합 방지
    """

    def __init__(self, config):
        """XGBoost 모델을 초기화합니다.

        Args:
            config: XGBoostConfig 인스턴스
        """
        super().__init__("XGBoost")
        self.config = config

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """XGBoost 모델을 학습합니다.

        Args:
            X_train: 학습 특성
            y_train: 학습 레이블
            X_val: 검증 특성 (early stopping용)
            y_val: 검증 레이블

        Returns:
            학습 결과 딕셔너리 (best_iteration, val_auc 등)
        """
        import xgboost as xgb

        # 클래스 불균형 보정
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = (
            neg_count / max(pos_count, 1)
            if self.config.scale_pos_weight == 1.0
            else self.config.scale_pos_weight
        )

        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.random_state,
            eval_metric="auc",
            tree_method="gpu_hist" if self.config.use_gpu else "hist",
            verbosity=0,
        )

        # Early stopping을 위한 검증 세트 설정
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True

        # 학습 결과 수집
        result = {"best_iteration": self.model.best_iteration if hasattr(self.model, "best_iteration") else -1}
        if X_val is not None:
            val_proba = self.predict_proba(X_val)
            result["val_auc"] = roc_auc_score(y_val, val_proba)
            logger.info(f"XGBoost 학습 완료 - Val AUC: {result['val_auc']:.4f}")

        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """양성 클래스의 예측 확률을 반환합니다.

        Args:
            X: 입력 특성

        Returns:
            양성 클래스 확률 (n_samples,)
        """
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, feature_names=None) -> Dict[str, float]:
        """특성 중요도를 반환합니다.

        Args:
            feature_names: 특성 이름 목록

        Returns:
            특성명-중요도 딕셔너리 (내림차순 정렬)
        """
        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        importance_dict = dict(zip(feature_names, importances))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


# ============================================================
# LightGBM 모델
# ============================================================
class LightGBMModel(BaseModel):
    """LightGBM 기반 혈액독성 예측 모델입니다.

    XGBoost보다 빠른 학습 속도와 메모리 효율성이 장점입니다.
    리프 단위 분할(Leaf-wise) 전략을 사용합니다.
    """

    def __init__(self, config):
        """LightGBM 모델을 초기화합니다.

        Args:
            config: LightGBMConfig 인스턴스
        """
        super().__init__("LightGBM")
        self.config = config

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """LightGBM 모델을 학습합니다.

        Args:
            X_train: 학습 특성
            y_train: 학습 레이블
            X_val: 검증 특성
            y_val: 검증 레이블

        Returns:
            학습 결과 딕셔너리
        """
        import lightgbm as lgb

        # 클래스 불균형 보정
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = (
            neg_count / max(pos_count, 1)
            if self.config.scale_pos_weight == 1.0
            else self.config.scale_pos_weight
        )

        self.model = lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_samples=self.config.min_child_samples,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.random_state,
            verbose=-1,
        )

        # Early stopping을 위한 검증 세트
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["eval_metric"] = "auc"

        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True

        result = {}
        if X_val is not None:
            val_proba = self.predict_proba(X_val)
            result["val_auc"] = roc_auc_score(y_val, val_proba)
            logger.info(f"LightGBM 학습 완료 - Val AUC: {result['val_auc']:.4f}")

        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """양성 클래스의 예측 확률을 반환합니다."""
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, feature_names=None) -> Dict[str, float]:
        """특성 중요도를 반환합니다."""
        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        importance_dict = dict(zip(feature_names, importances))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


# ============================================================
# Logistic Regression 모델
# ============================================================
class LogisticRegressionModel(BaseModel):
    """Logistic Regression 기반 혈액독성 예측 모델입니다.

    가장 기본적인 비교 모델로, 해석 가능성이 높습니다.
    회귀 계수를 통해 각 변수의 기여도를 직접 파악할 수 있습니다.
    """

    def __init__(self, config):
        """Logistic Regression 모델을 초기화합니다.

        Args:
            config: LogisticRegressionConfig 인스턴스
        """
        super().__init__("LogisticRegression")
        self.config = config

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """Logistic Regression 모델을 학습합니다.

        Args:
            X_train: 학습 특성
            y_train: 학습 레이블
            X_val: 검증 특성
            y_val: 검증 레이블

        Returns:
            학습 결과 딕셔너리
        """
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(
            C=self.config.C,
            penalty=self.config.penalty,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            class_weight=self.config.class_weight,
        )

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        result = {}
        if X_val is not None:
            val_proba = self.predict_proba(X_val)
            result["val_auc"] = roc_auc_score(y_val, val_proba)
            logger.info(
                f"Logistic Regression 학습 완료 - Val AUC: {result['val_auc']:.4f}"
            )

        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """양성 클래스의 예측 확률을 반환합니다."""
        return self.model.predict_proba(X)[:, 1]

    def get_coefficients(self, feature_names=None) -> Dict[str, float]:
        """회귀 계수를 반환합니다.

        양(+)의 계수는 혈액독성 위험 증가를 의미합니다.

        Args:
            feature_names: 특성 이름 목록

        Returns:
            특성명-계수 딕셔너리 (절대값 내림차순 정렬)
        """
        coeffs = self.model.coef_[0]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coeffs))]

        coeff_dict = dict(zip(feature_names, coeffs))
        return dict(sorted(coeff_dict.items(), key=lambda x: abs(x[1]), reverse=True))


# ============================================================
# 모델 팩토리
# ============================================================
def create_model(model_name: str, config) -> BaseModel:
    """모델 이름에 따라 적절한 모델 인스턴스를 생성합니다.

    Args:
        model_name: 모델 이름 ("xgboost", "lightgbm", "logistic_regression")
        config: 전체 Config 인스턴스

    Returns:
        BaseModel 인스턴스

    Raises:
        ValueError: 알 수 없는 모델 이름인 경우
    """
    models = {
        "xgboost": lambda: XGBoostModel(config.xgboost),
        "lightgbm": lambda: LightGBMModel(config.lightgbm),
        "logistic_regression": lambda: LogisticRegressionModel(
            config.logistic_regression
        ),
    }

    if model_name not in models:
        raise ValueError(
            f"알 수 없는 모델: {model_name}. "
            f"사용 가능한 모델: {list(models.keys())}"
        )

    return models[model_name]()
