import logging
from typing import Optional, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class LogisticModel:

    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict:
        self.params.setdefault("max_iter", 1000)
        self.params.setdefault("class_weight", "balanced")
        self.params.setdefault("random_state", 42)

        self.model = LogisticRegression(**self.params)
        self.model.fit(x_train, y_train)

        result = {}
        if x_val is not None and y_val is not None:
            val_proba = self.predict_proba(x_val)
            result["val_auc"] = roc_auc_score(y_val, val_proba)
            logger.info(f"LogisticRegression Validation AUC: {result['val_auc']:.4f}")
        return result

    def predict_proba(self, x : np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]

    def get_coefficients(self, feature_names : Optional[List] = None) -> Dict[str, float]:
        coefficients = self.model.coef_[0]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        coef_dict = dict(zip(feature_names, coefficients))
        return dict(sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True))

    def get_feature_importance(self, feature_names : Optional[List] = None) -> Dict[str, float]:
        coefficients = np.abs(self.model.coef_[0])
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        importance_dict = dict(zip(feature_names, coefficients))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))