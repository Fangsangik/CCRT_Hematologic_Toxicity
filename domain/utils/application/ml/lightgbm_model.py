import logging
from typing import Optional, Dict

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class LightGBMModel:
    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict:
        pos = y_train.sum()
        neg = len(y_train) - pos
        self.params.setdefault("scale_pos_weight", neg / max(pos, 1))
        self.params.setdefault("verbose", -1)
        self.params.setdefault("random_state", 42)

        self.model = lgb.LGBMClassifier(**self.params)

        fit_params = {}
        if x_val is not None and y_val is not None:
            fit_params["eval_set"] = [(x_val, y_val)]
            fit_params["eval_metric"] = "auc"
            fit_params["callbacks"] = [
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        self.model.fit(x_train, y_train, **fit_params)

        result = {}
        if x_val is not None and y_val is not None:
            val_proba = self.predict_proba(x_val)
            result["val_auc"] = roc_auc_score(y_val, val_proba)
            logger.info(f"LightGBM Validation AUC: {result['val_auc']:.4f}")
        return result

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]

    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        importance_dict = dict(zip(feature_names, importances))
        return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

