import logging
from typing import Type, Optional, Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class TrainPrediction:
    def __init__(self, n_folds: int = 5, seed: int = 42):
        self.n_folds = n_folds
        self.seed = seed

    def cross_validation(self, model_class: Type, x: np.ndarray, y: np.ndarray,
                         feature_names: Optional[List[str]] = None, **model_params) -> Dict:

        stratified_k_fold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        fold_aucs = []
        oof_predictions = np.zeros(len(y))
        all_importances = []

        for fold_idx, (train_idx, val_idx) in enumerate(stratified_k_fold.split(x, y)):
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            model = model_class(**model_params)
            model.fit(x_train, y_train, x_val, y_val)

            val_proba = model.predict_proba(x_val)
            oof_predictions[val_idx] = val_proba

            fold_auc = roc_auc_score(y_val, val_proba)
            fold_aucs.append(fold_auc)

            if hasattr(model, "get_feature_importance"):
                all_importances.append(model.get_feature_importance(feature_names))

            logger.info(f"  Fold {fold_idx + 1}/{self.n_folds}: AUC = {fold_auc:.4f}")

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        logger.info(f"  CV 결과: AUC = {mean_auc:.4f} ± {std_auc:.4f}")

        avg_importances = {}
        if all_importances:
            all_keys = set()
            for imp in all_importances:
                all_keys.update(imp.keys())
            for key in all_keys:
                avg_importances[key] = np.mean([imp.get(key, 0) for imp in all_importances])
            avg_importances = dict(sorted(avg_importances.items(), key=lambda x: x[1], reverse=True))

        return {
            "fold_aucs": fold_aucs,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "oof_predictions": oof_predictions,
            "feature_importances": avg_importances,
        }

    def train_final(self, model_class: Type, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                    y_test: np.ndarray,
                    x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, **model_params) -> Dict:

        cv_results = self.cross_validation(model_class, x_train, y_train, **model_params)

        model = model_class(**model_params)
        model.fit(x_train, y_train, x_val, y_val)

        test_proba = model.predict_proba(x_test)
        test_auc = roc_auc_score(y_test, test_proba)
        logger.info(f" 최종모델 test AUC = {test_auc:.4f}")

        return {
            "model": model,
            "test_proba": test_proba,
            "test_auc": test_auc,
            "cv_results": cv_results
        }
