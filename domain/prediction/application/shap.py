import logging
from pathlib import Path
import pandas as pd
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt

import shap

import numpy as np

logger = logging.getLogger(__name__)


class RunSHAP:
    """SHAP 분석 Use Case입니다."""

    def __init__(self, model, x: np.ndarray, feature_names: Optional[List[str]] = None):
        self.model = model
        self.x = x
        self.feature_names = feature_names or [f"feature_{i}" for i in range(x.shape[1])]
        self.shap_values = None
        self.explainer = None

    def compute(self) -> np.ndarray:
        """TreeExplainer로 SHAP 값 계산"""
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.x)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        logger.info(f"SHAP 값 계싼 완료 shape={self.shap_values.shape}")
        return self.shap_values

    def plot_summary(self, save_path: Optional[str] = None, plot_type: str = "dot",
                     max_display: int = 20) -> plt.Figure:
        """SHAP summary plot 생성"""
        if self.shap_values is None:
            self.compute()
        fig = plt.figure(figsize=(10, max(6, max_display * 0.4)))
        shap.summary_plot(self.shap_values, self.x, feature_names=self.feature_names, plot_type=plot_type,
                          max_display=max_display, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        return fig

    def get_top_features(self, top_n: int = 20) -> List[Tuple[str, float]]:
        if self.shap_values is None:
            self.compute()
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        pairs = sorted(zip(self.feature_names, mean_abs), key=lambda x: x[1], reverse=True)
        return pairs[:top_n]

    def export_values(self, filepath: str) -> pd.DataFrame:
        if self.shap_values is None:
            self.compute()
        df = pd.DataFrame(self.shap_values, columns=self.feature_names)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        return df
