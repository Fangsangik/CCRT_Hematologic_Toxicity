import logging
from pathlib import Path
import pandas as pd
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class CSVRepository:

    def load(self, filepath : str, **kwargs) -> pd.DataFrame:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

        suffix = path.suffix.lower()
        loaders = {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel,
            ".parquet": pd.read_parquet,
        }

        if suffix not in loaders :
            raise ValueError(f"지원하지 않는 파일 형식: {suffix}")

        df = loaders[suffix](path, **kwargs)
        logger.info(f"데이터 로드 완료: {path} ({len(df)}행 x {len(df.columns)}열)")
        return df

    def save(self, df : pd.DataFrame, filepath : str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"데이터 저장 완료: {path}")

    def handle_missing (self, df : pd.DataFrame, strategy : str = "median", max_missing_rate : float = 0.3) -> pd.DataFrame :
        df = df.copy()

        missing_rates = df.isnull().mean()
        cols_to_drop = missing_rates[missing_rates > max_missing_rate].index.tolist()
        if cols_to_drop :
            logger.warning(f"결측률 {max_missing_rate*100:.0f}% 초과 변수 제거: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if strategy == "median" :
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "mean" :
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols :
            if df[col].isnull().any() :
                df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def split (self, df : pd.DataFrame, target_col : str, test_size : float = 0.2, val_size : float = 0.1, random_state : int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] :
        stratify = df[target_col] if target_col in df.columns else None

        train_val, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)

        val_ratio = val_size / (1 - test_size)
        stratify_train_val = train_val[target_col] if target_col in train_val.columns else None

        train, val = train_test_split(
            train_val, test_size=val_ratio, random_state=random_state, stratify=stratify_train_val
        )

        logger.info("데이터 분할 완료: train=%d, val=%d, test=%d", len(train), len(val), len(test))
        return train, val, test
