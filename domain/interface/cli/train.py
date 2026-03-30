"""학습 CLI"""
import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "xgboost": "shared.infrastructure.ml.xgboost_model.XGBoostModel",
    "lightgbm": "shared.infrastructure.ml.lightgbm_model.LightGBMModel",
    "logistic_regression": "shared.infrastructure.ml.logistic_model.LogisticModel",
}


def _import_model_class(name: str):
    """모델 클래스를 동적 임포트합니다."""
    module_path, class_name = MODEL_MAP[name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@click.command()
@click.option("--mode", type=click.Choice(["prediction", "screening"]), default="prediction")
@click.option("--model", type=click.Choice(["xgboost", "lightgbm", "logistic_regression"]), default="xgboost")
@click.option("--data", default=None, help="데이터 파일 경로")
@click.option("--n-folds", default=5, help="CV 폴드 수")
@click.option("--seed", default=42, help="랜덤 시드")
def train(mode: str, model: str, data: str, n_folds: int, seed: int):
    """모델을 학습합니다."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from shared.infrastructure.repository.csv_repository import CSVRepository
    from prediction.domain import FeatureService
    from prediction.application.use_cases.train_prediction import TrainPrediction
    from shared.infrastructure.repository.model_repository import ModelRepository

    # 데이터 로드
    csv_repo = CSVRepository()
    if not data:
        click.echo("--data 옵션으로 데이터 파일 경로를 지정하세요.")
        raise SystemExit(1)
    data_path = data
    df = csv_repo.load(data_path)
    df = csv_repo.handle_missing(df)

    # Feature 추출
    feature_service = FeatureService()
    df, feature_names = feature_service.extract_all(df, mode="baseline_cbc")

    # 범주형 변수 원-핫 인코딩
    cat_cols = [c for c in ["sex", "stage", "t_stage", "n_stage", "ecog_ps", "chemo_regimen"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        # feature_names에서 원본 범주형 → 인코딩된 컬럼으로 교체
        encoded_cols = [c for c in df.columns if any(c.startswith(f"{cat}_") for cat in cat_cols)]
        feature_names = [f for f in feature_names if f not in cat_cols] + encoded_cols

    # 학습/테스트 분할
    train_df, val_df, test_df = csv_repo.split(df)

    target = "grade3_neutropenia"
    available_features = [f for f in feature_names if f in train_df.columns]

    x_train = train_df[available_features].values.astype(np.float32)
    y_train = train_df[target].values
    x_val = val_df[available_features].values.astype(np.float32)
    y_val = val_df[target].values
    x_test = test_df[available_features].values.astype(np.float32)
    y_test = test_df[target].values

    # 모델 학습
    model_class = _import_model_class(model)
    trainer = TrainPrediction(n_folds=n_folds, seed=seed)
    result = trainer.train_final(
        model_class, x_train, y_train, x_test, y_test, x_val, y_val,
    )

    click.echo(f"Test AUC: {result['test_auc']:.4f}")
    click.echo(f"CV AUC: {result['cv_results']['mean_auc']:.4f} ± {result['cv_results']['std_auc']:.4f}")

    # 모델 저장
    model_repo = ModelRepository()
    model_path = str(PROJECT_ROOT / "outputs" / "models" / f"{model}_{mode}.pkl")
    model_repo.save(result["model"], model_path)
    click.echo(f"모델 저장: {model_path}")


if __name__ == "__main__":
    train()
