"""
lstm_forecast.py - 시계열 예측 + 분류 방법론

LSTM 조기 예측 성능 개선을 위한 Forecast-then-Classify 전략:
    Step 1: Forecaster — Week 0-2 CBC → Week 3-7 CBC 예측 (회귀)
    Step 2: Classifier — 전체 Week 0-7 + baseline → 독성 분류

가설:
    Week 0-2 초기 변화 패턴에서 미래 CBC 궤적을 예측할 수 있다면,
    예측된 전체 시퀀스를 분류 모델에 제공하여 정확한 독성 예측이 가능하다.

사용법:
    python lstm_forecast.py --n_patients 200 --seed 42 --ext_patients 150 --ext_seed 9999
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader

from config import Config
from generate_emr_data import generate_patients, generate_cbc_timeseries
from src.data.data_loader import DataLoader as EMRDataLoader
from src.data.dataset import CCRTDataset, create_dataloaders
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessing import EMRPreprocessor
from src.evaluation.metrics import bootstrap_ci, compute_all_metrics
from src.models.baseline_models import create_model
from src.models.forecaster import (
    CBCForecasterV2, ForecastDatasetV2, ForecasterTrainerV2,
)
from src.models.lstm_model import LSTMPredictor
from src.models.trainer import LSTMTrainer
from src.utils.helpers import set_seed

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def prepare_data(config, n_patients=200, seed=42):
    """데이터 생성 → 전처리 → 분할 → 특성 공학을 수행합니다.

    Returns:
        tuple of prepared data components
    """
    set_seed(seed)

    # 1) 데이터 생성
    patients = generate_patients(n_patients, seed=seed)
    cbc_results, _ = generate_cbc_timeseries(patients, seed=seed)
    patients_df = pd.DataFrame(patients)
    cbc_df = pd.DataFrame(cbc_results)
    preprocessor = EMRPreprocessor(config)
    df = preprocessor.run_full_pipeline(patients_df, cbc_df)

    # 2) 데이터 로드 및 분할
    data_loader = EMRDataLoader(config)
    processed_path = config.paths.processed_data_dir / "emr_processed.csv"
    df.to_csv(processed_path, index=False)
    df = data_loader.load_data(str(processed_path))
    df = data_loader.handle_missing_values(df)
    train_df, val_df, test_df = data_loader.split_data(df)

    # 3) 특성 공학
    fe = FeatureEngineer(config)
    train_df = fe.create_cbc_temporal_features(train_df)
    val_df = fe.create_cbc_temporal_features(val_df)
    test_df = fe.create_cbc_temporal_features(test_df)

    train_df = fe.encode_categorical(train_df, fit=True)
    val_df = fe.encode_categorical(val_df, fit=False)
    test_df = fe.encode_categorical(test_df, fit=False)

    feature_cols = fe.get_feature_columns(train_df, mode="baseline_cbc")
    all_feature_cols = feature_cols["all"]

    # 4) 타겟 추출
    target = config.data.primary_target
    y_train = train_df[target].values
    y_val = val_df[target].values
    y_test = test_df[target].values

    # 5) Baseline 특성
    baseline_cols = [c for c in feature_cols["baseline"] if c in train_df.columns]
    X_base_train = train_df[baseline_cols].values.astype(np.float32)
    X_base_val = val_df[baseline_cols].values.astype(np.float32)
    X_base_test = test_df[baseline_cols].values.astype(np.float32)

    # 6) CBC 시퀀스 (Week 0-7)
    cbc_train = fe.prepare_lstm_sequences(train_df)
    cbc_val = fe.prepare_lstm_sequences(val_df)
    cbc_test = fe.prepare_lstm_sequences(test_df)

    # 7) 스케일링
    scaler_cbc = StandardScaler()
    n_s, seq_len, n_f = cbc_train.shape
    cbc_train = scaler_cbc.fit_transform(
        cbc_train.reshape(-1, n_f)
    ).reshape(n_s, seq_len, n_f)
    cbc_val = scaler_cbc.transform(
        cbc_val.reshape(-1, n_f)
    ).reshape(len(val_df), seq_len, n_f)
    cbc_test = scaler_cbc.transform(
        cbc_test.reshape(-1, n_f)
    ).reshape(len(test_df), seq_len, n_f)

    scaler_base = StandardScaler()
    X_base_train = scaler_base.fit_transform(X_base_train)
    X_base_val = scaler_base.transform(X_base_val)
    X_base_test = scaler_base.transform(X_base_test)

    return (
        train_df, val_df, test_df, fe,
        y_train, y_val, y_test,
        cbc_train, cbc_val, cbc_test,
        X_base_train, X_base_val, X_base_test,
        scaler_cbc, scaler_base, baseline_cols, all_feature_cols,
    )


def train_tree_models(X_train, y_train, X_val, y_val, config):
    """Tree 모델을 학습합니다."""
    models = {}
    for model_name in ["xgboost", "lightgbm", "logistic_regression"]:
        model = create_model(model_name, config)
        model.fit(X_train, y_train)
        models[model_name] = model
        logger.info(f"  {model_name} 학습 완료")
    return models


def train_forecaster(cbc_train, cbc_val, X_base_train, X_base_val, config):
    """CBC Forecaster V2 (Week 0-2 + Baseline → Week 3-7)를 학습합니다.

    개선 사항:
        - Seq2Seq + Attention: 인코더 전체 출력 활용
        - Autoregressive 디코더: Week별 순차 예측
        - Baseline 조건부: 환자 특성 반영
        - Delta 예측: 변화량 예측으로 안정성 향상
        - Teacher Forcing + Scheduled Sampling

    Args:
        cbc_train: (n_train, 8, 6) — scaled CBC 시퀀스
        cbc_val: (n_val, 8, 6) — scaled CBC 시퀀스
        X_base_train: (n_train, n_baseline) — scaled Baseline
        X_base_val: (n_val, n_baseline) — scaled Baseline

    Returns:
        학습된 ForecasterTrainerV2
    """
    logger.info("=" * 60)
    logger.info("Step 1: Forecaster V2 학습 (Seq2Seq + Attention + Baseline + Delta)")
    logger.info("=" * 60)

    n_cbc_features = cbc_train.shape[2]  # 6
    n_baseline_features = X_base_train.shape[1]

    # Dataset 생성 (Baseline 포함)
    train_ds = ForecastDatasetV2(cbc_train, X_base_train, input_weeks=3)
    val_ds = ForecastDatasetV2(cbc_val, X_base_val, input_weeks=3)

    train_loader = TorchDataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = TorchDataLoader(val_ds, batch_size=32, shuffle=False)

    # Forecaster V2 모델 생성
    forecaster = CBCForecasterV2(
        n_cbc_features=n_cbc_features,
        n_baseline_features=n_baseline_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        n_forecast_weeks=5,
    )

    n_params = sum(p.numel() for p in forecaster.parameters())
    logger.info(f"ForecasterV2 파라미터 수: {n_params:,}")

    f_trainer = ForecasterTrainerV2(
        forecaster, lr=1e-3, weight_decay=1e-4,
    )
    f_trainer.train(train_loader, val_loader, num_epochs=300, patience=30)

    # 검증: 예측 정확도 확인
    pred_train = f_trainer.predict(cbc_train[:, :3, :], X_base_train)
    actual_train = cbc_train[:, 3:, :]
    mse = np.mean((pred_train - actual_train) ** 2)
    logger.info(f"ForecasterV2 학습 데이터 MSE: {mse:.6f}")

    pred_val = f_trainer.predict(cbc_val[:, :3, :], X_base_val)
    actual_val = cbc_val[:, 3:, :]
    mse_val = np.mean((pred_val - actual_val) ** 2)
    logger.info(f"ForecasterV2 검증 데이터 MSE: {mse_val:.6f}")

    return f_trainer


def train_classifier(
    cbc_train, cbc_val,
    X_base_train, X_base_val,
    y_train, y_val,
    config,
):
    """분류 LSTM을 학습합니다 (전체 시퀀스, 마스킹 없음).

    Forecaster가 미래 데이터를 채워주므로, Classifier는
    항상 전체 Week 0-7 데이터를 보고 분류합니다.

    개선: Weighted BCE (pos_weight) — 양성 케이스 놓침 방지
    """
    logger.info("=" * 60)
    logger.info("Step 2: Classifier LSTM 학습 (Weighted BCE)")
    logger.info("=" * 60)

    lstm_config = config.lstm
    lstm_config.input_size = 6
    lstm_config.baseline_input_size = X_base_train.shape[1]

    train_ds = _SimpleSeqDataset(cbc_train, X_base_train, y_train)
    val_ds = _SimpleSeqDataset(cbc_val, X_base_val, y_val)

    train_loader = TorchDataLoader(train_ds, batch_size=lstm_config.batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_ds, batch_size=lstm_config.batch_size, shuffle=False)

    # pos_weight 계산: 음성/양성 비율 (양성 miss 페널티 증가)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    logger.info(f"  클래스 비율: 양성={n_pos:.0f}, 음성={n_neg:.0f}, pos_weight={pos_weight:.2f}")

    model = LSTMPredictor(lstm_config)
    trainer = LSTMTrainer(model, config, pos_weight=pos_weight)
    history = trainer.train(train_loader, val_loader)

    logger.info(f"Classifier 학습 완료 — Val AUC: {history['val_auc'][-1]:.4f}")
    return trainer


class _SimpleSeqDataset(torch.utils.data.Dataset):
    """마스크 채널 없이 CBC 시퀀스를 제공하는 간단한 Dataset.

    Forecast 방법론에서 Classifier는 항상 전체 시퀀스를 받으므로
    마스크 표시 채널이 불필요합니다.
    """

    def __init__(self, cbc_sequences, baseline_features, targets=None):
        self.cbc = torch.FloatTensor(cbc_sequences)
        self.baseline = torch.FloatTensor(baseline_features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.cbc)

    def __getitem__(self, idx):
        sample = {
            "cbc_seq": self.cbc[idx],
            "baseline": self.baseline[idx],
        }
        if self.targets is not None:
            sample["target"] = self.targets[idx]
        return sample


def evaluate_external(
    tree_models, classifier_trainer, forecaster_trainer,
    fe, scaler_cbc, scaler_base,
    baseline_cols, all_feature_cols,
    config,
    ext_patients=150, ext_seed=9999,
):
    """외부 검증: Forecast → Classify 파이프라인 평가."""
    logger.info("=" * 60)
    logger.info("External Validation 데이터 생성")
    logger.info("=" * 60)

    # 외부 데이터 생성 및 전처리
    ext_patients_list = generate_patients(ext_patients, seed=ext_seed)
    ext_cbc_list, _ = generate_cbc_timeseries(ext_patients_list, seed=ext_seed)
    ext_patients_df = pd.DataFrame(ext_patients_list)
    ext_cbc_df = pd.DataFrame(ext_cbc_list)
    preprocessor = EMRPreprocessor(config)
    ext_df = preprocessor.run_full_pipeline(ext_patients_df, ext_cbc_df)

    ext_path = config.paths.processed_data_dir / "forecast_external.csv"
    ext_df.to_csv(ext_path, index=False)

    # 특성 공학
    ext_df = fe.create_cbc_temporal_features(ext_df)
    ext_df = fe.encode_categorical(ext_df, fit=False)

    target = config.data.primary_target
    y_ext = ext_df[target].values

    logger.info(f"External 데이터: {len(ext_df)}명")
    logger.info(f"  양성: {y_ext.sum():.0f} ({y_ext.mean()*100:.1f}%)")

    # ----- Tree 모델 평가 -----
    for col in all_feature_cols:
        if col not in ext_df.columns:
            ext_df[col] = 0
    X_ext = ext_df[all_feature_cols].values.astype(np.float32)
    X_ext = np.nan_to_num(X_ext, nan=0.0)

    results = {}

    logger.info("")
    logger.info("=" * 60)
    logger.info("External Validation 결과")
    logger.info("=" * 60)

    for model_name, model in tree_models.items():
        ext_proba = model.predict_proba(X_ext)
        metrics = compute_all_metrics(y_ext, ext_proba, model_name=f"{model_name}_ext")
        _, ci_lower, ci_upper = bootstrap_ci(y_ext, ext_proba)
        metrics["auroc_ci"] = (ci_lower, ci_upper)
        results[model_name] = metrics

        logger.info(
            f"  {model_name:25s} | AUROC={metrics['auroc']:.4f} "
            f"({ci_lower:.4f}-{ci_upper:.4f}) | "
            f"Sens={metrics['sensitivity']:.4f} | Spec={metrics['specificity']:.4f}"
        )

    # ----- LSTM Forecast+Classify 평가 -----
    # Baseline 특성 준비 (Forecaster V2에서도 사용)
    for col in baseline_cols:
        if col not in ext_df.columns:
            ext_df[col] = 0
    X_base_ext = ext_df[baseline_cols].values.astype(np.float32)
    X_base_ext = np.nan_to_num(X_base_ext, nan=0.0)
    X_base_ext = scaler_base.transform(X_base_ext)

    # CBC 스케일링
    cbc_ext = fe.prepare_lstm_sequences(ext_df)
    n_s, seq_len, n_f = cbc_ext.shape
    cbc_ext_scaled = scaler_cbc.transform(
        cbc_ext.reshape(-1, n_f)
    ).reshape(n_s, seq_len, n_f)

    # Forecaster V2로 Week 3-7 예측 (Baseline 포함)
    cbc_input = cbc_ext_scaled[:, :3, :]  # Week 0-2 실제 데이터
    cbc_predicted = forecaster_trainer.predict(cbc_input, X_base_ext)  # Week 3-7 예측

    # [실제 Week 0-2] + [예측 Week 3-7] 결합
    cbc_combined = np.concatenate([cbc_input, cbc_predicted], axis=1)  # (n_s, 8, 6)
    logger.info(
        f"  Forecast 결합: 실제 Week 0-2 + 예측 Week 3-7 → shape={cbc_combined.shape}"
    )

    # Classifier에 전체 시퀀스 입력 (마스크 채널 없음)
    ext_ds = _SimpleSeqDataset(cbc_combined, X_base_ext, y_ext)
    ext_loader = TorchDataLoader(ext_ds, batch_size=32, shuffle=False)

    ext_proba, _ = classifier_trainer.predict(ext_loader)
    metrics = compute_all_metrics(y_ext, ext_proba, model_name="lstm_forecast_ext")
    _, ci_lower, ci_upper = bootstrap_ci(y_ext, ext_proba)
    metrics["auroc_ci"] = (ci_lower, ci_upper)
    results["lstm_forecast"] = metrics

    logger.info(
        f"  {'lstm_forecast':25s} | AUROC={metrics['auroc']:.4f} "
        f"({ci_lower:.4f}-{ci_upper:.4f}) | "
        f"Sens={metrics['sensitivity']:.4f} | Spec={metrics['specificity']:.4f}"
    )

    # ----- 다중 Threshold 분석 -----
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_ext, ext_proba)
    logger.info("")
    logger.info("  [Threshold 분석 — LSTM Forecast+Classify]")
    logger.info(f"  {'Threshold':>10s} | {'Sens':>6s} | {'Spec':>6s} | {'설명'}")
    logger.info("  " + "-" * 50)

    # Youden's J (현재 기본)
    logger.info(
        f"  {metrics['optimal_threshold']:10.3f} | "
        f"{metrics['sensitivity']:6.4f} | {metrics['specificity']:6.4f} | Youden's J"
    )

    # 목표 Sensitivity별 threshold
    for target_sens in [0.80, 0.85, 0.90]:
        idx = np.argmin(np.abs(tpr - target_sens))
        th = thresholds[idx]
        sens = tpr[idx]
        spec = 1 - fpr[idx]
        logger.info(f"  {th:10.3f} | {sens:6.4f} | {spec:6.4f} | Sens≥{target_sens:.0%}")

    # ----- 참고: Forecast 없이 실제 전체 데이터 사용 시 (Oracle) -----
    ext_ds_oracle = _SimpleSeqDataset(cbc_ext_scaled, X_base_ext, y_ext)
    oracle_loader = TorchDataLoader(ext_ds_oracle, batch_size=32, shuffle=False)
    oracle_proba, _ = classifier_trainer.predict(oracle_loader)
    oracle_metrics = compute_all_metrics(y_ext, oracle_proba, model_name="lstm_oracle")

    logger.info(
        f"  {'lstm_oracle (전체데이터)':25s} | AUROC={oracle_metrics['auroc']:.4f} | "
        f"Sens={oracle_metrics['sensitivity']:.4f} | Spec={oracle_metrics['specificity']:.4f}"
    )
    results["lstm_oracle"] = oracle_metrics

    return results


def main():
    parser = argparse.ArgumentParser(description="LSTM Forecast+Classify 방법론")
    parser.add_argument("--n_patients", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ext_patients", type=int, default=150)
    parser.add_argument("--ext_seed", type=int, default=9999)
    args = parser.parse_args()

    config = Config()
    config.paths.ensure_dirs()

    logger.info("=" * 60)
    logger.info("LSTM Forecast + Classify 방법론")
    logger.info("=" * 60)

    # 1) 데이터 준비
    (
        train_df, val_df, test_df, fe,
        y_train, y_val, y_test,
        cbc_train, cbc_val, cbc_test,
        X_base_train, X_base_val, X_base_test,
        scaler_cbc, scaler_base, baseline_cols, all_feature_cols,
    ) = prepare_data(config, args.n_patients, args.seed)

    # 2) Tree 모델 학습
    logger.info("")
    logger.info("Tree 모델 학습")
    X_train_flat = train_df[all_feature_cols].values.astype(np.float32)
    X_val_flat = val_df[all_feature_cols].values.astype(np.float32)
    X_train_flat = np.nan_to_num(X_train_flat, nan=0.0)

    tree_models = train_tree_models(X_train_flat, y_train, X_val_flat, y_val, config)

    # 3) Forecaster V2 학습 (Baseline 포함)
    forecaster_trainer = train_forecaster(
        cbc_train, cbc_val, X_base_train, X_base_val, config,
    )

    # 4) Classifier LSTM 학습 (전체 시퀀스, 마스킹 없음)
    classifier_trainer = train_classifier(
        cbc_train, cbc_val,
        X_base_train, X_base_val,
        y_train, y_val,
        config,
    )

    # 5) 내부 테스트: Forecast + Classify
    logger.info("")
    logger.info("내부 테스트 평가")

    # Forecast V2 방식 (Baseline 포함)
    cbc_test_input = cbc_test[:, :3, :]
    cbc_test_pred = forecaster_trainer.predict(cbc_test_input, X_base_test)
    cbc_test_combined = np.concatenate([cbc_test_input, cbc_test_pred], axis=1)

    test_ds = _SimpleSeqDataset(cbc_test_combined, X_base_test, y_test)
    test_loader = TorchDataLoader(test_ds, batch_size=32, shuffle=False)
    test_proba, _ = classifier_trainer.predict(test_loader)
    test_metrics = compute_all_metrics(y_test, test_proba, model_name="internal_forecast")
    logger.info(
        f"  Forecast+Classify AUROC={test_metrics['auroc']:.4f} | "
        f"Sens={test_metrics['sensitivity']:.4f} | Spec={test_metrics['specificity']:.4f}"
    )

    # Oracle (실제 전체 데이터)
    test_ds_oracle = _SimpleSeqDataset(cbc_test, X_base_test, y_test)
    oracle_loader = TorchDataLoader(test_ds_oracle, batch_size=32, shuffle=False)
    oracle_proba, _ = classifier_trainer.predict(oracle_loader)
    oracle_metrics = compute_all_metrics(y_test, oracle_proba, model_name="internal_oracle")
    logger.info(
        f"  Oracle (전체 데이터)    AUROC={oracle_metrics['auroc']:.4f} | "
        f"Sens={oracle_metrics['sensitivity']:.4f} | Spec={oracle_metrics['specificity']:.4f}"
    )

    # 6) 외부 검증
    results = evaluate_external(
        tree_models, classifier_trainer, forecaster_trainer,
        fe, scaler_cbc, scaler_base,
        baseline_cols, all_feature_cols,
        config,
        ext_patients=args.ext_patients, ext_seed=args.ext_seed,
    )

    # 7) 최종 요약
    logger.info("")
    logger.info("=" * 60)
    logger.info("최종 결과 요약 (External Validation)")
    logger.info("=" * 60)
    logger.info(f"{'모델':25s} | {'AUROC':>7s} | {'Sens':>6s} | {'Spec':>6s}")
    logger.info("-" * 55)
    for name, m in results.items():
        logger.info(
            f"{name:25s} | {m['auroc']:7.4f} | "
            f"{m['sensitivity']:6.4f} | {m['specificity']:6.4f}"
        )

    logger.info("")
    logger.info("Forecast+Classify 완료!")

    # 8) 시각화 및 Forecaster per-feature 분석
    plot_results(
        results, forecaster_trainer,
        cbc_test, X_base_test, y_test,
        scaler_cbc, config,
    )


def plot_results(
    results, forecaster_trainer,
    cbc_test, X_base_test, y_test,
    scaler_cbc, config,
):
    """시각화: ROC 곡선, 성능 비교, Forecaster per-feature MSE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    import matplotlib
    matplotlib.rcParams["font.family"] = "AppleGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "font.size": 12,
        "figure.dpi": 150,
        "figure.facecolor": "white",
    })

    out_dir = config.paths.figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cbc_features = config.data.cbc_features  # [WBC, ANC, ALC, AMC, PLT, Hb]

    # ============================================================
    # 1) Forecaster per-feature MSE (호중구 vs 단핵구 등)
    # ============================================================
    cbc_input = cbc_test[:, :3, :]
    cbc_pred = forecaster_trainer.predict(cbc_input, X_base_test)
    cbc_actual = cbc_test[:, 3:, :]

    # Inverse scale하여 원래 단위로 복원
    n_f = cbc_test.shape[2]
    pred_orig = scaler_cbc.inverse_transform(
        cbc_pred.reshape(-1, n_f)
    ).reshape(cbc_pred.shape)
    actual_orig = scaler_cbc.inverse_transform(
        cbc_actual.reshape(-1, n_f)
    ).reshape(cbc_actual.shape)

    # Per-feature MSE (scaled)
    per_feature_mse_scaled = np.mean((cbc_pred - cbc_actual) ** 2, axis=(0, 1))
    # Per-feature MAE (원래 단위)
    per_feature_mae_orig = np.mean(np.abs(pred_orig - actual_orig), axis=(0, 1))

    logger.info("")
    logger.info("=" * 60)
    logger.info("Forecaster Per-Feature 분석 (Week 3-7 예측 정확도)")
    logger.info("=" * 60)
    logger.info(f"{'Feature':>8s} | {'MSE(scaled)':>12s} | {'MAE(원래단위)':>14s}")
    logger.info("-" * 42)
    for i, feat in enumerate(cbc_features):
        logger.info(f"{feat:>8s} | {per_feature_mse_scaled[i]:12.4f} | {per_feature_mae_orig[i]:14.4f}")

    # ANC vs AMC 비교
    anc_idx = cbc_features.index("ANC")
    amc_idx = cbc_features.index("AMC")
    logger.info("")
    logger.info(f"호중구(ANC) 예측 MAE: {per_feature_mae_orig[anc_idx]:.4f} (10^3/uL)")
    logger.info(f"단핵구(AMC) 예측 MAE: {per_feature_mae_orig[amc_idx]:.4f} (10^3/uL)")

    # Per-feature MSE 바 차트
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if f in ["ANC", "WBC"] else "#3498db" if f == "AMC" else "#95a5a6"
              for f in cbc_features]
    bars = ax.bar(cbc_features, per_feature_mse_scaled, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("MSE (scaled)")
    ax.set_title("Forecaster Per-Feature MSE (Week 3-7 예측)")
    for bar, val in zip(bars, per_feature_mse_scaled):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.legend(
        handles=[
            plt.Rectangle((0,0), 1, 1, fc="#e74c3c", label="핵심 (ANC, WBC)"),
            plt.Rectangle((0,0), 1, 1, fc="#3498db", label="AMC"),
            plt.Rectangle((0,0), 1, 1, fc="#95a5a6", label="기타"),
        ],
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(out_dir / "forecaster_per_feature_mse.png")
    plt.close()
    logger.info(f"  저장: {out_dir / 'forecaster_per_feature_mse.png'}")

    # ============================================================
    # 2) Forecaster 예측 vs 실제 — ANC, AMC Week별 비교
    # ============================================================
    weeks = [3, 4, 5, 6, 7]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, feat_idx, feat_name, color in [
        (axes[0], anc_idx, "ANC (호중구)", "#e74c3c"),
        (axes[1], amc_idx, "AMC (단핵구)", "#3498db"),
    ]:
        actual_mean = actual_orig[:, :, feat_idx].mean(axis=0)
        actual_std = actual_orig[:, :, feat_idx].std(axis=0)
        pred_mean = pred_orig[:, :, feat_idx].mean(axis=0)
        pred_std = pred_orig[:, :, feat_idx].std(axis=0)

        ax.plot(weeks, actual_mean, "o-", color=color, label="실제", linewidth=2)
        ax.fill_between(weeks, actual_mean - actual_std, actual_mean + actual_std,
                         alpha=0.15, color=color)
        ax.plot(weeks, pred_mean, "s--", color="gray", label="예측", linewidth=2)
        ax.fill_between(weeks, pred_mean - pred_std, pred_mean + pred_std,
                         alpha=0.15, color="gray")
        ax.set_xlabel("Week")
        ax.set_ylabel(f"{feat_name} (10³/µL)")
        ax.set_title(f"{feat_name} — 실제 vs 예측")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "forecaster_anc_amc_comparison.png")
    plt.close()
    logger.info(f"  저장: {out_dir / 'forecaster_anc_amc_comparison.png'}")

    # ============================================================
    # 3) ROC 곡선 비교
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 8))

    # 외부 검증의 ROC 데이터는 results에서 가져올 수 없으므로
    # 내부 테스트 데이터로 ROC 플롯
    # — Forecast+Classify
    cbc_test_pred = forecaster_trainer.predict(cbc_test[:, :3, :], X_base_test)
    cbc_combined = np.concatenate([cbc_test[:, :3, :], cbc_test_pred], axis=1)

    from torch.utils.data import DataLoader as TDL
    # 이 함수는 main()에서 호출되므로 classifier_trainer에 접근 불가
    # → results dict의 AUROC 값으로 바 차트 생성

    model_names = list(results.keys())
    aurocs = [results[n]["auroc"] for n in model_names]
    senses = [results[n]["sensitivity"] for n in model_names]
    specs = [results[n]["specificity"] for n in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, aurocs, width, label="AUROC", color="#2ecc71", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, senses, width, label="Sensitivity", color="#e74c3c", edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, specs, width, label="Specificity", color="#3498db", edgecolor="black", linewidth=0.5)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in model_names], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("External Validation — 모델별 성능 비교")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_bar.png")
    plt.close()
    logger.info(f"  저장: {out_dir / 'model_comparison_bar.png'}")

    logger.info("시각화 완료!")


if __name__ == "__main__":
    main()
