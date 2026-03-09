"""
lstm_twostage.py - 2단계 학습 (Pre-train → Fine-tune) 방법론

LSTM 조기 예측 성능 개선을 위한 2단계 학습 전략:
    Phase 1: 전체 시퀀스(Week 0-7)로 궤적 패턴 완전 학습
    Phase 2: 마스킹(80%)으로 부분 데이터 예측 적응 (Fine-tune)

가설:
    LSTM이 전체 CBC 궤적을 먼저 학습하면, Week 0-2만 봐도
    미래 패턴을 내재적으로 추론하여 독성을 예측할 수 있다.

사용법:
    python lstm_twostage.py --n_patients 200 --seed 42 --ext_patients 150 --ext_seed 9999
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config import Config
from generate_emr_data import generate_patients, generate_cbc_timeseries
from src.data.data_loader import DataLoader as EMRDataLoader
from src.data.dataset import CCRTDataset, create_dataloaders
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessing import EMRPreprocessor
from src.evaluation.metrics import bootstrap_ci, compute_all_metrics
from src.models.baseline_models import create_model
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
        (train_df, val_df, test_df, fe, y_train, y_val, y_test,
         cbc_train, cbc_val, cbc_test,
         X_base_train, X_base_val, X_base_test,
         scaler_cbc, scaler_base, baseline_cols, all_feature_cols)
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
    """Tree 모델(XGBoost, LightGBM, LogReg)을 학습합니다."""
    models = {}
    for model_name in ["xgboost", "lightgbm", "logistic_regression"]:
        model = create_model(model_name, config)
        model.fit(X_train, y_train)
        models[model_name] = model
        logger.info(f"  {model_name} 학습 완료")
    return models


def train_lstm_twostage(
    cbc_train, cbc_val,
    X_base_train, X_base_val,
    y_train, y_val,
    config,
):
    """2단계 LSTM 학습을 수행합니다.

    Phase 1: 전체 시퀀스로 학습 (mask_future_prob=0.0)
    Phase 2: 마스킹 데이터로 fine-tune (mask_future_prob=0.8, lr=1e-4)

    Returns:
        학습된 LSTMTrainer
    """
    lstm_config = config.lstm
    lstm_config.baseline_input_size = X_base_train.shape[1]

    # ============================================================
    # Phase 1: 전체 궤적 학습 (마스킹 없음)
    # ============================================================
    logger.info("=" * 60)
    logger.info("Phase 1: 전체 시퀀스 학습 (Week 0-7, 마스킹 없음)")
    logger.info("=" * 60)

    # 마스킹 없는 Dataset
    train_ds_phase1 = CCRTDataset(cbc_train, X_base_train, y_train)
    val_ds_phase1 = CCRTDataset(cbc_val, X_base_val, y_val)

    dl_phase1 = create_dataloaders(
        train_ds_phase1, val_ds_phase1,
        batch_size=lstm_config.batch_size,
    )

    model = LSTMPredictor(lstm_config)
    trainer = LSTMTrainer(model, config)

    history1 = trainer.train(dl_phase1["train"], dl_phase1["val"])

    phase1_path = str(config.paths.model_dir / "lstm_twostage_phase1.pt")
    trainer.save_checkpoint(phase1_path)
    logger.info(f"Phase 1 완료 — Val AUC: {history1['val_auc'][-1]:.4f}")

    # ============================================================
    # Phase 2: 마스킹 데이터로 Fine-tune
    # ============================================================
    logger.info("=" * 60)
    logger.info("Phase 2: Fine-tune (mask_future_prob=0.8, lr=1e-4)")
    logger.info("=" * 60)

    # Optimizer/Scheduler/EarlyStopping 리셋 (lr=1e-4)
    trainer.reset_optimizer(lr=1e-4)

    # 80% 마스킹 Dataset
    train_ds_phase2 = CCRTDataset(
        cbc_train, X_base_train, y_train,
        mask_future_prob=0.8, mask_start_idx=3,
    )
    val_ds_phase2 = CCRTDataset(cbc_val, X_base_val, y_val)

    dl_phase2 = create_dataloaders(
        train_ds_phase2, val_ds_phase2,
        batch_size=lstm_config.batch_size,
    )

    history2 = trainer.train(dl_phase2["train"], dl_phase2["val"])

    phase2_path = str(config.paths.model_dir / "lstm_twostage_phase2.pt")
    trainer.save_checkpoint(phase2_path)
    logger.info(f"Phase 2 완료 — Val AUC: {history2['val_auc'][-1]:.4f}")

    return trainer


def evaluate_external(
    tree_models, lstm_trainer,
    fe, scaler_cbc, scaler_base,
    baseline_cols, all_feature_cols,
    config,
    ext_patients=150, ext_seed=9999,
):
    """외부 검증 데이터에서 모든 모델의 성능을 평가합니다."""
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

    ext_path = config.paths.processed_data_dir / "twostage_external.csv"
    ext_df.to_csv(ext_path, index=False)

    # 특성 공학 (학습 인코더 재사용)
    ext_df = fe.create_cbc_temporal_features(ext_df)
    ext_df = fe.encode_categorical(ext_df, fit=False)

    target = config.data.primary_target
    y_ext = ext_df[target].values

    logger.info(f"External 데이터: {len(ext_df)}명")
    logger.info(f"  양성: {y_ext.sum():.0f} ({y_ext.mean()*100:.1f}%)")

    # ----- 결측 컬럼 처리 -----
    for col in all_feature_cols:
        if col not in ext_df.columns:
            ext_df[col] = 0
    X_ext = ext_df[all_feature_cols].values.astype(np.float32)
    X_ext = np.nan_to_num(X_ext, nan=0.0)

    results = {}

    # ----- Tree 모델 평가 -----
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

    # ----- LSTM 2단계 평가 -----
    cbc_ext = fe.prepare_lstm_sequences(ext_df)
    n_s, seq_len, n_f = cbc_ext.shape
    cbc_ext = scaler_cbc.transform(
        cbc_ext.reshape(-1, n_f)
    ).reshape(n_s, seq_len, n_f)

    for col in baseline_cols:
        if col not in ext_df.columns:
            ext_df[col] = 0
    X_base_ext = ext_df[baseline_cols].values.astype(np.float32)
    X_base_ext = np.nan_to_num(X_base_ext, nan=0.0)
    X_base_ext = scaler_base.transform(X_base_ext)

    # mask_future_prob=1.0 → Week 3-7 100% 마스킹 (조기 예측 시뮬레이션)
    ext_dataset = CCRTDataset(
        cbc_ext, X_base_ext, y_ext,
        mask_future_prob=1.0, mask_start_idx=3,
    )
    ext_loader = torch.utils.data.DataLoader(
        ext_dataset, batch_size=32, shuffle=False,
    )

    ext_proba, _ = lstm_trainer.predict(ext_loader)
    metrics = compute_all_metrics(y_ext, ext_proba, model_name="lstm_twostage_ext")
    _, ci_lower, ci_upper = bootstrap_ci(y_ext, ext_proba)
    metrics["auroc_ci"] = (ci_lower, ci_upper)
    results["lstm_twostage"] = metrics

    logger.info(
        f"  {'lstm_twostage':25s} | AUROC={metrics['auroc']:.4f} "
        f"({ci_lower:.4f}-{ci_upper:.4f}) | "
        f"Sens={metrics['sensitivity']:.4f} | Spec={metrics['specificity']:.4f}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="LSTM 2단계 학습 방법론")
    parser.add_argument("--n_patients", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ext_patients", type=int, default=150)
    parser.add_argument("--ext_seed", type=int, default=9999)
    args = parser.parse_args()

    config = Config()
    config.paths.ensure_dirs()

    logger.info("=" * 60)
    logger.info("LSTM 2단계 학습 (Pre-train → Fine-tune)")
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

    # 3) LSTM 2단계 학습
    lstm_trainer = train_lstm_twostage(
        cbc_train, cbc_val,
        X_base_train, X_base_val,
        y_train, y_val,
        config,
    )

    # 4) 내부 테스트 평가
    logger.info("")
    logger.info("내부 테스트 세트 평가 (LSTM)")
    test_ds = CCRTDataset(cbc_test, X_base_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
    test_proba, _ = lstm_trainer.predict(test_loader)
    test_metrics = compute_all_metrics(y_test, test_proba, model_name="lstm_twostage_internal")
    logger.info(
        f"  Internal AUROC={test_metrics['auroc']:.4f} | "
        f"Sens={test_metrics['sensitivity']:.4f} | Spec={test_metrics['specificity']:.4f}"
    )

    # 5) 외부 검증
    results = evaluate_external(
        tree_models, lstm_trainer,
        fe, scaler_cbc, scaler_base,
        baseline_cols, all_feature_cols,
        config,
        ext_patients=args.ext_patients, ext_seed=args.ext_seed,
    )

    # 6) 최종 요약
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
    logger.info("2단계 학습 완료!")


if __name__ == "__main__":
    main()
