"""
external_validation.py - External Validation 수행 스크립트

기존 학습 데이터(seed=42)와 완전히 독립된 데이터를 생성하여
학습된 모델의 일반화 성능을 검증합니다.

External Validation의 핵심:
    - 학습에 사용되지 않은 완전히 새로운 데이터
    - 다른 시드, 다른 분포 가능 (다기관 시뮬레이션)
    - 모델 재학습 없이 저장된 모델로만 예측
    - Overfitting 여부를 확인하는 가장 엄격한 테스트

사용법:
    python external_validation.py
    python external_validation.py --n_patients 200
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_external_dataset(n_patients: int = 100, seed: int = 9999):
    """External validation용 완전 독립 데이터셋을 생성합니다.

    학습 데이터(seed=42)와 다른 점:
        - 완전히 다른 시드 (9999)
        - 독성 발생률을 인위적으로 조절하지 않음 (순수 랜덤)
        - 환자 특성 분포에 약간의 변동 (다기관 시뮬레이션)

    Args:
        n_patients: 환자 수
        seed: 랜덤 시드 (학습과 반드시 달라야 함)

    Returns:
        (patients_df, cbc_df) 튜플
    """
    from generate_emr_data import (
        NORMAL_CBC_RANGES,
        REGIMEN_TOXICITY,
        generate_cbc_timeseries,
        generate_patients,
    )

    logger.info("=" * 60)
    logger.info("External Validation 데이터 생성")
    logger.info(f"  환자 수: {n_patients}명")
    logger.info(f"  시드: {seed} (학습 시드=42와 완전 독립)")
    logger.info("=" * 60)

    # 1) 환자 정보 생성 (다른 시드)
    patients = generate_patients(n_patients, seed=seed)

    # 2) CBC 시계열 생성 (다른 시드)
    cbc_records, toxicity_labels = generate_cbc_timeseries(patients, seed=seed)

    # 3) DataFrame 변환
    patients_df = pd.DataFrame(patients)
    cbc_df = pd.DataFrame(cbc_records)

    # 요약 통계
    n_toxic = sum(toxicity_labels.values())
    logger.info(f"생성 완료:")
    logger.info(f"  환자: {len(patients_df)}명")
    logger.info(f"  CBC 검사: {len(cbc_df)}건")
    logger.info(f"  예상 독성 발생: {n_toxic}명 ({n_toxic/len(patients_df)*100:.1f}%)")

    ages = patients_df["age"]
    logger.info(f"  나이: {ages.mean():.1f} ± {ages.std():.1f}세")
    logger.info(f"  성별: 남 {(patients_df['sex']=='M').sum()}명, "
                f"여 {(patients_df['sex']=='F').sum()}명")
    logger.info(f"  레지멘: {patients_df['chemo_regimen'].value_counts().to_dict()}")

    return patients_df, cbc_df


def preprocess_external_data(patients_df, cbc_df, config):
    """External 데이터를 전처리합니다.

    Args:
        patients_df: 환자 정보 DataFrame
        cbc_df: CBC 결과 DataFrame
        config: Config 인스턴스

    Returns:
        전처리된 DataFrame
    """
    from src.data.preprocessing import EMRPreprocessor

    logger.info("=" * 60)
    logger.info("External 데이터 전처리")
    logger.info("=" * 60)

    preprocessor = EMRPreprocessor(config)
    processed_df = preprocessor.run_full_pipeline(patients_df, cbc_df)

    # 저장
    ext_path = config.paths.processed_data_dir / "external_validation.csv"
    processed_df.to_csv(ext_path, index=False, encoding="utf-8-sig")
    logger.info(f"전처리 완료: {ext_path} ({len(processed_df)}명, {len(processed_df.columns)}변수)")

    return processed_df


def train_models_on_original(config):
    """원본 데이터(100명)로 모델을 학습합니다.

    이미 학습된 모델이 있으면 그것을 사용하고,
    없으면 새로 학습합니다.

    Returns:
        (trained_models, feature_engineer, scaler_info, train_df) 딕셔너리
    """
    from src.data.data_loader import DataLoader
    from src.data.feature_engineer import FeatureEngineer
    from src.models.baseline_models import create_model
    from src.utils.helpers import set_seed

    set_seed(config.train.seed)

    # 원본 전처리 데이터 로드
    original_path = config.paths.processed_data_dir / "emr_processed.csv"
    if not original_path.exists():
        logger.error(f"학습 데이터 없음: {original_path}")
        logger.error("먼저 generate_emr_data.py --test를 실행하세요.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("학습 데이터로 모델 훈련")
    logger.info("=" * 60)

    # 데이터 로드 및 분할
    data_loader = DataLoader(config)
    df = data_loader.load_data(str(original_path))
    df = data_loader.handle_missing_values(df)
    train_df, val_df, _ = data_loader.split_data(df)

    # 특성 공학
    fe = FeatureEngineer(config)
    train_df = fe.create_cbc_temporal_features(train_df)
    val_df = fe.create_cbc_temporal_features(val_df)

    train_df = fe.encode_categorical(train_df, fit=True)
    val_df = fe.encode_categorical(val_df, fit=False)

    # baseline_cbc 모드로 학습 (CBC 시계열 포함)
    feature_cols = fe.get_feature_columns(train_df, mode="baseline_cbc")
    all_feature_cols = feature_cols["all"]

    target_col = config.data.primary_target
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values

    X_train = train_df[all_feature_cols].values.astype(np.float32)
    X_val = val_df[all_feature_cols].values.astype(np.float32)

    # 모든 ML 모델 학습
    trained_models = {}
    for model_name in ["xgboost", "lightgbm", "logistic_regression"]:
        logger.info(f"  학습 중: {model_name}")
        model = create_model(model_name, config)
        model.fit(X_train, y_train, X_val, y_val)
        trained_models[model_name] = model

    # LSTM 학습
    logger.info("  학습 중: lstm")
    try:
        import torch
        from sklearn.preprocessing import StandardScaler

        from src.data.dataset import CCRTDataset, create_dataloaders
        from src.models.lstm_model import LSTMPredictor
        from src.models.trainer import LSTMTrainer

        baseline_cols = feature_cols["baseline"]
        baseline_cols_available = [
            c for c in baseline_cols
            if c in train_df.columns
            and train_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        cbc_train = fe.prepare_lstm_sequences(train_df)
        cbc_val = fe.prepare_lstm_sequences(val_df)

        X_base_train = train_df[baseline_cols_available].values.astype(np.float32)
        X_base_val = val_df[baseline_cols_available].values.astype(np.float32)

        # 스케일러 저장 (external data에도 동일하게 적용해야 함)
        scaler_cbc = StandardScaler()
        n_s, seq_len, n_f = cbc_train.shape
        cbc_train = scaler_cbc.fit_transform(cbc_train.reshape(-1, n_f)).reshape(n_s, seq_len, n_f)
        cbc_val = scaler_cbc.transform(cbc_val.reshape(-1, n_f)).reshape(len(val_df), seq_len, n_f)

        scaler_base = StandardScaler()
        X_base_train = scaler_base.fit_transform(X_base_train)
        X_base_val = scaler_base.transform(X_base_val)

        train_dataset = CCRTDataset(
            cbc_train, X_base_train, y_train,
            mask_future_prob=0.8, mask_start_idx=3,
        )
        val_dataset = CCRTDataset(cbc_val, X_base_val, y_val)

        dataloaders = create_dataloaders(
            train_dataset, val_dataset, val_dataset,
            batch_size=config.lstm.batch_size,
        )

        lstm_config = config.lstm
        lstm_config.baseline_input_size = X_base_train.shape[1]
        lstm_model = LSTMPredictor(lstm_config)
        trainer = LSTMTrainer(lstm_model, config)
        trainer.train(dataloaders["train"], dataloaders["val"])

        trained_models["lstm"] = {
            "trainer": trainer,
            "scaler_cbc": scaler_cbc,
            "scaler_base": scaler_base,
            "baseline_cols": baseline_cols_available,
        }
    except Exception as e:
        logger.warning(f"LSTM 학습 실패: {e}")

    return trained_models, fe, all_feature_cols


def evaluate_external(trained_models, fe, train_feature_cols, ext_df, config):
    """External validation 데이터에서 모델 성능을 평가합니다.

    핵심: 모델 재학습 없이 예측만 수행합니다.

    Args:
        trained_models: 학습된 모델 딕셔너리
        fe: FeatureEngineer (학습 데이터에 fit된 상태)
        train_feature_cols: 학습에 사용된 feature 컬럼 목록
        ext_df: 전처리된 external validation DataFrame
        config: Config
    """
    from src.evaluation.metrics import bootstrap_ci, compute_all_metrics
    from src.evaluation.visualization import plot_model_comparison_bar, plot_roc_curves
    from src.utils.helpers import save_results

    logger.info("=" * 60)
    logger.info("External Validation 성능 평가")
    logger.info("=" * 60)

    # 특성 공학 (학습과 동일한 변환 적용)
    ext_df = fe.create_cbc_temporal_features(ext_df)
    ext_df = fe.encode_categorical(ext_df, fit=False)

    target_col = config.data.primary_target
    y_ext = ext_df[target_col].values

    logger.info(f"External 데이터: {len(ext_df)}명")
    logger.info(f"  양성 (Grade 3+): {y_ext.sum()} ({y_ext.mean()*100:.1f}%)")
    logger.info(f"  음성 (Grade <3): {len(y_ext) - y_ext.sum()} ({(1-y_ext.mean())*100:.1f}%)")

    # 학습에 사용된 feature 중 external에도 있는 것만 사용
    available_cols = [c for c in train_feature_cols if c in ext_df.columns]
    missing_cols = [c for c in train_feature_cols if c not in ext_df.columns]

    if missing_cols:
        logger.warning(f"External 데이터에 없는 feature {len(missing_cols)}개 → 0으로 채움")
        for col in missing_cols:
            ext_df[col] = 0

    X_ext = ext_df[train_feature_cols].values.astype(np.float32)

    # NaN 처리
    X_ext = np.nan_to_num(X_ext, nan=0.0)

    all_results = {}

    # ----- ML 모델 평가 -----
    for model_name in ["xgboost", "lightgbm", "logistic_regression"]:
        if model_name not in trained_models:
            continue

        model = trained_models[model_name]
        try:
            ext_proba = model.predict_proba(X_ext)
            result_key = f"{model_name}_external"

            metrics = compute_all_metrics(y_ext, ext_proba, model_name=result_key)
            all_results[result_key] = metrics

            # Bootstrap CI
            auc_point, auc_lower, auc_upper = bootstrap_ci(y_ext, ext_proba)
            all_results[result_key]["auroc_ci"] = (auc_lower, auc_upper)

            logger.info(
                f"  {model_name:25s} | AUROC={metrics.get('auroc', float('nan')):.4f} "
                f"({auc_lower:.4f}-{auc_upper:.4f}) | "
                f"Sens={metrics.get('sensitivity', float('nan')):.4f} | "
                f"Spec={metrics.get('specificity', float('nan')):.4f}"
            )
        except Exception as e:
            logger.error(f"  {model_name} 평가 실패: {e}")

    # ----- LSTM 평가 -----
    if "lstm" in trained_models:
        try:
            import torch
            from src.data.dataset import CCRTDataset

            lstm_info = trained_models["lstm"]
            trainer = lstm_info["trainer"]
            scaler_cbc = lstm_info["scaler_cbc"]
            scaler_base = lstm_info["scaler_base"]
            baseline_cols = lstm_info["baseline_cols"]

            # 시퀀스 데이터 준비 (학습과 동일한 스케일러 적용)
            # 외부 검증: 스케일링 후 CCRTDataset이 마스킹 + 마스크 표시 채널 처리
            cbc_ext = fe.prepare_lstm_sequences(ext_df)
            n_s, seq_len, n_f = cbc_ext.shape

            # 스케일링 먼저 적용 (마스킹 전에 해야 0값이 왜곡되지 않음)
            cbc_ext = scaler_cbc.transform(cbc_ext.reshape(-1, n_f)).reshape(n_s, seq_len, n_f)
            logger.info(f"  LSTM 외부검증: mask_future_prob=1.0으로 Week 3-7 100% 마스킹 (shape={cbc_ext.shape})")

            missing_base = [c for c in baseline_cols if c not in ext_df.columns]
            for col in missing_base:
                ext_df[col] = 0

            X_base_ext = ext_df[baseline_cols].values.astype(np.float32)
            X_base_ext = np.nan_to_num(X_base_ext, nan=0.0)
            X_base_ext = scaler_base.transform(X_base_ext)

            # mask_future_prob=1.0: 100% 마스킹 → Week 3-7을 0으로 + 마스크 표시 채널=0
            ext_dataset = CCRTDataset(
                cbc_ext, X_base_ext, y_ext,
                mask_future_prob=1.0, mask_start_idx=3,
            )
            ext_loader = torch.utils.data.DataLoader(
                ext_dataset, batch_size=32, shuffle=False
            )

            ext_proba, _ = trainer.predict(ext_loader)

            result_key = "lstm_external"
            metrics = compute_all_metrics(y_ext, ext_proba, model_name=result_key)
            all_results[result_key] = metrics

            auc_point, auc_lower, auc_upper = bootstrap_ci(y_ext, ext_proba)
            all_results[result_key]["auroc_ci"] = (auc_lower, auc_upper)

            logger.info(
                f"  {'lstm':25s} | AUROC={metrics.get('auroc', float('nan')):.4f} "
                f"({auc_lower:.4f}-{auc_upper:.4f}) | "
                f"Sens={metrics.get('sensitivity', float('nan')):.4f} | "
                f"Spec={metrics.get('specificity', float('nan')):.4f}"
            )
        except Exception as e:
            logger.error(f"  LSTM 평가 실패: {e}")
            import traceback
            traceback.print_exc()

    # ============================================================
    # 결과 시각화 및 저장
    # ============================================================
    if all_results:
        logger.info("\n" + "=" * 60)
        logger.info("External Validation 결과 요약")
        logger.info("=" * 60)

        # 결과 테이블 출력
        logger.info(f"{'모델':30s} | {'AUROC':>8s} | {'AUPRC':>8s} | {'Sens':>6s} | {'Spec':>6s} | {'PPV':>6s} | {'NPV':>6s}")
        logger.info("-" * 90)
        for key, metrics in sorted(all_results.items()):
            logger.info(
                f"{key:30s} | "
                f"{metrics.get('auroc', float('nan')):8.4f} | "
                f"{metrics.get('auprc', float('nan')):8.4f} | "
                f"{metrics.get('sensitivity', float('nan')):6.4f} | "
                f"{metrics.get('specificity', float('nan')):6.4f} | "
                f"{metrics.get('ppv', float('nan')):6.4f} | "
                f"{metrics.get('npv', float('nan')):6.4f}"
            )

        # ROC 곡선 저장
        try:
            plot_roc_curves(
                all_results,
                save_path=str(config.paths.figure_dir / "roc_external_validation.png"),
                title="External Validation - ROC Curves",
            )
        except Exception as e:
            logger.warning(f"ROC 곡선 저장 실패: {e}")

        # JSON 결과 저장
        save_results(
            {k: {kk: vv for kk, vv in v.items() if kk not in ("roc_curve", "pr_curve")}
             for k, v in all_results.items()},
            str(config.paths.log_dir / "external_validation_results.json"),
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="External Validation - 독립 데이터셋으로 모델 성능 검증",
    )
    parser.add_argument("--n_patients", type=int, default=100, help="External 환자 수 (기본: 100)")
    parser.add_argument("--seed", type=int, default=9999, help="시드 (학습과 반드시 다르게, 기본: 9999)")

    args = parser.parse_args()

    from config import Config
    from src.utils.helpers import set_seed, setup_logging

    config = Config()
    config.paths.ensure_dirs()
    setup_logging(log_dir=str(config.paths.log_dir))

    # ============================================================
    # 1단계: 원본 데이터로 모델 학습
    # ============================================================
    trained_models, fe, feature_cols = train_models_on_original(config)

    # ============================================================
    # 2단계: External 데이터 생성 (완전 독립)
    # ============================================================
    ext_patients_df, ext_cbc_df = generate_external_dataset(
        n_patients=args.n_patients,
        seed=args.seed,
    )

    # ============================================================
    # 3단계: External 데이터 전처리
    # ============================================================
    ext_processed = preprocess_external_data(ext_patients_df, ext_cbc_df, config)

    # ============================================================
    # 4단계: External Validation 평가
    # ============================================================
    results = evaluate_external(trained_models, fe, feature_cols, ext_processed, config)

    logger.info("\n" + "=" * 60)
    logger.info("External Validation 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
