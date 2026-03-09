"""
main.py - 프로젝트 메인 실행 파일

폐암 CCRT 혈액독성 예측 파이프라인의 전체 흐름을 관리합니다.

실행 단계:
    1. 데이터 로드 및 전처리
    2. 특성 공학 (CBC 시계열 파생 변수 생성)
    3. 모델 학습 (LSTM, XGBoost, LightGBM, Logistic Regression)
    4. 성능 평가 및 비교
    5. 결과 시각화 및 저장

사용법:
    # 합성 데이터로 전체 파이프라인 테스트
    python main.py --mode demo

    # 실제 데이터로 학습
    python main.py --mode train --data patients.csv

    # 특정 모델만 학습
    python main.py --mode train --data patients.csv --models lstm xgboost
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# 프로젝트 모듈 임포트
from config import Config
from src.data.data_loader import DataLoader
from src.data.dataset import CCRTDataset, create_dataloaders
from src.data.feature_engineer import FeatureEngineer
from src.evaluation.metrics import (
    bootstrap_ci,
    compare_models,
    compute_all_metrics,
    compute_incremental_value,
)
from src.evaluation.visualization import (
    plot_cbc_timeseries,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison_bar,
    plot_roc_curves,
    plot_training_history,
)
from src.models.baseline_models import create_model
from src.models.trainer import CrossValidator, LSTMTrainer
from src.utils.helpers import (
    print_data_summary,
    save_results,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def run_pipeline(config: Config, data_path: str = None, models_to_train: list = None):
    """전체 학습 파이프라인을 실행합니다.

    Args:
        config: 프로젝트 설정
        data_path: 데이터 파일 경로 (None이면 합성 데이터 사용)
        models_to_train: 학습할 모델 목록 (None이면 config 설정 사용)
    """
    # ============================================================
    # 1단계: 초기 설정
    # ============================================================
    set_seed(config.train.seed)
    config.paths.ensure_dirs()

    if models_to_train:
        config.train.models_to_train = models_to_train

    # ============================================================
    # 2단계: 데이터 로드
    # ============================================================
    logger.info("=" * 60)
    logger.info("1단계: 데이터 로드")
    logger.info("=" * 60)

    data_loader = DataLoader(config)

    if data_path:
        # 실제 데이터 파일 로드
        df = data_loader.load_data(data_path)
    else:
        # 합성 데이터 생성 (개발/테스트용)
        logger.info("합성 데이터를 생성합니다 (demo 모드)")
        df = data_loader.generate_synthetic_data(n_patients=300)

    # 데이터 요약 출력
    print_data_summary(df, config.data.primary_target)

    # ============================================================
    # 3단계: 결측치 처리
    # ============================================================
    logger.info("=" * 60)
    logger.info("2단계: 결측치 처리")
    logger.info("=" * 60)

    missing_summary = data_loader.explore_missing(df)
    df = data_loader.handle_missing_values(df)

    # ============================================================
    # 4단계: 데이터 분할
    # ============================================================
    logger.info("=" * 60)
    logger.info("3단계: 데이터 분할 (학습/검증/테스트)")
    logger.info("=" * 60)

    train_df, val_df, test_df = data_loader.split_data(df)

    # ============================================================
    # 5단계: 특성 공학
    # ============================================================
    logger.info("=" * 60)
    logger.info("4단계: 특성 공학 (CBC 시계열 파생 변수 생성)")
    logger.info("=" * 60)

    fe = FeatureEngineer(config)

    # CBC 시계열 파생 변수 생성
    train_df = fe.create_cbc_temporal_features(train_df)
    val_df = fe.create_cbc_temporal_features(val_df)
    test_df = fe.create_cbc_temporal_features(test_df)

    # 범주형 변수 인코딩 (학습 세트에 fit, 나머지에 transform)
    train_df = fe.encode_categorical(train_df, fit=True)
    val_df = fe.encode_categorical(val_df, fit=False)
    test_df = fe.encode_categorical(test_df, fit=False)

    # ============================================================
    # 6단계: CBC 시계열 시각화 (핵심 가설 확인)
    # ============================================================
    logger.info("=" * 60)
    logger.info("5단계: CBC 시계열 탐색적 분석")
    logger.info("=" * 60)

    # AMC 시계열을 양성/음성 그룹별로 시각화
    for feature in ["AMC", "ANC", "WBC"]:
        try:
            plot_cbc_timeseries(
                df,
                feature=feature,
                target_col=config.data.primary_target,
                save_path=str(config.paths.figure_dir / f"cbc_timeseries_{feature}.png"),
            )
        except Exception as e:
            logger.warning(f"{feature} 시계열 시각화 실패: {e}")

    # ============================================================
    # 7단계: 모델 학습 및 평가 (두 가지 실험 모드)
    # ============================================================
    all_results = {}  # 모든 모델의 평가 결과 저장

    for exp_mode in config.train.experiment_modes:
        logger.info("=" * 60)
        logger.info(f"6단계: 모델 학습 [{exp_mode}]")
        logger.info("=" * 60)

        # 실험 모드에 따른 특성 선택
        feature_cols = fe.get_feature_columns(train_df, mode=exp_mode)
        all_feature_cols = feature_cols["all"]

        # 타겟 추출
        target_col = config.data.primary_target
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        # ----- 전통적 ML 모델 학습 (XGBoost, LightGBM, LogReg) -----
        for model_name in config.train.models_to_train:
            if model_name == "lstm":
                continue  # LSTM은 아래에서 별도 처리

            logger.info(f"\n--- {model_name} ({exp_mode}) ---")

            try:
                # Tabular 특성 준비 (시계열 flatten 포함)
                X_train = train_df[all_feature_cols].values.astype(np.float32)
                X_val = val_df[all_feature_cols].values.astype(np.float32)
                X_test = test_df[all_feature_cols].values.astype(np.float32)

                # 모델 생성 및 학습
                model = create_model(model_name, config)
                model.fit(X_train, y_train, X_val, y_val)

                # 테스트 세트 예측
                test_proba = model.predict_proba(X_test)

                # 평가 지표 계산
                result_key = f"{model_name}_{exp_mode}"
                all_results[result_key] = compute_all_metrics(
                    y_test, test_proba,
                    model_name=result_key,
                )

                # 95% 신뢰구간 계산
                auc_point, auc_lower, auc_upper = bootstrap_ci(y_test, test_proba)
                all_results[result_key]["auroc_ci"] = (auc_lower, auc_upper)

                # 특성 중요도 (해당하는 경우)
                if hasattr(model, "get_feature_importance"):
                    importances = model.get_feature_importance(all_feature_cols)
                    all_results[result_key]["feature_importances"] = importances

                    # 특성 중요도 시각화
                    plot_feature_importance(
                        importances,
                        top_n=min(20, len(importances)),
                        save_path=str(
                            config.paths.figure_dir
                            / f"feature_importance_{result_key}.png"
                        ),
                        title=f"Feature Importance - {result_key}",
                    )

                # 모델 저장
                model.save(config.paths.model_dir / f"{result_key}.pkl")

            except Exception as e:
                logger.error(f"{model_name} 학습 실패: {e}")
                import traceback
                traceback.print_exc()

        # ----- LSTM 모델 학습 -----
        if "lstm" in config.train.models_to_train and exp_mode == "baseline_cbc":
            logger.info(f"\n--- LSTM ({exp_mode}) ---")

            try:
                import torch
                from src.models.lstm_model import LSTMPredictor

                # LSTM용 시계열 데이터 준비
                cbc_train = fe.prepare_lstm_sequences(train_df)
                cbc_val = fe.prepare_lstm_sequences(val_df)
                cbc_test = fe.prepare_lstm_sequences(test_df)

                # Baseline 특성 준비 (시계열 제외)
                baseline_cols = feature_cols["baseline"]
                baseline_cols_available = [
                    c for c in baseline_cols
                    if c in train_df.columns
                    and train_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
                ]

                X_base_train = train_df[baseline_cols_available].values.astype(np.float32)
                X_base_val = val_df[baseline_cols_available].values.astype(np.float32)
                X_base_test = test_df[baseline_cols_available].values.astype(np.float32)

                # 스케일링 (LSTM은 스케일에 민감)
                from sklearn.preprocessing import StandardScaler
                scaler_cbc = StandardScaler()
                n_samples_train, seq_len, n_feat = cbc_train.shape
                cbc_train_flat = cbc_train.reshape(-1, n_feat)
                cbc_train_flat = scaler_cbc.fit_transform(cbc_train_flat)
                cbc_train = cbc_train_flat.reshape(n_samples_train, seq_len, n_feat)

                cbc_val_flat = cbc_val.reshape(-1, n_feat)
                cbc_val = scaler_cbc.transform(cbc_val_flat).reshape(
                    len(val_df), seq_len, n_feat
                )
                cbc_test_flat = cbc_test.reshape(-1, n_feat)
                cbc_test = scaler_cbc.transform(cbc_test_flat).reshape(
                    len(test_df), seq_len, n_feat
                )

                scaler_base = StandardScaler()
                X_base_train = scaler_base.fit_transform(X_base_train)
                X_base_val = scaler_base.transform(X_base_val)
                X_base_test = scaler_base.transform(X_base_test)

                # Dataset 및 DataLoader 생성
                # mask_future_prob=0.8: 학습 시 80% 확률로 Week 3-7 마스킹
                # → LSTM이 조기 데이터(Week 0-2)만으로도 예측하는 능력을 강하게 학습
                train_dataset = CCRTDataset(
                    cbc_train, X_base_train, y_train,
                    mask_future_prob=0.8, mask_start_idx=3,
                )
                val_dataset = CCRTDataset(cbc_val, X_base_val, y_val)
                test_dataset = CCRTDataset(cbc_test, X_base_test, y_test)

                dataloaders = create_dataloaders(
                    train_dataset, val_dataset, test_dataset,
                    batch_size=config.lstm.batch_size,
                )

                # LSTM 모델 초기화
                lstm_config = config.lstm
                lstm_config.baseline_input_size = X_base_train.shape[1]
                model = LSTMPredictor(lstm_config)

                # 학습
                trainer = LSTMTrainer(model, config)
                history = trainer.train(dataloaders["train"], dataloaders["val"])

                # 학습 곡선 시각화
                plot_training_history(
                    history,
                    save_path=str(config.paths.figure_dir / "lstm_training_history.png"),
                )

                # 테스트 세트 예측
                test_proba, _ = trainer.predict(dataloaders["test"])

                # 평가
                result_key = f"lstm_{exp_mode}"
                all_results[result_key] = compute_all_metrics(
                    y_test, test_proba,
                    model_name=result_key,
                )

                # 95% 신뢰구간
                auc_point, auc_lower, auc_upper = bootstrap_ci(y_test, test_proba)
                all_results[result_key]["auroc_ci"] = (auc_lower, auc_upper)

                # 체크포인트 저장
                trainer.save_checkpoint(
                    str(config.paths.model_dir / "lstm_best.pt")
                )

            except Exception as e:
                logger.error(f"LSTM 학습 실패: {e}")
                import traceback
                traceback.print_exc()

    # ============================================================
    # 8단계: 모델 비교 및 결과 저장
    # ============================================================
    logger.info("=" * 60)
    logger.info("7단계: 모델 비교 및 결과 저장")
    logger.info("=" * 60)

    if all_results:
        # 모델 성능 비교 테이블
        comparison = compare_models(all_results)

        # ROC 곡선 비교
        plot_roc_curves(
            all_results,
            save_path=str(config.paths.figure_dir / "roc_comparison.png"),
        )

        # 막대 그래프 비교
        plot_model_comparison_bar(
            comparison,
            save_path=str(config.paths.figure_dir / "model_comparison.png"),
        )

        # Incremental value 계산 (baseline_only vs baseline_cbc)
        for model_name in config.train.models_to_train:
            if model_name == "lstm":
                continue
            base_key = f"{model_name}_baseline_only"
            enhanced_key = f"{model_name}_baseline_cbc"
            if base_key in all_results and enhanced_key in all_results:
                compute_incremental_value(
                    all_results[base_key],
                    all_results[enhanced_key],
                )

        # 혼동 행렬 (최적 모델)
        best_model_key = max(
            all_results.keys(),
            key=lambda k: all_results[k].get("auroc", 0),
        )
        best_result = all_results[best_model_key]
        optimal_threshold = best_result.get("optimal_threshold", 0.5)

        logger.info(f"\n최적 모델: {best_model_key} (AUROC={best_result['auroc']:.4f})")

        # 결과 저장
        save_results(
            {k: {kk: vv for kk, vv in v.items() if kk not in ("roc_curve", "pr_curve")}
             for k, v in all_results.items()},
            str(config.paths.log_dir / "experiment_results.json"),
        )

        # 설정 저장
        config.save(str(config.paths.log_dir / "config.yaml"))

    logger.info("=" * 60)
    logger.info("파이프라인 완료!")
    logger.info(f"결과 저장 위치: {config.paths.log_dir}")
    logger.info(f"시각화 저장 위치: {config.paths.figure_dir}")
    logger.info("=" * 60)

    return all_results


def main():
    """CLI 인자를 파싱하고 파이프라인을 실행합니다."""
    parser = argparse.ArgumentParser(
        description="폐암 CCRT 혈액독성 예측 모델 학습 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py --mode demo                                           # 합성 데이터로 데모
  python main.py --mode train --data data.csv                          # 전처리 완료된 데이터로 학습
  python main.py --mode emr --patients patients.csv --cbc cbc.csv      # EMR 원본 데이터 전처리 후 학습
  python main.py --mode template                                       # EMR 추출용 템플릿 생성
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "train", "emr", "template"],
        help="실행 모드: demo(합성), train(전처리 완료 데이터), emr(EMR 원본), template(템플릿 생성)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="전처리 완료된 데이터 파일 경로 (train 모드용)",
    )
    parser.add_argument(
        "--patients",
        type=str,
        default=None,
        help="환자 기본정보 CSV 경로 (emr 모드용)",
    )
    parser.add_argument(
        "--cbc",
        type=str,
        default=None,
        help="CBC 검사 결과 CSV 경로 (emr 모드용, Long 형식/날짜 기반)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="설정 파일 경로 (YAML/JSON)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["lstm", "xgboost", "lightgbm", "logistic_regression"],
        help="학습할 모델 목록",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본: 42)",
    )

    args = parser.parse_args()

    # 설정 로드
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()

    # CLI 인자로 설정 오버라이드
    config.train.seed = args.seed

    # 로깅 설정
    setup_logging(log_dir=str(config.paths.log_dir))

    # 파이프라인 실행
    if args.mode == "demo":
        logger.info("데모 모드: 합성 데이터로 전체 파이프라인을 실행합니다")
        run_pipeline(config, data_path=None, models_to_train=args.models)

    elif args.mode == "train":
        if not args.data:
            parser.error("--mode train 사용 시 --data 인자가 필요합니다")
        run_pipeline(config, data_path=args.data, models_to_train=args.models)

    elif args.mode == "emr":
        # EMR 원본 데이터(날짜 기반 CBC) → 전처리 → 학습
        if not args.patients or not args.cbc:
            parser.error("--mode emr 사용 시 --patients와 --cbc 인자가 필요합니다")

        import pandas as pd
        from src.data.preprocessing import EMRPreprocessor

        logger.info("EMR 모드: 원본 데이터를 전처리 후 학습합니다")

        # EMR 전처리
        preprocessor = EMRPreprocessor(config)
        patients_df = pd.read_csv(args.patients)
        cbc_df = pd.read_csv(args.cbc)

        processed_df = preprocessor.run_full_pipeline(patients_df, cbc_df)

        # 전처리 결과 저장
        processed_path = config.paths.processed_data_dir / "processed_data.csv"
        processed_df.to_csv(processed_path, index=False, encoding="utf-8-sig")
        logger.info(f"전처리 데이터 저장: {processed_path}")

        # 학습 파이프라인 실행
        run_pipeline(config, data_path=str(processed_path), models_to_train=args.models)

    elif args.mode == "template":
        # EMR 추출용 데이터 템플릿 생성
        from src.data.preprocessing import EMRPreprocessor
        preprocessor = EMRPreprocessor(config)
        preprocessor.generate_data_template(str(config.paths.raw_data_dir))
        logger.info("EMR 추출용 템플릿이 data/raw/ 디렉토리에 생성되었습니다")
        logger.info("  - template_patients.csv : 환자 기본정보")
        logger.info("  - template_cbc_results.csv : CBC 검사 결과 (Long 형식)")
        logger.info("  - data_dictionary.csv : 데이터 사전 (변수 설명)")


if __name__ == "__main__":
    main()
