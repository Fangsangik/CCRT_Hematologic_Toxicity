"""
config.py - 프로젝트 전역 설정 파일

폐암 CCRT 혈액독성 예측 모델의 모든 하이퍼파라미터와 경로를 중앙 관리합니다.
dataclass를 사용하여 타입 안전성과 IDE 자동완성을 지원합니다.
설정을 YAML/JSON으로 저장·로드할 수 있어 실험 재현성을 보장합니다.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import json
import yaml


# ============================================================
# 경로 설정
# ============================================================
@dataclass
class PathConfig:
    """프로젝트 내 디렉토리 및 파일 경로를 관리합니다."""

    # 프로젝트 루트 디렉토리 (이 파일 기준으로 자동 설정)
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)

    # 데이터 디렉토리
    raw_data_dir: Path = field(default=None)        # 원본 데이터 경로
    processed_data_dir: Path = field(default=None)  # 전처리된 데이터 경로

    # 출력 디렉토리
    model_dir: Path = field(default=None)       # 학습된 모델 저장 경로
    figure_dir: Path = field(default=None)      # 시각화 결과 저장 경로
    log_dir: Path = field(default=None)         # 로그 저장 경로

    def __post_init__(self):
        """기본 경로가 None이면 project_root 기준으로 자동 설정합니다."""
        if self.raw_data_dir is None:
            self.raw_data_dir = self.project_root / "data" / "raw"
        if self.processed_data_dir is None:
            self.processed_data_dir = self.project_root / "data" / "processed"
        if self.model_dir is None:
            self.model_dir = self.project_root / "outputs" / "models"
        if self.figure_dir is None:
            self.figure_dir = self.project_root / "outputs" / "figures"
        if self.log_dir is None:
            self.log_dir = self.project_root / "outputs" / "logs"

    def ensure_dirs(self):
        """모든 출력 디렉토리를 생성합니다."""
        for dir_path in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.model_dir,
            self.figure_dir,
            self.log_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================
# 데이터 설정
# ============================================================
@dataclass
class DataConfig:
    """데이터 관련 설정을 관리합니다."""

    # ----- CBC 시계열 변수 -----
    # 연구계획서 표3: WBC, ANC, ALC, AMC, PLT, Hb (Week 0, 1, 2)
    cbc_features: List[str] = field(default_factory=lambda: [
        "WBC",   # 백혈구 수 (10^3/uL)
        "ANC",   # 절대호중구수 (10^3/uL)
        "ALC",   # 절대림프구수 (10^3/uL)
        "AMC",   # 절대단핵구수 (10^3/uL) - 핵심 예측 변수
        "PLT",   # 혈소판 수 (10^3/uL)
        "Hb",    # 헤모글로빈 (g/dL)
    ])

    # CBC 전체 관찰 시점 (전처리 & Grade 계산용) - Week 0~7
    cbc_timepoints: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])

    # 모델 입력용 시점 (조기 예측 features) - Week 0~2만
    cbc_input_timepoints: List[int] = field(default_factory=lambda: [0, 1, 2])

    # ----- Baseline 임상 변수 -----
    # 연구계획서 표3: Age, Sex, BMI, PS, Stage, T, N, Cr, Albumin
    baseline_clinical_features: List[str] = field(default_factory=lambda: [
        "age",       # 나이 (연속형)
        "sex",       # 성별 (범주형: M/F)
        "bmi",       # 체질량지수 (연속형)
        "ecog_ps",   # ECOG Performance Status (순서형: 0-4)
        "stage",     # 병기 (범주형: IIIA, IIIB, IIIC 등)
        "t_stage",   # T 병기 (범주형)
        "n_stage",   # N 병기 (범주형)
        "creatinine",  # 크레아티닌 (연속형)
        "albumin",   # 알부민 (연속형)
    ])

    # ----- 치료 변수 -----
    # 연구계획서 표3: RT dose, Chemo regimen, Chemo dose/cycle
    treatment_features: List[str] = field(default_factory=lambda: [
        "rt_total_dose",    # 방사선 총 선량 (Gy)
        "rt_fraction_dose", # 분할 선량 (Gy)
        "chemo_regimen",    # 항암 레지멘 (범주형)
        "chemo_dose",       # 항암 용량 (연속형)
        "chemo_cycles",     # 항암 주기 수 (정수형)
    ])

    # ----- 범주형 변수 목록 (인코딩 대상) -----
    categorical_features: List[str] = field(default_factory=lambda: [
        "sex", "ecog_ps", "stage", "t_stage", "n_stage", "chemo_regimen",
    ])

    # ----- 타겟 변수 -----
    # CTCAE v5.0 기준 Grade 3 이상 혈액독성
    target_columns: List[str] = field(default_factory=lambda: [
        "grade3_neutropenia",       # Grade 3+ 호중구감소증
        "grade3_anemia",            # Grade 3+ 빈혈
        "grade3_thrombocytopenia",  # Grade 3+ 혈소판감소증
    ])

    # 주요 타겟 (단일 예측 시 사용)
    primary_target: str = "grade3_neutropenia"

    # ----- 데이터 분할 -----
    test_size: float = 0.2          # 테스트 세트 비율
    val_size: float = 0.15          # 검증 세트 비율 (학습 세트 내에서)
    random_state: int = 42          # 재현성을 위한 시드
    stratify: bool = True           # 층화 샘플링 여부 (클래스 불균형 대응)

    # ----- 결측치 처리 -----
    missing_strategy: str = "median"  # 결측치 대체 전략: "median", "mean", "knn", "mice"
    max_missing_rate: float = 0.3     # 변수별 최대 허용 결측률 (초과 시 변수 제거)


# ============================================================
# 모델 설정
# ============================================================
@dataclass
class LSTMConfig:
    """LSTM 모델 하이퍼파라미터를 관리합니다.

    시계열 패턴 학습을 위한 주요 모델입니다.
    CBC 시계열(Week 0-2)의 변화 패턴을 포착합니다.
    """

    # 네트워크 구조
    input_size: int = 7           # CBC 변수 6개 + 마스크 표시 채널 1개
    hidden_size: int = 64         # LSTM hidden state 차원
    num_layers: int = 2           # LSTM 레이어 수
    dropout: float = 0.3          # 드롭아웃 비율

    # Baseline 특성 결합을 위한 FC 레이어
    baseline_input_size: int = 14  # baseline + treatment 특성 수 (인코딩 후)
    fc_hidden_size: int = 32       # FC hidden 차원
    num_classes: int = 1           # 출력 클래스 수 (이진 분류)

    # 학습 설정
    learning_rate: float = 1e-3    # 학습률
    weight_decay: float = 1e-4     # L2 정규화
    batch_size: int = 32           # 배치 크기
    num_epochs: int = 100          # 최대 에폭 수
    early_stopping_patience: int = 15  # 조기 종료 인내심

    # 시계열 설정
    seq_length: int = 8            # 시계열 길이 (Week 0~7 = 학습용 전체 시계열)
    bidirectional: bool = False    # 양방향 LSTM 사용 여부


@dataclass
class XGBoostConfig:
    """XGBoost 하이퍼파라미터를 관리합니다.

    Baseline 비교 모델로 사용됩니다.
    시계열 데이터를 flattened features로 입력합니다.
    """

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    gamma: float = 0.1
    reg_alpha: float = 0.1       # L1 정규화
    reg_lambda: float = 1.0      # L2 정규화
    scale_pos_weight: float = 1.0  # 클래스 불균형 보정 (자동 계산 가능)
    random_state: int = 42
    use_gpu: bool = False         # GPU 사용 여부


@dataclass
class LightGBMConfig:
    """LightGBM 하이퍼파라미터를 관리합니다.

    XGBoost와 함께 비교 모델로 사용됩니다.
    """

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0
    random_state: int = 42


@dataclass
class LogisticRegressionConfig:
    """Logistic Regression 하이퍼파라미터를 관리합니다.

    가장 기본적인 비교 모델입니다.
    """

    C: float = 1.0                # 정규화 강도 (역수)
    penalty: str = "l2"           # 정규화 유형: "l1", "l2", "elasticnet"
    solver: str = "lbfgs"         # 최적화 알고리즘
    max_iter: int = 1000          # 최대 반복 횟수
    random_state: int = 42
    class_weight: Optional[str] = "balanced"  # 클래스 불균형 자동 보정


# ============================================================
# 학습 설정
# ============================================================
@dataclass
class TrainConfig:
    """학습 프로세스 전반의 설정을 관리합니다."""

    # 장치 설정
    device: str = "auto"           # "cpu", "cuda", "mps", "auto" (자동 감지)

    # 교차 검증
    n_folds: int = 5               # K-Fold 교차 검증 횟수
    use_stratified_kfold: bool = True  # 층화 K-Fold 사용 여부

    # 클래스 불균형 처리
    handle_imbalance: str = "smote"  # "smote", "class_weight", "oversampling", "none"

    # 모델 선택
    models_to_train: List[str] = field(default_factory=lambda: [
        "lstm",               # 주요 모델: LSTM (시계열)
        "xgboost",            # 비교 모델: XGBoost
        "lightgbm",           # 비교 모델: LightGBM
        "logistic_regression", # 비교 모델: Logistic Regression
    ])

    # 실험 모드 (Baseline-only vs Baseline+CBC 비교)
    experiment_modes: List[str] = field(default_factory=lambda: [
        "baseline_only",      # baseline 임상 + 치료 변수만 사용
        "baseline_cbc",       # baseline + CBC 시계열 결합
    ])

    # 재현성
    seed: int = 42


# ============================================================
# 전체 설정 통합
# ============================================================
@dataclass
class Config:
    """모든 설정을 하나로 통합하는 최상위 설정 클래스입니다.

    사용 예시:
        cfg = Config()
        cfg.paths.ensure_dirs()
        print(cfg.data.cbc_features)
        cfg.save("config.yaml")
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    logistic_regression: LogisticRegressionConfig = field(
        default_factory=LogisticRegressionConfig
    )
    train: TrainConfig = field(default_factory=TrainConfig)

    def save(self, filepath: str):
        """설정을 YAML 또는 JSON 파일로 저장합니다.

        Args:
            filepath: 저장할 파일 경로 (.yaml 또는 .json)
        """
        # Path 객체를 문자열로 변환하여 직렬화 가능하게 만듦
        config_dict = self._to_serializable(asdict(self))

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.suffix in (".yaml", ".yml"):
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        elif filepath.suffix == ".json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {filepath.suffix}")

    @classmethod
    def load(cls, filepath: str) -> "Config":
        """YAML 또는 JSON 파일에서 설정을 로드합니다.

        Args:
            filepath: 로드할 파일 경로

        Returns:
            로드된 Config 인스턴스
        """
        filepath = Path(filepath)
        if filepath.suffix in (".yaml", ".yml"):
            with open(filepath, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {filepath.suffix}")

        # 중첩 딕셔너리에서 Config 객체 재구성
        return cls(
            paths=PathConfig(**{
                k: Path(v) if k != "project_root" and isinstance(v, str) else
                (Path(v) if isinstance(v, str) else v)
                for k, v in config_dict.get("paths", {}).items()
            }),
            data=DataConfig(**config_dict.get("data", {})),
            lstm=LSTMConfig(**config_dict.get("lstm", {})),
            xgboost=XGBoostConfig(**config_dict.get("xgboost", {})),
            lightgbm=LightGBMConfig(**config_dict.get("lightgbm", {})),
            logistic_regression=LogisticRegressionConfig(
                **config_dict.get("logistic_regression", {})
            ),
            train=TrainConfig(**config_dict.get("train", {})),
        )

    @staticmethod
    def _to_serializable(obj):
        """중첩된 딕셔너리 내의 Path 객체를 문자열로 변환합니다."""
        if isinstance(obj, dict):
            return {k: Config._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Config._to_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj


# ============================================================
# CTCAE v5.0 혈액독성 등급 기준
# ============================================================
# 연구에서 사용하는 혈액독성 등급 정의 (참조용 상수)
CTCAE_THRESHOLDS = {
    "neutropenia": {
        # ANC 기준 (10^3/uL)
        "grade1": (1.5, 2.0),    # LLN - 1500/mm³
        "grade2": (1.0, 1.5),    # 1000 - 1500/mm³
        "grade3": (0.5, 1.0),    # 500 - 1000/mm³
        "grade4": (0.0, 0.5),    # < 500/mm³
    },
    "anemia": {
        # Hemoglobin 기준 (g/dL)
        "grade1": (10.0, 12.0),  # LLN - 10.0
        "grade2": (8.0, 10.0),   # 8.0 - 10.0
        "grade3": (6.5, 8.0),    # < 8.0 (수혈 필요)
        "grade4": (0.0, 6.5),    # 생명 위협
    },
    "thrombocytopenia": {
        # Platelet 기준 (10^3/uL)
        "grade1": (75, 150),     # LLN - 75,000
        "grade2": (50, 75),      # 50,000 - 75,000
        "grade3": (25, 50),      # 25,000 - 50,000
        "grade4": (0, 25),       # < 25,000
    },
}


if __name__ == "__main__":
    # 설정 테스트: 기본값 출력 및 저장
    cfg = Config()
    cfg.paths.ensure_dirs()
    print("=== 프로젝트 설정 ===")
    print(f"CBC 변수: {cfg.data.cbc_features}")
    print(f"Baseline 변수: {cfg.data.baseline_clinical_features}")
    print(f"타겟: {cfg.data.primary_target}")
    print(f"학습 모델: {cfg.train.models_to_train}")

    # 설정 파일 저장 예시
    cfg.save(str(cfg.paths.log_dir / "config.yaml"))
    print(f"\n설정 저장 완료: {cfg.paths.log_dir / 'config.yaml'}")
