"""평가 CLI"""
import logging
import sys
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@click.command()
@click.option("--mode", type=click.Choice(["prediction", "screening"]), default="prediction")
@click.option("--model-path", required=True, help="학습된 모델 경로")
@click.option("--data", required=True, help="평가 데이터 경로")
@click.option("--shap", "run_shap", is_flag=True, help="SHAP 분석 실행")
@click.option("--find-threshold", "find_thr", is_flag=True, help="임상 임계값 탐색")
def evaluate(mode: str, model_path: str, data: str, run_shap: bool, find_thr: bool):
    """모델을 평가합니다."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from shared.infrastructure.repository.csv_repository import CSVRepository
    from shared.infrastructure.repository.model_repository import ModelRepository
    from prediction.domain import FeatureService
    from prediction.application.use_cases.evaluate_prediction import EvaluatePrediction

    csv_repo = CSVRepository()
    model_repo = ModelRepository()

    model_obj = model_repo.load(model_path)
    df = csv_repo.load(data)
    df = csv_repo.handle_missing(df)

    feature_service = FeatureService()
    df, feature_names = feature_service.extract_all(df, mode="baseline_cbc")

    target = "grade3_neutropenia"
    available = [f for f in feature_names if f in df.columns]
    X = df[available].values
    y = df[target].values

    # 내부 모델 객체에서 predict_proba 호출
    inner_model = model_obj.model if hasattr(model_obj, "model") else model_obj
    y_prob = model_obj.predict_proba(X)

    evaluator = EvaluatePrediction()
    metrics = evaluator.compute_all_metrics(y, y_prob)

    click.echo(f"AUROC: {metrics['auroc']:.4f}")
    click.echo(f"Sensitivity: {metrics['sensitivity']:.4f}")
    click.echo(f"Specificity: {metrics['specificity']:.4f}")

    if find_thr:
        from screening.application.use_cases.find_threshold import FindThreshold
        thr, sens, spec = FindThreshold.execute(y, y_prob)
        click.echo(f"임상 임계값: {thr:.3f} (Sens={sens:.3f}, Spec={spec:.3f})")

    if run_shap:
        from prediction.application.use_cases.run_shap import RunSHAP
        shap_analyzer = RunSHAP(inner_model, X, available)
        save_path = str(PROJECT_ROOT / "outputs" / "figures" / "shap_summary.png")
        shap_analyzer.plot_summary(save_path=save_path)
        click.echo(f"SHAP plot 저장: {save_path}")
        top = shap_analyzer.get_top_features(10)
        for name, val in top:
            click.echo(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    evaluate()
