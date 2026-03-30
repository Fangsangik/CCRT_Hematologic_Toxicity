"""전처리 CLI"""
import logging
import sys
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@click.command()
@click.option("--data", required=True, help="입력 데이터 파일 경로 (CSV 또는 Excel)")
@click.option("--output", default=None, help="출력 파일 경로")
def preprocess(data: str, output: str):
    """데이터 전처리를 수행합니다."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from shared.infrastructure.repository.csv_repository import CSVRepository

    csv_repo = CSVRepository()
    df = csv_repo.load(data)
    df = csv_repo.handle_missing(df)

    output_path = output or str(PROJECT_ROOT / "data" / "processed" / "preprocessed.csv")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    click.echo(f"전처리 완료: {output_path} ({len(df)}행)")


if __name__ == "__main__":
    preprocess()
