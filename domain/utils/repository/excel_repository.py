import logging
from pathlib import Path
from typing import Dict

import yaml
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MAPPINT_FILE = _PROJECT_ROOT / "interfaces" / "config" / "column_mapping.yaml"

def load_column_mapping() -> Dict[str, str] :
    with open(_MAPPINT_FILE, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    mapping: Dict[str, str] = {}
    mapping.update(raw.get("baseline", {}))

    for kr, en in raw.get("cbc", {}).items():
        for week in range(8):
            mapping[f"{kr}_{week}주"] = f"{en}_week{week}"
            mapping[f"{kr}_week{week}"] = f"{en}_week{week}"

    return mapping

KOREAN_TO_ENGLISH: Dict[str, str] = load_column_mapping()

class ExcelRepository:
    def load(self, filepath: str, skip_unit_rows: int = 2, korean_map: Dict[str, str] = None) -> pd.DataFrame:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

        raw_header = pd.read_excel(path, header=0, nrows=0)
        raw_columns = [str(c).strip() for c in raw_header.columns]

        col_map = {**KOREAN_TO_ENGLISH, **(korean_map or {})}
        is_korean = any(c in col_map for c in raw_columns)
        if is_korean:
            mapped_columns = [col_map.get(c, c) for c in raw_columns]
            logger.info(f"한글 컬럼 감지, 매핑 적용: {dict(zip(raw_columns, mapped_columns))}")
        else:
            mapped_columns = raw_columns
            logger.info("영문 컬럼 감지, 매핑 생략")

        df = pd.read_excel(path, header=0, skiprows=range(1, 1 + skip_unit_rows))
        df.columns = mapped_columns

        num_numeric = {"patient_id", "chemo_regimen"}
        numeric_cols = [c for c in df.columns if c not in num_numeric]
        for col in numeric_cols :
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "grade3_neutropenia" in df.columns :
            df["grade3_neutropenia"] = df["grade3_neutropenia"].fillna(0).astype(int)

        logger.info(f"데이터 로드 : {path} ({len(df)}행 x {len(df.columns)}열)")
        return df

