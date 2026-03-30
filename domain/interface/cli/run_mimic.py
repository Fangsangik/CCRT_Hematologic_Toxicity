"""MIMIC-IV demo pipeline: cohort extraction → CBC timeseries → dataset preparation."""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MIMIC_DIR = PROJECT_ROOT / "data" / "mimic_demo" / "mimic-iv-clinical-database-demo-2.2" / "hosp"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

from datetime import datetime
log_file = LOG_DIR / f"mimic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

import pandas as pd

from mimic.application.use_cases.extract_cohort import ExtractCohort
from mimic.application.use_cases.build_timeseries import BuildTimeseries
from mimic.application.use_cases.prepare_dataset import PrepareDataset
from prediction.domain.feature_service import FeatureService


def main():
    logger.info("=" * 60)
    logger.info("MIMIC-IV Demo Pipeline")
    logger.info("=" * 60)

    # ── 1. Load MIMIC tables ─────────────────────────────────
    logger.info("1. Loading MIMIC-IV demo tables")
    patients = pd.read_csv(MIMIC_DIR / "patients.csv.gz")
    prescriptions = pd.read_csv(MIMIC_DIR / "prescriptions.csv.gz")
    labevents = pd.read_csv(MIMIC_DIR / "labevents.csv.gz")

    logger.info(f"  patients: {len(patients)} rows")
    logger.info(f"  prescriptions: {len(prescriptions)} rows")
    logger.info(f"  labevents: {len(labevents)} rows")

    # ── 2. Extract chemo cohort ──────────────────────────────
    logger.info("=" * 60)
    logger.info("2. Extracting chemo cohort")
    logger.info("=" * 60)

    extractor = ExtractCohort()
    cohort, chemo_start, cohort_stats = extractor.execute(prescriptions, patients)

    for k, v in cohort_stats.items():
        logger.info(f"  {k}: {v}")

    if len(cohort) == 0:
        logger.warning("No chemo patients found. Exiting.")
        return

    # ── 3. Build CBC timeseries ──────────────────────────────
    logger.info("=" * 60)
    logger.info("3. Building pseudo-weekly CBC timeseries")
    logger.info("=" * 60)

    builder = BuildTimeseries(input_weeks=[0, 1, 2], label_weeks=[3, 4, 5, 6])
    cbc_wide, ts_stats = builder.execute(labevents, chemo_start)

    for k, v in ts_stats.items():
        logger.info(f"  {k}: {v}")

    if len(cbc_wide) == 0:
        logger.warning("No CBC data found for chemo patients. Exiting.")
        return

    # ── 4. Prepare final dataset ─────────────────────────────
    logger.info("=" * 60)
    logger.info("4. Preparing final dataset")
    logger.info("=" * 60)

    preparer = PrepareDataset(
        input_weeks=[0, 1, 2],
        label_weeks=[3, 4, 5, 6],
        min_input_coverage=0.3,  # relaxed for demo
    )
    df, feature_names, prep_stats = preparer.execute(cohort, cbc_wide)

    for k, v in prep_stats.items():
        logger.info(f"  {k}: {v}")

    # ── 5. Feature engineering (delta, slope) ────────────────
    logger.info("=" * 60)
    logger.info("5. Feature engineering (shared FeatureService)")
    logger.info("=" * 60)

    feat_service = FeatureService()
    df_feat, cbc_feature_names = feat_service.extract_cbc_features(df)

    all_features = feature_names + cbc_feature_names
    available = [f for f in all_features if f in df_feat.columns]
    logger.info(f"  Total features: {len(available)} (baseline: {len(feature_names)}, derived: {len(cbc_feature_names)})")

    # ── 6. Save ──────────────────────────────────────────────
    output_path = OUTPUT_DIR / "mimic_demo.csv"
    df_feat.to_csv(output_path, index=False)
    logger.info(f"Dataset saved: {output_path} ({len(df_feat)} rows x {len(df_feat.columns)} cols)")

    # ── Summary ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    logger.info(f"  MIMIC patients: {len(patients)}")
    logger.info(f"  Chemo cohort: {cohort_stats['chemo_patients']}")
    logger.info(f"  CBC available: {ts_stats['patients_with_cbc']}")
    logger.info(f"  Final dataset: {len(df_feat)}")
    if "grade3_neutropenia" in df_feat.columns:
        pos = df_feat["grade3_neutropenia"].sum()
        logger.info(f"  Grade 3+ neutropenia: {pos}/{len(df_feat)} ({pos/len(df_feat):.1%})")
    logger.info(f"  Features: {len(available)}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Log: {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
