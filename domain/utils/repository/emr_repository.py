import logging
import pandas as pd

logger = logging.getLogger(__name__)

class EMRRepository :
    CTCAE_NEUTROPENIA = {
        "grade1": (1.0, 1.5),
        "grade2": (0.5, 1.0),
        "grade3": (0.1, 0.5),
        "grade4": (0.0, 0.1),
    }

    def assign_treatment_week (self, df : pd.DataFrame, treatment_start_col : str = "treatment_start_date", exam_date_col : str = "exam_date") -> pd.DataFrame :
        df = df.copy()
        df[treatment_start_col] = pd.to_datetime(df[treatment_start_col])
        df[exam_date_col] = pd.to_datetime(df[exam_date_col])
        df["days_from_start"] = (df[exam_date_col] - df[treatment_start_col]).dt.days
        df["treatment_week"] = (df["days_from_start"] / 7).round().astype(int)
        return df

    def convert_long_to_wide (self, df : pd.DataFrame, patient_col : str = "patient_id", week_col : str = "treatment_week", value_cols : list = None) -> pd.DataFrame :
        if value_cols is None :
            value_cols = ["WBC", "ANC", "ALC", "AMC", "PLT", "Hb"]

        pivoted = df.pivot_table (
            index = patient_col, columns = week_col, values = value_cols, aggfunc = "mean",
        )

        pivoted.columns = [f"{col}_week{week}" for col, week in pivoted.columns]
        return pivoted.reset_index()

    def calculate_ctcae_grade (self, anc_value : float) -> int :
        if anc_value < 0.1 :
            return 4
        elif anc_value < 0.5 :
            return 3
        elif anc_value < 1.0 :
            return 2
        elif anc_value < 1.5 :
            return 1
        return 0