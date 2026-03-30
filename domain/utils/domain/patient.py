from dataclasses import dataclass


@dataclass
class Patient:
    """환자 정보 객체입니다."""
    patient_id: str
    age: int
    sex: str
    bmi: float
    ecog_ps: int
    stage: str
    t_stage: str
    n_stage: str
    creatinine: float
    albumin: float
    rt_total_dose: float
    chemo_regimen: bool
