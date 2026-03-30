from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class PredictionOutput :
    model : Any
    test_proba : np.ndarray
    test_auc : float
    cv_results : dict
    metrics : Optional[dict] = None