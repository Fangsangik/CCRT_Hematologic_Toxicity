from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class PredictionInput :
    x_train : np.ndarray
    y_train : np.ndarray
    x_test : np.ndarray
    y_test : np.ndarray
    feature_names : Optional[List[str]] = None
    x_val : Optional[np.ndarray] = None
    y_val : Optional[np.ndarray] = None