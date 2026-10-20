import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import COLUMNS


@pytest.fixture(scope="session")
def synthetic_raw():
    """Small synthetic frame shaped like the raw UCI file (incl. NaNs and 0-4 target)."""
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(95, 200, n),
        "chol": rng.integers(125, 560, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n), "target": rng.integers(0, 5, n),
    })[COLUMNS]
    df.loc[3, "ca"] = np.nan
    df.loc[7, "thal"] = np.nan
    return df


@pytest.fixture(scope="session")
def synthetic_clean(synthetic_raw):
    from src.data import clean
    return clean(synthetic_raw)
