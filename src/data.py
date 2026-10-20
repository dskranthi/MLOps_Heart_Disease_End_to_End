"""Data loading and cleaning for the Heart Disease (Cleveland) dataset."""
from pathlib import Path

import pandas as pd

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
           "exang", "oldpeak", "slope", "ca", "thal", "target"]
FEATURES = COLUMNS[:-1]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
TARGET = "target"


def load_raw(path) -> pd.DataFrame:
    """Read the headerless UCI file; '?' marks missing values."""
    return pd.read_csv(Path(path), header=None, names=COLUMNS, na_values="?")


def binarize_target(series: pd.Series) -> pd.Series:
    """UCI target 0 = healthy, 1-4 = disease present -> 0/1."""
    return (series > 0).astype(int)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric, drop rows without target, binarize target, drop duplicates.

    Missing predictor values are kept: they are imputed inside the model pipeline
    so no information leaks from test folds.
    """
    out = df.copy()
    for col in COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[TARGET])
    out[TARGET] = binarize_target(out[TARGET])
    return out.drop_duplicates().reset_index(drop=True)
