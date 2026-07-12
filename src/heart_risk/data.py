from pathlib import Path

import numpy as np
import pandas as pd

from heart_risk.config import FEATURES, TARGET


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the UCI processed Cleveland file and assign its published column names."""
    columns = [*FEATURES, "severity"]
    return pd.read_csv(path, names=columns, na_values="?")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate types, remove duplicate records, and create a binary clinical target."""
    required = {*FEATURES, "severity"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clean = df.copy()
    for column in required:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=["severity"]).drop_duplicates().reset_index(drop=True)
    if not clean["severity"].between(0, 4).all():
        raise ValueError("Severity must be between 0 and 4")
    clean[TARGET] = (clean.pop("severity") > 0).astype(int)
    clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    return clean[[*FEATURES, TARGET]]


def save_processed_data(df: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
