"""
Dataset utilities for CardioAI.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Any
import pandas as pd
from .config import FEATURE_COLUMNS, TARGET_COLUMN, FEATURE_METADATA


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Loads and validates the heart disease CSV dataset.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    df = pd.read_csv(path)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    return df


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits DataFrame into feature matrix X and target series y.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    return X, y


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes statistical and distributional metrics for the dataset.
    """
    X, y = split_xy(df)
    total_records = len(df)
    target_counts = y.value_counts().to_dict()
    pos_count = int(target_counts.get(1, 0))
    neg_count = int(target_counts.get(0, 0))

    numeric_cols = [c for c in FEATURE_COLUMNS if FEATURE_METADATA[c]["type"] == "numeric"]
    stats = X[numeric_cols].describe().to_dict()

    missing_counts = df.isnull().sum().to_dict()

    return {
        "total_records": total_records,
        "features_count": len(FEATURE_COLUMNS),
        "target_distribution": {
            "no_disease_count": neg_count,
            "disease_count": pos_count,
            "no_disease_pct": round((neg_count / total_records) * 100, 1) if total_records else 0,
            "disease_pct": round((pos_count / total_records) * 100, 1) if total_records else 0,
        },
        "feature_stats": stats,
        "missing_values": missing_counts,
    }


def get_descriptive_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of the dataset with human-readable column titles.
    """
    rename_dict = {feat: FEATURE_METADATA[feat]["label"] for feat in FEATURE_COLUMNS if feat in df.columns}
    if TARGET_COLUMN in df.columns:
        rename_dict[TARGET_COLUMN] = "Cardiovascular Risk Target"
    return df.rename(columns=rename_dict)
