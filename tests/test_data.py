import pytest
from pathlib import Path
import pandas as pd
from src.data_utils import load_dataset, split_xy, get_dataset_summary, get_descriptive_dataframe
from src.config import FEATURE_COLUMNS, TARGET_COLUMN

DATA_PATH = Path("data/heart_sample.csv")


def test_load_dataset_success():
    df = load_dataset(DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in df.columns


def test_load_dataset_missing_file():
    with pytest.raises(FileNotFoundError):
        load_dataset("data/non_existent_file.csv")


def test_split_xy():
    df = load_dataset(DATA_PATH)
    X, y = split_xy(df)
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert len(y) == len(df)
    assert set(y.unique()).issubset({0, 1})


def test_get_dataset_summary():
    df = load_dataset(DATA_PATH)
    summary = get_dataset_summary(df)
    assert summary["total_records"] == len(df)
    assert summary["features_count"] == 13
    assert "target_distribution" in summary


def test_get_descriptive_dataframe():
    df = load_dataset(DATA_PATH)
    df_desc = get_descriptive_dataframe(df)
    assert "Age" in df_desc.columns
    assert "Serum Cholesterol" in df_desc.columns
