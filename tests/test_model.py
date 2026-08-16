import pytest
from pathlib import Path
import pandas as pd
from src.data_utils import load_dataset, split_xy
from src.model_utils import (
    build_pipeline,
    evaluate_model,
    train_and_save,
    compare_models,
    load_model,
    predict_single
)

DATA_PATH = Path("data/heart_sample.csv")


def test_build_pipeline():
    pipeline_rf = build_pipeline("random_forest")
    assert pipeline_rf is not None

    pipeline_lr = build_pipeline("logistic_regression")
    assert pipeline_lr is not None


def test_train_evaluate_and_predict(tmp_path):
    df = load_dataset(DATA_PATH)
    X, y = split_xy(df)

    pipeline, metrics, model_path = train_and_save(X, y, tmp_path, model_name="random_forest")

    assert model_path.exists()
    assert metrics["accuracy"] > 0.5
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0

    # Load model back and test prediction
    loaded_pipe = load_model(model_path)
    sample_input = {
        "age": 58, "sex": 1, "cp": 2, "trestbps": 132, "chol": 246,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.2,
        "slope": 1, "ca": 0, "thal": 2
    }
    pred, prob, risk_category = predict_single(loaded_pipe, sample_input)

    assert pred in (0, 1)
    assert 0.0 <= prob <= 1.0
    assert risk_category in ("Lower Risk", "Moderate Risk", "Higher Risk")


def test_compare_models():
    df = load_dataset(DATA_PATH)
    X, y = split_xy(df)
    df_res = compare_models(X, y)
    assert len(df_res) == 2
    assert "Model" in df_res.columns
    assert "Accuracy" in df_res.columns
