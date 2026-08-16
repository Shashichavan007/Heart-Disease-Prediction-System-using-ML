"""
Machine Learning pipeline, evaluation, comparison, and prediction utilities.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, FEATURE_METADATA

NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if FEATURE_METADATA[c]["type"] == "numeric"]
CATEGORICAL_FEATURES = [c for c in FEATURE_COLUMNS if FEATURE_METADATA[c]["type"] == "categorical"]


def build_pipeline(model_name: str = "random_forest") -> Pipeline:
    """
    Constructs a Scikit-Learn Pipeline with imputation, scaling, and estimator.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), FEATURE_COLUMNS),
        ],
        remainder="drop",
    )

    if model_name == "logistic_regression":
        model = LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            random_state=42
        )
    else:
        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluates a trained pipeline on test data and calculates comprehensive metrics.
    """
    preds = pipeline.predict(X_test)
    proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            proba = None

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    acc = float(accuracy_score(y_test, preds))

    cm = confusion_matrix(y_test, preds).tolist()
    class_report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "classification_report": class_report,
        "roc_auc": None,
    }

    if proba is not None and len(np.unique(y_test)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def train_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    model_dir: str | Path,
    model_name: str = "random_forest"
) -> Tuple[Pipeline, Dict[str, Any], Path]:
    """
    Splits dataset, trains pipeline, evaluates performance, exports model & metadata.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(model_name=model_name)
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "heart_disease_model.joblib"
    joblib.dump(pipeline, model_path)

    # Save model metadata
    metadata = {
        "model_name": model_name,
        "algorithm": "Random Forest Classifier" if model_name == "random_forest" else "Logistic Regression",
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features_count": len(FEATURE_COLUMNS),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_state": 42,
        "test_split_ratio": 0.2,
        "metrics": metrics,
    }

    metadata_path = model_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    metrics_path = model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Feature Importance Export (for Random Forest)
    if hasattr(pipeline.named_steps["model"], "feature_importances_"):
        importances = pipeline.named_steps["model"].feature_importances_
        fi_dict = {feat: float(imp) for feat, imp in zip(FEATURE_COLUMNS, importances)}
        # Sort descending
        fi_dict = dict(sorted(fi_dict.items(), key=lambda item: item[1], reverse=True))
        fi_path = model_dir / "feature_importance.json"
        fi_path.write_text(json.dumps(fi_dict, indent=2))

    return pipeline, metrics, model_path


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Trains and compares Random Forest vs. Logistic Regression on identical splits.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models_to_test = ["random_forest", "logistic_regression"]
    results = []

    for name in models_to_test:
        pipe = build_pipeline(model_name=name)
        pipe.fit(X_train, y_train)
        eval_metrics = evaluate_model(pipe, X_test, y_test)

        results.append({
            "Model": "Random Forest" if name == "random_forest" else "Logistic Regression",
            "Accuracy": eval_metrics["accuracy"],
            "Precision": eval_metrics["precision"],
            "Recall": eval_metrics["recall"],
            "F1 Score": eval_metrics["f1_score"],
            "ROC-AUC": eval_metrics["roc_auc"] if eval_metrics["roc_auc"] is not None else 0.0,
        })

    df_res = pd.DataFrame(results)
    return df_res


def load_model(model_path: str | Path) -> Pipeline:
    """
    Loads persisted Scikit-Learn pipeline from disk.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path.resolve()}")
    return joblib.load(path)


def predict_single(model: Pipeline, input_dict: Dict[str, Any]) -> Tuple[int, float, str]:
    """
    Performs inference on a single patient record input dictionary.

    Returns:
        prediction (int): 0 for lower risk, 1 for higher risk.
        probability (float): Model probability score between 0.0 and 1.0.
        risk_category (str): Human-friendly model risk classification label.
    """
    df_input = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
    pred = int(model.predict(df_input)[0])

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df_input)[0][1])
    else:
        prob = float(pred)

    # Categorize probability thresholds (Clearly labeled as model probability interpretation)
    if prob < 0.35:
        risk_category = "Lower Risk"
    elif prob <= 0.65:
        risk_category = "Moderate Risk"
    else:
        risk_category = "Higher Risk"

    return pred, prob, risk_category
