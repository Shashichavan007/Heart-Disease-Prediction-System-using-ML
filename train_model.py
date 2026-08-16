"""
CardioAI Model Training & Evaluation Pipeline Script.
"""

from __future__ import annotations
import json
from pathlib import Path

from src.data_utils import load_dataset, split_xy
from src.model_utils import train_and_save, compare_models

DATA_PATH = Path("data/heart_sample.csv")
MODEL_DIR = Path("models")
REPORT_PATH = Path("models/metrics.json")


def main():
    print("=" * 60)
    print("  CardioAI Model Training & Evaluation Pipeline")
    print("=" * 60)

    print(f"Loading dataset from: {DATA_PATH.resolve()}")
    df = load_dataset(DATA_PATH)
    X, y = split_xy(df)
    print(f"Dataset successfully loaded. Matrix shape: {X.shape}, Target positive class ratio: {y.mean():.2%}")

    print("\nTraining Random Forest model pipeline...")
    pipeline, metrics, model_path = train_and_save(X, y, MODEL_DIR, model_name="random_forest")

    print(f"Model successfully saved to: {model_path}")
    print(f"Metrics saved to: {REPORT_PATH}")

    print("\nEvaluated Metrics (Test Set):")
    print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall:    {metrics['recall']:.4f}")
    print(f"  - F1 Score:  {metrics['f1_score']:.4f}")
    if metrics.get("roc_auc"):
        print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nRunning comparative analysis across algorithms...")
    df_comparison = compare_models(X, y)
    print("\nModel Performance Comparison Table:")
    print(df_comparison.to_string(index=False))

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
