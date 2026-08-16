"""
CardioAI Command-Line Interface (CLI) Prediction Example.
"""

from __future__ import annotations
from pathlib import Path
import json

from src.model_utils import load_model, predict_single
from src.explainability import get_patient_key_factors
from src.config import MEDICAL_DISCLAIMER

MODEL_PATH = Path("models/heart_disease_model.joblib")

sample_input = {
    "age": 58,
    "sex": 1,
    "cp": 2,
    "trestbps": 132,
    "chol": 246,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 1,
    "ca": 0,
    "thal": 2,
}


def main():
    print("=" * 60)
    print("  CardioAI - CLI Risk Assessment Prediction")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print("Model file not found. Running training script first...")
        import train_model
        train_model.main()

    model = load_model(MODEL_PATH)

    print("\nSample Patient Input Features:")
    for k, v in sample_input.items():
        print(f"  - {k}: {v}")

    pred, prob, risk_category = predict_single(model, sample_input)

    print("\nInference Output:")
    print(f"  - Binary Prediction:    {pred} ({'Risk Detected' if pred == 1 else 'Lower Risk Detected'})")
    print(f"  - Model Probability:    {prob:.4f} ({prob * 100:.1f}%)")
    print(f"  - Model Risk Category:  {risk_category}")

    key_factors = get_patient_key_factors(model, sample_input, top_n=3)
    print("\nTop Contributing Model Factors:")
    for factor in key_factors:
        print(f"  {factor['rank']}. {factor['label']}: {factor['patient_value']} (Model Importance: {factor['model_importance']}%)")

    print("\n" + "-" * 60)
    print(f"DISCLAIMER: {MEDICAL_DISCLAIMER}")
    print("-" * 60)


if __name__ == "__main__":
    main()
