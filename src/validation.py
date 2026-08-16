"""
Validation utilities for patient input data and dataset integrity.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
from .config import FEATURE_METADATA, FEATURE_COLUMNS


def validate_patient_input(input_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Validates a dictionary of patient input features.

    Returns:
        is_valid (bool): True if all inputs pass critical validation.
        errors (List[str]): List of critical error messages blocking analysis.
        warnings (List[str]): List of physiological warning messages.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Check for missing required features
    missing = [feat for feat in FEATURE_COLUMNS if feat not in input_data]
    if missing:
        errors.append(f"Missing required input features: {', '.join(missing)}")

    for feat in FEATURE_COLUMNS:
        if feat not in input_data:
            continue

        val = input_data[feat]
        meta = FEATURE_METADATA.get(feat, {})

        if meta.get("type") == "numeric":
            try:
                num_val = float(val)
            except (ValueError, TypeError):
                errors.append(f"Invalid numeric value '{val}' for {meta.get('label', feat)}.")
                continue

            min_val = meta.get("min")
            max_val = meta.get("max")

            if min_val is not None and num_val < min_val:
                errors.append(f"{meta.get('label', feat)} cannot be below {min_val} {meta.get('unit', '')}.")

            if max_val is not None and num_val > max_val:
                errors.append(f"{meta.get('label', feat)} cannot exceed {max_val} {meta.get('unit', '')}.")

        elif meta.get("type") == "categorical":
            options = meta.get("options", {})
            try:
                cat_val = int(val)
                if options and cat_val not in options:
                    errors.append(f"Invalid selection code '{cat_val}' for {meta.get('label', feat)}.")
            except (ValueError, TypeError):
                errors.append(f"Invalid categorical value '{val}' for {meta.get('label', feat)}.")

    # Physiological plausibility checks & warnings
    if "age" in input_data and "thalach" in input_data:
        try:
            age = float(input_data["age"])
            thalach = float(input_data["thalach"])
            estimated_max_hr = 220.0 - age
            if thalach > estimated_max_hr + 25:
                warnings.append(
                    f"Maximum heart rate ({thalach:.0f} bpm) is unusually high for age {age:.0f} "
                    f"(estimated max: {estimated_max_hr:.0f} bpm)."
                )
        except (ValueError, TypeError):
            pass

    if "trestbps" in input_data:
        try:
            bp = float(input_data["trestbps"])
            if bp >= 180:
                warnings.append("Resting blood pressure ≥ 180 mmHg indicates severe hypertension (hypertensive crisis stage).")
            elif bp < 90:
                warnings.append("Resting blood pressure < 90 mmHg indicates hypotension.")
        except (ValueError, TypeError):
            pass

    if "chol" in input_data:
        try:
            chol = float(input_data["chol"])
            if chol >= 300:
                warnings.append("Serum cholesterol ≥ 300 mg/dL represents severe hypercholesterolemia.")
        except (ValueError, TypeError):
            pass

    is_valid = len(errors) == 0
    return is_valid, errors, warnings
