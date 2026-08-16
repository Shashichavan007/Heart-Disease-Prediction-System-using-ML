from src.validation import validate_patient_input


def test_validate_patient_input_valid():
    valid_input = {
        "age": 58, "sex": 1, "cp": 2, "trestbps": 132, "chol": 246,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.2,
        "slope": 1, "ca": 0, "thal": 2
    }
    is_valid, errors, warnings = validate_patient_input(valid_input)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_patient_input_out_of_bounds():
    invalid_input = {
        "age": 150,  # exceeds max 120
        "sex": 1, "cp": 2,
        "trestbps": 300, # exceeds max 250
        "chol": 246, "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 2
    }
    is_valid, errors, warnings = validate_patient_input(invalid_input)
    assert is_valid is False
    assert len(errors) > 0


def test_validate_patient_input_missing_field():
    incomplete_input = {
        "age": 58, "sex": 1, "cp": 2
    }
    is_valid, errors, warnings = validate_patient_input(incomplete_input)
    assert is_valid is False
    assert any("Missing required" in err for err in errors)
