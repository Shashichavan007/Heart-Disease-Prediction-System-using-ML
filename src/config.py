"""
CardioAI Application Configuration & Domain Metadata.
"""

PRODUCT_NAME = "CardioAI"
PRODUCT_SUBTITLE = "Intelligent Cardiovascular Risk Assessment"
PRODUCT_TAGLINE = "AI-assisted cardiovascular risk assessment for educational & research analytics"

MEDICAL_DISCLAIMER = (
    "For educational and research purposes only. This tool does not provide "
    "medical diagnosis or replace professional medical advice."
)

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

TARGET_COLUMN = "target"

# Feature metadata mapping for human-readable labels, units, defaults, and descriptions
FEATURE_METADATA = {
    "age": {
        "label": "Age",
        "unit": "years",
        "category": "Patient Profile",
        "min": 18,
        "max": 120,
        "default": 58,
        "description": "Patient age in years.",
        "type": "numeric"
    },
    "sex": {
        "label": "Biological Sex",
        "unit": "",
        "category": "Patient Profile",
        "options": {0: "Female (0)", 1: "Male (1)"},
        "default": 1,
        "description": "Biological sex at birth.",
        "type": "categorical"
    },
    "trestbps": {
        "label": "Resting Blood Pressure",
        "unit": "mmHg",
        "category": "Cardiovascular Measurements",
        "min": 80,
        "max": 250,
        "default": 132,
        "description": "Resting blood pressure measured upon admission to hospital.",
        "type": "numeric"
    },
    "chol": {
        "label": "Serum Cholesterol",
        "unit": "mg/dL",
        "category": "Cardiovascular Measurements",
        "min": 100,
        "max": 600,
        "default": 246,
        "description": "Total serum cholesterol level in mg/dL.",
        "type": "numeric"
    },
    "thalach": {
        "label": "Maximum Heart Rate Achieved",
        "unit": "bpm",
        "category": "Cardiovascular Measurements",
        "min": 60,
        "max": 220,
        "default": 150,
        "description": "Maximum heart rate achieved during exercise stress testing.",
        "type": "numeric"
    },
    "cp": {
        "label": "Chest Pain Type",
        "unit": "",
        "category": "Clinical Indicators",
        "options": {
            0: "Typical Angina (0)",
            1: "Atypical Angina (1)",
            2: "Non-anginal Pain (2)",
            3: "Asymptomatic (3)"
        },
        "default": 2,
        "description": "Type of chest pain experienced by the patient.",
        "type": "categorical"
    },
    "fbs": {
        "label": "Fasting Blood Sugar > 120 mg/dL",
        "unit": "",
        "category": "Clinical Indicators",
        "options": {0: "False - ≤ 120 mg/dL (0)", 1: "True - > 120 mg/dL (1)"},
        "default": 0,
        "description": "Fasting blood sugar level relative to 120 mg/dL threshold.",
        "type": "categorical"
    },
    "restecg": {
        "label": "Resting Electrocardiographic Results",
        "unit": "",
        "category": "Clinical Indicators",
        "options": {
            0: "Normal (0)",
            1: "ST-T Wave Abnormality (1)",
            2: "Left Ventricular Hypertrophy (2)"
        },
        "default": 1,
        "description": "Resting ECG findings.",
        "type": "categorical"
    },
    "exang": {
        "label": "Exercise-Induced Angina",
        "unit": "",
        "category": "Clinical Indicators",
        "options": {0: "No (0)", 1: "Yes (1)"},
        "default": 0,
        "description": "Chest pain induced by exertion or treadmill test.",
        "type": "categorical"
    },
    "oldpeak": {
        "label": "ST Depression (oldpeak)",
        "unit": "mm",
        "category": "Clinical Indicators",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
        "default": 1.2,
        "description": "ST depression induced by exercise relative to rest.",
        "type": "numeric"
    },
    "slope": {
        "label": "Slope of Peak Exercise ST Segment",
        "unit": "",
        "category": "Clinical Indicators",
        "options": {
            0: "Upsloping (0)",
            1: "Flat (1)",
            2: "Downsloping (2)"
        },
        "default": 1,
        "description": "Slope of the peak exercise ST segment.",
        "type": "categorical"
    },
    "ca": {
        "label": "Major Vessels Colored by Fluoroscopy",
        "unit": "vessels",
        "category": "Clinical Indicators",
        "options": {
            0: "0 Vessels (0)",
            1: "1 Vessel (1)",
            2: "2 Vessels (2)",
            3: "3 Vessels (3)",
            4: "4 Vessels (4)"
        },
        "default": 0,
        "description": "Number of major blood vessels (0-4) colored by fluoroscopy.",
        "type": "categorical"
    },
    "thal": {
        "label": "Thalassemia Result",
        "unit": "",
        "category": "Clinical Indicators",
        "options": {
            0: "Unknown / Null (0)",
            1: "Normal (1)",
            2: "Fixed Defect (2)",
            3: "Reversible Defect (3)"
        },
        "default": 2,
        "description": "Thalassemia blood disorder diagnostic classification.",
        "type": "categorical"
    }
}

# Theme Color Palette Tokens
THEME = {
    "primary": "#0284c7",       # Medical Sky Blue
    "primary_dark": "#0369a1",  # Deep Sky Blue
    "secondary": "#0d9488",     # Teal
    "background_light": "#f8fafc",
    "background_dark": "#0f172a",
    "text_dark": "#1e293b",
    "text_muted": "#64748b",
    "card_bg": "#ffffff",
    "card_border": "#e2e8f0",
    "risk_low": "#10b981",      # Emerald Green
    "risk_moderate": "#f59e0b", # Amber
    "risk_high": "#ef4444",     # Coral Red
}
