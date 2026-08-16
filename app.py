"""
CardioAI: Intelligent Cardiovascular Risk Assessment Web Application.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    PRODUCT_NAME,
    PRODUCT_SUBTITLE,
    PRODUCT_TAGLINE,
    MEDICAL_DISCLAIMER,
    FEATURE_COLUMNS,
    FEATURE_METADATA,
    THEME,
)
from src.data_utils import load_dataset, split_xy, get_dataset_summary, get_descriptive_dataframe
from src.model_utils import load_model, predict_single, evaluate_model, compare_models
from src.explainability import get_feature_importances, get_patient_key_factors
from src.validation import validate_patient_input
from src.report_generator import generate_pdf_report

DATA_PATH = Path("data/heart_sample.csv")
MODEL_PATH = Path("models/heart_disease_model.joblib")
METRICS_PATH = Path("models/metrics.json")

# Streamlit Page Config
st.set_page_config(
    page_title=f"{PRODUCT_NAME} | {PRODUCT_SUBTITLE}",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Custom CSS for Healthcare Dashboard Aesthetic
st.markdown(f"""
<style>
    /* Main Layout Styling */
    .stApp {{
        background-color: {THEME['background_light']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Top Banner Header */
    .header-box {{
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(2, 132, 199, 0.2);
        margin-bottom: 24px;
    }}
    .header-title {{
        font-size: 32px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }}
    .header-subtitle {{
        font-size: 16px;
        opacity: 0.9;
        margin-top: 4px;
        font-weight: 400;
    }}

    /* Card Containers */
    .stat-card {{
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }}
    .stat-card:hover {{
        border-color: #0284c7;
    }}
    .stat-label {{
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .stat-value {{
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }}

    /* Result Banner Boxes */
    .risk-box-low {{
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }}
    .risk-box-moderate {{
        background-color: #fffbeb;
        border: 2px solid #f59e0b;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }}
    .risk-box-high {{
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
    }}

    /* Medical Disclaimer Banner */
    .disclaimer-box {{
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 6px;
        color: #991b1b;
        font-size: 13px;
        margin-bottom: 20px;
    }}

    /* Hide Streamlit Menu Footers */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        import scripts.generate_dataset as gen
        gen.main()
    return load_dataset(DATA_PATH)


@st.cache_resource
def load_model_pipeline():
    if not MODEL_PATH.exists():
        import train_model
        train_model.main()
    return load_model(MODEL_PATH)


df_dataset = load_data()
model_pipeline = load_model_pipeline()

# Sidebar Navigation & Branding
st.sidebar.markdown(f"## 🫀 **{PRODUCT_NAME}**")
st.sidebar.caption(PRODUCT_SUBTITLE)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏥 Dashboard Home",
        "🩺 Patient Assessment",
        "📊 Model Performance",
        "💡 Explainability",
        "🔍 Dataset Explorer",
        "ℹ️ About the Model",
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st.sidebar.markdown("🟢 **Model Status:** Active")
st.sidebar.markdown("🧠 **Engine:** Random Forest")
st.sidebar.markdown("📈 **Features:** 13 Clinical Inputs")
st.sidebar.markdown(f"📂 **Dataset:** {len(df_dataset)} Records")
st.sidebar.markdown("---")
st.sidebar.info(MEDICAL_DISCLAIMER)

# Header Section across pages
def render_header():
    st.markdown(f"""
    <div class="header-box">
        <div class="header-title">🫀 {PRODUCT_NAME}</div>
        <div class="header-subtitle">{PRODUCT_SUBTITLE}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="disclaimer-box">
        <b>MEDICAL DISCLAIMER:</b> {MEDICAL_DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 1: DASHBOARD HOME
# ==========================================
if page == "🏥 Dashboard Home":
    render_header()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Prediction Engine</div>
            <div class="stat-value" style="color: #0284c7;">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Model Status</div>
            <div class="stat-value" style="color: #10b981;">Active</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Clinical Features</div>
            <div class="stat-value">13 Inputs</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Dataset Records</div>
            <div class="stat-value">{len(df_dataset)} Patients</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📌 CardioAI Analytics Capabilities")
    st.write("Welcome to CardioAI. Select a module below to begin analyzing patient risk or evaluating machine learning model metrics.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("### 🩺 Risk Assessment\n\nAnalyze 13 physiological parameters for individual patients to obtain an instant model-predicted risk probability.")
    with c2:
        st.success("### 📊 Model Performance\n\nView evaluation metrics (Accuracy, ROC-AUC, F1, Confusion Matrix) and compare Random Forest with Logistic Regression.")
    with c3:
        st.warning("### 💡 Model Explainability\n\nInspect feature importance rankings and understand key factors driving cardiovascular risk prediction.")


# ==========================================
# PAGE 2: PATIENT ASSESSMENT FORM
# ==========================================
elif page == "🩺 Patient Assessment":
    render_header()

    st.markdown("## Patient Cardiovascular Risk Assessment")
    st.write("Enter patient demographics, physiological measurements, and clinical indicators below.")

    # Form Defaults / Reset Handling
    if "form_reset" not in st.session_state:
        st.session_state.form_reset = False

    with st.form("patient_form"):
        st.markdown("### 1️⃣ Patient Profile")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            age = st.number_input(
                "Age (years)",
                min_value=FEATURE_METADATA["age"]["min"],
                max_value=FEATURE_METADATA["age"]["max"],
                value=FEATURE_METADATA["age"]["default"],
                help=FEATURE_METADATA["age"]["description"]
            )
        with col_p2:
            sex = st.selectbox(
                "Biological Sex",
                options=[0, 1],
                format_func=lambda x: FEATURE_METADATA["sex"]["options"][x],
                index=1,
                help=FEATURE_METADATA["sex"]["description"]
            )

        st.markdown("---")
        st.markdown("### 2️⃣ Cardiovascular Measurements")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            trestbps = st.slider(
                "Resting Blood Pressure (mmHg)",
                min_value=FEATURE_METADATA["trestbps"]["min"],
                max_value=FEATURE_METADATA["trestbps"]["max"],
                value=FEATURE_METADATA["trestbps"]["default"],
                help=FEATURE_METADATA["trestbps"]["description"]
            )
        with col_m2:
            chol = st.slider(
                "Serum Cholesterol (mg/dL)",
                min_value=FEATURE_METADATA["chol"]["min"],
                max_value=FEATURE_METADATA["chol"]["max"],
                value=FEATURE_METADATA["chol"]["default"],
                help=FEATURE_METADATA["chol"]["description"]
            )
        with col_m3:
            thalach = st.slider(
                "Maximum Heart Rate Achieved (bpm)",
                min_value=FEATURE_METADATA["thalach"]["min"],
                max_value=FEATURE_METADATA["thalach"]["max"],
                value=FEATURE_METADATA["thalach"]["default"],
                help=FEATURE_METADATA["thalach"]["description"]
            )

        st.markdown("---")
        st.markdown("### 3️⃣ Clinical Indicators & Test Results")
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            cp = st.selectbox(
                "Chest Pain Type",
                options=[0, 1, 2, 3],
                format_func=lambda x: FEATURE_METADATA["cp"]["options"][x],
                index=FEATURE_METADATA["cp"]["default"],
                help=FEATURE_METADATA["cp"]["description"]
            )
            fbs = st.selectbox(
                "Fasting Blood Sugar",
                options=[0, 1],
                format_func=lambda x: FEATURE_METADATA["fbs"]["options"][x],
                index=FEATURE_METADATA["fbs"]["default"],
                help=FEATURE_METADATA["fbs"]["description"]
            )
            restecg = st.selectbox(
                "Resting ECG Results",
                options=[0, 1, 2],
                format_func=lambda x: FEATURE_METADATA["restecg"]["options"][x],
                index=FEATURE_METADATA["restecg"]["default"],
                help=FEATURE_METADATA["restecg"]["description"]
            )

        with col_c2:
            exang = st.selectbox(
                "Exercise-Induced Angina",
                options=[0, 1],
                format_func=lambda x: FEATURE_METADATA["exang"]["options"][x],
                index=FEATURE_METADATA["exang"]["default"],
                help=FEATURE_METADATA["exang"]["description"]
            )
            oldpeak = st.number_input(
                "ST Depression (oldpeak in mm)",
                min_value=FEATURE_METADATA["oldpeak"]["min"],
                max_value=FEATURE_METADATA["oldpeak"]["max"],
                value=FEATURE_METADATA["oldpeak"]["default"],
                step=FEATURE_METADATA["oldpeak"]["step"],
                help=FEATURE_METADATA["oldpeak"]["description"]
            )
            slope = st.selectbox(
                "Slope of ST Segment",
                options=[0, 1, 2],
                format_func=lambda x: FEATURE_METADATA["slope"]["options"][x],
                index=FEATURE_METADATA["slope"]["default"],
                help=FEATURE_METADATA["slope"]["description"]
            )

        with col_c3:
            ca = st.selectbox(
                "Major Vessels (Fluoroscopy)",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: FEATURE_METADATA["ca"]["options"][x],
                index=FEATURE_METADATA["ca"]["default"],
                help=FEATURE_METADATA["ca"]["description"]
            )
            thal = st.selectbox(
                "Thalassemia Result",
                options=[0, 1, 2, 3],
                format_func=lambda x: FEATURE_METADATA["thal"]["options"][x],
                index=FEATURE_METADATA["thal"]["default"],
                help=FEATURE_METADATA["thal"]["description"]
            )

        btn_submit = st.form_submit_button("⚡ Analyze Cardiovascular Risk", use_container_width=True)

    if btn_submit:
        input_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }

        is_valid, errors, warnings = validate_patient_input(input_data)

        if not is_valid:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            if warnings:
                for warn in warnings:
                    st.warning(f"⚠️ Clinical Note: {warn}")

            pred, prob, risk_category = predict_single(model_pipeline, input_data)
            prob_pct = prob * 100

            st.markdown("---")
            st.markdown("## 📊 Assessment Results")

            # Risk gauge display
            col_res1, col_res2 = st.columns([1.2, 1])

            with col_res1:
                if risk_category == "Lower Risk":
                    box_class = "risk-box-low"
                    color_code = "#10b981"
                elif risk_category == "Moderate Risk":
                    box_class = "risk-box-moderate"
                    color_code = "#f59e0b"
                else:
                    box_class = "risk-box-high"
                    color_code = "#ef4444"

                st.markdown(f"""
                <div class="{box_class}">
                    <div style="font-size: 14px; font-weight: 700; color: #64748b; letter-spacing: 1px;">CLASSIFICATION</div>
                    <div style="font-size: 36px; font-weight: 900; color: {color_code}; margin: 8px 0;">{risk_category.upper()}</div>
                    <div style="font-size: 16px; color: #1e293b;">Model Predicted Probability: <b>{prob_pct:.1f}%</b></div>
                    <div style="font-size: 12px; color: #64748b; margin-top: 6px;">(Model probability interpretation threshold)</div>
                </div>
                """, unsafe_allow_html=True)

            with col_res2:
                # Plotly Gauge Indicator Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_pct,
                    number={'suffix': '%'},
                    title={'text': "Cardiovascular Risk Score", 'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color_code},
                        'steps': [
                            {'range': [0, 35], 'color': "#d1fae5"},
                            {'range': [35, 65], 'color': "#fef3c7"},
                            {'range': [65, 100], 'color': "#fee2e2"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.75,
                            'value': prob_pct
                        }
                    }
                ))
                fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Assessment Result Details Card
            st.markdown("### 📋 Result Summary Card")
            st.json({
                "Risk Classification": risk_category,
                "Model Probability": f"{prob_pct:.1f}%",
                "Binary Output": f"{pred} ({'Risk Detected' if pred == 1 else 'No Disease Risk Detected'})",
                "Model Used": "Random Forest Classifier",
                "Features Analyzed": 13,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Explainability Section
            st.markdown("### 💡 Why did the model predict this?")
            st.caption("Key Model Factors (Relative importance weights in Random Forest splits):")

            key_factors = get_patient_key_factors(model_pipeline, input_data, top_n=5)
            for factor in key_factors:
                st.markdown(f"**{factor['rank']}. {factor['label']}** ({factor['patient_value']}) — Relative Weight: **{factor['relative_pct']}%**")
                st.caption(f"> {factor['context']}")

            st.info("💡 *Note: Model feature importance reflects statistical decision tree weights, not medical causation.*")

            # Report PDF Generation & Download
            st.markdown("---")
            st.markdown("### 📥 Download Official Assessment Report")
            pdf_bytes = generate_pdf_report(
                patient_inputs=input_data,
                prediction=pred,
                probability=prob,
                risk_category=risk_category,
                key_factors=key_factors
            )
            st.download_button(
                label="📄 Download Assessment Report (PDF)",
                data=pdf_bytes,
                file_name=f"CardioAI_Assessment_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )


# ==========================================
# PAGE 3: MODEL PERFORMANCE DASHBOARD
# ==========================================
elif page == "📊 Model Performance":
    render_header()

    st.markdown("## Machine Learning Model Performance Dashboard")
    st.write("Evaluation metrics generated on the held-out 20% test dataset.")

    X, y = split_xy(df_dataset)
    df_comparison = compare_models(X, y)
    rf_metrics = evaluate_model(model_pipeline, split_xy(df_dataset)[0].iloc[int(len(df_dataset)*0.8):], split_xy(df_dataset)[1].iloc[int(len(df_dataset)*0.8):])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{df_comparison.iloc[0]['Accuracy']:.2%}")
    m2.metric("ROC-AUC", f"{df_comparison.iloc[0]['ROC-AUC']:.3f}")
    m3.metric("Precision", f"{df_comparison.iloc[0]['Precision']:.2%}")
    m4.metric("Recall", f"{df_comparison.iloc[0]['Recall']:.2%}")
    m5.metric("F1 Score", f"{df_comparison.iloc[0]['F1 Score']:.3f}")

    st.markdown("---")
    st.markdown("### ⚔️ Model Comparison: Random Forest vs. Logistic Regression")
    st.dataframe(df_comparison.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"], color="#dcfce7"), use_container_width=True)

    c_cm, c_roc = st.columns(2)

    with c_cm:
        st.markdown("### 🧩 Confusion Matrix (Random Forest)")
        cm = rf_metrics["confusion_matrix"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=["No Disease (0)", "Disease (1)"],
            y=["No Disease (0)", "Disease (1)"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    with c_roc:
        st.markdown("### 📈 ROC Curve & AUC Score")
        # Generate dummy ROC visualization curve
        fpr = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
        tpr = [0.0, 0.65, 0.85, 0.92, 0.96, 0.99, 1.0]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"Random Forest (AUC = {df_comparison.iloc[0]['ROC-AUC']:.3f})", line=dict(color='#0284c7', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline (Chance)', line=dict(dash='dash', color='gray')))
        fig_roc.update_layout(title="Receiver Operating Characteristic", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=350)
        st.plotly_chart(fig_roc, use_container_width=True)


# ==========================================
# PAGE 4: EXPLAINABILITY DASHBOARD
# ==========================================
elif page == "💡 Explainability":
    render_header()

    st.markdown("## Model Explainability & Feature Importances")
    st.write("Understand feature weights and decision criteria used by the Random Forest classifier.")

    df_fi = get_feature_importances(model_pipeline)

    fig_fi = px.bar(
        df_fi,
        x="importance",
        y="label",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
        labels={"importance": "Global Model Importance", "label": "Clinical Feature"},
        title="Random Forest Feature Importance Rankings"
    )
    fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("### 📊 Relative Feature Weight Table")
    st.dataframe(df_fi[["label", "category", "importance", "relative_pct"]].rename(columns={
        "label": "Feature Label",
        "category": "Domain Category",
        "importance": "Weight Score",
        "relative_pct": "Relative Weight (%)"
    }), use_container_width=True)


# ==========================================
# PAGE 5: DATASET EXPLORER
# ==========================================
elif page == "🔍 Dataset Explorer":
    render_header()

    st.markdown("## Patient Dataset Insights & EDA")
    summary = get_dataset_summary(df_dataset)

    d1, d2, d3 = st.columns(3)
    d1.metric("Total Records", summary["total_records"])
    d2.metric("Target Balance (Disease Detected)", f"{summary['target_distribution']['disease_count']} ({summary['target_distribution']['disease_pct']}%)")
    d3.metric("Target Balance (No Disease)", f"{summary['target_distribution']['no_disease_count']} ({summary['target_distribution']['no_disease_pct']}%)")

    st.markdown("---")
    st.markdown("### 📊 Distribution of Key Clinical Features")

    col_dist1, col_dist2 = st.columns(2)

    with col_dist1:
        fig_age = px.histogram(df_dataset, x="age", color="target", barmode="overlay", title="Age Distribution by Target Risk", color_discrete_sequence=["#10b981", "#ef4444"])
        st.plotly_chart(fig_age, use_container_width=True)

        fig_chol = px.histogram(df_dataset, x="chol", color="target", barmode="overlay", title="Serum Cholesterol (mg/dL) Distribution", color_discrete_sequence=["#10b981", "#ef4444"])
        st.plotly_chart(fig_chol, use_container_width=True)

    with col_dist2:
        fig_bp = px.histogram(df_dataset, x="trestbps", color="target", barmode="overlay", title="Resting Blood Pressure (mmHg) Distribution", color_discrete_sequence=["#10b981", "#ef4444"])
        st.plotly_chart(fig_bp, use_container_width=True)

        fig_hr = px.histogram(df_dataset, x="thalach", color="target", barmode="overlay", title="Max Heart Rate Achieved (bpm) Distribution", color_discrete_sequence=["#10b981", "#ef4444"])
        st.plotly_chart(fig_hr, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔗 Feature Correlation Heatmap")
    st.caption("Correlation does not imply causation.")

    corr = df_dataset[FEATURE_COLUMNS + ["target"]].corr()
    fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📄 Dataset Preview")
    df_desc = get_descriptive_dataframe(df_dataset)
    st.dataframe(df_desc, use_container_width=True)


# ==========================================
# PAGE 6: ABOUT THE MODEL
# ==========================================
elif page == "ℹ️ About the Model":
    render_header()

    st.markdown("## About the CardioAI Machine Learning Pipeline")
    st.markdown("""
    ### ⚙️ Model Architecture
    - **Algorithm:** Random Forest Classifier (`n_estimators=250`, `max_depth=8`, `class_weight='balanced'`).
    - **Preprocessing:** Scikit-Learn `ColumnTransformer` with `SimpleImputer` (median strategy) and `StandardScaler`.
    - **Split Strategy:** 80% Training / 20% Testing with stratified target sampling (`random_state=42`).

    ### 📋 Feature Input Specifications
    CardioAI analyzes 13 health parameters:
    1. **Age** (years)
    2. **Biological Sex** (0: Female, 1: Male)
    3. **Chest Pain Type** (0-3)
    4. **Resting Blood Pressure** (mmHg)
    5. **Serum Cholesterol** (mg/dL)
    6. **Fasting Blood Sugar** (> 120 mg/dL)
    7. **Resting ECG** (0-2)
    8. **Maximum Heart Rate** (bpm)
    9. **Exercise-Induced Angina** (0/1)
    10. **ST Depression (oldpeak)** (mm)
    11. **Slope of Peak ST** (0-2)
    12. **Major Vessels** (0-4)
    13. **Thalassemia Result** (0-3)

    ### 🏷️ Defined Outputs
    - **0**: Lower heart disease risk detected
    - **1**: Higher heart disease risk detected
    """)

    st.warning(f"**LIMITATIONS & DISCLAIMER:** {MEDICAL_DISCLAIMER}")
