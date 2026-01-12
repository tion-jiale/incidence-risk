import streamlit as st
import pandas as pd
import joblib
import json



# Load model artifacts
rf_model = joblib.load("rf_model.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
cb_model = joblib.load("catboost_model.pkl")

scaler = joblib.load("scaler.pkl")  # ONLY for Logistic Regression
model_columns = joblib.load("model_columns.pkl")

# Load precomputed metrics
with open("model_metrics.json") as f:
    model_metrics = json.load(f)


# App config

st.set_page_config(
    page_title="STD Risk Assessment System",
    layout="centered"
)

st.title("📊 STD Incidence Risk Assessment")
st.markdown("""
This system estimates **population-level STD risk**
based on demographic, socioeconomic, education, and crime indicators.
""")


# Model selection
st.header("Model Selection")

# Display → internal mapping
model_display_to_key = {
    "Random Forest (recommended)": "Random Forest",
    "Logistic Regression": "Logistic Regression",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost"
}

selected_display = st.selectbox(
    "Choose Prediction Model",
    list(model_display_to_key.keys()),
    help="Random Forest is recommended due to its stable and strong overall performance."
)

# Clean internal model name
model_choice = model_display_to_key[selected_display]

state = st.selectbox(
    "State",
    options=[
        "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
        "Pahang", "Perak", "Perlis", "Pulau Pinang",
        "Sabah", "Sarawak", "Selangor", "Terengganu", "WP Kuala Lumpur"
    ]
)

cases = st.number_input("Previous STD Cases", min_value=0)
incidence = st.number_input("Incidence Rate", min_value=0.0,step=1.0)
rape = st.number_input("Reported Rape Cases", min_value=0)
students = st.number_input("Post-secondary Student Enrolment", min_value=100000, step=1000)
income_mean = st.number_input("Mean Income (RM)", min_value=1500.0, step=100.0)
income_median = st.number_input("Median Income (RM)", min_value=3000.0, step=100.0)


# Build input dataframe
input_data = pd.DataFrame([{
    "cases": cases,
    "incidence": incidence,
    "rape": rape,
    "students": students,
    "income_mean": income_mean,
    "income_median": income_median
}])

# One-hot encode state
state_encoded = pd.get_dummies(pd.Series([state]), prefix="state")
input_data = pd.concat([input_data, state_encoded], axis=1)

# Ensure feature alignment
for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model_columns]


# Prediction
# Prediction
if st.button("Assess STD Risk"):

    if model_choice == "Logistic Regression":
        input_for_model = scaler.transform(input_data)
        model = lr_model

    elif model_choice == "XGBoost":
        input_for_model = input_data
        model = xgb_model

    elif model_choice == "CatBoost":
        input_for_model = input_data.values
        model = cb_model

    else:  # Random Forest
        input_for_model = input_data
        model = rf_model

    prediction = int(model.predict(input_for_model).item())
    probabilities = model.predict_proba(input_for_model)[0]
    confidence = probabilities[prediction]




    # Map risk levels
    risk_map = {0: " Low Risk", 1: " Moderate Risk", 2: " High Risk"}

    # Display results
    st.subheader("Risk Assessment Result")
    st.success(f"**Predicted Risk Level:** {risk_map[prediction]}")

    st.markdown(f"**Model Used:** `{model_choice}`")


    # Display evaluation metrics
    st.subheader(" Model Evaluation Metrics (Test Set)")

    metrics = model_metrics[model_choice]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precision (Macro)", f"{metrics['Precision']:.3f}")
        st.metric("Recall (Macro)", f"{metrics['Recall']:.3f}")
    with col2:
        st.metric("F1-score (Macro)", f"{metrics['F1-score']:.3f}")
        st.metric("ROC-AUC (OvR)", f"{metrics['ROC-AUC']:.3f}")

    st.markdown("""
    **Interpretation**
    -  Low Risk: Below-average STD incidence
    -  Moderate Risk: Requires monitoring
    -  High Risk: Priority for intervention and planning
    """)






