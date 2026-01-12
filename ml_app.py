import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# --------------------------------------------------
# Load data & artifacts
# --------------------------------------------------
data = pd.read_csv("final_data.csv")

X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

rf_model = joblib.load("rf_model.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
cb_model = joblib.load("catboost_model.pkl")

scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

with open("model_metrics.json") as f:
    model_metrics = json.load(f)


# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(page_title="STD Risk Assessment System", layout="centered")

st.title("📊 STD Incidence Risk Assessment")
st.markdown("""
This system estimates **population-level STD risk**
based on demographic, socioeconomic, education, and crime indicators.
""")


# --------------------------------------------------
# Navigation panel
# --------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["📊 EDA Dashboard", "🤖 Risk Prediction"]
)


# ==================================================
# 📊 EDA DASHBOARD
# ==================================================
if page == "📊 EDA Dashboard":

    st.header("Exploratory Data Analysis (EDA)")

    if st.checkbox("Show dataset"):
        st.dataframe(data)

    eda_option = st.selectbox(
        "Choose an EDA visualization",
        [
            "Incidence by Year",
            "Cases vs Income",
            "Rape Cases Yearly"
        ]
    )

    fig, ax = plt.subplots()

    if eda_option == "Incidence by Year":
        yearly_incidence = data.groupby("year")["incidence"].mean()

        ax.bar(yearly_incidence.index.astype(str), yearly_incidence.values)
        ax.set_title("Average Incidence Rate by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Incidence Rate")

    elif eda_option == "Cases vs Income":
        low_income_cases = data.loc[
            data["income_mean"] < data["income_mean"].median(), "cases"
        ]
        high_income_cases = data.loc[
            data["income_mean"] >= data["income_mean"].median(), "cases"
        ]

        ax.boxplot(
            [low_income_cases, high_income_cases],
            labels=["Low Income", "High Income"]
        )
        ax.set_xlabel("Income Level")
        ax.set_ylabel("STD Cases")
        ax.set_title("STD Cases by Income Group")

    elif eda_option == "Rape Cases Yearly":
        yearly_rape = data.groupby("year")["rape"].mean()

        ax.plot(yearly_rape.index.astype(str), yearly_rape.values)
        ax.set_title("Rape Cases Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Rape Cases")

    st.pyplot(fig)

# ==================================================
# 🤖 RISK PREDICTION
# ==================================================
if page == "🤖 Risk Prediction":

    # -----------------------------
    # Model selection
    # -----------------------------
    st.header("Model Selection")

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

    model_choice = model_display_to_key[selected_display]

    # -----------------------------
    # User inputs
    # -----------------------------
    st.header("Input Population Indicators")

    state = st.selectbox(
        "State",
        [
            "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
            "Pahang", "Perak", "Perlis", "Pulau Pinang",
            "Sabah", "Sarawak", "Selangor", "Terengganu", "WP Kuala Lumpur"
        ]
    )

    cases = st.number_input("Previous STD Cases", min_value=0)
    incidence = st.number_input("Incidence Rate", min_value=0.0, step=1.0)
    rape = st.number_input("Reported Rape Cases", min_value=0)
    students = st.number_input("Post-secondary Student Enrolment", min_value=50, step=1000)
    income_mean = st.number_input("Mean Income (RM)", min_value=1500.0, step=100.0)
    income_median = st.number_input("Median Income (RM)", min_value=3000.0, step=100.0)

    # -----------------------------
    # Build input dataframe
    # -----------------------------
    input_data = pd.DataFrame([{
        "cases": cases,
        "incidence": incidence,
        "rape": rape,
        "students": students,
        "income_mean": income_mean,
        "income_median": income_median
    }])

    state_encoded = pd.get_dummies(pd.Series([state]), prefix="state")
    input_data = pd.concat([input_data, state_encoded], axis=1)

    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[model_columns]

    # -----------------------------
    # Prediction
    # -----------------------------
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
        else:
            input_for_model = input_data
            model = rf_model

        prediction = int(model.predict(input_for_model)[0])
        probabilities = model.predict_proba(input_for_model)[0]
        confidence = probabilities[prediction]

        # -----------------------------
        # Color-coded output
        # -----------------------------
        st.subheader("Risk Assessment Result")

        if prediction == 0:
            st.success("🟢 **Low Risk**")
        elif prediction == 1:
            st.warning("🟡 **Moderate Risk**")
        else:
            st.error("🔴 **High Risk**")

        st.info(f"Model confidence: {confidence:.2%}")
        st.markdown(f"**Model Used:** `{model_choice}`")

        # -----------------------------
        # Evaluation metrics
        # -----------------------------
        st.subheader("Model Evaluation Metrics (Test Set)")

        metrics = model_metrics[model_choice]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precision (Macro)", f"{metrics['Precision']:.3f}")
            st.metric("Recall (Macro)", f"{metrics['Recall']:.3f}")
        with col2:
            st.metric("F1-score (Macro)", f"{metrics['F1-score']:.3f}")
            st.metric("ROC-AUC (OvR)", f"{metrics['ROC-AUC']:.3f}")

    # -----------------------------
    # Model comparison: ROC–AUC
    # -----------------------------
    st.header("Model Comparison: ROC–AUC Curves")

    fig, ax = plt.subplots()

    models = {
        "Random Forest": rf_model,
        "Logistic Regression": lr_model,
        "XGBoost": xgb_model,
        "CatBoost": cb_model
    }

    for name, mdl in models.items():
        if name == "Logistic Regression":
            X_input = scaler.transform(X_test)
        elif name == "CatBoost":
            X_input = X_test.values
        else:
            X_input = X_test

        y_score = mdl.predict_proba(X_input)
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC–AUC Curve Comparison")
    ax.legend()

    st.pyplot(fig)






