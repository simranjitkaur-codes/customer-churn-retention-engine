"""
Customer Churn Prediction & Retention Engine
Streamlit Dashboard
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"


# Allow imports from src/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendations import build_customer_report


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Retention Engine",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_artifacts():
    """Load the trained model and preprocessing pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same engineered features used during training."""

    df = df.copy()

    df["AvgMonthlyCharge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    return df


# ============================================================
# PREDICTION
# ============================================================

def predict_customer(model, preprocessor, customer_df):
    """Generate churn probability and prediction."""

    customer_features = add_engineered_features(customer_df)

    X_processed = preprocessor.transform(customer_features)

    probability = float(
        model.predict_proba(X_processed)[0, 1]
    )

    prediction = int(
        model.predict(X_processed)[0]
    )

    if probability >= 0.70:
        risk_level = "High Risk"
    elif probability >= 0.40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    return {
        "churn_probability": probability,
        "prediction": prediction,
        "risk_level": risk_level,
    }


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model, preprocessor = load_artifacts()
    model_loaded = True
except Exception as error:
    model_loaded = False
    model_error = error


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📉 Churn Engine")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Dashboard",
        "Predict Churn",
    ],
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.title("📉 Customer Churn Prediction & Retention Engine")

    st.markdown("---")

    st.header("Project Goal")

    st.write(
        "Predict which telecom customers are likely to churn "
        "and generate actionable retention recommendations "
        "using a machine learning model."
    )

    st.header("Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "Test ROC-AUC",
            "0.8411"
        )

    with col3:
        st.metric(
            "Test Recall",
            "0.6909"
        )

    st.header("Project Pipeline")

    st.markdown(
        """
        **Data Cleaning**
        → **EDA**
        → **Feature Engineering**
        → **Preprocessing**
        → **Model Training**
        → **Explainability**
        → **Retention Recommendations**
        """
    )

    st.info(
        "This dashboard demonstrates an end-to-end "
        "customer churn prediction workflow."
    )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "Dashboard":

    st.title("📊 Churn Analysis Dashboard")

    st.markdown("---")

    figures_path = PROJECT_ROOT / "reports" / "figures"

    charts = [
        (
            "Churn Distribution",
            "01_churn_distribution.png"
        ),
        (
            "Churn Rate by Contract Type",
            "02_contract_churn_rate.png"
        ),
        (
            "Tenure vs Churn",
            "03_tenure_vs_churn.png"
        ),
        (
            "Monthly Charges vs Churn",
            "04_monthly_charges_vs_churn.png"
        ),
        (
            "High-Risk Customer Segments",
            "09_risk_segment_comparison.png"
        ),
        (
            "Contract × Internet Service",
            "10_contract_internet_heatmap.png"
        ),
    ]

    for title, filename in charts:

        chart_path = figures_path / filename

        if chart_path.exists():

            st.subheader(title)

            st.image(
                str(chart_path),
                use_container_width=True
            )

        else:

            st.warning(
                f"Chart not found: {filename}"
            )

    st.caption(
        "EDA charts were generated from the training dataset."
    )


# ============================================================
# PREDICT CHURN
# ============================================================

elif page == "Predict Churn":

    st.title("🔮 Predict Customer Churn")

    st.markdown("---")

    if not model_loaded:

        st.error(
            f"Unable to load model artifacts: {model_error}"
        )

        st.stop()

    st.write(
        "Enter customer information to generate a churn "
        "prediction and retention recommendations."
    )

    with st.form("customer_form"):

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # CUSTOMER INFORMATION
        # ----------------------------------------------------

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.slider(
                "Tenure (months)",
                0,
                72,
                12
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple = st.selectbox(
                "Multiple Lines",
                [
                    "Yes",
                    "No",
                    "No phone service"
                ]
            )

            internet = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No"
                ]
            )

            security = st.selectbox(
                "Online Security",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            backup = st.selectbox(
                "Online Backup",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        # ----------------------------------------------------
        # BILLING / SERVICES
        # ----------------------------------------------------

        with col2:

            device = st.selectbox(
                "Device Protection",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            tech = st.selectbox(
                "Tech Support",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0
            )

            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=10000.0,
                value=1500.0
            )

        submitted = st.form_submit_button(
            "🔮 Predict Churn"
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if submitted:

        customer_df = pd.DataFrame([{

            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,

        }])

        result = predict_customer(
            model,
            preprocessor,
            customer_df
        )

        probability = result["churn_probability"]
        risk = result["risk_level"]

        customer_dict = customer_df.iloc[0].to_dict()

        report = build_customer_report(
            customer_dict,
            probability
        )

        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Churn Probability",
                f"{probability:.1%}"
            )

        with col2:
            st.metric(
                "Risk Level",
                risk
            )

        with col3:

            prediction_text = (
                "Likely to Churn"
                if result["prediction"] == 1
                else "Likely to Stay"
            )

            st.metric(
                "Prediction",
                prediction_text
            )

        if risk == "High Risk":

            st.error(
                "⚠️ High Risk Customer"
            )

        elif risk == "Medium Risk":

            st.warning(
                "⚠️ Medium Risk Customer"
            )

        else:

            st.success(
                "✅ Low Risk Customer"
            )

        st.subheader(
            "Recommended Retention Actions"
        )

        for i, recommendation in enumerate(
            report["recommendations"],
            start=1
        ):

            st.write(
                f"**{i}.** {recommendation}"
            )

        st.caption(
            "Recommendations are generated using "
            "transparent rule-based business logic."
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Customer Churn & Retention Engine"
)