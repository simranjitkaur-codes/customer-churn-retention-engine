"""
Customer Churn Retention Engine
Streamlit dashboard for single-customer and batch churn prediction.
"""

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_pipeline.joblib"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_metrics.json"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)


# Allow imports from src/
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendations import build_customer_report


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Retention Engine",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


# ============================================================
# LOAD PIPELINE / METRICS
# ============================================================

@st.cache_resource
def load_churn_pipeline():
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


model = load_churn_pipeline()
metrics = load_metrics()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_customer(customer_df: pd.DataFrame):
    """
    Run prediction through the complete saved pipeline.

    The pipeline handles:
    - feature engineering
    - preprocessing
    - classification
    """

    churn_class_index = list(
        model.classes_
    ).index(1)

    probability = float(
        model.predict_proba(
            customer_df
        )[0, churn_class_index]
    )

    prediction = int(
        model.predict(
            customer_df
        )[0]
    )

    return prediction, probability


def get_risk_display(probability: float):
    """
    Convert churn probability into risk category.
    """

    if probability >= 0.70:
        return "High Risk", "🔴"

    elif probability >= 0.40:
        return "Medium Risk", "🟠"

    else:
        return "Low Risk", "🟢"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Churn Engine")

st.sidebar.markdown(
    """
### Navigation

Use the tabs to:

- 👤 Predict individual customers
- 📂 Run batch predictions
- 📈 View model performance
- 🔍 Explore model insights

The application uses one saved pipeline containing
feature engineering, preprocessing, and the trained classifier.
"""
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Customer Churn Retention Engine"
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 Customer Churn Retention Engine")

st.markdown(
    """
Predict customer churn probability and generate
rule-based retention recommendations using customer
service, billing and contract information.
"""
)

st.markdown("---")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "👤 Single Customer",
        "📂 Batch Prediction",
        "📈 Model Performance",
        "🔍 Model Insights",
    ]
)


# ============================================================
# TAB 1 — SINGLE CUSTOMER
# ============================================================

with tab1:

    st.header("Individual Customer Prediction")

    st.write(
        "Enter the customer's information below."
    )

    with st.form("customer_prediction_form"):

        st.subheader("Customer Information")

        col1, col2, col3 = st.columns(3)

        # ----------------------------------------------------
        # COLUMN 1
        # ----------------------------------------------------

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x:
                    "Yes" if x == 1 else "No"
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12,
                step=1
            )

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

        # ----------------------------------------------------
        # COLUMN 2
        # ----------------------------------------------------

        with col2:

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        # ----------------------------------------------------
        # COLUMN 3
        # ----------------------------------------------------

        with col3:

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0,
                step=0.01
            )

            st.markdown("### Billing")

            st.info(
                "Total Charges are calculated automatically "
                "using tenure × monthly charges."
            )

            calculated_total = (
                tenure * monthly_charges
            )

            st.metric(
                "Estimated Total Charges",
                f"${calculated_total:,.2f}"
            )

        submitted = st.form_submit_button(
            "🔮 Predict Churn",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if submitted:

        customer = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": calculated_total,
        }

        raw_df = pd.DataFrame([customer])

        prediction, probability = predict_customer(
            raw_df
        )

        risk_level, risk_icon = get_risk_display(
            probability
        )

        report = build_customer_report(
            customer,
            probability
        )

        st.markdown("---")

        st.subheader("Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        # ----------------------------------------------------
        # CHURN SCORE
        # ----------------------------------------------------

        with result_col1:

            st.metric(
                label="Model churn score",
                value=f"{probability:.1%}",
            )

        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        with result_col2:

            st.metric(
                "Risk Level",
                f"{risk_icon} {risk_level}"
            )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with result_col3:

            prediction_text = (
                "Likely to Churn"
                if prediction == 1
                else "Likely to Stay"
            )

            st.metric(
                "Model Prediction",
                prediction_text
            )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

        st.caption(
            "Educational demo. Probability calibration and "
            "business risk cutoffs have not been validated. "
            "Retention suggestions are rule-based ideas, "
            "not proven interventions."
        )

        st.caption(
            "Current classifier: "
            f"{type(model.named_steps['classifier']).__name__}"
        )

        st.markdown("---")

        st.subheader("💡 Retention Recommendations")

        for recommendation in report["recommendations"]:

            st.info(
                f"• {recommendation}"
            )


# ============================================================
# TAB 2 — BATCH PREDICTION
# ============================================================

with tab2:

    st.header("📂 Batch Customer Prediction")

    st.write(
        "Upload a CSV containing customer records."
    )

    st.info(
        "The CSV should contain the 19 customer input "
        "features used by the model. `customerID` and "
        "`Churn` are optional."
    )

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

            st.subheader("Uploaded Data")

            st.dataframe(
                batch_df.head(10),
                use_container_width=True
            )

            missing_columns = [
                column
                for column in REQUIRED_COLUMNS
                if column not in batch_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            else:

                # Keep only the features expected by
                # the saved pipeline.
                prediction_input = batch_df[
                    REQUIRED_COLUMNS
                ].copy()

                # Ensure numeric columns are numeric.
                prediction_input["TotalCharges"] = (
                    pd.to_numeric(
                        prediction_input["TotalCharges"],
                        errors="coerce"
                    )
                )

                prediction_input["MonthlyCharges"] = (
                    pd.to_numeric(
                        prediction_input["MonthlyCharges"],
                        errors="coerce"
                    )
                )

                prediction_input["tenure"] = (
                    pd.to_numeric(
                        prediction_input["tenure"],
                        errors="coerce"
                    )
                )

                # The complete pipeline handles:
                # feature engineering
                # preprocessing
                # prediction

                churn_class_index = list(
                    model.classes_
                ).index(1)

                probabilities = (
                    model.predict_proba(
                        prediction_input
                    )[:, churn_class_index]
                )

                predictions = model.predict(
                    prediction_input
                )

                results_df = batch_df.copy()

                results_df[
                    "Churn_Probability"
                ] = probabilities

                results_df[
                    "Risk_Level"
                ] = [
                    get_risk_display(
                        probability
                    )[0]
                    for probability in probabilities
                ]

                results_df[
                    "Prediction"
                ] = [
                    "Likely to Churn"
                    if prediction == 1
                    else "Likely to Stay"
                    for prediction in predictions
                ]

                st.subheader(
                    "Prediction Results"
                )

                st.dataframe(
                    results_df,
                    use_container_width=True
                )

                st.subheader(
                    "Batch Summary"
                )

                batch_col1, batch_col2, batch_col3 = (
                    st.columns(3)
                )

                # ------------------------------------------------
                # CUSTOMERS
                # ------------------------------------------------

                with batch_col1:

                    st.metric(
                        "Customers",
                        len(results_df)
                    )

                # ------------------------------------------------
                # HIGH RISK
                # ------------------------------------------------

                with batch_col2:

                    st.metric(
                        "High Risk Customers",
                        int(
                            (
                                results_df[
                                    "Risk_Level"
                                ]
                                == "High Risk"
                            ).sum()
                        )
                    )

                # ------------------------------------------------
                # PREDICTED CHURN
                # ------------------------------------------------

                with batch_col3:

                    st.metric(
                        "Predicted Churn",
                        int(
                            (
                                results_df[
                                    "Prediction"
                                ]
                                == "Likely to Churn"
                            ).sum()
                        )
                    )

                csv_data = (
                    results_df
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                st.download_button(
                    label="⬇️ Download Predictions CSV",
                    data=csv_data,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                f"Could not process the uploaded CSV: {error}"
            )


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================

with tab3:

    st.header("📈 Model Performance")

    metrics_df = pd.DataFrame(
        metrics
    )

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("Selected Model")

    best_model_row = max(
        metrics,
        key=lambda row: row["ROC_AUC"]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Model",
            best_model_row["Model"]
        )

    with col2:

        st.metric(
            "Accuracy",
            f"{best_model_row['Accuracy']:.4f}"
        )

    with col3:

        st.metric(
            "F1 Score",
            f"{best_model_row['F1_Score']:.4f}"
        )

    with col4:

        st.metric(
            "ROC-AUC",
            f"{best_model_row['ROC_AUC']:.4f}"
        )

    st.markdown("---")

    model_comparison_path = (
        FIGURES_DIR
        / "08_model_comparison.png"
    )

    confusion_matrix_path = (
        FIGURES_DIR
        / "09_confusion_matrices.png"
    )

    if model_comparison_path.exists():

        st.subheader(
            "Model Comparison"
        )

        st.image(
            str(model_comparison_path),
            use_container_width=True
        )

    if confusion_matrix_path.exists():

        st.subheader(
            "Confusion Matrices"
        )

        st.image(
            str(confusion_matrix_path),
            use_container_width=True
        )


# ============================================================
# TAB 4 — MODEL INSIGHTS
# ============================================================

with tab4:

    st.header("🔍 Model Insights")

    insight_figures = [
        (
            "Feature Importance",
            "13_feature_importance.png"
        ),
        (
            "SHAP Summary",
            "14_shap_summary.png"
        ),
        (
            "SHAP Individual Explanation",
            "15_shap_individual_explanation.png"
        ),
    ]

    for title, filename in insight_figures:

        figure_path = (
            FIGURES_DIR
            / filename
        )

        if figure_path.exists():

            st.subheader(title)

            st.image(
                str(figure_path),
                use_container_width=True
            )

            st.markdown("---")

        else:

            st.warning(
                f"Figure not found: {filename}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Customer Churn Retention Engine • "
    "Machine Learning + Rule-Based Retention Recommendations"
)