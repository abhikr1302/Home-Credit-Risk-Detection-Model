import os
from typing import Dict, Any

import numpy as np
import requests
import streamlit as st


# ============================================================
# Configuration
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


# ============================================================
# Model Features
# Must exactly match the trained XGBoost pipeline
# ============================================================

FEATURES = [
    "ext_source_mean",
    "name_education_type",
    "name_income_type",
    "code_gender",
    "flag_own_car",
    "ext_source_3",
    "name_type_suite",
    "name_family_status",
    "name_housing_type",
    "name_contract_type",
    "installment_late_payment_rate",
    "flag_document_3",
    "ext_source_2",
    "previous_refusal_rate",
    "credit_card_avg_utilization",
    "previous_avg_credit_application_ratio",
    "region_rating_client",
    "pos_cash_loan_count",
    "age_years",
    "employment_years",
    "credit_card_avg_balance",
    "credit_card_max_utilization",
    "ext_source_1",
    "amt_goods_price",
    "def_60_cnt_social_circle",
    "amt_credit",
    "previous_approval_rate",
    "flag_own_realty",
    "amt_annuity",
    "installment_max_payment_delay",
    "installment_avg_payment_ratio",
    "def_30_cnt_social_circle",
    "bureau_max_overdue",
    "region_rating_client_w_city",
    "pos_cash_history_months",
    "previous_avg_annuity",
    "days_birth",
    "pos_cash_active_count",
    "days_employed",
    "bureau_avg_days_credit",
    "bureau_total_debt",
    "installment_previous_loans",
    "installment_total_records",
    "previous_refused_count",
    "credit_card_dpd_rate",
    "previous_approved_count",
    "pos_cash_completed_count",
    "annuity_to_income_ratio",
    "credit_to_income_ratio",
    "bureau_total_credit",
]


# ============================================================
# Helper Functions
# ============================================================

def safe_float(value: Any, default=np.nan) -> float:
    """Safely convert a value to float."""

    try:
        if value is None:
            return default

        if isinstance(value, str) and not value.strip():
            return default

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (ValueError, TypeError):
        return default


def make_json_safe(value: Any) -> Any:
    """
    Convert Python/NumPy values into JSON-safe values.

    JSON does not support NaN or Infinity.
    These values are converted to None, which becomes JSON null.
    """

    if value is None:
        return None

    if isinstance(value, (float, np.floating)):

        if not np.isfinite(value):
            return None

        return float(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    return value


def make_features_json_safe(
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert the complete feature dictionary to JSON-safe values."""

    return {
        feature: make_json_safe(value)
        for feature, value in features.items()
    }


# ============================================================
# Feature Preparation
# ============================================================

def calculate_features(
    raw: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert customer/bank-entered application information
    into the 50 features expected by the trained model.

    Historical features that normally come from:
        - Credit bureau
        - Previous applications
        - Installment history
        - Credit card history
        - POS/Cash history

    are left as NaN.

    The preprocessing pipeline on the API side handles
    missing values through its trained imputers.
    """

    # --------------------------------------------------------
    # Basic values
    # --------------------------------------------------------

    income = safe_float(
        raw.get("amt_income_total")
    )

    credit = safe_float(
        raw.get("amt_credit")
    )

    annuity = safe_float(
        raw.get("amt_annuity")
    )

    goods_price = safe_float(
        raw.get("amt_goods_price")
    )

    age = safe_float(
        raw.get("age_years")
    )

    employment_years = safe_float(
        raw.get("employment_years")
    )

    # --------------------------------------------------------
    # Start all model features as missing
    # --------------------------------------------------------

    values = {
        feature: np.nan
        for feature in FEATURES
    }

    # ========================================================
    # Applicant Information
    # ========================================================

    values["code_gender"] = raw.get(
        "code_gender",
        None
    )

    values["name_education_type"] = raw.get(
        "name_education_type",
        None
    )

    values["name_income_type"] = raw.get(
        "name_income_type",
        None
    )

    values["name_family_status"] = raw.get(
        "name_family_status",
        None
    )

    values["name_housing_type"] = raw.get(
        "name_housing_type",
        None
    )

    values["name_contract_type"] = raw.get(
        "name_contract_type",
        None
    )

    values["name_type_suite"] = raw.get(
        "name_type_suite",
        "Unaccompanied"
    )

    # ========================================================
    # Ownership
    # ========================================================

    values["flag_own_car"] = raw.get(
        "flag_own_car",
        np.nan
    )

    values["flag_own_realty"] = raw.get(
        "flag_own_realty",
        np.nan
    )

    # ========================================================
    # Financial Features
    # ========================================================

    values["amt_credit"] = credit

    values["amt_annuity"] = annuity

    values["amt_goods_price"] = goods_price

    # ========================================================
    # Age
    # ========================================================

    values["age_years"] = age

    if np.isfinite(age):
        values["days_birth"] = -(
            age * 365.25
        )

    # ========================================================
    # Employment
    # ========================================================

    values["employment_years"] = employment_years

    if np.isfinite(employment_years):
        values["days_employed"] = -(
            employment_years * 365.25
        )

    # ========================================================
    # Income Ratios
    # ========================================================

    if (
        np.isfinite(income)
        and income > 0
    ):

        if np.isfinite(credit):

            values["credit_to_income_ratio"] = (
                credit / income
            )

        if np.isfinite(annuity):

            values["annuity_to_income_ratio"] = (
                annuity / income
            )

    # ========================================================
    # External Credit Scores
    # ========================================================

    ext1 = safe_float(
        raw.get("ext_source_1")
    )

    ext2 = safe_float(
        raw.get("ext_source_2")
    )

    ext3 = safe_float(
        raw.get("ext_source_3")
    )

    values["ext_source_1"] = ext1

    values["ext_source_2"] = ext2

    values["ext_source_3"] = ext3

    external_scores = [
        score
        for score in [ext1, ext2, ext3]
        if np.isfinite(score)
    ]

    if external_scores:

        values["ext_source_mean"] = (
            sum(external_scores)
            / len(external_scores)
        )

    # ========================================================
    # Document Flag
    # ========================================================

    values["flag_document_3"] = raw.get(
        "flag_document_3",
        0
    )

    # ========================================================
    # Region Information
    # ========================================================

    values["region_rating_client"] = raw.get(
        "region_rating_client",
        np.nan
    )

    values["region_rating_client_w_city"] = raw.get(
        "region_rating_client_w_city",
        np.nan
    )

    # ========================================================
    # Final Feature Contract
    # ========================================================

    final_features = {
        feature: values.get(
            feature,
            np.nan
        )
        for feature in FEATURES
    }

    return final_features


# ============================================================
# Risk Interpretation
# ============================================================

def risk_result(
    probability: float
) -> Dict[str, str]:

    if probability >= 0.70:

        return {
            "level": "HIGH RISK",
            "message": (
                "The applicant has a high probability "
                "of repayment difficulty."
            ),
        }

    elif probability >= 0.40:

        return {
            "level": "MEDIUM RISK",
            "message": (
                "The applicant has a moderate probability "
                "of repayment difficulty."
            ),
        }

    else:

        return {
            "level": "LOW RISK",
            "message": (
                "The applicant has a relatively low "
                "probability of repayment difficulty."
            ),
        }


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="Home Credit Risk Prediction",
    page_icon="🏦",
    layout="wide",
)


# ============================================================
# Header
# ============================================================

st.title(
    "🏦 Home Credit Risk Prediction"
)

st.write(
    "Enter the applicant's basic information below "
    "to estimate their credit risk."
)

st.info(
    "Only information normally collected during a "
    "loan application is required. Historical credit, "
    "payment and loan information is handled by the "
    "model's preprocessing pipeline."
)


# ============================================================
# Sidebar - API Status
# ============================================================

with st.sidebar:

    st.header("System Status")

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )

        if health_response.status_code == 200:

            health_data = (
                health_response.json()
            )

            st.success(
                "API Connected"
            )

            st.caption(
                "Model: "
                + str(
                    health_data.get(
                        "model_type",
                        "XGBoost"
                    )
                )
            )

        else:

            st.error(
                "API is not responding correctly."
            )

    except requests.RequestException:

        st.error(
            "API is offline. Start FastAPI "
            "before making a prediction."
        )

    st.divider()

    st.caption(
        "Home Credit Risk Detection Model"
    )


# ============================================================
# Applicant Information
# ============================================================

st.header(
    "👤 Applicant Information"
)

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        [
            "M",
            "F",
        ],
        index=1,
    )

    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=80,
        value=30,
        step=1,
    )

    education = st.selectbox(
        "Education",
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ],
    )


with col2:

    family_status = st.selectbox(
        "Family Status",
        [
            "Married",
            "Single / not married",
            "Civil marriage",
            "Separated",
            "Widow",
        ],
    )

    income_type = st.selectbox(
        "Income Type",
        [
            "Working",
            "Commercial associate",
            "Pensioner",
            "State servant",
            "Student",
            "Unemployed",
            "Businessman",
            "Maternity leave",
        ],
    )

    housing_type = st.selectbox(
        "Housing Type",
        [
            "House / apartment",
            "With parents",
            "Municipal apartment",
            "Rented apartment",
            "Office apartment",
            "Co-op apartment",
        ],
    )


with col3:

    contract_type = st.selectbox(
        "Contract Type",
        [
            "Cash loans",
            "Revolving loans",
        ],
    )

    own_car = st.selectbox(
        "Own a Car?",
        [
            "No",
            "Yes",
        ],
    )

    own_realty = st.selectbox(
        "Own Property?",
        [
            "Yes",
            "No",
        ],
    )


# ============================================================
# Financial Information
# ============================================================

st.header(
    "💰 Financial Information"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    income = st.number_input(
        "Annual Income",
        min_value=10000.0,
        value=500000.0,
        step=10000.0,
        help="Applicant's total annual income.",
    )


with col2:

    credit = st.number_input(
        "Requested Credit Amount",
        min_value=1000.0,
        value=300000.0,
        step=10000.0,
        help="Amount of credit requested.",
    )


with col3:

    annuity = st.number_input(
        "Loan Annuity / EMI",
        min_value=100.0,
        value=20000.0,
        step=1000.0,
        help="Expected periodic loan payment.",
    )


with col4:

    goods_price = st.number_input(
        "Goods / Purchase Price",
        min_value=1000.0,
        value=300000.0,
        step=10000.0,
        help="Approximate price of the financed goods or asset.",
    )


# ============================================================
# Employment Information
# ============================================================

st.header(
    "💼 Employment Information"
)

employment_years = st.number_input(
    "Employment Experience (years)",
    min_value=0.0,
    max_value=60.0,
    value=5.0,
    step=0.5,
)


# ============================================================
# External Credit Information
# ============================================================

st.header(
    "📊 External Credit Information"
)

st.caption(
    "Enter external credit scores if they are available. "
    "Scores should normally be between 0 and 1."
)

col1, col2, col3 = st.columns(3)


with col1:

    ext_source_1 = st.number_input(
        "External Credit Score 1",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )


with col2:

    ext_source_2 = st.number_input(
        "External Credit Score 2",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )


with col3:

    ext_source_3 = st.number_input(
        "External Credit Score 3",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )


# ============================================================
# Optional Bank Information
# ============================================================

with st.expander(
    "🏦 Optional Bank Information"
):

    st.caption(
        "These fields can be populated by a bank employee "
        "when the information is available."
    )

    col1, col2 = st.columns(2)


    with col1:

        region_rating = st.selectbox(
            "Region Rating",
            [
                "Not Available",
                "1",
                "2",
                "3",
            ],
        )


    with col2:

        region_rating_city = st.selectbox(
            "Region Rating with City",
            [
                "Not Available",
                "1",
                "2",
                "3",
            ],
        )


# ============================================================
# Raw Applicant Input
# ============================================================

raw_input = {

    # Applicant
    "code_gender": gender,

    "name_education_type": education,

    "name_family_status": family_status,

    "name_income_type": income_type,

    "name_housing_type": housing_type,

    "name_contract_type": contract_type,

    "name_type_suite": "Unaccompanied",

    # Ownership
    "flag_own_car": (
        1
        if own_car == "Yes"
        else 0
    ),

    "flag_own_realty": (
        1
        if own_realty == "Yes"
        else 0
    ),

    # Financial
    "amt_income_total": income,

    "amt_credit": credit,

    "amt_annuity": annuity,

    "amt_goods_price": goods_price,

    # Age
    "age_years": age,

    # Employment
    "employment_years": employment_years,

    # External credit
    "ext_source_1": ext_source_1,

    "ext_source_2": ext_source_2,

    "ext_source_3": ext_source_3,

    # Document
    "flag_document_3": 0,

    # Region
    "region_rating_client": (
        np.nan
        if region_rating == "Not Available"
        else float(region_rating)
    ),

    "region_rating_client_w_city": (
        np.nan
        if region_rating_city == "Not Available"
        else float(region_rating_city)
    ),
}


# ============================================================
# Prediction Button
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Check Credit Risk",
    type="primary",
    use_container_width=True,
)


# ============================================================
# Prediction
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if income <= 0:

        st.error(
            "Annual income must be greater than zero."
        )

        st.stop()


    if credit <= 0:

        st.error(
            "Credit amount must be greater than zero."
        )

        st.stop()


    if annuity <= 0:

        st.error(
            "Loan annuity / EMI must be greater than zero."
        )

        st.stop()


    # --------------------------------------------------------
    # Generate model features
    # --------------------------------------------------------

    features = calculate_features(
        raw_input
    )


    # --------------------------------------------------------
    # Verify exactly 50 features
    # --------------------------------------------------------

    if len(features) != 50:

        st.error(
            "Model input error: expected 50 features, "
            f"but generated {len(features)}."
        )

        st.stop()


    if set(features.keys()) != set(FEATURES):

        st.error(
            "Model feature mismatch detected."
        )

        st.stop()


    # --------------------------------------------------------
    # Convert NaN / Infinity to JSON-safe null
    # --------------------------------------------------------

    json_safe_features = (
        make_features_json_safe(
            features
        )
    )


    # --------------------------------------------------------
    # Final payload
    # --------------------------------------------------------

    payload = {
        "features": json_safe_features
    }


    # --------------------------------------------------------
    # Prediction API
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing applicant credit risk..."
    ):

        try:

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

        except requests.RequestException as exc:

            st.error(
                "Unable to connect to the prediction API."
            )

            st.code(
                str(exc)
            )

            st.stop()


    # --------------------------------------------------------
    # Handle API errors
    # --------------------------------------------------------

    if response.status_code != 200:

        st.error(
            f"Prediction failed "
            f"(HTTP {response.status_code})."
        )

        try:

            st.json(
                response.json()
            )

        except ValueError:

            st.code(
                response.text
            )

        st.stop()


    # --------------------------------------------------------
    # Parse response
    # --------------------------------------------------------

    try:

        result = response.json()

    except ValueError:

        st.error(
            "The API returned an invalid response."
        )

        st.stop()


    # --------------------------------------------------------
    # Extract prediction
    # --------------------------------------------------------

    probability = safe_float(
        result.get(
            "default_probability"
        ),
        default=np.nan
    )

    predicted_class = result.get(
        "predicted_class"
    )


    if not np.isfinite(probability):

        st.error(
            "The API returned an invalid "
            "default probability."
        )

        st.stop()


    # --------------------------------------------------------
    # Risk assessment
    # --------------------------------------------------------

    risk = risk_result(
        probability
    )


    st.header(
        "📋 Credit Risk Assessment"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Default Probability",
            f"{probability * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Prediction",
            (
                "Higher Risk"
                if predicted_class == 1
                else "Lower Risk"
            )
        )


    with col3:

        st.metric(
            "Risk Level",
            risk["level"]
        )


    # --------------------------------------------------------
    # Risk message
    # --------------------------------------------------------

    if risk["level"] == "HIGH RISK":

        st.error(
            f"⚠️ {risk['message']}"
        )

    elif risk["level"] == "MEDIUM RISK":

        st.warning(
            f"⚠️ {risk['message']}"
        )

    else:

        st.success(
            f"✅ {risk['message']}"
        )


    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    st.subheader(
        "🏦 Recommendation"
    )


    if probability >= 0.70:

        st.write(
            "The application should undergo additional "
            "credit review. Consider verifying income, "
            "existing liabilities and credit history "
            "before approving the loan."
        )

    elif probability >= 0.40:

        st.write(
            "The application shows moderate risk. "
            "Additional verification of the applicant's "
            "financial and credit history is recommended."
        )

    else:

        st.write(
            "The application shows relatively low risk "
            "according to the model. Normal credit "
            "approval procedures can be followed."
        )


    # --------------------------------------------------------
    # Technical details
    # --------------------------------------------------------

    with st.expander(
        "🔧 Technical Details"
    ):

        st.write(
            "Features sent to model:",
            len(features)
        )

        st.write(
            "API endpoint:",
            f"{API_URL}/predict"
        )

        st.write(
            "Historical model features that are not "
            "available from the application form are "
            "sent as null and handled by the model's "
            "preprocessing pipeline."
        )