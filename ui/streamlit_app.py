import math
import os
from typing import Any

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

THRESHOLD = 0.60

FEATURES = [
    "ext_source_mean",
    "name_education_type",
    "code_gender",
    "name_income_type",
    "name_housing_type",
    "flag_own_car",
    "name_type_suite",
    "name_family_status",
    "name_contract_type",
    "flag_document_3",
    "credit_card_avg_utilization",
    "ext_source_3",
    "region_rating_client",
    "previous_refusal_rate",
    "flag_own_realty",
    "def_60_cnt_social_circle",
    "age_years",
    "pos_cash_loan_count",
    "employment_years",
    "active_bureau_account_ratio",
    "region_rating_client_w_city",
    "ext_source_1",
    "ext_source_2",
    "def_30_cnt_social_circle",
    "previous_approval_rate",
    "amt_credit",
    "previous_refused_count",
    "days_birth",
    "amt_annuity",
    "bureau_debt_to_annual_income_ratio",
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Home Credit Risk Detection",
    page_icon="🏦",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }

    .result-box {
        padding: 1.2rem;
        border-radius: 12px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .metric-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        background: #fafafa;
    }

    .small-text {
        font-size: 0.85rem;
        color: #666;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_number(value: Any, default: float = 0.0) -> float:
    """Convert a value safely to float."""

    try:
        number = float(value)

        if math.isnan(number) or math.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def make_json_safe(value: Any) -> Any:
    """
    Convert values into JSON-safe Python values.
    """

    if value is None:
        return None

    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        return value

    return value


def format_percentage(value: float) -> str:
    """Format decimal probability as percentage."""

    return f"{value * 100:.2f}%"


def get_api_error(response: requests.Response) -> str:
    """Extract useful API error information."""

    try:
        data = response.json()

        if isinstance(data, dict):

            if "detail" in data:
                detail = data["detail"]

                if isinstance(detail, list):
                    messages = []

                    for item in detail:
                        if isinstance(item, dict):
                            messages.append(
                                str(item.get("msg", item))
                            )
                        else:
                            messages.append(str(item))

                    return "; ".join(messages)

                return str(detail)

            if "message" in data:
                return str(data["message"])

        return response.text

    except Exception:
        return response.text


def check_api_health() -> bool:
    """Check whether FastAPI is available."""

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏦 Home Credit Risk Detection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Credit-risk assessment using applicant information, financial
    obligations, credit history and an XGBoost machine-learning model.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API STATUS
# ============================================================

if check_api_health():

    st.success(
        "FastAPI backend connected",
        icon="✅",
    )

else:

    st.error(
        "FastAPI backend is not available. "
        "Please start the API on http://127.0.0.1:8000.",
        icon="❌",
    )


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Applicant Information</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
        ],
    )

    family_status = st.selectbox(
        "Family Status",
        [
            "Single / not married",
            "Married",
            "Civil marriage",
            "Separated",
            "Widow",
            "Unknown",
        ],
    )

with col2:

    type_suite = st.selectbox(
        "Who accompanies the applicant?",
        [
            "Unaccompanied",
            "Family",
            "Spouse, partner",
            "Children",
            "Other_A",
            "Other_B",
            "Group of people",
        ],
    )

    education_type = st.selectbox(
        "Education",
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ],
    )

with col3:

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

    own_car = st.selectbox(
        "Owns Car",
        [
            "Yes",
            "No",
        ],
    )

    own_realty = st.selectbox(
        "Owns Property",
        [
            "Yes",
            "No",
        ],
    )


# ============================================================
# INCOME AND EMPLOYMENT
# ============================================================

st.markdown(
    '<div class="section-title">💰 Income & Employment</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:

    monthly_income = st.number_input(
        "Monthly Gross Income (₹)",
        min_value=0.0,
        value=50000.0,
        step=5000.0,
    )

with col2:

    income_type = st.selectbox(
        "Income Type",
        [
            "Working",
            "Commercial associate",
            "Pensioner",
            "State servant",
            "Businessman",
            "Student",
            "Maternity leave",
            "Unemployed",
        ],
    )

with col3:

    employment_years = st.number_input(
        "Employment Years",
        min_value=0.0,
        max_value=60.0,
        value=5.0,
        step=0.5,
    )


# ============================================================
# EXISTING FINANCIAL OBLIGATIONS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Existing Financial Obligations</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    existing_emi = st.number_input(
        "Existing Monthly EMI (₹)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
    )

with col2:

    household_expenses = st.number_input(
        "Monthly Household Expenses (₹)",
        min_value=0.0,
        value=20000.0,
        step=1000.0,
    )

with col3:

    active_loans = st.number_input(
        "Active Loans",
        min_value=0,
        max_value=50,
        value=0,
        step=1,
    )

with col4:

    total_outstanding_debt = st.number_input(
        "Total Outstanding Debt (₹)",
        min_value=0.0,
        value=0.0,
        step=5000.0,
    )


# ============================================================
# REQUESTED LOAN
# ============================================================

st.markdown(
    '<div class="section-title">🏦 Requested Loan</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    loan_amount = st.number_input(
        "Requested Loan Amount (₹)",
        min_value=0.0,
        value=300000.0,
        step=10000.0,
    )

with col2:

    loan_tenure = st.number_input(
        "Loan Tenure (Months)",
        min_value=1,
        max_value=360,
        value=60,
        step=12,
    )

with col3:

    proposed_emi = st.number_input(
        "Proposed Monthly EMI (₹)",
        min_value=0.0,
        value=7000.0,
        step=500.0,
    )

with col4:

    contract_type = st.selectbox(
        "Contract Type",
        [
            "Cash loans",
            "Revolving loans",
        ],
    )


# ============================================================
# CREDIT / BANKING INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">💳 Credit & Banking Information</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:

    total_credit_accounts = st.number_input(
        "Total Credit Accounts",
        min_value=0,
        max_value=100,
        value=3,
        step=1,
    )

    active_credit_accounts = st.number_input(
        "Active Credit Accounts",
        min_value=0,
        max_value=100,
        value=1,
        step=1,
    )

with col2:

    number_of_credit_cards = st.number_input(
        "Number of Credit Cards",
        min_value=0,
        max_value=50,
        value=1,
        step=1,
    )

    credit_card_limit = st.number_input(
        "Total Credit Card Limit (₹)",
        min_value=0.0,
        value=100000.0,
        step=5000.0,
    )

with col3:

    credit_card_balance = st.number_input(
        "Credit Card Balance (₹)",
        min_value=0.0,
        value=20000.0,
        step=1000.0,
    )


# ============================================================
# PREVIOUS APPLICATION HISTORY
# ============================================================

st.markdown(
    '<div class="section-title">📋 Previous Application History</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:

    previous_applications = st.number_input(
        "Previous Applications",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
    )

with col2:

    previous_approved = st.number_input(
        "Previously Approved",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
    )

with col3:

    previous_refused = st.number_input(
        "Previously Refused",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
    )


# ============================================================
# CALCULATED FINANCIAL INDICATORS
# ============================================================

annual_income = monthly_income * 12

total_monthly_debt = (
    existing_emi
    + proposed_emi
)

foir = (
    total_monthly_debt / monthly_income
    if monthly_income > 0
    else 0
)

disposable_income = (
    monthly_income
    - existing_emi
    - proposed_emi
    - household_expenses
)

loan_to_annual_income = (
    loan_amount / annual_income
    if annual_income > 0
    else 0
)

credit_card_utilization = (
    credit_card_balance / credit_card_limit
    if credit_card_limit > 0
    else 0
)

debt_to_income = (
    total_outstanding_debt / annual_income
    if annual_income > 0
    else 0
)


st.markdown(
    '<div class="section-title">📈 Financial Indicators</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "FOIR",
        f"{foir * 100:.1f}%",
    )

with col2:

    st.metric(
        "Disposable Income",
        f"₹{disposable_income:,.0f}",
    )

with col3:

    st.metric(
        "Loan / Annual Income",
        f"{loan_to_annual_income * 100:.1f}%",
    )

with col4:

    st.metric(
        "Credit Card Utilization",
        f"{credit_card_utilization * 100:.1f}%",
    )


# ============================================================
# BACKEND-ONLY DEFAULT VALUES
# ============================================================
#
# These fields remain part of the FastAPI model contract but are
# intentionally NOT exposed to the Streamlit user.
#
# They can later be replaced by:
# - bureau integrations
# - verified credit-bureau data
# - database lookups
# - automated feature engineering
#
# Do not remove these unless the API contract is changed.
# ============================================================

with st.expander("External scores and additional model inputs"):
    st.caption("Use verified normalized scores from 0 to 1. Defaults of 0.5 are placeholders; predictions depend on these values.")
    backend_external_score_1 = st.number_input("External score 1", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    backend_external_score_2 = st.number_input("External score 2", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    backend_external_score_3 = st.number_input("External score 3", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    backend_pos_cash_loans = st.number_input("Historical POS/Cash loans", min_value=0, value=0)
    backend_social_circle_30plus_defaults = st.number_input("Social circle 30+ day defaults", min_value=0, value=0)
    backend_social_circle_60plus_defaults = st.number_input("Social circle 60+ day defaults", min_value=0, value=0)
    backend_document_risk_flag = st.number_input("Document 3 flag", min_value=0, max_value=1, value=0)
    backend_region_rating_client = st.number_input("Region rating", min_value=1, max_value=3, value=2)
    backend_region_rating_client_w_city = st.number_input("Region rating including city", min_value=1, max_value=3, value=2)


# ============================================================
# PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Credit Risk Assessment</div>',
    unsafe_allow_html=True,
)

predict_clicked = st.button(
    "Check Credit Risk",
    type="primary",
    width="stretch",
)


if predict_clicked:

    if monthly_income <= 0:

        st.error(
            "Monthly income must be greater than zero."
        )

        st.stop()

    if loan_amount <= 0:

        st.error(
            "Requested loan amount must be greater than zero."
        )

        st.stop()

    if proposed_emi <= 0:

        st.error(
            "Proposed EMI must be greater than zero."
        )

        st.stop()

    if active_credit_accounts > total_credit_accounts:

        st.error(
            "Active credit accounts cannot exceed total credit accounts."
        )

        st.stop()

    if previous_approved + previous_refused > previous_applications:

        st.error(
            "Previously approved + previously refused "
            "cannot exceed previous applications."
        )

        st.stop()

    # --------------------------------------------------------
    # API PAYLOAD
    # --------------------------------------------------------
    #
    # Important:
    # Do NOT send:
    #
    #     {"features": model_features}
    #
    # The FastAPI backend accepts applicant-level fields and
    # performs the model feature transformation itself.
    # --------------------------------------------------------

    payload = {
        # Applicant
        "age": int(age),
        "gender": gender,
        "marital_status": family_status,
        "education": education_type,
        "housing_type": housing_type,
        "type_suite": type_suite,
        "owns_car": own_car,
        "owns_property": own_realty,

        # Income
        "income_type": income_type,
        "employment_years": float(employment_years),
        "annual_income": float(annual_income),

        # Loan
        "contract_type": contract_type,
        "loan_amount": float(loan_amount),
        "monthly_emi": float(proposed_emi),

        # Backend-only external information
        "external_score_1": float(backend_external_score_1),
        "external_score_2": float(backend_external_score_2),
        "external_score_3": float(backend_external_score_3),

        # Backend-only regional information
        "region_rating_client": int(
            backend_region_rating_client
        ),
        "region_rating_client_w_city": int(
            backend_region_rating_client_w_city
        ),

        # Banking / credit
        "total_credit_accounts": int(
            total_credit_accounts
        ),
        "active_credit_accounts": int(
            active_credit_accounts
        ),
        "total_outstanding_debt": float(
            total_outstanding_debt
        ),
        "number_of_credit_cards": int(
            number_of_credit_cards
        ),
        "credit_card_balance": float(
            credit_card_balance
        ),
        "credit_card_limit": float(
            credit_card_limit
        ),

        # Previous applications
        "previous_applications": int(
            previous_applications
        ),
        "previous_approved": int(
            previous_approved
        ),
        "previous_refused": int(
            previous_refused
        ),

        # Backend-only indicators
        "pos_cash_loans": int(
            backend_pos_cash_loans
        ),
        "social_circle_30plus_defaults": int(
            backend_social_circle_30plus_defaults
        ),
        "social_circle_60plus_defaults": int(
            backend_social_circle_60plus_defaults
        ),
        "document_risk_flag": int(
            backend_document_risk_flag
        ),
    }

    # --------------------------------------------------------
    # API REQUEST
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Running credit-risk assessment..."
        ):

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

        if response.status_code != 200:

            error_message = get_api_error(response)

            st.error(
                f"Prediction failed: {error_message}"
            )

            st.stop()

        result = response.json()

    except requests.RequestException as exc:

        st.error(
            "Unable to connect to the FastAPI backend."
        )

        st.code(str(exc))

        st.stop()

    # ========================================================
    # RESULT
    # ========================================================

    default_probability = safe_number(
        result["risk_probability"]
    )

    threshold = safe_number(
        result.get(
            "threshold",
            THRESHOLD,
        ),
        THRESHOLD,
    )

    risk_level = str(
        result.get(
            "risk_level",
            "UNKNOWN",
        )
    )

    decision = str(
        result.get(
            "decision",
            "UNKNOWN",
        )
    )

    risk_flags = result.get(
        "risk_flags",
        [],
    )

    if not isinstance(
        risk_flags,
        list,
    ):
        risk_flags = []


    # ========================================================
    # RISK RESULT DISPLAY
    # ========================================================

    risk_upper = risk_level.upper()

    if "LOW" in risk_upper:

        st.success(
            f"### 🟢 {risk_upper}",
        )

    elif "MEDIUM" in risk_upper:

        st.warning(
            f"### 🟡 {risk_upper}",
        )

    elif "HIGH" in risk_upper:

        st.error(
            f"### 🔴 {risk_upper}",
        )

    else:

        st.info(
            f"### {risk_upper}",
        )


    # ========================================================
    # MODEL METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Estimated Default Probability",
            format_percentage(
                default_probability
            ),
        )

    with col2:

        st.metric(
            "Model Threshold",
            format_percentage(
                threshold
            ),
        )

    with col3:

        st.metric(
            "Model Decision",
            decision,
        )


    st.caption(
        "The probability is the model's estimated default "
        "probability. It is not a guaranteed prediction of "
        "future repayment behaviour."
    )


    # ========================================================
    # FINANCIAL INDICATORS
    # ========================================================

    st.markdown(
        "### Financial Indicators"
    )

    if foir <= 0.40:

        st.success(
            "✓ Lower EMI burden"
        )

    elif foir <= 0.60:

        st.warning(
            "⚠ Moderate EMI burden"
        )

    else:

        st.error(
            "⚠ Higher EMI burden"
        )


    if disposable_income > 0:

        st.success(
            "✓ Positive calculated disposable income"
        )

    else:

        st.error(
            "⚠ Negative calculated disposable income"
        )


    # ========================================================
    # MODEL / INPUT RISK FLAGS
    # ========================================================

    st.markdown(
        "### Model/Input Risk Flags"
    )

    if risk_flags:

        for flag in risk_flags:

            st.warning(
                f"⚠ {flag}"
            )

    else:

        st.success(
            "No major rule-based risk flags detected"
        )


    # ========================================================
    # ASSESSMENT
    # ========================================================

    st.markdown(
        "### Assessment"
    )

    st.info(
        """
        The model result should be considered alongside verified
        income, existing liabilities, credit history and the
        institution's applicable credit policy.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Home Credit Risk Detection Model | "
    "XGBoost + FastAPI + Streamlit"
)

st.caption(
    "This application is a decision-support tool and does not "
    "constitute an automatic lending decision."
)