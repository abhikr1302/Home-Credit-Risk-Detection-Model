import os
from typing import Dict, Any

import numpy as np
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# These MUST remain aligned with the currently trained model.
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
# HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    """Convert empty/invalid values to None."""
    if value is None:
        return None

    try:
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except (ValueError, TypeError):
        return None


def make_json_safe(value):
    """Convert NaN/Infinity to JSON-safe None."""

    if isinstance(value, dict):
        return {
            key: make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

    if isinstance(value, np.floating):
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    if isinstance(value, np.integer):
        return int(value)

    return value


def calculate_ratios(
    monthly_income,
    existing_emi,
    household_expenses,
    proposed_emi,
    loan_amount,
):
    """Calculate common credit-analysis ratios."""

    monthly_income = safe_float(monthly_income) or 0
    existing_emi = safe_float(existing_emi) or 0
    household_expenses = safe_float(household_expenses) or 0
    proposed_emi = safe_float(proposed_emi) or 0
    loan_amount = safe_float(loan_amount) or 0

    total_emi = existing_emi + proposed_emi

    if monthly_income > 0:
        foir = (total_emi / monthly_income) * 100
        existing_dti = (existing_emi / monthly_income) * 100
        total_dti = (
            (total_emi + household_expenses)
            / monthly_income
        ) * 100
        loan_to_income = (
            loan_amount / (monthly_income * 12)
        )
    else:
        foir = None
        existing_dti = None
        total_dti = None
        loan_to_income = None

    disposable_income = (
        monthly_income
        - household_expenses
        - total_emi
    )

    return {
        "total_emi": total_emi,
        "foir": foir,
        "existing_dti": existing_dti,
        "total_dti": total_dti,
        "disposable_income": disposable_income,
        "loan_to_income": loan_to_income,
    }


def get_credit_capacity_message(foir):
    """Interpret EMI burden."""

    if foir is None:
        return "Unable to calculate EMI burden."

    if foir <= 30:
        return "Low EMI burden"

    if foir <= 40:
        return "Moderate EMI burden"

    if foir <= 50:
        return "High EMI burden"

    return "Very high EMI burden"


def risk_result(probability, threshold=0.50):
    """Convert model probability into a customer-friendly result."""

    if probability < 0.30:
        return (
            "LOW",
            "Lower estimated repayment difficulty",
            "The model estimates a relatively lower probability "
            "of repayment difficulty."
        )

    if probability < threshold:
        return (
            "MEDIUM",
            "Moderate estimated repayment difficulty",
            "The application may require additional review "
            "of income, liabilities and credit history."
        )

    return (
        "HIGH",
        "Higher estimated repayment difficulty",
        "The application should receive additional credit "
        "assessment before approval."
    )


def build_model_features(
    age,
    employment_years,
    monthly_income,
    loan_amount,
    proposed_emi,
    existing_emi,
    household_expenses,
    credit_score,
    max_dpd,
    previous_overdue,
    previous_loans,
    previous_approved,
    previous_refused,
    credit_card_balance,
    credit_card_limit,
    housing_type,
    education_type,
    income_type,
    family_status,
    gender,
    own_car,
    own_realty,
):
    """
    Build the currently trained model's 50-feature input.

    Important:
    The new bank-style UI fields are collected for the
    customer-facing analysis. Only features that already
    exist in the trained model are sent to the model.

    New fields will affect the ML prediction only after
    model retraining.
    """

    data: Dict[str, Any] = {feature: None for feature in FEATURES}

    # --------------------------------------------------------
    # Existing model features
    # --------------------------------------------------------

    data["age_years"] = safe_float(age)
    data["employment_years"] = safe_float(employment_years)

    data["amt_credit"] = safe_float(loan_amount)
    data["amt_annuity"] = safe_float(proposed_emi)

    data["name_housing_type"] = housing_type
    data["name_education_type"] = education_type
    data["name_income_type"] = income_type
    data["name_family_status"] = family_status
    data["code_gender"] = gender

    data["flag_own_car"] = own_car
    data["flag_own_realty"] = own_realty

    # Derived features used by current model
    income = safe_float(monthly_income)

    loan_amount_value = safe_float(loan_amount)
    proposed_emi_value = safe_float(proposed_emi)

    if income and income > 0 and loan_amount_value is not None:
        data["credit_to_income_ratio"] = (
            loan_amount_value / (income * 12)
        )

    if income and income > 0 and proposed_emi_value is not None:
        data["annuity_to_income_ratio"] = (
            proposed_emi_value / income
        )

    age_value = safe_float(age)
    if age_value is not None:
        data["days_birth"] = -(age_value * 365.25)

    employment_years_value = safe_float(employment_years)
    if employment_years_value is not None:
        data["days_employed"] = -(employment_years_value * 365.25)

    # --------------------------------------------------------
    # Credit-card information
    # --------------------------------------------------------

    card_balance = safe_float(credit_card_balance)
    card_limit = safe_float(credit_card_limit)

    if (
        card_balance is not None
        and card_limit is not None
        and card_limit > 0
    ):
        utilization = card_balance / card_limit

        data["credit_card_avg_utilization"] = utilization
        data["credit_card_max_utilization"] = utilization
        data["credit_card_avg_balance"] = card_balance

    # --------------------------------------------------------
    # Historical loan information
    # --------------------------------------------------------

    if previous_loans is not None:
        data["installment_previous_loans"] = safe_float(
            previous_loans
        )

    if previous_approved is not None:
        data["previous_approved_count"] = safe_float(
            previous_approved
        )

    if previous_refused is not None:
        data["previous_refused_count"] = safe_float(
            previous_refused
        )

    approved_count = safe_float(previous_approved)
    refused_count = safe_float(previous_refused)

    if approved_count is not None and refused_count is not None:
        total_previous = approved_count + refused_count

        if total_previous > 0:
            data["previous_approval_rate"] = (
                approved_count
                / total_previous
            )

            data["previous_refusal_rate"] = (
                refused_count
                / total_previous
            )

    if previous_overdue is not None:
        data["bureau_max_overdue"] = safe_float(
            previous_overdue
        )

    if max_dpd is not None:
        max_dpd_value = safe_float(max_dpd)
        data["installment_max_payment_delay"] = max_dpd_value

        if max_dpd_value is not None and max_dpd_value > 0:
            data["credit_card_dpd_rate"] = 1.0

    # --------------------------------------------------------
    # Existing model features without direct customer inputs
    # --------------------------------------------------------
    #
    # These remain None:
    #
    # ext_source_1
    # ext_source_2
    # ext_source_3
    # ext_source_mean
    #
    # bureau_total_credit
    # bureau_total_debt
    # bureau_avg_days_credit
    #
    # installment_late_payment_rate
    # installment_avg_payment_ratio
    #
    # POS cash features
    #
    # etc.
    #
    # FastAPI/model pipeline handles missing values.
    # --------------------------------------------------------

    return make_json_safe(data)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="🏦",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🏦 Credit Risk Assessment System")

st.markdown(
    """
    Enter the applicant's financial, employment, loan and
    credit information below. The system calculates key
    repayment-capacity indicators and then sends the available
    information to the machine-learning credit-risk model.
    """
)

st.info(
    "⚠️ This is a decision-support system, not an automatic "
    "loan approval/rejection system. Final lending decisions "
    "require appropriate human and institutional review."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("System Status")

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )

    if health_response.ok:
        st.sidebar.success("API: Online")
    else:
        st.sidebar.error("API: Error")

except requests.RequestException:
    st.sidebar.error("API: Offline")

st.sidebar.caption(f"API: {API_URL}")


# ============================================================
# 1. APPLICANT INFORMATION
# ============================================================

st.header("1️⃣ Applicant Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=30,
    )

with col2:
    gender = st.selectbox(
        "Gender",
        [
            "M",
            "F",
        ],
    )

with col3:
    family_status = st.selectbox(
        "Family Status",
        [
            "Single / not married",
            "Married",
            "Civil marriage",
            "Separated",
            "Widow / widower",
            "Unknown",
        ],
    )

with col4:
    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=15,
        value=0,
    )


col1, col2, col3, col4 = st.columns(4)

with col1:
    education_type = st.selectbox(
        "Education",
        [
            "Higher education",
            "Secondary / secondary special",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ],
    )

with col2:
    housing_type = st.selectbox(
        "Housing Type",
        [
            "House / apartment",
            "With parents",
            "Rented apartment",
            "Municipal apartment",
            "Office apartment",
            "Co-op apartment",
        ],
    )

with col3:
    own_car = st.checkbox(
        "Owns a car",
        value=False,
    )

with col4:
    own_realty = st.checkbox(
        "Owns property",
        value=False,
    )


# ============================================================
# 2. INCOME & EMPLOYMENT
# ============================================================

st.header("2️⃣ Income & Employment")

col1, col2, col3 = st.columns(3)

with col1:
    monthly_income = st.number_input(
        "Monthly Net Income (₹)",
        min_value=0.0,
        value=50000.0,
        step=5000.0,
    )

with col2:
    income_type = st.selectbox(
        "Income / Employment Type",
        [
            "Working",
            "Commercial associate",
            "Pensioner",
            "State servant",
            "Business owner",
            "Student",
            "Unemployed",
        ],
    )

with col3:
    employment_years = st.number_input(
        "Employment / Business Duration (Years)",
        min_value=0.0,
        max_value=50.0,
        value=3.0,
        step=0.5,
    )


# ============================================================
# 3. EXISTING FINANCIAL OBLIGATIONS
# ============================================================

st.header("3️⃣ Existing Financial Obligations")

col1, col2, col3 = st.columns(3)

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
        "Number of Active Loans",
        min_value=0,
        max_value=20,
        value=0,
    )


# ============================================================
# 4. REQUESTED LOAN
# ============================================================

st.header("4️⃣ Requested Loan")

col1, col2, col3 = st.columns(3)

with col1:
    loan_amount = st.number_input(
        "Requested Loan Amount (₹)",
        min_value=0.0,
        value=500000.0,
        step=25000.0,
    )

with col2:
    loan_tenure = st.number_input(
        "Loan Tenure (Months)",
        min_value=1,
        max_value=360,
        value=60,
    )

with col3:
    proposed_emi = st.number_input(
        "Expected / Proposed Monthly EMI (₹)",
        min_value=0.0,
        value=10000.0,
        step=500.0,
    )

loan_type = st.selectbox(
    "Loan Type",
    [
        "Personal Loan",
        "Home Loan",
        "Auto Loan",
        "Education Loan",
        "Consumer Loan",
        "Business Loan",
    ],
)


# ============================================================
# 5. CREDIT HISTORY
# ============================================================

st.header("5️⃣ Credit History")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=750,
    )

with col2:
    max_dpd = st.number_input(
        "Maximum Previous DPD (Days)",
        min_value=0,
        max_value=3650,
        value=0,
    )

with col3:
    previous_overdue = st.number_input(
        "Maximum Previous Overdue Amount (₹)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
    )


col1, col2, col3, col4 = st.columns(4)

with col1:
    previous_loans = st.number_input(
        "Previous Loans",
        min_value=0,
        max_value=100,
        value=0,
    )

with col2:
    previous_approved = st.number_input(
        "Previously Approved Loans",
        min_value=0,
        max_value=100,
        value=0,
    )

with col3:
    previous_refused = st.number_input(
        "Previously Refused Loans",
        min_value=0,
        max_value=100,
        value=0,
    )

with col4:
    credit_card_balance = st.number_input(
        "Credit Card Outstanding (₹)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
    )


credit_card_limit = st.number_input(
    "Total Credit Card Limit (₹)",
    min_value=0.0,
    value=100000.0,
    step=5000.0,
)


# ============================================================
# 6. FINANCIAL ANALYSIS
# ============================================================

st.header("6️⃣ Financial Analysis")

ratios = calculate_ratios(
    monthly_income=monthly_income,
    existing_emi=existing_emi,
    household_expenses=household_expenses,
    proposed_emi=proposed_emi,
    loan_amount=loan_amount,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if ratios["foir"] is not None:
        st.metric(
            "FOIR",
            f"{ratios['foir']:.1f}%",
        )
    else:
        st.metric("FOIR", "N/A")

with col2:
    if ratios["total_dti"] is not None:
        st.metric(
            "Total Debt Burden",
            f"{ratios['total_dti']:.1f}%",
        )
    else:
        st.metric("Total Debt Burden", "N/A")

with col3:
    st.metric(
        "Disposable Income",
        f"₹{ratios['disposable_income']:,.0f}",
    )

with col4:
    if ratios["loan_to_income"] is not None:
        st.metric(
            "Loan / Annual Income",
            f"{ratios['loan_to_income']:.2f}x",
        )
    else:
        st.metric(
            "Loan / Annual Income",
            "N/A",
        )


foir = ratios.get("foir")
capacity_message = (
    get_credit_capacity_message(foir)
    if foir is not None
    else "N/A"
)

st.caption(
    f"Repayment capacity indicator: **{capacity_message}**"
)

if ratios["disposable_income"] < 0:
    st.warning(
        "⚠️ The applicant's calculated monthly cash flow is "
        "negative after household expenses and total EMIs."
    )

if foir is not None and foir > 50:
    st.warning(
        "⚠️ Total EMI burden is high relative to monthly income."
    )


# ============================================================
# MODEL INPUT PREPARATION
# ============================================================

model_features = build_model_features(
    age=age,
    employment_years=employment_years,
    monthly_income=monthly_income,
    loan_amount=loan_amount,
    proposed_emi=proposed_emi,
    existing_emi=existing_emi,
    household_expenses=household_expenses,
    credit_score=credit_score,
    max_dpd=max_dpd,
    previous_overdue=previous_overdue,
    previous_loans=previous_loans,
    previous_approved=previous_approved,
    previous_refused=previous_refused,
    credit_card_balance=credit_card_balance,
    credit_card_limit=credit_card_limit,
    housing_type=housing_type,
    education_type=education_type,
    income_type=income_type,
    family_status=family_status,
    gender=gender,
    own_car=own_car,
    own_realty=own_realty,
)


# ============================================================
# 7. CREDIT RISK PREDICTION
# ============================================================

st.header("7️⃣ Credit Risk Assessment")

if st.button(
    "🔍 Check Credit Risk",
    type="primary",
    use_container_width=True,
):

    with st.spinner("Analyzing credit risk..."):

        try:

            payload = {
                "features": make_json_safe(
                    model_features
                )
            }

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

            if response.status_code != 200:

                st.error(
                    f"Prediction API returned "
                    f"HTTP {response.status_code}"
                )

                try:
                    st.code(
                        response.json()
                    )
                except Exception:
                    st.code(response.text)

            else:

                result = response.json()

                probability = float(
                    result.get(
                        "default_probability",
                        0,
                    )
                )

                threshold = float(
                    result.get(
                        "threshold",
                        0.50,
                    )
                )

                risk_level, risk_title, risk_description = (
                    risk_result(
                        probability,
                        threshold,
                    )
                )

                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                if risk_level == "LOW":
                    st.success(
                        f"### 🟢 {risk_level} RISK"
                    )

                elif risk_level == "MEDIUM":
                    st.warning(
                        f"### 🟡 {risk_level} RISK"
                    )

                else:
                    st.error(
                        f"### 🔴 {risk_level} RISK"
                    )

                st.subheader(risk_title)

                st.write(risk_description)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Model Probability",
                        f"{probability:.2%}",
                    )

                with col2:
                    st.metric(
                        "Model Threshold",
                        f"{threshold:.2%}",
                    )

                with col3:
                    st.metric(
                        "FOIR",
                        (
                            f"{ratios['foir']:.1f}%"
                            if ratios["foir"] is not None
                            else "N/A"
                        ),
                    )

                # ------------------------------------------------
                # CREDIT INDICATORS
                # ------------------------------------------------

                st.subheader("Credit Indicators")

                indicator_col1, indicator_col2 = st.columns(2)

                with indicator_col1:

                    if credit_score >= 750:
                        st.success(
                            "✓ Strong credit-score range"
                        )

                    elif credit_score >= 650:
                        st.warning(
                            "⚠ Moderate credit-score range"
                        )

                    else:
                        st.error(
                            "⚠ Low credit-score range"
                        )

                    if max_dpd == 0:
                        st.success(
                            "✓ No reported previous DPD"
                        )
                    elif max_dpd <= 30:
                        st.warning(
                            f"⚠ Previous DPD: {max_dpd} days"
                        )
                    else:
                        st.error(
                            f"⚠ Significant previous DPD: "
                            f"{max_dpd} days"
                        )

                with indicator_col2:

                    foor_value = ratios.get("foir")
                    disposable_income = ratios.get("disposable_income")

                    if foor_value is None:
                        st.info("ℹ FOIR unavailable for this scenario")
                    elif foor_value <= 30:
                        st.success(
                            "✓ EMI burden appears comfortable"
                        )

                    elif foor_value <= 40:
                        st.warning(
                            "⚠ EMI burden requires review"
                        )

                    elif foor_value <= 50:
                        st.warning(
                            "⚠ High EMI burden"
                        )

                    else:
                        st.error(
                            "⚠ Very high EMI burden"
                        )

                    if disposable_income is None:
                        st.info(
                            "ℹ Estimated disposable income unavailable"
                        )
                    elif disposable_income >= 0:
                        st.success(
                            "✓ Positive estimated disposable income"
                        )
                    else:
                        st.error(
                            "⚠ Negative estimated disposable income"
                        )

                # ------------------------------------------------
                # RECOMMENDATION
                # ------------------------------------------------

                st.subheader("Assessment Recommendation")

                foor_value = ratios.get("foir")
                disposable_income = ratios.get("disposable_income")

                if (
                    risk_level == "LOW"
                    and foor_value is not None
                    and foor_value <= 40
                    and disposable_income is not None
                    and disposable_income >= 0
                ):
                    st.success(
                        "The application appears relatively "
                        "comfortable based on the available "
                        "financial and model indicators. "
                        "Proceed with normal credit verification."
                    )

                elif risk_level == "HIGH":
                    st.error(
                        "Additional credit assessment is "
                        "recommended. Review income stability, "
                        "existing liabilities, credit history, "
                        "bank statements and repayment capacity "
                        "before making a lending decision."
                    )

                else:
                    st.warning(
                        "The application requires additional "
                        "review. Verify income, liabilities, "
                        "credit history and repayment capacity "
                        "before making a lending decision."
                    )

                # ------------------------------------------------
                # TECHNICAL DETAILS
                # ------------------------------------------------

                with st.expander(
                    "Technical Model Details"
                ):

                    st.json(result)

                    st.write(
                        "Features supplied to model:",
                        len(model_features),
                    )

                    st.write(
                        "Current trained model features:",
                        len(FEATURES),
                    )

                    st.caption(
                        "Note: Credit score, monthly income, "
                        "existing EMI, household expenses, "
                        "dependents and loan tenure are currently "
                        "collected for the bank-style UI and "
                        "financial analysis. They will affect "
                        "the XGBoost prediction only after the "
                        "model is retrained with these features."
                    )

        except requests.RequestException as exc:

            st.error(
                "Unable to connect to the FastAPI prediction service."
            )

            st.code(str(exc))

        except Exception as exc:

            st.error(
                "Unexpected error while processing the prediction."
            )

            st.code(str(exc))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Home Credit Risk Detection Model • "
    "Machine Learning Decision Support System"
)