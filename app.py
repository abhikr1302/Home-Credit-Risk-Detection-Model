from __future__ import annotations

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Home Credit Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GLOBAL
    ------------------------------------------------------- */

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }


    /* -------------------------------------------------------
       HEADER
    ------------------------------------------------------- */

    .main-header {
        padding: 22px 25px;
        border-radius: 12px;
        border: 1px solid #d9d9d9;
        margin-bottom: 20px;
    }

    .main-header h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 700;
    }

    .main-header p {
        margin: 6px 0 0 0;
        font-size: 16px;
    }


    /* -------------------------------------------------------
       SECTION HEADERS
    ------------------------------------------------------- */

    .section-header {
        padding: 10px 0px;
        margin-top: 12px;
        margin-bottom: 5px;
        border-bottom: 1px solid #dddddd;
    }

    .section-header h3 {
        margin: 0;
        font-size: 20px;
        font-weight: 650;
    }


    /* -------------------------------------------------------
       ASSESS BUTTON
    ------------------------------------------------------- */

    div.stButton > button {
        width: 100%;
        height: 55px;
        font-size: 18px;
        font-weight: 700;
        border-radius: 10px;
    }


    /* -------------------------------------------------------
       RESULT AREA
    ------------------------------------------------------- */

    .result-container {
        padding: 25px;
        border-radius: 14px;
        border: 1px solid #d8d8d8;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .risk-low {
        font-size: 28px;
        font-weight: 750;
    }

    .risk-medium {
        font-size: 28px;
        font-weight: 750;
    }

    .risk-high {
        font-size: 28px;
        font-weight: 750;
    }

    .probability-text {
        font-size: 18px;
        margin-top: 5px;
        margin-bottom: 20px;
    }


    /* -------------------------------------------------------
       RESULT METRICS
    ------------------------------------------------------- */

    .result-card {
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        min-height: 100px;
    }

    .result-card-title {
        font-size: 14px;
        margin-bottom: 8px;
    }

    .result-card-value {
        font-size: 22px;
        font-weight: 700;
    }


    /* -------------------------------------------------------
       PROFILE
    ------------------------------------------------------- */

    .profile-card {
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 18px;
        margin-top: 10px;
        margin-bottom: 10px;
    }


    /* -------------------------------------------------------
       RISK FLAGS
    ------------------------------------------------------- */

    .risk-flag {
        border: 1px solid #dddddd;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0px;
    }


    /* -------------------------------------------------------
       MODEL INTERPRETATION
    ------------------------------------------------------- */

    .model-card {
        border: 1px solid #dddddd;
        border-radius: 10px;
        padding: 18px;
        margin-top: 10px;
    }


    /* -------------------------------------------------------
       FOOTER
    ------------------------------------------------------- */

    .footer {
        text-align: center;
        padding-top: 25px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🏦 Home Credit Risk Assessment</h1>
        <p>
            AI-powered applicant credit-risk assessment using XGBoost
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PERSONAL INFORMATION
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h3>👤 Personal Information</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1,
    )

with col2:
    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Unknown",
        ],
    )

with col3:
    marital_status = st.selectbox(
        "Family Status",
        [
            "Married",
            "Single",
            "Separated",
            "Widow / Widower",
            "Unknown",
        ],
    )


col1, col2, col3 = st.columns(3)

with col1:
    education = st.selectbox(
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
    family_members = st.number_input(
        "Family Size",
        min_value=1,
        max_value=30,
        value=2,
        step=1,
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


col1, col2, col3 = st.columns(3)

with col1:
    residential_status = st.selectbox(
        "Residential Status",
        [
            "Owned",
            "Rented",
            "With parents",
            "Employer provided",
            "Other",
        ],
    )

with col2:
    owns_car = st.checkbox(
        "Owns Car",
        value=True,
    )

with col3:
    owns_property = st.checkbox(
        "Owns Property",
        value=True,
    )


# ============================================================
# EMPLOYMENT & INCOME
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h3>💼 Employment & Income</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    annual_income = st.number_input(
        "Annual Income (₹)",
        min_value=1.0,
        value=600000.0,
        step=10000.0,
    )

with col2:
    employment_years = st.number_input(
        "Employment (Years)",
        min_value=0.0,
        max_value=60.0,
        value=5.0,
        step=0.5,
    )

with col3:
    income_type = st.selectbox(
        "Income Type",
        [
            "Working",
            "Commercial associate",
            "Pensioner",
            "State servant",
            "Student",
            "Business owner",
            "Other",
        ],
    )


col1, col2 = st.columns(2)

with col1:
    other_monthly_income = st.number_input(
        "Other Monthly Income (₹)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
    )

with col2:
    income_stability_years = st.number_input(
        "Income Stability (Years)",
        min_value=0.0,
        max_value=60.0,
        value=3.0,
        step=0.5,
    )


# ============================================================
# CURRENT LOAN DETAILS
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h3>💰 Current Loan Details</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=1.0,
        value=300000.0,
        step=10000.0,
    )

with col2:
    monthly_emi = st.number_input(
        "Monthly Payment / EMI (₹)",
        min_value=1.0,
        value=7500.0,
        step=500.0,
    )

with col3:
    loan_term_months = st.number_input(
        "Loan Term (Months)",
        min_value=1,
        max_value=480,
        value=60,
        step=1,
    )


col1, col2, col3 = st.columns(3)

with col1:
    contract_type = st.selectbox(
        "Loan Contract",
        [
            "Cash loans",
            "Revolving loans",
        ],
    )

with col2:
    loan_type = st.selectbox(
        "Loan Type",
        [
            "Personal Loan",
            "Home Loan",
            "Auto Loan",
            "Education Loan",
            "Business Loan",
            "Consumer Loan",
            "Other",
        ],
    )

with col3:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        [
            "Home purchase",
            "Vehicle purchase",
            "Education",
            "Medical",
            "Business",
            "Debt consolidation",
            "Consumer purchase",
            "Other",
        ],
    )


secured_loan = st.checkbox(
    "Secured Loan",
    value=False,
)


# ============================================================
# EXTERNAL CREDIT SCORES
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h3>📊 External Credit Scores</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    external_score_1 = st.number_input(
        "Score 1",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01,
    )

with col2:
    external_score_2 = st.number_input(
        "Score 2",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01,
    )

with col3:
    external_score_3 = st.number_input(
        "Score 3",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01,
    )


credit_score = st.number_input(
    "Credit Score",
    min_value=0.0,
    max_value=900.0,
    value=700.0,
    step=1.0,
)


# ============================================================
# CREDIT BUREAU PROFILE
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h3>🏦 Credit Bureau Profile</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    total_credit_accounts = st.number_input(
        "Total Accounts",
        min_value=0,
        value=5,
        step=1,
    )

with col2:
    active_credit_accounts = st.number_input(
        "Active Accounts",
        min_value=0,
        value=2,
        step=1,
    )

with col3:
    closed_credit_accounts = st.number_input(
        "Closed Accounts",
        min_value=0,
        value=3,
        step=1,
    )


col1, col2, col3 = st.columns(3)

with col1:
    total_outstanding_debt = st.number_input(
        "Outstanding Debt (₹)",
        min_value=0.0,
        value=250000.0,
        step=10000.0,
    )

with col2:
    total_credit_limit = st.number_input(
        "Total Credit Limit (₹)",
        min_value=0.0,
        value=1000000.0,
        step=10000.0,
    )

with col3:
    total_overdue_amount = st.number_input(
        "Total Overdue (₹)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
    )


col1, col2, col3 = st.columns(3)

with col1:
    average_credit_history_months = st.number_input(
        "Credit History (Months)",
        min_value=0.0,
        value=48.0,
        step=1.0,
    )

with col2:
    maximum_days_past_due = st.number_input(
        "Maximum DPD",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

with col3:
    delinquent_accounts = st.number_input(
        "Delinquent Accounts",
        min_value=0,
        value=0,
        step=1,
    )


# ============================================================
# CREDIT CARD BEHAVIOUR
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h3>💳 Credit Card Behaviour</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    number_of_credit_cards = st.number_input(
        "Number of Cards",
        min_value=0,
        value=2,
        step=1,
    )

with col2:
    credit_card_balance = st.number_input(
        "Balance (₹)",
        min_value=0.0,
        value=50000.0,
        step=5000.0,
    )

with col3:
    credit_card_limit = st.number_input(
        "Limit (₹)",
        min_value=0.0,
        value=200000.0,
        step=5000.0,
    )

with col4:
    credit_card_max_utilization = st.number_input(
        "Utilization (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0,
    )


col1, col2 = st.columns(2)

with col1:
    credit_card_30plus_dpd = st.number_input(
        "30+ DPD Count",
        min_value=0,
        value=0,
        step=1,
    )

with col2:
    credit_card_average_balance = st.number_input(
        "Average Balance (₹)",
        min_value=0.0,
        value=25000.0,
        step=5000.0,
    )


# ============================================================
# PREVIOUS LOAN HISTORY
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h3>📋 Previous Loan History</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    previous_applications = st.number_input(
        "Applications",
        min_value=0,
        value=3,
        step=1,
    )

with col2:
    previous_approved = st.number_input(
        "Approved",
        min_value=0,
        value=2,
        step=1,
    )

with col3:
    previous_refused = st.number_input(
        "Refused",
        min_value=0,
        value=1,
        step=1,
    )


col1, col2, col3 = st.columns(3)

with col1:
    previous_average_loan_amount = st.number_input(
        "Average Loan Amount (₹)",
        min_value=0.0,
        value=200000.0,
        step=10000.0,
    )

with col2:
    previous_average_emi = st.number_input(
        "Average EMI (₹)",
        min_value=0.0,
        value=6000.0,
        step=500.0,
    )

with col3:
    previous_average_credit_application_ratio = st.number_input(
        "Credit/Application Ratio",
        min_value=0.0,
        value=0.70,
        step=0.01,
    )


# ============================================================
# REPAYMENT BEHAVIOUR
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <h3>⏱️ Repayment Behaviour</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    late_installment_rate = st.number_input(
        "Late Payment Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
    )

with col2:
    maximum_installment_delay = st.number_input(
        "Maximum Delay (Days)",
        min_value=0.0,
        value=5.0,
        step=1.0,
    )


col1, col2 = st.columns(2)

with col1:
    previous_loans_with_installments = st.number_input(
        "Previous Loans With Installments",
        min_value=0,
        value=3,
        step=1,
    )

with col2:
    installment_records = st.number_input(
        "Installment Records",
        min_value=0,
        value=36,
        step=1,
    )


average_payment_ratio = st.number_input(
    "Average Payment Ratio",
    min_value=0.0,
    value=0.98,
    step=0.01,
)


# ============================================================
# ADDITIONAL CREDIT RISK SIGNALS
# ============================================================

with st.expander(
    "Additional Credit Risk Information"
):

    col1, col2, col3 = st.columns(3)

    with col1:
        pos_cash_loans = st.number_input(
            "POS/Cash Loans",
            min_value=0,
            value=3,
            step=1,
        )

    with col2:
        pos_cash_active_loans = st.number_input(
            "Active POS/Cash Loans",
            min_value=0,
            value=1,
            step=1,
        )

    with col3:
        pos_cash_history_months = st.number_input(
            "POS/Cash History (Months)",
            min_value=0,
            value=36,
            step=1,
        )

    pos_cash_max_dpd = st.number_input(
        "POS/Cash Maximum DPD",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    col1, col2 = st.columns(2)

    with col1:
        social_circle_30plus_defaults = st.number_input(
            "Social Circle 30+ Defaults",
            min_value=0,
            value=0,
            step=1,
        )

    with col2:
        social_circle_60plus_defaults = st.number_input(
            "Social Circle 60+ Defaults",
            min_value=0,
            value=0,
            step=1,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        document_risk_flag = st.selectbox(
            "Document Risk Flag",
            [0, 1],
        )

    with col2:
        recent_credit_enquiries = st.number_input(
            "Recent Credit Enquiries",
            min_value=0,
            value=1,
            step=1,
        )

    with col3:
        fraud_flag = st.checkbox(
            "Fraud Risk Flag",
            value=False,
        )

    kyc_verified = st.checkbox(
        "KYC Verified",
        value=True,
    )


# ============================================================
# QUICK AFFORDABILITY SUMMARY
# ============================================================

monthly_income = (
    annual_income / 12
) + other_monthly_income

emi_income_ratio = (
    monthly_emi / monthly_income
    if monthly_income > 0
    else 0
)


st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Monthly Income",
        f"₹{monthly_income:,.0f}",
    )

with col2:
    st.metric(
        "Monthly EMI",
        f"₹{monthly_emi:,.0f}",
    )

with col3:
    st.metric(
        "EMI / Income",
        f"{emi_income_ratio * 100:.1f}%",
    )


# ============================================================
# ASSESS CREDIT RISK
# ============================================================

st.divider()

st.markdown(
    "<h3 style='text-align:center;'>🔍 Assess Credit Risk</h3>",
    unsafe_allow_html=True,
)

assess = st.button(
    "🔍 ASSESS CREDIT RISK",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PREDICTION
# ============================================================

if assess:

    payload = {

        # ----------------------------------------------------
        # Personal
        # ----------------------------------------------------

        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "family_members": family_members,
        "education": education,
        "housing_type": housing_type,
        "residential_status": residential_status,
        "owns_car": owns_car,
        "owns_property": owns_property,

        # ----------------------------------------------------
        # Contract
        # ----------------------------------------------------

        "contract_type": contract_type,

        # ----------------------------------------------------
        # Employment
        # ----------------------------------------------------

        "income_type": income_type,
        "employment_years": employment_years,
        "annual_income": annual_income,
        "other_monthly_income": other_monthly_income,
        "income_stability_years": income_stability_years,

        # ----------------------------------------------------
        # Loan
        # ----------------------------------------------------

        "loan_type": loan_type,
        "loan_purpose": loan_purpose,
        "secured_loan": secured_loan,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "monthly_emi": monthly_emi,

        # ----------------------------------------------------
        # Credit
        # ----------------------------------------------------

        "credit_score": credit_score,
        "external_score_1": external_score_1,
        "external_score_2": external_score_2,
        "external_score_3": external_score_3,

        # ----------------------------------------------------
        # Bureau
        # ----------------------------------------------------

        "total_credit_accounts":
            total_credit_accounts,

        "active_credit_accounts":
            active_credit_accounts,

        "closed_credit_accounts":
            closed_credit_accounts,

        "total_credit_limit":
            total_credit_limit,

        "total_outstanding_debt":
            total_outstanding_debt,

        "total_overdue_amount":
            total_overdue_amount,

        "average_credit_history_months":
            average_credit_history_months,

        "maximum_days_past_due":
            maximum_days_past_due,

        "delinquent_accounts":
            delinquent_accounts,

        # ----------------------------------------------------
        # Credit Card
        # ----------------------------------------------------

        "number_of_credit_cards":
            number_of_credit_cards,

        "credit_card_balance":
            credit_card_balance,

        "credit_card_limit":
            credit_card_limit,

        "credit_card_30plus_dpd":
            credit_card_30plus_dpd,

        "credit_card_max_utilization":
            credit_card_max_utilization,

        "credit_card_average_balance":
            credit_card_average_balance,

        # ----------------------------------------------------
        # Previous Applications
        # ----------------------------------------------------

        "previous_applications":
            previous_applications,

        "previous_approved":
            previous_approved,

        "previous_refused":
            previous_refused,

        "previous_average_loan_amount":
            previous_average_loan_amount,

        "previous_average_emi":
            previous_average_emi,

        "previous_average_credit_application_ratio":
            previous_average_credit_application_ratio,

        # ----------------------------------------------------
        # Installments
        # ----------------------------------------------------

        "previous_loans_with_installments":
            previous_loans_with_installments,

        "installment_records":
            installment_records,

        "late_installment_rate":
            late_installment_rate,

        "maximum_installment_delay":
            maximum_installment_delay,

        "average_payment_ratio":
            average_payment_ratio,

        # ----------------------------------------------------
        # POS/Cash
        # ----------------------------------------------------

        "pos_cash_loans":
            pos_cash_loans,

        "pos_cash_active_loans":
            pos_cash_active_loans,

        "pos_cash_history_months":
            pos_cash_history_months,

        "pos_cash_max_dpd":
            pos_cash_max_dpd,

        # ----------------------------------------------------
        # Additional
        # ----------------------------------------------------

        "social_circle_30plus_defaults":
            social_circle_30plus_defaults,

        "social_circle_60plus_defaults":
            social_circle_60plus_defaults,

        "document_risk_flag":
            document_risk_flag,

        "recent_credit_enquiries":
            recent_credit_enquiries,

        "fraud_flag":
            fraud_flag,

        "kyc_verified":
            kyc_verified,
    }

    try:

        with st.spinner(
            "Analyzing applicant credit profile..."
        ):

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=60,
            )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.status_code == 200:

            result = response.json()

            probability = float(
                result["risk_probability"]
            )

            risk_percentage = float(
                result["risk_percentage"]
            )

            risk_level = result[
                "risk_level"
            ]

            decision = result[
                "decision"
            ]

            threshold = float(
                result.get(
                    "threshold",
                    0.60,
                )
            )

            affordability = result.get(
                "affordability",
                {},
            )

            risk_flags = result.get(
                "risk_flags",
                [],
            )

            backend = result.get(
                "backend_processing",
                {},
            )

            # =================================================
            # RESULT HEADER
            # =================================================

            st.divider()

            st.markdown(
                "## 📊 Credit Risk Assessment Result"
            )

            # -------------------------------------------------
            # Risk Status
            # -------------------------------------------------

            if risk_level == "LOW":

                risk_icon = "🟢"

                st.markdown(
                    f"""
                    <div class="result-container">
                        <div class="risk-low">
                            {risk_icon} LOW RISK
                        </div>
                        <div class="probability-text">
                            Probability of repayment difficulty:
                            <b>{risk_percentage:.2f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            elif risk_level == "MEDIUM":

                risk_icon = "🟡"

                st.markdown(
                    f"""
                    <div class="result-container">
                        <div class="risk-medium">
                            {risk_icon} MEDIUM RISK
                        </div>
                        <div class="probability-text">
                            Probability of repayment difficulty:
                            <b>{risk_percentage:.2f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                risk_icon = "🔴"

                st.markdown(
                    f"""
                    <div class="result-container">
                        <div class="risk-high">
                            {risk_icon} HIGH RISK
                        </div>
                        <div class="probability-text">
                            Probability of repayment difficulty:
                            <b>{risk_percentage:.2f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # =================================================
            # RESULT CARDS
            # =================================================

            if risk_level == "LOW":

                prediction_text = "No Difficulty"

            elif risk_level == "MEDIUM":

                prediction_text = "Potential Difficulty"

            else:

                prediction_text = "Repayment Difficulty"

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-card-title">
                            Probability
                        </div>
                        <div class="result-card-value">
                            {risk_percentage:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-card-title">
                            Risk
                        </div>
                        <div class="result-card-value">
                            {risk_level}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col3:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-card-title">
                            Prediction
                        </div>
                        <div class="result-card-value">
                            {prediction_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col4:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-card-title">
                            Decision
                        </div>
                        <div class="result-card-value">
                            {decision}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # =================================================
            # PROBABILITY
            # =================================================

            st.markdown(
                "### Risk Probability"
            )

            st.progress(
                min(
                    max(
                        probability,
                        0.0,
                    ),
                    1.0,
                )
            )

            st.caption(
                f"Decision threshold: "
                f"{threshold * 100:.0f}%"
            )

            # =================================================
            # APPLICANT CREDIT PROFILE
            # =================================================

            st.markdown(
                "### 📋 Applicant Credit Profile"
            )

            profile_col1, profile_col2 = st.columns(2)

            with profile_col1:

                st.markdown(
                    """
                    <div class="profile-card">
                    """,
                    unsafe_allow_html=True,
                )

                st.write(
                    f"**Age:** {age}"
                )

                st.write(
                    f"**Employment:** "
                    f"{employment_years:.1f} years"
                )

                st.write(
                    f"**Annual Income:** "
                    f"₹{annual_income:,.0f}"
                )

                st.write(
                    f"**Credit Score:** "
                    f"{credit_score:.0f}"
                )

                st.write(
                    f"**Existing Credit Accounts:** "
                    f"{total_credit_accounts}"
                )

                st.write(
                    f"**Active Credit Accounts:** "
                    f"{active_credit_accounts}"
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

            with profile_col2:

                st.markdown(
                    """
                    <div class="profile-card">
                    """,
                    unsafe_allow_html=True,
                )

                st.write(
                    f"**Loan Amount:** "
                    f"₹{loan_amount:,.0f}"
                )

                st.write(
                    f"**Monthly EMI:** "
                    f"₹{monthly_emi:,.0f}"
                )

                st.write(
                    f"**Outstanding Debt:** "
                    f"₹{total_outstanding_debt:,.0f}"
                )

                st.write(
                    f"**Credit Card Utilization:** "
                    f"{credit_card_max_utilization:.1f}%"
                )

                st.write(
                    f"**Late Payment Rate:** "
                    f"{late_installment_rate * 100:.1f}%"
                )

                st.write(
                    f"**Previous Refused Applications:** "
                    f"{previous_refused}"
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

            # =================================================
            # AFFORDABILITY
            # =================================================

            st.markdown(
                "### 💰 Affordability Analysis"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Monthly Income",
                    f"₹{affordability.get('monthly_income', 0):,.0f}",
                )

            with col2:

                st.metric(
                    "Monthly EMI",
                    f"₹{affordability.get('monthly_emi', 0):,.0f}",
                )

            with col3:

                st.metric(
                    "Total Obligation",
                    f"₹{affordability.get('total_monthly_obligation', 0):,.0f}",
                )

            with col4:

                st.metric(
                    "DTI",
                    f"{affordability.get('debt_to_income_percentage', 0):.1f}%",
                )

            # =================================================
            # KEY RISK INDICATORS
            # =================================================

            st.markdown(
                "### ⚠️ Key Risk Indicators"
            )

            if not risk_flags:

                st.success(
                    "No major risk indicators detected."
                )

            else:

                for flag in risk_flags:

                    if (
                        flag
                        == "No major rule-based risk flags detected"
                    ):

                        st.success(
                            f"✓ {flag}"
                        )

                    else:

                        st.warning(
                            f"⚠️ {flag}"
                        )

            # =================================================
            # MODEL INTERPRETATION
            # =================================================

            st.markdown(
                "### 🤖 Model Interpretation"
            )

            with st.expander(
                "View model interpretation"
            ):

                st.write(
                    """
                    The XGBoost model evaluates the applicant
                    using a combination of personal, income,
                    credit-history, repayment and existing-debt
                    characteristics.
                    """
                )

                st.write(
                    f"""
                    **Predicted probability of repayment
                    difficulty:** {risk_percentage:.2f}%
                    """
                )

                st.write(
                    f"""
                    **Decision threshold:** 
                    {threshold * 100:.0f}%
                    """
                )

                if probability >= threshold:

                    st.write(
                        """
                        The predicted risk is above the
                        configured decision threshold.
                        The application should therefore be
                        treated as high risk and may require
                        additional manual credit review.
                        """
                    )

                else:

                    st.write(
                        """
                        The predicted risk is below the
                        configured decision threshold.
                        The application is therefore classified
                        as lower risk under the current model
                        policy.
                        """
                    )

                st.write(
                    """
                    Important: the prediction is a machine
                    learning risk score and should support,
                    not replace, responsible human credit
                    assessment and applicable lending policy.
                    """
                )

            # =================================================
            # BACKEND MODEL INFORMATION
            # =================================================

            with st.expander(
                "🔧 Technical Model Information"
            ):

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Banking Inputs",
                        backend.get(
                            "raw_banking_inputs",
                            0,
                        ),
                    )

                with col2:

                    st.metric(
                        "Engineered Features",
                        backend.get(
                            "engineered_model_features",
                            0,
                        ),
                    )

                with col3:

                    st.metric(
                        "Model Features",
                        backend.get(
                            "features_sent_to_model",
                            0,
                        ),
                    )

                st.info(
                    """
                    The frontend collects business-level
                    banking information. The backend converts
                    these inputs into the 50 features expected
                    by the trained XGBoost model.
                    """
                )

        # ----------------------------------------------------
        # API ERROR
        # ----------------------------------------------------

        else:

            st.error(
                "❌ Prediction API returned an error."
            )

            try:

                st.json(
                    response.json()
                )

            except Exception:

                st.code(
                    response.text
                )

    # --------------------------------------------------------
    # CONNECTION ERROR
    # --------------------------------------------------------

    except requests.exceptions.ConnectionError:

        st.error(
            """
            ❌ Cannot connect to the FastAPI backend.

            Start the API first:

            `uvicorn src.api:app --reload`
            """
        )

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    except requests.exceptions.Timeout:

        st.error(
            "❌ The prediction request timed out."
        )

    # --------------------------------------------------------
    # OTHER ERROR
    # --------------------------------------------------------

    except Exception as exc:

        st.error(
            f"❌ Unexpected error: {exc}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        Home Credit Risk Detection Model |
        XGBoost Credit Risk Assessment
    </div>
    """,
    unsafe_allow_html=True,
)