"""Single Streamlit entry point: python -m streamlit run app.py."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="Home Credit Risk", layout="wide")
st.title("Home Credit Risk Assessment")
st.caption("Enter applicant details to assess risk with the trained XGBoost model.")

with st.form("assessment"):
    st.subheader("Applicant and loan")
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        annual_income = st.number_input(
            "Annual income", min_value=1.0, value=600000.0, step=10000.0
        )
        loan_amount = st.number_input(
            "Loan amount", min_value=1.0, value=300000.0, step=10000.0
        )
        monthly_emi = st.number_input(
            "Monthly EMI", min_value=1.0, value=7500.0, step=500.0
        )
        employment_years = st.number_input(
            "Employment years", min_value=0.0, max_value=60.0, value=5.0
        )
    with right:
        gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
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
        marital_status = st.selectbox(
            "Family status",
            [
                "Married",
                "Single / not married",
                "Civil marriage",
                "Separated",
                "Widow",
                "Unknown",
            ],
        )
        housing_type = st.selectbox(
            "Housing type",
            [
                "House / apartment",
                "With parents",
                "Municipal apartment",
                "Rented apartment",
                "Office apartment",
                "Co-op apartment",
            ],
        )
        income_type = st.selectbox(
            "Income type",
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
        owns_car = st.checkbox("Owns a car")
        owns_property = st.checkbox("Owns property", value=True)

    st.subheader("Credit history")
    left, middle, right = st.columns(3)
    with left:
        total_credit_accounts = st.number_input(
            "Total credit accounts", min_value=0, value=3
        )
        active_credit_accounts = st.number_input(
            "Active credit accounts", min_value=0, value=1
        )
        total_outstanding_debt = st.number_input(
            "Outstanding debt", min_value=0.0, value=0.0, step=5000.0
        )
    with middle:
        credit_card_limit = st.number_input(
            "Credit card limit", min_value=0.0, value=100000.0, step=5000.0
        )
        credit_card_balance = st.number_input(
            "Credit card balance", min_value=0.0, value=20000.0, step=1000.0
        )
    with right:
        previous_applications = st.number_input(
            "Previous applications", min_value=0, value=0
        )
        previous_approved = st.number_input("Previously approved", min_value=0, value=0)
        previous_refused = st.number_input("Previously refused", min_value=0, value=0)

    with st.expander("External scores and additional inputs"):
        st.caption(
            "Use verified normalized scores (0–1). Defaults of 0.5 are placeholders and affect the prediction."
        )
        scores = [
            st.number_input(
                f"External score {i}",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
            )
            for i in range(1, 4)
        ]
        type_suite = st.selectbox(
            "Accompanied by",
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
        contract_type = st.selectbox("Contract type", ["Cash loans", "Revolving loans"])
        pos_cash_loans = st.number_input(
            "Historical POS/Cash loans", min_value=0, value=0
        )
        social_30 = st.number_input(
            "Social circle 30+ day defaults", min_value=0, value=0
        )
        social_60 = st.number_input(
            "Social circle 60+ day defaults", min_value=0, value=0
        )
        document_flag = st.number_input(
            "Document 3 flag", min_value=0, max_value=1, value=0
        )
        region_rating = st.number_input(
            "Region rating", min_value=1, max_value=3, value=2
        )
        city_rating = st.number_input(
            "Region rating including city", min_value=1, max_value=3, value=2
        )
    submitted = st.form_submit_button(
        "Check credit risk", type="primary", width="stretch"
    )

if submitted:
    payload = {
        "age": age,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "monthly_emi": monthly_emi,
        "employment_years": employment_years,
        "gender": gender,
        "education": education,
        "marital_status": marital_status,
        "housing_type": housing_type,
        "income_type": income_type,
        "owns_car": owns_car,
        "owns_property": owns_property,
        "total_credit_accounts": total_credit_accounts,
        "active_credit_accounts": active_credit_accounts,
        "total_outstanding_debt": total_outstanding_debt,
        "credit_card_limit": credit_card_limit,
        "credit_card_balance": credit_card_balance,
        "previous_applications": previous_applications,
        "previous_approved": previous_approved,
        "previous_refused": previous_refused,
        "type_suite": type_suite,
        "contract_type": contract_type,
        "pos_cash_loans": pos_cash_loans,
        "social_circle_30plus_defaults": social_30,
        "social_circle_60plus_defaults": social_60,
        "document_risk_flag": document_flag,
        "region_rating_client": region_rating,
        "region_rating_client_w_city": city_rating,
        **{f"external_score_{i}": value for i, value in enumerate(scores, 1)},
    }
    try:
        with st.spinner("Assessing credit risk..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        if response.status_code != 200:
            detail = response.json().get("detail", "Prediction failed.")
            if isinstance(detail, list):
                detail = "; ".join(item["msg"] for item in detail)
            st.error(str(detail))
            st.stop()
        result = response.json()
    except requests.RequestException:
        st.error(
            "Cannot reach the API. Start it with: python -m uvicorn src.api:app --port 8000"
        )
        st.stop()

    alert = {"LOW": st.success, "MEDIUM": st.warning, "HIGH": st.error}[
        result["risk_level"]
    ]
    alert(f"{result['risk_level']} risk")
    left, middle, right = st.columns(3)
    left.metric("Estimated Default Probability", f"{result['risk_probability']:.2%}")
    middle.metric("Model Threshold", f"{result['threshold']:.0%}")
    right.metric("Model Decision", result["decision"])
    st.write("Risk flags:", result["risk_flags"])
    st.caption(
        f"Estimated monthly debt burden: {result['affordability']['debt_to_income_percentage']:.1f}% of income."
    )

st.caption(
    "Decision support only. Scores are not calibrated probabilities or automatic lending decisions. Use the same financial units as the training data."
)
