import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Home Credit Risk Prediction",
    page_icon="🏦",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
FEATURE_FILE = Path("reports/recommended_features.json")

# The UI presents human-friendly fields and sends the model's 50 selected
# features to the FastAPI prediction endpoint.
FEATURES = [
    "ext_source_mean", "name_education_type", "name_income_type", "code_gender",
    "name_housing_type", "flag_own_car", "name_family_status", "ext_source_3",
    "name_type_suite", "name_contract_type", "installment_late_payment_rate",
    "ext_source_2", "region_rating_client", "credit_card_avg_utilization",
    "pos_cash_loan_count", "previous_refusal_rate", "previous_avg_credit_application_ratio",
    "flag_document_3", "flag_own_realty", "credit_card_avg_balance", "employment_years",
    "age_years", "amt_goods_price", "credit_card_max_utilization", "def_30_cnt_social_circle",
    "ext_source_1", "def_60_cnt_social_circle", "amt_credit", "region_rating_client_w_city",
    "installment_avg_payment_ratio", "installment_previous_loans", "amt_annuity",
    "pos_cash_history_months", "days_birth", "previous_avg_annuity", "installment_total_records",
    "pos_cash_active_count", "installment_max_payment_delay", "bureau_avg_days_credit",
    "previous_approval_rate", "bureau_total_debt", "annuity_to_income_ratio",
    "bureau_total_overdue", "days_last_phone_change", "bureau_total_credit",
    "bureau_total_accounts", "days_employed", "credit_to_income_ratio",
    "installment_late_payment_count", "previous_avg_credit_amount",
]

LABELS = {
    "amt_income_total": "Annual Income",
    "amt_credit": "Credit Amount",
    "amt_annuity": "Loan Annuity",
    "amt_goods_price": "Goods Price",
    "age_years": "Age (years)",
    "employment_years": "Employment (years)",
    "ext_source_1": "External Score 1",
    "ext_source_2": "External Score 2",
    "ext_source_3": "External Score 3",
    "ext_source_mean": "External Score Average",
}

CATEGORIES = {
    "code_gender": ["M", "F"],
    "flag_own_car": [0, 1],
    "flag_own_realty": [0, 1],
    "flag_document_3": [0, 1],
    "name_contract_type": ["Cash loans", "Revolving loans"],
    "name_income_type": ["Working", "Commercial associate", "Pensioner", "State servant", "Student", "Other"],
    "name_education_type": ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
    "name_family_status": ["Married", "Single / not married", "Civil marriage", "Separated", "Widow", "Unknown"],
    "name_housing_type": ["House / apartment", "With parents", "Rented apartment", "Municipal apartment", "Office apartment", "Co-op apartment"],
    "name_type_suite": ["Unaccompanied", "Family", "Spouse, partner", "Children", "Other_A", "Other_B", "Group of people"],
}


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_features(raw):
    income = max(safe_float(raw.get("amt_income_total")), 1.0)
    family = max(safe_float(raw.get("cnt_fam_members"), 1.0), 1.0)
    credit = safe_float(raw.get("amt_credit"))
    annuity = safe_float(raw.get("amt_annuity"))

    values = dict(raw)
    values["age_years"] = safe_float(raw.get("age_years"))
    values["employment_years"] = safe_float(raw.get("employment_years"))
    values["credit_to_income_ratio"] = credit / income
    values["annuity_to_income_ratio"] = annuity / income
    values["income_per_family_member"] = income / family

    ext = [safe_float(raw.get(k), 0.0) for k in ("ext_source_1", "ext_source_2", "ext_source_3")]
    values["ext_source_mean"] = sum(ext) / 3.0
    return {feature: values.get(feature, 0) for feature in FEATURES}


def field(label, key, default=0.0, step=0.01, min_value=None, max_value=None):
    kwargs = {"label": label, "value": default, "step": step}
    if min_value is not None:
        kwargs["min_value"] = min_value
    if max_value is not None:
        kwargs["max_value"] = max_value
    return st.number_input(**kwargs)


def select(label, key, options, index=0):
    return st.selectbox(label, options, index=index)


def build_form():
    raw = {}

    st.subheader("👤 Applicant Profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        raw["code_gender"] = select("Gender", "gender", CATEGORIES["code_gender"])
        raw["cnt_children"] = field("Number of Children", "children", 0.0, 1.0, 0.0)
        raw["cnt_fam_members"] = field("Family Members", "family", 2.0, 1.0, 1.0)
    with c2:
        raw["name_education_type"] = select("Education", "education", CATEGORIES["name_education_type"], 0)
        raw["name_family_status"] = select("Family Status", "family_status", CATEGORIES["name_family_status"], 0)
        raw["name_type_suite"] = select("Type of Suite", "suite", CATEGORIES["name_type_suite"], 0)
    with c3:
        raw["name_income_type"] = select("Income Type", "income_type", CATEGORIES["name_income_type"], 0)
        raw["name_housing_type"] = select("Housing Type", "housing", CATEGORIES["name_housing_type"], 0)
        raw["name_contract_type"] = select("Contract Type", "contract", CATEGORIES["name_contract_type"], 0)

    st.subheader("💰 Financial Profile")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        raw["amt_income_total"] = field("Annual Income", "income", 180000.0, 1000.0, 0.0)
    with c2:
        raw["amt_credit"] = field("Credit Amount", "credit", 500000.0, 1000.0, 0.0)
    with c3:
        raw["amt_annuity"] = field("Loan Annuity", "annuity", 25000.0, 500.0, 0.0)
    with c4:
        raw["amt_goods_price"] = field("Goods Price", "goods", 450000.0, 1000.0, 0.0)

    c1, c2, c3 = st.columns(3)
    with c1:
        raw["age_years"] = field("Age (years)", "age", 35.0, 1.0, 18.0, 100.0)
    with c2:
        raw["employment_years"] = field("Employment (years)", "employment", 5.0, 0.5, 0.0, 60.0)
    with c3:
        raw["days_last_phone_change"] = field("Days Since Phone Change", "phone_change", 100.0, 1.0)

    st.subheader("📊 External Credit Signals")
    c1, c2, c3 = st.columns(3)
    for col, key in zip((c1, c2, c3), ("ext_source_1", "ext_source_2", "ext_source_3")):
        with col:
            raw[key] = field(LABELS[key], key, 0.50, 0.01, 0.0, 1.0)

    st.subheader("🏦 Credit & Bureau History")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        raw["bureau_total_accounts"] = field("Bureau Accounts", "bureau_accounts", 5.0, 1.0, 0.0)
    with c2:
        raw["bureau_total_credit"] = field("Bureau Total Credit", "bureau_credit", 250000.0, 1000.0, 0.0)
    with c3:
        raw["bureau_total_debt"] = field("Bureau Total Debt", "bureau_debt", 80000.0, 1000.0, 0.0)
    with c4:
        raw["bureau_total_overdue"] = field("Bureau Overdue", "bureau_overdue", 0.0, 100.0, 0.0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        raw["bureau_avg_days_credit"] = field("Avg. Credit Age (days)", "bureau_age", -800.0, 10.0)
    with c2:
        raw["previous_approval_rate"] = field("Previous Approval Rate", "approval_rate", 0.60, 0.01, 0.0, 1.0)
    with c3:
        raw["previous_refusal_rate"] = field("Previous Refusal Rate", "refusal_rate", 0.20, 0.01, 0.0, 1.0)
    with c4:
        raw["previous_avg_credit_amount"] = field("Previous Avg. Credit", "previous_credit", 200000.0, 1000.0, 0.0)

    st.subheader("💳 Payment Behaviour")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        raw["installment_late_payment_rate"] = field("Installment Late Rate", "late_rate", 0.10, 0.01, 0.0, 1.0)
    with c2:
        raw["installment_avg_payment_ratio"] = field("Payment Ratio", "payment_ratio", 0.95, 0.01, 0.0, 2.0)
    with c3:
        raw["installment_previous_loans"] = field("Previous Loans", "installment_loans", 3.0, 1.0, 0.0)
    with c4:
        raw["installment_total_records"] = field("Installment Records", "installment_records", 20.0, 1.0, 0.0)

    st.subheader("💳 Credit Card & POS Cash")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        raw["credit_card_avg_utilization"] = field("Card Avg. Utilization", "card_util", 0.30, 0.01, 0.0, 5.0)
    with c2:
        raw["credit_card_max_utilization"] = field("Card Max Utilization", "card_max_util", 0.60, 0.01, 0.0, 10.0)
    with c3:
        raw["credit_card_avg_balance"] = field("Card Avg. Balance", "card_balance", 50000.0, 1000.0, 0.0)
    with c4:
        raw["pos_cash_loan_count"] = field("POS Cash Loans", "pos_loans", 2.0, 1.0, 0.0)

    c1, c2, c3 = st.columns(3)
    with c1:
        raw["pos_cash_history_months"] = field("POS History (months)", "pos_history", 12.0, 1.0, 0.0)
    with c2:
        raw["pos_cash_active_count"] = field("Active POS Loans", "pos_active", 1.0, 1.0, 0.0)
    with c3:
        raw["installment_max_payment_delay"] = field("Max Payment Delay (days)", "max_delay", 5.0, 1.0)

    st.subheader("⚙️ Other Indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        raw["region_rating_client"] = field("Region Rating", "region_rating", 2.0, 1.0, 1.0, 3.0)
    with c2:
        raw["region_rating_client_w_city"] = field("Region/City Rating", "region_city", 2.0, 1.0, 1.0, 3.0)
    with c3:
        raw["def_30_cnt_social_circle"] = field("30-Day Social Defaults", "social30", 0.0, 1.0, 0.0)
    with c4:
        raw["def_60_cnt_social_circle"] = field("60-Day Social Defaults", "social60", 0.0, 1.0, 0.0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        raw["flag_own_car"] = select("Owns Car", "car", [0, 1])
    with c2:
        raw["flag_own_realty"] = select("Owns Property", "realty", [0, 1])
    with c3:
        raw["flag_document_3"] = select("Document 3 Flag", "doc3", [0, 1])
    with c4:
        raw["days_employed"] = field("Days Employed", "days_employed", -1800.0, 10.0)

    return calculate_features(raw)


def risk_result(result):
    probability = float(result["default_probability"])
    percent = probability * 100
    risk = str(result["risk_level"]).upper()

    st.divider()
    st.subheader("🎯 Prediction Result")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Default Probability", f"{percent:.2f}%")
    with c2:
        st.metric("Risk Level", risk)
    with c3:
        st.metric("Prediction Class", result["predicted_class"])

    st.progress(min(max(probability, 0.0), 1.0), text=f"Risk probability: {percent:.2f}%")

    if risk == "HIGH":
        st.error("⚠️ High Risk — the model predicts a higher probability of repayment difficulty.")
    else:
        st.success("✅ Low Risk — the model predicts a lower probability of repayment difficulty.")

    st.caption(
        f"Model threshold: {result['threshold']:.2f} | "
        f"Features supplied: {result['supplied_feature_count']} | "
        f"Missing features: {result['missing_feature_count']}"
    )


def main():
    st.markdown("# 🏦 Home Credit Risk Prediction")
    st.markdown("### AI-powered applicant repayment-risk assessment")

    with st.sidebar:
        st.header("⚙️ Model Connection")
        api_url = st.text_input("FastAPI URL", API_URL).rstrip("/")
        st.info("The UI sends the selected 50 model features to the FastAPI prediction endpoint.")
        if st.button("Check API"):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.ok:
                    st.success("API is healthy")
                else:
                    st.error(f"API returned {response.status_code}")
            except requests.RequestException as exc:
                st.error(f"API unavailable: {exc}")

    applicant_id = st.number_input("Applicant ID (optional)", min_value=0, value=100001, step=1)
    features = build_form()

    st.divider()
    left, center, right = st.columns([1, 1, 1])
    with center:
        predict_clicked = st.button("🔍 PREDICT CREDIT RISK", type="primary", use_container_width=True)

    if predict_clicked:
        payload = {"applicant_id": int(applicant_id), "features": features}
        try:
            with st.spinner("Running XGBoost prediction..."):
                response = requests.post(f"{api_url}/predict", json=payload, timeout=30)
            if response.ok:
                risk_result(response.json())
            else:
                st.error(f"Prediction failed ({response.status_code})")
                st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"Could not connect to FastAPI: {exc}")

    with st.expander("🔧 Model Feature Payload"):
        st.caption(f"Exactly {len(FEATURES)} selected model features are sent to the API.")
        st.dataframe(pd.DataFrame({"Feature": FEATURES, "Value": [features[f] for f in FEATURES]}), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
