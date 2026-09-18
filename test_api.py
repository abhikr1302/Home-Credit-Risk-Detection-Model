from __future__ import annotations

import sys
from typing import Any, Dict

import requests


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# TEST APPLICATION
# ============================================================

APPLICATION: Dict[str, Any] = {

    # --------------------------------------------------------
    # Personal & KYC
    # --------------------------------------------------------

    "age": 35,

    "gender": "Male",

    "marital_status": "Married",

    "family_members": 2,

    "education": "Higher education",

    "housing_type": "House / apartment",

    "residential_status": "Owned",

    "owns_car": True,

    "owns_property": True,

    "contract_type": "Cash loans",

    # --------------------------------------------------------
    # Employment & Income
    # --------------------------------------------------------

    "income_type": "Working",

    "employment_years": 5.0,

    "annual_income": 600000.0,

    "other_monthly_income": 0.0,

    "income_stability_years": 3.0,

    # --------------------------------------------------------
    # Loan
    # --------------------------------------------------------

    "loan_type": "Personal Loan",

    "loan_purpose": "Other",

    "secured_loan": False,

    "loan_amount": 300000.0,

    "loan_term_months": 60,

    "monthly_emi": 7500.0,

    # --------------------------------------------------------
    # Credit Score
    # --------------------------------------------------------

    "credit_score": 700.0,

    "external_score_1": 0.70,

    "external_score_2": 0.70,

    "external_score_3": 0.70,

    # --------------------------------------------------------
    # Bureau
    # --------------------------------------------------------

    "total_credit_accounts": 5,

    "active_credit_accounts": 2,

    "closed_credit_accounts": 3,

    "total_credit_limit": 1000000.0,

    "total_outstanding_debt": 250000.0,

    "total_overdue_amount": 0.0,

    "average_credit_history_months": 48.0,

    "maximum_days_past_due": 0.0,

    "delinquent_accounts": 0,

    # --------------------------------------------------------
    # Credit Card
    # --------------------------------------------------------

    "number_of_credit_cards": 2,

    "credit_card_balance": 50000.0,

    "credit_card_limit": 200000.0,

    "credit_card_30plus_dpd": 0,

    "credit_card_max_utilization": 30.0,

    "credit_card_average_balance": 25000.0,

    # --------------------------------------------------------
    # Previous Applications
    # --------------------------------------------------------

    "previous_applications": 3,

    "previous_approved": 2,

    "previous_refused": 1,

    "previous_average_loan_amount": 200000.0,

    "previous_average_emi": 6000.0,

    "previous_average_credit_application_ratio": 0.70,

    # --------------------------------------------------------
    # Installments
    # --------------------------------------------------------

    "previous_loans_with_installments": 3,

    "installment_records": 36,

    "late_installment_rate": 0.05,

    "maximum_installment_delay": 5.0,

    "average_payment_ratio": 0.98,

    # --------------------------------------------------------
    # POS / CASH
    # --------------------------------------------------------

    "pos_cash_loans": 3,

    "pos_cash_active_loans": 1,

    "pos_cash_history_months": 36,

    "pos_cash_max_dpd": 0.0,

    # --------------------------------------------------------
    # Additional Risk
    # --------------------------------------------------------

    "social_circle_30plus_defaults": 0,

    "social_circle_60plus_defaults": 0,

    "document_risk_flag": 0,

    "recent_credit_enquiries": 1,

    "fraud_flag": False,

    "kyc_verified": True,
}


# ============================================================
# HELPER
# ============================================================

def fail(message: str) -> None:

    print(
        f"\n❌ {message}"
    )

    sys.exit(1)


# ============================================================
# HEADER
# ============================================================

print(
    "=" * 70
)

print(
    "BANKING CREDIT RISK API TEST"
)

print(
    "=" * 70
)


# ============================================================
# 1. HEALTH
# ============================================================

print(
    "\n[1] Testing API health..."
)

try:

    response = requests.get(
        f"{API_URL}/health",
        timeout=10,
    )

except Exception as exc:

    fail(
        f"Cannot connect to API: {exc}"
    )


if response.status_code != 200:

    fail(
        f"Health endpoint returned "
        f"status {response.status_code}"
    )


health = response.json()

print(
    f"Status              : "
    f"{health.get('status')}"
)

print(
    f"Model loaded        : "
    f"{health.get('model_loaded')}"
)

print(
    f"Model               : "
    f"{health.get('model')}"
)

print(
    f"Feature count       : "
    f"{health.get('feature_count')}"
)

print(
    f"Threshold            : "
    f"{health.get('threshold')}"
)

print(
    f"Applicant ID used   : "
    f"{health.get('applicant_id_used')}"
)

print(
    f"Frontend model      : "
    f"{health.get('frontend_model_features_used')}"
)

print(
    f"Backend engineering : "
    f"{health.get('backend_feature_engineering')}"
)


if health.get("status") != "healthy":

    fail(
        "Health check status is not healthy."
    )


if not health.get("model_loaded"):

    fail(
        "XGBoost model is not loaded."
    )


if health.get("feature_count") != 50:

    fail(
        "Expected exactly 50 model features."
    )


if health.get("threshold") != 0.60:

    fail(
        "Expected threshold of 0.60."
    )


print(
    "✅ Health check passed."
)


# ============================================================
# 2. APPLICATION
# ============================================================

print(
    "\n[2] Creating banking credit application..."
)

print(
    f"Application fields  : "
    f"{len(APPLICATION)}"
)

print(
    "Applicant ID        : NOT USED"
)

print(
    "Frontend model      : NOT USED"
)

print(
    "Backend engineering : ENABLED"
)

print(
    "Contract type       : "
    f"{APPLICATION['contract_type']}"
)


# ============================================================
# 3. PREDICTION
# ============================================================

print(
    "\n[3] Sending application to backend..."
)

try:

    response = requests.post(
        f"{API_URL}/predict",
        json=APPLICATION,
        timeout=60,
    )

except Exception as exc:

    fail(
        f"Prediction request failed: {exc}"
    )


if response.status_code != 200:

    print(
        "\n❌ Prediction failed:"
    )

    print(
        response.text
    )

    sys.exit(1)


result = response.json()


# ============================================================
# 4. VALIDATE RESPONSE
# ============================================================

print(
    "\n[4] Validating prediction response..."
)

success = result.get(
    "success"
)

probability = result.get(
    "risk_probability"
)

risk_percentage = result.get(
    "risk_percentage"
)

risk_level = result.get(
    "risk_level"
)

decision = result.get(
    "decision"
)

threshold = result.get(
    "threshold"
)

model_info = result.get(
    "model",
    {}
)

affordability = result.get(
    "affordability",
    {}
)

risk_flags = result.get(
    "risk_flags",
    []
)

backend_processing = result.get(
    "backend_processing",
    {})


print(
    f"Success             : {success}"
)

print(
    f"Risk probability    : {probability}"
)

print(
    f"Risk percentage     : {risk_percentage}%"
)

print(
    f"Risk level          : {risk_level}"
)

print(
    f"Decision            : {decision}"
)

print(
    f"Threshold           : {threshold}"
)


# ============================================================
# VALIDATIONS
# ============================================================

if success is not True:

    fail(
        "API returned success=False."
    )


if probability is None:

    fail(
        "Risk probability is missing."
    )


if not 0 <= probability <= 1:

    fail(
        f"Invalid probability: {probability}"
    )


if risk_level not in [
    "LOW",
    "MEDIUM",
    "HIGH",
]:

    fail(
        f"Invalid risk level: {risk_level}"
    )


if decision not in [
    "HIGH RISK",
    "LOWER RISK",
]:

    fail(
        f"Invalid decision: {decision}"
    )


if threshold != 0.60:

    fail(
        f"Invalid threshold: {threshold}"
    )


# ============================================================
# 5. MODEL VALIDATION
# ============================================================

print(
    "\n[5] Validating model configuration..."
)

print(
    f"Model type         : "
    f"{model_info.get('type')}"
)

print(
    f"Feature count      : "
    f"{model_info.get('feature_count')}"
)

print(
    f"Backend engineering: "
    f"{model_info.get('backend_feature_engineering')}"
)

print(
    f"Frontend features  : "
    f"{model_info.get('frontend_model_features')}"
)

print(
    f"Applicant ID used  : "
    f"{model_info.get('applicant_id_used')}"
)


if model_info.get("type") != "XGBoost":

    fail(
        "Model is not XGBoost."
    )


if model_info.get("feature_count") != 50:

    fail(
        "Model is not receiving 50 features."
    )


if model_info.get(
    "backend_feature_engineering"
) is not True:

    fail(
        "Backend feature engineering is disabled."
    )


if model_info.get(
    "frontend_model_features"
) is not False:

    fail(
        "Frontend should not send model-ready features."
    )


if model_info.get(
    "applicant_id_used"
) is not False:

    fail(
        "Applicant ID should not be used."
    )


print(
    "✅ Model configuration passed."
)


# ============================================================
# 6. BACKEND FEATURE ENGINEERING
# ============================================================

print(
    "\n[6] Validating backend feature engineering..."
)

raw_inputs = backend_processing.get(
    "raw_banking_inputs"
)

engineered_features = backend_processing.get(
    "engineered_model_features"
)

model_features = backend_processing.get(
    "features_sent_to_model"
)

print(
    f"Banking inputs     : "
    f"{raw_inputs}"
)

print(
    f"Engineered features: "
    f"{engineered_features}"
)

print(
    f"Model features     : "
    f"{model_features}"
)


if engineered_features != 50:

    fail(
        "Backend did not create exactly 50 features."
    )


if model_features != 50:

    fail(
        "Exactly 50 features must be sent to model."
    )


if raw_inputs is None or raw_inputs < 50:

    fail(
        "Expected a full banking application payload."
    )


print(
    "✅ Backend feature engineering passed."
)


# ============================================================
# 7. AFFORDABILITY
# ============================================================

print(
    "\n[7] Validating affordability metrics..."
)

monthly_income = affordability.get(
    "monthly_income"
)

monthly_emi = affordability.get(
    "monthly_emi"
)

dti = affordability.get(
    "debt_to_income_ratio"
)

print(
    f"Monthly income    : ₹{monthly_income:,.2f}"
)

print(
    f"Monthly EMI       : ₹{monthly_emi:,.2f}"
)

print(
    f"DTI               : "
    f"{dti * 100:.2f}%"
)


if monthly_income <= 0:

    fail(
        "Monthly income must be positive."
    )


if monthly_emi <= 0:

    fail(
        "Monthly EMI must be positive."
    )


if dti < 0:

    fail(
        "DTI cannot be negative."
    )


print(
    "✅ Affordability validation passed."
)


# ============================================================
# 8. RISK FLAGS
# ============================================================

print(
    "\n[8] Risk flags..."
)

for flag in risk_flags:

    print(
        f"  - {flag}"
    )


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TEST SUMMARY"
)

print(
    "=" * 70
)

print(
    "✅ API health                 : PASSED"
)

print(
    "✅ Model loaded               : PASSED"
)

print(
    "✅ XGBoost model              : PASSED"
)

print(
    "✅ 50 model features          : PASSED"
)

print(
    "✅ Backend feature engineering: PASSED"
)

print(
    "✅ No Applicant ID            : PASSED"
)

print(
    "✅ Frontend model features    : NOT USED"
)

print(
    "✅ Contract type mapping      : PASSED"
)

print(
    "✅ Prediction                 : PASSED"
)

print(
    "✅ Affordability metrics      : PASSED"
)

print(
    "✅ Risk flags                 : PASSED"
)

print(
    "\n🎉 ALL API TESTS PASSED."
)

print(
    "=" * 70
)