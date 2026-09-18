from __future__ import annotations

from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_PATH


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Home Credit Risk Detection API",
    description=(
        "Banking-style credit risk prediction API. "
        "The frontend accepts real-world banking inputs while "
        "the backend engineers the features required by the "
        "trained XGBoost Home Credit model."
    ),
    version="2.0.0",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

THRESHOLD = 0.60

EXPECTED_FEATURES: List[str] = [
    "ext_source_mean",
    "name_education_type",
    "name_income_type",
    "code_gender",
    "name_family_status",
    "flag_own_car",
    "name_housing_type",
    "ext_source_3",
    "name_type_suite",
    "ext_source_2",
    "name_contract_type",
    "installment_late_payment_rate",
    "previous_refusal_rate",
    "flag_document_3",
    "region_rating_client",
    "credit_card_avg_utilization",
    "previous_avg_credit_application_ratio",
    "pos_cash_loan_count",
    "age_years",
    "credit_card_balance_to_monthly_income_ratio",
    "credit_card_max_utilization",
    "employment_years",
    "active_bureau_account_ratio",
    "flag_own_realty",
    "ext_source_1",
    "def_60_cnt_social_circle",
    "def_30_cnt_social_circle",
    "installment_avg_payment_ratio",
    "pos_cash_active_count",
    "amt_credit",
    "credit_card_avg_balance",
    "previous_approval_rate",
    "region_rating_client_w_city",
    "days_birth",
    "bureau_active_accounts",
    "amt_annuity",
    "previous_avg_annuity",
    "pos_cash_history_months",
    "bureau_debt_to_annual_income_ratio",
    "bureau_avg_days_credit",
    "installment_previous_loans",
    "installment_max_payment_delay",
    "loan_annuity_to_monthly_income_ratio",
    "installment_total_records",
    "days_employed",
    "bureau_total_credit",
    "bureau_total_overdue",
    "annuity_to_income_ratio",
    "pos_cash_max_dpd",
    "credit_card_dpd_30_count",
]


# ============================================================
# LOAD MODEL
# ============================================================

model = None
model_load_error = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    model_load_error = str(exc)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """
    Safely divide two numbers.

    Returns default when denominator is zero,
    missing, invalid, or the result is not finite.
    """
    try:
        numerator = float(numerator)
        denominator = float(denominator)

        if not np.isfinite(numerator):
            return default

        if not np.isfinite(denominator) or denominator == 0:
            return default

        result = numerator / denominator

        if not np.isfinite(result):
            return default

        return float(result)

    except (TypeError, ValueError):
        return default


def clean_numeric(value: Any, default: float = 0.0) -> float:
    """
    Convert input to a finite float.
    """
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def normalize_text(value: Any, default: str = "Unknown") -> str:
    """
    Normalize categorical input.
    """
    if value is None:
        return default

    value = str(value).strip()

    return value if value else default


def calculate_risk_level(probability: float) -> str:
    """
    Convert probability into business-friendly risk band.
    """
    if probability < 0.30:
        return "LOW"

    if probability < THRESHOLD:
        return "MEDIUM"

    return "HIGH"


def calculate_decision(probability: float) -> str:
    """
    Business decision based on configured threshold.
    """
    if probability >= THRESHOLD:
        return "HIGH RISK"

    return "LOWER RISK"


def calculate_risk_flags(data: Dict[str, Any]) -> List[str]:
    """
    Generate explainable business-level risk flags.
    """

    flags: List[str] = []

    credit_score = clean_numeric(data.get("credit_score"))
    total_debt = clean_numeric(data.get("total_outstanding_debt"))
    annual_income = clean_numeric(data.get("annual_income"))
    overdue = clean_numeric(data.get("total_overdue_amount"))
    max_dpd = clean_numeric(data.get("maximum_days_past_due"))
    delinquent_accounts = clean_numeric(
        data.get("delinquent_accounts")
    )
    credit_card_utilization = clean_numeric(
        data.get("credit_card_max_utilization")
    )
    late_installment_rate = clean_numeric(
        data.get("late_installment_rate")
    )
    recent_enquiries = clean_numeric(
        data.get("recent_credit_enquiries")
    )
    previous_refused = clean_numeric(
        data.get("previous_refused")
    )
    fraud_flag = bool(data.get("fraud_flag", False))
    kyc_verified = bool(data.get("kyc_verified", True))

    debt_to_income = safe_divide(
        total_debt,
        annual_income,
    )

    if credit_score > 0 and credit_score < 600:
        flags.append("Low credit score")

    if debt_to_income > 0.50:
        flags.append("High existing debt relative to income")

    if overdue > 0:
        flags.append("Existing overdue amount")

    if max_dpd >= 30:
        flags.append("Previous serious payment delinquency")

    if delinquent_accounts > 0:
        flags.append("Existing delinquent accounts")

    if credit_card_utilization >= 80:
        flags.append("High credit card utilization")

    if late_installment_rate >= 0.20:
        flags.append("High installment late-payment rate")

    if recent_enquiries >= 5:
        flags.append("High number of recent credit enquiries")

    if previous_refused > 0:
        flags.append("Previous loan applications were refused")

    if fraud_flag:
        flags.append("Fraud risk flag present")

    if not kyc_verified:
        flags.append("KYC verification incomplete")

    if not flags:
        flags.append("No major rule-based risk flags detected")

    return flags


# ============================================================
# REQUEST MODEL
# ============================================================

class CreditApplication(BaseModel):
    # --------------------------------------------------------
    # Personal & KYC
    # --------------------------------------------------------

    age: int = Field(..., ge=18, le=100)

    gender: str = "Male"

    marital_status: str = "Married"

    family_members: int = Field(
        default=1,
        ge=1,
        le=30,
    )

    education: str = "Higher education"

    housing_type: str = "House / apartment"

    residential_status: str = "Owned"

    owns_car: bool = False

    owns_property: bool = True

    contract_type: str = "Cash loans"

    # --------------------------------------------------------
    # Employment & Income
    # --------------------------------------------------------

    income_type: str = "Working"

    employment_years: float = Field(
        default=5.0,
        ge=0,
        le=60,
    )

    annual_income: float = Field(
        ...,
        gt=0,
    )

    other_monthly_income: float = Field(
        default=0.0,
        ge=0,
    )

    income_stability_years: float = Field(
        default=3.0,
        ge=0,
        le=60,
    )

    # --------------------------------------------------------
    # Loan Application
    # --------------------------------------------------------

    loan_type: str = "Personal Loan"

    loan_purpose: str = "Other"

    secured_loan: bool = False

    loan_amount: float = Field(
        ...,
        gt=0,
    )

    loan_term_months: int = Field(
        default=60,
        ge=1,
        le=480,
    )

    monthly_emi: float = Field(
        ...,
        gt=0,
    )

    # --------------------------------------------------------
    # Credit Score / External Risk
    # --------------------------------------------------------

    credit_score: float = Field(
        default=700,
        ge=0,
        le=900,
    )

    external_score_1: float = Field(
        default=0.5,
        ge=0,
        le=1,
    )

    external_score_2: float = Field(
        default=0.5,
        ge=0,
        le=1,
    )

    external_score_3: float = Field(
        default=0.5,
        ge=0,
        le=1,
    )

    # --------------------------------------------------------
    # Bureau / Existing Credit
    # --------------------------------------------------------

    total_credit_accounts: int = Field(
        default=0,
        ge=0,
    )

    active_credit_accounts: int = Field(
        default=0,
        ge=0,
    )

    closed_credit_accounts: int = Field(
        default=0,
        ge=0,
    )

    total_credit_limit: float = Field(
        default=0.0,
        ge=0,
    )

    total_outstanding_debt: float = Field(
        default=0.0,
        ge=0,
    )

    total_overdue_amount: float = Field(
        default=0.0,
        ge=0,
    )

    average_credit_history_months: float = Field(
        default=0.0,
        ge=0,
    )

    maximum_days_past_due: float = Field(
        default=0.0,
        ge=0,
    )

    delinquent_accounts: int = Field(
        default=0,
        ge=0,
    )

    # --------------------------------------------------------
    # Credit Card
    # --------------------------------------------------------

    number_of_credit_cards: int = Field(
        default=0,
        ge=0,
    )

    credit_card_balance: float = Field(
        default=0.0,
        ge=0,
    )

    credit_card_limit: float = Field(
        default=0.0,
        ge=0,
    )

    credit_card_30plus_dpd: int = Field(
        default=0,
        ge=0,
    )

    credit_card_max_utilization: float = Field(
        default=0.0,
        ge=0,
    )

    credit_card_average_balance: float = Field(
        default=0.0,
        ge=0,
    )

    # --------------------------------------------------------
    # Previous Applications
    # --------------------------------------------------------

    previous_applications: int = Field(
        default=0,
        ge=0,
    )

    previous_approved: int = Field(
        default=0,
        ge=0,
    )

    previous_refused: int = Field(
        default=0,
        ge=0,
    )

    previous_average_loan_amount: float = Field(
        default=0.0,
        ge=0,
    )

    previous_average_emi: float = Field(
        default=0.0,
        ge=0,
    )

    previous_average_credit_application_ratio: float = Field(
        default=0.0,
        ge=0,
    )

    # --------------------------------------------------------
    # Installment Repayment
    # --------------------------------------------------------

    previous_loans_with_installments: int = Field(
        default=0,
        ge=0,
    )

    installment_records: int = Field(
        default=0,
        ge=0,
    )

    late_installment_rate: float = Field(
        default=0.0,
        ge=0,
        le=1,
    )

    maximum_installment_delay: float = Field(
        default=0.0,
        ge=0,
    )

    average_payment_ratio: float = Field(
        default=1.0,
        ge=0,
    )

    # --------------------------------------------------------
    # POS / CASH
    # --------------------------------------------------------

    pos_cash_loans: int = Field(
        default=0,
        ge=0,
    )

    pos_cash_active_loans: int = Field(
        default=0,
        ge=0,
    )

    pos_cash_history_months: int = Field(
        default=0,
        ge=0,
    )

    pos_cash_max_dpd: float = Field(
        default=0.0,
        ge=0,
    )

    # --------------------------------------------------------
    # Additional Risk Signals
    # --------------------------------------------------------

    social_circle_30plus_defaults: int = Field(
        default=0,
        ge=0,
    )

    social_circle_60plus_defaults: int = Field(
        default=0,
        ge=0,
    )

    document_risk_flag: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    recent_credit_enquiries: int = Field(
        default=0,
        ge=0,
    )

    fraud_flag: bool = False

    kyc_verified: bool = True


# ============================================================
# BACKEND FEATURE ENGINEERING
# ============================================================

def engineer_features(
    application: CreditApplication,
) -> pd.DataFrame:
    """
    Convert real-world banking application inputs into
    the exact 50 features expected by the trained model.

    Applicant ID is intentionally NOT used.
    """

    data = application.model_dump()

    # ========================================================
    # BASIC CLEANING
    # ========================================================

    age = clean_numeric(data["age"], 18)

    employment_years = clean_numeric(
        data["employment_years"]
    )

    annual_income = clean_numeric(
        data["annual_income"]
    )

    other_monthly_income = clean_numeric(
        data["other_monthly_income"]
    )

    loan_amount = clean_numeric(
        data["loan_amount"]
    )

    monthly_emi = clean_numeric(
        data["monthly_emi"]
    )

    total_monthly_income = (
        annual_income / 12.0
    ) + other_monthly_income

    # ========================================================
    # EXTERNAL SCORES
    # ========================================================

    ext_source_1 = clean_numeric(
        data["external_score_1"],
        0.5,
    )

    ext_source_2 = clean_numeric(
        data["external_score_2"],
        0.5,
    )

    ext_source_3 = clean_numeric(
        data["external_score_3"],
        0.5,
    )

    ext_source_mean = np.mean(
        [
            ext_source_1,
            ext_source_2,
            ext_source_3,
        ]
    )

    # ========================================================
    # PREVIOUS APPLICATIONS
    # ========================================================

    previous_applications = clean_numeric(
        data["previous_applications"]
    )

    previous_approved = clean_numeric(
        data["previous_approved"]
    )

    previous_refused = clean_numeric(
        data["previous_refused"]
    )

    previous_approval_rate = safe_divide(
        previous_approved,
        previous_applications,
    )

    previous_refusal_rate = safe_divide(
        previous_refused,
        previous_applications,
    )

    # ========================================================
    # CREDIT CARD
    # ========================================================

    credit_card_balance = clean_numeric(
        data["credit_card_balance"]
    )

    credit_card_limit = clean_numeric(
        data["credit_card_limit"]
    )

    credit_card_utilization = safe_divide(
        credit_card_balance,
        credit_card_limit,
    )

    credit_card_max_utilization = clean_numeric(
        data["credit_card_max_utilization"]
    )

    # Convert percentage-style utilization to ratio
    if credit_card_max_utilization > 1:
        credit_card_max_utilization /= 100.0

    credit_card_max_utilization = min(
        max(credit_card_max_utilization, 0.0),
        1.0,
    )

    # ========================================================
    # BUREAU
    # ========================================================

    total_credit_accounts = clean_numeric(
        data["total_credit_accounts"]
    )

    active_credit_accounts = clean_numeric(
        data["active_credit_accounts"]
    )

    closed_credit_accounts = clean_numeric(
        data["closed_credit_accounts"]
    )

    total_credit_limit = clean_numeric(
        data["total_credit_limit"]
    )

    total_outstanding_debt = clean_numeric(
        data["total_outstanding_debt"]
    )

    total_overdue_amount = clean_numeric(
        data["total_overdue_amount"]
    )

    active_bureau_account_ratio = safe_divide(
        active_credit_accounts,
        total_credit_accounts,
    )

    bureau_debt_to_annual_income_ratio = safe_divide(
        total_outstanding_debt,
        annual_income,
    )

    bureau_credit_to_annual_income_ratio = safe_divide(
        total_credit_limit,
        annual_income,
    )

    bureau_overdue_to_annual_income_ratio = safe_divide(
        total_overdue_amount,
        annual_income,
    )

    # ========================================================
    # INSTALLMENTS
    # ========================================================

    late_installment_rate = clean_numeric(
        data["late_installment_rate"]
    )

    late_installment_rate = min(
        max(late_installment_rate, 0.0),
        1.0,
    )

    average_payment_ratio = clean_numeric(
        data["average_payment_ratio"],
        1.0,
    )

    maximum_installment_delay = clean_numeric(
        data["maximum_installment_delay"]
    )

    installment_records = clean_numeric(
        data["installment_records"]
    )

    previous_loans_with_installments = clean_numeric(
        data["previous_loans_with_installments"]
    )

    # ========================================================
    # POS / CASH
    # ========================================================

    pos_cash_loans = clean_numeric(
        data["pos_cash_loans"]
    )

    pos_cash_active_loans = clean_numeric(
        data["pos_cash_active_loans"]
    )

    pos_cash_history_months = clean_numeric(
        data["pos_cash_history_months"]
    )

    pos_cash_max_dpd = clean_numeric(
        data["pos_cash_max_dpd"]
    )

    # ========================================================
    # BANK-STYLE AFFORDABILITY FEATURES
    # ========================================================

    annuity_to_income_ratio = safe_divide(
        monthly_emi,
        total_monthly_income,
    )

    loan_annuity_to_monthly_income_ratio = safe_divide(
        monthly_emi,
        total_monthly_income,
    )

    credit_card_balance_to_monthly_income_ratio = safe_divide(
        credit_card_balance,
        total_monthly_income,
    )

    # ========================================================
    # REGION PROXY
    # ========================================================

    # The original Home Credit model contains
    # region_rating_client and region_rating_client_w_city.
    #
    # Since the new banking UI does not ask for geographic
    # risk ratings, we derive a conservative proxy from
    # external credit signals.
    #
    # This is a compatibility mapping, not a genuine
    # geographic risk score.

    region_rating_client = int(
        round(
            max(
                1,
                min(
                    3,
                    3 - (ext_source_mean * 2),
                ),
            )
        )
    )

    region_rating_client_w_city = region_rating_client

    # ========================================================
    # MODEL FEATURE DICTIONARY
    # ========================================================

    features: Dict[str, Any] = {

        # ----------------------------------------------------
        # External Risk
        # ----------------------------------------------------

        "ext_source_mean": ext_source_mean,

        "ext_source_1": ext_source_1,

        "ext_source_2": ext_source_2,

        "ext_source_3": ext_source_3,

        # ----------------------------------------------------
        # Personal
        # ----------------------------------------------------

        "name_education_type":
            normalize_text(
                data["education"]
            ),

        "code_gender":
            normalize_text(
                data["gender"]
            ),

        "name_family_status":
            normalize_text(
                data["marital_status"]
            ),

        "name_housing_type":
            normalize_text(
                data["housing_type"]
            ),

        "name_type_suite":
            normalize_text(
                data["residential_status"]
            ),

        "flag_own_car":
            int(bool(data["owns_car"])),

        "flag_own_realty":
            int(bool(data["owns_property"])),

        "age_years":
            age,

        "days_birth":
            -(age * 365.25),

        # ----------------------------------------------------
        # Employment
        # ----------------------------------------------------

        "name_income_type":
            normalize_text(
                data["income_type"]
            ),

        "employment_years":
            employment_years,

        "days_employed":
            -(employment_years * 365.25),

        # ----------------------------------------------------
        # Loan
        # ----------------------------------------------------

        "name_contract_type":
            normalize_text(
                data["contract_type"]
            ),

        "amt_credit":
            loan_amount,

        "amt_annuity":
            monthly_emi,

        "annuity_to_income_ratio":
            annuity_to_income_ratio,

        "loan_annuity_to_monthly_income_ratio":
            loan_annuity_to_monthly_income_ratio,

        # ----------------------------------------------------
        # Installment Behaviour
        # ----------------------------------------------------

        "installment_late_payment_rate":
            late_installment_rate,

        "installment_avg_payment_ratio":
            average_payment_ratio,

        "installment_previous_loans":
            previous_loans_with_installments,

        "installment_max_payment_delay":
            maximum_installment_delay,

        "installment_total_records":
            installment_records,

        # ----------------------------------------------------
        # Previous Applications
        # ----------------------------------------------------

        "previous_refusal_rate":
            previous_refusal_rate,

        "previous_approval_rate":
            previous_approval_rate,

        "previous_avg_credit_application_ratio":
            clean_numeric(
                data[
                    "previous_average_credit_application_ratio"
                ]
            ),

        "previous_avg_annuity":
            clean_numeric(
                data["previous_average_emi"]
            ),

        # ----------------------------------------------------
        # Credit Card
        # ----------------------------------------------------

        "credit_card_avg_utilization":
            credit_card_utilization,

        "credit_card_max_utilization":
            credit_card_max_utilization,

        "credit_card_balance_to_monthly_income_ratio":
            credit_card_balance_to_monthly_income_ratio,

        "credit_card_avg_balance":
            clean_numeric(
                data["credit_card_average_balance"]
            ),

        "credit_card_dpd_30_count":
            clean_numeric(
                data["credit_card_30plus_dpd"]
            ),

        # ----------------------------------------------------
        # Bureau
        # ----------------------------------------------------

        "bureau_total_accounts":
            total_credit_accounts,

        "bureau_active_accounts":
            active_credit_accounts,

        "bureau_closed_accounts":
            closed_credit_accounts,

        "bureau_total_credit":
            total_credit_limit,

        "bureau_total_overdue":
            total_overdue_amount,

        "bureau_avg_days_credit":
            clean_numeric(
                data["average_credit_history_months"]
            ) * 30.4375,

        "bureau_debt_to_annual_income_ratio":
            bureau_debt_to_annual_income_ratio,

        "active_bureau_account_ratio":
            active_bureau_account_ratio,

        # ----------------------------------------------------
        # POS / CASH
        # ----------------------------------------------------

        "pos_cash_loan_count":
            pos_cash_loans,

        "pos_cash_active_count":
            pos_cash_active_loans,

        "pos_cash_history_months":
            pos_cash_history_months,

        "pos_cash_max_dpd":
            pos_cash_max_dpd,

        # ----------------------------------------------------
        # Social Circle
        # ----------------------------------------------------

        "def_30_cnt_social_circle":
            clean_numeric(
                data["social_circle_30plus_defaults"]
            ),

        "def_60_cnt_social_circle":
            clean_numeric(
                data["social_circle_60plus_defaults"]
            ),

        # ----------------------------------------------------
        # Document Risk
        # ----------------------------------------------------

        "flag_document_3":
            int(data["document_risk_flag"]),

        # ----------------------------------------------------
        # Region
        # ----------------------------------------------------

        "region_rating_client":
            region_rating_client,

        "region_rating_client_w_city":
            region_rating_client_w_city,
    }

    # ========================================================
    # FINAL FEATURE VALIDATION
    # ========================================================

    missing_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    extra_features = [
        feature
        for feature in features
        if feature not in EXPECTED_FEATURES
    ]

    # Keep ONLY the exact model features.
    features = {
        feature: features[feature]
        for feature in EXPECTED_FEATURES
    }

    df = pd.DataFrame(
        [features],
        columns=EXPECTED_FEATURES,
    )

    # Ensure numerical infinities are removed.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Numeric columns
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Categorical columns
    categorical_columns = df.select_dtypes(
        exclude=[np.number]
    ).columns

    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown")

    # Final validation
    if list(df.columns) != EXPECTED_FEATURES:
        raise ValueError(
            "Final feature order does not match the trained model."
        )

    return df


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health() -> Dict[str, Any]:

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model": "XGBoost",
        "feature_count": len(EXPECTED_FEATURES),
        "threshold": THRESHOLD,
        "applicant_id_used": False,
        "frontend_model_features_used": False,
        "backend_feature_engineering": True,
        "model_path": str(MODEL_PATH),
        "model_load_error": model_load_error,
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info() -> Dict[str, Any]:

    return {
        "model_loaded": model is not None,
        "model_type": "XGBoost",
        "feature_count": len(EXPECTED_FEATURES),
        "threshold": THRESHOLD,
        "features": EXPECTED_FEATURES,
        "architecture": {
            "frontend": "Important real-world banking inputs",
            "backend": "Complete feature engineering",
            "model_input": "Selected 50 trained features",
            "prediction": "XGBoost",
        },
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(application: CreditApplication) -> Dict[str, Any]:

    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Model is not loaded.",
                "error": model_load_error,
            },
        )

    try:

        # ----------------------------------------------------
        # Convert request
        # ----------------------------------------------------

        application_data = application.model_dump()

        # ----------------------------------------------------
        # Backend Feature Engineering
        # ----------------------------------------------------

        try:

            feature_df = engineer_features(
                application
            )

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail={
                    "message":
                        "Backend feature engineering failed.",
                    "error":
                        str(exc),
                },
            )

        # ----------------------------------------------------
        # Model Prediction
        # ----------------------------------------------------

        probability = float(
            model.predict_proba(feature_df)[0, 1]
        )

        probability = max(
            0.0,
            min(
                1.0,
                probability,
            ),
        )

        risk_level = calculate_risk_level(
            probability
        )

        decision = calculate_decision(
            probability
        )

        # ----------------------------------------------------
        # Risk Flags
        # ----------------------------------------------------

        risk_flags = calculate_risk_flags(
            application_data
        )

        # ----------------------------------------------------
        # Affordability Metrics
        # ----------------------------------------------------

        annual_income = clean_numeric(
            application_data["annual_income"]
        )

        monthly_income = (
            annual_income / 12.0
        ) + clean_numeric(
            application_data[
                "other_monthly_income"
            ]
        )

        monthly_emi = clean_numeric(
            application_data["monthly_emi"]
        )

        existing_debt = clean_numeric(
            application_data[
                "total_outstanding_debt"
            ]
        )

        existing_monthly_debt_estimate = (
            existing_debt / 60.0
            if existing_debt > 0
            else 0.0
        )

        total_monthly_obligation = (
            monthly_emi
            + existing_monthly_debt_estimate
        )

        dti = safe_divide(
            total_monthly_obligation,
            monthly_income,
        )

        loan_to_annual_income = safe_divide(
            clean_numeric(
                application_data["loan_amount"]
            ),
            annual_income,
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "success": True,

            "risk_probability": round(
                probability,
                6,
            ),

            "risk_percentage": round(
                probability * 100,
                2,
            ),

            "risk_level": risk_level,

            "decision": decision,

            "threshold": THRESHOLD,

            "model": {
                "type": "XGBoost",
                "feature_count": len(
                    EXPECTED_FEATURES
                ),
                "backend_feature_engineering": True,
                "frontend_model_features": False,
                "applicant_id_used": False,
            },

            "affordability": {
                "monthly_income": round(
                    monthly_income,
                    2,
                ),

                "monthly_emi": round(
                    monthly_emi,
                    2,
                ),

                "estimated_existing_monthly_debt": round(
                    existing_monthly_debt_estimate,
                    2,
                ),

                "total_monthly_obligation": round(
                    total_monthly_obligation,
                    2,
                ),

                "debt_to_income_ratio": round(
                    dti,
                    4,
                ),

                "debt_to_income_percentage": round(
                    dti * 100,
                    2,
                ),

                "loan_to_annual_income_ratio": round(
                    loan_to_annual_income,
                    4,
                ),
            },

            "risk_flags": risk_flags,

            "backend_processing": {
                "raw_banking_inputs": len(
                    application_data
                ),
                "engineered_model_features": len(
                    EXPECTED_FEATURES
                ),
                "features_sent_to_model": len(
                    feature_df.columns
                ),
            },
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction failed.",
                "error": str(exc),
            },
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "application": "Home Credit Risk Detection",
        "api": "FastAPI",
        "model": "XGBoost",
        "version": "2.0.0",
        "status": "running",
        "architecture": (
            "Banking Inputs -> Backend Feature Engineering "
            "-> 50 Model Features -> XGBoost -> Risk Decision"
        ),
    }