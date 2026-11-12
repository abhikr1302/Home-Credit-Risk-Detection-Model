from __future__ import annotations

from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import MODEL_PATH, CLASSIFICATION_THRESHOLD
from src.model_contract import PRODUCTION_FEATURES


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Home Credit Risk Detection API",
    description=(
        "Credit risk prediction API using the finalized "
        "30-feature XGBoost Home Credit model."
    ),
    version="3.0.0",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

THRESHOLD = CLASSIFICATION_THRESHOLD

EXPECTED_FEATURES = PRODUCTION_FEATURES.copy()


# ============================================================
# LOAD MODEL
# ============================================================

model = None
model_load_error = None

try:
    model = joblib.load(MODEL_PATH)
    if list(model.feature_names_in_) != EXPECTED_FEATURES:
        raise ValueError("Saved model does not match the production feature contract. Run python -m src.model_training.")
except Exception as exc:
    model = None
    model_load_error = str(exc)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
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


def clean_numeric(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def normalize_text(
    value: Any,
    default: str = "Unknown",
) -> str:
    if value is None:
        return default

    value = str(value).strip()

    return value if value else default


def normalize_gender(value: Any) -> str:
    value = normalize_text(value, "M").upper()

    mapping = {
        "MALE": "M",
        "M": "M",
        "FEMALE": "F",
        "F": "F",
    }

    return mapping.get(value, "XNA")


def normalize_car_ownership(value: Any) -> str:
    if isinstance(value, bool):
        return "Y" if value else "N"

    value = str(value).strip().upper()

    return "Y" if value in {"Y", "YES", "TRUE", "1", "OWNED"} else "N"


def normalize_property_ownership(value: Any) -> str:
    if isinstance(value, bool):
        return "Y" if value else "N"

    value = str(value).strip().upper()

    return "Y" if value in {"Y", "YES", "TRUE", "1", "OWNED", "HOUSE / APARTMENT"} else "N"


def calculate_risk_level(probability: float) -> str:
    if probability < min(0.30, THRESHOLD):
        return "LOW"

    if probability < THRESHOLD:
        return "MEDIUM"

    return "HIGH"


def calculate_decision(probability: float) -> str:
    if probability >= THRESHOLD:
        return "HIGH RISK"

    return "LOWER RISK"


# ============================================================
# REQUEST MODEL
# ============================================================

class CreditApplication(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.active_credit_accounts > self.total_credit_accounts:
            raise ValueError("Active credit accounts cannot exceed total credit accounts.")
        if self.previous_approved + self.previous_refused > self.previous_applications:
            raise ValueError("Approved and refused counts cannot exceed previous applications.")
        if self.credit_card_balance > 0 and self.credit_card_limit == 0:
            raise ValueError("A positive card balance requires a positive credit limit.")
        return self


    # --------------------------------------------------------
    # Personal
    # --------------------------------------------------------

    age: int = Field(
        ...,
        ge=18,
        le=100,
    )

    gender: str = "M"

    marital_status: str = "Married"

    education: str = "Secondary / secondary special"

    housing_type: str = "House / apartment"

    type_suite: str = "Unaccompanied"

    owns_car: bool = False

    owns_property: bool = True

    # --------------------------------------------------------
    # Employment
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

    # --------------------------------------------------------
    # Loan
    # --------------------------------------------------------

    contract_type: str = "Cash loans"

    loan_amount: float = Field(
        ...,
        gt=0,
    )

    monthly_emi: float = Field(
        ...,
        gt=0,
    )

    # --------------------------------------------------------
    # External Risk Scores
    # --------------------------------------------------------

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
    # Region Risk
    # --------------------------------------------------------

    region_rating_client: int = Field(
        default=2,
        ge=1,
        le=3,
    )

    region_rating_client_w_city: int = Field(
        default=2,
        ge=1,
        le=3,
    )

    # --------------------------------------------------------
    # Bureau
    # --------------------------------------------------------

    total_credit_accounts: int = Field(
        default=0,
        ge=0,
    )

    active_credit_accounts: int = Field(
        default=0,
        ge=0,
    )

    total_outstanding_debt: float = Field(
        default=0.0,
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

    # --------------------------------------------------------
    # POS / CASH
    # --------------------------------------------------------

    pos_cash_loans: int = Field(
        default=0,
        ge=0,
    )

    # --------------------------------------------------------
    # Social Circle
    # --------------------------------------------------------

    social_circle_30plus_defaults: int = Field(
        default=0,
        ge=0,
    )

    social_circle_60plus_defaults: int = Field(
        default=0,
        ge=0,
    )

    # --------------------------------------------------------
    # Document Risk
    # --------------------------------------------------------

    document_risk_flag: int = Field(
        default=0,
        ge=0,
        le=1,
    )


# ============================================================
# BACKEND FEATURE ENGINEERING
# ============================================================

def engineer_features(
    application: CreditApplication,
) -> pd.DataFrame:

    data = application.model_dump()

    # --------------------------------------------------------
    # Basic values
    # --------------------------------------------------------

    age = clean_numeric(data["age"])

    employment_years = clean_numeric(
        data["employment_years"]
    )

    annual_income = clean_numeric(
        data["annual_income"]
    )

    loan_amount = clean_numeric(
        data["loan_amount"]
    )

    monthly_emi = clean_numeric(
        data["monthly_emi"]
    )

    # --------------------------------------------------------
    # External scores
    # --------------------------------------------------------

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

    ext_source_mean = float(
        np.mean(
            [
                ext_source_1,
                ext_source_2,
                ext_source_3,
            ]
        )
    )

    # --------------------------------------------------------
    # Previous applications
    # --------------------------------------------------------

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
        default=np.nan,
    )

    previous_refusal_rate = safe_divide(
        previous_refused,
        previous_applications,
        default=np.nan,
    )

    # --------------------------------------------------------
    # Credit card utilization
    # --------------------------------------------------------

    credit_card_balance = clean_numeric(
        data["credit_card_balance"]
    )

    credit_card_limit = clean_numeric(
        data["credit_card_limit"]
    )

    credit_card_avg_utilization = safe_divide(
        credit_card_balance,
        credit_card_limit,
        default=np.nan,
    )

    credit_card_avg_utilization = min(
        max(credit_card_avg_utilization, 0.0),
        5.0,
    )

    # --------------------------------------------------------
    # Bureau
    # --------------------------------------------------------

    total_credit_accounts = clean_numeric(
        data["total_credit_accounts"]
    )

    active_credit_accounts = clean_numeric(
        data["active_credit_accounts"]
    )

    total_outstanding_debt = clean_numeric(
        data["total_outstanding_debt"]
    )

    active_bureau_account_ratio = safe_divide(
        active_credit_accounts,
        total_credit_accounts,
        default=np.nan,
    )

    bureau_debt_to_annual_income_ratio = safe_divide(
        total_outstanding_debt,
        annual_income,
    )

    # --------------------------------------------------------
    # Model features
    # --------------------------------------------------------

    features: Dict[str, Any] = {

        # External risk
        "ext_source_mean": ext_source_mean,
        "ext_source_1": ext_source_1,
        "ext_source_2": ext_source_2,
        "ext_source_3": ext_source_3,

        # Personal
        "name_education_type": normalize_text(
            data["education"]
        ),

        "code_gender": normalize_gender(
            data["gender"]
        ),

        "name_family_status": normalize_text(
            data["marital_status"]
        ),

        "name_housing_type": normalize_text(
            data["housing_type"]
        ),

        "name_type_suite": normalize_text(
            data["type_suite"]
        ),

        "flag_own_car": normalize_car_ownership(
            data["owns_car"]
        ),

        "flag_own_realty": normalize_property_ownership(
            data["owns_property"]
        ),

        "age_years": age,

        "days_birth": -(age * 365.25),

        # Employment
        "name_income_type": normalize_text(
            data["income_type"]
        ),

        "employment_years": employment_years,

        # Contract / loan
        "name_contract_type": normalize_text(
            data["contract_type"]
        ),

        "amt_credit": loan_amount,

        "amt_annuity": monthly_emi,

        # Credit card
        "credit_card_avg_utilization":
            credit_card_avg_utilization,

        # Region
        "region_rating_client":
            clean_numeric(
                data["region_rating_client"]
            ),

        "region_rating_client_w_city":
            clean_numeric(
                data["region_rating_client_w_city"]
            ),

        # Previous applications
        "previous_refusal_rate":
            previous_refusal_rate,

        "previous_approval_rate":
            previous_approval_rate,

        "previous_refused_count":
            previous_refused,

        # POS / CASH
        "pos_cash_loan_count":
            clean_numeric(
                data["pos_cash_loans"]
            ),

        # Bureau
        "active_bureau_account_ratio":
            active_bureau_account_ratio,

        "bureau_debt_to_annual_income_ratio":
            bureau_debt_to_annual_income_ratio,

        # Social circle
        "def_30_cnt_social_circle":
            clean_numeric(
                data["social_circle_30plus_defaults"]
            ),

        "def_60_cnt_social_circle":
            clean_numeric(
                data["social_circle_60plus_defaults"]
            ),

        # Document
        "flag_document_3":
            clean_numeric(
                data["document_risk_flag"]
            ),
    }

    # --------------------------------------------------------
    # Validate exact model contract
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    features = {
        feature: features[feature]
        for feature in EXPECTED_FEATURES
    }

    df = pd.DataFrame(
        [features],
        columns=EXPECTED_FEATURES,
    )

    # Replace invalid numerical values.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    # Preserve undefined ratios as NaN for the fitted training-time imputer.

    categorical_columns = df.select_dtypes(
        exclude=[np.number]
    ).columns

    for column in categorical_columns:
        df[column] = (
            df[column].fillna("Unknown")
        )

    if list(df.columns) != EXPECTED_FEATURES:
        raise ValueError(
            "Final feature order does not match "
            "the trained model."
        )

    return df


# ============================================================
# RISK FLAGS
# ============================================================

def calculate_risk_flags(
    data: Dict[str, Any],
) -> List[str]:

    flags: List[str] = []

    annual_income = clean_numeric(
        data["annual_income"]
    )

    total_debt = clean_numeric(
        data["total_outstanding_debt"]
    )

    previous_refused = clean_numeric(
        data["previous_refused"]
    )

    credit_card_balance = clean_numeric(
        data["credit_card_balance"]
    )

    credit_card_limit = clean_numeric(
        data["credit_card_limit"]
    )

    social_30 = clean_numeric(
        data["social_circle_30plus_defaults"]
    )

    social_60 = clean_numeric(
        data["social_circle_60plus_defaults"]
    )

    debt_to_income = safe_divide(
        total_debt,
        annual_income,
    )

    card_utilization = safe_divide(
        credit_card_balance,
        credit_card_limit,
    )

    if debt_to_income > 0.50:
        flags.append(
            "High existing debt relative to income"
        )

    if card_utilization >= 0.80:
        flags.append(
            "High credit card utilization"
        )

    if previous_refused > 0:
        flags.append(
            "Previous loan applications were refused"
        )

    if social_30 > 0:
        flags.append(
            "30+ day social-circle defaults reported"
        )

    if social_60 > 0:
        flags.append(
            "60+ day social-circle defaults reported"
        )

    if not flags:
        flags.append(
            "No major rule-based risk flags detected"
        )

    return flags


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health() -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model": "XGBoost",
        "feature_count": len(EXPECTED_FEATURES),
        "threshold": THRESHOLD,
        "applicant_id_used": False,
        "backend_feature_engineering": True,
        "model_path": str(MODEL_PATH),
        "model_load_error": model_load_error,
    }


# ============================================================
# MODEL INFORMATION
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
            "frontend": "Real-world banking inputs",
            "backend": "Finalized feature engineering",
            "model_input": "30 selected trained features",
            "prediction": "XGBoost",
        },
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    application: CreditApplication,
) -> Dict[str, Any]:

    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Model is not loaded.",
                "error": model_load_error,
            },
        )

    try:

        application_data = application.model_dump()

        # ----------------------------------------------------
        # Feature engineering
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
                    "error": str(exc),
                },
            )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        probability = float(
            model.predict_proba(
                feature_df
            )[0, 1]
        )

        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability.")

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
        # Affordability
        # ----------------------------------------------------

        annual_income = clean_numeric(
            application_data["annual_income"]
        )

        monthly_income = (
            annual_income / 12.0
        )

        monthly_emi = clean_numeric(
            application_data["monthly_emi"]
        )

        total_debt = clean_numeric(
            application_data[
                "total_outstanding_debt"
            ]
        )

        existing_monthly_debt_estimate = (
            total_debt / 60.0
            if total_debt > 0
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

        loan_to_income = safe_divide(
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
                    loan_to_income,
                    4,
                ),
            },

            "risk_flags": calculate_risk_flags(
                application_data
            ),

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
        "version": "3.0.0",
        "status": "running",
        "architecture": (
            "Banking Inputs -> Backend Feature Engineering "
            "-> 30 Model Features -> XGBoost -> Risk Decision"
        ),
    }
