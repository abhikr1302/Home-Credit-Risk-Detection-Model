"""Banking API. Start with: python -m uvicorn src.api:app --port 8000."""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import CLASSIFICATION_THRESHOLD, MODEL_PATH, PRODUCTION_FEATURES

app = FastAPI(title="Home Credit Risk Detection", version="4.0.0")
THRESHOLD = CLASSIFICATION_THRESHOLD
EXPECTED_FEATURES = PRODUCTION_FEATURES
model = None
model_load_error = None
try:
    model = joblib.load(MODEL_PATH)
    if list(model.feature_names_in_) != EXPECTED_FEATURES:
        raise ValueError("Saved model features differ. Run python -m src.train_model.")
except Exception as error:
    model = None
    model_load_error = str(error)


class CreditApplication(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    age: int = Field(ge=18, le=100)
    annual_income: float = Field(gt=0)
    loan_amount: float = Field(gt=0)
    monthly_emi: float = Field(gt=0)
    gender: str = "M"
    marital_status: str = "Married"
    education: str = "Secondary / secondary special"
    housing_type: str = "House / apartment"
    type_suite: str = "Unaccompanied"
    owns_car: bool = False
    owns_property: bool = True
    income_type: str = "Working"
    employment_years: float = Field(default=5, ge=0, le=60)
    contract_type: str = "Cash loans"
    external_score_1: float = Field(default=0.5, ge=0, le=1)
    external_score_2: float = Field(default=0.5, ge=0, le=1)
    external_score_3: float = Field(default=0.5, ge=0, le=1)
    region_rating_client: int = Field(default=2, ge=1, le=3)
    region_rating_client_w_city: int = Field(default=2, ge=1, le=3)
    total_credit_accounts: int = Field(default=0, ge=0)
    active_credit_accounts: int = Field(default=0, ge=0)
    total_outstanding_debt: float = Field(default=0, ge=0)
    number_of_credit_cards: int = Field(default=0, ge=0)
    credit_card_balance: float = Field(default=0, ge=0)
    credit_card_limit: float = Field(default=0, ge=0)
    previous_applications: int = Field(default=0, ge=0)
    previous_approved: int = Field(default=0, ge=0)
    previous_refused: int = Field(default=0, ge=0)
    pos_cash_loans: int = Field(default=0, ge=0)
    social_circle_30plus_defaults: int = Field(default=0, ge=0)
    social_circle_60plus_defaults: int = Field(default=0, ge=0)
    document_risk_flag: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.active_credit_accounts > self.total_credit_accounts:
            raise ValueError(
                "Active credit accounts cannot exceed total credit accounts."
            )
        if self.previous_approved + self.previous_refused > self.previous_applications:
            raise ValueError(
                "Approved and refused counts cannot exceed previous applications."
            )
        if self.credit_card_balance > 0 and self.credit_card_limit == 0:
            raise ValueError(
                "A positive card balance requires a positive credit limit."
            )
        return self


def safe_divide(numerator, denominator, default=0.0):
    return numerator / denominator if denominator else default


def engineer_features(application):
    data = application.model_dump()
    scores = [data[f"external_score_{i}"] for i in range(1, 4)]
    gender = {"MALE": "M", "M": "M", "FEMALE": "F", "F": "F"}.get(
        data["gender"].strip().upper(), "XNA"
    )
    features = {
        "ext_source_mean": float(np.mean(scores)),
        **{f"ext_source_{i}": score for i, score in enumerate(scores, 1)},
        "code_gender": gender,
        "flag_own_car": "Y" if data["owns_car"] else "N",
        "flag_own_realty": "Y" if data["owns_property"] else "N",
        "age_years": data["age"],
        "days_birth": -data["age"] * 365.25,
        "employment_years": data["employment_years"],
        "amt_credit": data["loan_amount"],
        "amt_annuity": data["monthly_emi"],
        "credit_card_avg_utilization": np.clip(
            safe_divide(data["credit_card_balance"], data["credit_card_limit"], np.nan),
            0,
            5,
        ),
        "previous_approval_rate": safe_divide(
            data["previous_approved"], data["previous_applications"], np.nan
        ),
        "previous_refusal_rate": safe_divide(
            data["previous_refused"], data["previous_applications"], np.nan
        ),
        "previous_refused_count": data["previous_refused"],
        "active_bureau_account_ratio": safe_divide(
            data["active_credit_accounts"], data["total_credit_accounts"], np.nan
        ),
        "bureau_debt_to_annual_income_ratio": data["total_outstanding_debt"]
        / data["annual_income"],
        "region_rating_client": data["region_rating_client"],
        "region_rating_client_w_city": data["region_rating_client_w_city"],
        "pos_cash_loan_count": data["pos_cash_loans"],
        "def_30_cnt_social_circle": data["social_circle_30plus_defaults"],
        "def_60_cnt_social_circle": data["social_circle_60plus_defaults"],
        "flag_document_3": data["document_risk_flag"],
    }
    for feature, field in {
        "name_education_type": "education",
        "name_family_status": "marital_status",
        "name_housing_type": "housing_type",
        "name_type_suite": "type_suite",
        "name_income_type": "income_type",
        "name_contract_type": "contract_type",
    }.items():
        features[feature] = data[field].strip() or "Unknown"
    return pd.DataFrame([features], columns=EXPECTED_FEATURES)


def calculate_risk_level(probability):
    if probability < min(0.3, THRESHOLD):
        return "LOW"
    return "MEDIUM" if probability < THRESHOLD else "HIGH"


def calculate_decision(probability):
    return "HIGH RISK" if probability >= THRESHOLD else "LOWER RISK"


def calculate_risk_flags(data):
    checks = [
        (
            data["total_outstanding_debt"] / data["annual_income"] > 0.5,
            "High existing debt relative to income",
        ),
        (
            safe_divide(data["credit_card_balance"], data["credit_card_limit"]) >= 0.8,
            "High credit card utilization",
        ),
        (data["previous_refused"] > 0, "Previous loan applications were refused"),
        (
            data["social_circle_30plus_defaults"] > 0,
            "30+ day social-circle defaults reported",
        ),
        (
            data["social_circle_60plus_defaults"] > 0,
            "60+ day social-circle defaults reported",
        ),
    ]
    return [message for condition, message in checks if condition] or [
        "No major rule-based risk flags detected"
    ]


@app.get("/")
def root():
    return {"application": "Home Credit Risk Detection", "documentation": "/docs"}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_count": len(EXPECTED_FEATURES),
        "threshold": THRESHOLD,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_loaded": model is not None,
        "model_type": "XGBoost",
        "feature_count": len(EXPECTED_FEATURES),
        "threshold": THRESHOLD,
        "features": EXPECTED_FEATURES,
    }


@app.post("/predict")
def predict(application: CreditApplication):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run python -m src.train_model.",
        )
    try:
        features = engineer_features(application)
        probability = float(model.predict_proba(features)[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability.")
    except Exception as error:
        raise HTTPException(status_code=500, detail="Prediction failed.") from error

    monthly_income = application.annual_income / 12
    existing_debt = application.total_outstanding_debt / 60
    monthly_obligation = application.monthly_emi + existing_debt
    dti = monthly_obligation / monthly_income
    return {
        "success": True,
        "risk_probability": round(probability, 6),
        "risk_percentage": round(probability * 100, 2),
        "risk_level": calculate_risk_level(probability),
        "decision": calculate_decision(probability),
        "threshold": THRESHOLD,
        "model": {
            "type": "XGBoost",
            "feature_count": len(EXPECTED_FEATURES),
            "backend_feature_engineering": True,
            "applicant_id_used": False,
        },
        "affordability": {
            "monthly_income": round(monthly_income, 2),
            "monthly_emi": round(application.monthly_emi, 2),
            "estimated_existing_monthly_debt": round(existing_debt, 2),
            "total_monthly_obligation": round(monthly_obligation, 2),
            "debt_to_income_ratio": round(dti, 4),
            "debt_to_income_percentage": round(dti * 100, 2),
            "loan_to_annual_income_ratio": round(
                application.loan_amount / application.annual_income, 4
            ),
        },
        "risk_flags": calculate_risk_flags(application.model_dump()),
        "backend_processing": {
            "raw_banking_inputs": len(CreditApplication.model_fields),
            "engineered_model_features": len(EXPECTED_FEATURES),
            "features_sent_to_model": len(features.columns),
        },
    }
