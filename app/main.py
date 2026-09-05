"""
FastAPI application for Home Credit Risk Prediction.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from app.schemas import (
    ModelInformation,
    PredictionRequest,
    PredictionResponse,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "models/xgboost_home_credit_pipeline.joblib",
    )
)

CLASSIFICATION_THRESHOLD = float(
    os.getenv(
        "CLASSIFICATION_THRESHOLD",
        "0.50",
    )
)


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def get_expected_features(
    model_pipeline,
) -> list[str]:
    """
    Retrieve the original feature names expected by the fitted
    sklearn pipeline.
    """

    if hasattr(
        model_pipeline,
        "feature_names_in_",
    ):

        return list(
            model_pipeline.feature_names_in_
        )

    if hasattr(
        model_pipeline,
        "named_steps",
    ):

        preprocessor = (
            model_pipeline
            .named_steps
            .get("preprocessor")
        )

        if preprocessor is not None:

            if hasattr(
                preprocessor,
                "feature_names_in_",
            ):

                return list(
                    preprocessor
                    .feature_names_in_
                )

    raise ValueError(
        "Unable to determine expected "
        "feature names from the fitted model."
    )


# ---------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found: "
            f"{MODEL_PATH.resolve()}"
        )

    model_pipeline = joblib.load(
        MODEL_PATH
    )

    expected_features = (
        get_expected_features(
            model_pipeline
        )
    )

    app.state.model_pipeline = (
        model_pipeline
    )

    app.state.expected_features = (
        expected_features
    )

    yield

    app.state.model_pipeline = None


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="Home Credit Risk Prediction API",
    description=(
        "Predicts the probability of repayment "
        "difficulty for a Home Credit applicant."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, str]:

    return {
        "message": (
            "Home Credit Risk Prediction API"
        ),
        "documentation": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "features": "/features",
    }


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------

@app.get("/health")
def health(
    request: Request,
) -> dict[str, str]:

    model_pipeline = getattr(
        request.app.state,
        "model_pipeline",
        None,
    )

    if model_pipeline is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return {
        "status": "healthy",
        "model_status": "loaded",
    }


# ---------------------------------------------------------------------
# Model information
# ---------------------------------------------------------------------

@app.get(
    "/model-info",
    response_model=ModelInformation,
)
def model_information(
    request: Request,
) -> ModelInformation:

    model_pipeline = (
        request.app.state
        .model_pipeline
    )

    model = (
        model_pipeline
        .named_steps["model"]
    )

    expected_features = (
        request.app.state
        .expected_features
    )

    return ModelInformation(
        model_type=type(model).__name__,
        expected_feature_count=(
            len(expected_features)
        ),
        threshold=(
            CLASSIFICATION_THRESHOLD
        ),
        prediction_target=(
            "Probability of repayment difficulty"
        ),
    )


# ---------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------

@app.get("/features")
def list_features(
    request: Request,
) -> dict[str, object]:

    expected_features = (
        request.app.state
        .expected_features
    )

    return {
        "feature_count": (
            len(expected_features)
        ),
        "features": expected_features,
    }


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:

    model_pipeline = (
        request.app.state
        .model_pipeline
    )

    expected_features = (
        request.app.state
        .expected_features
    )

    supplied_features = (
        payload.features
    )

    # ---------------------------------------------------------------
    # Detect unknown features
    # ---------------------------------------------------------------

    unexpected_features = sorted(
        set(supplied_features)
        - set(expected_features)
    )

    if unexpected_features:

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Unexpected feature names supplied."
                ),
                "unexpected_features": (
                    unexpected_features
                ),
            },
        )

    # ---------------------------------------------------------------
    # Create model input.
    #
    # Missing features are represented as NaN.
    # The sklearn pipeline handles imputation.
    # ---------------------------------------------------------------

    applicant_data = {
        feature: supplied_features.get(
            feature,
            np.nan,
        )
        for feature in expected_features
    }

    applicant_frame = pd.DataFrame(
        [applicant_data],
        columns=expected_features,
    )

    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------

    try:

        probability = float(
            model_pipeline
            .predict_proba(
                applicant_frame
            )[0, 1]
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "Prediction failed. "
                "Check the supplied feature values "
                f"and data types. Error: {error}"
            ),
        ) from error

    # ---------------------------------------------------------------
    # Classification
    # ---------------------------------------------------------------

    predicted_class = int(
        probability
        >= CLASSIFICATION_THRESHOLD
    )

    risk_level = (
        "high"
        if predicted_class == 1
        else "low"
    )

    # ---------------------------------------------------------------
    # Missing feature count
    # ---------------------------------------------------------------

    missing_feature_count = (
        len(expected_features)
        - len(supplied_features)
    )

    # ---------------------------------------------------------------
    # Response
    # ---------------------------------------------------------------

    return PredictionResponse(
        applicant_id=payload.applicant_id,
        default_probability=round(
            probability,
            6,
        ),
        predicted_class=(
            predicted_class
        ),
        risk_level=risk_level,
        threshold=(
            CLASSIFICATION_THRESHOLD
        ),
        supplied_feature_count=(
            len(supplied_features)
        ),
        missing_feature_count=(
            missing_feature_count
        ),
    )
