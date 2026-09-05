"""
Pydantic schemas for the Home Credit Risk API.
"""

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class PredictionRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    applicant_id: int | None = Field(
        default=None,
        description=(
            "Optional applicant identifier."
        ),
    )

    features: dict[str, Any] = Field(
        description=(
            "Applicant feature values expected "
            "by the trained model."
        )
    )

    @field_validator("features")
    @classmethod
    def validate_features(
        cls,
        features: dict[str, Any],
    ) -> dict[str, Any]:

        if not features:

            raise ValueError(
                "At least one feature must be supplied."
            )

        return features


class PredictionResponse(BaseModel):

    applicant_id: int | None

    default_probability: float

    predicted_class: int

    risk_level: str

    threshold: float

    supplied_feature_count: int

    missing_feature_count: int


class ModelInformation(BaseModel):

    model_type: str

    expected_feature_count: int

    threshold: float

    prediction_target: str