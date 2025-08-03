"""
Pydantic models for request/response.

This module defines Pydantic models for API request and response schemas.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any


class PredictionRequest(BaseModel):
    """
    Request schema for single property value prediction.

    Contains all the necessary features to predict a property's value
    including location, size, and property characteristics.
    """
    features: Dict[str, Any] = Field(
        ...,
        description="Property features dictionary containing all required fields for prediction",
        example={
            "type": "departamento",
            "sector": "las condes",
            "net_usable_area": 120.5,
            "net_area": 150.0,
            "n_rooms": 3,
            "n_bathroom": 2,
            "latitude": -33.4172,
            "longitude": -70.5476
        }
    )

    @validator('features')
    def validate_features(cls, v):
        """Validate that features is not empty."""
        if not v:
            raise ValueError("Features cannot be empty")

        required_fields = ['type', 'sector', 'net_usable_area', 'net_area', 'n_rooms', 'n_bathroom', 'latitude', 'longitude']
        missing_fields = [field for field in required_fields if field not in v]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "type": "departamento",
                    "sector": "las condes",
                    "net_usable_area": 120.5,
                    "net_area": 150.0,
                    "n_rooms": 3,
                    "n_bathroom": 2,
                    "latitude": -33.4172,
                    "longitude": -70.5476
                }
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema for single property value prediction.

    Contains the predicted property value along with metadata about
    the prediction process and model used.
    """
    prediction: float = Field(
        ...,
        description="Predicted property value in Chilean Pesos (CLP)",
        example=185000000,
        ge=0
    )
    prediction_time: float = Field(
        ...,
        description="Processing time for the prediction in seconds",
        example=0.0234,
        ge=0
    )
    model_version: str = Field(
        ...,
        description="Identifier of the model version used for this prediction",
        example="best_mae_pipeline"
    )

    class Config:
        schema_extra = {
            "example": {
                "prediction": 185000000,
                "prediction_time": 0.0234,
                "model_version": "best_mae_pipeline"
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Request schema for batch property value predictions.

    Allows multiple properties to be evaluated in a single request
    for efficient processing.
    """
    instances: List[Dict[str, Any]] = Field(
        ...,
        description="Array of property feature dictionaries for batch prediction",
        min_items=1,
        max_items=100,
        example=[
            {
                "type": "departamento",
                "sector": "las condes",
                "net_usable_area": 120.5,
                "net_area": 150.0,
                "n_rooms": 3,
                "n_bathroom": 2,
                "latitude": -33.4172,
                "longitude": -70.5476
            },
            {
                "type": "casa",
                "sector": "vitacura",
                "net_usable_area": 200.0,
                "net_area": 280.0,
                "n_rooms": 4,
                "n_bathroom": 3,
                "latitude": -33.3791,
                "longitude": -70.5435
            }
        ]
    )

    @validator('instances')
    def validate_instances(cls, v):
        """Validate that instances is not empty and contains valid data."""
        if not v:
            raise ValueError("Instances cannot be empty")
        if len(v) > 100:
            raise ValueError("Maximum 100 instances allowed per batch")

        required_fields = ['type', 'sector', 'net_usable_area', 'net_area', 'n_rooms', 'n_bathroom', 'latitude', 'longitude']
        for i, instance in enumerate(v):
            missing_fields = [field for field in required_fields if field not in instance]
            if missing_fields:
                raise ValueError(f"Instance {i}: Missing required fields: {missing_fields}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "type": "departamento",
                        "sector": "las condes",
                        "net_usable_area": 120.5,
                        "net_area": 150.0,
                        "n_rooms": 3,
                        "n_bathroom": 2,
                        "latitude": -33.4172,
                        "longitude": -70.5476
                    },
                    {
                        "type": "casa",
                        "sector": "vitacura",
                        "net_usable_area": 200.0,
                        "net_area": 280.0,
                        "n_rooms": 4,
                        "n_bathroom": 3,
                        "latitude": -33.3791,
                        "longitude": -70.5435
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Response schema for batch property value predictions.

    Contains an array of predicted values corresponding to the input
    instances in the same order.
    """
    predictions: List[float] = Field(
        ...,
        description="Array of predicted property values in Chilean Pesos (CLP), ordered same as input",
        example=[185000000, 220000000]
    )
    prediction_time: float = Field(
        ...,
        description="Total processing time for all predictions in seconds",
        example=0.0456,
        ge=0
    )
    model_version: str = Field(
        ...,
        description="Identifier of the model version used for predictions",
        example="best_mae_pipeline"
    )

    class Config:
        schema_extra = {
            "example": {
                "predictions": [185000000, 220000000],
                "prediction_time": 0.0456,
                "model_version": "best_mae_pipeline"
            }
        }


class ModelInfo(BaseModel):
    """
    Schema for model information and metadata.

    Provides comprehensive information about the currently active
    model including its capabilities and requirements.
    """
    name: str = Field(
        ...,
        description="Model name identifier",
        example="property_valuation"
    )
    version: str = Field(
        ...,
        description="Current model version identifier",
        example="best_mae_pipeline"
    )
    description: str = Field(
        ...,
        description="Human-readable description of the model's purpose",
        example="Real estate property valuation model for Chile"
    )
    features: List[str] = Field(
        ...,
        description="List of required input features for predictions",
        example=["type", "sector", "net_usable_area", "net_area", "n_rooms", "n_bathroom", "latitude", "longitude"]
    )

    class Config:
        schema_extra = {
            "example": {
                "name": "property_valuation",
                "version": "best_mae_pipeline",
                "description": "Real estate property valuation model for Chile",
                "features": [
                    "type", "sector", "net_usable_area", "net_area",
                    "n_rooms", "n_bathroom", "latitude", "longitude"
                ]
            }
        }


class ErrorResponse(BaseModel):
    """
    Schema for API error responses.

    Standardized error format providing clear information about
    what went wrong and when it occurred.
    """
    detail: str = Field(
        ...,
        description="Detailed error message explaining what went wrong",
        example="Model not found or failed to load"
    )

    class Config:
        schema_extra = {
            "example": {
                "detail": "Model not found or failed to load"
            }
        }



