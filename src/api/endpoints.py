"""
API endpoints.

This module defines the API endpoints for the FastAPI application.
"""

from fastapi import APIRouter, HTTPException
import logging
import time
import pandas as pd
import os
import json
from datetime import datetime

from src.api.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    ModelInfo, ErrorResponse
)
from src.models.serialization import load_best_model, get_best_model
from src.config import  MODEL_METRIC, MODEL_LOAD_TARGET, PIPELINE_SCHEMA_DIR, JUPYTER_SCHEMA_DIR, DATA_VERSION
from src.utils.logging import log_model_prediction, log_schema_validation
from src.data.generate_schema import validate_against_schema

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(
    prefix=f"/api/{DATA_VERSION}",
    tags=["Property Valuation"],
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

# Cache for loaded models and metadata
model_cache = {}


def get_model_with_metadata():
    """
    Get the best model and its metadata from cache or load from disk.

    Returns:
        tuple: (model, model_path, model_version)

    Raises:
        HTTPException: If the model cannot be loaded.
    """
    model_key = "best"

    # Check if model is in cache
    if model_key in model_cache:
        logger.debug(f"Using cached best model")
        return model_cache[model_key]

    try:
        # Get the best model path
        best_model_path = get_best_model(
            metric=MODEL_METRIC,
            load_target=MODEL_LOAD_TARGET
        )

        if not best_model_path:
            raise HTTPException(status_code=500, detail="No best model found")

        # Extract version from model path
        model_filename = os.path.basename(best_model_path)
        version_parts = model_filename.split('_')
        model_version = version_parts[0] if len(version_parts) >= 1 else "unknown"

        # Load the best model
        logger.info(f"Loading best model using metric={MODEL_METRIC}, load_target={MODEL_LOAD_TARGET}")
        model = load_best_model(
            metric=MODEL_METRIC,
            load_target=MODEL_LOAD_TARGET
        )

        # Cache model with metadata
        model_data = (model, best_model_path, model_version)
        model_cache[model_key] = model_data
        logger.info(f"Loaded best model using metric={MODEL_METRIC}, load_target={MODEL_LOAD_TARGET}")

        return model_data
    except Exception as e:
        logger.error(f"Error loading best model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    summary="Single Property Valuation",
    description="**Generate accurate property valuation using machine learning models**",
    response_description="Property value prediction with processing metrics and model version",
    responses={
        200: {
            "model": PredictionResponse,
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": 185000000,
                        "prediction_time": 0.0234,
                        "model_version": "best_mae_pipeline"
                    }
                }
            }
        },
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "features", "net_usable_area"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        },
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict(request: PredictionRequest):
    """
    Generate accurate property valuation using advanced machine learning models.

    This endpoint provides professional-grade property valuations for the Chilean real estate market
    using validated machine learning algorithms trained on comprehensive market data.

    **Request Example:**
    ```json
    {
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
    ```

    **Required Parameters:**
    - **type**: Property classification (departamento, casa, oficina)
    - **sector**: Geographic sector within Santiago metropolitan area
    - **net_usable_area**: Usable floor area in square meters
    - **net_area**: Total property area in square meters
    - **n_rooms**: Number of bedrooms
    - **n_bathroom**: Number of bathrooms
    - **latitude**: Geographic coordinate (decimal degrees)
    - **longitude**: Geographic coordinate (decimal degrees)

    **Response:**
    - **prediction**: Property valuation in Chilean Pesos (CLP)
    - **prediction_time**: Processing time in seconds
    - **model_version**: Model version identifier
    """
    start_time = time.time()

    try:
        # Get model and metadata in single call
        model, best_model_path, model_version = get_model_with_metadata()

        # Determine schema directory based on load target
        schema_dir = PIPELINE_SCHEMA_DIR if MODEL_LOAD_TARGET.lower() == 'pipeline' else JUPYTER_SCHEMA_DIR
        schema_path = os.path.join(schema_dir, f"{model_version}_schema_train.json")

        # Convert features to DataFrame for validation
        features_df = pd.DataFrame([request.features])

        # Create a typical property record with price for validation
        typical_property = pd.DataFrame({
            'type': [request.features.get('type', 'departamento')],
            'sector': [request.features.get('sector', 'las condes')],
            'net_usable_area': [request.features.get('net_usable_area', 120.0)],
            'net_area': [request.features.get('net_area', 150.0)],
            'n_rooms': [request.features.get('n_rooms', 3.0)],
            'n_bathroom': [request.features.get('n_bathroom', 2.0)],
            'latitude': [request.features.get('latitude', -33.4)],
            'longitude': [request.features.get('longitude', -70.55)],
            'price': [15000]  # Dummy price value for validation
        })

        # Validate against schema
        is_valid, violations = validate_against_schema(typical_property, schema_path)

        # Create validation metadata
        validation_metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "model_path": best_model_path,
            "schema_path": schema_path,
            "is_valid": is_valid,
            "violations": violations,
            "input_data": request.features
        }

        # Log validation metadata
        log_schema_validation('single', validation_metadata)

        # Log validation results
        if is_valid:
            logger.info(f"Input data validated successfully against schema: {schema_path}")
        else:
            logger.warning(f"Input data validation failed against schema: {schema_path}")
            logger.warning(f"Violations: {violations}")

        # Make prediction
        prediction = model.predict(features_df)[0]
        prediction_time = time.time() - start_time

        logger.info(f"Prediction made in {prediction_time:.4f} seconds")

        model_version_str = f"best_{MODEL_METRIC}_{MODEL_LOAD_TARGET}"

        log_model_prediction(
            model_name=f"property_valuation_{model_version_str}",
            input_data=request.features,
            prediction=prediction,
            model_version=model_version_str,
            processing_time=prediction_time
        )

        return PredictionResponse(
            prediction=prediction,
            prediction_time=prediction_time,
            model_version=model_version_str
        )
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predictions/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Property Valuation",
    description="**Process multiple property valuations efficiently in a single request**",
    response_description="Batch property valuations with performance metrics and model version",
    responses={
        200: {
            "model": BatchPredictionResponse,
            "description": "Successful batch prediction",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [185000000, 220000000, 165000000],
                        "prediction_time": 0.0456,
                        "model_version": "best_mae_pipeline"
                    }
                }
            }
        },
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "instances"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        },
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def batch_predict(request: BatchPredictionRequest):
    """
    Process multiple property valuations efficiently in a single request.

    This endpoint enables bulk property valuation for portfolio analysis and large-scale
    real estate assessments with optimized processing performance.

    **Request Example:**
    ```json
    {
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
                "sector": "providencia",
                "net_usable_area": 200.0,
                "net_area": 250.0,
                "n_rooms": 4,
                "n_bathroom": 3,
                "latitude": -33.4372,
                "longitude": -70.6276
            }
        ]
    }
    ```

    **Response:**
    - **predictions**: Array of property valuations in Chilean Pesos (CLP)
    - **prediction_time**: Total batch processing time in seconds
    - **model_version**: Model version identifier

    **Note:** Results are returned in the same order as input properties.
    """
    start_time = time.time()

    try:
        # Get model and metadata in single call
        model, best_model_path, model_version = get_model_with_metadata()

        # Determine schema directory and construct path
        schema_dir = PIPELINE_SCHEMA_DIR if MODEL_LOAD_TARGET.lower() == 'pipeline' else JUPYTER_SCHEMA_DIR
        schema_path = os.path.join(schema_dir, f"{model_version}_schema_train.json")

        # Convert instances to DataFrame for validation
        features_df = pd.DataFrame(request.instances)

        # Validate each instance in the batch
        all_valid = True
        all_violations = []

        for i, instance in enumerate(request.instances):
            # Create a typical property record with price for validation
            typical_property = pd.DataFrame({
                'type': [instance.get('type', 'departamento')],
                'sector': [instance.get('sector', 'las condes')],
                'net_usable_area': [instance.get('net_usable_area', 120.0)],
                'net_area': [instance.get('net_area', 150.0)],
                'n_rooms': [instance.get('n_rooms', 3.0)],
                'n_bathroom': [instance.get('n_bathroom', 2.0)],
                'latitude': [instance.get('latitude', -33.4)],
                'longitude': [instance.get('longitude', -70.55)],
                'price': [15000]  # Dummy price value for validation
            })

            # Validate against schema
            is_valid, violations = validate_against_schema(typical_property, schema_path)

            if not is_valid:
                all_valid = False
                all_violations.append({
                    "instance_index": i,
                    "violations": violations
                })

        # Create validation metadata
        validation_metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "model_path": best_model_path,
            "schema_path": schema_path,
            "is_valid": all_valid,
            "batch_size": len(request.instances),
            "violations": all_violations,
            "input_data": request.instances
        }

        # Log validation metadata
        log_schema_validation('batch', validation_metadata)

        # Log validation results
        if all_valid:
            logger.info(f"All batch instances validated successfully against schema: {schema_path}")
        else:
            logger.warning(f"Some batch instances failed validation against schema: {schema_path}")
            logger.warning(f"Violations: {all_violations}")

        # Make predictions
        predictions = model.predict(features_df).tolist()
        prediction_time = time.time() - start_time

        logger.info(f"Batch prediction made for {len(request.instances)} instances in {prediction_time:.4f} seconds")

        model_version_str = f"best_{MODEL_METRIC}_{MODEL_LOAD_TARGET}"

        # Log each prediction in the batch individually for detailed monitoring
        for i, (instance, prediction) in enumerate(zip(request.instances, predictions)):
            log_model_prediction(
                model_name=f"property_valuation_{model_version_str}",
                input_data=instance,
                prediction=prediction,
                model_version=model_version_str,
                processing_time=prediction_time
            )

        return BatchPredictionResponse(
            predictions=predictions,
            prediction_time=prediction_time,
            model_version=model_version_str
        )
    except Exception as e:
        logger.error(f"Error making batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/model/info",
    response_model=ModelInfo,
    summary="Model Information",
    description="**Retrieve comprehensive model metadata and capabilities**",
    response_description="Model specifications including version, features, and description",
    tags=["Model Management"],
    responses={
        200: {
            "model": ModelInfo,
            "description": "Model information retrieved successfully",
            "content": {
                "application/json": {
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
            }
        },
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_model_info():
    """
    Retrieve comprehensive metadata about the active valuation model.

    This endpoint provides essential information about the current model including
    version details, supported features, and technical specifications for integration purposes.

    **Response:**
    - **name**: Model identifier
    - **version**: Active model version
    - **description**: Model purpose and scope
    - **features**: Required input parameters

    **Use Cases:**
    - Model version verification
    - Integration parameter validation
    - System compatibility checks
    """
    try:
        # Get model metadata
        _, _, model_version = get_model_with_metadata()
        model_version_str = f"best_{MODEL_METRIC}_{MODEL_LOAD_TARGET}"

        return ModelInfo(
            name="property_valuation",
            version=model_version_str,
            description="Professional real estate valuation model for Chilean market",
            features=[
                "type", "sector", "net_usable_area", "net_area",
                "n_rooms", "n_bathroom", "latitude", "longitude"
            ]
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/health",
    summary="System Health Check",
    description="**Verify API service operational status and component health**",
    response_description="System health metrics and operational status",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-15T10:30:45.123456"
                    }
                }
            }
        },
        503: {
            "description": "Service unavailable",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Service unavailable"
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Perform comprehensive health check of the API service.

    This endpoint verifies system operational status including service availability
    and model loading capabilities for production readiness assessment.

    **Use Cases:**
    - Load balancer health verification
    - System monitoring and alerting
    - Deployment readiness checks
    - Integration testing validation

    **Response:**
    - **status**: Service operational status
    - **timestamp**: Current server timestamp

    **Status Codes:**
    - **200**: System operational and model accessible
    - **503**: Service unavailable or model loading failure
    """
    try:
        # Try to get model to ensure it's loadable
        get_model_with_metadata()
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")
