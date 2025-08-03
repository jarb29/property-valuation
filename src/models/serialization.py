"""
Model serialization/deserialization.

This module provides functionality for saving and loading machine learning models.
"""


import logging
from typing import Any, Optional, Dict, Union
import json
from datetime import datetime

from src.config import MODEL_DIR, DATA_VERSION
import os
import pickle
import joblib
from typing import Any, Optional

# Adjust the imports according to your actual project structure
from src.config import (
    MODEL_DIR,
    PIPELINE_MODELS_DIR,
    JUPYTER_MODELS_DIR
)
from src.utils.helpers import get_versioned_filename

logger = logging.getLogger(__name__)

def save_model(
    model: Any,
    path: Optional[str] = None,
    method: str = 'pickle',
    model_type: str = 'model',
    base_name: Optional[str] = None,
    save_target: str = 'pipeline'
) -> str:
    """
    Save a model to disk with versioning, optionally choosing between pipeline vs jupyter model dir.

    Args:
        model (Any): The model to save.
        path (str, optional): If provided, the exact save path (versioning is skipped).
        method (str, optional): The serialization method ('pickle' or 'joblib').
        model_type (str, optional): The type of model for versioning.
        base_name (str, optional): Base name for the versioned file.
        save_target (str, optional): Where to save the model if path is not given:
            - 'pipeline' -> saves to PIPELINE_MODELS_DIR
            - 'jupyter' -> saves to JUPYTER_MODELS_DIR
            - anything else -> uses MODEL_DIR (the default).

    Returns:
        str: The path where the model was saved.

    Raises:
        ValueError: If the serialization method is not supported.
    """
    # Decide where to save if path not provided
    if path is None:
        # Determine the directory based on save_target
        if save_target.lower() == 'pipeline':
            directory = PIPELINE_MODELS_DIR
        elif save_target.lower() == 'jupyter':
            directory = JUPYTER_MODELS_DIR
        else:
            directory = MODEL_DIR

        # Fallback base_name if none specified
        if base_name is None:
            base_name = 'model'

        # Pick extension
        extension = 'pkl' if method.lower() == 'pickle' else 'joblib'

        # Build versioned filename
        path = get_versioned_filename(
            base_name=base_name,
            file_type=model_type,
            directory=directory,
            extension=extension
        )

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save model using the specified serialization method
    if method.lower() == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    elif method.lower() == 'joblib':
        joblib.dump(model, path)
    else:
        raise ValueError(f"Unsupported serialization method: {method}")

    logger.info(f"Model saved to {path} using {method}")
    return path


def load_model(path: str, method: str = 'pickle') -> Any:
    """
    Load a model from disk.

    Args:
        path (str): The path where the model is saved.
        method (str, optional): The serialization method used ('pickle' or 'joblib'). Defaults to 'pickle'.

    Returns:
        Any: The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the serialization method is not supported.
    """
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # Load model using the specified method
    if method.lower() == 'pickle':
        with open(path, 'rb') as f:
            model = pickle.load(f)
    elif method.lower() == 'joblib':
        model = joblib.load(path)
    else:
        raise ValueError(f"Unsupported serialization method: {method}")

    logger.info(f"Model loaded from {path} using {method}")

    return model


def save_model_metadata(model_path: str, metadata: Dict[str, Any], metadata_path: Optional[str] = None) -> str:
    """
    Save model metadata to a JSON file with versioning.

    Args:
        model_path (str): The path where the model is saved.
        metadata (Dict[str, Any]): The metadata to save.
        metadata_path (Optional[str], optional): The path where the metadata will be saved.
            If provided, versioning is not used. Defaults to None.

    Returns:
        str: The path where the metadata was saved.
    """
    # Generate metadata path if not provided
    if metadata_path is None:
        # Extract model type and base name from model path for consistent versioning
        model_filename = os.path.basename(model_path)
        parts = model_filename.split('_')

        if len(parts) >= 3 and parts[0].startswith(DATA_VERSION):
            # If model filename follows our versioning pattern: v1.1_model_base.pkl
            # Extract the model type and base name
            model_type = parts[1]
            base_name = '_'.join(parts[2:]).split('.')[0]
        else:
            # Default values if model filename doesn't follow our pattern
            model_type = 'model'
            base_name = os.path.basename(model_path).split('.')[0]

        # Generate a versioned filename for metadata
        metadata_path = get_versioned_filename(
            base_name=base_name,
            file_type=f"{model_type}_metadata",
            directory=os.path.dirname(model_path),
            extension='json'
        )

    # Ensure directory exists
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Add timestamp and model path
    metadata['timestamp'] = datetime.now().isoformat()
    metadata['model_path'] = model_path

    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Model metadata saved to {metadata_path}")

    return metadata_path


def load_model_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Load model metadata from a JSON file.

    Args:
        metadata_path (str): The path where the metadata is saved.

    Returns:
        Dict[str, Any]: The loaded metadata.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    # Check if file exists
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    logger.info(f"Model metadata loaded from {metadata_path}")

    return metadata


def list_saved_models(model_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    List all saved models in the specified directory.

    Args:
        model_dir (Optional[str], optional): The directory to search for models. Defaults to None.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping model paths to their metadata.
    """
    # Use default model directory if not provided
    if model_dir is None:
        model_dir = MODEL_DIR

    # Check if directory exists
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory not found: {model_dir}")
        return {}

    # Find all model files
    model_files = []
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.pkl') or file.endswith('.joblib'):
                model_files.append(os.path.join(root, file))

    # Load metadata for each model
    models = {}
    for model_path in model_files:
        model_name = os.path.basename(model_path).split('.')[0]

        # Try all possible metadata filename patterns
        metadata_paths = []

        # List all files in the directory for debugging
        model_dir = os.path.dirname(model_path)
        logger.debug(f"Files in directory {model_dir}:")
        for file in os.listdir(model_dir):
            logger.debug(f"  {file}")

        # Standard pattern: model_name_metadata.json
        standard_pattern = os.path.join(model_dir, f"{model_name}_metadata.json")
        metadata_paths.append(standard_pattern)
        logger.debug(f"Trying standard pattern: {standard_pattern}")

        # Direct approach: try to find the exact metadata file
        # First, look for exact matches with the model name
        exact_matches = []
        for file in os.listdir(model_dir):
            if file.endswith('.json'):
                # Check if this is an exact match for the model
                model_base = model_name.split('.')[0]  # Remove extension if present
                file_base = file.split('.')[0]  # Remove extension

                # Check for exact match with model name in the metadata filename
                if model_base in file_base:
                    direct_pattern = os.path.join(model_dir, file)
                    exact_matches.append(direct_pattern)
                    logger.debug(f"Found potential exact match: {direct_pattern}")

        # Sort exact matches by similarity to model name (most similar first)
        exact_matches.sort(key=lambda x: abs(len(os.path.basename(x).split('.')[0]) - len(model_name)))

        # Add exact matches to metadata_paths (most similar first)
        for match in exact_matches:
            if match not in metadata_paths:
                metadata_paths.append(match)
                logger.debug(f"Trying direct pattern (exact match): {match}")

        # Then add any other JSON files that contain the model name
        for file in os.listdir(model_dir):
            if file.endswith('.json') and model_name in file:
                direct_pattern = os.path.join(model_dir, file)
                if direct_pattern not in metadata_paths:
                    metadata_paths.append(direct_pattern)
                    logger.debug(f"Trying direct pattern: {direct_pattern}")

        # Extract parts from model name
        parts = model_name.split('_')
        if len(parts) >= 3 and parts[0].startswith('v'):
            version = parts[0]
            model_type = parts[1]
            base_name = '_'.join(parts[2:])

            # Alternative pattern 1: version_model_type_metadata_base_name.json
            alt_pattern1 = os.path.join(
                model_dir,
                f"{version}_{model_type}_metadata_{base_name}.json"
            )
            metadata_paths.append(alt_pattern1)
            logger.debug(f"Trying alternative pattern 1: {alt_pattern1}")

            # Alternative pattern 2: directly look for files with similar name
            logger.debug(f"Looking for files starting with {version} and containing 'metadata'")
            for file in os.listdir(model_dir):
                if file.startswith(version) and file.endswith('.json') and 'metadata' in file:
                    # Check if this file corresponds to our model
                    model_base_name = os.path.basename(model_path).split('.')[0]
                    # Remove version prefix and file extension
                    file_without_version = file[len(version)+1:]
                    file_without_extension = file_without_version.split('.')[0]

                    # Check if the file contains the model type and base name
                    if model_type in file_without_extension and base_name in file_without_extension:
                        alt_pattern2 = os.path.join(model_dir, file)
                        metadata_paths.append(alt_pattern2)
                        logger.debug(f"Found potential metadata file: {alt_pattern2}")

            # Alternative pattern 3: special case for "metadata" in the middle
            # Pattern like "v1.1_gradient_metadata_boosting_property_valuation.json"
            alt_pattern3 = os.path.join(
                model_dir,
                f"{version}_{model_type}_metadata_{base_name}.json"
            )
            metadata_paths.append(alt_pattern3)
            logger.debug(f"Trying alternative pattern 3: {alt_pattern3}")

            # Alternative pattern 3b: special case for "metadata" in the middle of model_type
            # Pattern like "v1.1_gradient_metadata_boosting_property_valuation.json"
            model_type_parts = model_type.split('_')
            if len(model_type_parts) > 1:
                model_type_prefix = model_type_parts[0]
                model_type_suffix = '_'.join(model_type_parts[1:])
                alt_pattern3b = os.path.join(
                    model_dir,
                    f"{version}_{model_type_prefix}_metadata_{model_type_suffix}_{base_name}.json"
                )
                metadata_paths.append(alt_pattern3b)
                logger.debug(f"Trying alternative pattern 3b: {alt_pattern3b}")

            # Alternative pattern 3c: special case for when model_type is a single word
            # and metadata is between model_type and base_name
            # Pattern like "v1.1_gradient_metadata_boosting_property_valuation.json"
            alt_pattern3c = os.path.join(
                model_dir,
                f"{version}_{model_type}_metadata_{base_name}.json"
            )
            if alt_pattern3c != alt_pattern3:  # Avoid duplicates
                metadata_paths.append(alt_pattern3c)
                logger.debug(f"Trying alternative pattern 3c: {alt_pattern3c}")

            # Alternative pattern 3d: specific pattern for the exact format we're seeing
            # Pattern like "v1.1_gradient_metadata_boosting_property_valuation.json"
            if model_type == "gradient_boosting":
                alt_pattern3d = os.path.join(
                    model_dir,
                    f"{version}_gradient_metadata_boosting_{base_name}.json"
                )
                metadata_paths.append(alt_pattern3d)
                logger.debug(f"Trying alternative pattern 3d: {alt_pattern3d}")

            # Alternative pattern 4: try all JSON files with the same version prefix
            for file in os.listdir(model_dir):
                if file.startswith(version) and file.endswith('.json'):
                    alt_pattern4 = os.path.join(model_dir, file)
                    if alt_pattern4 not in metadata_paths:
                        metadata_paths.append(alt_pattern4)
                        logger.debug(f"Trying alternative pattern 4: {alt_pattern4}")

        # Try each metadata path
        metadata_path = None

        # First, try to find a metadata file that matches the model version exactly
        model_filename = os.path.basename(model_path)
        # Extract the version part (e.g., v1.1, v1.2, v1.3) from the model filename
        # The version is the part before the first underscore
        model_version_parts = model_filename.split('_')[0].split('.')
        if len(model_version_parts) >= 2:
            # Include the version number (e.g., v1.1, v1.2, v1.3)
            model_version = model_version_parts[0] + '.' + model_version_parts[1]
        else:
            # Fallback to just the first part (e.g., v1)
            model_version = model_version_parts[0]
        logger.debug(f"Model filename: {model_filename}")
        logger.debug(f"Model version: {model_version}")

        # Look for metadata files that match the model version
        version_matches = []
        for path in metadata_paths:
            if os.path.exists(path):
                # Extract the version from the metadata path
                metadata_filename = os.path.basename(path)
                if metadata_filename.startswith(model_version):
                    version_matches.append(path)
                    logger.debug(f"Found version match: {path}")

        # If we found version matches, use the one that matches the model version exactly
        exact_match = None
        if version_matches:
            for path in version_matches:
                metadata_filename = os.path.basename(path)
                # Check if the metadata filename starts with the exact model version
                if metadata_filename.startswith(model_version + "_"):
                    exact_match = path
                    logger.debug(f"Found exact version match: {path}")
                    break

            # If we found an exact match, use it
            if exact_match:
                metadata_path = exact_match
                logger.debug(f"Using exact version match: {metadata_path}")
            # Otherwise, use the first version match
            else:
                metadata_path = version_matches[0]
                logger.debug(f"Using version match: {metadata_path}")
        # Otherwise, fall back to the first metadata path that exists
        else:
            for path in metadata_paths:
                if os.path.exists(path):
                    metadata_path = path
                    logger.debug(f"Found metadata file: {metadata_path}")
                    break

        if metadata_path is not None and os.path.exists(metadata_path):
            try:
                metadata = load_model_metadata(metadata_path)
                models[model_path] = metadata
            except Exception as e:
                logger.warning(f"Could not load metadata for {model_path}: {str(e)}")
                models[model_path] = {"error": str(e)}
        else:
            logger.warning(f"No metadata file found for {model_path}")
            models[model_path] = {"timestamp": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()}

    return models


def get_latest_model(model_dir: Optional[str] = None, model_type: Optional[str] = None) -> Optional[str]:
    """
    Get the path to the latest saved model.

    Args:
        model_dir (Optional[str], optional): The directory to search for models. Defaults to None.
        model_type (Optional[str], optional): The type of model to search for. Defaults to None.

    Returns:
        Optional[str]: The path to the latest model, or None if no models are found.
    """
    # List all saved models
    models = list_saved_models(model_dir)

    if not models:
        return None

    # Filter by model type if specified
    if model_type is not None:
        models = {path: meta for path, meta in models.items()
                 if meta.get('model_type', '').lower() == model_type.lower()}

        if not models:
            return None

    # Sort by timestamp
    sorted_models = sorted(models.items(),
                          key=lambda x: x[1].get('timestamp', ''),
                          reverse=True)

    return sorted_models[0][0]



def get_best_model(
        model_dir: Optional[str] = None,
        model_type: Optional[str] = None,
        metric: str = 'rmse',
        load_target: str = 'pipeline'
) -> Optional[str]:
    """
    Get the path (URL) to the best model based on the specified metric.
    If model_dir is not provided, it will look in pipeline or jupyter directories
    or a default directory (MODEL_DIR) depending on 'load_target'.

    Args:
        model_dir (Optional[str]): The directory to search for models.
            If provided, overrides 'load_target'. Defaults to None.
        model_type (Optional[str]): Filter by model type (e.g., "gradient_boosting").
            Defaults to None.
        metric (str): Metric to sort by (rmse, mape, mae). Defaults to 'rmse'.
        load_target (str): If model_dir is None, determines which directory to check:
            - 'pipeline' -> PIPELINE_MODELS_DIR
            - 'jupyter'  -> JUPYTER_MODELS_DIR
            - anything else -> MODEL_DIR

    Returns:
        Optional[str]: The path to the best model, or None if no models are found.
    """
    # Decide where to look if model_dir is not given
    if model_dir is None:
        if load_target.lower() == 'pipeline':
            model_dir = PIPELINE_MODELS_DIR
        elif load_target.lower() == 'jupyter':
            model_dir = JUPYTER_MODELS_DIR
        else:
            model_dir = MODEL_DIR

    # List all saved models in the directory
    models = list_saved_models(model_dir)
    if not models:
        logger.warning(f"No models found in directory: {model_dir}")
        return None

    # (Optional) Filter by model_type
    if model_type:
        filtered_models = {}
        for path, meta in models.items():
            if meta.get('model_type', '').lower() == model_type.lower():
                filtered_models[path] = meta
        if not filtered_models:
            logger.warning(f"No models of type '{model_type}' found in {model_dir}")
            return None
        models = filtered_models

    # Only keep models that have the chosen metric
    models_with_metrics = {}
    for path, meta in models.items():
        eval_metrics = meta.get('evaluation_metrics', {})
        if eval_metrics and metric.lower() in eval_metrics:
            models_with_metrics[path] = meta

    if not models_with_metrics:
        logger.warning(f"No models with evaluation metric '{metric}' found in {model_dir}")
        return None

    # Sort (ascending) by the chosen metric to find the best (lowest) value
    sorted_models = sorted(
        models_with_metrics.items(),
        key=lambda x: x[1]['evaluation_metrics'][metric.lower()]
    )
    best_model_path, best_meta = sorted_models[0]
    best_value = best_meta['evaluation_metrics'][metric.lower()]
    logger.info(f"Best model found in '{model_dir}' with {metric}={best_value}: {best_model_path}")

    # Return only the path
    return best_model_path


def load_best_model(
        model_dir: Optional[str] = None,
        model_type: Optional[str] = None,
        metric: str = 'rmse',
        method: str = 'pickle',
        load_target: str = 'pipeline'
) -> Any:
    """
    Load the best model by first getting its path with get_best_model, then
    reading it from disk.

    Args:
        model_dir (Optional[str]): The directory to search in. If provided,
            overrides 'load_target'. Defaults to None.
        model_type (Optional[str]): The type of model to filter by. Defaults to None.
        metric (str): The metric used to determine the best model. Defaults to 'rmse'.
        method (str): The serialization method ('pickle' or 'joblib'). Defaults to 'pickle'.
        load_target (str): If model_dir is None, determines where to look:
            - 'pipeline' -> PIPELINE_MODELS_DIR
            - 'jupyter'  -> JUPYTER_MODELS_DIR
            - anything else -> MODEL_DIR

    Returns:
        Any: The loaded model.

    Raises:
        FileNotFoundError: If no models are found for the specified metric.
    """
    best_model_path = get_best_model(
        model_dir=model_dir,
        model_type=model_type,
        metric=metric,
        load_target=load_target
    )

    if not best_model_path:
        raise FileNotFoundError(
            f"No models found with metric '{metric}' in '{model_dir}' or load_target='{load_target}'")

    # Here we assume you have a helper function named load_model
    # that knows how to open the file based on the method
    return load_model(best_model_path, method)
