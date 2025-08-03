"""
Model definition and pipeline.

This module defines the machine learning model and its pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
import logging

from src.data.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating machine learning models."""

    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseEstimator:
        """
        Create a machine learning model of the specified type.

        Args:
            model_type (str): The type of model to create.
            **kwargs: Additional arguments to pass to the model constructor.

        Returns:
            BaseEstimator: The created model.

        Raises:
            ValueError: If the model type is not supported.
        """
        model_type = model_type.lower()

        if model_type == "random_forest":
            return RandomForestRegressor(**kwargs)
        elif model_type == "gradient_boosting":
            return GradientBoostingRegressor(**kwargs)
        elif model_type == "linear_regression":
            return LinearRegression(**kwargs)
        elif model_type == "ridge":
            return Ridge(**kwargs)
        elif model_type == "lasso":
            return Lasso(**kwargs)
        elif model_type == "svr":
            return SVR(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class ModelPipeline:
    """Class for creating and managing the model pipeline."""

    def __init__(self, model_type: str, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the model pipeline.

        Args:
            model_type (str): The type of model to use.
            model_params (Optional[Dict[str, Any]], optional): Parameters for the model. Defaults to None.
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.data_processor = DataProcessor()
        self.model = None
        self.pipeline = None

    def build_pipeline(self, categorical_features: List[str], numerical_features: List[str]) -> None:
        """
        Build the model pipeline.

        Args:
            categorical_features (List[str]): List of categorical feature column names.
            numerical_features (List[str]): List of numerical feature column names.
        """
        # Create the model
        self.model = ModelFactory.create_model(self.model_type, **self.model_params)

        # Set up the data processor
        self.data_processor = DataProcessor()

        # Store feature information
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

        logger.info(f"Built pipeline with model type: {self.model_type}")

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> 'ModelPipeline':
        """
        Fit the model pipeline to the training data.

        Args:
            X_train (pd.DataFrame): The training features.
            y_train (np.ndarray): The training target.

        Returns:
            ModelPipeline: The fitted model pipeline.
        """
        if not hasattr(self, 'categorical_features') or not hasattr(self, 'numerical_features'):
            raise ValueError("Pipeline must be built before fitting")

        # Fit the data processor
        self.data_processor.fit(
            X_train,
            self.categorical_features,
            self.numerical_features
        )

        # Transform the data
        X_train_processed = self.data_processor.transform(X_train)

        # Fit the model
        self.model.fit(X_train_processed, y_train)

        logger.info("Model pipeline fitted successfully")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted model pipeline.

        Args:
            X (pd.DataFrame): The features to predict on.

        Returns:
            np.ndarray: The predictions.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before predicting")

        # Transform the data
        X_processed = self.data_processor.transform(X)

        # Make predictions
        return self.model.predict(X_processed)

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get the feature importances from the model.

        Returns:
            Dict[str, float]: A dictionary mapping feature names to their importance.

        Raises:
            ValueError: If the model does not support feature importances.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importances")

        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not support feature importances")

        # Get feature names from the data processor
        feature_names = self.data_processor.get_feature_names()

        # Get feature importances from the model
        importances = self.model.feature_importances_

        # Create a dictionary mapping feature names to importances
        return {name: importance for name, importance in zip(feature_names, importances)}

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the model.

        Returns:
            Dict[str, Any]: The model parameters.
        """
        if self.model is None:
            raise ValueError("Model must be created before getting parameters")

        return self.model.get_params()

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.

        Returns:
            Dict[str, Any]: A dictionary with model information.
        """
        if self.model is None:
            raise ValueError("Model must be created before getting summary")

        return {
            "model_type": self.model_type,
            "model_params": self.model_params,
            "categorical_features": getattr(self, 'categorical_features', []),
            "numerical_features": getattr(self, 'numerical_features', [])
        }