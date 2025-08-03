"""
Data preprocessing and feature engineering.

This module provides functionality for preprocessing data and engineering features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Class for preprocessing data and engineering features."""

    def __init__(self):
        """Initialize the data processor."""
        self.preprocessor = None
        self.categorical_features = []
        self.numerical_features = []
        self.target_column = None

    def fit(self, data: pd.DataFrame,
            categorical_features: List[str],
            numerical_features: List[str],
            target_column: Optional[str] = None) -> 'DataProcessor':
        """
        Fit the data processor to the data.

        Args:
            data (pd.DataFrame): The data to fit the processor to.
            categorical_features (List[str]): List of categorical feature column names.
            numerical_features (List[str]): List of numerical feature column names.
            target_column (Optional[str], optional): Name of the target column. Defaults to None.

        Returns:
            DataProcessor: The fitted data processor.
        """
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target_column = target_column

        # Create preprocessing pipelines for both categorical and numeric data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Fit the preprocessor to the data
        self.preprocessor.fit(data[categorical_features + numerical_features])

        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform the data using the fitted preprocessor.

        Args:
            data (pd.DataFrame): The data to transform.

        Returns:
            np.ndarray: The transformed data.
        """
        if self.preprocessor is None:
            raise ValueError("DataProcessor must be fitted before transform")

        return self.preprocessor.transform(data[self.categorical_features + self.numerical_features])

    def fit_transform(self, data: pd.DataFrame,
                     categorical_features: List[str],
                     numerical_features: List[str],
                     target_column: Optional[str] = None) -> np.ndarray:
        """
        Fit the data processor to the data and transform it.

        Args:
            data (pd.DataFrame): The data to fit and transform.
            categorical_features (List[str]): List of categorical feature column names.
            numerical_features (List[str]): List of numerical feature column names.
            target_column (Optional[str], optional): Name of the target column. Defaults to None.

        Returns:
            np.ndarray: The transformed data.
        """
        self.fit(data, categorical_features, numerical_features, target_column)
        return self.transform(data)

    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features after transformation.

        Returns:
            List[str]: The feature names.
        """
        if self.preprocessor is None:
            raise ValueError("DataProcessor must be fitted before getting feature names")

        return self.preprocessor.get_feature_names_out()

    def prepare_data_for_training(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training by separating features and target.

        Args:
            data (pd.DataFrame): The data to prepare.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The features and target.
        """
        if self.target_column is None:
            raise ValueError("Target column must be specified for training data preparation")

        X = self.transform(data)
        y = data[self.target_column].values

        return X, y