"""
Model Pipeline Module.

This module implements the model pipeline for the property valuation model.
It includes functions for model training, evaluation, and persistence.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple

from src.models.model import ModelPipeline, ModelFactory
from src.models.evaluate import evaluate_model
from src.models.tfma_like_evaluator import evaluate_model_with_slicing
from src.models.serialization import save_model, save_model_metadata, get_best_model, load_best_model
from src.utils.helpers import get_versioned_filename
from src.config import MODEL_DIR, OUTPUT_PIPELINE_DIR, PIPELINE_DATA_DIR

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    Model training pipeline for the property valuation model.

    This class handles the model training, evaluation, and persistence
    steps of the pipeline.
    """

    def __init__(self, model_dir: str = None, output_dir: str = None):
        """
        Initialize the model training pipeline.

        Args:
            model_dir (str, optional): Directory to save model files. Defaults to MODEL_DIR from config.
            output_dir (str, optional): Directory to save output files. Defaults to OUTPUT_PIPELINE_DIR from config.
        """
        self.model_dir = model_dir if model_dir is not None else MODEL_DIR
        self.output_dir = output_dir if output_dir is not None else OUTPUT_PIPELINE_DIR

        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Initialized ModelTrainingPipeline with model_dir={self.model_dir}, output_dir={self.output_dir}")

    def train_model(self, X_train: pd.DataFrame, y_train: np.ndarray,
                   model_type: str = 'gradient_boosting',
                   model_params: Optional[Dict[str, Any]] = None) -> ModelPipeline:
        """
        Train a model on the given data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training target.
            model_type (str): Type of model to train.
            model_params (Optional[Dict[str, Any]]): Parameters for the model.

        Returns:
            ModelPipeline: The trained model pipeline.
        """
        logger.info(f"Training {model_type} model")

        # Set default model parameters if not provided
        if model_params is None:
            if model_type == 'gradient_boosting':
                model_params = {
                    "learning_rate": 0.01,
                    "n_estimators": 300,
                    "max_depth": 5,
                    "loss": "absolute_error"
                }
            else:
                model_params = {}

        # Identify categorical and numerical features
        categorical_features = [col for col in X_train.columns if X_train[col].dtype == 'object']
        numerical_features = [col for col in X_train.columns if col not in categorical_features]

        logger.info(f"Categorical features: {categorical_features}")
        logger.info(f"Numerical features: {numerical_features}")

        # Create and train the model pipeline
        model_pipeline = ModelPipeline(model_type, model_params)
        model_pipeline.build_pipeline(categorical_features, numerical_features)
        model_pipeline.fit(X_train, y_train)

        logger.info(f"Model training completed")

        return model_pipeline

    def evaluate_model(self, model_pipeline: ModelPipeline, X_test: pd.DataFrame, y_test: np.ndarray,
                      is_regression: bool = True) -> Dict[str, Any]:
        """
        Evaluate a trained model.

        Args:
            model_pipeline (ModelPipeline): The trained model pipeline.
            X_test (pd.DataFrame): Test features.
            y_test (np.ndarray): Test target.
            is_regression (bool): Whether the model is a regression model.

        Returns:
            Dict[str, Any]: Evaluation results.
        """
        logger.info(f"Evaluating model")

        # Generate versioned output path
        output_path = get_versioned_filename(
            base_name=model_pipeline.model_type,
            file_type='evaluation',
            directory=PIPELINE_DATA_DIR,
            extension='json'
        )

        # Evaluate the model
        evaluation_results = evaluate_model(
            model_pipeline=model_pipeline,
            X_test=X_test,
            y_test=y_test,
            is_regression=is_regression,
            save_results=True,
            output_path=output_path
        )

        logger.info(f"Model evaluation completed")

        return evaluation_results

    def evaluate_model_with_slicing(self, model_pipeline: ModelPipeline, X_test: pd.DataFrame, y_test: np.ndarray,
                                  slice_features: List[str], is_regression: bool = True) -> Dict[str, Any]:
        """
        Evaluate a trained model across different slices of data.

        Args:
            model_pipeline (ModelPipeline): The trained model pipeline.
            X_test (pd.DataFrame): Test features.
            y_test (np.ndarray): Test target.
            slice_features (List[str]): Features to slice the data by.
            is_regression (bool): Whether the model is a regression model.

        Returns:
            Dict[str, Any]: Evaluation results.
        """
        logger.info(f"Evaluating model with slicing on features: {slice_features}")

        # Generate versioned output path
        output_path = get_versioned_filename(
            base_name=model_pipeline.model_type,
            file_type='slice_evaluation',
            directory=PIPELINE_DATA_DIR,
            extension='json'
        )

        # Evaluate the model with slicing
        slice_evaluation_results = evaluate_model_with_slicing(
            model_pipeline=model_pipeline,
            X_test=X_test,
            y_test=y_test,
            slice_features=slice_features,
            is_regression=is_regression,
            save_results=True,
            output_path=output_path
        )

        logger.info(f"Model evaluation with slicing completed")

        return slice_evaluation_results

    def save_model(self, model_pipeline: ModelPipeline, base_name: str = 'property_valuation',
              evaluation_results: Optional[Dict[str, Any]] = None,
              target: str = 'price',
              training_data_shape: Optional[List[int]] = None) -> Tuple[str, str]:
        """
        Save a trained model and its metadata.

        Args:
            model_pipeline (ModelPipeline): The trained model pipeline.
            base_name (str): Base name for the model file.
            evaluation_results (Optional[Dict[str, Any]]): Evaluation results from model evaluation.
            target (str): The target variable name.
            training_data_shape (Optional[List[int]]): Shape of the training data [rows, columns].

        Returns:
            Tuple[str, str]: (model_path, metadata_path)
        """
        logger.info(f"Saving model with base_name={base_name}")

        # Save the model
        model_path = save_model(
            model=model_pipeline,
            model_type=model_pipeline.model_type,
            base_name=base_name,
            save_target = 'pipeline'
        )

        # Get basic model summary
        model_summary = model_pipeline.get_model_summary()

        # Create combined features list
        features = model_summary.get("categorical_features", []) + model_summary.get("numerical_features", [])

        # Create metadata in the required format
        metadata = {
            "model_type": model_summary.get("model_type", "unknown"),
            "features": features,
            "target": target,
            "hyperparameters": model_summary.get("model_params", {}),
            "categorical_features": model_summary.get("categorical_features", []),
            "numerical_features": model_summary.get("numerical_features", []),
            "training_data_shape": training_data_shape or [],
            "description": "Property valuation model trained on real estate data"
        }

        # Add evaluation metrics if available
        if evaluation_results and "evaluation_results" in evaluation_results:
            metrics = evaluation_results["evaluation_results"]
            metadata["evaluation_metrics"] = {
                "rmse": float(metrics.get("rmse", 0)),
                "mape": float(metrics.get("mape", 0)) if "mape" in metrics else 0,
                "mae": float(metrics.get("mae", 0))
            }

        # Save metadata
        metadata_path = save_model_metadata(model_path, metadata)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Model metadata saved to {metadata_path}")

        return model_path, metadata_path

    def load_best_model(self, metric: str = 'rmse') -> ModelPipeline:
        """
        Load the best model based on the specified evaluation metric.

        Args:
            metric (str): The metric to use for comparison.

        Returns:
            ModelPipeline: The loaded model pipeline.
        """
        logger.info(f"Loading best model based on {metric}")

        # Get the path to the best model
        best_model_path = get_best_model(model_dir=self.model_dir, metric=metric)

        if best_model_path is None:
            logger.warning(f"No models found with metric {metric}")
            return None

        # Load the model
        best_model = load_best_model(model_dir=self.model_dir, metric=metric)

        logger.info(f"Best model loaded from {best_model_path}")

        return best_model

    def run(self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray,
           model_type: str = 'gradient_boosting', model_params: Optional[Dict[str, Any]] = None,
           slice_features: Optional[List[str]] = None, is_regression: bool = True,
           save_model: bool = True, base_name: str = 'property_valuation') -> Dict[str, Any]:
        """
        Run the full model training pipeline.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (np.ndarray): Training target.
            X_test (pd.DataFrame): Test features.
            y_test (np.ndarray): Test target.
            model_type (str): Type of model to train.
            model_params (Optional[Dict[str, Any]]): Parameters for the model.
            slice_features (Optional[List[str]]): Features to slice the data by for evaluation.
            is_regression (bool): Whether the model is a regression model.
            save_model (bool): Whether to save the trained model.
            base_name (str): Base name for the model file.

        Returns:
            Dict[str, Any]: Results of the pipeline run.
        """
        logger.info(f"Running model training pipeline")

        # Step 1: Train the model
        model_pipeline = self.train_model(X_train, y_train, model_type, model_params)

        # Step 2: Evaluate the model
        evaluation_results = self.evaluate_model(model_pipeline, X_test, y_test, is_regression)

        # Step 3: Evaluate the model with slicing (if slice_features is provided)
        slice_evaluation_results = None
        if slice_features is not None:
            slice_evaluation_results = self.evaluate_model_with_slicing(
                model_pipeline, X_test, y_test, slice_features, is_regression
            )

        # Step 4: Save the model (if requested)
        model_path = None
        metadata_path = None
        if save_model:
            # Get training data shape
            training_data_shape = [len(X_train), len(X_train.columns) + 1]  # +1 for target column

            model_path, metadata_path = self.save_model(
                model_pipeline,
                base_name=base_name,
                evaluation_results=evaluation_results,
                target='price',
                training_data_shape=training_data_shape
            )

        # Return the results
        results = {
            "model_pipeline": model_pipeline,
            "evaluation_results": evaluation_results,
            "slice_evaluation_results": slice_evaluation_results,
            "model_path": model_path,
            "metadata_path": metadata_path
        }

        logger.info(f"Model training pipeline completed successfully")

        return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    from src.pipeline.data_pipeline import DataPipeline

    # Run the data pipeline to get the processed data
    data_pipeline = DataPipeline()
    train_data = data_pipeline.run(data_type='train', use_latest=False, validate=True)
    test_data = data_pipeline.run(data_type='test', use_latest=False, validate=True)

    # Prepare features and target
    features = [col for col in train_data.columns if col not in ['id', 'price']]
    target = 'price'

    X_train = train_data[features]
    y_train = train_data[target].values
    X_test = test_data[features]
    y_test = test_data[target].values

    # Run the model training pipeline
    model_pipeline = ModelTrainingPipeline()
    results = model_pipeline.run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type='gradient_boosting',
        slice_features=['type', 'sector'],
        is_regression=True,
        save_model=True
    )

    print(f"Model evaluation results:")
    for metric, value in results['evaluation_results']['evaluation_results'].items():
        print(f"  {metric}: {value}")
