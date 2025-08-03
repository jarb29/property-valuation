"""
TensorFlow Model Analysis (TFMA) like evaluator.

This module provides functionality for evaluating machine learning models across different slices of data,
similar to TensorFlow Model Analysis (TFMA).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,

)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import os
from datetime import datetime

from src.models.model import ModelPipeline
from src.models.evaluate import ModelEvaluator
from src.config import JUPYTER_LOGS_DIR
from src.utils.helpers import get_versioned_filename

logger = logging.getLogger(__name__)


class TFMALikeEvaluator:
    """Class for evaluating machine learning models across different slices of data."""

    def __init__(self, model_pipeline: Union[ModelPipeline, Pipeline]):
        """
        Initialize the TFMA-like evaluator.

        Args:
            model_pipeline (Union[ModelPipeline, Pipeline]): The trained model pipeline to evaluate.
                Can be either a ModelPipeline object or a scikit-learn Pipeline object.
        """
        self.model_pipeline = model_pipeline
        self.evaluation_results = {}
        self.slice_results = {}
        self.is_regression = True  # Default to regression

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray,
                 slice_features: List[str], is_regression: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model on the entire test set and on slices of the test set.

        Args:
            X_test (pd.DataFrame): The test features.
            y_test (np.ndarray): The test target.
            slice_features (List[str]): Features to slice the data by.
            is_regression (bool, optional): Whether the model is a regression model. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary with evaluation results.
        """
        self.is_regression = is_regression

        # Create a standard evaluator for the overall evaluation
        evaluator = ModelEvaluator(self.model_pipeline)

        # Evaluate on the entire test set
        try:
            if is_regression:
                overall_results = evaluator.evaluate_regression(X_test, y_test)
            else:
                overall_results = evaluator.evaluate_classification(X_test, y_test)
        except AttributeError as e:
            # Handle the case where ModelEvaluator can't work with a regular Pipeline
            logger.warning(f"ModelEvaluator encountered an error: {str(e)}")
            logger.warning("Falling back to direct evaluation")

            # Fallback to direct evaluation
            if is_regression:
                # Make predictions
                predictions = self.model_pipeline.predict(X_test)

                # Calculate metrics
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

                # Store results
                overall_results = {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                }
            else:
                # Make predictions
                predictions = self.model_pipeline.predict(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
                recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
                f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

                # Store results
                overall_results = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }

        self.evaluation_results = overall_results

        # Evaluate on slices of the test set
        self.slice_results = {}

        for feature in slice_features:
            if feature not in X_test.columns:
                logger.warning(f"Feature {feature} not found in test data. Skipping.")
                continue

            # Get unique values for the feature
            unique_values = X_test[feature].unique()

            # Evaluate on each slice
            feature_results = {}
            for value in unique_values:
                # Create a mask for the current slice
                mask = X_test[feature] == value

                # Skip if the slice is too small
                if mask.sum() < 10:
                    logger.warning(f"Slice {feature}={value} has fewer than 10 samples. Skipping.")
                    continue

                try:
                    # Create a slice evaluator
                    slice_evaluator = ModelEvaluator(self.model_pipeline)

                    # Evaluate on the slice
                    if is_regression:
                        slice_result = slice_evaluator.evaluate_regression(X_test[mask], y_test[mask])
                    else:
                        slice_result = slice_evaluator.evaluate_classification(X_test[mask], y_test[mask])
                except AttributeError as e:
                    # Handle the case where ModelEvaluator can't work with a regular Pipeline
                    logger.warning(f"ModelEvaluator encountered an error for slice {feature}={value}: {str(e)}")
                    logger.warning("Falling back to direct evaluation")

                    # Fallback to direct evaluation
                    if is_regression:
                        # Make predictions
                        predictions = self.model_pipeline.predict(X_test[mask])

                        # Calculate metrics
                        mse = mean_squared_error(y_test[mask], predictions)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test[mask], predictions)
                        r2 = r2_score(y_test[mask], predictions)

                        # Store results
                        slice_result = {
                            "mse": mse,
                            "rmse": rmse,
                            "mae": mae,
                            "r2": r2
                        }
                    else:
                        # Make predictions
                        predictions = self.model_pipeline.predict(X_test[mask])

                        # Calculate metrics
                        accuracy = accuracy_score(y_test[mask], predictions)
                        precision = precision_score(y_test[mask], predictions, average='weighted', zero_division=0)
                        recall = recall_score(y_test[mask], predictions, average='weighted', zero_division=0)
                        f1 = f1_score(y_test[mask], predictions, average='weighted', zero_division=0)

                        # Store results
                        slice_result = {
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1
                        }

                # Store the result
                feature_results[str(value)] = slice_result

            self.slice_results[feature] = feature_results

        # Return a combined result
        return {
            "overall": self.evaluation_results,
            "slices": self.slice_results
        }

    def plot_slice_metrics(self, metric: str = None, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot evaluation metrics across different slices.

        Args:
            metric (str, optional): The metric to plot. If None, uses RMSE for regression and accuracy for classification.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 8).
        """
        if not self.slice_results:
            raise ValueError("Model must be evaluated before plotting slice metrics")

        # Determine the metric to plot
        if metric is None:
            if self.is_regression:
                metric = "rmse"
            else:
                metric = "accuracy"

        # Create a figure
        plt.figure(figsize=figsize)

        # Plot metrics for each slice feature
        num_features = len(self.slice_results)
        for i, (feature, results) in enumerate(self.slice_results.items()):
            # Create a subplot
            plt.subplot(num_features, 1, i + 1)

            # Extract values and metrics
            values = list(results.keys())
            metrics = [result[metric] for result in results.values()]

            # Create a bar plot
            sns.barplot(x=values, y=metrics)

            # Add a horizontal line for the overall metric
            overall_metric = self.evaluation_results[metric]
            plt.axhline(y=overall_metric, color='r', linestyle='--',
                        label=f'Overall {metric}: {overall_metric:.4f}')

            # Add labels and title
            plt.xlabel(feature)
            plt.ylabel(metric)
            plt.title(f'{metric} by {feature}')
            plt.legend()

            # Rotate x-axis labels if there are many values
            if len(values) > 5:
                plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_slice_comparison(self, slice_feature: str, metric: str = None,
                             figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot a comparison of a specific metric across slices of a feature.

        Args:
            slice_feature (str): The feature to slice by.
            metric (str, optional): The metric to plot. If None, uses RMSE for regression and accuracy for classification.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
        """
        if not self.slice_results or slice_feature not in self.slice_results:
            raise ValueError(f"Model must be evaluated on slice feature {slice_feature} before plotting")

        # Determine the metric to plot
        if metric is None:
            if self.is_regression:
                metric = "rmse"
            else:
                metric = "accuracy"

        # Extract values and metrics
        results = self.slice_results[slice_feature]
        values = list(results.keys())
        metrics = [result[metric] for result in results.values()]

        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'Slice': values,
            metric: metrics
        })

        # Sort by metric value
        df = df.sort_values(by=metric, ascending=False)

        # Create a figure
        plt.figure(figsize=figsize)

        # Create a bar plot
        sns.barplot(x='Slice', y=metric, data=df)

        # Add a horizontal line for the overall metric
        overall_metric = self.evaluation_results[metric]
        plt.axhline(y=overall_metric, color='r', linestyle='--',
                    label=f'Overall {metric}: {overall_metric:.4f}')

        # Add labels and title
        plt.xlabel(slice_feature)
        plt.ylabel(metric)
        plt.title(f'{metric} by {slice_feature}')
        plt.legend()

        # Rotate x-axis labels if there are many values
        if len(values) > 5:
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def get_slice_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the evaluation results across slices.

        Returns:
            Dict[str, Any]: A dictionary with evaluation information.
        """
        if not self.evaluation_results or not self.slice_results:
            raise ValueError("Model must be evaluated before getting summary")

        # Check if model_pipeline is a ModelPipeline object with get_model_summary method
        if hasattr(self.model_pipeline, 'get_model_summary'):
            # Get model summary from ModelPipeline object
            model_summary = self.model_pipeline.get_model_summary()
        else:
            # Create a basic model summary for scikit-learn Pipeline objects
            model_summary = {
                "model_type": getattr(self.model_pipeline, 'model_type', 'unknown_model'),
                "model_params": {},
                "categorical_features": [],
                "numerical_features": []
            }
            # Try to extract some information from the Pipeline if possible
            if hasattr(self.model_pipeline, 'steps'):
                # Get the last step which is typically the model
                last_step_name, last_step_estimator = self.model_pipeline.steps[-1]
                if hasattr(last_step_estimator, 'get_params'):
                    model_summary["model_params"] = last_step_estimator.get_params()

        # Combine with evaluation results
        return {
            **model_summary,
            "overall_evaluation": self.evaluation_results,
            "slice_evaluation": self.slice_results
        }

    def save_evaluation_results(self, output_path: Optional[str] = None) -> str:
        """
        Save evaluation results to a JSON file.

        Args:
            output_path (Optional[str], optional): Path to save the results. Defaults to None.

        Returns:
            str: Path to the saved results.
        """
        if not self.evaluation_results or not self.slice_results:
            raise ValueError("Model must be evaluated before saving results")

        # Generate output path if not provided
        if output_path is None:
            # Check if model_pipeline has model_type attribute (ModelPipeline object)
            # or use a default value (for scikit-learn Pipeline objects)
            model_type = getattr(self.model_pipeline, 'model_type', 'unknown_model')
            # Use versioned filename pattern
            output_path = get_versioned_filename(
                base_name=model_type,
                file_type='evaluation',
                directory=JUPYTER_LOGS_DIR,
                extension='json'
            )

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        results = {
            "overall": {},
            "slices": {}
        }

        # Process overall results
        for k, v in self.evaluation_results.items():
            if isinstance(v, np.ndarray):
                results["overall"][k] = v.tolist()
            else:
                results["overall"][k] = v

        # Process slice results
        for feature, feature_results in self.slice_results.items():
            results["slices"][feature] = {}
            for value, value_results in feature_results.items():
                results["slices"][feature][value] = {}
                for k, v in value_results.items():
                    if isinstance(v, np.ndarray):
                        results["slices"][feature][value][k] = v.tolist()
                    else:
                        results["slices"][feature][value][k] = v

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Evaluation results saved to {output_path}")

        return output_path


def evaluate_model_with_slicing(model_pipeline: Union[ModelPipeline, Pipeline], X_test: pd.DataFrame, y_test: np.ndarray,
                              slice_features: List[str], is_regression: bool = True,
                              save_results: bool = True, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate a model across different slices of data.

    Args:
        model_pipeline (Union[ModelPipeline, Pipeline]): The trained model pipeline to evaluate.
            Can be either a ModelPipeline object or a scikit-learn Pipeline object.
        X_test (pd.DataFrame): The test features.
        y_test (np.ndarray): The test target.
        slice_features (List[str]): Features to slice the data by.
        is_regression (bool, optional): Whether the model is a regression model. Defaults to True.
        save_results (bool, optional): Whether to save the evaluation results. Defaults to True.
        output_path (Optional[str], optional): Path to save the results. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary with evaluation results.
    """
    # Create evaluator
    evaluator = TFMALikeEvaluator(model_pipeline)

    # Evaluate model
    evaluator.evaluate(X_test, y_test, slice_features, is_regression)

    # Save results if requested
    if save_results:
        evaluator.save_evaluation_results(output_path)

    # Get evaluation summary
    evaluation_summary = evaluator.get_slice_summary()

    return evaluation_summary
