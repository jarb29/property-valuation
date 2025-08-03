"""
Evaluation metrics and analysis.

This module provides functionality for evaluating machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import os
from datetime import datetime

from src.models.model import ModelPipeline
from src.config import JUPYTER_LOGS_DIR
from src.utils.helpers import get_versioned_filename

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for evaluating machine learning models."""

    def __init__(self, model_pipeline: ModelPipeline):
        """
        Initialize the model evaluator.

        Args:
            model_pipeline (ModelPipeline): The trained model pipeline to evaluate.
        """
        self.model_pipeline = model_pipeline
        self.evaluation_results = {}
        self.predictions = None
        self.actual_values = None
        self.is_regression = True  # Default to regression

    def evaluate_regression(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a regression model.

        Args:
            X_test (pd.DataFrame): The test features.
            y_test (np.ndarray): The test target.

        Returns:
            Dict[str, float]: A dictionary with evaluation metrics.
        """
        # Make predictions
        self.predictions = self.model_pipeline.predict(X_test)
        self.actual_values = y_test
        self.is_regression = True

        # Calculate metrics
        mse = mean_squared_error(y_test, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, self.predictions)
        r2 = r2_score(y_test, self.predictions)

        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = mean_absolute_percentage_error(y_test, self.predictions)

        # Store results
        results = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape
        }

        self.evaluation_results = results

        logger.info(f"Regression evaluation results: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        return results

    def evaluate_classification(self, X_test: pd.DataFrame, y_test: np.ndarray,
                               average: str = 'weighted') -> Dict[str, float]:
        """
        Evaluate a classification model.

        Args:
            X_test (pd.DataFrame): The test features.
            y_test (np.ndarray): The test target.
            average (str, optional): Averaging method for multi-class metrics. Defaults to 'weighted'.

        Returns:
            Dict[str, float]: A dictionary with evaluation metrics.
        """
        # Make predictions
        self.predictions = self.model_pipeline.predict(X_test)
        self.actual_values = y_test
        self.is_regression = False

        # Calculate metrics
        accuracy = accuracy_score(y_test, self.predictions)
        precision = precision_score(y_test, self.predictions, average=average, zero_division=0)
        recall = recall_score(y_test, self.predictions, average=average, zero_division=0)
        f1 = f1_score(y_test, self.predictions, average=average, zero_division=0)

        # Store results
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        # Add confusion matrix
        cm = confusion_matrix(y_test, self.predictions)
        results["confusion_matrix"] = cm.tolist()

        # Try to calculate ROC AUC if possible (binary classification or predict_proba available)
        try:
            if hasattr(self.model_pipeline.model, 'predict_proba'):
                y_prob = self.model_pipeline.model.predict_proba(
                    self.model_pipeline.data_processor.transform(X_test)
                )

                # Binary classification
                if y_prob.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                    results["roc_auc"] = roc_auc
                # Multi-class
                else:
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average=average)
                    results["roc_auc"] = roc_auc
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {str(e)}")

        self.evaluation_results = results

        logger.info(f"Classification evaluation results: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        return results

    def plot_regression_results(self, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot regression evaluation results.

        Args:
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 10).
        """
        if self.predictions is None or self.actual_values is None:
            raise ValueError("Model must be evaluated before plotting results")

        if not self.is_regression:
            raise ValueError("This method is only for regression models")

        plt.figure(figsize=figsize)

        # Actual vs Predicted
        plt.subplot(2, 2, 1)
        plt.scatter(self.actual_values, self.predictions, alpha=0.5)
        plt.plot([self.actual_values.min(), self.actual_values.max()],
                [self.actual_values.min(), self.actual_values.max()],
                'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')

        # Residuals
        residuals = self.actual_values - self.predictions

        # Residual plot
        plt.subplot(2, 2, 2)
        plt.scatter(self.predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')

        # Residual histogram
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title('Residual Histogram')

        # QQ plot
        plt.subplot(2, 2, 4)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot')

        plt.tight_layout()
        plt.show()

    def plot_classification_results(self, class_names: Optional[List[str]] = None,
                                   figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot classification evaluation results.

        Args:
            class_names (Optional[List[str]], optional): Names of the classes. Defaults to None.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 10).
        """
        if self.predictions is None or self.actual_values is None:
            raise ValueError("Model must be evaluated before plotting results")

        if self.is_regression:
            raise ValueError("This method is only for classification models")

        plt.figure(figsize=figsize)

        # Confusion Matrix
        cm = confusion_matrix(self.actual_values, self.predictions)
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        # Try to plot ROC curve for binary classification
        try:
            if hasattr(self.model_pipeline.model, 'predict_proba'):
                # Get probabilities
                y_prob = self.model_pipeline.model.predict_proba(
                    self.model_pipeline.data_processor.transform(X_test)
                )

                # Binary classification
                if y_prob.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(self.actual_values, y_prob[:, 1])

                    plt.subplot(2, 2, 2)
                    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(self.actual_values, y_prob[:, 1]):.4f})')
                    plt.plot([0, 1], [0, 1], 'r--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend()

                    # Precision-Recall curve
                    precision, recall, _ = precision_recall_curve(self.actual_values, y_prob[:, 1])

                    plt.subplot(2, 2, 3)
                    plt.plot(recall, precision)
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title('Precision-Recall Curve')
        except Exception as e:
            logger.warning(f"Could not plot ROC or PR curves: {str(e)}")

        plt.tight_layout()
        plt.show()

    def plot_feature_importances(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot feature importances.

        Args:
            top_n (int, optional): Number of top features to show. Defaults to 20.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 8).
        """
        try:
            importances = self.model_pipeline.get_feature_importances()

            # Sort importances
            importances = {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}

            # Take top N
            if len(importances) > top_n:
                importances = dict(list(importances.items())[:top_n])

            # Plot
            plt.figure(figsize=figsize)
            plt.barh(list(importances.keys()), list(importances.values()))
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Top {len(importances)} Feature Importances')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"Could not plot feature importances: {str(e)}")

    def save_evaluation_results(self, output_path: Optional[str] = None) -> str:
        """
        Save evaluation results to a JSON file.

        Args:
            output_path (Optional[str], optional): Path to save the results. Defaults to None.

        Returns:
            str: Path to the saved results.
        """
        if not self.evaluation_results:
            raise ValueError("Model must be evaluated before saving results")

        # Generate output path if not provided
        if output_path is None:
            model_type = self.model_pipeline.model_type
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
        results = {}
        for k, v in self.evaluation_results.items():
            if isinstance(v, np.ndarray):
                results[k] = v.tolist()
            else:
                results[k] = v

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"Evaluation results saved to {output_path}")

        return output_path

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the evaluation results.

        Returns:
            Dict[str, Any]: A dictionary with evaluation information.
        """
        if not self.evaluation_results:
            raise ValueError("Model must be evaluated before getting summary")

        # Get model summary
        model_summary = self.model_pipeline.get_model_summary()

        # Combine with evaluation results
        return {
            **model_summary,
            "evaluation_results": self.evaluation_results
        }


def evaluate_model(model_pipeline: ModelPipeline, X_test: pd.DataFrame, y_test: np.ndarray,
                  is_regression: bool = True, save_results: bool = True,
                  output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate a model using the complete evaluation pipeline.

    Args:
        model_pipeline (ModelPipeline): The trained model pipeline to evaluate.
        X_test (pd.DataFrame): The test features.
        y_test (np.ndarray): The test target.
        is_regression (bool, optional): Whether the model is a regression model. Defaults to True.
        save_results (bool, optional): Whether to save the evaluation results. Defaults to True.
        output_path (Optional[str], optional): Path to save the results. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary with evaluation results.
    """
    # Create evaluator
    evaluator = ModelEvaluator(model_pipeline)

    # Evaluate model
    if is_regression:
        evaluator.evaluate_regression(X_test, y_test)
    else:
        evaluator.evaluate_classification(X_test, y_test)

    # Save results if requested
    if save_results:
        evaluator.save_evaluation_results(output_path)

    # Get evaluation summary
    evaluation_summary = evaluator.get_evaluation_summary()

    return evaluation_summary
