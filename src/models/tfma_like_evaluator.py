import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import os

from src.config import JUPYTER_LOGS_DIR
from src.utils.helpers import get_versioned_filename

logger = logging.getLogger(__name__)


class TFMALikeEvaluator:
    def __init__(self, model_pipeline):
        self.model_pipeline = model_pipeline
        self.evaluation_results = {}
        self.slice_results = {}

    def _evaluate_slice(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        predictions = self.model_pipeline.predict(X)
        mse = mean_squared_error(y, predictions)
        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y, predictions),
            "r2": r2_score(y, predictions)
        }

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray, slice_features: List[str]) -> Dict[str, Any]:
        # Overall evaluation
        self.evaluation_results = self._evaluate_slice(X_test, y_test)
        
        # Slice evaluation
        self.slice_results = {}
        for feature in slice_features:
            if feature not in X_test.columns:
                continue
                
            feature_results = {}
            for value in X_test[feature].unique():
                mask = X_test[feature] == value
                if mask.sum() >= 10:  # Minimum samples per slice
                    feature_results[str(value)] = self._evaluate_slice(X_test[mask], y_test[mask])
            
            self.slice_results[feature] = feature_results

        return {"overall": self.evaluation_results, "slices": self.slice_results}

    def plot_slice_metrics(self, metric='rmse', figsize=(12, 8)):
        """TFX-style slice metrics visualization."""
        if not self.slice_results:
            raise ValueError("Model must be evaluated before plotting slice metrics")
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        num_features = len(self.slice_results)
        fig, axes = plt.subplots(num_features, 1, figsize=(figsize[0], figsize[1] * num_features // 2))
        
        if num_features == 1:
            axes = [axes]
        
        for i, (feature, results) in enumerate(self.slice_results.items()):
            ax = axes[i]
            
            # Extract values and metrics
            slices = list(results.keys())
            values = [result[metric] for result in results.values()]
            
            # Create horizontal bar chart (TFX style)
            bars = ax.barh(slices, values, color='steelblue', alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar, val in zip(bars, values):
                width = bar.get_width()
                ax.text(width + width * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Add overall metric line
            overall_metric = self.evaluation_results[metric]
            ax.axvline(x=overall_metric, color='red', linestyle='--', linewidth=2,
                      label=f'Overall {metric.upper()}: {overall_metric:.3f}')
            
            ax.set_title(f'{metric.upper()} by {feature}', fontweight='bold', pad=15)
            ax.set_xlabel(metric.upper(), fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        return fig

    def get_slice_summary(self) -> Dict[str, Any]:
        model_summary = getattr(self.model_pipeline, 'get_model_summary', lambda: {
            "model_type": getattr(self.model_pipeline, 'model_type', 'unknown'),
            "model_params": {},
            "categorical_features": [],
            "numerical_features": []
        })()
        
        return {
            **model_summary,
            "overall_evaluation": self.evaluation_results,
            "slice_evaluation": self.slice_results
        }

    def save_evaluation_results(self, output_path: Optional[str] = None) -> str:
        if output_path is None:
            model_type = getattr(self.model_pipeline, 'model_type', 'unknown')
            output_path = get_versioned_filename(
                base_name=model_type,
                file_type='evaluation',
                directory=JUPYTER_LOGS_DIR,
                extension='json'
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results = {"overall": self.evaluation_results, "slices": self.slice_results}
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return output_path


def evaluate_model_with_slicing(model_pipeline, X_test: pd.DataFrame, y_test: np.ndarray,
                              slice_features: List[str], is_regression: bool = True,
                              save_results: bool = True, output_path: Optional[str] = None) -> Dict[str, Any]:
    evaluator = TFMALikeEvaluator(model_pipeline)
    evaluator.evaluate(X_test, y_test, slice_features)
    
    if save_results:
        evaluator.save_evaluation_results(output_path)
    
    return evaluator.get_slice_summary()