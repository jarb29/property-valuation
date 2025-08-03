import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import json
import os

from src.models.model import ModelPipeline
from src.config import JUPYTER_LOGS_DIR
from src.utils.helpers import get_versioned_filename


class ModelEvaluator:
    def __init__(self, model_pipeline: ModelPipeline):
        self.model_pipeline = model_pipeline
        self.evaluation_results = {}

    def evaluate_regression(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        predictions = self.model_pipeline.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        results = {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions),
            "mape": mean_absolute_percentage_error(y_test, predictions)
        }
        
        self.evaluation_results = results
        return results

    def save_evaluation_results(self, output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = get_versioned_filename(
                base_name=self.model_pipeline.model_type,
                file_type='evaluation',
                directory=JUPYTER_LOGS_DIR,
                extension='json'
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=4)
        
        return output_path

    def get_evaluation_summary(self) -> Dict[str, Any]:
        model_summary = self.model_pipeline.get_model_summary()
        return {**model_summary, "evaluation_results": self.evaluation_results}


def evaluate_model(model_pipeline: ModelPipeline, X_test: pd.DataFrame, y_test: np.ndarray,
                  is_regression: bool = True, save_results: bool = True,
                  output_path: Optional[str] = None) -> Dict[str, Any]:
    evaluator = ModelEvaluator(model_pipeline)
    evaluator.evaluate_regression(X_test, y_test)
    
    if save_results:
        evaluator.save_evaluation_results(output_path)
    
    return evaluator.get_evaluation_summary()