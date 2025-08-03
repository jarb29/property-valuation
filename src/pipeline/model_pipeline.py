import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from src.models.model import ModelPipeline
from src.models.evaluate import evaluate_model
from src.models.tfma_like_evaluator import evaluate_model_with_slicing
from src.models.serialization import save_model, save_model_metadata, load_best_model
from src.utils.helpers import get_versioned_filename
from src.config import MODEL_DIR, PIPELINE_DATA_DIR

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)

    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        return {
            "learning_rate": 0.01,
            "n_estimators": 300,
            "max_depth": 5,
            "loss": "absolute_error"
        } if model_type == 'gradient_boosting' else {}

    def _train_model(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                    model_type: str, model_params: Optional[Dict[str, Any]]) -> ModelPipeline:
        model_params = model_params or self._get_default_params(model_type)
        
        categorical_features = [col for col in X_train.columns if X_train[col].dtype == 'object']
        numerical_features = [col for col in X_train.columns if col not in categorical_features]
        
        model_pipeline = ModelPipeline(model_type, model_params)
        model_pipeline.build_pipeline(categorical_features, numerical_features)
        model_pipeline.fit(X_train, y_train)
        
        return model_pipeline

    def _evaluate_model(self, model_pipeline: ModelPipeline, X_test: pd.DataFrame, 
                       y_test: np.ndarray) -> Dict[str, Any]:
        output_path = get_versioned_filename(
            base_name=model_pipeline.model_type,
            file_type='evaluation',
            directory=PIPELINE_DATA_DIR,
            extension='json'
        )
        
        return evaluate_model(
            model_pipeline=model_pipeline,
            X_test=X_test,
            y_test=y_test,
            is_regression=True,
            save_results=True,
            output_path=output_path
        )

    def _evaluate_with_slicing(self, model_pipeline: ModelPipeline, X_test: pd.DataFrame,
                              y_test: np.ndarray, slice_features: List[str]) -> Dict[str, Any]:
        output_path = get_versioned_filename(
            base_name=model_pipeline.model_type,
            file_type='slice_evaluation',
            directory=PIPELINE_DATA_DIR,
            extension='json'
        )
        
        return evaluate_model_with_slicing(
            model_pipeline=model_pipeline,
            X_test=X_test,
            y_test=y_test,
            slice_features=slice_features,
            is_regression=True,
            save_results=True,
            output_path=output_path
        )

    def _save_model(self, model_pipeline: ModelPipeline, evaluation_results: Dict[str, Any],
                   training_data_shape: List[int]) -> Tuple[str, str]:
        model_path = save_model(
            model=model_pipeline,
            model_type=model_pipeline.model_type,
            base_name='property_valuation',
            save_target='pipeline'
        )
        
        model_summary = model_pipeline.get_model_summary()
        features = model_summary.get("categorical_features", []) + model_summary.get("numerical_features", [])
        
        metadata = {
            "model_type": model_summary.get("model_type", "unknown"),
            "features": features,
            "target": "price",
            "hyperparameters": model_summary.get("model_params", {}),
            "categorical_features": model_summary.get("categorical_features", []),
            "numerical_features": model_summary.get("numerical_features", []),
            "training_data_shape": training_data_shape,
            "description": "Property valuation model trained on real estate data"
        }
        
        if evaluation_results and "evaluation_results" in evaluation_results:
            metrics = evaluation_results["evaluation_results"]
            metadata["evaluation_metrics"] = {
                "rmse": float(metrics.get("rmse", 0)),
                "mape": float(metrics.get("mape", 0)) if "mape" in metrics else 0,
                "mae": float(metrics.get("mae", 0))
            }
        
        metadata_path = save_model_metadata(model_path, metadata)
        return model_path, metadata_path

    def load_best_model(self, metric: str = 'rmse') -> ModelPipeline:
        return load_best_model(model_dir=self.model_dir, metric=metric)

    def run(self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray,
           model_type: str = 'gradient_boosting', model_params: Optional[Dict[str, Any]] = None,
           slice_features: Optional[List[str]] = None, save_model: bool = True) -> Dict[str, Any]:
        
        model_pipeline = self._train_model(X_train, y_train, model_type, model_params)
        evaluation_results = self._evaluate_model(model_pipeline, X_test, y_test)
        
        slice_evaluation_results = None
        if slice_features:
            slice_evaluation_results = self._evaluate_with_slicing(model_pipeline, X_test, y_test, slice_features)
        
        model_path = metadata_path = None
        if save_model:
            training_data_shape = [len(X_train), len(X_train.columns) + 1]
            model_path, metadata_path = self._save_model(model_pipeline, evaluation_results, training_data_shape)
        
        return {
            "model_pipeline": model_pipeline,
            "evaluation_results": evaluation_results,
            "slice_evaluation_results": slice_evaluation_results,
            "model_path": model_path,
            "metadata_path": metadata_path
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from src.pipeline.data_pipeline import DataPipeline
    
    data_pipeline = DataPipeline()
    train_data = data_pipeline.run('train', False, True)
    test_data = data_pipeline.run('test', False, True)
    
    features = [col for col in train_data.columns if col not in ['id', 'price']]
    X_train, y_train = train_data[features], train_data['price'].values
    X_test, y_test = test_data[features], test_data['price'].values
    
    model_pipeline = ModelTrainingPipeline()
    results = model_pipeline.run(X_train, y_train, X_test, y_test, slice_features=['type', 'sector'])
    
    for metric, value in results['evaluation_results']['evaluation_results'].items():
        print(f"{metric}: {value}")