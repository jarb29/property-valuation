import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR

from src.data.data_processor import DataProcessor


class ModelFactory:
    @staticmethod
    def create_model(model_type: str, **kwargs):
        models = {
            "random_forest": RandomForestRegressor,
            "gradient_boosting": GradientBoostingRegressor,
            "linear_regression": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "svr": SVR
        }
        
        model_class = models.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model_class(**kwargs)


class ModelPipeline:
    def __init__(self, model_type: str, model_params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.data_processor = DataProcessor()
        self.model = None

    def build_pipeline(self, categorical_features: List[str], numerical_features: List[str]) -> None:
        self.model = ModelFactory.create_model(self.model_type, **self.model_params)
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> 'ModelPipeline':
        self.data_processor.fit(X_train, self.categorical_features, self.numerical_features)
        X_train_processed = self.data_processor.transform(X_train)
        self.model.fit(X_train_processed, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_processed = self.data_processor.transform(X)
        return self.model.predict(X_processed)

    def get_feature_importances(self) -> Dict[str, float]:
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not support feature importances")
        
        feature_names = self.data_processor.get_feature_names()
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))

    def get_model_params(self) -> Dict[str, Any]:
        return self.model.get_params()

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_params": self.model_params,
            "categorical_features": getattr(self, 'categorical_features', []),
            "numerical_features": getattr(self, 'numerical_features', [])
        }