import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class DataProcessor:
    def __init__(self):
        self.preprocessor = None
        self.categorical_features = []
        self.numerical_features = []

    def fit(self, data: pd.DataFrame, categorical_features: List[str], 
            numerical_features: List[str], target_column: Optional[str] = None) -> 'DataProcessor':
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        self.preprocessor = ColumnTransformer([
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        self.preprocessor.fit(data[categorical_features + numerical_features])
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(data[self.categorical_features + self.numerical_features])

    def fit_transform(self, data: pd.DataFrame, categorical_features: List[str],
                     numerical_features: List[str], target_column: Optional[str] = None) -> np.ndarray:
        self.fit(data, categorical_features, numerical_features, target_column)
        return self.transform(data)

    def get_feature_names(self) -> List[str]:
        return self.preprocessor.get_feature_names_out()