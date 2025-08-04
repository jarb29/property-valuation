import os
import sys
import unittest
import pandas as pd
import numpy as np
import pytest
import shutil
from unittest.mock import patch, MagicMock, ANY
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.model_pipeline import ModelTrainingPipeline
from src.models.serialization import list_saved_models, get_best_model, load_best_model
from src.config import MODEL_DIR
from src.utils.helpers import load_latest_data


class TestModelTrainingPipeline(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.join('tests', 'temp_outputs', 'model_pipeline')
        self.test_model_dir = os.path.join('tests', 'temp_outputs', 'models')
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.test_model_dir, exist_ok=True)
        
        self.pipeline = ModelTrainingPipeline(model_dir=self.test_model_dir)
        
        self.X_train = pd.DataFrame({
            'type': ['house', 'apartment', 'house', 'apartment'],
            'sector': ['sector1', 'sector2', 'sector1', 'sector3'],
            'net_usable_area': [100.0, 80.0, 120.0, 70.0],
            'net_area': [120.0, 90.0, 140.0, 80.0],
            'n_rooms': [3.0, 2.0, 4.0, 1.0],
            'n_bathroom': [2.0, 1.0, 2.0, 1.0],
            'latitude': [-33.4, -33.5, -33.4, -33.6],
            'longitude': [-70.6, -70.7, -70.6, -70.8]
        })
        self.y_train = np.array([150000, 120000, 180000, 90000])
        
        self.X_test = pd.DataFrame({
            'type': ['house', 'apartment'],
            'sector': ['sector2', 'sector1'],
            'net_usable_area': [110.0, 75.0],
            'net_area': [130.0, 85.0],
            'n_rooms': [3.0, 2.0],
            'n_bathroom': [2.0, 1.0],
            'latitude': [-33.5, -33.4],
            'longitude': [-70.7, -70.6]
        })
        self.y_test = np.array([160000, 110000])

    def tearDown(self):
        for dir_path in [self.test_output_dir, self.test_model_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

    @patch('src.pipeline.model_pipeline.ModelPipeline')
    def test_train_model(self, mock_model_pipeline):
        mock_instance = MagicMock()
        mock_model_pipeline.return_value = mock_instance
        
        result = self.pipeline._train_model(
            X_train=self.X_train,
            y_train=self.y_train,
            model_type='gradient_boosting',
            model_params=None
        )
        
        self.assertEqual(result, mock_instance)
        mock_model_pipeline.assert_called_once()
        mock_instance.build_pipeline.assert_called_once()
        mock_instance.fit.assert_called_once_with(self.X_train, self.y_train)

    @patch('src.pipeline.model_pipeline.evaluate_model')
    @patch('src.pipeline.model_pipeline.get_versioned_filename')
    def test_evaluate_model(self, mock_get_versioned_filename, mock_evaluate):
        mock_model = MagicMock()
        mock_model.model_type = 'gradient_boosting'
        mock_results = {'evaluation_results': {'rmse': 10000}}
        mock_evaluate.return_value = mock_results
        versioned_path = os.path.join(self.test_output_dir, 'v3.1_evaluation_gradient_boosting.json')
        mock_get_versioned_filename.return_value = versioned_path
        
        result = self.pipeline._evaluate_model(
            model_pipeline=mock_model,
            X_test=self.X_test,
            y_test=self.y_test
        )
        
        self.assertEqual(result, mock_results)
        mock_get_versioned_filename.assert_called_once()
        mock_evaluate.assert_called_once()

    @patch('src.pipeline.model_pipeline.evaluate_model_with_slicing')
    @patch('src.pipeline.model_pipeline.get_versioned_filename')
    def test_evaluate_model_with_slicing(self, mock_get_versioned_filename, mock_evaluate_slicing):
        mock_model = MagicMock()
        mock_model.model_type = 'gradient_boosting'
        mock_results = {'slice_results': {'type': {'house': {'rmse': 9000}}}}
        mock_evaluate_slicing.return_value = mock_results
        versioned_path = os.path.join(self.test_output_dir, 'v3.1_slice_evaluation_gradient_boosting.json')
        mock_get_versioned_filename.return_value = versioned_path
        
        result = self.pipeline._evaluate_with_slicing(
            model_pipeline=mock_model,
            X_test=self.X_test,
            y_test=self.y_test,
            slice_features=['type', 'sector']
        )
        
        self.assertEqual(result, mock_results)
        mock_get_versioned_filename.assert_called_once()
        mock_evaluate_slicing.assert_called_once()

    @patch('src.pipeline.model_pipeline.save_model')
    @patch('src.pipeline.model_pipeline.save_model_metadata')
    def test_save_model(self, mock_save_metadata, mock_save_model):
        mock_model = MagicMock()
        mock_model.model_type = 'gradient_boosting'
        mock_model.get_model_summary.return_value = {'model_type': 'gradient_boosting'}
        
        model_path = os.path.join(self.test_model_dir, 'model.pkl')
        metadata_path = os.path.join(self.test_model_dir, 'model_metadata.json')
        mock_save_model.return_value = model_path
        mock_save_metadata.return_value = metadata_path
        
        result_path, result_metadata_path = self.pipeline._save_model(
            model_pipeline=mock_model,
            evaluation_results={'evaluation_results': {'rmse': 10000}},
            training_data_shape=[4, 9]
        )
        
        self.assertEqual(result_path, model_path)
        self.assertEqual(result_metadata_path, metadata_path)
        mock_save_model.assert_called_once()
        mock_save_metadata.assert_called_once_with(model_path, ANY)

    @patch('src.models.serialization.get_best_model')
    @patch('src.models.serialization.load_model')
    def test_load_best_model(self, mock_load_model, mock_get_best):
        model_path = os.path.join(self.test_model_dir, 'model.pkl')
        mock_get_best.return_value = model_path
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            f.write(b'dummy model data')
        
        result = self.pipeline.load_best_model(metric='rmse')
        
        self.assertEqual(result, mock_model)
        mock_get_best.assert_called_once_with(self.test_model_dir, 'rmse', 'pipeline')
        mock_load_model.assert_called_once_with(model_path)

    @patch('src.pipeline.model_pipeline.ModelTrainingPipeline._train_model')
    @patch('src.pipeline.model_pipeline.ModelTrainingPipeline._evaluate_model')
    @patch('src.pipeline.model_pipeline.ModelTrainingPipeline._evaluate_with_slicing')
    @patch('src.pipeline.model_pipeline.ModelTrainingPipeline._save_model')
    def test_run(self, mock_save, mock_evaluate_slicing, mock_evaluate, mock_train):
        mock_model = MagicMock()
        mock_train.return_value = mock_model
        eval_results = {'evaluation_results': {'rmse': 10000}}
        mock_evaluate.return_value = eval_results
        slice_results = {'slice_results': {'type': {'house': {'rmse': 9000}}}}
        mock_evaluate_slicing.return_value = slice_results
        model_path = os.path.join(self.test_model_dir, 'model.pkl')
        metadata_path = os.path.join(self.test_model_dir, 'model_metadata.json')
        mock_save.return_value = (model_path, metadata_path)
        
        result = self.pipeline.run(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            model_type='gradient_boosting',
            slice_features=['type', 'sector'],
            save_model=True
        )
        
        self.assertEqual(result['model_pipeline'], mock_model)
        self.assertEqual(result['evaluation_results'], eval_results)
        self.assertEqual(result['slice_evaluation_results'], slice_results)
        self.assertEqual(result['model_path'], model_path)
        self.assertEqual(result['metadata_path'], metadata_path)
        
        for mock in [mock_train, mock_evaluate, mock_evaluate_slicing, mock_save]:
            mock.assert_called_once()


def print_model_metrics(models):
    print("\nModel Evaluation Metrics:")
    print("-" * 80)
    print(f"{'Model Path':<60} {'RMSE':<10} {'MAPE':<10} {'MAE':<10}")
    print("-" * 80)
    
    for path, meta in models.items():
        metrics = meta.get('evaluation_metrics', {})
        filename = os.path.basename(path)
        if metrics:
            rmse = metrics.get('rmse', 'N/A')
            mape = metrics.get('mape', 'N/A')
            mae = metrics.get('mae', 'N/A')
            print(f"{filename:<60} {rmse:<10.2f} {mape:<10.4f} {mae:<10.2f}")
        else:
            print(f"{filename:<60} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    print("-" * 80)


@pytest.fixture
def models():
    return list_saved_models(MODEL_DIR)

@pytest.fixture
def test_data():
    data = load_latest_data(data_type='test')
    features = [col for col in data.columns if col not in ['id', 'price']]
    return data[features], data['price'], features


def test_list_saved_models():
    models = list_saved_models(MODEL_DIR)
    assert isinstance(models, dict)
    print_model_metrics(models)

@pytest.mark.parametrize("metric", ["rmse", "mape", "mae"])
def test_get_best_model(metric):
    best_model_path = get_best_model(metric=metric)
    if best_model_path is None:
        pytest.skip(f"No models with {metric.upper()} metric found")
    
    assert isinstance(best_model_path, str)
    assert os.path.exists(best_model_path)
    print(f"Best model ({metric.upper()}): {os.path.basename(best_model_path)}")


@pytest.mark.parametrize("metric", ["rmse", "mape", "mae"])
def test_load_best_model(metric, test_data):
    best_model_path = get_best_model(metric=metric)
    if best_model_path is None:
        pytest.skip(f"No models with {metric.upper()} metric found")
    
    best_model = load_best_model(metric=metric)
    assert best_model is not None
    print(f"Best model loaded successfully: {type(best_model)}")
    
    X_test, y_test, _ = test_data
    y_pred = best_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Model performance on test data:")
    print(f"RMSE: {rmse:.2f}, MAPE: {mape:.4f}, MAE: {mae:.2f}")

if __name__ == '__main__':
    unittest.main()
