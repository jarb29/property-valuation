"""
Tests for model-related functionality.

This module contains tests for model pipeline and best model selection.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.pipeline.model_pipeline import ModelTrainingPipeline
from src.models.serialization import (
    list_saved_models,
    get_best_model,
    load_best_model
)
from src.config import MODEL_DIR
from src.utils.helpers import load_latest_data


class TestModelTrainingPipeline(unittest.TestCase):
    """Tests for the ModelTrainingPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_output_dir = os.path.join('tests', 'temp_outputs', 'model_pipeline')
        self.test_model_dir = os.path.join('tests', 'temp_outputs', 'models')
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.test_model_dir, exist_ok=True)

        # Create a test pipeline
        self.pipeline = ModelTrainingPipeline(
            model_dir=self.test_model_dir,
            output_dir=self.test_output_dir
        )

        # Create sample data for testing
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
        """Tear down test fixtures."""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
        if os.path.exists(self.test_model_dir):
            shutil.rmtree(self.test_model_dir)

    @patch('src.pipeline.model_pipeline.ModelPipeline')
    def test_train_model(self, mock_model_pipeline):
        """Test training a model."""
        # Set up the mock
        mock_instance = MagicMock()
        mock_model_pipeline.return_value = mock_instance

        # Call the method
        result = self.pipeline.train_model(
            X_train=self.X_train,
            y_train=self.y_train,
            model_type='gradient_boosting'
        )

        # Check the result
        self.assertEqual(result, mock_instance)
        mock_model_pipeline.assert_called_once()
        mock_instance.build_pipeline.assert_called_once()
        mock_instance.fit.assert_called_once_with(self.X_train, self.y_train)

    @patch('src.pipeline.model_pipeline.evaluate_model')
    @patch('src.pipeline.model_pipeline.get_versioned_filename')
    def test_evaluate_model(self, mock_get_versioned_filename, mock_evaluate):
        """Test evaluating a model."""
        # Set up the mocks
        mock_model = MagicMock()
        mock_model.model_type = 'gradient_boosting'
        mock_results = {'evaluation_results': {'rmse': 10000}}
        mock_evaluate.return_value = mock_results

        # Mock the versioned filename
        versioned_path = os.path.join(self.test_output_dir, 'v3.1_evaluation_gradient_boosting.json')
        mock_get_versioned_filename.return_value = versioned_path

        # Call the method
        result = self.pipeline.evaluate_model(
            model_pipeline=mock_model,
            X_test=self.X_test,
            y_test=self.y_test,
            is_regression=True
        )

        # Check the result
        self.assertEqual(result, mock_results)
        mock_get_versioned_filename.assert_called_once_with(
            base_name='gradient_boosting',
            file_type='evaluation',
            directory='/Users/joserubio/Desktop/proyectos/bain/outputs/pipeline/data',
            extension='json'
        )
        mock_evaluate.assert_called_once_with(
            model_pipeline=mock_model,
            X_test=self.X_test,
            y_test=self.y_test,
            is_regression=True,
            save_results=True,
            output_path=versioned_path
        )

    @patch('src.pipeline.model_pipeline.evaluate_model_with_slicing')
    @patch('src.pipeline.model_pipeline.get_versioned_filename')
    def test_evaluate_model_with_slicing(self, mock_get_versioned_filename, mock_evaluate_slicing):
        """Test evaluating a model with slicing."""
        # Set up the mocks
        mock_model = MagicMock()
        mock_model.model_type = 'gradient_boosting'
        mock_results = {'slice_results': {'type': {'house': {'rmse': 9000}}}}
        mock_evaluate_slicing.return_value = mock_results
        slice_features = ['type', 'sector']

        # Mock the versioned filename
        versioned_path = os.path.join(self.test_output_dir, 'v3.1_slice_evaluation_gradient_boosting.json')
        mock_get_versioned_filename.return_value = versioned_path

        # Call the method
        result = self.pipeline.evaluate_model_with_slicing(
            model_pipeline=mock_model,
            X_test=self.X_test,
            y_test=self.y_test,
            slice_features=slice_features,
            is_regression=True
        )

        # Check the result
        self.assertEqual(result, mock_results)
        mock_get_versioned_filename.assert_called_once_with(
            base_name='gradient_boosting',
            file_type='slice_evaluation',
            directory='/Users/joserubio/Desktop/proyectos/bain/outputs/pipeline/data',
            extension='json'
        )
        mock_evaluate_slicing.assert_called_once_with(
            model_pipeline=mock_model,
            X_test=self.X_test,
            y_test=self.y_test,
            slice_features=slice_features,
            is_regression=True,
            save_results=True,
            output_path=versioned_path
        )

    @patch('src.pipeline.model_pipeline.save_model')
    @patch('src.pipeline.model_pipeline.save_model_metadata')
    def test_save_model(self, mock_save_metadata, mock_save_model):
        """Test saving a model."""
        # Set up the mocks
        mock_model = MagicMock()
        mock_model.model_type = 'gradient_boosting'
        mock_model.get_model_summary.return_value = {'model_type': 'gradient_boosting'}

        model_path = os.path.join(self.test_model_dir, 'model.pkl')
        metadata_path = os.path.join(self.test_model_dir, 'model_metadata.json')
        mock_save_model.return_value = model_path
        mock_save_metadata.return_value = metadata_path

        # Call the method
        result_path, result_metadata_path = self.pipeline.save_model(
            model_pipeline=mock_model,
            base_name='property_valuation'
        )

        # Check the result
        self.assertEqual(result_path, model_path)
        self.assertEqual(result_metadata_path, metadata_path)
        mock_save_model.assert_called_once_with(
            model=mock_model,
            model_type='gradient_boosting',
            base_name='property_valuation',
            save_target='pipeline'
        )
        # Use ANY to match any metadata dictionary
        from unittest.mock import ANY
        mock_save_metadata.assert_called_once_with(model_path, ANY)

    @patch('src.pipeline.model_pipeline.get_best_model')
    @patch('src.pipeline.model_pipeline.load_best_model')
    def test_load_best_model(self, mock_load_best, mock_get_best):
        """Test loading the best model."""
        # Set up the mocks
        model_path = os.path.join(self.test_model_dir, 'model.pkl')
        mock_get_best.return_value = model_path
        mock_model = MagicMock()
        mock_load_best.return_value = mock_model

        # Call the method
        result = self.pipeline.load_best_model(metric='rmse')

        # Check the result
        self.assertEqual(result, mock_model)
        mock_get_best.assert_called_once_with(model_dir=self.test_model_dir, metric='rmse')
        mock_load_best.assert_called_once_with(model_dir=self.test_model_dir, metric='rmse')

    @patch('src.pipeline.model_pipeline.ModelTrainingPipeline.train_model')
    @patch('src.pipeline.model_pipeline.ModelTrainingPipeline.evaluate_model')
    @patch('src.pipeline.model_pipeline.ModelTrainingPipeline.evaluate_model_with_slicing')
    @patch('src.pipeline.model_pipeline.ModelTrainingPipeline.save_model')
    def test_run(self, mock_save, mock_evaluate_slicing, mock_evaluate, mock_train):
        """Test running the full pipeline."""
        # Set up the mocks
        mock_model = MagicMock()
        mock_train.return_value = mock_model

        eval_results = {'evaluation_results': {'rmse': 10000}}
        mock_evaluate.return_value = eval_results

        slice_results = {'slice_results': {'type': {'house': {'rmse': 9000}}}}
        mock_evaluate_slicing.return_value = slice_results

        model_path = os.path.join(self.test_model_dir, 'model.pkl')
        metadata_path = os.path.join(self.test_model_dir, 'model_metadata.json')
        mock_save.return_value = (model_path, metadata_path)

        slice_features = ['type', 'sector']

        # Call the method
        result = self.pipeline.run(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            model_type='gradient_boosting',
            slice_features=slice_features,
            is_regression=True,
            save_model=True,
            base_name='property_valuation'
        )

        # Check the result
        self.assertEqual(result['model_pipeline'], mock_model)
        self.assertEqual(result['evaluation_results'], eval_results)
        self.assertEqual(result['slice_evaluation_results'], slice_results)
        self.assertEqual(result['model_path'], model_path)
        self.assertEqual(result['metadata_path'], metadata_path)

        # Use ANY to match any arguments
        from unittest.mock import ANY
        mock_train.assert_called_once()
        mock_evaluate.assert_called_once_with(
            model_pipeline=mock_model,
            X_test=self.X_test,
            y_test=self.y_test,
            is_regression=True
        )
        mock_evaluate_slicing.assert_called_once_with(
            model_pipeline=mock_model,
            X_test=self.X_test,
            y_test=self.y_test,
            slice_features=slice_features,
            is_regression=True
        )
        mock_save.assert_called_once_with(
            model_pipeline=mock_model,
            base_name='property_valuation'
        )


# Tests for best model selection
def print_model_metrics(models):
    """
    Print evaluation metrics for all models.

    Args:
        models (Dict[str, Dict[str, Any]]): Dictionary of models and their metadata.
    """
    print("\nModel Evaluation Metrics:")
    print("-" * 80)
    print(f"{'Model Path':<60} {'RMSE':<10} {'MAPE':<10} {'MAE':<10}")
    print("-" * 80)

    for path, meta in models.items():
        metrics = meta.get('evaluation_metrics', {})
        if metrics:
            rmse = metrics.get('rmse', 'N/A')
            mape = metrics.get('mape', 'N/A')
            mae = metrics.get('mae', 'N/A')

            # Format the path to show only the filename
            filename = os.path.basename(path)

            print(f"{filename:<60} {rmse:<10.2f} {mape:<10.4f} {mae:<10.2f}")
        else:
            filename = os.path.basename(path)
            print(f"{filename:<60} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

    print("-" * 80)


@pytest.fixture
def models():
    """Fixture to get all saved models."""
    return list_saved_models(MODEL_DIR)


@pytest.fixture
def test_data():
    """Fixture to load the latest test data."""
    # Load the test dataset from the latest versioned file
    data = load_latest_data(data_type='test')

    # Define features and target
    features = [col for col in data.columns if col not in ['id', 'price']]
    target = 'price'

    X_test = data[features]
    y_test = data[target]

    return X_test, y_test, features


def test_list_saved_models():
    """Test that list_saved_models returns a dictionary."""
    models = list_saved_models(MODEL_DIR)
    assert isinstance(models, dict)

    # Print models for debugging
    print_model_metrics(models)


def test_get_best_model_rmse():
    """Test finding the best model based on RMSE."""
    best_model_path = get_best_model(metric='rmse')

    # Skip test if no models with RMSE metric are found
    if best_model_path is None:
        pytest.skip("No models with RMSE metric found")

    assert isinstance(best_model_path, str)
    assert os.path.exists(best_model_path)
    print(f"Best model (RMSE): {os.path.basename(best_model_path)}")


def test_get_best_model_mape():
    """Test finding the best model based on MAPE."""
    best_model_path = get_best_model(metric='mape')

    # Skip test if no models with MAPE metric are found
    if best_model_path is None:
        pytest.skip("No models with MAPE metric found")

    assert isinstance(best_model_path, str)
    assert os.path.exists(best_model_path)
    print(f"Best model (MAPE): {os.path.basename(best_model_path)}")


def test_get_best_model_mae():
    """Test finding the best model based on MAE."""
    best_model_path = get_best_model(metric='mae')

    # Skip test if no models with MAE metric are found
    if best_model_path is None:
        pytest.skip("No models with MAE metric found")

    assert isinstance(best_model_path, str)
    assert os.path.exists(best_model_path)
    print(f"Best model (MAE): {os.path.basename(best_model_path)}")


def test_load_best_model_rmse(test_data):
    """Test loading the best model based on RMSE."""
    best_model_path = get_best_model(metric='rmse')

    # Skip test if no models with RMSE metric are found
    if best_model_path is None:
        pytest.skip("No models with RMSE metric found")

    best_model = load_best_model(metric='rmse')
    assert best_model is not None
    print(f"Best model loaded successfully: {type(best_model)}")

    # Test the model on the test data
    X_test, y_test, _ = test_data
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Model performance on test data:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")


def test_load_best_model_mape(test_data):
    """Test loading the best model based on MAPE."""
    best_model_path = get_best_model(metric='mape')

    # Skip test if no models with MAPE metric are found
    if best_model_path is None:
        pytest.skip("No models with MAPE metric found")

    best_model = load_best_model(metric='mape')
    assert best_model is not None
    print(f"Best model loaded successfully: {type(best_model)}")

    # Test the model on the test data
    X_test, y_test, _ = test_data
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Model performance on test data:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")


def test_load_best_model_mae(test_data):
    """Test loading the best model based on MAE."""
    best_model_path = get_best_model(metric='mae')

    # Skip test if no models with MAE metric are found
    if best_model_path is None:
        pytest.skip("No models with MAE metric found")

    best_model = load_best_model(metric='mae')
    assert best_model is not None
    print(f"Best model loaded successfully: {type(best_model)}")

    # Test the model on the test data
    X_test, y_test, _ = test_data
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Model performance on test data:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")


if __name__ == '__main__':
    unittest.main()
