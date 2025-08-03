"""
Tests for data-related functionality.

This module contains tests for data loading, data pipeline, and schema validation.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.utils.helpers import load_latest_data, load_original_data
from src.config import DATA_VERSION, TRAIN_DATA_PATH, TEST_DATA_PATH
from src.pipeline.data_pipeline import DataPipeline
from src.data.generate_schema import validate_against_schema


class TestDataLoading(unittest.TestCase):
    """Tests for data loading functionality."""

    def test_data_loading(self):
        """Test data loading functionality for both functions."""
        # Test load_original_data
        train_data_original = load_original_data(data_type='train')
        self.assertIsInstance(train_data_original, pd.DataFrame)
        self.assertGreater(len(train_data_original), 0)

        # Test load_latest_data
        train_data_latest = load_latest_data(data_type='train')
        self.assertIsInstance(train_data_latest, pd.DataFrame)
        self.assertGreater(len(train_data_latest), 0)


class TestDataPipeline(unittest.TestCase):
    """Tests for the DataPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_output_dir = os.path.join('tests', 'temp_outputs', 'data_pipeline')
        os.makedirs(self.test_output_dir, exist_ok=True)

        # Create a test pipeline
        self.pipeline = DataPipeline(output_dir=self.test_output_dir)

        # Create a sample dataframe for testing
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'type': ['house', 'apartment', 'house', 'apartment', 'house'],
            'sector': ['sector1', 'sector2', 'sector1', 'sector3', 'sector2'],
            'net_usable_area': [100.0, 80.0, 120.0, 0.0, 90.0],  # One zero value
            'net_area': [120.0, 90.0, 140.0, 70.0, 110.0],
            'n_rooms': [3.0, 2.0, 4.0, 1.0, 3.0],
            'n_bathroom': [2.0, 1.0, 2.0, 1.0, 2.0],
            'latitude': [-33.4, -33.5, -33.4, -33.6, -33.5],
            'longitude': [-70.6, -70.7, -70.6, -70.8, -70.7],
            'price': [150000, 120000, 180000, 90000, 140000]
        })

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    @patch('src.pipeline.data_pipeline.load_original_data')
    def test_load_data_original(self, mock_load_original):
        """Test loading original data."""
        # Set up the mock
        mock_load_original.return_value = self.sample_data

        # Call the method
        result = self.pipeline.load_data(data_type='train', use_latest=False)

        # Check the result
        self.assertEqual(len(result), 5)
        mock_load_original.assert_called_once_with(data_type='train')

    @patch('src.pipeline.data_pipeline.load_latest_data')
    def test_load_data_latest(self, mock_load_latest):
        """Test loading latest data."""
        # Set up the mock
        mock_load_latest.return_value = self.sample_data

        # Call the method
        result = self.pipeline.load_data(data_type='train', use_latest=True)

        # Check the result
        self.assertEqual(len(result), 5)
        mock_load_latest.assert_called_once_with(data_type='train', outputs_dir='/Users/joserubio/Desktop/proyectos/bain/outputs/pipeline/data')

    def test_clean_data(self):
        """Test cleaning data (removing zero values)."""
        # Call the method
        result = self.pipeline.clean_data(self.sample_data)

        # Check the result
        self.assertEqual(len(result), 4)  # One row should be removed
        self.assertNotIn(0.0, result['net_usable_area'].values)

    @patch('src.pipeline.data_pipeline.separate_outliers_and_save')
    def test_detect_and_handle_outliers(self, mock_separate_outliers):
        """Test outlier detection and handling."""
        # Set up the mock
        clean_data = self.sample_data[self.sample_data['net_usable_area'] > 0]
        outlier_data = pd.DataFrame()
        mock_separate_outliers.return_value = (clean_data, outlier_data)

        # Call the method
        clean_result, outlier_result = self.pipeline.detect_and_handle_outliers(self.sample_data)

        # Check the result
        self.assertEqual(len(clean_result), 4)
        self.assertEqual(len(outlier_result), 0)
        mock_separate_outliers.assert_called_once_with(data_path=None, save_target='pipeline')

    @patch('src.pipeline.data_pipeline.generate_schema')
    @patch('src.pipeline.data_pipeline.schema_to_json')
    def test_generate_and_save_schema(self, mock_schema_to_json, mock_generate_schema):
        """Test schema generation and saving."""
        # Set up the mocks
        mock_schema = {'dataset_name': 'test_data', 'columns': {}}
        mock_generate_schema.return_value = mock_schema
        mock_schema_to_json.return_value = os.path.join(self.test_output_dir, 'schema.json')

        # Call the method
        result = self.pipeline.generate_and_save_schema(
            self.sample_data,
            dataset_name='test_data',
            base_name='test'
        )

        # Check the result
        self.assertEqual(result, mock_schema)
        mock_generate_schema.assert_called_once_with(self.sample_data, 'test_data')
        mock_schema_to_json.assert_called_once_with(mock_schema, base_name='test', save_target='pipeline')

    @patch('src.pipeline.data_pipeline.validate_against_schema')
    @patch('src.pipeline.data_pipeline.get_latest_versioned_file')
    def test_validate_data(self, mock_get_latest, mock_validate):
        """Test data validation."""
        # Set up the mocks
        schema_path = os.path.join(self.test_output_dir, 'schema.json')
        mock_get_latest.return_value = schema_path
        mock_validate.return_value = (True, [])

        # Call the method
        result = self.pipeline.validate_data(self.sample_data)

        # Check the result
        self.assertTrue(result)
        mock_get_latest.assert_called_once()
        mock_validate.assert_called_once_with(self.sample_data, schema_path, log_violations=True)

    @patch('src.pipeline.data_pipeline.DataPipeline.load_data')
    @patch('src.pipeline.data_pipeline.DataPipeline.clean_data')
    @patch('src.pipeline.data_pipeline.DataPipeline.detect_and_handle_outliers')
    @patch('src.pipeline.data_pipeline.DataPipeline.generate_and_save_schema')
    @patch('src.pipeline.data_pipeline.DataPipeline.validate_data')
    def test_run_pipeline_train(self, mock_validate, mock_schema, mock_outliers, mock_clean, mock_load):
        """Test running the full pipeline for training data."""
        # Set up the mocks
        mock_load.return_value = self.sample_data
        mock_clean.return_value = self.sample_data[self.sample_data['net_usable_area'] > 0]
        clean_data = self.sample_data[self.sample_data['net_usable_area'] > 0]
        outlier_data = pd.DataFrame()
        mock_outliers.return_value = (clean_data, outlier_data)
        mock_schema.return_value = {'dataset_name': 'train_data', 'columns': {}}
        mock_validate.return_value = True

        # Call the method
        result = self.pipeline.run(data_type='train', use_latest=False, validate=True)

        # Check the result
        self.assertEqual(len(result), 4)
        mock_load.assert_called_once_with(data_type='train', use_latest=False)
        mock_clean.assert_called_once()
        mock_outliers.assert_called_once()
        mock_schema.assert_called_once()
        mock_validate.assert_called_once()


class TestSchemaValidation(unittest.TestCase):
    """Tests for schema validation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary schema file for testing
        self.schema_path = os.path.join('tests', 'temp_schema.json')
        import json
        schema = {
            "dataset_name": "test_data",
            "columns": {
                "type": {
                    "data_type": "object",
                    "unique_values_list": ["departamento", "casa"],
                    "min_value": None,
                    "max_value": None
                },
                "sector": {
                    "data_type": "object",
                    "unique_values_list": ["las condes", "vitacura", "providencia"],
                    "min_value": None,
                    "max_value": None
                },
                "net_usable_area": {
                    "data_type": "float64",
                    "unique_values_list": None,
                    "numerical": {
                        "min": 0.0,
                        "max": 500.0
                    }
                },
                "net_area": {
                    "data_type": "float64",
                    "unique_values_list": None,
                    "numerical": {
                        "min": 0.0,
                        "max": 1000.0
                    }
                },
                "n_rooms": {
                    "data_type": "float64",
                    "unique_values_list": None,
                    "numerical": {
                        "min": 0.0,
                        "max": 10.0
                    }
                },
                "n_bathroom": {
                    "data_type": "float64",
                    "unique_values_list": None,
                    "numerical": {
                        "min": 0.0,
                        "max": 10.0
                    }
                },
                "latitude": {
                    "data_type": "float64",
                    "unique_values_list": None,
                    "numerical": {
                        "min": -90.0,
                        "max": 90.0
                    }
                },
                "longitude": {
                    "data_type": "float64",
                    "unique_values_list": None,
                    "numerical": {
                        "min": -180.0,
                        "max": 180.0
                    }
                },
                "price": {
                    "data_type": "int64",
                    "unique_values_list": None,
                    "numerical": {
                        "min": 0,
                        "max": 1000000
                    }
                }
            }
        }
        with open(self.schema_path, 'w') as f:
            json.dump(schema, f)

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary files
        if os.path.exists(self.schema_path):
            os.remove(self.schema_path)

    def test_single_valid_value(self):
        """Test validation with a single valid value."""
        single_valid = pd.DataFrame({
            'type': ['departamento'],
            'sector': ['las condes'],
            'net_usable_area': [120.0],
            'net_area': [150.0],
            'n_rooms': [3.0],
            'n_bathroom': [2.0],
            'latitude': [-33.4],
            'longitude': [-70.55],
            'price': [15000]
        })

        is_valid, violations = validate_against_schema(single_valid, self.schema_path)
        self.assertTrue(is_valid)
        self.assertEqual(len(violations), 0)

    def test_single_invalid_value(self):
        """Test validation with a single invalid value."""
        single_invalid = pd.DataFrame({
            'type': ['mansion'],
            'sector': ['invalid_sector'],
            'net_usable_area': [500000.0],
            'net_area': [200.0],
            'n_rooms': [3.0],
            'n_bathroom': [2.0],
            'latitude': [-33.4],
            'longitude': [-70.55],
            'price': [15000]
        })

        is_valid, violations = validate_against_schema(single_invalid, self.schema_path)
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)

    def test_batch_valid_values(self):
        """Test validation with batch valid values."""
        batch_valid = pd.DataFrame({
            'type': ['departamento', 'casa', 'departamento'],
            'sector': ['las condes', 'vitacura', 'providencia'],
            'net_usable_area': [120.0, 200.0, 80.0],
            'net_area': [150.0, 300.0, 100.0],
            'n_rooms': [3.0, 4.0, 2.0],
            'n_bathroom': [2.0, 3.0, 1.0],
            'latitude': [-33.4, -33.35, -33.42],
            'longitude': [-70.55, -70.52, -70.58],
            'price': [15000, 25000, 12000]
        })

        is_valid, violations = validate_against_schema(batch_valid, self.schema_path)
        self.assertTrue(is_valid)
        self.assertEqual(len(violations), 0)

    def test_batch_mixed_values(self):
        """Test validation with batch mixed values."""
        batch_mixed = pd.DataFrame({
            'type': ['departamento', 'mansion', 'casa'],
            'sector': ['las condes', 'vitacura', 'invalid_place'],
            'net_usable_area': [120.0, 200.0, 999999.0],
            'net_area': [150.0, 300.0, 100.0],
            'n_rooms': [3.0, 4.0, 2.0],
            'n_bathroom': [2.0, 3.0, 1.0],
            'latitude': [-33.4, -33.35, -33.42],
            'longitude': [-70.55, -70.52, -70.58],
            'price': [15000, 25000, 12000]
        })

        is_valid, violations = validate_against_schema(batch_mixed, self.schema_path)
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)


if __name__ == '__main__':
    unittest.main()
