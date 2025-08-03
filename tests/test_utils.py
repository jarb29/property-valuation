"""
Tests for utility functionality.

This module contains tests for prediction logging and versioning.
"""

import os
import sys
import json
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.utils.logging import log_model_prediction
from src.config import LOG_DIR, DATA_VERSION, SCHEMA_DIR
from src.data.generate_schema import generate_schema, schema_to_json, schema_to_csv
from src.data.outlier_handler import separate_outliers_and_save
from src.models.serialization import save_model, save_model_metadata


class TestPredictionLogging(unittest.TestCase):
    """Tests for prediction logging."""

    def setUp(self):
        """Set up test fixtures."""
        # Create LOG_DIR if it doesn't exist
        os.makedirs(LOG_DIR, exist_ok=True)

    def test_single_prediction_logging(self):
        """Test logging a single prediction."""
        # Simulate input data for a property valuation
        input_data = {
            "type": "house",
            "sector": "sector1",
            "net_usable_area": 100.0,
            "net_area": 120.0,
            "n_rooms": 3.0,
            "n_bathroom": 2.0,
            "latitude": -33.4,
            "longitude": -70.6
        }

        # Simulate a prediction
        prediction = 150000.0

        # Log the prediction
        log_model_prediction(
            model_name="property_valuation_test",
            input_data=input_data,
            prediction=prediction
        )

        # Check if the log file exists
        log_file = os.path.join(LOG_DIR, 'predictions.log')
        self.assertTrue(os.path.exists(log_file))

        # Read and check the last line of the log file
        with open(log_file, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            last_log = lines[-1]

            # Parse the JSON log entry
            log_data = json.loads(last_log)
            self.assertIn('timestamp', log_data)
            self.assertEqual(log_data.get('model'), 'property_valuation_test')
            self.assertIn('input', log_data)
            self.assertEqual(log_data.get('prediction'), 150000.0)

    def test_batch_prediction_logging(self):
        """Test logging batch predictions."""
        # Simulate batch input data
        batch_input = [
            {
                "type": "house",
                "sector": "sector1",
                "net_usable_area": 100.0,
                "net_area": 120.0,
                "n_rooms": 3.0,
                "n_bathroom": 2.0,
                "latitude": -33.4,
                "longitude": -70.6
            },
            {
                "type": "apartment",
                "sector": "sector2",
                "net_usable_area": 80.0,
                "net_area": 90.0,
                "n_rooms": 2.0,
                "n_bathroom": 1.0,
                "latitude": -33.5,
                "longitude": -70.7
            }
        ]

        # Simulate batch predictions
        batch_predictions = [150000.0, 120000.0]

        # Log each prediction in the batch
        for i, (instance, prediction) in enumerate(zip(batch_input, batch_predictions)):
            log_model_prediction(
                model_name="property_valuation_test",
                input_data=instance,
                prediction=prediction
            )

        # Check if the log file exists and count entries
        log_file = os.path.join(LOG_DIR, 'predictions.log')
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, 'r') as f:
            lines = f.readlines()
            self.assertGreaterEqual(len(lines), 2)


class TestVersioning(unittest.TestCase):
    """Tests for versioning system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories for test outputs
        self.test_output_dir = os.path.join('tests', 'temp_outputs', 'versioning')
        self.test_schema_dir = os.path.join(self.test_output_dir, 'schema')
        self.test_model_dir = os.path.join(self.test_output_dir, 'models')
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.test_schema_dir, exist_ok=True)
        os.makedirs(self.test_model_dir, exist_ok=True)

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_schema_versioning(self):
        """Test schema generation and versioning."""
        # Create a simple DataFrame
        df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.rand(100)
        })

        # Generate schema
        schema = generate_schema(df, 'test_data')

        # Save schema with versioning
        json_path = schema_to_json(schema, base_name='test', schema_dir=self.test_schema_dir)
        csv_path = schema_to_csv(schema, base_name='test', schema_dir=self.test_schema_dir)

        # Check that files were created
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(csv_path))

        # Save again to test versioning increment
        json_path2 = schema_to_json(schema, base_name='test', schema_dir=self.test_schema_dir)
        csv_path2 = schema_to_csv(schema, base_name='test', schema_dir=self.test_schema_dir)

        # Check that new files were created with incremented version
        self.assertTrue(os.path.exists(json_path2))
        self.assertTrue(os.path.exists(csv_path2))
        self.assertNotEqual(json_path, json_path2)
        self.assertNotEqual(csv_path, csv_path2)

    def test_outlier_processing_versioning(self):
        """Test outlier processing and versioning."""
        # Create a simple DataFrame with outliers
        np.random.seed(42)
        df = pd.DataFrame({
            'type': ['type1'] * 95 + ['type2'] * 5,
            'sector': ['sector1'] * 95 + ['sector2'] * 5,
            'net_usable_area': np.random.rand(100) * 100,
            'net_area': np.random.rand(100) * 150,
            'n_rooms': np.random.randint(1, 5, 100),
            'n_bathroom': np.random.randint(1, 3, 100),
            'latitude': np.random.rand(100) * 10,
            'longitude': np.random.rand(100) * 10,
            'price': np.random.rand(100) * 1000
        })

        # Add some outliers
        df.loc[95:, 'price'] = df.loc[95:, 'price'] * 10

        # Save the DataFrame to a temporary file
        temp_file = os.path.join(self.test_output_dir, 'temp_test_data.csv')
        df.to_csv(temp_file, index=False)

        # Process outliers with versioning
        clean_data, outlier_data = separate_outliers_and_save(
            data_path=temp_file,
            output_dir=self.test_output_dir
        )

        # Check results
        self.assertIsInstance(clean_data, pd.DataFrame)
        self.assertIsInstance(outlier_data, pd.DataFrame)
        self.assertGreater(len(clean_data), 0)
        self.assertGreaterEqual(len(outlier_data), 0)

    def test_model_versioning(self):
        """Test model saving and versioning."""
        # Create a simple model
        model = LinearRegression()
        X = np.random.rand(100, 2)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.rand(100)
        model.fit(X, y)

        # Save model with versioning
        model_path = save_model(
            model=model,
            model_type='linear_regression',
            base_name='linear_regression',
            model_dir=self.test_model_dir
        )

        # Check that model file was created
        self.assertTrue(os.path.exists(model_path))

        # Save metadata
        metadata = {
            'model_type': 'linear_regression',
            'features': ['feature1', 'feature2'],
            'target': 'target',
            'metrics': {
                'r2': 0.95,
                'mse': 0.05
            }
        }
        metadata_path = save_model_metadata(model_path, metadata)

        # Check that metadata file was created
        self.assertTrue(os.path.exists(metadata_path))

        # Save again to test versioning increment
        model_path2 = save_model(
            model=model,
            model_type='linear_regression',
            base_name='linear_regression',
            model_dir=self.test_model_dir
        )

        # Check that new model file was created with incremented version
        self.assertTrue(os.path.exists(model_path2))
        self.assertNotEqual(model_path, model_path2)

        # Save metadata again
        metadata_path2 = save_model_metadata(model_path2, metadata)

        # Check that new metadata file was created
        self.assertTrue(os.path.exists(metadata_path2))
        self.assertNotEqual(metadata_path, metadata_path2)


if __name__ == '__main__':
    unittest.main()