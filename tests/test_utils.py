import os
import sys
import unittest
import pandas as pd
import numpy as np
import shutil
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logging import log_model_prediction
from src.config import LOG_DIR, PREDICTION_LOGS_DIR
from src.data.generate_schema import generate_schema, schema_to_json
from src.data.outlier_handler import separate_outliers_and_save
from src.models.serialization import save_model, save_model_metadata


class TestPredictionLogging(unittest.TestCase):
    def setUp(self):
        os.makedirs(LOG_DIR, exist_ok=True)

    def test_batch_prediction_logging(self):
        batch_input = [
            {"type": "house", "sector": "sector1", "net_usable_area": 100.0,
             "net_area": 120.0, "n_rooms": 3.0, "n_bathroom": 2.0,
             "latitude": -33.4, "longitude": -70.6},
            {"type": "apartment", "sector": "sector2", "net_usable_area": 80.0,
             "net_area": 90.0, "n_rooms": 2.0, "n_bathroom": 1.0,
             "latitude": -33.5, "longitude": -70.7}
        ]
        batch_predictions = [150000.0, 120000.0]
        
        for instance, prediction in zip(batch_input, batch_predictions):
            log_model_prediction(
                model_name="property_valuation_test",
                input_data=instance,
                prediction=prediction,
                model_version="test_version",
                processing_time=0.1
            )
        
        log_file = os.path.join(PREDICTION_LOGS_DIR, 'predictions.log')
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, 'r') as f:
            lines = f.readlines()
            self.assertGreaterEqual(len(lines), 2)


class TestVersioning(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.join('tests', 'temp_outputs', 'versioning')
        self.test_schema_dir = os.path.join(self.test_output_dir, 'schema')
        self.test_model_dir = os.path.join(self.test_output_dir, 'models')
        for dir_path in [self.test_output_dir, self.test_schema_dir, self.test_model_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_schema_versioning(self):
        df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.rand(100)
        })
        
        schema = generate_schema(df, 'test_data')
        
        json_path = schema_to_json(schema, base_name='test', 
                                 filename=os.path.join(self.test_schema_dir, 'test_schema.json'))
        json_path2 = schema_to_json(schema, base_name='test', 
                                  filename=os.path.join(self.test_schema_dir, 'test_schema2.json'))
        
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(json_path2))
        self.assertNotEqual(json_path, json_path2)

    def test_outlier_processing_versioning(self):
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
        df.loc[95:, 'price'] = df.loc[95:, 'price'] * 10
        
        temp_file = os.path.join(self.test_output_dir, 'temp_test_data.csv')
        df.to_csv(temp_file, index=False)
        
        clean_data, outlier_data = separate_outliers_and_save(
            data_path=temp_file, save_target='pipeline'
        )
        
        self.assertIsInstance(clean_data, pd.DataFrame)
        self.assertIsInstance(outlier_data, pd.DataFrame)
        self.assertGreater(len(clean_data), 0)
        self.assertGreaterEqual(len(outlier_data), 0)

    def test_model_versioning(self):
        model = LinearRegression()
        X = np.random.rand(100, 2)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.rand(100)
        model.fit(X, y)
        
        model_path = save_model(
            model=model,
            model_type='linear_regression',
            base_name='linear_regression',
            path=os.path.join(self.test_model_dir, 'linear_regression.pkl')
        )
        
        metadata = {
            'model_type': 'linear_regression',
            'features': ['feature1', 'feature2'],
            'target': 'target',
            'metrics': {'r2': 0.95, 'mse': 0.05}
        }
        metadata_path = save_model_metadata(model_path, metadata)
        
        model_path2 = save_model(
            model=model,
            model_type='linear_regression',
            base_name='linear_regression',
            path=os.path.join(self.test_model_dir, 'linear_regression2.pkl')
        )
        metadata_path2 = save_model_metadata(model_path2, metadata)
        
        for path in [model_path, metadata_path, model_path2, metadata_path2]:
            self.assertTrue(os.path.exists(path))
        
        self.assertNotEqual(model_path, model_path2)
        self.assertNotEqual(metadata_path, metadata_path2)

if __name__ == '__main__':
    unittest.main()
