import os
import sys
import unittest
import pandas as pd
from unittest.mock import patch
import json
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.helpers import load_latest_data, load_original_data
from src.pipeline.data_pipeline import DataPipeline
from src.data.generate_schema import validate_against_schema


class TestDataLoading(unittest.TestCase):
    def test_data_loading(self):
        for load_func in [load_original_data, load_latest_data]:
            data = load_func(data_type='train')
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.join('tests', 'temp_outputs', 'data_pipeline')
        os.makedirs(self.test_output_dir, exist_ok=True)
        self.pipeline = DataPipeline(output_dir=self.test_output_dir, schema_dir=self.test_output_dir)
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'type': ['house', 'apartment', 'house', 'apartment', 'house'],
            'sector': ['sector1', 'sector2', 'sector1', 'sector3', 'sector2'],
            'net_usable_area': [100.0, 80.0, 120.0, 0.0, 90.0],
            'net_area': [120.0, 90.0, 140.0, 70.0, 110.0],
            'n_rooms': [3.0, 2.0, 4.0, 1.0, 3.0],
            'n_bathroom': [2.0, 1.0, 2.0, 1.0, 2.0],
            'latitude': [-33.4, -33.5, -33.4, -33.6, -33.5],
            'longitude': [-70.6, -70.7, -70.6, -70.8, -70.7],
            'price': [150000, 120000, 180000, 90000, 140000]
        })

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    @patch('src.pipeline.data_pipeline.load_original_data')
    @patch('src.pipeline.data_pipeline.load_latest_data')
    def test_load_data(self, mock_load_latest, mock_load_original):
        mock_load_original.return_value = self.sample_data
        mock_load_latest.return_value = self.sample_data
        
        result_original = self.pipeline._load_data(data_type='train', use_latest=False)
        result_latest = self.pipeline._load_data(data_type='train', use_latest=True)
        
        self.assertEqual(len(result_original), 5)
        self.assertEqual(len(result_latest), 5)
        mock_load_original.assert_called_once_with(data_type='train')
        mock_load_latest.assert_called_once()

    def test_clean_data(self):
        result = self.pipeline._clean_data(self.sample_data)
        self.assertEqual(len(result), 4)
        self.assertNotIn(0.0, result['net_usable_area'].values)

    @patch('src.pipeline.data_pipeline.separate_outliers_and_save')
    def test_detect_and_handle_outliers(self, mock_separate_outliers):
        clean_data = self.sample_data[self.sample_data['net_usable_area'] > 0]
        mock_separate_outliers.return_value = (clean_data, pd.DataFrame())
        
        clean_result, outlier_result = mock_separate_outliers(data_path=None, save_target='pipeline')
        
        self.assertGreater(len(clean_result), 0)
        self.assertIsInstance(clean_result, pd.DataFrame)
        self.assertIsInstance(outlier_result, pd.DataFrame)
        mock_separate_outliers.assert_called_once_with(data_path=None, save_target='pipeline')

    @patch('src.pipeline.data_pipeline.generate_schema')
    @patch('src.pipeline.data_pipeline.schema_to_json')
    def test_generate_and_save_schema(self, mock_schema_to_json, mock_generate_schema):
        mock_schema = {'dataset_name': 'test_data', 'columns': {}}
        mock_generate_schema.return_value = mock_schema
        mock_schema_to_json.return_value = os.path.join(self.test_output_dir, 'schema.json')
        
        result = self.pipeline._generate_schema(self.sample_data, base_name='test')
        
        self.assertEqual(result, mock_schema)
        mock_generate_schema.assert_called_once_with(self.sample_data, 'test_data')
        mock_schema_to_json.assert_called_once_with(mock_schema, base_name='test', save_target='pipeline')

    @patch('src.pipeline.data_pipeline.validate_against_schema')
    @patch('src.pipeline.data_pipeline.get_latest_versioned_file')
    def test_validate_data(self, mock_get_latest, mock_validate):
        schema_path = os.path.join(self.test_output_dir, 'schema.json')
        mock_get_latest.return_value = schema_path
        mock_validate.return_value = (True, [])
        
        result = self.pipeline._validate_data(self.sample_data)
        
        self.assertTrue(result)
        mock_get_latest.assert_called_once()
        mock_validate.assert_called_once_with(self.sample_data, schema_path, log_violations=True)

    @patch('src.pipeline.data_pipeline.DataPipeline._load_data')
    @patch('src.pipeline.data_pipeline.DataPipeline._clean_data')
    @patch('src.pipeline.data_pipeline.separate_outliers_and_save')
    @patch('src.pipeline.data_pipeline.DataPipeline._generate_schema')
    @patch('src.pipeline.data_pipeline.DataPipeline._validate_data')
    def test_run_pipeline_train(self, mock_validate, mock_schema, mock_outliers, mock_clean, mock_load):
        clean_data = self.sample_data[self.sample_data['net_usable_area'] > 0]
        mock_load.return_value = self.sample_data
        mock_clean.return_value = clean_data
        mock_outliers.return_value = (clean_data, pd.DataFrame())
        mock_schema.return_value = {'dataset_name': 'train_data', 'columns': {}}
        mock_validate.return_value = True
        
        result = self.pipeline.run(data_type='train', use_latest=False, validate=True)
        
        self.assertEqual(len(result), 4)
        for mock in [mock_load, mock_clean, mock_schema, mock_validate]:
            mock.assert_called_once()
        mock_outliers.assert_called_once_with(data_path=None, save_target='pipeline')


class TestSchemaValidation(unittest.TestCase):
    def setUp(self):
        self.schema_path = os.path.join('tests', 'temp_schema.json')
        schema = {
            "dataset_name": "test_data",
            "columns": {
                "type": {"data_type": "object", "unique_values_list": ["departamento", "casa"]},
                "sector": {"data_type": "object", "unique_values_list": ["las condes", "vitacura", "providencia"]},
                "net_usable_area": {"data_type": "float64", "numerical": {"min": 0.0, "max": 500.0}},
                "net_area": {"data_type": "float64", "numerical": {"min": 0.0, "max": 1000.0}},
                "n_rooms": {"data_type": "float64", "numerical": {"min": 0.0, "max": 10.0}},
                "n_bathroom": {"data_type": "float64", "numerical": {"min": 0.0, "max": 10.0}},
                "latitude": {"data_type": "float64", "numerical": {"min": -90.0, "max": 90.0}},
                "longitude": {"data_type": "float64", "numerical": {"min": -180.0, "max": 180.0}},
                "price": {"data_type": "int64", "numerical": {"min": 0, "max": 1000000}}
            }
        }
        with open(self.schema_path, 'w') as f:
            json.dump(schema, f)

    def tearDown(self):
        if os.path.exists(self.schema_path):
            os.remove(self.schema_path)

    def test_validation_single_values(self):
        valid_data = pd.DataFrame({
            'type': ['departamento'], 'sector': ['las condes'], 'net_usable_area': [120.0],
            'net_area': [150.0], 'n_rooms': [3.0], 'n_bathroom': [2.0],
            'latitude': [-33.4], 'longitude': [-70.55], 'price': [15000]
        })
        invalid_data = pd.DataFrame({
            'type': ['mansion'], 'sector': ['invalid_sector'], 'net_usable_area': [500000.0],
            'net_area': [200.0], 'n_rooms': [3.0], 'n_bathroom': [2.0],
            'latitude': [-33.4], 'longitude': [-70.55], 'price': [15000]
        })
        
        is_valid, violations = validate_against_schema(valid_data, self.schema_path)
        self.assertTrue(is_valid)
        self.assertEqual(len(violations), 0)
        
        is_valid, violations = validate_against_schema(invalid_data, self.schema_path)
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)

    def test_validation_batch_values(self):
        valid_batch = pd.DataFrame({
            'type': ['departamento', 'casa', 'departamento'],
            'sector': ['las condes', 'vitacura', 'providencia'],
            'net_usable_area': [120.0, 200.0, 80.0], 'net_area': [150.0, 300.0, 100.0],
            'n_rooms': [3.0, 4.0, 2.0], 'n_bathroom': [2.0, 3.0, 1.0],
            'latitude': [-33.4, -33.35, -33.42], 'longitude': [-70.55, -70.52, -70.58],
            'price': [15000, 25000, 12000]
        })
        mixed_batch = pd.DataFrame({
            'type': ['departamento', 'mansion', 'casa'],
            'sector': ['las condes', 'vitacura', 'invalid_place'],
            'net_usable_area': [120.0, 200.0, 999999.0], 'net_area': [150.0, 300.0, 100.0],
            'n_rooms': [3.0, 4.0, 2.0], 'n_bathroom': [2.0, 3.0, 1.0],
            'latitude': [-33.4, -33.35, -33.42], 'longitude': [-70.55, -70.52, -70.58],
            'price': [15000, 25000, 12000]
        })
        
        is_valid, violations = validate_against_schema(valid_batch, self.schema_path)
        self.assertTrue(is_valid)
        self.assertEqual(len(violations), 0)
        
        is_valid, violations = validate_against_schema(mixed_batch, self.schema_path)
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)


if __name__ == '__main__':
    unittest.main()
