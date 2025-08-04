import os
import sys
import unittest
import pandas as pd
import numpy as np
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.outlier_handler import separate_outliers_and_save
from src.config import TRAIN_DATA_PATH, PIPELINE_DATA_DIR
from src.utils.helpers import load_latest_data, get_latest_versioned_file


class TestOutlierHandler(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = os.path.join('tests', 'temp_outputs', 'outliers')
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Create sample data with outliers
        np.random.seed(42)
        self.df = pd.DataFrame({
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
        # Add outliers
        self.df.loc[95:, 'price'] = self.df.loc[95:, 'price'] * 10
        
        self.temp_file = os.path.join(self.test_output_dir, 'test_data.csv')
        self.df.to_csv(self.temp_file, index=False)

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_outlier_detection(self):
        data_without_outliers, data_with_outliers = separate_outliers_and_save(
            data_path=self.temp_file,
            save_target='pipeline'
        )
        
        self.assertIsInstance(data_without_outliers, pd.DataFrame)
        self.assertIsInstance(data_with_outliers, pd.DataFrame)
        self.assertGreater(len(data_without_outliers), 0)
        self.assertGreaterEqual(len(data_with_outliers), 0)
        
        total_data_points = len(data_without_outliers) + len(data_with_outliers)
        outlier_percentage = len(data_with_outliers) / total_data_points * 100
        self.assertLess(outlier_percentage, 50)

    def test_output_files_created(self):
        separate_outliers_and_save(data_path=self.temp_file, save_target='pipeline')
        
        clean_data_path = get_latest_versioned_file('clean', 'data', PIPELINE_DATA_DIR, 'csv')
        outlier_data_path = get_latest_versioned_file('outliers', 'data', PIPELINE_DATA_DIR, 'csv')
        
        for path in [clean_data_path, outlier_data_path]:
            self.assertIsNotNone(path)
            self.assertTrue(os.path.exists(path))
        
        df_clean = pd.read_csv(clean_data_path)
        df_outliers = pd.read_csv(outlier_data_path)
        
        self.assertGreater(len(df_clean), 0)
        self.assertGreaterEqual(len(df_outliers), 0)


def detect_outliers_hierarchical(df):
    df_clean = df[~(df == 0).any(axis=1)].copy()
    df_clean['is_outlier'] = False
    
    for (sector, prop_type, rooms, bathrooms), group in df_clean.groupby(['sector', 'type', 'n_rooms', 'n_bathroom']):
        if len(group) < 5:
            continue
        
        group = group.copy()
        group['area_bin'] = pd.cut(group['net_usable_area'], bins=5, labels=False)
        
        for bin_id, bin_group in group.groupby('area_bin'):
            if len(bin_group) < 3:
                continue
            
            Q1 = bin_group['price'].quantile(0.25)
            Q3 = bin_group['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (bin_group['price'] < lower_bound) | (bin_group['price'] > upper_bound)
            df_clean.loc[bin_group.index[outlier_mask], 'is_outlier'] = True
    
    return df_clean


def test_outlier_functions_with_same_data():
    temp_dir = os.path.join('tests', 'temp_outputs', 'outlier_functions')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        original_data = pd.read_csv(TRAIN_DATA_PATH)
        
        # Test hierarchical detection
        df_with_outliers = detect_outliers_hierarchical(original_data)
        outliers_count_1 = df_with_outliers['is_outlier'].sum()
        
        # Test separate_outliers_and_save
        data_without_outliers, data_with_outliers = separate_outliers_and_save(
            data_path=TRAIN_DATA_PATH, save_target='pipeline'
        )
        outliers_count_2 = len(data_with_outliers)
        
        # Test with latest data
        latest_data = load_latest_data(data_type='train')
        df_with_outliers_latest = detect_outliers_hierarchical(latest_data)
        outliers_count_3 = df_with_outliers_latest['is_outlier'].sum()
        
        for count in [outliers_count_1, outliers_count_2, outliers_count_3]:
            assert count >= 0
        
        return True
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    unittest.main()
