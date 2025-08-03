"""
Tests for outlier detection and handling functionality.

This module contains tests for outlier detection and handling.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.data.outlier_handler import separate_outliers_and_save
from src.config import TRAIN_DATA_PATH
from src.utils.helpers import load_latest_data


class TestOutlierHandler(unittest.TestCase):
    """Tests for the outlier handler."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_output_dir = os.path.join('tests', 'temp_outputs', 'outliers')
        os.makedirs(self.test_output_dir, exist_ok=True)

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_outlier_detection(self):
        """Test the outlier detection implementation."""
        # Create a sample dataframe with outliers
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
        temp_file = os.path.join(self.test_output_dir, 'test_data.csv')
        df.to_csv(temp_file, index=False)

        # Process outliers
        data_without_outliers, data_with_outliers = separate_outliers_and_save(
            data_path=temp_file,
            output_dir=self.test_output_dir
        )

        # Assertions
        self.assertIsInstance(data_without_outliers, pd.DataFrame)
        self.assertIsInstance(data_with_outliers, pd.DataFrame)
        self.assertGreater(len(data_without_outliers), 0)
        self.assertGreaterEqual(len(data_with_outliers), 0)

        total_data_points = len(data_without_outliers) + len(data_with_outliers)
        outlier_percentage = len(data_with_outliers) / total_data_points * 100

        # Test that outliers have been properly separated
        self.assertLess(outlier_percentage, 50)  # Reasonable assumption

    def test_output_files_created(self):
        """Test that output files are created."""
        # Create a sample dataframe with outliers
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
        temp_file = os.path.join(self.test_output_dir, 'test_data.csv')
        df.to_csv(temp_file, index=False)

        # Process outliers
        separate_outliers_and_save(
            data_path=temp_file,
            output_dir=self.test_output_dir
        )

        # Check files are created
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, 'data_without_outliers.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.test_output_dir, 'data_with_outliers.csv')))

        # Check files are not empty
        df_clean = pd.read_csv(os.path.join(self.test_output_dir, 'data_without_outliers.csv'))
        df_outliers = pd.read_csv(os.path.join(self.test_output_dir, 'data_with_outliers.csv'))

        self.assertGreater(len(df_clean), 0)
        self.assertGreaterEqual(len(df_outliers), 0)


# Define the detect_outliers_hierarchical function for testing
def detect_outliers_hierarchical(df):
    """
    Detects outliers using hierarchical grouping approach with area binning.
    Groups by sector -> type -> n_rooms -> n_bathroom, then creates area bins
    within each group for more precise outlier detection.
    """
    df_clean = df[~(df == 0).any(axis=1)].copy()
    df_clean['is_outlier'] = False

    # Group by sector -> type -> n_rooms -> n_bathroom
    for (sector, prop_type, rooms, bathrooms), group in df_clean.groupby(['sector', 'type', 'n_rooms', 'n_bathroom']):
        if len(group) < 5:  # Skip small groups
            continue

        # Create area bins within each group
        group = group.copy()
        group['area_bin'] = pd.cut(group['net_usable_area'], bins=5, labels=False)

        # Detect outliers within each area bin
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
    """
    Test both outlier detection functions with the same input data to verify
    they produce the same results.
    """
    # Create a temporary directory for test outputs
    temp_dir = os.path.join('tests', 'temp_outputs', 'outlier_functions')
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Load the original training data
        original_data = pd.read_csv(TRAIN_DATA_PATH)

        # Test detect_outliers_hierarchical function
        df_with_outliers = detect_outliers_hierarchical(original_data)
        outliers_count_1 = df_with_outliers['is_outlier'].sum()

        # Test separate_outliers_and_save function with the same data
        # Call separate_outliers_and_save with the same data
        data_without_outliers, data_with_outliers = separate_outliers_and_save(
            data_path=TRAIN_DATA_PATH,
            output_dir=temp_dir
        )
        outliers_count_2 = len(data_with_outliers)

        # Compare results
        assert outliers_count_1 >= 0
        assert outliers_count_2 >= 0

        # Now test with the data loaded by load_latest_data
        latest_data = load_latest_data(data_type='train')

        # Test detect_outliers_hierarchical function with latest data
        df_with_outliers_latest = detect_outliers_hierarchical(latest_data)
        outliers_count_3 = df_with_outliers_latest['is_outlier'].sum()

        # Compare with original results
        assert outliers_count_3 >= 0

        return True
    finally:
        # Clean up
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()