"""
Data Pipeline Module.

This module implements the data pipeline for the property valuation model.
It includes functions for data loading, cleaning, outlier detection, and schema validation.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any

from src.data.outlier_handler import separate_outliers_and_save
from src.data.generate_schema import generate_schema, schema_to_json, validate_against_schema
from src.utils.helpers import load_original_data, load_latest_data, get_latest_versioned_file
from src.config import PIPELINE_DATA_DIR, PIPELINE_SCHEMA_DIR

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Data pipeline for the property valuation model.

    This class handles the data loading, cleaning, outlier detection, and schema validation
    steps of the pipeline.
    """

    def __init__(self, output_dir: str = None, schema_dir: str = None):
        """
        Initialize the data pipeline.

        Args:
            output_dir (str, optional): Directory to save output files. Defaults to PIPELINE_DATA_DIR from config.
            schema_dir (str, optional): Directory to save schema files. Defaults to PIPELINE_SCHEMA_DIR from config.
        """
        self.output_dir = output_dir if output_dir is not None else PIPELINE_DATA_DIR
        self.schema_dir = schema_dir if schema_dir is not None else PIPELINE_SCHEMA_DIR

        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.schema_dir, exist_ok=True)

        logger.info(f"Initialized DataPipeline with output_dir={self.output_dir}, schema_dir={self.schema_dir}")

    def load_data(self, data_type: str = 'train', use_latest: bool = True) -> pd.DataFrame:
        """
        Load data for the pipeline.

        Args:
            data_type (str): Type of data to load ('train' or 'test').
            use_latest (bool): Whether to use the latest processed data or the original data.

        Returns:
            pd.DataFrame: The loaded data.
        """
        if use_latest:
            logger.info(f"Loading latest {data_type} data")
            return load_latest_data(data_type=data_type, outputs_dir=PIPELINE_DATA_DIR)
        else:
            logger.info(f"Loading original {data_type} data")
            return load_original_data(data_type=data_type)

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by removing zero values.

        Args:
            data (pd.DataFrame): The data to clean.

        Returns:
            pd.DataFrame: The cleaned data.
        """
        logger.info("Cleaning data (removing zero values)")
        return data[~(data == 0).any(axis=1)].copy()

    def detect_and_handle_outliers(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect and handle outliers in the data.

        Args:
            data (pd.DataFrame): The data to process.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (data_without_outliers, data_with_outliers)
        """
        logger.info("Detecting and handling outliers")
        return separate_outliers_and_save(data_path=None, save_target='pipeline')

    def generate_and_save_schema(self, data: pd.DataFrame, dataset_name: str, base_name: str) -> Dict[str, Any]:
        """
        Generate and save a schema for the data.

        Args:
            data (pd.DataFrame): The data to generate a schema for.
            dataset_name (str): Name of the dataset.
            base_name (str): Base name for the schema file.

        Returns:
            Dict[str, Any]: The generated schema.
        """
        logger.info(f"Generating schema for {dataset_name}")
        schema = generate_schema(data, dataset_name)


        # Save schema to the pipeline schema directory
        schema_path = schema_to_json(schema, base_name=base_name, save_target='pipeline')
        logger.info(f"Schema saved to {schema_path}")
        return schema

    def validate_data(self, data: pd.DataFrame, schema_path: Optional[str] = None) -> bool:
        """
        Validate data against a schema.

        Args:
            data (pd.DataFrame): The data to validate.
            schema_path (str, optional): Path to the schema file. If None, the latest schema is used.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        if schema_path is None:
            # First try to find schema in the pipeline schema directory
            pipeline_schema_dir = os.path.join(PIPELINE_DATA_DIR, "schema")
            schema_path = get_latest_versioned_file(
                base_name='train',
                file_type='schema',
                directory=pipeline_schema_dir,
                extension='json'
            )

            # If not found, try the default schema directory
            if schema_path is None:
                schema_path = get_latest_versioned_file(
                    base_name='train',
                    file_type='schema',
                    directory=self.schema_dir,
                    extension='json'
                )

        if schema_path is None:
            logger.warning("No schema file found. Skipping validation.")
            return True

        logger.info(f"Validating data against schema {schema_path}")
        is_valid, violations = validate_against_schema(data, schema_path, log_violations=True)

        if not is_valid:
            logger.warning(f"Data validation failed with {len(violations)} violations")
        else:
            logger.info("Data validation passed")

        return is_valid

    def run(self, data_type: str = 'train', use_latest: bool = False, validate: bool = True) -> pd.DataFrame:
        """
        Run the full data pipeline.

        Args:
            data_type (str): Type of data to process ('train' or 'test').
            use_latest (bool): Whether to use the latest processed data or the original data.
            validate (bool): Whether to validate the data against a schema.

        Returns:
            pd.DataFrame: The processed data.
        """
        logger.info(f"Running data pipeline for {data_type} data")

        # Step 1: Load data
        data = self.load_data(data_type=data_type, use_latest=use_latest)
        logger.info(f"Loaded data with shape {data.shape}")

        # Step 2: Clean data
        data_clean = self.clean_data(data)
        logger.info(f"Cleaned data with shape {data_clean.shape}")

        # Step 3: Detect and handle outliers (for training data only)
        if data_type == 'train':
            data_without_outliers, data_with_outliers = self.detect_and_handle_outliers(data_clean)
            logger.info(f"Separated data into {len(data_without_outliers)} clean samples and {len(data_with_outliers)} outliers")
            processed_data = data_without_outliers
        else:
            processed_data = data_clean

        # Step 4: Generate and save schema (for training data only)
        if data_type == 'train':
            schema = self.generate_and_save_schema(
                processed_data,
                dataset_name=f"{data_type}_data",
                base_name=data_type
            )

        # Step 5: Validate data (if requested)
        if validate:
            is_valid = self.validate_data(processed_data)
            if not is_valid:
                logger.warning("Data validation failed, but continuing with pipeline")

        logger.info(f"Data pipeline completed successfully for {data_type} data")
        return processed_data


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the pipeline
    pipeline = DataPipeline()
    train_data = pipeline.run(data_type='train', use_latest=False, validate=True)
    test_data = pipeline.run(data_type='test', use_latest=False, validate=True)

    print(f"Processed training data shape: {train_data.shape}")
    print(f"Processed test data shape: {test_data.shape}")
