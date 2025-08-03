import os
import logging
import pandas as pd
from typing import Optional

from src.data.outlier_handler import separate_outliers_and_save
from src.data.generate_schema import generate_schema, schema_to_json, validate_against_schema
from src.utils.helpers import load_original_data, load_latest_data, get_latest_versioned_file
from src.config import PIPELINE_DATA_DIR, PIPELINE_SCHEMA_DIR

logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(self, output_dir: str = None, schema_dir: str = None):
        self.output_dir = output_dir or PIPELINE_DATA_DIR
        self.schema_dir = schema_dir or PIPELINE_SCHEMA_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.schema_dir, exist_ok=True)

    def _load_data(self, data_type: str, use_latest: bool) -> pd.DataFrame:
        return load_latest_data(data_type=data_type, outputs_dir=PIPELINE_DATA_DIR) if use_latest else load_original_data(data_type=data_type)

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[~(data == 0).any(axis=1)].copy()

    def _generate_schema(self, data: pd.DataFrame, base_name: str):
        schema = generate_schema(data, f"{base_name}_data")
        schema_to_json(schema, base_name=base_name, save_target='pipeline')
        return schema

    def _validate_data(self, data: pd.DataFrame, schema_path: Optional[str] = None) -> bool:
        if not schema_path:
            for directory in [os.path.join(PIPELINE_DATA_DIR, "schema"), self.schema_dir]:
                schema_path = get_latest_versioned_file('train', 'schema', directory, 'json')
                if schema_path:
                    break
        
        if not schema_path:
            return True
        
        is_valid, _ = validate_against_schema(data, schema_path, log_violations=True)
        return is_valid

    def run(self, data_type: str = 'train', use_latest: bool = False, validate: bool = True) -> pd.DataFrame:
        data = self._load_data(data_type, use_latest)
        
        if (use_latest and data_type == 'train') or data_type == 'test':
            if validate:
                self._validate_data(data)
            return data
        
        # Full processing for training data
        data = self._clean_data(data)
        data, _ = separate_outliers_and_save(data_path=None, save_target='pipeline')
        self._generate_schema(data, data_type)
        
        if validate:
            self._validate_data(data)
        
        return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = DataPipeline()
    train_data = pipeline.run('train', False, True)
    test_data = pipeline.run('test', False, True)
    print(f"Train: {train_data.shape}, Test: {test_data.shape}")