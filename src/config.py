"""
Configuration settings and environment variables for the ML project.

This module handles loading environment variables, setting default values,
and providing a centralized configuration for the entire application.
"""

import os
import glob
from pathlib import Path
from dotenv import load_dotenv

# 1. Load environment variables from .env
load_dotenv()

# 2. Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent

# 3. Original Data Directories
DATA_DIR = os.path.join(BASE_DIR, "data")


# 4. Outputs (two main subfolders: pipeline & jupyter)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
# LOG_DIR is used for general application logs (API logs, prediction logs, etc.)
# that are not specific to pipeline or jupyter components
LOG_DIR = os.path.join(OUTPUT_DIR, "predictions")


# Pipeline-specific output directories
OUTPUT_PIPELINE_DIR = os.path.join(OUTPUT_DIR, "pipeline")
PIPELINE_MODELS_DIR = os.path.join(OUTPUT_PIPELINE_DIR, "models")
PIPELINE_SCHEMA_DIR = os.path.join(OUTPUT_PIPELINE_DIR, "schema")
PIPELINE_DATA_DIR = os.path.join(OUTPUT_PIPELINE_DIR, "data")
PIPELINE_LOGS_DIR = os.path.join(OUTPUT_PIPELINE_DIR, "logs")  # For pipeline execution logs

# Jupyter-specific output directories
OUTPUT_JUPYTER_DIR = os.path.join(OUTPUT_DIR, "jupyter")
JUPYTER_MODELS_DIR = os.path.join(OUTPUT_JUPYTER_DIR, "models")
JUPYTER_SCHEMA_DIR = os.path.join(OUTPUT_JUPYTER_DIR, "schema")
JUPYTER_DATA_DIR = os.path.join(OUTPUT_JUPYTER_DIR, "data")
JUPYTER_LOGS_DIR = os.path.join(OUTPUT_JUPYTER_DIR, "logs")  # For jupyter notebook logs and results

# Prediction schema logs
PREDICTION_SCHEMA = os.path.join(LOG_DIR, "schema")
# 5. Ensure Directories Exist

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

os.makedirs(OUTPUT_PIPELINE_DIR, exist_ok=True)
os.makedirs(PIPELINE_MODELS_DIR, exist_ok=True)
os.makedirs(PIPELINE_SCHEMA_DIR, exist_ok=True)
os.makedirs(PIPELINE_DATA_DIR, exist_ok=True)
os.makedirs(PIPELINE_LOGS_DIR, exist_ok=True)

os.makedirs(OUTPUT_JUPYTER_DIR, exist_ok=True)
os.makedirs(JUPYTER_MODELS_DIR, exist_ok=True)
os.makedirs(JUPYTER_SCHEMA_DIR, exist_ok=True)
os.makedirs(JUPYTER_DATA_DIR, exist_ok=True)
os.makedirs(JUPYTER_LOGS_DIR, exist_ok=True)

os.makedirs(PREDICTION_SCHEMA, exist_ok=True)


data_version_dirs = [
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("v")
]
print(f"Found data version folders: {data_version_dirs}")
# 7. Data Version Settings
DATA_VERSION = os.getenv("DATA_VERSION", "v3")

# 8. Model Settings
API_KEY = os.getenv("API_KEY", "default_api_key")
MODEL_VERSION = os.getenv("MODEL_VERSION", DATA_VERSION)
MODEL_METRIC = os.getenv("MODEL_METRIC", "rmse")
MODEL_LOAD_TARGET = os.getenv("MODEL_LOAD_TARGET", "pipeline")
# If no version folders are found and user hasn't explicitly set DATA_VERSION, pick the first
if not data_version_dirs:
    if "DATA_VERSION" not in os.environ and data_version_dirs:
        DATA_VERSION = data_version_dirs[0]
else:
    # Create the default version folder if it doesn't exist
    os.makedirs(os.path.join(DATA_DIR, DATA_VERSION), exist_ok=True)

# 9. API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() in ("true", "1", "t")

# Here we define MODEL_DIR to fix any undefined references:
MODEL_DIR = PIPELINE_MODELS_DIR
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, f"model_{MODEL_VERSION}.pkl")

# 10. Data Settings
DATA_VERSION_DIR = os.path.join(DATA_DIR, DATA_VERSION)
print(f"Data version directory: {DATA_VERSION_DIR}")
TRAIN_DATA_PATH = os.path.join(DATA_VERSION_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_VERSION_DIR, "test.csv")

# 11. Logging Settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB by default
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))  # 5 backup files by default
