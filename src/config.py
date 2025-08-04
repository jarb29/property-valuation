import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Pipeline directories
OUTPUT_PIPELINE_DIR = os.path.join(OUTPUT_DIR, "pipeline")
PIPELINE_MODELS_DIR = os.path.join(OUTPUT_PIPELINE_DIR, "models")
PIPELINE_SCHEMA_DIR = os.path.join(OUTPUT_PIPELINE_DIR, "schema")
PIPELINE_DATA_DIR = os.path.join(OUTPUT_PIPELINE_DIR, "data")
PIPELINE_LOGS_DIR = os.path.join(OUTPUT_PIPELINE_DIR, "logs")

# Jupyter directories
OUTPUT_JUPYTER_DIR = os.path.join(OUTPUT_DIR, "jupyter")
JUPYTER_MODELS_DIR = os.path.join(OUTPUT_JUPYTER_DIR, "models")
JUPYTER_SCHEMA_DIR = os.path.join(OUTPUT_JUPYTER_DIR, "schema")
JUPYTER_DATA_DIR = os.path.join(OUTPUT_JUPYTER_DIR, "data")
JUPYTER_LOGS_DIR = os.path.join(OUTPUT_JUPYTER_DIR, "logs")

# Prediction logs - organized by type
LOG_DIR = os.path.join(OUTPUT_DIR, "predictions")
API_LOGS_DIR = os.path.join(LOG_DIR, "api")
PREDICTION_LOGS_DIR = os.path.join(LOG_DIR, "predictions")
SCHEMA_VALIDATION_LOGS_DIR = os.path.join(LOG_DIR, "schema_validation")
ERROR_LOGS_DIR = os.path.join(LOG_DIR, "errors")
# Create directories
for directory in [OUTPUT_DIR, LOG_DIR, API_LOGS_DIR, PREDICTION_LOGS_DIR,
                 SCHEMA_VALIDATION_LOGS_DIR, ERROR_LOGS_DIR,
                 OUTPUT_PIPELINE_DIR, PIPELINE_MODELS_DIR, PIPELINE_SCHEMA_DIR,
                 PIPELINE_DATA_DIR, PIPELINE_LOGS_DIR, OUTPUT_JUPYTER_DIR,
                 JUPYTER_MODELS_DIR, JUPYTER_SCHEMA_DIR, JUPYTER_DATA_DIR,
                 JUPYTER_LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Environment variables
DATA_VERSION = os.getenv("DATA_VERSION", "v2")
MODEL_VERSION = os.getenv("MODEL_VERSION", DATA_VERSION)
MODEL_METRIC = os.getenv("MODEL_METRIC", "rmse")
MODEL_LOAD_TARGET = os.getenv("MODEL_LOAD_TARGET", "pipeline")
API_KEY = os.getenv("API_KEY", "default_api_key")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() in ("true", "1", "t")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# Data paths
DATA_VERSION_DIR = os.path.join(DATA_DIR, DATA_VERSION)
os.makedirs(DATA_VERSION_DIR, exist_ok=True)
TRAIN_DATA_PATH = os.path.join(DATA_VERSION_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_VERSION_DIR, "test.csv")

# Model paths
MODEL_DIR = PIPELINE_MODELS_DIR
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, f"model_{MODEL_VERSION}.pkl")