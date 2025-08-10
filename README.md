# Property Valuation ML System

Machine learning system for Chilean real estate property valuation.

## 🚀 Quick Start

Get up and running in minutes:

```bash
git clone https://github.com/jarb29/property-valuation
cd property-valuation

# Create data directory and add your data
mkdir -p data/v1
# Copy your train.csv and test.csv to data/v1/

# Create minimal .env file
cat > .env << EOF
API_HOST=0.0.0.0
API_PORT=8000
EOF

# Run the ML pipeline first
docker-compose --profile pipeline up pipeline

# Then start the API
docker-compose up api
```

## 🔧 Model Configuration

The system uses environment variables for configuration. The minimal `.env` file above contains only the essential settings needed for Docker operation.

## Overview

This project implements an end-to-end machine learning solution for accurate property valuation. It combines data processing, model training, and a RESTful API to provide reliable property price predictions based on various features such as location, size, and property characteristics.

## Features

- **Machine Learning Pipeline**

  - Automated data processing and feature engineering
  - Multiple model options (Gradient Boosting, Random Forest, Linear Regression)
  - Comprehensive model evaluation and performance metrics
  - Version-controlled data and model management

- **RESTful API**

  - Single property valuation endpoint
  - Batch prediction for multiple properties
  - Model information and health check endpoints
  - Comprehensive input validation and error handling

- **Robust Infrastructure**
  - Containerized deployment with Docker
  - Configurable through environment variables
  - Extensive logging and monitoring
  - Development and production environments

## Architecture

The system is organized into several key components:

```
├── data/               # Version-controlled datasets
│   ├── v1/             # Data version 1 (train.csv, test.csv)
│   ├── v2/             # Data version 2
│   └── v3/             # Data version 3 (current default)
├── docs/               # Documentation files
├── notebooks/          # Jupyter notebooks for exploration
├── outputs/            # Model outputs, logs, and schemas
│   ├── pipeline/       # Pipeline-generated artifacts
│   │   ├── models/     # Trained models (model_v3.pkl)
│   │   ├── schema/     # Data schemas (v3_schema_train.json)
│   │   ├── data/       # Processed data files
│   │   └── logs/       # Pipeline execution logs
│   ├── jupyter/        # Notebook-generated artifacts
│   └── predictions/    # API prediction logs
├── scripts/            # Execution scripts
├── src/                # Source code
│   ├── api/            # API implementation
│   ├── data/           # Data processing modules
│   ├── models/         # ML model implementations
│   ├── pipeline/       # Pipeline components
│   └── utils/          # Utility functions
└── tests/              # Test suite
```

## Installation

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)

### Local Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/jarb29/property-valuation
   cd property-valuation
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (copy from example):
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Docker Setup

1. Build and start the containers:

   ```bash
   docker-compose up -d
   ```

2. For development mode with hot reload:
   ```bash
   docker-compose --profile dev up -d
   ```

## Usage

### Running the API

```bash
# Local development with auto-reload
python scripts/run_api.py --reload

# Specify host and port
python scripts/run_api.py --host 0.0.0.0 --port 8000

# Production mode with multiple workers
python scripts/run_api.py --workers 4
```

### Training Models

```bash
# Run the default pipeline (uses DATA_VERSION=v3 by default)
python scripts/pipeline.py

# Specify model type
python scripts/pipeline.py --model-type random_forest

# Validate data during processing
python scripts/pipeline.py --validate-data

# Use a specific data version (set environment variable)
DATA_VERSION=v2 python scripts/pipeline.py
```

### Using Docker Profiles

```bash
# Run the API service
docker-compose up api
or # Use port 8001 instead
API_PORT=8001 docker-compose up api
# Run the development API with hot reload
docker-compose --profile dev up api-dev

# Run the full ML pipeline
docker-compose --profile pipeline up pipeline



# Specify a different data version
docker-compose -e DATA_VERSION=v2 --profile pipeline up pipeline
```

## API Documentation

### Endpoints

> **Note**: The API endpoints include the data version in their paths (e.g., `/api/v3/...`). This version matches the `DATA_VERSION` environment variable.

- **POST /api/v3/predictions**

  - Single property valuation
  - Requires property features (type, sector, area, rooms, etc.)
  - Returns predicted price and model metadata

- **POST /api/v3/predictions/batch**

  - Batch property valuation
  - Accepts multiple property records
  - Returns array of predictions with metadata

- **GET /api/v3/model/info**

  - Retrieves model information
  - Returns model version, features, and description

- **GET /api/v3/health**
  - System health check
  - Verifies API and model availability

When changing the `DATA_VERSION` (e.g., to `v2`), the API endpoints will automatically update to match (e.g., `/api/v2/predictions`).

### Example Request (Single Prediction)

```json
{
  "features": {
    "type": "departamento",
    "sector": "las condes",
    "net_usable_area": 120.5,
    "net_area": 150.0,
    "n_rooms": 3,
    "n_bathroom": 2,
    "latitude": -33.4172,
    "longitude": -70.5476
  }
}
```

### Example Response

```json
{
  "prediction": 185000000,
  "prediction_time": 0.0234,
  "model_version": "best_mae_pipeline"
}
```

## Data Versioning System

The system implements a robust data versioning mechanism that ensures consistency between datasets, models, and API endpoints:

### How It Works

1. **Version-Based Directory Structure**:

   - Data is organized in version-specific directories: `data/v1`, `data/v2`, `data/v3`, etc.
   - Each version directory contains its own `train.csv` and `test.csv` files
   - This structure allows multiple data versions to coexist without conflicts

2. **Version Selection**:

   - The active data version is controlled by the `DATA_VERSION` environment variable (default: `v3`)
   - When the system starts, it automatically detects available data versions
   - If the specified version doesn't exist, it's created automatically

3. **Version-Based Outputs**:

   - All outputs (models, schemas, processed data) are named according to the active data version
   - Models are saved as `model_{MODEL_VERSION}.pkl` (e.g., `model_v3.pkl`)
   - API endpoints include the version in their paths (e.g., `/api/v3/predictions`)

4. **Version Relationships**:
   ```
   DATA_VERSION (e.g., v3)
        │
        ├─── Determines data paths: data/v3/train.csv, data/v3/test.csv
        │
        ├─── Defaults MODEL_VERSION (if not explicitly set)
        │     │
        │     └─── Determines model file names: model_v3.pkl
        │
        └─── Determines API endpoint paths: /api/v3/predictions
   ```

### Benefits

- **Reproducibility**: Each version's complete pipeline can be reproduced independently
- **Parallel Development**: Multiple data versions can be developed and tested simultaneously
- **Backward Compatibility**: Older model versions remain accessible even as new ones are developed
- **Traceability**: Clear lineage from data to models to API endpoints

### Usage Examples

To use a specific data version:

```bash
# Set in .env file
DATA_VERSION=v2

# Or set as environment variable
export DATA_VERSION=v2
python scripts/pipeline.py

# Or pass directly to Docker
docker-compose -e DATA_VERSION=v2 up pipeline
```

To use different versions for data and models:

```bash
# Use v2 data but v1 model version
export DATA_VERSION=v2
export MODEL_VERSION=v1
python scripts/pipeline.py
```

## Configuration

The system is highly configurable through environment variables:

### Data and Model Settings

| Variable            | Description                | Default                | Notes                                          |
| ------------------- | -------------------------- | ---------------------- | ---------------------------------------------- |
| `DATA_VERSION`      | Version of data to use     | `v3`                   | Controls which data folder is used (`data/v3`) |
| `MODEL_VERSION`     | Version of model to use    | Same as `DATA_VERSION` | Used in model filenames (`model_v3.pkl`)       |
| `MODEL_METRIC`      | Metric for model selection | `rmse`                 | Options: `rmse`, `mae`, `r2`                   |
| `MODEL_LOAD_TARGET` | Source of models           | `pipeline`             | Options: `pipeline`, `jupyter`                 |

### API Settings

| Variable      | Description                | Default           |
| ------------- | -------------------------- | ----------------- |
| `API_HOST`    | Host for the API server    | `0.0.0.0`         |
| `API_PORT`    | Port for the API server    | `8000`            |
| `API_WORKERS` | Number of worker processes | `1`               |
| `API_DEBUG`   | Enable debug mode          | `False`           |
| `API_KEY`     | API authentication key     | `default_api_key` |

### Logging Settings

| Variable           | Description                | Default           |
| ------------------ | -------------------------- | ----------------- |
| `LOG_LEVEL`        | Logging level              | `INFO`            |
| `LOG_MAX_BYTES`    | Maximum log file size      | `10485760` (10MB) |
| `LOG_BACKUP_COUNT` | Number of backup log files | `5`               |

## Docker Configuration

The project includes a Docker setup with multiple services:

- **api**: Main API service for production

  - Resource limits: 1 CPU, 1GB memory
  - Production environment with 2 workers
  - Read-only access to data directory
  - Built-in health check

- **api-dev**: Development API with hot reload

  - Debug mode enabled
  - Hot reload for code changes
  - Detailed logging (DEBUG level)
  - Mounted source code for development

- **pipeline**: Service for running the full ML pipeline
  - Resource limits: 2 CPUs, 2GB memory
  - Runs the complete data processing and model training pipeline
  - Persistent storage for outputs and models

### Advanced Docker Features

The Docker configuration includes several advanced features:

- **Multi-stage builds**: Reduces final image size by separating build and runtime dependencies
- **Named volumes**: Ensures data persistence between container restarts
  - `model_data`: Stores trained models
  - `app_outputs`: Stores application outputs
- **Custom network**: Isolates container communication
- **Resource limits**: Prevents resource contention
- **Logging configuration**: Manages log file size and rotation
- **Health checks**: Monitors service availability
- **Container naming**: Simplifies container management

## Documentation

This project uses MkDocs with the Material theme to provide comprehensive, searchable, and well-organized documentation.

### MkDocs Setup

The documentation is built using [MkDocs](https://www.mkdocs.org/), a fast and simple static site generator designed specifically for project documentation:

1. **Installation**:

   ```bash
   # Install MkDocs and required plugins
   pip install -r requirements-docs.txt
   ```

2. **Local Development**:

   ```bash
   # Start the MkDocs development server
   mkdocs serve
   ```

   This will start a local server at `http://127.0.0.1:8000/` that automatically reloads when you make changes.

3. **Building the Documentation**:
   ```bash
   # Build the static site
   mkdocs build
   ```
   This creates a `site` directory with the built HTML files.

### Documentation Structure

- `mkdocs.yml`: Configuration file in the project root
- `docs/`: Documentation source files
  - `index.md`: Main landing page
  - `getting-started.md`: Quick start guide
  - `installation-guide.md`: Detailed installation instructions
  - `user-manual.md`: Comprehensive user guide
  - `api-documentation.md`: API reference
  - `github-pages-deployment.md`: Guide for deploying to GitHub Pages
  - `Challenge.md`: Original project requirements
  - `assets/css/extra.css`: Custom CSS styles

### GitHub Pages Deployment

The documentation can be deployed to GitHub Pages with a single command:

```bash
# Deploy to GitHub Pages
mkdocs gh-deploy
```

This command:

1. Builds your documentation
2. Creates or updates a branch called `gh-pages`
3. Pushes the built site to that branch
4. GitHub automatically serves the site from this branch

For more details, see the [GitHub Pages Deployment Guide](docs/github-pages-deployment.md).

### Versioning

The project uses a comprehensive versioning system as described in the [Data Versioning System](#data-versioning-system) section. This ensures consistency between:

- Data versions (v1, v2, v3, etc.)
- Model versions
- API endpoint versions

When deploying to GitHub Pages, the documentation will reflect the latest stable version by default.

### Docker Overview

The project uses Docker for containerization as detailed in the [Docker Configuration](#docker-configuration) section. This provides:

- Isolated, reproducible environments
- Easy deployment across different platforms
- Scalable services with resource management
- Separate configurations for development and production

For more details on the Docker setup, refer to the Dockerfile and docker-compose.yml files in the repository.

├── Dockerfile
├── README.md
├── **pycache**
│ └── test_outlier_handler.cpython-39-pytest-7.3.1.pyc
├── data
│ ├── README.md
│ ├── v1
│ │ ├── test.csv
│ │ └── train.csv
│ ├── v2
│ │ ├── test.csv
│ │ └── train.csv
│ └── v3
│ ├── test.csv
│ └── train.csv
├── docker-compose.yml
├── docs
│ ├── Challenge.md
│ ├── README.md
│ ├── api-documentation.md
│ ├── assets
│ │ ├── css
│ │ │ └── extra.css
│ │ ├── images
│ │ │ └── logo.svg
│ │ └── js
│ │ └── custom.js
│ ├── getting-started.md
│ ├── index.md
│ ├── installation-guide.md
│ └── user-manual.md
├── mkdocs.yml
├── notebooks
│ ├── Property-Friends-basic-model.ipynb
│ ├── **pycache**
│ │ ├── test_schema.cpython-39-pytest-7.3.1.pyc
│ │ └── test_schema_validation.cpython-39-pytest-7.3.1.pyc
│ ├── exploratory_analysis.ipynb
│ └── model_evaluation.ipynb
├── outputs
│ ├── jupyter
│ │ ├── data
│ │ ├── logs
│ │ ├── models
│ │ └── schema
│ ├── pipeline
│ │ ├── data
│ │ │ ├── v1.1_data_clean.csv
│ │ │ ├── v1.1_data_outliers.csv
│ │ │ ├── v1.1_evaluation_gradient_boosting.json
│ │ │ ├── v1.2_data_clean.csv
│ │ │ ├── v1.2_data_outliers.csv
│ │ │ ├── v1.2_evaluation_gradient_boosting.json
│ │ │ ├── v2.1_data_clean.csv
│ │ │ ├── v2.1_data_outliers.csv
│ │ │ ├── v2.1_evaluation_gradient_boosting.json
│ │ │ ├── v2.2_data_clean.csv
│ │ │ ├── v2.2_data_outliers.csv
│ │ │ ├── v2.2_evaluation_gradient_boosting.json
│ │ │ ├── v2.3_data_clean.csv
│ │ │ ├── v2.3_data_outliers.csv
│ │ │ └── v2.3_evaluation_gradient_boosting.json
│ │ ├── logs
│ │ │ └── PropertyValuationPipeline.log
│ │ ├── models
│ │ │ ├── v1.1_gradient_boosting_property_valuation.pkl
│ │ │ ├── v1.1_gradient_metadata_boosting_property_valuation.json
│ │ │ ├── v1.2_gradient_boosting_property_valuation.pkl
│ │ │ ├── v1.2_gradient_metadata_boosting_property_valuation.json
│ │ │ ├── v2.1_gradient_boosting_property_valuation.pkl
│ │ │ ├── v2.1_gradient_metadata_boosting_property_valuation.json
│ │ │ ├── v2.2_gradient_boosting_property_valuation.pkl
│ │ │ ├── v2.2_gradient_metadata_boosting_property_valuation.json
│ │ │ ├── v2.3_gradient_boosting_property_valuation.pkl
│ │ │ └── v2.3_gradient_metadata_boosting_property_valuation.json
│ │ └── schema
│ │ ├── v1.1_schema_train.json
│ │ ├── v1.2_schema_train.json
│ │ ├── v2.1_schema_train.json
│ │ ├── v2.2_schema_train.json
│ │ └── v2.3_schema_train.json
│ └── predictions
│ ├── api
│ │ └── api.log
│ ├── errors
│ │ └── error.log
│ ├── predictions
│ │ └── predictions.log
│ └── schema_validation
│ └── schema_validation.log
├── requirements-docs.txt
├── requirements.txt
├── scripts
│ ├── **pycache**
│ │ └── test_outlier_handler.cpython-39-pytest-7.3.1.pyc
│ ├── pipeline.py
│ └── run_api.py
├── src
│ ├── **init**.py
│ ├── **pycache**
│ │ ├── **init**.cpython-39.pyc
│ │ └── config.cpython-39.pyc
│ ├── api
│ │ ├── **init**.py
│ │ ├── **pycache**
│ │ │ ├── **init**.cpython-39.pyc
│ │ │ ├── auth.cpython-39.pyc
│ │ │ ├── endpoints.cpython-39.pyc
│ │ │ ├── logging.cpython-39.pyc
│ │ │ ├── main.cpython-39.pyc
│ │ │ ├── middleware.cpython-39.pyc
│ │ │ └── schemas.cpython-39.pyc
│ │ ├── auth.py
│ │ ├── endpoints.py
│ │ ├── logging.py
│ │ ├── main.py
│ │ ├── middleware.py
│ │ └── schemas.py
│ ├── config.py
│ ├── data
│ │ ├── **init**.py
│ │ ├── **pycache**
│ │ │ ├── **init**.cpython-39.pyc
│ │ │ ├── data_processor.cpython-39.pyc
│ │ │ ├── generate_schema.cpython-39.pyc
│ │ │ └── outlier_handler.cpython-39.pyc
│ │ ├── data_orm.py
│ │ ├── data_processor.py
│ │ ├── generate_schema.py
│ │ └── outlier_handler.py
│ ├── models
│ │ ├── **init**.py
│ │ ├── **pycache**
│ │ │ ├── **init**.cpython-39.pyc
│ │ │ ├── evaluate.cpython-39.pyc
│ │ │ ├── model.cpython-39.pyc
│ │ │ ├── serialization.cpython-39.pyc
│ │ │ └── tfma_like_evaluator.cpython-39.pyc
│ │ ├── evaluate.py
│ │ ├── model.py
│ │ ├── serialization.py
│ │ └── tfma_like_evaluator.py
│ ├── pipeline
│ │ ├── **pycache**
│ │ │ ├── data_pipeline.cpython-39.pyc
│ │ │ └── model_pipeline.cpython-39.pyc
│ │ ├── data_pipeline.py
│ │ └── model_pipeline.py
│ └── utils
│ ├── **init**.py
│ ├── **pycache**
│ │ ├── **init**.cpython-39.pyc
│ │ ├── helpers.cpython-39.pyc
│ │ └── logging.cpython-39.pyc
│ ├── helpers.py
│ ├── logging.py
│ └── plot_styles.py
└── tests
├── **init**.py
├── **pycache**
│ ├── **init**.cpython-39.pyc
│ ├── test_api.cpython-39-pytest-7.3.1.pyc
│ ├── test_api.cpython-39.pyc
│ ├── test_data.cpython-39-pytest-7.3.1.pyc
│ ├── test_data.cpython-39.pyc
│ ├── test_model.cpython-39-pytest-7.3.1.pyc
│ ├── test_model.cpython-39.pyc
│ ├── test_outlier_handler.cpython-39-pytest-7.3.1.pyc
│ ├── test_outliers.cpython-39-pytest-7.3.1.pyc
│ ├── test_outliers.cpython-39.pyc
│ ├── test_prediction_logging.cpython-39-pytest-7.3.1.pyc
│ ├── test_schema_validation.cpython-39-pytest-7.3.1.pyc
│ ├── test_utils.cpython-39-pytest-7.3.1.pyc
│ └── test_utils.cpython-39.pyc
├── test_api.py
├── test_data.py
├── test_model.py
├── test_outliers.py
└── test_utils.py

├── Dockerfile # Container configuration
├── README.md # Project documentation
├── docker-compose.yml # Multi-service orchestration
├── requirements.txt # Python dependencies
├── mkdocs.yml # Documentation config

├── data/ # Versioned datasets
│ ├── v1/
│ │ └── train.csv # Training data example
│ ├── v2/
│ └── v3/

├── docs/ # Documentation files
│ ├── index.md # Main documentation
│ ├── api-documentation.md # API reference
│ └── assets/
│ └── css/
│ └── extra.css # Custom styles

├── notebooks/ # Jupyter analysis
│ ├── exploratory_analysis.ipynb # Data exploration
│ └── model_evaluation.ipynb # Model assessment

├── outputs/ # Generated artifacts
│ ├── pipeline/
│ │ ├── data/
│ │ │ └── v2.3_data_clean.csv # Processed data example
│ │ ├── models/
│ │ │ └── v2.3_gradient_boosting_property_valuation.pkl # Trained model
│ │ ├── schema/
│ │ │ └── v2.3_schema_train.json # Data schema
│ │ └── logs/
│ │ └── PropertyValuationPipeline.log # Pipeline logs
│ └── predictions/
│ └── api/
│ └── api.log # API logs

├── scripts/ # Execution scripts
│ ├── pipeline.py # ML pipeline runner
│ └── run_api.py # API server launcher

├── src/ # Source code
│ ├── api/
│ │ └── main.py # FastAPI application
│ ├── data/
│ │ └── data_processor.py # Data processing logic
│ ├── models/
│ │ └── model.py # ML model implementation
│ ├── pipeline/
│ │ └── data_pipeline.py # Pipeline orchestration
│ └── utils/
│ └── logging.py # Utility functions

└── tests/ # Test suite
├── test_api.py # API endpoint tests
├── test_data.py # Data processing tests
└── test_model.py # Model functionality tests
