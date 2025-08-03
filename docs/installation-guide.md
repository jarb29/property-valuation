
# Installation Guide

This guide provides detailed instructions for installing and configuring the Property Valuation ML System in different environments.

## System Requirements

### Minimum Requirements
- CPU: 2 cores
- RAM: 4GB
- Disk: 10GB free space
- Operating System: Linux, macOS, or Windows 10+
- Python 3.8+
- Docker 20.10+ and Docker Compose 2.0+ (for containerized deployment)

### Recommended Requirements
- CPU: 4+ cores
- RAM: 8GB+
- Disk: 20GB+ free space
- SSD storage for improved performance

## Installation Methods

### Method 1: Docker Installation (Recommended)

Docker provides the easiest and most consistent way to deploy the system across different environments.

#### Prerequisites
1. Install [Docker](https://docs.docker.com/get-docker/)
2. Install [Docker Compose](https://docs.docker.com/compose/install/)

#### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/jarb29/property-valuation
   cd property-valuation
   ```

2. Build the Docker images:
   ```bash
   docker-compose build
   ```

3. Start the services:
   ```bash
   # Start the API service
   docker-compose up api

   # Or start in detached mode
   docker-compose up -d api
   ```

4. Verify the installation:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

#### Docker Compose Profiles

The system includes several Docker Compose profiles for different use cases:

- **Default**: Runs the API service
  ```bash
  docker-compose up api
  ```

- **Development**: Runs the API with hot reload for development
  ```bash
  docker-compose --profile dev up api-dev
  ```

- **Pipeline**: Runs the ML pipeline for training models
  ```bash
  docker-compose --profile pipeline up pipeline
  ```

### Method 2: Local Python Installation

For development or when Docker is not available, you can install the system directly on your machine.

#### Prerequisites
1. Install [Python 3.8+](https://www.python.org/downloads/)
2. Install [pip](https://pip.pypa.io/en/stable/installation/)
3. (Optional) Install [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html)

#### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/jarb29/property-valuation
   cd property-valuation
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv

   # On Linux/macOS
   source .venv/bin/activate

   # On Windows
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Run the API:
   ```bash
   python scripts/run_api.py
   ```

6. Verify the installation:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

## Configuration

### Environment Variables

The system is configured using environment variables. You can set these in a `.env` file or directly in your environment.

#### Core Settings
| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `DATA_VERSION` | Version of data to use | `v1` | Controls which data folder is used (`data/v1`) |
| `MODEL_VERSION` | Version of model to use | Same as `DATA_VERSION` | Used in model filenames (`model_v1.pkl`) |
| `MODEL_METRIC` | Metric for model selection | `rmse` | Options: `rmse`, `mae`, `r2` |
| `MODEL_LOAD_TARGET` | Source of models | `pipeline` | Options: `pipeline`, `jupyter` |

#### API Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `API_HOST` | Host for the API server | `0.0.0.0` |
| `API_PORT` | Port for the API server | `8000` |
| `API_WORKERS` | Number of worker processes | `1` |
| `API_DEBUG` | Enable debug mode | `False` |
| `API_KEY` | API authentication key | `default_api_key` |

#### Logging Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_MAX_BYTES` | Maximum log file size | `10485760` (10MB) |
| `LOG_BACKUP_COUNT` | Number of backup log files | `5` |

### Configuration File

For more advanced configuration, you can modify the `src/config.py` file. This file contains default values for all settings and loads environment variables.

## Directory Structure

Understanding the directory structure helps with configuration:

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

## Troubleshooting

### Common Issues

#### Docker Issues
- **Error: Cannot connect to the Docker daemon**
  - Make sure Docker is running
  - Try restarting the Docker service

- **Error: Port already in use**
  - Change the port in the `.env` file or use the `-p` flag with Docker

#### Python Issues
- **Error: Module not found**
  - Make sure you've installed all dependencies: `pip install -r requirements.txt`
  - Check that you're in the correct directory

- **Error: Permission denied**
  - Check file permissions for scripts and data directories
  - Try running with sudo (Linux/macOS) or as administrator (Windows)

#### Model Issues
- **Error: Model file not found**
  - Run the pipeline to generate the model: `python scripts/pipeline.py`
  - Check that the `DATA_VERSION` and `MODEL_VERSION` are set correctly

### Logs

Check the log files for detailed error information:

- API logs: `outputs/predictions/api.log`
- Pipeline logs: `outputs/pipeline/logs/pipeline.log`

## Updating

To update the system to the latest version:

1. Pull the latest changes:
   ```bash
   git pull origin main
   ```

2. Rebuild Docker images (if using Docker):
   ```bash
   docker-compose build
   ```

3. Restart the services:
   ```bash
   docker-compose down
   docker-compose up -d api
   ```

## Next Steps

After installation, you might want to:

- Read the [Getting Started Guide](getting-started.md) for a quick introduction
- Explore the [API Documentation](api-documentation.md) for details on available endpoints
- Check the [User Manual](user-manual.md) for in-depth usage instructions
