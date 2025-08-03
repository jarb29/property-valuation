# Property Valuation ML System

Machine learning system for Chilean real estate property valuation.

## 🚀 Quick Start

Get up and running in minutes:

=== "Docker"
    ```bash
    git clone https://github.com/jarb29/property-valuation
    cd property-valuation
    
    # Create data directory and add your data
    mkdir -p data/v1
    # Copy your train.csv and test.csv to data/v1/
    
    # Run the ML pipeline first
    docker-compose --profile pipeline up pipeline
    
    # Then start the API
    docker-compose up api
    ```

=== "Local"
    ```bash
    git clone https://github.com/jarb29/property-valuation
    cd property-valuation
    
    # Create data directory and add your data
    mkdir -p data/v1
    # Copy your train.csv and test.csv to data/v1/
    
    # Set up environment
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    
    # Run the ML pipeline first
    DATA_VERSION=v1 python scripts/pipeline.py
    
    # Then start the API
    python scripts/run_api.py
    ```

## ✨ Features

- **🤖 ML Pipeline** - Automated training and evaluation
- **🚀 REST API** - Production-ready endpoints
- **📊 Data Versioning** - Complete traceability
- **🐳 Docker Ready** - Containerized deployment
- **📈 Monitoring** - Comprehensive logging
- **🎯 Smart Model Selection** - Automatic best model selection by metric

## 🔧 Model Configuration

The system automatically selects the best model based on your criteria:

```bash
# Configure model selection
MODEL_METRIC=rmse        # Choose best model by RMSE
MODEL_LOAD_TARGET=pipeline  # Load from pipeline outputs
DATA_VERSION=v1          # Use v1 dataset
```

**Available Metrics:** `rmse`, `mae`, `r2`  
**Model Sources:** `pipeline`, `jupyter`  
**Data Versions:** `v1`, `v2`, `v3`...

## 📊 Performance

| Metric | Current | Target | Selection |
|--------|---------|--------|----------|
| RMSE | 5,710 CLP | < 6,000 | `MODEL_METRIC=rmse` |
| MAE | 2,625 CLP | < 3,000 | `MODEL_METRIC=mae` |
| R² | 0.847 | > 0.80 | `MODEL_METRIC=r2` |
| Response Time | 23ms | < 50ms | - |

## 🎯 Test It Now

```bash
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: default_api_key" \
  -d '{"features": {"type": "departamento", "sector": "las condes", "net_usable_area": 120.5, "net_area": 150.0, "n_rooms": 3, "n_bathroom": 2, "latitude": -33.4172, "longitude": -70.5476}}'
```

## 📚 Documentation

- **[Getting Started](getting-started.md)** - Setup and installation
- **[User Manual](user-manual.md)** - Complete system guide  
- **[API Reference](api-documentation.md)** - Endpoint documentation
- **[Challenge](Challenge.md)** - Original requirements