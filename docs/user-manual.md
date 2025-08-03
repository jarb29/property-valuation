---
layout: default
title: User Manual
---

# User Manual

This comprehensive guide provides in-depth information about the Property Valuation ML System's architecture, components, and how to use them effectively.

## System Architecture

The Property Valuation ML System is designed with a modular architecture that separates concerns and allows for flexibility and scalability. The system consists of several key components:

### 1. Data Processing Layer

The data processing layer handles data loading, validation, cleaning, and feature engineering:

- **Data Loaders**: Load data from various sources (CSV files, databases)
- **Data Validators**: Ensure data quality and schema compliance
- **Feature Engineers**: Transform raw data into model-ready features
- **Data Versioning**: Manage different versions of datasets

### 2. Model Layer

The model layer is responsible for training, evaluating, and serving machine learning models:

- **Model Trainers**: Train different types of models (Gradient Boosting, Random Forest, etc.)
- **Model Evaluators**: Assess model performance using various metrics
- **Model Registry**: Store and version trained models
- **Model Selectors**: Select the best model based on performance metrics

### 3. API Layer

The API layer provides RESTful endpoints for interacting with the system:

- **Prediction Endpoints**: Serve model predictions
- **Model Information Endpoints**: Provide metadata about models
- **Health Check Endpoints**: Monitor system health
- **Authentication**: Secure API access

### 4. Infrastructure Layer

The infrastructure layer provides the foundation for running the system:

- **Docker Containers**: Isolate and package the application
- **Logging**: Record system activities and errors
- **Configuration**: Manage system settings
- **Monitoring**: Track system performance

## Data Versioning System

### Overview

The data versioning system ensures consistency between datasets, models, and API endpoints:

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

### Using Different Data Versions

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

### Creating a New Data Version

To create a new data version:

1. Create a new directory in the `data/` folder (e.g., `data/v4/`)
2. Add your dataset files (`train.csv`, `test.csv`)
3. Set `DATA_VERSION=v4` in your environment
4. Run the pipeline to train models on the new data

## ML Pipeline

### Pipeline Components

The ML pipeline consists of several stages:

1. **Data Loading**: Load raw data from the specified data version
2. **Data Validation**: Validate data against schema (optional)
3. **Data Preprocessing**: Clean and prepare data for modeling
4. **Feature Engineering**: Create and transform features
5. **Model Training**: Train multiple model types
6. **Model Evaluation**: Evaluate models using specified metrics
7. **Model Selection**: Select the best model based on performance
8. **Model Persistence**: Save the selected model and metadata

### Running the Pipeline

To run the complete pipeline:

```bash
# Using Python
python scripts/pipeline.py

# Using Docker
docker-compose --profile pipeline up pipeline
```

### Pipeline Options

The pipeline supports several options:

```bash
# Specify model type
python scripts/pipeline.py --model-type random_forest

# Validate data during processing
python scripts/pipeline.py --validate-data

# Use a specific data version
DATA_VERSION=v2 python scripts/pipeline.py

# Specify evaluation metric
MODEL_METRIC=mae python scripts/pipeline.py
```

## API Usage

### Authentication

All API requests require an API key:

```bash
curl -X POST http://localhost:8000/api/v3/predictions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{"features": {...}}'
```

The API key is configured using the `API_KEY` environment variable.

### Single Property Valuation

To predict the value of a single property:

```bash
curl -X POST http://localhost:8000/api/v3/predictions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
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
  }'
```

### Batch Property Valuation

To predict values for multiple properties:

```bash
curl -X POST http://localhost:8000/api/v3/predictions/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "properties": [
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
      },
      {
        "features": {
          "type": "casa",
          "sector": "providencia",
          "net_usable_area": 200.0,
          "net_area": 250.0,
          "n_rooms": 4,
          "n_bathroom": 3,
          "latitude": -33.4298,
          "longitude": -70.6345
        }
      }
    ]
  }'
```

### Model Information

To retrieve information about the current model:

```bash
curl -X GET http://localhost:8000/api/v3/model/info \
  -H "X-API-Key: your_api_key_here"
```

### Health Check

To check the health of the API:

```bash
curl -X GET http://localhost:8000/api/v3/health \
  -H "X-API-Key: your_api_key_here"
```

## Advanced Usage

### Custom Feature Engineering

To implement custom feature engineering:

1. Create a new feature engineering class in `src/data/feature_engineering.py`
2. Register your feature engineering class in `src/pipeline/data_pipeline.py`
3. Update the pipeline configuration to use your custom feature engineering

Example:

```python
# src/data/feature_engineering.py
class CustomFeatureEngineer(BaseFeatureEngineer):
    def transform(self, df):
        # Your custom feature engineering logic
        df['custom_feature'] = df['feature1'] * df['feature2']
        return df

# src/pipeline/data_pipeline.py
feature_engineers = {
    'default': DefaultFeatureEngineer(),
    'custom': CustomFeatureEngineer()
}
```

### Custom Models

To add a custom model:

1. Create a new model class in `src/models/models.py`
2. Register your model in `src/pipeline/model_pipeline.py`
3. Update the pipeline configuration to use your custom model

Example:

```python
# src/models/models.py
class CustomModel(BaseModel):
    def train(self, X, y):
        # Your custom model training logic
        self.model = YourCustomAlgorithm()
        self.model.fit(X, y)
        return self

# src/pipeline/model_pipeline.py
models = {
    'gradient_boosting': GradientBoostingModel(),
    'random_forest': RandomForestModel(),
    'custom': CustomModel()
}
```

### Logging and Monitoring

The system includes comprehensive logging:

- **API Logs**: `outputs/predictions/api.log`
- **Pipeline Logs**: `outputs/pipeline/logs/pipeline.log`
- **Error Logs**: `outputs/predictions/error.log`

To configure logging:

```bash
# Set log level
LOG_LEVEL=DEBUG python scripts/run_api.py

# Configure log file size
LOG_MAX_BYTES=20971520 LOG_BACKUP_COUNT=10 python scripts/run_api.py
```

## Best Practices

### Production Deployment

For production deployment:

1. Use Docker with resource limits
2. Set a strong API key
3. Configure proper logging
4. Use multiple workers for the API
5. Set up monitoring and alerting

Example production Docker command:

```bash
docker-compose -e API_WORKERS=4 -e LOG_LEVEL=WARNING up -d api
```

### Data Management

Best practices for data management:

1. Keep data versions in separate directories
2. Document data schema changes
3. Validate data before processing
4. Back up data regularly
5. Use consistent naming conventions

### Model Management

Best practices for model management:

1. Version models consistently
2. Document model changes
3. Track model performance metrics
4. Compare models before deployment
5. Monitor model performance in production

## Troubleshooting

### Common API Issues

- **401 Unauthorized**: Check your API key
- **400 Bad Request**: Verify your request format
- **500 Internal Server Error**: Check API logs for details

### Common Pipeline Issues

- **Missing data files**: Ensure data files exist in the correct version directory
- **Model training failures**: Check pipeline logs for details
- **Feature engineering errors**: Verify data format and feature engineering logic

## Appendix

### API Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 500 | Internal Server Error |

### Model Performance Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R² | Coefficient of Determination |

### Environment Variables Reference

See the [Installation Guide](installation-guide.md#environment-variables) for a complete list of environment variables.

### File Formats

- **Data Files**: CSV format with headers
- **Model Files**: Pickle (.pkl) format
- **Schema Files**: JSON format
- **Log Files**: Text format
