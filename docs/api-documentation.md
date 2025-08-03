---
layout: default
title: API Documentation
---

# API Documentation

This document provides detailed information about the Property Valuation API endpoints, request/response formats, and authentication.

## Authentication

The API uses API key authentication. You need to include your API key in the request headers:

```
X-API-Key: your_api_key_here
```

The API key can be configured using the `API_KEY` environment variable (default: `default_api_key`).

## Base URL

The base URL for all API endpoints includes the data version:

```
http://{host}:{port}/api/{data_version}/
```

Where:
- `host` is the API host (default: `0.0.0.0`)
- `port` is the API port (default: `8000`)
- `data_version` is the current data version (e.g., `v3`)

## Endpoints

### 1. Single Property Valuation

**Endpoint:** `POST /api/v3/predictions`

**Description:** Predicts the value of a single property based on its features.

**Request Body:**
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

**Response:**
```json
{
  "prediction": 185000000,
  "prediction_time": 0.0234,
  "model_version": "best_mae_pipeline"
}
```

### 2. Batch Property Valuation

**Endpoint:** `POST /api/v3/predictions/batch`

**Description:** Predicts the values of multiple properties in a single request.

**Request Body:**
```json
{
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
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": 185000000,
      "prediction_time": 0.0234,
      "model_version": "best_mae_pipeline"
    },
    {
      "prediction": 320000000,
      "prediction_time": 0.0245,
      "model_version": "best_mae_pipeline"
    }
  ],
  "total_time": 0.0479
}
```

### 3. Model Information

**Endpoint:** `GET /api/v3/model/info`

**Description:** Retrieves information about the currently loaded model.

**Response:**
```json
{
  "model_version": "best_mae_pipeline",
  "data_version": "v3",
  "features": [
    "type",
    "sector",
    "net_usable_area",
    "net_area",
    "n_rooms",
    "n_bathroom",
    "latitude",
    "longitude"
  ],
  "model_type": "gradient_boosting",
  "training_date": "2023-06-15T14:30:45",
  "metrics": {
    "rmse": 15234567.89,
    "mae": 9876543.21,
    "r2": 0.87
  }
}
```

### 4. Health Check

**Endpoint:** `GET /api/v3/health`

**Description:** Checks the health status of the API and model.

**Response:**
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "model_loaded": true,
  "model_version": "best_mae_pipeline",
  "data_version": "v3",
  "uptime": 3600
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- **400 Bad Request**: Invalid input data
- **401 Unauthorized**: Missing or invalid API key
- **404 Not Found**: Endpoint not found
- **500 Internal Server Error**: Server-side error

Example error response:
```json
{
  "error": "Invalid input data",
  "detail": "Missing required field: 'features'",
  "status_code": 400
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse. By default, clients are limited to 100 requests per minute.

## Versioning

The API endpoints include the data version in their paths (e.g., `/api/v3/...`). This version matches the `DATA_VERSION` environment variable.

When changing the `DATA_VERSION` (e.g., to `v2`), the API endpoints will automatically update to match (e.g., `/api/v2/predictions`).
