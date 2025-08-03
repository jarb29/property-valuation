# API Reference

Complete reference for the Property Valuation ML System REST API. This API provides enterprise-grade property valuation services with comprehensive validation, authentication, and monitoring.

---

## 🔐 Authentication

The API uses API key authentication for secure access:

```http
X-API-Key: your_api_key_here
```

!!! info "API Key Configuration"
    The API key is configured via the `API_KEY` environment variable (default: `default_api_key`). 
    For production deployments, always use a secure, randomly generated key.

### Authentication Example

```bash
curl -X GET http://localhost:8000/api/v3/health \
  -H "X-API-Key: your_secure_api_key"
```

---

## 🌐 Base URL Structure

The API follows a versioned URL structure that automatically matches your data version:

```
http://{host}:{port}/api/{data_version}/
```

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `host` | API server host | `0.0.0.0` | `localhost` |
| `port` | API server port | `8000` | `8080` |
| `data_version` | Current data version | `v3` | `v2`, `v3` |

!!! tip "Version Synchronization"
    The API version automatically matches your `DATA_VERSION` environment variable. 
    When you change data versions, endpoints update accordingly (e.g., `/api/v2/predictions`).

---

## 📊 Endpoints Overview

| Endpoint | Method | Description | Rate Limit |
|----------|--------|-------------|------------|
| `/predictions` | POST | Single property valuation | 100/min |
| `/predictions/batch` | POST | Batch property valuation | 20/min |
| `/model/info` | GET | Model information | 200/min |
| `/health` | GET | System health check | Unlimited |

---

## 🏠 Single Property Valuation

### Endpoint

<div class="endpoint">
  <span class="endpoint-method post">POST</span>
  <span class="endpoint-path">/api/v3/predictions</span>
</div>

Generate accurate property valuation using advanced machine learning models trained on comprehensive Chilean real estate market data.

### Request Format

```json
{
  "features": {
    "type": "string",
    "sector": "string", 
    "net_usable_area": "number",
    "net_area": "number",
    "n_rooms": "integer",
    "n_bathroom": "integer",
    "latitude": "number",
    "longitude": "number"
  }
}
```

### Required Parameters

| Parameter | Type | Description | Validation |
|-----------|------|-------------|------------|
| `type` | string | Property classification | `departamento`, `casa`, `oficina` |
| `sector` | string | Geographic sector in Santiago | Valid Santiago sector name |
| `net_usable_area` | number | Usable floor area (m²) | > 0, < 1000 |
| `net_area` | number | Total property area (m²) | > 0, < 2000 |
| `n_rooms` | integer | Number of bedrooms | 0-10 |
| `n_bathroom` | integer | Number of bathrooms | 0-10 |
| `latitude` | number | Geographic coordinate | -90 to 90 |
| `longitude` | number | Geographic coordinate | -180 to 180 |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v3/predictions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: default_api_key" \
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

### Response Format

```json
{
  "prediction": 185000000,
  "prediction_time": 0.0234,
  "model_version": "best_rmse_pipeline"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | number | Property valuation in Chilean Pesos (CLP) |
| `prediction_time` | number | Processing time in seconds |
| `model_version` | string | Model version identifier |

---

## 🏘️ Batch Property Valuation

### Endpoint

<div class="endpoint">
  <span class="endpoint-method post">POST</span>
  <span class="endpoint-path">/api/v3/predictions/batch</span>
</div>

Process multiple property valuations in a single request for improved efficiency.

### Request Format

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

### Batch Limits

!!! warning "Batch Processing Limits"
    - **Maximum properties per request**: 100
    - **Request timeout**: 30 seconds
    - **Rate limit**: 20 requests per minute

### Example Request

```bash
curl -X POST http://localhost:8000/api/v3/predictions/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: default_api_key" \
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

### Response Format

```json
{
  "predictions": [
    {
      "prediction": 185000000,
      "prediction_time": 0.0234,
      "model_version": "best_rmse_pipeline"
    },
    {
      "prediction": 320000000,
      "prediction_time": 0.0245,
      "model_version": "best_rmse_pipeline"
    }
  ],
  "total_time": 0.0479,
  "processed_count": 2
}
```

---

## 📋 Model Information

### Endpoint

<div class="endpoint">
  <span class="endpoint-method get">GET</span>
  <span class="endpoint-path">/api/v3/model/info</span>
</div>

Retrieve comprehensive information about the currently loaded model, including metadata, features, and performance metrics.

### Example Request

```bash
curl -X GET http://localhost:8000/api/v3/model/info \
  -H "X-API-Key: default_api_key"
```

### Response Format

```json
{
  "model_version": "best_rmse_pipeline",
  "data_version": "v3",
  "model_type": "gradient_boosting",
  "training_date": "2024-01-15T14:30:45Z",
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
  "metrics": {
    "rmse": 15234567.89,
    "mae": 9876543.21,
    "r2": 0.87
  },
  "training_samples": 50000,
  "validation_samples": 12500
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_version` | string | Model version identifier |
| `data_version` | string | Data version used for training |
| `model_type` | string | Algorithm type (gradient_boosting, random_forest, etc.) |
| `training_date` | string | ISO 8601 timestamp of training completion |
| `features` | array | List of input features |
| `metrics` | object | Model performance metrics |
| `training_samples` | integer | Number of training samples |
| `validation_samples` | integer | Number of validation samples |

---

## 🏥 Health Check

### Endpoint

<div class="endpoint">
  <span class="endpoint-method get">GET</span>
  <span class="endpoint-path">/api/v3/health</span>
</div>

Monitor system health and availability status.

### Example Request

```bash
curl -X GET http://localhost:8000/api/v3/health \
  -H "X-API-Key: default_api_key"
```

### Response Format

```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "model_loaded": true,
  "model_version": "best_rmse_pipeline",
  "data_version": "v3",
  "uptime": 3600,
  "memory_usage": {
    "used": "512MB",
    "available": "2GB"
  },
  "disk_usage": {
    "used": "1.2GB", 
    "available": "10GB"
  }
}
```

### Health Status Values

| Status | Description |
|--------|-------------|
| `healthy` | All systems operational |
| `degraded` | Some non-critical issues |
| `unhealthy` | Critical issues detected |

---

## ⚠️ Error Handling

The API returns standard HTTP status codes with detailed error information:

### HTTP Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid input data |
| 401 | Unauthorized | Missing or invalid API key |
| 422 | Unprocessable Entity | Validation errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |

### Error Response Format

```json
{
  "error": "Validation Error",
  "detail": "Invalid property type. Must be one of: departamento, casa, oficina",
  "status_code": 422,
  "timestamp": "2024-01-15T14:30:45Z",
  "request_id": "req_123456789"
}
```

### Common Error Examples

=== "Invalid API Key"
    ```json
    {
      "error": "Unauthorized",
      "detail": "Invalid or missing API key",
      "status_code": 401
    }
    ```

=== "Validation Error"
    ```json
    {
      "error": "Validation Error",
      "detail": [
        {
          "loc": ["body", "features", "net_usable_area"],
          "msg": "ensure this value is greater than 0",
          "type": "value_error.number.not_gt"
        }
      ],
      "status_code": 422
    }
    ```

=== "Rate Limit Exceeded"
    ```json
    {
      "error": "Rate Limit Exceeded",
      "detail": "Too many requests. Limit: 100 per minute",
      "status_code": 429,
      "retry_after": 60
    }
    ```

---

## 🚦 Rate Limiting

The API implements rate limiting to ensure fair usage and system stability:

| Endpoint | Rate Limit | Window |
|----------|------------|--------|
| `/predictions` | 100 requests | 1 minute |
| `/predictions/batch` | 20 requests | 1 minute |
| `/model/info` | 200 requests | 1 minute |
| `/health` | Unlimited | - |

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642262400
```

---

## 🔧 SDK and Integration Examples

### Python SDK Example

```python
import requests
import json

class PropertyValuationClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/json',
            'X-API-Key': api_key
        }
    
    def predict(self, features):
        response = requests.post(
            f"{self.base_url}/api/v3/predictions",
            headers=self.headers,
            json={"features": features}
        )
        return response.json()
    
    def batch_predict(self, properties):
        response = requests.post(
            f"{self.base_url}/api/v3/predictions/batch",
            headers=self.headers,
            json={"properties": properties}
        )
        return response.json()

# Usage
client = PropertyValuationClient("http://localhost:8000", "your_api_key")
result = client.predict({
    "type": "departamento",
    "sector": "las condes",
    "net_usable_area": 120.5,
    "net_area": 150.0,
    "n_rooms": 3,
    "n_bathroom": 2,
    "latitude": -33.4172,
    "longitude": -70.5476
})
print(f"Predicted value: ${result['prediction']:,} CLP")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

class PropertyValuationClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json',
            'X-API-Key': apiKey
        };
    }
    
    async predict(features) {
        try {
            const response = await axios.post(
                `${this.baseUrl}/api/v3/predictions`,
                { features },
                { headers: this.headers }
            );
            return response.data;
        } catch (error) {
            throw new Error(`Prediction failed: ${error.response.data.detail}`);
        }
    }
}

// Usage
const client = new PropertyValuationClient('http://localhost:8000', 'your_api_key');
const result = await client.predict({
    type: 'departamento',
    sector: 'las condes',
    net_usable_area: 120.5,
    net_area: 150.0,
    n_rooms: 3,
    n_bathroom: 2,
    latitude: -33.4172,
    longitude: -70.5476
});
console.log(`Predicted value: $${result.prediction.toLocaleString()} CLP`);
```

---

## 📊 Interactive API Documentation

For hands-on API exploration, visit our interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces provide:
- ✅ Interactive request/response testing
- ✅ Complete schema documentation  
- ✅ Authentication testing
- ✅ Response format examples

---

## 🔍 Monitoring and Observability

### Request Logging

All API requests are logged with structured data:

```json
{
  "timestamp": "2024-01-15T14:30:45Z",
  "request_id": "req_123456789",
  "method": "POST",
  "endpoint": "/api/v3/predictions",
  "status_code": 200,
  "response_time": 0.0234,
  "user_agent": "PropertyValuationClient/1.0",
  "ip_address": "192.168.1.100"
}
```

### Performance Metrics

Monitor API performance through logs:

- **Response Time**: Average < 50ms
- **Throughput**: 1000+ requests/minute
- **Error Rate**: < 0.1%
- **Availability**: 99.9% uptime

---

## 📞 Support

Need help with the API? We're here to assist:

- **📖 Documentation**: Complete API reference and guides
- **🐛 GitHub Issues**: Bug reports and feature requests
- **📧 Email**: api-support@property-valuation.com
- **💬 Community**: Join our developer community

<div class="text-center">
  <p><strong>Build amazing applications with our Property Valuation API!</strong></p>
</div>