<div class="hero">
  <h1>Property Valuation ML System</h1>
  <p>Enterprise-grade machine learning for Chilean real estate valuation</p>
</div>

## 🚀 Quick Start

Get up and running in minutes:

=== "Docker"
    ```bash
    git clone <repository-url>
    cd property-valuation
    docker-compose up api
    ```

=== "Local"
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/run_api.py
    ```

## ✨ Features

- **🤖 ML Pipeline** - Automated training and evaluation
- **🚀 REST API** - Production-ready endpoints
- **📊 Data Versioning** - Complete traceability
- **🐳 Docker Ready** - Containerized deployment
- **📈 Monitoring** - Comprehensive logging

## 📊 Performance

| Metric | Current | Target |
|--------|---------|--------|
| RMSE | 5,710 CLP | < 6,000 |
| MAE | 2,625 CLP | < 3,000 |
| Response Time | 23ms | < 50ms |

## 🎯 Test It Now

```bash
curl -X POST http://localhost:8000/api/v3/predictions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: default_api_key" \
  -d '{"features": {"type": "departamento", "sector": "las condes", "net_usable_area": 120.5, "net_area": 150.0, "n_rooms": 3, "n_bathroom": 2, "latitude": -33.4172, "longitude": -70.5476}}'
```

## 📚 Documentation

- **[Getting Started](getting-started.md)** - Setup and installation
- **[User Manual](user-manual.md)** - Complete system guide  
- **[API Reference](api-documentation.md)** - Endpoint documentation
- **[Challenge](Challenge.md)** - Original requirements