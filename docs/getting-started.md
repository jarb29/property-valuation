---
layout: default
title: Getting Started
---

# Getting Started with Property Valuation ML System

This guide will help you quickly get started with the Property Valuation ML System.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)
- Git

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd property-valuation
```

### 2. Choose Your Setup Method

You can set up the system using either Docker (recommended) or a local Python environment.

#### Option A: Docker Setup (Recommended)

1. Build and start the API service:

```bash
docker-compose up api
```

2. The API will be available at `http://localhost:8000`

#### Option B: Local Python Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the API:

```bash
python scripts/run_api.py
```

## Making Your First API Request

Once the API is running, you can make a prediction request:

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

You should receive a response like:

```json
{
  "prediction": 185000000,
  "prediction_time": 0.0234,
  "model_version": "best_mae_pipeline"
}
```

## Next Steps

Now that you have the system up and running, you might want to:

1. **Explore the API Documentation**: Check out the [API Documentation](api-documentation.md) for details on all available endpoints.

2. **Train a Custom Model**: Learn how to train your own model with custom data:

```bash
# Run the ML pipeline with default settings
docker-compose --profile pipeline up pipeline

# Or with custom settings
DATA_VERSION=v2 docker-compose --profile pipeline up pipeline
```

3. **Configure the System**: Review the [Installation Guide](installation-guide.md) for detailed configuration options.

4. **Understand the Architecture**: Explore the [User Manual](user-manual.md) to understand the system's architecture and components.

## Troubleshooting

If you encounter any issues:

1. Check the logs:
   - API logs: `outputs/predictions/api.log`
   - Pipeline logs: `outputs/pipeline/logs/pipeline.log`

2. Ensure all environment variables are set correctly

3. Verify Docker is running properly if using containerized deployment

4. Check the model file exists: `outputs/pipeline/models/model_v3.pkl` (or your specific version)

## Getting Help

If you need further assistance, please:

1. Check the [full documentation](https://github.com/username/property-valuation)
2. Open an issue on GitHub
3. Contact the maintainers
