# Getting Started

Welcome to the Property Valuation ML System! This guide will help you get up and running quickly with our enterprise-grade machine learning platform.

---

## 📋 Prerequisites

Before you begin, ensure your system meets these requirements:

!!! info "System Requirements"
    - **Python**: 3.8 or higher
    - **Docker**: Latest version (recommended)
    - **Memory**: Minimum 4GB RAM
    - **Storage**: 2GB free space
    - **OS**: Linux, macOS, or Windows

### Required Tools

=== "Essential"
    - [Python 3.8+](https://python.org/downloads/)
    - [Git](https://git-scm.com/downloads)

=== "Recommended"
    - [Docker Desktop](https://docker.com/products/docker-desktop/)
    - [VS Code](https://code.visualstudio.com/) with Python extension
    - [Postman](https://postman.com/) for API testing

---

## 🚀 Quick Installation

Choose your preferred installation method:

### Option A: Docker Setup (Recommended)

Docker provides the fastest and most reliable setup experience:

```bash
# 1. Clone the repository
git clone https://github.com/jarb29/property-valuation
cd property-valuation

# 2. Prepare your data
mkdir -p data/v1
# Copy your train.csv and test.csv files to data/v1/

# 3. Run the ML pipeline
docker-compose --profile pipeline up pipeline

# 4. Start the API service
docker-compose up api

# 5. Verify installation
curl http://localhost:8000/api/v1/health
```

!!! success "Docker Benefits"
    - ✅ No dependency conflicts
    - ✅ Consistent environment
    - ✅ Production-ready configuration
    - ✅ Easy scaling and deployment

### Option B: Local Python Setup

For development and customization:

```bash
# 1. Clone and navigate
git clone https://github.com/jarb29/property-valuation
cd property-valuation

# 2. Prepare your data
mkdir -p data/v1
# Copy your train.csv and test.csv files to data/v1/

# 3. Create virtual environment
python -m venv .venv

# 4. Activate environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Configure environment
cp .env.example .env
# Edit .env to set DATA_VERSION=v1

# 7. Run the ML pipeline
python scripts/pipeline.py

# 8. Start the API
python scripts/run_api.py
```

---

## 🔧 Configuration

### Environment Variables

The system uses environment variables for configuration. Key settings:

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DATA_VERSION` | Data version to use | `v1` | `v1`, `v2`, `v3` |
| `MODEL_METRIC` | Model selection metric | `rmse` | `mae`, `r2` |
| `API_HOST` | API server host | `0.0.0.0` | `localhost` |
| `API_PORT` | API server port | `8000` | `8080` |
| `API_KEY` | Authentication key | `default_api_key` | `your_secure_key` |

### Sample Configuration

Create your `.env` file:

```bash
# Data Configuration
DATA_VERSION=v1
MODEL_VERSION=v1
MODEL_METRIC=rmse
MODEL_LOAD_TARGET=pipeline

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your_secure_api_key
API_DEBUG=False

# Logging Configuration
LOG_LEVEL=INFO
```

### Model Configuration Details

The system uses these key variables (defined in `src/config`) for intelligent model selection:

| Variable | Purpose | Options | Example |
|----------|---------|---------|----------|
| `API_KEY` | API authentication | Any string | `"my_secure_key"` |
| `MODEL_VERSION` | Model version to load | Defaults to `DATA_VERSION` | `v1`, `v2`, `v3` |
| `MODEL_METRIC` | Best model selection criteria | `rmse`, `mae`, `r2` | `rmse` |
| `MODEL_LOAD_TARGET` | Model source location | `pipeline`, `jupyter` | `pipeline` |

**How Model Selection Works:**

1. **Training Phase**: Multiple models are trained and saved with performance metrics
2. **Selection Phase**: System chooses the best model based on `MODEL_METRIC`
3. **Loading Phase**: API loads the selected model from `MODEL_LOAD_TARGET` location
4. **Versioning**: All artifacts are saved with version numbers for traceability

**Example Workflow:**
```bash
# Train models with different configurations
MODEL_METRIC=rmse python scripts/pipeline.py  # Selects best RMSE model
MODEL_METRIC=mae python scripts/pipeline.py   # Selects best MAE model

# API automatically loads the best model based on your configuration
MODEL_LOAD_TARGET=pipeline MODEL_METRIC=rmse python scripts/run_api.py
```

---

## 🧪 Verify Your Installation

### 1. Health Check

Test that the API is running:

```bash
curl -X GET http://localhost:8000/api/v1/health \
  -H "X-API-Key: default_api_key"
```

Expected response:
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "model_loaded": true,
  "model_version": "best_rmse_pipeline",
  "data_version": "v3",
  "uptime": 60
}
```

### 2. Model Information

Check the loaded model:

```bash
curl -X GET http://localhost:8000/api/v1/model/info \
  -H "X-API-Key: default_api_key"
```

### 3. Sample Prediction

Make your first prediction:

```bash
curl -X POST http://localhost:8000/api/v1/predictions \
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

Expected response:
```json
{
  "prediction": 185000000,
  "prediction_time": 0.0234,
  "model_version": "best_rmse_pipeline"
}
```

---

## 🎯 Your First Workflow

Now that everything is set up, let's walk through a complete workflow:

### Step 1: Explore the Data

```bash
# Check available data versions
ls data/

# View sample data
head -5 data/v1/train.csv
```

### Step 2: Train a Model

```bash
# Run the ML pipeline with default settings
docker-compose --profile pipeline up pipeline

# Or with custom configuration
DATA_VERSION=v1 MODEL_METRIC=mae docker-compose --profile pipeline up pipeline
```

### Step 3: Start the API

After training, start the API server:

```bash
# Start the API service
docker-compose up api

# Or use a different port if 8000 is busy
API_PORT=8080 docker-compose up api
```

**Verify the API is running:**
```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Expected response: {"status": "healthy", "model_loaded": true}
```

### Step 4: Make Predictions

Use the API to make predictions for different property types:

=== "Apartment"
    ```json
    {
      "features": {
        "type": "departamento",
        "sector": "providencia",
        "net_usable_area": 85.0,
        "net_area": 100.0,
        "n_rooms": 2,
        "n_bathroom": 1,
        "latitude": -33.4298,
        "longitude": -70.6345
      }
    }
    ```

=== "House"
    ```json
    {
      "features": {
        "type": "casa",
        "sector": "las condes",
        "net_usable_area": 200.0,
        "net_area": 300.0,
        "n_rooms": 4,
        "n_bathroom": 3,
        "latitude": -33.4172,
        "longitude": -70.5476
      }
    }
    ```

### Step 5: Batch Processing

For multiple properties:

```bash
curl -X POST http://localhost:8000/api/v1/predictions/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: default_api_key" \
  -d '{
    "properties": [
      {"features": {...}},
      {"features": {...}}
    ]
  }'
```

---

## 🔍 Development Tools

### API Documentation

Access interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Monitoring and Logs

Monitor your system:

```bash
# View API logs
tail -f outputs/predictions/api.log

# View pipeline logs
tail -f outputs/pipeline/logs/pipeline.log

# Docker logs
docker-compose logs -f api
```

### Development Mode

For active development:

```bash
# Start with hot reload
python scripts/run_api.py --reload

# Or with Docker
docker-compose --profile dev up api-dev
```

---

## 🚨 Troubleshooting

### Common Issues

!!! warning "Port Already in Use"
    **Problem**: Port 8000 is already in use (Docker error: "port is already allocated")
    
    **Solutions**: 
    ```bash
    # Option 1: Use a different port (recommended)
    API_PORT=8080 docker-compose up api
    
    # Option 2: Kill the process using port 8000
    lsof -ti:8000 | xargs kill -9
    
    # Option 3: Stop all Docker containers first
    docker-compose down
    
    # Option 4: Check what's using the port
    lsof -i :8000
    
    # Option 5: For Python API (auto-finds available port)
    python scripts/run_api.py --auto-port
    ```
    
    !!! tip "Smart Port Detection"
        The Python API script automatically finds available ports and shows helpful messages:
        ```
        WARNING: Port 8000 is already in use. Using port 8001 instead.
        ```

!!! warning "Model Not Found"
    **Problem**: No model file found
    
    **Solution**: 
    ```bash
    # Train a model first
    python scripts/pipeline.py
    # Or check if model exists
    ls outputs/pipeline/models/
    ```

!!! warning "Permission Denied"
    **Problem**: Docker permission issues
    
    **Solution**: 
    ```bash
    # Add user to docker group (Linux)
    sudo usermod -aG docker $USER
    # Or run with sudo
    sudo docker-compose up api
    ```

### Getting Help

If you encounter issues:

1. **Check the logs** for detailed error messages
2. **Verify environment variables** are set correctly
3. **Ensure all dependencies** are installed
4. **Check Docker status** if using containers

---

## 🎉 Next Steps

Congratulations! You now have a working Property Valuation ML System. Here's what to explore next:

| Next Step | Description | Link |
|-----------|-------------|------|
| **API Deep Dive** | Learn all API endpoints and features | [API Documentation](api-documentation.md) |
| **System Architecture** | Understand the complete system design | [User Manual](user-manual.md) |
| **Custom Models** | Train models with your own data | [User Manual - Training](user-manual.md#model-training) |
| **Production Deployment** | Deploy to production environments | [User Manual - Deployment](user-manual.md#deployment) |

---

## 📞 Support

Need help? We're here for you:

- **📖 Documentation**: Complete guides and references
- **🐛 GitHub Issues**: Bug reports and feature requests  
- **📧 Email**: team@property-valuation.com
- **💬 Community**: Join our developer community

<div class="text-center">
  <p><strong>Ready to build amazing property valuation applications!</strong></p>
</div>