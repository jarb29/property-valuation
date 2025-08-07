# PropertyAI - Flask Web Application

A modern, enterprise-grade Flask web application for Chilean real estate property valuation using ONNX-optimized machine learning models with interactive map interface.

## 🚀 Quick Start

### Local Development
```bash
cd flask-app
./run-local.sh
```

### Docker Deployment (Recommended)
```bash
cd flask-app
./run-docker.sh
```

Visit `http://localhost:5002` to use the web interface.

## 📁 Self-Contained Architecture

### Directory Structure
```
flask-app/
├── models/
│   ├── predictor.py         # Main predictor class
│   └── bestmodel/           # Embedded best model (auto-generated)
│       ├── model.onnx       # ONNX model for fast inference
│       ├── preprocessor.pkl # Data preprocessing pipeline
│       └── metadata.json    # Model information & metrics
├── templates/
│   ├── base.html            # Modern enterprise layout
│   ├── index.html           # Interactive form with map
│   ├── result.html          # Prediction results
│   └── model_info.html      # Model information page
├── static/
│   ├── css/
│   │   ├── style.css        # Modern enterprise styling
│   │   └── results.css      # Results modal styling
│   └── js/
│       ├── app.js           # Interactive map & form logic
│       └── debug.js         # Debug utilities
├── prepare-model.py         # Best model preparation script
├── run-local.sh            # Local development script
├── run-docker.sh           # Docker deployment script
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Service orchestration
├── gunicorn.conf.py        # Production server config
├── requirements.txt        # Python dependencies
└── app.py                  # Flask application
```

## 🔄 Simplified Data Flow

```
1. prepare-model.py → Selects best model from pipeline
2. Creates bestmodel/ → 3 files (ONNX, preprocessor, metadata)
3. Flask app loads → From local bestmodel/ directory
4. User request → Preprocessor + ONNX inference → Response
```

## 🌟 Key Features

### 🎨 Modern Enterprise UI
- **Interactive Map**: Leaflet-based map for location selection
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Draggable Modal**: Resizable and draggable results display
- **Professional Styling**: Clean, modern enterprise design
- **Real-time Validation**: Instant feedback on form inputs

### 📊 Smart Property Valuation
- **6 Santiago Sectors**: Las Condes, Vitacura, Lo Barnechea, Ñuñoa, Providencia, La Reina
- **15,352+ Training Properties**: Comprehensive dataset coverage
- **Automatic Best Model**: Selects optimal model based on configured metric
- **Price per m² Calculation**: Automatic area-based pricing

### 🚀 Self-Contained Architecture
- **No External Dependencies**: Everything embedded in bestmodel/ folder
- **ONNX Optimization**: 10-50x faster inference than sklearn
- **Automatic Model Selection**: Uses metric from `src/config.py`
- **Zero Configuration**: Hardcoded defaults for simplicity

### 🔍 Intelligent Model Management
- **Dynamic Selection**: Automatically finds best performing model
- **Metadata-Driven**: All info comes from model metadata
- **Version Agnostic**: Works with any model version (v2.1, v2.2, v2.3, etc.)

## 🏗️ Model Preparation Process

### Automatic Best Model Selection
```bash
# Run prepare-model.py (automatically called by run scripts)
🔍 Scanning for available models...
📊 Found 3 models
🏆 Selected best model: v2.3_gradient_boosting_property_valuation.pkl (metric: mae)
📦 Loading model...
💾 Saving model components...
🔄 Converting to ONNX...
✅ Best model prepared successfully
```

### What Gets Created
```
models/bestmodel/
├── model.onnx          # Fast ONNX inference model
├── preprocessor.pkl    # Sklearn preprocessing pipeline  
└── metadata.json       # Complete model information
```

## 🐳 Docker Deployment

### Self-Contained Container
```bash
./run-docker.sh
├── Prepares best model locally
├── Embeds model in Docker image
├── Builds self-contained container
└── Runs with zero external dependencies
```

### Production Features
- **Embedded Models**: No volume mounts needed
- **Gunicorn WSGI**: Production server with 2 workers
- **Health Checks**: Automatic container monitoring
- **Auto Restart**: Container restarts on failure
- **Instant Startup**: Pre-converted ONNX models

### Container Structure
```
Container /app/
├── src/config.py              # Configuration from main project
├── models/bestmodel/          # Embedded best model
│   ├── model.onnx            # Fast inference
│   ├── preprocessor.pkl      # Data processing
│   └── metadata.json         # Model info
├── app.py                    # Flask application
└── All dependencies included
```

## 🌐 API Endpoints

- **`GET /`** - Main web interface with interactive map
- **`POST /predict`** - Property valuation API (JSON)
- **`GET /health`** - Health check endpoint
- **`GET /model/info`** - Model information page
- **`GET /api/model/info`** - Model metadata API

## 📊 Model Information Display

The `/model/info` page shows:
- **Model Type**: gradient_boosting
- **Inference Engine**: ONNX Runtime (fast predictions)
- **Preprocessing Engine**: Scikit-learn Pipeline (data processing)
- **Selection Metric**: MAE/RMSE (from config)
- **Best Score**: Actual performance metric
- **Original Model Path**: Source location
- **Performance Metrics**: RMSE, MAE, MAPE
- **Features**: All input features
- **Timestamp**: Model training time

## 📈 Performance

- **ONNX Inference**: ~10-50x faster than sklearn
- **Sub-second Response**: < 100ms prediction time
- **Memory Efficient**: Optimized model format
- **Concurrent Users**: Supports multiple simultaneous requests
- **Instant Startup**: Pre-converted models

## 🎯 Usage Examples

### Local Development
```bash
cd flask-app
./run-local.sh
# Automatically prepares best model and starts Flask
```

### Docker Production
```bash
cd flask-app
./run-docker.sh
# Prepares model, builds container, and deploys
```

### API Usage
```bash
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
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

### Health Check
```bash
curl http://localhost:5002/health
# Returns: {"status": "healthy", "model_loaded": true, ...}
```

## 🔧 Configuration

### Model Selection
The best model is automatically selected based on the metric defined in `src/config.py`:
```python
MODEL_METRIC = os.getenv("MODEL_METRIC", "mae")  # mae, rmse, or mape
```

### Hardcoded Settings (Zero Config)
- **Host**: 0.0.0.0
- **Port**: 5000 (mapped to 5002 in Docker)
- **Workers**: 2 (Gunicorn)
- **Debug**: False (Production)

## 🎯 Key Benefits

1. **Zero Configuration**: Works out of the box
2. **Self-Contained**: No external dependencies
3. **Automatic**: Best model selection and preparation
4. **Fast**: ONNX-optimized inference
5. **Portable**: Docker image works anywhere
6. **Production Ready**: Gunicorn, health checks, monitoring

## 🔗 Related Documentation

- **[Main Project README](../README.md)** - Complete ML pipeline documentation
- **[API Documentation](../docs/api-documentation.md)** - Full API reference