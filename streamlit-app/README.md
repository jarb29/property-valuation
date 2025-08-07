# PropertyAI - Streamlit Application

A modern Streamlit web application for Chilean real estate property valuation using ONNX-optimized machine learning models.

## 🚀 Quick Start

### Local Development
```bash
# 1. Prepare the best model
python prepare-model.py

# 2. Run the Streamlit app
streamlit run app.py
```

### Streamlit Cloud Deployment
1. **Prepare Model**: Run `python prepare-model.py` locally
2. **Push to GitHub**: Commit all files including `models/bestmodel/`
3. **Deploy**: Connect repository to Streamlit Cloud
4. **Access**: Your app will be available at `https://your-app.streamlit.app`

## 📁 Application Structure

```
streamlit-app/
├── app.py                  # Main Streamlit application
├── pages/                  # Multi-page app structure
│   ├── 01_🏠_Property_Valuation.py
│   ├── 02_📊_Model_Information.py
│   └── 03_📈_Analytics.py
├── models/
│   ├── predictor.py        # Streamlit-specific predictor
│   └── bestmodel/          # Pre-prepared model files
│       ├── model.onnx      # ONNX model for inference
│       ├── preprocessor.pkl # Data preprocessing pipeline
│       └── metadata.json   # Model information
├── utils/
│   ├── ui_utils.py         # UI components and formatting
│   └── map_utils.py        # Map utilities and coordinates
├── prepare-model.py        # Model preparation script
├── requirements.txt        # Streamlit Cloud dependencies
├── .streamlit/
│   └── config.toml         # App configuration
└── README.md
```

## 🌟 Features

### 🏠 Property Valuation
- **Interactive Form**: Easy-to-use property input form
- **Real-time Validation**: Instant feedback on inputs
- **Interactive Map**: Visual property location selection
- **Instant Results**: Fast ONNX-powered predictions
- **Detailed Breakdown**: Price per m², prediction time, model info

### 📊 Model Information
- **Model Overview**: Detailed model specifications
- **Performance Metrics**: RMSE, MAE, MAPE visualization
- **Technical Details**: Architecture and optimization info
- **Interactive Charts**: Plotly-powered visualizations

### 📈 Analytics Dashboard
- **Market Overview**: Key market statistics
- **Sector Analysis**: Price distribution by area
- **Interactive Filters**: Explore data by criteria
- **Market Insights**: Premium and value sectors

## 🔧 Model Preparation

The `prepare-model.py` script automatically:

1. **Scans** `../outputs/pipeline/models/` for available models
2. **Selects** the best model based on configured metric
3. **Converts** to ONNX format for fast inference
4. **Saves** to `models/bestmodel/` directory:
   - `model.onnx` - Optimized inference model
   - `preprocessor.pkl` - Data preprocessing pipeline
   - `metadata.json` - Model information and metrics

## 🌐 Streamlit Cloud Deployment

### Prerequisites
- GitHub repository with the streamlit-app code
- Streamlit Cloud account (free)

### Deployment Steps

1. **Prepare Model Locally**:
   ```bash
   cd streamlit-app
   python prepare-model.py
   ```

2. **Commit to GitHub**:
   ```bash
   git add .
   git commit -m "Add Streamlit PropertyAI app"
   git push origin main
   ```

3. **Deploy to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file: `app.py`
   - Deploy!

4. **Access Your App**:
   - Your app will be available at: `https://[username]-[repo-name]-[branch]-[hash].streamlit.app`

### Configuration
- **Python Version**: 3.9+ (automatically detected)
- **Dependencies**: Installed from `requirements.txt`
- **Configuration**: Applied from `.streamlit/config.toml`

## 📊 Performance

### Streamlit Cloud Specs
- **Memory**: 1GB RAM limit
- **CPU**: Shared resources
- **Storage**: Temporary file system
- **Timeout**: 5-minute inactivity timeout

### Optimization Features
- **Model Caching**: `@st.cache_resource` for model loading
- **Data Caching**: `@st.cache_data` for static data
- **ONNX Runtime**: 10-50x faster than sklearn
- **Minimal Dependencies**: Optimized requirements

## 🎯 Usage Examples

### Property Valuation
1. Select property type (Apartment/House)
2. Choose sector from dropdown
3. Enter property details (area, rooms, bathrooms)
4. Set coordinates using the interactive map
5. Click "Calculate Property Value" for instant results

### Model Information
- View detailed model specifications
- Explore performance metrics with interactive charts
- Access technical details and architecture info

### Analytics Dashboard
- Explore market trends and insights
- Filter properties by sector, price, and area
- Compare different sectors and property types

## 🔗 Related Applications

- **Flask App**: Production API at `../flask-app/`
- **Main Pipeline**: ML training pipeline at `../`
- **Documentation**: Complete project docs at `../docs/`

## 💡 Key Benefits

### For Users
- ✅ **User-Friendly**: Intuitive Streamlit interface
- ✅ **Fast**: ONNX-optimized predictions
- ✅ **Interactive**: Real-time maps and visualizations
- ✅ **Accessible**: No installation required

### For Developers
- ✅ **Zero Infrastructure**: Streamlit Cloud hosting
- ✅ **Easy Deployment**: Git push to deploy
- ✅ **Cost-Effective**: Free hosting tier
- ✅ **Scalable**: Automatic scaling by Streamlit

### For Business
- ✅ **Professional**: Modern, clean interface
- ✅ **Reliable**: Production-ready ML models
- ✅ **Shareable**: Public URL for demos
- ✅ **Analytics**: Built-in market insights

## 🚀 Next Steps

1. **Test Locally**: Run `streamlit run app.py`
2. **Customize**: Modify UI, add features, enhance analytics
3. **Deploy**: Push to GitHub and deploy to Streamlit Cloud
4. **Share**: Use the public URL for demos and presentations

The Streamlit app provides a user-friendly interface for the PropertyAI system, perfect for demos, client presentations, and end-user interactions!