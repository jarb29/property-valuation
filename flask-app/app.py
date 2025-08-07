import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))
# Add src directory to Python path
src_path = str(Path(__file__).resolve().parent.parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from flask import Flask, render_template, request, jsonify
from models.predictor import PropertyPredictor

app = Flask(__name__)


# Initialize predictor (will auto-select best model)
try:
    predictor = PropertyPredictor()
    print("✅ Model loaded successfully")
    print(f"📊 Model info: {predictor.get_model_info()}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    predictor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not predictor:
        return jsonify({'error': 'Model not available'}), 503

    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid request format'}), 400

        features = data['features']

        # Validate required fields
        required_fields = predictor.get_feature_names()
        for field in required_fields:
            if field not in features:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Make prediction
        import time
        start_time = time.time()
        prediction = predictor.predict(features)
        prediction_time = time.time() - start_time

        model_info = predictor.get_model_info()

        return jsonify({
            'prediction': prediction,
            'prediction_time': prediction_time,
            'model_version': model_info.get('model_type', 'unknown'),
            'features': features
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    if predictor:
        model_info = predictor.get_model_info()
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "model_metrics": model_info['metrics'],
            "model_type": model_info['model_type']
        })
    else:
        return jsonify({"status": "unhealthy", "model_loaded": False}), 503

@app.route('/model/info')
def model_info_page():
    """Model information page"""
    try:
        if predictor:
            model_info = predictor.get_model_info()
            print(f"Model info: {model_info}")  # Debug
            return render_template('model_info.html', model_info=model_info)
        else:
            return render_template('index.html')
    except Exception as e:
        print(f"Model info error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500

@app.route('/api/model/info')
def model_info_api():
    """Get detailed model information via API"""
    if predictor:
        return jsonify(predictor.get_model_info())
    else:
        return jsonify({"error": "Model not loaded"}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)