import sys
import os
from pathlib import Path

print("=== APP RUNNER DEBUG START ===", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"Current working directory: {Path.cwd()}", flush=True)
print(f"__file__ location: {__file__}", flush=True)
print(f"Files in current dir: {os.listdir('.')}", flush=True)

# Add current directory to Python path
print("Adding paths to sys.path...", flush=True)
sys.path.insert(0, str(Path(__file__).parent))
print(f"Added to path: {str(Path(__file__).parent)}", flush=True)

# Add src directory to Python path
src_path = str(Path(__file__).resolve().parent.parent / 'src')
print(f"Checking src path: {src_path}", flush=True)
print(f"Src path exists: {Path(src_path).exists()}", flush=True)
if src_path not in sys.path:
    sys.path.append(src_path)
    print(f"Added src to path: {src_path}", flush=True)

print(f"Final Python path: {sys.path[:5]}...", flush=True)

print("Importing Flask...", flush=True)
from flask import Flask, render_template, request, jsonify
print("Flask imported successfully", flush=True)

print("Importing PropertyPredictor...", flush=True)
try:
    from models.predictor import PropertyPredictor
    print("PropertyPredictor imported successfully", flush=True)
except Exception as e:
    print(f"ERROR importing PropertyPredictor: {e}", flush=True)
    import traceback
    traceback.print_exc()
    PropertyPredictor = None

print("Creating Flask app...", flush=True)
app = Flask(__name__)
print("Flask app created", flush=True)

print("🚀 Starting Flask application...", flush=True)
print(f"📁 Current working directory: {Path.cwd()}", flush=True)
print(f"🐍 Python path: {sys.path[:3]}...", flush=True)  # Show first 3 paths

# Initialize predictor (will auto-select best model)
try:
    print("📦 Loading ML model...", flush=True)
    predictor = PropertyPredictor()
    print("✅ Model loaded successfully", flush=True)
    print(f"📊 Model info: {predictor.get_model_info()}", flush=True)
except Exception as e:
    print(f"❌ Error loading model: {e}", flush=True)
    import traceback
    traceback.print_exc()
    predictor = None

print("🌐 Flask app initialized, ready to serve requests", flush=True)

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

        # Validate required fields and collect missing ones
        required_fields = predictor.get_feature_names()
        missing_fields = []
        empty_fields = []
        
        for field in required_fields:
            if field not in features:
                missing_fields.append(field)
            elif not features[field] or str(features[field]).strip() == '':
                empty_fields.append(field)
        
        # Return validation errors with field highlighting info
        if missing_fields or empty_fields:
            error_fields = missing_fields + empty_fields
            return jsonify({
                'error': 'Please fill in all required fields',
                'validation_errors': {
                    'missing_fields': missing_fields,
                    'empty_fields': empty_fields,
                    'error_fields': error_fields
                }
            }), 400

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

@app.route('/map')
def advanced_map():
    """Advanced 3D map visualization page"""
    return render_template('map.html')

if __name__ == '__main__':
    print("Starting Flask server on 0.0.0.0:8080...", flush=True)
    app.run(host='0.0.0.0', port=8080, debug=False)