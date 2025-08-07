import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class PropertyPredictor:
    def __init__(self):
        # Add src to path for preprocessor dependencies
        src_path = Path(__file__).parent.parent.parent / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        # Import MODEL_METRIC from config
        try:
            from config import MODEL_METRIC
            self.model_metric = MODEL_METRIC.lower()
        except ImportError:
            self.model_metric = "mae"  # fallback
        
        # Load from bestmodel directory
        bestmodel_dir = Path(__file__).parent / "bestmodel"
        
        if not bestmodel_dir.exists():
            raise FileNotFoundError(f"Best model directory not found: {bestmodel_dir}")
        
        # Load preprocessor
        preprocessor_path = bestmodel_dir / "preprocessor.pkl"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        # Load metadata
        metadata_path = bestmodel_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Load ONNX model
        if ort is None:
            raise Exception("ONNX runtime not available. Install with: pip install onnxruntime")
        
        onnx_path = bestmodel_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        self.model = ort.InferenceSession(str(onnx_path))
        
        print(f"✅ Model loaded successfully from: {bestmodel_dir}")
    
    def predict(self, features: Dict[str, Any]) -> int:
        """Make prediction for property valuation using ONNX"""
        # Preprocess
        data = pd.DataFrame([features])
        processed_data = self.preprocessor.transform(data)
        input_data = {"float_input": processed_data.astype(np.float32)}
        
        # Predict
        result = self.model.run(None, input_data)
        return int(result[0][0])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metrics"""
        # Use selection metric from config
        selection_metric = self.model_metric
        
        metrics = self.metadata.get('evaluation_metrics', {})
        best_metric_value = metrics.get(selection_metric, 'N/A')
        
        return {
            'preprocessor_path': 'models/bestmodel/preprocessor.pkl',
            'onnx_model_path': 'models/bestmodel/model.onnx',
            'original_model_path': self.metadata.get('model_path', 'unknown'),
            'model_type': self.metadata.get('model_type', 'unknown'),
            'inference_engine': 'ONNX Runtime',
            'preprocessing_engine': 'Scikit-learn Pipeline',
            'selection_metric': selection_metric.upper(),
            'best_metric_value': best_metric_value,
            'metrics': metrics,
            'timestamp': self.metadata.get('timestamp', 'unknown'),
            'features': self.metadata.get('features', [])
        }
    
    def get_feature_names(self):
        """Get expected feature names"""
        return self.metadata.get('features', [
            'type', 'sector', 'net_usable_area', 'net_area', 
            'n_rooms', 'n_bathroom', 'latitude', 'longitude'
        ])