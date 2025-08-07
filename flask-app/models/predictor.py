from typing import Dict, Any

from pathlib import Path
import pickle
import json
try:
    import onnxruntime as ort
except ImportError:
    ort = None

class PropertyPredictor:
    def __init__(self):
        # Setup paths for imports
        import sys
        flask_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(flask_dir.parent / "src"))
        sys.path.insert(0, str(flask_dir.parent))
        
        # Load from bestmodel directory
        bestmodel_dir = Path(__file__).parent / "bestmodel"
        
        # Load preprocessor only
        with open(bestmodel_dir / "preprocessor.pkl", 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        # Load metadata
        with open(bestmodel_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load ONNX model
        if ort is None:
            raise Exception("ONNX runtime not available")
        self.model = ort.InferenceSession(str(bestmodel_dir / "model.onnx"))
        
        print(f"✅ Using best model from: {bestmodel_dir}")
    

    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metrics"""
        # Import config to get the selection metric
        import sys
        flask_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(flask_dir.parent / "src"))
        from config import MODEL_METRIC
        
        metrics = self.metadata.get('evaluation_metrics', {})
        selection_metric = MODEL_METRIC.lower()
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
    
    def predict(self, features):
        """Make prediction for property valuation using ONNX"""
        import pandas as pd
        import numpy as np
        
        # Preprocess
        data = pd.DataFrame([features])
        processed_data = self.preprocessor.transform(data)
        input_data = {"float_input": processed_data.astype(np.float32)}
        
        # Predict
        result = self.model.run(None, input_data)
        return int(result[0][0])
    
    def get_feature_names(self):
        """Get expected feature names"""
        return self.metadata.get('features', [])