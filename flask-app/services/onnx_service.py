import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

try:
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from config.settings import Config
from exceptions.model_exceptions import ONNXConversionError, PredictionError


class ONNXService:
    """Service for ONNX model conversion and inference"""
    
    def __init__(self):
        if not ONNX_AVAILABLE:
            raise ONNXConversionError("ONNX libraries required. Install with: pip install onnxruntime skl2onnx")
    
    @staticmethod
    def convert_to_onnx(model_path: str, pipeline_model) -> str:
        """Convert pickle model to ONNX format"""
        onnx_path = model_path.replace('.pkl', '.onnx')
        
        try:
            print(f"Converting {model_path} to ONNX...")
            
            # Get processed feature count by fitting on dummy data
            dummy_data = pd.DataFrame({
                'type': ['departamento'],
                'sector': ['las condes'],
                'net_usable_area': [100.0],
                'net_area': [120.0],
                'n_rooms': [3],
                'n_bathroom': [2],
                'latitude': [-33.4],
                'longitude': [-70.5]
            })
            
            processed_dummy = pipeline_model.data_processor.transform(dummy_data)
            n_features = processed_dummy.shape[1]
            
            # Convert only the model part (after preprocessing)
            initial_types = [('float_input', FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(pipeline_model.model, initial_types=initial_types)
            
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"✅ Model converted to ONNX: {onnx_path}")
            print(f"Features after preprocessing: {n_features}")
            return onnx_path
            
        except Exception as e:
            raise ONNXConversionError(f"ONNX conversion failed: {str(e)}")
    
    @staticmethod
    def load_onnx_model(onnx_path: str):
        """Load ONNX model"""
        try:
            return ort.InferenceSession(onnx_path)
        except Exception as e:
            raise ONNXConversionError(f"Error loading ONNX model: {str(e)}")
    
    @staticmethod
    def prepare_onnx_input(features: Dict[str, Any], preprocessor) -> Dict[str, np.ndarray]:
        """Prepare input data for ONNX model"""
        try:
            # Create DataFrame and preprocess using the stored preprocessor
            data = pd.DataFrame([features])
            processed_data = preprocessor.transform(data)
            
            # Return as dictionary for ONNX
            return {"float_input": processed_data.astype(np.float32)}
        except Exception as e:
            raise PredictionError(f"Input preparation failed: {str(e)}")
    
    @staticmethod
    def predict_with_onnx(model, input_data: Dict[str, np.ndarray]) -> int:
        """Make prediction using ONNX model"""
        try:
            result = model.run(None, input_data)
            return int(result[0][0])
        except Exception as e:
            raise PredictionError(f"ONNX prediction failed: {str(e)}")