#!/usr/bin/env python3
import sys
import json
import pickle
import shutil
from pathlib import Path

import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def setup_paths():
    """Setup Python paths for imports"""
    flask_dir = Path(__file__).parent
    sys.path.insert(0, str(flask_dir.parent / "src"))
    sys.path.insert(0, str(flask_dir.parent))
    return flask_dir


def find_model_files(models_dir):
    """Find all model files with their metadata"""
    models = {}
    
    for pkl_file in models_dir.glob("*.pkl"):
        # Generate metadata filename
        base_name = pkl_file.stem
        parts = base_name.split('_')
        
        if len(parts) >= 3:
            parts.insert(2, 'metadata')
            metadata_filename = '_'.join(parts) + '.json'
        else:
            metadata_filename = f"{base_name}_metadata.json"
            
        metadata_path = models_dir / metadata_filename
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            models[str(pkl_file)] = metadata
    
    return models


def select_best_model(models, metric):
    """Select the best model based on the given metric"""
    models_with_metrics = {
        path: meta for path, meta in models.items()
        if meta.get('evaluation_metrics', {}).get(metric)
    }
    
    if not models_with_metrics:
        # Fallback to first available model
        return Path(list(models.keys())[0])
    
    best_model_path, _ = min(
        models_with_metrics.items(), 
        key=lambda x: x[1]['evaluation_metrics'][metric]
    )
    
    return Path(best_model_path)


def get_metadata_file(model_file):
    """Get corresponding metadata file for a model"""
    base_name = model_file.stem
    parts = base_name.split('_')
    
    if len(parts) >= 3:
        parts.insert(2, 'metadata')
        metadata_filename = '_'.join(parts) + '.json'
    else:
        metadata_filename = f"{base_name}_metadata.json"
        
    return model_file.parent / metadata_filename


def convert_to_onnx(pipeline_model, output_path):
    """Convert sklearn model to ONNX format"""
    # Create dummy data to determine feature count
    dummy_data = pd.DataFrame({
        'type': ['departamento'], 'sector': ['las condes'],
        'net_usable_area': [100.0], 'net_area': [120.0],
        'n_rooms': [3], 'n_bathroom': [2],
        'latitude': [-33.4], 'longitude': [-70.5]
    })
    
    processed_dummy = pipeline_model.data_processor.transform(dummy_data)
    n_features = processed_dummy.shape[1]
    
    # Convert to ONNX
    initial_types = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(pipeline_model.model, initial_types=initial_types)
    
    # Save ONNX model
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def prepare_best_model():
    """Main function to prepare the best model for Flask app"""
    # Setup
    flask_dir = setup_paths()
    from config import MODEL_METRIC
    
    # Paths
    pipeline_models_dir = flask_dir.parent / "outputs" / "pipeline" / "models"
    bestmodel_dir = flask_dir / "models" / "bestmodel"
    bestmodel_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and select best model
    print("🔍 Scanning for available models...")
    models = find_model_files(pipeline_models_dir)
    
    if not models:
        raise FileNotFoundError(f"No models found in {pipeline_models_dir}")
    
    print(f"📊 Found {len(models)} models")
    
    metric = MODEL_METRIC.lower()
    model_file = select_best_model(models, metric)
    metadata_file = get_metadata_file(model_file)
    
    print(f"🏆 Selected best model: {model_file.name} (metric: {metric})")
    
    # Load model
    print("📦 Loading model...")
    with open(model_file, 'rb') as f:
        pipeline_model = pickle.load(f)
    
    # Save components
    print("💾 Saving model components...")
    
    # 1. Copy metadata
    if metadata_file.exists():
        shutil.copy2(metadata_file, bestmodel_dir / "metadata.json")
    
    # 2. Save preprocessor
    with open(bestmodel_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(pipeline_model.data_processor, f)
    
    # 3. Convert and save ONNX model
    print("🔄 Converting to ONNX...")
    convert_to_onnx(pipeline_model, bestmodel_dir / "model.onnx")
    
    # Summary
    files = list(bestmodel_dir.glob('*'))
    print("✅ Best model prepared successfully")
    print(f"📁 Created {len(files)} files: {[f.name for f in files]}")


if __name__ == "__main__":
    prepare_best_model()