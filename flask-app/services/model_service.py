import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple

from config.settings import Config
from exceptions.model_exceptions import ModelNotFoundError, ModelLoadError


class ModelService:
    """Service for model discovery and metadata management"""
    
    @staticmethod
    def list_saved_models(model_dir: Path) -> Dict[str, Dict[str, Any]]:
        """List all models with their metadata"""
        models = {}
        
        if not model_dir.exists():
            return models
            
        for file in model_dir.glob('*.pkl'):
            metadata_path = ModelService._get_metadata_path(file)
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    models[str(file)] = metadata
                except Exception as e:
                    print(f"Warning: Could not load metadata for {file.name}: {e}")
                        
        return models
    
    @staticmethod
    def _get_metadata_path(model_path: Path) -> Path:
        """Generate metadata file path from model path"""
        base_name = model_path.stem
        parts = base_name.split('_')
        if len(parts) >= 3:
            parts.insert(2, 'metadata')
            metadata_filename = '_'.join(parts) + '.json'
        else:
            metadata_filename = base_name + '_metadata.json'
        
        return model_path.parent / metadata_filename
    
    @staticmethod
    def get_best_model() -> Tuple[str, Dict[str, Any]]:
        """Get best model path based on metadata and metrics"""
        model_dir = Config.get_model_dir()
        models = ModelService.list_saved_models(model_dir)
        
        if not models:
            raise ModelNotFoundError(f"No models found in {model_dir}")
        
        # Filter models with the target metric
        models_with_metrics = {
            path: meta for path, meta in models.items()
            if meta.get('evaluation_metrics', {}).get(Config.MODEL_METRIC.lower())
        }
        
        if not models_with_metrics:
            model_path = list(models.keys())[0]
            return model_path, models[model_path]
        
        # Select best model based on metric
        best_model_path, best_metadata = min(
            models_with_metrics.items(), 
            key=lambda x: x[1]['evaluation_metrics'][Config.MODEL_METRIC.lower()]
        )
        
        return best_model_path, best_metadata
    
    @staticmethod
    def load_model(model_path: str):
        """Load pickle model"""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {model_path}: {str(e)}")