import os
import pickle
import json
import logging
from typing import Any, Optional, Dict
from datetime import datetime

from src.config import MODEL_DIR, PIPELINE_MODELS_DIR, JUPYTER_MODELS_DIR
from src.utils.helpers import get_versioned_filename

logger = logging.getLogger(__name__)


def save_model(model: Any, path: Optional[str] = None, model_type: str = 'model',
               base_name: Optional[str] = None, save_target: str = 'pipeline') -> str:
    if path is None:
        directory = PIPELINE_MODELS_DIR if save_target == 'pipeline' else JUPYTER_MODELS_DIR if save_target == 'jupyter' else MODEL_DIR
        path = get_versioned_filename(
            base_name=base_name or 'model',
            file_type=model_type,
            directory=directory,
            extension='pkl'
        )
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    
    return path


def load_model(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_model_metadata(model_path: str, metadata: Dict[str, Any], metadata_path: Optional[str] = None) -> str:
    if metadata_path is None:
        model_filename = os.path.basename(model_path)
        parts = model_filename.split('_')
        
        if len(parts) >= 3:
            model_type = parts[1]
            base_name = '_'.join(parts[2:]).split('.')[0]
        else:
            model_type = 'model'
            base_name = os.path.basename(model_path).split('.')[0]
        
        metadata_path = get_versioned_filename(
            base_name=base_name,
            file_type=f"{model_type}_metadata",
            directory=os.path.dirname(model_path),
            extension='json'
        )
    
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    metadata['timestamp'] = datetime.now().isoformat()
    metadata['model_path'] = model_path
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return metadata_path


def load_model_metadata(metadata_path: str) -> Dict[str, Any]:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def list_saved_models(model_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    if model_dir is None:
        model_dir = MODEL_DIR
    
    if not os.path.exists(model_dir):
        return {}
    
    models = {}
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith('.pkl'):
                model_path = os.path.join(root, file)
                model_name = os.path.basename(model_path).split('.')[0]
                
                # Try to find metadata file
                metadata_path = None
                for metadata_file in os.listdir(root):
                    if metadata_file.endswith('.json') and model_name in metadata_file:
                        metadata_path = os.path.join(root, metadata_file)
                        break
                
                if metadata_path and os.path.exists(metadata_path):
                    try:
                        models[model_path] = load_model_metadata(metadata_path)
                    except Exception:
                        models[model_path] = {"timestamp": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()}
                else:
                    models[model_path] = {"timestamp": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()}
    
    return models


def get_best_model(model_dir: Optional[str] = None, metric: str = 'rmse', load_target: str = 'pipeline') -> Optional[str]:
    if model_dir is None:
        model_dir = PIPELINE_MODELS_DIR if load_target == 'pipeline' else JUPYTER_MODELS_DIR if load_target == 'jupyter' else MODEL_DIR
    
    models = list_saved_models(model_dir)
    if not models:
        return None
    
    models_with_metrics = {
        path: meta for path, meta in models.items()
        if meta.get('evaluation_metrics', {}).get(metric.lower())
    }
    
    if not models_with_metrics:
        return None
    
    return min(models_with_metrics.items(), key=lambda x: x[1]['evaluation_metrics'][metric.lower()])[0]


def load_best_model(model_dir: Optional[str] = None, metric: str = 'rmse', load_target: str = 'pipeline') -> Any:
    best_model_path = get_best_model(model_dir, metric, load_target)
    if not best_model_path:
        raise FileNotFoundError(f"No models found with metric '{metric}'")
    return load_model(best_model_path)