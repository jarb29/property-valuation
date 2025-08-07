import sys
import os
from pathlib import Path

from config.settings import Config


class PathManager:
    """Utility class for managing Python paths and imports"""
    
    @staticmethod
    def setup_paths():
        """Setup Python paths for imports"""
        paths_to_add = [
            str(Config.SRC_DIR),
            str(Config.BASE_DIR)
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # Set PYTHONPATH environment variable
        current_pythonpath = os.environ.get('PYTHONPATH', '')
        new_pythonpath = ':'.join(paths_to_add + [current_pythonpath] if current_pythonpath else paths_to_add)
        os.environ['PYTHONPATH'] = new_pythonpath
    
    @staticmethod
    def safe_import(module_name: str, fallback_value=None):
        """Safely import a module with fallback"""
        try:
            module = __import__(module_name, fromlist=[''])
            return module
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")
            return fallback_value