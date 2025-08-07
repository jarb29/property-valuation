class ModelError(Exception):
    """Base exception for model-related errors"""
    pass

class ModelNotFoundError(ModelError):
    """Raised when no models are found"""
    pass

class ModelLoadError(ModelError):
    """Raised when model loading fails"""
    pass

class ONNXConversionError(ModelError):
    """Raised when ONNX conversion fails"""
    pass

class PredictionError(ModelError):
    """Raised when prediction fails"""
    pass