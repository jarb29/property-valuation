import os
import logging
import logging.handlers
from typing import Optional, Dict, Any, List
import platform
import time
import traceback
import psutil
import json

from src.config import (PIPELINE_LOGS_DIR, LOG_MAX_BYTES, LOG_BACKUP_COUNT,
                        PREDICTION_LOGS_DIR, SCHEMA_VALIDATION_LOGS_DIR)

# Color codes for console output
COLORS_ENABLED = platform.system() != "Windows" or os.environ.get("ANSICON") is not None
RESET = '\033[0m' if COLORS_ENABLED else ''
BRIGHT_GREEN = '\033[92m' if COLORS_ENABLED else ''
BRIGHT_YELLOW = '\033[93m' if COLORS_ENABLED else ''
BRIGHT_RED = '\033[91m' if COLORS_ENABLED else ''
BOLD = '\033[1m' if COLORS_ENABLED else ''


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'INFO': BRIGHT_GREEN,
        'WARNING': BRIGHT_YELLOW,
        'ERROR': BRIGHT_RED,
        'CRITICAL': BOLD + BRIGHT_RED,
    }

    def format(self, record):
        formatted_message = super().format(record)
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            level_start = formatted_message.find(record.levelname)
            if level_start != -1:
                level_end = level_start + len(record.levelname)
                formatted_message = (
                    formatted_message[:level_start] +
                    color + record.levelname + RESET +
                    formatted_message[level_end:]
                )
        return formatted_message


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None,
                 log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                 log_to_console: bool = True) -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handlers = []

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(ColoredFormatter(log_format))
        handlers.append(console_handler)

    # File handler with rotation
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    for handler in handlers:
        root_logger.addHandler(handler)


def get_logger(name: str, log_level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, log_level.upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
    return logger


def setup_pipeline_logging(log_level: str = 'INFO', process_name: str = 'Pipeline',
                        log_to_console: bool = True) -> str:
    process_name_clean = process_name.replace(' ', '')
    log_file = os.path.join(PIPELINE_LOGS_DIR, f"{process_name_clean}.log")
    
    setup_logging(log_level, log_file, log_to_console=log_to_console)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {process_name} - Log file: {log_file}")
    
    return log_file


def configure_logging(app_name: str = 'PropertyValuation', log_level: str = 'INFO',
                     log_to_console: bool = True, log_to_file: bool = True,
                     log_file: Optional[str] = None, log_format: Optional[str] = None,
                     date_format: Optional[str] = None, use_colors: bool = True,
                     max_bytes: Optional[int] = None, backup_count: Optional[int] = None,
                     exclude_modules: Optional[List[str]] = None) -> str:
    """Configure logging with custom parameters."""
    if log_to_file and log_file is None:
        app_name_clean = app_name.replace(' ', '')
        log_file = os.path.join(PIPELINE_LOGS_DIR, f"{app_name_clean}.log")
    
    # Use default format if not provided
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Use provided parameters or defaults
    max_bytes = max_bytes or LOG_MAX_BYTES
    backup_count = backup_count or LOG_BACKUP_COUNT
    
    # Set up logging with custom parameters
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handlers = []

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        if use_colors:
            console_handler.setFormatter(ColoredFormatter(log_format))
        else:
            console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)

    # File handler with custom rotation settings
    if log_to_file and log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    for handler in handlers:
        root_logger.addHandler(handler)
    
    return log_file if log_to_file else ""


class ContextLogger:
    """Simple context logger for compatibility."""
    def __init__(self, logger: logging.Logger, context: Optional[dict] = None):
        self.logger = logger
        self.context = context or {}
    
    def _format_context(self) -> str:
        if not self.context:
            return ""
        context_items = [f"{k}={v}" for k, v in self.context.items()]
        return f"[{' '.join(context_items)}] "
    
    def info(self, msg: str, *args, **kwargs) -> None:
        self.logger.info(f"{self._format_context()}{msg}", *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        self.logger.warning(f"{self._format_context()}{msg}", *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        self.logger.error(f"{self._format_context()}{msg}", *args, **kwargs)


class PerformanceLoggerContext:
    """Context manager for performance logging."""
    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name} in {duration:.2f}s")
        else:
            self.logger.error(f"Failed {self.operation_name} after {duration:.2f}s")


class LogSummary:
    """Simple log summary collector."""
    def __init__(self, title: str):
        self.title = title
        self.entries: List[str] = []
    
    def add_info(self, message: str) -> None:
        self.entries.append(f"✓ {message}")
    
    def add_error(self, message: str) -> None:
        self.entries.append(f"✗ {message}")
    
    def log_summary(self, logger: logging.Logger) -> None:
        logger.info(f"\n{self.title}:")
        for entry in self.entries:
            logger.info(f"  {entry}")


def log_memory_usage(logger: logging.Logger, context: str = "Memory usage") -> None:
    """Log current memory usage."""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"{context}: {memory_mb:.1f} MB")
    except Exception:
        logger.debug("Could not retrieve memory usage")


def log_exception(logger: logging.Logger, message: str, exception: Exception) -> None:
    """Log exception with traceback."""
    logger.error(f"{message}: {str(exception)}")
    logger.debug(f"Traceback: {traceback.format_exc()}")


def log_model_prediction(model_name: str, input_data: dict, prediction: float, 
                        model_version: str, processing_time: float) -> None:
    """Log model prediction for monitoring."""
    logger = logging.getLogger('model_predictions')
    
    # Ensure the logger has file handlers
    if not logger.handlers:
        _setup_prediction_logger(logger)
    
    log_entry = {
        'timestamp': time.time(),
        'model_name': model_name,
        'model_version': model_version,
        'input_data': input_data,
        'prediction': prediction,
        'processing_time': processing_time
    }
    logger.info(json.dumps(log_entry))


def log_schema_validation(validation_type: str, metadata: dict) -> None:
    """Log schema validation results."""
    logger = logging.getLogger('schema_validation')
    
    # Ensure the logger has file handlers
    if not logger.handlers:
        _setup_validation_logger(logger)
    
    log_entry = {
        'validation_type': validation_type,
        'metadata': metadata
    }
    logger.info(json.dumps(log_entry))


def _setup_prediction_logger(logger: logging.Logger) -> None:
    """Setup file handler for prediction logger."""
    handler = logging.handlers.RotatingFileHandler(
        os.path.join(PREDICTION_LOGS_DIR, 'predictions.log'),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _setup_validation_logger(logger: logging.Logger) -> None:
    """Setup file handler for validation logger."""
    handler = logging.handlers.RotatingFileHandler(
        os.path.join(SCHEMA_VALIDATION_LOGS_DIR, 'schema_validation.log'),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)