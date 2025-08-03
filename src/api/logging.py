"""
Logging configuration for the API.

Provides structured JSON logging with request tracking and error handling.
"""

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, Any

from src.config import (LOG_DIR, LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT,
                        API_LOGS_DIR, ERROR_LOGS_DIR)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    EXCLUDED_ATTRS = frozenset([
        'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
        'funcName', 'id', 'levelname', 'levelno', 'lineno', 'module',
        'msecs', 'message', 'msg', 'name', 'pathname', 'process',
        'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'
    ])

    def __init__(self, pretty: bool = False) -> None:
        super().__init__()
        self.pretty = pretty

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        for key, value in record.__dict__.items():
            if key not in self.EXCLUDED_ATTRS:
                log_data[key] = value
        
        return json.dumps(log_data, indent=2 if self.pretty else None, sort_keys=True)


def setup_logging() -> None:
    """Configure logging with console and file handlers."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(console_handler)
    
    _add_file_handlers(root_logger)
    _configure_third_party_loggers()
    
    logging.info(f"Logging configured with level {LOG_LEVEL}")


def _add_file_handlers(logger: logging.Logger) -> None:
    """Add rotating file handlers for general and error logs."""
    general_handler = RotatingFileHandler(
        os.path.join(API_LOGS_DIR, 'api.log'),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    general_handler.setFormatter(JsonFormatter(pretty=True))
    logger.addHandler(general_handler)
    
    error_handler = RotatingFileHandler(
        os.path.join(ERROR_LOGS_DIR, 'error.log'),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JsonFormatter(pretty=True))
    logger.addHandler(error_handler)


def _configure_third_party_loggers() -> None:
    """Configure third-party library loggers to reduce noise."""
    for logger_name in ['uvicorn', 'fastapi']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


class RequestIdFilter(logging.Filter):
    """Filter to add request ID to log records."""

    def __init__(self, request_id: str) -> None:
        super().__init__()
        self.request_id = request_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id to log record."""
        record.request_id = self.request_id
        return True


def get_request_logger(request_id: str) -> logging.Logger:
    """Get logger with request ID filter attached."""
    logger = logging.getLogger('src.api.request')
    logger.addFilter(RequestIdFilter(request_id))
    return logger
