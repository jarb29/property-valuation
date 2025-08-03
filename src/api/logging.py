"""
Logging configuration.

This module provides logging configuration for the API.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

from src.config import LOG_DIR, LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the log record.

    This formatter can output in two formats:
    1. Standard JSON (default): One-line JSON for machine processing
    2. Pretty JSON: Indented, human-readable JSON with formatted timestamp
    """

    def __init__(self, pretty=False):
        """
        Initialize the formatter.

        Args:
            pretty (bool): Whether to use pretty formatting for human readability.
                           Default is False (compact JSON for machine processing).
        """
        super().__init__()
        self.pretty = pretty

    def format_timestamp(self, dt):
        """Format timestamp in a human-readable way."""
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def format(self, record):
        """
        Format the log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            str: The formatted log record as a JSON string.
        """
        # Get current time
        now = datetime.utcnow()

        # Create the log object with basic fields
        logobj = {
            'timestamp': self.format_timestamp(now) if self.pretty else now.isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
        }

        # Add exception info if available
        if record.exc_info:
            logobj['exception'] = self.formatException(record.exc_info)

        # Add request_id if available
        if hasattr(record, 'request_id'):
            logobj['request_id'] = record.request_id

        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                          'funcName', 'id', 'levelname', 'levelno', 'lineno', 'module',
                          'msecs', 'message', 'msg', 'name', 'pathname', 'process',
                          'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName']:
                logobj[key] = value

        # Format as JSON with or without pretty printing
        if self.pretty:
            return json.dumps(logobj, indent=2, sort_keys=True)
        else:
            return json.dumps(logobj)


def setup_logging():
    """
    Set up logging configuration.
    """
    # Create log directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set log level
    root_logger.setLevel(getattr(logging, LOG_LEVEL.upper()))

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(console_handler)

    # Create file handler for general logs
    # api.log contains all API-related logs including uvicorn server logs
    general_log_path = os.path.join(LOG_DIR, 'api.log')
    general_file_handler = RotatingFileHandler(
        general_log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
    )
    general_file_handler.setFormatter(JsonFormatter(pretty=True))
    root_logger.addHandler(general_file_handler)

    # Log the location of the log file for user reference
    logging.info(f"API logs are being written to: {general_log_path}")

    # Create file handler for error logs
    # error.log contains only ERROR and higher level logs for easier troubleshooting
    error_log_path = os.path.join(LOG_DIR, 'error.log')
    error_file_handler = RotatingFileHandler(
        error_log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(JsonFormatter(pretty=True))
    root_logger.addHandler(error_file_handler)

    # Set up specific loggers
    setup_specific_loggers()

    # Log that logging has been set up
    logging.info(f"Logging set up with level {LOG_LEVEL}")


def setup_specific_loggers():
    """
    Set up specific loggers for different components.
    """
    # Set up API logger
    api_logger = logging.getLogger('src.api')
    api_logger.setLevel(getattr(logging, LOG_LEVEL.upper()))

    # Set up models logger
    models_logger = logging.getLogger('src.models')
    models_logger.setLevel(getattr(logging, LOG_LEVEL.upper()))

    # Set up data logger
    data_logger = logging.getLogger('src.data')
    data_logger.setLevel(getattr(logging, LOG_LEVEL.upper()))

    # Set up third-party loggers
    # Set uvicorn and fastapi to WARNING level to reduce noise
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.WARNING)


class RequestIdFilter(logging.Filter):
    """
    Filter that adds request_id to log records.
    """

    def __init__(self, request_id):
        """
        Initialize the filter with a request ID.

        Args:
            request_id: The request ID to add to log records.
        """
        super().__init__()
        self.request_id = request_id

    def filter(self, record):
        """
        Add request_id to the log record.

        Args:
            record: The log record to filter.

        Returns:
            bool: Always returns True to include the record.
        """
        record.request_id = self.request_id
        return True


def get_request_logger(request_id):
    """
    Get a logger with the request ID added to log records.

    Args:
        request_id: The request ID to add to log records.

    Returns:
        logging.Logger: A logger with the request ID filter.
    """
    logger = logging.getLogger('src.api.request')
    logger.addFilter(RequestIdFilter(request_id))
    return logger
