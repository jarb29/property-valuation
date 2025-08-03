"""
Logging Configuration Module.

This module provides functions for configuring logging for the property valuation pipeline.
"""

import os
import logging
import logging.handlers
from typing import Optional, List, Dict, Any, Union, Callable
import json
from datetime import datetime
import time
import sys
import platform
import traceback
import psutil
import functools
import re
import uuid

from src.config import LOG_DIR, PIPELINE_LOGS_DIR, LOG_MAX_BYTES, LOG_BACKUP_COUNT, PREDICTION_SCHEMA

# ANSI color codes for colored console output
# Only use colors on platforms that support them (not Windows unless ANSICON or equivalent is used)
COLORS_ENABLED = platform.system() != "Windows" or os.environ.get("ANSICON") is not None

# Color codes
RESET = '\033[0m' if COLORS_ENABLED else ''
BOLD = '\033[1m' if COLORS_ENABLED else ''
BLACK = '\033[30m' if COLORS_ENABLED else ''
RED = '\033[31m' if COLORS_ENABLED else ''
GREEN = '\033[32m' if COLORS_ENABLED else ''
YELLOW = '\033[33m' if COLORS_ENABLED else ''
BLUE = '\033[34m' if COLORS_ENABLED else ''
MAGENTA = '\033[35m' if COLORS_ENABLED else ''
CYAN = '\033[36m' if COLORS_ENABLED else ''
WHITE = '\033[37m' if COLORS_ENABLED else ''
BRIGHT_BLACK = '\033[90m' if COLORS_ENABLED else ''
BRIGHT_RED = '\033[91m' if COLORS_ENABLED else ''
BRIGHT_GREEN = '\033[92m' if COLORS_ENABLED else ''
BRIGHT_YELLOW = '\033[93m' if COLORS_ENABLED else ''
BRIGHT_BLUE = '\033[94m' if COLORS_ENABLED else ''
BRIGHT_MAGENTA = '\033[95m' if COLORS_ENABLED else ''
BRIGHT_CYAN = '\033[96m' if COLORS_ENABLED else ''
BRIGHT_WHITE = '\033[97m' if COLORS_ENABLED else ''


class ColoredFormatter(logging.Formatter):
    """
    A formatter that adds colors to log messages based on their level.

    This formatter adds ANSI color codes to log messages to make them more
    distinguishable in the console output. Different colors are used for
    different log levels (e.g., red for ERROR, yellow for WARNING, etc.).
    """

    # Color mapping for different log levels
    COLORS = {
        'DEBUG': BRIGHT_BLACK,
        'INFO': BRIGHT_GREEN,
        'WARNING': BRIGHT_YELLOW,
        'ERROR': BRIGHT_RED,
        'CRITICAL': BOLD + BRIGHT_RED,
    }

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        """
        Initialize the formatter with the specified format string.

        Args:
            fmt (str, optional): Format string. Defaults to None.
            datefmt (str, optional): Date format string. Defaults to None.
            style (str, optional): Style of the format string. Defaults to '%'.
            validate (bool, optional): Whether to validate the format string. Defaults to True.
        """
        super().__init__(fmt, datefmt, style, validate)

    def format(self, record):
        """
        Format the specified record as text.

        Args:
            record (LogRecord): The log record to format.

        Returns:
            str: The formatted log record.
        """
        # Get the original formatted message
        formatted_message = super().format(record)

        # Add color based on the log level
        if record.levelname in self.COLORS:
            # Add color code before the level name and reset after the message
            color = self.COLORS[record.levelname]
            # Find the level name in the formatted message and add color
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
    """
    Set up logging configuration.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file (Optional[str]): Path to log file. If None, logs are only printed to console.
        log_format (str): Format string for log messages.
        log_to_console (bool): Whether to log to console.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handlers
    handlers = []

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        # Use ColoredFormatter for console output to improve readability
        console_handler.setFormatter(ColoredFormatter(log_format))
        handlers.append(console_handler)

    # File handler with rotation
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Use RotatingFileHandler to automatically rotate logs when they reach a certain size
        # Use size and backup count from config
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)


def setup_json_logging(log_level: str = 'INFO', log_file: Optional[str] = None,
                      log_to_console: bool = True) -> None:
    """
    Set up JSON logging configuration.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file (Optional[str]): Path to log file. If None, logs are only printed to console.
        log_to_console (bool): Whether to log to console.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create JSON formatter
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'name': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }

            # Add exception info if available
            if record.exc_info:
                log_record['exception'] = self.formatException(record.exc_info)

            return json.dumps(log_record)

    # Create handlers
    handlers = []

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(JsonFormatter())
        handlers.append(console_handler)

    # File handler with rotation
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Use RotatingFileHandler to automatically rotate logs when they reach a certain size
        # Use size and backup count from config
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(JsonFormatter())
        handlers.append(file_handler)

    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)


def get_logger(name: str, log_level: str = 'INFO') -> logging.Logger:
    """
    Get a logger with the specified name and level.

    Args:
        name (str): Name of the logger.
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, log_level.upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
    return logger


def log_memory_usage(logger: logging.Logger, message: str = "Current memory usage", level: int = logging.INFO) -> Dict[str, float]:
    """
    Log the current memory usage of the process.

    Args:
        logger (logging.Logger): The logger to use.
        message (str, optional): The message to log. Defaults to "Current memory usage".
        level (int, optional): The log level. Defaults to logging.INFO.

    Returns:
        Dict[str, float]: A dictionary with memory usage information.
    """
    try:
        # Get the current process
        process = psutil.Process(os.getpid())

        # Get memory info
        memory_info = process.memory_info()

        # Calculate memory usage in MB
        rss_mb = memory_info.rss / (1024 * 1024)  # Resident Set Size
        vms_mb = memory_info.vms / (1024 * 1024)  # Virtual Memory Size

        # Get system memory info
        system_memory = psutil.virtual_memory()
        system_memory_total_mb = system_memory.total / (1024 * 1024)
        system_memory_available_mb = system_memory.available / (1024 * 1024)
        system_memory_used_percent = system_memory.percent

        # Create memory usage dictionary
        memory_usage = {
            'rss_mb': rss_mb,
            'vms_mb': vms_mb,
            'system_total_mb': system_memory_total_mb,
            'system_available_mb': system_memory_available_mb,
            'system_used_percent': system_memory_used_percent
        }

        # Log memory usage
        logger.log(level, f"{message}: RSS={rss_mb:.2f}MB, VMS={vms_mb:.2f}MB, System={system_memory_used_percent:.1f}%")

        return memory_usage
    except Exception as e:
        logger.warning(f"Failed to log memory usage: {str(e)}")
        return {}


def log_performance_decorator(logger: logging.Logger, level: int = logging.INFO):
    """
    Decorator to log the execution time and memory usage of a function.

    Args:
        logger (logging.Logger): The logger to use.
        level (int, optional): The log level. Defaults to logging.INFO.

    Returns:
        Callable: The decorator function.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log start message
            logger.log(level, f"Starting {func.__name__}")

            # Log initial memory usage
            start_memory = log_memory_usage(logger, f"Memory before {func.__name__}", level)

            # Record start time
            start_time = time.time()

            try:
                # Call the function
                result = func(*args, **kwargs)

                # Record end time
                end_time = time.time()

                # Calculate execution time
                execution_time = end_time - start_time

                # Log final memory usage
                end_memory = log_memory_usage(logger, f"Memory after {func.__name__}", level)

                # Calculate memory difference
                memory_diff = {}
                for key in start_memory:
                    if key in end_memory:
                        memory_diff[key] = end_memory[key] - start_memory[key]

                # Log performance metrics
                logger.log(level, f"Completed {func.__name__} in {execution_time:.4f}s")
                if memory_diff:
                    logger.log(level, f"Memory change during {func.__name__}: RSS={memory_diff.get('rss_mb', 0):.2f}MB")

                return result
            except Exception as e:
                # Record end time even if an exception occurred
                end_time = time.time()
                execution_time = end_time - start_time

                # Log error with performance information
                logger.error(f"Error in {func.__name__} after {execution_time:.4f}s: {str(e)}", exc_info=e)

                # Re-raise the exception
                raise

        return wrapper

    return decorator


class LogSummary:
    """
    Collect and summarize important log messages.

    This class collects important log messages and can generate a summary at the end
    of a process or operation. This is useful for monitoring the application and
    identifying issues.

    Example usage:
    ```
    # Create a log summary
    summary = LogSummary()

    # Add messages to the summary
    summary.add_info("Process started")
    summary.add_warning("Missing optional data")
    summary.add_error("Failed to connect to database")

    # Generate and log the summary
    summary.log_summary(logger)
    ```
    """

    def __init__(self, title: str = "Operation Summary"):
        """
        Initialize the log summary.

        Args:
            title (str, optional): Title for the summary. Defaults to "Operation Summary".
        """
        self.title = title
        self.start_time = time.time()
        self.messages = {
            'info': [],
            'warning': [],
            'error': [],
            'critical': []
        }
        self.counts = {
            'info': 0,
            'warning': 0,
            'error': 0,
            'critical': 0
        }

    def add_message(self, level: str, message: str, timestamp: Optional[float] = None) -> None:
        """
        Add a message to the summary.

        Args:
            level (str): Message level ('info', 'warning', 'error', 'critical').
            message (str): The message to add.
            timestamp (Optional[float], optional): Message timestamp. Defaults to current time.
        """
        if level not in self.messages:
            raise ValueError(f"Invalid level: {level}. Must be one of {list(self.messages.keys())}")

        if timestamp is None:
            timestamp = time.time()

        self.messages[level].append({
            'message': message,
            'timestamp': timestamp,
            'time_str': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        })
        self.counts[level] += 1

    def add_info(self, message: str, timestamp: Optional[float] = None) -> None:
        """Add an info message to the summary."""
        self.add_message('info', message, timestamp)

    def add_warning(self, message: str, timestamp: Optional[float] = None) -> None:
        """Add a warning message to the summary."""
        self.add_message('warning', message, timestamp)

    def add_error(self, message: str, timestamp: Optional[float] = None) -> None:
        """Add an error message to the summary."""
        self.add_message('error', message, timestamp)

    def add_critical(self, message: str, timestamp: Optional[float] = None) -> None:
        """Add a critical message to the summary."""
        self.add_message('critical', message, timestamp)

    def get_summary(self) -> str:
        """
        Generate a text summary of the collected messages.

        Returns:
            str: The summary text.
        """
        duration = time.time() - self.start_time

        lines = [
            f"=== {self.title} ===",
            f"Duration: {duration:.2f}s",
            f"Message counts: {self.counts['info']} info, {self.counts['warning']} warnings, "
            f"{self.counts['error']} errors, {self.counts['critical']} critical",
            ""
        ]

        # Add critical messages
        if self.messages['critical']:
            lines.append("CRITICAL MESSAGES:")
            for msg in self.messages['critical']:
                lines.append(f"  [{msg['time_str']}] {msg['message']}")
            lines.append("")

        # Add error messages
        if self.messages['error']:
            lines.append("ERROR MESSAGES:")
            for msg in self.messages['error']:
                lines.append(f"  [{msg['time_str']}] {msg['message']}")
            lines.append("")

        # Add warning messages
        if self.messages['warning']:
            lines.append("WARNING MESSAGES:")
            for msg in self.messages['warning']:
                lines.append(f"  [{msg['time_str']}] {msg['message']}")
            lines.append("")

        # Add info messages (limited to 10)
        if self.messages['info']:
            info_count = len(self.messages['info'])
            lines.append(f"INFO MESSAGES (showing {min(10, info_count)} of {info_count}):")
            for msg in self.messages['info'][:10]:
                lines.append(f"  [{msg['time_str']}] {msg['message']}")

            if info_count > 10:
                lines.append(f"  ... and {info_count - 10} more info messages")
            lines.append("")

        lines.append(f"=== End of {self.title} ===")

        return "\n".join(lines)

    def log_summary(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        """
        Log the summary using the provided logger.

        Args:
            logger (logging.Logger): The logger to use.
            level (int, optional): Log level for the summary. Defaults to logging.INFO.
        """
        summary = self.get_summary()
        for line in summary.split('\n'):
            logger.log(level, line)

    def clear(self) -> None:
        """Clear all collected messages."""
        for level in self.messages:
            self.messages[level] = []
            self.counts[level] = 0
        self.start_time = time.time()


class ContextLogger:
    """
    Context-based logger for tracking operations across multiple functions.

    This logger maintains context information (such as a request ID, operation name,
    or user ID) and adds it to all log messages. This helps with tracing the flow of
    execution and correlating log messages from different parts of the code.

    Example usage:
    ```
    # Create a context logger
    ctx_logger = ContextLogger(logger, {'request_id': '123', 'user_id': '456'})

    # Log messages with context
    ctx_logger.info("Processing request")

    # Add more context
    ctx_logger.add_context('operation', 'data_processing')
    ctx_logger.info("Starting data processing")

    # Create a child context for a sub-operation
    child_ctx = ctx_logger.create_child('sub_operation', 'validation')
    child_ctx.info("Validating data")
    ```
    """

    def __init__(self, logger: logging.Logger, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the context logger.

        Args:
            logger (logging.Logger): The base logger to use.
            context (Optional[Dict[str, Any]], optional): Initial context information. Defaults to None.
        """
        self.logger = logger
        self.context = context or {}

    def add_context(self, key: str, value: Any) -> None:
        """
        Add a new context item or update an existing one.

        Args:
            key (str): Context key.
            value (Any): Context value.
        """
        self.context[key] = value

    def remove_context(self, key: str) -> None:
        """
        Remove a context item.

        Args:
            key (str): Context key to remove.
        """
        if key in self.context:
            del self.context[key]

    def create_child(self, key: str, value: Any) -> 'ContextLogger':
        """
        Create a child context logger with additional context.

        Args:
            key (str): Context key for the additional context.
            value (Any): Context value for the additional context.

        Returns:
            ContextLogger: A new context logger with the combined context.
        """
        child_context = self.context.copy()
        child_context[key] = value
        return ContextLogger(self.logger, child_context)

    def _format_context(self) -> str:
        """
        Format the context information for inclusion in log messages.

        Returns:
            str: Formatted context string.
        """
        if not self.context:
            return ""

        context_items = [f"{k}={v}" for k, v in self.context.items()]
        return f"[{' '.join(context_items)}] "

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log a debug message with context."""
        self.logger.debug(f"{self._format_context()}{msg}", *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log an info message with context."""
        self.logger.info(f"{self._format_context()}{msg}", *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log a warning message with context."""
        self.logger.warning(f"{self._format_context()}{msg}", *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log an error message with context."""
        self.logger.error(f"{self._format_context()}{msg}", *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log a critical message with context."""
        self.logger.critical(f"{self._format_context()}{msg}", *args, **kwargs)

    def exception(self, msg: str, *args, exc_info=True, **kwargs) -> None:
        """Log an exception message with context."""
        self.logger.exception(f"{self._format_context()}{msg}", *args, exc_info=exc_info, **kwargs)

    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Log a message with the specified level and context."""
        self.logger.log(level, f"{self._format_context()}{msg}", *args, **kwargs)


class PerformanceLoggerContext:
    """
    Context manager to log the execution time and memory usage of a block of code.

    Example usage:
    ```
    with PerformanceLoggerContext(logger, "Processing data"):
        # Code to measure
        process_data()
    ```
    """

    def __init__(self, logger: logging.Logger, operation_name: str, level: int = logging.INFO):
        """
        Initialize the context manager.

        Args:
            logger (logging.Logger): The logger to use.
            operation_name (str): Name of the operation being measured.
            level (int, optional): The log level. Defaults to logging.INFO.
        """
        self.logger = logger
        self.operation_name = operation_name
        self.level = level
        self.start_time = 0
        self.start_memory = {}

    def __enter__(self):
        """Enter the context manager."""
        # Log start message
        self.logger.log(self.level, f"Starting {self.operation_name}")

        # Log initial memory usage
        self.start_memory = log_memory_usage(self.logger, f"Memory before {self.operation_name}", self.level)

        # Record start time
        self.start_time = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        # Record end time
        end_time = time.time()

        # Calculate execution time
        execution_time = end_time - self.start_time

        # Log final memory usage
        end_memory = log_memory_usage(self.logger, f"Memory after {self.operation_name}", self.level)

        # Calculate memory difference
        memory_diff = {}
        for key in self.start_memory:
            if key in end_memory:
                memory_diff[key] = end_memory[key] - self.start_memory[key]

        if exc_type is None:
            # Log success message with performance metrics
            self.logger.log(self.level, f"Completed {self.operation_name} in {execution_time:.4f}s")
            if memory_diff:
                self.logger.log(self.level, f"Memory change during {self.operation_name}: RSS={memory_diff.get('rss_mb', 0):.2f}MB")
        else:
            # Log error with performance information
            self.logger.error(f"Error in {self.operation_name} after {execution_time:.4f}s: {str(exc_val)}", exc_info=exc_val)

        # Don't suppress exceptions
        return False


def log_exception(logger: logging.Logger, message: str, exc_info: Exception, level: int = logging.ERROR,
                 extra_context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an exception with detailed information.

    This function logs an exception with detailed information, including the exception type,
    message, traceback, and context. This is very helpful for debugging.

    Args:
        logger (logging.Logger): The logger to use.
        message (str): The message to log.
        exc_info (Exception): The exception to log.
        level (int, optional): The log level. Defaults to logging.ERROR.
        extra_context (Optional[Dict[str, Any]], optional): Extra context to include in the log. Defaults to None.
    """
    # Create a dictionary with exception details
    exc_details = {
        'exception_type': type(exc_info).__name__,
        'exception_message': str(exc_info),
        'traceback': traceback.format_exc()
    }

    # Add extra context if provided
    if extra_context:
        exc_details.update(extra_context)

    # Log the exception with the detailed information
    logger.log(level, f"{message}: {exc_details['exception_type']}: {exc_details['exception_message']}",
               exc_info=exc_info, extra=exc_details)


# Dictionaries to store loggers by name to avoid creating new loggers for each log entry
_prediction_loggers = {}
_validation_loggers = {}

def log_model_prediction(model_name: str, input_data: Dict[str, Any], prediction: Any,
                        model_version: str = "unknown", processing_time: Optional[float] = None,
                        log_file: Optional[str] = None) -> None:
    """
    Log a model prediction for monitoring purposes.

    This function logs model predictions to a dedicated log file for monitoring and analysis.
    The logs are stored in JSON format for easy parsing and analysis.

    These logs can be used for:
    - Monitoring model performance over time
    - Detecting drift in input data or predictions
    - Auditing model usage and decisions
    - Debugging issues with specific predictions

    Args:
        model_name (str): Name of the model.
        input_data (Dict[str, Any]): Input data for the prediction.
        prediction (Any): Prediction result.
        model_version (str, optional): Version of the model. Defaults to "unknown".
        processing_time (Optional[float], optional): Time taken to make the prediction in seconds. Defaults to None.
        log_file (Optional[str]): Path to log file. If None, uses default path.
    """
    if log_file is None:
        # predictions.log contains all model prediction data in JSON format
        # for monitoring, analysis, and debugging purposes
        log_file = os.path.join(LOG_DIR, 'predictions.log')

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Use a singleton logger pattern to avoid creating a new logger for each prediction
    logger_name = f'prediction.{model_name}'

    # Check if we already have a logger for this model
    if logger_name not in _prediction_loggers:
        # Create a new logger if one doesn't exist
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create JSON formatter with enhanced metadata
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_record = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'model': model_name,
                    'model_version': getattr(record, 'model_version', 'unknown'),
                    'input': record.input_data,
                    'prediction': record.prediction,
                    'prediction_id': getattr(record, 'prediction_id', str(uuid.uuid4())),
                    'processing_time': getattr(record, 'processing_time', None)
                }
                return json.dumps(log_record)

        # Create file handler with rotation to prevent the log file from growing too large
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOG_MAX_BYTES,  # Use value from config
            backupCount=LOG_BACKUP_COUNT  # Use value from config
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

        # Store the logger in our dictionary for future use
        _prediction_loggers[logger_name] = logger
    else:
        # Use the existing logger
        logger = _prediction_loggers[logger_name]

    # Create log record
    log_record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg='',
        args=(),
        exc_info=None
    )
    log_record.input_data = input_data
    log_record.prediction = prediction
    log_record.prediction_id = str(uuid.uuid4())
    log_record.model_version = model_version
    log_record.processing_time = processing_time

    # Log the record
    logger.handle(log_record)

    # Log to application logger as well (at debug level)
    app_logger = logging.getLogger('src.api')
    if app_logger.isEnabledFor(logging.DEBUG):
        app_logger.debug(f"Prediction logged for model {model_name} with ID {log_record.prediction_id}")


def log_schema_validation(validation_type: str, validation_metadata: Dict[str, Any], log_file: Optional[str] = None) -> None:
    """
    Log schema validation results to a rotating log file.

    This function logs schema validation metadata to a dedicated log file for monitoring and analysis.
    The logs are stored in JSON format for easy parsing and analysis.

    These logs can be used for:
    - Monitoring data quality over time
    - Detecting drift in input data
    - Auditing validation results
    - Debugging issues with specific validations

    Args:
        validation_type (str): Type of validation (e.g., 'single', 'batch').
        validation_metadata (Dict[str, Any]): Validation metadata to log.
        log_file (Optional[str]): Path to log file. If None, uses default path.
    """
    if log_file is None:
        # schema_validation.log contains all schema validation data in JSON format
        # for monitoring, analysis, and debugging purposes
        log_file = os.path.join(PREDICTION_SCHEMA, 'schema_validation.log')

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Use a singleton logger pattern to avoid creating a new logger for each validation
    logger_name = f'validation.{validation_type}'

    # Check if we already have a logger for this validation type
    if logger_name not in _validation_loggers:
        # Create a new logger if one doesn't exist
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create JSON formatter
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                # Add timestamp if not already present
                validation_data = record.validation_data.copy()
                if 'timestamp' not in validation_data:
                    validation_data['timestamp'] = datetime.utcnow().isoformat()

                # Add validation ID if not already present
                if 'validation_id' not in validation_data:
                    validation_data['validation_id'] = str(uuid.uuid4())

                return json.dumps(validation_data)

        # Create file handler with rotation to prevent the log file from growing too large
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOG_MAX_BYTES,  # Use value from config
            backupCount=LOG_BACKUP_COUNT  # Use value from config
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

        # Store the logger in our dictionary for future use
        _validation_loggers[logger_name] = logger
    else:
        # Use the existing logger
        logger = _validation_loggers[logger_name]

    # Create log record
    log_record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg='',
        args=(),
        exc_info=None
    )
    log_record.validation_data = validation_metadata

    # Log the record
    logger.handle(log_record)

    # Log to application logger as well (at debug level)
    app_logger = logging.getLogger('src.api')
    if app_logger.isEnabledFor(logging.DEBUG):
        app_logger.debug(f"Schema validation logged for type {validation_type}")


def setup_pipeline_logging(log_level: str = 'INFO', process_name: str = 'Pipeline',
                        log_to_console: bool = True) -> str:
    """
    Set up comprehensive logging for the pipeline with a detailed format and unique log file.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        process_name (str): Name of the process being logged (used in log file name).
        log_to_console (bool): Whether to log to console.

    Returns:
        str: Path to the created log file.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create a fixed log file name based on the process name
    process_name_clean = process_name.replace(' ', '')
    log_file = os.path.join(PIPELINE_LOGS_DIR, f"{process_name_clean}.log")

    # Ensure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S,%f'[:-3])

    # Create handlers
    handlers = []

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        # Use ColoredFormatter for console output to improve readability
        console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S,%f'[:-3]))
        handlers.append(console_handler)

    # File handler with rotation
    # Use RotatingFileHandler to automatically rotate logs when they reach a certain size
    # Use size and backup count from config
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)

    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {process_name} - Log file: {log_file}")

    return log_file


class ProcessingLogger:
    """
    Helper class for logging file processing with detailed information and progress tracking.
    """

    def __init__(self, logger_name: str, total_files: int = 0):
        """
        Initialize the processing logger.

        Args:
            logger_name (str): Name for the logger.
            total_files (int): Total number of files to process (for progress tracking).
        """
        self.logger = logging.getLogger(logger_name)
        self.total_files = total_files
        self.processed_files = 0
        self.start_time = time.time()
        self.file_start_time = 0
        self.current_file = ""

    def log_arguments(self, **kwargs):
        """
        Log command line arguments or configuration parameters.

        Args:
            **kwargs: Key-value pairs of arguments to log.
        """
        args_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(f"Arguments: {args_str}")

    def log_database_loaded(self, num_files: int, elapsed_time: float):
        """
        Log information about a loaded database.

        Args:
            num_files (int): Number of files in the database.
            elapsed_time (float): Time taken to load the database.
        """
        self.logger.info(f"Loaded database with {num_files} processed files in {elapsed_time:.2f}s")

    def log_files_found(self, directory: str, file_count: int, total_size_mb: float, elapsed_time: float,
                       file_distribution: Optional[Dict[str, int]] = None):
        """
        Log information about files found for processing.

        Args:
            directory (str): Directory where files were found.
            file_count (int): Number of files found.
            total_size_mb (float): Total size of files in MB.
            elapsed_time (float): Time taken to find files.
            file_distribution (Optional[Dict[str, int]]): Distribution of file types.
        """
        self.total_files = file_count
        self.logger.info(f"Loading OCR files from {directory}...")
        self.logger.info(f"Found {file_count} OCR files ({total_size_mb:.1f}MB total) in {elapsed_time:.2f}s")

        if file_distribution:
            dist_str = ", ".join([f"{count} {file_type} files" for file_type, count in file_distribution.items()])
            self.logger.info(f"File distribution: {dist_str}")

        self.logger.info(f"Starting processing of {file_count} files...")

    def start_file_processing(self, file_index: int, filename: str, file_size_kb: float,
                             char_count: int, file_type: str):
        """
        Log the start of processing for a file.

        Args:
            file_index (int): Index of the file (1-based).
            filename (str): Name of the file.
            file_size_kb (float): Size of the file in KB.
            char_count (int): Number of characters in the file.
            file_type (str): Type or classification of the file.
        """
        self.processed_files = file_index
        self.current_file = filename
        self.file_start_time = time.time()

        self.logger.info(f"[{file_index}/{self.total_files}] PROCESSING: {filename} "
                        f"({file_size_kb:.1f}KB, {char_count} chars, {file_type})")

    def log_extraction_results(self, file_index: int, rows: int, words: int, date: str, raw_rows: int,
                              elapsed_time: float):
        """
        Log the results of data extraction from a file.

        Args:
            file_index (int): Index of the file (1-based).
            rows (int): Number of rows extracted.
            words (int): Number of words extracted.
            date (str): Date extracted from the file.
            raw_rows (int): Number of raw rows in the file.
            elapsed_time (float): Time taken for extraction.
        """
        self.logger.info(f"[{file_index}/{self.total_files}] EXTRACTED: "
                        f"rows:{rows}, words:{words}, date:{date}, raw_rows:{raw_rows} in {elapsed_time:.2f}s")

    def complete_file_processing(self, file_index: int, filename: str, elapsed_time: float,
                                extract_time: float, io_time: float, has_data: bool):
        """
        Log the completion of processing for a file.

        Args:
            file_index (int): Index of the file (1-based).
            filename (str): Name of the file.
            elapsed_time (float): Total time taken to process the file.
            extract_time (float): Time taken for data extraction.
            io_time (float): Time taken for I/O operations.
            has_data (bool): Whether data was successfully extracted.
        """
        self.logger.info(f"[{file_index}/{self.total_files}] COMPLETED: {filename} in {elapsed_time:.2f}s "
                        f"(extract:{extract_time:.1f}s, io:{io_time:.1f}s, data:{'yes' if has_data else 'no'})")

    def log_processing_summary(self, successful_files: int, failed_files: int, elapsed_time: float):
        """
        Log a summary of the processing results.

        Args:
            successful_files (int): Number of successfully processed files.
            failed_files (int): Number of files that failed processing.
            elapsed_time (float): Total time taken for processing.
        """
        total = successful_files + failed_files
        success_rate = (successful_files / total) * 100 if total > 0 else 0

        self.logger.info(f"Processing completed: {successful_files} successful, {failed_files} failed "
                        f"({success_rate:.1f}% success rate) in {elapsed_time:.2f}s")

        if total > 0:
            avg_time = elapsed_time / total
            self.logger.info(f"Average processing time: {avg_time:.3f}s per file")

    def log_error(self, file_index: int, filename: str, error_message: str, exc_info=None):
        """
        Log an error that occurred during file processing.

        Args:
            file_index (int): Index of the file (1-based).
            filename (str): Name of the file.
            error_message (str): Error message.
            exc_info (Exception, optional): Exception information. Defaults to None.
        """
        if exc_info:
            self.logger.error(
                f"[{file_index}/{self.total_files}] ERROR: {filename} - {error_message}",
                exc_info=exc_info
            )
        else:
            self.logger.error(f"[{file_index}/{self.total_files}] ERROR: {filename} - {error_message}")


class ModuleFilter(logging.Filter):
    """
    Filter that allows messages only from specified modules.

    This filter can be used to include or exclude messages from specific modules,
    which is useful for controlling verbosity and focusing on relevant messages.
    """

    def __init__(self, module_patterns: List[str], exclude: bool = False):
        """
        Initialize the filter with module patterns.

        Args:
            module_patterns (List[str]): List of module name patterns to filter.
                Patterns can include '*' as a wildcard.
            exclude (bool, optional): If True, exclude matching modules; if False, include only matching modules.
                Defaults to False.
        """
        super().__init__()
        self.module_patterns = module_patterns
        self.exclude = exclude

    def filter(self, record):
        """
        Filter log records based on module patterns.

        Args:
            record (LogRecord): The log record to filter.

        Returns:
            bool: True if the record should be logged, False otherwise.
        """
        # If no patterns are specified, include all records
        if not self.module_patterns:
            return True

        # Check if the record's module name matches any of the patterns
        module_name = record.name
        matches = False

        for pattern in self.module_patterns:
            # Convert glob pattern to regex pattern
            regex_pattern = pattern.replace('.', '\\.').replace('*', '.*')
            if re.match(f"^{regex_pattern}$", module_name):
                matches = True
                break

        # Return based on exclude flag
        return not matches if self.exclude else matches


class LevelRangeFilter(logging.Filter):
    """
    Filter that allows messages only within a specified level range.

    This filter can be used to include messages within a specific level range,
    which is useful for controlling verbosity.
    """

    def __init__(self, min_level: int, max_level: int):
        """
        Initialize the filter with level range.

        Args:
            min_level (int): Minimum log level (inclusive).
            max_level (int): Maximum log level (inclusive).
        """
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record):
        """
        Filter log records based on level range.

        Args:
            record (LogRecord): The log record to filter.

        Returns:
            bool: True if the record should be logged, False otherwise.
        """
        return self.min_level <= record.levelno <= self.max_level


def configure_logging(
    app_name: str = 'PropertyValuation',
    log_level: str = 'INFO',
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_file: Optional[str] = None,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    date_format: str = '%Y-%m-%d %H:%M:%S,%f',
    use_json: bool = False,
    use_colors: bool = True,
    max_bytes: int = LOG_MAX_BYTES,  # Use value from config
    backup_count: int = LOG_BACKUP_COUNT,  # Use value from config
    module_filters: Optional[Dict[str, List[str]]] = None,
    exclude_modules: Optional[List[str]] = None
) -> str:
    """
    Centralized logging configuration for consistent logging across the application.

    This function combines the best features of the existing logging setup functions
    and provides a single entry point for configuring logging in the application.

    Args:
        app_name (str, optional): Name of the application. Defaults to 'PropertyValuation'.
        log_level (str, optional): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to 'INFO'.
        log_to_console (bool, optional): Whether to log to console. Defaults to True.
        log_to_file (bool, optional): Whether to log to file. Defaults to True.
        log_file (Optional[str], optional): Path to log file. If None, a timestamped file is created. Defaults to None.
        log_format (str, optional): Format string for log messages. Defaults to '%(asctime)s - %(name)s - %(levelname)s - %(message)s'.
        date_format (str, optional): Format string for dates in log messages. Defaults to '%Y-%m-%d %H:%M:%S,%f'.
        use_json (bool, optional): Whether to use JSON formatting for logs. Defaults to False.
        use_colors (bool, optional): Whether to use colors for console output. Defaults to True.
        max_bytes (int, optional): Maximum size of log files before rotation. Defaults to 10MB.
        backup_count (int, optional): Number of backup files to keep. Defaults to 5.
        module_filters (Optional[Dict[str, List[str]]], optional): Dictionary mapping handler names ('console', 'file')
            to lists of module patterns to include. Patterns can include '*' as a wildcard. Defaults to None.
        exclude_modules (Optional[List[str]], optional): List of module patterns to exclude from all handlers.
            Patterns can include '*' as a wildcard. Defaults to None.

    Returns:
        str: Path to the log file, or empty string if not logging to file.
    """
    # Convert log level string to numeric value
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handlers
    handlers = []

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        # Choose formatter based on settings
        if use_json:
            # Create JSON formatter
            class JsonFormatter(logging.Formatter):
                def format(self, record):
                    log_record = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'level': record.levelname,
                        'name': record.name,
                        'message': record.getMessage(),
                        'module': record.module,
                        'function': record.funcName,
                        'line': record.lineno
                    }

                    # Add exception info if available
                    if record.exc_info:
                        log_record['exception'] = self.formatException(record.exc_info)

                    return json.dumps(log_record)

            console_handler.setFormatter(JsonFormatter())
        elif use_colors:
            # Use ColoredFormatter for console output
            console_handler.setFormatter(ColoredFormatter(log_format, datefmt=date_format[:-3]))
        else:
            # Use standard formatter
            console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format[:-3]))

        handlers.append(console_handler)

    # File handler
    if log_to_file:
        # Generate log file path if not provided
        if log_file is None:
            app_name_clean = app_name.replace(' ', '')
            log_file = os.path.join(PIPELINE_LOGS_DIR, f"{app_name_clean}.log")

        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)

        # Choose formatter based on settings
        if use_json:
            # Create JSON formatter
            class JsonFormatter(logging.Formatter):
                def format(self, record):
                    log_record = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'level': record.levelname,
                        'name': record.name,
                        'message': record.getMessage(),
                        'module': record.module,
                        'function': record.funcName,
                        'line': record.lineno
                    }

                    # Add exception info if available
                    if record.exc_info:
                        log_record['exception'] = self.formatException(record.exc_info)

                    return json.dumps(log_record)

            file_handler.setFormatter(JsonFormatter())
        else:
            # Use standard formatter
            file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format[:-3]))

        handlers.append(file_handler)

    # Apply module filters if specified
    if module_filters:
        for handler_name, patterns in module_filters.items():
            if handler_name == 'console' and log_to_console and handlers:
                # Apply filter to console handler (first handler)
                console_filter = ModuleFilter(patterns)
                handlers[0].addFilter(console_filter)
                logger = logging.getLogger(__name__)
                logger.debug(f"Applied module filter to console handler: {patterns}")

            if handler_name == 'file' and log_to_file and len(handlers) > 1:
                # Apply filter to file handler (second handler)
                file_filter = ModuleFilter(patterns)
                handlers[1].addFilter(file_filter)
                logger = logging.getLogger(__name__)
                logger.debug(f"Applied module filter to file handler: {patterns}")

    # Apply exclude modules filter if specified
    if exclude_modules:
        exclude_filter = ModuleFilter(exclude_modules, exclude=True)
        for handler in handlers:
            handler.addFilter(exclude_filter)
        logger = logging.getLogger(__name__)
        logger.debug(f"Applied exclude module filter: {exclude_modules}")

    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)

    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for {app_name}")
    if log_to_file:
        logger.info(f"Log file: {log_file}")

    # Log filter information
    if module_filters:
        logger.debug(f"Module filters: {module_filters}")
    if exclude_modules:
        logger.debug(f"Excluded modules: {exclude_modules}")

    return log_file if log_to_file else ""


if __name__ == "__main__":
    # Example usage of centralized logging configuration
    log_file = configure_logging(
        app_name='LoggingExample',
        log_level='INFO',
        log_to_console=True,
        log_to_file=True,
        use_colors=True
    )

    logger = get_logger(__name__)

    # Basic logging examples
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Example of exception logging
    try:
        1 / 0
    except Exception as e:
        log_exception(logger, "An error occurred", e)

    # Example of memory usage logging
    log_memory_usage(logger)

    # Example of performance logging with decorator
    @log_performance_decorator(logger)
    def slow_function():
        logger.info("Doing some work...")
        time.sleep(1)
        return "Done"

    slow_function()

    # Example of performance logging with context manager
    with PerformanceLoggerContext(logger, "Processing data"):
        logger.info("Processing...")
        time.sleep(0.5)

    # Example of model prediction logging
    log_model_prediction(
        model_name='property_valuation',
        input_data={'area': 100, 'rooms': 3, 'bathrooms': 2},
        prediction=150000
    )

    # Example of processing logger
    proc_logger = ProcessingLogger("data_extraction", total_files=5)

    proc_logger.log_arguments(input="/path/to/data", limit=None)
    proc_logger.log_database_loaded(0, 0.00)
    proc_logger.log_files_found(
        "/path/to/data",
        5, 2.3, 0.1,
        {"txt": 3, "csv": 2}
    )

    # Example file processing
    proc_logger.start_file_processing(1, "file1.txt", 1.2, 950, "TEXT")
    proc_logger.log_extraction_results(1, 1, 43, "2023-01-01", 3, 0.00)
    proc_logger.complete_file_processing(1, "file1.txt", 0.00, 0.0, 0.0, True)

    logger.info(f"Examples completed. Check the log file: {log_file}")
