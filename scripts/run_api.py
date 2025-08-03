#!/usr/bin/env python
"""
Script to run the API server.

This script runs the FastAPI application using uvicorn.
"""

import os
import sys
import argparse
import logging
import uvicorn
import socket
from contextlib import closing

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.logging import setup_logging
from src.config import API_HOST, API_PORT, API_WORKERS, API_DEBUG, LOG_DIR


def is_port_available(host, port):
    """
    Check if a port is available on the specified host.

    Args:
        host (str): The host to check.
        port (int): The port to check.

    Returns:
        bool: True if the port is available, False otherwise.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        return result != 0


def find_available_port(host, start_port, max_attempts=10):
    """
    Find an available port starting from the specified port.

    Args:
        host (str): The host to check.
        start_port (int): The port to start checking from.
        max_attempts (int): Maximum number of ports to check.

    Returns:
        int: An available port, or None if no port is available after max_attempts.
    """
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(host, port):
            return port
    return None


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run the API server')

    parser.add_argument('--host', type=str, default=API_HOST,
                        help=f'Host to bind the server to (default: {API_HOST})')

    parser.add_argument('--port', type=int, default=API_PORT,
                        help=f'Port to bind the server to (default: {API_PORT})')

    parser.add_argument('--auto-port', action='store_true',
                        help='Automatically find an available port if the specified port is in use')

    parser.add_argument('--no-auto-port', action='store_true',
                        help='Disable automatic port finding')

    parser.add_argument('--workers', type=int, default=API_WORKERS,
                        help=f'Number of worker processes (default: {API_WORKERS})')

    parser.add_argument('--debug', action='store_true', default=API_DEBUG,
                        help='Enable debug mode')

    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload on code changes (for development)')

    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')

    return parser.parse_args()


def main():
    """
    Main function to run the API server.
    """
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Check if the specified port is available
    port = args.port
    if not is_port_available(args.host, port):
        if not args.no_auto_port:
            # Try to find an available port (default behavior)
            new_port = find_available_port(args.host, port + 1)
            if new_port:
                logging.warning(f"Port {port} is already in use. Using port {new_port} instead.")
                port = new_port
            else:
                logging.error(f"Port {port} is already in use and no available ports found in range {port+1}-{port+10}.")
                logging.error(f"Please try the following:")
                logging.error(f"1. Use a different port with --port option: python scripts/run_api.py --port {port+20}")
                logging.error(f"2. Stop the process using port {port} and try again")
                logging.error(f"3. To find processes using this port, run: lsof -i :{port}")
                sys.exit(1)
        else:
            # Port is not available and auto-port is disabled
            logging.error(f"Port {port} is already in use. Please try the following:")
            logging.error(f"1. Use a different port with --port option: python scripts/run_api.py --port {port+1}")
            logging.error(f"2. Remove --no-auto-port to automatically find an available port")
            logging.error(f"3. Stop the process using port {port} and try again")
            logging.error(f"4. To find processes using this port, run: lsof -i :{port}")
            sys.exit(1)

    # Log the start of the API server
    logging.info(f"Starting API server on {args.host}:{port}")

    try:
        # Ensure log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)

        # Configure uvicorn logging
        # Note: Uvicorn is the ASGI server that runs the FastAPI application.
        # It's a production-grade server that handles HTTP requests efficiently.
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(asctime)s [uvicorn] %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "json": {
                    "()": "src.api.logging.JsonFormatter",
                    "pretty": True,
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "api_file": {
                    "formatter": "json",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": os.path.join(LOG_DIR, "api.log"),
                    "maxBytes": 10 * 1024 * 1024,  # 10 MB
                    "backupCount": 5,
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default", "api_file"], "level": args.log_level},
                "uvicorn.error": {"level": "INFO"},
                "uvicorn.access": {"handlers": ["default", "api_file"], "level": "INFO", "propagate": False},
            },
        }

        # Run the API server
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=port,
            workers=args.workers,
            log_level=args.log_level.lower(),
            reload=args.reload,
            log_config=log_config
        )

    except OSError as e:
        # This is a fallback in case our port checking didn't catch the issue
        if e.errno == 48 or "address already in use" in str(e).lower():
            logging.error(f"Port {port} is already in use. Please try the following:")
            logging.error(f"1. Use a different port with --port option: python scripts/run_api.py --port {port+1}")
            logging.error(f"2. Remove --no-auto-port to automatically find an available port")
            logging.error(f"3. Stop the process using port {port} and try again")
            logging.error(f"4. To find processes using this port, run: lsof -i :{port}")
            sys.exit(1)
        else:
            logging.exception(f"Error running API server: {str(e)}")
            sys.exit(1)
    except Exception as e:
        logging.exception(f"Error running API server: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
