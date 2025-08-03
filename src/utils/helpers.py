"""
Helper functions.

This module provides various helper functions used throughout the application.
"""

import os
import pandas as pd
import glob
import re
from typing import List, Dict, Any, Optional, Union
import logging
import json
import time
from functools import wraps
from datetime import datetime


# Adjust these imports according to your actual project structure
from src.config import (
    DATA_VERSION_DIR,
    OUTPUT_DIR,
    PIPELINE_DATA_DIR,
    JUPYTER_DATA_DIR,
    DATA_VERSION
)

logger = logging.getLogger(__name__)


def timer_decorator(func):
    """
    Decorator to measure the execution time of a function.

    Args:
        func: The function to measure.

    Returns:
        callable: The wrapped function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def retry_decorator(max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    Decorator to retry a function on failure.

    Args:
        max_retries (int, optional): Maximum number of retries. Defaults to 3.
        delay (int, optional): Initial delay between retries in seconds. Defaults to 1.
        backoff (int, optional): Backoff multiplier. Defaults to 2.
        exceptions (tuple, optional): Exceptions to catch. Defaults to (Exception,).

    Returns:
        callable: The decorator function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_retries, delay
            while mtries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    mtries -= 1
                    if mtries == 0:
                        raise

                    logger.warning(f"Retrying {func.__name__} in {mdelay} seconds after error: {str(e)}")
                    time.sleep(mdelay)
                    mdelay *= backoff
        return wrapper
    return decorator


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: The loaded JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        return json.load(f)


def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data (Dict[str, Any]): The data to save.
        file_path (str): Path where the JSON file will be saved.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Args:
        d (Dict[str, Any]): The dictionary to flatten.
        parent_key (str, optional): The parent key. Defaults to ''.
        sep (str, optional): Separator between keys. Defaults to '_'.

    Returns:
        Dict[str, Any]: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_file_extension(file_path: str) -> str:
    """
    Get the extension of a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: The file extension.
    """
    return os.path.splitext(file_path)[1].lower()


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory_path (str): Path to the directory.
    """
    os.makedirs(directory_path, exist_ok=True)


def format_timestamp(timestamp: Optional[float] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a timestamp as a string.

    Args:
        timestamp (Optional[float], optional): The timestamp to format. Defaults to None (current time).
        format_str (str, optional): The format string. Defaults to "%Y-%m-%d %H:%M:%S".

    Returns:
        str: The formatted timestamp.
    """
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(timestamp).strftime(format_str)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks.

    Args:
        lst (List[Any]): The list to split.
        chunk_size (int): The size of each chunk.

    Returns:
        List[List[Any]]: A list of chunks.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: Union[int, float] = 0) -> Union[int, float]:
    """
    Safely divide two numbers, returning a default value if the denominator is zero.

    Args:
        numerator (Union[int, float]): The numerator.
        denominator (Union[int, float]): The denominator.
        default (Union[int, float], optional): The default value to return if the denominator is zero. Defaults to 0.

    Returns:
        Union[int, float]: The result of the division, or the default value if the denominator is zero.
    """
    return numerator / denominator if denominator != 0 else default


def get_memory_usage(obj: Any) -> float:
    """
    Get the memory usage of an object in MB.

    Args:
        obj (Any): The object to measure.

    Returns:
        float: The memory usage in MB.
    """
    import sys
    if isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / (1024 * 1024)
    else:
        return sys.getsizeof(obj) / (1024 * 1024)


def get_versioned_filename(base_name: str, file_type: str, directory: str, extension: str = 'csv') -> str:
    """
    Generate a versioned filename based on the data version and existing files.

    The function checks for existing files with the same base name and file type,
    and increments the version number accordingly. The format is:
    {DATA_VERSION}.{incremental_version}_{file_type}_{base_name}.{extension}

    For example: v1.3_schema_train.json

    Args:
        base_name (str): The base name of the file (e.g., 'train', 'test')
        file_type (str): The type of file (e.g., 'schema', 'model', 'data')
        directory (str): The directory where the file will be saved
        extension (str, optional): The file extension. Defaults to 'csv'.

    Returns:
        str: The versioned filename
    """
    # Ensure directory exists
    ensure_directory_exists(directory)

    # Create the pattern to search for existing files
    pattern = f"{DATA_VERSION}.*_{file_type}_{base_name}.{extension}"
    search_pattern = os.path.join(directory, pattern)

    # Find all matching files
    matching_files = glob.glob(search_pattern)

    # Extract version numbers from existing files
    version_numbers = []
    for file_path in matching_files:
        file_name = os.path.basename(file_path)
        match = re.match(f"{DATA_VERSION}\.(\d+)_{file_type}_{base_name}\.{extension}", file_name)
        if match:
            version_numbers.append(int(match.group(1)))

    # Determine the next version number
    next_version = 1
    if version_numbers:
        next_version = max(version_numbers) + 1

    # Generate the versioned filename
    versioned_filename = f"{DATA_VERSION}.{next_version}_{file_type}_{base_name}.{extension}"
    full_path = os.path.join(directory, versioned_filename)

    logger.info(f"Generated versioned filename: {versioned_filename}")
    return full_path


def get_latest_versioned_file(base_name: str, file_type: str, directory: str, extension: str = 'csv', data_version: str = None) -> str:
    """
    Find the latest versioned file for a given data version, file type, and base name.

    The function searches for files with the pattern:
    {data_version}.{incremental_version}_{file_type}_{base_name}.{extension}
    and returns the path to the file with the highest incremental version.

    For example, if there are files v1.1_data_clean.csv, v1.2_data_clean.csv, v1.3_data_clean.csv,
    this function will return the path to v1.3_data_clean.csv.

    Args:
        base_name (str): The base name of the file (e.g., 'clean', 'outliers')
        file_type (str): The type of file (e.g., 'schema', 'model', 'data')
        directory (str): The directory to search in
        extension (str, optional): The file extension. Defaults to 'csv'.
        data_version (str, optional): The data version to search for. Defaults to DATA_VERSION from config.

    Returns:
        str: The path to the latest versioned file, or None if no matching files are found
    """
    # Use DATA_VERSION from config if data_version is not provided
    if data_version is None:
        data_version = DATA_VERSION

    # Create the pattern to search for existing files
    pattern = f"{data_version}.*_{file_type}_{base_name}.{extension}"
    search_pattern = os.path.join(directory, pattern)

    # Find all matching files
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        logger.warning(f"No matching files found for pattern: {pattern} in directory: {directory}")
        return None

    # Extract version numbers from existing files
    versioned_files = []
    for file_path in matching_files:
        file_name = os.path.basename(file_path)
        match = re.match(f"{data_version}\.(\d+)_{file_type}_{base_name}\.{extension}", file_name)
        if match:
            version = int(match.group(1))
            versioned_files.append((file_path, version))

    if not versioned_files:
        logger.warning(f"No valid versioned files found for pattern: {pattern} in directory: {directory}")
        return None

    # Sort by version number (descending) and return the path to the file with the highest version
    latest_file = sorted(versioned_files, key=lambda x: x[1], reverse=True)[0][0]
    logger.info(f"Found latest versioned file: {latest_file}")
    return latest_file


def load_original_data(data_type='train'):
    """
    Load the original data file directly from the data version directory.

    This function loads the original data file (train.csv or test.csv) from the
    data version directory specified in config.py (DATA_VERSION_DIR).

    Args:
        data_type (str, optional): The type of data to load ('train' or 'test'). Defaults to 'train'.

    Returns:
        pandas.DataFrame: The loaded data
    """
    import pandas as pd
    from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH


    # Determine the file path based on data_type
    if data_type == 'train':
        file_path = TRAIN_DATA_PATH
    else:
        file_path = TEST_DATA_PATH

    # Load and return the data
    logger.info(f"Loading original {data_type} data from: {file_path}")
    return pd.read_csv(file_path)




def load_latest_data(
    data_type='train',
    outputs_dir=None,
    fallback_dir=None,
    load_target='pipeline'
):
    """
    Load the latest versioned data file for training or testing.

    This function first tries to find the latest versioned data file in the chosen
    outputs directory. If no file is found, it falls back to the original data
    file in the data version directory.

    Args:
        data_type (str, optional): The type of data to load ('train' or 'test').
            Defaults to 'train'.
        outputs_dir (str, optional): Directory where versioned output files are stored.
            If provided, it overrides the 'load_target' choice.
        fallback_dir (str, optional): Directory to fall back to if no versioned file
            is found. Defaults to None, which uses DATA_VERSION_DIR from config.
        load_target (str, optional): Where to load the data from if outputs_dir is not
            provided:
            - 'pipeline' -> uses OUTPUT_PIPELINE_DIR
            - 'jupyter'  -> uses OUTPUT_JUPYTER_DIR
            - anything else -> uses OUTPUT_DIR

    Returns:
        pandas.DataFrame: The loaded data
    """
    # If outputs_dir isn't specified, derive it from load_target
    if outputs_dir is None:
        if load_target.lower() == 'pipeline':
            outputs_dir = PIPELINE_DATA_DIR
        elif load_target.lower() == 'jupyter':
            outputs_dir = JUPYTER_DATA_DIR
        else:
            outputs_dir = OUTPUT_DIR

    # If fallback_dir isn't explicitly given, default to the versioned data dir
    if fallback_dir is None:
        fallback_dir = DATA_VERSION_DIR

    # Try to find the latest versioned data file
    if data_type == 'train':
        # For training data, look for the latest clean data file
        latest_file = get_latest_versioned_file(
            base_name='clean',
            file_type='data',
            directory=outputs_dir,
            extension='csv'
        )
    else:
        # For test data, look for the latest test data file
        latest_file = get_latest_versioned_file(
            base_name='test',
            file_type='data',
            directory=outputs_dir,
            extension='csv'
        )

        # If no versioned test file is found, try the clean data file logic
        if latest_file is None:
            logger.info(f"No versioned test file found in {outputs_dir}. "
                        "Looking for clean data file as fallback.")
            latest_file = get_latest_versioned_file(
                base_name='clean',
                file_type='data',
                directory=outputs_dir,
                extension='csv'
            )

            # If a clean data file is found, we create a versioned test file from original
            if latest_file is not None:
                logger.info(f"Found clean data file: {latest_file}")
                fallback_file = os.path.join(fallback_dir, f"{data_type}.csv")
                logger.info(f"Checking for original test file: {fallback_file}")
                if os.path.exists(fallback_file):
                    logger.info("Original test file found. Creating versioned test file...")
                    test_data = pd.read_csv(fallback_file)
                    versioned_test_file = get_versioned_filename(
                        base_name='test',
                        file_type='data',
                        directory=outputs_dir,
                        extension='csv'
                    )
                    test_data.to_csv(versioned_test_file, index=False)
                    logger.info(f"Created versioned test file: {versioned_test_file}")
                    latest_file = versioned_test_file
                else:
                    logger.warning(f"Original test file not found at: {fallback_file}")

    # If a versioned file is found, load it
    if latest_file is not None:
        logger.info(f"Loading data from versioned file: {latest_file}")
        return pd.read_csv(latest_file)

    # Otherwise, fall back to the original data file
    fallback_file = os.path.join(fallback_dir, f"{data_type}.csv")
    logger.warning(f"No versioned file found. Falling back to: {fallback_file}")
    return pd.read_csv(fallback_file)
