"""
API key authentication.

This module provides API key authentication for the FastAPI application.
"""

from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
import logging

from src.config import API_KEY

logger = logging.getLogger(__name__)

# API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Validate the API key.

    Args:
        api_key (str, optional): The API key from the request header. Defaults to Security(api_key_header).

    Returns:
        str: The validated API key.

    Raises:
        HTTPException: If the API key is invalid or missing.
    """
    if not api_key:
        logger.warning("API key missing")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="API key missing"
        )

    if api_key != API_KEY:
        logger.warning("Invalid API key")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid API key"
        )

    return api_key


def verify_api_key(api_key: str) -> bool:
    """
    Verify if an API key is valid.

    Args:
        api_key (str): The API key to verify.

    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    return api_key == API_KEY