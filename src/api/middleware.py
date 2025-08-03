"""
API middleware.

This module provides middleware for the FastAPI application for logging and error handling.
"""

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import json
import uuid

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""

    async def dispatch(self, request: Request, call_next):
        """
        Process the request and log information about it.

        Args:
            request (Request): The incoming request.
            call_next (callable): The next middleware or route handler.

        Returns:
            Response: The response from the next middleware or route handler.
        """
        # Generate request ID
        request_id = str(uuid.uuid4())

        # Add request ID to request state
        request.state.request_id = request_id

        # Log request
        await self._log_request(request, request_id)

        # Process request and measure time
        start_time = time.time()

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log response
            self._log_response(request, response, process_time, request_id)

            # Add custom headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id

            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.exception(f"Request {request_id} failed after {process_time:.4f}s: {str(e)}")
            raise

    async def _log_request(self, request: Request, request_id: str):
        """
        Log information about the request.

        Args:
            request (Request): The incoming request.
            request_id (str): The unique request ID.
        """
        # Get client IP
        client_host = request.client.host if request.client else "unknown"

        # Log basic request info
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path} from {client_host}"
        )

        # Log headers if debug
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Request {request_id} headers: {dict(request.headers)}")

        # Log body if debug and not a GET request
        if logger.isEnabledFor(logging.DEBUG) and request.method != "GET":
            try:
                body = await request.body()
                if body:
                    try:
                        # Try to parse as JSON
                        body_str = body.decode("utf-8")
                        json_body = json.loads(body_str)
                        logger.debug(f"Request {request_id} body: {json_body}")
                    except:
                        # Log as string if not JSON
                        logger.debug(f"Request {request_id} body: {body}")
            except Exception as e:
                logger.debug(f"Could not log request {request_id} body: {str(e)}")

    def _log_response(self, request: Request, response: Response, process_time: float, request_id: str):
        """
        Log information about the response.

        Args:
            request (Request): The incoming request.
            response (Response): The outgoing response.
            process_time (float): The time taken to process the request.
            request_id (str): The unique request ID.
        """
        # Log basic response info
        logger.info(
            f"Response {request_id}: {request.method} {request.url.path} "
            f"completed in {process_time:.4f}s with status {response.status_code}"
        )

        # Log headers if debug
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Response {request_id} headers: {dict(response.headers)}")


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling errors."""

    async def dispatch(self, request: Request, call_next):
        """
        Process the request and handle any errors.

        Args:
            request (Request): The incoming request.
            call_next (callable): The next middleware or route handler.

        Returns:
            Response: The response from the next middleware or route handler.
        """
        try:
            return await call_next(request)
        except Exception as e:
            # Log the error
            logger.exception(f"Unhandled exception in request: {str(e)}")

            # Return a JSON response with error information
            from fastapi.responses import JSONResponse
            from src.api.schemas import ErrorResponse
            from datetime import datetime

            error_response = ErrorResponse(
                error="InternalServerError",
                detail=str(e),
                timestamp=datetime.now()
            )

            return JSONResponse(
                status_code=500,
                content=error_response.dict()
            )