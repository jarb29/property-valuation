"""
Tests for the API endpoints.

This module contains tests for the API endpoints using FastAPI's TestClient.
"""

import os
import unittest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import app
from src.config import API_KEY, DATA_VERSION

# Create API prefix based on DATA_VERSION
API_PREFIX = f"/api/v{DATA_VERSION.replace('v', '')}"


class TestAPI(unittest.TestCase):
    """Tests for the API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.api_key = API_KEY
        self.headers = {"X-API-Key": self.api_key}

    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("timestamp", data)
        self.assertIn("version", data)

    def test_predict_without_api_key(self):
        """Test the predict endpoint without an API key."""
        response = self.client.post(
            f"{API_PREFIX}/predict",
            json={"features": {"feature1": 1.0, "feature2": "value", "feature3": 2.5}}
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "API key missing")

    def test_predict_with_invalid_api_key(self):
        """Test the predict endpoint with an invalid API key."""
        response = self.client.post(
            f"{API_PREFIX}/predict",
            headers={"X-API-Key": "invalid_key"},
            json={"features": {"feature1": 1.0, "feature2": "value", "feature3": 2.5}}
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Invalid API key")

    @patch("src.api.endpoints.get_model")
    def test_predict(self, mock_get_model):
        """Test the predict endpoint."""
        # Set up the mock
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.75]
        mock_get_model.return_value = mock_model

        # Make the request
        response = self.client.post(
            f"{API_PREFIX}/predict",
            headers=self.headers,
            json={"features": {"feature1": 1.0, "feature2": "value", "feature3": 2.5}}
        )

        # Check the response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["prediction"], 0.75)
        self.assertIn("prediction_time", data)
        self.assertIn("model_version", data)

        # Check that the mock was called correctly
        mock_get_model.assert_called_once()
        mock_model.predict.assert_called_once()

    @patch("src.api.endpoints.get_model")
    def test_batch_predict(self, mock_get_model):
        """Test the batch predict endpoint."""
        # Set up the mock
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.75, 0.85]
        mock_get_model.return_value = mock_model

        # Make the request
        response = self.client.post(
            f"{API_PREFIX}/batch-predict",
            headers=self.headers,
            json={
                "instances": [
                    {"feature1": 1.0, "feature2": "value1", "feature3": 2.5},
                    {"feature1": 2.0, "feature2": "value2", "feature3": 3.5}
                ]
            }
        )

        # Check the response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["predictions"], [0.75, 0.85])
        self.assertIn("prediction_time", data)
        self.assertIn("model_version", data)

        # Check that the mock was called correctly
        mock_get_model.assert_called_once()
        mock_model.predict.assert_called_once()

    @patch("src.api.endpoints.list_saved_models")
    def test_list_models(self, mock_list_models):
        """Test the list models endpoint."""
        # Set up the mock
        mock_list_models.return_value = {
            "/path/to/model1.pkl": {
                "model_type": "gradient_boosting",
                "timestamp": "2023-01-01T12:00:00",
                "evaluation_results": {"rmse": 10000}
            },
            "/path/to/model2.pkl": {
                "model_type": "random_forest",
                "timestamp": "2023-01-02T12:00:00",
                "evaluation_results": {"rmse": 9000}
            }
        }

        # Make the request
        response = self.client.get(
            f"{API_PREFIX}/models",
            headers=self.headers
        )

        # Check the response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["model_type"], "gradient_boosting")
        self.assertEqual(data[1]["model_type"], "random_forest")

        # Check that the mock was called correctly
        mock_list_models.assert_called_once()

    @patch("src.api.endpoints.list_saved_models")
    def test_get_model_info(self, mock_list_models):
        """Test the get model info endpoint."""
        # Set up the mock
        mock_list_models.return_value = {
            "/path/to/model1.pkl": {
                "model_type": "gradient_boosting",
                "timestamp": "2023-01-01T12:00:00",
                "evaluation_results": {"rmse": 10000}
            }
        }

        # Make the request
        response = self.client.get(
            f"{API_PREFIX}/models/model1",
            headers=self.headers
        )

        # Check the response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_type"], "gradient_boosting")
        self.assertEqual(data["model_version"], "model1")
        self.assertEqual(data["performance_metrics"]["rmse"], 10000)

        # Check that the mock was called correctly
        mock_list_models.assert_called_once()

    @patch("src.api.endpoints.list_saved_models")
    def test_get_model_info_not_found(self, mock_list_models):
        """Test the get model info endpoint with a non-existent model."""
        # Set up the mock
        mock_list_models.return_value = {}

        # Make the request
        response = self.client.get(
            f"{API_PREFIX}/models/non_existent_model",
            headers=self.headers
        )

        # Check the response
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Model not found: non_existent_model")

        # Check that the mock was called correctly
        mock_list_models.assert_called_once()


if __name__ == "__main__":
    unittest.main()
