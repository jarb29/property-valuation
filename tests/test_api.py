import unittest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import app
from src.config import API_KEY, DATA_VERSION

API_PREFIX = f"/api/v{DATA_VERSION.replace('v', '')}"


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.headers = {"X-API-Key": API_KEY}

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
        self.assertIn("version", data)

    def test_predict_auth_errors(self):
        test_data = {"features": {"feature1": 1.0}}
        
        # No API key
        response = self.client.post(f"{API_PREFIX}/predictions", json=test_data)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "API key missing")
        
        # Invalid API key
        response = self.client.post(
            f"{API_PREFIX}/predictions",
            headers={"X-API-Key": "invalid_key"},
            json=test_data
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Invalid API key")

    @patch("src.api.endpoints.get_model")
    def test_predict(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.75]
        mock_get_model.return_value = mock_model

        response = self.client.post(
            f"{API_PREFIX}/predictions",
            headers=self.headers,
            json={"features": {
                "type": "departamento",
                "sector": "las condes",
                "net_usable_area": 120.5,
                "net_area": 150.0,
                "n_rooms": 3,
                "n_bathroom": 2,
                "latitude": -33.4172,
                "longitude": -70.5476
            }}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["prediction"], 0.75)
        self.assertIn("prediction_time", data)
        self.assertIn("model_version", data)
        mock_get_model.assert_called_once()
        mock_model.predict.assert_called_once()

    @patch("src.api.endpoints.get_model")
    def test_batch_predict(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.75, 0.85])
        mock_get_model.return_value = mock_model

        response = self.client.post(
            f"{API_PREFIX}/predictions/batch",
            headers=self.headers,
            json={"instances": [
                {"type": "departamento", "sector": "las condes", "net_usable_area": 120.5,
                 "net_area": 150.0, "n_rooms": 3, "n_bathroom": 2,
                 "latitude": -33.4172, "longitude": -70.5476},
                {"type": "casa", "sector": "providencia", "net_usable_area": 200.0,
                 "net_area": 250.0, "n_rooms": 4, "n_bathroom": 3,
                 "latitude": -33.4372, "longitude": -70.6276}
            ]}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["predictions"], [0.75, 0.85])
        self.assertIn("prediction_time", data)
        self.assertIn("model_version", data)
        mock_get_model.assert_called_once()
        mock_model.predict.assert_called_once()

    @patch("src.api.endpoints.get_best_model")
    def test_get_model_info(self, mock_get_best_model):
        mock_get_best_model.return_value = "/path/to/model1.pkl"

        response = self.client.get(f"{API_PREFIX}/model/info", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "property_valuation")
        self.assertIn("version", data)
        self.assertIn("description", data)
        self.assertIn("features", data)
        mock_get_best_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
