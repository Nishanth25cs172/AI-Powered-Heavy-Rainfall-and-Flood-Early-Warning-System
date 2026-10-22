import unittest
import json
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))

from backend.app import app

class TestWeatherInputValidation(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()
        self.endpoint = '/api/predict-rainfall'

    def post(self, payload):
        response = self.client.post(
            self.endpoint,
            data=json.dumps(payload),
            content_type='application/json'
        )
        return response.status_code, response.get_json()

    def test_valid_typical_weather_input(self):
        payload = {
            "avg_temp": 28.5, "wind_speed": 14.2, "air_pressure": 1008.5,
            "latitude": 19.076, "longitude": 72.877, "elevation": 14.0
        }
        status, data = self.post(payload)
        self.assertEqual(status, 200)
        self.assertTrue(data.get('success'))

    def test_minimum_boundary_values(self):
        payload = {
            "avg_temp": -20.0, "wind_speed": 0.0, "air_pressure": 800.0,
            "latitude": 6.0, "longitude": 68.0, "elevation": 0.0
        }
        status, data = self.post(payload)
        self.assertEqual(status, 200)
        self.assertTrue(data.get('success'))

    def test_maximum_boundary_values(self):
        payload = {
            "avg_temp": 55.0, "wind_speed": 150.0, "air_pressure": 1080.0,
            "latitude": 37.5, "longitude": 97.5, "elevation": 8848.0
        }
        status, data = self.post(payload)
        self.assertEqual(status, 200)
        self.assertTrue(data.get('success'))

    def test_temp_below_min(self):
        payload = {
            "avg_temp": -20.1, "wind_speed": 10.0, "air_pressure": 1010.0,
            "latitude": 20.0, "longitude": 78.0, "elevation": 100.0
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_wind_below_min(self):
        payload = {
            "avg_temp": 25.0, "wind_speed": -0.1, "air_pressure": 1010.0,
            "latitude": 20.0, "longitude": 78.0, "elevation": 100.0
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_pressure_below_min(self):
        payload = {
            "avg_temp": 25.0, "wind_speed": 10.0, "air_pressure": 799.9,
            "latitude": 20.0, "longitude": 78.0, "elevation": 100.0
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_latitude_below_min(self):
        payload = {
            "avg_temp": 25.0, "wind_speed": 10.0, "air_pressure": 1010.0,
            "latitude": 5.9, "longitude": 78.0, "elevation": 100.0
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_longitude_below_min(self):
        payload = {
            "avg_temp": 25.0, "wind_speed": 10.0, "air_pressure": 1010.0,
            "latitude": 20.0, "longitude": 67.9, "elevation": 100.0
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_elevation_below_min(self):
        payload = {
            "avg_temp": 25.0, "wind_speed": 10.0, "air_pressure": 1010.0,
            "latitude": 20.0, "longitude": 78.0, "elevation": -0.1
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_temp_above_max(self):
        payload = {
            "avg_temp": 55.1, "wind_speed": 10.0, "air_pressure": 1010.0,
            "latitude": 20.0, "longitude": 78.0, "elevation": 100.0
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_wind_above_max(self):
        payload = {
            "avg_temp": 25.0, "wind_speed": 150.1, "air_pressure": 1010.0,
            "latitude": 20.0, "longitude": 78.0, "elevation": 100.0
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_pressure_above_max(self):
        payload = {
            "avg_temp": 25.0, "wind_speed": 10.0, "air_pressure": 1080.1,
            "latitude": 20.0, "longitude": 78.0, "elevation": 100.0
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_invalid_string_field(self):
        payload = {
            "avg_temp": "invalid_temp", "wind_speed": 10.0, "air_pressure": 1010.0,
            "latitude": 20.0, "longitude": 78.0, "elevation": 100.0
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_out_of_training_range_warning(self):
        # Temp 44.0 is within physical [-20, 55] but outside training max (43.4 °C)
        payload = {
            "avg_temp": 44.0, "wind_speed": 10.0, "air_pressure": 1010.0,
            "latitude": 20.0, "longitude": 78.0, "elevation": 100.0
        }
        status, data = self.post(payload)
        self.assertEqual(status, 200)
        self.assertTrue(data.get('success'))
        warnings = data.get('training_range_warnings', data.get('range_warnings', data.get('warnings', [])))
        self.assertTrue(len(warnings) > 0)

if __name__ == '__main__':
    unittest.main()
