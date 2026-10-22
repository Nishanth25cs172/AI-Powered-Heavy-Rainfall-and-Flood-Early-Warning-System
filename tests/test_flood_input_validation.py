import unittest
import json
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))

from backend.app import app

class TestFloodInputValidation(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()
        self.endpoint = '/api/predict-flood'

    def post(self, payload):
        response = self.client.post(
            self.endpoint,
            data=json.dumps(payload),
            content_type='application/json'
        )
        return response.status_code, response.get_json()

    def test_valid_typical_flood_input(self):
        payload = {
            "Rainfall (mm)": 140.0, "Water Level (m)": 6.2, "River Discharge (m³/s)": 2800.0,
            "Elevation (m)": 14.0, "Latitude": 19.076, "Longitude": 72.877,
            "Soil Type": "Clay", "Land Cover": "Urban"
        }
        status, data = self.post(payload)
        self.assertEqual(status, 200)
        self.assertTrue(data.get('success'))

    def test_minimum_physical_boundary(self):
        payload = {
            "Rainfall (mm)": 0.0, "Water Level (m)": 0.0, "River Discharge (m³/s)": 0.0,
            "Elevation (m)": 0.0, "Latitude": 6.0, "Longitude": 68.0,
            "Soil Type": "Clay", "Land Cover": "Forest"
        }
        status, data = self.post(payload)
        self.assertEqual(status, 200)

    def test_maximum_physical_boundary(self):
        payload = {
            "Rainfall (mm)": 1000.0, "Water Level (m)": 30.0, "River Discharge (m³/s)": 50000.0,
            "Elevation (m)": 8848.0, "Latitude": 37.5, "Longitude": 97.5,
            "Soil Type": "Sandy", "Land Cover": "Water Body"
        }
        status, data = self.post(payload)
        self.assertEqual(status, 200)

    def test_rainfall_below_min(self):
        payload = {
            "Rainfall (mm)": -0.1, "Water Level (m)": 5.0, "River Discharge (m³/s)": 1000.0,
            "Elevation (m)": 50.0, "Latitude": 20.0, "Longitude": 78.0,
            "Soil Type": "Clay", "Land Cover": "Urban"
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_water_level_below_min(self):
        payload = {
            "Rainfall (mm)": 50.0, "Water Level (m)": -0.1, "River Discharge (m³/s)": 1000.0,
            "Elevation (m)": 50.0, "Latitude": 20.0, "Longitude": 78.0,
            "Soil Type": "Clay", "Land Cover": "Urban"
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_discharge_below_min(self):
        payload = {
            "Rainfall (mm)": 50.0, "Water Level (m)": 5.0, "River Discharge (m³/s)": -1.0,
            "Elevation (m)": 50.0, "Latitude": 20.0, "Longitude": 78.0,
            "Soil Type": "Clay", "Land Cover": "Urban"
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_elevation_below_min(self):
        payload = {
            "Rainfall (mm)": 50.0, "Water Level (m)": 5.0, "River Discharge (m³/s)": 1000.0,
            "Elevation (m)": -0.1, "Latitude": 20.0, "Longitude": 78.0,
            "Soil Type": "Clay", "Land Cover": "Urban"
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_invalid_categorical_soil_type(self):
        payload = {
            "Rainfall (mm)": 50.0, "Water Level (m)": 5.0, "River Discharge (m³/s)": 1000.0,
            "Elevation (m)": 50.0, "Latitude": 20.0, "Longitude": 78.0,
            "Soil Type": "Volcanic Ash", "Land Cover": "Urban"
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_invalid_categorical_land_cover(self):
        payload = {
            "Rainfall (mm)": 50.0, "Water Level (m)": 5.0, "River Discharge (m³/s)": 1000.0,
            "Elevation (m)": 50.0, "Latitude": 20.0, "Longitude": 78.0,
            "Soil Type": "Clay", "Land Cover": "Cyberpunk Megacity"
        }
        status, data = self.post(payload)
        self.assertEqual(status, 400)

    def test_out_of_training_range_warning(self):
        # Rainfall 650.0 mm is within physical [0, 1000] but exceeds training dataset max (~500 mm)
        payload = {
            "Rainfall (mm)": 650.0, "Water Level (m)": 5.0, "River Discharge (m³/s)": 1000.0,
            "Elevation (m)": 50.0, "Latitude": 20.0, "Longitude": 78.0,
            "Soil Type": "Clay", "Land Cover": "Urban"
        }
        status, data = self.post(payload)
        self.assertEqual(status, 200)
        self.assertTrue(data.get('success'))
        warnings = data.get('training_range_warnings', data.get('range_warnings', data.get('warnings', [])))
        self.assertTrue(len(warnings) > 0)

if __name__ == '__main__':
    unittest.main()
