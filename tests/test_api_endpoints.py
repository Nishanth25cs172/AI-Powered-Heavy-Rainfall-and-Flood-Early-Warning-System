import unittest
import json
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))

from backend.app import app

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    def test_health_check(self):
        res = self.client.get('/api/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'healthy')

    def test_dashboard_endpoint(self):
        res = self.client.get('/api/dashboard')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue('alerts' in data or 'metrics' in data or 'records' in data)

    def test_rainfall_analytics(self):
        res = self.client.get('/api/rainfall-analytics')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('monthly_seasonality', data)
        self.assertIn('yearly_trends', data)

    def test_analytics(self):
        res = self.client.get('/api/analytics')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue('datasets' in data or 'flood_counts' in data)

    def test_risk_map(self):
        res = self.client.get('/api/risk-map')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertIn('locations', data)
        self.assertGreater(len(data.get('locations', [])), 0)

    def test_datasets_info(self):
        res = self.client.get('/api/datasets/info')
        self.assertEqual(res.status_code, 200)

if __name__ == '__main__':
    unittest.main()
