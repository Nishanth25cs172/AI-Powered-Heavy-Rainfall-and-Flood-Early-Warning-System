import unittest
import os
import sys
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from ml.config import FLOOD_MODEL_PATH, RAINFALL_MODEL_PATH, FLOOD_META_PATH, RAINFALL_META_PATH

class TestMLPipelines(unittest.TestCase):
    def test_flood_model_file_exists(self):
        self.assertTrue(os.path.exists(FLOOD_MODEL_PATH), f"Flood model not found at {FLOOD_MODEL_PATH}")

    def test_rainfall_model_file_exists(self):
        self.assertTrue(os.path.exists(RAINFALL_MODEL_PATH), f"Rainfall model not found at {RAINFALL_MODEL_PATH}")

    def test_smoke_load_flood_model(self):
        model = joblib.load(FLOOD_MODEL_PATH)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

    def test_smoke_load_rainfall_model(self):
        model = joblib.load(RAINFALL_MODEL_PATH)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

if __name__ == '__main__':
    unittest.main()
