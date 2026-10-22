"""
Module B: Flood Inundation Prediction Pipeline Entry Point
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from ml.select_model import run_flood_pipeline

if __name__ == '__main__':
    run_flood_pipeline()
