"""
Module A: Heavy Rainfall Weather Prediction Pipeline Entry Point
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from ml.select_model import run_rainfall_pipeline

if __name__ == '__main__':
    run_rainfall_pipeline()
