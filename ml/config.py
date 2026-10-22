"""
SIH26071 – Centralized Project Configuration
Provides unified path resolution, random seeds, model hyperparameter grids, and threshold constants.
"""

import os
import hashlib

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
ML_DIR = os.path.join(BASE_DIR, 'ml')
MODELS_DIR = os.path.join(ML_DIR, 'models')
FLOOD_MODELS_DIR = os.path.join(MODELS_DIR, 'flood')
RAINFALL_MODELS_DIR = os.path.join(MODELS_DIR, 'rainfall')
REPORTS_DIR = os.path.join(ML_DIR, 'reports')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

# Dataset File Paths
FLOOD_DATASET_PATH = os.path.join(RAW_DATA_DIR, 'flood_risk.csv')
RAINFALL_DAILY_PATH = os.path.join(RAW_DATA_DIR, 'Daily Rainfall Data - India (2009-2024).csv')
WEATHER_DATASET_CSV = os.path.join(RAW_DATA_DIR, 'Indian Rainfall and Weather Prediction Dataset.csv')
WEATHER_DATASET_XLSX = os.path.join(RAW_DATA_DIR, 'Indian Rainfall and Weather Prediction Dataset.xlsx')

# Model File Paths
FLOOD_MODEL_PATH = os.path.join(MODELS_DIR, 'flood_decision_tree_model.joblib')
FLOOD_MODEL_CANONICAL = os.path.join(FLOOD_MODELS_DIR, 'flood_model.joblib')
RAINFALL_MODEL_PATH = os.path.join(MODELS_DIR, 'rainfall_random_forest_model.joblib')
RAINFALL_MODEL_CANONICAL = os.path.join(RAINFALL_MODELS_DIR, 'rainfall_model.joblib')

# Reports & Metadata Paths
FLOOD_META_PATH = os.path.join(REPORTS_DIR, 'flood_model_metadata.json')
RAINFALL_META_PATH = os.path.join(REPORTS_DIR, 'rainfall_model_metadata.json')
MODEL_COMPARISON_CSV = os.path.join(REPORTS_DIR, 'model_comparison.csv')
FLOOD_BENCHMARK_CSV = os.path.join(REPORTS_DIR, 'flood_model_benchmark.csv')
RAINFALL_BENCHMARK_CSV = os.path.join(REPORTS_DIR, 'rainfall_model_benchmark.csv')

# Global Constants & Seeds
RANDOM_SEED = 42
IMD_HEAVY_RAINFALL_THRESHOLD_MM = 64.5

# Ensure Required Directories Exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FLOOD_MODELS_DIR, RAINFALL_MODELS_DIR, REPORTS_DIR, DOCS_DIR]:
    os.makedirs(d, exist_ok=True)

def compute_file_hash(filepath: str) -> str:
    """Computes SHA-256 hash of a file for dataset version tracking."""
    if not os.path.exists(filepath):
        return "N/A"
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]
