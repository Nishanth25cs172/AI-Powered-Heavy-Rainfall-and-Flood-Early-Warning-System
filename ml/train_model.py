"""
SIH26071 – AI/ML-Based Integrated Heavy Rainfall Early Warning and Inundation Prediction System
Unified Model Training & Benchmarking Suite (Phase 3)
Executes training & benchmarking pipelines for both Module A (Rainfall) and Module B (Flood Risk).
"""

import logging
from typing import Dict, Any
from train_rainfall_model import train_and_evaluate_rainfall_model
from train_flood_model import train_and_evaluate_flood_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def train_all_models() -> None:
    """Orchestrates end-to-end training, benchmarking, and metadata exports for both Rainfall & Flood models."""
    logging.info("=================================================================")
    logging.info("STARTING PHASE 3 ML PIPELINE TRAINING & BENCHMARKING SUITE")
    logging.info("=================================================================")

    logging.info("--- 1. Training & Benchmarking Rainfall Prediction Pipeline ---")
    rainfall_meta = train_and_evaluate_rainfall_model()

    logging.info("--- 2. Training & Benchmarking Flood Inundation Pipeline ---")
    flood_meta = train_and_evaluate_flood_model()

    logging.info("=================================================================")
    logging.info("PHASE 3 ML PIPELINE TRAINING & BENCHMARKING COMPLETE")
    logging.info(f"Rainfall Best Model : {rainfall_meta['primary_algorithm']} (F1: {rainfall_meta['primary_evaluation_metrics']['f1_score']})")
    logging.info(f"Flood Best Model    : {flood_meta['primary_algorithm']} (F1: {flood_meta['primary_evaluation_metrics']['f1_score']})")
    logging.info("=================================================================")

if __name__ == '__main__':
    train_all_models()
