"""
SIH26071 – Model Selection & Cross-Validation Engine
Runs StratifiedKFold CV for Flood and TimeSeriesSplit CV for Rainfall, evaluates multiple candidate models,
selects optimal pipelines according to docs/model_selection_criteria.md, and exports comprehensive reports.
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple

from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.config import (
    FLOOD_DATASET_PATH, WEATHER_DATASET_XLSX, WEATHER_DATASET_CSV,
    FLOOD_MODEL_PATH, FLOOD_MODEL_CANONICAL, RAINFALL_MODEL_PATH, RAINFALL_MODEL_CANONICAL,
    FLOOD_META_PATH, RAINFALL_META_PATH, MODEL_COMPARISON_CSV, FLOOD_BENCHMARK_CSV, RAINFALL_BENCHMARK_CSV,
    RANDOM_SEED, compute_file_hash
)
from ml.preprocessing import build_flood_preprocessor, build_rainfall_preprocessor
from ml.data_quality import audit_dataset_quality

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def evaluate_models_cv(
    X: pd.DataFrame, y: pd.Series, preprocessor: Any, cv_splitter: Any, is_timeseries: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """Evaluates candidate classifiers via Cross-Validation reporting mean and std per metric."""
    candidate_classifiers = {
        'Baseline Dummy': DummyClassifier(strategy='most_frequent'),
        'Decision Tree Classifier': DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=10, min_samples_leaf=5, class_weight='balanced', random_state=RANDOM_SEED),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1),
        'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_SEED),
        'Extra Trees Classifier': ExtraTreesClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
    }

    results = []
    fitted_pipelines = {}

    for name, clf in candidate_classifiers.items():
        logging.info(f"Running CV evaluation for: {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        accs, precs, recs, f1s, aucs = [], [], [], [], []

        for train_idx, val_idx in cv_splitter.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_val)
            
            accs.append(accuracy_score(y_val, preds))
            precs.append(precision_score(y_val, preds, zero_division=0))
            recs.append(recall_score(y_val, preds, zero_division=0))
            f1s.append(f1_score(y_val, preds, zero_division=0))

            if hasattr(pipeline, "predict_proba"):
                try:
                    probs = pipeline.predict_proba(X_val)[:, 1]
                    aucs.append(roc_auc_score(y_val, probs))
                except Exception:
                    aucs.append(0.5)
            else:
                aucs.append(0.5)

        # Fit on full dataset for final deployment
        pipeline.fit(X, y)
        fitted_pipelines[name] = pipeline

        mean_f1 = float(np.mean(f1s))
        mean_rec = float(np.mean(recs))
        mean_auc = float(np.mean(aucs))
        selection_rank = round((0.50 * mean_f1) + (0.35 * mean_rec) + (0.15 * mean_auc), 4)

        results.append({
            'algorithm': name,
            'cv_acc_mean': round(float(np.mean(accs)), 4),
            'cv_acc_std': round(float(np.std(accs)), 4),
            'cv_precision_mean': round(float(np.mean(precs)), 4),
            'cv_precision_std': round(float(np.std(precs)), 4),
            'cv_recall_mean': round(mean_rec, 4),
            'cv_recall_std': round(float(np.std(recs)), 4),
            'cv_f1_mean': round(mean_f1, 4),
            'cv_f1_std': round(float(np.std(f1s)), 4),
            'cv_auc_mean': round(mean_auc, 4),
            'cv_auc_std': round(float(np.std(aucs)), 4),
            'selection_rank': selection_rank
        })

    df_results = pd.DataFrame(results).sort_values(by='selection_rank', ascending=False)
    return df_results, fitted_pipelines

def run_flood_pipeline() -> Dict[str, Any]:
    """Runs complete Flood ML pipeline with Data Quality, Stratified CV, Model Selection & Metadata output."""
    logging.info("=================================================================")
    logging.info("RUNNING MODULE B: FLOOD RISK ML PIPELINE")
    logging.info("=================================================================")
    
    df = pd.read_csv(FLOOD_DATASET_PATH)
    data_hash = compute_file_hash(FLOOD_DATASET_PATH)
    audit_summary = audit_dataset_quality(df, target_col='Flood Occurred', name="Flood Risk Dataset")

    num_cols = ['Latitude', 'Longitude', 'Rainfall (mm)', 'River Discharge (m³/s)', 'Water Level (m)', 'Elevation (m)']
    cat_cols = ['Land Cover', 'Soil Type']
    features = num_cols + cat_cols
    X = df[features]
    y = df['Flood Occurred'].astype(int)

    preprocessor = build_flood_preprocessor(num_cols, cat_cols)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    df_benchmark, fitted_pipelines = evaluate_models_cv(X, y, preprocessor, cv)
    df_benchmark.to_csv(FLOOD_BENCHMARK_CSV, index=False)
    logging.info(f"Saved Flood Benchmark to: {FLOOD_BENCHMARK_CSV}")

    best_name = df_benchmark.iloc[0]['algorithm']
    best_pipeline = fitted_pipelines[best_name]
    best_row = df_benchmark.iloc[0].to_dict()

    # Save Canonical Model
    joblib.dump(best_pipeline, FLOOD_MODEL_PATH)
    joblib.dump(best_pipeline, FLOOD_MODEL_CANONICAL)
    logging.info(f"Saved Best Flood Model ({best_name}) to {FLOOD_MODEL_PATH}")

    # Metadata
    metadata = {
        'model_name': 'Flood Risk Early Warning Classifier',
        'primary_algorithm': best_name,
        'version': '2.0.0',
        'training_timestamp': datetime.now().isoformat(),
        'dataset_hash': data_hash,
        'selection_criteria': 'Best CV Selection Rank (0.50*F1 + 0.35*Recall + 0.15*AUC)',
        'target_definition': 'Flood Occurred (1=Yes, 0=No)',
        'dataset_source': 'flood_risk.csv',
        'dataset_total_samples': len(df),
        'columns': features,
        'numeric_columns': num_cols,
        'categorical_columns': cat_cols,
        'numeric_features': num_cols,
        'categorical_features': cat_cols,
        'cross_validation_metrics': best_row,
        'primary_evaluation_metrics': {
            'accuracy': best_row['cv_acc_mean'],
            'precision': best_row['cv_precision_mean'],
            'recall': best_row['cv_recall_mean'],
            'f1_score': best_row['cv_f1_mean'],
            'roc_auc': best_row['cv_auc_mean']
        },
        'benchmark_summary': df_benchmark.to_dict(orient='records'),
        'data_quality_audit': audit_summary
    }

    with open(FLOOD_META_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Saved Flood Metadata to: {FLOOD_META_PATH}")
    return metadata

def run_rainfall_pipeline() -> Dict[str, Any]:
    """Runs complete Rainfall ML pipeline with Data Quality, TimeSeries CV, Model Selection & Metadata output."""
    logging.info("=================================================================")
    logging.info("RUNNING MODULE A: HEAVY RAINFALL ML PIPELINE")
    logging.info("=================================================================")

    if os.path.exists(WEATHER_DATASET_CSV):
        df = pd.read_csv(WEATHER_DATASET_CSV)
        data_hash = compute_file_hash(WEATHER_DATASET_CSV)
    else:
        df = pd.read_excel(WEATHER_DATASET_XLSX)
        data_hash = compute_file_hash(WEATHER_DATASET_XLSX)

    df = df.dropna(subset=['rainfall']).copy()
    df['Rain_Target'] = (df['rainfall'] > 0).astype(int)
    
    if 'date_of_record' in df.columns:
        df['parsed_date'] = pd.to_datetime(df['date_of_record'], errors='coerce')
        df = df.sort_values('parsed_date').reset_index(drop=True)

    audit_summary = audit_dataset_quality(df, target_col='Rain_Target', name="Weather Prediction Dataset")

    features = ['avg_temp', 'wind_speed', 'air_pressure', 'latitude', 'longitude', 'elevation']
    X = df[features]
    y = df['Rain_Target']

    preprocessor = build_rainfall_preprocessor(features)
    cv = TimeSeriesSplit(n_splits=5)

    df_benchmark, fitted_pipelines = evaluate_models_cv(X, y, preprocessor, cv, is_timeseries=True)
    df_benchmark.to_csv(RAINFALL_BENCHMARK_CSV, index=False)
    logging.info(f"Saved Rainfall Benchmark to: {RAINFALL_BENCHMARK_CSV}")

    best_name = df_benchmark.iloc[0]['algorithm']
    best_pipeline = fitted_pipelines[best_name]
    best_row = df_benchmark.iloc[0].to_dict()

    # Save Canonical Model
    joblib.dump(best_pipeline, RAINFALL_MODEL_PATH)
    joblib.dump(best_pipeline, RAINFALL_MODEL_CANONICAL)
    logging.info(f"Saved Best Rainfall Model ({best_name}) to {RAINFALL_MODEL_PATH}")

    # Metadata
    metadata = {
        'model_name': 'Heavy Rainfall Weather Classifier',
        'primary_algorithm': best_name,
        'version': '2.0.0',
        'training_timestamp': datetime.now().isoformat(),
        'dataset_hash': data_hash,
        'selection_criteria': 'Best TimeSeriesSplit CV Selection Rank (0.50*F1 + 0.35*Recall + 0.15*AUC)',
        'target_definition': 'Rain_Target (1=Rain, 0=No Rain)',
        'dataset_source': 'Indian Rainfall and Weather Prediction Dataset',
        'dataset_total_samples': len(df),
        'columns': features,
        'numeric_columns': features,
        'categorical_columns': [],
        'numeric_features': features,
        'categorical_features': [],
        'cross_validation_metrics': best_row,
        'primary_evaluation_metrics': {
            'accuracy': best_row['cv_acc_mean'],
            'precision': best_row['cv_precision_mean'],
            'recall': best_row['cv_recall_mean'],
            'f1_score': best_row['cv_f1_mean'],
            'roc_auc': best_row['cv_auc_mean']
        },
        'benchmark_summary': df_benchmark.to_dict(orient='records'),
        'data_quality_audit': audit_summary
    }

    with open(RAINFALL_META_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Saved Rainfall Metadata to: {RAINFALL_META_PATH}")
    return metadata

def select_and_export_all():
    """Runs both pipelines and exports combined model_comparison.csv."""
    flood_meta = run_flood_pipeline()
    rainfall_meta = run_rainfall_pipeline()

    # Create Combined model_comparison.csv
    flood_df = pd.DataFrame(flood_meta['benchmark_summary'])
    flood_df['module'] = 'Module B: Flood Inundation'
    
    rainfall_df = pd.DataFrame(rainfall_meta['benchmark_summary'])
    rainfall_df['module'] = 'Module A: Heavy Rainfall'

    combined = pd.concat([rainfall_df, flood_df], ignore_index=True)
    combined.to_csv(MODEL_COMPARISON_CSV, index=False)
    logging.info(f"Saved Combined Model Comparison CSV to: {MODEL_COMPARISON_CSV}")
    logging.info("=================================================================")
    logging.info("ALL ML PIPELINES, CROSS-VALIDATIONS & BENCHMARKS COMPLETE")
    logging.info("=================================================================")

if __name__ == '__main__':
    select_and_export_all()
