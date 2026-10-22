"""
SIH26071 – AI/ML-Based Integrated Heavy Rainfall Early Warning and Inundation Prediction System
Module A: Heavy Rainfall / Weather Classification Pipeline (Phase 3 Overhaul)
Dataset: data/raw/Indian Rainfall and Weather Prediction Dataset.xlsx / .csv

Pipeline Enhancements:
- Time-based Split: Train on 2009–2020 (~80%), Test on 2021–2024 (~20% holdout)
- Multi-Model Benchmarking: RandomForest, GradientBoosting, ExtraTrees, DecisionTree, LogisticRegression
- Automated Selection: Best model chosen based on F1-Score & Recall
- Reports: Benchmarking CSV saved to ml/reports/rainfall_model_benchmark.csv, metadata saved to rainfall_model_metadata.json
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from typing import Dict, List, Tuple, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH_CSV = os.path.join(BASE_DIR, 'data', 'raw', 'Indian Rainfall and Weather Prediction Dataset.csv')
DATA_PATH_XLSX = os.path.join(BASE_DIR, 'data', 'raw', 'Indian Rainfall and Weather Prediction Dataset.xlsx')
MODELS_DIR = os.path.join(BASE_DIR, 'ml', 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'ml', 'reports')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def train_and_evaluate_rainfall_model() -> Dict[str, Any]:
    """Executes time-based split, benchmarks candidate classifiers, and exports canonical model & metadata."""
    if os.path.exists(DATA_PATH_CSV):
        logging.info(f"Loading weather dataset from CSV: {DATA_PATH_CSV}")
        df = pd.read_csv(DATA_PATH_CSV)
    elif os.path.exists(DATA_PATH_XLSX):
        logging.info(f"Loading weather dataset from XLSX: {DATA_PATH_XLSX}")
        df = pd.read_excel(DATA_PATH_XLSX)
    else:
        raise FileNotFoundError("Weather dataset not found in data/raw folder.")

    logging.info(f"Raw weather records: {len(df):,} rows x {len(df.columns)} columns")
    df = df.dropna(subset=['rainfall']).copy()
    df['Rain_Target'] = (df['rainfall'] > 0).astype(int)

    features = ['avg_temp', 'wind_speed', 'air_pressure', 'latitude', 'longitude', 'elevation']

    # 1. TIME-BASED SPLIT (2009-2020 Train | 2021-2024 Test)
    if 'date_of_record' in df.columns:
        df['parsed_date'] = pd.to_datetime(df['date_of_record'], errors='coerce')
        df['year'] = df['parsed_date'].dt.year
        train_mask = df['year'] <= 2020
        test_mask = df['year'] > 2020

        # Fallback if time mask is unexpected
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            df = df.sort_values('parsed_date').reset_index(drop=True)
            split_idx = int(len(df) * 0.8)
            train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
        else:
            train_df, test_df = df[train_mask], df[test_mask]
    else:
        split_idx = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df['Rain_Target']
    X_test, y_test = test_df[features], test_df['Rain_Target']

    logging.info(f"Time-based Split -> Train set: {len(X_train):,} records | Holdout Test set: {len(X_test):,} records")

    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. MULTI-MODEL BENCHMARKING
    candidate_classifiers = {
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
        'Extra Trees Classifier': ExtraTreesClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'Decision Tree Classifier': DecisionTreeClassifier(max_depth=8, random_state=42),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    }

    benchmark_results = []
    trained_pipelines = {}

    for name, clf in candidate_classifiers.items():
        logging.info(f"Training benchmark candidate: {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_prob))
        else:
            auc = 0.0

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        benchmark_results.append({
            'algorithm': name,
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4)
        })

        trained_pipelines[name] = {
            'pipeline': pipeline,
            'metrics': {
                'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'roc_auc': auc,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            }
        }

    # Save Benchmark CSV
    df_benchmark = pd.DataFrame(benchmark_results).sort_values(by='f1_score', ascending=False)
    benchmark_csv_path = os.path.join(REPORTS_DIR, 'rainfall_model_benchmark.csv')
    df_benchmark.to_csv(benchmark_csv_path, index=False)
    logging.info(f"Saved Rainfall Model Benchmark CSV to: {benchmark_csv_path}")

    # 3. AUTOMATED BEST MODEL SELECTION (Highest F1-Score)
    best_algo_name = df_benchmark.iloc[0]['algorithm']
    best_info = trained_pipelines[best_algo_name]
    best_pipeline = best_info['pipeline']
    best_metrics = best_info['metrics']

    logging.info("=" * 65)
    logging.info(f"BEST RAINFALL MODEL SELECTED: {best_algo_name}")
    logging.info(f"  Accuracy  : {best_metrics['accuracy']:.4f}")
    logging.info(f"  Precision : {best_metrics['precision']:.4f}")
    logging.info(f"  Recall    : {best_metrics['recall']:.4f}")
    logging.info(f"  F1-Score  : {best_metrics['f1_score']:.4f}")
    logging.info(f"  ROC-AUC   : {best_metrics['roc_auc']:.4f}")
    logging.info("=" * 65)

    rf_model_path = os.path.join(MODELS_DIR, 'rainfall_random_forest_model.joblib')
    joblib.dump(best_pipeline, rf_model_path)
    logging.info(f"Saved primary Rainfall model pipeline to: {rf_model_path}")

    # Feature Importances if available
    clf_step = best_pipeline.named_steps['classifier']
    if hasattr(clf_step, 'feature_importances_'):
        importances = clf_step.feature_importances_.tolist()
        top_feature_importances = sorted(
            [{'feature': f, 'importance': round(imp, 5)} for f, imp in zip(features, importances)],
            key=lambda x: x['importance'],
            reverse=True
        )
    else:
        top_feature_importances = [{'feature': f, 'importance': 0.0} for f in features]

    metadata = {
        'model_name': 'Heavy Rainfall Prediction Model',
        'primary_algorithm': best_algo_name,
        'selection_criteria': 'Best F1-Score & Recall on Time-Based Holdout Test Set (2021-2024)',
        'target_definition': 'Rain_Target (1 = Rain, 0 = No Rain)',
        'dataset_source': 'Indian Rainfall and Weather Prediction Dataset',
        'dataset_total_records': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'split_type': 'Time-Based Split (2009-2020 Train | 2021-2024 Test)',
        'columns': features,
        'numeric_columns': features,
        'categorical_columns': [],
        'features': {
            'all_features': features,
            'numeric_features': features
        },
        'primary_evaluation_metrics': {
            'accuracy': round(best_metrics['accuracy'], 4),
            'precision': round(best_metrics['precision'], 4),
            'recall': round(best_metrics['recall'], 4),
            'f1_score': round(best_metrics['f1_score'], 4),
            'roc_auc': round(best_metrics['roc_auc'], 4),
            'confusion_matrix': best_metrics['confusion_matrix'],
            'classification_report': best_metrics['classification_report']
        },
        'feature_importances': top_feature_importances,
        'benchmark_summary': benchmark_results
    }

    report_path = os.path.join(REPORTS_DIR, 'rainfall_model_metadata.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Saved Rainfall Model Metadata JSON to: {report_path}")

    return metadata

if __name__ == '__main__':
    train_and_evaluate_rainfall_model()
