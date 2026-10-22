"""
SIH26071 – AI/ML-Based Integrated Heavy Rainfall Early Warning and Inundation Prediction System
Module B: Flood / Inundation Prediction Pipeline (Phase 3 Overhaul)
Dataset: data/raw/flood_risk.csv

Pipeline Enhancements:
- Stratified 80/20 Train/Test Split (random_state=42)
- Multi-Model Benchmarking: DecisionTree, RandomForest, GradientBoosting, ExtraTrees, LogisticRegression
- Automated Selection: Best model chosen based on F1-Score & Recall
- Reports: Benchmarking CSV saved to ml/reports/flood_model_benchmark.csv, metadata saved to flood_model_metadata.json
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
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
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'flood_risk.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'ml', 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'ml', 'reports')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def train_and_evaluate_flood_model() -> Dict[str, Any]:
    """Executes stratified split, benchmarks candidate flood classifiers, and exports canonical model & metadata."""
    logging.info(f"Loading flood risk dataset from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Flood dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    logging.info(f"Dataset loaded: {df.shape[0]:,} records x {df.shape[1]} columns")

    target = 'Flood Occurred'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {DATA_PATH}")

    num_cols = ['Latitude', 'Longitude', 'Rainfall (mm)', 'River Discharge (m³/s)', 'Water Level (m)', 'Elevation (m)']
    cat_cols = ['Land Cover', 'Soil Type']
    features = num_cols + cat_cols

    X = df[features]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logging.info(f"Train set: {len(X_train):,} records | Holdout Test set: {len(X_test):,} records")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

    candidate_classifiers = {
        'Decision Tree Classifier': DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=10, min_samples_leaf=5, class_weight='balanced', random_state=42),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'Extra Trees Classifier': ExtraTreesClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    }

    benchmark_results = []
    trained_pipelines = {}

    for name, clf in candidate_classifiers.items():
        logging.info(f"Training flood model candidate: {name}...")
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
    benchmark_csv_path = os.path.join(REPORTS_DIR, 'flood_model_benchmark.csv')
    df_benchmark.to_csv(benchmark_csv_path, index=False)
    logging.info(f"Saved Flood Model Benchmark CSV to: {benchmark_csv_path}")

    # Automated Best Model Selection
    best_algo_name = df_benchmark.iloc[0]['algorithm']
    best_info = trained_pipelines[best_algo_name]
    best_pipeline = best_info['pipeline']
    best_metrics = best_info['metrics']

    logging.info("=" * 65)
    logging.info(f"BEST FLOOD MODEL SELECTED: {best_algo_name}")
    logging.info(f"  Accuracy  : {best_metrics['accuracy']:.4f}")
    logging.info(f"  Precision : {best_metrics['precision']:.4f}")
    logging.info(f"  Recall    : {best_metrics['recall']:.4f}")
    logging.info(f"  F1-Score  : {best_metrics['f1_score']:.4f}")
    logging.info(f"  ROC-AUC   : {best_metrics['roc_auc']:.4f}")
    logging.info("=" * 65)

    flood_model_path = os.path.join(MODELS_DIR, 'flood_decision_tree_model.joblib')
    joblib.dump(best_pipeline, flood_model_path)
    logging.info(f"Saved primary Flood model pipeline to: {flood_model_path}")

    # Extract Feature Importances if available
    clf_step = best_pipeline.named_steps['classifier']
    prep_step = best_pipeline.named_steps['preprocessor']
    
    try:
        cat_feature_names = prep_step.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
        all_encoded_features = num_cols + cat_feature_names
        if hasattr(clf_step, 'feature_importances_'):
            importances = clf_step.feature_importances_.tolist()
            top_feature_importances = sorted(
                [{'feature': f, 'importance': round(imp, 5)} for f, imp in zip(all_encoded_features, importances)],
                key=lambda x: x['importance'],
                reverse=True
            )
        else:
            top_feature_importances = [{'feature': f, 'importance': 0.0} for f in all_encoded_features]
    except Exception:
        top_feature_importances = [{'feature': f, 'importance': 0.0} for f in features]

    metadata = {
        'model_name': 'Flood Risk Decision Tree Early Warning Classifier',
        'primary_algorithm': best_algo_name,
        'selection_criteria': 'Best F1-Score & Recall on Stratified Holdout Test Set',
        'target_definition': 'Flood Occurred (1 = Flood, 0 = No Flood)',
        'dataset_source': 'flood_risk.csv',
        'dataset_total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'columns': features,
        'numeric_columns': num_cols,
        'categorical_columns': cat_cols,
        'numeric_features': num_cols,
        'categorical_features': cat_cols,
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

    report_path = os.path.join(REPORTS_DIR, 'flood_model_metadata.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Saved Flood Model Metadata JSON to: {report_path}")

    return metadata

if __name__ == '__main__':
    train_and_evaluate_flood_model()
