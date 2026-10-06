"""
SIH26071 – AI/ML-Based Integrated Heavy Rainfall Early Warning and Inundation Prediction System
ML Model Training & Evaluation Pipeline
Dataset: data/flood_risk.csv
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'flood_risk.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'ml', 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'ml', 'reports')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def train_and_evaluate():
    logging.info(f"Loading primary flood risk dataset from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Primary dataset not found at {DATA_PATH}")

    # Read with utf-8 encoding to preserve symbols
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    logging.info(f"Dataset loaded: {df.shape[0]:,} records x {df.shape[1]} columns")

    target = 'Flood Occurred'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {DATA_PATH}")

    # Verify no missing values or handle them
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logging.warning(f"Handling missing values: {null_counts[null_counts > 0].to_dict()}")
        df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    logging.info(f"Numeric features ({len(numeric_columns)}): {numeric_columns}")
    logging.info(f"Categorical features ({len(categorical_columns)}): {categorical_columns}")
    logging.info(f"Class distribution: {y.value_counts().to_dict()} (Mean positive: {y.mean():.4f})")

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
        ]
    )

    # Base Classifier with balanced weights
    base_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
    ])

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    logging.info(f"Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    # Hyperparameter Optimization via RandomizedSearchCV with 5-fold Stratified CV
    param_dist = {
        'classifier__n_estimators': [100, 150, 200, 250],
        'classifier__max_depth': [None, 8, 12, 16, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logging.info("Initiating Hyperparameter Optimization (RandomizedSearchCV, 5-fold CV)...")
    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_dist,
        n_iter=6,
        scoring='f1',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    logging.info(f"Best Hyperparameters: {search.best_params_}")

    # Evaluate on holdout test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Feature Importance extraction from Random Forest
    rf_step = best_model.named_steps['classifier']
    cat_step = best_model.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_names = cat_step.get_feature_names_out(categorical_columns).tolist() if hasattr(cat_step, 'get_feature_names_out') else []
    all_feature_names = numeric_columns + encoded_cat_names
    importances = rf_step.feature_importances_.tolist()
    feature_importance_map = sorted(
        [{'feature': f, 'importance': round(imp, 5)} for f, imp in zip(all_feature_names, importances)],
        key=lambda x: x['importance'],
        reverse=True
    )

    metrics = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'roc_auc': round(auc, 4),
        'confusion_matrix': cm,
        'classification_report': clf_report,
        'best_params': search.best_params_,
        'best_cv_score': round(float(search.best_score_), 4),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_importances': feature_importance_map[:10]
    }

    logging.info("=" * 60)
    logging.info("ACTUAL MODEL EVALUATION METRICS (Test Set):")
    logging.info(f"  Accuracy        : {acc:.4f} ({acc*100:.2f}%)")
    logging.info(f"  Precision       : {prec:.4f}")
    logging.info(f"  Recall          : {rec:.4f}")
    logging.info(f"  F1-Score        : {f1:.4f}")
    logging.info(f"  ROC-AUC         : {auc:.4f}")
    logging.info(f"  Confusion Matrix: {cm}")
    logging.info("=" * 60)

    # Save trained pipeline model
    model_save_path = os.path.join(MODELS_DIR, 'flood_risk_model.joblib')
    joblib.dump(best_model, model_save_path)
    logging.info(f"Trained model saved to: {model_save_path}")

    # Compute feature ranges and metadata
    metadata = {
        'dataset_name': 'flood_risk.csv',
        'dataset_path': 'data/flood_risk.csv',
        'columns': X.columns.tolist(),
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'categorical_options': {
            col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in categorical_columns
        },
        'ranges': {
            col: {'min': float(df[col].min()), 'max': float(df[col].max())} for col in numeric_columns
        },
        'target': target,
        'target_distribution': y.value_counts().to_dict(),
        'target_rate': round(float(y.mean()), 4),
        'rows': len(df),
        'metrics': metrics
    }

    metadata_path = os.path.join(REPORTS_DIR, 'model_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Model metadata & reports saved to: {metadata_path}")

    return metrics

if __name__ == '__main__':
    train_and_evaluate()
