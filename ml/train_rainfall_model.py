"""
SIH26071 – AI/ML-Based Integrated Heavy Rainfall Early Warning and Inundation Prediction System
Module A: Heavy Rainfall Binary Classification Pipeline
Dataset: data/Daily Rainfall Data - India (2009-2024).csv

Logistic Regression Rainfall Classifier:
- Inputs: State, Date, Normal Rainfall, RFS, Lag 1, Lag 3, Lag 7
- Data Leakage Audit: Excludes 'deviation' and 'actual' (current day) from prediction inputs.
- Algorithm: Logistic Regression (class_weight='balanced', solver='lbfgs')
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- Threshold: IMD Standard (Heavy Rainfall >= 64.5 mm in 24 hours)
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Daily Rainfall Data - India (2009-2024).csv')
MODELS_DIR = os.path.join(BASE_DIR, 'ml', 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'ml', 'reports')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

IMD_HEAVY_RAINFALL_THRESHOLD_MM = 64.5

def train_and_evaluate_rainfall_model():
    logging.info(f"Loading Daily Rainfall Dataset from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Rainfall dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logging.info(f"Raw rainfall records: {len(df):,} rows x {len(df.columns)} columns")

    # 1. Cleaning & Temporal Sorting
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'actual']).copy()
    df = df.sort_values(['state_name', 'date']).reset_index(drop=True)
    logging.info(f"Valid records with observed actual rainfall: {len(df):,}")

    min_date = df['date'].min().strftime('%Y-%m-%d')
    max_date = df['date'].max().strftime('%Y-%m-%d')
    logging.info(f"Observation timeline: {min_date} to {max_date}")

    # 2. Binary Target Creation based on IMD Heavy Rainfall Threshold (>= 64.5 mm)
    df['Heavy_Rainfall'] = (df['actual'] >= IMD_HEAVY_RAINFALL_THRESHOLD_MM).astype(int)
    pos_count = int(df['Heavy_Rainfall'].sum())
    neg_count = int(len(df) - pos_count)
    pos_rate = float(df['Heavy_Rainfall'].mean())

    logging.info(f"Target 'Heavy_Rainfall' distribution: 0={neg_count:,}, 1={pos_count:,} ({pos_rate*100:.3f}% heavy rain events)")

    # 3. Chronological Feature Engineering (Zero Lookahead / Zero Data Leakage)
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofyear'] = df['date'].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)

    # Shifted Lags: strictly antecedent observed rainfall
    df['lag_1'] = df.groupby('state_name')['actual'].shift(1)
    df['lag_3'] = df.groupby('state_name')['actual'].shift(3)
    df['lag_7'] = df.groupby('state_name')['actual'].shift(7)

    # Clean missing values in features
    df['normal_filled'] = df['normal'].fillna(0.0)
    df['rfs'] = df['rfs'].fillna(0.0)
    df['lag_1'] = df['lag_1'].fillna(df['normal_filled']).fillna(0.0)
    df['lag_3'] = df['lag_3'].fillna(df['normal_filled']).fillna(0.0)
    df['lag_7'] = df['lag_7'].fillna(df['normal_filled']).fillna(0.0)

    # 4. Strict Chronological Train/Test Split
    # Past data (2009-2021) -> Training
    # Future holdout (2022-2024) -> Testing
    split_cutoff = '2021-12-31'
    train_mask = df['date'] <= split_cutoff
    test_mask = df['date'] > split_cutoff

    cat_features = ['state_name']
    num_features = [
        'month', 'day', 'dayofyear', 'sin_doy', 'cos_doy',
        'is_monsoon', 'normal_filled', 'rfs', 'lag_1', 'lag_3', 'lag_7'
    ]
    features = cat_features + num_features

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, 'Heavy_Rainfall']
    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, 'Heavy_Rainfall']

    logging.info(f"Chronological Train (2009-2021): {len(X_train):,} records | Heavy rain: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
    logging.info(f"Chronological Test  (2022-2024): {len(X_test):,} records | Heavy rain: {y_test.sum()} ({y_test.mean()*100:.3f}%)")

    # 5. Preprocessing Pipeline: StandardScaler for numerical, OneHotEncoder for state
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    # 6. PRIMARY MODEL: Logistic Regression
    logging.info("Training PRIMARY MODEL: Logistic Regression...")
    logistic_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            C=1.0,
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        ))
    ])

    logistic_pipeline.fit(X_train, y_train)

    y_pred_log = logistic_pipeline.predict(X_test)
    y_prob_log = logistic_pipeline.predict_proba(X_test)[:, 1]

    acc_log = float(accuracy_score(y_test, y_pred_log))
    prec_log = float(precision_score(y_test, y_pred_log, zero_division=0))
    rec_log = float(recall_score(y_test, y_pred_log, zero_division=0))
    f1_log = float(f1_score(y_test, y_pred_log, zero_division=0))
    auc_log = float(roc_auc_score(y_test, y_prob_log))
    cm_log = confusion_matrix(y_test, y_pred_log).tolist()
    clf_report_log = classification_report(y_test, y_pred_log, output_dict=True, zero_division=0)

    logging.info("=" * 65)
    logging.info("PRIMARY MODEL: LOGISTIC REGRESSION EVALUATION (Test 2022-2024):")
    logging.info(f"  Accuracy        : {acc_log:.4f} ({acc_log*100:.2f}%)")
    logging.info(f"  Precision       : {prec_log:.4f}")
    logging.info(f"  Recall (Priority): {rec_log:.4f} ({rec_log*100:.2f}% of heavy rain events caught)")
    logging.info(f"  F1-Score        : {f1_log:.4f}")
    logging.info(f"  ROC-AUC         : {auc_log:.4f}")
    logging.info(f"  Confusion Matrix: {cm_log}")
    logging.info("=" * 65)

    # 7. MODEL COMPARISON: Decision Tree & Random Forest
    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(class_weight='balanced', max_depth=8, min_samples_leaf=15, random_state=42))
    ])
    dt_pipeline.fit(X_train, y_train)
    y_pred_dt = dt_pipeline.predict(X_test)
    y_prob_dt = dt_pipeline.predict_proba(X_test)[:, 1]
    acc_dt = float(accuracy_score(y_test, y_pred_dt))
    rec_dt = float(recall_score(y_test, y_pred_dt, zero_division=0))
    f1_dt = float(f1_score(y_test, y_pred_dt, zero_division=0))
    auc_dt = float(roc_auc_score(y_test, y_prob_dt))

    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, random_state=42, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    acc_rf = float(accuracy_score(y_test, y_pred_rf))
    rec_rf = float(recall_score(y_test, y_pred_rf, zero_division=0))
    f1_rf = float(f1_score(y_test, y_pred_rf, zero_division=0))
    auc_rf = float(roc_auc_score(y_test, y_prob_rf))

    # 8. Save Model Artifacts
    primary_model_path = os.path.join(MODELS_DIR, 'rainfall_logistic_model.joblib')
    joblib.dump(logistic_pipeline, primary_model_path)
    joblib.dump(logistic_pipeline, os.path.join(MODELS_DIR, 'rainfall_prediction_model.joblib'))
    logging.info(f"Primary Logistic Regression model saved to: {primary_model_path}")

    # Extract Coefficients for Explainability
    log_clf = logistic_pipeline.named_steps['classifier']
    cat_enc = logistic_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_state_names = cat_enc.get_feature_names_out(cat_features).tolist() if hasattr(cat_enc, 'get_feature_names_out') else []
    all_feature_names = num_features + encoded_state_names
    coefficients = log_clf.coef_[0].tolist()

    top_pos_coefs = sorted(
        [{'feature': f, 'coefficient': round(c, 4)} for f, c in zip(all_feature_names, coefficients)],
        key=lambda x: x['coefficient'],
        reverse=True
    )[:8]

    # Compute State Climatological Normal Lookup
    state_monthly_norm = df.groupby(['state_name', 'month'])['normal'].mean().round(2).unstack().to_dict()
    state_norm_lookup = {}
    states_list = sorted(df['state_name'].dropna().unique().tolist())
    for st in states_list:
        state_norm_lookup[st] = {int(m): float(state_monthly_norm.get(m, {}).get(st, 15.0)) for m in range(1, 13)}

    # 9. Save Detailed Metadata & Technical Report
    metadata = {
        'model_name': 'Heavy Rainfall Binary Classification Model',
        'primary_algorithm': 'Logistic Regression',
        'target_definition': 'Heavy_Rainfall (1 = Observed Daily Rainfall >= 64.5 mm, 0 = Below 64.5 mm)',
        'meteorological_threshold': {
            'threshold_mm': IMD_HEAVY_RAINFALL_THRESHOLD_MM,
            'agency': 'India Meteorological Department (IMD)',
            'criterion': 'Daily 24h precipitation >= 64.5 mm classified as Heavy Rainfall'
        },
        'data_leakage_audit': {
            'deviation_status': 'EXCLUDED to prevent data leakage (deviation = actual - normal, actual is target)',
            'antecedent_lags_used': ['lag_1', 'lag_3', 'lag_7'],
            'forecast_features_used': ['rfs', 'normal_filled']
        },
        'dataset_source': 'Daily Rainfall Data - India (2009-2024).csv',
        'dataset_total_records': len(df),
        'temporal_split': {
            'strategy': 'Chronological (2009-2021 Train, 2022-2024 Test)',
            'training_period': f'{min_date} to 2021-12-31',
            'testing_period': f'2022-01-01 to {max_date}',
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        },
        'features': {
            'all_features': features,
            'numeric_features': num_features,
            'categorical_features': cat_features
        },
        'primary_evaluation_metrics': {
            'accuracy': round(acc_log, 4),
            'precision': round(prec_log, 4),
            'recall': round(rec_log, 4),
            'f1_score': round(f1_log, 4),
            'roc_auc': round(auc_log, 4),
            'confusion_matrix': cm_log,
            'classification_report': clf_report_log
        },
        'algorithm_comparison': {
            'Logistic Regression (Primary)': {
                'accuracy': round(acc_log, 4),
                'recall': round(rec_log, 4),
                'precision': round(prec_log, 4),
                'f1_score': round(f1_log, 4),
                'roc_auc': round(auc_log, 4)
            },
            'Decision Tree (Comparison)': {
                'accuracy': round(acc_dt, 4),
                'recall': round(rec_dt, 4),
                'precision': round(float(precision_score(y_test, y_pred_dt, zero_division=0)), 4),
                'f1_score': round(f1_dt, 4),
                'roc_auc': round(auc_dt, 4)
            },
            'Random Forest (Comparison)': {
                'accuracy': round(acc_rf, 4),
                'recall': round(rec_rf, 4),
                'precision': round(float(precision_score(y_test, y_pred_rf, zero_division=0)), 4),
                'f1_score': round(f1_rf, 4),
                'roc_auc': round(auc_rf, 4)
            }
        },
        'explainability_top_positive_weights': top_pos_coefs,
        'states': states_list,
        'climatological_normal_lookup': state_norm_lookup
    }

    report_path = os.path.join(REPORTS_DIR, 'rainfall_model_metadata.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Rainfall metadata and reports successfully saved to: {report_path}")

    return metadata

if __name__ == '__main__':
    train_and_evaluate_rainfall_model()
