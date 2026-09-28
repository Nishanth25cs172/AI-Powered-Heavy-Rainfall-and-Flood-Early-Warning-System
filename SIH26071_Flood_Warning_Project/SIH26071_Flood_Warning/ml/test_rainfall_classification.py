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
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

print("Loading rainfall dataset...")
df = pd.read_csv('data/Daily Rainfall Data - India (2009-2024).csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date', 'actual']).sort_values(['state_name', 'date']).reset_index(drop=True)

# IMD Meteorological Standard Threshold for Heavy Rainfall: >= 64.5 mm in 24 hours
IMD_HEAVY_THRESHOLD = 64.5
df['Heavy_Rainfall'] = (df['actual'] >= IMD_HEAVY_THRESHOLD).astype(int)

print(f"Total valid records: {len(df):,}")
print("Target counts:")
print(df['Heavy_Rainfall'].value_counts())
print(f"Heavy rainfall proportion: {df['Heavy_Rainfall'].mean()*100:.3f}%")

# Chronological Feature Engineering (strictly past information per state)
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofyear'] = df['date'].dt.dayofyear
df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)

# Lag features per state (shift(1) ensures no lookahead leakage)
df['lag_1'] = df.groupby('state_name')['actual'].shift(1)
df['lag_2'] = df.groupby('state_name')['actual'].shift(2)
df['lag_3'] = df.groupby('state_name')['actual'].shift(3)
# Rolling 7-day past average excluding today
df['rolling_7d'] = df.groupby('state_name')['actual'].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())

# Impute initial lags with state normal or 0
df['lag_1'] = df['lag_1'].fillna(df['normal']).fillna(0)
df['lag_2'] = df['lag_2'].fillna(df['normal']).fillna(0)
df['lag_3'] = df['lag_3'].fillna(df['normal']).fillna(0)
df['rolling_7d'] = df['rolling_7d'].fillna(df['normal']).fillna(0)
df['normal_filled'] = df['normal'].fillna(df['actual']).fillna(0)

# Chronological Split: 2009-2021 Train, 2022-2024 Test
train_mask = df['date'] <= '2021-12-31'
test_mask = df['date'] >= '2022-01-01'

features = ['state_name', 'month', 'day', 'dayofyear', 'sin_doy', 'cos_doy', 'is_monsoon', 'normal_filled', 'lag_1', 'lag_2', 'lag_3', 'rolling_7d']
num_features = ['month', 'day', 'dayofyear', 'sin_doy', 'cos_doy', 'is_monsoon', 'normal_filled', 'lag_1', 'lag_2', 'lag_3', 'rolling_7d']
cat_features = ['state_name']

X_train = df.loc[train_mask, features]
y_train = df.loc[train_mask, 'Heavy_Rainfall']

X_test = df.loc[test_mask, features]
y_test = df.loc[test_mask, 'Heavy_Rainfall']

print(f"\nTrain set (2009-2021): {len(X_train):,} rows, {y_train.sum()} heavy rain events ({y_train.mean()*100:.3f}%)")
print(f"Test set (2022-2024): {len(X_test):,} rows, {y_test.sum()} heavy rain events ({y_test.mean()*100:.3f}%)")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ]
)

# 1. Primary Algorithm: Logistic Regression (Unit III Syllabus)
print("\n--- Training Primary Model: Logistic Regression (class_weight='balanced') ---")
log_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

log_pipe.fit(X_train, y_train)

y_pred_log = log_pipe.predict(X_test)
y_prob_log = log_pipe.predict_proba(X_test)[:, 1]

print("Logistic Regression Holdout Results:")
print("Accuracy :", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred_log, zero_division=0))
print("F1-score :", f1_score(y_test, y_pred_log, zero_division=0))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log, zero_division=0))

# 2. Decision Tree Comparison
print("\n--- Training Comparison: Decision Tree Classifier ---")
dt_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(class_weight='balanced', max_depth=8, random_state=42))
])
dt_pipe.fit(X_train, y_train)
y_pred_dt = dt_pipe.predict(X_test)
y_prob_dt = dt_pipe.predict_proba(X_test)[:, 1]
print("Decision Tree Holdout Results:")
print("Accuracy :", accuracy_score(y_test, y_pred_dt))
print("Recall   :", recall_score(y_test, y_pred_dt, zero_division=0))
print("F1-score :", f1_score(y_test, y_pred_dt, zero_division=0))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_dt))
