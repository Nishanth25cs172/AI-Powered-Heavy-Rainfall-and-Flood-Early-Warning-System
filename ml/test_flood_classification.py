import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

print("Loading flood risk dataset...")
df = pd.read_csv('data/flood_risk.csv')
target = 'Flood Occurred'
X = df.drop(columns=[target])
y = df[target].astype(int)

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

print(f"Total rows: {len(df):,}")
print(f"Features: {X.columns.tolist()}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Stratified 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

# 1. Primary Model: Decision Tree Classifier (Unit III Syllabus)
print("\n--- Training Primary Flood Model: Decision Tree Classifier ---")
dt_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=6, min_samples_split=20, min_samples_leaf=10, random_state=42))
])

dt_pipe.fit(X_train, y_train)

y_pred_dt = dt_pipe.predict(X_test)
y_prob_dt = dt_pipe.predict_proba(X_test)[:, 1]

acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, zero_division=0)
rec_dt = recall_score(y_test, y_pred_dt, zero_division=0)
f1_dt = f1_score(y_test, y_pred_dt, zero_division=0)
auc_dt = roc_auc_score(y_test, y_prob_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)

print("Decision Tree Classifier Evaluation (Test Set):")
print(f"  Accuracy : {acc_dt:.4f} ({acc_dt*100:.2f}%)")
print(f"  Precision: {prec_dt:.4f}")
print(f"  Recall   : {rec_dt:.4f}")
print(f"  F1-Score : {f1_dt:.4f}")
print(f"  ROC-AUC  : {auc_dt:.4f}")
print("Confusion Matrix:\n", cm_dt)
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt, zero_division=0))

# 2. Ensemble Comparison: Random Forest Classifier
print("\n--- Training Comparison: Random Forest Classifier ---")
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1))
])
rf_pipe.fit(X_train, y_train)
y_pred_rf = rf_pipe.predict(X_test)
y_prob_rf = rf_pipe.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print("Random Forest Evaluation (Test Set):")
print(f"  Accuracy : {acc_rf:.4f} ({acc_rf*100:.2f}%)")
print(f"  Recall   : {rec_rf:.4f}")
print(f"  F1-Score : {f1_rf:.4f}")
print(f"  ROC-AUC  : {auc_rf:.4f}")
