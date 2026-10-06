import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data_path = 'data/Daily Rainfall Data - India (2009-2024).csv'
print(f"Loading {data_path}...")
df = pd.read_csv(data_path)

# Drop missing target values
df = df.dropna(subset=['actual']).copy()
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

print(f"Valid records with actual rainfall: {len(df):,}")

# Feature Engineering strictly from the dataset
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofyear'] = df['date'].dt.dayofyear

# Cyclical seasonal encoding
df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

# Monsoon indicator: June to September is standard Indian SW Monsoon
df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)

# Normal rainfall: fill missing normal with state-month average normal
state_month_normal = df.groupby(['state_name', 'month'])['normal'].transform('mean')
df['normal_filled'] = df['normal'].fillna(state_month_normal).fillna(df['normal'].mean())

features = ['state_name', 'month', 'day', 'dayofyear', 'sin_doy', 'cos_doy', 'is_monsoon', 'normal_filled']
target = 'actual'

print(f"Features: {features}")
print(f"Target: {target}")

# Chronological Train-Test Split (Train: 2009-2021, Test: 2022-2024)
train_mask = df['year'] <= 2021
test_mask = df['year'] >= 2022

X_train = df.loc[train_mask, features]
y_train = df.loc[train_mask, target]

X_test = df.loc[test_mask, features]
y_test = df.loc[test_mask, target]

print(f"Train records (2009-2021): {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
print(f"Test records (2022-2024): {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")

cat_cols = ['state_name']
num_cols = ['month', 'day', 'dayofyear', 'sin_doy', 'cos_doy', 'is_monsoon', 'normal_filled']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

# Test models
models = {
    'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=42),
}

for name, reg in models.items():
    print(f"\nTraining {name} pipeline...")
    pipe = Pipeline([
        ('prep', preprocessor),
        ('reg', reg)
    ])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)  # Rainfall cannot be negative
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name} Results on Holdout Test Set (2022-2024):")
    print(f"  MAE:  {mae:.4f} mm")
    print(f"  RMSE: {rmse:.4f} mm")
    print(f"  R²:   {r2:.4f}")
