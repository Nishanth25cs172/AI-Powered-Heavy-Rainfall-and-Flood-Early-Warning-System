import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data/Daily Rainfall Data - India (2009-2024).csv')
df = df.dropna(subset=['actual']).copy()
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).sort_values(['state_name', 'date']).reset_index(drop=True)

# Derive lag_1 (previous day's actual rainfall in that state)
df['prev_date'] = df.groupby('state_name')['date'].shift(1)
df['date_diff'] = (df['date'] - df['prev_date']).dt.days
df['lag_1'] = df.groupby('state_name')['actual'].shift(1)
# Only keep lag_1 if date_diff == 1, otherwise fill with normal
df.loc[df['date_diff'] != 1, 'lag_1'] = np.nan

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofyear'] = df['date'].dt.dayofyear
df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)

state_month_normal = df.groupby(['state_name', 'month'])['normal'].transform('mean')
df['normal_filled'] = df['normal'].fillna(state_month_normal).fillna(df['normal'].mean())
df['lag_1'] = df['lag_1'].fillna(df['normal_filled'])

# Chronological split
train_mask = df['year'] <= 2021
test_mask = df['year'] >= 2022

features = ['state_name', 'month', 'day', 'dayofyear', 'sin_doy', 'cos_doy', 'is_monsoon', 'normal_filled', 'lag_1']
target = 'actual'

X_train = df.loc[train_mask, features]
y_train = df.loc[train_mask, target]
X_test = df.loc[test_mask, features]
y_test = df.loc[test_mask, target]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['month', 'day', 'dayofyear', 'sin_doy', 'cos_doy', 'is_monsoon', 'normal_filled', 'lag_1']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['state_name'])
    ]
)

pipe = Pipeline([
    ('prep', preprocessor),
    ('reg', HistGradientBoostingRegressor(max_iter=200, learning_rate=0.07, max_leaf_nodes=40, random_state=42))
])
pipe.fit(X_train, y_train)

y_pred = np.clip(pipe.predict(X_test), 0, None)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model with Lag-1 on Holdout Test Set (2022-2024):")
print(f"  MAE:  {mae:.4f} mm")
print(f"  RMSE: {rmse:.4f} mm")
print(f"  R²:   {r2:.4f}")
