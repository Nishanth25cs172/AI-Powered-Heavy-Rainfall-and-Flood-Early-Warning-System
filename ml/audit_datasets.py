import pandas as pd
import numpy as np

print("=== 1. RAINFALL DATASET AUDIT ===")
rf_path = 'data/Daily Rainfall Data - India (2009-2024).csv'
df_rf = pd.read_csv(rf_path)
print(f"Total rows: {len(df_rf):,}")
print("Columns:", df_rf.columns.tolist())
print(df_rf.head(2))

df_rf['date'] = pd.to_datetime(df_rf['date'], errors='coerce')
print(f"Date range: {df_rf['date'].min()} to {df_rf['date'].max()}")
print("Null count:\n", df_rf.isnull().sum())

# Actual rainfall statistics
act = pd.to_numeric(df_rf['actual'], errors='coerce').dropna()
print(f"Actual stats: Min={act.min()}, Median={act.median()}, Mean={act.mean():.2f}, Max={act.max()}")
for thresh in [15.6, 35.5, 64.5, 115.6]:
    count = (act >= thresh).sum()
    pct = (act >= thresh).mean() * 100
    print(f"Actual >= {thresh} mm: {count:,} records ({pct:.2f}%)")

print("\n=== 2. FLOOD RISK DATASET AUDIT ===")
fl_path = 'data/flood_risk.csv'
df_fl = pd.read_csv(fl_path)
print(f"Total rows: {len(df_fl):,}")
print("Columns:", df_fl.columns.tolist())
print(df_fl.head(2))
print("Target distribution:\n", df_fl['Flood Occurred'].value_counts(normalize=True))
print("Correlation with target:")
num_cols = df_fl.select_dtypes(include=[np.number]).columns
corrs = df_fl[num_cols].corr()['Flood Occurred'].sort_values(ascending=False)
print(corrs)
