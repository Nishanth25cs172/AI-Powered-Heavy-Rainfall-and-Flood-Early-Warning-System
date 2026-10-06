import os
import json
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')

p_flood = os.path.join(data_dir, 'flood_risk.csv')
p_rain = os.path.join(data_dir, 'Daily Rainfall Data - India (2009-2024).csv')

print("=" * 80)
print("SIH26071 FLOOD WARNING PROJECT: DETAILED DATASET INSPECTION")
print("=" * 80)

# Check bytes/encoding of flood_risk.csv
with open(p_flood, 'rb') as f:
    raw_head = f.readline()
    print("\nRaw header bytes of flood_risk.csv:")
    print(raw_head)

# Try utf-8 vs latin1 vs cp1252
try:
    df_flood = pd.read_csv(p_flood, encoding='utf-8')
    print("flood_risk.csv read successfully with utf-8")
except Exception as e:
    print(f"utf-8 failed: {e}")
    df_flood = pd.read_csv(p_flood, encoding='latin1')
    print("flood_risk.csv read with latin1")

print("\n" + "-" * 40)
print("DATASET 1: flood_risk.csv")
print("-" * 40)
print(f"File Path: {p_flood}")
print(f"File Size: {os.path.getsize(p_flood):,} bytes")
print(f"Dimensions: {df_flood.shape[0]:,} rows x {df_flood.shape[1]} columns")
print("\nColumns and Data Types:")
for c in df_flood.columns:
    print(f"  - '{c}': dtype={df_flood[c].dtype}, null_count={df_flood[c].isnull().sum()}")

print("\nSummary Statistics of flood_risk.csv:")
print(df_flood.describe().T)

print("\nCategorical columns unique values:")
for c in df_flood.select_dtypes(include=['object']).columns:
    print(f"  - {c}: {df_flood[c].unique().tolist()}")

if 'Flood Occurred' in df_flood.columns:
    print("\nTarget Column 'Flood Occurred' distribution:")
    print(df_flood['Flood Occurred'].value_counts(dropna=False))
    print(f"Positive class ratio: {df_flood['Flood Occurred'].mean():.4f}")

print("\n" + "-" * 40)
print("DATASET 2: Daily Rainfall Data - India (2009-2024).csv")
print("-" * 40)
print(f"File Path: {p_rain}")
print(f"File Size: {os.path.getsize(p_rain):,} bytes")

# Read Daily Rainfall Data
df_rain = pd.read_csv(p_rain)
print(f"Dimensions: {df_rain.shape[0]:,} rows x {df_rain.shape[1]} columns")
print("\nColumns and Data Types:")
for c in df_rain.columns:
    print(f"  - '{c}': dtype={df_rain[c].dtype}, null_count={df_rain[c].isnull().sum()} ({df_rain[c].isnull().mean()*100:.2f}%)")

print("\nRainfall Date range:")
df_rain['date'] = pd.to_datetime(df_rain['date'], errors='coerce')
print(f"  Min Date: {df_rain['date'].min()}")
print(f"  Max Date: {df_rain['date'].max()}")

print("\nRainfall States (sample 10):")
print(f"  Total unique states: {df_rain['state_name'].nunique()}")
print(f"  States: {df_rain['state_name'].dropna().unique()[:10].tolist()}")

print("\nRainfall Summary Statistics:")
print(df_rain[['actual', 'rfs', 'normal', 'deviation']].describe().T)

print("\n" + "=" * 80)
print("CROSS-DATASET COMMON KEY COMPATIBILITY ANALYSIS")
print("=" * 80)
flood_cols = set(df_flood.columns)
rain_cols = set(df_rain.columns)
common_cols = flood_cols.intersection(rain_cols)
print(f"Direct common column names: {common_cols}")
print(f"flood_risk columns: {list(df_flood.columns)}")
print(f"rainfall columns: {list(df_rain.columns)}")

has_geo_flood = any(k in c.lower() for c in df_flood.columns for k in ['lat', 'lon', 'state', 'district', 'city'])
has_date_flood = any('date' in c.lower() or 'year' in c.lower() for c in df_flood.columns)
print(f"flood_risk has geographic fields: {has_geo_flood} (Latitude, Longitude present: {'Latitude' in df_flood.columns and 'Longitude' in df_flood.columns})")
print(f"flood_risk has date/time fields: {has_date_flood}")
print(f"rainfall has date/time fields: True ('date' present)")
print(f"rainfall has geographic fields: True ('state_code', 'state_name' present, but NO lat/long)")

if len(common_cols) == 0 and not has_date_flood:
    print("\nCONCLUSION ON JOINING:")
    print("-> NO reliable common key exists between the two datasets.")
    print("-> flood_risk.csv has pointwise Latitude/Longitude coordinates and environmental measurements with no timestamps or state names.")
    print("-> Daily Rainfall Data has daily state-aggregated time series with state names and dates, but no pointwise latitude/longitude.")
    print("-> As per instructions: DO NOT force a join.")
    print("-> Keep both datasets separate.")
    print("-> Use them together in the AIML system through coordinated analysis:")
    print("   - flood_risk.csv for flood-risk prediction and geospatial risk modeling.")
    print("   - Daily Rainfall Data for historical rainfall intelligence, monsoon trends, state-wise patterns, and extreme weather hazard baselines.")
