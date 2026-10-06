import os
import sys
import json
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))

from app import app, meta

def run_tests():
    print("=" * 70)
    print("AI-POWERED HEAVY RAINFALL AND FLOOD EARLY-WARNING SYSTEM: INTEGRATION & SYSTEM TEST")
    print("=" * 70)

    # 1. Dataset Loading
    print("\n[TEST 1] EXACT OFFICIAL DATASETS VALIDATION")
    p1 = os.path.join(BASE_DIR, 'data', 'Daily Rainfall Data - India (2009-2024).csv')
    p2 = os.path.join(BASE_DIR, 'data', 'flood_risk.csv')
    
    assert os.path.exists(p1), f"Rainfall dataset missing at {p1}"
    assert os.path.exists(p2), f"Flood risk dataset missing at {p2}"
    
    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2, encoding='utf-8')
    print(f"  [OK] Dataset 1 (Daily Rainfall Data - India): {len(df1):,} records, {len(df1.columns)} columns")
    print(f"  [OK] Dataset 2 (flood_risk.csv): {len(df2):,} records, {len(df2.columns)} columns")
    assert len(df1) == 204876, f"Expected 204,876 rows in Dataset 1, got {len(df1)}"
    assert len(df2) == 10000, f"Expected 10,000 rows in Dataset 2, got {len(df2)}"

    # 2. Preprocessing & Temporal Extraction
    print("\n[TEST 2] DATA PREPROCESSING & SEASONALITY")
    df1['date'] = pd.to_datetime(df1['date'], errors='coerce')
    df1['year'] = df1['date'].dt.year
    df1['month'] = df1['date'].dt.month
    monthly_mean = df1.groupby('month')['actual'].mean()
    yearly_mean = df1.groupby('year')['actual'].mean()
    print(f"  [OK] Dataset 1: {len(monthly_mean)} months seasonality, {len(yearly_mean)} years timeline (2009–2024)")
    print(f"  [OK] Dataset 2: 'Flood Occurred' distribution: {dict(df2['Flood Occurred'].value_counts())}")

    # 3. ML Prediction (Local Pipeline)
    print("\n[TEST 3] ML PIPELINE PREDICTION")
    model_path = os.path.join(BASE_DIR, 'ml', 'models', 'flood_risk_model.joblib')
    meta_path = os.path.join(BASE_DIR, 'ml', 'reports', 'model_metadata.json')
    assert os.path.exists(model_path), f"Model file missing at {model_path}"
    assert os.path.exists(meta_path), f"Metadata file missing at {meta_path}"

    model = joblib.load(model_path)
    meta_data = json.load(open(meta_path, encoding='utf-8'))
    
    sample_row = {
        'Latitude': 25.32, 'Longitude': 82.97, 'Rainfall (mm)': 278.4, 'Temperature (°C)': 26.8,
        'Humidity (%)': 96.5, 'River Discharge (m³/s)': 4120.0, 'Water Level (m)': 8.92,
        'Elevation (m)': 68.0, 'Land Cover': 'Urban', 'Soil Type': 'Clay',
        'Population Density': 6850.0, 'Infrastructure': 1.0, 'Historical Floods': 1.0
    }
    x_test = pd.DataFrame([sample_row])
    prob = float(model.predict_proba(x_test)[0, 1])
    print(f"  [OK] Model loaded: Pipeline with {len(meta_data['columns'])} features")
    print(f"  [OK] Sample inference: Flood Probability = {prob*100:.2f}% (Risk: {'CRITICAL' if prob>=.75 else 'HIGH'})")

    # 4. Backend REST API Endpoints via Flask Test Client
    print("\n[TEST 4] BACKEND REST API ENDPOINTS")
    client = app.test_client()
    endpoints = [
        ('GET', '/api/datasets/info', None),
        ('GET', '/api/rainfall-analytics', None),
        ('GET', '/api/dashboard', None),
        ('GET', '/api/analytics', None),
        ('GET', '/api/config', None),
        ('POST', '/api/login', {'username': 'admin', 'password': 'admin123'}),
        ('POST', '/api/predict', sample_row)
    ]
    for method, path, body in endpoints:
        if method == 'GET':
            res = client.get(path)
        else:
            res = client.post(path, json=body)
        assert res.status_code == 200, f"Expected 200 for {method} {path}, got {res.status_code}"
        parsed = json.loads(res.data.decode('utf-8'))
        print(f"  [OK] {method} {path} -> HTTP 200 OK (Keys: {list(parsed.keys())[:4]}...)")

    # 5. Frontend Pages & Markup Verification
    print("\n[TEST 5] FRONTEND PAGES & MARKUP")
    pages = [
        'login.html', 'dashboard.html', 'prediction.html',
        'results.html', 'risk-map.html', 'alerts.html', 'analytics.html'
    ]
    for p in pages:
        full_p = os.path.join(BASE_DIR, 'frontend', p)
        assert os.path.exists(full_p), f"Required page {p} missing in frontend!"
        print(f"  [OK] Page exists: {p} ({os.path.getsize(full_p):,} bytes)")

    dash_html = open(os.path.join(BASE_DIR, 'frontend', 'dashboard.html'), encoding='utf-8').read()
    assert "Daily Rainfall Data - India (2009-2024)" in dash_html, "Dataset 1 missing in dashboard.html"
    assert "flood_risk.csv" in dash_html, "Dataset 2 missing in dashboard.html"
    assert "SECTION 1" in dash_html and "FLOOD RISK ANALYTICS" in dash_html, "Section 1 missing in dashboard.html"
    assert "SECTION 2" in dash_html and "HISTORICAL RAINFALL ANALYTICS" in dash_html, "Section 2 missing in dashboard.html"
    print("  [OK] Dashboard HTML contains both official datasets and separated Section 1 & Section 2 analytics")

    # 6. Error Handling
    print("\n[TEST 6] ERROR HANDLING")
    bad_res = client.post('/api/predict', json={'Latitude': 12.3})
    assert bad_res.status_code == 400, f"Expected 400 for missing fields, got {bad_res.status_code}"
    err_body = json.loads(bad_res.data.decode('utf-8'))
    print(f"  [OK] Incomplete payload caught: HTTP 400 with message '{err_body['error']}' and {len(err_body['fields'])} missing fields reported")

    print("\n" + "=" * 70)
    print("ALL 6 INTEGRATION & VERIFICATION TEST SUITES PASSED (100% SUCCESS)!")
    print("=" * 70)

if __name__ == '__main__':
    run_tests()
