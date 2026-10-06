import urllib.request
import json

base = 'http://127.0.0.1:5000'

def request(url, method='GET', data=None):
    req = urllib.request.Request(url, method=method)
    if data:
        req.add_header('Content-Type', 'application/json')
        body = json.dumps(data).encode('utf-8')
    else:
        body = None
    with urllib.request.urlopen(req, data=body, timeout=6) as res:
        return res.status, json.loads(res.read().decode('utf-8'))

print("=== 1. TEST GET /api/health ===")
status, h = request(f"{base}/api/health")
print(f"Status: {status}")
print("Health payload:", json.dumps(h, indent=2))
assert h['status'] == 'healthy'
assert 'Decision Tree' in h['models']['flood_inundation_primary']
assert 'Logistic Regression' in h['models']['heavy_rainfall_primary']

print("\n=== 2. TEST POST /api/predict-rainfall (Logistic Regression) ===")
status, rf = request(f"{base}/api/predict-rainfall", method='POST', data={
    'state_name': 'Maharashtra',
    'date': '2024-07-22',
    'normal': 22.4,
    'recent_rainfall': 75.0
})
print(f"Status: {status}")
print("Heavy Rainfall Result:", json.dumps(rf, indent=2))
assert 'heavy_rainfall' in rf
assert 'prediction_label' in rf
assert 'probability' in rf
assert 'threshold_criteria' in rf
print(f"-> Heavy Rainfall: {rf['prediction_label']} (Prob: {rf['probability_pct']}%), Risk: {rf['risk_level']}")

print("\n=== 3. TEST POST /api/predict-flood (Decision Tree) ===")
status, fl = request(f"{base}/api/predict-flood", method='POST', data={
    'Latitude': 19.076,
    'Longitude': 72.877,
    'Rainfall (mm)': 140.0,
    'Temperature (°C)': 29.0,
    'Humidity (%)': 88.0,
    'River Discharge (m³/s)': 2800.0,
    'Water Level (m)': 6.2,
    'Elevation (m)': 14.0,
    'Land Cover': 'Urban',
    'Soil Type': 'Clay',
    'Population Density': 4500.0,
    'Infrastructure': 0.8,
    'Historical Floods': 1.0
})
print(f"Status: {status}")
print("Flood Result:", json.dumps(fl, indent=2))
assert 'flood_risk' in fl
assert 'primary_model' in fl
assert 'Decision Tree' in fl['primary_model']
print(f"-> Flood Risk: {fl['prediction_label']} (Prob: {fl['probability_pct']}%), Risk: {fl['risk_level']}")

print("\n=== 4. TEST POST /api/predict-risk (Integrated Risk Engine) ===")
status, rk = request(f"{base}/api/predict-risk", method='POST', data={
    'rainfall': {
        'state_name': 'Assam',
        'date': '2024-07-15',
        'recent_rainfall': 88.0
    },
    'flood': {
        'Latitude': 26.20,
        'Longitude': 92.93,
        'River Discharge (m³/s)': 4200.0,
        'Water Level (m)': 7.8,
        'Elevation (m)': 45.0,
        'Land Cover': 'Agricultural',
        'Soil Type': 'Silt',
        'Population Density': 1200.0,
        'Infrastructure': 0.5,
        'Historical Floods': 1.0
    }
})
print(f"Status: {status}")
print("Integrated Risk Result:", json.dumps(rk, indent=2))
assert 'rainfall_prediction' in rk
assert 'flood_prediction' in rk
assert 'overall_risk' in rk
assert 'warning_bulletin' in rk
print(f"-> Compound Risk: {rk['overall_risk']} (Score: {rk['compound_risk_score']}%)")
print(f"-> Bulletin: {rk['warning_bulletin']}")

print("\n=== ALL RECTIFIED BACKEND ENDPOINTS PASSED CLEANLY! ===")
