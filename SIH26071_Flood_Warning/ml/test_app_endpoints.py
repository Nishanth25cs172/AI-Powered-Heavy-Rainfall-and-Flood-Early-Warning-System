import os
import sys
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'backend'))

from app import app, meta

print("=" * 70)
print("TESTING FLASK APPLICATION ENDPOINTS VIA TEST CLIENT")
print("=" * 70)

client = app.test_client()

# 1. Test Static Landing / Login
res = client.get('/')
print(f"GET / -> Status: {res.status_code} (HTML length: {len(res.data)})")
assert res.status_code == 200

# 2. Test Login API
res = client.post('/api/login', json={'username': 'admin', 'password': 'admin123'})
data = json.loads(res.data.decode('utf-8'))
print(f"POST /api/login -> Status: {res.status_code}, Success: {data.get('success')}")
assert res.status_code == 200 and data.get('success') is True

# 3. Test Config API
res = client.get('/api/config')
data = json.loads(res.data.decode('utf-8'))
print(f"GET /api/config -> Status: {res.status_code}, Columns: {len(data['columns'])}, Datasets: {list(data['datasets'].keys())}")
assert res.status_code == 200

# 4. Test Datasets Info API
res = client.get('/api/datasets/info')
data = json.loads(res.data.decode('utf-8'))
print(f"GET /api/datasets/info -> Status: {res.status_code}, D1: {data['dataset1']['name']}, D2: {data['dataset2']['name']}")
assert res.status_code == 200
assert 'Daily Rainfall Data - India (2009-2024)' in data['dataset1']['name']
assert 'flood_risk' in data['dataset2']['name']

# 5. Test Rainfall Analytics API
res = client.get('/api/rainfall-analytics')
data = json.loads(res.data.decode('utf-8'))
print(f"GET /api/rainfall-analytics -> Status: {res.status_code}, Records: {data['records_count']}, Years: {len(data['yearly_trends']['years'])}, States: {data['unique_states_count']}")
assert res.status_code == 200
assert data['records_count'] == 204876

# 6. Test Dashboard API
res = client.get('/api/dashboard')
data = json.loads(res.data.decode('utf-8'))
print(f"GET /api/dashboard -> Status: {res.status_code}, Flood Records: {data['records']}, Map Points: {len(data['points'])}, Alerts: {len(data['alerts'])}")
assert res.status_code == 200

# 7. Test Analytics API (Section 1 and Section 2)
res = client.get('/api/analytics')
data = json.loads(res.data.decode('utf-8'))
print(f"GET /api/analytics -> Status: {res.status_code}")
print(f"  Section 1 (Flood Risk): Samples={data['section1_flood_risk']['total_samples']}, Accuracy={data['section1_flood_risk']['model_performance']['accuracy']}")
print(f"  Section 2 (Rainfall Analytics): Records={data['section2_historical_rainfall']['total_records']}, Peak Month={data['section2_historical_rainfall']['monthly_seasonality']['peak_month']}")
assert res.status_code == 200
assert 'section1_flood_risk' in data
assert 'section2_historical_rainfall' in data

# 8. Test ML Prediction API
sample_payload = {
    'Latitude': 25.32,
    'Longitude': 82.97,
    'Rainfall (mm)': 278.4,
    'Temperature (°C)': 26.8,
    'Humidity (%)': 96.5,
    'River Discharge (m³/s)': 4120.0,
    'Water Level (m)': 8.92,
    'Elevation (m)': 68.0,
    'Land Cover': 'Urban',
    'Soil Type': 'Clay',
    'Population Density': 6850.0,
    'Infrastructure': 1.0,
    'Historical Floods': 1.0
}
res = client.post('/api/predict', json=sample_payload)
data = json.loads(res.data.decode('utf-8'))
print(f"POST /api/predict -> Status: {res.status_code}")
print(f"  Probability: {data['flood_probability']}%, Risk Level: {data['risk_level']}, Warning: {data['warning_status']}")
print(f"  Warning Message: {data['warning_message']}")
print(f"  Evacuation: {data['evacuation_recommendation']}")
print(f"  Contributing Factors: {data['contributing_factors']}")
assert res.status_code == 200

# 9. Test ML Prediction with alternative key names (e.g. without degrees/cubed symbols)
alt_payload = {
    'latitude': 20.5,
    'longitude': 78.9,
    'rainfall': 140.0,
    'temperature': 28.0,
    'humidity': 80.0,
    'river discharge': 2200.0,
    'water level': 4.5,
    'elevation': 250.0,
    'land cover': 'Agricultural',
    'soil type': 'Loam',
    'population density': 3500.0,
    'infrastructure': 1,
    'historical floods': 0
}
res = client.post('/api/predict', json=alt_payload)
data = json.loads(res.data.decode('utf-8'))
print(f"POST /api/predict (normalized aliases) -> Status: {res.status_code}, Risk: {data['risk_level']}")
assert res.status_code == 200

# 10. Test Missing Fields Error Handling
res = client.post('/api/predict', json={'Latitude': 12.3})
data = json.loads(res.data.decode('utf-8'))
print(f"POST /api/predict (invalid payload) -> Status: {res.status_code}, Error: {data.get('error')}, Missing: {len(data.get('fields', []))} fields")
assert res.status_code == 400

print("\n" + "=" * 70)
print("ALL 10 API TEST SUITES PASSED FLAWLESSLY!")
print("=" * 70)
