import urllib.request
import json

base_url = 'http://127.0.0.1:5000'
pages = [
    '/login.html',
    '/dashboard.html',
    '/prediction.html',
    '/results.html',
    '/risk-map.html',
    '/alerts.html',
    '/analytics.html'
]

print("=== CHECKING FRONTEND PAGES ===")
for p in pages:
    url = f"{base_url}{p}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            print(f"[OK] {p} - Status: {resp.status} ({len(content)} bytes)")
    except Exception as e:
        print(f"[FAIL] {p} - Error: {e}")

apis = [
    '/api/config',
    '/api/dashboard',
    '/api/analytics',
    '/api/rainfall-analytics',
    '/api/datasets/info'
]

print("\n=== CHECKING BACKEND API ENDPOINTS ===")
for a in apis:
    url = f"{base_url}{a}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print(f"[OK] {a} - Status: {resp.status}, Top Keys: {list(data.keys())[:5]}")
    except Exception as e:
        print(f"[FAIL] {a} - Error: {e}")

print("\n=== TESTING PREDICTION POST API ===")
predict_url = f"{base_url}/api/predict"
sample_payload = {
    "Latitude": 19.07,
    "Longitude": 72.87,
    "Rainfall (mm)": 245.5,
    "Temperature (°C)": 28.5,
    "Humidity (%)": 92.0,
    "River Discharge (m³/s)": 3450.0,
    "Water Level (m)": 7.8,
    "Elevation (m)": 15.0,
    "Land Cover": "Urban",
    "Soil Type": "Clay",
    "Population Density": 7500.0,
    "Infrastructure": 1.0,
    "Historical Floods": 1.0
}
data_bytes = json.dumps(sample_payload).encode('utf-8')
req = urllib.request.Request(predict_url, data=data_bytes, headers={'Content-Type': 'application/json'}, method='POST')
try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode('utf-8'))
        print(f"[OK] /api/predict - Status: {resp.status}")
        print(f"     Risk Level: {result.get('risk_level')}")
        print(f"     Warning Status: {result.get('warning_status')}")
        print(f"     Probability: {result.get('flood_probability')}%")
        print(f"     Evacuation: {result.get('evacuation_recommendation')}")
        print(f"     Contributing Factors: {result.get('contributing_factors')}")
except urllib.error.HTTPError as he:
    print(f"[FAIL] /api/predict - HTTP Error {he.code}: {he.read().decode('utf-8')}")
except Exception as e:
    print(f"[FAIL] /api/predict - Error: {e}")

print("\n=== ALL LIVE HTTP VERIFICATIONS COMPLETE ===")
