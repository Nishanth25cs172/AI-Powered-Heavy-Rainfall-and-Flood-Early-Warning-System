import urllib.request
import json

base = "http://127.0.0.1:5000"

print("=== TESTING /api/rainfall-prediction-config ===")
req = urllib.request.Request(f"{base}/api/rainfall-prediction-config")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))
    print(f"Status: {resp.status}")
    print(f"Available States: {len(data['states'])}")
    print(f"Metrics: {data['metrics']}")
    print(f"IMD Categories: {list(data['imd_categories'].keys())}")

print("\n=== TESTING /api/predict-rainfall (POST) ===")
sample_requests = [
    {"state_name": "Maharashtra", "date": "2024-07-20", "normal": 18.5, "recent_rainfall": 45.0},
    {"state_name": "Kerala", "date": "2024-08-12", "normal": 22.0, "recent_rainfall": 85.0},
    {"state_name": "Rajasthan", "date": "2024-01-10", "normal": 0.2, "recent_rainfall": 0.0},
    {"state_name": "Assam", "date": "2024-06-28"} # auto-defaults normal and lag_1
]

for p in sample_requests:
    b = json.dumps(p).encode('utf-8')
    r = urllib.request.Request(f"{base}/api/predict-rainfall", data=b, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(r) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print(f"[OK] {p['state_name']:15} on {p['date']} -> Predicted: {res['predicted_rainfall_mm']:6.2f} mm | Status: {res['prediction_status']:12} | Category: {res['imd_category']}")
        print(f"     Normal: {res['normal_baseline_mm']} mm | Deviation: {res['deviation_pct']}% | Advisory: {res['warning_message'][:60]}...")

print("\n=== TESTING EXISTING FLOOD PREDICT API (REGRESSION CHECK) ===")
flood_payload = {
    "Latitude": 19.07, "Longitude": 72.87, "Rainfall (mm)": 245.5,
    "Temperature (°C)": 28.5, "Humidity (%)": 92.0, "River Discharge (m³/s)": 3450.0,
    "Water Level (m)": 7.8, "Elevation (m)": 15.0, "Land Cover": "Urban",
    "Soil Type": "Clay", "Population Density": 7500.0, "Infrastructure": 1.0,
    "Historical Floods": 1.0
}
b = json.dumps(flood_payload).encode('utf-8')
r = urllib.request.Request(f"{base}/api/predict", data=b, headers={'Content-Type': 'application/json'}, method='POST')
with urllib.request.urlopen(r) as resp:
    res = json.loads(resp.read().decode('utf-8'))
    print(f"[OK] Flood Risk Predict -> Risk: {res['risk_level']} ({res['flood_probability']}%) | Warning: {res['warning_status']}")

print("\nALL BACKEND API TESTS PASSED 100%!")
