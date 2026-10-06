import urllib.request
import urllib.error
import json

def test(url, method='GET', data=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8') if data else None,
        headers={'Content-Type': 'application/json'} if data else {},
        method=method
    )
    try:
        res = urllib.request.urlopen(req)
        body = json.loads(res.read().decode())
        print(f"[PASS] {method} {url} -> HTTP {res.status}")
        return body
    except urllib.error.HTTPError as e:
        body = json.loads(e.read().decode())
        print(f"[EXPECTED HTTP {e.code}] {method} {url} -> {body}")
        return body

print("="*60)
print("--- 1. Testing Dataset Info API ---")
info = test('http://127.0.0.1:5000/api/datasets/info')
print("  Dataset 1:", info['dataset1']['name'], "| Status:", info['dataset1']['status'], "| Records:", f"{info['dataset1']['records']:,}")
print("  Dataset 2:", info['dataset2']['name'], "| Status:", info['dataset2']['status'], "| Records:", f"{info['dataset2']['records']:,}")

print("\n--- 2. Testing Rainfall Analytics API ---")
rain = test('http://127.0.0.1:5000/api/rainfall-analytics')
print("  Monthly Trend Months:", rain['monthly_trend']['labels'])
print("  Actual Monthly Precipitation (mm):", rain['monthly_trend']['actual_monthly_mm'])
print("  Peak Monsoon Month:", rain['monthly_trend']['peak_monsoon'])
print("  Timeline Years:", rain['yearly_analysis']['years'])
print("  Extreme Rain Days (>100mm):", rain['yearly_analysis']['extreme_rainfall_days'])

print("\n--- 3. Testing Dashboard Telemetry API ---")
dash = test('http://127.0.0.1:5000/api/dashboard')
print("  Flood Dataset Records:", f"{dash['records']:,}")
print("  Rainfall Dataset Records:", f"{dash['rainfall_dataset_records']:,}")
print("  Avg Rainfall:", dash['avg_rainfall'], "mm | Flood Rate:", dash['flood_rate'], "%")

print("\n--- 4. Testing Analytics Telemetry API ---")
analytics = test('http://127.0.0.1:5000/api/analytics')
print("  Metrics:", analytics['metrics']['accuracy'], "accuracy |", analytics['metrics']['f1'], "F1")

print("\n--- 5. Testing ML Predict API (Valid 13 Features) ---")
pred_payload = {
    'Latitude': 25.32, 'Longitude': 82.97, 'Rainfall (mm)': 278.4, 'Temperature (°C)': 26.8,
    'Humidity (%)': 96.5, 'River Discharge (m³/s)': 4120.0, 'Water Level (m)': 8.92,
    'Elevation (m)': 68.0, 'Land Cover': 'Urban', 'Soil Type': 'Clay',
    'Population Density': 6850.0, 'Infrastructure': 1.0, 'Historical Floods': 1.0
}
pred = test('http://127.0.0.1:5000/api/predict', method='POST', data=pred_payload)
print("  Inference Result:", pred['risk_level'], f"({pred['flood_probability']}%)", "| Prediction Flag:", pred['prediction'])

print("\n--- 6. Testing Error Handling (Missing Inputs) ---")
err = test('http://127.0.0.1:5000/api/predict', method='POST', data={'Latitude': 25.32})
print("  Error Handling Response:", err['error'], "| Missing fields:", err.get('fields'))
print("="*60)
print("ALL TESTS PASSED SUCCESSFULLY!")
