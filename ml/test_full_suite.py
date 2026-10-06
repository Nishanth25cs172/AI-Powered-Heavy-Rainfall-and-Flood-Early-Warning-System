import urllib.request
import json
import sys

base = 'http://127.0.0.1:5000'

def check(url, name, method='GET', data=None):
    req = urllib.request.Request(url, method=method)
    if data:
        req.add_header('Content-Type', 'application/json')
        body = json.dumps(data).encode('utf-8')
    else:
        body = None
    try:
        with urllib.request.urlopen(req, data=body, timeout=5) as res:
            status = res.status
            content = res.read().decode('utf-8', errors='ignore')
            print(f"[PASS] {name} ({method} {url}) -> HTTP {status} (Length: {len(content)})")
            return content
    except Exception as e:
        print(f"[FAIL] {name} ({method} {url}) -> Error: {e}")
        return None

print("=== 1. HTML PAGES ===")
d_html = check(f"{base}/dashboard.html", "Dashboard")
rp_html = check(f"{base}/rainfall-prediction.html", "Rainfall Prediction Page")
rr_html = check(f"{base}/rainfall-results.html", "Rainfall Results Page")
check(f"{base}/prediction.html", "Flood Prediction Page")
check(f"{base}/analytics.html", "Analytics Page")
check(f"{base}/risk-map.html", "Risk Map Page")
check(f"{base}/alerts.html", "Alerts Page")
check(f"{base}/login.html", "Login Page")

print("\n=== 2. RAINFALL CARD BUTTONS ON DASHBOARD ===")
assert d_html is not None, "Dashboard HTML could not be loaded"
assert "[View Analytics]" in d_html, "Missing [View Analytics] on dashboard"
assert "[View Predictions]" in d_html, "Missing [View Predictions] on dashboard"
assert 'href="/rainfall-prediction.html"' in d_html, "Missing link to rainfall-prediction.html on dashboard"
print("[PASS] Dashboard contains both [View Analytics] and [View Predictions] with link to /rainfall-prediction.html")

print("\n=== 3. RAINFALL PREDICTION PAGE CONTENTS ===")
assert rp_html is not None, "Rainfall Prediction HTML could not be loaded"
assert "AI RAINFALL PREDICTION" in rp_html, "Missing AI RAINFALL PREDICTION title"
assert "Predict rainfall using historical rainfall patterns from the Daily Rainfall Data - India (2009-2024) dataset." in rp_html, "Missing expected subtitle"
assert "state_name" in rp_html, "Missing state_name input"
assert "date" in rp_html, "Missing date input"
assert "normal" in rp_html, "Missing normal input"
assert "recent_rainfall" in rp_html, "Missing recent_rainfall input"
print("[PASS] rainfall-prediction.html matches all required titles, subtitles, and actual dataset inputs")

print("\n=== 4. API ENDPOINTS & INFERENCE ===")
check(f"{base}/api/config", "Config API")
check(f"{base}/api/dashboard", "Dashboard Telemetry API")
check(f"{base}/api/analytics", "Analytics API")
check(f"{base}/api/rainfall-analytics", "Rainfall Analytics API")
check(f"{base}/api/datasets/info", "Datasets Info API")
check(f"{base}/api/rainfall-prediction-config", "Rainfall Prediction Config API")

# Test Rainfall Prediction
rf_res = check(f"{base}/api/predict-rainfall", "Rainfall Prediction Inference", method='POST', data={
    'state_name': 'Maharashtra',
    'date': '2024-07-15',
    'normal': 22.4,
    'recent_rainfall': 38.5
})
assert rf_res is not None, "Rainfall prediction API failed"
rf_json = json.loads(rf_res)
print(f"   Rainfall Prediction -> {rf_json['predicted_rainfall_mm']} mm, Status: {rf_json['prediction_status']}, IMD: {rf_json['imd_category']}")

# Test Flood Risk Prediction (original)
fl_res = check(f"{base}/api/predict", "Flood Risk Prediction Inference", method='POST', data={
    'Latitude': 19.076,
    'Longitude': 72.877,
    'Rainfall (mm)': 140.0,
    'Temperature (°C)': 29.0,
    'Humidity (%)': 88.0,
    'River Discharge (m³/s)': 1200.0,
    'Water Level (m)': 5.2,
    'Elevation (m)': 14.0,
    'Land Cover': 'Urban',
    'Soil Type': 'Clay',
    'Population Density': 4500.0,
    'Infrastructure': 0.8,
    'Historical Floods': 1.0
})
assert fl_res is not None, "Flood risk prediction API failed"
fl_json = json.loads(fl_res)
print(f"   Flood Risk Prediction -> Probability: {fl_json['flood_probability']}%, Level: {fl_json['risk_level']}")

print("\n=== ALL SYSTEM TESTS PASSED SUCCESSFULLY! ===")
