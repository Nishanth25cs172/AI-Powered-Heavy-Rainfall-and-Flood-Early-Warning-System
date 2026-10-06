import joblib
import json
import pandas as pd
import numpy as np
import datetime

model = joblib.load('ml/models/rainfall_prediction_model.joblib')
with open('ml/reports/rainfall_model_metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

def predict_rainfall(state_name, date_str, normal=None, recent_rainfall=None):
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    month = dt.month
    day = dt.day
    doy = dt.timetuple().tm_yday
    sin_doy = np.sin(2 * np.pi * doy / 365.25)
    cos_doy = np.cos(2 * np.pi * doy / 365.25)
    is_monsoon = 1 if month in [6, 7, 8, 9] else 0

    norm_table = meta.get('climatological_normal_lookup', {})
    state_table = norm_table.get(state_name, {})
    default_normal = state_table.get(str(month), 4.65)

    normal_val = float(normal) if (normal is not None and str(normal).strip() != '') else float(default_normal)
    lag_val = float(recent_rainfall) if (recent_rainfall is not None and str(recent_rainfall).strip() != '') else normal_val

    row = {
        'state_name': state_name,
        'month': month,
        'day': day,
        'dayofyear': doy,
        'sin_doy': sin_doy,
        'cos_doy': cos_doy,
        'is_monsoon': is_monsoon,
        'normal_filled': normal_val,
        'lag_1': lag_val
    }

    df_single = pd.DataFrame([row])
    pred = float(model.predict(df_single)[0])
    pred = max(0.0, round(pred, 2))

    if pred < 2.5:
        category, status, color = 'No Rain / Trace', 'LOW', '#10b981'
    elif pred < 15.6:
        category, status, color = 'Light Rain', 'LIGHT', '#38bdf8'
    elif pred < 64.5:
        category, status, color = 'Moderate Rain', 'MODERATE', '#f59e0b'
    elif pred < 115.6:
        category, status, color = 'Heavy Rain', 'HEAVY', '#f97316'
    elif pred < 204.5:
        category, status, color = 'Very Heavy Rain', 'VERY HEAVY', '#ef4444'
    else:
        category, status, color = 'Extremely Heavy Rain', 'EXTREMELY HEAVY', '#dc2626'

    deviation = round(((pred - normal_val) / (normal_val + 1e-4)) * 100, 1)

    return {
        'predicted_rainfall_mm': pred,
        'prediction_status': status,
        'imd_category': category,
        'badge_color': color,
        'normal_baseline_mm': normal_val,
        'deviation_pct': deviation,
        'input_summary': {
            'state_name': state_name,
            'date': date_str,
            'month': month,
            'day': day,
            'recent_rainfall_mm': lag_val,
            'climatological_normal_mm': normal_val
        }
    }

# Test sample scenarios
scenarios = [
    ('Maharashtra', '2024-07-15', None, 45.0), # Monsoon deluge scenario
    ('Kerala', '2024-08-10', 25.0, 80.0),      # Heavy monsoon scenario
    ('Rajasthan', '2024-01-15', None, 0.0),    # Winter dry scenario
    ('Assam', '2024-06-25', None, 30.0)        # Pre-monsoon surge
]

print("=== TESTING RAINFALL PREDICTION FUNCTION ===")
for state, date, norm, lag in scenarios:
    res = predict_rainfall(state, date, norm, lag)
    print(f"State: {state:15} Date: {date} -> Predicted: {res['predicted_rainfall_mm']:6.2f} mm | Status: {res['prediction_status']:10} | Cat: {res['imd_category']}")
print("Function test PASSED!")
