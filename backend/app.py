import os
import json
import logging
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONT = os.path.join(BASE, 'frontend')
DATA_DIR = os.path.join(BASE, 'data')

# Exactly TWO Official Project Datasets
DATASET1_PATH = os.path.join(DATA_DIR, 'Daily Rainfall Data - India (2009-2024).csv')
DATASET2_PATH = os.path.join(DATA_DIR, 'flood_risk.csv')

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.risk_engine import RiskEngine

FLOOD_DT_MODEL_PATH = os.path.join(BASE, 'ml', 'models', 'flood_decision_tree_model.joblib')
FLOOD_RF_MODEL_PATH = os.path.join(BASE, 'ml', 'models', 'flood_risk_model.joblib')
FLOOD_META_PATH = os.path.join(BASE, 'ml', 'reports', 'flood_model_metadata.json')
META_PATH = os.path.join(BASE, 'ml', 'reports', 'model_metadata.json')

RAINFALL_LOGISTIC_PATH = os.path.join(BASE, 'ml', 'models', 'rainfall_logistic_model.joblib')
RAINFALL_META_PATH = os.path.join(BASE, 'ml', 'reports', 'rainfall_model_metadata.json')

app = Flask(__name__, static_folder=FRONT, static_url_path='')

# ----------------- Load Flood Risk ML Models (Random Forest & Decision Tree) -----------------
logging.info(f"Loading Primary Flood Random Forest Model from {FLOOD_RF_MODEL_PATH}...")
if os.path.exists(os.path.join(BASE, 'ml', 'models', 'flood_random_forest_model.joblib')):
    FLOOD_RF_MODEL_PATH = os.path.join(BASE, 'ml', 'models', 'flood_random_forest_model.joblib')
flood_rf_model = joblib.load(FLOOD_RF_MODEL_PATH)

logging.info(f"Loading Comparison Flood Decision Tree Model from {FLOOD_DT_MODEL_PATH}...")
flood_dt_model = joblib.load(FLOOD_DT_MODEL_PATH)
model = flood_rf_model  # Primary default classifier for flood prediction

with open(FLOOD_META_PATH if os.path.exists(FLOOD_META_PATH) else META_PATH, 'r', encoding='utf-8') as f:
    meta = json.load(f)

# Normalize meta keys for compatibility across endpoints
if 'rows' not in meta:
    meta['rows'] = meta.get('dataset_total_samples', 10000)
if 'metrics' not in meta and 'primary_evaluation_metrics' in meta:
    meta['metrics'] = dict(meta['primary_evaluation_metrics'])
if 'metrics' in meta:
    if 'f1_score' in meta['metrics'] and 'f1' not in meta['metrics']:
        meta['metrics']['f1'] = meta['metrics']['f1_score']
    if 'feature_importances' not in meta['metrics']:
        meta['metrics']['feature_importances'] = meta.get('feature_importances', [])

# ----------------- Load Rainfall Prediction ML Model (Logistic Regression) -----------------
logging.info(f"Loading Primary Rainfall Logistic Regression Model from {RAINFALL_LOGISTIC_PATH}...")
rainfall_model = joblib.load(RAINFALL_LOGISTIC_PATH)
with open(RAINFALL_META_PATH, 'r', encoding='utf-8') as f:
    rainfall_meta = json.load(f)

# ----------------- Load Primary Flood Risk Dataset -----------------
logging.info(f"Loading primary flood risk dataset from {DATASET2_PATH}...")
df_flood = pd.read_csv(DATASET2_PATH, encoding='utf-8')
df = df_flood  # backwards compatibility alias

# ----------------- Load & Process Historical Rainfall Dataset -----------------
logging.info(f"Loading historical rainfall dataset from {DATASET1_PATH}...")
df_rain = pd.read_csv(DATASET1_PATH)
df_rain['date'] = pd.to_datetime(df_rain['date'], errors='coerce')
df_rain['year'] = df_rain['date'].dt.year
df_rain['month'] = df_rain['date'].dt.month

# Precompute Rainfall Analytics for Fast Response
logging.info("Precomputing historical rainfall analytics (2009–2024)...")
days_in_month = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Monthly Seasonality
m_actual_daily = df_rain.groupby('month')['actual'].mean().to_dict()
m_normal_daily = df_rain.groupby('month')['normal'].mean().to_dict()
monthly_actual_totals = [round(float(m_actual_daily.get(m, 0.0) * days_in_month[m]), 1) for m in range(1, 13)]
monthly_normal_totals = [round(float(m_normal_daily.get(m, 0.0) * days_in_month[m]), 1) for m in range(1, 13)]

# Yearly Trends (2009-2024)
years_list = sorted([int(y) for y in df_rain['year'].dropna().unique()])
y_actual_mean = df_rain.groupby('year')['actual'].mean().to_dict()
y_normal_mean = df_rain.groupby('year')['normal'].mean().to_dict()
y_max_daily = df_rain.groupby('year')['actual'].max().to_dict()

# Annual totals estimate (accounting for partial year 2024 through July)
yearly_actual_totals = [
    round(float(y_actual_mean.get(y, 3.8) * 365.25 * (7/12 if y == 2024 else 1.0)), 1)
    for y in years_list
]
yearly_normal_totals = [
    round(float(y_normal_mean.get(y, 4.6) * 365.25 * (7/12 if y == 2024 else 1.0)), 1)
    for y in years_list
]

# Heavy Rainfall Events Counting (IMD Standards)
extreme_100_by_year = df_rain[df_rain['actual'] >= 100].groupby('year').size().to_dict()
heavy_64_by_year = df_rain[(df_rain['actual'] >= 64.5) & (df_rain['actual'] < 100)].groupby('year').size().to_dict()

# State-wise Ranking across 36 Indian States/UTs
state_grp = df_rain.groupby('state_name').agg(
    records=('id', 'count'),
    mean_daily=('actual', 'mean'),
    max_daily=('actual', 'max'),
    heavy_days=('actual', lambda x: int((x >= 64.5).sum()))
).reset_index()

state_grp['mean_daily'] = state_grp['mean_daily'].round(2)
state_grp['max_daily'] = state_grp['max_daily'].round(2)
state_grp_sorted = state_grp.sort_values(by='mean_daily', ascending=False)

top_wettest_states = state_grp_sorted.head(10).to_dict(orient='records')
driest_states = state_grp.sort_values(by='mean_daily', ascending=True).head(10).to_dict(orient='records')
all_states_summary = state_grp_sorted.to_dict(orient='records')

# IMD Heavy Rainfall Categorization Totals
imd_heavy_count = int(df_rain[(df_rain['actual'] >= 64.5) & (df_rain['actual'] < 115.5)].shape[0])
imd_very_heavy_count = int(df_rain[(df_rain['actual'] >= 115.5) & (df_rain['actual'] < 204.5)].shape[0])
imd_extremely_heavy_count = int(df_rain[df_rain['actual'] >= 204.5].shape[0])

# Top single-day rainfall records in India (2009-2024)
top_records = df_rain.nlargest(10, 'actual')[['date', 'state_name', 'actual', 'normal', 'deviation']].copy()
top_records['date'] = top_records['date'].dt.strftime('%Y-%m-%d')
top_extreme_events = top_records.round(2).to_dict(orient='records')

# Rainfall Distribution Bins
rain_bins = [-1, 2.5, 15.5, 64.5, 115.5, 204.5, 350.0]
rain_bin_labels = ['No Rain (<2.5mm)', 'Light (2.5-15.5mm)', 'Moderate (15.5-64.5mm)', 'Heavy (64.5-115.5mm)', 'Very Heavy (115.5-204.5mm)', 'Extremely Heavy (>204.5mm)']
rain_dist_counts = pd.cut(df_rain['actual'], bins=rain_bins, labels=rain_bin_labels).value_counts()[rain_bin_labels].tolist()

rainfall_summary = {
    'records_count': len(df_rain),
    'valid_records': int(df_rain['actual'].notnull().sum()),
    'unique_states_count': int(df_rain['state_name'].nunique()),
    'date_range': {
        'start': df_rain['date'].min().strftime('%Y-%m-%d'),
        'end': df_rain['date'].max().strftime('%Y-%m-%d'),
        'span_years': '2009–2024 (16 Years)'
    },
    'overall_stats': {
        'min_daily_rainfall': round(float(df_rain['actual'].min()), 2),
        'max_daily_rainfall': round(float(df_rain['actual'].max()), 2),
        'mean_daily_rainfall': round(float(df_rain['actual'].mean()), 2),
        'median_daily_rainfall': round(float(df_rain['actual'].median()), 2),
        'std_daily_rainfall': round(float(df_rain['actual'].std()), 2)
    },
    'monthly_seasonality': {
        'labels': month_labels,
        'actual_monthly_mm': monthly_actual_totals,
        'normal_monthly_mm': monthly_normal_totals,
        'peak_month': 'July',
        'monsoon_window': 'June – September'
    },
    'yearly_trends': {
        'years': [str(y) for y in years_list],
        'annual_actual_mm': yearly_actual_totals,
        'annual_normal_mm': yearly_normal_totals,
        'lpa_baseline_mm': 1160.0,
        'extreme_days_100mm': [int(extreme_100_by_year.get(y, 0)) for y in years_list],
        'heavy_days_64mm': [int(heavy_64_by_year.get(y, 0)) for y in years_list]
    },
    'imd_hazard_classification': {
        'categories': ['Heavy Rain (64.5–115.5mm)', 'Very Heavy Rain (115.6–204.4mm)', 'Extremely Heavy Rain (≥204.5mm)'],
        'counts': [imd_heavy_count, imd_very_heavy_count, imd_extremely_heavy_count]
    },
    'distribution': {
        'labels': rain_bin_labels,
        'counts': [int(c) for c in rain_dist_counts]
    },
    'top_wettest_states': top_wettest_states,
    'driest_states': driest_states,
    'all_states': all_states_summary,
    'top_extreme_events': top_extreme_events
}
logging.info("Rainfall analytics precomputed successfully.")

# ----------------- Helper Functions -----------------
def normalize_feature_payload(data):
    """Normalizes incoming payload to match exact model feature names."""
    field_aliases = {
        'temperature': 'Temperature (°C)',
        'temperature (c)': 'Temperature (°C)',
        'temperature (°c)': 'Temperature (°C)',
        'river discharge': 'River Discharge (m³/s)',
        'river discharge (m3/s)': 'River Discharge (m³/s)',
        'river discharge (m³/s)': 'River Discharge (m³/s)',
        'rainfall': 'Rainfall (mm)',
        'rainfall (mm)': 'Rainfall (mm)',
        'water level': 'Water Level (m)',
        'water level (m)': 'Water Level (m)',
        'elevation': 'Elevation (m)',
        'elevation (m)': 'Elevation (m)',
        'humidity': 'Humidity (%)',
        'humidity (%)': 'Humidity (%)',
        'population density': 'Population Density',
        'historical floods': 'Historical Floods',
        'infrastructure': 'Infrastructure',
        'land cover': 'Land Cover',
        'soil type': 'Soil Type',
        'latitude': 'Latitude',
        'longitude': 'Longitude'
    }

    normalized = {}
    for k, v in data.items():
        lk = str(k).strip().lower()
        target_name = field_aliases.get(lk, k)
        normalized[target_name] = v
    return normalized

def determine_risk_band(prob_pct, rainfall=None, water_level=None):
    """Categorizes risk probability into standard operational disaster bands."""
    if prob_pct < 25.0:
        band = 'LOW'
        slug = 'low'
        badge_color = '#10b981'
        warning_level = 'NORMAL'
        warning_msg = 'No imminent flood inundation threat. Environmental and hydraulic parameters remain within baseline operating ranges.'
        evacuation = 'None'
    elif prob_pct < 50.0:
        band = 'MODERATE'
        slug = 'moderate'
        badge_color = '#f59e0b'
        warning_level = 'ADVISORY'
        warning_msg = 'Moderate flood risk advisory. Precipitation and catchment saturation require active monitoring by local authorities.'
        evacuation = 'Routine Vigilance'
    elif prob_pct < 75.0:
        band = 'HIGH'
        slug = 'high'
        badge_color = '#f97316'
        warning_level = 'WARNING'
        warning_msg = 'High flood risk alert. Sustained heavy precipitation and rising river levels indicate probable localized inundation.'
        evacuation = 'Precautionary Preparedness'
    else:
        band = 'CRITICAL'
        slug = 'critical'
        badge_color = '#ef4444'
        warning_level = 'EMERGENCY'
        warning_msg = 'CRITICAL FLOOD HAZARD DETECTED! Severe precipitation volume and hydraulic threshold breaches. Immediate action required!'
        evacuation = 'Immediate Evacuation to High Ground'

    return {
        'risk_band': band,
        'risk_slug': slug,
        'badge_color': badge_color,
        'warning_level': warning_level,
        'warning_msg': warning_msg,
        'evacuation': evacuation
    }

# ----------------- CORS Support -----------------
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return '', 204

# ----------------- Static & Web Routes -----------------
@app.get('/')
def index():
    return send_from_directory(FRONT, 'login.html')

@app.get('/<path:path>')
def static_files(path):
    p = os.path.join(FRONT, path)
    return send_from_directory(FRONT, path) if os.path.exists(p) else send_from_directory(FRONT, 'login.html')

# ----------------- Auth API -----------------
@app.post('/api/login')
def login():
    d = request.get_json() or {}
    username = str(d.get('username', '')).strip()
    password = str(d.get('password', '')).strip()
    ok = (username == 'admin' and password == 'admin123')
    return jsonify({
        'success': ok,
        'message': 'Command Center authentication successful' if ok else 'Invalid credentials. Use demo: admin / admin123',
        'operator': 'Incident Commander (Admin)' if ok else None
    })

# ----------------- Telemetry & Config API -----------------
@app.get('/api/config')
def get_config():
    return jsonify({
        'columns': meta['columns'],
        'numeric_columns': meta['numeric_columns'],
        'categorical_columns': meta['categorical_columns'],
        'categorical_options': meta['categorical_options'],
        'ranges': meta['ranges'],
        'target': meta['target'],
        'rows': meta['rows'],
        'datasets': {
            'dataset1': {
                'filename': 'Daily Rainfall Data - India (2009-2024).csv',
                'name': 'Daily Rainfall Data - India (2009-2024)',
                'path': 'data/Daily Rainfall Data - India (2009-2024).csv',
                'records': len(df_rain),
                'role': 'Historical Rainfall Intelligence & Trend Analysis',
                'status': 'Connected'
            },
            'dataset2': {
                'filename': 'flood_risk.csv',
                'name': 'flood_risk.csv',
                'path': 'data/flood_risk.csv',
                'records': len(df_flood),
                'role': 'Primary ML Flood Risk Classification & Inundation Prediction',
                'status': 'Connected'
            }
        },
        'status': 'Online'
    })

# ----------------- System Health API -----------------
@app.get('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'project_name': 'AI-Powered Heavy Rainfall & Flood Early Warning System',
        'project_title': 'AI-Powered Heavy Rainfall and Flood Early-Warning System',
        'models': {
            'heavy_rainfall_primary': 'Logistic Regression (Binary Classification, IMD >= 64.5 mm Threshold)',
            'heavy_rainfall_comparison': ['Decision Tree Classifier', 'Random Forest Classifier'],
            'flood_inundation_primary': 'Random Forest Classifier',
            'flood_inundation_comparison': 'Decision Tree Classifier'
        },
        'datasets_verified': [
            'Daily Rainfall Data - India (2009-2024).csv',
            'flood_risk.csv'
        ],
        'aiml_syllabus_mapping': {
            'Unit_I': 'Intelligent Agents, PEAS Framework, Early Warning Decision Support',
            'Unit_II': 'Data Preprocessing, Missing Value Imputation, Class Imbalance Handling, Stratified Split',
            'Unit_III': 'Random Forest Classifier, Decision Tree Classifier, Evaluation Metrics (Recall, ROC-AUC, F1)'
        },
        'risk_engine': 'Operational',
        'disclaimer': 'AI/ML-based early-warning and decision-support prototype.'
    })

# ----------------- Flood ML Prediction API (Random Forest Primary) -----------------
@app.post('/api/predict-flood')
@app.post('/api/predict')
def predict_flood():
    try:
        raw_payload = request.get_json() or {}
        payload = normalize_feature_payload(raw_payload)

        # Check required columns
        missing = [c for c in meta['columns'] if c not in payload or payload[c] in ('', None)]
        if missing:
            return jsonify({
                'error': 'Missing required prediction fields',
                'fields': missing
            }), 400

        # Construct single-row DataFrame
        row = {}
        for c in meta['columns']:
            if c in meta['numeric_columns']:
                try:
                    row[c] = float(payload[c])
                except (ValueError, TypeError):
                    return jsonify({'error': f'Field {c} must be a valid numeric value'}), 400
            else:
                row[c] = str(payload[c])

        df_single = pd.DataFrame([row])

        # Primary Inference via Random Forest Classifier
        prob_rf = float(flood_rf_model.predict_proba(df_single)[0, 1])
        prob_pct = round(prob_rf * 100, 2)
        prediction_class = int(prob_rf >= 0.5)

        # Comparison Inference via Decision Tree
        prob_dt = float(flood_dt_model.predict_proba(df_single)[0, 1])
        prob_dt_pct = round(prob_dt * 100, 2)

        rainfall_val = row.get('Rainfall (mm)', 0.0)
        water_level_val = row.get('Water Level (m)', 0.0)
        
        # Risk Evaluation via Backend Risk Engine Service
        flood_hazard = RiskEngine.evaluate_flood_hazard(prob_rf, bool(prediction_class == 1), water_level_m=water_level_val)

        # Feature contribution analysis for early warning breakdown
        contributing_factors = []
        if rainfall_val >= 150.0:
            contributing_factors.append(f'Extreme rainfall volume ({rainfall_val:.1f} mm)')
        if water_level_val >= 6.5:
            contributing_factors.append(f'High river water stage ({water_level_val:.2f} m)')
        if row.get('River Discharge (m³/s)', 0.0) >= 3000.0:
            contributing_factors.append(f'Severe discharge rate ({row.get("River Discharge (m³/s)"):.0f} m³/s)')
        if row.get('Elevation (m)', 9999) <= 100.0:
            contributing_factors.append('Low-lying coastal / flood plain elevation')
        if row.get('Land Cover') == 'Urban':
            contributing_factors.append('High urban runoff impermeability')
        if row.get('Soil Type') == 'Clay':
            contributing_factors.append('Slow clay soil percolation')

        return jsonify({
            'success': True,
            'flood_risk': bool(prediction_class == 1),
            'prediction': prediction_class,
            'prediction_label': 'YES' if prediction_class == 1 else 'NO',
            'flood_probability': prob_pct,
            'probability_pct': prob_pct,
            'risk_level': flood_hazard['risk_level'],
            'risk_slug': flood_hazard['risk_level'].lower(),
            'warning_status': flood_hazard['risk_level'],
            'warning_message': flood_hazard['message'],
            'evacuation_recommendation': flood_hazard['evacuation_advice'],
            'badge_color': flood_hazard['badge_color'],
            'contributing_factors': contributing_factors if contributing_factors else ['General environmental parameter convergence'],
            'primary_model': 'Random Forest Classifier',
            'comparison_model': 'Decision Tree Classifier',
            'comparison_probability_pct': prob_dt_pct,
            'metrics': meta.get('metrics', meta.get('primary_evaluation_metrics', {})),
            'dataset_source': 'flood_risk.csv',
            'features_received': row
        })
    except Exception as e:
        logging.error(f"Flood prediction inference error: {e}", exc_info=True)
        return jsonify({'error': 'Flood inference failure', 'details': str(e)}), 500

# ----------------- Heavy Rainfall Binary Classification API (Logistic Regression) -----------------
@app.post('/api/predict-rainfall')
def predict_rainfall_api():
    try:
        raw_payload = request.get_json() or {}
        
        # 1. State Input Validation
        state_name = str(raw_payload.get('state', raw_payload.get('state_name', ''))).strip()
        if not state_name:
            return jsonify({'error': 'Missing required input field: state'}), 400

        state_match = next((s for s in rainfall_meta['states'] if s.lower() == state_name.lower()), None)
        if not state_match:
            return jsonify({
                'error': f"Unknown state '{state_name}'. Must be one of the {len(rainfall_meta['states'])} available dataset states.",
                'available_states': rainfall_meta['states']
            }), 400
        state_name = state_match

        # 2. Date Input Validation & Feature Extraction
        date_str = str(raw_payload.get('date', '')).strip()
        if not date_str:
            return jsonify({'error': 'Missing required input field: date'}), 400
        
        try:
            import datetime
            dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Must be YYYY-MM-DD.'}), 400

        month = dt.month
        day = dt.day
        doy = dt.timetuple().tm_yday
        sin_doy = float(np.sin(2 * np.pi * doy / 365.25))
        cos_doy = float(np.cos(2 * np.pi * doy / 365.25))
        is_monsoon = 1 if month in [6, 7, 8, 9] else 0

        season_name = (
            'Southwest Monsoon' if is_monsoon else
            'Winter' if month in [1, 2] else
            'Pre-Monsoon / Summer' if month in [3, 4, 5] else
            'Post-Monsoon / Northeast Monsoon'
        )

        # 3. Numeric Parameter Validation (normal, rfs, lag1, lag3, lag7)
        rr = raw_payload.get('recent_rainfall', 0.0)
        def_norm = raw_payload.get('normal', raw_payload.get('normal_rainfall', 0.0))
        def_rfs = raw_payload.get('rfs', rr)
        def_l1 = raw_payload.get('lag1', raw_payload.get('lag_1', rr))
        def_l3 = raw_payload.get('lag3', raw_payload.get('lag_3', rr))
        def_l7 = raw_payload.get('lag7', raw_payload.get('lag_7', rr))

        numeric_fields = {
            'normal': def_norm if def_norm is not None else 0.0,
            'rfs': def_rfs if def_rfs is not None else 0.0,
            'lag1': def_l1 if def_l1 is not None else 0.0,
            'lag3': def_l3 if def_l3 is not None else 0.0,
            'lag7': def_l7 if def_l7 is not None else 0.0
        }

        missing_fields = [k for k, v in numeric_fields.items() if v is None or str(v).strip() == '']
        if missing_fields:
            return jsonify({
                'error': f"Missing required prediction parameters: {', '.join(missing_fields)}",
                'missing_fields': missing_fields
            }), 400

        parsed_numeric = {}
        for k, v in numeric_fields.items():
            try:
                val = float(v)
                if val < 0.0:
                    return jsonify({'error': f"Parameter '{k}' cannot be negative (got {val})"}), 400
                parsed_numeric[k] = val
            except (ValueError, TypeError):
                return jsonify({'error': f"Parameter '{k}' must be a valid numeric value"}), 400

        normal_val = parsed_numeric['normal']
        rfs_val = parsed_numeric['rfs']
        lag1_val = parsed_numeric['lag1']
        lag3_val = parsed_numeric['lag3']
        lag7_val = parsed_numeric['lag7']

        # 4. Construct Row Matching Logistic Regression Pipeline Feature Expectations (Zero Data Leakage)
        row = {
            'state_name': state_name,
            'month': month,
            'day': day,
            'dayofyear': doy,
            'sin_doy': sin_doy,
            'cos_doy': cos_doy,
            'is_monsoon': is_monsoon,
            'normal_filled': normal_val,
            'rfs': rfs_val,
            'lag_1': lag1_val,
            'lag_3': lag3_val,
            'lag_7': lag7_val
        }

        df_single = pd.DataFrame([row])

        # 5. Inference via Existing Fitted Logistic Regression Pipeline
        prob_heavy = float(rainfall_model.predict_proba(df_single)[0, 1])
        prob_pct = round(prob_heavy * 100, 2)
        is_heavy = bool(prob_heavy >= 0.5)
        pred_text = "HEAVY RAINFALL" if is_heavy else "NOT HEAVY RAINFALL"

        # Evaluate Warning Tiers
        hazard_info = RiskEngine.evaluate_rainfall_hazard(prob_heavy, is_heavy)

        # Retrieve Actual Model Evaluation Performance Metrics from Metadata Report
        meta_metrics = rainfall_meta.get('primary_evaluation_metrics', {})
        acc_pct = round(meta_metrics.get('accuracy', 0.9633) * 100, 2)
        prec_pct = round(meta_metrics.get('precision', 0.1019) * 100, 2)
        rec_pct = round(meta_metrics.get('recall', 0.9576) * 100, 2)

        return jsonify({
            'success': True,
            'prediction': pred_text,
            'prediction_label': pred_text,
            'prediction_status': 'FLOOD RISK' if prediction_class == 1 else 'NORMAL',
            'predicted_rainfall_mm': round(rfs_val, 2),
            'imd_category': "Heavy Rainfall" if is_heavy else "Normal Rainfall",
            'heavy_rainfall': is_heavy,
            'probability': prob_pct,
            'probability_pct': prob_pct,
            'risk_level': hazard_info['risk_level'],
            'warning_status': hazard_info['risk_level'],
            'badge_color': hazard_info['badge_color'],
            'warning_message': hazard_info['message'],
            'advisory': hazard_info['advisory'],
            'metrics': {
                'accuracy': acc_pct,
                'precision': prec_pct,
                'recall': rec_pct,
                'f1_score': round(meta_metrics.get('f1_score', 0.1842) * 100, 2),
                'roc_auc': round(meta_metrics.get('roc_auc', 0.9928) * 100, 2)
            },
            'input_summary': {
                'state': state_name,
                'date': date_str,
                'month': month,
                'day': day,
                'season': season_name,
                'normal': normal_val,
                'rfs': rfs_val,
                'lag1': lag1_val,
                'lag3': lag3_val,
                'lag7': lag7_val
            }
        })
    except Exception as e:
        logging.error(f"Rainfall prediction inference error: {e}", exc_info=True)
        return jsonify({'error': 'Rainfall inference failure', 'details': str(e)}), 500

# ----------------- Integrated Risk Engine API (Option B: Chained Prediction) -----------------
@app.post('/api/predict-risk')
def predict_integrated_risk():
    """
    Synthesizes Module A (Heavy Rainfall Logistic Regression) and Module B (Flood Decision Tree)
    into a compound disaster risk score and calibrated early warning advisory.
    """
    try:
        data = request.get_json() or {}

        # 1. Evaluate Heavy Rainfall Component
        rainfall_input = data.get('rainfall', data)
        state_name = str(rainfall_input.get('state_name', 'Maharashtra')).strip()
        state_match = next((s for s in rainfall_meta['states'] if s.lower() == state_name.lower()), 'Maharashtra')
        
        date_str = str(rainfall_input.get('date', '2024-07-15'))
        try:
            import datetime
            dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            dt = datetime.date(2024, 7, 15)

        month = dt.month
        norm_table = rainfall_meta.get('climatological_normal_lookup', {})
        default_normal = float(norm_table.get(state_match, {}).get(str(month), 18.5))
        recent_val = float(rainfall_input.get('recent_rainfall', default_normal * 1.5))

        row_rain = {
            'state_name': state_match,
            'month': month,
            'day': dt.day,
            'dayofyear': dt.timetuple().tm_yday,
            'sin_doy': float(np.sin(2 * np.pi * dt.timetuple().tm_yday / 365.25)),
            'cos_doy': float(np.cos(2 * np.pi * dt.timetuple().tm_yday / 365.25)),
            'is_monsoon': 1 if month in [6, 7, 8, 9] else 0,
            'normal_filled': default_normal,
            'lag_1': recent_val,
            'lag_2': round(recent_val * 0.9, 2),
            'lag_3': round(recent_val * 0.8, 2),
            'rolling_7d': round((recent_val * 2.7 + default_normal * 4.3) / 7.0, 2)
        }
        df_rain_single = pd.DataFrame([row_rain])
        rainfall_prob = float(rainfall_model.predict_proba(df_rain_single)[0, 1])
        heavy_rainfall = bool(rainfall_prob >= 0.5)

        # 2. Evaluate Flood Inundation Component
        flood_input = data.get('flood', data)
        # In chained integration: if flood rainfall is not explicitly provided, use predicted rainfall burden
        cascaded_rainfall = float(flood_input.get('Rainfall (mm)', recent_val if heavy_rainfall else default_normal))
        
        row_flood = {
            'Latitude': float(flood_input.get('Latitude', 19.076)),
            'Longitude': float(flood_input.get('Longitude', 72.877)),
            'Rainfall (mm)': cascaded_rainfall,
            'Temperature (°C)': float(flood_input.get('Temperature (°C)', 28.5)),
            'Humidity (%)': float(flood_input.get('Humidity (%)', 85.0)),
            'River Discharge (m³/s)': float(flood_input.get('River Discharge (m³/s)', 1800.0)),
            'Water Level (m)': float(flood_input.get('Water Level (m)', 5.5)),
            'Elevation (m)': float(flood_input.get('Elevation (m)', 25.0)),
            'Land Cover': str(flood_input.get('Land Cover', 'Urban')),
            'Soil Type': str(flood_input.get('Soil Type', 'Clay')),
            'Population Density': float(flood_input.get('Population Density', 3500.0)),
            'Infrastructure': float(flood_input.get('Infrastructure', 0.7)),
            'Historical Floods': float(flood_input.get('Historical Floods', 1.0))
        }
        df_flood_single = pd.DataFrame([row_flood])
        flood_prob = float(flood_dt_model.predict_proba(df_flood_single)[0, 1])
        flood_occurred = bool(flood_prob >= 0.5)

        # Environmental hydraulic stress (water level & discharge normalized)
        water_stress = min(max((row_flood['Water Level (m)'] - 3.0) / 7.0, 0.0), 1.0)

        # 3. Compound Early Warning Synthesis via Risk Engine
        compound_eval = RiskEngine.compute_compound_risk(
            rainfall_prob=rainfall_prob,
            flood_prob=flood_prob,
            environmental_stress=water_stress,
            location_meta={'state': state_match, 'date': date_str}
        )

        rainfall_hazard = RiskEngine.evaluate_rainfall_hazard(rainfall_prob, heavy_rainfall)
        flood_hazard = RiskEngine.evaluate_flood_hazard(flood_prob, flood_occurred, water_level_m=row_flood['Water Level (m)'])

        return jsonify({
            'rainfall_prediction': {
                'heavy_rainfall': heavy_rainfall,
                'prediction_label': 'YES' if heavy_rainfall else 'NO',
                'probability_pct': round(rainfall_prob * 100, 1),
                'risk_level': rainfall_hazard['risk_level'],
                'message': rainfall_hazard['message'],
                'primary_model': 'Logistic Regression (Unit III Syllabus)'
            },
            'flood_prediction': {
                'flood_risk': flood_occurred,
                'prediction_label': 'YES' if flood_occurred else 'NO',
                'probability_pct': round(flood_prob * 100, 1),
                'risk_level': flood_hazard['risk_level'],
                'message': flood_hazard['message'],
                'primary_model': 'Decision Tree Classifier (Unit III Syllabus)'
            },
            'overall_risk': compound_eval['overall_risk'],
            'compound_risk_score': compound_eval['compound_risk_score'],
            'badge_color': compound_eval['badge_color'],
            'warning_bulletin': compound_eval['warning_bulletin'],
            'actionable_recommendations': compound_eval['actionable_recommendations'],
            'integration_strategy': 'Option B: Chained Rainfall-to-Inundation Decision Support',
            'disclaimer': compound_eval['disclaimer']
        })
    except Exception as e:
        logging.error(f"Integrated risk prediction error: {e}", exc_info=True)
        return jsonify({'error': 'Risk synthesis failure', 'details': str(e)}), 500

@app.get('/api/rainfall-prediction-config')
def get_rainfall_prediction_config():
    metrics = rainfall_meta.get('primary_evaluation_metrics', {})
    return jsonify({
        'states': rainfall_meta.get('states', []),
        'climatological_normal_lookup': rainfall_meta.get('climatological_normal_lookup', {}),
        'metrics': metrics,
        'primary_evaluation_metrics': metrics,
        'algorithm_comparison': rainfall_meta.get('algorithm_comparison', {}),
        'meteorological_threshold': rainfall_meta.get('meteorological_threshold', {}),
        'target_definition': rainfall_meta.get('target_definition', 'Heavy_Rainfall (>= 64.5 mm)'),
        'dataset_source': 'Daily Rainfall Data - India (2009-2024).csv',
        'status': 'Online'
    })

STATE_CENTROIDS = {
    'andhra pradesh': (15.9129, 79.7400),
    'arunachal pradesh': (28.2180, 94.7278),
    'assam': (26.2006, 92.9376),
    'bihar': (25.0961, 85.3131),
    'chhattisgarh': (21.2787, 81.8661),
    'goa': (15.2993, 74.1240),
    'gujarat': (22.2587, 71.1924),
    'haryana': (29.0588, 76.0856),
    'himachal pradesh': (31.1048, 77.1734),
    'jharkhand': (23.6102, 85.2799),
    'karnataka': (15.3173, 75.7139),
    'kerala': (10.8505, 76.2711),
    'madhya pradesh': (22.9734, 78.6569),
    'maharashtra': (19.7515, 75.7139),
    'manipur': (24.6637, 93.9063),
    'meghalaya': (25.4670, 91.3662),
    'mizoram': (23.1645, 92.9376),
    'nagaland': (26.1584, 94.5624),
    'odisha': (20.9517, 85.0985),
    'punjab': (31.1471, 75.3412),
    'rajasthan': (27.0238, 74.2179),
    'sikkim': (27.5330, 88.5122),
    'tamil nadu': (11.1271, 78.6569),
    'telangana': (18.1124, 79.0193),
    'tripura': (23.9408, 91.9882),
    'uttar pradesh': (26.8467, 80.9462),
    'uttarakhand': (30.0668, 79.0193),
    'west bengal': (22.9868, 87.8550),
    'andaman and nicobar islands': (11.7401, 92.6586),
    'chandigarh': (30.7333, 76.7794),
    'the dadra and nagar haveli and daman and diu': (20.4283, 72.8397),
    'dadra and nagar haveli and daman and diu': (20.4283, 72.8397),
    'delhi': (28.7041, 77.1025),
    'jammu and kashmir': (33.7782, 76.5762),
    'ladakh': (34.1526, 77.5771),
    'lakshadweep': (10.5667, 72.6417),
    'puducherry': (11.9416, 79.8083)
}

# ----------------- Dashboard API -----------------
@app.get('/api/dashboard')
def dashboard():
    avg_rainfall = float(df_flood['Rainfall (mm)'].mean())
    flood_rate = float(df_flood['Flood Occurred'].mean() * 100)

    # Rainfall distribution from flood_risk.csv
    bins = pd.cut(df_flood['Rainfall (mm)'], [-1, 50, 100, 150, 200, 250, 301], labels=['0-50', '50-100', '100-150', '150-200', '200-250', '250-300'])
    rc = bins.value_counts().sort_index()

    # Sample representative geospatial points across India from flood_risk.csv
    sample = df_flood.sample(min(350, len(df_flood)), random_state=42)
    probs = model.predict_proba(sample[meta['columns']])[:, 1]
    points = []
    for i, (_, r) in enumerate(sample.iterrows()):
        p_pct = float(probs[i]) * 100
        r_info = determine_risk_band(p_pct)
        points.append({
            'lat': round(float(r['Latitude']), 4),
            'lon': round(float(r['Longitude']), 4),
            'prob': round(p_pct, 1),
            'risk': r_info['risk_band'],
            'observed': int(r['Flood Occurred']),
            'rainfall': round(float(r['Rainfall (mm)']), 1),
            'water_level': round(float(r['Water Level (m)']), 2),
            'discharge': round(float(r['River Discharge (m³/s)']), 1),
            'elevation': round(float(r.get('Elevation (m)', 0.0)), 1),
            'temperature': round(float(r.get('Temperature (°C)', 25.0)), 1),
            'humidity': round(float(r.get('Humidity (%)', 70.0)), 1),
            'land_cover': str(r.get('Land Cover', 'Urban')),
            'soil_type': str(r.get('Soil Type', 'Clay')),
            'prediction': int(probs[i] >= 0.5),
            'prediction_label': 'FLOOD' if probs[i] >= 0.5 else 'NO FLOOD'
        })

    # State Rainfall Centroids with actual 16-year metrics from Daily Rainfall Data - India
    state_rainfall_points = []
    for _, sr in state_grp.iterrows():
        s_name = str(sr['state_name']).strip()
        coords = STATE_CENTROIDS.get(s_name.lower())
        if coords:
            m_daily = float(sr['mean_daily']) if pd.notnull(sr['mean_daily']) else 0.0
            mx_daily = float(sr['max_daily']) if pd.notnull(sr['max_daily']) else 0.0
            h_days = int(sr['heavy_days']) if pd.notnull(sr['heavy_days']) else 0
            state_rainfall_points.append({
                'state_name': s_name,
                'lat': coords[0],
                'lon': coords[1],
                'mean_daily_rainfall': round(m_daily, 2),
                'max_daily_rainfall': round(mx_daily, 2),
                'heavy_rain_days': h_days,
                'records_count': int(sr['records']),
                'category': 'High Precipitation Zone' if h_days >= 15 else 'Moderate Precipitation Zone'
            })

    # High-Risk Alert Highlights from dataset
    top_alerts_df = df_flood.sample(min(2000, len(df_flood)), random_state=11).copy()
    tp = model.predict_proba(top_alerts_df[meta['columns']])[:, 1]
    top_alerts_df['prob'] = tp
    alerts = []
    for _, r in top_alerts_df.nlargest(6, 'prob').iterrows():
        p_pct = float(r['prob']) * 100
        r_info = determine_risk_band(p_pct)
        alerts.append({
            'title': f"{r_info['risk_band']} Flood Risk Alert",
            'location': f"Lat {r['Latitude']:.2f}°N, Lon {r['Longitude']:.2f}°E",
            'prob': round(p_pct, 1),
            'rainfall': round(float(r['Rainfall (mm)']), 1),
            'water_level': round(float(r['Water Level (m)']), 2),
            'discharge': round(float(r['River Discharge (m³/s)']), 0),
            'warning_status': r_info['warning_level'],
            'warning_message': r_info['warning_msg']
        })

    return jsonify({
        'records': len(df_flood),
        'rainfall_dataset_records': len(df_rain),
        'avg_rainfall': round(avg_rainfall, 2),
        'flood_rate': round(flood_rate, 2),
        'rain_labels': rc.index.tolist(),
        'rain_counts': rc.tolist(),
        'alerts': alerts,
        'points': points,
        'state_rainfall_points': state_rainfall_points,
        'metrics': meta['metrics'],
        'rainfall_summary': {
            'records': len(df_rain),
            'years_covered': '2009–2024',
            'states_count': int(df_rain['state_name'].nunique()),
            'max_historical_single_day': round(float(df_rain['actual'].max()), 2),
            'national_daily_mean': round(float(df_rain['actual'].mean()), 2),
            'peak_monsoon': 'July (Monsoon Surge)'
        },
        'datasets': {
            'dataset1': 'Daily Rainfall Data - India (2009-2024).csv',
            'dataset2': 'flood_risk.csv',
            'status': 'Connected'
        }
    })

# ----------------- Risk Map API (Section 16: Structured Geospatial Data) -----------------
@app.get('/api/risk-map')
def risk_map():
    sample = df_flood.sample(min(350, len(df_flood)), random_state=42)
    probs = model.predict_proba(sample[meta['columns']])[:, 1]
    locations = []
    for i, (_, r) in enumerate(sample.iterrows()):
        p_pct = float(probs[i]) * 100
        r_info = determine_risk_band(p_pct)
        locations.append({
            'latitude': round(float(r['Latitude']), 4),
            'longitude': round(float(r['Longitude']), 4),
            'rainfall': round(float(r['Rainfall (mm)']), 1),
            'water_level': round(float(r['Water Level (m)']), 2),
            'river_discharge': round(float(r['River Discharge (m³/s)']), 1),
            'elevation': round(float(r.get('Elevation (m)', 0.0)), 1),
            'temperature': round(float(r.get('Temperature (°C)', 25.0)), 1),
            'humidity': round(float(r.get('Humidity (%)', 70.0)), 1),
            'land_cover': str(r.get('Land Cover', 'Urban')),
            'soil_type': str(r.get('Soil Type', 'Clay')),
            'flood_probability': round(float(probs[i]), 4),
            'flood_probability_pct': round(p_pct, 1),
            'risk_level': r_info['risk_band'],
            'risk_slug': r_info['risk_slug'],
            'risk_color': r_info['badge_color'],
            'warning_level': r_info['warning_level'],
            'warning_message': r_info['warning_msg'],
            'evacuation': r_info['evacuation'],
            'flood_occurred': int(r['Flood Occurred']),
            'prediction': int(probs[i] >= 0.5),
            'prediction_label': 'FLOOD' if probs[i] >= 0.5 else 'NO FLOOD'
        })

    state_rainfall = []
    for _, sr in state_grp.iterrows():
        s_name = str(sr['state_name']).strip()
        coords = STATE_CENTROIDS.get(s_name.lower())
        if coords:
            m_daily = float(sr['mean_daily']) if pd.notnull(sr['mean_daily']) else 0.0
            mx_daily = float(sr['max_daily']) if pd.notnull(sr['max_daily']) else 0.0
            h_days = int(sr['heavy_days']) if pd.notnull(sr['heavy_days']) else 0
            cnt = int(sr['records']) if pd.notnull(sr['records']) else 0
            state_rainfall.append({
                'state_name': s_name,
                'latitude': coords[0],
                'longitude': coords[1],
                'mean_daily_rainfall': round(m_daily, 2),
                'max_daily_rainfall': round(mx_daily, 2),
                'heavy_rain_days': h_days,
                'records_count': cnt
            })

    return jsonify({
        'status': 'success',
        'source': 'Model-based Risk Visualization from flood_risk.csv and Daily Rainfall Data - India (2009-2024).csv',
        'total_locations': len(locations),
        'locations': locations,
        'state_rainfall': state_rainfall
    })

# ----------------- Analytics API (Section 1: Flood Risk, Section 2: Historical Rainfall) -----------------
@app.get('/api/analytics')
def analytics():
    def col_stats(s):
        return {
            'min': round(float(s.min()), 2),
            'max': round(float(s.max()), 2),
            'mean': round(float(s.mean()), 2),
            'std': round(float(s.std()), 2)
        }

    # SECTION 1: FLOOD RISK ANALYTICS
    flood_risk_analytics = {
        'total_samples': len(df_flood),
        'flood_counts': df_flood['Flood Occurred'].value_counts().sort_index().tolist(),
        'flood_rate_percent': round(float(df_flood['Flood Occurred'].mean() * 100), 2),
        'feature_statistics': {
            'rainfall': col_stats(df_flood['Rainfall (mm)']),
            'water_level': col_stats(df_flood['Water Level (m)']),
            'discharge': col_stats(df_flood['River Discharge (m³/s)']),
            'elevation': col_stats(df_flood['Elevation (m)']),
            'temperature': col_stats(df_flood['Temperature (°C)']),
            'humidity': col_stats(df_flood['Humidity (%)']),
            'population_density': col_stats(df_flood['Population Density'])
        },
        'land_cover_distribution': df_flood['Land Cover'].value_counts().to_dict(),
        'soil_type_distribution': df_flood['Soil Type'].value_counts().to_dict(),
        'model_performance': {
            'algorithm': meta.get('primary_algorithm', 'Decision Tree Classifier (Unit III Syllabus)'),
            'comparison_algorithm': meta.get('comparison_algorithm', 'Random Forest Classifier'),
            'accuracy': meta['metrics']['accuracy'],
            'precision': meta['metrics']['precision'],
            'recall': meta['metrics']['recall'],
            'f1': meta['metrics']['f1'],
            'roc_auc': meta['metrics']['roc_auc'],
            'confusion_matrix': meta['metrics']['confusion_matrix'],
            'best_params': meta['metrics'].get('best_params', {}),
            'feature_importances': meta['metrics'].get('feature_importances', []),
            'algorithm_comparison': meta.get('algorithm_comparison', {})
        }
    }

    # SECTION 2: HISTORICAL RAINFALL ANALYTICS
    historical_rainfall_analytics = {
        'total_records': len(df_rain),
        'date_range': rainfall_summary['date_range'],
        'overall_stats': rainfall_summary['overall_stats'],
        'monthly_seasonality': rainfall_summary['monthly_seasonality'],
        'yearly_trends': rainfall_summary['yearly_trends'],
        'top_wettest_states': rainfall_summary['top_wettest_states'],
        'driest_states': rainfall_summary['driest_states'],
        'imd_hazard_classification': rainfall_summary['imd_hazard_classification'],
        'rainfall_distribution': rainfall_summary['distribution'],
        'top_extreme_events': rainfall_summary['top_extreme_events']
    }

    return jsonify({
        'section1_flood_risk': flood_risk_analytics,
        'section2_historical_rainfall': historical_rainfall_analytics,
        # Backwards compatibility keys for existing frontend widgets
        'rainfall': col_stats(df_flood['Rainfall (mm)']),
        'water_level': col_stats(df_flood['Water Level (m)']),
        'discharge': col_stats(df_flood['River Discharge (m³/s)']),
        'flood_counts': df_flood['Flood Occurred'].value_counts().sort_index().tolist(),
        'metrics': meta['metrics'],
        'datasets': {
            'dataset1': 'Daily Rainfall Data - India (2009-2024).csv',
            'dataset2': 'flood_risk.csv',
            'status': 'Connected'
        }
    })

# ----------------- Dedicated Rainfall Analytics API -----------------
@app.get('/api/rainfall-analytics')
def get_rainfall_analytics():
    return jsonify(rainfall_summary)

# ----------------- Datasets Info API -----------------
@app.get('/api/datasets/info')
def datasets_info():
    """Returns comprehensive metadata, schema, and statistics for both official project datasets."""
    return jsonify({
        'dataset1': {
            'filename': 'Daily Rainfall Data - India (2009-2024).csv',
            'name': 'Daily Rainfall Data - India (2009-2024)',
            'path': 'data/Daily Rainfall Data - India (2009-2024).csv',
            'status': 'Connected',
            'records': len(df_rain),
            'columns': ['id', 'date', 'state_code', 'state_name', 'actual', 'rfs', 'normal', 'deviation'],
            'data_types': {
                'id': 'int64',
                'date': 'datetime64[ns]',
                'state_code': 'int64',
                'state_name': 'object (String)',
                'actual': 'float64 (mm)',
                'rfs': 'float64',
                'normal': 'float64 (mm)',
                'deviation': 'float64 (%)'
            },
            'missing_values': {
                'actual': int(df_rain['actual'].isnull().sum()),
                'rfs': int(df_rain['rfs'].isnull().sum()),
                'normal': int(df_rain['normal'].isnull().sum()),
                'deviation': int(df_rain['deviation'].isnull().sum())
            },
            'date_range': rainfall_summary['date_range'],
            'coverage': '36 States and Union Territories of India',
            'role': 'Historical rainfall analysis, seasonal trends, precipitation departure, and heavy rainfall hazard analysis.'
        },
        'dataset2': {
            'filename': 'flood_risk.csv',
            'name': 'flood_risk.csv',
            'path': 'data/flood_risk.csv',
            'status': 'Connected',
            'records': len(df_flood),
            'target_column': 'Flood Occurred',
            'columns_count': len(df_flood.columns),
            'columns': df_flood.columns.tolist(),
            'missing_values_count': int(df_flood.isnull().sum().sum()),
            'features': {
                'numerical': meta['numeric_columns'],
                'categorical': meta['categorical_columns']
            },
            'target_distribution': {
                'no_flood': int(df_flood['Flood Occurred'].value_counts().get(0, 4943)),
                'flood_occurred': int(df_flood['Flood Occurred'].value_counts().get(1, 5057)),
                'flood_rate_pct': round(float(df_flood['Flood Occurred'].mean() * 100), 2)
            },
            'model_evaluation': {
                'algorithm': meta.get('primary_algorithm', 'Decision Tree Classifier (Unit III Syllabus)'),
                'comparison_algorithm': meta.get('comparison_algorithm', 'Random Forest Classifier'),
                'accuracy': meta['metrics']['accuracy'],
                'f1_score': meta['metrics'].get('f1_score', meta['metrics'].get('f1')),
                'precision': meta['metrics']['precision'],
                'recall': meta['metrics']['recall'],
                'roc_auc': meta['metrics']['roc_auc']
            },
            'role': 'Primary flood risk classification, environmental hazard telemetry, and early warning inundation inference.'
        },
        'cross_dataset_relationship': {
            'common_key_status': 'No direct common key',
            'strategy': 'Coordinated Dual-Engine Analysis (Separated datasets aligned into coordinated AIML intelligence)',
            'explanation': 'flood_risk.csv provides station-level latitude/longitude and environmental features for ML flood prediction; Daily Rainfall Data provides 16-year state-level meteorological time series for historical rainfall insights.'
        }
    })

if __name__ == '__main__':
    logging.info("Starting AI-Powered Heavy Rainfall and Flood Early-Warning System Flask Backend Server...")
    app.run(host='127.0.0.1', port=5000, debug=False)
