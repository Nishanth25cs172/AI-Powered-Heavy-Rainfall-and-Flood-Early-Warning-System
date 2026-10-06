/**
 * AI-Powered Heavy Rainfall and Flood Early-Warning System - Configuration & Constants
 * SIH26071 AI/ML-Based Integrated Decision Support & Early Warning System
 */

const CONFIG = {
  appName: 'AI-Powered Heavy Rainfall and Flood Early-Warning System',
  subtitle: 'AI/ML-Based Integrated Decision Support & Early Warning System',
  datasetName: 'Daily Rainfall Data - India (2009-2024).csv',
  floodDatasetName: 'flood_risk.csv',
  modelName: 'Logistic Regression & Decision Tree (Unit III Syllabus)',
  dataStatus: '● Datasets Connected',

  // Exactly TWO Official Project Datasets
  DATASETS: {
    rainfall: {
      id: 'rainfall_india',
      label: 'Rainfall Dataset',
      name: 'Daily Rainfall Data - India (2009-2024).csv',
      timeframe: '2009–2024',
      status: '● Connected',
      purpose: 'Historical Rainfall Analysis & Multi-Year Trends',
      recordsCount: '204,876 Daily Records (2009–2024)',
      icon: '🌧️'
    },
    flood_risk: {
      id: 'flood_risk_india',
      label: 'Flood Risk Dataset',
      name: 'flood_risk.csv',
      timeframe: '10,000 Station Observations across India',
      status: '● Connected',
      purpose: 'AI Flood Risk Prediction & Inundation Modeling',
      recordsCount: '10,000 Observations',
      icon: '🌊'
    }
  },
  
  // Backend REST API Endpoints (Auto-detects Flask port 5000 vs Live Server port 5500)
  API: (() => {
    const isDirectFlask = (typeof window !== 'undefined' && window.location && window.location.port === '5000');
    const base = isDirectFlask ? '/api' : 'http://127.0.0.1:5000/api';
    return {
      BASE: base,
      HEALTH: `${base}/health`,
      CONFIG: `${base}/config`,
      LOGIN: `${base}/login`,
      PREDICT: `${base}/predict`,
      PREDICT_FLOOD: `${base}/predict-flood`,
      PREDICT_RAINFALL: `${base}/predict-rainfall`,
      PREDICT_RISK: `${base}/predict-risk`,
      RAINFALL_PREDICTION_CONFIG: `${base}/rainfall-prediction-config`,
      DASHBOARD: `${base}/dashboard`,
      ANALYTICS: `${base}/analytics`,
      DATASETS_INFO: `${base}/datasets/info`,
      RAINFALL_ANALYTICS: `${base}/rainfall-analytics`,
      RISK_MAP: `${base}/risk-map`
    };
  })(),

  // Polling / Auto-refresh interval (milliseconds)
  REFRESH_INTERVAL_MS: 30000,

  // Risk Classification Bands
  RISK_LEVELS: {
    LOW: { label: 'LOW', slug: 'low', color: '#10b981', glow: 'rgba(16, 185, 129, 0.4)', min: 0, max: 25 },
    MODERATE: { label: 'MODERATE', slug: 'moderate', color: '#f59e0b', glow: 'rgba(245, 158, 11, 0.4)', min: 25, max: 50 },
    HIGH: { label: 'HIGH', slug: 'high', color: '#f97316', glow: 'rgba(249, 115, 22, 0.4)', min: 50, max: 75 },
    CRITICAL: { label: 'CRITICAL', slug: 'critical', color: '#ef4444', glow: 'rgba(239, 68, 68, 0.5)', min: 75, max: 100 }
  },

  // Disaster Types Monitored
  DISASTER_TYPES: [
    { id: 'flood', name: 'Flood & Inundation', icon: '🌧️', baseSeverity: 'HIGH', metricKey: 'Water Level (m)' },
    { id: 'storm', name: 'Severe Storm', icon: '⛈️', baseSeverity: 'HIGH', metricKey: 'Rainfall (mm)' },
    { id: 'cyclone', name: 'Tropical Cyclone', icon: '🌪️', baseSeverity: 'MODERATE', metricKey: 'Wind Speed' },
    { id: 'heat', name: 'Extreme Heat', icon: '🔥', baseSeverity: 'LOW', metricKey: 'Temperature (°C)' },
    { id: 'heavy_rain', name: 'Heavy Rainfall', icon: '🌊', baseSeverity: 'CRITICAL', metricKey: 'Rainfall (mm)' },
    { id: 'lightning', name: 'Lightning Storm', icon: '⚡', baseSeverity: 'HIGH', metricKey: 'Humidity (%)' }
  ],

  // Emergency Hotlines & Relief Contact Directory
  EMERGENCY_CONTACTS: [
    { title: 'National Disaster Response Force (NDRF)', number: '1078 / 011-24363260', type: 'Rescue & Evacuation', status: '24x7 Active', badge: 'National' },
    { title: 'State Disaster Management Authority (SDMA)', number: '1070', type: 'State Coordination Center', status: '24x7 Active', badge: 'State' },
    { title: 'District Disaster Management Cell (DDMA)', number: '1077', type: 'Local Response Unit', status: '24x7 Active', badge: 'District' },
    { title: 'Emergency Medical & Ambulance', number: '108 / 112', type: 'Medical Response', status: '24x7 Active', badge: 'Medical' },
    { title: 'Central Water Commission (CWC) Flood Cell', number: '1800-180-1551', type: 'Hydrological Warning', status: 'Active', badge: 'Hydrology' }
  ],

  // Prediction Scenarios Presets for Live Demonstrations
  PRESETS: {
    monsoon_deluge: {
      name: '🚨 Monsoon Deluge & Severe Inundation',
      data: {
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
    },
    flash_flood: {
      name: '⚠️ Flash Flood & Cloudburst Alert',
      data: {
        'Latitude': 30.31,
        'Longitude': 78.03,
        'Rainfall (mm)': 225.0,
        'Temperature (°C)': 22.4,
        'Humidity (%)': 91.0,
        'River Discharge (m³/s)': 3300.0,
        'Water Level (m)': 7.45,
        'Elevation (m)': 450.0,
        'Land Cover': 'Agricultural',
        'Soil Type': 'Loam',
        'Population Density': 3200.0,
        'Infrastructure': 1.0,
        'Historical Floods': 1.0
      }
    },
    moderate_risk: {
      name: '⚡ Moderate Monsoon Surge',
      data: {
        'Latitude': 22.57,
        'Longitude': 88.36,
        'Rainfall (mm)': 142.0,
        'Temperature (°C)': 29.5,
        'Humidity (%)': 78.0,
        'River Discharge (m³/s)': 2100.0,
        'Water Level (m)': 4.80,
        'Elevation (m)': 12.0,
        'Land Cover': 'Urban',
        'Soil Type': 'Silt',
        'Population Density': 8200.0,
        'Infrastructure': 1.0,
        'Historical Floods': 0.0
      }
    },
    low_risk: {
      name: '☀️ Normal Clear Weather (Low Risk)',
      data: {
        'Latitude': 26.91,
        'Longitude': 75.78,
        'Rainfall (mm)': 12.5,
        'Temperature (°C)': 34.2,
        'Humidity (%)': 35.0,
        'River Discharge (m³/s)': 350.0,
        'Water Level (m)': 1.20,
        'Elevation (m)': 430.0,
        'Land Cover': 'Desert',
        'Soil Type': 'Sandy',
        'Population Density': 2100.0,
        'Infrastructure': 0.0,
        'Historical Floods': 0.0
      }
    }
  },

  // Presets for AI Rainfall Prediction Demonstrations
  RAINFALL_PRESETS: {
    monsoon_deluge: {
      name: '🌧️ Heavy Monsoon Deluge (Maharashtra)',
      data: {
        state_name: 'Maharashtra',
        date: '2024-07-22',
        normal: 19.8,
        recent_rainfall: 65.0
      }
    },
    coastal_surge: {
      name: '🌊 Coastal Active Monsoon (Kerala)',
      data: {
        state_name: 'Kerala',
        date: '2024-08-14',
        normal: 24.5,
        recent_rainfall: 82.0
      }
    },
    northeast_monsoon: {
      name: '⚡ Brahmaputra Basin Surge (Assam)',
      data: {
        state_name: 'Assam',
        date: '2024-06-28',
        normal: 18.2,
        recent_rainfall: 45.0
      }
    },
    himalayan_alert: {
      name: '⚠️ Cloudburst Alert (Uttarakhand)',
      data: {
        state_name: 'Uttarakhand',
        date: '2024-08-08',
        normal: 16.5,
        recent_rainfall: 72.0
      }
    },
    dry_winter: {
      name: '☀️ Arid Dry Winter (Rajasthan)',
      data: {
        state_name: 'Rajasthan',
        date: '2024-01-15',
        normal: 0.2,
        recent_rainfall: 0.0
      }
    }
  }
};

// Export to window
window.CONFIG = CONFIG;
