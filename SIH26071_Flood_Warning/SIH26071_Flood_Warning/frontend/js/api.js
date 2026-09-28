/**
 * AI-Powered Heavy Rainfall and Flood Early-Warning System - Unified REST API Client
 * Connects frontend directly to the existing Flask Backend
 */

const DisasterAPI = {
  status: {
    isOnline: false,
    latencyMs: 0,
    lastChecked: null,
    subscribers: []
  },

  cache: {
    config: null,
    dashboard: null,
    analytics: null,
    lastPrediction: null
  },

  /**
   * Subscribe to connectivity status updates
   */
  onStatusChange(callback) {
    if (typeof callback === 'function') {
      this.status.subscribers.push(callback);
      callback(this.status);
    }
  },

  _notifyStatus() {
    this.status.subscribers.forEach(cb => {
      try { cb(this.status); } catch (e) { console.error(e); }
    });
  },

  /**
   * Wrapper for fetch with configurable timeout
   */
  async request(url, options = {}, timeoutMs = 8000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    const start = performance.now();

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...(options.headers || {})
        }
      });

      clearTimeout(timeoutId);
      const latency = Math.round(performance.now() - start);

      if (!response.ok) {
        let errMessage = `HTTP error ${response.status}`;
        try {
          const errData = await response.json();
          errMessage = errData.error || (errData.fields ? `Missing: ${errData.fields.join(', ')}` : errMessage);
        } catch (_) {}
        throw new Error(errMessage);
      }

      this.status.isOnline = true;
      this.status.latencyMs = latency;
      this.status.lastChecked = new Date();
      this._notifyStatus();

      return await response.json();
    } catch (err) {
      clearTimeout(timeoutId);
      this.status.isOnline = false;
      this.status.lastChecked = new Date();
      this._notifyStatus();
      throw err;
    }
  },

  /**
   * Ping / Check backend health
   */
  async ping() {
    try {
      const data = await this.request(CONFIG.API.CONFIG, { method: 'GET' }, 4000);
      this.cache.config = data;
      return true;
    } catch (e) {
      return false;
    }
  },

  /**
   * Fetch backend dataset configuration and feature metadata
   */
  async getConfig() {
    if (this.cache.config) return this.cache.config;
    try {
      const data = await this.request(CONFIG.API.CONFIG);
      this.cache.config = data;
      return data;
    } catch (err) {
      console.warn('API: Failed to fetch config, using fallback defaults', err);
      // Fallback fallback if offline
      return {
        columns: ['Latitude', 'Longitude', 'Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)', 'River Discharge (m³/s)', 'Water Level (m)', 'Elevation (m)', 'Land Cover', 'Soil Type', 'Population Density', 'Infrastructure', 'Historical Floods'],
        numeric_columns: ['Latitude', 'Longitude', 'Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)', 'River Discharge (m³/s)', 'Water Level (m)', 'Elevation (m)', 'Population Density', 'Infrastructure', 'Historical Floods'],
        categorical_columns: ['Land Cover', 'Soil Type'],
        categorical_options: {
          'Land Cover': ['Agricultural', 'Desert', 'Forest', 'Urban', 'Water Body'],
          'Soil Type': ['Clay', 'Loam', 'Peat', 'Sandy', 'Silt']
        },
        ranges: {
          'Latitude': { min: 8.0, max: 37.0 },
          'Longitude': { min: 68.0, max: 97.0 },
          'Rainfall (mm)': { min: 0.0, max: 300.0 },
          'Temperature (°C)': { min: 15.0, max: 45.0 },
          'Humidity (%)': { min: 20.0, max: 100.0 },
          'River Discharge (m³/s)': { min: 0.0, max: 5000.0 },
          'Water Level (m)': { min: 0.0, max: 10.0 },
          'Elevation (m)': { min: 1.0, max: 8848.0 },
          'Population Density': { min: 2.0, max: 10000.0 },
          'Infrastructure': { min: 0.0, max: 1.0 },
          'Historical Floods': { min: 0.0, max: 1.0 }
        },
        rows: 10000,
        target: 'Flood Occurred'
      };
    }
  },

  /**
   * Fetch Dashboard metrics, sampled map coordinates, and alerts
   */
  async getDashboard() {
    try {
      const data = await this.request(CONFIG.API.DASHBOARD);
      this.cache.dashboard = data;
      return data;
    } catch (err) {
      console.warn('API: Failed to fetch dashboard data', err);
      if (this.cache.dashboard) return this.cache.dashboard;
      throw err;
    }
  },

  /**
   * Fetch Analytics metrics, distributions, and stats
   */
  async getAnalytics() {
    try {
      const data = await this.request(CONFIG.API.ANALYTICS);
      this.cache.analytics = data;
      return data;
    } catch (err) {
      console.warn('API: Failed to fetch analytics data', err);
      if (this.cache.analytics) return this.cache.analytics;
      throw err;
    }
  },

  /**
   * Fetch complete specifications for both official datasets
   */
  async getDatasetsInfo() {
    try {
      return await this.request(CONFIG.API.DATASETS_INFO);
    } catch (err) {
      console.warn('API: Failed to fetch datasets info', err);
      return null;
    }
  },

  /**
   * Fetch precomputed multi-year aggregation from Dataset 1
   */
  async getRainfallAnalytics() {
    try {
      return await this.request(CONFIG.API.RAINFALL_ANALYTICS);
    } catch (err) {
      console.warn('API: Failed to fetch rainfall analytics', err);
      return null;
    }
  },

  /**
   * Submit 13 features to the ML Model for inference
   */
  async predictRisk(featureData) {
    return await this.predictFlood(featureData);
  },

  /**
   * Submit 13 Environmental Features to Primary Decision Tree Flood Model
   */
  async predictFlood(featureData) {
    const endpoint = (CONFIG.API.PREDICT_FLOOD) ? CONFIG.API.PREDICT_FLOOD : CONFIG.API.PREDICT;
    const response = await this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(featureData)
    });
    this.cache.lastPrediction = { ...response, input: featureData, timestamp: new Date().toISOString() };
    try {
      localStorage.setItem('sih_last_prediction', JSON.stringify(this.cache.lastPrediction));
    } catch (_) {}
    return response;
  },

  /**
   * Submit combined meteorology and hydrology to Risk Engine for compound early warning
   */
  async predictIntegratedRisk(combinedData) {
    const response = await this.request(CONFIG.API.PREDICT_RISK, {
      method: 'POST',
      body: JSON.stringify(combinedData)
    });
    return response;
  },

  /**
   * Fetch system health & syllabus compliance
   */
  async getHealth() {
    try {
      return await this.request(CONFIG.API.HEALTH);
    } catch (err) {
      return { status: 'offline' };
    }
  },

  /**
   * Submit State, Date & Normal Precipitation to the Rainfall ML Model
   */
  async predictRainfall(rainfallData) {
    const response = await this.request(CONFIG.API.PREDICT_RAINFALL, {
      method: 'POST',
      body: JSON.stringify(rainfallData)
    });
    this.cache.lastRainfallPrediction = { ...response, input: rainfallData, timestamp: new Date().toISOString() };
    try {
      localStorage.setItem('sih_last_rainfall_prediction', JSON.stringify(this.cache.lastRainfallPrediction));
    } catch (_) {}
    return response;
  },

  /**
   * Fetch Rainfall Prediction configuration & state normals
   */
  async getRainfallPredictionConfig() {
    try {
      return await this.request(CONFIG.API.RAINFALL_PREDICTION_CONFIG);
    } catch (err) {
      console.warn('API: Failed to fetch rainfall prediction config', err);
      return null;
    }
  },

  /**
   * Fetch structured risk map locations & state rainfall data (Section 16)
   */
  async getRiskMap() {
    try {
      return await this.request(CONFIG.API.RISK_MAP);
    } catch (err) {
      console.warn('API: Failed to fetch /api/risk-map, falling back to cached telemetry', err);
      return null;
    }
  },

  /**
   * Admin Authentication
   */
  async login(username, password) {
    return await this.request(CONFIG.API.LOGIN, {
      method: 'POST',
      body: JSON.stringify({ username, password })
    });
  }
};

window.DisasterAPI = DisasterAPI;
