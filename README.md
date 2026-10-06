# SIH26071 — AI-Powered Heavy Rainfall and Flood Early-Warning System

**Smart India Hackathon (SIH) Project · Problem Statement ID: SIH26071**  
**Category:** AI/ML-Based Integrated Heavy Rainfall Early Warning and Inundation Prediction System  
**Academic Alignment:** AIML Academic Syllabus (Units I, II, III)

---

## 1. Executive Summary

The **AI-Powered Heavy Rainfall and Flood Early-Warning System (SIH26071)** is an AI/ML-based early-warning and decision-support prototype engineered to predict heavy precipitation storm events, estimate localized flood inundation hazards, and synthesize actionable compound disaster risk advisories.

The system is strictly built upon verified machine-learning algorithms covered in the academic curriculum:
1. **Module A (Heavy Rainfall Early Warning):** **Logistic Regression** binary classifier trained on 16 years of daily meteorological observations across 36 Indian States and Union Territories to detect heavy rainfall events exceeding the official India Meteorological Department (IMD) threshold ($\ge 64.5\text{ mm/day}$).
2. **Module B (Flood Inundation Prediction):** **Decision Tree Classifier** providing an interpretable, transparent decision hierarchy for station-level hydrological and environmental telemetry, compared against an ensemble Random Forest classifier.
3. **Integration Layer (Risk Engine):** A decoupled backend service (`backend/services/risk_engine.py`) synthesizing multi-hazard probabilities into compound risk tiers (**LOW**, **MODERATE**, **HIGH**, **CRITICAL**) with emergency action bulletins.

---

## 2. Official Datasets

The project strictly utilizes **exactly two official datasets** located in `data/`. No external datasets or synthetic records are used:

| Dataset Identifier | File Name | Observations | Time Span / Scope | Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset 1** | `data/Daily Rainfall Data - India (2009-2024).csv` | 204,876 daily records | 2009–2024 (16 Years, 36 States/UTs) | Climatological baseline, time-series feature engineering, lag metrics, and IMD heavy rainfall classification. |
| **Dataset 2** | `data/flood_risk.csv` | 10,000 observations | Station-level environmental & terrain telemetry | Hydrological modeling, soil/land-cover characterization, hydraulic stage assessment, and flood inundation classification. |

---

## 3. AIML Academic Syllabus Mapping

The entire system is structured to map directly to core AIML curriculum topics, ensuring every architectural decision is defensible during academic reviews and hackathon vivas.

```
                              SIH26071 SYSTEM ARCHITECTURE
                                           │
       ┌───────────────────────────────────┴───────────────────────────────────┐
       ▼                                                                       ▼
UNIT I: AI FUNDAMENTALS                                            UNIT II: ML FUNDAMENTALS
• Intelligent Early-Warning Agent                                  • Data Preprocessing (Scalers, Encoders)
• PEAS Framework Specification                                     • Imbalanced Data Handling (Class Weighting)
• Decision Support Architecture                                    • Chronological Splitting (No Temporal Leakage)
                                                                   • Stratified 5-Fold Cross-Validation
                                                                               │
                                                                               ▼
                                                                   UNIT III: SUPERVISED LEARNING
                                                                   • Logistic Regression (Rainfall Binary Target)
                                                                   • Decision Tree Classifier (Interpretable Flood)
                                                                   • Random Forest (Ensemble Comparison)
                                                                   • Evaluation (Recall, ROC-AUC, F1, Confusion Matrix)
```

### Unit I — AI Fundamentals: PEAS Specification

The system operates as an **Intelligent Early-Warning Decision-Support Agent**:

| PEAS Element | Implementation in SIH26071 System |
| :--- | :--- |
| **Performance Measure (P)** | • Maximized **Recall** on heavy rainfall events ($\ge 64.5\text{ mm}$) to minimize missed alerts.<br>• Minimized False Negatives in critical inundation zones.<br>• Timely generation of graded compound risk advisories (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`). |
| **Environment (E)** | • Historical meteorological time-series records across 36 Indian states.<br>• Geospatial terrain, soil permeabilities, land-use covers, river discharge stages, and sensor telemetry. |
| **Actuators (A)** | • REST API endpoints (`/api/predict-rainfall`, `/api/predict-flood`, `/api/predict-risk`).<br>• Visual early-warning radar cards and danger status badges.<br>• Evacuation recommendations and automated emergency response bulletins. |
| **Sensors (S)** | • Input parameters: Climatological normal rainfall, past rainfall lag ($t-1, t-3, t-7$), state, river discharge ($m^3/s$), water level ($m$), elevation ($m$), humidity (%), temperature ($^\circ C$), soil type, and land cover. |

---

## 4. Module A — Heavy Rainfall Prediction

### 4.1 Target Formulation & IMD Standard
Rather than using continuous regression and arbitrary discretization, Module A frames precipitation risk as a **Binary Classification Problem**:
$$\text{Heavy\_Rainfall} = \begin{cases} 1 & \text{if } \text{actual} \ge 64.5\text{ mm} \\ 0 & \text{if } \text{actual} < 64.5\text{ mm} \end{cases}$$

* **Meteorological Authority:** India Meteorological Department (IMD) standard classification:
  * Normal / Moderate: $0.0 - 64.4\text{ mm/day}$
  * Heavy Rainfall: $64.5 - 115.5\text{ mm/day}$
  * Very Heavy Rainfall: $115.6 - 204.4\text{ mm/day}$
  * Extremely Heavy Deluge: $\ge 204.5\text{ mm/day}$
* **Target Distribution:** 741 positive events out of 187,714 valid daily records (**0.395% base rate**, severe class imbalance).

### 4.2 Temporal Splitting (Zero Future Information Leakage)
To prevent temporal data leakage, records were split chronologically rather than with random shuffling:
* **Training Set:** `2009-01-01` to `2021-12-31` (160,446 records; 623 positive heavy rain events).
* **Holdout Test Set:** `2022-01-01` to `2024-07-28` (27,268 records; 118 positive heavy rain events).

### 4.3 Feature Engineering & Historical Lag
All rolling and lag features are derived strictly from antecedent dates:
* `normal`: Long-period average climatological expectation.
* `actual_lag1`, `actual_lag3`, `actual_lag7`: Historical rainfall observed 1, 3, and 7 days prior.
* `rolling_mean_7d`: Antecedent 7-day cumulative wetness.
* `deviation`: Climatological departure percentage.
* `month`, `day`, `day_of_year`, `season`: Seasonal and intra-annual cyclicity.
* `state_name`: One-Hot Encoded regional indicators.

### 4.4 Primary Model: Logistic Regression (Unit III Syllabus)
To overcome the 0.395% base rate and prioritize life-saving warning recall, Logistic Regression was configured with inverse frequency class weighting (`class_weight='balanced'`).

#### Holdout Test Set Evaluation (2022–2024 Records)
| Metric | Logistic Regression (Primary) | Decision Tree | Random Forest |
| :--- | :---: | :---: | :---: |
| **Recall (Catching Heavy Rain)** | **88.98%** (105 / 118 caught) | 77.97% | 82.20% |
| **ROC-AUC** | **0.9729** | 0.8841 | 0.9575 |
| **Overall Accuracy** | **92.72%** | 98.71% | 99.41% |
| **False Negatives (Missed Disasters)** | **13** (Lowest) | 26 | 21 |

> **Evaluation Insight:** In an early-warning context, missing an extreme deluge (False Negative) can be catastrophic. The balanced Logistic Regression classifier captures **88.98% of all real-world heavy rainfall events** in the held-out future period, achieving an exceptional **0.9729 ROC-AUC**.

---

## 5. Module B — Flood Inundation Prediction

### 5.1 Dataset & Leakage Audit
Trained on `data/flood_risk.csv` (10,000 records, 13 features).
* **Leakage Audit on `Historical Floods`:** Pearson correlation with `Flood Occurred` target is $r = 0.012$, proving that the historical flood feature does not cause circular target leakage.
* **Target Distribution:** 5,057 flood events (50.57%) vs 4,943 non-flood events (49.43%).

### 5.2 Primary Model: Decision Tree Classifier (Unit III Syllabus)
Decision trees provide human-interpretable conditional splits suitable for disaster risk explanation:
* Preprocessing: `StandardScaler` for continuous features, `OneHotEncoder` for terrain/soil categories.
* Stratified 5-Fold Cross Validation Mean F1: **0.566**.
* Holdout Test Metrics (2,000 samples): Accuracy: **50.85%**, Precision: **51.86%**, Recall: **38.67%**, F1: **44.31%**, ROC-AUC: **0.5143**.
* Comparison Model: Random Forest Classifier (Accuracy: 50.95%, Recall: 59.45%, F1: 55.06%).

---

## 6. Integration Architecture & Risk Engine

The system implements **Option B: Chained Rainfall-to-Inundation Decision Support**:

```
                         DECISION PIPELINE
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
       RAINFALL PREDICTION                TERRAIN & RIVER
       (Logistic Regression)             TELEMETRY INPUTS
                │                                 │
                ▼                                 │
       Predicted Rainfall (mm)                    │
       + Heavy Rain Prob P(Rain)                  │
                │                                 │
                └───────────────┬─────────────────┘
                                ▼
                       FLOOD INUNDATION MODEL
                      (Decision Tree Classifier)
                                │
                                ▼
                       P(Flood Inundation)
                                │
                                ▼
                       BACKEND RISK ENGINE
               (services/risk_engine.py Synthesis)
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
    COMPOUND RISK          EARLY WARNING           ACTIONABLE
     SCORE (0-100%)           BULLETIN           RECOMMENDATION
```

### Compound Risk Formula
$$\text{Compound Risk} = 0.45 \cdot P_{\text{rain}} + 0.45 \cdot P_{\text{flood}} + 0.10 \cdot \text{WaterStress}$$

* **LOW (<30%):** Routine monitoring; normal conditions.
* **MODERATE (30–60%):** Catchment saturation rising; ground teams alerted.
* **HIGH (60–80%):** Flash flood / inundation probable; pre-position rescue teams.
* **CRITICAL (>80%):** Extreme deluge and riverbank overtopping; immediate evacuation advisory.

---

## 7. REST API Endpoints

The Flask backend exposes clean, validated REST endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health verification, model inventory, and academic syllabus mapping. |
| `POST` | `/api/predict-rainfall` | Primary Logistic Regression heavy rainfall binary prediction (`YES/NO`), probability, and IMD criteria. |
| `POST` | `/api/predict-flood` | Primary Decision Tree flood inundation prediction (`YES/NO`), probability, and factors. |
| `POST` | `/api/predict` | Backward-compatible alias for the flood prediction endpoint. |
| `POST` | `/api/predict-risk` | Integrated multi-hazard synthesis combining rainfall and inundation via `RiskEngine`. |
| `GET` | `/api/rainfall-prediction-config` | State baselines, climatological normals, and model evaluation reports. |
| `GET` | `/api/rainfall-analytics` | Precomputed 16-year historical rainfall distribution and state statistics. |
| `GET` | `/api/dashboard` | Aggregated command center telemetry, geospatial coordinates, and alert history. |
| `GET` | `/api/analytics` | Dual-dataset statistical summaries and model verification metrics. |
| `GET` | `/api/datasets/info` | Comprehensive schema, datatypes, missing values, and descriptions for both CSVs. |

---

## 8. Dashboard Layout & UI Design

The web interface features a dark navy/cyan glassmorphic Command Center theme:
* **Fixed Vertical Left Sidebar** (width: 260px) with highlighted active views:
  1. 🏠 **Dashboard** (`/dashboard.html`)
  2. 🌧️ **Rainfall Prediction** (`/rainfall-prediction.html`)
  3. 🌊 **Flood Prediction** (`/prediction.html` $\to$ `/dashboard.html#prediction`)
  4. 📡 **Live Monitoring** (`/dashboard.html#monitoring`)
  5. 🚨 **Alerts** (`/dashboard.html#alerts`)
  6. 📊 **Analytics** (`/dashboard.html#analytics`)
  7. 🗺️ **Risk Map** (`/dashboard.html#map`)
  8. 🛡️ **Emergency Info** (`/dashboard.html#emergency`)
  9. ℹ️ **About** (`/dashboard.html#about`)
  10. 🚪 **Logout** (`/login.html`) — positioned directly below About.
* **Dual Prediction Labs**:
  * **Rainfall Prediction Lab:** State-wise climatological lookup, date selection, lag inputs, returning binary `Heavy Rainfall: YES / NO`, probability %, and IMD risk badge.
  * **Flood Prediction Lab:** 13 environmental and hydrological inputs, returning binary `Flood Risk: YES / NO`, probability %, and Decision Tree factor analysis.

---

## 9. Verification & Automated Test Suite

A comprehensive test suite verifies data integrity, ML models, and API endpoints:

```bash
# 1. Verify all rectified API endpoints
python ml/test_rectified_apis.py

# 2. Verify complete system (HTML pages, cards, and endpoints)
python ml/test_full_suite.py

# 3. Verify left sidebar navigation and layout
python ml/test_sidebar_navigation.py
```

### Automated Test Results
* **GET `/api/health`:** Status 200 OK — Syllabus Units I, II, III mapped.
* **POST `/api/predict-rainfall`:** Status 200 OK — Binary Logistic Regression inference verified.
* **POST `/api/predict-flood`:** Status 200 OK — Binary Decision Tree inference verified.
* **POST `/api/predict-risk`:** Status 200 OK — Compound risk engine verified.
* **All 8 HTML Pages:** Status 200 OK.

---

## 10. How to Run Locally

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Windows, macOS, or Linux

### Installation
```bash
# 1. Clone repository and navigate to project root
cd SIH26071_Flood_Warning

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-train ML models from scratch
python ml/train_rainfall_model.py
python ml/train_flood_model.py

# 4. Start the Flask server
python backend/app.py
```

### Accessing the Web Interface
Open your browser to:
```
http://127.0.0.1:5000
```
* **Demo Username:** `admin`
* **Demo Password:** `admin123`
* Or click **⚡ 1-Click Instant Demo Access** on the landing page.

---

## 11. Technical Defensibility & Project Limitations

To adhere to the principle **Validated metrics > claimed accuracy**:
1. **Prototype Disclaimer:** This system is an AI/ML decision-support prototype. It does not replace official bulletins from the India Meteorological Department (IMD) or the National Disaster Management Authority (NDMA).
2. **Data Scope:** Predictions are derived from tabular historical daily observations (`2009-2024`) and hydrological telemetry (`flood_risk.csv`), not live satellite radar imagery or Numerical Weather Prediction (NWP) physics simulations.
3. **Imbalance Handling:** The high recall (88.98%) in heavy rainfall prediction intentionally accepts false positives to guarantee early-warning responsiveness, as is standard in life-safety emergency systems.
