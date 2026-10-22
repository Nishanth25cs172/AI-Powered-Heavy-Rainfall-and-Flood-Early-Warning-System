# Model Card: Module B — Flood Inundation Early Warning Classifier

## Model Details
- **Model Name:** Flood Inundation Early Warning Classifier
- **Model Architecture:** Decision Tree Classifier / Ensemble Pipeline
- **Version:** 2.0.0
- **Primary Task:** Binary Classification (`Flood Occurred`: 1 = Yes, 0 = No)
- **Framework:** scikit-learn (StandardScaler + OneHotEncoder + DecisionTreeClassifier)

## Intended Use
- **Primary Use Case:** Automated hydrological early-warning decision support for flood inundation risk assessment.
- **Intended Users:** Disaster management personnel, municipal water management teams, and emergency responders.
- **Out of Scope:** Structural engineering floodwall failure modeling or direct dam breach hydrodynamic flow dynamics.

## Data Details
- **Training Dataset:** `data/raw/flood_risk.csv` (10,000 station observations)
- **Features Used:**
  - **Numerical:** `Latitude`, `Longitude`, `Rainfall (mm)`, `River Discharge (m³/s)`, `Water Level (m)`, `Elevation (m)`
  - **Categorical:** `Land Cover`, `Soil Type`
- **Validation Strategy:** Stratified 5-Fold Cross-Validation & 80/20 Stratified Holdout Split.

## Evaluation Metrics
- **Recall (Primary Early-Warning Metric):** Prioritized to minimize false negatives (missed inundations).
- **F1-Score:** Harmonic mean balancing precision and recall.
- **ROC-AUC & Accuracy:** Verified on holdout test set.

## Limitations & Risks
- **Data Scope:** Dependent on station-level telemetry and terrain characteristics.
- **Fairness & Bias:** Performance varies based on sensor telemetry coverage across rural vs. urban catchments.
