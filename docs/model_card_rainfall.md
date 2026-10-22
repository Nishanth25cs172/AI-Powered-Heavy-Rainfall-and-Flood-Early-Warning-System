# Model Card: Module A — Heavy Rainfall Weather Classifier

## Model Details
- **Model Name:** Heavy Rainfall Weather Prediction Classifier
- **Model Architecture:** Gradient Boosting Classifier / Random Forest Pipeline
- **Version:** 2.0.0
- **Primary Task:** Binary Classification (`Rain_Target`: 1 = Precipitation Expected, 0 = No Precipitation)
- **Framework:** scikit-learn (SimpleImputer + StandardScaler + GradientBoostingClassifier)

## Intended Use
- **Primary Use Case:** Meteorological heavy rainfall risk detection and precipitation storm early warning.
- **IMD Criterion:** Detection of heavy rainfall events aligned with India Meteorological Department (IMD) standards ($\ge 64.5\text{ mm/day}$).
- **Out of Scope:** Micro-scale convective cloud physics or numerical weather prediction (NWP) atmospheric physics modeling.

## Data Details
- **Training Dataset:** `data/raw/Indian Rainfall and Weather Prediction Dataset.xlsx` (10,000 meteorological records spanning 2009–2024)
- **Features Used:** `avg_temp`, `wind_speed`, `air_pressure`, `latitude`, `longitude`, `elevation`
- **Validation Strategy:** `TimeSeriesSplit` 5-Fold Cross-Validation & Chronological Time-Based Holdout Split (2009–2020 Train | 2021–2024 Test).

## Evaluation Metrics
- **Recall (Primary Early-Warning Metric):** High priority on capturing precipitation events to minimize missed storm alerts.
- **F1-Score & ROC-AUC:** Evaluated across TimeSeriesSplit cross-validation folds.

## Limitations & Risks
- **Temporal Extrapolation:** Meteorological patterns shift over decadal climate cycles; models should be periodically re-fit on recent observation windows.
