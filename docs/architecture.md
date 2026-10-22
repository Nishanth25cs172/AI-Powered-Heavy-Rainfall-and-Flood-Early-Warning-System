# System Architecture & Technical Specifications

**Project:** AI-Powered Heavy Rainfall and Flood Early Warning System  
**System Architecture Specification**

---

## 1. Modular Architecture Overview

The system is designed as a decoupled, multi-tiered AI decision-support architecture:

```
+-------------------------------------------------------------------------+
|                           PRESENTATION LAYER                            |
|    Frontend Dashboard, Leaflet Risk Map, Rainfall & Flood Prediction Labs|
+-------------------------------------------------------------------------+
                                     |
                                REST APIs (JSON)
                                     v
+-------------------------------------------------------------------------+
|                          BACKEND SERVICE LAYER                          |
|               Flask REST API (app.py, routes/, schemas.py)              |
|                                    |                                    |
|              Risk Engine Service (services/risk_engine.py)              |
+-------------------------------------------------------------------------+
                                     |
                             Inference Pipelines
                                     v
+-------------------------------------------------------------------------+
|                           MACHINE LEARNING LAYER                        |
|   Module A (Rainfall Pipeline)    |    Module B (Flood Pipeline)      |
|   GradientBoosting / RF Pipeline  |    DecisionTree / Pipeline       |
+-------------------------------------------------------------------------+
                                     |
                           Raw Datasets & Preprocessing
                                     v
+-------------------------------------------------------------------------+
|                              DATA LAYER                                 |
|            data/raw/ (flood_risk.csv, Indian Rainfall...)               |
+-------------------------------------------------------------------------+
```

---

## 2. Component Breakdown

1. **Data Layer (`data/raw/`)**: Read-only store for historical daily rainfall and station telemetry datasets.
2. **ML Pipeline (`ml/`)**: Contains centralized config (`config.py`), quality auditing (`data_quality.py`), sklearn preprocessors (`preprocessing.py`), model selection (`select_model.py`), and model artifacts (`ml/models/`, `ml/reports/`).
3. **Backend Service (`backend/`)**: Modularized Flask application using Pydantic schema validation (`schemas.py`), decoupled blueprint routes (`routes/`), and compound risk engine (`services/risk_engine.py`).
4. **Frontend Presentation (`frontend/`)**: Modern glassmorphic Web UI with ES JavaScript modules (`frontend/js/modules/`).
5. **Testing Framework (`tests/`)**: Modular test suite split into `tests/ml/`, `tests/api/`, and `tests/validation/`.
