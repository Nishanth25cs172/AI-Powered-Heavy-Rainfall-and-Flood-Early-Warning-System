# Model Selection Criteria Specification

**Project:** AI-Powered Heavy Rainfall and Flood Early Warning System  
**Document Status:** Pre-defined Selection Criteria (Established prior to benchmark execution)

---

## 1. Objective & Decision Rationale

In early-warning decision-support systems for natural hazards (heavy rainfall and flood inundation), missing a critical disaster event (**False Negative**) carries severe safety risks. Therefore, model selection prioritizes **Recall** and **F1-Score** over raw Accuracy.

---

## 2. Selection Metrics & Weights

Candidate models are evaluated against the following prioritized criteria:

1. **Primary Metric: F1-Score & Recall on Holdout Test Set**
   - For **Module A (Rainfall)**: Evaluated on chronological time-based holdout test set (2021–2024). Target is capturing extreme events without over-fitting past years.
   - For **Module B (Flood Inundation)**: Evaluated on stratified holdout test set. Target is maximizing F1-Score while balancing precision and recall.

2. **Secondary Metric: Cross-Validation Generalization & Stability**
   - Minimum standard deviation across folds (`TimeSeriesSplit` for Rainfall, `StratifiedKFold` for Flood).
   - Low generalization gap between train CV score and test score ($| \text{CV}_{\text{score}} - \text{Test}_{\text{score}} | < 0.10$).

3. **Operational Constraints**
   - Fast inference latency (< 50 ms per request for real-time API response).
   - Compact model binary size (< 100 MB).
   - Sklearn Pipeline compatibility with `predict_proba` support.

---

## 3. Decision Rule Matrix

$$\text{Selection Rank} = 0.50 \cdot \text{F1\_Score} + 0.35 \cdot \text{Recall} + 0.15 \cdot \text{ROC\_AUC}$$

The algorithm achieving the highest Selection Rank that satisfies all operational constraints is selected as the primary deployed model for that module.
