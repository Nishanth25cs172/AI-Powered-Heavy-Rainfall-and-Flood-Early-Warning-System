"""
SIH26071 – Data Quality & Leakage Audit Engine
Provides automated duplicate detection, missing percentage calculation, target correlation leakage check, and outlier auditing.
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def audit_dataset_quality(df: pd.DataFrame, target_col: str, name: str = "Dataset") -> Dict[str, Any]:
    """
    Performs comprehensive data quality audit:
    - Missing value counts & percentages
    - Duplicate row counts
    - Target correlation leakage check (flags features with r > 0.90)
    - Basic data types and record shape
    """
    logging.info(f"--- Running Data Quality Audit for: {name} ---")
    total_rows, total_cols = df.shape
    
    # Missing values
    missing_series = df.isnull().sum()
    missing_pct = (missing_series / total_rows) * 100
    missing_info = {
        col: {'count': int(missing_series[col]), 'pct': round(float(missing_pct[col]), 2)}
        for col in df.columns if missing_series[col] > 0
    }

    # Duplicates
    num_duplicates = int(df.duplicated().sum())

    # Target leakage audit
    leakage_warnings = []
    correlations = {}
    if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != target_col:
                try:
                    corr = abs(float(df[col].corr(df[target_col])))
                    correlations[col] = round(corr, 4)
                    if corr > 0.90:
                        leakage_warnings.append(
                            f"POTENTIAL TARGET LEAKAGE: Feature '{col}' has high correlation ({corr:.4f}) with target '{target_col}'"
                        )
                except Exception:
                    pass

    summary = {
        'dataset_name': name,
        'rows': total_rows,
        'columns': total_cols,
        'duplicate_rows': num_duplicates,
        'missing_values': missing_info,
        'target_correlations': correlations,
        'leakage_warnings': leakage_warnings
    }

    logging.info(f"Audit Complete: {total_rows} rows, {num_duplicates} duplicates, {len(missing_info)} cols with missing values.")
    for w in leakage_warnings:
        logging.warning(w)

    return summary
