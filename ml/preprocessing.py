"""
SIH26071 – Machine Learning Preprocessing Pipelines
Builds scikit-learn ColumnTransformers and Pipelines for Flood and Rainfall modules.
"""

from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def build_flood_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Builds ColumnTransformer for Flood Inundation dataset."""
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

def build_rainfall_preprocessor(features: List[str]) -> Pipeline:
    """Builds Sklearn Pipeline for Heavy Rainfall weather dataset."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
