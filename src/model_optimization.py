"""Módulo de ingeniería de pipelines, validación cruzada y optimización de hiperparámetros.
Diseñado para la Evidencia de Aprendizaje N.º 3 - Desarrollo de IA (ISPC).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_NUM_COLS: List[str] = ['rim', 'width', 'ratio']
FEATURE_CAT_COLS: List[str] = ['Brand']


def build_preprocessor(with_scaler: bool = False, reference_brand: str = 'BRAND_A') -> ColumnTransformer:
    """Construye un ColumnTransformer para imputación y codificación libre de data leakage.
    
    Parámetros:
    -----------
    with_scaler : bool
        Si es True, aplica StandardScaler a las variables numéricas (indispensable para
        modelos lineales regularizados como Ridge y Lasso). Si es False, mantiene la escala
        original (para modelos basados en árboles y OLS).
    reference_brand : str
        Categoría omitida para One-Hot Encoding para evitar multicolinealidad perfecta
        (por defecto 'BRAND_A', manteniendo consistencia con Evidencia 2).
    
    Retorna:
    --------
    ColumnTransformer configurado.
    """
    if with_scaler:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    else:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ])

    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(drop=[reference_brand], sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, FEATURE_NUM_COLS),
            ('cat', cat_pipeline, FEATURE_CAT_COLS)
        ],
        remainder='drop'
    )
    return preprocessor


def build_pipeline(
    regressor: BaseEstimator,
    with_scaler: bool = False,
    reference_brand: str = 'BRAND_A',
    log_target: bool = False
) -> BaseEstimator:
    """Construye un Pipeline integrado con preprocesamiento y modelo estimador.
    
    Si log_target es True, envuelve el pipeline en un TransformedTargetRegressor con log1p y expm1
    para modelar la asimetría y heterocedasticidad en la variable objetivo.
    """
    preprocessor = build_preprocessor(with_scaler=with_scaler, reference_brand=reference_brand)
    
    base_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    if log_target:
        return TransformedTargetRegressor(
            regressor=base_pipe,
            func=np.log1p,
            inverse_func=np.expm1
        )
    return base_pipe


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Obtiene los nombres de las columnas transformadas luego del ajuste."""
    num_names = list(FEATURE_NUM_COLS)
    cat_step = preprocessor.named_transformers_['cat']
    ohe = cat_step.named_steps['onehot']
    cat_names = list(ohe.get_feature_names_out(FEATURE_CAT_COLS))
    return num_names + cat_names
